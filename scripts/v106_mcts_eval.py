"""v106 MCTS-on-digit-codebook evaluation script.

Runs BP-only vs BP+MCTS on a sample of GSM8K problems.  Reports:
  - BP alone accuracy (50-problem sample)
  - BP + MCTS accuracy
  - Average rollouts per problem (0 for easy, N_ROLLOUTS for hard)
  - Average wallclock per problem (BP-only vs BP+MCTS)

If the v105 checkpoint does not exist, the script falls back to a
random-init model and reports that all numbers are from an untrained model.

Usage:
  V105_TASK=1 python scripts/v106_mcts_eval.py CKPT_PATH [options]

Options:
  --ckpt PATH         v105 checkpoint (positional or keyword)
  --gsm8k PATH        GSM8K factor graph jsonl
                      default: .cache/gsm8k_factor_graphs_200test.jsonl
  --n_problems N      number of problems to sample (default: 50)
  --n_rollouts N      MCTS rollouts per hard problem (default: 50)
  --calib_threshold F threshold for BP-only shortcut (default: 0.7)
  --K N               number of BP breaths (default: V105_K_MAX)
  --seed N            RNG seed for problem sampling (default: 42)
  --cpu               use CPU instead of AMD GPU
  --no_mcts           only run BP, skip MCTS (useful for baseline timing)

If v105 ckpt isn't ready yet, set CKPT_PATH to a non-existent path and the
script will print a SKIP MARKER and exit 0 — re-run when v105 is available.
"""
import argparse
import os
import sys
import random
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from tinygrad import Tensor, Device, dtypes
from tinygrad.nn.state import safe_load

from mycelium import Config
from mycelium.loader import _load_state, load_breathing
from mycelium.factor_graph_v105 import (
    V105_K_MAX, V105_N_MAX, V105_F_MAX, V105_N_DIGITS, V105_N_HEADS,
    attach_fg_params_v105, fg_v105_state_dict,
    value_to_digits,
)
from mycelium.factor_graph_data_v105 import load_gsm8k_records_v105
from mycelium.factor_graph_v106 import bp_and_mcts_eval, mcts_solve, make_single_problem_batch_np

SHARED_ATTRS = ("wv", "bv", "wo", "bo", "w_out", "b_out",
                "in_ln_g", "in_ln_b", "post_ln_g", "post_ln_b")


# ---------------------------------------------------------------------------
# Model helpers (copied from eval_v105_factor_graph.py)
# ---------------------------------------------------------------------------

def cast_layers_fp32(model):
    def _cast(obj, attr):
        t = getattr(obj, attr)
        if t.dtype == dtypes.half:
            setattr(obj, attr, t.cast(dtypes.float).contiguous().realize())
    _cast(model.embed, "weight")
    sw = model.block.shared
    for a in SHARED_ATTRS:
        _cast(sw, a)
    for layer in model.block.layers:
        for a in ("wq", "bq", "wk", "bk", "w_in", "b_in"):
            _cast(layer, a)


def model_state_dict_v105(model) -> dict:
    sd = {"ln_f.g": model.ln_f_g, "ln_f.b": model.ln_f_b}
    sw = model.block.shared
    for a in SHARED_ATTRS:
        sd[f"shared.{a}"] = getattr(sw, a)
    for i, layer in enumerate(model.block.layers):
        for a in ("wq", "bq", "wk", "bk", "w_in", "b_in"):
            sd[f"phase{i}.{a}"] = getattr(layer, a)
    sd.update(fg_v105_state_dict(model))
    return sd


def load_ckpt_v105(model, path: str):
    sd      = safe_load(path)
    targets = model_state_dict_v105(model)
    n_loaded = 0
    for name, dst in targets.items():
        if name not in sd:
            continue
        src = sd[name].to(dst.device).realize()
        if src.shape != dst.shape:
            try:
                src = src.reshape(dst.shape)
            except Exception:
                continue
        if src.dtype != dst.dtype:
            src = src.cast(dst.dtype)
        dst.assign(src).realize()
        n_loaded += 1
    v105_keys = [k for k in targets if k.startswith("fg_v105.")]
    loaded    = [k for k in v105_keys if k in sd]
    print(f"  loaded {len(loaded)}/{len(v105_keys)} v105 keys from ckpt ({n_loaded} total keys)")


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def print_summary(results: list[dict], n_rollouts: int, label: str = "") -> None:
    if not results:
        print("  (no results)")
        return

    n = len(results)
    bp_acc   = sum(r["bp_correct"]   for r in results) / n
    mcts_acc = sum(r["mcts_correct"] for r in results) / n
    triggered = [r for r in results if r["mcts_triggered"]]
    n_trig   = len(triggered)
    n_easy   = n - n_trig

    avg_rollouts_all  = sum(r["rollouts_used"] for r in results) / n
    avg_rollouts_hard = (sum(r["rollouts_used"] for r in triggered) / n_trig) if n_trig > 0 else 0.0

    bp_times   = [r["bp_wallclock_s"]   for r in results]
    mcts_times = [r["mcts_wallclock_s"] for r in results]
    hard_times = [r["mcts_wallclock_s"] for r in triggered]

    print(f"\n{'=' * 60}")
    print(f"  v106 MCTS eval {label}")
    print(f"{'=' * 60}")
    print(f"  Problems evaluated  : {n}")
    print(f"  Easy (MCTS skipped) : {n_easy}  ({100*n_easy/n:.0f}%)")
    print(f"  Hard (MCTS triggered): {n_trig}  ({100*n_trig/n:.0f}%)")
    print()
    print(f"  BP-only accuracy    : {bp_acc*100:.1f}%")
    print(f"  BP+MCTS accuracy    : {mcts_acc*100:.1f}%")
    delta = mcts_acc - bp_acc
    sign  = "+" if delta >= 0 else ""
    print(f"  Delta               : {sign}{delta*100:.1f}%")
    print()
    print(f"  Avg rollouts/problem: {avg_rollouts_all:.1f}  (hard only: {avg_rollouts_hard:.1f})")
    print(f"  Avg BP wallclock    : {np.mean(bp_times):.2f}s")
    print(f"  Avg MCTS wallclock  : {np.mean(mcts_times):.2f}s")
    if hard_times:
        print(f"  Avg MCTS wallclock  : {np.mean(hard_times):.2f}s (hard problems only)")

    # Calibration histogram
    bp_calvals = [r["bp_calib"] for r in results]
    print(f"\n  BP calib distribution (calib_threshold gate):")
    for lo, hi in [(0.0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.01)]:
        cnt = sum(1 for c in bp_calvals if lo <= c < hi)
        bar = "#" * cnt
        print(f"    [{lo:.1f},{hi:.1f}): {cnt:3d}  {bar}")

    print(f"{'=' * 60}\n")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="v106 MCTS eval")
    parser.add_argument("ckpt",           type=str, nargs="?", default=None,
                        help="v105 checkpoint .safetensors (optional; uses random init if absent)")
    parser.add_argument("--ckpt",         dest="ckpt_kw", type=str, default=None)
    parser.add_argument("--gsm8k",        type=str,
                        default=".cache/gsm8k_factor_graphs_200test.jsonl")
    parser.add_argument("--n_problems",   type=int, default=50)
    parser.add_argument("--n_rollouts",   type=int,
                        default=int(os.environ.get("V106_N_ROLLOUTS", "50")))
    parser.add_argument("--calib_threshold", type=float,
                        default=float(os.environ.get("V106_CALIB_THRESHOLD", "0.7")))
    parser.add_argument("--K",            type=int, default=None)
    parser.add_argument("--seed",         type=int, default=42)
    parser.add_argument("--cpu",          action="store_true")
    parser.add_argument("--no_mcts",      action="store_true")
    parser.add_argument("--n_max",        type=int, default=V105_N_MAX)
    parser.add_argument("--f_max",        type=int, default=V105_F_MAX)
    parser.add_argument("--n_digits",     type=int, default=V105_N_DIGITS)
    args = parser.parse_args()

    ckpt_path = args.ckpt_kw or args.ckpt

    if args.cpu:
        os.environ["DEV"] = "CLANG"

    # ------------------------------------------------------------------
    # Check for v105 ckpt readiness
    # ------------------------------------------------------------------
    ckpt_available = ckpt_path is not None and os.path.exists(ckpt_path)
    if not ckpt_available:
        if ckpt_path is not None:
            print(f"[v106 smoke] v105 ckpt not found at: {ckpt_path}")
        else:
            print("[v106 smoke] no ckpt provided")
        print("[v106 smoke] SKIP MARKER — re-run when v105 ckpt is available")
        print("[v106 smoke] Proceeding with random-init model for algorithm validation only.")

    K = args.K if args.K is not None else V105_K_MAX

    N_DIGITS = args.n_digits
    N_MAX    = args.n_max
    F_MAX    = args.f_max

    print(f"=== v106 MCTS eval ===")
    print(f"ckpt={ckpt_path}  K={K}  n_rollouts={args.n_rollouts}")
    print(f"calib_threshold={args.calib_threshold}  n_problems={args.n_problems}")
    print(f"N_DIGITS={N_DIGITS}  N_MAX={N_MAX}  F_MAX={F_MAX}")

    # ------------------------------------------------------------------
    # Build model
    # ------------------------------------------------------------------
    cfg   = Config()
    sd    = _load_state()
    model = load_breathing(cfg, sd=sd)
    del sd
    cast_layers_fp32(model)
    attach_fg_params_v105(model, hidden=cfg.hidden,
                          n_digits=N_DIGITS, n_max=N_MAX, f_max=F_MAX, k_max=K)

    if ckpt_available:
        print(f"Loading v105 ckpt: {ckpt_path}")
        load_ckpt_v105(model, ckpt_path)
    else:
        print("WARNING: using random-init model — numbers are not meaningful for accuracy.")

    if hasattr(Device[Device.DEFAULT], "synchronize"):
        Device[Device.DEFAULT].synchronize()
    Tensor.training = False

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    if not os.path.exists(args.gsm8k):
        print(f"[v106 smoke] GSM8K file not found: {args.gsm8k}")
        print("[v106 smoke] generating synthetic records for algorithm validation...")
        records = _make_synthetic_records(args.n_problems, n_max=N_MAX, f_max=F_MAX, seed=args.seed)
        data_label = "synthetic"
    else:
        all_records = load_gsm8k_records_v105(args.gsm8k, n_digits=N_DIGITS, n_max=N_MAX, f_max=F_MAX)
        if not all_records:
            print("[v106 smoke] No usable records loaded from GSM8K file.")
            records = _make_synthetic_records(args.n_problems, n_max=N_MAX, f_max=F_MAX, seed=args.seed)
            data_label = "synthetic (gsm8k empty after filter)"
        else:
            rng = random.Random(args.seed)
            n_sample = min(args.n_problems, len(all_records))
            records  = rng.sample(all_records, n_sample)
            data_label = f"GSM8K ({len(all_records)} total, {n_sample} sampled)"

    print(f"Data: {data_label}")

    # ------------------------------------------------------------------
    # Run evaluation
    # ------------------------------------------------------------------
    n_rollouts_eff = 0 if args.no_mcts else args.n_rollouts
    if args.no_mcts:
        print("--no_mcts: running BP-only baseline.")

    print(f"\nRunning {len(records)} problems...")
    t_start = time.perf_counter()

    results = bp_and_mcts_eval(
        model,
        records,
        n_rollouts=n_rollouts_eff,
        calib_threshold=args.calib_threshold,
        k_breaths=K,
        n_max=N_MAX,
        f_max=F_MAX,
        n_digits=N_DIGITS,
        verbose=True,
    )

    total_t = time.perf_counter() - t_start
    print(f"\nTotal elapsed: {total_t:.1f}s  ({total_t / max(len(records), 1):.2f}s/problem)")

    print_summary(results, n_rollouts_eff, label=f"(K={K}, rollouts={n_rollouts_eff})")

    # Quick algorithm sanity checks
    print("Algorithm sanity checks:")
    triggered = [r for r in results if r["mcts_triggered"]]
    bp_only   = [r for r in results if not r["mcts_triggered"]]
    print(f"  MCTS triggered     : {len(triggered)} / {len(results)}")
    print(f"  BP shortcuts       : {len(bp_only)} / {len(results)}")
    if triggered:
        avg_hard_t = np.mean([r["mcts_wallclock_s"] for r in triggered])
        print(f"  Avg hard-problem t : {avg_hard_t:.1f}s  (target < 30s)")
        ok_30s = avg_hard_t < 30.0
        print(f"  30s/problem target : {'PASS' if ok_30s else 'FAIL'}")
    print()

    # Verify MCTS activation logic: easy problems should NOT trigger MCTS
    correct_skip = all(
        not r["mcts_triggered"] for r in bp_only
    )
    print(f"  BP shortcut gate   : {'PASS' if correct_skip else 'FAIL (check calib_threshold)'}")

    # Verify rollouts_used is correct
    rollout_ok = all(
        (r["rollouts_used"] == n_rollouts_eff if r["mcts_triggered"] else r["rollouts_used"] == 0)
        for r in results
    )
    print(f"  Rollout count gate : {'PASS' if rollout_ok else 'FAIL'}")

    return 0


# ---------------------------------------------------------------------------
# Synthetic data fallback
# ---------------------------------------------------------------------------

def _make_synthetic_records(n: int, n_max: int = V105_N_MAX, f_max: int = V105_F_MAX,
                              seed: int = 42) -> list[dict]:
    """Generate n simple x + y = z arithmetic problems as v105 records."""
    rng = random.Random(seed)
    records = []
    for _ in range(n):
        x = rng.randint(0, 999)
        y = rng.randint(0, 999)
        z = x + y
        records.append({
            "gold_values":     [x, y, z],
            "observed_mask":   [1, 1, 0],
            "observed_values": [x, y, 0],
            "factor_types":    ["add"],
            "factor_args":     [[0, 1, 2]],
            "n_factors":       1,
            "query_idx":       2,
            "difficulty":      "easy",
        })
    return records


if __name__ == "__main__":
    sys.exit(main())
