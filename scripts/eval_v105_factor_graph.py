"""v105 factor graph evaluation — digit-by-digit codebook architecture.

Usage:
  python scripts/eval_v105_factor_graph.py CKPT_PATH [options]

Options:
  --test PATH         validation set jsonl (default: .cache/factor_graph_test.jsonl)
  --gsm8k PATH        GSM8K test set (default: .cache/gsm8k_factor_graphs_200test.jsonl)
  --K N               number of breaths (default: V105_K_MAX)
  --K_sweep 1,2,5,8   run eval at multiple K values
  --batch N           eval batch size (default: 8)
  --n_digits N        digit positions per variable (default: V105_N_DIGITS=5)
  --cpu               run on CPU (default: AMD GPU)

Diagnostics printed:
  - Per-difficulty cell accuracy and query accuracy
  - Per-digit accuracy (how often each digit position is correct)
  - Value-range breakdown (observed values bucketed by magnitude)
  - K-sweep table if --K_sweep given
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from tinygrad import Tensor, Device, dtypes
from tinygrad.nn.state import safe_load

from mycelium import Config, BreathingTransformer
from mycelium.loader import _load_state, load_breathing
from mycelium.factor_graph_v105 import (
    V105_K_MAX, V105_N_MAX, V105_F_MAX, V105_N_DIGITS, V105_N_HEADS,
    attach_fg_params_v105, fg_v105_state_dict,
    fg_breathing_forward_v105, fg_accuracy_v105,
    _compile_jit_fg_eval_v105,
    value_to_digits, digits_to_value,
)
from mycelium.factor_graph_data_v105 import (
    FactorGraphLoaderV105, load_gsm8k_records_v105,
    _records_to_batch_v105, batch_to_tensors_v105,
)

DIFFICULTIES = ["easy", "medium", "hard", "gsm8k"]
SHARED_ATTRS = ("wv", "bv", "wo", "bo", "w_out", "b_out",
                "in_ln_g", "in_ln_b", "post_ln_g", "post_ln_b")


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
    v105_keys = [k for k in targets if k.startswith("fg_v105.")]
    loaded    = [k for k in v105_keys if k in sd]
    print(f"  loaded {len(loaded)}/{len(v105_keys)} v105 keys from ckpt")


def run_eval(
    model,
    records: list[dict],
    K: int,
    batch_size: int,
    n_max: int = V105_N_MAX,
    f_max: int = V105_F_MAX,
    n_digits: int = V105_N_DIGITS,
    eval_fn=None,
    max_batches: int | None = None,
) -> dict:
    """Evaluate over a list of records. Returns stats dict."""
    Tensor.training = False

    agg = {
        "n_unobs": 0, "n_correct_unobs": 0,
        "query_correct": 0, "n_puzzles": 0,
        "digit_correct": 0, "digit_total": 0,
    }
    # Per-digit-position accuracy
    digit_pos_correct = np.zeros(n_digits, dtype=np.int64)
    digit_pos_total   = np.zeros(n_digits, dtype=np.int64)

    n = len(records)
    n_batches = 0
    for start in range(0, n, batch_size):
        batch_recs = records[start : start + batch_size]
        while len(batch_recs) < batch_size:
            batch_recs.append(records[0])
        batch_np = _records_to_batch_v105(batch_recs, n_max, f_max, k_max=K, n_heads=V105_N_HEADS, n_digits=n_digits)
        batch    = batch_to_tensors_v105(batch_np)

        if eval_fn is not None:
            pred_dg_t, _ = eval_fn(
                batch["digit_init"], batch["node_kinds"],
                batch["staging_mask"], batch["head_op_mask"],
                batch["gold_digits"], batch["observed_mask"],
            )
            pred_dg = pred_dg_t.numpy()
        else:
            dig_lh, _, _ = fg_breathing_forward_v105(
                model, batch["digit_init"], batch["node_kinds"],
                batch["staging_mask"], batch["head_op_mask"],
                K=K, n_max=n_max, f_max=f_max, n_digits=n_digits,
            )
            pred_dg = dig_lh[-1].argmax(axis=-1).realize().numpy()

        gold_dg = batch_np["gold_digits"]   # (B, N_MAX, N_DIGITS)
        obs_np  = batch_np["observed_mask"]  # (B, N_MAX)
        qi_np   = batch_np["query_idx"]
        nv_np   = batch_np["n_vars_total"]

        B_local = len(batch_recs)
        for b in range(B_local):
            nv = int(nv_np[b])
            qi = int(qi_np[b])
            for vi in range(min(nv, n_max)):
                if obs_np[b, vi] == 0:
                    agg["n_unobs"] += 1
                    if np.all(pred_dg[b, vi] == gold_dg[b, vi]):
                        agg["n_correct_unobs"] += 1
                    for p in range(n_digits):
                        digit_pos_total[p]   += 1
                        if pred_dg[b, vi, p] == gold_dg[b, vi, p]:
                            digit_pos_correct[p] += 1
            if qi < n_max and np.all(pred_dg[b, qi] == gold_dg[b, qi]):
                agg["query_correct"] += 1
            agg["n_puzzles"] += 1

        n_batches += 1
        if max_batches is not None and n_batches >= max_batches:
            break

    cell_acc  = agg["n_correct_unobs"] / max(agg["n_unobs"], 1)
    query_acc = agg["query_correct"]   / max(agg["n_puzzles"], 1)
    per_digit_acc = (digit_pos_correct / (digit_pos_total + 1e-8)).tolist()

    Tensor.training = True
    return {
        "cell_acc":     cell_acc,
        "query_acc":    query_acc,
        "n_puzzles":    agg["n_puzzles"],
        "n_unobs":      agg["n_unobs"],
        "per_digit_acc": per_digit_acc,  # list[float] length n_digits
    }


def main():
    parser = argparse.ArgumentParser(description="v105 factor graph eval")
    parser.add_argument("ckpt",         type=str,  help="checkpoint path (.safetensors)")
    parser.add_argument("--test",       type=str,  default=".cache/factor_graph_test.jsonl")
    parser.add_argument("--gsm8k",      type=str,  default=".cache/gsm8k_factor_graphs_200test.jsonl")
    parser.add_argument("--K",          type=int,  default=None)
    parser.add_argument("--K_sweep",    type=str,  default="")
    parser.add_argument("--batch",      type=int,  default=8)
    parser.add_argument("--n_digits",   type=int,  default=V105_N_DIGITS)
    parser.add_argument("--n_max",      type=int,  default=V105_N_MAX)
    parser.add_argument("--f_max",      type=int,  default=V105_F_MAX)
    parser.add_argument("--max_batches",type=int,  default=None)
    parser.add_argument("--cpu",        action="store_true")
    args = parser.parse_args()

    if args.cpu:
        os.environ["DEV"] = "CLANG"

    K_default = args.K if args.K is not None else V105_K_MAX
    K_sweep   = [int(k) for k in args.K_sweep.split(",") if k.strip()] if args.K_sweep else [K_default]
    K_max_needed = max(K_sweep)

    N_DIGITS = args.n_digits
    N_MAX    = args.n_max
    F_MAX    = args.f_max

    print(f"=== v105 eval: {args.ckpt} ===")
    print(f"K={K_default}  K_sweep={K_sweep}  N_DIGITS={N_DIGITS}  T={N_MAX*N_DIGITS+F_MAX}")

    cfg   = Config()
    sd    = _load_state()
    model = load_breathing(cfg, sd=sd)
    del sd
    cast_layers_fp32(model)

    attach_fg_params_v105(model, hidden=cfg.hidden,
                          n_digits=N_DIGITS, n_max=N_MAX, f_max=F_MAX, k_max=K_max_needed)
    load_ckpt_v105(model, args.ckpt)
    Device[Device.DEFAULT].synchronize()

    # Load eval data
    import json as _json
    def load_jsonl(path):
        out = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(_json.loads(line))
        return out
    test_records = load_jsonl(args.test) if os.path.exists(args.test) else []
    gsm8k_records = load_gsm8k_records_v105(args.gsm8k, n_digits=N_DIGITS, n_max=N_MAX, f_max=F_MAX) \
        if os.path.exists(args.gsm8k) else []
    print(f"test records: {len(test_records)}   gsm8k records (kept): {len(gsm8k_records)}")

    Tensor.training = False
    eval_fn = _compile_jit_fg_eval_v105(model, K=K_default, B=args.batch,
                                         n_max=N_MAX, f_max=F_MAX, n_digits=N_DIGITS)

    # --- Synthetic test eval ---
    if test_records:
        print(f"\n--- Synthetic test set (N={len(test_records)}) ---")
        r = run_eval(model, test_records, K=K_default, batch_size=args.batch,
                     n_max=N_MAX, f_max=F_MAX, n_digits=N_DIGITS,
                     eval_fn=eval_fn, max_batches=args.max_batches)
        print(f"  cell_acc={r['cell_acc']:.3f}  query_acc={r['query_acc']:.3f}  n={r['n_puzzles']}")
        dag = [f"d{p}={r['per_digit_acc'][p]:.3f}" for p in range(N_DIGITS)]
        print(f"  per_digit_acc: {' '.join(dag)}")

    # --- GSM8K eval ---
    if gsm8k_records:
        print(f"\n--- GSM8K test set (N={len(gsm8k_records)}) ---")
        r = run_eval(model, gsm8k_records, K=K_default, batch_size=args.batch,
                     n_max=N_MAX, f_max=F_MAX, n_digits=N_DIGITS,
                     eval_fn=eval_fn, max_batches=args.max_batches)
        print(f"  cell_acc={r['cell_acc']:.3f}  query_acc={r['query_acc']:.3f}  n={r['n_puzzles']}")
        dag = [f"d{p}={r['per_digit_acc'][p]:.3f}" for p in range(N_DIGITS)]
        print(f"  per_digit_acc: {' '.join(dag)}")

    # --- K sweep ---
    if len(K_sweep) > 1 and test_records:
        print(f"\n--- K sweep on synthetic test (N={len(test_records)}) ---")
        print(f"{'K':>4}  {'cell_acc':>9}  {'query_acc':>10}")
        for k in K_sweep:
            eval_fn_k = _compile_jit_fg_eval_v105(model, K=k, B=args.batch,
                                                    n_max=N_MAX, f_max=F_MAX, n_digits=N_DIGITS)
            r = run_eval(model, test_records, K=k, batch_size=args.batch,
                         n_max=N_MAX, f_max=F_MAX, n_digits=N_DIGITS,
                         eval_fn=eval_fn_k, max_batches=args.max_batches)
            print(f"{k:>4}  {r['cell_acc']:>9.3f}  {r['query_acc']:>10.3f}")


if __name__ == "__main__":
    main()
