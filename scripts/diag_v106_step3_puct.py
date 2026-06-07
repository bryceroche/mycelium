"""v106 → v110-step3 PUCT diagnostic.

Loads the v110-step3 balanced champion ckpt, evaluates BP-only and BP+PUCT
on a hard subset of the val set, and reports the delta + wall-clock.

Usage:
  V110_STEP3_TASK=1 .venv/bin/python scripts/diag_v106_step3_puct.py

Env overrides:
  CKPT                          (.cache/fg_v110_step3_ckpts/v110_step3_cont8_step1000.safetensors)
  VAL_PATH                      (.cache/factor_graph_test.jsonl)
  V106_S3_N_ROLLOUTS=30
  V106_S3_CALIB_THRESHOLD=0.85
  V106_S3_N_PUZZLES=50          (how many hard puzzles to evaluate)
  V106_S3_DIFFICULTY=hard
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set env BEFORE importing the v110-step3 modules so constants are correct
os.environ.setdefault("V110_STEP3_TASK", "1")
os.environ.setdefault("V110_STEP3_K_MAX", "8")
os.environ.setdefault("V110_STEP3_N_DIGITS", "5")
os.environ.setdefault("V110_STEP3_N_MAX", "16")
os.environ.setdefault("V110_STEP3_F_MAX", "8")
os.environ.setdefault("V110_STEP3_WAIST_DIM", "512")
os.environ.setdefault("V110_STEP3_ALTERNATION", "1")
os.environ.setdefault("V110_STEP3_PHASE_SCALE", "1.0")
os.environ.setdefault("V110_STEP3_HARD_BREATH_LEVEL", "0")
# Also expose to v110-step constants (which v106_step3 reads at import time)
os.environ.setdefault("V110_STEP_K_MAX",    os.environ["V110_STEP3_K_MAX"])
os.environ.setdefault("V110_STEP_N_DIGITS", os.environ["V110_STEP3_N_DIGITS"])
os.environ.setdefault("V110_STEP_N_MAX",    os.environ["V110_STEP3_N_MAX"])
os.environ.setdefault("V110_STEP_F_MAX",    os.environ["V110_STEP3_F_MAX"])
os.environ.setdefault("V110_STEP_WAIST_DIM", os.environ["V110_STEP3_WAIST_DIM"])
os.environ.setdefault("V110_STEP_ALTERNATION", os.environ["V110_STEP3_ALTERNATION"])
os.environ.setdefault("V110_STEP_PHASE_SCALE", os.environ["V110_STEP3_PHASE_SCALE"])

import numpy as np
from tinygrad import Device, Tensor, dtypes

from mycelium import Config
from mycelium.loader import _load_state, load_breathing
from mycelium.factor_graph_v110_step3 import (
    V110_STEP3_K_MAX, V110_STEP3_N_MAX, V110_STEP3_F_MAX, V110_STEP3_N_HEADS,
    V110_STEP3_N_DIGITS,
    V110_STEP3_WAIST_DIM, V110_STEP3_ALTERNATION, V110_STEP3_PHASE_SCALE,
    V110_STEP3_CODEBOOK_N, V110_STEP3_IB_CENTROIDS,
    V110_STEP3_GATE_PROFILE, V110_STEP3_PHOTON_ALPHA,
    attach_fg_params_v110_step3,
)
from mycelium.factor_graph_v108 import bins_to_digits_msd, digits_to_value_msd
from mycelium.factor_graph_data_v107 import (
    FactorGraphLoaderV107, _records_to_batch_v107,
)
from scripts.v108_factor_graph_train import cast_layers_fp32
from scripts.v110_step3_factor_graph_train import load_ckpt_v110_step3

from mycelium.factor_graph_v106_step3 import (
    puct_solve, extract_single_problem, argmax_decode_values,
)


def main():
    CKPT = os.environ.get(
        "CKPT",
        ".cache/fg_v110_step3_ckpts/v110_step3_cont8_step1000.safetensors",
    )
    VAL_PATH = os.environ.get("VAL_PATH", ".cache/factor_graph_test.jsonl")
    SEED = int(os.environ.get("SEED", "42"))
    N_PUZZLES = int(os.environ.get("V106_S3_N_PUZZLES", "50"))
    DIFFICULTY = os.environ.get("V106_S3_DIFFICULTY", "hard")
    N_ROLLOUTS = int(os.environ.get("V106_S3_N_ROLLOUTS", "30"))
    CALIB_THRESHOLD = float(os.environ.get("V106_S3_CALIB_THRESHOLD", "0.85"))

    print(f"=== v106 → v110-step3 PUCT diagnostic ===")
    print(f"  ckpt: {CKPT}")
    print(f"  val:  {VAL_PATH}")
    print(f"  difficulty: {DIFFICULTY}")
    print(f"  N_puzzles: {N_PUZZLES}")
    print(f"  N_rollouts: {N_ROLLOUTS}")
    print(f"  calib_threshold: {CALIB_THRESHOLD}")
    print(f"  K_max: {V110_STEP3_K_MAX}, alternation: {V110_STEP3_ALTERNATION}, "
          f"phase_scale: {V110_STEP3_PHASE_SCALE}")
    print()

    np.random.seed(SEED)
    Tensor.manual_seed(SEED)

    cfg = Config()
    print("loading Pythia-410M...")
    sd = _load_state()
    model = load_breathing(cfg, sd=sd)
    del sd
    cast_layers_fp32(model)

    attach_fg_params_v110_step3(
        model, hidden=cfg.hidden,
        n_max=V110_STEP3_N_MAX, f_max=V110_STEP3_F_MAX, k_max=V110_STEP3_K_MAX,
        n_digits=V110_STEP3_N_DIGITS, n_code=V110_STEP3_CODEBOOK_N,
        ib_centroids_path=V110_STEP3_IB_CENTROIDS,
        waist_dim=V110_STEP3_WAIST_DIM,
    )
    Device[Device.DEFAULT].synchronize()

    print(f"loading ckpt: {CKPT}")
    load_ckpt_v110_step3(model, CKPT)
    print()

    # Load val loader, filter to specified difficulty
    val_loader = FactorGraphLoaderV107(
        VAL_PATH, batch_size=1,
        difficulty_filter=DIFFICULTY, curriculum=False,
        n_max=V110_STEP3_N_MAX, f_max=V110_STEP3_F_MAX, k_max=V110_STEP3_K_MAX,
        n_heads=V110_STEP3_N_HEADS,
        seed=SEED + 2,
    )

    # Pre-pick deterministic subset: take first N_PUZZLES records of that difficulty
    records = val_loader.records[:N_PUZZLES]
    print(f"  will evaluate {len(records)} {DIFFICULTY} puzzles\n", flush=True)

    # ------------------------------------------------------------------
    # Loop: per-puzzle BP-only, BP+PUCT
    # ------------------------------------------------------------------
    Tensor.training = False

    bp_cell_correct = 0
    bp_cell_total = 0
    bp_query_correct = 0
    bp_query_total = 0
    puct_cell_correct = 0
    puct_cell_total = 0
    puct_query_correct = 0
    puct_query_total = 0

    bp_wallclock_total = 0.0
    puct_wallclock_total = 0.0

    mcts_triggered_count = 0
    n_clamps_total = 0

    t_loop_start = time.time()

    for i, rec in enumerate(records):
        # Build a B=1 batch_np for this single record
        batch_np = _records_to_batch_v107(
            [rec],
            n_max=V110_STEP3_N_MAX, f_max=V110_STEP3_F_MAX,
            k_max=V110_STEP3_K_MAX, n_heads=V110_STEP3_N_HEADS,
        )
        # Gold values per variable (n_total)
        gold_values_np = batch_np["gold_values"][0]    # (N_MAX,) int
        obs_mask_np = batch_np["observed_mask"][0]
        n_total = int(batch_np["n_vars_total"][0])
        query_idx = int(batch_np["query_idx"][0])

        # BP-only
        t_bp0 = time.perf_counter()
        bp_res = puct_solve(
            model, batch_np,
            n_rollouts=0,                       # forces BP-only path
            calib_threshold=CALIB_THRESHOLD,
            k_breaths=V110_STEP3_K_MAX,
            n_max=V110_STEP3_N_MAX, f_max=V110_STEP3_F_MAX,
            n_digits=V110_STEP3_N_DIGITS,
            alternation=V110_STEP3_ALTERNATION,
            phase_scale=V110_STEP3_PHASE_SCALE,
            gate_profile=V110_STEP3_GATE_PROFILE,
            photon_alpha=V110_STEP3_PHOTON_ALPHA,
        )
        bp_dt = time.perf_counter() - t_bp0
        bp_wallclock_total += bp_dt

        # BP+PUCT
        t_p0 = time.perf_counter()
        puct_res = puct_solve(
            model, batch_np,
            n_rollouts=N_ROLLOUTS,
            calib_threshold=CALIB_THRESHOLD,
            k_breaths=V110_STEP3_K_MAX,
            n_max=V110_STEP3_N_MAX, f_max=V110_STEP3_F_MAX,
            n_digits=V110_STEP3_N_DIGITS,
            alternation=V110_STEP3_ALTERNATION,
            phase_scale=V110_STEP3_PHASE_SCALE,
            gate_profile=V110_STEP3_GATE_PROFILE,
            photon_alpha=V110_STEP3_PHOTON_ALPHA,
        )
        puct_dt = time.perf_counter() - t_p0
        puct_wallclock_total += puct_dt

        if puct_res["mcts_triggered"]:
            mcts_triggered_count += 1
        n_clamps_total += len(puct_res["final_clamps"])

        # Per-puzzle accuracy: cell (all unobserved vars correct?) and query
        # Note: gold_values vs predicted_values comparison via integer equality
        bp_pred = bp_res["predicted_values"]
        puct_pred = puct_res["predicted_values"]

        # Compute per-cell correctness (per unobserved variable)
        for vi in range(min(n_total, V110_STEP3_N_MAX)):
            if obs_mask_np[vi] != 0:
                continue
            bp_cell_total += 1
            puct_cell_total += 1
            if int(bp_pred[vi]) == int(gold_values_np[vi]):
                bp_cell_correct += 1
            if int(puct_pred[vi]) == int(gold_values_np[vi]):
                puct_cell_correct += 1

        # Query accuracy
        if 0 <= query_idx < V110_STEP3_N_MAX:
            bp_query_total += 1
            puct_query_total += 1
            if int(bp_pred[query_idx]) == int(gold_values_np[query_idx]):
                bp_query_correct += 1
            if int(puct_pred[query_idx]) == int(gold_values_np[query_idx]):
                puct_query_correct += 1

        if (i + 1) % 5 == 0 or i == 0:
            bp_acc_so_far = bp_cell_correct / max(bp_cell_total, 1)
            puct_acc_so_far = puct_cell_correct / max(puct_cell_total, 1)
            trig = "MCTS" if puct_res["mcts_triggered"] else "BP "
            elapsed = time.time() - t_loop_start
            print(
                f"  [{i+1:3d}/{len(records)}] {trig}  "
                f"bp_cell={bp_acc_so_far:.3f}  puct_cell={puct_acc_so_far:.3f}  "
                f"bp_t={bp_dt:.1f}s  puct_t={puct_dt:.1f}s  "
                f"clamps={len(puct_res['final_clamps'])}  "
                f"(elapsed={elapsed:.0f}s)",
                flush=True,
            )

    # --------------------------------------------------------------
    # Summary
    # --------------------------------------------------------------
    n = len(records)
    bp_cell = bp_cell_correct / max(bp_cell_total, 1)
    pct_cell = puct_cell_correct / max(puct_cell_total, 1)
    bp_q = bp_query_correct / max(bp_query_total, 1)
    pct_q = puct_query_correct / max(puct_query_total, 1)
    mean_bp_t = bp_wallclock_total / max(n, 1)
    mean_puct_t = puct_wallclock_total / max(n, 1)

    print("\n" + "=" * 70)
    print(f"=== SUMMARY (n={n} puzzles, difficulty={DIFFICULTY}) ===")
    print("=" * 70)
    print(f"  BP-only   cell_acc: {bp_cell:.4f}  ({bp_cell_correct}/{bp_cell_total})")
    print(f"  BP+PUCT   cell_acc: {pct_cell:.4f}  ({puct_cell_correct}/{puct_cell_total})")
    print(f"  Δ cell_acc        : {pct_cell - bp_cell:+.4f}")
    print()
    print(f"  BP-only   query_acc: {bp_q:.4f}  ({bp_query_correct}/{bp_query_total})")
    print(f"  BP+PUCT   query_acc: {pct_q:.4f}  ({puct_query_correct}/{puct_query_total})")
    print(f"  Δ query_acc        : {pct_q - bp_q:+.4f}")
    print()
    print(f"  Mean wall-clock per puzzle: BP-only={mean_bp_t:.2f}s  "
          f"PUCT={mean_puct_t:.2f}s  (×{mean_puct_t/max(mean_bp_t, 1e-3):.1f})")
    print(f"  Total wall-clock           : BP-only={bp_wallclock_total:.1f}s  "
          f"PUCT={puct_wallclock_total:.1f}s")
    print(f"  MCTS triggered on  : {mcts_triggered_count}/{n} puzzles "
          f"(calib_threshold={CALIB_THRESHOLD})")
    if mcts_triggered_count > 0:
        avg_clamps = n_clamps_total / mcts_triggered_count
        print(f"  Avg clamps applied : {avg_clamps:.1f} per MCTS-triggered puzzle")
    print()


if __name__ == "__main__":
    main()
