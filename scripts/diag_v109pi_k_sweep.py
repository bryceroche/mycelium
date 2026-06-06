"""v109pi K-sweep diagnostic.

Tests whether per-breath Q rotation preserves (or improves) v109's
non-decreasing K-sweep dynamics.

v108 baseline (no waist, no alternation):
  pos4 hard: K=2 → 0.191, K=4 → 0.138, K=8 → 0.132  (DECREASES with K)

v109 (waist + alternation):
  Flipped v108's K-sweep to non-decreasing at smoke (step 500).

v109pi (waist + alternation + per-breath Q rotation by k·π/K_max):
  Open question. The rotation is calibrated to K=K_max=8 — at K<K_max
  the phase grid is non-uniform.

Usage:
  CKPT=.cache/fg_v109pi_ckpts/v109pi_cont9_step500.safetensors \
    .venv/bin/python scripts/diag_v109pi_k_sweep.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("V109PI_TASK", "1")
os.environ.setdefault("V109PI_K_MAX", "8")
os.environ.setdefault("V109PI_N_DIGITS", "5")
os.environ.setdefault("V109PI_N_MAX", "16")
os.environ.setdefault("V109PI_F_MAX", "8")
os.environ.setdefault("V109PI_WAIST_DIM", "512")
os.environ.setdefault("V109PI_ALTERNATION", "1")
os.environ.setdefault("V109PI_PHASE_SCALE", "1.0")
os.environ.setdefault("V109PI_HARD_BREATH_LEVEL", "0")

import numpy as np
from tinygrad import Device, Tensor, dtypes

from mycelium import Config
from mycelium.loader import _load_state, load_breathing
from mycelium.factor_graph_v109pi import (
    V109PI_K_MAX, V109PI_N_MAX, V109PI_F_MAX, V109PI_N_HEADS, V109PI_N_DIGITS,
    V109PI_WAIST_DIM, V109PI_ALTERNATION, V109PI_PHASE_SCALE,
    V109PI_CODEBOOK_N, V109PI_IB_CENTROIDS,
    attach_fg_params_v109pi,
    _compile_jit_fg_eval_v109pi,
)
from mycelium.factor_graph_data_v107 import FactorGraphLoaderV107
from scripts.v108_factor_graph_train import cast_layers_fp32
from scripts.v109pi_factor_graph_train import load_ckpt_v109pi, evaluate_v109pi


def main():
    CKPT = os.environ.get(
        "CKPT", ".cache/fg_v109pi_ckpts/v109pi_cont9_step500.safetensors"
    )
    BATCH = int(os.environ.get("BATCH", "4"))
    EVAL_BATCHES = int(os.environ.get("EVAL_BATCHES", "20"))
    VAL_PATH = os.environ.get("VAL_PATH", ".cache/factor_graph_test.jsonl")
    SEED = int(os.environ.get("SEED", "42"))
    K_VALUES = [int(k) for k in os.environ.get("K_VALUES", "2,4,8").split(",")]
    PHASE_SCALE = float(os.environ.get("V109PI_PHASE_SCALE", "1.0"))

    print(f"=== v109pi K-sweep diagnostic ===")
    print(f"  ckpt: {CKPT}")
    print(f"  val:  {VAL_PATH}")
    print(f"  alternation: {V109PI_ALTERNATION}")
    print(f"  phase_scale: {PHASE_SCALE}")
    print(f"  K values: {K_VALUES}")
    print()

    np.random.seed(SEED)
    Tensor.manual_seed(SEED)

    cfg = Config()
    print("loading Pythia-410M...")
    sd = _load_state()
    model = load_breathing(cfg, sd=sd)
    del sd
    cast_layers_fp32(model)

    attach_fg_params_v109pi(
        model, hidden=cfg.hidden,
        n_max=V109PI_N_MAX, f_max=V109PI_F_MAX, k_max=V109PI_K_MAX,
        n_digits=V109PI_N_DIGITS, n_code=V109PI_CODEBOOK_N,
        ib_centroids_path=V109PI_IB_CENTROIDS,
        waist_dim=V109PI_WAIST_DIM,
    )
    Device[Device.DEFAULT].synchronize()

    print(f"loading ckpt: {CKPT}")
    load_ckpt_v109pi(model, CKPT)
    print()

    val_loader = FactorGraphLoaderV107(
        VAL_PATH, batch_size=BATCH,
        difficulty_filter=None, curriculum=False,
        n_max=V109PI_N_MAX, f_max=V109PI_F_MAX, k_max=V109PI_K_MAX,
        n_heads=V109PI_N_HEADS,
        seed=SEED + 2,
    )

    Tensor.training = False

    results_by_k = {}
    for K in K_VALUES:
        print(f"\n=== EVAL at K={K} (phase_scale={PHASE_SCALE}) ===")
        eval_fn = _compile_jit_fg_eval_v109pi(
            model, K=K, B=BATCH,
            n_max=V109PI_N_MAX, f_max=V109PI_F_MAX, n_digits=V109PI_N_DIGITS,
            alternation=V109PI_ALTERNATION, phase_scale=PHASE_SCALE,
        )
        t0 = time.time()
        results = evaluate_v109pi(
            model, val_loader, K=K, max_batches=EVAL_BATCHES, eval_fn=eval_fn,
            n_max=V109PI_N_MAX, f_max=V109PI_F_MAX, n_digits=V109PI_N_DIGITS,
        )
        dt = time.time() - t0
        print(f"  ({dt:.1f}s)")
        for d in ("easy", "medium", "hard"):
            if d not in results: continue
            v = results[d]
            pp = " ".join(f"{p:.3f}" for p in v["per_pos_acc"])
            print(f"  val[{d:6s}]: cell={v['cell_acc']:.3f} q={v['query_acc']:.3f} "
                  f"digit={v['digit_acc']:.3f} per_pos=[{pp}] n={v['n_puzzles']}")
        results_by_k[K] = results

    print("\n=== K-SWEEP COMPARISON ===\n")
    for d in ("easy", "medium", "hard"):
        print(f"--- val[{d}] ---")
        print(f"  {'K':>3} | {'pos0':>6} {'pos1':>6} {'pos2':>6} {'pos3':>6} {'pos4':>6} | digit  cell")
        for K in K_VALUES:
            if d not in results_by_k[K]: continue
            v = results_by_k[K][d]
            pp = "  ".join(f"{p:.3f}" for p in v["per_pos_acc"])
            print(f"  {K:>3} | {pp} | {v['digit_acc']:.3f} {v['cell_acc']:.3f}")
        print()

    print("=== TREND: pos4 (ones digit) acc vs K ===\n")
    for d in ("easy", "medium", "hard"):
        print(f"  {d:6s}:", end=" ")
        for K in K_VALUES:
            if d not in results_by_k[K]: continue
            pos4 = results_by_k[K][d]["per_pos_acc"][4]
            print(f"K={K}→{pos4:.3f}", end="  ")
        print()
    print()

    print("=== TREND: cell_acc vs K ===\n")
    for d in ("easy", "medium", "hard"):
        print(f"  {d:6s}:", end=" ")
        for K in K_VALUES:
            if d not in results_by_k[K]: continue
            ca = results_by_k[K][d]["cell_acc"]
            print(f"K={K}→{ca:.3f}", end="  ")
        print()
    print()

    # Compare to v108 baseline (hard-coded from the prior K-sweep)
    print("=== COMPARISON TO V108 BASELINE (pos4 acc by K) ===\n")
    v108_pos4 = {
        "easy":   {2: 0.100, 4: 0.100, 8: 0.157},
        "medium": {2: 0.142, 4: 0.142, 8: 0.189},
        "hard":   {2: 0.191, 4: 0.138, 8: 0.132},
    }
    print(f"  {'diff':<8} | {'K=2 Δ':>8} {'K=4 Δ':>8} {'K=8 Δ':>8} | v108 trend     | v109pi trend")
    for d in ("easy", "medium", "hard"):
        if d not in results_by_k[K_VALUES[0]]: continue
        deltas = []
        seq = []
        for K in K_VALUES:
            v_p4 = results_by_k[K][d]["per_pos_acc"][4]
            v108_p4 = v108_pos4[d].get(K, 0.0)
            deltas.append(v_p4 - v108_p4)
            seq.append(v_p4)
        v108_trend = "rise" if v108_pos4[d][8] > v108_pos4[d][2] else "fall"
        if seq[-1] > seq[0]:
            v_trend = "rise"
        elif seq[-1] < seq[0]:
            v_trend = "fall"
        else:
            v_trend = "flat"
        dd = "  ".join(f"{x:+.3f}" for x in deltas)
        print(f"  {d:<8} | {dd}      | {v108_trend:13s} | {v_trend}")

    print()
    print("Interpretation:")
    print("  v109pi pos4 hard RISES → π-rotation extends v109's healthy dynamics")
    print("  v109pi pos4 hard FLAT  → rotation neutral on K-sweep (alternation does it all)")
    print("  v109pi pos4 hard FALLS → rotation tuned to K=K_max only; phase-grid mismatch at K<K_max")


if __name__ == "__main__":
    main()
