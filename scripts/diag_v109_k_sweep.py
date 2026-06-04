"""v109 K-sweep diagnostic.

Tests whether alternation flips v108's K-sweep anomaly.

v108 K-sweep (baseline, alternation absent):
  pos4 hard: K=2 → 0.191, K=4 → 0.138, K=8 → 0.132  (DECREASES with K)

v109 K-sweep prediction:
  If pos4 hard RISES monotonically → alternation makes the wave work
  If still flat/decreasing → alternation is decorative

Usage:
  CKPT=.cache/fg_v109_ckpts/v109_smoke_step500.safetensors \
    .venv/bin/python scripts/diag_v109_k_sweep.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("V109_TASK", "1")
os.environ.setdefault("V109_K_MAX", "8")
os.environ.setdefault("V109_N_DIGITS", "5")
os.environ.setdefault("V109_N_MAX", "16")
os.environ.setdefault("V109_F_MAX", "8")
os.environ.setdefault("V109_WAIST_DIM", "512")
os.environ.setdefault("V109_ALTERNATION", "1")
os.environ.setdefault("V109_HARD_BREATH_LEVEL", "0")

import numpy as np
from tinygrad import Device, Tensor, dtypes

from mycelium import Config, BreathingTransformer
from mycelium.loader import _load_state, load_breathing
from mycelium.factor_graph_v109 import (
    V109_K_MAX, V109_N_MAX, V109_F_MAX, V109_N_HEADS, V109_N_DIGITS,
    V109_WAIST_DIM, V109_ALTERNATION,
    V109_CODEBOOK_N, V109_IB_CENTROIDS,
    attach_fg_params_v109,
    _compile_jit_fg_eval_v109,
)
from mycelium.factor_graph_data_v107 import FactorGraphLoaderV107
from scripts.v108_factor_graph_train import cast_layers_fp32
from scripts.v109_factor_graph_train import load_ckpt_v109, evaluate_v109


def main():
    CKPT = os.environ.get(
        "CKPT", ".cache/fg_v109_ckpts/v109_smoke_step500.safetensors"
    )
    BATCH = int(os.environ.get("BATCH", "4"))
    EVAL_BATCHES = int(os.environ.get("EVAL_BATCHES", "20"))
    VAL_PATH = os.environ.get("VAL_PATH", ".cache/factor_graph_test.jsonl")
    SEED = int(os.environ.get("SEED", "42"))
    K_VALUES = [int(k) for k in os.environ.get("K_VALUES", "2,4,8").split(",")]

    print(f"=== v109 K-sweep diagnostic ===")
    print(f"  ckpt: {CKPT}")
    print(f"  val:  {VAL_PATH}")
    print(f"  alternation: {V109_ALTERNATION}")
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

    attach_fg_params_v109(
        model, hidden=cfg.hidden,
        n_max=V109_N_MAX, f_max=V109_F_MAX, k_max=V109_K_MAX,
        n_digits=V109_N_DIGITS, n_code=V109_CODEBOOK_N,
        ib_centroids_path=V109_IB_CENTROIDS,
        waist_dim=V109_WAIST_DIM,
    )
    Device[Device.DEFAULT].synchronize()

    print(f"loading ckpt: {CKPT}")
    load_ckpt_v109(model, CKPT)
    print()

    val_loader = FactorGraphLoaderV107(
        VAL_PATH, batch_size=BATCH,
        difficulty_filter=None, curriculum=False,
        n_max=V109_N_MAX, f_max=V109_F_MAX, k_max=V109_K_MAX,
        n_heads=V109_N_HEADS,
        seed=SEED + 2,
    )

    Tensor.training = False

    results_by_k = {}
    for K in K_VALUES:
        print(f"\n=== EVAL at K={K} ===")
        eval_fn = _compile_jit_fg_eval_v109(
            model, K=K, B=BATCH,
            n_max=V109_N_MAX, f_max=V109_F_MAX, n_digits=V109_N_DIGITS,
            alternation=V109_ALTERNATION,
        )
        t0 = time.time()
        results = evaluate_v109(
            model, val_loader, K=K, max_batches=EVAL_BATCHES, eval_fn=eval_fn,
            n_max=V109_N_MAX, f_max=V109_F_MAX, n_digits=V109_N_DIGITS,
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

    # =========================================================================
    # Compare across K
    # =========================================================================
    print("\n=== K-SWEEP COMPARISON ===")
    print()
    for d in ("easy", "medium", "hard"):
        print(f"--- val[{d}] ---")
        print(f"  {'K':>3} | {'pos0':>6} {'pos1':>6} {'pos2':>6} {'pos3':>6} {'pos4':>6} | digit  cell")
        for K in K_VALUES:
            if d not in results_by_k[K]: continue
            v = results_by_k[K][d]
            pp = "  ".join(f"{p:.3f}" for p in v["per_pos_acc"])
            print(f"  {K:>3} | {pp} | {v['digit_acc']:.3f} {v['cell_acc']:.3f}")
        print()

    print("=== TREND: pos4 (ones digit) acc vs K ===")
    print()
    for d in ("easy", "medium", "hard"):
        print(f"  {d:6s}:", end=" ")
        for K in K_VALUES:
            if d not in results_by_k[K]: continue
            pos4 = results_by_k[K][d]["per_pos_acc"][4]
            print(f"K={K}→{pos4:.3f}", end="  ")
        print()
    print()

    # Compare to v108 baseline (hard-coded from the prior K-sweep)
    print("=== COMPARISON TO V108 BASELINE (pos4 acc by K) ===")
    print()
    v108_pos4 = {
        "easy":   {2: 0.100, 4: 0.100, 8: 0.157},
        "medium": {2: 0.142, 4: 0.142, 8: 0.189},
        "hard":   {2: 0.191, 4: 0.138, 8: 0.132},
    }
    print(f"  {'diff':<8} | {'K=2 Δ':>8} {'K=4 Δ':>8} {'K=8 Δ':>8} | v108 trend     | v109 trend")
    for d in ("easy", "medium", "hard"):
        if d not in results_by_k[K_VALUES[0]]: continue
        deltas = []
        v109_pos4_seq = []
        for K in K_VALUES:
            v109_p4 = results_by_k[K][d]["per_pos_acc"][4]
            v108_p4 = v108_pos4[d].get(K, 0.0)
            deltas.append(v109_p4 - v108_p4)
            v109_pos4_seq.append(v109_p4)
        v108_trend = "rise" if v108_pos4[d][8] > v108_pos4[d][2] else "fall"
        v109_trend = "rise" if v109_pos4_seq[-1] > v109_pos4_seq[0] else "fall"
        if v109_pos4_seq[-1] == v109_pos4_seq[0]:
            v109_trend = "flat"
        dd = "  ".join(f"{x:+.3f}" for x in deltas)
        print(f"  {d:<8} | {dd}      | {v108_trend:13s} | {v109_trend}")

    print()
    print("Interpretation:")
    print("  v109 pos4 hard RISES with K → alternation hypothesis confirmed")
    print("  v109 pos4 hard FLAT with K  → waist helps; alternation may be neutral")
    print("  v109 pos4 hard FALLS with K → same as v108; alternation didn't change dynamics")


if __name__ == "__main__":
    main()
