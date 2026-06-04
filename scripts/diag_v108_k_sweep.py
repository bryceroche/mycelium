"""v108 K-sweep diagnostic.

Tests the Monte Carlo sampling theory predicition: does per-position accuracy
(especially pos4, the ones digit) rise monotonically with K?

Procedure:
  1. Load v108 step 500 ckpt.
  2. Compile eval JITs at K=2, K=4, K=8.
  3. Run each on the same val set.
  4. Compare per-position accuracy across K.

Interpretation:
  Monotonic pos4 rise with K → MC sampling theory has legs (K=16+ motivated)
  Flat pos4 across K           → capacity ceiling (MC framing decorative)

The model was trained with K_max=8. Running at K=2, K=4 uses the first 2 or 4
of the 8 trained breath_embed positions. No retraining required.

Usage:
  .venv/bin/python scripts/diag_v108_k_sweep.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("V108_TASK", "1")
os.environ.setdefault("V108_K_MAX", "8")
os.environ.setdefault("V108_N_DIGITS", "5")
os.environ.setdefault("V108_N_MAX", "16")
os.environ.setdefault("V108_F_MAX", "8")
os.environ.setdefault("V108_HARD_BREATH_LEVEL", "0")

import numpy as np
from tinygrad import Device, Tensor, dtypes

from mycelium import Config, BreathingTransformer
from mycelium.loader import _load_state, load_breathing
from mycelium.factor_graph_v108 import (
    V108_K_MAX, V108_N_MAX, V108_F_MAX, V108_N_HEADS, V108_N_DIGITS,
    V108_CODEBOOK_N, V108_IB_CENTROIDS,
    attach_fg_params_v108,
    _compile_jit_fg_eval_v108,
)
from mycelium.factor_graph_data_v107 import FactorGraphLoaderV107
from scripts.v108_factor_graph_train import (
    cast_layers_fp32, load_ckpt_v108, evaluate_v108,
)


def main():
    CKPT = os.environ.get(
        "CKPT", ".cache/fg_v108_ckpts/v108_smoke_step500.safetensors"
    )
    BATCH = int(os.environ.get("BATCH", "4"))
    EVAL_BATCHES = int(os.environ.get("EVAL_BATCHES", "20"))
    VAL_PATH = os.environ.get("VAL_PATH", ".cache/factor_graph_test.jsonl")
    SEED = int(os.environ.get("SEED", "42"))
    K_VALUES = [int(k) for k in os.environ.get("K_VALUES", "2,4,8").split(",")]

    print(f"=== v108 K-sweep diagnostic ===")
    print(f"  ckpt: {CKPT}")
    print(f"  val:  {VAL_PATH}")
    print(f"  batch={BATCH}  batches={EVAL_BATCHES}")
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

    # Attach with K_MAX so breath_embed has full 8 positions
    attach_fg_params_v108(
        model, hidden=cfg.hidden,
        n_max=V108_N_MAX, f_max=V108_F_MAX, k_max=V108_K_MAX,
        n_digits=V108_N_DIGITS, n_code=V108_CODEBOOK_N,
        ib_centroids_path=V108_IB_CENTROIDS,
    )
    Device[Device.DEFAULT].synchronize()

    print(f"loading ckpt: {CKPT}")
    load_ckpt_v108(model, CKPT)
    print()

    val_loader = FactorGraphLoaderV107(
        VAL_PATH, batch_size=BATCH,
        difficulty_filter=None, curriculum=False,
        n_max=V108_N_MAX, f_max=V108_F_MAX, k_max=V108_K_MAX,
        n_heads=V108_N_HEADS,
        seed=SEED + 2,
    )

    Tensor.training = False

    # Compile one eval JIT per K and run
    results_by_k = {}
    for K in K_VALUES:
        print(f"\n=== EVAL at K={K} ===")
        eval_fn = _compile_jit_fg_eval_v108(
            model, K=K, B=BATCH,
            n_max=V108_N_MAX, f_max=V108_F_MAX, n_digits=V108_N_DIGITS,
        )
        t0 = time.time()
        results = evaluate_v108(
            model, val_loader, K=K, max_batches=EVAL_BATCHES, eval_fn=eval_fn,
            n_max=V108_N_MAX, f_max=V108_F_MAX, n_digits=V108_N_DIGITS,
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
    # Compare per-position acc across K
    # =========================================================================
    print("\n=== K-SWEEP COMPARISON: per-position acc by K ===")
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

    # Trend analysis
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

    print("Interpretation:")
    print("  Monotonic rise of pos4 with K → MC sampling theory applies")
    print("  Flat pos4 across K            → capacity is the ceiling")


if __name__ == "__main__":
    main()
