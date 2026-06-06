"""v109pi OOD eval — 5-digit numbers [10000, 99999], unseen at train time.

Training distribution: operands in [0, 9999] (loguniform).
OOD test set: factor graphs with gold values in [10000, 99999].

The 5-level tree codebook has capacity for 5-digit numbers (codebook depth 5),
but the model has never seen a non-zero leading digit (pos0) during training.
This tests whether the digit-level supervision composes geometrically:
can the model produce pos0 ∈ {1..9} when train always had pos0 = 0?

Reports per-position accuracy and cell accuracy at K=8 (the trained K).

Usage:
  CKPT=.cache/fg_v109pi_ckpts/v109pi_cont9_step500.safetensors \
    .venv/bin/python scripts/diag_v109pi_ood.py
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
    EVAL_BATCHES = int(os.environ.get("EVAL_BATCHES", "30"))
    OOD_PATH = os.environ.get("OOD_PATH", ".cache/factor_graph_test_ood_5digit.jsonl")
    IND_PATH = os.environ.get("IND_PATH", ".cache/factor_graph_test.jsonl")
    SEED = int(os.environ.get("SEED", "42"))
    K = int(os.environ.get("K", str(V109PI_K_MAX)))
    PHASE_SCALE = float(os.environ.get("V109PI_PHASE_SCALE", "1.0"))

    print(f"=== v109pi OOD diagnostic ===")
    print(f"  ckpt:    {CKPT}")
    print(f"  IND val: {IND_PATH}")
    print(f"  OOD val: {OOD_PATH}")
    print(f"  K:       {K}")
    print(f"  phase:   {PHASE_SCALE}")
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

    Tensor.training = False

    eval_fn = _compile_jit_fg_eval_v109pi(
        model, K=K, B=BATCH,
        n_max=V109PI_N_MAX, f_max=V109PI_F_MAX, n_digits=V109PI_N_DIGITS,
        alternation=V109PI_ALTERNATION, phase_scale=PHASE_SCALE,
    )

    for label, path in [("IND", IND_PATH), ("OOD", OOD_PATH)]:
        print(f"\n=== {label} eval ({path}) ===")
        loader = FactorGraphLoaderV107(
            path, batch_size=BATCH,
            difficulty_filter=None, curriculum=False,
            n_max=V109PI_N_MAX, f_max=V109PI_F_MAX, k_max=V109PI_K_MAX,
            n_heads=V109PI_N_HEADS,
            seed=SEED + 2,
        )
        t0 = time.time()
        results = evaluate_v109pi(
            model, loader, K=K, max_batches=EVAL_BATCHES, eval_fn=eval_fn,
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

    print("\nInterpretation:")
    print("  OOD pos0 acc > chance(0.1) → leading-digit composition works")
    print("  OOD pos4 acc ~ IND pos4    → ones-digit composes (the easy case)")
    print("  OOD cell_acc > 0           → at least some 5-digit answers right")


if __name__ == "__main__":
    main()
