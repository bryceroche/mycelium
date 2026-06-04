"""IB codebook ablation diagnostic for v108.

Tests whether the IB semantic codebook (32 centroids × 1024d, gated by
delta_gate_quant) is acting as a low-pass filter that suppresses low-order
digit prediction.

Procedure:
  1. Load v108 step 500 ckpt.
  2. Run evaluation normally — get per-position accuracy.
  3. Zero out delta_gate_quant (disable IB codebook mixing).
  4. Re-run evaluation — get per-position accuracy.
  5. Compare. If pos3/pos4 jump with IB disabled, the codebook was the smoother.

Usage:
  .venv/bin/python scripts/diag_v108_ib_ablation.py
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
from tinygrad.nn.state import safe_load

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
    CKPT  = os.environ.get(
        "CKPT", ".cache/fg_v108_ckpts/v108_smoke_step500.safetensors"
    )
    BATCH = int(os.environ.get("BATCH", "4"))
    EVAL_BATCHES = int(os.environ.get("EVAL_BATCHES", "20"))
    VAL_PATH = os.environ.get("VAL_PATH", ".cache/factor_graph_test.jsonl")
    SEED = int(os.environ.get("SEED", "42"))
    K = V108_K_MAX

    print(f"=== v108 IB codebook ablation diagnostic ===")
    print(f"  ckpt: {CKPT}")
    print(f"  val:  {VAL_PATH}")
    print(f"  batch={BATCH} batches={EVAL_BATCHES}")
    print()

    np.random.seed(SEED)
    Tensor.manual_seed(SEED)

    cfg = Config()
    print("loading Pythia-410M...")
    sd = _load_state()
    model = load_breathing(cfg, sd=sd)
    del sd
    cast_layers_fp32(model)

    attach_fg_params_v108(
        model, hidden=cfg.hidden,
        n_max=V108_N_MAX, f_max=V108_F_MAX, k_max=K,
        n_digits=V108_N_DIGITS, n_code=V108_CODEBOOK_N,
        ib_centroids_path=V108_IB_CENTROIDS,
    )
    Device[Device.DEFAULT].synchronize()

    print(f"loading ckpt: {CKPT}")
    load_ckpt_v108(model, CKPT)
    print()

    # Inspect what delta_gate_quant looks like in the trained ckpt
    dgq = model.fg_v107_delta_gate_quant.numpy()
    print(f"  delta_gate_quant trained values: {' '.join(f'{v:.4f}' for v in dgq)}")
    print(f"  mean={dgq.mean():.4f}  max={dgq.max():.4f}  min={dgq.min():.4f}")
    print()

    val_loader = FactorGraphLoaderV107(
        VAL_PATH, batch_size=BATCH,
        difficulty_filter=None, curriculum=False,
        n_max=V108_N_MAX, f_max=V108_F_MAX, k_max=K, n_heads=V108_N_HEADS,
        seed=SEED + 2,
    )

    Tensor.training = False
    eval_fn = _compile_jit_fg_eval_v108(
        model, K=K, B=BATCH,
        n_max=V108_N_MAX, f_max=V108_F_MAX, n_digits=V108_N_DIGITS,
    )

    # =========================================================================
    # Run 1: AS-IS (IB codebook engaged)
    # =========================================================================
    print("=== EVAL 1: AS-IS (IB codebook engaged with trained gate) ===")
    t0 = time.time()
    results_normal = evaluate_v108(
        model, val_loader, K=K, max_batches=EVAL_BATCHES, eval_fn=eval_fn,
        n_max=V108_N_MAX, f_max=V108_F_MAX, n_digits=V108_N_DIGITS,
    )
    dt = time.time() - t0
    print(f"  ({dt:.1f}s)")
    for d in ("easy", "medium", "hard"):
        if d not in results_normal: continue
        v = results_normal[d]
        pp = " ".join(f"{p:.2f}" for p in v["per_pos_acc"])
        print(f"  val[{d:6s}]: cell={v['cell_acc']:.3f} q={v['query_acc']:.3f} "
              f"digit={v['digit_acc']:.3f} per_pos=[{pp}] n={v['n_puzzles']}")
    print()

    # =========================================================================
    # Run 2: IB DISABLED (delta_gate_quant = 0)
    # =========================================================================
    print("=== EVAL 2: IB CODEBOOK DISABLED (delta_gate_quant = 0) ===")
    # Force gate to zero — the IB recon won't contribute to h_quant
    zero_gate = Tensor.zeros((K,), dtype=dtypes.float).contiguous().realize()
    model.fg_v107_delta_gate_quant.assign(zero_gate).realize()
    gate_after = model.fg_v107_delta_gate_quant.numpy()
    print(f"  delta_gate_quant after zero: {' '.join(f'{v:.4f}' for v in gate_after)}")
    print()

    # Need to recompile JIT because the model param changed... or do we?
    # The JIT captures the Tensor reference. If we .assign() the underlying
    # data, the cached JIT should use the new values. Let's verify.
    t0 = time.time()
    results_ablated = evaluate_v108(
        model, val_loader, K=K, max_batches=EVAL_BATCHES, eval_fn=eval_fn,
        n_max=V108_N_MAX, f_max=V108_F_MAX, n_digits=V108_N_DIGITS,
    )
    dt = time.time() - t0
    print(f"  ({dt:.1f}s)")
    for d in ("easy", "medium", "hard"):
        if d not in results_ablated: continue
        v = results_ablated[d]
        pp = " ".join(f"{p:.2f}" for p in v["per_pos_acc"])
        print(f"  val[{d:6s}]: cell={v['cell_acc']:.3f} q={v['query_acc']:.3f} "
              f"digit={v['digit_acc']:.3f} per_pos=[{pp}] n={v['n_puzzles']}")
    print()

    # =========================================================================
    # Compare
    # =========================================================================
    print("=== COMPARISON: per-position acc with IB ON vs OFF ===")
    print(f"  {'diff':<8} {'pos0':>6} {'pos1':>6} {'pos2':>6} {'pos3':>6} {'pos4':>6}  | digit  cell")
    for d in ("easy", "medium", "hard"):
        if d not in results_normal or d not in results_ablated: continue
        n = results_normal[d]
        a = results_ablated[d]
        deltas = [a["per_pos_acc"][p] - n["per_pos_acc"][p] for p in range(V108_N_DIGITS)]
        dd = " ".join(f"{x:+.3f}" for x in deltas)
        d_digit = a["digit_acc"] - n["digit_acc"]
        d_cell  = a["cell_acc"]  - n["cell_acc"]
        print(f"  {d:<8} {dd} | {d_digit:+.3f} {d_cell:+.3f}")
    print()

    print("Interpretation:")
    print("  If pos3/pos4 deltas are POSITIVE (>+0.05): IB codebook WAS smoothing low-order digits.")
    print("  If pos3/pos4 deltas are near zero: IB codebook is not the low-pass cause.")
    print("  If pos3/pos4 deltas are NEGATIVE: IB was actually helping (e.g., providing constraint signal).")


if __name__ == "__main__":
    main()
