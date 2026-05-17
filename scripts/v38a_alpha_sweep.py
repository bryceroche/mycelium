"""Stage a: BFIELD_ALPHA inference sweep on v38 step 1500 ckpt.

Loads the converged v38 ARITH champion (61/59/58 at α=1) and runs accuracy eval
at multiple α values. α multiplies the B-field bottleneck's decompressed
contribution at the residual:

    final = x + α · decompressed
          = (1-α)·uncond + α·cond       [uncond=x, cond=x+decompressed]

Tests whether CFG-style residual amplification on the B-field gives a free
inference win — analogous to v6's +10.8% from logit-CFG.

Usage:
    DEV=PCI+AMD BFIELD_WAIST=256 \\
      /home/bryce/mycelium/.venv/bin/python scripts/v38a_alpha_sweep.py
"""
import os
import sys
import time

os.environ.setdefault("DEV", "PCI+AMD")
os.environ.setdefault("BFIELD_WAIST", "256")
# Match v38's training-time geometry exactly
os.environ.setdefault("PER_HEAD_PITCH", "1")
os.environ.setdefault("LOOKUP_VALUE_INJECT", "1")
os.environ.setdefault("LOOKUP_VALUES_INIT_PATH", ".cache/ib_centroids_per_layer/centroids_n16.npy")
os.environ.setdefault("LOOKUP_TEMP", "20")
os.environ.setdefault("LOOKUP_VALUE_SCALE", "1.0")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from tinygrad import Tensor, Device, dtypes
from tinygrad.helpers import getenv
from tinygrad.nn.state import safe_load

from mycelium import Config
from mycelium.loader import _load_state, load_breathing
from mycelium.data import load_tokenizer
from mycelium.l3_data import generate_math, split_train_eval
from mycelium.l3_training import accuracy_at_loops_multi


def cast_fp32(model):
    def _c(o, a):
        t = getattr(o, a)
        if t.dtype == dtypes.half:
            setattr(o, a, t.cast(dtypes.float).contiguous().realize())
    _c(model.embed, "weight")
    _c(model, "embed_out")
    sw = model.block.shared
    for a in ("wv", "bv", "wo", "bo", "w_out", "b_out"):
        _c(sw, a)
    for layer in model.block.layers:
        for a in ("wq", "bq", "wk", "bk", "w_in", "b_in"):
            _c(layer, a)


def main():
    ckpt = getenv("CKPT", "/home/bryce/mycelium/.cache/arith_ckpts/v38_bfield_w256_step1500.safetensors")
    alphas_str = getenv("ALPHAS", "0,0.5,1.0,1.5,2.0,2.5,3.0,4.0,5.0")
    alphas = [float(a) for a in alphas_str.split(",")]
    eval_loops_str = getenv("EVAL_LOOPS", "1,4,8")
    eval_loops = [int(l) for l in eval_loops_str.split(",")]
    n_eval = getenv("NUM_EVAL", 100)
    eval_batch = getenv("EVAL_BATCH", 64)
    fixed_len = getenv("FIXED_LEN", 32)
    seed = getenv("SEED", 42)

    cfg = Config()
    print(f"=== Stage a: BFIELD_ALPHA inference sweep on v38 ===")
    print(f"  ckpt: {os.path.basename(ckpt)}")
    print(f"  α values: {alphas}")
    print(f"  loops: {eval_loops}  N_eval={n_eval}  cache_len={fixed_len + 40}")
    print()

    print("loading Pythia → breathing transformer...")
    t0 = time.perf_counter()
    sd = _load_state()
    model = load_breathing(cfg, sd=sd)
    cast_fp32(model)
    del sd
    Device[Device.DEFAULT].synchronize()
    print(f"  loaded base ({time.perf_counter()-t0:.1f}s)")

    print(f"\nloading v38 ckpt: {os.path.basename(ckpt)}")
    state = safe_load(ckpt)
    info = model.load_state_dict(state, strict=False)
    Device[Device.DEFAULT].synchronize()
    if info.get("missing"):
        print(f"  missing keys (expected — old ckpt): {info['missing'][:3]}")

    tok = load_tokenizer()
    print(f"\ngenerating {n_eval} ARITH eval problems...")
    all_examples = generate_math("ARITH", n_eval + 50, seed=seed, digit_spacing=True)
    _, eval_examples = split_train_eval(all_examples, n_eval=n_eval, seed=seed)
    print(f"  eval set: {len(eval_examples)}")

    # Sweep
    results = {}
    print(f"\n=== α sweep ===")
    for alpha in alphas:
        model.block.bfield_alpha.assign(Tensor.ones((1,), dtype=dtypes.float) * float(alpha)).realize()
        row = {}
        print(f"\nα = {alpha:.2f}")
        for nl in eval_loops:
            t1 = time.perf_counter()
            acc, _ = accuracy_at_loops_multi(model, tok, eval_examples, n_loops=nl,
                                              batch_size=eval_batch, cache_max_len=fixed_len + 40)
            t = time.perf_counter() - t1
            row[nl] = acc
            print(f"  acc @ A={nl}: {acc*100:.1f}%  ({t:.1f}s)")
        results[alpha] = row

    # Summary
    print(f"\n=== SUMMARY ===")
    header = "  α    " + "  ".join(f"A={l}".rjust(7) for l in eval_loops) + "  mean"
    print(header)
    print("  " + "-" * (len(header) - 2))
    best_mean = -1.0
    best_alpha = None
    for alpha in alphas:
        row = results[alpha]
        accs = [row[l] for l in eval_loops]
        mean_acc = sum(accs) / len(accs)
        line = f"  {alpha:.2f}  " + "  ".join(f"{a*100:5.1f}%".rjust(7) for a in accs) + f"  {mean_acc*100:.1f}%"
        if alpha == 1.0:
            line += "  (baseline)"
        print(line)
        if mean_acc > best_mean:
            best_mean = mean_acc
            best_alpha = alpha

    base = results.get(1.0, {})
    base_mean = sum(base[l] for l in eval_loops) / len(eval_loops) if base else 0.0
    print(f"\n  best α = {best_alpha:.2f}  mean acc = {best_mean*100:.1f}%")
    if base:
        print(f"  baseline (α=1) mean acc = {base_mean*100:.1f}%")
        print(f"  CFG lift over baseline = {(best_mean - base_mean)*100:+.1f} pts")


if __name__ == "__main__":
    main()
