"""Pythia-410M L0-L3 baseline eval (no fine-tuning, no architectural extras).

Loads the model from Pythia weights only, all architectural additions OFF,
n_loops=1. Tests what raw 4-layer Pythia-init can do on our curricula.

This is the "no architecture" baseline for measuring how much of our v24c
gains come from architecture vs raw transformer + Pythia weights.

Usage:
    DEV=PCI+AMD LEVEL=L4_MIXED LOOPS=1,4,8 \\
        python scripts/eval_pythia_baseline.py

Env vars:
    LEVEL  (L4_MIXED)  — curriculum to evaluate
    LOOPS  (1,4,8)     — n_loops sweep
    NUM_EVAL  (100)    — held-out problems
    SEED      (42)     — eval split seed
"""
import os
import sys
import time

# Make sure all features are off BEFORE importing breathing
os.environ["NOTEBOOK_V24"] = "0"
os.environ["NOTEBOOK_DUAL"] = "0"
os.environ["PER_HEAD_PITCH"] = "0"
os.environ["CROSS_BREATH_HANDOFF"] = "0"
os.environ["BREATH_TIME_EMBED"] = "0"
os.environ["CONSTANT_RADIUS"] = "0"
os.environ["SINE_TEMP"] = "0"
os.environ["CROSS_CYCLE_NOTEBOOK"] = "0"
os.environ["PER_BREATH_TEMP"] = "0"
os.environ["BREATH_NORM_OSC"] = "0"
os.environ["NOTEBOOK_STATE_INIT_SCALE"] = "0.0"
os.environ.setdefault("ABLATE_BREATH_ROTATION", "0")  # keep base π-cycled RoPE
os.environ.setdefault("SPACE_DIGITS", "1")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from tinygrad import Tensor, Device, dtypes
from tinygrad.helpers import getenv

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
    level = getenv("LEVEL", "L4_MIXED")
    loops_str = getenv("LOOPS", "1,4,8")
    n_eval = getenv("NUM_EVAL", 100)
    space_digits = bool(getenv("SPACE_DIGITS", 1))
    seed = getenv("SEED", 42)
    eval_loops_list = [int(x) for x in loops_str.split(",")]
    num_problems = getenv("NUM_PROBLEMS", 20000)
    fixed_len_map = {"ARITH": 32, "L3": 64, "L4": 96, "L4_MIXED": 96, "L4_BORROW": 96, "L4.5": 160}
    fixed_len = fixed_len_map.get(level, 96)

    cfg = Config()
    print(f"=== Pythia-410M L0-L3 baseline on {level} ===")
    print(f"loops={eval_loops_list}  num_eval={n_eval}  space_digits={space_digits}  fixed_len={fixed_len}")
    print(f"all architectural extras OFF (no notebook, no per-head pitch, no handoff, etc.)")
    print()

    print(f"generating {level} problems...")
    t0 = time.perf_counter()
    all_examples = generate_math(level, num_problems, seed=seed, digit_spacing=space_digits)
    _, eval_examples = split_train_eval(all_examples, n_eval=n_eval, seed=seed)
    print(f"  eval set: {len(eval_examples)} examples  ({time.perf_counter()-t0:.1f}s)")

    print("\nloading Pythia-410M L0-L3 (no ckpt, no fine-tuning)...")
    t0 = time.perf_counter()
    sd = _load_state()
    model = load_breathing(cfg, sd=sd)
    cast_fp32(model)
    del sd
    Device[Device.DEFAULT].synchronize()
    print(f"  loaded in {time.perf_counter()-t0:.1f}s")

    tok = load_tokenizer()
    eval_batch = getenv("EVAL_BATCH", 64)
    cache_len = fixed_len + 40

    print(f"\n=== Running eval ===")
    for nl in eval_loops_list:
        t0 = time.perf_counter()
        acc, rows = accuracy_at_loops_multi(model, tok, eval_examples,
                                             n_loops=[nl, 1],
                                             batch_size=eval_batch,
                                             cache_max_len=cache_len)
        gt = time.perf_counter() - t0
        print(f"  acc @ A={nl} C=1: {acc*100:.1f}%  ({gt:.1f}s)")
        # Show one example
        if rows:
            ex, parsed, gen = rows[0]
            print(f"    Q: {ex.problem}")
            print(f"    gen: {gen.strip()[:200]!r}")
            print(f"    parsed: {parsed}, gold: {ex.answer}, {'OK' if parsed == ex.answer else 'WRONG'}")
        print()


if __name__ == "__main__":
    main()
