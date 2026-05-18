"""Eval a checkpoint on any curriculum level. CLI args:
  CKPT (env var): path to safetensors
  LEVEL (env var): L3, L4, L4_BORROW, L4_MIXED, L4.5, ARITH, ...
Uses the v24c-era env knobs set by the caller (PER_HEAD_PITCH=1, NOTEBOOK_V24=1, etc.).
"""
import os
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from tinygrad import Tensor, Device, dtypes
from tinygrad.nn.state import safe_load

from mycelium import Config
from mycelium.loader import _load_state, load_breathing
from mycelium.data import load_tokenizer
from mycelium.l3_data import generate_math, split_train_eval
from mycelium.l3_training import accuracy_at_loops_multi

DEFAULT_FIXED_LEN = {
    "ARITH": 32, "ARITH_HARD": 32, "ARITH_MIXED": 32, "ARITH_BORROW": 32,
    "L3": 64, "L4": 96, "L4_BORROW": 96, "L4_MIXED": 96, "L4.5": 160,
}


def cast_model_fp32(model):
    def _cast(obj, attr):
        t = getattr(obj, attr)
        if t.dtype == dtypes.half:
            setattr(obj, attr, t.cast(dtypes.float).contiguous().realize())
    _cast(model.embed, "weight")
    _cast(model, "embed_out")
    sw = model.block.shared
    for a in ("wv", "bv", "wo", "bo", "w_out", "b_out"):
        _cast(sw, a)
    for layer in model.block.layers:
        for a in ("wq", "bq", "wk", "bk", "w_in", "b_in"):
            _cast(layer, a)


def main():
    cfg = Config()
    LEVEL = os.environ.get("LEVEL", "L4_MIXED")
    CKPT = os.environ.get("CKPT", "")
    NUM_EVAL = int(os.environ.get("NUM_EVAL", "100"))
    EVAL_LOOPS = [int(x) for x in os.environ.get("EVAL_LOOPS", "1,4,8").split(",")]
    EVAL_BATCH = int(os.environ.get("EVAL_BATCH", "64"))
    fixed_len = int(os.environ.get("FIXED_LEN", DEFAULT_FIXED_LEN.get(LEVEL, 96)))
    cache_max_len = int(os.environ.get("EVAL_CACHE_LEN", "0")) or (fixed_len + 40)

    print(f"=== eval ckpt on {LEVEL} ===")
    print(f"  ckpt: {CKPT}")
    print(f"  num_eval: {NUM_EVAL}  fixed_len: {fixed_len}  cache_max_len: {cache_max_len}")
    print(f"  eval_loops: {EVAL_LOOPS}  batch: {EVAL_BATCH}")
    print(f"  device: {Device.DEFAULT}")

    print(f"\ngenerating {LEVEL} eval set (seed=42)...")
    all_examples = generate_math(LEVEL, 20000, seed=42, digit_spacing=True)
    _, eval_examples = split_train_eval(all_examples, n_eval=NUM_EVAL, seed=42)
    print(f"  eval count: {len(eval_examples)}")
    if eval_examples:
        print(f"  sample: {eval_examples[0].problem!r}")
    tok = load_tokenizer()

    print("\nloading Pythia-410M -> breathing transformer...")
    sd = _load_state()
    model = load_breathing(cfg, sd=sd)
    cast_model_fp32(model)
    Device[Device.DEFAULT].synchronize()
    del sd

    if not CKPT:
        print("WARNING: CKPT not set — evaluating Pythia-only init (random baseline).")
    else:
        print(f"\nloading checkpoint: {CKPT}")
        ckpt_sd = safe_load(CKPT)
        info = model.load_state_dict(ckpt_sd, strict=False)
        print(f"  missing keys: {len(info['missing'])} ({info['missing'][:3]}...)")
        print(f"  unexpected keys: {len(info['unexpected'])}")
        del ckpt_sd

    Tensor.training = False
    print(f"\n=== accuracy on {LEVEL} ({NUM_EVAL} held-out) ===")
    for nl in EVAL_LOOPS:
        t0 = time.perf_counter()
        acc, rows = accuracy_at_loops_multi(
            model, tok, eval_examples,
            n_loops=[nl, 1],
            batch_size=EVAL_BATCH,
            cache_max_len=cache_max_len,
        )
        dt = time.perf_counter() - t0
        print(f"  acc @ A={nl} C=1: {acc*100:.1f}%  ({dt:.1f}s)")
        for ex, parsed, gen in rows[:1]:
            print(f"    Q: {ex.problem!r}")
            print(f"    gen: {gen.strip()!r}")
            print(f"    parsed: {parsed}, gold: {ex.answer}, {'OK' if parsed == ex.answer else 'WRONG'}")
        print()


if __name__ == "__main__":
    main()
