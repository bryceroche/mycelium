"""One-shot diagnostic: load v24c step 500 ckpt, run accuracy eval, no training.

If acc != 96/94/91 → the inference path itself has a bug (introduced since v24c
was trained), and we've been chasing the wrong hypothesis. If acc == 96/94/91
→ training (even minimal) is somehow breaking the model.
"""
import os
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# v24c env knobs (read at import time by mycelium.breathing) — must be set
# BEFORE importing mycelium.
os.environ.setdefault("PER_HEAD_PITCH", "1")
os.environ.setdefault("SINE_TEMP", "1")
os.environ.setdefault("SINE_TEMP_MAX", "2.0")
os.environ.setdefault("SINE_TEMP_MIN", "0.7")
os.environ.setdefault("CONSTANT_RADIUS", "1")
os.environ.setdefault("BREATH_TIME_EMBED", "1")
os.environ.setdefault("BREATH_TIME_INIT_SCALE", "0.0")
os.environ.setdefault("CROSS_BREATH_HANDOFF", "1")
os.environ.setdefault("ABLATE_BREATH_ROTATION", "1")
os.environ.setdefault("NOTEBOOK_V24", "1")
os.environ.setdefault("NOTEBOOK_DUAL", "1")
os.environ.setdefault("NOTEBOOK_POOL_MODE", "attn")
os.environ.setdefault("NOTEBOOK_INIT_SCALE", "0.02")
os.environ.setdefault("DEV", "PCI+AMD")

import numpy as np
from tinygrad import Tensor, Device, dtypes
from tinygrad.nn.state import safe_load

from mycelium import Config
from mycelium.loader import _load_state, load_breathing
from mycelium.data import load_tokenizer
from mycelium.l3_data import generate_math, split_train_eval
from mycelium.l3_training import accuracy_at_loops_multi


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
    print(f"device={Device.DEFAULT}")

    print("\ngenerating L4_MIXED eval set (same seed=42 as training)...")
    all_examples = generate_math("L4_MIXED", 20000, seed=42, digit_spacing=True)
    _, eval_examples = split_train_eval(all_examples, n_eval=100, seed=42)
    print(f"  eval count: {len(eval_examples)}")
    tok = load_tokenizer()

    print("\nloading Pythia-410M -> breathing transformer...")
    sd = _load_state()
    model = load_breathing(cfg, sd=sd)
    cast_model_fp32(model)
    Device[Device.DEFAULT].synchronize()
    del sd

    ckpt_path = ".cache/l4_mixed_ckpts/l4_mixed_v24c_dual_notebook_step500.safetensors"
    print(f"\nloading v24c step-500 ckpt: {ckpt_path}")
    ckpt_sd = safe_load(ckpt_path)
    info = model.load_state_dict(ckpt_sd, strict=False)
    print(f"  missing keys: {len(info['missing'])} ({info['missing'][:3]}...)")
    print(f"  unexpected keys: {len(info['unexpected'])}")
    del ckpt_sd

    Tensor.training = False

    print("\n=== accuracy eval (the same one the training loop runs) ===")
    for nl in [1, 4, 8]:
        t0 = time.perf_counter()
        acc, rows = accuracy_at_loops_multi(
            model, tok, eval_examples,
            n_loops=[nl, 1],     # [phase_A, phase_C] — matches v24c's [nl, 1]
            batch_size=64,
            cache_max_len=136,   # FIXED_LEN(96) + 40
        )
        dt = time.perf_counter() - t0
        print(f"  acc @ A={nl} C=1: {acc*100:.1f}%  ({dt:.1f}s)")
        # Show 2 sample generations
        for ex, parsed, gen in rows[:2]:
            print(f"    Q: {ex.problem!r}")
            print(f"    gen: {gen.strip()!r}")
            print(f"    parsed: {parsed}, gold: {ex.answer}, {'OK' if parsed == ex.answer else 'WRONG'}")
        print()

    print("=== if v24c ckpt gives 96/94/91 → training has been breaking the model")
    print("=== if v24c ckpt gives 0% → inference path itself is broken")


if __name__ == "__main__":
    main()
