"""Oracle ensemble test for two phase shifts.

Runs accuracy_at_loops_multi at shift=0 and shift=π/4 (where the per-shift
sweep showed both still work at 90%+). Compares per-example correctness:

  acc_0:    correct at shift=0 only
  acc_pi4:  correct at shift=π/4 only
  oracle:   correct at either shift (the upper bound any ensemble can reach)
  both:     correct at both shifts (the safe overlap)

If oracle ≫ max(acc_0, acc_pi4) → shifts get DIFFERENT examples right.
Ensembling has real upside. Run a proper logit-averaging eval next.

If oracle ≈ max(acc_0, acc_pi4) → shifts give essentially the same answers.
Ensembling won't help; quadrature views aren't independent.

Env vars: same as eval_phase_shift_sweep.py.
"""
import os
import sys
import time
import math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from tinygrad import Tensor, Device, dtypes
from tinygrad.nn.state import safe_load

from mycelium import Config
from mycelium.loader import _load_state, load_breathing
from mycelium.data import load_tokenizer
from mycelium.l3_data import generate_math, split_train_eval, parse_int_answer
from mycelium.l3_training import accuracy_at_loops_multi

DEFAULT_FIXED_LEN = {"L4_MIXED": 96, "L4.5": 160, "L4": 96, "L3": 64}


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


def set_phase_shift(model, original_pitch_np, shift_rad: float):
    cfg = model.cfg
    shifted = original_pitch_np + shift_rad
    cos_np = np.cos(shifted).reshape(cfg.n_phases, 1, cfg.n_heads, 1, 1).astype(np.float32)
    sin_np = np.sin(shifted).reshape(cfg.n_phases, 1, cfg.n_heads, 1, 1).astype(np.float32)
    model.block.per_head_pitch_cos.assign(Tensor(cos_np, dtype=dtypes.float).contiguous())
    model.block.per_head_pitch_sin.assign(Tensor(sin_np, dtype=dtypes.float).contiguous())
    model.block.per_head_pitch.assign(Tensor(shifted, dtype=dtypes.float).contiguous())


def per_example_correctness(model, tok, examples, n_loops_pair, batch_size, cache_max_len):
    """Run accuracy_at_loops_multi, return parallel list of per-example bools."""
    acc, rows = accuracy_at_loops_multi(model, tok, examples, n_loops=n_loops_pair,
                                         batch_size=batch_size, cache_max_len=cache_max_len)
    # rows is list of (ex, parsed, gen) per example, same order as examples
    correctness = []
    for (ex, parsed, gen) in rows:
        correctness.append(parsed == ex.answer)
    return acc, correctness


def main():
    cfg = Config()
    CKPT = os.environ.get("CKPT", "")
    LEVEL = os.environ.get("LEVEL", "L4.5")
    NUM_EVAL = int(os.environ.get("NUM_EVAL", "100"))
    EVAL_LOOPS = [int(x) for x in os.environ.get("EVAL_LOOPS", "1,4,8").split(",")]
    EVAL_BATCH = int(os.environ.get("EVAL_BATCH", "64"))
    fixed_len = int(os.environ.get("FIXED_LEN", DEFAULT_FIXED_LEN.get(LEVEL, 160)))
    cache_max_len = int(os.environ.get("EVAL_CACHE_LEN", "0")) or (fixed_len + 40)

    print(f"=== Oracle ensemble test on {LEVEL} ===")
    print(f"  ckpt: {CKPT}")
    print(f"  shifts: 0.0  and π/4 ({math.pi/4:.3f} rad, 45.0°)")

    print(f"\ngenerating {LEVEL} eval set...")
    all_examples = generate_math(LEVEL, 20000, seed=42, digit_spacing=True)
    _, eval_examples = split_train_eval(all_examples, n_eval=NUM_EVAL, seed=42)
    tok = load_tokenizer()

    print(f"\nloading Pythia + ckpt...")
    sd = _load_state()
    model = load_breathing(cfg, sd=sd)
    cast_model_fp32(model)
    Device[Device.DEFAULT].synchronize()
    del sd
    ckpt_sd = safe_load(CKPT)
    model.load_state_dict(ckpt_sd, strict=False)
    del ckpt_sd

    original_pitch_np = model.block.per_head_pitch.numpy().copy()

    Tensor.training = False
    print()
    print(f"{'A':>3s} | {'acc@0':>7s} {'acc@π/4':>8s} {'oracle':>8s} {'both':>7s} {'diff':>6s}")
    print("-" * 50)

    for nl in EVAL_LOOPS:
        # shift=0
        set_phase_shift(model, original_pitch_np, 0.0)
        t0 = time.perf_counter()
        acc0, correct0 = per_example_correctness(model, tok, eval_examples,
                                                  [nl, 1], EVAL_BATCH, cache_max_len)
        t_shift0 = time.perf_counter() - t0
        # shift=π/4
        set_phase_shift(model, original_pitch_np, math.pi / 4)
        t0 = time.perf_counter()
        acc_pi4, correct_pi4 = per_example_correctness(model, tok, eval_examples,
                                                       [nl, 1], EVAL_BATCH, cache_max_len)
        t_shift_pi4 = time.perf_counter() - t0

        # Per-example bools
        c0 = np.array(correct0, dtype=bool)
        c1 = np.array(correct_pi4, dtype=bool)
        n = len(c0)
        oracle = (c0 | c1).sum() / n * 100  # union — correct at either shift
        both = (c0 & c1).sum() / n * 100    # intersection — correct at both
        diff = ((c0 ^ c1)).sum() / n * 100   # XOR — disagrees on these examples

        print(f"A={nl} | {acc0*100:>6.1f}%  {acc_pi4*100:>7.1f}%  {oracle:>7.1f}%  {both:>6.1f}%  {diff:>5.1f}%   "
              f"(time: {t_shift0+t_shift_pi4:.1f}s)", flush=True)

    print()
    print("Reading:")
    print("  oracle == max(acc@0, acc@π/4)  → shifts agree on which examples are hard. Ensembling won't help.")
    print("  oracle  > max(acc@0, acc@π/4)  → shifts have different strengths. Ensembling has real headroom.")
    print("  high 'diff'                    → predictions differ a lot. Logit-averaging worth trying.")


if __name__ == "__main__":
    main()
