"""Inference-time phase-shift sweep diagnostic.

Loads a checkpoint once, then for each PHASE_SHIFT value in the sweep:
1. Modifies the per-head pitch (uniformly shifts ALL heads by the offset)
2. Reassigns per_head_pitch_cos / per_head_pitch_sin tables (used in attention)
3. Runs accuracy_at_loops_multi against L4.5 eval set

Answers: how far can we phase-shift the model at inference before it breaks?

If acc at PHASE_SHIFT=π/2 is comparable to baseline → quadrature views are
viable; an ensemble could combine them. If acc collapses at π/16 or π/8 →
the trained representations are tightly coupled to the exact training phase
and any phase-shift augmentation is dead.

This is the cheap diagnostic for whether quadrature has *any* legs.

Env vars:
  CKPT (required)
  LEVEL (default L4.5)
  NUM_EVAL (default 100)
  PHASE_SHIFTS (default '0,0.196,0.393,0.785,1.571'  ≈ 0, π/16, π/8, π/4, π/2)
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
from mycelium.l3_data import generate_math, split_train_eval
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


def apply_uniform_phase_shift(model, shift_rad: float):
    """Uniformly add shift_rad to per_head_pitch values for ALL heads in ALL layers.
    Recompute cos/sin tables and assign in place — JIT graph picks up the new
    values on next replay (same mechanism as the quadrature ramp in training)."""
    cfg = model.cfg
    half_heads = cfg.n_heads // 2
    pitch_range = 2 * math.pi  # use full circle — uniform shift, this just affects the base ranges
    # Read out current per_head_pitch (which has whatever offsets ckpt provided)
    current = model.block.per_head_pitch.numpy()  # (n_phases, n_heads)
    shifted = current + shift_rad  # uniform shift
    cos_np = np.cos(shifted).reshape(cfg.n_phases, 1, cfg.n_heads, 1, 1).astype(np.float32)
    sin_np = np.sin(shifted).reshape(cfg.n_phases, 1, cfg.n_heads, 1, 1).astype(np.float32)
    model.block.per_head_pitch_cos.assign(Tensor(cos_np, dtype=dtypes.float).contiguous())
    model.block.per_head_pitch_sin.assign(Tensor(sin_np, dtype=dtypes.float).contiguous())


def main():
    cfg = Config()
    CKPT = os.environ.get("CKPT", "")
    LEVEL = os.environ.get("LEVEL", "L4.5")
    NUM_EVAL = int(os.environ.get("NUM_EVAL", "100"))
    EVAL_LOOPS = [int(x) for x in os.environ.get("EVAL_LOOPS", "1,4,8").split(",")]
    EVAL_BATCH = int(os.environ.get("EVAL_BATCH", "64"))
    fixed_len = int(os.environ.get("FIXED_LEN", DEFAULT_FIXED_LEN.get(LEVEL, 160)))
    cache_max_len = int(os.environ.get("EVAL_CACHE_LEN", "0")) or (fixed_len + 40)
    shifts_str = os.environ.get("PHASE_SHIFTS", f"0,{math.pi/16:.4f},{math.pi/8:.4f},{math.pi/4:.4f},{math.pi/2:.4f}")
    shifts = [float(s) for s in shifts_str.split(",")]

    print(f"=== Phase-shift sweep on {LEVEL} ===")
    print(f"  ckpt: {CKPT}")
    print(f"  shifts (radians): {shifts}")
    print(f"  shifts (degrees): {[f'{s*180/math.pi:.1f}°' for s in shifts]}")

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

    # Snapshot the original per_head_pitch for reset between shifts
    original_pitch = model.block.per_head_pitch.numpy().copy()

    Tensor.training = False
    print()
    print(f"{'shift':>10s} {'deg':>8s} | " + " | ".join(f"A={nl:1d}" for nl in EVAL_LOOPS))
    print("-" * (24 + 8 * len(EVAL_LOOPS)))

    results = {}
    for shift in shifts:
        # Reset to original pitch, then apply uniform shift
        model.block.per_head_pitch.assign(Tensor(original_pitch, dtype=dtypes.float).contiguous())
        # Reset cos/sin too
        cos_np = np.cos(original_pitch).reshape(cfg.n_phases, 1, cfg.n_heads, 1, 1).astype(np.float32)
        sin_np = np.sin(original_pitch).reshape(cfg.n_phases, 1, cfg.n_heads, 1, 1).astype(np.float32)
        model.block.per_head_pitch_cos.assign(Tensor(cos_np, dtype=dtypes.float).contiguous())
        model.block.per_head_pitch_sin.assign(Tensor(sin_np, dtype=dtypes.float).contiguous())
        # Now apply the test shift
        apply_uniform_phase_shift(model, shift)

        accs = []
        for nl in EVAL_LOOPS:
            t0 = time.perf_counter()
            acc, _ = accuracy_at_loops_multi(
                model, tok, eval_examples,
                n_loops=[nl, 1],
                batch_size=EVAL_BATCH,
                cache_max_len=cache_max_len,
            )
            accs.append(acc * 100)
            print(f"    shift={shift:.3f} ({shift*180/math.pi:.1f}°) A={nl}: {acc*100:.1f}%  ({time.perf_counter()-t0:.1f}s)", flush=True)
        results[shift] = accs

    print()
    print(f"=== summary ===")
    print(f"{'shift':>10s} {'deg':>8s} | " + " | ".join(f"A={nl:1d}".rjust(6) for nl in EVAL_LOOPS))
    print("-" * (24 + 9 * len(EVAL_LOOPS)))
    for shift, accs in results.items():
        row = f"{shift:>10.3f} {shift*180/math.pi:>7.1f}° | " + " | ".join(f"{a:>5.1f}%" for a in accs)
        print(row)


if __name__ == "__main__":
    main()
