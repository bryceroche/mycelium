"""Per-layer-offset configurable eval — sweep multiple 4-element offset configs.

For each config, sets per-layer offsets exactly as specified (all 16 heads
within a layer get the same offset), recomputes cos/sin tables, runs accuracy
eval. No training.

Tests whether *non-uniform* per-layer configurations within the model's phase
tolerance (from the uniform-shift sweep) preserve accuracy, or whether the
model is brittle to any per-layer pattern other than its trained one.

Env vars:
  CKPT (required)
  LEVEL (default L4.5)
  NUM_EVAL (default 100)
  EVAL_LOOPS (default '1,4,8')
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


def set_layer_offsets(model, offsets: list[float]):
    """Set per-layer pitch to `offsets` (one per layer). All heads within a
    layer get the same offset. Recompute cos/sin tables and assign."""
    cfg = model.cfg
    assert len(offsets) == cfg.n_phases, f"need {cfg.n_phases} offsets, got {len(offsets)}"
    ph_np = np.zeros((cfg.n_phases, cfg.n_heads), dtype=np.float32)
    for l in range(cfg.n_phases):
        ph_np[l, :] = offsets[l]
    cos_np = np.cos(ph_np).reshape(cfg.n_phases, 1, cfg.n_heads, 1, 1).astype(np.float32)
    sin_np = np.sin(ph_np).reshape(cfg.n_phases, 1, cfg.n_heads, 1, 1).astype(np.float32)
    model.block.per_head_pitch_cos.assign(Tensor(cos_np, dtype=dtypes.float).contiguous())
    model.block.per_head_pitch_sin.assign(Tensor(sin_np, dtype=dtypes.float).contiguous())
    model.block.per_head_pitch.assign(Tensor(ph_np, dtype=dtypes.float).contiguous())


def fmt_offsets(offsets):
    """Format offset list as multiples of π for readability."""
    parts = []
    for o in offsets:
        if abs(o) < 1e-6:
            parts.append("0")
        else:
            frac = o / math.pi
            # find a simple representation
            for denom in [2, 3, 4, 6, 8, 16, 32, 64]:
                if abs(frac * denom - round(frac * denom)) < 0.001:
                    num = round(frac * denom)
                    if num == 0:
                        parts.append("0")
                    elif denom == 1:
                        parts.append(f"{num}π")
                    else:
                        parts.append(f"{num}π/{denom}" if num != 1 else f"π/{denom}")
                    break
            else:
                parts.append(f"{o:.3f}")
    return "{" + ", ".join(parts) + "}"


def main():
    cfg = Config()
    CKPT = os.environ.get("CKPT", "")
    LEVEL = os.environ.get("LEVEL", "L4.5")
    NUM_EVAL = int(os.environ.get("NUM_EVAL", "100"))
    EVAL_LOOPS = [int(x) for x in os.environ.get("EVAL_LOOPS", "1,4,8").split(",")]
    EVAL_BATCH = int(os.environ.get("EVAL_BATCH", "64"))
    fixed_len = int(os.environ.get("FIXED_LEN", DEFAULT_FIXED_LEN.get(LEVEL, 160)))
    cache_max_len = int(os.environ.get("EVAL_CACHE_LEN", "0")) or (fixed_len + 40)

    # Define configs to test. Each is a 4-element list of per-layer offsets.
    pi = math.pi
    base_step = pi / (cfg.n_phases * cfg.n_heads)
    configs = [
        ("v23a baseline (matches v46b ckpt)", [l * base_step for l in range(cfg.n_phases)]),
        ("alternating π/8", [0, pi/8, 0, pi/8]),
        ("alternating π/4", [0, pi/4, 0, pi/4]),
        ("symmetric alternating ±π/8", [-pi/8, pi/8, -pi/8, pi/8]),
        ("triangle small (peak π/8)", [0, pi/16, pi/8, pi/16]),
        ("triangle medium (peak π/4)", [0, pi/8, pi/4, pi/8]),
    ]

    print(f"=== Per-layer offset sweep on {LEVEL} ({NUM_EVAL} examples) ===")
    print(f"  ckpt: {CKPT}")

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

    Tensor.training = False
    print()
    print(f"{'config':<50s} | " + " | ".join(f"A={nl}" for nl in EVAL_LOOPS))
    print("-" * (52 + 8 * len(EVAL_LOOPS)))

    results = []
    for name, offsets in configs:
        set_layer_offsets(model, offsets)
        accs = []
        for nl in EVAL_LOOPS:
            t0 = time.perf_counter()
            acc, _ = accuracy_at_loops_multi(model, tok, eval_examples,
                                              n_loops=[nl, 1],
                                              batch_size=EVAL_BATCH,
                                              cache_max_len=cache_max_len)
            accs.append(acc * 100)
        row_label = f"{name:<50s}"
        row_vals = " | ".join(f"{a:>5.1f}%" for a in accs)
        offset_str = fmt_offsets(offsets)
        print(f"{row_label} | {row_vals}   {offset_str}", flush=True)
        results.append((name, offsets, accs))

    print()
    print("=== summary ===")
    print(f"Baseline (v23a): A=1/4/8 should be 92/92/88")
    print(f"Configs ≥85 across all loops are candidates for training experiments.")


if __name__ == "__main__":
    main()
