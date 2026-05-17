"""Quick diagnostic: does v24c (L4_MIXED champion) generalize to single-cycle ARITH?

v24c was trained on L4_MIXED (multi-cycle word problems with +, -, ×, ÷ as intermediate
cycle ops). Each cycle in L4_MIXED is essentially a single-cycle arithmetic problem.
Question: do v24c's weights transfer to single-cycle ARITH directly?

If yes (and especially if +/- accuracy is non-zero, unlike v36/v38), v24c becomes our
balanced foundation. Cold-starting from Pythia consistently produces the +/- blindspot
because the model converges to × ÷ local minima on uniform ARITH data.

Usage:
    DEV=PCI+AMD CKPT=/path/to/v24c.safetensors python scripts/eval_v24c_on_arith.py
"""
import os
import sys
import time

os.environ.setdefault("DEV", "PCI+AMD")
# v24c training env (from l4_mixed_v24c_dual_notebook.log)
os.environ.setdefault("PER_HEAD_PITCH", "1")
os.environ.setdefault("ABLATE_BREATH_ROTATION", "1")
os.environ.setdefault("CONSTANT_RADIUS", "1")
os.environ.setdefault("SINE_TEMP", "1")
os.environ.setdefault("SINE_TEMP_MAX", "2.0")
os.environ.setdefault("SINE_TEMP_MIN", "0.7")
os.environ.setdefault("BREATH_TIME_EMBED", "1")
os.environ.setdefault("BREATH_TIME_INIT_SCALE", "0.0")
os.environ.setdefault("CROSS_BREATH_HANDOFF", "1")
os.environ.setdefault("NOTEBOOK_V24", "1")
os.environ.setdefault("NOTEBOOK_DUAL", "1")
os.environ.setdefault("NOTEBOOK_POOL_MODE", "attn")
os.environ.setdefault("NOTEBOOK_INIT_SCALE", "0.02")
os.environ.setdefault("LOOKUP_VALUE_INJECT", "1")
os.environ.setdefault("BFIELD_WAIST", "0")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from tinygrad import Tensor, Device, dtypes
from tinygrad.helpers import getenv
from tinygrad.nn.state import safe_load

from mycelium import Config
from mycelium.loader import _load_state, load_breathing
from mycelium.data import load_tokenizer
from mycelium.l3_data import generate_math, parse_int_answer
from mycelium.lookup_table import op_label_from_text
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
    ckpt = getenv("CKPT", "/home/bryce/mycelium/.cache/l4_mixed_ckpts/l4_mixed_v24c_dual_notebook_step500.safetensors")
    n_eval = getenv("N_EVAL", 100)
    n_loops = getenv("LOOPS", 8)
    fixed_len = getenv("FIXED_LEN", 32)
    seed = getenv("SEED", 42)

    cfg = Config()
    print(f"=== v24c → ARITH diagnostic ===")
    print(f"  ckpt: {os.path.basename(ckpt)}")
    print(f"  N: {n_eval}  loops: {n_loops}  fixed_len: {fixed_len}")
    print(f"  env: NOTEBOOK_V24+DUAL=on, PER_HEAD_PITCH=on, ABLATE_BREATH_ROTATION=on, etc.")
    print()

    print("loading Pythia → breathing transformer...")
    t0 = time.perf_counter()
    sd = _load_state()
    model = load_breathing(cfg, sd=sd)
    cast_fp32(model)
    Device[Device.DEFAULT].synchronize()
    print(f"  loaded base ({time.perf_counter()-t0:.1f}s)")

    print(f"\nloading v24c ckpt: {os.path.basename(ckpt)}")
    state = safe_load(ckpt)
    info = model.load_state_dict(state, strict=False)
    Device[Device.DEFAULT].synchronize()
    if info.get("missing"):
        print(f"  missing keys: {len(info['missing'])} ({info['missing'][:3]}...)")

    tok = load_tokenizer()
    print(f"\ngenerating {n_eval} ARITH problems...")
    examples = generate_math("ARITH", n_eval + 200, seed=seed, digit_spacing=True)
    examples = examples[:n_eval]
    print(f"  generated {len(examples)} examples")

    Tensor.training = False
    print(f"\nrunning batched accuracy eval at A={n_loops}...")
    t1 = time.perf_counter()
    acc, rows = accuracy_at_loops_multi(model, tok, examples, n_loops=n_loops,
                                          batch_size=64, cache_max_len=fixed_len + 40)
    print(f"  overall accuracy: {acc*100:.1f}%  ({time.perf_counter()-t1:.1f}s)")

    # Per-op breakdown
    per_op_total = [0, 0, 0, 0]
    per_op_correct = [0, 0, 0, 0]
    for ex, parsed, _gen in rows:
        op = op_label_from_text(ex.problem)
        if 0 <= op < 4:
            per_op_total[op] += 1
            if parsed is not None and parsed == ex.answer:
                per_op_correct[op] += 1

    print(f"\n=== Per-op breakdown ===")
    op_names = ['+', '-', '*', '/']
    for op in range(4):
        if per_op_total[op] > 0:
            pct = per_op_correct[op] / per_op_total[op] * 100
            print(f"  {op_names[op]}: {per_op_correct[op]}/{per_op_total[op]}  ({pct:.0f}%)")
        else:
            print(f"  {op_names[op]}: 0/0 (no examples)")

    # Show 5 example outputs (one per op if possible) so we can see WHAT v24c outputs
    print(f"\n=== Sample outputs ===")
    shown = {0: False, 1: False, 2: False, 3: False}
    for ex, parsed, gen_text in rows[:50]:
        op = op_label_from_text(ex.problem)
        if 0 <= op < 4 and not shown[op]:
            shown[op] = True
            mark = "✓" if (parsed is not None and parsed == ex.answer) else "✗"
            short_gen = gen_text[:30].strip().replace("\n", " ")
            print(f"  [{op_names[op]}] {ex.problem!r}  →  {short_gen!r}  (gold: {ex.answer})  {mark}")
            if all(shown.values()):
                break


if __name__ == "__main__":
    main()
