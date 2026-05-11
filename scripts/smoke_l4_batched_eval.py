"""A/B verification: batched vs sequential L4 multi-cycle accuracy eval.

Loads the resumed L3 checkpoint (the same starting point the v1 L4 run used),
generates a tiny L4 eval set, and runs accuracy_at_loops_multi twice — once
through the new batched code path (the patched else-branch), once through
explicit sequential multi_cycle_generate calls (what the old else-branch did).

Both paths are deterministic argmax-greedy decoding, so per-example parsed
answers should match exactly. If accuracy and per-example parses match, the
batched patch is correct and we can relaunch training.

Usage:
  DEV=PCI+AMD .venv/bin/python scripts/smoke_l4_batched_eval.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from tinygrad import Tensor, Device, dtypes
from tinygrad.helpers import getenv

from mycelium import Config
from mycelium.loader import _load_state, load_breathing
from mycelium.data import load_tokenizer
from mycelium.l3_data import generate_math
from mycelium.l3_training import (
    accuracy_at_loops_multi, multi_cycle_generate, parse_int_answer,
)
from tinygrad.nn.state import safe_load


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
    N = getenv("N", 10)
    PHASE_A = getenv("PHASE_A", 8)
    PHASE_C = getenv("PHASE_C", 1)
    CKPT = getenv("CKPT", "/home/bryce/mycelium/.cache/l3_ckpts/l3_ctrl_v5_step375.safetensors")
    SEED = getenv("SEED", 42)

    print(f"device={Device.DEFAULT}")
    print(f"A/B: N={N} L4  n_loops=[{PHASE_A}, {PHASE_C}]  ckpt={os.path.basename(CKPT)}")
    print()

    print("generating L4 examples...")
    all_examples = generate_math("L4", N, seed=SEED, digit_spacing=True)
    examples = all_examples[:N]
    print(f"  got {len(examples)} examples (digit-spaced)")

    print("\nloading Pythia-410M -> breathing transformer...")
    sd = _load_state()
    model = load_breathing(cfg, sd=sd)
    cast_model_fp32(model)
    Device[Device.DEFAULT].synchronize()
    del sd

    print(f"resuming from {CKPT}")
    ck_sd = safe_load(CKPT)
    info = model.load_state_dict(ck_sd, strict=False)
    if info["missing"]:
        print(f"  (ckpt missing {len(info['missing'])} keys)")
    if info["unexpected"]:
        print(f"  (ignoring {len(info['unexpected'])} extra ckpt keys)")
    del ck_sd

    tok = load_tokenizer()
    Tensor.training = False

    n_loops_list = [int(PHASE_A), int(PHASE_C)]

    # ---- NEW path: batched ----
    print(f"\n--- NEW (batched) path ---", flush=True)
    t0 = time.perf_counter()
    new_acc, new_rows = accuracy_at_loops_multi(
        model, tok, examples,
        n_loops=n_loops_list,
        batch_size=min(N, 16),
        cache_max_len=200,
        max_new_per_cycle=40,
    )
    new_dt = time.perf_counter() - t0
    print(f"  new_acc = {new_acc*100:.1f}%   ({new_dt:.1f}s)")

    new_parses = [(parsed, ex.answer) for (ex, parsed, _gen) in new_rows]

    # ---- OLD path: sequential per-example multi_cycle_generate ----
    print(f"\n--- OLD (sequential) path ---", flush=True)
    t0 = time.perf_counter()
    correct = 0
    old_parses = []
    for ex in examples:
        prompt_ids = tok.encode(ex.problem).ids
        cycle_outs = multi_cycle_generate(
            model, tok, prompt_ids, n_loops=n_loops_list,
            n_cycles=len(ex.gen_targets), max_new_per_cycle=40,
            use_kv_cache=True,
        )
        last_text = tok.decode(cycle_outs[-1])
        parsed = parse_int_answer(last_text)
        old_parses.append((parsed, ex.answer))
        if parsed == ex.answer:
            correct += 1
    old_acc = correct / len(examples)
    old_dt = time.perf_counter() - t0
    print(f"  old_acc = {old_acc*100:.1f}%   ({old_dt:.1f}s)")

    # ---- Compare ----
    print(f"\n--- A/B verdict ---")
    print(f"  speedup: {old_dt/new_dt:.1f}× (old {old_dt:.1f}s vs new {new_dt:.1f}s)")
    matches = sum(1 for (np_, _), (op_, _) in zip(new_parses, old_parses) if np_ == op_)
    print(f"  per-example parse match: {matches}/{len(examples)}")
    if matches != len(examples):
        print("  MISMATCH — diffs:")
        for i, ((np_, gold), (op_, _)) in enumerate(zip(new_parses, old_parses)):
            if np_ != op_:
                print(f"    ex{i}: new={np_}  old={op_}  gold={gold}")
    if new_acc == old_acc and matches == len(examples):
        print(f"\n  PASS — batched path is correct. {old_dt/new_dt:.1f}× faster.")
    else:
        print(f"\n  FAIL — investigate.")


if __name__ == "__main__":
    main()
