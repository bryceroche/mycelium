"""Standalone accuracy eval on a saved checkpoint.

Decouples eval from training so a slow eval path doesn't block the training
loop. Works for any LEVEL (L3, L4, L4.5) at any n_loops sweep.

Usage:
    DEV=AMD CKPT=/path/to/ckpt.safetensors LEVEL=L4 LOOPS=1,8 \\
        python scripts/eval_l4.py

Env vars (defaults in parens):
    CKPT             — path to .safetensors checkpoint to evaluate
    LEVEL  (L4)      — curriculum level for problem generation
    LOOPS  (1,8)     — comma-separated n_loops values to evaluate at
    NUM_EVAL  (100)  — number of held-out problems
    SPACE_DIGITS (1) — digit-spacing on data (matches training)
    PHASE_C_LOOPS(1) — n_loops for execution cycles (3-phase scheduling)
    SEED      (42)   — for problem generation
"""
import sys, os, time
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
    _c(model.embed, "weight"); _c(model, "embed_out")
    sw = model.block.shared
    for a in ("wv","bv","wo","bo","w_out","b_out"): _c(sw, a)
    for layer in model.block.layers:
        for a in ("wq","bq","wk","bk","w_in","b_in"): _c(layer, a)


def main():
    ckpt = getenv("CKPT", "")
    assert ckpt and os.path.exists(ckpt), f"set CKPT=<path>, got {ckpt!r}"
    level = getenv("LEVEL", "L4")
    loops_str = getenv("LOOPS", "1,8")
    n_eval = getenv("NUM_EVAL", 100)
    space_digits = bool(getenv("SPACE_DIGITS", 1))
    phase_c_loops = getenv("PHASE_C_LOOPS", 1)
    seed = getenv("SEED", 42)
    eval_loops_list = [int(x) for x in loops_str.split(",")]
    num_problems = getenv("NUM_PROBLEMS", 20000)

    cfg = Config()
    print(f"=== eval {os.path.basename(ckpt)} on {level} ===")
    print(f"loops={eval_loops_list}  num_eval={n_eval}  space_digits={space_digits}  phase_c={phase_c_loops}")
    print()

    # Build held-out eval set the same way training does
    print(f"generating {level} problems...")
    t0 = time.perf_counter()
    all_examples = generate_math(level, num_problems, seed=seed, digit_spacing=space_digits)
    _, eval_examples = split_train_eval(all_examples, n_eval=n_eval, seed=seed)
    print(f"  eval set: {len(eval_examples)} examples  ({time.perf_counter()-t0:.1f}s)")

    print("\nloading model + checkpoint...")
    t0 = time.perf_counter()
    sd = _load_state()
    model = load_breathing(cfg, sd=sd); cast_fp32(model); del sd
    sd_ck = safe_load(ckpt)
    info = model.load_state_dict(sd_ck, strict=False)
    print(f"  loaded in {time.perf_counter()-t0:.1f}s")
    if info["missing"]:
        print(f"  (ckpt missing {len(info['missing'])} keys, kept default init)")

    # Sample sequence-length to size cache reasonably
    sample = eval_examples[0]
    sample_ids = load_tokenizer().encode(sample.problem).ids
    fixed_len = max(96, len(sample_ids) + 40)  # safe for both L3 (64) and L4 (96)
    eval_cache_len = fixed_len + 40
    eval_batch = 64

    tok = load_tokenizer()
    print(f"\neval_batch={eval_batch}  cache_max_len={eval_cache_len}\n")

    Tensor.training = False
    for nl in eval_loops_list:
        t0 = time.perf_counter()
        acc, rows = accuracy_at_loops_multi(
            model, tok, eval_examples,
            n_loops=[nl, phase_c_loops],
            batch_size=eval_batch,
            cache_max_len=eval_cache_len,
        )
        dt = time.perf_counter() - t0
        print(f"acc @ A={nl} C={phase_c_loops}: {acc*100:.1f}%  ({dt:.1f}s)")
        # Show 3 examples for sanity
        for ex, parsed, gen in rows[:3]:
            ok = "OK" if parsed == ex.answer else "WRONG"
            print(f"    Q: {ex.problem!r}")
            print(f"    gen: {gen.strip()!r}")
            print(f"    parsed={parsed} gold={ex.answer} [{ok}]")
        print()


if __name__ == "__main__":
    main()
