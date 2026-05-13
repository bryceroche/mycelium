"""Classify L4 v4 errors by failure mode.

Runs the L4 v4 ckpt on 100 held-out eval problems, captures every wrong answer,
and classifies by mode:
  - off_by_10:        parsed = gold ± 10 (tens-place borrow/carry)
  - off_by_100:       parsed = gold ± 100 (hundreds-place error)
  - cycle0_wrong:     intermediate (after first ####) doesn't match expected
  - cycle1_wrong:     intermediate correct, final off
  - mangled:          parsing failure (parsed=None)
  - other:            wrong but doesn't fit above categories

Tells us whether the "tens-place borrow subtraction" diagnosis from N=2 samples
holds across the full eval set, or whether errors are distributed across multiple
modes.

Usage:
    DEV=PCI+AMD SINE_TEMP=1 SINE_TEMP_MAX=2.0 SINE_TEMP_MIN=0.7 \\
        CKPT=.cache/l4_ckpts/l4_v4_step1500.safetensors \\
        LOOPS=1 N=100 .venv/bin/python scripts/classify_l4_errors.py
"""
import sys, os, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from tinygrad import Tensor, Device, dtypes
from tinygrad.helpers import getenv
from tinygrad.nn.state import safe_load

from mycelium import Config
from mycelium.loader import _load_state, load_breathing
from mycelium.data import load_tokenizer
from mycelium.l3_data import generate_math, split_train_eval, parse_int_answer
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


def get_gold_intermediate(ex) -> int | None:
    """Extract the answer of the first cycle (the intermediate result) from
    gen_targets[0]. e.g., '8 5 * 2 = 1 7 0 cookies now.' → 170."""
    if not ex.gen_targets:
        return None
    cycle0 = ex.gen_targets[0]
    # Find last "= X" pattern in cycle0
    matches = re.findall(r'=\s*([\d\s]+?)(?:cookies|shells|toys|stickers|toy cars|\.|$)', cycle0)
    if not matches:
        return None
    last = matches[-1].strip()
    # Collapse digit-spaced into integer
    digits_only = re.sub(r'\s+', '', last)
    if digits_only.isdigit():
        return int(digits_only)
    return None


def parse_cycle0_intermediate(gen_text: str) -> int | None:
    """Parse the intermediate value from the generated text (before first ####)."""
    parts = gen_text.split("####")
    if len(parts) < 2:
        return None
    cycle0 = parts[0]
    # Same regex as gold
    matches = re.findall(r'=\s*([\d\s]+?)(?:cookies|shells|toys|stickers|toy cars|\.|$)', cycle0)
    if not matches:
        return None
    last = matches[-1].strip()
    digits_only = re.sub(r'\s+', '', last)
    if digits_only.isdigit():
        return int(digits_only)
    return None


def main():
    ckpt = getenv("CKPT", ".cache/l4_ckpts/l4_v4_step1500.safetensors")
    LEVEL = getenv("LEVEL", "L4")
    N = getenv("N", 100)
    LOOPS = getenv("LOOPS", 1)
    PHASE_C_LOOPS = getenv("PHASE_C_LOOPS", 1)
    SEED = getenv("SEED", 42)
    SPACE_DIGITS = bool(getenv("SPACE_DIGITS", 1))

    cfg = Config()
    print(f"=== L4 v4 error classification ===")
    print(f"CKPT={os.path.basename(ckpt)}  N={N}  LOOPS={LOOPS}\n")

    # Build held-out set the same way training does
    all_examples = generate_math(LEVEL, 20000, seed=SEED, digit_spacing=SPACE_DIGITS)
    _, eval_examples = split_train_eval(all_examples, n_eval=N, seed=SEED)
    print(f"eval set: {len(eval_examples)} problems\n")

    # Load model
    print("loading model + ckpt...")
    sd = _load_state(); model = load_breathing(cfg, sd=sd); cast_fp32(model); del sd
    info = model.load_state_dict(safe_load(ckpt), strict=False)
    print(f"  loaded.\n")
    tok = load_tokenizer()
    Tensor.training = False

    # Run accuracy_at_loops_multi with single loop count to get all rows
    print(f"running accuracy eval at A={LOOPS}...")
    import time
    t0 = time.perf_counter()
    acc, rows = accuracy_at_loops_multi(
        model, tok, eval_examples,
        n_loops=[LOOPS, PHASE_C_LOOPS],
        batch_size=64,
        cache_max_len=136,
    )
    print(f"  acc: {acc*100:.1f}% in {time.perf_counter()-t0:.1f}s\n")
    print(f"total wrong: {sum(1 for ex, parsed, _ in rows if parsed != ex.answer)}\n")

    # Classify each wrong row
    categories = {
        "off_by_10":      [],
        "off_by_100":     [],
        "cycle0_wrong":   [],
        "cycle1_wrong":   [],
        "mangled":        [],
        "other_small":    [],   # off by 1-9
        "other_large":    [],   # off by 11-99 or 101+
    }

    for ex, parsed, gen_text in rows:
        if parsed == ex.answer:
            continue
        gold = ex.answer
        gold_inter = get_gold_intermediate(ex)
        model_inter = parse_cycle0_intermediate(gen_text)

        if parsed is None:
            categories["mangled"].append((ex, parsed, gen_text, gold_inter, model_inter))
            continue

        diff = parsed - gold

        # First: is cycle 0 right or wrong?
        cycle0_correct = (gold_inter is not None and model_inter == gold_inter)

        if not cycle0_correct and model_inter is not None:
            categories["cycle0_wrong"].append((ex, parsed, gen_text, gold_inter, model_inter))
            continue

        # Cycle 0 was right (or we couldn't determine); the error is in cycle 1
        if abs(diff) == 10:
            categories["off_by_10"].append((ex, parsed, gen_text, gold_inter, model_inter))
        elif abs(diff) == 100:
            categories["off_by_100"].append((ex, parsed, gen_text, gold_inter, model_inter))
        elif abs(diff) < 10:
            categories["other_small"].append((ex, parsed, gen_text, gold_inter, model_inter))
        else:
            # cycle 0 right (or unknown) but cycle 1 off by some other amount
            categories["cycle1_wrong"].append((ex, parsed, gen_text, gold_inter, model_inter))

    # Report
    print("--- error classification ---")
    total_wrong = sum(len(v) for v in categories.values())
    print(f"total errors: {total_wrong}\n")
    for name, items in categories.items():
        if not items: continue
        print(f"  {name}: {len(items)} ({len(items)/max(1, total_wrong)*100:.0f}%)")
        for ex, parsed, gen_text, gi, mi in items[:3]:
            print(f"    Q: {ex.problem!r}")
            print(f"    gen: {gen_text[:160].strip()!r}{'...' if len(gen_text) > 160 else ''}")
            print(f"    parsed={parsed}  gold={ex.answer}  diff={parsed-ex.answer if parsed else 'N/A'}")
            print(f"    gold_intermediate={gi}  model_intermediate={mi}")
            print()

    # Tens-place borrow specifically: cycle 0 correct AND parsed = gold ± 10
    print("--- tens-place borrow hypothesis ---")
    n_off_by_10 = len(categories["off_by_10"])
    pct = n_off_by_10 / max(1, total_wrong) * 100
    print(f"  off-by-10 (cycle0 correct, final wrong by tens place): {n_off_by_10}/{total_wrong} ({pct:.0f}%)")
    if pct > 50:
        print(f"  → DOMINANT failure mode. Targeted borrow training (Option B) justified.")
    elif pct > 30:
        print(f"  → SIGNIFICANT but not dominant. Targeted training may help, but other modes also matter.")
    else:
        print(f"  → MINOR mode. Tens-place borrow is not the main bottleneck.")


if __name__ == "__main__":
    main()
