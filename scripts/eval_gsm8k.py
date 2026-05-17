"""Zero-shot GSM8K eval. Two modes:
  raw: no preprocessing — apples-to-apples vs Pythia-410M baseline (0% public)
  spaced: digit-spaced preprocessing — gives our digit-tokenized model a fair shot

Usage:
    DEV=PCI+AMD CKPT=/path/to/ckpt LOOPS=1,4,8 NUM_EVAL=100 MODE=spaced \\
        python scripts/eval_gsm8k.py
"""
import os
import sys
import time
import re

os.environ.setdefault("DEV", "PCI+AMD")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pyarrow.parquet as pq
from tinygrad import Tensor, Device, dtypes
from tinygrad.helpers import getenv
from tinygrad.nn.state import safe_load

from mycelium import Config
from mycelium.loader import _load_state, load_breathing
from mycelium.data import load_tokenizer
from mycelium.l3_data import MathExample, parse_int_answer
from mycelium.l3_training import accuracy_at_loops_multi


GSM8K_TEST = "/home/bryce/mycelium/.cache/gsm8k/datasets--openai--gsm8k/snapshots/740312add88f781978c0658806c59bc2815b9866/main/test-00000-of-00001.parquet"


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


def parse_gsm8k_answer(text: str):
    """Extract integer answer from GSM8K answer field (ends with '#### N')."""
    m = re.search(r"####\s*(-?\d[\d,]*\.?\d*)", text)
    if not m:
        return None
    val = m.group(1).replace(",", "")
    try:
        return int(float(val))
    except ValueError:
        return None


def digit_space(s: str) -> str:
    """Insert spaces between consecutive digits. '144' → '1 4 4'."""
    return re.sub(r"(\d)(?=\d)", r"\1 ", s)


def make_math_example(question: str, answer_text: str, mode: str = "raw") -> MathExample:
    """Convert GSM8K problem to MathExample format.

    In mode=spaced, all digits in question + answer chain are space-separated.
    In mode=raw, kept as-is.
    """
    gold = parse_gsm8k_answer(answer_text)
    if gold is None:
        return None
    # Strip the calculator annotations and the #### marker for the "gen" target
    chain = re.sub(r"<<[^>]*>>", "", answer_text)
    chain = re.sub(r"####.*$", "", chain, flags=re.MULTILINE).strip()
    # Convert to one cycle per line (each line = one reasoning step)
    lines = [ln.strip() for ln in chain.split("\n") if ln.strip()]
    if not lines:
        return None
    if mode == "spaced":
        question = digit_space(question)
        lines = [digit_space(ln) for ln in lines]
        gold_str = digit_space(str(gold))
    else:
        gold_str = str(gold)
    # Build gen as the full reasoning + final answer line
    gen_targets = [ln for ln in lines]
    # Append final answer as own cycle for our pipeline
    gen_targets.append(f"The answer is {gold_str}.")
    return MathExample(
        problem=question,
        gen_targets=gen_targets,
        answer=gold,
        level="GSM8K",
    )


def main():
    ckpt = getenv("CKPT", "/home/bryce/mycelium/.cache/l4_mixed_ckpts/l4_mixed_v24c_dual_notebook_step500.safetensors")
    mode = getenv("MODE", "spaced")  # raw or spaced
    loops_str = getenv("LOOPS", "1,4,8")
    n_eval = getenv("NUM_EVAL", 100)
    seed = getenv("SEED", 42)
    eval_loops_list = [int(x) for x in loops_str.split(",")]
    eval_batch = getenv("EVAL_BATCH", 32)
    fixed_len = getenv("FIXED_LEN", 384)   # GSM8K problems are longer
    cache_len = fixed_len + 100

    cfg = Config()
    print(f"=== GSM8K zero-shot eval on {os.path.basename(ckpt)} ===")
    print(f"  mode={mode}  loops={eval_loops_list}  num_eval={n_eval}  fixed_len={fixed_len}  eval_batch={eval_batch}")
    print()

    print(f"loading GSM8K test set ({GSM8K_TEST})...")
    t = pq.read_table(GSM8K_TEST)
    questions = t["question"].to_pylist()
    answers = t["answer"].to_pylist()
    print(f"  {len(questions)} problems total")

    # Shuffle and pick eval subset
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(questions))[:n_eval]
    eval_examples = []
    for i in indices:
        ex = make_math_example(questions[i], answers[i], mode=mode)
        if ex is not None:
            eval_examples.append(ex)
    print(f"  {len(eval_examples)} examples (after parsing)")
    print()
    # Show a sample
    if eval_examples:
        ex = eval_examples[0]
        print(f"Sample (mode={mode}):")
        print(f"  Q: {ex.problem[:200]}")
        print(f"  A (gold): {ex.answer}")
        print(f"  gen_targets[0]: {ex.gen_targets[0][:200]}")
        print()

    print("loading model + ckpt...")
    t0 = time.perf_counter()
    sd = _load_state()
    model = load_breathing(cfg, sd=sd)
    cast_fp32(model)
    del sd
    state = safe_load(ckpt)
    info = model.load_state_dict(state, strict=False)
    print(f"  loaded in {time.perf_counter()-t0:.1f}s")
    if info.get("missing"):
        print(f"  ({len(info['missing'])} missing keys, kept default init)")

    tok = load_tokenizer()
    print(f"\n=== Eval ===")
    for nl in eval_loops_list:
        t0 = time.perf_counter()
        acc, rows = accuracy_at_loops_multi(model, tok, eval_examples,
                                             n_loops=[nl, 1],
                                             batch_size=eval_batch,
                                             cache_max_len=cache_len)
        dt = time.perf_counter() - t0
        print(f"  acc @ A={nl} C=1: {acc*100:.1f}%  ({dt:.1f}s)")
        # Show a few examples for the first eval
        if nl == eval_loops_list[0]:
            for ex, parsed, gen in rows[:3]:
                print(f"    Q: {ex.problem[:150]}")
                print(f"    gen: {gen.strip()[:300]!r}")
                print(f"    parsed: {parsed}, gold: {ex.answer}, {'OK' if parsed == ex.answer else 'WRONG'}")
                print()


if __name__ == "__main__":
    main()
