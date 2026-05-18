"""Eval a checkpoint on GSM8K with digit-spacing applied to the questions.

Why digit-spaced: our breathing transformer was trained on digit-spaced
arithmetic ("1 4 1" not "141"). Raw GSM8K uses BPE tokenization where
multi-digit numbers are subwords — model has never seen those. By spacing
the digits in the prompt, we keep the model's primary capability intact
while testing on the harder natural-English benchmark.

Env vars:
  CKPT (required): path to checkpoint safetensors
  NUM_EVAL (default 200): number of test examples (max 1319)
  EVAL_LOOPS (default '1,4,8')
  EVAL_BATCH (default 64)
  FIXED_LEN (default 256): big enough for p95 prompt + generation
  EVAL_CACHE_LEN (default FIXED_LEN+80)
"""
import os
import re
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pyarrow.parquet as pq
from tinygrad import Tensor, Device, dtypes
from tinygrad.nn.state import safe_load

from mycelium import Config
from mycelium.loader import _load_state, load_breathing
from mycelium.data import load_tokenizer
from mycelium.l3_data import MathExample, space_digits
from mycelium.l3_training import accuracy_at_loops_multi


GSM8K_TEST = (".cache/gsm8k/datasets--openai--gsm8k/snapshots/"
              "740312add88f781978c0658806c59bc2815b9866/main/test-00000-of-00001.parquet")


def load_gsm8k_digit_spaced(num: int, max_prompt_words: int = 130) -> list[MathExample]:
    """Load GSM8K test set, apply digit-spacing, return MathExample list.

    Filters out the longest tail (max_prompt_words guards against examples
    that won't fit in our fixed sequence length).
    """
    table = pq.read_table(GSM8K_TEST)
    data = table.to_pydict()
    out: list[MathExample] = []
    for q, a in zip(data["question"], data["answer"]):
        q_spaced = space_digits(q)
        if len(q_spaced.split()) > max_prompt_words:
            continue
        m = re.search(r"####\s*(-?\d[\d,]*)\s*$", a)
        if not m:
            continue
        gold = int(m.group(1).replace(",", ""))
        out.append(MathExample(
            problem=q_spaced,
            gen_targets=[""],
            answer=gold,
            level="GSM8K_SPACED",
        ))
        if len(out) >= num:
            break
    return out


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
    CKPT = os.environ.get("CKPT", "")
    NUM_EVAL = int(os.environ.get("NUM_EVAL", "200"))
    EVAL_LOOPS = [int(x) for x in os.environ.get("EVAL_LOOPS", "1,4,8").split(",")]
    EVAL_BATCH = int(os.environ.get("EVAL_BATCH", "64"))
    FIXED_LEN = int(os.environ.get("FIXED_LEN", "256"))
    EVAL_CACHE_LEN = int(os.environ.get("EVAL_CACHE_LEN", "0")) or (FIXED_LEN + 80)

    print(f"=== GSM8K (digit-spaced) eval ===")
    print(f"  ckpt: {CKPT}")
    print(f"  num_eval: {NUM_EVAL}  fixed_len: {FIXED_LEN}  cache_max_len: {EVAL_CACHE_LEN}")
    print(f"  eval_loops: {EVAL_LOOPS}  batch: {EVAL_BATCH}")
    print(f"  device: {Device.DEFAULT}")

    print(f"\nloading GSM8K test set + digit-spacing...")
    eval_examples = load_gsm8k_digit_spaced(NUM_EVAL)
    print(f"  loaded {len(eval_examples)} examples (filtered for prompt-fit)")
    if eval_examples:
        ex = eval_examples[0]
        print(f"  sample question: {ex.problem!r}")
        print(f"  sample answer:   {ex.answer}")
    tok = load_tokenizer()

    print(f"\nloading Pythia-410M -> breathing transformer...")
    sd = _load_state()
    model = load_breathing(cfg, sd=sd)
    cast_model_fp32(model)
    Device[Device.DEFAULT].synchronize()
    del sd

    if not CKPT:
        print("ERROR: CKPT env var required")
        sys.exit(1)
    print(f"\nloading checkpoint: {CKPT}")
    ckpt_sd = safe_load(CKPT)
    info = model.load_state_dict(ckpt_sd, strict=False)
    print(f"  missing keys: {len(info['missing'])}, unexpected: {len(info['unexpected'])}")
    del ckpt_sd

    Tensor.training = False
    print(f"\n=== accuracy on GSM8K-spaced ({len(eval_examples)} examples) ===")
    for nl in EVAL_LOOPS:
        t0 = time.perf_counter()
        acc, rows = accuracy_at_loops_multi(
            model, tok, eval_examples,
            n_loops=[nl, 1],
            batch_size=EVAL_BATCH,
            cache_max_len=EVAL_CACHE_LEN,
        )
        dt = time.perf_counter() - t0
        n_correct = int(acc * len(eval_examples))
        # binomial standard error
        p = acc
        se = (p * (1 - p) / max(len(eval_examples), 1)) ** 0.5
        print(f"  acc @ A={nl} C=1: {acc*100:.1f}%  ({n_correct}/{len(eval_examples)}, ±{se*100*1.96:.1f}% 95% CI)  ({dt:.1f}s)")
        for ex, parsed, gen in rows[:2]:
            short_q = ex.problem if len(ex.problem) < 120 else ex.problem[:117] + "..."
            short_g = gen.strip() if len(gen.strip()) < 200 else gen.strip()[:197] + "..."
            print(f"    Q: {short_q}")
            print(f"    gen: {short_g!r}")
            print(f"    parsed: {parsed}, gold: {ex.answer}, {'OK' if parsed == ex.answer else 'WRONG'}")
        print()


if __name__ == "__main__":
    main()
