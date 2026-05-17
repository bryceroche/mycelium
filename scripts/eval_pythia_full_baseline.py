"""Full Pythia-410M (all 24 layers) zero-shot baseline eval.

Builds the complete Pythia-410M model via load_pythia_baseline(n_layers=24),
loads embed_out separately, runs simple autoregressive greedy generation.
No fine-tuning. Tests what a full pretrained Pythia-410M gets on our math
curricula zero-shot.

Usage:
    DEV=PCI+AMD LEVEL=L4_MIXED NUM_EVAL=20 \\
        python scripts/eval_pythia_full_baseline.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from tinygrad import Tensor, Device, dtypes
from tinygrad.helpers import getenv

from mycelium import Config
from mycelium.loader import _load_state, load_pythia_baseline
from mycelium.data import load_tokenizer
from mycelium.l3_data import generate_math, split_train_eval, parse_int_answer


def main():
    level = getenv("LEVEL", "L4_MIXED")
    n_eval = getenv("NUM_EVAL", 20)
    space_digits = bool(getenv("SPACE_DIGITS", 1))
    seed = getenv("SEED", 42)
    max_new = getenv("MAX_NEW", 80)
    num_problems = getenv("NUM_PROBLEMS", 20000)

    cfg = Config()
    print(f"=== Full Pythia-410M (24 layers) zero-shot baseline on {level} ===")
    print(f"num_eval={n_eval}  max_new_tokens={max_new}  space_digits={space_digits}")
    print()

    print(f"generating {level} problems...")
    t0 = time.perf_counter()
    all_examples = generate_math(level, num_problems, seed=seed, digit_spacing=space_digits)
    _, eval_examples = split_train_eval(all_examples, n_eval=n_eval, seed=seed)
    print(f"  eval set: {len(eval_examples)} examples  ({time.perf_counter()-t0:.1f}s)")

    print("\nloading Pythia-410M (24 layers)...")
    t0 = time.perf_counter()
    sd = _load_state()
    model = load_pythia_baseline(cfg, n_layers=24, sd=sd)
    # Load embed_out (lm_head) — separate from embed_in for Pythia
    embed_out_key = "embed_out.weight" if "embed_out.weight" in sd else None
    if embed_out_key is None:
        # Fallback: try other naming
        for k in sd:
            if "embed_out" in k or "lm_head" in k:
                embed_out_key = k
                break
    print(f"  embed_out key: {embed_out_key}")
    embed_out_w = sd[embed_out_key]
    # Convert to numpy on CPU. Output projection happens in numpy each step to
    # sidestep a tinygrad rangeify bug in the slice+matmul path.
    embed_out_np = embed_out_w.numpy().astype(np.float32)
    if embed_out_np.shape[0] == cfg.vocab_size:
        embed_out_np = embed_out_np.T  # → (hidden, vocab)
    print(f"  embed_out (numpy) shape: {embed_out_np.shape}")
    del sd
    Device[Device.DEFAULT].synchronize()
    print(f"  loaded in {time.perf_counter()-t0:.1f}s")

    tok = load_tokenizer()
    vocab_active = 50277  # match the training-time effective vocab

    print(f"\n=== Running eval (greedy generation, max_new={max_new}) ===")
    correct = 0
    total = 0
    samples_shown = 0
    t0 = time.perf_counter()
    for ex in eval_examples:
        enc = tok.encode(ex.problem, add_special_tokens=False)
        prompt_ids = enc.ids if hasattr(enc, "ids") else list(enc)
        ids = list(prompt_ids)
        # Greedy autoregressive generation. Fresh tensor each step (no growing graph).
        for _ in range(max_new):
            ids_np = np.array([ids], dtype=np.int32)
            tokens = Tensor(ids_np, dtype=dtypes.int).realize()
            h = model(tokens).realize()  # (1, T, hidden)
            h_last_np = h.numpy()[0, -1, :].astype(np.float32)  # (hidden,)
            logits_np = h_last_np @ embed_out_np  # (vocab,)
            next_id = int(np.argmax(logits_np[:vocab_active]))
            ids.append(next_id)
        gen_ids = ids[len(prompt_ids):]
        try:
            gen_text = tok.decode(gen_ids, skip_special_tokens=True)
        except TypeError:
            gen_text = tok.decode(gen_ids)
        parsed = parse_int_answer(gen_text)
        is_correct = (parsed == ex.answer)
        if is_correct:
            correct += 1
        total += 1
        if samples_shown < 3:
            print(f"  Q: {ex.problem[:120]}")
            print(f"  gen: {gen_text[:200]!r}")
            print(f"  parsed: {parsed}, gold: {ex.answer}, {'OK' if is_correct else 'WRONG'}")
            print()
            samples_shown += 1
    dt = time.perf_counter() - t0
    print(f"\n  ACCURACY: {correct}/{total} = {correct/total*100:.1f}%  ({dt:.1f}s)")


if __name__ == "__main__":
    main()
