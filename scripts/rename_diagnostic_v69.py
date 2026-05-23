"""Rename diagnostic for v69 (USE_KV_CACHE=0 safe).

Replaces first proper noun in each GSM8K question with 'Xyz123', runs the model
on original vs renamed, and reports:
  - % cases where renamed output contains 'Xyz...' (grounded)
  - % cases where renamed output contains the original name (memorized template)
  - % cases where it emits neither (neither grounded nor memorized)

Forked from scripts/rename_diagnostic.py — the only change is that when
USE_KV_CACHE=0 it calls segmented_generate_batch instead of
segmented_generate_kv_batch. This is required for v69 because the Stage 1
JIT is not yet v69-aware (waist shape changed 512→128).

Env vars: same as eval_ckpt_controller_segmented.py. Plus:
    DIAG_OUT     Output JSONL path (default /tmp/rename_diag_v69.jsonl)
    USE_KV_CACHE 0 or 1 (default 0 for v69)
"""
import os
import sys
import json
import re
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from tinygrad import Tensor, Device, dtypes
from tinygrad.nn.state import safe_load

from mycelium import Config
from mycelium.loader import _load_state, load_breathing
from mycelium.data import load_tokenizer
from mycelium.l3_data import load_gsm8k_steps, parse_int_answer
from scripts.eval_ckpt_controller_segmented import (
    cast_model_fp32, segmented_generate_kv_batch, segmented_generate_batch
)


REPLACEMENT = "Xyz123"

# Words that ARE capitalized but shouldn't count as proper nouns
STOP_WORDS = {
    "The", "A", "An", "How", "Each", "If", "And", "But", "For", "What",
    "While", "She", "He", "It", "They", "We", "I", "On", "In", "At", "To",
    "After", "Before", "Over", "There", "When", "Where", "Why", "Then",
    "This", "That", "These", "Those", "Every", "All", "Some", "Many",
    "Today", "Tomorrow", "Yesterday", "Monday", "Tuesday", "Wednesday",
    "Thursday", "Friday", "Saturday", "Sunday",
}


def find_first_proper_noun(text: str) -> str | None:
    """Return the first capitalized word in text that isn't a stop word."""
    for word in text.split():
        w = re.sub(r"[^A-Za-z]", "", word)
        if (w and w[0].isupper() and len(w) > 1
            and w not in STOP_WORDS):
            return w
    return None


def main():
    CKPT         = os.environ.get("CKPT", "")
    NUM_EVAL     = int(os.environ.get("NUM_EVAL", "20"))
    FIXED_LEN    = int(os.environ.get("FIXED_LEN", "400"))
    MAX_NEW      = int(os.environ.get("MAX_NEW", "120"))
    BATCH        = int(os.environ.get("BATCH", "2"))
    USE_KV_CACHE = int(os.environ.get("USE_KV_CACHE", "0")) > 0
    DIAG_OUT     = os.environ.get("DIAG_OUT", "/tmp/rename_diag_v69.jsonl")

    print(f"=== rename diagnostic (v69) ===")
    print(f"  ckpt:         {CKPT}")
    print(f"  num_eval:     {NUM_EVAL}")
    print(f"  fixed_len:    {FIXED_LEN}")
    print(f"  use_kv_cache: {USE_KV_CACHE}")
    print(f"  replacement:  {REPLACEMENT!r}")

    path = os.environ.get("GSM8K_STEPS_PATH", ".cache/gsm8k_steps_v1_test.jsonl")
    all_examples = load_gsm8k_steps(path, min_k=2, max_k=6, bucket_by_k=False)

    # Find examples where the first proper noun is identifiable and stable
    selected = []
    for ex in all_examples:
        if len(selected) >= NUM_EVAL:
            break
        name = find_first_proper_noun(ex.problem)
        if name is None or len(name) < 3:
            continue
        if ex.problem.count(name) < 2:
            continue
        selected.append((ex, name))

    print(f"  selected {len(selected)} examples with stable proper noun")
    for ex, name in selected[:3]:
        print(f"    {name!r}: {ex.problem[:80]}")

    # Load model
    cfg = Config()
    sd = _load_state()
    model = load_breathing(cfg, sd=sd)
    cast_model_fp32(model)
    Device[Device.DEFAULT].synchronize()
    del sd

    if CKPT:
        ckpt_sd = safe_load(CKPT)
        info = model.load_state_dict(ckpt_sd, strict=False)
        print(f"  loaded; missing={len(info['missing'])}, unexpected={len(info['unexpected'])}")
        del ckpt_sd

    Tensor.training = False
    tok = load_tokenizer()

    gen_fn = segmented_generate_kv_batch if USE_KV_CACHE else segmented_generate_batch

    # Generate originals and renamed in pairs
    results = []
    t0 = time.perf_counter()
    for ex, name in selected:
        K = len(ex.gen_targets)
        orig_prompt    = ex.problem
        renamed_prompt = re.sub(r'\b' + re.escape(name) + r'\b', REPLACEMENT, ex.problem)

        # Run model on both (batch of 2)
        prompts    = [orig_prompt, renamed_prompt]
        prompt_ids = [tok.encode(p).ids for p in prompts]
        gen_per_ex = gen_fn(model, prompt_ids, tok, K=K,
                            fixed_len=FIXED_LEN, max_new=MAX_NEW)
        orig_gen    = tok.decode(gen_per_ex[0]).strip()
        renamed_gen = tok.decode(gen_per_ex[1]).strip()

        orig_has_name           = name in orig_gen
        renamed_has_name        = name in renamed_gen       # memorized?
        renamed_has_xyz         = REPLACEMENT in renamed_gen
        renamed_has_xyz_partial = "Xyz" in renamed_gen      # grounded?

        result = {
            "k": K,
            "name": name,
            "orig_prompt": orig_prompt[:200],
            "renamed_prompt": renamed_prompt[:200],
            "orig_gen": orig_gen[:300],
            "renamed_gen": renamed_gen[:300],
            "orig_has_name": orig_has_name,
            "renamed_has_name": renamed_has_name,
            "renamed_has_xyz": renamed_has_xyz,
            "renamed_has_xyz_partial": renamed_has_xyz_partial,
        }
        results.append(result)
        print(f"\n[{name!r}] K={K}")
        print(f"  orig:    {orig_gen[:120]!r}")
        print(f"  renamed: {renamed_gen[:120]!r}")
        print(f"  orig has '{name}': {orig_has_name}  "
              f"renamed has '{name}': {renamed_has_name}  "
              f"renamed has 'Xyz': {renamed_has_xyz_partial}")

    dt = time.perf_counter() - t0
    with open(DIAG_OUT, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"\n=== summary ({dt:.1f}s) ===")
    n = len(results)
    n_orig_grounded      = sum(1 for r in results if r["orig_has_name"])
    n_renamed_memorized  = sum(1 for r in results if r["renamed_has_name"])
    n_renamed_grounded   = sum(1 for r in results if r["renamed_has_xyz_partial"])
    n_renamed_neither    = sum(1 for r in results
                               if not r["renamed_has_name"] and not r["renamed_has_xyz_partial"])
    print(f"Originals:  {n_orig_grounded}/{n} contained the question's name")
    print(f"Renamed examples:")
    print(f"  emitted ORIG NAME (memorized): {n_renamed_memorized}/{n}")
    print(f"  emitted 'Xyz' (grounded):      {n_renamed_grounded}/{n}")
    print(f"  emitted NEITHER:               {n_renamed_neither}/{n}")
    print(f"  results written to: {DIAG_OUT}")

    # Compare to v66 baseline
    print(f"\nv66 baseline: 0/{NUM_EVAL} grounded (memorized templates)")


if __name__ == "__main__":
    main()
