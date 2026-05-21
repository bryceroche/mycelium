"""Run a ckpt on N GSM8K test problems and dump structured per-example output
for manual failure analysis. Forks scripts/eval_ckpt_controller_segmented.py
but writes ALL examples (not just first 3) to a JSONL.

Env: same as the eval script. Plus DIAG_OUT (default /tmp/v60_diag.jsonl).
"""
import os
import sys
import json
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tinygrad import Tensor, Device, dtypes
from tinygrad.nn.state import safe_load

from mycelium import Config
from mycelium.loader import _load_state, load_breathing
from mycelium.data import load_tokenizer
from mycelium.l3_data import generate_math, split_train_eval, parse_int_answer

# reuse the eval script's gen functions
from scripts.eval_ckpt_controller_segmented import (
    segmented_generate_kv_batch, segmented_generate_batch, cast_model_fp32,
)


def main():
    CKPT = os.environ.get("CKPT", "")
    LEVEL = os.environ.get("LEVEL", "GSM8K_STEPS")
    NUM_EVAL = int(os.environ.get("NUM_EVAL", "100"))
    FIXED_LEN = int(os.environ.get("FIXED_LEN", "400"))
    MAX_NEW = int(os.environ.get("MAX_NEW", "120"))
    BATCH = int(os.environ.get("BATCH", "2"))
    USE_KV_CACHE = int(os.environ.get("USE_KV_CACHE", "1")) > 0
    DIAG_OUT = os.environ.get("DIAG_OUT", "/tmp/gsm8k_diag.jsonl")

    print(f"=== diagnostic dump on {LEVEL} ===")
    print(f"  ckpt: {CKPT}")
    print(f"  out: {DIAG_OUT}")
    print(f"  num_eval: {NUM_EVAL}  fixed_len: {FIXED_LEN}  max_new: {MAX_NEW}")

    if LEVEL == "GSM8K_STEPS":
        from mycelium.l3_data import load_gsm8k_steps
        path = os.environ.get("GSM8K_STEPS_PATH", ".cache/gsm8k_steps_v1_test.jsonl")
        min_k = int(os.environ.get("GSM8K_STEPS_MIN_K", "2"))
        max_k = int(os.environ.get("GSM8K_STEPS_MAX_K", "6"))
        all_examples = load_gsm8k_steps(path, min_k=min_k, max_k=max_k, bucket_by_k=False)
        _, eval_examples = split_train_eval(all_examples, n_eval=NUM_EVAL, seed=42)
    else:
        all_examples = generate_math(LEVEL, 20000, seed=42, digit_spacing=True)
        _, eval_examples = split_train_eval(all_examples, n_eval=NUM_EVAL, seed=42)

    print(f"  {len(eval_examples)} examples loaded")

    cfg = Config()
    sd = _load_state()
    model = load_breathing(cfg, sd=sd)
    cast_model_fp32(model)
    Device[Device.DEFAULT].synchronize()
    del sd
    if CKPT:
        ckpt_sd = safe_load(CKPT)
        info = model.load_state_dict(ckpt_sd, strict=False)
        print(f"  loaded; missing {len(info['missing'])}, unexpected {len(info['unexpected'])}")
        del ckpt_sd

    Tensor.training = False
    gen_fn = segmented_generate_kv_batch if USE_KV_CACHE else segmented_generate_batch
    tok = load_tokenizer()

    # group by K for JIT cache efficiency
    examples_by_k: dict[int, list] = {}
    for ex in eval_examples:
        examples_by_k.setdefault(len(ex.gen_targets), []).append(ex)

    t0 = time.perf_counter()
    correct = 0
    total = 0
    with open(DIAG_OUT, "w") as f:
        for group_K in sorted(examples_by_k):
            group = examples_by_k[group_K]
            for batch_start in range(0, len(group), BATCH):
                batch = group[batch_start:batch_start + BATCH]
                prompt_ids = [tok.encode(ex.problem).ids for ex in batch]
                gen_per_ex = gen_fn(model, prompt_ids, tok, K=group_K,
                                     fixed_len=FIXED_LEN, max_new=MAX_NEW)
                for i, ex in enumerate(batch):
                    gen_text = tok.decode(gen_per_ex[i])
                    parsed = parse_int_answer(gen_text)
                    ok = (parsed == ex.answer)
                    if ok:
                        correct += 1
                    total += 1
                    f.write(json.dumps({
                        "k": group_K,
                        "question": ex.problem,
                        "gold_answer": ex.answer,
                        "gold_steps": ex.gen_targets,
                        "gen_text": gen_text.strip(),
                        "parsed": parsed,
                        "ok": ok,
                    }) + "\n")
            print(f"  K={group_K}: dumped {len(group)}")

    dt = time.perf_counter() - t0
    acc = correct / max(total, 1) * 100
    print(f"\n=== {acc:.1f}% ({correct}/{total}) in {dt:.1f}s ===")
    print(f"=== {total} examples written to {DIAG_OUT} ===")


if __name__ == "__main__":
    main()
