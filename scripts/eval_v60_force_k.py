"""Force-K decode: mask `####` token (id 1835) logits to -inf until segment count
reaches K-1. Tests "if we FORCE the model to emit K segments, how good is content?"

Result determines overnight strategy:
- > 15% → architecture fix on 410M (segment timing is the bottleneck)
- < 5%  → capacity bet on v62 Pythia-1B
- 5-15% → hybrid

Env: same as eval_ckpt_controller_segmented. Plus DIAG_OUT (default
/tmp/force_k_eval.jsonl).
"""
import os
import sys
import json
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from tinygrad import Tensor, Device, dtypes
from tinygrad.engine.jit import TinyJit
from tinygrad.nn.state import safe_load

from mycelium import Config
from mycelium.loader import _load_state, load_breathing
from mycelium.data import load_tokenizer
from mycelium.l3_data import generate_math, split_train_eval, parse_int_answer, load_gsm8k_steps
from scripts.eval_ckpt_controller_segmented import cast_model_fp32


HASH_TOKEN_ID = 1835  # "####"
_JIT_CACHE: dict = {}


def _compile_jit_logits_at_pos(model, K: int, fixed_len: int, B: int):
    """Forward returning per-breath LOGITS at the t_pos position (instead of argmax).
    Shape: (K, B, vocab_active=50277). Caller masks `####` and argmaxes in Python.
    """
    key = (id(model), K, fixed_len, B)
    if key in _JIT_CACHE:
        return _JIT_CACHE[key]
    print(f"[JIT] compile force-K forward: K={K} B={B} fixed_len={fixed_len}", flush=True)

    @TinyJit
    def _fwd(tokens, t_pos_t):
        _final, _mw, _pbx, waist_per_breath = model.breathe_with_lookup(
            tokens, n_loops=K, return_per_breath_x=True, return_waist_compressed=True)
        prompt_emb = model.embed(tokens).cast(dtypes.float)
        positions = Tensor.arange(fixed_len)
        gather_mask = (positions.reshape(1, fixed_len) == t_pos_t.reshape(B, 1)).reshape(B, fixed_len, 1).cast(dtypes.float)
        per_breath_logits = []
        for k in range(K):
            wk = waist_per_breath[k].cast(dtypes.float)
            wk_at_pos = (wk * gather_mask).sum(axis=1, keepdim=True)
            lk_at_pos = model.waist_controller.forward(wk_at_pos, prompt_emb, model.embed_out,
                                                         k_idx=k, K_total=K)
            # (B, 1, vocab) → (B, vocab_active)
            per_breath_logits.append(lk_at_pos[:, 0, :50277])
        stacked = Tensor.stack(*per_breath_logits, dim=0)  # (K, B, 50277)
        return stacked.realize()

    _JIT_CACHE[key] = _fwd
    return _fwd


def force_k_generate_batch(model, prompt_ids_list, tok, K, fixed_len,
                              max_new=120, eos_id=0):
    """Greedy decode with `####` token masked until segment count reaches K-1."""
    B = len(prompt_ids_list)
    current_lens = [len(p) for p in prompt_ids_list]
    max_prompt = max(current_lens)
    assert max_prompt + max_new <= fixed_len, f"need fixed_len ≥ {max_prompt + max_new}"

    tokens_np = np.zeros((B, fixed_len), dtype=np.int32)
    for b in range(B):
        tokens_np[b, :current_lens[b]] = prompt_ids_list[b]
    generated_per_ex = [[] for _ in range(B)]
    active = [True] * B
    current_step = [0] * B
    hashes_seen = [0] * B
    step_tokens_since_hash = [0] * B  # tokens emitted since last #### (or start of generation)

    fwd = _compile_jit_logits_at_pos(model, K, fixed_len, B)
    t_pos_np = np.zeros((B,), dtype=np.int32)
    t_pos_t = Tensor(t_pos_np, dtype=dtypes.int).contiguous().realize()

    for _step in range(max_new):
        if not any(active):
            break
        for b in range(B):
            t_pos_np[b] = current_lens[b] - 1
        t_pos_t.assign(Tensor(t_pos_np, dtype=dtypes.int)).realize()
        tokens = Tensor(tokens_np, dtype=dtypes.int).realize()
        # (K, B, vocab_active)
        logits_np = fwd(tokens, t_pos_t).numpy()

        for b in range(B):
            if not active[b]:
                continue
            k_b = min(current_step[b], K - 1)
            # Get logits for this breath, this example
            l = logits_np[k_b, b].copy()  # (vocab_active,)
            # Force-K: BIAS #### logit upward proportional to remaining segments needed.
            # If we need K-1 segments and have emitted h, boost #### by β × (K-1-h).
            # Combined with min segment length: only allow #### if we've emitted
            # >= MIN_SEG_TOKENS tokens since the last ####.
            remaining = (K - 1) - hashes_seen[b]
            tokens_since_hash = step_tokens_since_hash[b]
            MIN_SEG_TOKENS = 10  # minimum tokens per segment before #### allowed
            if remaining > 0 and tokens_since_hash >= MIN_SEG_TOKENS:
                # Bias by 5 × remaining (strong push). Each remaining segment adds 5 to logit.
                l[HASH_TOKEN_ID] += 5.0 * remaining
            elif tokens_since_hash < MIN_SEG_TOKENS:
                # Suppress premature #### within current segment
                l[HASH_TOKEN_ID] = -1e9
            next_tok = int(np.argmax(l))
            generated_per_ex[b].append(next_tok)
            if current_lens[b] < fixed_len:
                tokens_np[b, current_lens[b]] = next_tok
                current_lens[b] += 1
            else:
                active[b] = False
                continue
            if next_tok == eos_id:
                active[b] = False
                continue
            # Detect #### emission
            if next_tok == HASH_TOKEN_ID:
                hashes_seen[b] += 1
                step_tokens_since_hash[b] = 0
                current_step[b] = min(hashes_seen[b], K - 1)
                if hashes_seen[b] >= K:
                    active[b] = False
            else:
                step_tokens_since_hash[b] += 1

    return generated_per_ex


def main():
    CKPT = os.environ.get("CKPT", "")
    LEVEL = os.environ.get("LEVEL", "GSM8K_STEPS")
    NUM_EVAL = int(os.environ.get("NUM_EVAL", "100"))
    FIXED_LEN = int(os.environ.get("FIXED_LEN", "400"))
    MAX_NEW = int(os.environ.get("MAX_NEW", "120"))
    BATCH = int(os.environ.get("BATCH", "2"))
    DIAG_OUT = os.environ.get("DIAG_OUT", "/tmp/force_k_eval.jsonl")

    print(f"=== Force-K eval on {LEVEL} ===")
    print(f"  ckpt: {CKPT}")
    print(f"  num_eval: {NUM_EVAL}  fixed_len: {FIXED_LEN}  max_new: {MAX_NEW}")
    print(f"  #### token id: {HASH_TOKEN_ID}")

    if LEVEL == "GSM8K_STEPS":
        path = os.environ.get("GSM8K_STEPS_PATH", ".cache/gsm8k_steps_v1_test.jsonl")
        min_k = int(os.environ.get("GSM8K_STEPS_MIN_K", "2"))
        max_k = int(os.environ.get("GSM8K_STEPS_MAX_K", "6"))
        all_examples = load_gsm8k_steps(path, min_k=min_k, max_k=max_k, bucket_by_k=False)
        _, eval_examples = split_train_eval(all_examples, n_eval=NUM_EVAL, seed=42)
    else:
        all_examples = generate_math(LEVEL, 20000, seed=42, digit_spacing=True)
        _, eval_examples = split_train_eval(all_examples, n_eval=NUM_EVAL, seed=42)

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
    tok = load_tokenizer()

    examples_by_k: dict[int, list] = {}
    for ex in eval_examples:
        examples_by_k.setdefault(len(ex.gen_targets), []).append(ex)

    t0 = time.perf_counter()
    correct = 0
    total = 0
    per_k = {}
    with open(DIAG_OUT, "w") as f:
        for K in sorted(examples_by_k):
            group = examples_by_k[K]
            gc, gt = 0, 0
            for batch_start in range(0, len(group), BATCH):
                batch = group[batch_start:batch_start + BATCH]
                prompt_ids = [tok.encode(ex.problem).ids for ex in batch]
                gen_per_ex = force_k_generate_batch(model, prompt_ids, tok, K=K,
                                                      fixed_len=FIXED_LEN, max_new=MAX_NEW)
                for i, ex in enumerate(batch):
                    gen_text = tok.decode(gen_per_ex[i])
                    parsed = parse_int_answer(gen_text)
                    ok = (parsed == ex.answer)
                    if ok:
                        correct += 1; gc += 1
                    total += 1; gt += 1
                    f.write(json.dumps({
                        "k": K, "question": ex.problem, "gold_answer": ex.answer,
                        "gen_text": gen_text.strip(), "parsed": parsed, "ok": ok,
                    }) + "\n")
            per_k[K] = (gc, gt)
            print(f"  K={K}: {gc}/{gt}")

    dt = time.perf_counter() - t0
    acc = correct / max(total, 1) * 100
    print(f"\n=== Force-K acc: {acc:.1f}% ({correct}/{total})  ({dt:.1f}s) ===")
    print("per-K breakdown:")
    for k, (c, t) in sorted(per_k.items()):
        pct = c / max(t, 1) * 100
        print(f"  K={k}: {pct:.1f}% ({c}/{t})")


if __name__ == "__main__":
    main()
