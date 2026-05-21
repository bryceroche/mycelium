"""CFG-style prompt amplification at inference. Runs WaistController twice per
token: once with normal prompt embeddings (cond) and once with prompt embeddings
zeroed (uncond). Combines: logits_cfg = uncond + alpha * (cond - uncond).

Env: same as eval script. Plus CFG_ALPHA (default 1.5), DIAG_OUT (default
/tmp/cfg_eval.jsonl).
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


_JIT_CACHE: dict = {}


def _compile_cfg_forward(model, K: int, fixed_len: int, B: int, alpha: float):
    key = (id(model), K, fixed_len, B, alpha)
    if key in _JIT_CACHE:
        return _JIT_CACHE[key]
    print(f"[JIT] compile CFG forward: K={K} B={B} fixed_len={fixed_len} alpha={alpha}", flush=True)

    @TinyJit
    def _fwd(tokens, t_pos_t):
        _final, _mw, _pbx, waist_per_breath = model.breathe_with_lookup(
            tokens, n_loops=K, return_per_breath_x=True, return_waist_compressed=True)
        prompt_emb = model.embed(tokens).cast(dtypes.float)            # (B, T, H)
        prompt_emb_zero = Tensor.zeros_like(prompt_emb)                # uncond input
        positions = Tensor.arange(fixed_len)
        gather_mask = (positions.reshape(1, fixed_len) == t_pos_t.reshape(B, 1)).reshape(B, fixed_len, 1).cast(dtypes.float)
        per_breath_argmax = []
        for k in range(K):
            wk = waist_per_breath[k].cast(dtypes.float)
            wk_at_pos = (wk * gather_mask).sum(axis=1, keepdim=True)   # (B, 1, waist_dim)
            l_cond   = model.waist_controller.forward(wk_at_pos, prompt_emb,      model.embed_out)
            l_uncond = model.waist_controller.forward(wk_at_pos, prompt_emb_zero, model.embed_out)
            # CFG combine: uncond + alpha * (cond - uncond)
            l_cfg = l_uncond + alpha * (l_cond - l_uncond)
            tk = l_cfg[:, :, :50277].argmax(axis=-1).reshape(B)
            per_breath_argmax.append(tk)
        stacked = Tensor.stack(*per_breath_argmax, dim=0)              # (K, B) int
        return stacked.realize()

    _JIT_CACHE[key] = _fwd
    return _fwd


def cfg_generate_batch(model, prompt_ids_list, tok, K, fixed_len, alpha,
                        max_new=120, eos_id=0):
    """CFG-amplified segmented decode. Two controller forward passes per token."""
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
    prompt_lens = current_lens[:]

    fwd = _compile_cfg_forward(model, K, fixed_len, B, alpha)
    t_pos_np = np.zeros((B,), dtype=np.int32)
    t_pos_t = Tensor(t_pos_np, dtype=dtypes.int).contiguous().realize()

    for _step in range(max_new):
        if not any(active):
            break
        for b in range(B):
            t_pos_np[b] = current_lens[b] - 1
        t_pos_t.assign(Tensor(t_pos_np, dtype=dtypes.int)).realize()
        tokens = Tensor(tokens_np, dtype=dtypes.int).realize()
        stacked = fwd(tokens, t_pos_t)
        stacked_np = stacked.numpy()
        for b in range(B):
            if not active[b]:
                continue
            k_b = min(current_step[b], K - 1)
            next_tok = int(stacked_np[k_b, b])
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
            # Check #### count in generated tokens (since prompt) for step advancement
            decoded = tok.decode(generated_per_ex[b])
            n_hash = decoded.count("####")
            if n_hash > hashes_seen[b]:
                hashes_seen[b] = n_hash
                current_step[b] = min(n_hash, K - 1)
                if n_hash >= K:
                    active[b] = False

    return generated_per_ex


def main():
    CKPT = os.environ.get("CKPT", "")
    LEVEL = os.environ.get("LEVEL", "GSM8K_STEPS")
    NUM_EVAL = int(os.environ.get("NUM_EVAL", "100"))
    FIXED_LEN = int(os.environ.get("FIXED_LEN", "400"))
    MAX_NEW = int(os.environ.get("MAX_NEW", "120"))
    BATCH = int(os.environ.get("BATCH", "2"))
    CFG_ALPHA = float(os.environ.get("CFG_ALPHA", "1.5"))
    DIAG_OUT = os.environ.get("DIAG_OUT", "/tmp/cfg_eval.jsonl")

    print(f"=== CFG eval on {LEVEL}, alpha={CFG_ALPHA} ===")
    print(f"  ckpt: {CKPT}")
    print(f"  num_eval: {NUM_EVAL}  fixed_len: {FIXED_LEN}  max_new: {MAX_NEW}")

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
                gen_per_ex = cfg_generate_batch(model, prompt_ids, tok, K=K,
                                                  fixed_len=FIXED_LEN, alpha=CFG_ALPHA,
                                                  max_new=MAX_NEW)
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
                        "alpha": CFG_ALPHA,
                    }) + "\n")
            per_k[K] = (gc, gt)
            print(f"  K={K}: {gc}/{gt}")

    dt = time.perf_counter() - t0
    acc = correct / max(total, 1) * 100
    print(f"\n=== CFG α={CFG_ALPHA} acc: {acc:.1f}% ({correct}/{total})  ({dt:.1f}s) ===")
    print("per-K breakdown:")
    for k, (c, t) in sorted(per_k.items()):
        pct = c / max(t, 1) * 100
        print(f"  K={k}: {pct:.1f}% ({c}/{t})")


if __name__ == "__main__":
    main()
