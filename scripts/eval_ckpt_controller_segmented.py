"""Per-breath segmented decode — train/eval matched for the K-breath paradigm.

The standard aligned eval (eval_ckpt_controller_l4.py) uses the LAST breath's
waist to decode every generated token. But training supervises breath k at
step-k positions only. So at inference, we should use breath k's waist while
generating step-k tokens, then switch to breath k+1's waist after the `####`
boundary.

This script implements the matched-train/eval version:
  - Track current_step per example (starts at 0).
  - Generate each token using `waist_compressed_per_breath[current_step]`.
  - Increment current_step after detecting `####` in the decoded text.
  - Saturate at K-1 (don't index past the last available breath).

Env vars: same as the training script. Plus LEVEL (default L4.5), K (default 3),
MAX_NEW (default 80), FIXED_LEN (default 192).
"""
import os
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from tinygrad import Tensor, Device, dtypes
from tinygrad.engine.jit import TinyJit
from tinygrad.nn.state import safe_load

from mycelium import Config
from mycelium.loader import _load_state, load_breathing
from mycelium.data import load_tokenizer
from mycelium.l3_data import generate_math, split_train_eval, parse_int_answer


_EVAL_JIT_CACHE: dict = {}
_DECODE_JIT_CACHE: dict = {}


def _compile_jit_controller_decode_from_waist(model, K: int, max_len: int, B: int):
    """JIT'd controller decode used by the KV-cached path.

    Inputs:
      t_pos_t:           (B,) int — current position per example
      prompt_emb_buf:    (B, max_len, hidden) fp32 — full prompt + generated embeddings
      *waist_per_breath: K Tensors, each (B, max_len, waist_dim) fp32
    Returns:
      (K, B) int — argmax token ID for each breath at t_pos[b].

    The caller picks the right breath per example via current_step[b].
    """
    key = (id(model), int(K), int(max_len), int(B))
    if key in _DECODE_JIT_CACHE:
        return _DECODE_JIT_CACHE[key]

    @TinyJit
    def _decode(t_pos_t, prompt_emb_buf, *waist_per_breath):
        positions = Tensor.arange(max_len)
        gather_mask = (positions.reshape(1, max_len) == t_pos_t.reshape(B, 1)).reshape(B, max_len, 1).cast(dtypes.float)
        tokens_per_breath = []
        for k in range(K):
            wk_at_pos = (waist_per_breath[k] * gather_mask).sum(axis=1, keepdim=True)  # (B, 1, waist_dim)
            lk = model.waist_controller.forward(wk_at_pos, prompt_emb_buf, model.embed_out)
            tk = lk[:, :, :50277].argmax(axis=-1).reshape(B)
            tokens_per_breath.append(tk)
        return Tensor.stack(*tokens_per_breath, dim=0).realize()  # (K, B) int

    _DECODE_JIT_CACHE[key] = _decode
    return _decode


def _compile_jit_segmented_forward(model, K: int, fixed_len: int, B: int):
    """JIT'd forward: (tokens, t_pos_t) → (K, B) argmax token IDs at t_pos per example.

    Phase 1 optimization (2026-05-20): pass per-example t_pos as a Tensor input;
    gather waist[k] at t_pos[b] BEFORE the controller cross-attn. Controller
    runs at Q with 1 position (the current generation position), not T positions.
    This drops the controller's per-step compute by ~T× (224× at fixed_len=224).
    At K=4 the controller was the dominant cost (~3.2B ops/step), so this is
    the biggest single eval-time speedup before the KV-cache build.

    Stable inputs:
      tokens: (B, fixed_len) int
      t_pos_t: (B,) int — current position per example
    Returns: (K, B) int — argmax token id at t_pos[b] per breath per example.
    """
    key = (id(model), int(K), int(fixed_len), int(B))
    if key in _EVAL_JIT_CACHE:
        return _EVAL_JIT_CACHE[key]

    @TinyJit
    def _fwd(tokens, t_pos_t):
        _final, _mw, _pbx, waist_per_breath = model.breathe_with_lookup(
            tokens, n_loops=K, return_per_breath_x=True, return_waist_compressed=True)
        prompt_emb = model.embed(tokens).cast(dtypes.float)
        # Build per-example position mask once: (B, T, 1) one-hot at t_pos.
        positions = Tensor.arange(fixed_len)  # (T,)
        gather_mask = (positions.reshape(1, fixed_len) == t_pos_t.reshape(B, 1)).reshape(B, fixed_len, 1).cast(dtypes.float)
        tokens_per_breath = []
        for k in range(K):
            wk = waist_per_breath[k].cast(dtypes.float)  # (B, T, waist_dim)
            # Gather waist at t_pos[b]: zero out non-current positions, sum over T → (B, 1, waist_dim)
            wk_at_pos = (wk * gather_mask).sum(axis=1, keepdim=True)
            # Controller runs cross-attn with Q=1 position (vs T positions): T× cheaper.
            lk_at_pos = model.waist_controller.forward(wk_at_pos, prompt_emb, model.embed_out)
            # (B, 1, vocab) → argmax over active vocab → (B, 1) int
            tk = lk_at_pos[:, :, :50277].argmax(axis=-1).reshape(B)
            tokens_per_breath.append(tk)
        stacked = Tensor.stack(*tokens_per_breath, dim=0)  # (K, B) int
        return stacked.realize()

    _EVAL_JIT_CACHE[key] = _fwd
    return _fwd


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


def segmented_generate_kv_batch(model, prompt_ids_list, tok, K, fixed_len,
                                  max_new=80, eos_id=0):
    """KV-cached segmented decode via model.cached_generate_segmented.

    Uses the new prefill/decode split: K breaths on the prompt build KV cache +
    capture waist at all prompt positions; per-token decode runs K breaths over
    just the new position using cached K/V. Controller decode runs at Q-position=1
    (Phase 1 optimization).

    Returns: list of lists of generated token IDs (one per example).
    """
    B = len(prompt_ids_list)
    max_prompt = max(len(p) for p in prompt_ids_list)
    assert max_prompt + max_new <= fixed_len, f"need fixed_len ≥ {max_prompt + max_new}"

    # Track per-example prompt lengths so we measure #### only against generated tokens.
    prompt_lens = [len(p) for p in prompt_ids_list]

    def decode_fn(argmax_per_breath, t_pos_per, decoded_so_far):
        # argmax_per_breath: Tensor (K, B) int — argmax token IDs per breath per example
        # Determine current_step per example from #### in tokens generated SINCE the prompt.
        current_step = [0] * B
        for b in range(B):
            n_hash = tok.decode(decoded_so_far[b][prompt_lens[b]:]).count("####")
            current_step[b] = min(n_hash, K - 1)
        stacked_np = argmax_per_breath.numpy()
        next_ids = np.zeros(B, dtype=np.int32)
        for b in range(B):
            next_ids[b] = stacked_np[current_step[b], b]
        return next_ids

    outs = model.cached_generate_segmented(
        prompt_ids_list, n_loops=K, max_new=max_new,
        decode_fn=decode_fn,
        stop_token_ids=[eos_id],
        cache_max_len=fixed_len,
    )
    # outs is list[list[int]] of generated tokens (excluding prompt).
    return outs


def segmented_generate_batch(model, prompt_ids_list, tok, K, fixed_len,
                              max_new=80, eos_id=0):
    """Per-breath segmented greedy decode.

    For each example, maintain a current_step counter starting at 0. At each
    generation position, decode using `waist_compressed_per_breath[current_step]`.
    Increment current_step after the running decoded text contains another
    occurrence of `####` (saturate at K-1).

    Returns: list of lists of generated token IDs (one per example).
    """
    B = len(prompt_ids_list)
    current_lens = [len(p) for p in prompt_ids_list]
    max_prompt = max(current_lens)
    assert max_prompt + max_new <= fixed_len, f"need fixed_len ≥ {max_prompt + max_new}"
    tokens_np = np.zeros((B, fixed_len), dtype=np.int32)
    for b in range(B):
        tokens_np[b, :current_lens[b]] = prompt_ids_list[b]
    generated_per_ex = [[] for _ in range(B)]
    active = [True] * B
    # Per-example step index (0..K-1) and #### count seen so far.
    current_step = [0] * B
    hashes_seen = [0] * B

    fwd = _compile_jit_segmented_forward(model, K, fixed_len, B)
    # Persistent t_pos buffer; updated in-place each step.
    t_pos_np = np.zeros((B,), dtype=np.int32)
    t_pos_t = Tensor(t_pos_np, dtype=dtypes.int).contiguous().realize()

    for _step in range(max_new):
        if not any(active):
            break
        # Update t_pos: each example's CURRENT last position (predicts next token).
        for b in range(B):
            t_pos_np[b] = current_lens[b] - 1
        t_pos_t.assign(Tensor(t_pos_np, dtype=dtypes.int)).realize()
        tokens = Tensor(tokens_np, dtype=dtypes.int).realize()
        stacked = fwd(tokens, t_pos_t)  # (K, B) int — argmax at t_pos per example
        stacked_np = stacked.numpy()
        for b in range(B):
            if not active[b]:
                continue
            k_b = min(current_step[b], K - 1)
            next_tok = int(stacked_np[k_b, b])
            generated_per_ex[b].append(next_tok)
            current_lens[b] += 1
            if current_lens[b] < fixed_len:
                tokens_np[b, current_lens[b] - 1] = next_tok
            # Detect step boundary: decode running text and count `####` occurrences.
            decoded = tok.decode(generated_per_ex[b])
            n_hash = decoded.count("####")
            if n_hash > hashes_seen[b]:
                hashes_seen[b] = n_hash
                current_step[b] = min(current_step[b] + 1, K - 1)
            if next_tok == eos_id or current_lens[b] >= fixed_len:
                active[b] = False
    return generated_per_ex


def main():
    # Config overrides via env vars (must match the trained ckpt's dims).
    cfg_kwargs = {}
    for env_key, cfg_key in [("HIDDEN", "hidden"), ("N_HEADS", "n_heads"),
                              ("HEAD_DIM", "head_dim"), ("FFN", "ffn"),
                              ("CONTROLLER_HIDDEN", "controller_hidden")]:
        v = os.environ.get(env_key)
        if v is not None:
            cfg_kwargs[cfg_key] = int(v)
    cfg = Config(**cfg_kwargs)
    if cfg_kwargs:
        print(f"[Config overrides] {cfg_kwargs}")
    CKPT = os.environ.get("CKPT", "")
    LEVEL = os.environ.get("LEVEL", "L4.5")
    NUM_EVAL = int(os.environ.get("NUM_EVAL", "100"))
    K = int(os.environ.get("K", "3"))
    FIXED_LEN = int(os.environ.get("FIXED_LEN", "192"))
    BATCH = int(os.environ.get("BATCH", "16"))
    MAX_NEW = int(os.environ.get("MAX_NEW", "80"))
    USE_KV_CACHE = bool(int(os.environ.get("USE_KV_CACHE", "1")))

    print(f"=== per-breath segmented eval on {LEVEL} (K={K} breaths, breath-k decodes step-k) ===")
    print(f"  ckpt: {CKPT}")
    print(f"  num_eval: {NUM_EVAL}  fixed_len: {FIXED_LEN}  batch: {BATCH}  max_new: {MAX_NEW}  kv_cache: {USE_KV_CACHE}")

    if LEVEL == "GSM8K_STEPS":
        # GSM8K with Haiku-distilled per-step targets. K varies per example;
        # we'll bucket the eval and run per-K below. Path defaults to the train
        # JSONL but can be overridden (typically the test split for held-out eval).
        from mycelium.l3_data import load_gsm8k_steps
        path = os.environ.get("GSM8K_STEPS_PATH", ".cache/gsm8k_steps_v1_test.jsonl")
        max_k = int(os.environ.get("GSM8K_STEPS_MAX_K", "6"))
        min_k = int(os.environ.get("GSM8K_STEPS_MIN_K", "2"))
        print(f"\nloading GSM8K-steps eval from {path} (K range {min_k}..{max_k})...")
        eval_buckets = load_gsm8k_steps(path, min_k=min_k, max_k=max_k, bucket_by_k=True)
        # Cap per-bucket eval to NUM_EVAL/n_buckets so total ≈ NUM_EVAL.
        n_buckets = len(eval_buckets)
        per_bucket = max(1, NUM_EVAL // max(1, n_buckets))
        eval_examples = []
        for k in sorted(eval_buckets):
            eval_examples.extend(eval_buckets[k][:per_bucket])
        print(f"  loaded {len(eval_examples)} examples across K={sorted(eval_buckets)}")
    else:
        print(f"\ngenerating {LEVEL} eval set (seed=42 for split parity)...")
        all_examples = generate_math(LEVEL, 20000, seed=42, digit_spacing=True)
        _, eval_examples = split_train_eval(all_examples, n_eval=NUM_EVAL, seed=42)
    tok = load_tokenizer()

    print(f"\nloading Pythia + ckpt...")
    sd = _load_state()
    model = load_breathing(cfg, sd=sd)
    cast_model_fp32(model)
    Device[Device.DEFAULT].synchronize()
    del sd
    if not CKPT:
        print("WARNING: CKPT not set — evaluating untrained model.")
    else:
        ckpt_sd = safe_load(CKPT)
        info = model.load_state_dict(ckpt_sd, strict=False)
        print(f"  loaded; missing {len(info['missing'])}, unexpected {len(info['unexpected'])}")
        del ckpt_sd

    Tensor.training = False
    print(f"\n=== running segmented eval (batched, B={BATCH}) ===")
    correct = 0
    total = 0
    per_k_stats: dict[int, tuple[int, int]] = {}  # K → (correct, total)
    t0 = time.perf_counter()
    samples_to_show = 3
    gen_fn = segmented_generate_kv_batch if USE_KV_CACHE else segmented_generate_batch

    # For GSM8K_STEPS, group by K so each batch is uniform-K (per_breath / cached JIT
    # caches per K and would recompile mid-batch otherwise).
    if LEVEL == "GSM8K_STEPS":
        examples_by_k: dict[int, list] = {}
        for ex in eval_examples:
            examples_by_k.setdefault(len(ex.gen_targets), []).append(ex)
        eval_groups = [(k, examples_by_k[k]) for k in sorted(examples_by_k)]
    else:
        eval_groups = [(K, eval_examples)]

    for group_K, group_examples in eval_groups:
        group_correct, group_total = 0, 0
        for batch_start in range(0, len(group_examples), BATCH):
            batch = group_examples[batch_start:batch_start + BATCH]
            prompt_ids = [tok.encode(ex.problem).ids for ex in batch]
            gen_per_ex = gen_fn(model, prompt_ids, tok, K=group_K,
                                 fixed_len=FIXED_LEN, max_new=MAX_NEW)
            for i, ex in enumerate(batch):
                gen_text = tok.decode(gen_per_ex[i])
                parsed = parse_int_answer(gen_text)
                ok = (parsed == ex.answer)
                if ok:
                    correct += 1
                    group_correct += 1
                total += 1
                group_total += 1
                if samples_to_show > 0 and batch_start == 0 and i < samples_to_show:
                    print(f"  [K={group_K}] Q: {ex.problem[:80]!r}")
                    print(f"  gen: {gen_text.strip()!r}")
                    print(f"  parsed: {parsed}, gold: {ex.answer}, {'OK' if ok else 'WRONG'}")
                    samples_to_show -= 1
        per_k_stats[group_K] = (group_correct, group_total)
    dt = time.perf_counter() - t0
    acc = correct / max(total, 1) * 100
    print(f"\n=== segmented acc: {acc:.1f}% ({correct}/{total})  ({dt:.1f}s) ===")
    if len(per_k_stats) > 1:
        print(f"per-K breakdown:")
        for k, (c, t) in sorted(per_k_stats.items()):
            pct = c / max(t, 1) * 100
            print(f"  K={k}: {pct:.1f}% ({c}/{t})")


if __name__ == "__main__":
    main()
