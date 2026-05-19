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


def _compile_jit_segmented_forward(model, K: int, fixed_len: int, B: int):
    """JIT'd forward: tokens → (K, B, T) argmax token IDs.

    Argmax happens INSIDE the JIT — returns int token IDs not logits. This is
    ~25,000× less host transfer than returning the (K, B, T, vocab) tensor
    (50304 → 1 per position). Required for scaling beyond K=4: at K=8 the
    full-logits return is ~13GB; argmax-in-JIT brings it to ~1MB.

    Stable inputs: tokens shape (B, fixed_len). Returns (K, B, fixed_len) int.
    """
    key = (id(model), int(K), int(fixed_len), int(B))
    if key in _EVAL_JIT_CACHE:
        return _EVAL_JIT_CACHE[key]

    @TinyJit
    def _fwd(tokens):
        _final, _mw, _pbx, waist_per_breath = model.breathe_with_lookup(
            tokens, n_loops=K, return_per_breath_x=True, return_waist_compressed=True)
        prompt_emb = model.embed(tokens).cast(dtypes.float)
        tokens_per_breath = []
        for k in range(K):
            wk = waist_per_breath[k].cast(dtypes.float)
            lk = model.waist_controller.forward(wk, prompt_emb, model.embed_out)
            # Slice to active vocab, then argmax inside the JIT. Output is int32.
            tk = lk[:, :, :50277].argmax(axis=-1)  # (B, T) int
            tokens_per_breath.append(tk)
        stacked = Tensor.stack(*tokens_per_breath, dim=0)  # (K, B, T) int
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

    for _step in range(max_new):
        if not any(active):
            break
        tokens = Tensor(tokens_np, dtype=dtypes.int).realize()
        stacked = fwd(tokens)  # (K, B, T) int — argmax inside JIT
        stacked_np = stacked.numpy()
        for b in range(B):
            if not active[b]:
                continue
            k_b = min(current_step[b], K - 1)
            pos = current_lens[b] - 1
            next_tok = int(stacked_np[k_b, b, pos])
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
    cfg = Config()
    CKPT = os.environ.get("CKPT", "")
    LEVEL = os.environ.get("LEVEL", "L4.5")
    NUM_EVAL = int(os.environ.get("NUM_EVAL", "100"))
    K = int(os.environ.get("K", "3"))
    FIXED_LEN = int(os.environ.get("FIXED_LEN", "192"))
    BATCH = int(os.environ.get("BATCH", "16"))
    MAX_NEW = int(os.environ.get("MAX_NEW", "80"))

    print(f"=== per-breath segmented eval on {LEVEL} (K={K} breaths, breath-k decodes step-k) ===")
    print(f"  ckpt: {CKPT}")
    print(f"  num_eval: {NUM_EVAL}  fixed_len: {FIXED_LEN}  batch: {BATCH}  max_new: {MAX_NEW}")

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
    t0 = time.perf_counter()
    samples_to_show = 3
    for batch_start in range(0, len(eval_examples), BATCH):
        batch = eval_examples[batch_start:batch_start + BATCH]
        prompt_ids = [tok.encode(ex.problem).ids for ex in batch]
        gen_per_ex = segmented_generate_batch(model, prompt_ids, tok, K=K,
                                                fixed_len=FIXED_LEN, max_new=MAX_NEW)
        for i, ex in enumerate(batch):
            gen_text = tok.decode(gen_per_ex[i])
            parsed = parse_int_answer(gen_text)
            ok = (parsed == ex.answer)
            if ok:
                correct += 1
            total += 1
            if samples_to_show > 0 and batch_start == 0 and i < samples_to_show:
                print(f"  Q: {ex.problem[:80]!r}")
                print(f"  gen: {gen_text.strip()!r}")
                print(f"  parsed: {parsed}, gold: {ex.answer}, {'OK' if ok else 'WRONG'}")
                samples_to_show -= 1
    dt = time.perf_counter() - t0
    acc = correct / max(total, 1) * 100
    print(f"\n=== segmented acc: {acc:.1f}% ({correct}/{total})  ({dt:.1f}s) ===")


if __name__ == "__main__":
    main()
