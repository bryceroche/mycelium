"""v54-aligned eval — single cycle, K breaths, controller-decoded final answer.

The standard `accuracy_at_loops_multi` runs the model in multi-cycle eval
(autoregressive with breaths per generated token, using the standard ln_f +
embed_out decode chain). That's misaligned with v54's training paradigm:
  - v54 training: ONE forward through prompt + gen_targets, K=2 breaths,
    each breath's compressed waist decoded via the WaistController.
  - Misaligned eval: multi-cycle with breaths-per-token, ignores controller.

This script does the aligned version:
  1. For each example, encode prompt only.
  2. Greedy autoregressive: at each step, run main model + K breaths, take
     last breath's compressed waist, decode via controller, pick argmax at
     the current input's last position → next token. Append, repeat.
  3. Parse the generated text for an integer, compare to gold.

Env vars: same as the training script (PER_BREATH_DECODE=1, CONTROLLER_DECODE=1,
BFIELD_WAIST=512, BFIELD_END_OF_BREATH=1, NOTEBOOK_V24/REPLACE, etc.).
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


def _compile_jit_aligned_forward(model, K: int, fixed_len: int, B: int):
    """JIT'd forward: tokens → (B, T) argmax token IDs (last breath only).

    Argmax happens INSIDE the JIT — returns int token IDs not logits. Tiny
    host transfer regardless of K.
    """
    key = (id(model), int(K), int(fixed_len), int(B))
    if key in _EVAL_JIT_CACHE:
        return _EVAL_JIT_CACHE[key]

    @TinyJit
    def _fwd(tokens):
        _final, _mw, _pbx, waist_per_breath = model.breathe_with_lookup(
            tokens, n_loops=K, return_per_breath_x=True, return_waist_compressed=True)
        last_waist = waist_per_breath[-1].cast(dtypes.float)
        prompt_emb = model.embed(tokens).cast(dtypes.float)
        logits = model.waist_controller.forward(last_waist, prompt_emb, model.embed_out)
        return logits[:, :, :50277].argmax(axis=-1).realize()  # (B, T) int

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


def aligned_generate_batch(model, prompt_ids_list: list, K: int, fixed_len: int,
                            max_new: int = 12, eos_id: int = 0) -> list:
    """Batched aligned greedy decode via the controller.

    For each example: K breaths of main model on prompt → last waist →
    controller decode → next token at current last position.

    Returns: list of lists of generated token IDs (one per example).
    """
    B = len(prompt_ids_list)
    current_lens = [len(p) for p in prompt_ids_list]
    max_prompt = max(current_lens)
    assert max_prompt + max_new <= fixed_len, f"need fixed_len ≥ {max_prompt + max_new}"
    # Pad to fixed_len initially
    tokens_np = np.zeros((B, fixed_len), dtype=np.int32)
    for b in range(B):
        tokens_np[b, :current_lens[b]] = prompt_ids_list[b]
    generated_per_ex = [[] for _ in range(B)]
    active = [True] * B

    fwd = _compile_jit_aligned_forward(model, K, fixed_len, B)

    for _step in range(max_new):
        if not any(active):
            break
        tokens = Tensor(tokens_np, dtype=dtypes.int).realize()
        next_toks = fwd(tokens)  # (B, T) int — argmax inside JIT
        next_toks_np = next_toks.numpy()
        for b in range(B):
            if not active[b]:
                continue
            pos = current_lens[b] - 1
            next_tok = int(next_toks_np[b, pos])
            generated_per_ex[b].append(next_tok)
            current_lens[b] += 1
            if current_lens[b] < fixed_len:
                tokens_np[b, current_lens[b] - 1] = next_tok
            if next_tok == eos_id or current_lens[b] >= fixed_len:
                active[b] = False
    return generated_per_ex


def main():
    cfg = Config()
    CKPT = os.environ.get("CKPT", "")
    LEVEL = os.environ.get("LEVEL", "L4")
    NUM_EVAL = int(os.environ.get("NUM_EVAL", "100"))
    K = int(os.environ.get("K", "2"))
    FIXED_LEN = int(os.environ.get("FIXED_LEN", "96"))
    BATCH = int(os.environ.get("BATCH", "32"))
    MAX_NEW = int(os.environ.get("MAX_NEW", "12"))

    print(f"=== v54-aligned eval on {LEVEL} (K={K} breaths, controller decode) ===")
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
    print(f"\n=== running aligned eval (batched, B={BATCH}) ===")
    correct = 0
    total = 0
    t0 = time.perf_counter()
    samples_to_show = 3
    for batch_start in range(0, len(eval_examples), BATCH):
        batch = eval_examples[batch_start:batch_start + BATCH]
        prompt_ids = [tok.encode(ex.problem).ids for ex in batch]
        gen_per_ex = aligned_generate_batch(model, prompt_ids, K=K,
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
    print(f"\n=== aligned acc: {acc:.1f}% ({correct}/{total})  ({dt:.1f}s) ===")
    print(f"compared to standard misaligned eval on this ckpt (A=1/A=2 from training log)")


if __name__ == "__main__":
    main()
