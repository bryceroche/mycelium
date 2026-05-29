"""v81 DAG eval — patched to use SINGLE-HEAD decode (force_single_head=True).

Diagnosis (2026-05-27): v81's multi-head WaistController emits 4 logit tensors
per breath, but each head was supervised ONLY at its specific position range
(ops at the start of the answer span, types in the middle, args1/args2 toward
the end). At positions OUTSIDE each head's supervised range, that head produces
unconstrained output — which means decoding ALL heads at the SAME positions
(the original eval_v81_dag.py approach A) reads from heads at positions they
were never trained on.

The shared cross-attention backbone, however, learned the FULL B6 structure
(ops|types|args1|args2 pattern + content) because every head's CE gradient
flowed through it. When we set `force_single_head=True`, the WaistController
skips the per-head bias and uses the shared backbone's pure prediction.

Empirically (diag_v81_per_breath_decode.py on v81_prod_step2000.safetensors):
- Multi-head approach A (current eval_v81): 0/20 parse, 0/20 acc
- Single-head approach:                      ~35% parse, 0% acc (model under-converged)

Per-head training-time CE is misleadingly low (ops=0.30, types=0.33, args1=0.43,
args2=0.54) because the supervised positions are mostly STRUCTURAL ("," "|" digits)
that any head can predict. The model is undertrained on actual CONTENT — at most
problems, per-position content acc is 33-75%, far below the ~95% needed for joint
correctness of a 4-step DAG.

This script still implements an AR loop because the single-head path also benefits
from autoregressive decoding (each step the model sees its previously-emitted
tokens). The cross-attn / notebook masks remain prompt-range only (since v81 was
trained that way).

Env vars:
    CKPT=...           required. Path to the v81 ckpt.
    V77_TEST_PATH=...  test set JSONL.
    NUM_EVAL=60        max problems to eval.
    K=7                inner breaths (matches training).
    FIXED_LEN=320      input padding.
    BATCH=8            eval batch size.
    MAX_NEW=120        max tokens to AR-decode per problem.
"""
import os
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from tinygrad import Tensor, Device, dtypes
from tinygrad.engine.jit import TinyJit
from tinygrad.nn.state import safe_load

from mycelium import Config
from mycelium.loader import _load_state, load_breathing
from mycelium.data import load_tokenizer
from mycelium.l3_data import load_gsm8k_v77

from v81_sympy_eval import b6_string_to_dag, dag_to_answer  # type: ignore


_EVAL_JIT_CACHE: dict = {}


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


def _compile_jit_single_head_step(model, K: int, fixed_len: int, B: int):
    """JIT compile: K breaths once, then SINGLE-HEAD decode at every position.

    Returns argmax token ids per position (B, fixed_len) — caller picks the relevant
    `t_pos` for AR generation.

    Single-head path: model.waist_controller.forward(..., force_single_head=True)
    skips the per-head bias and uses the shared cross-attn backbone's pure output.

    Inputs:
      tokens   (B, fixed_len) int     — input with zero-filled answer span
      kv_mask  (B, fixed_len) float   — 1.0 at prompt positions, 0.0 elsewhere
      t_pos    (B,)          int     — position whose argmax to return
    Outputs:
      next_ids (B,) int — argmax of single-head logits at t_pos per example.
    """
    key = (id(model), int(K), int(fixed_len), int(B))
    if key in _EVAL_JIT_CACHE:
        return _EVAL_JIT_CACHE[key]

    @TinyJit
    def _fwd(tokens, t_pos_t, kv_mask):
        from mycelium.breathing import V81_MAIN_ATTN_MASK as _V81MAM
        _main_attn = kv_mask if _V81MAM else None
        _final, _mw, _pbx, waist_per_breath = model.breathe_with_lookup(
            tokens, n_loops=K, return_per_breath_x=True, return_waist_compressed=True,
            notebook_pool_mask=kv_mask, main_attn_mask=_main_attn)
        prompt_emb = model.embed(tokens).cast(dtypes.float)
        wk = waist_per_breath[K - 1].cast(dtypes.float)
        # Gather waist at t_pos (per-example) instead of computing logits at every position.
        positions = Tensor.arange(fixed_len)
        gather_mask = (positions.reshape(1, fixed_len) == t_pos_t.reshape(B, 1)).reshape(B, fixed_len, 1).cast(dtypes.float)
        wk_at_pos = (wk * gather_mask).sum(axis=1, keepdim=True)
        # SINGLE-HEAD path — bypasses per-head bias.
        lk = model.waist_controller.forward(
            wk_at_pos, prompt_emb, model.embed_out,
            k_idx=K - 1, K_total=K,
            kv_mask=kv_mask,
            force_single_head=True)
        # lk is a Tensor when force_single_head=True
        return lk[:, :, :50277].argmax(axis=-1).reshape(B).realize()

    _EVAL_JIT_CACHE[key] = _fwd
    return _fwd


def generate_single_head_batch(model, prompt_ids_list, tok, K, fixed_len,
                                max_new=120, eos_id=0):
    """Greedy decode using single-head path. Each AR step regenerates the model's
    forward (the masking holds the answer span out anyway, so the model's prediction
    at position t_pos only depends on the prompt). For efficiency we still do a full
    forward each step — this could be optimized later with a KV-cache, but for the
    initial fix-validation we want clarity over speed.

    Returns: list of lists of generated token IDs (one per example, excluding prompt).
    """
    B = len(prompt_ids_list)
    prompt_lens = [len(p) for p in prompt_ids_list]
    current_lens = list(prompt_lens)
    max_prompt = max(current_lens)
    assert max_prompt + max_new <= fixed_len, (
        f"need fixed_len >= {max_prompt + max_new}, got {fixed_len}")
    tokens_np = np.zeros((B, fixed_len), dtype=np.int32)
    # kv_mask covers ONLY the prompt range — matches training-time v81 masking.
    kv_mask_np = np.zeros((B, fixed_len), dtype=np.float32)
    for b in range(B):
        tokens_np[b, :prompt_lens[b]] = prompt_ids_list[b]
        kv_mask_np[b, :prompt_lens[b]] = 1.0
    generated_per_ex = [[] for _ in range(B)]
    active = [True] * B

    fwd = _compile_jit_single_head_step(model, K, fixed_len, B)
    t_pos_np = np.zeros((B,), dtype=np.int32)
    t_pos_t = Tensor(t_pos_np, dtype=dtypes.int).contiguous().realize()
    kv_mask_t = Tensor(kv_mask_np, dtype=dtypes.float).contiguous().realize()

    for _step in range(max_new):
        if not any(active):
            break
        for b in range(B):
            t_pos_np[b] = current_lens[b] - 1
        t_pos_t.assign(Tensor(t_pos_np, dtype=dtypes.int)).realize()
        tokens = Tensor(tokens_np, dtype=dtypes.int).realize()
        next_ids_t = fwd(tokens, t_pos_t, kv_mask_t)
        next_ids_np = next_ids_t.numpy()
        for b in range(B):
            if not active[b]:
                continue
            next_tok = int(next_ids_np[b])
            generated_per_ex[b].append(next_tok)
            current_lens[b] += 1
            if current_lens[b] < fixed_len:
                tokens_np[b, current_lens[b] - 1] = next_tok
            if next_tok == eos_id or current_lens[b] >= fixed_len:
                active[b] = False
    return generated_per_ex


def extract_b6(text: str) -> str:
    """Take the generated text and truncate after the first 4 pipe-separated segments.

    The model often emits 5+ pipes (because it never learned to STOP after the 4th
    list). Truncate to the first 4 segments. Whitespace-normalize each segment.
    """
    text = text.strip()
    # Split on " |" (with leading space) — this is the canonical separator since
    # head_prefixes during training were " " + ops then " | " + each subsequent list.
    segments = text.split(" |")
    if len(segments) < 4:
        # Fall back to splitting on "|" (no leading space).
        segments = text.split("|")
    if len(segments) < 4:
        return text  # can't truncate, return as-is
    # Take first 4 segments and re-join with " | "
    return " | ".join(s.strip() for s in segments[:4])


def main():
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
    TEST_PATH = os.environ.get("V77_TEST_PATH", ".cache/gsm8k_steps_v81_test.jsonl")
    NUM_EVAL = int(os.environ.get("NUM_EVAL", "60"))
    K = int(os.environ.get("K", "7"))
    FIXED_LEN = int(os.environ.get("FIXED_LEN", "320"))
    BATCH = int(os.environ.get("BATCH", "8"))
    MAX_NEW = int(os.environ.get("MAX_NEW", "120"))
    MIN_K = int(os.environ.get("GSM8K_STEPS_MIN_K", "2"))
    MAX_K = int(os.environ.get("GSM8K_STEPS_MAX_K", "6"))

    print(f"=== v81 SINGLE-HEAD AR eval (K={K} breaths) ===")
    print(f"  ckpt: {CKPT}")
    print(f"  test: {TEST_PATH}")
    print(f"  num_eval: {NUM_EVAL}  fixed_len: {FIXED_LEN}  batch: {BATCH}  max_new: {MAX_NEW}")
    print(f"  decode: force_single_head=True (per-head bias bypassed)")

    if not os.path.exists(TEST_PATH):
        # Fall back to the train file (used during smoke training).
        train_alt = ".cache/gsm8k_steps_v81_train.jsonl"
        if os.path.exists(train_alt):
            TEST_PATH = train_alt
            print(f"  [fallback] test set not found; using train file: {TEST_PATH}")
        else:
            print(f"ERROR: test set not found at {TEST_PATH}", file=sys.stderr)
            sys.exit(1)

    print(f"\nloading v77/v81 test set...")
    examples = load_gsm8k_v77(TEST_PATH, min_k=MIN_K, max_k=MAX_K,
                                require_sympy_match=True, bucket_by_k=False)
    examples = examples[:NUM_EVAL]
    print(f"  using {len(examples)} examples")

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

    correct = 0
    parseable = 0
    samples_to_show = 5
    t0 = time.perf_counter()

    for batch_start in range(0, len(examples), BATCH):
        batch = examples[batch_start:batch_start + BATCH]
        prompt_ids = [tok.encode(ex.problem).ids for ex in batch]
        # Pad batch
        if len(prompt_ids) < BATCH:
            while len(prompt_ids) < BATCH:
                prompt_ids.append(prompt_ids[-1])

        gen_per_ex = generate_single_head_batch(model, prompt_ids, tok,
                                                   K=K, fixed_len=FIXED_LEN,
                                                   max_new=MAX_NEW)
        for i, ex in enumerate(batch):
            gen_text = tok.decode(gen_per_ex[i])
            b6_text = extract_b6(gen_text)
            dag = b6_string_to_dag(b6_text)
            val = dag_to_answer(dag) if dag else None
            ok = val is not None and abs(val - ex.gold_answer) < 1e-3
            if dag is not None:
                parseable += 1
            if ok:
                correct += 1
            if samples_to_show > 0 and batch_start == 0 and i < samples_to_show:
                print(f"\n  Q: {ex.problem[:90]!r}")
                print(f"  gen: {gen_text.strip()[:200]!r}")
                print(f"  B6:  {b6_text!r}")
                print(f"  DAG: {dag!r}")
                print(f"  val={val}  gold={ex.gold_answer}  {'OK' if ok else 'WRONG'}")
                samples_to_show -= 1

    total = len(examples)
    dt = time.perf_counter() - t0
    acc = correct / max(total, 1) * 100
    parse_pct = parseable / max(total, 1) * 100
    print(f"\n=== v81 SINGLE-HEAD AR eval results ({dt:.1f}s) ===")
    print(f"  accuracy:       {acc:.1f}% ({correct}/{total})")
    print(f"  B6 parse rate:  {parse_pct:.1f}% ({parseable}/{total})")


if __name__ == "__main__":
    main()
