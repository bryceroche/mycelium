"""v81 DAG eval — multi-head WaistController decoding.

v81 paradigm: the final breath emits 4 PARALLEL LISTS:
    "<ops_list> | <types_list> | <args1_list> | <args2_list>"

At each answer-span position, ALL 4 heads produce logits. At training, only ONE
head was supervised per position (the others got -100). At inference we have a
choice:
  (A) Decode each head independently using only its own list segment positions.
  (B) Decode autoregressively, switching the "active" head when " | " is emitted.

This script implements approach (A) with a simpler trick: we run a single forward
pass with the input being `[prompt, zeros, EOS, zeros]` (no teacher-forcing — the
masking already zeroed answer-span embeddings during training), then ARGMAX each
head at EACH answer-span position. We then walk each head's argmax sequence to
extract its list, stopping at structural delimiters or EOS.

Output assembly: combine the 4 head outputs into a single B6-style text, then call
`v81_sympy_eval.b6_string_to_dag` to construct the DAG, then `dag_to_answer` to
execute via SymPy.

Env vars (mirror eval_v77_dag):
    CKPT=...           required. Path to the v81 ckpt.
    V77_TEST_PATH=...  test set JSONL.
    NUM_EVAL=60        max problems to eval.
    K=7                inner breaths (matches training).
    FIXED_LEN=320      input padding.
    BATCH=8            eval batch size.
    MAX_LIST=40        max tokens per list per head.
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


def _compile_jit_final_breath_all_heads(model, K: int, fixed_len: int, B: int):
    """JIT: run K breaths once, then for the FINAL breath gather ALL 4 head argmaxes
    across ALL answer-span positions in a single pass.

    Inputs:
      tokens   (B, fixed_len) int     — input with zero-filled answer span
      kv_mask  (B, fixed_len) float   — 1.0 at prompt positions, 0.0 elsewhere
    Outputs:
      4 tensors of shape (B, fixed_len) int — argmax per position for each head.
    """
    key = (id(model), int(K), int(fixed_len), int(B))
    if key in _EVAL_JIT_CACHE:
        return _EVAL_JIT_CACHE[key]

    @TinyJit
    def _fwd(tokens, kv_mask):
        from mycelium.breathing import V81_MAIN_ATTN_MASK as _V81MAM
        _main_attn = kv_mask if _V81MAM else None
        _final, _mw, _pbx, waist_per_breath = model.breathe_with_lookup(
            tokens, n_loops=K, return_per_breath_x=True, return_waist_compressed=True,
            notebook_pool_mask=kv_mask, main_attn_mask=_main_attn)
        prompt_emb = model.embed(tokens).cast(dtypes.float)
        wk = waist_per_breath[K - 1].cast(dtypes.float)
        # Full-length T_q = fixed_len so we get logits at every position in one pass.
        out = model.waist_controller.forward(
            wk, prompt_emb, model.embed_out,
            k_idx=K - 1, K_total=K,
            kv_mask=kv_mask)
        # out is a dict {ops, types, args1, args2} each (B, fixed_len, vocab)
        ops_ids   = out["ops"][:, :, :50277].argmax(axis=-1).cast(dtypes.int).realize()
        types_ids = out["types"][:, :, :50277].argmax(axis=-1).cast(dtypes.int).realize()
        a1_ids    = out["args1"][:, :, :50277].argmax(axis=-1).cast(dtypes.int).realize()
        a2_ids    = out["args2"][:, :, :50277].argmax(axis=-1).cast(dtypes.int).realize()
        return ops_ids, types_ids, a1_ids, a2_ids

    _EVAL_JIT_CACHE[key] = _fwd
    return _fwd


def _decode_head_list(argmax_ids, prompt_len, eos_id=0, pipe_token_id=1040, max_len=40):
    """Walk a head's argmax sequence starting at prompt_len-1 (= predicts first
    answer token). Stop when we emit ` | ` (which is just token 1040, " |"), EOS,
    or hit max_len.

    The first answer position is prompt_len-1 in `pred` (which predicts argmax_ids[prompt_len]
    in the token sequence, but here argmax_ids itself is indexed the same as predictions
    when we use full-T_q forward; the JIT returns (B, fixed_len) where index p IS the
    head's pred for the token AT position p, derived from waist[p] in the final breath).

    Wait — let me reconcile. WaistController(waist, ...) returns logits at T_q = fixed_len.
    The logit at index p IS the model's prediction for "what token belongs at position p".
    So argmax_ids[p] is the head's preferred token AT position p, conditioned on the
    waist at p (which only sees prompt context via masking).

    So to extract a head's list we walk argmax_ids starting at position prompt_len-1
    (which predicts the FIRST answer token; the model's loss at training was on label
    placed at prompt_len-1 to match tokens[prompt_len]).

    Actually — re-reading the JIT: each head's MLP residual + tied embed_out applied
    at every T_q position. The CE loss in training is `pred = logits[:, :-1, :]` ie
    pred[p] predicts tokens[p+1]. So:
      pred[prompt_len-1] = head's argmax for tokens[prompt_len] = first answer token.

    Returns: list of int token ids belonging to this head's list (excluding delimiter).
    """
    out = []
    p = prompt_len - 1  # position whose argmax IS the first answer token
    for _ in range(max_len):
        if p >= len(argmax_ids):
            break
        tid = int(argmax_ids[p])
        if tid == eos_id:
            break
        if tid == pipe_token_id:
            break
        out.append(tid)
        p += 1
    return out


def assemble_b6_text(tok, prompt_len, ops_ids, types_ids, a1_ids, a2_ids):
    """Decode each head's list and assemble into a v81-style B6 string."""
    EOS_ID = 0
    PIPE_ID = 1040  # " |" in Pythia tokenizer
    ops_toks = _decode_head_list(ops_ids, prompt_len, eos_id=EOS_ID, pipe_token_id=PIPE_ID)
    types_toks = _decode_head_list(types_ids, prompt_len, eos_id=EOS_ID, pipe_token_id=PIPE_ID)
    a1_toks = _decode_head_list(a1_ids, prompt_len, eos_id=EOS_ID, pipe_token_id=PIPE_ID)
    a2_toks = _decode_head_list(a2_ids, prompt_len, eos_id=EOS_ID, pipe_token_id=PIPE_ID)
    ops_str = tok.decode(ops_toks).strip()
    types_str = tok.decode(types_toks).strip()
    a1_str = tok.decode(a1_toks).strip()
    a2_str = tok.decode(a2_toks).strip()
    return f"{ops_str} | {types_str} | {a1_str} | {a2_str}", (ops_str, types_str, a1_str, a2_str)


def eval_batch(model, tok, examples, K, fixed_len, batch_size):
    """Run multi-head eval on a batch of examples. Returns list of (text, dag, val, ok)."""
    # Pad batch to batch_size for stable JIT shape.
    B = batch_size
    prompt_ids_list = []
    for ex in examples:
        prompt_ids_list.append(tok.encode(ex.problem).ids)
    while len(prompt_ids_list) < B:
        prompt_ids_list.append(prompt_ids_list[-1])
    prompt_lens = [len(p) for p in prompt_ids_list]

    tokens_np = np.zeros((B, fixed_len), dtype=np.int32)
    kv_mask_np = np.zeros((B, fixed_len), dtype=np.float32)
    for b in range(B):
        L = min(prompt_lens[b], fixed_len)
        tokens_np[b, :L] = prompt_ids_list[b][:L]
        kv_mask_np[b, :L] = 1.0

    tokens_t = Tensor(tokens_np, dtype=dtypes.int).realize()
    kv_mask_t = Tensor(kv_mask_np, dtype=dtypes.float).realize()
    fwd = _compile_jit_final_breath_all_heads(model, K, fixed_len, B)
    ops_t, types_t, a1_t, a2_t = fwd(tokens_t, kv_mask_t)
    ops_np = ops_t.numpy()
    types_np = types_t.numpy()
    a1_np = a1_t.numpy()
    a2_np = a2_t.numpy()
    results = []
    for i, ex in enumerate(examples):
        text, segs = assemble_b6_text(tok, prompt_lens[i],
                                        ops_np[i], types_np[i], a1_np[i], a2_np[i])
        dag = b6_string_to_dag(text)
        val = dag_to_answer(dag) if dag else None
        ok = val is not None and abs(val - ex.gold_answer) < 1e-3
        results.append({
            "problem": ex.problem,
            "text": text,
            "segs": segs,
            "dag": dag,
            "val": val,
            "gold": ex.gold_answer,
            "ok": ok,
        })
    return results


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
    TEST_PATH = os.environ.get("V77_TEST_PATH", ".cache/gsm8k_steps_v81_train.jsonl")
    NUM_EVAL = int(os.environ.get("NUM_EVAL", "60"))
    K = int(os.environ.get("K", "7"))
    FIXED_LEN = int(os.environ.get("FIXED_LEN", "320"))
    BATCH = int(os.environ.get("BATCH", "8"))
    MIN_K = int(os.environ.get("GSM8K_STEPS_MIN_K", "2"))
    MAX_K = int(os.environ.get("GSM8K_STEPS_MAX_K", "6"))

    print(f"=== v81 multi-head eval (K={K} breaths, 4 heads decoded in parallel) ===")
    print(f"  ckpt: {CKPT}")
    print(f"  test: {TEST_PATH}")
    print(f"  num_eval: {NUM_EVAL}  fixed_len: {FIXED_LEN}  batch: {BATCH}")
    if not os.path.exists(TEST_PATH):
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
        results = eval_batch(model, tok, batch, K=K, fixed_len=FIXED_LEN, batch_size=BATCH)
        for i, r in enumerate(results):
            if r["dag"] is not None:
                parseable += 1
            if r["ok"]:
                correct += 1
            if samples_to_show > 0:
                print(f"\n  Q: {r['problem'][:100]!r}")
                print(f"  B6: {r['text']!r}")
                print(f"  segs: ops={r['segs'][0]!r}  types={r['segs'][1]!r}  a1={r['segs'][2]!r}  a2={r['segs'][3]!r}")
                print(f"  DAG: {r['dag']!r}")
                print(f"  val={r['val']}  gold={r['gold']}  {'OK' if r['ok'] else 'WRONG'}")
                samples_to_show -= 1

    total = len(examples)
    dt = time.perf_counter() - t0
    acc = correct / max(total, 1) * 100
    parse_pct = parseable / max(total, 1) * 100
    print(f"\n=== v81 multi-head eval results ({dt:.1f}s) ===")
    print(f"  accuracy:       {acc:.1f}% ({correct}/{total})")
    print(f"  B6 parse rate:  {parse_pct:.1f}% ({parseable}/{total})")


if __name__ == "__main__":
    main()
