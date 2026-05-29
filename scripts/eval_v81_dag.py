"""v81 DAG eval — multi-head WaistController decoding (BUG-FIXED 2026-05-27).

v81 paradigm: the final breath emits 4 PARALLEL LISTS:
    "<ops_list> | <types_list> | <args1_list> | <args2_list>"

Each head has a per-head additive bias on top of a SHARED cross-attn backbone.
At training, each head was supervised at a DIFFERENT position range:
  - ops head:   positions [prompt_len-1 .. +len(' ops_str')-1]
  - types head: positions starting AFTER ops's range, for len(' | types_str')
  - args1 head: starting AFTER types's range
  - args2 head: starting AFTER args1's range
Outside its supervised range, each head's loss was -100 (no gradient signal).

**Bug found 2026-05-27 (diagnostic: scripts/diag_v81_per_breath_decode.py):**
The original "approach A" eval below argmaxes ALL 4 heads at the SAME starting
position (prompt_len - 1) and walks each independently. This is fundamentally
broken: at positions OUTSIDE each head's supervised range, the head produces
UNCONSTRAINED output (no training signal applied there). So types/args1/args2
all produce garbage at position prompt_len - 1 (where only OPS was supervised).

**Fix:** Bypass the per-head bias entirely (force_single_head=True) and use the
SHARED backbone's output. The cross-attn backbone learned the full B6 structure
because every head's CE gradient flowed through it. The shared backbone produces
the canonical sequence `ops | types | args1 | args2`. We truncate to the first
4 pipe-separated segments and pass to b6_string_to_dag for SymPy execution.

Env vars (mirror eval_v77_dag):
    CKPT=...           required. Path to the v81 ckpt.
    V77_TEST_PATH=...  test set JSONL.
    NUM_EVAL=60        max problems to eval.
    K=7                inner breaths (matches training).
    FIXED_LEN=320      input padding.
    BATCH=8            eval batch size.
    MAX_NEW=80         max AR tokens (B6 fits well under 80).
    MULTI_HEAD_DECODE  if "1", uses the LEGACY multi-head argmax (broken; for diag only).
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
    """LEGACY (broken): JIT compiles a full-T_q multi-head forward and returns the
    argmax per position for each of the 4 heads. Use only with MULTI_HEAD_DECODE=1
    for diagnostic comparison; the SINGLE-HEAD path (default) supersedes this.

    Inputs:
      tokens   (B, fixed_len) int     — input with zero-filled answer span
      kv_mask  (B, fixed_len) float   — 1.0 at prompt positions, 0.0 elsewhere
    Outputs:
      4 tensors of shape (B, fixed_len) int — argmax per position for each head.
    """
    key = ("multihead", id(model), int(K), int(fixed_len), int(B))
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
        out = model.waist_controller.forward(
            wk, prompt_emb, model.embed_out,
            k_idx=K - 1, K_total=K,
            kv_mask=kv_mask)
        ops_ids   = out["ops"][:, :, :50277].argmax(axis=-1).cast(dtypes.int).realize()
        types_ids = out["types"][:, :, :50277].argmax(axis=-1).cast(dtypes.int).realize()
        a1_ids    = out["args1"][:, :, :50277].argmax(axis=-1).cast(dtypes.int).realize()
        a2_ids    = out["args2"][:, :, :50277].argmax(axis=-1).cast(dtypes.int).realize()
        return ops_ids, types_ids, a1_ids, a2_ids

    _EVAL_JIT_CACHE[key] = _fwd
    return _fwd


def _compile_jit_single_head_step(model, K: int, fixed_len: int, B: int):
    """JIT: K breaths once, return argmax token IDs at t_pos[b] using the SINGLE-HEAD
    path (force_single_head=True — bypasses per-head bias). Returns (B,) int — argmax
    of single-head logits at t_pos per example.

    Inputs:
      tokens   (B, fixed_len) int
      t_pos_t  (B,)          int  — per-example current position
      kv_mask  (B, fixed_len) float — 1.0 at valid (prompt) positions, 0.0 elsewhere
    """
    key = ("single", id(model), int(K), int(fixed_len), int(B))
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
        positions = Tensor.arange(fixed_len)
        gather_mask = (positions.reshape(1, fixed_len) == t_pos_t.reshape(B, 1)).reshape(B, fixed_len, 1).cast(dtypes.float)
        wk_at_pos = (wk * gather_mask).sum(axis=1, keepdim=True)
        lk = model.waist_controller.forward(
            wk_at_pos, prompt_emb, model.embed_out,
            k_idx=K - 1, K_total=K,
            kv_mask=kv_mask,
            force_single_head=True)
        return lk[:, :, :50277].argmax(axis=-1).reshape(B).realize()

    _EVAL_JIT_CACHE[key] = _fwd
    return _fwd


def _decode_head_list(argmax_ids, prompt_len, eos_id=0, pipe_token_id=1040, max_len=40):
    """Walk a head's argmax sequence starting at prompt_len-1 (= predicts first
    answer token). Stop when we emit ` | ` (token 1040), EOS, or hit max_len.
    """
    out = []
    p = prompt_len - 1
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


def assemble_b6_text_multihead(tok, prompt_len, ops_ids, types_ids, a1_ids, a2_ids):
    """LEGACY (broken). Walks each head independently from prompt_len-1 until pipe/eos.
    Kept for diagnostic comparison under MULTI_HEAD_DECODE=1."""
    EOS_ID = 0
    PIPE_ID = 1040
    ops_toks = _decode_head_list(ops_ids, prompt_len, eos_id=EOS_ID, pipe_token_id=PIPE_ID)
    types_toks = _decode_head_list(types_ids, prompt_len, eos_id=EOS_ID, pipe_token_id=PIPE_ID)
    a1_toks = _decode_head_list(a1_ids, prompt_len, eos_id=EOS_ID, pipe_token_id=PIPE_ID)
    a2_toks = _decode_head_list(a2_ids, prompt_len, eos_id=EOS_ID, pipe_token_id=PIPE_ID)
    ops_str = tok.decode(ops_toks).strip()
    types_str = tok.decode(types_toks).strip()
    a1_str = tok.decode(a1_toks).strip()
    a2_str = tok.decode(a2_toks).strip()
    return f"{ops_str} | {types_str} | {a1_str} | {a2_str}", (ops_str, types_str, a1_str, a2_str)


def extract_b6_from_single_head_text(text: str) -> str:
    """Truncate single-head decoded text to the first 4 pipe-separated segments.

    The model often emits 5+ pipes (never learned to STOP after the 4th list).
    We split on " |" (with leading space, per the training format `... | ...`)
    and take the first 4 segments.
    """
    text = text.strip()
    segments = text.split(" |")
    if len(segments) < 4:
        segments = text.split("|")
    if len(segments) < 4:
        return text  # can't truncate, return as-is
    return " | ".join(s.strip() for s in segments[:4])


def generate_single_head_batch(model, prompt_ids_list, K, fixed_len, max_new=80, eos_id=0):
    """Greedy AR decode using the single-head path (bypasses per-head bias)."""
    B = len(prompt_ids_list)
    prompt_lens = [len(p) for p in prompt_ids_list]
    current_lens = list(prompt_lens)
    assert max(current_lens) + max_new <= fixed_len, (
        f"need fixed_len >= {max(current_lens) + max_new}, got {fixed_len}")
    tokens_np = np.zeros((B, fixed_len), dtype=np.int32)
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


def eval_batch_singlehead(model, tok, examples, K, fixed_len, batch_size, max_new):
    """Run SINGLE-HEAD AR eval on a batch."""
    B = batch_size
    prompt_ids_list = []
    for ex in examples:
        prompt_ids_list.append(tok.encode(ex.problem).ids)
    while len(prompt_ids_list) < B:
        prompt_ids_list.append(prompt_ids_list[-1])

    gen_per_ex = generate_single_head_batch(model, prompt_ids_list, K, fixed_len, max_new=max_new)
    results = []
    for i, ex in enumerate(examples):
        gen_text = tok.decode(gen_per_ex[i])
        b6_text = extract_b6_from_single_head_text(gen_text)
        dag = b6_string_to_dag(b6_text)
        val = dag_to_answer(dag) if dag else None
        ok = val is not None and abs(val - ex.gold_answer) < 1e-3
        results.append({
            "problem": ex.problem,
            "gen": gen_text,
            "text": b6_text,
            "segs": tuple(b6_text.split(" | ")) if " | " in b6_text else (b6_text,),
            "dag": dag,
            "val": val,
            "gold": ex.gold_answer,
            "ok": ok,
        })
    return results


def eval_batch_multihead(model, tok, examples, K, fixed_len, batch_size):
    """LEGACY (broken) multi-head eval — kept for diagnostic comparison only.

    Argmaxes each of the 4 heads at every position; walks each independently
    from prompt_len-1 until pipe/eos. This produces 0% accuracy in practice
    because heads have no training signal outside their own position range
    (see eval_v81_dag.py docstring + scripts/diag_v81_per_breath_decode.py).
    """
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
        text, segs = assemble_b6_text_multihead(tok, prompt_lens[i],
                                                  ops_np[i], types_np[i], a1_np[i], a2_np[i])
        dag = b6_string_to_dag(text)
        val = dag_to_answer(dag) if dag else None
        ok = val is not None and abs(val - ex.gold_answer) < 1e-3
        results.append({
            "problem": ex.problem,
            "gen": "",
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
    MAX_NEW = int(os.environ.get("MAX_NEW", "80"))
    MULTI_HEAD_DECODE = int(os.environ.get("MULTI_HEAD_DECODE", "0"))
    MIN_K = int(os.environ.get("GSM8K_STEPS_MIN_K", "2"))
    MAX_K = int(os.environ.get("GSM8K_STEPS_MAX_K", "6"))

    decode_mode = "MULTI-HEAD (legacy, broken)" if MULTI_HEAD_DECODE else "SINGLE-HEAD AR (bug-fixed 2026-05-27)"
    print(f"=== v81 DAG eval — {decode_mode} ===")
    print(f"  ckpt: {CKPT}")
    print(f"  test: {TEST_PATH}")
    print(f"  num_eval: {NUM_EVAL}  fixed_len: {FIXED_LEN}  batch: {BATCH}  max_new: {MAX_NEW}")
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
        if MULTI_HEAD_DECODE:
            results = eval_batch_multihead(model, tok, batch, K=K, fixed_len=FIXED_LEN, batch_size=BATCH)
        else:
            results = eval_batch_singlehead(model, tok, batch, K=K, fixed_len=FIXED_LEN,
                                              batch_size=BATCH, max_new=MAX_NEW)
        for i, r in enumerate(results):
            if r["dag"] is not None:
                parseable += 1
            if r["ok"]:
                correct += 1
            if samples_to_show > 0:
                print(f"\n  Q: {r['problem'][:100]!r}")
                if r["gen"]:
                    print(f"  gen: {r['gen'].strip()[:200]!r}")
                print(f"  B6:  {r['text']!r}")
                print(f"  DAG: {r['dag']!r}")
                print(f"  val={r['val']}  gold={r['gold']}  {'OK' if r['ok'] else 'WRONG'}")
                samples_to_show -= 1

    total = len(examples)
    dt = time.perf_counter() - t0
    acc = correct / max(total, 1) * 100
    parse_pct = parseable / max(total, 1) * 100
    print(f"\n=== v81 DAG eval results ({decode_mode}) ({dt:.1f}s) ===")
    print(f"  accuracy:       {acc:.1f}% ({correct}/{total})")
    print(f"  B6 parse rate:  {parse_pct:.1f}% ({parseable}/{total})")


if __name__ == "__main__":
    main()
