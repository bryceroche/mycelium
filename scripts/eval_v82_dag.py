"""v82 DAG eval — SINGLE-HEAD AR decoding on the 3-list parallel-diffusion format.

v82 paradigm: the final breath (B6) emits the FULL 3-list sequence

    "<ops_csv> | <types_csv> | <args_csv>"

The v82 SINGLE-HEAD path is the default (no MULTI_HEAD_WAIST split). At training,
each breath is supervised with single-head full-sequence CE on its breath-specific
target (precision schedule P0..P3 across breaths). At eval we run K breaths once
and AR-decode tokens from the LAST breath's single-head logits.

Env vars (mirror eval_v81_dag):
    CKPT=...           required. Path to the v82 ckpt.
    V77_TEST_PATH=...  test set JSONL (default v82 test).
    NUM_EVAL=60        max problems to eval.
    K=7                inner breaths (matches training).
    FIXED_LEN=256      input padding.
    BATCH=4            eval batch size.
    MAX_NEW=80         max AR tokens.

The launcher should also set MULTI_HEAD_WAIST=0 so the WaistController emits a
single-head tensor (not a 4-key dict). All 4 v81 masks are still applied at eval
for train/eval consistency.
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

from v82_sympy_eval import b6_string_to_dag, dag_to_answer  # type: ignore


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
    """JIT: K breaths once, return argmax token IDs at t_pos[b] using the single-head
    WaistController path. force_single_head=True is benign when MULTI_HEAD_WAIST=0
    (the same path runs); explicit so the eval graph is stable across env states.

    Inputs:
      tokens   (B, fixed_len) int
      t_pos_t  (B,)          int  — per-example current position
      kv_mask  (B, fixed_len) float — 1.0 at valid (prompt) positions, 0.0 elsewhere
    Returns: (B,) int — argmax of single-head logits at t_pos per example.
    """
    key = ("v82_single", id(model), int(K), int(fixed_len), int(B))
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


def extract_b6_from_text(text: str) -> str:
    """Truncate decoded text to the first 3 pipe-separated segments.

    The model may emit more pipes (didn't learn a stop signal). Split on ' |' and
    take the first 3 segments.
    """
    text = text.strip()
    segments = text.split(" |")
    if len(segments) < 3:
        segments = text.split("|")
    if len(segments) < 3:
        return text
    return " | ".join(s.strip() for s in segments[:3])


def generate_single_head_batch(model, prompt_ids_list, K, fixed_len, max_new=80, eos_id=0):
    """Greedy AR decode using the single-head path."""
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


def eval_batch(model, tok, examples, K, fixed_len, batch_size, max_new):
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
        b6_text = extract_b6_from_text(gen_text)
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
    TEST_PATH = os.environ.get("V77_TEST_PATH", ".cache/gsm8k_steps_v82_test.jsonl")
    NUM_EVAL = int(os.environ.get("NUM_EVAL", "60"))
    K = int(os.environ.get("K", "7"))
    FIXED_LEN = int(os.environ.get("FIXED_LEN", "256"))
    BATCH = int(os.environ.get("BATCH", "4"))
    MAX_NEW = int(os.environ.get("MAX_NEW", "80"))
    MIN_K = int(os.environ.get("GSM8K_STEPS_MIN_K", "2"))
    MAX_K = int(os.environ.get("GSM8K_STEPS_MAX_K", "6"))

    print(f"=== v82 DAG eval — SINGLE-HEAD AR (parallel-diffusion paradigm) ===")
    print(f"  ckpt: {CKPT}")
    print(f"  test: {TEST_PATH}")
    print(f"  num_eval: {NUM_EVAL}  fixed_len: {FIXED_LEN}  batch: {BATCH}  max_new: {MAX_NEW}")
    if not os.path.exists(TEST_PATH):
        print(f"ERROR: test set not found at {TEST_PATH}", file=sys.stderr)
        sys.exit(1)

    print(f"\nloading v77/v82 test set...")
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
        results = eval_batch(model, tok, batch, K=K, fixed_len=FIXED_LEN,
                              batch_size=BATCH, max_new=MAX_NEW)
        for i, r in enumerate(results):
            if r["dag"] is not None:
                parseable += 1
            if r["ok"]:
                correct += 1
            if samples_to_show > 0:
                print(f"\n  Q: {r['problem'][:100]!r}")
                print(f"  gen: {r['gen'].strip()[:200]!r}")
                print(f"  B6:  {r['text']!r}")
                print(f"  DAG: {r['dag']!r}")
                print(f"  val={r['val']}  gold={r['gold']}  {'OK' if r['ok'] else 'WRONG'}")
                samples_to_show -= 1

    total = len(examples)
    dt = time.perf_counter() - t0
    acc = correct / max(total, 1) * 100
    parse_pct = parseable / max(total, 1) * 100
    print(f"\n=== v82 DAG eval results ({dt:.1f}s) ===")
    print(f"  accuracy:       {acc:.1f}% ({correct}/{total})")
    print(f"  B6 parse rate:  {parse_pct:.1f}% ({parseable}/{total})")


if __name__ == "__main__":
    main()
