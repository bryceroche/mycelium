"""v85 DAG eval — structured slot decoding with SymPy execution.

v85 paradigm: K breaths, slot decoder fires at every breath, final breath's slot
mixtures get argmax'd and executed via SymPy.

Per-problem:
  1. Encode the problem; build numbers_emb via token-span pooling.
  2. Run K breaths, capture waist_compressed[K-1].
  3. Call v85_slot_decoder.forward(waist_compressed[K-1].pool, numbers_emb, mask).
  4. Argmax ops_logits, types_logits, args1_logits, args2_logits per slot.
  5. Threshold is_active_logits at 0.5 (sigmoid).
  6. Render the DAG: for each active slot, emit
       "xk = <a1> <op> <a2>"
     where args are either literal number values (from numbers list) or x_k refs.
  7. Execute via dag_to_answer; compare to gold.

Env vars:
  CKPT=...           required.
  V77_TEST_PATH=...  test set JSONL (default v85 test).
  NUM_EVAL=60        max problems to eval.
  K=5                inner breaths.
  FIXED_LEN=256      input padding.
  BATCH=4            eval batch size (currently sequential — JIT recompiles
                      problematic with the slot decoder; future work).
"""
import os
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from tinygrad import Tensor, Device, dtypes
from tinygrad.nn.state import safe_load

from mycelium import Config
from mycelium.loader import _load_state, load_breathing
from mycelium.data import load_tokenizer
from mycelium.l3_data import load_gsm8k_v85
from mycelium.l3_training import _v85_encode_batch

from v77_sympy_eval import dag_to_answer  # type: ignore


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


OP_TABLE = ["+", "-", "*", "/"]


def _render_dag(active_mask, ops, args1_idx, args2_idx, numbers_values, n_real,
                K_max, N_max):
    """Build a SymPy-executable DAG string from per-slot decisions.

    active_mask: list[bool] length K_max
    ops, args1_idx, args2_idx: list[int] length K_max
    numbers_values: list[float] length N_max (only first n_total are valid)
    n_real: int — len(numbers) (real prompt numbers); rest are implicit.

    Returns: dag string or None if any active slot can't render.

    v90 (2026-05-27): if no slots are active at all (all False), fall back to
    using slot 0 so we still emit a parseable DAG. This avoids penalizing the
    eval when the active head defaults to False on every slot.
    """
    n_total = len(numbers_values)

    # v90 fallback: if all slots inactive, force slot 0 active so we emit
    # something the renderer can finalize. Without this the whole problem
    # silently returns None and counts as a parse failure.
    if not any(active_mask):
        active_mask = list(active_mask)
        active_mask[0] = True

    parts = []
    last_x = -1
    for k in range(K_max):
        if not active_mask[k]:
            continue
        op = ops[k]
        if not (0 <= op < 4):
            return None
        sym = OP_TABLE[op]

        def render_arg(idx):
            if 0 <= idx < N_max:
                # numbers pointer
                if 0 <= idx < n_total:
                    return str(numbers_values[idx])
                return None
            elif N_max <= idx < N_max + K_max:
                # dag pointer
                k_ref = idx - N_max
                if 0 <= k_ref < k and active_mask[k_ref]:
                    return f"x{k_ref}"
                return None
            return None

        a1 = render_arg(args1_idx[k])
        a2 = render_arg(args2_idx[k])
        if a1 is None or a2 is None:
            return None
        parts.append(f"x{k} = {a1} {sym} {a2}")
        last_x = k

    if last_x < 0:
        return None
    parts.append(f"answer = x{last_x}")
    return " ; ".join(parts)


def eval_one_problem(model, tok, ex, K, fixed_len, K_max, N_max):
    """Evaluate a single problem end-to-end. Returns dict with:
        problem, gold, val, dag, ok, components.
    """
    # Build encoded batch with B=1.
    enc = _v85_encode_batch([ex], tok, fixed_len, K_max, N_max)
    tokens = Tensor(enc["tokens_np"], dtype=dtypes.int).realize()
    number_span_idx = Tensor(enc["number_span_token_idx_np"], dtype=dtypes.int).realize()
    numbers_mask = Tensor(enc["numbers_mask_np"], dtype=dtypes.float).realize()
    kv_mask = Tensor(enc["kv_mask_np"], dtype=dtypes.float).realize()

    Tensor.training = False
    # Forward (eager — not JIT; eval is one-shot per problem).
    _final, _mw, _pbx, waist_per_breath = model.breathe_with_lookup(
        tokens, K, return_per_breath_x=True, return_waist_compressed=True,
        notebook_pool_mask=kv_mask, main_attn_mask=kv_mask)

    # Build numbers_emb (same as in training).
    B = 1
    T = tokens.shape[1]
    H = model.cfg.hidden
    prompt_emb_full = model.embed(tokens).cast(dtypes.float)
    prompt_emb_masked = prompt_emb_full * kv_mask.cast(dtypes.float).reshape(B, T, 1)
    positions = Tensor.arange(T).reshape(1, 1, T)
    start_idx = number_span_idx[:, :, 0:1]
    end_idx = number_span_idx[:, :, 1:2]
    span_mask = ((positions >= start_idx) & (positions < end_idx)).cast(dtypes.float)
    span_len = span_mask.sum(axis=-1, keepdim=True) + 1e-6
    numbers_emb = (span_mask @ prompt_emb_masked) / span_len
    numbers_emb = numbers_emb * numbers_mask.reshape(B, -1, 1)

    # Pool final-breath waist over the prompt.
    waist_k = waist_per_breath[K - 1].cast(dtypes.float)
    waist_k_masked = waist_k * kv_mask.cast(waist_k.dtype).reshape(B, -1, 1)
    mask_count = kv_mask.cast(dtypes.float).sum(axis=1, keepdim=True) + 1e-6
    waist_pooled = waist_k_masked.sum(axis=1, keepdim=True) * (1.0 / mask_count.reshape(B, 1, 1))

    # v86: pass waist_full + kv_mask so the slot decoder can do per-slot
    # cross-attn over the full waist. When V86_ARGS_CROSS_ATTN=0 these are
    # ignored.
    decoder_out = model.v85_slot_decoder.forward(
        waist_pooled, numbers_emb, numbers_mask,
        waist_full=waist_k, waist_full_mask=kv_mask)

    ops_logits = decoder_out["ops_logits"].numpy()[0]      # (K_max, 4)
    args1_logits = decoder_out["args1_logits"].numpy()[0]  # (K_max, N_max + K_max)
    args2_logits = decoder_out["args2_logits"].numpy()[0]
    active_logits = decoder_out["active_logits"].numpy()[0]  # (K_max,)

    ops_pred = ops_logits.argmax(axis=-1).tolist()
    args1_pred = args1_logits.argmax(axis=-1).tolist()
    args2_pred = args2_logits.argmax(axis=-1).tolist()
    # sigmoid threshold (v92: tunable via V92_ACTIVE_EVAL_THRESHOLD; default 0.5)
    _active_thresh = float(os.environ.get("V92_ACTIVE_EVAL_THRESHOLD", "0.5"))
    active_pred = (1.0 / (1.0 + np.exp(-active_logits))) > _active_thresh
    active_pred = active_pred.tolist()

    # Build numbers_values list combining real + implicit.
    real_vals = [n["value"] for n in ex.numbers]
    imp_vals = [n["value"] for n in ex.implicit_numbers]
    numbers_values = real_vals + imp_vals

    dag = _render_dag(active_pred, ops_pred, args1_pred, args2_pred,
                       numbers_values, n_real=len(real_vals),
                       K_max=K_max, N_max=N_max)
    val = dag_to_answer(dag) if dag else None
    ok = (val is not None) and (abs(val - ex.gold_answer) < 1e-3)

    return {
        "problem": ex.problem,
        "gold": ex.gold_answer,
        "dag": dag,
        "val": val,
        "ok": ok,
        "active_pred": active_pred,
        "ops_pred": ops_pred,
        "args1_pred": args1_pred,
        "args2_pred": args2_pred,
        "numbers_values": numbers_values,
        "n_steps_gold": ex.n_steps,
        "ops_gold": [{"ADD": 0, "SUB": 1, "MUL": 2, "DIV": 3}[s["op"]] for s in ex.dag_slots],
    }


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
    TEST_PATH = os.environ.get("V77_TEST_PATH", ".cache/gsm8k_steps_v85_test.jsonl")
    if not os.path.exists(TEST_PATH):
        # Fallback: try train file (smoke).
        if os.path.exists(".cache/gsm8k_steps_v85_smoke.jsonl"):
            TEST_PATH = ".cache/gsm8k_steps_v85_smoke.jsonl"
        elif os.path.exists(".cache/gsm8k_steps_v85_train.jsonl"):
            TEST_PATH = ".cache/gsm8k_steps_v85_train.jsonl"
    NUM_EVAL = int(os.environ.get("NUM_EVAL", "60"))
    K = int(os.environ.get("K", "5"))
    FIXED_LEN = int(os.environ.get("FIXED_LEN", "256"))
    K_MAX = int(os.environ.get("V85_K_MAX", "10"))
    N_MAX = int(os.environ.get("V85_N_MAX", "20"))
    MIN_K = int(os.environ.get("GSM8K_STEPS_MIN_K", "2"))
    MAX_K = int(os.environ.get("GSM8K_STEPS_MAX_K", "6"))

    print(f"=== v85 DAG eval — structured slot decoder ===")
    print(f"  ckpt: {CKPT}")
    print(f"  test: {TEST_PATH}")
    print(f"  num_eval: {NUM_EVAL}  K: {K}  K_max: {K_MAX}  N_max: {N_MAX}  fixed_len: {FIXED_LEN}")
    if not os.path.exists(TEST_PATH):
        print(f"ERROR: test set not found at {TEST_PATH}", file=sys.stderr)
        sys.exit(1)

    examples = load_gsm8k_v85(TEST_PATH, min_k=MIN_K, max_k=MAX_K,
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
    sum_ops_correct = 0
    n_ops_compared = 0
    # v90 diagnostics: total True-vs-False count + gold n_steps tracking.
    sum_active_true = 0
    sum_active_slots = 0  # K_max * num_problems
    sum_gold_steps = 0

    for i, ex in enumerate(examples):
        r = eval_one_problem(model, tok, ex, K=K, fixed_len=FIXED_LEN,
                              K_max=K_MAX, N_max=N_MAX)
        if r["dag"] is not None:
            parseable += 1
        if r["ok"]:
            correct += 1

        # v90: tally active-fraction.
        sum_active_true += int(sum(1 for v in r["active_pred"] if v))
        sum_active_slots += K_MAX
        sum_gold_steps += int(r["n_steps_gold"])

        # Per-slot ops accuracy (gold slots only).
        for k, gold_op in enumerate(r["ops_gold"]):
            if r["active_pred"][k]:  # only count active predictions
                if r["ops_pred"][k] == gold_op:
                    sum_ops_correct += 1
                n_ops_compared += 1

        if samples_to_show > 0:
            print(f"\n  Q: {r['problem'][:100]!r}")
            print(f"  numbers: {r['numbers_values']}")
            print(f"  active_pred: {r['active_pred']}")
            print(f"  ops_pred:    {r['ops_pred']}")
            print(f"  args1_pred:  {r['args1_pred']}")
            print(f"  args2_pred:  {r['args2_pred']}")
            print(f"  DAG: {r['dag']!r}")
            print(f"  val={r['val']}  gold={r['gold']}  {'OK' if r['ok'] else 'WRONG'}")
            samples_to_show -= 1

    total = len(examples)
    dt = time.perf_counter() - t0
    acc = correct / max(total, 1) * 100
    parse_pct = parseable / max(total, 1) * 100
    ops_acc = (sum_ops_correct / max(n_ops_compared, 1)) * 100
    active_frac = (sum_active_true / max(sum_active_slots, 1)) * 100
    gold_frac = (sum_gold_steps / max(sum_active_slots, 1)) * 100
    print(f"\n=== v85 DAG eval results ({dt:.1f}s) ===")
    print(f"  accuracy:       {acc:.1f}% ({correct}/{total})")
    print(f"  DAG parse rate: {parse_pct:.1f}% ({parseable}/{total})")
    print(f"  ops slot acc:   {ops_acc:.1f}% (n={n_ops_compared})")
    print(f"  active=True frac: {active_frac:.1f}% ({sum_active_true}/{sum_active_slots})  "
          f"gold n_steps frac: {gold_frac:.1f}% ({sum_gold_steps}/{sum_active_slots})")


if __name__ == "__main__":
    main()
