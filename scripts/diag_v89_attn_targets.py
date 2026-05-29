"""v89 sanity check — verify attention targets are computed correctly.

For each problem in a small sample, prints:
  - problem text + number spans
  - per-slot gold args1/args2 source/index
  - computed attention target token positions
  - prompt tokens at those positions (to verify the position maps to the right number)

Also: load a v88 ckpt and dump current cross-attn attn peaks per slot to compare
against the GOLD positions — shows how far off the current attention is from
the targets.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

os.environ.setdefault("V85_QUERYABLE", "1")
os.environ.setdefault("V86_ARGS_CROSS_ATTN", "1")
os.environ.setdefault("V87_SLOT_POS_INIT_SCALE", "0.5")
os.environ.setdefault("V89_SUPERVISED_ATTN", "1")
os.environ.setdefault("V89_PROJ_INIT_SCALE", "0.02")

import numpy as np
from tinygrad import Tensor, Device, dtypes
from tinygrad.nn.state import safe_load

from mycelium import Config
from mycelium.loader import _load_state, load_breathing
from mycelium.data import load_tokenizer
from mycelium.l3_data import load_gsm8k_v85
from mycelium.l3_training import _v85_encode_batch


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


def main():
    print("=== v89 attention target sanity check ===")

    cfg = Config()
    tok = load_tokenizer()

    K = 5
    FIXED_LEN = 224
    K_MAX = 10
    N_MAX = 20

    # Load 3 problems.
    examples = load_gsm8k_v85(
        ".cache/gsm8k_steps_v85_train.jsonl",
        min_k=2, max_k=6,
        require_sympy_match=True, bucket_by_k=False)[:3]

    # First, just verify that the encode pipeline produces sane attn targets,
    # without loading model.
    enc = _v85_encode_batch(examples, tok, FIXED_LEN, K_MAX, N_MAX)
    tokens_np = enc["tokens_np"]
    args1_targets = enc["args1_targets_np"]
    args2_targets = enc["args2_targets_np"]
    args_attn_target = enc["args_attn_target_np"]
    number_spans = enc["number_span_token_idx_np"]
    numbers_mask = enc["numbers_mask_np"]
    prompt_lens = enc["prompt_lens_np"]

    print(f"\nargs_attn_target shape: {args_attn_target.shape}")
    print(f"number of valid (non -100) attn targets per problem:")
    for b in range(len(examples)):
        n_valid = int((args_attn_target[b] != -100).sum())
        print(f"  problem {b}: {n_valid} / {2 * K_MAX} positions")

    # For each problem, dump expected target positions and prompt tokens.
    for b, ex in enumerate(examples):
        print(f"\n--- Problem {b} ---")
        print(f"  text: {ex.problem[:120]!r}")
        prompt_ids = tokens_np[b, :prompt_lens[b]].tolist()
        decoded = tok.decode(prompt_ids)
        print(f"  prompt_len = {prompt_lens[b]}")

        # Decode each token individually to show position->text.
        per_tok = []
        for ti, tid in enumerate(prompt_ids):
            tstr = tok.decode([tid])
            per_tok.append((ti, tid, tstr))

        # Show numbers.
        print(f"  numbers (n_real={len(ex.numbers)}):")
        for i, n in enumerate(ex.numbers):
            mask_v = numbers_mask[b, i]
            span = number_spans[b, i].tolist()
            span_text = tok.decode(prompt_ids[span[0]:span[1]]) if span[1] > span[0] else "(degenerate)"
            print(f"    [{i}] value={n.get('value')}  span_char=[{n.get('span_start_char')},{n.get('span_end_char')})  span_tok={span}  text={span_text!r}  mask={mask_v}")
        if ex.implicit_numbers:
            print(f"  implicit numbers:")
            for j, n in enumerate(ex.implicit_numbers):
                print(f"    [{len(ex.numbers) + j}] value={n.get('value')} (implicit)")

        # Show DAG slots + their attention targets.
        print(f"  dag_slots (n_steps={ex.n_steps}):")
        for k, slot in enumerate(ex.dag_slots):
            if k >= K_MAX:
                break
            args = slot.get("args", [])
            t1 = int(args_attn_target[b, k, 0])
            t2 = int(args_attn_target[b, k, 1])
            t1_text = tok.decode([prompt_ids[t1]]) if 0 <= t1 < len(prompt_ids) else "(N/A)"
            t2_text = tok.decode([prompt_ids[t2]]) if 0 <= t2 < len(prompt_ids) else "(N/A)"
            print(f"    [k={k}] op={slot.get('op')}")
            print(f"          args[0]: src={args[0].get('source')} idx={args[0].get('index')} value={args[0].get('value')} → attn_target={t1} tok={t1_text!r}")
            if len(args) > 1:
                print(f"          args[1]: src={args[1].get('source')} idx={args[1].get('index')} value={args[1].get('value')} → attn_target={t2} tok={t2_text!r}")

    # Now (optional) — load v88 ckpt and dump current attn peaks to compare.
    CKPT = os.environ.get("CKPT", "")
    if not CKPT:
        print("\n(no CKPT set — skipping model attn peak diag)")
        return

    print(f"\n=== loading model + ckpt {CKPT} ===")
    sd = _load_state()
    model = load_breathing(cfg, sd=sd)
    cast_model_fp32(model)
    Device[Device.DEFAULT].synchronize()
    del sd
    ckpt_sd = safe_load(CKPT)
    info = model.load_state_dict(ckpt_sd, strict=False)
    print(f"  loaded ckpt; missing={len(info['missing'])} unexpected={len(info['unexpected'])}")
    del ckpt_sd

    # Inherit v86 K/V into v89 args1/args2 (mimics what V89_INHERIT_V86=1 does).
    dec = model.v85_slot_decoder
    k_src = dec.v86_args_k_proj.detach().contiguous()
    v_src = dec.v86_args_v_proj.detach().contiguous()
    dec.v89_args1_k_proj.assign(k_src.contiguous()).realize()
    dec.v89_args2_k_proj.assign(k_src.contiguous()).realize()
    dec.v89_args1_v_proj.assign(v_src.contiguous()).realize()
    dec.v89_args2_v_proj.assign(v_src.contiguous()).realize()
    print("[diag] v89 args1/args2 K/V inherited from v86 — args1 and args2 should give IDENTICAL attn at init")

    Tensor.training = False

    for b, ex in enumerate(examples):
        tokens = Tensor(tokens_np[b:b+1], dtype=dtypes.int).realize()
        kv_mask = Tensor(enc["kv_mask_np"][b:b+1], dtype=dtypes.float).realize()
        numbers_mask_t = Tensor(numbers_mask[b:b+1], dtype=dtypes.float).realize()
        number_span_idx = Tensor(number_spans[b:b+1], dtype=dtypes.int).realize()

        out = model.breathe_with_lookup(
            tokens, K, return_per_breath_x=True, return_waist_compressed=True,
            notebook_pool_mask=kv_mask, main_attn_mask=kv_mask)
        if isinstance(out, tuple):
            waist_per_breath = out[3] if len(out) >= 4 else out[-1]
        elif isinstance(out, dict):
            waist_per_breath = out["waist_per_breath"]
        else:
            print("ERROR")
            return

        T = tokens.shape[1]
        prompt_emb_full = model.embed(tokens).cast(dtypes.float)
        prompt_emb_masked = prompt_emb_full * kv_mask.cast(dtypes.float).reshape(1, T, 1)
        positions = Tensor.arange(T).reshape(1, 1, T)
        start_idx = number_span_idx[:, :, 0:1]
        end_idx = number_span_idx[:, :, 1:2]
        span_mask = ((positions >= start_idx) & (positions < end_idx)).cast(dtypes.float)
        span_len = span_mask.sum(axis=-1, keepdim=True) + 1e-6
        numbers_emb = (span_mask @ prompt_emb_masked) / span_len
        numbers_emb = numbers_emb * numbers_mask_t.reshape(1, -1, 1)

        waist_k = waist_per_breath[K - 1].cast(dtypes.float)
        waist_k_masked = waist_k * kv_mask.cast(waist_k.dtype).reshape(1, -1, 1)
        mask_count = kv_mask.cast(dtypes.float).sum(axis=1, keepdim=True) + 1e-6
        waist_pooled = waist_k_masked.sum(axis=1, keepdim=True) * (1.0 / mask_count.reshape(1, 1, 1))

        decoder_out = model.v85_slot_decoder.forward(
            waist_pooled, numbers_emb, numbers_mask_t,
            waist_full=waist_k, waist_full_mask=kv_mask)

        if "args1_attn" not in decoder_out:
            print("WARN: args1_attn not in decoder_out — V89_SUPERVISED_ATTN not active?")
            return

        a1_attn = decoder_out["args1_attn"].numpy()[0]   # (K_max, T_full)
        a2_attn = decoder_out["args2_attn"].numpy()[0]
        plen = int(prompt_lens[b])
        a1_p = a1_attn[:, :plen]
        a2_p = a2_attn[:, :plen]
        a1_p = a1_p / (a1_p.sum(axis=1, keepdims=True) + 1e-12)
        a2_p = a2_p / (a2_p.sum(axis=1, keepdims=True) + 1e-12)

        print(f"\n--- Problem {b} model attn vs gold ---")
        for k in range(min(K_MAX, len(ex.dag_slots) + 2)):
            t1 = int(args_attn_target[b, k, 0])
            t2 = int(args_attn_target[b, k, 1])
            peak1 = int(np.argmax(a1_p[k]))
            peak2 = int(np.argmax(a2_p[k]))
            peak1_v = float(a1_p[k, peak1])
            peak2_v = float(a2_p[k, peak2])
            mass_at_t1 = float(a1_p[k, t1]) if 0 <= t1 < plen else 0.0
            mass_at_t2 = float(a2_p[k, t2]) if 0 <= t2 < plen else 0.0
            hit1 = "HIT" if peak1 == t1 else "miss"
            hit2 = "HIT" if peak2 == t2 else "miss"
            print(f"  k={k:2d}  a1: target={t1:3d}  peak={peak1:3d}({peak1_v:.3f})  mass@target={mass_at_t1:.3f}  {hit1}"
                  f"   a2: target={t2:3d}  peak={peak2:3d}({peak2_v:.3f})  mass@target={mass_at_t2:.3f}  {hit2}")


if __name__ == "__main__":
    main()
