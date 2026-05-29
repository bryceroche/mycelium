"""v88 diagnostic — verify K/V projection reinit breaks args cross-attn dead zone.

Loads a ckpt (e.g. v87 step300 to baseline, or v88 mid-train to show progress).
Optionally REINITIALIZES v86_args_k_proj and v86_args_v_proj at scale
V88_KV_PROJ_INIT_SCALE so the diag can show pre- vs post-reinit L2s + attention
JSD across slots over a few sample problems.

Compare:
  - V88_REINIT_KV_PROJ=0: baseline (whatever the ckpt has)
  - V88_REINIT_KV_PROJ=1 with scale 0.02: K L2 from ~0.1 to ~5.7, JSD should
    lift from ~0.001 to something measurable (>0.01) at init time.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from tinygrad import Tensor, Device, dtypes
from tinygrad.nn.state import safe_load

# Force the modes on for the diagnostic so the env reads inside the module pick them up.
os.environ["V86_ARGS_CROSS_ATTN"] = "1"
os.environ["V85_QUERYABLE"] = "1"

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
    print("=== v88 K/V proj reinit diagnostic ===")

    cfg = Config()
    tok = load_tokenizer()
    sd = _load_state()
    model = load_breathing(cfg, sd=sd)
    cast_model_fp32(model)
    Device[Device.DEFAULT].synchronize()
    del sd

    CKPT = os.environ.get("CKPT", "")
    if CKPT:
        ckpt_sd = safe_load(CKPT)
        info = model.load_state_dict(ckpt_sd, strict=False)
        print(f"loaded ckpt {CKPT}; missing {len(info['missing'])}, unexpected {len(info['unexpected'])}")
        del ckpt_sd
    else:
        print("WARNING: CKPT not set — using untrained init.")

    dec = model.v85_slot_decoder

    # Report PRE-reinit L2s.
    print("\n=== slot decoder param L2 (PRE-reinit) ===")
    for name in ["slot_pos_embed", "v86_args_slot_pos", "v86_args_q_proj",
                 "v86_args_k_proj", "v86_args_v_proj"]:
        t = getattr(dec, name)
        n = float((t.cast(dtypes.float) ** 2).sum().sqrt().numpy())
        s = tuple(t.shape)
        print(f"  {name:24s} shape={s} L2={n:.4f}")

    # Apply v88 reinit if requested.
    V88_REINIT = bool(int(os.environ.get("V88_REINIT_KV_PROJ", "0")))
    V88_SCALE = float(os.environ.get("V88_KV_PROJ_INIT_SCALE", "0.02"))
    if V88_REINIT and V88_SCALE > 0.0:
        K_shape = dec.v86_args_k_proj.shape
        V_shape = dec.v86_args_v_proj.shape
        new_k = (Tensor.randn(*K_shape) * V88_SCALE).cast(
            dec.v86_args_k_proj.dtype).to(dec.v86_args_k_proj.device).contiguous()
        dec.v86_args_k_proj.assign(new_k).realize()
        new_v = (Tensor.randn(*V_shape) * V88_SCALE).cast(
            dec.v86_args_v_proj.dtype).to(dec.v86_args_v_proj.device).contiguous()
        dec.v86_args_v_proj.assign(new_v).realize()
        print(f"\n[v88] reinitialized K/V proj at randn-scale {V88_SCALE}")

        # Re-report L2s.
        print("\n=== slot decoder param L2 (POST-reinit) ===")
        for name in ["slot_pos_embed", "v86_args_slot_pos", "v86_args_q_proj",
                     "v86_args_k_proj", "v86_args_v_proj"]:
            t = getattr(dec, name)
            n = float((t.cast(dtypes.float) ** 2).sum().sqrt().numpy())
            s = tuple(t.shape)
            print(f"  {name:24s} shape={s} L2={n:.4f}")
    else:
        print(f"\n[v88] no reinit (V88_REINIT_KV_PROJ={int(V88_REINIT)} scale={V88_SCALE})")

    Tensor.training = False

    # Encode 3 sample problems.
    examples = load_gsm8k_v85(
        ".cache/gsm8k_steps_v85_train.jsonl",
        min_k=2, max_k=6,
        require_sympy_match=True, bucket_by_k=False)
    if len(examples) < 3:
        examples = load_gsm8k_v85(
            ".cache/gsm8k_steps_v85_train.jsonl",
            min_k=2, max_k=6,
            require_sympy_match=False, bucket_by_k=False)
    examples = examples[:3]
    print(f"\n=== running v86/v87/v88 args attn on {len(examples)} samples ===")

    K = 5
    FIXED_LEN = 224
    K_MAX = 10
    N_MAX = 20

    pjsds_all = []
    for ei, ex in enumerate(examples):
        enc = _v85_encode_batch([ex], tok, FIXED_LEN, K_MAX, N_MAX)
        tokens = Tensor(enc["tokens_np"], dtype=dtypes.int).realize()
        number_span_idx = Tensor(enc["number_span_token_idx_np"], dtype=dtypes.int).realize()
        numbers_mask = Tensor(enc["numbers_mask_np"], dtype=dtypes.float).realize()
        kv_mask = Tensor(enc["kv_mask_np"], dtype=dtypes.float).realize()

        out = model.breathe_with_lookup(
            tokens, K, return_per_breath_x=True, return_waist_compressed=True,
            notebook_pool_mask=kv_mask, main_attn_mask=kv_mask)
        if isinstance(out, tuple):
            waist_per_breath = out[3] if len(out) >= 4 else out[-1]
        elif isinstance(out, dict):
            waist_per_breath = out["waist_per_breath"]
        else:
            print("ERROR: unexpected breathe_with_lookup return type", type(out))
            return

        if waist_per_breath is None:
            print("ERROR: waist_per_breath is None")
            return

        # Build numbers_emb.
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

        waist_k = waist_per_breath[K - 1].cast(dtypes.float)
        waist_k_masked = waist_k * kv_mask.cast(waist_k.dtype).reshape(B, -1, 1)
        mask_count = kv_mask.cast(dtypes.float).sum(axis=1, keepdim=True) + 1e-6
        waist_pooled = waist_k_masked.sum(axis=1, keepdim=True) * (1.0 / mask_count.reshape(B, 1, 1))

        decoder_out = model.v85_slot_decoder.forward(
            waist_pooled, numbers_emb, numbers_mask,
            waist_full=waist_k, waist_full_mask=kv_mask)

        if "args_attn" not in decoder_out:
            print(f"  [problem {ei}] V86_ARGS_CROSS_ATTN not active.")
            return

        args_attn = decoder_out["args_attn"].numpy()
        prompt_len = int(enc["prompt_lens_np"][0])

        print(f"\n  --- Problem {ei}: {ex.problem[:80]!r} (prompt_len={prompt_len}) ---")
        slot_dists = args_attn[0]
        slot_dists_p = slot_dists[:, :prompt_len]
        slot_dists_p = slot_dists_p / (slot_dists_p.sum(axis=1, keepdims=True) + 1e-12)

        eps = 1e-12
        ref = slot_dists_p[0]
        for k in range(K_MAX):
            row = slot_dists_p[k]
            top_idx = np.argsort(-row)[:3]
            peak = row.max()
            if k == 0:
                jsd = 0.0
            else:
                m = 0.5 * (row + ref)
                kl_p = (row * (np.log(row + eps) - np.log(m + eps))).sum()
                kl_q = (ref * (np.log(ref + eps) - np.log(m + eps))).sum()
                jsd = 0.5 * (kl_p + kl_q)
            print(f"    slot {k:2d}  peak={peak:.4f}  top_pos={top_idx.tolist()}  jsd_vs_slot0={jsd:.4f}")

        K_act = K_MAX
        pairwise_jsds = []
        for a in range(K_act):
            for b in range(a + 1, K_act):
                pa = slot_dists_p[a] + eps
                pb = slot_dists_p[b] + eps
                m = 0.5 * (pa + pb)
                kl_a = (pa * (np.log(pa) - np.log(m))).sum()
                kl_b = (pb * (np.log(pb) - np.log(m))).sum()
                pairwise_jsds.append(0.5 * (kl_a + kl_b))
        mean_pjsd = float(np.mean(pairwise_jsds))
        max_pjsd = float(np.max(pairwise_jsds))
        print(f"    >>> mean pairwise JSD: {mean_pjsd:.4f}  max: {max_pjsd:.4f}")
        pjsds_all.append(mean_pjsd)

    print(f"\n=== SUMMARY ===")
    print(f"V88_REINIT_KV_PROJ={int(V88_REINIT)}  V88_KV_PROJ_INIT_SCALE={V88_SCALE}")
    print(f"mean pairwise JSD per problem: {[round(x, 6) for x in pjsds_all]}")
    print(f"overall mean pairwise JSD: {np.mean(pjsds_all):.6f}")
    print(f"target: > 0.1; v87 baseline ~0.0005; uniform ceiling ln(2)={np.log(2):.4f}")


if __name__ == "__main__":
    main()
