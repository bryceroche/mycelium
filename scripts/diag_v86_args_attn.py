"""v86 diagnostic — verify per-slot args cross-attention patterns are non-uniform.

With a fresh model (v85_smoke_step500 warm-start) and V86_ARGS_CROSS_ATTN=1,
load a few v85 examples and compute the args cross-attention weights for each
slot. Confirm:

  1. Different slots produce DIFFERENT attention patterns (not all same).
  2. The patterns are not uniform across the prompt (some positions higher than
     others).
  3. The patterns respect the kv_mask (zero weight outside the prompt).

NOTE: v86_args_k_proj and v86_args_v_proj are zero-init at the start, so the
attention will be UNIFORM over valid positions when run on an unfit model.
That's a feature, not a bug — it means the cross-attn contribution is initially
zero (slot_args_ctx = 0) so warm-start behavior is byte-identical to v85.

This diagnostic checks two cases:
  (a) zero-init (just-loaded ckpt): attention should be uniform across prompt.
  (b) after a few training steps: attention should differentiate per slot.

For just diagnostic visibility, we also dump the per-slot pos_embed norms to
confirm slots are differentiated at the slot_query level (slot_pos_embed should
be non-zero from v85 init).
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from tinygrad import Tensor, Device, dtypes
from tinygrad.nn.state import safe_load

# Force v86 on for the diagnostic
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
    print("=== v86 args cross-attn diagnostic ===")

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

    Tensor.training = False

    # Inspect param norms.
    print("\n=== v85/v86 slot decoder param norms ===")
    dec = model.v85_slot_decoder
    for name in ["slot_pos_embed", "waist_pool_proj_w", "ops_codebook", "types_codebook",
                 "args1_q_w", "args2_q_w", "args_k_w",
                 "v86_args_q_proj", "v86_args_k_proj", "v86_args_v_proj",
                 "v86_args_slot_pos", "active_head_w", "active_head_b"]:
        t = getattr(dec, name)
        n = float((t.cast(dtypes.float) ** 2).sum().sqrt().numpy())
        s = tuple(t.shape)
        print(f"  {name:24s} shape={s} L2={n:.4f}")

    # Inspect slot_pos_embed pairwise cosine similarity to verify slots are
    # differentiated already at the slot_query level.
    pos = dec.slot_pos_embed.cast(dtypes.float).numpy()  # (K_max, H)
    K_max = pos.shape[0]
    pos_norm = pos / (np.linalg.norm(pos, axis=-1, keepdims=True) + 1e-8)
    cos = pos_norm @ pos_norm.T
    print(f"\n=== slot_pos_embed pairwise cosine ===")
    print(f"  mean off-diag: {(cos.sum() - K_max) / (K_max * (K_max - 1)):.4f}")
    print(f"  max off-diag:  {(cos - np.eye(K_max)).max():.4f}")
    print(f"  min off-diag:  {(cos - np.eye(K_max) * 100).min():.4f}")

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
    print(f"\n=== running v86 args attn on {len(examples)} samples ===")

    K = 5  # n breaths
    FIXED_LEN = 224
    K_MAX = 10
    N_MAX = 20

    for ei, ex in enumerate(examples):
        enc = _v85_encode_batch([ex], tok, FIXED_LEN, K_MAX, N_MAX)
        tokens = Tensor(enc["tokens_np"], dtype=dtypes.int).realize()
        number_span_idx = Tensor(enc["number_span_token_idx_np"], dtype=dtypes.int).realize()
        numbers_mask = Tensor(enc["numbers_mask_np"], dtype=dtypes.float).realize()
        kv_mask = Tensor(enc["kv_mask_np"], dtype=dtypes.float).realize()

        _final, _mw, _pbx, waist_per_breath = model.breathe_with_lookup(
            tokens, K, return_per_breath_x=True, return_waist_compressed=True,
            notebook_pool_mask=kv_mask, main_attn_mask=kv_mask)

        # Build numbers_emb (token-span pool).
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

        # Use final-breath waist.
        waist_k = waist_per_breath[K - 1].cast(dtypes.float)
        waist_k_masked = waist_k * kv_mask.cast(waist_k.dtype).reshape(B, -1, 1)
        mask_count = kv_mask.cast(dtypes.float).sum(axis=1, keepdim=True) + 1e-6
        waist_pooled = waist_k_masked.sum(axis=1, keepdim=True) * (1.0 / mask_count.reshape(B, 1, 1))

        decoder_out = model.v85_slot_decoder.forward(
            waist_pooled, numbers_emb, numbers_mask,
            waist_full=waist_k, waist_full_mask=kv_mask)

        if "args_attn" not in decoder_out:
            print(f"  [problem {ei}] V86_ARGS_CROSS_ATTN not active — no args_attn returned. Aborting.")
            return

        args_attn = decoder_out["args_attn"].numpy()  # (1, K_max, T_full)
        prompt_len = int(enc["prompt_lens_np"][0])

        print(f"\n  --- Problem {ei}: {ex.problem[:80]!r} (prompt_len={prompt_len}) ---")
        # For each of slots 0..K_max-1, report:
        #   - top-3 attended positions in the prompt
        #   - peak weight value
        #   - JSD vs slot 0's pattern (as a "differentiation" metric)
        slot_dists = args_attn[0]  # (K_max, T_full)

        # Restrict to prompt positions for clarity.
        slot_dists_p = slot_dists[:, :prompt_len]
        slot_dists_p = slot_dists_p / (slot_dists_p.sum(axis=1, keepdims=True) + 1e-12)

        eps = 1e-12
        ref = slot_dists_p[0]
        for k in range(K_MAX):
            row = slot_dists_p[k]
            top_idx = np.argsort(-row)[:3]
            peak = row.max()
            # JSD vs slot 0
            if k == 0:
                jsd = 0.0
            else:
                m = 0.5 * (row + ref)
                kl_p = (row * (np.log(row + eps) - np.log(m + eps))).sum()
                kl_q = (ref * (np.log(ref + eps) - np.log(m + eps))).sum()
                jsd = 0.5 * (kl_p + kl_q)
            print(f"    slot {k:2d}  peak={peak:.4f}  top_pos={top_idx.tolist()}  jsd_vs_slot0={jsd:.4f}")

        # Aggregate metric: mean pairwise JSD across slots.
        # Uniform attn → mean pairwise JSD = 0.
        sd_p = slot_dists_p
        K_act = K_MAX
        pairwise_jsds = []
        for a in range(K_act):
            for b in range(a + 1, K_act):
                pa = sd_p[a] + eps
                pb = sd_p[b] + eps
                m = 0.5 * (pa + pb)
                kl_a = (pa * (np.log(pa) - np.log(m))).sum()
                kl_b = (pb * (np.log(pb) - np.log(m))).sum()
                pairwise_jsds.append(0.5 * (kl_a + kl_b))
        mean_pjsd = float(np.mean(pairwise_jsds))
        max_pjsd = float(np.max(pairwise_jsds))
        print(f"    >>> mean pairwise JSD over {len(pairwise_jsds)} slot pairs: {mean_pjsd:.4f}")
        print(f"    >>> max pairwise JSD: {max_pjsd:.4f}")
        print(f"    (uniform attn -> ~0; well-differentiated -> non-trivial; max ln 2={np.log(2):.4f})")


if __name__ == "__main__":
    main()
