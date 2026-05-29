"""v91 gradient norm diagnostic — verify ops/args grad parity after simplification.

Compares to the v90 baseline:
  ops_codebook grad_L2 ≈ 8.6 vs args_k_w grad_L2 ≈ 0.88 (10x attenuation)
  args1_q_w / args2_q_w / v89_*_proj / v86_* all in the 0.3-0.5 range.

Under v91, the args pathway collapses to a single arg_pos_emb (2, H) + the
existing waist_pool / slot_pos / ops_codebook / types_codebook chain. We expect
the args gradient to flow at roughly the SAME magnitude as ops (since both now
go through a single matmul into a softmax).
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from tinygrad import Tensor, Device, dtypes
from tinygrad.nn.state import safe_load

os.environ.setdefault("V85_QUERYABLE", "1")
os.environ.setdefault("V85_K_MAX", "10")
os.environ.setdefault("V85_N_MAX", "20")
os.environ.setdefault("V85_TYPES_N", "32")
os.environ.setdefault("V86_ARGS_CROSS_ATTN", "1")
os.environ.setdefault("V86_ACTIVE_POS_WEIGHT", "1.0")
os.environ.setdefault("V87_SLOT_POS_INIT_SCALE", "0.5")
os.environ.setdefault("V89_SUPERVISED_ATTN", "0")
os.environ.setdefault("V91_SIMPLIFIED_ARGS", "1")
os.environ.setdefault("BFIELD_WAIST", "512")
os.environ.setdefault("BFIELD_END_OF_BREATH", "1")
os.environ.setdefault("BFIELD_ALPHA", "1.0")
os.environ.setdefault("WAIST_CODEBOOK_N", "64")
os.environ.setdefault("CONTROLLER_DECODE", "1")
os.environ.setdefault("PER_BREATH_DECODE", "1")
os.environ.setdefault("NOTEBOOK_V24", "1")
os.environ.setdefault("NOTEBOOK_DUAL", "1")
os.environ.setdefault("NOTEBOOK_POOL_MODE", "attn")
os.environ.setdefault("PER_HEAD_PITCH", "1")
os.environ.setdefault("SINE_TEMP", "1")
os.environ.setdefault("CONSTANT_RADIUS", "1")
os.environ.setdefault("BREATH_TIME_EMBED", "1")
os.environ.setdefault("CROSS_BREATH_HANDOFF", "1")
os.environ.setdefault("ABLATE_BREATH_ROTATION", "1")
os.environ.setdefault("CONTROLLER_N_LAYERS", "4")
os.environ.setdefault("V78_HEAD_CODEBOOK", "1")
os.environ.setdefault("V78_HEAD_CODEBOOK_N", "32")
os.environ.setdefault("MAX_STEP_BASE", "2.0")
os.environ.setdefault("MAX_STEP_MIN", "0.1")
os.environ.setdefault("NOTEBOOK_ACCUMULATE_ENABLED", "1")
os.environ.setdefault("NOTEBOOK_NO_DETACH", "1")
os.environ.setdefault("BREATH_EMBED_ORTHO_INIT", "2.0")
os.environ.setdefault("PER_BREATH_TEMP", "1")
os.environ.setdefault("BREATH_NORM_OSC", "1")
os.environ.setdefault("V79_CAUSAL_MASKS", "1")
os.environ.setdefault("V81_MAIN_ATTN_MASK", "1")
os.environ.setdefault("V83_GRADUATION", "1")
os.environ.setdefault("BFIELD_WAIST_SCHEDULE", "64,256,384,512,512")
os.environ.setdefault("PROMPT_REFRESH_ALPHA", "0.1")
os.environ.setdefault("DEV", "PCI+AMD")

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


def l2_norm(t: Tensor) -> float:
    arr = t.cast(dtypes.float).numpy()
    return float(np.sqrt((arr.astype(np.float64) ** 2).sum()))


def main():
    cfg = Config()
    CKPT = os.environ.get("CKPT", ".cache/gsm8k_steps_ckpts/v90_smoke_step100.safetensors")
    DATA = os.environ.get("DATA", ".cache/gsm8k_steps_v85_train.jsonl")
    K = int(os.environ.get("K", "5"))
    FIXED_LEN = int(os.environ.get("FIXED_LEN", "224"))
    K_MAX = int(os.environ.get("V85_K_MAX", "10"))
    N_MAX = int(os.environ.get("V85_N_MAX", "20"))

    print(f"=== v91 grad-norm diagnostic ===")
    print(f"  ckpt: {CKPT}")
    print(f"  data: {DATA}")

    tok = load_tokenizer()
    sd = _load_state()
    model = load_breathing(cfg, sd=sd)
    cast_model_fp32(model)
    Device[Device.DEFAULT].synchronize()
    del sd
    if os.path.exists(CKPT):
        ckpt_sd = safe_load(CKPT)
        info = model.load_state_dict(ckpt_sd, strict=False)
        print(f"  loaded; missing {len(info['missing'])}, unexpected {len(info['unexpected'])}")
        del ckpt_sd

    dec = model.v85_slot_decoder

    examples = load_gsm8k_v85(DATA, min_k=2, max_k=6,
                              require_sympy_match=True, bucket_by_k=False)
    examples = examples[:4]
    print(f"  using {len(examples)} examples for forward/backward pass")

    enc = _v85_encode_batch(examples, tok, FIXED_LEN, K_MAX, N_MAX)
    tokens = Tensor(enc["tokens_np"], dtype=dtypes.int).realize()
    number_span_idx = Tensor(enc["number_span_token_idx_np"], dtype=dtypes.int).realize()
    numbers_mask = Tensor(enc["numbers_mask_np"], dtype=dtypes.float).realize()
    ops_targets = Tensor(enc["ops_targets_np"], dtype=dtypes.int).realize()
    types_targets = Tensor(enc["types_targets_np"], dtype=dtypes.int).realize()
    args1_targets = Tensor(enc["args1_targets_np"], dtype=dtypes.int).realize()
    args2_targets = Tensor(enc["args2_targets_np"], dtype=dtypes.int).realize()
    active_targets = Tensor(enc["active_targets_np"], dtype=dtypes.float).realize()
    active_valid = Tensor(enc["active_valid_np"], dtype=dtypes.float).realize()
    kv_mask = Tensor(enc["kv_mask_np"], dtype=dtypes.float).realize()
    B = len(examples)

    Tensor.training = True
    for p in model.parameters():
        p.requires_grad = True
        if hasattr(p, "grad"):
            p.grad = None

    _final, _mw, _per_breath_x, waist_per_breath = model.breathe_with_lookup(
        tokens, K, return_per_breath_x=True, return_waist_compressed=True,
        notebook_pool_mask=kv_mask, main_attn_mask=kv_mask)

    H_ = cfg.hidden
    T = tokens.shape[1]
    prompt_emb_full = model.embed(tokens).cast(dtypes.float)
    prompt_emb_masked = prompt_emb_full * kv_mask.cast(dtypes.float).reshape(B, T, 1)
    positions = Tensor.arange(T).reshape(1, 1, T)
    start_idx = number_span_idx[:, :, 0:1]
    end_idx = number_span_idx[:, :, 1:2]
    span_mask = ((positions >= start_idx) & (positions < end_idx)).cast(dtypes.float)
    span_len = span_mask.sum(axis=-1, keepdim=True) + 1e-6
    numbers_emb = (span_mask @ prompt_emb_masked) / span_len
    numbers_emb = numbers_emb * numbers_mask.reshape(B, -1, 1)

    per_breath_losses = []
    for k in range(K):
        waist_k = waist_per_breath[k].cast(dtypes.float)
        waist_k_masked = waist_k * kv_mask.cast(waist_k.dtype).reshape(B, -1, 1)
        mask_count = kv_mask.cast(dtypes.float).sum(axis=1, keepdim=True) + 1e-6
        waist_pooled = waist_k_masked.sum(axis=1, keepdim=True) * (1.0 / mask_count.reshape(B, 1, 1))

        decoder_out = model.v85_slot_decoder.forward(
            waist_pooled, numbers_emb, numbers_mask,
            waist_full=waist_k, waist_full_mask=kv_mask)
        ops_logits = decoder_out["ops_logits"]
        types_logits = decoder_out["types_logits"]
        args1_logits = decoder_out["args1_logits"]
        args2_logits = decoder_out["args2_logits"]
        active_logits = decoder_out["active_logits"]

        ops_ce = ops_logits.sparse_categorical_crossentropy(
            ops_targets, ignore_index=-100, reduction="mean")
        types_ce = types_logits.sparse_categorical_crossentropy(
            types_targets, ignore_index=-100, reduction="mean")
        args1_ce = args1_logits.sparse_categorical_crossentropy(
            args1_targets, ignore_index=-100, reduction="mean")
        args2_ce = args2_logits.sparse_categorical_crossentropy(
            args2_targets, ignore_index=-100, reduction="mean")
        args_ce = (args1_ce + args2_ce) * 0.5

        z = active_logits
        y = active_targets
        log1p_exp_neg_abs = (1.0 + (-z.abs()).exp()).log()
        sp_neg = (-z).maximum(0.0) + log1p_exp_neg_abs
        sp_pos = z.maximum(0.0) + log1p_exp_neg_abs
        pos_w = 1.0
        bce_per = pos_w * y * sp_neg + (1.0 - y) * sp_pos
        active_ce = (bce_per * active_valid).sum() / (active_valid.sum() + 1.0)

        breath_loss = ops_ce + types_ce + args_ce + active_ce
        per_breath_losses.append(breath_loss)

    total_loss = sum(per_breath_losses[1:], per_breath_losses[0]) / float(K)
    print(f"  total_loss (mean over {K} breaths): {float(total_loss.numpy()):.4f}")
    total_loss.backward()
    Device[Device.DEFAULT].synchronize()

    def report_param(name, p):
        if not hasattr(p, "grad") or p.grad is None:
            print(f"    {name:38s} grad=None   shape={tuple(p.shape)}")
            return None
        n = l2_norm(p.grad)
        print(f"    {name:38s} grad_L2={n:.3e}  shape={tuple(p.shape)}")
        return n

    print("\nOps pathway (waist-side to softmax-side):")
    g_waist_pool_w = report_param("waist_pool_proj_w", dec.waist_pool_proj_w)
    g_slot_pos = report_param("slot_pos_embed", dec.slot_pos_embed)
    g_slot_ln_g = report_param("slot_ln_g", dec.slot_ln_g)
    g_ops_codebook = report_param("ops_codebook", dec.ops_codebook)
    g_types_codebook = report_param("types_codebook", dec.types_codebook)

    print("\nArgs pathway (v91 — only arg_pos_emb is new):")
    g_arg_pos_emb = report_param("arg_pos_emb", dec.arg_pos_emb)

    print("\nDeprecated tensors (should have grad=None or near-zero):")
    report_param("args1_q_w",        dec.args1_q_w)
    report_param("args2_q_w",        dec.args2_q_w)
    report_param("args_k_w",         dec.args_k_w)
    report_param("v86_args_q_proj",  dec.v86_args_q_proj)
    report_param("v86_args_k_proj",  dec.v86_args_k_proj)
    report_param("v86_args_v_proj",  dec.v86_args_v_proj)
    report_param("v86_args_slot_pos", dec.v86_args_slot_pos)
    report_param("v89_args1_k_proj", dec.v89_args1_k_proj)
    report_param("v89_args1_v_proj", dec.v89_args1_v_proj)

    print("\n" + "=" * 60)
    print("SUMMARY (compare to v90 baseline)")
    print("=" * 60)
    print(f"  v90 baseline (audit log):")
    print(f"    ops_codebook  : 8.577e+00")
    print(f"    args_k_w      : 8.749e-01   (~10x attenuation)")
    print(f"    args1_q_w     : 5.187e-01   (~16x)")
    print(f"    args2_q_w     : 4.212e-01   (~20x)")
    print()
    print(f"  v91 (current run):")
    print(f"    ops_codebook  : {g_ops_codebook:.3e}")
    print(f"    arg_pos_emb   : {g_arg_pos_emb:.3e}")
    print(f"    ops/arg ratio : {(g_ops_codebook / max(g_arg_pos_emb, 1e-12)):.2f}x")
    # Also report types_codebook for comparison (should be similar to ops_codebook
    # since both are single matmul + softmax).
    print(f"    types_codebook: {g_types_codebook:.3e}  (parallel reference)")


if __name__ == "__main__":
    main()
