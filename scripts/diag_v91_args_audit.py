"""v91 ARGS pathway audit — gradient norms and projection counts.

Validates the "deep chain bootstrap" hypothesis by comparing ops vs args pathways
on the v90_smoke_step100 checkpoint. Read-only.

Output:
  1. List of projections in each pathway (ops vs args) with shapes.
  2. L2 grad norms for each param in each pathway after ONE forward+backward.
  3. Verification that ops_codebook and args_q/k projections are separate params.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import numpy as np
from tinygrad import Tensor, Device, dtypes
from tinygrad.nn.state import safe_load

# Enable v85+v86+v89 paths — must be set BEFORE breathing.py imports module-level constants.
os.environ.setdefault("V85_QUERYABLE", "1")
os.environ.setdefault("V85_K_MAX", "10")
os.environ.setdefault("V85_N_MAX", "20")
os.environ.setdefault("V85_TYPES_N", "32")
os.environ.setdefault("V86_ARGS_CROSS_ATTN", "1")
os.environ.setdefault("V86_ACTIVE_POS_WEIGHT", "1.0")
os.environ.setdefault("V87_SLOT_POS_INIT_SCALE", "0.5")
os.environ.setdefault("V89_SUPERVISED_ATTN", "1")
os.environ.setdefault("V89_PROJ_INIT_SCALE", "0.02")
os.environ.setdefault("V89_SUPERVISED_ATTN_WEIGHT", "0.5")
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
    """Return scalar L2 norm of a tensor."""
    arr = t.cast(dtypes.float).numpy()
    return float(np.sqrt((arr.astype(np.float64) ** 2).sum()))


def main():
    cfg = Config()
    CKPT = os.environ.get(
        "CKPT", ".cache/gsm8k_steps_ckpts/v90_smoke_step100.safetensors")
    DATA = os.environ.get("DATA", ".cache/gsm8k_steps_v85_train.jsonl")
    K = int(os.environ.get("K", "5"))
    FIXED_LEN = int(os.environ.get("FIXED_LEN", "224"))
    K_MAX = int(os.environ.get("V85_K_MAX", "10"))
    N_MAX = int(os.environ.get("V85_N_MAX", "20"))

    print(f"=== v91 args pathway audit ===")
    print(f"  ckpt: {CKPT}")
    print(f"  data: {DATA}")

    # ==== Load model ====
    tok = load_tokenizer()
    sd = _load_state()
    model = load_breathing(cfg, sd=sd)
    cast_model_fp32(model)
    Device[Device.DEFAULT].synchronize()
    del sd
    ckpt_sd = safe_load(CKPT)
    info = model.load_state_dict(ckpt_sd, strict=False)
    print(f"  loaded; missing {len(info['missing'])}, unexpected {len(info['unexpected'])}")
    del ckpt_sd

    # ==== Section 1: enumerate projections ====
    print("\n" + "=" * 70)
    print("SECTION 1 — Projection inventory (waist -> softmax)")
    print("=" * 70)

    dec = model.v85_slot_decoder
    H = cfg.hidden
    W = 512  # BFIELD_WAIST
    Kp = dec.h_pointer
    Km = dec.K_max
    Nm = dec.N_max

    print("\nShared (both pathways traverse these to build slot_query):")
    shared_path = [
        ("waist_pool_proj_w", (W, H), "linear  (B, T, W) mean-> (B, W) -> (B, H)"),
        ("waist_pool_proj_b", (H,), "bias add"),
        ("slot_pos_embed", (Km, H), "additive learned per-slot vector"),
        ("slot_ln_g/b + LN", (H,), "layernorm (gain + bias)"),
        ("GELU", (), "non-linearity (no params, but a non-trivial chain link)"),
    ]
    for name, shape, desc in shared_path:
        print(f"    {name:30s} {str(shape):20s} {desc}")

    print("\nOPS pathway (after slot_query):")
    ops_path = [
        ("ops_codebook", (4, H), "(B,Km,H) @ codebook.T -> (B,Km,4)"),
        ("ops_head_b", (4,), "bias add"),
        ("softmax (4)", (), "decision over 4 entries"),
    ]
    for name, shape, desc in ops_path:
        print(f"    {name:30s} {str(shape):20s} {desc}")
    N_OPS = 1  # ops_codebook is the only matmul after shared chain
    print(f"  Trainable transforms downstream of waist: 1 matmul (ops_codebook.T)")

    print("\nARGS1 pathway (after slot_query):")
    args_path = [
        ("v86_args_slot_pos", (Km, H), "additive per-slot pos (also init for slot_query)"),
        ("v86_args_q_proj", (H, Kp), "Q projection for cross-attn"),
        ("v89_args1_k_proj", (W, Kp), "K projection for cross-attn"),
        ("v89_args1_v_proj", (W, H), "V projection for cross-attn"),
        ("scores = Q @ K^T", (), "matmul (no params)"),
        ("softmax_attn (T_full)", (), "first softmax inside the chain"),
        ("ctx = attn @ V", (), "matmul (no params) -> (B, Km, H)"),
        ("(residual) slot_query + ctx", (), "skip add"),
        ("args1_q_w", (H, Kp), "pointer query projection"),
        ("args_k_w", (H, Kp), "shared pointer key projection (applied to BOTH numbers and slots)"),
        ("scores2 = slot_q @ all_k^T", (), "matmul (no params)"),
        ("mask + clip + softmax(30)", (), "final softmax over 20 numbers + 10 dag slots"),
    ]
    for name, shape, desc in args_path:
        print(f"    {name:30s} {str(shape):20s} {desc}")
    N_ARGS = 5  # q_proj, k_proj, v_proj, args1_q_w, args_k_w = five matmul-style projections
    # plus v86_args_slot_pos additive embedding (counted as a trainable "transformation" too)
    print(f"  Trainable matmul-style transforms downstream of waist: 5")
    print(f"  PLUS additive v86_args_slot_pos (1 add)")
    print(f"  PLUS TWO softmaxes inside the chain (cross-attn AND final pointer)")

    print("\nARGS2 pathway (after slot_query):")
    print(f"    same shape; uses v89_args2_k_proj, v89_args2_v_proj, args2_q_w, shared args_k_w")
    print(f"    Trainable matmul-style transforms downstream of waist: 5")

    print(f"\nRatio (args / ops): {N_ARGS}/{N_OPS} = {N_ARGS/N_OPS:.1f}x")
    print(f"Softmaxes in chain (args): 2 (cross-attn + pointer)")
    print(f"Softmaxes in chain (ops):  1 (final only)")

    # ==== Section 3: verify separation ====
    print("\n" + "=" * 70)
    print("SECTION 3 — Param separation check (ops vs args)")
    print("=" * 70)
    ops_ids = {id(dec.ops_codebook), id(dec.ops_head_b)}
    args_ids = {
        id(dec.args1_q_w), id(dec.args2_q_w), id(dec.args_k_w),
        id(dec.v86_args_q_proj), id(dec.v86_args_k_proj), id(dec.v86_args_v_proj),
        id(dec.v86_args_slot_pos),
        id(dec.v89_args1_k_proj), id(dec.v89_args1_v_proj),
        id(dec.v89_args2_k_proj), id(dec.v89_args2_v_proj),
    }
    overlap = ops_ids & args_ids
    print(f"  ops params: {len(ops_ids)}")
    print(f"  args params: {len(args_ids)}")
    print(f"  overlap: {len(overlap)}")
    print(f"  Separate: {len(overlap) == 0}")

    # ==== Section 2: gradient norms ====
    print("\n" + "=" * 70)
    print("SECTION 2 — Gradient norm diagnostic (4 examples, 1 fwd+bwd)")
    print("=" * 70)

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
    args_attn_target = Tensor(enc["args_attn_target_np"], dtype=dtypes.int).realize()
    B = len(examples)

    # Clear any prior grads. Tinygrad only fills .grad on tensors with
    # requires_grad=True. AdamW would set this implicitly; we set it manually
    # so we don't have to actually run an opt step.
    Tensor.training = True
    for p in model.parameters():
        p.requires_grad = True
        if hasattr(p, "grad"):
            p.grad = None

    # FORWARD (eager — same as eval, replicating the v85 train step body).
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

    a1_attn_target = args_attn_target[:, :, 0]
    a2_attn_target = args_attn_target[:, :, 1]

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

        if "args1_attn_scores" in decoder_out:
            a1_scores = decoder_out["args1_attn_scores"]
            a2_scores = decoder_out["args2_attn_scores"]
            a1_attn_ce = a1_scores.sparse_categorical_crossentropy(
                a1_attn_target, ignore_index=-100, reduction="mean")
            a2_attn_ce = a2_scores.sparse_categorical_crossentropy(
                a2_attn_target, ignore_index=-100, reduction="mean")
            attn_aux_ce = (a1_attn_ce + a2_attn_ce) * 0.5
        else:
            attn_aux_ce = Tensor.zeros((), dtype=dtypes.float)

        v89_w = 0.5
        breath_loss = ops_ce + types_ce + args_ce + active_ce + v89_w * attn_aux_ce
        per_breath_losses.append(breath_loss)

    total_loss = sum(per_breath_losses[1:], per_breath_losses[0]) / float(K)
    print(f"  total_loss (mean over {K} breaths): {float(total_loss.numpy()):.4f}")
    total_loss.backward()
    Device[Device.DEFAULT].synchronize()

    # Now print per-parameter grad norms.
    def report_param(name, p, depth_idx):
        if not hasattr(p, "grad") or p.grad is None:
            print(f"    [{depth_idx:2d}]  {name:38s} grad=None   shape={p.shape}")
            return None
        n = l2_norm(p.grad)
        print(f"    [{depth_idx:2d}]  {name:38s} grad_L2={n:.3e}  shape={tuple(p.shape)}")
        return n

    print("\nOps pathway gradients (waist-side to softmax-side):")
    ops_norms = {}
    ops_norms["waist_pool_proj_w"] = report_param("waist_pool_proj_w", dec.waist_pool_proj_w, 0)
    ops_norms["waist_pool_proj_b"] = report_param("waist_pool_proj_b", dec.waist_pool_proj_b, 1)
    ops_norms["slot_pos_embed"]    = report_param("slot_pos_embed",    dec.slot_pos_embed, 2)
    ops_norms["slot_ln_g"]         = report_param("slot_ln_g",         dec.slot_ln_g, 3)
    ops_norms["slot_ln_b"]         = report_param("slot_ln_b",         dec.slot_ln_b, 4)
    ops_norms["ops_codebook"]      = report_param("ops_codebook",      dec.ops_codebook, 5)
    ops_norms["ops_head_b"]        = report_param("ops_head_b",        dec.ops_head_b, 6)

    print("\nArgs pathway gradients (waist-side to softmax-side):")
    args_norms = {}
    # shared chain (already reported above but also relevant for args)
    args_norms["[shared] waist_pool_proj_w"] = ops_norms["waist_pool_proj_w"]
    args_norms["[shared] slot_pos_embed"]    = ops_norms["slot_pos_embed"]
    args_norms["v86_args_slot_pos"]     = report_param("v86_args_slot_pos",  dec.v86_args_slot_pos, 0)
    args_norms["v86_args_q_proj"]       = report_param("v86_args_q_proj",    dec.v86_args_q_proj, 1)
    args_norms["v89_args1_k_proj"]      = report_param("v89_args1_k_proj",   dec.v89_args1_k_proj, 2)
    args_norms["v89_args1_v_proj"]      = report_param("v89_args1_v_proj",   dec.v89_args1_v_proj, 3)
    args_norms["v89_args2_k_proj"]      = report_param("v89_args2_k_proj",   dec.v89_args2_k_proj, 4)
    args_norms["v89_args2_v_proj"]      = report_param("v89_args2_v_proj",   dec.v89_args2_v_proj, 5)
    args_norms["args1_q_w"]             = report_param("args1_q_w",          dec.args1_q_w, 6)
    args_norms["args2_q_w"]             = report_param("args2_q_w",          dec.args2_q_w, 7)
    args_norms["args_k_w"]              = report_param("args_k_w",           dec.args_k_w, 8)

    # Summary side-by-side.
    print("\n" + "=" * 70)
    print("SIDE-BY-SIDE SUMMARY")
    print("=" * 70)
    print(f"  ops_codebook  grad_L2:         {ops_norms['ops_codebook']:.3e}")
    print(f"  ops_head_b    grad_L2:         {ops_norms['ops_head_b']:.3e}")
    print(f"  ----")
    print(f"  args_k_w      grad_L2:         {args_norms['args_k_w']:.3e}")
    print(f"  args1_q_w     grad_L2:         {args_norms['args1_q_w']:.3e}")
    print(f"  args2_q_w     grad_L2:         {args_norms['args2_q_w']:.3e}")
    print(f"  v89_args1_k_proj grad_L2:      {args_norms['v89_args1_k_proj']:.3e}")
    print(f"  v89_args1_v_proj grad_L2:      {args_norms['v89_args1_v_proj']:.3e}")
    print(f"  v89_args2_k_proj grad_L2:      {args_norms['v89_args2_k_proj']:.3e}")
    print(f"  v89_args2_v_proj grad_L2:      {args_norms['v89_args2_v_proj']:.3e}")
    print(f"  v86_args_q_proj  grad_L2:      {args_norms['v86_args_q_proj']:.3e}")


if __name__ == "__main__":
    main()
