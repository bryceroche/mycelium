"""v109pi factor graph — v109 + per-breath Q rotation (Option A π-cycled).

Architecture: v109 base (single-token-per-variable, waist + alternation, tree
codebook output) + ONE new mechanism: at each breath k, Q is rotated pairwise
across head_dim by phase k·π/K_max BEFORE the Q@K product. K stays unrotated.

  At breath k = 0: phase = 0, cos=1, sin=0 → Q unchanged → byte-identical to v109
  At breath k = K-1: phase = (K-1)·π/K → max Q rotation, most different attn

The rotation is applied PAIRWISE on head_dim (same pattern as RoPE) but
UNIFORMLY across sequence positions. So this is NOT position-dependent RoPE —
it's just per-breath Q rotation that makes attention patterns differ between
breaths.

Mathematically the per-breath rotation is:
  q_pi[..., 2i]   = cos(k·π/K) · q[..., 2i]   - sin(k·π/K) · q[..., 2i+1]
  q_pi[..., 2i+1] = sin(k·π/K) · q[..., 2i]   + cos(k·π/K) · q[..., 2i+1]

This adds another per-breath diversity mechanism alongside breath_embed[k],
delta_gate[k], and the alternation (commit/expand). The hypothesis: with v109's
waist+alternation already providing commit-propagate rhythm, π-cycling the Q
rotation adds attention-pattern diversity that lets each breath compute a
different inner-product geometry.

Env vars:
  V109PI_TASK=1                  — enable v109pi forward path
  V109PI_K_MAX=8                 — number of breaths
  V109PI_N_DIGITS=5
  V109PI_WAIST_DIM=512
  V109PI_ALTERNATION=1
  V109PI_PHASE_SCALE=1.0         — overall phase scale (1.0 = full π·k/K_max)
"""
from __future__ import annotations

import math
import os
from typing import Any

import numpy as np
from tinygrad import Tensor, dtypes
from tinygrad.engine.jit import TinyJit

from mycelium.config import Config
from mycelium.factor_graph_v100 import (
    embed_factor_graph_v100_aligned,
    V100_N_HEADS,
)
from mycelium.factor_graph_v107 import V107_N_HEADS
from mycelium.factor_graph_v108 import (
    V108_N_MAX, V108_F_MAX, V108_N_HEADS, V108_N_DIGITS, V108_K_MAX,
    V108_CODEBOOK_N, V108_IB_CENTROIDS,
    V108_CALIB_WEIGHT, V108_FACTOR_AUX_WEIGHT,
    bins_to_digits_msd,
)
from mycelium.factor_graph_v109 import (
    V109_K_MAX, V109_N_DIGITS, V109_WAIST_DIM, V109_ALTERNATION,
    V109_VAR_LOSS_WEIGHT, V109_N_MAX, V109_F_MAX, V109_N_HEADS,
    V109_HARD_BREATH_LEVEL,
    _apply_waist_v109,
    attach_fg_params_v109, fg_v109_parameters, fg_v109_state_dict,
)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

V109PI_TASK              = int(os.environ.get("V109PI_TASK", "0")) > 0
V109PI_K_MAX             = int(os.environ.get("V109PI_K_MAX",             "8"))
V109PI_N_DIGITS          = int(os.environ.get("V109PI_N_DIGITS",          "5"))
V109PI_WAIST_DIM         = int(os.environ.get("V109PI_WAIST_DIM",         "512"))
V109PI_ALTERNATION       = int(os.environ.get("V109PI_ALTERNATION",       "1")) > 0
V109PI_HARD_BREATH_LEVEL = int(os.environ.get("V109PI_HARD_BREATH_LEVEL", "0")) > 0
V109PI_VAR_LOSS_WEIGHT   = float(os.environ.get("V109PI_VAR_LOSS_WEIGHT", "1.0"))
V109PI_N_MAX             = int(os.environ.get("V109PI_N_MAX",             str(V109_N_MAX)))
V109PI_F_MAX             = int(os.environ.get("V109PI_F_MAX",             str(V109_F_MAX)))
V109PI_T_MAX             = V109PI_N_MAX + V109PI_F_MAX
V109PI_CODEBOOK_N        = int(os.environ.get("V109PI_CODEBOOK_N",        str(V108_CODEBOOK_N)))
V109PI_IB_CENTROIDS      = os.environ.get("V109PI_IB_CENTROIDS",          V108_IB_CENTROIDS)
V109PI_CALIB_WEIGHT      = float(os.environ.get("V109PI_CALIB_WEIGHT",    str(V108_CALIB_WEIGHT)))
V109PI_FACTOR_AUX_WEIGHT = float(os.environ.get("V109PI_FACTOR_AUX_WEIGHT", str(V108_FACTOR_AUX_WEIGHT)))
V109PI_N_HEADS           = V109_N_HEADS
V109PI_PHASE_SCALE       = float(os.environ.get("V109PI_PHASE_SCALE",     "1.0"))


# ---------------------------------------------------------------------------
# Per-breath Q rotation (Option A π-cycled)
# ---------------------------------------------------------------------------

def _rotate_q_pi(q: Tensor, cos_k: float, sin_k: float) -> Tensor:
    """Rotate Q pairwise across head_dim by a single (cos, sin).

    Input q: (B, n_heads, S, head_dim)
    Output:  same shape, with pairs (2i, 2i+1) rotated by angle k·π/K.

    Uniform rotation across sequence positions (no position-dependence).
    The rotation is applied as:
      q_rot[..., 2i]   = cos · q[..., 2i]   - sin · q[..., 2i+1]
      q_rot[..., 2i+1] = sin · q[..., 2i]   + cos · q[..., 2i+1]
    """
    H_dim = q.shape[-1]
    assert H_dim % 2 == 0, f"head_dim must be even, got {H_dim}"
    n_pairs = H_dim // 2

    # Reshape into pairs: (..., n_pairs, 2)
    q_pairs = q.reshape(*q.shape[:-1], n_pairs, 2)
    q_even = q_pairs[..., 0]                       # (..., n_pairs)
    q_odd  = q_pairs[..., 1]                       # (..., n_pairs)

    cos_t = Tensor([cos_k], dtype=q.dtype).reshape(*([1] * (q.ndim - 1)))
    sin_t = Tensor([sin_k], dtype=q.dtype).reshape(*([1] * (q.ndim - 1)))

    new_even = cos_t * q_even - sin_t * q_odd
    new_odd  = sin_t * q_even + cos_t * q_odd

    # Recombine: stack pairs and flatten
    out_pairs = new_even.unsqueeze(-1).cat(new_odd.unsqueeze(-1), dim=-1)
    return out_pairs.reshape(*q.shape)


def fg_layer_forward_v109pi(
    layer: Any,
    x: Tensor,
    attn_bias: Tensor,
    q_rot_cos: float,
    q_rot_sin: float,
) -> Tensor:
    """Modified v100-style layer forward: rotate Q pairwise by (cos, sin) before Q@K."""
    cfg = layer.cfg
    B, S, H = x.shape
    n_heads  = cfg.n_heads
    head_dim = cfg.head_dim

    from mycelium.breathing import _layernorm
    attn_in = _layernorm(x, layer.shared.in_ln_g, layer.shared.in_ln_b, cfg.layer_norm_eps)
    mlp_in  = _layernorm(x, layer.shared.post_ln_g, layer.shared.post_ln_b, cfg.layer_norm_eps)
    attn_in_dt = attn_in.cast(x.dtype) if attn_in.dtype != x.dtype else attn_in
    mlp_in_dt  = mlp_in.cast(x.dtype) if mlp_in.dtype != x.dtype else mlp_in

    q = (attn_in_dt @ layer.wq + layer.bq).reshape(B, S, n_heads, head_dim).transpose(1, 2)
    k = (attn_in_dt @ layer.wk + layer.bk).reshape(B, S, n_heads, head_dim).transpose(1, 2)
    v = (attn_in_dt @ layer.shared.wv + layer.shared.bv).reshape(B, S, n_heads, head_dim).transpose(1, 2)

    # === v109pi: per-breath Q rotation ===
    # At k=0: cos=1, sin=0 → identity (byte-safe warm-start from v109)
    q = _rotate_q_pi(q, q_rot_cos, q_rot_sin)

    scale  = 1.0 / math.sqrt(head_dim)
    scores = q @ k.transpose(-2, -1) * scale

    scores = scores + attn_bias.cast(scores.dtype)
    attn = scores.clip(-1e4, 1e4).softmax(-1)
    ctx  = (attn @ v).transpose(1, 2).reshape(B, S, H)
    attn_out = ctx @ layer.shared.wo + layer.shared.bo

    ff = (mlp_in_dt @ layer.w_in + layer.b_in).gelu()
    ffn_out = ff @ layer.shared.w_out + layer.shared.b_out

    return x + attn_out + ffn_out


# ---------------------------------------------------------------------------
# Forward pass (v109 substrate + per-breath Q rotation)
# ---------------------------------------------------------------------------

def fg_breathing_forward_v109pi(
    model: Any,
    domain_init: Tensor,
    node_kinds: Tensor,
    staging_mask: Tensor,
    head_op_mask: Tensor,
    K: int,
    n_max: int = V109PI_N_MAX,
    f_max: int = V109PI_F_MAX,
    n_digits: int = V109PI_N_DIGITS,
    alternation: bool = V109PI_ALTERNATION,
    phase_scale: float = V109PI_PHASE_SCALE,
) -> tuple[list[Tensor], list[Tensor], list[Tensor], list[Tensor]]:
    """Run K breaths with per-breath Q rotation on top of v109 substrate."""
    assert hasattr(model, "fg_v107_domain_codebook"), "missing v107 backbone"
    assert hasattr(model, "fg_v108_tree_codebook"), "missing v108 tree codebook"
    assert hasattr(model, "fg_v109_W_compress"), "missing v109 waist"

    domain_codebook  = model.fg_v107_domain_codebook
    var_state_embed  = model.fg_v107_var_state_embed
    var_pos_embed    = model.fg_v107_var_pos_embed
    factor_pos_embed = model.fg_v107_factor_pos_embed
    node_kind_embed  = model.fg_v107_node_kind_embed
    breath_embed     = model.fg_v107_breath_embed
    delta_gate       = model.fg_v107_delta_gate
    calib_head_w     = model.fg_v107_calib_head_w
    calib_head_b     = model.fg_v107_calib_head_b
    semantic_codebook = model.fg_v107_semantic_codebook
    delta_gate_quant  = model.fg_v107_delta_gate_quant
    temperature       = model.fg_v107_temperature
    tree_codebook = model.fg_v108_tree_codebook

    W_compress = model.fg_v109_W_compress
    b_compress = model.fg_v109_b_compress
    W_expand   = model.fg_v109_W_expand
    b_expand   = model.fg_v109_b_expand

    H = int(tree_codebook.shape[-1])
    B = int(domain_init.shape[0])
    T = n_max + f_max

    x = embed_factor_graph_v100_aligned(
        domain_init, node_kinds,
        var_pos_embed, factor_pos_embed, node_kind_embed,
        domain_codebook, var_state_embed,
    )
    x = x.cast(dtypes.half) if x.dtype != dtypes.half else x

    layers = list(model.block.layers)
    K_max  = int(breath_embed.shape[0])
    assert K <= K_max, f"K={K} > K_max={K_max}"

    # Precompute per-breath rotation angles for Q
    # phase_k = phase_scale * k * π / K_max  (so k=0 → 0, k=K_max-1 → ~π)
    # cos_k, sin_k are constants per breath (not Tensors).
    breath_phases = [phase_scale * k * math.pi / float(K_max) for k in range(K_max)]
    breath_cos = [math.cos(p) for p in breath_phases]
    breath_sin = [math.sin(p) for p in breath_phases]

    from mycelium.breathing import _layernorm

    tree_logits_history   = []
    var_logits_history    = []
    factor_logits_history = []
    calib_history         = []

    tree_cb_flat = tree_codebook.reshape(n_digits * 10, H)

    for k in range(K):
        be_k  = breath_embed[k].reshape(1, 1, -1).cast(x.dtype)
        x_in  = x + be_k
        x_pre = x

        stk      = staging_mask[:, k, :, :]
        stk_h    = stk.reshape(B, 1, T, T).expand(B, V109PI_N_HEADS, T, T)
        combined = stk_h.cast(x.dtype) + head_op_mask.cast(x.dtype)

        # Pythia layers with per-breath Q rotation
        cos_k = breath_cos[k]
        sin_k = breath_sin[k]
        h = x_in
        for layer in layers[:4]:
            h = fg_layer_forward_v109pi(layer, h, combined, cos_k, sin_k)

        # IB codebook (same as v109)
        cb  = semantic_codebook.cast(h.dtype)
        tmp = temperature.cast(h.dtype)
        scores   = h @ cb.T / tmp.reshape(1, 1, 1)
        weights  = scores.clip(-1e4, 1e4).softmax(-1)
        recon    = weights @ cb
        quantize = recon - h
        gate_quant_k = delta_gate_quant[k].cast(h.dtype).reshape(1, 1, 1)
        h_quant = h + gate_quant_k * quantize

        # Alternation: waist on even breaths
        is_commit_breath = (k % 2 == 0)
        if not alternation or is_commit_breath:
            h_quant = _apply_waist_v109(
                h_quant, W_compress, b_compress, W_expand, b_expand,
            )

        gate_k = delta_gate[k].cast(h_quant.dtype).reshape(1, 1, 1)
        delta  = h_quant - x_pre
        x      = x_pre + gate_k * delta

        # Readout (same as v109)
        x_ln = _layernorm(x, model.ln_f_g, model.ln_f_b,
                          model.cfg.layer_norm_eps).cast(dtypes.float)
        var_x = x_ln[:, :n_max, :]

        tree_logits_flat = var_x @ tree_cb_flat.T.cast(dtypes.float)
        tree_logits_k    = tree_logits_flat.reshape(B, n_max, n_digits, 10)
        tree_logits_history.append(tree_logits_k)

        var_logits_k = var_x @ domain_codebook.T.cast(dtypes.float)
        var_logits_history.append(var_logits_k)

        fac_x = x_ln[:, n_max:n_max + f_max, :]
        fac_logits_k = fac_x @ domain_codebook.T.cast(dtypes.float)
        factor_logits_history.append(fac_logits_k)

        pool        = x_ln.mean(axis=1)
        calib_logit = pool @ calib_head_w.cast(dtypes.float) + calib_head_b.cast(dtypes.float)
        calib_history.append(calib_logit.reshape(-1).sigmoid())

    return tree_logits_history, var_logits_history, factor_logits_history, calib_history


# ---------------------------------------------------------------------------
# Parameter attachment (same as v109 — no new params for π-cycling)
# ---------------------------------------------------------------------------

def attach_fg_params_v109pi(
    model: Any,
    hidden: int,
    n_max: int = V109PI_N_MAX,
    f_max: int = V109PI_F_MAX,
    k_max: int | None = None,
    n_digits: int = V109PI_N_DIGITS,
    n_code: int = V109PI_CODEBOOK_N,
    ib_centroids_path: str = V109PI_IB_CENTROIDS,
    waist_dim: int = V109PI_WAIST_DIM,
) -> None:
    """No new trainable params — π-cycling uses constant per-breath rotations."""
    if k_max is None:
        k_max = V109PI_K_MAX
    attach_fg_params_v109(
        model, hidden=hidden,
        n_max=n_max, f_max=f_max, k_max=k_max,
        n_digits=n_digits, n_code=n_code, ib_centroids_path=ib_centroids_path,
        waist_dim=waist_dim,
    )
    phase_max = V109PI_PHASE_SCALE * (k_max - 1) * math.pi / float(k_max)
    print(
        f"[v109pi] π-cycled Q rotation enabled: phase_scale={V109PI_PHASE_SCALE} "
        f"→ phases [0, π·{V109PI_PHASE_SCALE:.2f}·(K-1)/K = {phase_max:.3f}] across {k_max} breaths",
        flush=True,
    )


fg_v109pi_parameters = fg_v109_parameters
fg_v109pi_state_dict = fg_v109_state_dict


# ---------------------------------------------------------------------------
# JIT training step
# ---------------------------------------------------------------------------

_JIT_V109PI_CACHE: dict = {}


def _compile_jit_fg_step_v109pi(
    model: Any,
    opt: Any,
    K: int,
    B: int,
    factor_aux_weight: float = V109PI_FACTOR_AUX_WEIGHT,
    calib_weight: float = V109PI_CALIB_WEIGHT,
    var_loss_weight: float = V109PI_VAR_LOSS_WEIGHT,
    hard_breath_level: bool = V109PI_HARD_BREATH_LEVEL,
    alternation: bool = V109PI_ALTERNATION,
    phase_scale: float = V109PI_PHASE_SCALE,
    n_max: int = V109PI_N_MAX,
    f_max: int = V109PI_F_MAX,
    n_digits: int = V109PI_N_DIGITS,
    grad_clip: float = 1.0,
):
    key = ("v109pi", id(model), id(opt), int(K), int(B),
           float(factor_aux_weight), float(calib_weight), float(var_loss_weight),
           bool(hard_breath_level), bool(alternation), float(phase_scale),
           int(n_max), int(f_max), int(n_digits), float(grad_clip))
    if key in _JIT_V109PI_CACHE:
        return _JIT_V109PI_CACHE[key]

    fw, aw, vw, gc = float(factor_aux_weight), float(calib_weight), \
                     float(var_loss_weight), float(grad_clip)
    params = opt.params
    print(f"[JIT] compile v109pi fg step: K={K} B={B} n_digits={n_digits} "
          f"alternation={alternation} phase_scale={phase_scale} vw={vw}...",
          flush=True)

    @TinyJit
    def _step(
        domain_init, node_kinds, staging_mask, head_op_mask,
        gold_digits, gold_bins, observed_mask, factor_gold_bin, factor_valid,
    ):
        opt.zero_grad()

        tree_lh, var_lh, fac_lh, calib_h = fg_breathing_forward_v109pi(
            model, domain_init, node_kinds, staging_mask, head_op_mask,
            K=K, n_max=n_max, f_max=f_max, n_digits=n_digits,
            alternation=alternation, phase_scale=phase_scale,
        )

        unobs_float = (1 - observed_mask.cast(dtypes.float)).reshape(B * n_max)
        n_unobs_sum = unobs_float.sum() + 1e-8
        gd_flat = gold_digits.cast(dtypes.int).reshape(B * n_max, n_digits)

        var_loss_sum   = Tensor.zeros((), dtype=dtypes.float).contiguous()
        var_weight_sum = 0.0
        per_breath_ce_t: list[Tensor] = []

        for k, tree_logits_k in enumerate(tree_lh):
            weight_k = 1.0 + float(k) / float(max(K - 1, 1))
            tl_flat  = tree_logits_k.reshape(B * n_max, n_digits, 10)
            if hard_breath_level:
                levels_to_use = [k] if k < n_digits else list(range(n_digits))
            else:
                levels_to_use = list(range(n_digits))
            ce_breath_sum = Tensor.zeros((), dtype=dtypes.float).contiguous()
            for level in levels_to_use:
                level_logits = tl_flat[:, level, :]
                level_gold   = gd_flat[:, level]
                log_probs    = level_logits.log_softmax(axis=-1)
                gold_oh      = level_gold.one_hot(10).cast(log_probs.dtype)
                nll          = -(log_probs * gold_oh).sum(axis=-1)
                masked_nll   = nll * unobs_float.cast(nll.dtype)
                ce_level     = masked_nll.sum() / n_unobs_sum
                ce_breath_sum = ce_breath_sum + ce_level
            ce_k = ce_breath_sum / float(len(levels_to_use))
            per_breath_ce_t.append(ce_k)
            var_loss_sum   = var_loss_sum + ce_k * weight_k
            var_weight_sum += weight_k
        var_loss = var_loss_sum / float(var_weight_sum)

        # Factor aux (200-way) — same as v109
        n_valid_factors = factor_valid.cast(dtypes.float).sum() + 1e-8
        gold_fac_flat   = factor_gold_bin.cast(dtypes.int).reshape(B * f_max)
        gold_fac_oh     = gold_fac_flat.one_hot(200).cast(dtypes.float)
        valid_flat      = factor_valid.cast(dtypes.float).reshape(B * f_max)
        factor_aux_sum   = Tensor.zeros((), dtype=dtypes.float).contiguous()
        factor_aux_w_sum = 0.0
        for k_aux, fac_logits_k in enumerate(fac_lh):
            w_k_aux = 1.0 + float(k_aux) / float(max(K - 1, 1))
            fac_flat  = fac_logits_k.reshape(B * f_max, 200)
            fac_lp    = fac_flat.log_softmax(axis=-1)
            fac_nll   = -(fac_lp * gold_fac_oh).sum(axis=-1)
            fac_masked= fac_nll * valid_flat
            fac_ce_k  = fac_masked.sum() / n_valid_factors
            factor_aux_sum   = factor_aux_sum + fac_ce_k * w_k_aux
            factor_aux_w_sum += w_k_aux
        factor_aux_loss = factor_aux_sum / float(factor_aux_w_sum)

        # Calibration
        final_tree = tree_lh[-1]
        pred_digits_final = final_tree.argmax(axis=-1).detach()
        eq_per_pos = (pred_digits_final == gold_digits.cast(dtypes.int)).cast(dtypes.float)
        eq         = eq_per_pos.prod(axis=-1)
        unobs_2d   = (1 - observed_mask.cast(dtypes.float))
        eq_unobs   = eq * unobs_2d
        n_unobs_per = unobs_2d.sum(axis=-1) + 1e-8
        correct     = eq_unobs.sum(axis=-1) / n_unobs_per

        calib_loss_sum = Tensor.zeros((), dtype=dtypes.float).contiguous()
        for kc, calib_k in enumerate(calib_h):
            prog       = float(kc) / float(max(K - 1, 1))
            target_k   = 0.5 + (correct - 0.5) * prog
            calib_loss_sum = calib_loss_sum + ((calib_k - target_k) ** 2).mean()
        calib_loss = calib_loss_sum / float(K)

        cell_acc  = (eq_unobs.sum() / (unobs_2d.sum() + 1e-8)).detach()
        query_acc = correct.mean().detach()

        total_ce = vw * var_loss + fw * factor_aux_loss + aw * calib_loss
        total_ce.backward()

        healthy = total_ce.isfinite().cast(dtypes.float)
        for p in params:
            if p.grad is not None:
                p.grad = p.grad * healthy.cast(p.grad.dtype)

        if gc > 0:
            sq_sum = Tensor.zeros((), dtype=dtypes.float).contiguous()
            for p in params:
                if p.grad is not None:
                    sq_sum = sq_sum + p.grad.cast(dtypes.float).square().sum()
            grad_norm = (sq_sum + 1e-12).sqrt()
            clip_coef = (Tensor(gc, dtype=dtypes.float) / (grad_norm + 1e-6)).minimum(
                Tensor(1.0, dtype=dtypes.float)
            )
            for p in params:
                if p.grad is not None:
                    p.grad = p.grad * clip_coef.cast(p.grad.dtype)

        opt.step()

        return (
            total_ce.realize(), healthy.realize(),
            var_loss.realize(), factor_aux_loss.realize(),
            calib_loss.realize(),
            cell_acc.realize(), query_acc.realize(),
            *(ce.realize() for ce in per_breath_ce_t),
        )

    _JIT_V109PI_CACHE[key] = _step
    print(f"[JIT] v109pi fg step ready (cache={len(_JIT_V109PI_CACHE)})", flush=True)
    return _step


def _compile_jit_fg_eval_v109pi(
    model: Any,
    K: int,
    B: int,
    n_max: int = V109PI_N_MAX,
    f_max: int = V109PI_F_MAX,
    n_digits: int = V109PI_N_DIGITS,
    alternation: bool = V109PI_ALTERNATION,
    phase_scale: float = V109PI_PHASE_SCALE,
):
    key = ("eval_v109pi", id(model), int(K), int(B), int(n_max), int(f_max),
           int(n_digits), bool(alternation), float(phase_scale))
    if key in _JIT_V109PI_CACHE:
        return _JIT_V109PI_CACHE[key]

    print(f"[JIT] compile v109pi fg eval: K={K} B={B} phase_scale={phase_scale}...",
          flush=True)

    @TinyJit
    def _eval(
        domain_init, node_kinds, staging_mask, head_op_mask,
        gold_digits, observed_mask,
    ):
        tree_lh, _, _, _ = fg_breathing_forward_v109pi(
            model, domain_init, node_kinds, staging_mask, head_op_mask,
            K=K, n_max=n_max, f_max=f_max, n_digits=n_digits,
            alternation=alternation, phase_scale=phase_scale,
        )
        final_tree = tree_lh[-1]
        pred_digits = final_tree.argmax(axis=-1)
        eq_per_pos = (pred_digits == gold_digits.cast(dtypes.int)).cast(dtypes.float)
        eq = eq_per_pos.prod(axis=-1)
        unobs = (1 - observed_mask.cast(dtypes.float))
        cell_acc = (eq * unobs).sum() / (unobs.sum() + 1e-8)
        return pred_digits.realize(), cell_acc.realize()

    _JIT_V109PI_CACHE[key] = _eval
    print(f"[JIT] v109pi eval ready (cache={len(_JIT_V109PI_CACHE)})", flush=True)
    return _eval
