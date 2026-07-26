"""v110-acc: v109pi + write-once ACCUMULATE notebook.

Architecture (warm-start from v109pi_cont9_step500):
    v109pi base (waist + alternation + per-breath Q rotation)
  + ACCUMULATE notebook:
      - K_max slots of dim H, one per breath
      - At end of breath k, write pool(h_post_waist) → slot k (write-once)
      - At start of breath k+1, causal cross-attn reads slots 0..k → adds
        to residual via zero-init W_o
      - Breath 0 reads empty notebook → no contribution → byte-safe at step 0

This is the v61-style DAG notebook concept restored on top of v109pi.

Distinct from REPLACE notebook: each breath writes ITS OWN slot, never
overwrites. Later breaths see ALL prior commitments. The residual stream
is the carrier; the notebook is a write-once memory.

Byte-safe at step 0: W_o is zero-init, so cross-attn output is zero,
so x is unchanged, so forward is byte-identical to v109pi.
"""
from __future__ import annotations

import math
import os
from typing import Any

import numpy as np
from tinygrad import Tensor, dtypes
from tinygrad.engine.jit import TinyJit
from tinygrad.helpers import getenv

from mycelium.factor_graph_v107 import embed_factor_graph_v100_aligned
from mycelium.factor_graph_v108 import (
    V108_K_MAX, V108_N_MAX, V108_F_MAX, V108_N_DIGITS, V108_N_HEADS,
    V108_CODEBOOK_N, V108_IB_CENTROIDS, V108_CALIB_WEIGHT,
    V108_FACTOR_AUX_WEIGHT,
    bins_to_digits_msd, values_to_digits_msd,
)
from mycelium.factor_graph_v109 import (
    V109_K_MAX, V109_N_MAX, V109_F_MAX, V109_N_HEADS, V109_N_DIGITS,
    V109_WAIST_DIM, V109_ALTERNATION,
    V109_CODEBOOK_N, V109_IB_CENTROIDS,
    attach_fg_params_v109, fg_v109_parameters, fg_v109_state_dict,
    _apply_waist_v109,
)
from mycelium.factor_graph_v109pi import (
    V109PI_K_MAX, V109PI_N_MAX, V109PI_F_MAX, V109PI_N_HEADS, V109PI_N_DIGITS,
    V109PI_WAIST_DIM, V109PI_ALTERNATION, V109PI_PHASE_SCALE,
    V109PI_CODEBOOK_N, V109PI_IB_CENTROIDS,
    attach_fg_params_v109pi,
    fg_layer_forward_v109pi,
)


# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

V110_ACC_TASK              = int(os.environ.get("V110_ACC_TASK", "0")) > 0
V110_ACC_K_MAX             = int(os.environ.get("V110_ACC_K_MAX", str(V109PI_K_MAX)))
V110_ACC_N_DIGITS          = int(os.environ.get("V110_ACC_N_DIGITS", str(V109PI_N_DIGITS)))
V110_ACC_WAIST_DIM         = int(os.environ.get("V110_ACC_WAIST_DIM", str(V109PI_WAIST_DIM)))
V110_ACC_ALTERNATION       = int(os.environ.get("V110_ACC_ALTERNATION", "1")) > 0
V110_ACC_HARD_BREATH_LEVEL = int(os.environ.get("V110_ACC_HARD_BREATH_LEVEL", "0")) > 0
V110_ACC_VAR_LOSS_WEIGHT   = float(os.environ.get("V110_ACC_VAR_LOSS_WEIGHT", "1.0"))
V110_ACC_N_MAX             = int(os.environ.get("V110_ACC_N_MAX", str(V109PI_N_MAX)))
V110_ACC_F_MAX             = int(os.environ.get("V110_ACC_F_MAX", str(V109PI_F_MAX)))
V110_ACC_T_MAX             = V110_ACC_N_MAX + V110_ACC_F_MAX
V110_ACC_CODEBOOK_N        = int(os.environ.get("V110_ACC_CODEBOOK_N", str(V109PI_CODEBOOK_N)))
V110_ACC_IB_CENTROIDS      = os.environ.get("V110_ACC_IB_CENTROIDS", V109PI_IB_CENTROIDS)
V110_ACC_CALIB_WEIGHT      = float(os.environ.get("V110_ACC_CALIB_WEIGHT", str(V108_CALIB_WEIGHT)))
V110_ACC_FACTOR_AUX_WEIGHT = float(os.environ.get("V110_ACC_FACTOR_AUX_WEIGHT", str(V108_FACTOR_AUX_WEIGHT)))
V110_ACC_N_HEADS           = V109PI_N_HEADS
V110_ACC_PHASE_SCALE       = float(os.environ.get("V110_ACC_PHASE_SCALE", str(V109PI_PHASE_SCALE)))


# ---------------------------------------------------------------------------
# ACCUMULATE notebook cross-attention
# ---------------------------------------------------------------------------

def _acc_notebook_read(
    x: Tensor,               # (B, T, H) current residual
    slots: list[Tensor],     # list of (B, H) committed slot vectors, length k
    acc_W_q: Tensor,         # (H, H)
    acc_W_k: Tensor,         # (H, H)
    acc_W_v: Tensor,         # (H, H)
    acc_W_o: Tensor,         # (H, H) zero-init
    acc_b_o: Tensor,         # (H,)   zero-init
) -> Tensor:
    """Cross-attend current residual to prior notebook commitments.

    Returns delta_x (B, T, H) to be ADDED to x. At init (W_o=0), delta_x=0.
    """
    if len(slots) == 0:
        # Breath 0: no prior commitments. Return zero contribution.
        return x * 0.0  # preserves shape and dtype, contributes nothing

    H = int(x.shape[-1])
    B = int(x.shape[0])
    T = int(x.shape[1])
    k = len(slots)

    # Stack slots → (B, k, H)
    nb = Tensor.stack(*slots, dim=1).cast(x.dtype)  # (B, k, H)

    # Single-head cross-attn (no positional embeddings on the slot axis —
    # the slot index IS the breath index, but we let the model learn
    # whatever positional structure it needs through the slot contents).
    Wq = acc_W_q.cast(x.dtype)
    Wk = acc_W_k.cast(x.dtype)
    Wv = acc_W_v.cast(x.dtype)
    Wo = acc_W_o.cast(x.dtype)
    bo = acc_b_o.reshape(1, 1, -1).cast(x.dtype)

    q = x  @ Wq                      # (B, T, H)
    K = nb @ Wk                      # (B, k, H)
    V = nb @ Wv                      # (B, k, H)

    scale = 1.0 / math.sqrt(H)
    scores  = (q @ K.transpose(-1, -2)) * scale          # (B, T, k)
    weights = scores.clip(-1e4, 1e4).softmax(-1)         # (B, T, k)
    ctx     = weights @ V                                # (B, T, H)
    delta   = ctx @ Wo + bo                              # (B, T, H) zero-init
    return delta


def _acc_notebook_write(
    h: Tensor,              # (B, T, H) post-waist residual
    acc_W_write: Tensor,    # (H, H) projection from pooled hidden to slot dim
    acc_b_write: Tensor,    # (H,)
) -> Tensor:
    """Pool post-waist residual to a single commit vector for this breath.

    Returns (B, H) slot vector.
    """
    pool = h.mean(axis=1)                            # (B, H)
    Ww = acc_W_write.cast(h.dtype)
    bw = acc_b_write.reshape(1, -1).cast(h.dtype)
    slot = pool @ Ww + bw                            # (B, H)
    return slot


# ---------------------------------------------------------------------------
# Forward pass — v109pi + accumulate notebook
# ---------------------------------------------------------------------------

def fg_breathing_forward_v110_acc(
    model: Any,
    domain_init: Tensor,
    node_kinds: Tensor,
    staging_mask: Tensor,
    head_op_mask: Tensor,
    K: int,
    n_max: int = V110_ACC_N_MAX,
    f_max: int = V110_ACC_F_MAX,
    n_digits: int = V110_ACC_N_DIGITS,
    alternation: bool = V110_ACC_ALTERNATION,
    phase_scale: float = V110_ACC_PHASE_SCALE,
):
    """v109pi forward + write-once accumulate notebook."""
    assert hasattr(model, "fg_v107_domain_codebook"), "missing v107 backbone"
    assert hasattr(model, "fg_v108_tree_codebook"), "missing v108 tree codebook"
    assert hasattr(model, "fg_v109_W_compress"), "missing v109 waist"
    assert hasattr(model, "fg_v110_acc_W_q"), "missing v110-acc notebook params"

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

    acc_W_q     = model.fg_v110_acc_W_q
    acc_W_k     = model.fg_v110_acc_W_k
    acc_W_v     = model.fg_v110_acc_W_v
    acc_W_o     = model.fg_v110_acc_W_o
    acc_b_o     = model.fg_v110_acc_b_o
    acc_W_write = model.fg_v110_acc_W_write
    acc_b_write = model.fg_v110_acc_b_write

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

    breath_phases = [phase_scale * k * math.pi / float(K_max) for k in range(K_max)]
    breath_cos = [math.cos(p) for p in breath_phases]
    breath_sin = [math.sin(p) for p in breath_phases]

    from mycelium.breathing import _layernorm

    tree_logits_history   = []
    var_logits_history    = []
    factor_logits_history = []
    calib_history         = []

    tree_cb_flat = tree_codebook.reshape(n_digits * 10, H)

    # ACCUMULATE notebook: list of (B, H) tensors, one per breath
    notebook_slots: list[Tensor] = []

    for k in range(K):
        # Read from prior notebook slots (causal: only slots 0..k-1)
        x_acc_delta = _acc_notebook_read(
            x, notebook_slots,
            acc_W_q, acc_W_k, acc_W_v, acc_W_o, acc_b_o,
        )
        x = x + x_acc_delta

        be_k  = breath_embed[k].reshape(1, 1, -1).cast(x.dtype)
        x_in  = x + be_k
        x_pre = x

        stk      = staging_mask[:, k, :, :]
        stk_h    = stk.reshape(B, 1, T, T).expand(B, V110_ACC_N_HEADS, T, T)
        combined = stk_h.cast(x.dtype) + head_op_mask.cast(x.dtype)

        cos_k = breath_cos[k]
        sin_k = breath_sin[k]
        h = x_in
        for layer in layers[:4]:
            h = fg_layer_forward_v109pi(layer, h, combined, cos_k, sin_k)

        # IB codebook
        cb  = semantic_codebook.cast(h.dtype)
        tmp = temperature.cast(h.dtype)
        scores   = h @ cb.T / tmp.reshape(1, 1, 1)
        weights  = scores.clip(-1e4, 1e4).softmax(-1)
        recon    = weights @ cb
        quantize = recon - h
        gate_quant_k = delta_gate_quant[k].cast(h.dtype).reshape(1, 1, 1)
        h_quant = h + gate_quant_k * quantize

        # Alternation
        is_commit_breath = (k % 2 == 0)
        if not alternation or is_commit_breath:
            h_quant = _apply_waist_v109(
                h_quant, W_compress, b_compress, W_expand, b_expand,
            )

        # delta_gate blend
        gate_k = delta_gate[k].cast(h_quant.dtype).reshape(1, 1, 1)
        delta  = h_quant - x_pre
        x      = x_pre + gate_k * delta

        # WRITE to notebook: pool post-waist residual to a single slot vector
        slot_k = _acc_notebook_write(x, acc_W_write, acc_b_write)  # (B, H)
        notebook_slots.append(slot_k)

        # Readout
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
# Parameter attachment
# ---------------------------------------------------------------------------

def attach_fg_params_v110_acc(
    model: Any,
    hidden: int,
    n_max: int = V110_ACC_N_MAX,
    f_max: int = V110_ACC_F_MAX,
    k_max: int | None = None,
    n_digits: int = V110_ACC_N_DIGITS,
    n_code: int = V110_ACC_CODEBOOK_N,
    ib_centroids_path: str = V110_ACC_IB_CENTROIDS,
    waist_dim: int = V110_ACC_WAIST_DIM,
) -> None:
    """Attach v109pi params + accumulate notebook params.

    Notebook params:
      acc_W_q / acc_W_k / acc_W_v : (H, H) small Gaussian init
      acc_W_o                     : (H, H) ZERO-init (byte-safe warm start)
      acc_b_o                     : (H,)   ZERO-init
      acc_W_write                 : (H, H) small Gaussian init
      acc_b_write                 : (H,)   ZERO-init
    """
    if k_max is None:
        k_max = V110_ACC_K_MAX

    # v109pi params (which calls attach_fg_params_v109)
    attach_fg_params_v109pi(
        model, hidden=hidden,
        n_max=n_max, f_max=f_max, k_max=k_max,
        n_digits=n_digits, n_code=n_code, ib_centroids_path=ib_centroids_path,
        waist_dim=waist_dim,
    )

    H = hidden
    rng = np.random.RandomState(110001)
    scale_init = 0.02

    Wq = (rng.randn(H, H).astype(np.float32) * scale_init)
    Wk = (rng.randn(H, H).astype(np.float32) * scale_init)
    Wv = (rng.randn(H, H).astype(np.float32) * scale_init)
    Wo = np.zeros((H, H), dtype=np.float32)   # zero-init for byte-safe warm-start
    bo = np.zeros((H,), dtype=np.float32)
    Ww = (rng.randn(H, H).astype(np.float32) * scale_init)
    bw = np.zeros((H,), dtype=np.float32)

    model.fg_v110_acc_W_q     = Tensor(Wq, dtype=dtypes.float).contiguous()
    model.fg_v110_acc_W_k     = Tensor(Wk, dtype=dtypes.float).contiguous()
    model.fg_v110_acc_W_v     = Tensor(Wv, dtype=dtypes.float).contiguous()
    model.fg_v110_acc_W_o     = Tensor(Wo, dtype=dtypes.float).contiguous()
    model.fg_v110_acc_b_o     = Tensor(bo, dtype=dtypes.float).contiguous()
    model.fg_v110_acc_W_write = Tensor(Ww, dtype=dtypes.float).contiguous()
    model.fg_v110_acc_b_write = Tensor(bw, dtype=dtypes.float).contiguous()

    n_params = 5 * H * H + 2 * H
    print(
        f"[v110-acc] accumulate notebook attached: {n_params:,} new params "
        f"(5 (H,H) + 2 (H,) at H={H})",
        flush=True,
    )


def fg_v110_acc_parameters(model: Any) -> list[Tensor]:
    """All trainable params: v109pi + accumulate notebook."""
    base = fg_v109_parameters(model)
    extras = [
        model.fg_v110_acc_W_q,
        model.fg_v110_acc_W_k,
        model.fg_v110_acc_W_v,
        model.fg_v110_acc_W_o,
        model.fg_v110_acc_b_o,
        model.fg_v110_acc_W_write,
        model.fg_v110_acc_b_write,
    ]
    return base + extras


def fg_v110_acc_state_dict(model: Any) -> dict:
    sd = fg_v109_state_dict(model)
    sd.update({
        "fg_v110_acc_W_q":     model.fg_v110_acc_W_q,
        "fg_v110_acc_W_k":     model.fg_v110_acc_W_k,
        "fg_v110_acc_W_v":     model.fg_v110_acc_W_v,
        "fg_v110_acc_W_o":     model.fg_v110_acc_W_o,
        "fg_v110_acc_b_o":     model.fg_v110_acc_b_o,
        "fg_v110_acc_W_write": model.fg_v110_acc_W_write,
        "fg_v110_acc_b_write": model.fg_v110_acc_b_write,
    })
    return sd


# ---------------------------------------------------------------------------
# JIT step/eval — wrap v109pi training step machinery
# ---------------------------------------------------------------------------

_JIT_V110_ACC_CACHE: dict = {}


def _compile_jit_fg_step_v110_acc(
    model: Any,
    opt: Any,
    K: int,
    B: int,
    factor_aux_weight: float = V110_ACC_FACTOR_AUX_WEIGHT,
    calib_weight: float = V110_ACC_CALIB_WEIGHT,
    var_loss_weight: float = V110_ACC_VAR_LOSS_WEIGHT,
    hard_breath_level: bool = V110_ACC_HARD_BREATH_LEVEL,
    alternation: bool = V110_ACC_ALTERNATION,
    phase_scale: float = V110_ACC_PHASE_SCALE,
    n_max: int = V110_ACC_N_MAX,
    f_max: int = V110_ACC_F_MAX,
    n_digits: int = V110_ACC_N_DIGITS,
    grad_clip: float = 1.0,
):
    """Mirror of v109pi step, with forward swapped to v110-acc."""
    key = ("v110_acc", id(model), id(opt), int(K), int(B),
           float(factor_aux_weight), float(calib_weight), float(var_loss_weight),
           bool(hard_breath_level), bool(alternation), float(phase_scale),
           int(n_max), int(f_max), int(n_digits), float(grad_clip))
    if key in _JIT_V110_ACC_CACHE:
        return _JIT_V110_ACC_CACHE[key]

    fw, aw, vw, gc = float(factor_aux_weight), float(calib_weight), \
                     float(var_loss_weight), float(grad_clip)
    params = opt.params
    print(
        f"[JIT] compile v110-acc fg step: K={K} B={B} n_digits={n_digits} "
        f"alternation={alternation} phase={phase_scale} vw={vw}...",
        flush=True,
    )

    @TinyJit
    def _step(
        domain_init, node_kinds, staging_mask, head_op_mask,
        gold_digits, gold_bins, observed_mask, factor_gold_bin, factor_valid,
    ):
        opt.zero_grad()

        tree_lh, var_lh, fac_lh, calib_h = fg_breathing_forward_v110_acc(
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

    _JIT_V110_ACC_CACHE[key] = _step
    print(f"[JIT] v110-acc step ready (cache={len(_JIT_V110_ACC_CACHE)})", flush=True)
    return _step


def _compile_jit_fg_eval_v110_acc(
    model: Any,
    K: int,
    B: int,
    n_max: int = V110_ACC_N_MAX,
    f_max: int = V110_ACC_F_MAX,
    n_digits: int = V110_ACC_N_DIGITS,
    alternation: bool = V110_ACC_ALTERNATION,
    phase_scale: float = V110_ACC_PHASE_SCALE,
):
    key = ("eval_v110_acc", id(model), int(K), int(B), int(n_max), int(f_max),
           int(n_digits), bool(alternation), float(phase_scale))
    if key in _JIT_V110_ACC_CACHE:
        return _JIT_V110_ACC_CACHE[key]

    print(
        f"[JIT] compile v110-acc fg eval: K={K} B={B} phase={phase_scale}...",
        flush=True,
    )

    @TinyJit
    def _eval(
        domain_init, node_kinds, staging_mask, head_op_mask,
        gold_digits, observed_mask,
    ):
        tree_lh, _, _, _ = fg_breathing_forward_v110_acc(
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

    _JIT_V110_ACC_CACHE[key] = _eval
    print(f"[JIT] v110-acc eval ready (cache={len(_JIT_V110_ACC_CACHE)})", flush=True)
    return _eval
