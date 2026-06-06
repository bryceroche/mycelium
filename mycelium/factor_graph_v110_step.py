"""v110-step: v110-full + step-size variance penalty (Goldilocks bite-sized steps).

Architectural intent:
  Each of K breaths produces a step contribution `gate_k * (h_quant - x_pre)`.
  In v110-full, the per-breath gate values can drift to extremes — some
  breaths take huge steps, others take negligible ones. The load is uneven.

  v110-step penalizes the COEFFICIENT OF VARIATION (std/mean) of per-breath
  step magnitudes. Minimizing CoV pushes all breaths toward similar step
  sizes — bite-sized, not too big, not too small, JUST right. Distributes
  the load across all K breath cycles.

This connects the per-breath calibration head to its original ODE-integrator
purpose: the calibration is the Dopri5-style step controller. The CoV penalty
is the simplest scalar regulator — a stepping stone toward fully adaptive
per-breath step sizes driven by calibration.

The forward returns step magnitudes per breath (in addition to the usual
tree/var/factor/calib histories). The JIT step computes step_balance_loss
from these and adds it to total_ce.
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
    V108_CALIB_WEIGHT, V108_FACTOR_AUX_WEIGHT,
    bins_to_digits_msd,
)
from mycelium.factor_graph_v109 import (
    V109_N_HEADS, V109_WAIST_DIM,
    fg_v109_parameters, fg_v109_state_dict,
    _apply_waist_v109,
)
from mycelium.factor_graph_v109pi import (
    V109PI_K_MAX, V109PI_N_MAX, V109PI_F_MAX, V109PI_N_HEADS, V109PI_N_DIGITS,
    V109PI_WAIST_DIM, V109PI_ALTERNATION, V109PI_PHASE_SCALE,
    V109PI_CODEBOOK_N, V109PI_IB_CENTROIDS,
    fg_layer_forward_v109pi,
)
from mycelium.factor_graph_v110_acc import (
    attach_fg_params_v110_acc,
    fg_v110_acc_parameters, fg_v110_acc_state_dict,
    _acc_notebook_read, _acc_notebook_write,
)
from mycelium.factor_graph_v110_photon import _photon_gate


# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

V110_STEP_TASK              = int(os.environ.get("V110_STEP_TASK", "0")) > 0
V110_STEP_K_MAX             = int(os.environ.get("V110_STEP_K_MAX", str(V109PI_K_MAX)))
V110_STEP_N_DIGITS          = int(os.environ.get("V110_STEP_N_DIGITS", str(V109PI_N_DIGITS)))
V110_STEP_WAIST_DIM         = int(os.environ.get("V110_STEP_WAIST_DIM", str(V109PI_WAIST_DIM)))
V110_STEP_ALTERNATION       = int(os.environ.get("V110_STEP_ALTERNATION", "1")) > 0
V110_STEP_HARD_BREATH_LEVEL = int(os.environ.get("V110_STEP_HARD_BREATH_LEVEL", "0")) > 0
V110_STEP_VAR_LOSS_WEIGHT   = float(os.environ.get("V110_STEP_VAR_LOSS_WEIGHT", "1.0"))
V110_STEP_N_MAX             = int(os.environ.get("V110_STEP_N_MAX", str(V109PI_N_MAX)))
V110_STEP_F_MAX             = int(os.environ.get("V110_STEP_F_MAX", str(V109PI_F_MAX)))
V110_STEP_T_MAX             = V110_STEP_N_MAX + V110_STEP_F_MAX
V110_STEP_CODEBOOK_N        = int(os.environ.get("V110_STEP_CODEBOOK_N", str(V109PI_CODEBOOK_N)))
V110_STEP_IB_CENTROIDS      = os.environ.get("V110_STEP_IB_CENTROIDS", V109PI_IB_CENTROIDS)
V110_STEP_CALIB_WEIGHT      = float(os.environ.get("V110_STEP_CALIB_WEIGHT", str(V108_CALIB_WEIGHT)))
V110_STEP_FACTOR_AUX_WEIGHT = float(os.environ.get("V110_STEP_FACTOR_AUX_WEIGHT", str(V108_FACTOR_AUX_WEIGHT)))
V110_STEP_N_HEADS           = V109PI_N_HEADS
V110_STEP_PHASE_SCALE       = float(os.environ.get("V110_STEP_PHASE_SCALE", str(V109PI_PHASE_SCALE)))

V110_STEP_GATE_PROFILE = os.environ.get("V110_STEP_GATE_PROFILE", "sin2_pi")
V110_STEP_PHOTON_ALPHA = float(os.environ.get("V110_STEP_PHOTON_ALPHA", "0.5"))

# Step-size regulation weight — coefficient of variation aux loss.
# At weight=0 → identical to v110-full. At weight=0.1 → mild push toward uniform.
V110_STEP_BALANCE_WEIGHT = float(os.environ.get("V110_STEP_BALANCE_WEIGHT", "0.05"))


# ---------------------------------------------------------------------------
# Forward pass — v110-full + per-breath step magnitude tracking
# ---------------------------------------------------------------------------

def fg_breathing_forward_v110_step(
    model: Any,
    domain_init: Tensor,
    node_kinds: Tensor,
    staging_mask: Tensor,
    head_op_mask: Tensor,
    K: int,
    n_max: int = V110_STEP_N_MAX,
    f_max: int = V110_STEP_F_MAX,
    n_digits: int = V110_STEP_N_DIGITS,
    alternation: bool = V110_STEP_ALTERNATION,
    phase_scale: float = V110_STEP_PHASE_SCALE,
    gate_profile: str = V110_STEP_GATE_PROFILE,
    photon_alpha: float = V110_STEP_PHOTON_ALPHA,
):
    """v110-full forward + per-breath step magnitude tracking.

    Returns one extra history list: step_mags_history. Each element is a
    scalar Tensor = mean square of (gate_k * delta_k) for breath k.
    """
    assert hasattr(model, "fg_v107_domain_codebook"), "missing v107 backbone"
    assert hasattr(model, "fg_v108_tree_codebook"), "missing v108 tree codebook"
    assert hasattr(model, "fg_v109_W_compress"), "missing v109 waist"
    assert hasattr(model, "fg_v110_acc_W_q"), "missing v110-acc notebook"

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

    photon_gates = [_photon_gate(k, K_max, gate_profile) for k in range(K_max)]
    binary_gates = [_photon_gate(k, K_max, "binary")    for k in range(K_max)]

    from mycelium.breathing import _layernorm

    tree_logits_history   = []
    var_logits_history    = []
    factor_logits_history = []
    calib_history         = []
    step_mags_history     = []  # NEW: per-breath step magnitude scalars

    tree_cb_flat = tree_codebook.reshape(n_digits * 10, H)
    notebook_slots: list[Tensor] = []

    for k in range(K):
        x_acc_delta = _acc_notebook_read(
            x, notebook_slots,
            acc_W_q, acc_W_k, acc_W_v, acc_W_o, acc_b_o,
        )
        x = x + x_acc_delta

        be_k  = breath_embed[k].reshape(1, 1, -1).cast(x.dtype)
        x_in  = x + be_k
        x_pre = x

        stk      = staging_mask[:, k, :, :]
        stk_h    = stk.reshape(B, 1, T, T).expand(B, V110_STEP_N_HEADS, T, T)
        combined = stk_h.cast(x.dtype) + head_op_mask.cast(x.dtype)

        cos_k = breath_cos[k]
        sin_k = breath_sin[k]
        h = x_in
        for layer in layers[:4]:
            h = fg_layer_forward_v109pi(layer, h, combined, cos_k, sin_k)

        cb  = semantic_codebook.cast(h.dtype)
        tmp = temperature.cast(h.dtype)
        scores   = h @ cb.T / tmp.reshape(1, 1, 1)
        weights  = scores.clip(-1e4, 1e4).softmax(-1)
        recon    = weights @ cb
        quantize = recon - h
        gate_quant_k = delta_gate_quant[k].cast(h.dtype).reshape(1, 1, 1)
        h_quant = h + gate_quant_k * quantize

        if alternation:
            gate_k_amp = (1.0 - photon_alpha) * binary_gates[k] + photon_alpha * photon_gates[k]
        else:
            gate_k_amp = 1.0

        if gate_k_amp > 0.0:
            h_quant_waist = _apply_waist_v109(
                h_quant, W_compress, b_compress, W_expand, b_expand,
            )
            if gate_k_amp >= 1.0:
                h_quant = h_quant_waist
            else:
                h_quant = (1.0 - gate_k_amp) * h_quant + gate_k_amp * h_quant_waist

        gate_k = delta_gate[k].cast(h_quant.dtype).reshape(1, 1, 1)
        delta  = h_quant - x_pre
        step   = gate_k * delta                            # (B, T, H) — the actual step
        x      = x_pre + step

        # Per-breath step magnitude: mean of squared step values across (B, T, H)
        step_mag_k = step.cast(dtypes.float).square().mean()
        step_mags_history.append(step_mag_k)

        slot_k = _acc_notebook_write(x, acc_W_write, acc_b_write)
        notebook_slots.append(slot_k)

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

    return (tree_logits_history, var_logits_history, factor_logits_history,
            calib_history, step_mags_history)


# ---------------------------------------------------------------------------
# Parameter attachment — same as v110-full (no new params)
# ---------------------------------------------------------------------------

def attach_fg_params_v110_step(
    model: Any,
    hidden: int,
    n_max: int = V110_STEP_N_MAX,
    f_max: int = V110_STEP_F_MAX,
    k_max: int | None = None,
    n_digits: int = V110_STEP_N_DIGITS,
    n_code: int = V110_STEP_CODEBOOK_N,
    ib_centroids_path: str = V110_STEP_IB_CENTROIDS,
    waist_dim: int = V110_STEP_WAIST_DIM,
) -> None:
    if k_max is None:
        k_max = V110_STEP_K_MAX
    attach_fg_params_v110_acc(
        model, hidden=hidden,
        n_max=n_max, f_max=f_max, k_max=k_max,
        n_digits=n_digits, n_code=n_code, ib_centroids_path=ib_centroids_path,
        waist_dim=waist_dim,
    )

    profile = V110_STEP_GATE_PROFILE
    alpha   = V110_STEP_PHOTON_ALPHA
    bw      = V110_STEP_BALANCE_WEIGHT
    gates = [_photon_gate(k, k_max, profile) for k in range(k_max)]
    binary = [_photon_gate(k, k_max, "binary") for k in range(k_max)]
    blended = [(1 - alpha) * binary[k] + alpha * gates[k] for k in range(k_max)]
    gates_s = " ".join(f"{g:.3f}" for g in gates)
    blended_s = " ".join(f"{g:.3f}" for g in blended)
    print(f"[v110-step] photon gate_profile={profile} alpha={alpha:.3f} "
          f"balance_weight={bw}", flush=True)
    print(f"            pure_profile_gates: [{gates_s}]", flush=True)
    print(f"            blended (warm-start mix): [{blended_s}]", flush=True)


fg_v110_step_parameters = fg_v110_acc_parameters
fg_v110_step_state_dict = fg_v110_acc_state_dict


# ---------------------------------------------------------------------------
# JIT step/eval
# ---------------------------------------------------------------------------

_JIT_V110_STEP_CACHE: dict = {}


def _compile_jit_fg_step_v110_step(
    model: Any,
    opt: Any,
    K: int,
    B: int,
    factor_aux_weight: float = V110_STEP_FACTOR_AUX_WEIGHT,
    calib_weight: float = V110_STEP_CALIB_WEIGHT,
    var_loss_weight: float = V110_STEP_VAR_LOSS_WEIGHT,
    balance_weight: float = V110_STEP_BALANCE_WEIGHT,
    hard_breath_level: bool = V110_STEP_HARD_BREATH_LEVEL,
    alternation: bool = V110_STEP_ALTERNATION,
    phase_scale: float = V110_STEP_PHASE_SCALE,
    n_max: int = V110_STEP_N_MAX,
    f_max: int = V110_STEP_F_MAX,
    n_digits: int = V110_STEP_N_DIGITS,
    gate_profile: str = V110_STEP_GATE_PROFILE,
    photon_alpha: float = V110_STEP_PHOTON_ALPHA,
    grad_clip: float = 1.0,
):
    key = ("v110_step", id(model), id(opt), int(K), int(B),
           float(factor_aux_weight), float(calib_weight), float(var_loss_weight),
           float(balance_weight),
           bool(hard_breath_level), bool(alternation), float(phase_scale),
           int(n_max), int(f_max), int(n_digits),
           str(gate_profile), float(photon_alpha), float(grad_clip))
    if key in _JIT_V110_STEP_CACHE:
        return _JIT_V110_STEP_CACHE[key]

    fw, aw, vw, bw, gc = (float(factor_aux_weight), float(calib_weight),
                          float(var_loss_weight), float(balance_weight),
                          float(grad_clip))
    params = opt.params
    print(
        f"[JIT] compile v110-step fg step: K={K} B={B} "
        f"profile={gate_profile} alpha={photon_alpha} balance={bw}...",
        flush=True,
    )

    @TinyJit
    def _step(
        domain_init, node_kinds, staging_mask, head_op_mask,
        gold_digits, gold_bins, observed_mask, factor_gold_bin, factor_valid,
    ):
        opt.zero_grad()

        tree_lh, var_lh, fac_lh, calib_h, step_mh = fg_breathing_forward_v110_step(
            model, domain_init, node_kinds, staging_mask, head_op_mask,
            K=K, n_max=n_max, f_max=f_max, n_digits=n_digits,
            alternation=alternation, phase_scale=phase_scale,
            gate_profile=gate_profile, photon_alpha=photon_alpha,
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

        # STEP-SIZE BALANCE LOSS: coefficient of variation across breaths
        # step_mh is a list of K scalar Tensors. Stack and compute std/mean.
        step_stack = Tensor.stack(*step_mh, dim=0)            # (K,)
        step_mean  = step_stack.mean()                        # scalar
        step_var   = ((step_stack - step_mean) ** 2).mean()   # scalar
        step_std   = (step_var + 1e-12).sqrt()                # scalar
        step_cov   = step_std / (step_mean + 1e-8)            # coefficient of variation
        step_balance_loss = step_cov

        cell_acc  = (eq_unobs.sum() / (unobs_2d.sum() + 1e-8)).detach()
        query_acc = correct.mean().detach()

        total_ce = (vw * var_loss + fw * factor_aux_loss + aw * calib_loss
                    + bw * step_balance_loss)
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
            step_balance_loss.realize(),
            *(ce.realize() for ce in per_breath_ce_t),
            *(s.realize() for s in step_mh),
        )

    _JIT_V110_STEP_CACHE[key] = _step
    print(f"[JIT] v110-step step ready (cache={len(_JIT_V110_STEP_CACHE)})", flush=True)
    return _step


def _compile_jit_fg_eval_v110_step(
    model: Any,
    K: int,
    B: int,
    n_max: int = V110_STEP_N_MAX,
    f_max: int = V110_STEP_F_MAX,
    n_digits: int = V110_STEP_N_DIGITS,
    alternation: bool = V110_STEP_ALTERNATION,
    phase_scale: float = V110_STEP_PHASE_SCALE,
    gate_profile: str = V110_STEP_GATE_PROFILE,
    photon_alpha: float = V110_STEP_PHOTON_ALPHA,
):
    key = ("eval_v110_step", id(model), int(K), int(B), int(n_max), int(f_max),
           int(n_digits), bool(alternation), float(phase_scale),
           str(gate_profile), float(photon_alpha))
    if key in _JIT_V110_STEP_CACHE:
        return _JIT_V110_STEP_CACHE[key]

    print(
        f"[JIT] compile v110-step fg eval: K={K} B={B} "
        f"profile={gate_profile} alpha={photon_alpha}...",
        flush=True,
    )

    @TinyJit
    def _eval(
        domain_init, node_kinds, staging_mask, head_op_mask,
        gold_digits, observed_mask,
    ):
        tree_lh, _, _, _, _ = fg_breathing_forward_v110_step(
            model, domain_init, node_kinds, staging_mask, head_op_mask,
            K=K, n_max=n_max, f_max=f_max, n_digits=n_digits,
            alternation=alternation, phase_scale=phase_scale,
            gate_profile=gate_profile, photon_alpha=photon_alpha,
        )
        final_tree = tree_lh[-1]
        pred_digits = final_tree.argmax(axis=-1)
        eq_per_pos = (pred_digits == gold_digits.cast(dtypes.int)).cast(dtypes.float)
        eq = eq_per_pos.prod(axis=-1)
        unobs = (1 - observed_mask.cast(dtypes.float))
        cell_acc = (eq * unobs).sum() / (unobs.sum() + 1e-8)
        return pred_digits.realize(), cell_acc.realize()

    _JIT_V110_STEP_CACHE[key] = _eval
    print(f"[JIT] v110-step eval ready (cache={len(_JIT_V110_STEP_CACHE)})", flush=True)
    return _eval
