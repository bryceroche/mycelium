"""v110-step3: Calibration-head-driven Goldilocks penalty.

v110-step  : penalty = CoV(step_mag_k)              (raw step uniformity)
v110-step2 : penalty = CoV(step_mag_k / CE_k)        (Dopri5 with per-breath CE)
v110-step3 : penalty = CoV(step_mag_k / (1 - calib_k))   ← THIS

The calibration head output calib_k ∈ [0, 1] is the model's CONFIDENCE
that breath k's answer is correct (trained with detached argmax-equals-gold
target). Per the architecture spec, calib_k is the "Dopri5-style error
estimator" — currently decorative, trained with weight 0.05.

v110-step3 puts it to work:
  uncertainty_k = 1 - calib_k     (remaining work)
  normalized_k  = step_mag_k / max(uncertainty_k, 0.05)
  penalty       = std(normalized) / mean(normalized)

Healthy invariant: step_k ∝ uncertainty_k.
  - High uncertainty (calib_k low) → big step OK (lots of work to do)
  - High confidence (calib_k high) → small step (just polish)

calib_k is DETACHED so backprop doesn't go through it — we don't want
the model lowering calibration to satisfy the constraint. We DO want the
calibration head to become PREDICTIVE: if it's noisy/decorative, the
penalty signal is weak; if it's well-calibrated, the penalty drives a
clean dynamics. The mechanism rewards calibration accuracy as a side
effect.
"""
from __future__ import annotations

import math
import os
from typing import Any

import numpy as np
from tinygrad import Tensor, dtypes
from tinygrad.engine.jit import TinyJit
from tinygrad.helpers import getenv

# Reuse v110-step's forward + helpers (same architecture, only loss differs)
from mycelium.factor_graph_v110_step import (
    V110_STEP_K_MAX as V110_STEP3_K_MAX,
    V110_STEP_N_MAX as V110_STEP3_N_MAX,
    V110_STEP_F_MAX as V110_STEP3_F_MAX,
    V110_STEP_T_MAX as V110_STEP3_T_MAX,
    V110_STEP_N_HEADS as V110_STEP3_N_HEADS,
    V110_STEP_N_DIGITS as V110_STEP3_N_DIGITS,
    V110_STEP_WAIST_DIM as V110_STEP3_WAIST_DIM,
    V110_STEP_ALTERNATION as V110_STEP3_ALTERNATION,
    V110_STEP_HARD_BREATH_LEVEL as V110_STEP3_HARD_BREATH_LEVEL,
    V110_STEP_VAR_LOSS_WEIGHT as V110_STEP3_VAR_LOSS_WEIGHT,
    V110_STEP_CODEBOOK_N as V110_STEP3_CODEBOOK_N,
    V110_STEP_IB_CENTROIDS as V110_STEP3_IB_CENTROIDS,
    V110_STEP_CALIB_WEIGHT as V110_STEP3_CALIB_WEIGHT,
    V110_STEP_FACTOR_AUX_WEIGHT as V110_STEP3_FACTOR_AUX_WEIGHT,
    V110_STEP_PHASE_SCALE as V110_STEP3_PHASE_SCALE,
    V110_STEP_GATE_PROFILE as V110_STEP3_GATE_PROFILE,
    V110_STEP_PHOTON_ALPHA as V110_STEP3_PHOTON_ALPHA,
    fg_breathing_forward_v110_step,
    attach_fg_params_v110_step,
    fg_v110_step_parameters,
    fg_v110_step_state_dict,
)
from mycelium.factor_graph_v110_photon import _photon_gate


V110_STEP3_TASK = int(os.environ.get("V110_STEP3_TASK", "0")) > 0
V110_STEP3_BALANCE_WEIGHT = float(os.environ.get("V110_STEP3_BALANCE_WEIGHT", "0.05"))
# Clip for (1 - calib_k) to avoid divide-by-near-zero when calib is high
V110_STEP3_UNCERTAINTY_MIN = float(os.environ.get("V110_STEP3_UNCERTAINTY_MIN", "0.05"))


def attach_fg_params_v110_step3(
    model: Any,
    hidden: int,
    n_max: int = V110_STEP3_N_MAX,
    f_max: int = V110_STEP3_F_MAX,
    k_max: int | None = None,
    n_digits: int = V110_STEP3_N_DIGITS,
    n_code: int = V110_STEP3_CODEBOOK_N,
    ib_centroids_path: str = V110_STEP3_IB_CENTROIDS,
    waist_dim: int = V110_STEP3_WAIST_DIM,
) -> None:
    if k_max is None:
        k_max = V110_STEP3_K_MAX
    attach_fg_params_v110_step(
        model, hidden=hidden,
        n_max=n_max, f_max=f_max, k_max=k_max,
        n_digits=n_digits, n_code=n_code, ib_centroids_path=ib_centroids_path,
        waist_dim=waist_dim,
    )
    bw = V110_STEP3_BALANCE_WEIGHT
    um = V110_STEP3_UNCERTAINTY_MIN
    print(f"[v110-step3] calibration-driven Goldilocks penalty weight={bw}",
          flush=True)
    print(f"             penalty = CoV(step_mag_k / max(1-calib_k, {um}))",
          flush=True)


fg_v110_step3_parameters = fg_v110_step_parameters
fg_v110_step3_state_dict = fg_v110_step_state_dict


_JIT_V110_STEP3_CACHE: dict = {}


def _compile_jit_fg_step_v110_step3(
    model: Any,
    opt: Any,
    K: int,
    B: int,
    factor_aux_weight: float = V110_STEP3_FACTOR_AUX_WEIGHT,
    calib_weight: float = V110_STEP3_CALIB_WEIGHT,
    var_loss_weight: float = V110_STEP3_VAR_LOSS_WEIGHT,
    balance_weight: float = V110_STEP3_BALANCE_WEIGHT,
    uncertainty_min: float = V110_STEP3_UNCERTAINTY_MIN,
    hard_breath_level: bool = V110_STEP3_HARD_BREATH_LEVEL,
    alternation: bool = V110_STEP3_ALTERNATION,
    phase_scale: float = V110_STEP3_PHASE_SCALE,
    n_max: int = V110_STEP3_N_MAX,
    f_max: int = V110_STEP3_F_MAX,
    n_digits: int = V110_STEP3_N_DIGITS,
    gate_profile: str = V110_STEP3_GATE_PROFILE,
    photon_alpha: float = V110_STEP3_PHOTON_ALPHA,
    grad_clip: float = 1.0,
):
    key = ("v110_step3", id(model), id(opt), int(K), int(B),
           float(factor_aux_weight), float(calib_weight), float(var_loss_weight),
           float(balance_weight), float(uncertainty_min),
           bool(hard_breath_level), bool(alternation), float(phase_scale),
           int(n_max), int(f_max), int(n_digits),
           str(gate_profile), float(photon_alpha), float(grad_clip))
    if key in _JIT_V110_STEP3_CACHE:
        return _JIT_V110_STEP3_CACHE[key]

    fw, aw, vw, bw, um, gc = (float(factor_aux_weight), float(calib_weight),
                              float(var_loss_weight), float(balance_weight),
                              float(uncertainty_min), float(grad_clip))
    params = opt.params
    print(
        f"[JIT] compile v110-step3 fg step: K={K} B={B} "
        f"profile={gate_profile} alpha={photon_alpha} balance={bw} u_min={um}...",
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

        # CALIBRATION-DRIVEN STEP BALANCE LOSS
        # Per-breath uncertainty = 1 - mean confidence across batch.
        # step_mag_k should scale with uncertainty_k. CoV of the ratio
        # should be small. Calib is DETACHED — we don't want the model
        # lowering calibration to satisfy the constraint.
        # calib_h[k] is (B,) — pool the batch and detach.
        calib_per_breath = [c.mean().detach() for c in calib_h]   # K scalars
        calib_stack = Tensor.stack(*calib_per_breath, dim=0)      # (K,)
        uncertainty_k = (1.0 - calib_stack).clip(um, 1.0)         # (K,) in [u_min, 1]
        step_stack = Tensor.stack(*step_mh, dim=0)                # (K,)
        normalized = step_stack / uncertainty_k                   # (K,)
        norm_mean  = normalized.mean()
        norm_var   = ((normalized - norm_mean) ** 2).mean()
        norm_std   = (norm_var + 1e-12).sqrt()
        step_balance_loss = norm_std / (norm_mean + 1e-8)

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
            *(c for c in calib_per_breath),
        )

    _JIT_V110_STEP3_CACHE[key] = _step
    print(f"[JIT] v110-step3 step ready (cache={len(_JIT_V110_STEP3_CACHE)})", flush=True)
    return _step


def _compile_jit_fg_eval_v110_step3(
    model: Any,
    K: int,
    B: int,
    n_max: int = V110_STEP3_N_MAX,
    f_max: int = V110_STEP3_F_MAX,
    n_digits: int = V110_STEP3_N_DIGITS,
    alternation: bool = V110_STEP3_ALTERNATION,
    phase_scale: float = V110_STEP3_PHASE_SCALE,
    gate_profile: str = V110_STEP3_GATE_PROFILE,
    photon_alpha: float = V110_STEP3_PHOTON_ALPHA,
):
    key = ("eval_v110_step3", id(model), int(K), int(B), int(n_max), int(f_max),
           int(n_digits), bool(alternation), float(phase_scale),
           str(gate_profile), float(photon_alpha))
    if key in _JIT_V110_STEP3_CACHE:
        return _JIT_V110_STEP3_CACHE[key]

    print(
        f"[JIT] compile v110-step3 fg eval: K={K} B={B} "
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

    _JIT_V110_STEP3_CACHE[key] = _eval
    print(f"[JIT] v110-step3 eval ready (cache={len(_JIT_V110_STEP3_CACHE)})", flush=True)
    return _eval
