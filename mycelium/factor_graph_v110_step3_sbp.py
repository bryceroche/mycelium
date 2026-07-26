"""v110-step3 + Stochastic BP consistency training.

Hypothesis (Jun 6, after MC-BP inference showed +0.0061 cell on hard with
noise=0.01): the inference-time noise tolerance window is narrow (0.005-
0.015) because v110-step3 was never trained with noise. Train v110-step3
with a stochastic-BP consistency loss to WIDEN the noise tolerance →
larger MC-BP signal at inference.

Mechanism (per training step):
  1. forward_det   = forward(x, noise_scale=0)         deterministic
  2. forward_noisy = forward(x, noise_scale=σ_train)   stochastic
  3. CE ladder + calib + factor_aux + balance applied to FORWARD_DET only
  4. Consistency loss: KL(softmax(det.tree_logits).detach, softmax(noisy.tree_logits))
     -- asymmetric teacher-student. Det is the teacher (only trained by CE),
        noisy is pulled toward teacher's posterior under perturbation.
  5. Total = CE_det + λ_cons * KL_cons

The detach in step (4) means: gradient on det weights comes only from CE.
Gradient on noisy weights comes from BOTH CE (via shared params) AND KL
(student pulled toward teacher). Net effect: weights converge to a
posterior shape that is BOTH gold-matching AND noise-tolerant.

RBM connection (for context):
  This is the simplest instance of a Contrastive-Divergence-like training
  signal. Det forward = positive phase. Noisy forward = "perturbed positive"
  (NOT a true negative phase — that would require Gibbs sampling from
  model). Consistency loss = pull perturbed phase toward clean phase.

  Each breath = one approximate block-Gibbs step on a continuous Boltzmann
  machine (visible = residual stream, hidden = waist activation).
  Adding noise per breath = Langevin sampling on the same energy landscape.
"""
from __future__ import annotations

import os
from typing import Any

from tinygrad import Tensor, dtypes
from tinygrad.engine.jit import TinyJit

from mycelium.factor_graph_v110_step3 import (
    V110_STEP3_N_MAX, V110_STEP3_F_MAX, V110_STEP3_N_DIGITS, V110_STEP3_N_HEADS,
    V110_STEP3_K_MAX, V110_STEP3_WAIST_DIM,
    V110_STEP3_ALTERNATION, V110_STEP3_PHASE_SCALE,
    V110_STEP3_GATE_PROFILE, V110_STEP3_PHOTON_ALPHA,
    V110_STEP3_HARD_BREATH_LEVEL,
    V110_STEP3_VAR_LOSS_WEIGHT, V110_STEP3_CALIB_WEIGHT,
    V110_STEP3_FACTOR_AUX_WEIGHT, V110_STEP3_BALANCE_WEIGHT,
    V110_STEP3_UNCERTAINTY_MIN,
    V110_STEP3_CODEBOOK_N, V110_STEP3_IB_CENTROIDS,
    attach_fg_params_v110_step3,
    fg_v110_step3_parameters, fg_v110_step3_state_dict,
)
from mycelium.factor_graph_v110_step3_mcbp import (
    fg_breathing_forward_v110_step3_mcbp,
)


# ---------------------------------------------------------------------------
# SBP-specific constants
# ---------------------------------------------------------------------------

# Training-time noise scale on x at start of breath. Sweet spot from MC-BP
# inference diagnostic was 0.01 — so we train at 2x that to encourage wider
# tolerance window without overwhelming the gradient.
V110_STEP3_SBP_NOISE_SCALE = float(os.environ.get(
    "V110_STEP3_SBP_NOISE_SCALE", "0.02"))

# Consistency loss weight. Start at 0.1 (mild pull). Higher = stronger
# noise tolerance but may slow CE convergence.
V110_STEP3_SBP_CONS_WEIGHT = float(os.environ.get(
    "V110_STEP3_SBP_CONS_WEIGHT", "0.1"))


_JIT_V110_STEP3_SBP_CACHE: dict = {}


def _compile_jit_fg_step_v110_step3_sbp_simple(
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
    """Simple SBP: one forward per step, noise tensor + scale as JIT inputs.

    Caller alternates noise={zeros, randn*σ} or samples noise_scale per step
    from a distribution. Single-forward graph → fits AMD memory.
    """
    key = ("v110_step3_sbp_simple", id(model), id(opt), int(K), int(B),
           float(factor_aux_weight), float(calib_weight), float(var_loss_weight),
           float(balance_weight), float(uncertainty_min),
           bool(hard_breath_level), bool(alternation), float(phase_scale),
           int(n_max), int(f_max), int(n_digits),
           str(gate_profile), float(photon_alpha), float(grad_clip))
    if key in _JIT_V110_STEP3_SBP_CACHE:
        return _JIT_V110_STEP3_SBP_CACHE[key]

    fw, aw, vw, bw, um, gc = (float(factor_aux_weight), float(calib_weight),
                              float(var_loss_weight), float(balance_weight),
                              float(uncertainty_min), float(grad_clip))
    params = opt.params
    print(
        f"[JIT] compile v110-step3 SBP-simple step: K={K} B={B} "
        f"profile={gate_profile} alpha={photon_alpha} balance={bw} u_min={um}...",
        flush=True,
    )

    @TinyJit
    def _step(
        domain_init, node_kinds, staging_mask, head_op_mask,
        gold_digits, gold_bins, observed_mask, factor_gold_bin, factor_valid,
        noise, noise_scale,
    ):
        opt.zero_grad()

        tree_lh, _, fac_lh, calib_h, step_mh = (
            fg_breathing_forward_v110_step3_mcbp(
                model, domain_init, node_kinds, staging_mask, head_op_mask,
                noise, noise_scale,
                K=K, n_max=n_max, f_max=f_max, n_digits=n_digits,
                alternation=alternation, phase_scale=phase_scale,
                gate_profile=gate_profile, photon_alpha=photon_alpha,
            )
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

        calib_per_breath = [c.mean().detach() for c in calib_h]
        calib_stack = Tensor.stack(*calib_per_breath, dim=0)
        uncertainty_k = (1.0 - calib_stack).clip(um, 1.0)
        step_stack = Tensor.stack(*step_mh, dim=0)
        normalized = step_stack / uncertainty_k
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

    _JIT_V110_STEP3_SBP_CACHE[key] = _step
    print(f"[JIT] v110-step3 SBP-simple step ready (cache={len(_JIT_V110_STEP3_SBP_CACHE)})",
          flush=True)
    return _step


def _compile_jit_fg_step_v110_step3_sbp(
    model: Any,
    opt: Any,
    K: int,
    B: int,
    noise_zeros: Tensor,           # (K_max, B, T, H) all-zero, realized once
    ns_det_tensor: Tensor,         # scalar Tensor [0.0], realized once
    ns_noisy_tensor: Tensor,       # scalar Tensor [sigma], realized once
    cons_weight: float = V110_STEP3_SBP_CONS_WEIGHT,
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
    key = ("v110_step3_sbp", id(model), id(opt), int(K), int(B),
           float(cons_weight),
           float(factor_aux_weight), float(calib_weight), float(var_loss_weight),
           float(balance_weight), float(uncertainty_min),
           bool(hard_breath_level), bool(alternation), float(phase_scale),
           int(n_max), int(f_max), int(n_digits),
           str(gate_profile), float(photon_alpha), float(grad_clip))
    if key in _JIT_V110_STEP3_SBP_CACHE:
        return _JIT_V110_STEP3_SBP_CACHE[key]

    fw, aw, vw, bw, um, gc = (float(factor_aux_weight), float(calib_weight),
                              float(var_loss_weight), float(balance_weight),
                              float(uncertainty_min), float(grad_clip))
    cw = float(cons_weight)
    params = opt.params
    print(
        f"[JIT] compile v110-step3 SBP step: K={K} B={B} cons_weight={cw} "
        f"profile={gate_profile} alpha={photon_alpha} balance={bw} u_min={um}...",
        flush=True,
    )

    @TinyJit
    def _step(
        domain_init, node_kinds, staging_mask, head_op_mask,
        gold_digits, gold_bins, observed_mask, factor_gold_bin, factor_valid,
        noise_noisy,   # (K_max, B, T, H) — fresh per step
    ):
        opt.zero_grad()

        # --- Det forward (noise_scale = 0) -------------------------------
        det_tree_lh, det_var_lh, det_fac_lh, det_calib_h, det_step_mh = (
            fg_breathing_forward_v110_step3_mcbp(
                model, domain_init, node_kinds, staging_mask, head_op_mask,
                noise_zeros, ns_det_tensor,
                K=K, n_max=n_max, f_max=f_max, n_digits=n_digits,
                alternation=alternation, phase_scale=phase_scale,
                gate_profile=gate_profile, photon_alpha=photon_alpha,
            )
        )

        # --- Noisy forward (noise_scale = sigma) -------------------------
        noisy_tree_lh, _, _, _, _ = (
            fg_breathing_forward_v110_step3_mcbp(
                model, domain_init, node_kinds, staging_mask, head_op_mask,
                noise_noisy, ns_noisy_tensor,
                K=K, n_max=n_max, f_max=f_max, n_digits=n_digits,
                alternation=alternation, phase_scale=phase_scale,
                gate_profile=gate_profile, photon_alpha=photon_alpha,
            )
        )

        # --- CE ladder on det --------------------------------------------
        unobs_float = (1 - observed_mask.cast(dtypes.float)).reshape(B * n_max)
        n_unobs_sum = unobs_float.sum() + 1e-8
        gd_flat = gold_digits.cast(dtypes.int).reshape(B * n_max, n_digits)

        var_loss_sum   = Tensor.zeros((), dtype=dtypes.float).contiguous()
        var_weight_sum = 0.0
        per_breath_ce_t: list[Tensor] = []

        for k, tree_logits_k in enumerate(det_tree_lh):
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

        # --- Factor aux on det -------------------------------------------
        n_valid_factors = factor_valid.cast(dtypes.float).sum() + 1e-8
        gold_fac_flat   = factor_gold_bin.cast(dtypes.int).reshape(B * f_max)
        gold_fac_oh     = gold_fac_flat.one_hot(200).cast(dtypes.float)
        valid_flat      = factor_valid.cast(dtypes.float).reshape(B * f_max)
        factor_aux_sum   = Tensor.zeros((), dtype=dtypes.float).contiguous()
        factor_aux_w_sum = 0.0
        for k_aux, fac_logits_k in enumerate(det_fac_lh):
            w_k_aux = 1.0 + float(k_aux) / float(max(K - 1, 1))
            fac_flat  = fac_logits_k.reshape(B * f_max, 200)
            fac_lp    = fac_flat.log_softmax(axis=-1)
            fac_nll   = -(fac_lp * gold_fac_oh).sum(axis=-1)
            fac_masked= fac_nll * valid_flat
            fac_ce_k  = fac_masked.sum() / n_valid_factors
            factor_aux_sum   = factor_aux_sum + fac_ce_k * w_k_aux
            factor_aux_w_sum += w_k_aux
        factor_aux_loss = factor_aux_sum / float(factor_aux_w_sum)

        # --- Calibration target on det -----------------------------------
        final_tree = det_tree_lh[-1]
        pred_digits_final = final_tree.argmax(axis=-1).detach()
        eq_per_pos = (pred_digits_final == gold_digits.cast(dtypes.int)).cast(dtypes.float)
        eq         = eq_per_pos.prod(axis=-1)
        unobs_2d   = (1 - observed_mask.cast(dtypes.float))
        eq_unobs   = eq * unobs_2d
        n_unobs_per = unobs_2d.sum(axis=-1) + 1e-8
        correct     = eq_unobs.sum(axis=-1) / n_unobs_per

        calib_loss_sum = Tensor.zeros((), dtype=dtypes.float).contiguous()
        for kc, calib_k in enumerate(det_calib_h):
            prog       = float(kc) / float(max(K - 1, 1))
            target_k   = 0.5 + (correct - 0.5) * prog
            calib_loss_sum = calib_loss_sum + ((calib_k - target_k) ** 2).mean()
        calib_loss = calib_loss_sum / float(K)

        # --- Goldilocks step balance loss on det -------------------------
        calib_per_breath = [c.mean().detach() for c in det_calib_h]
        calib_stack = Tensor.stack(*calib_per_breath, dim=0)
        uncertainty_k = (1.0 - calib_stack).clip(um, 1.0)
        step_stack = Tensor.stack(*det_step_mh, dim=0)
        normalized = step_stack / uncertainty_k
        norm_mean  = normalized.mean()
        norm_var   = ((normalized - norm_mean) ** 2).mean()
        norm_std   = (norm_var + 1e-12).sqrt()
        step_balance_loss = norm_std / (norm_mean + 1e-8)

        # --- Consistency loss (asymmetric KL teacher→student) -----------
        # KL(p_det || p_noisy) = sum_i p_det * (log p_det - log p_noisy)
        # Det is teacher (detached). Noisy is student (gets gradient).
        cons_loss_sum = Tensor.zeros((), dtype=dtypes.float).contiguous()
        for k_c in range(K):
            tl_det   = det_tree_lh[k_c].reshape(B * n_max, n_digits, 10).detach()
            tl_noisy = noisy_tree_lh[k_c].reshape(B * n_max, n_digits, 10)
            p_det      = tl_det.softmax(axis=-1)
            log_p_det  = tl_det.log_softmax(axis=-1)
            log_p_nz   = tl_noisy.log_softmax(axis=-1)
            kl_pp = (p_det * (log_p_det - log_p_nz)).sum(axis=-1)  # (B*n_max, n_digits)
            # Mask to unobserved variables only
            kl_masked = kl_pp.mean(axis=-1) * unobs_float.cast(kl_pp.dtype)
            kl_k = kl_masked.sum() / n_unobs_sum
            cons_loss_sum = cons_loss_sum + kl_k
        cons_loss = cons_loss_sum / float(K)

        cell_acc  = (eq_unobs.sum() / (unobs_2d.sum() + 1e-8)).detach()
        query_acc = correct.mean().detach()

        total_ce = (vw * var_loss + fw * factor_aux_loss + aw * calib_loss
                    + bw * step_balance_loss + cw * cons_loss)
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
            cons_loss.realize(),
            *(ce.realize() for ce in per_breath_ce_t),
            *(s.realize() for s in det_step_mh),
            *(c for c in calib_per_breath),
        )

    _JIT_V110_STEP3_SBP_CACHE[key] = _step
    print(f"[JIT] v110-step3 SBP step ready (cache={len(_JIT_V110_STEP3_SBP_CACHE)})",
          flush=True)
    return _step
