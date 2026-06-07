"""v110-step3 + Tree-logits feedback channel.

Adds a "variable-to-future-self" message channel: each breath's
soft-argmax of tree_logits is fed back as additive embedding to the
NEXT breath's residual stream at variable positions.

Architectural intent:
  Currently the tree_codebook readout (digit predictions) is used ONLY
  for the per-breath CE loss, then discarded. The next breath can't see
  the discrete commitment from the previous breath — only the residual
  stream, which contains it implicitly via the codebook projection.

  This adds an explicit feedback channel: the model's own committed
  beliefs become input to the next breath. In BP terms, this is the
  variable→future-self message that complements the existing
  factor↔variable messages.

Mechanism (per breath k, after tree_logits_k computed):
  probs_k       = softmax(tree_logits_k)             # (B, n_max, 5, 10)
  fb            = probs_k @ tree_codebook            # (B, n_max, H)
                                                       sum over (level, digit)
  fb_ln         = layernorm(fb, ln_g, ln_b)
  feedback_k    = gate * fb_ln

  At start of breath k+1:
  x[:, :n_max, :] += feedback_k

Warm-start safety:
  feedback_gate is initialized to ZERO → feedback_k=0 → step 0 forward
  is byte-identical to v110-step3 (mcbp with noise=0).

SBP composability:
  Forward also accepts (noise, noise_scale) tensors — the noise is added
  to x at start of breath, same as mcbp/SBP. Setting noise_scale=0 gives
  pure tree-feedback (Path B); setting noise_scale=σ gives feedback+SBP
  (Path C).

New parameters (~2K, trivial):
  fg_v111_feedback_gate   : scalar (1,)  zero-init
  fg_v111_feedback_ln_g   : (H,)         ones
  fg_v111_feedback_ln_b   : (H,)         zeros
"""
from __future__ import annotations

import math
import os
from typing import Any

import numpy as np
from tinygrad import Tensor, dtypes
from tinygrad.engine.jit import TinyJit

from mycelium.factor_graph_v107 import embed_factor_graph_v100_aligned
from mycelium.factor_graph_v109pi import fg_layer_forward_v109pi
from mycelium.factor_graph_v109 import _apply_waist_v109
from mycelium.factor_graph_v110_acc import (
    _acc_notebook_read, _acc_notebook_write,
)
from mycelium.factor_graph_v110_photon import _photon_gate
from mycelium.factor_graph_v110_step import (
    V110_STEP_N_MAX, V110_STEP_F_MAX, V110_STEP_N_DIGITS,
    V110_STEP_K_MAX, V110_STEP_N_HEADS, V110_STEP_ALTERNATION,
    V110_STEP_PHASE_SCALE, V110_STEP_GATE_PROFILE, V110_STEP_PHOTON_ALPHA,
)
from mycelium.factor_graph_v110_step3 import (
    V110_STEP3_N_MAX, V110_STEP3_F_MAX, V110_STEP3_N_DIGITS,
    V110_STEP3_K_MAX, V110_STEP3_WAIST_DIM,
    V110_STEP3_ALTERNATION, V110_STEP3_PHASE_SCALE,
    V110_STEP3_GATE_PROFILE, V110_STEP3_PHOTON_ALPHA,
    attach_fg_params_v110_step3,
)


# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

V110_STEP3_FB_GATE_INIT = float(os.environ.get("V110_STEP3_FB_GATE_INIT", "0.0"))


def attach_fg_params_v110_step3_fb(
    model: Any,
    hidden: int,
    n_max: int = V110_STEP3_N_MAX,
    f_max: int = V110_STEP3_F_MAX,
    k_max: int = V110_STEP3_K_MAX,
    n_digits: int = V110_STEP3_N_DIGITS,
    n_code: int = 32,
    ib_centroids_path: str = ".cache/ib_codebook_gsm8k_K32.npy",
    waist_dim: int = V110_STEP3_WAIST_DIM,
) -> None:
    """Attach v110-step3 + 3 new feedback params to model."""
    attach_fg_params_v110_step3(
        model, hidden=hidden,
        n_max=n_max, f_max=f_max, k_max=k_max,
        n_digits=n_digits, n_code=n_code, ib_centroids_path=ib_centroids_path,
        waist_dim=waist_dim,
    )

    if hasattr(model, "fg_v111_feedback_gate"):
        return

    # Zero-init gate → byte-identical to v110-step3 at step 0.
    model.fg_v111_feedback_gate = Tensor(
        np.array([V110_STEP3_FB_GATE_INIT], dtype=np.float32)
    ).contiguous().realize()

    # LayerNorm params (default identity-init)
    model.fg_v111_feedback_ln_g = Tensor(
        np.ones((hidden,), dtype=np.float32)
    ).contiguous().realize()
    model.fg_v111_feedback_ln_b = Tensor(
        np.zeros((hidden,), dtype=np.float32)
    ).contiguous().realize()

    n_new = 1 + 2 * hidden
    print(f"[v111-fb] tree-feedback params attached: gate={V110_STEP3_FB_GATE_INIT} "
          f"(zero-init → byte-identical warm-start)  +{n_new:,} params",
          flush=True)


def fg_v110_step3_fb_parameters(model: Any) -> list[Tensor]:
    """v110-step3 (acc) params + 3 new feedback params."""
    from scripts.v110_acc_factor_graph_train import collect_fg_params_v110_acc
    params = collect_fg_params_v110_acc(model)
    params = list(params)
    params.extend([
        model.fg_v111_feedback_gate,
        model.fg_v111_feedback_ln_g,
        model.fg_v111_feedback_ln_b,
    ])
    return params


def fg_v110_step3_fb_state_dict(model: Any) -> dict[str, Tensor]:
    """v110-step3 (acc) state_dict + 3 new feedback entries."""
    from scripts.v110_acc_factor_graph_train import model_state_dict_v110_acc
    sd = dict(model_state_dict_v110_acc(model))
    sd["fg_v111_feedback_gate"]   = model.fg_v111_feedback_gate
    sd["fg_v111_feedback_ln_g"]   = model.fg_v111_feedback_ln_g
    sd["fg_v111_feedback_ln_b"]   = model.fg_v111_feedback_ln_b
    return sd


def _layernorm_fb(x: Tensor, g: Tensor, b: Tensor, eps: float = 1e-5) -> Tensor:
    """Per-position LayerNorm over hidden dim. Equivalent to nn.LayerNorm(H)."""
    x = x.cast(dtypes.float)
    mean = x.mean(axis=-1, keepdim=True)
    var  = ((x - mean) ** 2).mean(axis=-1, keepdim=True)
    x_n  = (x - mean) / (var + eps).sqrt()
    return x_n * g.cast(dtypes.float) + b.cast(dtypes.float)


def fg_breathing_forward_v110_step3_fb(
    model: Any,
    domain_init: Tensor,
    node_kinds: Tensor,
    staging_mask: Tensor,
    head_op_mask: Tensor,
    noise: Tensor,           # (K_max, B, T, H) — for SBP composability
    noise_scale: Tensor,     # scalar Tensor — sweep without recompile
    K: int,
    n_max: int = V110_STEP3_N_MAX,
    f_max: int = V110_STEP3_F_MAX,
    n_digits: int = V110_STEP3_N_DIGITS,
    alternation: bool = V110_STEP3_ALTERNATION,
    phase_scale: float = V110_STEP3_PHASE_SCALE,
    gate_profile: str = V110_STEP3_GATE_PROFILE,
    photon_alpha: float = V110_STEP3_PHOTON_ALPHA,
):
    """v110-step3 + tree-logits feedback channel + SBP-compatible noise input.

    Returns (tree_logits_history, var_logits_history, factor_logits_history,
             calib_history, step_mags_history) — same shape as v110_step.

    With gate=0 and noise_scale=0, output is byte-identical to v110-step3.
    """
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

    fb_gate = model.fg_v111_feedback_gate
    fb_ln_g = model.fg_v111_feedback_ln_g
    fb_ln_b = model.fg_v111_feedback_ln_b

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
    step_mags_history     = []
    tree_cb_flat = tree_codebook.reshape(n_digits * 10, H)
    notebook_slots: list[Tensor] = []

    # Tree-feedback buffer: holds previous breath's feedback for next breath.
    # None on first breath (no feedback to read).
    feedback_buffer: Tensor | None = None

    for k in range(K):
        x_acc_delta = _acc_notebook_read(
            x, notebook_slots,
            acc_W_q, acc_W_k, acc_W_v, acc_W_o, acc_b_o,
        )
        x = x + x_acc_delta

        # *** Tree-logits feedback: add previous breath's commitment ***
        if feedback_buffer is not None:
            # Pad feedback (B, n_max, H) to (B, T, H) with zeros at factor positions
            fb_zero_pad = Tensor.zeros(B, f_max, H, dtype=dtypes.half)
            fb_padded   = Tensor.cat(feedback_buffer.cast(dtypes.half), fb_zero_pad, dim=1)
            x = x + fb_padded.cast(x.dtype)

        # *** Monte Carlo BP noise injection on residual stream ***
        x = x + (noise[k] * noise_scale).cast(x.dtype)

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
        step   = gate_k * delta
        x      = x_pre + step

        step_mag_k = step.cast(dtypes.float).square().mean()
        step_mags_history.append(step_mag_k)

        slot_k = _acc_notebook_write(x, acc_W_write, acc_b_write)
        notebook_slots.append(slot_k)

        x_ln  = _layernorm(x, model.ln_f_g, model.ln_f_b,
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

        # *** Write tree-feedback for next breath ***
        # probs_k: (B, n_max, n_digits, 10) -- soft argmax over digits per level
        # Sum over (level, digit) weighted by probability:
        #   fb[b, v, h] = sum_{level, digit} probs[b,v,level,digit] * codebook[level,digit,h]
        probs_k = tree_logits_k.softmax(axis=-1)
        probs_flat = probs_k.reshape(B, n_max, n_digits * 10).cast(dtypes.float)
        fb_raw     = probs_flat @ tree_cb_flat.cast(dtypes.float)   # (B, n_max, H)

        # LayerNorm + gate
        fb_ln = _layernorm_fb(fb_raw, fb_ln_g, fb_ln_b)
        feedback_buffer = fb_gate.reshape(1, 1, 1).cast(fb_ln.dtype) * fb_ln

    return (tree_logits_history, var_logits_history, factor_logits_history,
            calib_history, step_mags_history)


_JIT_V110_STEP3_FB_CACHE: dict = {}


def _compile_jit_fg_step_v110_step3_fb(
    model: Any,
    opt: Any,
    K: int,
    B: int,
    factor_aux_weight: float,
    calib_weight: float,
    var_loss_weight: float,
    balance_weight: float,
    uncertainty_min: float,
    hard_breath_level: bool,
    alternation: bool,
    phase_scale: float,
    n_max: int,
    f_max: int,
    n_digits: int,
    gate_profile: str,
    photon_alpha: float,
    grad_clip: float = 1.0,
):
    """Single-forward train step with feedback architecture.

    Mirrors _compile_jit_fg_step_v110_step3_sbp_simple but uses the
    feedback-enabled forward. Compatible with SBP-style noise injection
    via the noise/noise_scale JIT inputs.
    """
    key = ("v110_step3_fb_simple", id(model), id(opt), int(K), int(B),
           float(factor_aux_weight), float(calib_weight), float(var_loss_weight),
           float(balance_weight), float(uncertainty_min),
           bool(hard_breath_level), bool(alternation), float(phase_scale),
           int(n_max), int(f_max), int(n_digits),
           str(gate_profile), float(photon_alpha), float(grad_clip))
    if key in _JIT_V110_STEP3_FB_CACHE:
        return _JIT_V110_STEP3_FB_CACHE[key]

    fw, aw, vw, bw, um, gc = (float(factor_aux_weight), float(calib_weight),
                              float(var_loss_weight), float(balance_weight),
                              float(uncertainty_min), float(grad_clip))
    params = opt.params
    print(
        f"[JIT] compile v110-step3 FB step: K={K} B={B} "
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
            fg_breathing_forward_v110_step3_fb(
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

    _JIT_V110_STEP3_FB_CACHE[key] = _step
    print(f"[JIT] v110-step3 FB step ready (cache={len(_JIT_V110_STEP3_FB_CACHE)})",
          flush=True)
    return _step


def compile_jit_mcbp_fb(
    model: Any,
    K: int = V110_STEP3_K_MAX,
    n_max: int = V110_STEP3_N_MAX,
    f_max: int = V110_STEP3_F_MAX,
    n_digits: int = V110_STEP3_N_DIGITS,
    alternation: bool = V110_STEP3_ALTERNATION,
    phase_scale: float = V110_STEP3_PHASE_SCALE,
    gate_profile: str = V110_STEP3_GATE_PROFILE,
    photon_alpha: float = V110_STEP3_PHOTON_ALPHA,
):
    """MC-BP-style JIT eval with fb forward: returns final-breath tree_logits."""
    @TinyJit
    def _eval(
        domain_init: Tensor,
        node_kinds: Tensor,
        staging_mask: Tensor,
        head_op_mask: Tensor,
        noise: Tensor,
        noise_scale: Tensor,
    ) -> Tensor:
        Tensor.training = False
        tree_lh, _, _, _, _ = fg_breathing_forward_v110_step3_fb(
            model, domain_init, node_kinds, staging_mask, head_op_mask,
            noise, noise_scale,
            K=K, n_max=n_max, f_max=f_max, n_digits=n_digits,
            alternation=alternation, phase_scale=phase_scale,
            gate_profile=gate_profile, photon_alpha=photon_alpha,
        )
        return tree_lh[-1].realize()
    return _eval


def compile_jit_eval_v110_step3_fb(
    model: Any,
    K: int = V110_STEP3_K_MAX,
    B: int = 8,
    n_max: int = V110_STEP3_N_MAX,
    f_max: int = V110_STEP3_F_MAX,
    n_digits: int = V110_STEP3_N_DIGITS,
    alternation: bool = V110_STEP3_ALTERNATION,
    phase_scale: float = V110_STEP3_PHASE_SCALE,
    gate_profile: str = V110_STEP3_GATE_PROFILE,
    photon_alpha: float = V110_STEP3_PHOTON_ALPHA,
):
    """JIT'd eval forward — signature matches evaluate_v109 (6 inputs, 2 outputs).

    Noise is baked-in as zeros (deterministic eval). Feedback channel still
    active via the gate. Returns (pred_digits, cell_acc).
    """
    T = n_max + f_max
    H = 1024
    # Pre-realized constants captured in JIT closure
    eval_noise_zeros = Tensor.zeros(K, B, T, H, dtype=dtypes.half).contiguous().realize()
    eval_noise_scale = Tensor(
        np.array([0.0], dtype=np.float16)
    ).cast(dtypes.half).contiguous().realize()

    @TinyJit
    def _eval(
        domain_init: Tensor,
        node_kinds: Tensor,
        staging_mask: Tensor,
        head_op_mask: Tensor,
        gold_digits: Tensor,
        observed_mask: Tensor,
    ):
        Tensor.training = False
        tree_lh, _, _, _, _ = fg_breathing_forward_v110_step3_fb(
            model, domain_init, node_kinds, staging_mask, head_op_mask,
            eval_noise_zeros, eval_noise_scale,
            K=K, n_max=n_max, f_max=f_max, n_digits=n_digits,
            alternation=alternation, phase_scale=phase_scale,
            gate_profile=gate_profile, photon_alpha=photon_alpha,
        )
        final_tree = tree_lh[-1]
        pred_digits = final_tree.argmax(axis=-1)
        eq_per_pos  = (pred_digits == gold_digits.cast(dtypes.int)).cast(dtypes.float)
        eq          = eq_per_pos.prod(axis=-1)
        unobs       = (1 - observed_mask.cast(dtypes.float))
        cell_acc    = (eq * unobs).sum() / (unobs.sum() + 1e-8)
        return pred_digits.realize(), cell_acc.realize()
    return _eval
