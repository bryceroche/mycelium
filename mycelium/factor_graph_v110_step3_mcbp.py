"""v110-step3 + Monte Carlo BP: stochastic noise injection on h_quant per breath.

Hypothesis (Jun 6): PUCT failed because samples are correlated (tree path
inherits BP attractor). True Monte Carlo needs INDEPENDENT samples.
Inject Gaussian noise on h_quant per breath, run N forwards, average
tree_logits. Continuous posterior estimate from discrete stochastic
trajectories — the π-estimate pattern.

The forward is byte-identical to fg_breathing_forward_v110_step when
noise_scale=0. Same v110-step3 ckpt loads; no retraining required.

Noise injection location: on h_quant (the consolidated BP message)
AFTER waist blend, BEFORE delta computation. This perturbs the belief
that gets blended into the residual stream, propagating to subsequent
breaths via the K-iteration loop.
"""
from __future__ import annotations

import math
from typing import Any

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


def fg_breathing_forward_v110_step3_mcbp(
    model: Any,
    domain_init: Tensor,
    node_kinds: Tensor,
    staging_mask: Tensor,
    head_op_mask: Tensor,
    noise: Tensor,           # (K_max, B, T, H) — fresh per call
    noise_scale: Tensor,     # scalar Tensor — sweep without recompile
    K: int,
    n_max: int = V110_STEP_N_MAX,
    f_max: int = V110_STEP_F_MAX,
    n_digits: int = V110_STEP_N_DIGITS,
    alternation: bool = V110_STEP_ALTERNATION,
    phase_scale: float = V110_STEP_PHASE_SCALE,
    gate_profile: str = V110_STEP_GATE_PROFILE,
    photon_alpha: float = V110_STEP_PHOTON_ALPHA,
):
    """v110-step forward + per-breath Gaussian noise on h_quant.

    Returns tree_logits_history (only — eval doesn't need others).
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

    for k in range(K):
        # *** Monte Carlo BP noise injection on residual stream ***
        # noise[k] is (B, T, H); noise_scale is scalar Tensor.
        # When noise_scale == 0, x is unchanged (deterministic).
        # Injecting on x at start of breath propagates the noise through the
        # WHOLE breath (read, transformer, waist, blend) and into next breath.
        x = x + (noise[k] * noise_scale).cast(x.dtype)

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

    return (tree_logits_history, var_logits_history, factor_logits_history,
            calib_history, step_mags_history)


def compile_jit_mcbp(
    model: Any,
    K: int = V110_STEP_K_MAX,
    n_max: int = V110_STEP_N_MAX,
    f_max: int = V110_STEP_F_MAX,
    n_digits: int = V110_STEP_N_DIGITS,
    alternation: bool = V110_STEP_ALTERNATION,
    phase_scale: float = V110_STEP_PHASE_SCALE,
    gate_profile: str = V110_STEP_GATE_PROFILE,
    photon_alpha: float = V110_STEP_PHOTON_ALPHA,
):
    """Return a JIT-compiled MC-BP forward: returns final-breath tree_logits."""
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
        tree_lh, _, _, _, _ = fg_breathing_forward_v110_step3_mcbp(
            model, domain_init, node_kinds, staging_mask, head_op_mask,
            noise, noise_scale,
            K=K, n_max=n_max, f_max=f_max, n_digits=n_digits,
            alternation=alternation, phase_scale=phase_scale,
            gate_profile=gate_profile, photon_alpha=photon_alpha,
        )
        return tree_lh[-1].realize()
    return _eval
