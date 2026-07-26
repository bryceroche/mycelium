"""v112: Dual-notebook architecture.

TWO accumulate-style notebooks instead of one. Each notebook is the
v110-acc cross-attention pattern (causal cross-attn over slots, zero-init
W_o), but they're written conditionally based on the breath phase:

  COMMIT notebook (variable→factor decisions):
    Written ONLY on collapse (even) breaths — phase where waist is ON.
    Read on every breath.

  PROPAGATION notebook (carry/observation messages):
    Written ONLY on expand (odd) breaths — phase where waist is OFF.
    Read on every breath.

Both notebooks use causal cross-attention identical to v110-acc. Both
have zero-init W_o so delta=0 at initialization (the new modules are
inert at step 0).

This is the v110-step3 base architecture (calibration-driven Goldilocks +
step balance + per-breath step magnitude tracking + waist alternation +
photon gating + π-cycled Q rotation) PLUS two notebooks that specialize
on collapse vs expand semantics.

Cold-start friendly: with zero-init W_o on both notebooks, the model
starts as v110-step3 forward and learns to use the notebooks from
scratch through task gradient.
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
# COMMIT notebook helpers — mirror v110-acc structure
# ---------------------------------------------------------------------------

def _commit_notebook_read(
    x: Tensor,               # (B, T, H) current residual
    slots: list[Tensor],     # list of (B, H) committed slot vectors
    commit_W_q: Tensor,      # (H, H)
    commit_W_k: Tensor,      # (H, H)
    commit_W_v: Tensor,      # (H, H)
    commit_W_o: Tensor,      # (H, H) zero-init
    commit_b_o: Tensor,      # (H,)   zero-init
) -> Tensor:
    """Causal cross-attn over COMMIT notebook slots. Returns delta_x (B, T, H).
    Empty slots → returns zero contribution."""
    if len(slots) == 0:
        return x * 0.0

    H = int(x.shape[-1])
    nb = Tensor.stack(*slots, dim=1).cast(x.dtype)  # (B, k, H)

    Wq = commit_W_q.cast(x.dtype)
    Wk = commit_W_k.cast(x.dtype)
    Wv = commit_W_v.cast(x.dtype)
    Wo = commit_W_o.cast(x.dtype)
    bo = commit_b_o.reshape(1, 1, -1).cast(x.dtype)

    q = x  @ Wq
    K = nb @ Wk
    V = nb @ Wv

    scale = 1.0 / math.sqrt(H)
    scores  = (q @ K.transpose(-1, -2)) * scale
    weights = scores.clip(-1e4, 1e4).softmax(-1)
    ctx     = weights @ V
    delta   = ctx @ Wo + bo
    return delta


def _commit_notebook_write(
    h: Tensor,                # (B, T, H) post-step residual
    commit_W_write: Tensor,   # (H, H)
    commit_b_write: Tensor,   # (H,)
) -> Tensor:
    """Pool post-step residual to a single COMMIT slot vector. Returns (B, H)."""
    pool = h.mean(axis=1)
    Ww = commit_W_write.cast(h.dtype)
    bw = commit_b_write.reshape(1, -1).cast(h.dtype)
    slot = pool @ Ww + bw
    return slot


# ---------------------------------------------------------------------------
# PROPAGATION notebook helpers — same structure, separate params
# ---------------------------------------------------------------------------

def _prop_notebook_read(
    x: Tensor,
    slots: list[Tensor],
    prop_W_q: Tensor,
    prop_W_k: Tensor,
    prop_W_v: Tensor,
    prop_W_o: Tensor,
    prop_b_o: Tensor,
) -> Tensor:
    """Causal cross-attn over PROPAGATION notebook slots."""
    if len(slots) == 0:
        return x * 0.0

    H = int(x.shape[-1])
    nb = Tensor.stack(*slots, dim=1).cast(x.dtype)

    Wq = prop_W_q.cast(x.dtype)
    Wk = prop_W_k.cast(x.dtype)
    Wv = prop_W_v.cast(x.dtype)
    Wo = prop_W_o.cast(x.dtype)
    bo = prop_b_o.reshape(1, 1, -1).cast(x.dtype)

    q = x  @ Wq
    K = nb @ Wk
    V = nb @ Wv

    scale = 1.0 / math.sqrt(H)
    scores  = (q @ K.transpose(-1, -2)) * scale
    weights = scores.clip(-1e4, 1e4).softmax(-1)
    ctx     = weights @ V
    delta   = ctx @ Wo + bo
    return delta


def _prop_notebook_write(
    h: Tensor,
    prop_W_write: Tensor,
    prop_b_write: Tensor,
) -> Tensor:
    """Pool post-step residual to a single PROPAGATION slot vector."""
    pool = h.mean(axis=1)
    Ww = prop_W_write.cast(h.dtype)
    bw = prop_b_write.reshape(1, -1).cast(h.dtype)
    slot = pool @ Ww + bw
    return slot


# ---------------------------------------------------------------------------
# Parameter attachment
# ---------------------------------------------------------------------------

def attach_fg_params_v112(
    model: Any,
    hidden: int,
    n_max: int = V110_STEP3_N_MAX,
    f_max: int = V110_STEP3_F_MAX,
    k_max: int = V110_STEP3_K_MAX,
    n_digits: int = V110_STEP3_N_DIGITS,
    n_code: int = 32,
    ib_centroids_path: str = ".cache/ib_centroids_gsm8k_partial.npz",
    waist_dim: int = V110_STEP3_WAIST_DIM,
) -> None:
    """Attach v110-step3 base + v112 commit + prop notebook params.

    14 new params total (7 per notebook):
      fg_v114_commit_W_q / _W_k / _W_v       : (H, H) small Gaussian init
      fg_v114_commit_W_o                     : (H, H) ZERO-init (safe start)
      fg_v114_commit_b_o                     : (H,)   ZERO-init
      fg_v114_commit_W_write                 : (H, H) small Gaussian init
      fg_v114_commit_b_write                 : (H,)   ZERO-init
      fg_v114_prop_W_q / _W_k / _W_v         : (H, H) small Gaussian init
      fg_v114_prop_W_o                       : (H, H) ZERO-init
      fg_v114_prop_b_o                       : (H,)   ZERO-init
      fg_v114_prop_W_write                   : (H, H) small Gaussian init
      fg_v114_prop_b_write                   : (H,)   ZERO-init
    """
    attach_fg_params_v110_step3(
        model, hidden=hidden,
        n_max=n_max, f_max=f_max, k_max=k_max,
        n_digits=n_digits, n_code=n_code, ib_centroids_path=ib_centroids_path,
        waist_dim=waist_dim,
    )
    if hasattr(model, "fg_v114_commit_W_q"):
        return

    H = hidden
    scale_init = 0.02

    # COMMIT notebook params
    rng_c = np.random.RandomState(112001)
    Wq_c = (rng_c.randn(H, H).astype(np.float32) * scale_init)
    Wk_c = (rng_c.randn(H, H).astype(np.float32) * scale_init)
    Wv_c = (rng_c.randn(H, H).astype(np.float32) * scale_init)
    Wo_c = np.zeros((H, H), dtype=np.float32)
    bo_c = np.zeros((H,), dtype=np.float32)
    Ww_c = (rng_c.randn(H, H).astype(np.float32) * scale_init)
    bw_c = np.zeros((H,), dtype=np.float32)

    model.fg_v114_commit_W_q     = Tensor(Wq_c, dtype=dtypes.float).contiguous()
    model.fg_v114_commit_W_k     = Tensor(Wk_c, dtype=dtypes.float).contiguous()
    model.fg_v114_commit_W_v     = Tensor(Wv_c, dtype=dtypes.float).contiguous()
    model.fg_v114_commit_W_o     = Tensor(Wo_c, dtype=dtypes.float).contiguous()
    model.fg_v114_commit_b_o     = Tensor(bo_c, dtype=dtypes.float).contiguous()
    model.fg_v114_commit_W_write = Tensor(Ww_c, dtype=dtypes.float).contiguous()
    model.fg_v114_commit_b_write = Tensor(bw_c, dtype=dtypes.float).contiguous()

    # PROPAGATION notebook params — independent RNG seed
    rng_p = np.random.RandomState(112002)
    Wq_p = (rng_p.randn(H, H).astype(np.float32) * scale_init)
    Wk_p = (rng_p.randn(H, H).astype(np.float32) * scale_init)
    Wv_p = (rng_p.randn(H, H).astype(np.float32) * scale_init)
    Wo_p = np.zeros((H, H), dtype=np.float32)
    bo_p = np.zeros((H,), dtype=np.float32)
    Ww_p = (rng_p.randn(H, H).astype(np.float32) * scale_init)
    bw_p = np.zeros((H,), dtype=np.float32)

    model.fg_v114_prop_W_q       = Tensor(Wq_p, dtype=dtypes.float).contiguous()
    model.fg_v114_prop_W_k       = Tensor(Wk_p, dtype=dtypes.float).contiguous()
    model.fg_v114_prop_W_v       = Tensor(Wv_p, dtype=dtypes.float).contiguous()
    model.fg_v114_prop_W_o       = Tensor(Wo_p, dtype=dtypes.float).contiguous()
    model.fg_v114_prop_b_o       = Tensor(bo_p, dtype=dtypes.float).contiguous()
    model.fg_v114_prop_W_write   = Tensor(Ww_p, dtype=dtypes.float).contiguous()
    model.fg_v114_prop_b_write   = Tensor(bw_p, dtype=dtypes.float).contiguous()

    n_params_per = 5 * H * H + 2 * H
    n_params = 2 * n_params_per
    print(
        f"[v112] dual-notebook attached: {n_params:,} new params total "
        f"({n_params_per:,} per notebook at H={H})",
        flush=True,
    )
    print(
        f"[v112] commit notebook writes on EVEN (collapse) breaths; "
        f"prop notebook writes on ODD (expand) breaths; both READ every breath.",
        flush=True,
    )


def fg_v112_parameters(model: Any) -> list[Tensor]:
    """Backbone + v107/v108/v109 + v112 dual notebook params.

    NOTE: v110-acc notebook params (fg_v110_acc_*) are attached by the chain
    (attach_fg_params_v110_step3 → v110_step → v110_acc → v109pi → v109)
    but are NOT used in v112's forward — v112 uses its own commit + prop
    notebooks. Excluding them here keeps the optimizer's param list aligned
    with what actually receives gradients.
    """
    from mycelium.factor_graph_v109 import fg_v109_parameters
    params = []
    sw = model.block.shared
    params += [sw.wv, sw.bv, sw.wo, sw.bo, sw.w_out, sw.b_out,
               sw.in_ln_g, sw.in_ln_b, sw.post_ln_g, sw.post_ln_b]
    for layer in model.block.layers:
        params += [layer.wq, layer.bq, layer.wk, layer.bk, layer.w_in, layer.b_in]
    params += [model.ln_f_g, model.ln_f_b]
    params += fg_v109_parameters(model)  # v107 backbone + v108 tree + v109 waist
    params.extend([
        model.fg_v114_commit_W_q,
        model.fg_v114_commit_W_k,
        model.fg_v114_commit_W_v,
        model.fg_v114_commit_W_o,
        model.fg_v114_commit_b_o,
        model.fg_v114_commit_W_write,
        model.fg_v114_commit_b_write,
        model.fg_v114_prop_W_q,
        model.fg_v114_prop_W_k,
        model.fg_v114_prop_W_v,
        model.fg_v114_prop_W_o,
        model.fg_v114_prop_b_o,
        model.fg_v114_prop_W_write,
        model.fg_v114_prop_b_write,
    ])
    return params


def fg_v112_state_dict(model: Any) -> dict[str, Tensor]:
    """State dict containing backbone + all fg families incl. v112 dual notebooks."""
    from scripts.v110_acc_factor_graph_train import model_state_dict_v110_acc
    sd = dict(model_state_dict_v110_acc(model))
    sd["fg_v114_commit_W_q"]     = model.fg_v114_commit_W_q
    sd["fg_v114_commit_W_k"]     = model.fg_v114_commit_W_k
    sd["fg_v114_commit_W_v"]     = model.fg_v114_commit_W_v
    sd["fg_v114_commit_W_o"]     = model.fg_v114_commit_W_o
    sd["fg_v114_commit_b_o"]     = model.fg_v114_commit_b_o
    sd["fg_v114_commit_W_write"] = model.fg_v114_commit_W_write
    sd["fg_v114_commit_b_write"] = model.fg_v114_commit_b_write
    sd["fg_v114_prop_W_q"]       = model.fg_v114_prop_W_q
    sd["fg_v114_prop_W_k"]       = model.fg_v114_prop_W_k
    sd["fg_v114_prop_W_v"]       = model.fg_v114_prop_W_v
    sd["fg_v114_prop_W_o"]       = model.fg_v114_prop_W_o
    sd["fg_v114_prop_b_o"]       = model.fg_v114_prop_b_o
    sd["fg_v114_prop_W_write"]   = model.fg_v114_prop_W_write
    sd["fg_v114_prop_b_write"]   = model.fg_v114_prop_b_write
    return sd


# ---------------------------------------------------------------------------
# Forward pass — v110-step3 base + dual notebooks
# ---------------------------------------------------------------------------

def fg_breathing_forward_v112(
    model: Any,
    domain_init: Tensor,
    node_kinds: Tensor,
    staging_mask: Tensor,
    head_op_mask: Tensor,
    noise: Tensor,           # (K_max, B, T, H) — SBP-composable
    noise_scale: Tensor,     # scalar Tensor
    K: int,
    n_max: int = V110_STEP3_N_MAX,
    f_max: int = V110_STEP3_F_MAX,
    n_digits: int = V110_STEP3_N_DIGITS,
    alternation: bool = V110_STEP3_ALTERNATION,
    phase_scale: float = V110_STEP3_PHASE_SCALE,
    gate_profile: str = V110_STEP3_GATE_PROFILE,
    photon_alpha: float = V110_STEP3_PHOTON_ALPHA,
):
    """v110-step3 forward + DUAL accumulate notebooks (commit + prop).

    Returns (tree_logits_history, var_logits_history, factor_logits_history,
             calib_history, step_mags_history) — same shape as v110-step.

    At init (both notebooks W_o=0), the contribution from both notebooks
    is zero, so forward is byte-identical to v110-step3 base.
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

    # Note: v110-acc notebook params still exist on model but are NOT used
    # by v112 forward — v112 uses its own commit + prop notebooks below.

    commit_W_q     = model.fg_v114_commit_W_q
    commit_W_k     = model.fg_v114_commit_W_k
    commit_W_v     = model.fg_v114_commit_W_v
    commit_W_o     = model.fg_v114_commit_W_o
    commit_b_o     = model.fg_v114_commit_b_o
    commit_W_write = model.fg_v114_commit_W_write
    commit_b_write = model.fg_v114_commit_b_write

    prop_W_q     = model.fg_v114_prop_W_q
    prop_W_k     = model.fg_v114_prop_W_k
    prop_W_v     = model.fg_v114_prop_W_v
    prop_W_o     = model.fg_v114_prop_W_o
    prop_b_o     = model.fg_v114_prop_b_o
    prop_W_write = model.fg_v114_prop_W_write
    prop_b_write = model.fg_v114_prop_b_write

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

    # Dual notebooks: two separate slot lists
    commit_slots: list[Tensor] = []
    prop_slots:   list[Tensor] = []

    for k in range(K):
        # is_collapse: even breath where waist is ON (commit phase)
        # is_expand:   odd breath where waist is OFF (propagate phase)
        is_collapse = (k % 2 == 0)

        # READ both notebooks (each handles empty slots correctly)
        x_commit_delta = _commit_notebook_read(
            x, commit_slots,
            commit_W_q, commit_W_k, commit_W_v, commit_W_o, commit_b_o,
        )
        x_prop_delta = _prop_notebook_read(
            x, prop_slots,
            prop_W_q, prop_W_k, prop_W_v, prop_W_o, prop_b_o,
        )
        x = x + x_commit_delta + x_prop_delta

        # SBP-composable noise injection on residual stream
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

        # IB codebook
        cb  = semantic_codebook.cast(h.dtype)
        tmp = temperature.cast(h.dtype)
        scores   = h @ cb.T / tmp.reshape(1, 1, 1)
        weights  = scores.clip(-1e4, 1e4).softmax(-1)
        recon    = weights @ cb
        quantize = recon - h
        gate_quant_k = delta_gate_quant[k].cast(h.dtype).reshape(1, 1, 1)
        h_quant = h + gate_quant_k * quantize

        # Waist alternation (binary + photon blend) — same as v110-step3
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

        # delta_gate blend
        gate_k = delta_gate[k].cast(h_quant.dtype).reshape(1, 1, 1)
        delta  = h_quant - x_pre
        step   = gate_k * delta
        x      = x_pre + step

        # Per-breath step magnitude tracking
        step_mag_k = step.cast(dtypes.float).square().mean()
        step_mags_history.append(step_mag_k)

        # WRITE conditionally based on phase (Python-side static, JIT-safe)
        if is_collapse:
            commit_slot_k = _commit_notebook_write(x, commit_W_write, commit_b_write)
            commit_slots.append(commit_slot_k)
        else:
            prop_slot_k = _prop_notebook_write(x, prop_W_write, prop_b_write)
            prop_slots.append(prop_slot_k)

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

    return (tree_logits_history, var_logits_history, factor_logits_history,
            calib_history, step_mags_history)


# ---------------------------------------------------------------------------
# JIT step/eval — mirror v110-step3 structure
# ---------------------------------------------------------------------------

_JIT_V112_CACHE: dict = {}


def _compile_jit_fg_step_v112(
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
    """Train step JIT for v112 — mirrors v110-step3 loss structure."""
    key = ("v112", id(model), id(opt), int(K), int(B),
           float(factor_aux_weight), float(calib_weight), float(var_loss_weight),
           float(balance_weight), float(uncertainty_min),
           bool(hard_breath_level), bool(alternation), float(phase_scale),
           int(n_max), int(f_max), int(n_digits),
           str(gate_profile), float(photon_alpha), float(grad_clip))
    if key in _JIT_V112_CACHE:
        return _JIT_V112_CACHE[key]

    fw, aw, vw, bw, um, gc = (float(factor_aux_weight), float(calib_weight),
                              float(var_loss_weight), float(balance_weight),
                              float(uncertainty_min), float(grad_clip))
    params = opt.params
    print(
        f"[JIT] compile v112 step: K={K} B={B} "
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

        tree_lh, _, fac_lh, calib_h, step_mh = fg_breathing_forward_v112(
            model, domain_init, node_kinds, staging_mask, head_op_mask,
            noise, noise_scale,
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

        # Calibration-driven Goldilocks step balance loss
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

    _JIT_V112_CACHE[key] = _step
    print(f"[JIT] v112 step ready (cache={len(_JIT_V112_CACHE)})", flush=True)
    return _step


def compile_jit_mcbp_v112(
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
    """MC-BP eval JIT with noise + noise_scale inputs. Returns final tree_logits."""
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
        tree_lh, _, _, _, _ = fg_breathing_forward_v112(
            model, domain_init, node_kinds, staging_mask, head_op_mask,
            noise, noise_scale,
            K=K, n_max=n_max, f_max=f_max, n_digits=n_digits,
            alternation=alternation, phase_scale=phase_scale,
            gate_profile=gate_profile, photon_alpha=photon_alpha,
        )
        return tree_lh[-1].realize()
    return _eval


def compile_jit_eval_v112(
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

    Noise is baked-in as zeros (deterministic eval). Returns (pred_digits, cell_acc).
    """
    T = n_max + f_max
    H = 1024
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
        tree_lh, _, _, _, _ = fg_breathing_forward_v112(
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
