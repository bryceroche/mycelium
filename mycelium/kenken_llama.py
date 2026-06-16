"""KenKen Llama-backbone variant — v98 mechanism, bigger/shallower backbone.

A CAPACITY/BACKBONE swap of the WORKING v98 KenKen architecture (mycelium/kenken.py).
The v98 MECHANISM is preserved EXACTLY — cells live in the residual stream (49 cells
= tokens), the custom row/col/cage STRUCTURED ATTENTION MASK, the K-breath iterative-
prefill loop, per-breath additive marker, verification inlet, per-breath delta_gate,
the value-domain-masked codebook readout, the calibration head, the per-breath weighted
CE ladder, the convergence instrument (Property 2), and the Property-2 emit. ONLY the
backbone changes:

  FROM (mycelium/kenken.py): 4 SHARED Pythia-410M L0-L3 layers, hidden=1024, 16 heads.
  TO   (this file):          2 SHARED Llama-2048 (SmolLM2-1.7B) layers, hidden=2048,
                             32 heads, PLUS a 512-dim WAIST per breath.

THIS IS NOT THE PERCEIVER. Cells stay PRIMARY in the residual stream; there is NO
latent state, NO cross-attention, NO READ/THINK/WRITE. The perceiver (kenken_v300*)
is dead — this module does not import or touch it. The only thing borrowed from the
v200/v300 stack is the Llama LAYER + loader (mycelium/llama_loader.py).

WHAT IS REUSED FROM mycelium/kenken.py UNCHANGED (all hidden-/backbone-agnostic):
  - build_kenken_attn_bias        — the (B, n_heads, 49, 49) row/col/cage/global bias;
                                     it adapts the head-group split to the head COUNT
                                     (5/5/5/1 ratio over n_heads) via _build_kenken_fixed_masks,
                                     so at n_heads=32 it produces a 10/10/10/2 split and
                                     the SAME structured topology — this plugs straight
                                     into LlamaLayer's attn_mask param.
  - build_verification_inlet      — per-cell op/target/size inlet, RMSNorm'd, LIVE (§1A.E.8)
  - embed_kenken                  — state + position embeds (built at H=2048 here)
  - convergence_instrument        — Property 2 artifact (gold-free min-based + JSD-floor)
  - kenken_loss / per_breath_ce / kenken_accuracy / kenken_constraint_energy

WHAT IS NEW HERE:
  - kenken_llama_forward          — the breath loop running 2 shared Llama-2048 layers
                                     (with the structured mask + RoPE) + the waist.
  - attach_kenken_llama_params    — allocates the KenKen heads at H=2048 (via the same
                                     attach_kenken_params) + the 512-d waist params.
  - the 2048->512->2048 WAIST     — env-selectable SOFT (default, v38 B-field additive,
                                     W_up zero-init → identity at step 0) / HARD (no-skip
                                     replace, behind KENKEN_WAIST_MODE=hard but NOT default).

SUBSTRATE LAWS (new backbone → applied):
  - DETACHED-scale RMSNorm at the waist seam (the divisor is a constant w.r.t. autograd,
    so pre-norm amplitude is not a flat loss direction — the #237 trapdoor fix, §1A.E.14).
  - fp16 activation carry, weights fp32 (the validated v98 recipe; KEPT — the Llama
    layers are bigger so watch the norm at GPU-smoke time, but no fp32-carry by default).
  - scores.clip(-1e4, 1e4) inside attention (already in LlamaLayer.__call__).
  - no dtypes.float32 literal in any JIT body (the trainer's where()-gated NaN guard
    + the forward use Tensor.zeros((), dtype=dtypes.float) only — fine).

Env vars (additive to the kenken.py set):
  KENKEN_BACKBONE=llama        — selects this forward in the trainer (pythia is default)
  KENKEN_N_LLAMA_LAYERS=2      — shared Llama layers per breath
  KENKEN_WAIST_DIM=512         — bottleneck width
  KENKEN_WAIST_MODE=soft       — soft (default, additive B-field) / hard (no-skip replace)
"""
from __future__ import annotations

import math
import os
from typing import Any

import numpy as np
from tinygrad import Tensor, dtypes

from mycelium.kenken_data import N_MAX, N_CELLS
from mycelium.kenken import (
    embed_kenken,
    build_kenken_attn_bias,
    build_verification_inlet,
    attach_kenken_params,
    kenken_parameters,
    kenken_state_dict,
)
from mycelium.llama_loader import _rms_norm


# ---- env knobs (this backbone only) -----------------------------------------

KENKEN_N_LLAMA_LAYERS = int(os.environ.get("KENKEN_N_LLAMA_LAYERS", "2"))
KENKEN_WAIST_DIM      = int(os.environ.get("KENKEN_WAIST_DIM", "512"))
KENKEN_WAIST_MODE     = os.environ.get("KENKEN_WAIST_MODE", "soft").strip().lower()


# ---- detached-scale RMSNorm (the waist-seam substrate law, §1A.E.14) ----------

def _rms_norm_detached(x: Tensor, weight: Tensor, eps: float = 1e-5) -> Tensor:
    """RMSNorm with a DETACHED divisor (mirror factor_graph_v200._rms_norm_detached).

    Forward IDENTICAL to llama_loader._rms_norm. Backward difference: the divisor is
    a constant w.r.t. autograd, so pre-norm amplitude is no longer a flat direction
    of the loss (standard RMSNorm has ∂out/∂scale = 0 — the #237 trapdoor: amplitude
    free to grow, organ starved to fp16 zero). Applied at the waist seam only; the
    Llama internals keep standard RMSNorm (bound the seam, not the organ).
    """
    rms = (x.float().pow(2).mean(axis=-1, keepdim=True) + eps).sqrt().detach()
    return (x.float() / rms * weight.float()).cast(x.dtype)


# ---- the 2048 -> 512 -> 2048 WAIST -------------------------------------------

def _apply_waist(model: Any, x: Tensor, eps: float) -> Tensor:
    """Apply the per-breath bottleneck waist to the (B, 49, H) residual stream.

    SOFT (default, KENKEN_WAIST_MODE=soft) — v38 B-field additive style:
        z   = GELU( rmsnorm_det(x) @ W_down )          # (B, 49, waist_dim)
        out = x + (z @ W_up) * post_norm_w             # W_up ZERO-INIT → out == x at step 0
      The skip (`x +`) means the residual stream passes through UNCHANGED at init, and
      the waist's gradient is LIVE from step 1 because W_down is QR-init (non-zero), so
      ∂L/∂W_up = z.T @ (upstream) ≠ 0. IB-regularization without the collapse risk.

    HARD (KENKEN_WAIST_MODE=hard) — the v105/perceiver-scarred no-skip REPLACE:
        out = GELU( rmsnorm_det(x) @ W_down ) @ W_up   # forces everything through 512d
      Available behind the knob (NOT default) — it is the collapse-prone codec form.
    """
    W_down     = model.kenken_waist_w_down       # (H, waist_dim)
    W_up       = model.kenken_waist_w_up         # (waist_dim, H)  ZERO-INIT (soft)
    post_norm_w = model.kenken_waist_post_w      # (H,) scalar-per-channel post scale

    x_n = _rms_norm_detached(x, model.kenken_waist_seam_w, eps)        # detached-scale seam
    z = (x_n @ W_down.cast(x.dtype)).gelu()                            # (B, 49, waist_dim)
    up = z @ W_up.cast(x.dtype)                                        # (B, 49, H)

    if model.kenken_waist_mode == "hard":
        # no-skip REPLACE — forces the residual through the 512d bottleneck.
        return up * post_norm_w.cast(x.dtype)
    # SOFT additive (default): identity at step 0 (W_up zero-init), residual passes through.
    return x + up * post_norm_w.cast(x.dtype)


# ---- one Llama-layer forward over the 49-cell residual stream ----------------

def _kenken_llama_layer(layer: Any, x: Tensor, rope_cos: Tensor, rope_sin: Tensor,
                        attn_bias: Tensor) -> Tensor:
    """Run ONE shared LlamaLayer over (B, 49, H) with the structured mask.

    layer:     a mycelium.llama_loader.LlamaLayer (SmolLM2-1.7B weights, 2048d).
    x:         (B, 49, H) residual stream (cells are the sequence positions).
    attn_bias: (B, n_heads, 49, 49) PER-BATCH additive bias (0 allow / -1e4 block),
               from build_kenken_attn_bias — the row/col/cage/global topology.

    LlamaLayer.__call__ already: pre-RMSNorm → q/k/v → RoPE → scores + attn_mask →
    scores.clip(-1e4,1e4).softmax → o_proj → residual → pre-RMSNorm → SwiGLU → residual.
    The structured mask plugs straight into its attn_mask param (same (B,nh,S,S) shape).
    """
    return layer(x, rope_cos, rope_sin, attn_mask=attn_bias)


# ---- iterative prefill loop (Llama backbone + waist) -------------------------

def kenken_llama_forward(model: Any, batch: Any, K: int, stoch_keep=None):
    """Run K breaths of constraint propagation on a KenKen batch (Llama backbone).

    EXACT mirror of mycelium.kenken.kenken_breathing_forward — same per-breath
    structure, same masks, same ladder-feeding history shapes — with TWO backbone
    deltas: (1) the inner stack is N shared Llama-2048 layers (default 2) with RoPE
    + the structured mask instead of 4 Pythia layers; (2) a 512-d waist is applied
    after the layers, before the delta_gate convex blend.

    Per-breath structure:
      1. Add per-breath additive marker (model.kenken_breath_embed[k]).
      2. Add the verification inlet (RMSNorm'd, LIVE at init — not gated; §1A.E.8).
      3. Run N shared Llama-2048 layers with the per-batch structured attn_bias + RoPE.
      4. Apply the 512-d waist (soft additive default / hard replace behind the knob).
      5. Learnable per-breath delta_gate convex residual blend (+ optional stoch depth).
      6. Readout: cell logits (value-domain masked) via the 2048d codebook + calibration.

    stoch_keep: OPTIONAL (K,) per-breath keep-SCALE Tensor (train-only stochastic
                depth; identical semantics to kenken_breathing_forward). None => full.

    Returns:
      cell_logits_history: list of K (B, 49, N_MAX) float Tensors (value-domain masked).
      calib_history:       list of K (B,) float Tensors (sigmoid'd).
    """
    assert hasattr(model, "kenken_state_embed"), \
        "model has no kenken params; was attach_kenken_llama_params called?"
    assert hasattr(model, "llama_layers"), \
        "model has no llama_layers; was attach_llama_layers called?"

    state_embed    = model.kenken_state_embed      # (8, H)
    position_embed = model.kenken_position_embed    # (49, H)
    breath_embed   = model.kenken_breath_embed      # (K_max, H)
    delta_gate     = model.kenken_delta_gate        # (K_max,)
    value_codebook = model.kenken_value_codebook    # (N_MAX, H)
    calib_head_w   = model.kenken_calib_head_w      # (H, 1)
    calib_head_b   = model.kenken_calib_head_b      # (1,)

    rope_cos = model.llama_rope_cos
    rope_sin = model.llama_rope_sin
    eps = float(model.llama_cfg.rms_norm_eps)

    input_cells       = batch.input_cells           # (B, 49) int
    cage_mask         = batch.cage_mask             # (B, 49, 49)
    cell_valid        = batch.cell_valid            # (B, 49)
    value_domain_mask = batch.value_domain_mask     # (B, 49, N_MAX)

    # Per-batch structured attention bias (row/col fixed + cage per-instance + validity).
    # build_kenken_attn_bias reads model.kenken_fixed_mask (n_heads=32 here) + head_split,
    # producing (B, n_heads, 49, 49) — exactly the attn_mask LlamaLayer expects.
    attn_bias = build_kenken_attn_bias(model, cage_mask, cell_valid)         # (B,nh,49,49)

    # Verification inlet (per-cell, RMSNorm'd, LIVE). Built once; added each breath.
    inlet = build_verification_inlet(
        model, batch.cage_op, batch.cage_target, batch.cage_size, batch.cell_cage_id,
    )                                                                        # (B, 49, H)

    # Value-domain bias for readout: illegal values get -1e4 added to their logit.
    value_bias = (1.0 - value_domain_mask) * (-1e4)                          # (B, 49, N_MAX)

    x = embed_kenken(input_cells, state_embed, position_embed)               # (B, 49, H)
    x = x.cast(dtypes.half) if x.dtype != dtypes.half else x                 # fp16 carry

    llama_layers = list(model.llama_layers)
    n_layers = len(llama_layers)
    assert n_layers >= 1, f"expected >=1 Llama layer; got {n_layers}"
    K_max = int(breath_embed.shape[0])
    assert K <= K_max, f"K={K} exceeds K_max={K_max} for kenken_breath_embed"

    inlet_h = inlet.cast(x.dtype)
    B0 = int(cell_valid.shape[0])
    cell_valid_col = cell_valid.reshape(B0, N_CELLS, 1)

    cell_logits_history = []
    calib_history = []

    for k in range(K):
        be_k = breath_embed[k].reshape(1, 1, -1).cast(x.dtype)              # (1,1,H)
        x_in = x + be_k + inlet_h                                           # add LIVE inlet

        x_pre = x
        h = x_in
        for layer in llama_layers:
            h = _kenken_llama_layer(layer, h, rope_cos, rope_sin, attn_bias)

        # 512-d waist (soft additive default / hard replace behind KENKEN_WAIST_MODE).
        h = _apply_waist(model, h, eps)

        gate_k = delta_gate[k].cast(h.dtype).reshape(1, 1, 1)
        delta = h - x_pre
        if stoch_keep is not None:
            keep_k = stoch_keep[k].cast(h.dtype).reshape(1, 1, 1)
            x = x_pre + (gate_k * keep_k) * delta
        else:
            x = x_pre + gate_k * delta

        # Readout: RMSNorm (Llama-native) then project each cell to an N_MAX-way logit.
        x_ln = _rms_norm(x, model.kenken_readout_norm_w, eps).cast(dtypes.float)
        cell_logits_k = x_ln @ value_codebook.T.cast(dtypes.float)          # (B, 49, N_MAX)
        cell_logits_k = cell_logits_k + value_bias.cast(dtypes.float)       # value-domain mask
        cell_logits_history.append(cell_logits_k)

        # Calibration: mean-pool over VALID cells only.
        pool_num = (x_ln * cell_valid_col.cast(dtypes.float)).sum(axis=1)   # (B, H)
        pool_den = cell_valid_col.cast(dtypes.float).sum(axis=1) + 1e-6     # (B, 1)
        pool = pool_num / pool_den
        calib_logit = pool @ calib_head_w.cast(dtypes.float) + calib_head_b.cast(dtypes.float)
        calib_k = calib_logit.reshape(-1).sigmoid()
        calib_history.append(calib_k)

    return cell_logits_history, calib_history


# ---- param attach (KenKen heads at H=2048 + the waist) -----------------------

def attach_kenken_llama_params(model: Any, hidden: int, n_heads: int,
                               k_max: int | None = None,
                               waist_dim: int | None = None,
                               waist_mode: str | None = None) -> None:
    """Allocate KenKen params on `model` for the Llama backbone.

    Reuses attach_kenken_params (the SAME state_embed / position_embed / value_codebook
    / calib / breath_embed / delta_gate / fixed_mask / verification-inlet allocator) at
    hidden=2048, n_heads=32 — so the structured-mask head-group split adapts to 32 heads
    (5/5/5/1 ratio → 10/10/10/2). Then attaches:

      kenken_waist_w_down   (H, waist_dim)   — QR-init scale 0.01 (so W_up grad is LIVE)
      kenken_waist_w_up     (waist_dim, H)   — ZERO-INIT (soft: identity at step 0)
      kenken_waist_post_w   (H,)             — post-up per-channel scale (ones init)
      kenken_waist_seam_w   (H,)             — detached-scale RMSNorm weight (ones init)
      kenken_readout_norm_w (H,)             — Llama-native RMSNorm before codebook (ones)
      kenken_waist_mode     str              — "soft" (default) | "hard"
    """
    if k_max is None:
        k_max = int(os.environ.get("KENKEN_K_MAX", "20"))
    if waist_dim is None:
        waist_dim = KENKEN_WAIST_DIM
    if waist_mode is None:
        waist_mode = KENKEN_WAIST_MODE
    waist_mode = str(waist_mode).strip().lower()
    assert waist_mode in ("soft", "hard"), f"KENKEN_WAIST_MODE must be soft|hard, got {waist_mode}"

    # All the v98 KenKen heads at H=2048 / n_heads=32 (mask head-split adapts to 32).
    attach_kenken_params(model, hidden=hidden, n_heads=n_heads, k_max=k_max)

    # ---- the 512-d waist ----
    rng_w = np.random.RandomState(99113)
    # W_down: QR-init scale 0.01 → z is non-zero, so ∂L/∂W_up ≠ 0 at step 1 (LIVE grad).
    wd_raw = rng_w.randn(hidden, waist_dim).astype(np.float32)
    Q_wd, _ = np.linalg.qr(wd_raw if hidden >= waist_dim else wd_raw.T)
    if hidden >= waist_dim:
        wd_init = (Q_wd[:, :waist_dim] * 0.01).astype(np.float32)
    else:
        wd_init = (Q_wd[:waist_dim, :].T * 0.01).astype(np.float32)
    model.kenken_waist_w_down = Tensor(wd_init, dtype=dtypes.float).contiguous()
    # W_up: ZERO-INIT → soft waist is identity at step 0 (residual passes through).
    model.kenken_waist_w_up = Tensor(
        np.zeros((waist_dim, hidden), dtype=np.float32), dtype=dtypes.float).contiguous()
    # Post-up per-channel scale (ones init) + detached-scale seam RMSNorm + readout norm.
    model.kenken_waist_post_w   = Tensor.ones((hidden,), dtype=dtypes.float).contiguous()
    model.kenken_waist_seam_w   = Tensor.ones((hidden,), dtype=dtypes.float).contiguous()
    model.kenken_readout_norm_w = Tensor.ones((hidden,), dtype=dtypes.float).contiguous()
    model.kenken_waist_mode = waist_mode
    model.kenken_waist_dim = int(waist_dim)


def kenken_llama_waist_parameters(model: Any) -> list[Tensor]:
    """Just the waist + readout-norm params (the new trainable tensors for this backbone)."""
    return [
        model.kenken_waist_w_down,
        model.kenken_waist_w_up,
        model.kenken_waist_post_w,
        model.kenken_waist_seam_w,
        model.kenken_readout_norm_w,
    ]


def kenken_llama_parameters(model: Any) -> list[Tensor]:
    """Trainable KenKen-specific params for the Llama backbone: the v98 heads (at 2048d,
    minus the frozen fixed mask) + the waist. The Llama LAYER params are collected
    separately in the trainer (they are the shared backbone, like the Pythia path)."""
    return kenken_parameters(model) + kenken_llama_waist_parameters(model)


def kenken_llama_state_dict(model: Any) -> dict[str, Tensor]:
    """State-dict entries for the Llama-backbone KenKen params (v98 heads + waist).

    The Llama LAYER weights are saved by the trainer's backbone-aware state dict
    (mirrors how the Pythia path saves shared/phase weights)."""
    sd = dict(kenken_state_dict(model))   # v98 heads at 2048d
    sd.update({
        "kenken.waist_w_down":   model.kenken_waist_w_down,
        "kenken.waist_w_up":     model.kenken_waist_w_up,
        "kenken.waist_post_w":   model.kenken_waist_post_w,
        "kenken.waist_seam_w":   model.kenken_waist_seam_w,
        "kenken.readout_norm_w": model.kenken_readout_norm_w,
    })
    return sd
