"""v121: Perceiver-as-mechanistic-observer — energy-selected notebook writes.

v118-v120 all FAILED by adding new cross-token attention channels that the
model rejected (4 attempts). v121 is structurally different on three axes:

1. READS INTERMEDIATE LAYERS, NOT FINAL STATE.
   During each breath, Pythia processes L0 → L1 → L2 → L3. The intermediate
   states (L0_out, L1_out, L2_out) are DISCARDED after each breath — only
   the final residual survives. v121 captures these intermediate states.
   This is GENUINELY NEW information that nothing else preserves.

2. ENERGY-BASED SELECTION.
   Per-position, per-layer energy = ||h_l - h_{l-1}||. High energy positions
   = "significant computation happened here." Soft selection via
   temperature-scaled softmax (differentiable, no hard topk needed in JIT).
   This is principled compression, not learned latent queries.

3. WRITES THROUGH THE EXISTING ACCUMULATE NOTEBOOK (validated channel).
   No new cross-attention back to residual. The notebook read path is
   unchanged. The perceiver just makes the notebook WRITE step smarter:
   selecting high-energy slices instead of pooling the full residual.

Parameters added:
  fg_v121_W_select      (H, H)   ZERO-INIT — projects energy-weighted slice
  fg_v121_energy_temp   (1,)     init 1.0  — softmax sharpness scalar

Total new params: H*H + 1 = 1,048,577 (~1M) at H=1024.

Byte-identity at step 0: W_select=0 → v121_write_delta=0 →
write_source unchanged → forward identical to v112b.

Gradient analysis:
  ∂L/∂W_select = selected.T @ ∂L/∂write_source
  selected = energy_weights @ h_after_L3    (non-zero at init)
  ∂L/∂write_source flows through notebook → future breaths → loss (non-zero)
  → ∂L/∂W_select is non-zero from step 1. W_select can grow immediately.

AMD/JIT safety:
  - .cast(dtypes.float) not dtypes.float32
  - .clip(-1e4, 1e4) before softmax
  - All v121 tensors .contiguous().realize()
  - No new cross-attention pathways to residual (the v118-v120 failure mode)
  - No Llama base (Pythia only)
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
from mycelium.factor_graph_v112b import (
    V112B_TOPOLOGY_DIM, V112B_BIAS_SCALE_INIT,
    attach_fg_params_v112b,
    fg_v112b_parameters, fg_v112b_state_dict,
)


# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

V121_ENERGY_TEMP_INIT = float(os.environ.get("V121_ENERGY_TEMP_INIT", "1.0"))


# ---------------------------------------------------------------------------
# Parameter attachment
# ---------------------------------------------------------------------------

def attach_fg_params_v121(
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
    """Attach v112b base params + 2 new v121 energy-selection params."""
    attach_fg_params_v112b(
        model, hidden=hidden,
        n_max=n_max, f_max=f_max, k_max=k_max,
        n_digits=n_digits, n_code=n_code, ib_centroids_path=ib_centroids_path,
        waist_dim=waist_dim,
    )

    if hasattr(model, "fg_v121_W_select"):
        return  # already attached (e.g. resuming from v121 ckpt)

    H = hidden

    # W_select: (H, H) ZERO-INIT — zero at init → v121_write_delta=0 → byte-identical
    W_select_np = np.zeros((H, H), dtype=np.float32)
    model.fg_v121_W_select = Tensor(
        W_select_np
    ).contiguous().realize()

    # energy_temp: scalar init 1.0 (shape (1,) for Tensor compat)
    # Controls sharpness of softmax over positions. No sigmoid — raw multiplier.
    energy_temp_np = np.array([V121_ENERGY_TEMP_INIT], dtype=np.float32)
    model.fg_v121_energy_temp = Tensor(
        energy_temp_np
    ).contiguous().realize()

    n_new = H * H + 1
    print(
        f"[v121] energy-selection params attached: H={H}  "
        f"energy_temp_init={V121_ENERGY_TEMP_INIT}  "
        f"(W_select=zero → byte-identical warm-start)  "
        f"+{n_new:,} params",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Parameter collection + state dict
# ---------------------------------------------------------------------------

def fg_v121_parameters(model: Any) -> list[Tensor]:
    """v112b params + 2 new v121 params."""
    params = list(fg_v112b_parameters(model))
    params.extend([
        model.fg_v121_W_select,
        model.fg_v121_energy_temp,
    ])
    return params


def fg_v121_state_dict(model: Any) -> dict[str, Tensor]:
    """v112b state_dict + 2 new v121 entries."""
    sd = dict(fg_v112b_state_dict(model))
    sd["fg_v121_W_select"]     = model.fg_v121_W_select
    sd["fg_v121_energy_temp"]  = model.fg_v121_energy_temp
    return sd


# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------

def fg_breathing_forward_v121(
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
    """v112b forward + energy-selected perceiver writes to accumulate notebook.

    Returns (tree_logits_history, var_logits_history, factor_logits_history,
             calib_history, step_mags_history) — same shape as v112b.

    With W_select=0, output is byte-identical to v112b.

    The perceiver mechanism:
      1. Capture h after each of the 4 Pythia layers (L0..L3).
      2. Compute per-position energy = sum of per-layer ||delta_h||.
      3. Soft-select via temperature-scaled softmax.
      4. Energy-weighted average of final residual → project via W_select.
      5. ADD to the existing acc-notebook write_source (augments, not replaces).
    """
    # === Load all params from model =========================================
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

    # v112b topology params
    node_topology   = model.fg_v115_node_topology
    W_res_gate      = model.fg_v115_W_res_gate
    attn_bias_scale = model.fg_v115_attn_bias_scale

    # v121 energy-selection params
    W_select     = model.fg_v121_W_select      # (H, H) zero-init
    energy_temp  = model.fg_v121_energy_temp   # (1,) scalar

    # === Shapes ==============================================================
    H = int(tree_codebook.shape[-1])
    B = int(domain_init.shape[0])
    T = n_max + f_max

    # === Embedding ===========================================================
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

    # === v112b precomputations (outside K loop) ==============================
    topology_f = node_topology.cast(dtypes.float)                         # (T, latent)
    attn_bias_full = (topology_f @ topology_f.T) * attn_bias_scale.reshape(1, 1).cast(dtypes.float)
    attn_bias_btht = attn_bias_full.reshape(1, 1, T, T)                    # (1,1,T,T) broadcasts

    gate_per_pos     = (topology_f @ W_res_gate.cast(dtypes.float)).tanh() # (T, H)
    gate_multiplier_3d = (1.0 + gate_per_pos).reshape(1, T, H)             # (1, T, H)
    # =========================================================================

    # History lists
    tree_logits_history   = []
    var_logits_history    = []
    factor_logits_history = []
    calib_history         = []
    step_mags_history     = []

    tree_cb_flat = tree_codebook.reshape(n_digits * 10, H)

    # ACCUMULATE notebook: list of (B, H) tensors, one per breath
    notebook_slots: list[Tensor] = []

    # Pre-cast W_select + energy_temp once (outside loop for efficiency)
    W_select_f   = W_select.cast(dtypes.float)    # (H, H)
    energy_temp_f = energy_temp.cast(dtypes.float) # (1,)

    for k in range(K):
        # Read from prior notebook slots (causal: only slots 0..k-1)
        x_acc_delta = _acc_notebook_read(
            x, notebook_slots,
            acc_W_q, acc_W_k, acc_W_v, acc_W_o, acc_b_o,
        )
        x = x + x_acc_delta

        # SBP noise injection
        x = x + (noise[k] * noise_scale).cast(x.dtype)

        # v112b: per-position residual gate (zero-init → multiplier=1 at start)
        x = x * gate_multiplier_3d.cast(x.dtype)

        be_k  = breath_embed[k].reshape(1, 1, -1).cast(x.dtype)
        x_in  = x + be_k
        x_pre = x

        stk      = staging_mask[:, k, :, :]
        stk_h    = stk.reshape(B, 1, T, T).expand(B, V110_STEP_N_HEADS, T, T)
        combined = stk_h.cast(x.dtype) + head_op_mask.cast(x.dtype)
        # v112b: add learnable attn bias (zero-init scale → no-op at start)
        combined = combined + attn_bias_btht.cast(combined.dtype)

        cos_k = breath_cos[k]
        sin_k = breath_sin[k]

        # === v121 core: capture intermediate states through all 4 layers =====
        h_in = x_in   # (B, T, H) — input to L0

        h_after_L0 = fg_layer_forward_v109pi(layers[0], h_in,      combined, cos_k, sin_k)
        h_after_L1 = fg_layer_forward_v109pi(layers[1], h_after_L0, combined, cos_k, sin_k)
        h_after_L2 = fg_layer_forward_v109pi(layers[2], h_after_L1, combined, cos_k, sin_k)
        h_after_L3 = fg_layer_forward_v109pi(layers[3], h_after_L2, combined, cos_k, sin_k)

        # Per-position, per-layer energy (L2 norm of layer-induced change)
        # (B, T) — each is the magnitude of the change this layer made
        e0 = (h_after_L0 - h_in     ).cast(dtypes.float).square().sum(-1).sqrt()  # (B, T)
        e1 = (h_after_L1 - h_after_L0).cast(dtypes.float).square().sum(-1).sqrt() # (B, T)
        e2 = (h_after_L2 - h_after_L1).cast(dtypes.float).square().sum(-1).sqrt() # (B, T)
        e3 = (h_after_L3 - h_after_L2).cast(dtypes.float).square().sum(-1).sqrt() # (B, T)

        # Total per-position energy (B, T)
        total_energy = e0 + e1 + e2 + e3

        # Soft selection via temperature-scaled softmax (differentiable)
        energy_logits  = total_energy * energy_temp_f.reshape(1, 1)                # (B, T)
        energy_weights = energy_logits.clip(-1e4, 1e4).softmax(-1)                 # (B, T)

        # Energy-weighted average of the final residual at high-energy positions
        # selected: (B, H) — energy-weighted sum over positions
        selected = (energy_weights.unsqueeze(-1).cast(h_after_L3.dtype) * h_after_L3).sum(1)  # (B, H)

        # Project through zero-init W_select → delta is zero at init
        v121_write_delta = selected.cast(dtypes.float) @ W_select_f  # (B, H)
        # =====================================================================

        h = h_after_L3  # final output of the 4 Pythia layers

        # IB codebook quantization
        cb  = semantic_codebook.cast(h.dtype)
        tmp = temperature.cast(h.dtype)
        scores   = h @ cb.T / tmp.reshape(1, 1, 1)
        weights  = scores.clip(-1e4, 1e4).softmax(-1)
        recon    = weights @ cb
        quantize = recon - h
        gate_quant_k = delta_gate_quant[k].cast(h.dtype).reshape(1, 1, 1)
        h_quant = h + gate_quant_k * quantize

        # Waist (photon alternation)
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

        # delta_gate residual blend
        gate_k = delta_gate[k].cast(h_quant.dtype).reshape(1, 1, 1)
        delta  = h_quant - x_pre
        step   = gate_k * delta
        x      = x_pre + step

        step_mag_k = step.cast(dtypes.float).square().mean()
        step_mags_history.append(step_mag_k)

        # === v121 augmented notebook write ====================================
        # Original v110-acc write source: pool(x) @ W_write + b_write
        # v121 augment: add the energy-selected delta BEFORE the write projection.
        # Because W_select=0, v121_write_delta=0 at step 0 → byte-identical.
        #
        # Implementation: we compute the standard write slot, then add the
        # projected v121 delta directly to the slot vector.
        # This is equivalent to:
        #   write_source = pool(x) + v121_write_delta_projected_back
        # but avoids the intermediate: just add after the full write computation.
        slot_k = _acc_notebook_write(x, acc_W_write, acc_b_write)  # (B, H)
        # Augment slot_k with v121 energy-selected signal (zero at step 0)
        slot_k = slot_k + v121_write_delta.cast(slot_k.dtype)       # (B, H) augmented
        # =====================================================================

        notebook_slots.append(slot_k)

        # Readout
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


# ---------------------------------------------------------------------------
# JIT compile helpers
# ---------------------------------------------------------------------------

_JIT_V121_CACHE: dict = {}


def _compile_jit_fg_step_v121(
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
    """Train step for v121. Same loss as v112b, with v121 energy-perceiver forward."""
    key = ("v121", id(model), id(opt), int(K), int(B),
           float(factor_aux_weight), float(calib_weight), float(var_loss_weight),
           float(balance_weight), float(uncertainty_min),
           bool(hard_breath_level), bool(alternation), float(phase_scale),
           int(n_max), int(f_max), int(n_digits),
           str(gate_profile), float(photon_alpha), float(grad_clip))
    if key in _JIT_V121_CACHE:
        return _JIT_V121_CACHE[key]

    fw, aw, vw, bw, um, gc = (float(factor_aux_weight), float(calib_weight),
                              float(var_loss_weight), float(balance_weight),
                              float(uncertainty_min), float(grad_clip))
    params = opt.params
    print(
        f"[JIT] compile v121 step: K={K} B={B} "
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
            fg_breathing_forward_v121(
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

    _JIT_V121_CACHE[key] = _step
    print(f"[JIT] v121 step ready (cache={len(_JIT_V121_CACHE)})", flush=True)
    return _step


def compile_jit_eval_v121(
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
    """JIT'd eval forward — noise baked-in as zeros (deterministic).

    Returns (pred_digits, cell_acc). Matches evaluate_v109 / evaluate_v110_acc
    call signature.
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
        tree_lh, _, _, _, _ = fg_breathing_forward_v121(
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
