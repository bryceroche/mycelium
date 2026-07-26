"""v110-photon: v109pi with smooth cos² waist gate (B-field) phase-locked to Q sin rotation (E-field).

v109pi waist gating: alternating binary [1, 0, 1, 0, 1, 0, 1, 0]
v110-photon gating:  smooth cos²(k · π / K_max - φ) profile, learned alpha

The "photon" framing: per-breath Q rotation provides cos_k/sin_k constants
(the "E-field" component). The waist amplitude provides the "B-field"
component. v109pi has discontinuous waist gating; v110-photon makes it
smooth and aligns it phase-wise with the Q rotation.

Three profile options (V110_PHOTON_GATE_PROFILE):
  "binary"     — matches v109pi alternation (CONTROL for byte-safe warm-start)
  "cos2_pi2"   — gate = cos²(k · π/2 / (K_max-1)) — smooth decay from 1 → 0
  "cos2_pi"    — gate = cos²(k · π / (K_max-1))   — full oscillation
  "sin2_pi"    — gate = sin²(k · π / K_max)        — peaks mid-K (B-field mode)
  "alt_smooth" — gate = (1 + cos(k · π)) / 2       — same nodes as binary, smooth

Byte-safe warm-start: PHOTON_ALPHA controls how much we blend toward the
smooth profile. PHOTON_ALPHA=0 → identical to v109pi alternation.
PHOTON_ALPHA=1 → fully smooth gate. Default 0.5 means waist contribution
is half binary half smooth at start, ramped via training.
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
    V109_N_HEADS, V109_WAIST_DIM, V109_ALTERNATION,
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

V110_PHOTON_TASK              = int(os.environ.get("V110_PHOTON_TASK", "0")) > 0
V110_PHOTON_K_MAX             = int(os.environ.get("V110_PHOTON_K_MAX", str(V109PI_K_MAX)))
V110_PHOTON_N_DIGITS          = int(os.environ.get("V110_PHOTON_N_DIGITS", str(V109PI_N_DIGITS)))
V110_PHOTON_WAIST_DIM         = int(os.environ.get("V110_PHOTON_WAIST_DIM", str(V109PI_WAIST_DIM)))
V110_PHOTON_ALTERNATION       = int(os.environ.get("V110_PHOTON_ALTERNATION", "1")) > 0
V110_PHOTON_HARD_BREATH_LEVEL = int(os.environ.get("V110_PHOTON_HARD_BREATH_LEVEL", "0")) > 0
V110_PHOTON_VAR_LOSS_WEIGHT   = float(os.environ.get("V110_PHOTON_VAR_LOSS_WEIGHT", "1.0"))
V110_PHOTON_N_MAX             = int(os.environ.get("V110_PHOTON_N_MAX", str(V109PI_N_MAX)))
V110_PHOTON_F_MAX             = int(os.environ.get("V110_PHOTON_F_MAX", str(V109PI_F_MAX)))
V110_PHOTON_T_MAX             = V110_PHOTON_N_MAX + V110_PHOTON_F_MAX
V110_PHOTON_CODEBOOK_N        = int(os.environ.get("V110_PHOTON_CODEBOOK_N", str(V109PI_CODEBOOK_N)))
V110_PHOTON_IB_CENTROIDS      = os.environ.get("V110_PHOTON_IB_CENTROIDS", V109PI_IB_CENTROIDS)
V110_PHOTON_CALIB_WEIGHT      = float(os.environ.get("V110_PHOTON_CALIB_WEIGHT", str(V108_CALIB_WEIGHT)))
V110_PHOTON_FACTOR_AUX_WEIGHT = float(os.environ.get("V110_PHOTON_FACTOR_AUX_WEIGHT", str(V108_FACTOR_AUX_WEIGHT)))
V110_PHOTON_N_HEADS           = V109PI_N_HEADS
V110_PHOTON_PHASE_SCALE       = float(os.environ.get("V110_PHOTON_PHASE_SCALE", str(V109PI_PHASE_SCALE)))

V110_PHOTON_GATE_PROFILE = os.environ.get("V110_PHOTON_GATE_PROFILE", "alt_smooth")
V110_PHOTON_ALPHA        = float(os.environ.get("V110_PHOTON_ALPHA", "0.5"))


# ---------------------------------------------------------------------------
# Per-breath gate profile (constant per breath — no Tensor ops)
# ---------------------------------------------------------------------------

def _photon_gate(k: int, K_max: int, profile: str) -> float:
    """Compute the waist gate amplitude for breath k under a given profile.

    All profiles return values in [0, 1]. "binary" matches v109pi alternation
    exactly. "alt_smooth" preserves the same peaks at even breaths but
    smooths the transitions (cos profile with period K=2).

    V110_PHOTON_FREQ_MULT (default 1.0) multiplies the sin/cos angle for the
    continuous profiles (cos2_pi2, cos2_pi, sin2_pi). 1.0 = original
    half-cycle over K breaths. 2.0 = full cycle. 4.0 at K=8 aliases to
    {0,1,0,1,...} binary at integer breaths. No-op for "binary" and
    "alt_smooth".
    """
    freq = float(os.environ.get("V110_PHOTON_FREQ_MULT", "1.0"))

    if profile == "binary":
        return 1.0 if k % 2 == 0 else 0.0
    if profile == "cos2_pi2":
        # cos² from 0 to π/2 across K breaths: starts at 1, ends at 0
        denom = max(K_max - 1, 1)
        return math.cos(freq * k * math.pi / 2.0 / denom) ** 2
    if profile == "cos2_pi":
        # cos² from 0 to π across K breaths: starts at 1, ends at 1, dips at K/2
        denom = max(K_max - 1, 1)
        return math.cos(freq * k * math.pi / denom) ** 2
    if profile == "sin2_pi":
        # sin² from 0 to π across K breaths: starts at 0, peaks at K/2, ends at 0
        # B-field 90° offset from cos2_pi (the "E-field")
        return math.sin(freq * k * math.pi / float(K_max)) ** 2
    if profile == "alt_smooth":
        # Smooth alternation: cos²(k·π/2) → 1 at k even, 0 at k odd
        # but smoothed in continuous sense. Identical to "binary" at integer k.
        # So this is a no-op for K integer breaths. Switch to a small offset
        # so adjacent breaths get a smooth interpolation:
        # gate(k) = 0.5 * (1 + cos(k · π))  — exact same as binary at integers
        # To genuinely smooth, use a phase offset:
        return 0.5 * (1.0 + math.cos(k * math.pi - math.pi / 4.0))
    if profile == "piecewise_4phase":
        # v118: 4-phase piecewise envelope for K=8 breaths.
        # Phase 0=EXPLORE(0,1): 0.0, 0.0  — full rank, no compression
        # Phase 1=COMPRESS(2,3): 1.0, 0.8 — heavy → fading compression
        # Phase 2=COMMIT(4,5): 0.5, 0.5   — moderate
        # Phase 3=REFINE(6,7): 0.1, 0.0   — fading → full rank
        # For K != 8 (generalised): divide K into 4 even phases, clamp.
        _PIECEWISE_8 = [0.0, 0.0, 1.0, 0.8, 0.5, 0.5, 0.1, 0.0]
        if K_max == 8 and k < 8:
            return _PIECEWISE_8[k]
        # Generic: divide K_max into 4 equal phases, lookup within phase
        phase_size = max(K_max // 4, 1)
        phase = min(k // phase_size, 3)
        pos_in_phase = k % phase_size
        # Use hard-coded phase-level gate based on position within phase
        _PHASE_GATES = [[0.0, 0.0], [1.0, 0.8], [0.5, 0.5], [0.1, 0.0]]
        gates = _PHASE_GATES[phase]
        if len(gates) == 1 or phase_size == 1:
            return gates[0]
        # Linear interp within phase
        frac = pos_in_phase / max(phase_size - 1, 1)
        return gates[0] + frac * (gates[-1] - gates[0])
    raise ValueError(f"unknown V110_PHOTON_GATE_PROFILE: {profile}")


# ---------------------------------------------------------------------------
# Forward pass — v109pi with photon-gated waist
# ---------------------------------------------------------------------------

def fg_breathing_forward_v110_photon(
    model: Any,
    domain_init: Tensor,
    node_kinds: Tensor,
    staging_mask: Tensor,
    head_op_mask: Tensor,
    K: int,
    n_max: int = V110_PHOTON_N_MAX,
    f_max: int = V110_PHOTON_F_MAX,
    n_digits: int = V110_PHOTON_N_DIGITS,
    alternation: bool = V110_PHOTON_ALTERNATION,
    phase_scale: float = V110_PHOTON_PHASE_SCALE,
    gate_profile: str = V110_PHOTON_GATE_PROFILE,
    photon_alpha: float = V110_PHOTON_ALPHA,
):
    """v109pi forward + photon-shaped waist gate.

    The gate replaces the binary (k%2==0) alternation with a smooth profile.
    photon_alpha controls the blend: alpha=0 → identical to v109pi binary
    alternation; alpha=1 → fully smooth profile.
    """
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

    # V110_PHOTON_FOLD=1: triangle-wave fold rotation phases into [0, π/2].
    # Tests Bryce's "valley as pure cost" hypothesis: if alternation between
    # backbone-led (cos=1) and waist-led (cos=0) is the active ingredient,
    # folding should match or beat unfolded. Anti-alignment (cos<0) eliminated.
    # Pre-registered kill criterion (Jun 9): folded must beat cont9_step500
    # control (hard 0.397) by >+0.02 without easy regression beyond -0.01.
    fold = int(os.environ.get("V110_PHOTON_FOLD", "0")) > 0
    breath_phases = [phase_scale * k * math.pi / float(K_max) for k in range(K_max)]
    if fold:
        half_pi = math.pi / 2.0
        breath_phases = [half_pi - abs(half_pi - p) for p in breath_phases]
    breath_cos = [math.cos(p) for p in breath_phases]
    breath_sin = [math.sin(p) for p in breath_phases]

    # Per-breath photon gate values (constants)
    photon_gates = [_photon_gate(k, K_max, gate_profile) for k in range(K_max)]
    binary_gates = [_photon_gate(k, K_max, "binary")    for k in range(K_max)]

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
        stk_h    = stk.reshape(B, 1, T, T).expand(B, V110_PHOTON_N_HEADS, T, T)
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

        # Photon-gated waist: blend binary alternation with smooth profile
        # alpha=0 → byte-identical to v109pi binary alternation
        # alpha=1 → fully smooth profile
        # Continuous gating: h_quant_after = (1 - gate) * h_quant + gate * waist(h_quant)
        if alternation:
            gate_k_amp = (1.0 - photon_alpha) * binary_gates[k] + photon_alpha * photon_gates[k]
        else:
            # No alternation = waist every breath = gate amplitude 1.0
            gate_k_amp = 1.0

        # Always compute the waist; modulate the contribution by gate_k_amp.
        # When gate_k_amp = 0, h_quant unchanged (skip-waist breath).
        # When gate_k_amp = 1, h_quant fully replaced by waist output.
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
        x      = x_pre + gate_k * delta

        # Readout (same as v109pi)
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
# Parameter attachment (no new params — gate is computed from constants)
# ---------------------------------------------------------------------------

def attach_fg_params_v110_photon(
    model: Any,
    hidden: int,
    n_max: int = V110_PHOTON_N_MAX,
    f_max: int = V110_PHOTON_F_MAX,
    k_max: int | None = None,
    n_digits: int = V110_PHOTON_N_DIGITS,
    n_code: int = V110_PHOTON_CODEBOOK_N,
    ib_centroids_path: str = V110_PHOTON_IB_CENTROIDS,
    waist_dim: int = V110_PHOTON_WAIST_DIM,
) -> None:
    """v110-photon adds no new trainable params — the gate is computed
    from per-breath constants. PHOTON_ALPHA controls the blend at config time.
    """
    if k_max is None:
        k_max = V110_PHOTON_K_MAX
    attach_fg_params_v109pi(
        model, hidden=hidden,
        n_max=n_max, f_max=f_max, k_max=k_max,
        n_digits=n_digits, n_code=n_code, ib_centroids_path=ib_centroids_path,
        waist_dim=waist_dim,
    )
    profile = V110_PHOTON_GATE_PROFILE
    alpha   = V110_PHOTON_ALPHA
    gates = [_photon_gate(k, k_max, profile) for k in range(k_max)]
    binary = [_photon_gate(k, k_max, "binary") for k in range(k_max)]
    blended = [(1 - alpha) * binary[k] + alpha * gates[k] for k in range(k_max)]
    gates_s   = " ".join(f"{g:.3f}" for g in gates)
    blended_s = " ".join(f"{g:.3f}" for g in blended)
    print(
        f"[v110-photon] gate_profile={profile} alpha={alpha:.3f}", flush=True,
    )
    print(f"             pure_profile_gates: [{gates_s}]", flush=True)
    print(f"             blended (warm-start mix): [{blended_s}]", flush=True)


fg_v110_photon_parameters = fg_v109_parameters
fg_v110_photon_state_dict = fg_v109_state_dict


# ---------------------------------------------------------------------------
# JIT step/eval — mirror v109pi exactly, swap forward
# ---------------------------------------------------------------------------

_JIT_V110_PHOTON_CACHE: dict = {}


def _compile_jit_fg_step_v110_photon(
    model: Any,
    opt: Any,
    K: int,
    B: int,
    factor_aux_weight: float = V110_PHOTON_FACTOR_AUX_WEIGHT,
    calib_weight: float = V110_PHOTON_CALIB_WEIGHT,
    var_loss_weight: float = V110_PHOTON_VAR_LOSS_WEIGHT,
    hard_breath_level: bool = V110_PHOTON_HARD_BREATH_LEVEL,
    alternation: bool = V110_PHOTON_ALTERNATION,
    phase_scale: float = V110_PHOTON_PHASE_SCALE,
    n_max: int = V110_PHOTON_N_MAX,
    f_max: int = V110_PHOTON_F_MAX,
    n_digits: int = V110_PHOTON_N_DIGITS,
    gate_profile: str = V110_PHOTON_GATE_PROFILE,
    photon_alpha: float = V110_PHOTON_ALPHA,
    grad_clip: float = 1.0,
):
    key = ("v110_photon", id(model), id(opt), int(K), int(B),
           float(factor_aux_weight), float(calib_weight), float(var_loss_weight),
           bool(hard_breath_level), bool(alternation), float(phase_scale),
           int(n_max), int(f_max), int(n_digits),
           str(gate_profile), float(photon_alpha), float(grad_clip))
    if key in _JIT_V110_PHOTON_CACHE:
        return _JIT_V110_PHOTON_CACHE[key]

    fw, aw, vw, gc = float(factor_aux_weight), float(calib_weight), \
                     float(var_loss_weight), float(grad_clip)
    params = opt.params
    print(
        f"[JIT] compile v110-photon fg step: K={K} B={B} n_digits={n_digits} "
        f"alternation={alternation} profile={gate_profile} alpha={photon_alpha}...",
        flush=True,
    )

    @TinyJit
    def _step(
        domain_init, node_kinds, staging_mask, head_op_mask,
        gold_digits, gold_bins, observed_mask, factor_gold_bin, factor_valid,
    ):
        opt.zero_grad()

        tree_lh, var_lh, fac_lh, calib_h = fg_breathing_forward_v110_photon(
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

    _JIT_V110_PHOTON_CACHE[key] = _step
    print(f"[JIT] v110-photon step ready (cache={len(_JIT_V110_PHOTON_CACHE)})", flush=True)
    return _step


def _compile_jit_fg_eval_v110_photon(
    model: Any,
    K: int,
    B: int,
    n_max: int = V110_PHOTON_N_MAX,
    f_max: int = V110_PHOTON_F_MAX,
    n_digits: int = V110_PHOTON_N_DIGITS,
    alternation: bool = V110_PHOTON_ALTERNATION,
    phase_scale: float = V110_PHOTON_PHASE_SCALE,
    gate_profile: str = V110_PHOTON_GATE_PROFILE,
    photon_alpha: float = V110_PHOTON_ALPHA,
):
    key = ("eval_v110_photon", id(model), int(K), int(B), int(n_max), int(f_max),
           int(n_digits), bool(alternation), float(phase_scale),
           str(gate_profile), float(photon_alpha))
    if key in _JIT_V110_PHOTON_CACHE:
        return _JIT_V110_PHOTON_CACHE[key]

    print(
        f"[JIT] compile v110-photon fg eval: K={K} B={B} "
        f"profile={gate_profile} alpha={photon_alpha}...",
        flush=True,
    )

    @TinyJit
    def _eval(
        domain_init, node_kinds, staging_mask, head_op_mask,
        gold_digits, observed_mask,
    ):
        tree_lh, _, _, _ = fg_breathing_forward_v110_photon(
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

    _JIT_V110_PHOTON_CACHE[key] = _eval
    print(f"[JIT] v110-photon eval ready (cache={len(_JIT_V110_PHOTON_CACHE)})", flush=True)
    return _eval
