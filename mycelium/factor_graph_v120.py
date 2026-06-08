"""v120: Perceiver-as-cross-breath-NOTEBOOK with IB-anchored latent initialization.

Why v118/v119 FILTER failed and why NOTEBOOK is different:
  v118/v119 (FILTER): at each breath, compress 24 tokens → 16 latents → expand back.
    Latents created fresh, discarded after the breath. The model correctly concluded
    "why round-trip through 16 when 24 already fits in attention?" — pointless.

  v120 (NOTEBOOK): latents PERSIST across all K=8 breaths within one forward.
    Each breath:
      1. READ:  residual (B, T, H) cross-attends to latents → context delta added to x
      2. BREATH: standard v112b backbone (topology gate, 4 Pythia layers, waist, etc.)
      3. WRITE: latents updated by attending to residual after breath

    Latents store EVOLVING ABSTRACT GRAPH STATE — what's resolved, pending, the carry
    value progression — that the residual stream has only implicitly.

New params (~2M):
  fg_v120_latents       (16, 1024)     IB-centroid init (first 16 of 32)
  fg_v120_read_gate     (1,)           zero-init scalar (used as 1+g amplifier)
  fg_v120_write_gate    (1,)           zero-init scalar (used as 1+g amplifier)
  fg_v120_read_W_out    (1024, 1024)   ZERO-INIT — critical for bootstrap
  fg_v120_write_W_out   (1024, 1024)   ZERO-INIT — critical for bootstrap

Gradient bootstrap analysis:
  At step 0: W_out=0 → delta=ctx@0=0 → x unchanged → byte-identical to v112b.
  ∂L/∂read_W_out = read_ctx.T @ (1+g) * ∂L/∂out
    read_ctx is NON-ZERO at init (IB-meaningful latents → meaningful cross-attn).
    (1+g) at init = 1 (not 0), so gradient flows WITHOUT waiting for gate to open.
    read_W_out moves first; once read_delta ≠ 0, write_ctx carries information, and
    write_W_out gradient also becomes non-zero.
  read_gate / write_gate: ∂L/∂g = ∂L/∂out * delta — non-zero once W_out grows.
  Bootstrap order: read_W_out → write_W_out → gates (all self-unlocking).

AMD JIT safety:
  - No .cast(dtypes.float32) inside JIT (AM driver segfault) → .cast(dtypes.float)
  - .clip(-1e4, 1e4) for softmax stability
  - All new tensors .contiguous().realize() at attach time
  - latents_b allocated per-forward from model.fg_v120_latents.expand(...) — OK in JIT
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
)
from mycelium.factor_graph_v104 import load_ib_centroids


# ---------------------------------------------------------------------------
# v120 constants
# ---------------------------------------------------------------------------

V120_N_LATENTS = int(os.environ.get("V120_N_LATENTS", "16"))
V120_IB_CENTROIDS = os.environ.get(
    "V120_IB_CENTROIDS", ".cache/ib_centroids_gsm8k_partial.npz"
)


# ---------------------------------------------------------------------------
# Parameter attachment
# ---------------------------------------------------------------------------

def attach_fg_params_v120(
    model: Any,
    hidden: int,
    n_max: int = V110_STEP3_N_MAX,
    f_max: int = V110_STEP3_F_MAX,
    k_max: int = V110_STEP3_K_MAX,
    n_digits: int = V110_STEP3_N_DIGITS,
    n_code: int = 32,
    ib_centroids_path: str = ".cache/ib_centroids_gsm8k_partial.npz",
    waist_dim: int = V110_STEP3_WAIST_DIM,
    n_latents: int = V120_N_LATENTS,
) -> None:
    """Attach v112b backbone + v120 persistent-notebook params.

    All new v120 params are zero-init (W_outs and gates) except latents which
    are initialized from the first n_latents IB centroids. At step 0 the forward
    is byte-identical to v112b because W_out=0 → delta=0 → x unchanged.
    """
    # Step 1: v112b backbone (includes v110-step3 base)
    attach_fg_params_v112b(
        model, hidden=hidden,
        n_max=n_max, f_max=f_max, k_max=k_max,
        n_digits=n_digits, n_code=n_code, ib_centroids_path=ib_centroids_path,
        waist_dim=waist_dim,
    )

    if hasattr(model, "fg_v120_latents"):
        print("[v120] Notebook params already attached.", flush=True)
        return

    L = n_latents

    # Step 2: IB-centroid init for latents (first L of 32 IB centroids)
    try:
        # load_ib_centroids returns (n_code, hidden) — we take first L rows
        centroids_np = load_ib_centroids(
            ib_centroids_path, n_code=L, hidden=hidden,
            noise_scale=0.0,  # no noise: we want exact IB init for meaningful cross-attn
            seed=42,
        )
        lat_norms = np.linalg.norm(centroids_np, axis=1)
        print(
            f"[v120] Latents init from IB centroids: "
            f"norms={lat_norms.mean():.3f}, std={lat_norms.std():.3f}",
            flush=True,
        )
    except Exception as e:
        print(f"[v120] WARNING: IB centroid load failed ({e}); falling back to QR init", flush=True)
        _rng = np.random.RandomState(12345)
        _rand = _rng.randn(hidden, L).astype(np.float32)
        _q, _ = np.linalg.qr(_rand)   # (hidden, L) orthonormal
        centroids_np = _q.T[:L].astype(np.float32)  # (L, hidden)

    model.fg_v120_latents = Tensor(
        centroids_np.astype(np.float32)
    ).contiguous().realize()

    # Step 3: zero-init gate scalars (used as 1+g amplifiers — +1 ensures gradient flows)
    model.fg_v120_read_gate = Tensor(
        np.zeros((1,), dtype=np.float32)
    ).contiguous().realize()
    model.fg_v120_write_gate = Tensor(
        np.zeros((1,), dtype=np.float32)
    ).contiguous().realize()

    # Step 4: ZERO-INIT output projections — the critical bootstrap mechanism.
    # At init: delta = ctx @ W_out = ctx @ 0 = 0 → x unchanged.
    # But ∂L/∂W_out = ctx.T @ ∂L/∂x_after is non-zero (ctx is IB-meaningful),
    # so W_out receives gradient from step 1 onward.
    model.fg_v120_read_W_out = Tensor(
        np.zeros((hidden, hidden), dtype=np.float32)
    ).contiguous().realize()
    model.fg_v120_write_W_out = Tensor(
        np.zeros((hidden, hidden), dtype=np.float32)
    ).contiguous().realize()

    n_new = (L * hidden          # latents
             + 1 + 1             # read_gate, write_gate
             + hidden * hidden   # read_W_out
             + hidden * hidden)  # write_W_out
    print(
        f"[v120] Notebook params attached: L={L} latents (IB-init) + "
        f"2 × (H×H) zero-init W_out + 2 zero-init gates  "
        f"+{n_new:,} params  (~{n_new/1e6:.2f}M)",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Parameter collection + state dict
# ---------------------------------------------------------------------------

def fg_v120_parameters(model: Any) -> list[Tensor]:
    """v112b params + all v120-specific notebook params."""
    from mycelium.factor_graph_v112b import fg_v112b_parameters
    params = list(fg_v112b_parameters(model))
    for key in (
        "fg_v120_latents",
        "fg_v120_read_gate",
        "fg_v120_write_gate",
        "fg_v120_read_W_out",
        "fg_v120_write_W_out",
    ):
        if hasattr(model, key):
            params.append(getattr(model, key))
    return params


def fg_v120_state_dict(model: Any) -> dict[str, Tensor]:
    """v112b state_dict + all v120-specific notebook tensors."""
    from mycelium.factor_graph_v112b import fg_v112b_state_dict
    sd = dict(fg_v112b_state_dict(model))
    for key in (
        "fg_v120_latents",
        "fg_v120_read_gate",
        "fg_v120_write_gate",
        "fg_v120_read_W_out",
        "fg_v120_write_W_out",
    ):
        if hasattr(model, key):
            sd[key] = getattr(model, key)
    return sd


# ---------------------------------------------------------------------------
# Cross-breath notebook: read + write helpers
# ---------------------------------------------------------------------------

def _notebook_read(
    x: Tensor,             # (B, T, H) residual stream
    latents_b: Tensor,     # (B, L, H) persistent latent state
    layer0: Any,           # Pythia L0: provides wq, bq, wk, bk, shared.wv, shared.bv
    W_out: Tensor,         # (H, H) zero-init — output projection
    gate: Tensor,          # (1,) zero-init scalar — amplifier
) -> Tensor:
    """READ: residual attends to latents, returns updated residual.

    At init (W_out=0): delta = ctx @ 0 = 0 → returns x unchanged.
    ∂L/∂W_out = read_ctx.T @ (1+g) * ∂L/∂out — NON-ZERO because latents are IB-meaningful.
    """
    B = int(x.shape[0])
    T = int(x.shape[1])
    H = int(x.shape[2])
    L = int(latents_b.shape[1])
    dt = x.dtype

    # Q from residual
    x_flat = x.reshape(B * T, H).cast(dt)
    q_res = (x_flat @ layer0.wq.cast(dt) + layer0.bq.cast(dt)).reshape(B, T, H)

    # K, V from latents
    lat_flat = latents_b.reshape(B * L, H).cast(dt)
    k_lat = (lat_flat @ layer0.wk.cast(dt) + layer0.bk.cast(dt)).reshape(B, L, H)
    v_lat = (lat_flat @ layer0.shared.wv.cast(dt) + layer0.shared.bv.cast(dt)).reshape(B, L, H)

    scale = 1.0 / math.sqrt(H)
    scores = (q_res @ k_lat.transpose(1, 2)) * scale     # (B, T, L)
    weights = scores.clip(-1e4, 1e4).softmax(-1)          # (B, T, L)
    read_ctx = weights @ v_lat                             # (B, T, H) — non-zero at init

    # Pass through zero-init W_out
    read_delta = read_ctx @ W_out.cast(dt)                # zero at init
    # (1 + gate) amplifier: at init gate=0, amp=1, delta=0 → x unchanged.
    # The +1 ensures ∂L/∂W_out = read_ctx.T @ ∂L/∂x_after is non-zero from step 1.
    amp = (gate.cast(dtypes.float) + 1.0).cast(dt).reshape(1, 1, 1)
    return x + amp * read_delta


def _notebook_write(
    x: Tensor,             # (B, T, H) residual stream AFTER breath
    latents_b: Tensor,     # (B, L, H) current latent state
    layer0: Any,           # Pythia L0 projections
    W_out: Tensor,         # (H, H) zero-init output projection
    gate: Tensor,          # (1,) zero-init scalar amplifier
) -> Tensor:
    """WRITE: latents attend to residual, return updated latents.

    At init (W_out=0): write_delta=0 → latents unchanged.
    Once read_W_out grows (after a few steps), x carries information,
    write_ctx becomes meaningful, and write_W_out receives non-zero gradient.
    """
    B = int(x.shape[0])
    T = int(x.shape[1])
    H = int(x.shape[2])
    L = int(latents_b.shape[1])
    dt = x.dtype

    # Q from latents
    lat_flat = latents_b.reshape(B * L, H).cast(dt)
    q_lat = (lat_flat @ layer0.wq.cast(dt) + layer0.bq.cast(dt)).reshape(B, L, H)

    # K, V from residual
    x_flat = x.reshape(B * T, H).cast(dt)
    k_res = (x_flat @ layer0.wk.cast(dt) + layer0.bk.cast(dt)).reshape(B, T, H)
    v_res = (x_flat @ layer0.shared.wv.cast(dt) + layer0.shared.bv.cast(dt)).reshape(B, T, H)

    scale = 1.0 / math.sqrt(H)
    scores = (q_lat @ k_res.transpose(1, 2)) * scale     # (B, L, T)
    weights = scores.clip(-1e4, 1e4).softmax(-1)          # (B, L, T)
    write_ctx = weights @ v_res                            # (B, L, H)

    # Zero-init W_out + (1+gate) amplifier (same pattern as read)
    write_delta = write_ctx @ W_out.cast(dt)               # zero at init
    amp = (gate.cast(dtypes.float) + 1.0).cast(dt).reshape(1, 1, 1)
    return latents_b + amp * write_delta


# ---------------------------------------------------------------------------
# Main forward pass
# ---------------------------------------------------------------------------

def fg_breathing_forward_v120(
    model: Any,
    domain_init: Tensor,
    node_kinds: Tensor,
    staging_mask: Tensor,
    head_op_mask: Tensor,
    noise: Tensor,        # (K_max, B, T, H)
    noise_scale: Tensor,  # scalar
    K: int,
    n_max: int = V110_STEP3_N_MAX,
    f_max: int = V110_STEP3_F_MAX,
    n_digits: int = V110_STEP3_N_DIGITS,
    alternation: bool = V110_STEP3_ALTERNATION,
    phase_scale: float = V110_STEP3_PHASE_SCALE,
    gate_profile: str = V110_STEP3_GATE_PROFILE,
    photon_alpha: float = V110_STEP3_PHOTON_ALPHA,
):
    """v120 forward: v112b backbone + persistent cross-breath latent notebook.

    Returns (tree_logits_history, var_logits_history, factor_logits_history,
             calib_history, step_mags_history) — same shape as v112b.

    With fg_v120_read_W_out=0 and fg_v120_write_W_out=0 (init):
      output is byte-identical to v112b.
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

    node_topology   = model.fg_v115_node_topology
    W_res_gate      = model.fg_v115_W_res_gate
    attn_bias_scale = model.fg_v115_attn_bias_scale

    # v120 notebook params
    latents_1L_H  = model.fg_v120_latents       # (L, H)
    read_gate     = model.fg_v120_read_gate      # (1,)
    write_gate    = model.fg_v120_write_gate     # (1,)
    read_W_out    = model.fg_v120_read_W_out     # (H, H)
    write_W_out   = model.fg_v120_write_W_out    # (H, H)

    H = int(tree_codebook.shape[-1])
    B = int(domain_init.shape[0])
    T = n_max + f_max
    L = int(latents_1L_H.shape[0])

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

    # === v112b precomputations (outside K loop) ===
    topology_f = node_topology.cast(dtypes.float)
    attn_bias_full = (topology_f @ topology_f.T) * attn_bias_scale.reshape(1, 1).cast(dtypes.float)
    attn_bias_btht = attn_bias_full.reshape(1, 1, T, T)
    gate_per_pos = (topology_f @ W_res_gate.cast(dtypes.float)).tanh()
    gate_multiplier_3d = (1.0 + gate_per_pos).reshape(1, T, H)

    # === v120: initialize persistent latents per batch (not per step!) ===
    # latents_b lives across K breaths for this single forward call.
    # Shape: (B, L, H). Initialize from model params (shared across batch,
    # then each batch element's evolution diverges through the write steps).
    latents_b = latents_1L_H.reshape(1, L, H).expand(B, L, H).cast(dtypes.half)

    layer0 = layers[0]  # Pythia L0 — used for notebook cross-attention projections

    for k in range(K):
        # Accumulate notebook read (v110-acc cross-breath history)
        x_acc_delta = _acc_notebook_read(
            x, notebook_slots,
            acc_W_q, acc_W_k, acc_W_v, acc_W_o, acc_b_o,
        )
        x = x + x_acc_delta

        # SBP noise
        x = x + (noise[k] * noise_scale).cast(x.dtype)

        # v112b per-position residual gate
        x = x * gate_multiplier_3d.cast(x.dtype)

        # ===== READ: residual gets context from latents =====
        x = _notebook_read(x, latents_b, layer0, read_W_out, read_gate)

        be_k  = breath_embed[k].reshape(1, 1, -1).cast(x.dtype)
        x_in  = x + be_k
        x_pre = x

        stk      = staging_mask[:, k, :, :]
        stk_h    = stk.reshape(B, 1, T, T).expand(B, V110_STEP_N_HEADS, T, T)
        combined = stk_h.cast(x.dtype) + head_op_mask.cast(x.dtype)
        combined = combined + attn_bias_btht.cast(combined.dtype)

        cos_k = breath_cos[k]
        sin_k = breath_sin[k]

        # ===== STANDARD BREATH: v112b 4-layer forward =====
        h = x_in
        for layer in layers[:4]:
            h = fg_layer_forward_v109pi(layer, h, combined, cos_k, sin_k)

        # IB semantic codebook
        cb  = semantic_codebook.cast(h.dtype)
        tmp = temperature.cast(h.dtype)
        scores_cb = h @ cb.T / tmp.reshape(1, 1, 1)
        weights_cb = scores_cb.clip(-1e4, 1e4).softmax(-1)
        recon    = weights_cb @ cb
        quantize = recon - h
        gate_quant_k = delta_gate_quant[k].cast(h.dtype).reshape(1, 1, 1)
        h_quant = h + gate_quant_k * quantize

        # Photon-gated waist
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

        step_mag_k = step.cast(dtypes.float).square().mean()
        step_mags_history.append(step_mag_k)

        # ===== WRITE: latents updated from residual AFTER breath =====
        latents_b = _notebook_write(x, latents_b, layer0, write_W_out, write_gate)

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


# ---------------------------------------------------------------------------
# JIT compile helpers
# ---------------------------------------------------------------------------

_JIT_V120_CACHE: dict = {}


def _compile_jit_fg_step_v120(
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
    """Train step for v120. Same loss as v112b, with v120 persistent-notebook forward."""
    key = ("v120", id(model), id(opt), int(K), int(B),
           float(factor_aux_weight), float(calib_weight), float(var_loss_weight),
           float(balance_weight), float(uncertainty_min),
           bool(hard_breath_level), bool(alternation), float(phase_scale),
           int(n_max), int(f_max), int(n_digits),
           str(gate_profile), float(photon_alpha), float(grad_clip))
    if key in _JIT_V120_CACHE:
        return _JIT_V120_CACHE[key]

    fw, aw, vw, bw, um, gc = (
        float(factor_aux_weight), float(calib_weight),
        float(var_loss_weight), float(balance_weight),
        float(uncertainty_min), float(grad_clip),
    )
    params = opt.params
    print(
        f"[JIT] compile v120 step: K={K} B={B} "
        f"profile={gate_profile} alpha={photon_alpha}...",
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
            fg_breathing_forward_v120(
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
            fac_masked = fac_nll * valid_flat
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

        # AMD-safe: single-kernel isfinite check
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

    _JIT_V120_CACHE[key] = _step
    print(f"[JIT] v120 step ready (cache={len(_JIT_V120_CACHE)})", flush=True)
    return _step


def compile_jit_eval_v120(
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
    """JIT'd eval forward for v120 — noise baked-in as zeros (deterministic)."""
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
        tree_lh, _, _, _, _ = fg_breathing_forward_v120(
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
