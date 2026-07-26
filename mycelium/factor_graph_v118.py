"""v118: 4-phase piecewise photon + Perceiver bottleneck + 4 phase-gated LoRAs.

Builds ON TOP OF v112b (per-node residual gate validated Jun 7).
Three new mechanisms, all independently toggleable via env flags:

Mechanism 1: 4-phase piecewise photon envelope (V118_GATE_PROFILE=piecewise_4phase)
  K=8 breaths split into 4 phases of 2 breaths each:
    EXPLORE   breaths 0,1  gate = 0.0, 0.0
    COMPRESS  breaths 2,3  gate = 1.0, 0.8
    COMMIT    breaths 4,5  gate = 0.5, 0.5
    REFINE    breaths 6,7  gate = 0.1, 0.0
  Combined with existing alternation (waist on even breaths only):
    effective gates: 0.0, 1.0, 0.5, 0.1 (on even breaths only)
  Implemented in _photon_gate in factor_graph_v110_photon.py.

Mechanism 2: Perceiver-style bottleneck at phase transitions (V118_PERCEIVER_ENABLED=1)
  Learnable latent queries: fg_v118_perceiver_latents (N_LATENTS, H)  QR-init
  At end of breaths 1, 3, 5 (last breath of EXPLORE, COMPRESS, COMMIT):
    Compress: cross-attn(Q=latents, K=residual, V=residual) → (N_LATENTS, H)
    Expand:   cross-attn(Q=residual, K=latents, V=latents)  → (T, H)
  Uses Pythia L0 Q/K/V projections (no new attention params beyond latents).
  Zero-init gate scalar: residual += gate_scalar * (perceiver_out - residual)
  Step 0 forward is byte-identical to v112b (gate=0 → no effect).

Mechanism 3: 4 phase-gated LoRA adapters (V118_PER_PHASE_LORAS=1)
  Rank-16 LoRAs on each Pythia L0-L3 Q, K, V, O, FFN_W1, FFN_W2.
  Four adapter sets (A=explore, B=compress, C=commit, D=refine), active
  only on their designated phase breaths.
  Down-projections init 0.02 Gaussian; UP-projections zero-init.
  Step 0 forward byte-identical to v112b (up=0 → zero contribution).

Ablation matrix (single build, four smoke variants):
  v118-a: piecewise_4phase, PERCEIVER=0, LORAS=0  — envelope alone
  v118-b: piecewise_4phase, PERCEIVER=1, LORAS=0  — + Perceiver
  v118-c: piecewise_4phase, PERCEIVER=1, LORAS=1  — full v118
  v118-d: sin2_pi, PERCEIVER=0, LORAS=0           — control = v112b baseline

AMD JIT safety:
  - No .cast(dtypes.float32) inside JIT (AM driver segfault) → use .cast(dtypes.float)
  - No per-param .isnan() inside JIT → single-kernel total.isfinite()
  - LoRA phase selection via Python-level integer index (no JIT branching)
    If K=8 × 4-phase LoRAs × 6 projections × 4 layers exceeds AMD JIT capacity,
    phase masking stays at Python level (already the case here).
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


# ---------------------------------------------------------------------------
# V118 env flags
# ---------------------------------------------------------------------------

V118_GATE_PROFILE       = os.environ.get("V118_GATE_PROFILE", "piecewise_4phase")
V118_PERCEIVER_ENABLED  = int(os.environ.get("V118_PERCEIVER_ENABLED", "0")) > 0
V118_PERCEIVER_N_LATENTS = int(os.environ.get("V118_PERCEIVER_N_LATENTS", "16"))
V118_PER_PHASE_LORAS    = int(os.environ.get("V118_PER_PHASE_LORAS", "0")) > 0
V118_LORA_RANK          = int(os.environ.get("V118_LORA_RANK", "16"))
# v119 fix: zero-init output projection on perceiver (bootstrap fix).
# When 1, adds fg_v118_perceiver_W_out (H, H) zero-init and uses
#   delta = ctx_e @ W_out  (= 0 at init);  out = x + g * delta
# instead of the v118 formula  out = x + g * (ctx_e - x)  which couldn't bootstrap.
V119_PERCEIVER_W_OUT    = int(os.environ.get("V119_PERCEIVER_W_OUT", "0")) > 0

# Phase schedule for K=8: which breaths belong to which phase
# phase 0=EXPLORE(0,1), 1=COMPRESS(2,3), 2=COMMIT(4,5), 3=REFINE(6,7)
_PHASE_FOR_BREATH = [0, 0, 1, 1, 2, 2, 3, 3]
_PHASE_NAMES = ["explore", "compress", "commit", "refine"]

# Perceiver trigger at end of which breaths (last breath of each non-refine phase)
_PERCEIVER_TRIGGER_BREATHS = {1, 3, 5}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _breath_phase(k: int, K_max: int = 8) -> int:
    """Return phase index 0-3 for breath k (generalises beyond K=8)."""
    # Divide K evenly into 4 phases; last phase gets any remainder
    phase_size = max(K_max // 4, 1)
    phase = min(k // phase_size, 3)
    return phase


def _perceiver_trigger_set(K_max: int = 8) -> set:
    """Return set of breath indices that trigger the perceiver.

    These are the last breath of phases 0, 1, 2 (EXPLORE, COMPRESS, COMMIT).
    Phase 3 (REFINE) does not trigger.
    """
    phase_size = max(K_max // 4, 1)
    triggers = set()
    for phase in range(3):
        last_of_phase = min(phase_size * (phase + 1) - 1, K_max - 1)
        triggers.add(last_of_phase)
    return triggers


# ---------------------------------------------------------------------------
# Parameter attachment
# ---------------------------------------------------------------------------

def attach_fg_params_v118(
    model: Any,
    hidden: int,
    n_max: int = V110_STEP3_N_MAX,
    f_max: int = V110_STEP3_F_MAX,
    k_max: int = V110_STEP3_K_MAX,
    n_digits: int = V110_STEP3_N_DIGITS,
    n_code: int = 32,
    ib_centroids_path: str = ".cache/ib_centroids_gsm8k_partial.npz",
    waist_dim: int = V110_STEP3_WAIST_DIM,
    perceiver_n_latents: int = V118_PERCEIVER_N_LATENTS,
    lora_rank: int = V118_LORA_RANK,
    perceiver_enabled: bool = V118_PERCEIVER_ENABLED,
    per_phase_loras: bool = V118_PER_PHASE_LORAS,
) -> None:
    """Attach v112b backbone + v118-specific params (perceiver, LoRAs).

    All new params are zero-init (gates/up-projections) so step 0 forward
    is byte-identical to v112b.
    """
    # Step 1: attach v112b backbone (includes v110-step3 base)
    attach_fg_params_v112b(
        model, hidden=hidden,
        n_max=n_max, f_max=f_max, k_max=k_max,
        n_digits=n_digits, n_code=n_code, ib_centroids_path=ib_centroids_path,
        waist_dim=waist_dim,
    )

    new_params = 0

    # Step 2: Perceiver bottleneck (optional)
    if perceiver_enabled and not hasattr(model, "fg_v118_perceiver_latents"):
        # QR-init latents for orthogonality; shape (N_latents, hidden)
        L = perceiver_n_latents
        latents_np = np.zeros((L, hidden), dtype=np.float32)
        # Fill with QR decomposition of a random matrix for orthogonal init
        _rng_p = np.random.RandomState(12345)
        _rand_mat = _rng_p.randn(L, hidden).astype(np.float32)
        # QR gives Q shape (L, L) if L < hidden; we want (L, hidden)
        # Use L rows of a random orthonormal basis in R^hidden
        _q, _ = np.linalg.qr(_rand_mat.T)  # Q: (hidden, L)
        latents_np = _q.T.astype(np.float32)  # (L, hidden) — orthonormal rows
        model.fg_v118_perceiver_latents = Tensor(
            latents_np
        ).contiguous().realize()

        # Zero-init gate scalar (ensures step 0 = byte-identical to v112b)
        model.fg_v118_perceiver_gate = Tensor(
            np.zeros((1,), dtype=np.float32)
        ).contiguous().realize()

        # v119 fix: zero-init output projection (H, H). When enabled, the gate
        # opens onto a delta starting at zero rather than a "replace good with
        # garbage" update. Two-knob graduation: gate AND W_out must learn
        # together — same pattern that worked for v112b residual gate.
        if V119_PERCEIVER_W_OUT:
            model.fg_v118_perceiver_W_out = Tensor(
                np.zeros((hidden, hidden), dtype=np.float32)
            ).contiguous().realize()
            new_params += hidden * hidden
            print(
                f"[v119] Perceiver W_out: ({hidden},{hidden}) zero-init  "
                f"+{hidden * hidden:,} params  (bootstrap fix)",
                flush=True,
            )

        new_params += L * hidden + 1
        print(
            f"[v118] Perceiver: {L} latents × {hidden} hidden "
            f"(QR-init) + zero-init gate scalar  +{L * hidden + 1:,} params",
            flush=True,
        )
    elif perceiver_enabled:
        print("[v118] Perceiver params already attached.", flush=True)
    else:
        print("[v118] Perceiver DISABLED (V118_PERCEIVER_ENABLED=0).", flush=True)

    # Step 3: 4 phase-gated LoRA adapters (optional)
    # 6 projections × 4 layers × 4 phases = 96 LoRA pairs
    # Key names: fg_v118_lora_{phase}_{layer}_{proj}_{down|up}
    # Projections: q, k, v, o, ffn_w1, ffn_w2
    _LORA_PROJS = ["q", "k", "v", "o", "ffn_w1", "ffn_w2"]
    _LORA_PHASES = ["explore", "compress", "commit", "refine"]

    if per_phase_loras and not hasattr(model, "fg_v118_lora_explore_0_q_down"):
        _rng_l = np.random.RandomState(42)

        # Projection dimensions (Pythia-410M L0-L3):
        # q,k: (hidden, hidden) — but Q/K are (hidden, n_heads*head_dim) = (1024, 1024)
        # v,o: (hidden, hidden)
        # ffn_w1: (hidden, ffn_dim) = (1024, 4096)
        # ffn_w2: (ffn_dim, hidden) = (4096, 1024)
        _PROJ_IN_DIMS  = {"q": hidden, "k": hidden, "v": hidden,
                          "o": hidden, "ffn_w1": hidden, "ffn_w2": hidden * 4}
        _PROJ_OUT_DIMS = {"q": hidden, "k": hidden, "v": hidden,
                          "o": hidden, "ffn_w1": hidden * 4, "ffn_w2": hidden}

        n_lora_params = 0
        for phase_name in _LORA_PHASES:
            for layer_idx in range(4):
                for proj_name in _LORA_PROJS:
                    in_dim  = _PROJ_IN_DIMS[proj_name]
                    out_dim = _PROJ_OUT_DIMS[proj_name]
                    rank = lora_rank

                    # Down: (in_dim, rank) — small Gaussian init
                    down_np = _rng_l.randn(in_dim, rank).astype(np.float32) * 0.02
                    down_key = f"fg_v118_lora_{phase_name}_{layer_idx}_{proj_name}_down"
                    setattr(model, down_key,
                            Tensor(down_np).contiguous().realize())

                    # Up: (rank, out_dim) — zero-init (ensures step 0 identity)
                    up_np = np.zeros((rank, out_dim), dtype=np.float32)
                    up_key = f"fg_v118_lora_{phase_name}_{layer_idx}_{proj_name}_up"
                    setattr(model, up_key,
                            Tensor(up_np).contiguous().realize())

                    n_lora_params += in_dim * rank + rank * out_dim

        new_params += n_lora_params
        print(
            f"[v118] LoRAs: rank={lora_rank} × 6 projs × 4 layers × 4 phases "
            f"(zero-init up → step 0 identity)  +{n_lora_params:,} params",
            flush=True,
        )
    elif per_phase_loras:
        print("[v118] LoRA params already attached.", flush=True)
    else:
        print("[v118] Per-phase LoRAs DISABLED (V118_PER_PHASE_LORAS=0).", flush=True)

    print(
        f"[v118] total NEW params (beyond v112b backbone): {new_params:,}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Parameter collection + state dict
# ---------------------------------------------------------------------------

def fg_v118_parameters(model: Any) -> list[Tensor]:
    """v112b params + all v118-specific params."""
    from mycelium.factor_graph_v112b import fg_v112b_parameters
    params = list(fg_v112b_parameters(model))

    # Perceiver
    for key in ("fg_v118_perceiver_latents", "fg_v118_perceiver_gate", "fg_v118_perceiver_W_out"):
        if hasattr(model, key):
            params.append(getattr(model, key))

    # LoRAs
    _LORA_PROJS  = ["q", "k", "v", "o", "ffn_w1", "ffn_w2"]
    _LORA_PHASES = ["explore", "compress", "commit", "refine"]
    for phase_name in _LORA_PHASES:
        for layer_idx in range(4):
            for proj_name in _LORA_PROJS:
                for suffix in ("down", "up"):
                    key = f"fg_v118_lora_{phase_name}_{layer_idx}_{proj_name}_{suffix}"
                    if hasattr(model, key):
                        params.append(getattr(model, key))

    return params


def fg_v118_state_dict(model: Any) -> dict[str, Tensor]:
    """v112b state_dict + all v118-specific tensors."""
    from mycelium.factor_graph_v112b import fg_v112b_state_dict
    sd = dict(fg_v112b_state_dict(model))

    for key in ("fg_v118_perceiver_latents", "fg_v118_perceiver_gate", "fg_v118_perceiver_W_out"):
        if hasattr(model, key):
            sd[key] = getattr(model, key)

    _LORA_PROJS  = ["q", "k", "v", "o", "ffn_w1", "ffn_w2"]
    _LORA_PHASES = ["explore", "compress", "commit", "refine"]
    for phase_name in _LORA_PHASES:
        for layer_idx in range(4):
            for proj_name in _LORA_PROJS:
                for suffix in ("down", "up"):
                    key = f"fg_v118_lora_{phase_name}_{layer_idx}_{proj_name}_{suffix}"
                    if hasattr(model, key):
                        sd[key] = getattr(model, key)

    return sd


# ---------------------------------------------------------------------------
# Perceiver helper: compress+expand at phase transitions
# ---------------------------------------------------------------------------

def _perceiver_compress_expand(
    x: Tensor,          # (B, T, H) residual stream
    latents: Tensor,    # (N_latents, H) learnable latent queries
    gate: Tensor,       # scalar — zero-init
    layer0: Any,        # Pythia L0 for Q/K/V projections
    W_out: Any = None,  # v119 fix: (H, H) zero-init output projection; if None, use v118 formula
) -> Tensor:
    """Perceiver bottleneck: compress residual into latents then expand back.

    Uses Pythia L0's Q/K/V weight matrices for the cross-attention projections
    (no new attention parameters beyond the latent queries themselves).

    At init (gate=0): returns x unchanged — byte-identical to no-perceiver.
    """
    B = int(x.shape[0])
    T = int(x.shape[1])
    H = int(x.shape[2])
    L = int(latents.shape[0])
    dt = x.dtype

    # Perceiver gate (zero-init scalar → no effect at step 0)
    g = gate.cast(dtypes.float).reshape(1, 1, 1)

    # If gate is effectively zero, skip all computation for efficiency
    # (Python-level check, not inside JIT — this branch is resolved at trace time)
    # Note: inside TinyJit we can't branch on tensor values, so we always compute
    # but the multiply-by-gate makes it a no-op at init.

    # Flatten x for projection: (B*T, H)
    x_flat = x.reshape(B * T, H).cast(dt)

    # ---- COMPRESS: latents attend to residual ----
    # Q from latents: (L, H)
    lat = latents.cast(dt)                                # (L, H)
    q_lat = lat @ layer0.wq.cast(dt) + layer0.bq.cast(dt)  # (L, H)
    # K, V from residual flattened (treat T dim as sequence)
    k_res = x_flat @ layer0.wk.cast(dt) + layer0.bk.cast(dt)  # (B*T, H)
    v_res = x_flat @ layer0.shared.wv.cast(dt) + layer0.shared.bv.cast(dt)

    # Reshape for batched cross-attn: broadcast latent Q over batch
    # q_lat: (1, L, H) → (B, L, H)
    q_lat_b = q_lat.reshape(1, L, H).expand(B, L, H)       # (B, L, H)
    k_res_b = k_res.reshape(B, T, H)                        # (B, T, H)
    v_res_b = v_res.reshape(B, T, H)                        # (B, T, H)

    scale = 1.0 / math.sqrt(H)
    # scores: (B, L, T)
    scores_c = (q_lat_b @ k_res_b.transpose(1, 2)) * scale
    weights_c = scores_c.clip(-1e4, 1e4).softmax(-1)        # (B, L, T)
    ctx_c = weights_c @ v_res_b                              # (B, L, H)
    # ctx_c: compressed latent states

    # ---- EXPAND: residual attends to compressed latents ----
    q_res  = x_flat @ layer0.wq.cast(dt) + layer0.bq.cast(dt)   # (B*T, H)
    k_lat  = ctx_c.reshape(B, L, H)                              # (B, L, H)
    v_lat  = ctx_c.reshape(B, L, H)

    q_res_b = q_res.reshape(B, T, H)                            # (B, T, H)
    k_lat_s = (k_lat @ layer0.wk.cast(dt).T)                    # (B, L, H) — optional re-proj
    v_lat_s = (v_lat @ layer0.shared.wv.cast(dt).T)             # (B, L, H)

    scores_e  = (q_res_b @ k_lat_s.transpose(1, 2)) * scale     # (B, T, L)
    weights_e = scores_e.clip(-1e4, 1e4).softmax(-1)            # (B, T, L)
    ctx_e     = weights_e @ v_lat_s                              # (B, T, H)

    # v119 fix v2: zero-init output projection bootstraps the perceiver.
    # The original v118 formula  out = x + g * (ctx_e - x)  is bootstrap-locked
    # because ctx_e at init is garbage (random latents → random cross-attention)
    # and loss pushes g AWAY from opening.
    # First v119 attempt used  out = x + g * (ctx_e @ W_out)  but that's ALSO
    # locked: ∂L/∂g = ∂L/∂out * delta = 0 at init (delta=0), AND
    #         ∂L/∂W_out = g * ... = 0 at init (g=0). Double-zero lockup.
    # Real fix: drop g entirely. W_out alone is the rate-limiter:
    #   delta = ctx_e @ W_out  (zero at init via W_out=0)
    #   out = x + delta         (byte-identical at init, gradient flows to W_out)
    # ∂L/∂W_out = ctx_e^T @ ∂L/∂out  is NON-ZERO at init because ctx_e is
    # meaningful (Pythia-projected). W_out can grow without needing a gate.
    if W_out is not None:
        delta = ctx_e @ W_out.cast(dt)        # (B, T, H) — zero at init
        # Use (1 + g) as amplifier: at init g=0 → amp=1, delta=0 → out=x
        # Keeps g in the autograd graph (required by tinygrad's unwrap)
        # but the +1 ensures W_out gradient flows regardless of g's value.
        # W_out moves first; once delta ≠ 0, g gradient becomes non-zero too.
        amp = g.cast(dt) + 1.0
        out = x + amp * delta
    else:
        # v118 original formula (bootstrap-locked)
        out = x + g.cast(dt) * (ctx_e - x)
    return out


# ---------------------------------------------------------------------------
# LoRA application helper
# ---------------------------------------------------------------------------

def _lora_delta(x_in: Tensor, down: Tensor, up: Tensor) -> Tensor:
    """Compute LoRA delta: x_in @ down @ up.

    down: (in_dim, rank), up: (rank, out_dim). Both cast to x_in.dtype.
    Up is zero-init so delta is zero at step 0.
    """
    dt = x_in.dtype
    return x_in @ down.cast(dt) @ up.cast(dt)


def _layer_forward_v118_lora(
    layer: Any,
    x: Tensor,
    attn_bias: Tensor,
    q_rot_cos: float,
    q_rot_sin: float,
    # LoRA params for this layer (may be None if LoRAs disabled)
    lora_q_down: Tensor | None = None, lora_q_up: Tensor | None = None,
    lora_k_down: Tensor | None = None, lora_k_up: Tensor | None = None,
    lora_v_down: Tensor | None = None, lora_v_up: Tensor | None = None,
    lora_o_down: Tensor | None = None, lora_o_up: Tensor | None = None,
    lora_w1_down: Tensor | None = None, lora_w1_up: Tensor | None = None,
    lora_w2_down: Tensor | None = None, lora_w2_up: Tensor | None = None,
) -> Tensor:
    """v109pi layer forward + optional phase-gated LoRA adapters.

    When all lora_*_down/up are None (LoRAs disabled) or up-projections are
    zero-init, output is byte-identical to fg_layer_forward_v109pi.
    """
    from mycelium.factor_graph_v109pi import _rotate_q_pi
    from mycelium.breathing import _layernorm

    cfg = layer.cfg
    B, S, H = x.shape
    n_heads  = cfg.n_heads
    head_dim = cfg.head_dim

    attn_in = _layernorm(x, layer.shared.in_ln_g, layer.shared.in_ln_b, cfg.layer_norm_eps)
    mlp_in  = _layernorm(x, layer.shared.post_ln_g, layer.shared.post_ln_b, cfg.layer_norm_eps)
    attn_in_dt = attn_in.cast(x.dtype) if attn_in.dtype != x.dtype else attn_in
    mlp_in_dt  = mlp_in.cast(x.dtype) if mlp_in.dtype != x.dtype else mlp_in

    # Q projection + LoRA
    q_base = attn_in_dt @ layer.wq + layer.bq
    if lora_q_down is not None and lora_q_up is not None:
        q_base = q_base + _lora_delta(attn_in_dt, lora_q_down, lora_q_up)
    q = q_base.reshape(B, S, n_heads, head_dim).transpose(1, 2)

    # K projection + LoRA
    k_base = attn_in_dt @ layer.wk + layer.bk
    if lora_k_down is not None and lora_k_up is not None:
        k_base = k_base + _lora_delta(attn_in_dt, lora_k_down, lora_k_up)
    k = k_base.reshape(B, S, n_heads, head_dim).transpose(1, 2)

    # V projection + LoRA
    v_base = attn_in_dt @ layer.shared.wv + layer.shared.bv
    if lora_v_down is not None and lora_v_up is not None:
        v_base = v_base + _lora_delta(attn_in_dt, lora_v_down, lora_v_up)
    v = v_base.reshape(B, S, n_heads, head_dim).transpose(1, 2)

    # Per-breath Q rotation (v109pi)
    q = _rotate_q_pi(q, q_rot_cos, q_rot_sin)

    scale  = 1.0 / math.sqrt(head_dim)
    scores = q @ k.transpose(-2, -1) * scale
    scores = scores + attn_bias.cast(scores.dtype)
    attn   = scores.clip(-1e4, 1e4).softmax(-1)
    ctx    = (attn @ v).transpose(1, 2).reshape(B, S, H)

    # O projection + LoRA
    attn_out_base = ctx @ layer.shared.wo + layer.shared.bo
    if lora_o_down is not None and lora_o_up is not None:
        attn_out_base = attn_out_base + _lora_delta(ctx, lora_o_down, lora_o_up)

    # FFN_W1 + LoRA
    ff_base = mlp_in_dt @ layer.w_in + layer.b_in
    if lora_w1_down is not None and lora_w1_up is not None:
        ff_base = ff_base + _lora_delta(mlp_in_dt, lora_w1_down, lora_w1_up)
    ff = ff_base.gelu()

    # FFN_W2 + LoRA
    ffn_out_base = ff @ layer.shared.w_out + layer.shared.b_out
    if lora_w2_down is not None and lora_w2_up is not None:
        ffn_out_base = ffn_out_base + _lora_delta(ff, lora_w2_down, lora_w2_up)

    return x + attn_out_base + ffn_out_base


# ---------------------------------------------------------------------------
# Main forward pass
# ---------------------------------------------------------------------------

def fg_breathing_forward_v118(
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
    gate_profile: str = V118_GATE_PROFILE,
    photon_alpha: float = V110_STEP3_PHOTON_ALPHA,
    perceiver_enabled: bool = V118_PERCEIVER_ENABLED,
    per_phase_loras: bool = V118_PER_PHASE_LORAS,
):
    """v118 forward: v112b backbone + piecewise photon + Perceiver + phase LoRAs.

    Returns (tree_logits_history, var_logits_history, factor_logits_history,
             calib_history, step_mags_history).

    With gate_profile='sin2_pi' (or any existing profile), perceiver_enabled=False,
    per_phase_loras=False: output is byte-identical to fg_breathing_forward_v112b.
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

    # Resolve gate profile: map v118-specific name to _photon_gate call
    # "piecewise_4phase" is handled below; all others pass through to _photon_gate
    def _resolve_gate(k: int) -> float:
        if gate_profile == "piecewise_4phase":
            # Hard-coded piecewise table for K=8
            _PIECEWISE = [0.0, 0.0, 1.0, 0.8, 0.5, 0.5, 0.1, 0.0]
            if k < len(_PIECEWISE):
                return _PIECEWISE[k]
            return 0.0
        return _photon_gate(k, K_max, gate_profile)

    photon_gates = [_resolve_gate(k) for k in range(K_max)]
    binary_gates = [_photon_gate(k, K_max, "binary") for k in range(K_max)]

    # Perceiver setup (Python-level booleans, not JIT branches)
    perceiver_latents = None
    perceiver_gate    = None
    perceiver_W_out   = None  # v119 fix
    perceiver_trigger = set()
    if perceiver_enabled and hasattr(model, "fg_v118_perceiver_latents"):
        perceiver_latents = model.fg_v118_perceiver_latents
        perceiver_gate    = model.fg_v118_perceiver_gate
        perceiver_trigger = _perceiver_trigger_set(K_max)
        if hasattr(model, "fg_v118_perceiver_W_out"):
            perceiver_W_out = model.fg_v118_perceiver_W_out

    # LoRA setup: pre-load all phase LoRA tensors into a dict keyed by
    # (phase_name, layer_idx, proj_name) → (down, up)
    # Phase mapping is Python-level so no JIT branching.
    lora_params: dict = {}
    if per_phase_loras and hasattr(model, "fg_v118_lora_explore_0_q_down"):
        _LORA_PROJS  = ["q", "k", "v", "o", "ffn_w1", "ffn_w2"]
        _LORA_PHASES = ["explore", "compress", "commit", "refine"]
        for phase_name in _LORA_PHASES:
            for layer_idx in range(4):
                for proj_name in _LORA_PROJS:
                    dk = f"fg_v118_lora_{phase_name}_{layer_idx}_{proj_name}_down"
                    uk = f"fg_v118_lora_{phase_name}_{layer_idx}_{proj_name}_up"
                    if hasattr(model, dk) and hasattr(model, uk):
                        lora_params[(phase_name, layer_idx, proj_name)] = (
                            getattr(model, dk), getattr(model, uk)
                        )

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

    for k in range(K):
        # Accumulate notebook read
        x_acc_delta = _acc_notebook_read(
            x, notebook_slots,
            acc_W_q, acc_W_k, acc_W_v, acc_W_o, acc_b_o,
        )
        x = x + x_acc_delta

        # SBP noise
        x = x + (noise[k] * noise_scale).cast(x.dtype)

        # v112b per-position residual gate
        x = x * gate_multiplier_3d.cast(x.dtype)

        be_k  = breath_embed[k].reshape(1, 1, -1).cast(x.dtype)
        x_in  = x + be_k
        x_pre = x

        stk      = staging_mask[:, k, :, :]
        stk_h    = stk.reshape(B, 1, T, T).expand(B, V110_STEP_N_HEADS, T, T)
        combined = stk_h.cast(x.dtype) + head_op_mask.cast(x.dtype)
        combined = combined + attn_bias_btht.cast(combined.dtype)

        cos_k = breath_cos[k]
        sin_k = breath_sin[k]

        # Determine which phase this breath belongs to (Python-level)
        phase_idx  = _breath_phase(k, K_max)
        phase_name = _PHASE_NAMES[phase_idx]

        h = x_in
        for layer_idx, layer in enumerate(layers[:4]):
            if lora_params:
                # Gather LoRA params for this phase and layer
                _lq  = lora_params.get((phase_name, layer_idx, "q"))
                _lk  = lora_params.get((phase_name, layer_idx, "k"))
                _lv  = lora_params.get((phase_name, layer_idx, "v"))
                _lo  = lora_params.get((phase_name, layer_idx, "o"))
                _lw1 = lora_params.get((phase_name, layer_idx, "ffn_w1"))
                _lw2 = lora_params.get((phase_name, layer_idx, "ffn_w2"))
                h = _layer_forward_v118_lora(
                    layer, h, combined, cos_k, sin_k,
                    lora_q_down=_lq[0] if _lq else None,
                    lora_q_up=_lq[1] if _lq else None,
                    lora_k_down=_lk[0] if _lk else None,
                    lora_k_up=_lk[1] if _lk else None,
                    lora_v_down=_lv[0] if _lv else None,
                    lora_v_up=_lv[1] if _lv else None,
                    lora_o_down=_lo[0] if _lo else None,
                    lora_o_up=_lo[1] if _lo else None,
                    lora_w1_down=_lw1[0] if _lw1 else None,
                    lora_w1_up=_lw1[1] if _lw1 else None,
                    lora_w2_down=_lw2[0] if _lw2 else None,
                    lora_w2_up=_lw2[1] if _lw2 else None,
                )
            else:
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

        # Perceiver bottleneck at phase transitions (Python-level gating)
        if perceiver_latents is not None and k in perceiver_trigger:
            h_quant = _perceiver_compress_expand(
                h_quant, perceiver_latents, perceiver_gate, layers[0],
                W_out=perceiver_W_out,
            )

        # delta_gate blend
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


# ---------------------------------------------------------------------------
# JIT compile helpers
# ---------------------------------------------------------------------------

_JIT_V118_CACHE: dict = {}


def _compile_jit_fg_step_v118(
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
    perceiver_enabled: bool = V118_PERCEIVER_ENABLED,
    per_phase_loras: bool = V118_PER_PHASE_LORAS,
    grad_clip: float = 1.0,
):
    """Train step for v118. Same loss as v112b, with v118 forward."""
    key = ("v118", id(model), id(opt), int(K), int(B),
           float(factor_aux_weight), float(calib_weight), float(var_loss_weight),
           float(balance_weight), float(uncertainty_min),
           bool(hard_breath_level), bool(alternation), float(phase_scale),
           int(n_max), int(f_max), int(n_digits),
           str(gate_profile), float(photon_alpha),
           bool(perceiver_enabled), bool(per_phase_loras),
           float(grad_clip))
    if key in _JIT_V118_CACHE:
        return _JIT_V118_CACHE[key]

    fw, aw, vw, bw, um, gc = (
        float(factor_aux_weight), float(calib_weight),
        float(var_loss_weight), float(balance_weight),
        float(uncertainty_min), float(grad_clip),
    )
    params = opt.params
    print(
        f"[JIT] compile v118 step: K={K} B={B} "
        f"profile={gate_profile} alpha={photon_alpha} "
        f"perceiver={perceiver_enabled} loras={per_phase_loras}...",
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
            fg_breathing_forward_v118(
                model, domain_init, node_kinds, staging_mask, head_op_mask,
                noise, noise_scale,
                K=K, n_max=n_max, f_max=f_max, n_digits=n_digits,
                alternation=alternation, phase_scale=phase_scale,
                gate_profile=gate_profile, photon_alpha=photon_alpha,
                perceiver_enabled=perceiver_enabled,
                per_phase_loras=per_phase_loras,
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

        # AMD-safe: single-kernel isfinite, no per-param isnan
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

    _JIT_V118_CACHE[key] = _step
    print(f"[JIT] v118 step ready (cache={len(_JIT_V118_CACHE)})", flush=True)
    return _step


def compile_jit_eval_v118(
    model: Any,
    K: int = V110_STEP3_K_MAX,
    B: int = 8,
    n_max: int = V110_STEP3_N_MAX,
    f_max: int = V110_STEP3_F_MAX,
    n_digits: int = V110_STEP3_N_DIGITS,
    alternation: bool = V110_STEP3_ALTERNATION,
    phase_scale: float = V110_STEP3_PHASE_SCALE,
    gate_profile: str = V118_GATE_PROFILE,
    photon_alpha: float = V110_STEP3_PHOTON_ALPHA,
    perceiver_enabled: bool = V118_PERCEIVER_ENABLED,
    per_phase_loras: bool = V118_PER_PHASE_LORAS,
):
    """JIT'd eval forward for v118 — same signature as v112b eval."""
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
        tree_lh, _, _, _, _ = fg_breathing_forward_v118(
            model, domain_init, node_kinds, staging_mask, head_op_mask,
            eval_noise_zeros, eval_noise_scale,
            K=K, n_max=n_max, f_max=f_max, n_digits=n_digits,
            alternation=alternation, phase_scale=phase_scale,
            gate_profile=gate_profile, photon_alpha=photon_alpha,
            perceiver_enabled=perceiver_enabled,
            per_phase_loras=per_phase_loras,
        )
        final_tree = tree_lh[-1]
        pred_digits = final_tree.argmax(axis=-1)
        eq_per_pos  = (pred_digits == gold_digits.cast(dtypes.int)).cast(dtypes.float)
        eq          = eq_per_pos.prod(axis=-1)
        unobs       = (1 - observed_mask.cast(dtypes.float))
        cell_acc    = (eq * unobs).sum() / (unobs.sum() + 1e-8)
        return pred_digits.realize(), cell_acc.realize()

    return _eval
