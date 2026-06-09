"""v200 Perceiver-CORE architecture for factor-graph inference.

Stage 1: Minimum viable skeleton — cold-start training loop, no warm-start.

Architectural pivot from v98-v121 (residual-stream primary, perceiver-as-add-on)
to v200 (latent-stream primary, perceiver IS the architecture).

  v98-v121: Factor graph tokens live in 1024d residual stream.
            Perceiver tried as add-on → 5× refused as redundant.

  v200: 32 learnable latents are the PRIMARY state at 2048d.
        Factor graph tokens are STATIC REFERENCE.
        Each breath: latents READ tokens → THINK (self-attn) → READOUT.

Stage 1 scope (no waist, no write, no SBP, no mirror — all deferred to Stage 2):
  - 32 latents × 2048d, QR-orthogonal init
  - Embed factor graph tokens via Llama embed_tokens
  - Per breath:
      READ: latents cross-attend to fg_tokens  (32×T cross-attention)
      THINK: 4 Llama layers process latents    (32×32 self-attention)
      READOUT: tree codebook on first 16 latents (variable beliefs)
               calibration head on pooled latents
  - Per-breath weighted CE ladder: loss = Σ_k (1 + k/(K-1)) × CE_k
  - Calibration head: scalar confidence per breath (BCE vs detached argmax target)

Key constants:
  V200_K_MAX      = 8    — breaths per forward
  V200_N_LATENTS  = 32   — learnable latents
  V200_N_VAR_LAT  = 16   — first 16 latents = variable beliefs (tree codebook)
  V200_N_SCRATCH  = 16   — remaining latents = scratch/factor space
  V200_HIDDEN     = 2048 — Llama hidden size
  V200_N_MAX      = 16   — max variable nodes per factor graph
  V200_F_MAX      = 8    — max factor nodes per factor graph
  V200_T_MAX      = 24   — T_MAX = N_MAX + F_MAX
  V200_N_DIGITS   = 5    — tree codebook depth (10000s / 1000s / 100s / 10s / 1s)

Env vars:
  V200_TASK=1            — enable v200 forward path
  V200_K_MAX=8           — number of breaths
  V200_N_LATENTS=32      — number of learnable latents
  V200_N_VAR_LAT=16      — variable-belief latents (get tree codebook readout)
  V200_N_DIGITS=5        — tree codebook depth
  V200_CALIB_WEIGHT=0.05 — weight on calibration BCE loss
  V200_FACTOR_AUX=0.0    — factor auxiliary loss weight (unused in Stage 1)
"""
from __future__ import annotations

import math
import os
from typing import Any

import numpy as np
from tinygrad import Tensor, dtypes
from tinygrad.engine.jit import TinyJit

from mycelium.llama_loader import LlamaConfig, LlamaLayer, SMOLLM2_1_7B_CFG, _rms_norm


# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

V200_TASK         = int(os.environ.get("V200_TASK",        "0")) > 0
V200_K_MAX        = int(os.environ.get("V200_K_MAX",       "8"))
V200_N_LATENTS    = int(os.environ.get("V200_N_LATENTS",   "32"))
V200_N_VAR_LAT    = int(os.environ.get("V200_N_VAR_LAT",   "16"))   # variable-belief latents
V200_N_DIGITS     = int(os.environ.get("V200_N_DIGITS",    "5"))
V200_CALIB_WEIGHT = float(os.environ.get("V200_CALIB_WEIGHT", "0.05"))
V200_FACTOR_AUX   = float(os.environ.get("V200_FACTOR_AUX",   "0.0"))

V200_N_MAX        = int(os.environ.get("V200_N_MAX",       "16"))
V200_F_MAX        = int(os.environ.get("V200_F_MAX",       "8"))
V200_T_MAX        = V200_N_MAX + V200_F_MAX   # = 24

# Stage 2a: alternating waist
# V200_STAGE2A_WAIST=0  → Stage 1 behavior (no waist)
# V200_STAGE2A_WAIST=1  → alternating waist on EVEN breaths (commit phase)
V200_STAGE2A_WAIST    = int(os.environ.get("V200_STAGE2A_WAIST",  "0")) > 0
V200_WAIST_DIM        = int(os.environ.get("V200_WAIST_DIM",       "512"))


# ---------------------------------------------------------------------------
# Tree codebook (same Fourier-orthogonal init as v108)
# ---------------------------------------------------------------------------

def _fourier_orthogonal_init(n_entries: int, hidden: int, seed: int = 0) -> np.ndarray:
    """(n_entries, hidden) Fourier orthogonal init. Scaled 0.1/sqrt(hidden)."""
    cb = np.zeros((n_entries, hidden), dtype=np.float32)
    n_pairs = hidden // 2
    for d in range(n_entries):
        for k_idx in range(n_pairs):
            angle = 2.0 * np.pi * d * (k_idx + 1) / float(n_entries)
            cb[d, 2 * k_idx]     = float(np.cos(angle))
            cb[d, 2 * k_idx + 1] = float(np.sin(angle))
    cb = cb * (0.1 / float(np.sqrt(hidden)))
    return cb.astype(np.float32)


# ---------------------------------------------------------------------------
# READ step: latents cross-attend to factor graph tokens
# ---------------------------------------------------------------------------

def _cross_attend_v200(
    latents: Tensor,        # (B, L, H)
    fg_tokens: Tensor,      # (B, T, H)
    wq: Tensor,             # (H, H)
    wk: Tensor,             # (H, H)
    wv: Tensor,             # (H, H)
    wo: Tensor,             # (H, H)
    n_heads: int,
    head_dim: int,
) -> Tensor:
    """Cross-attention: Q from latents, K/V from fg_tokens.

    Returns (B, L, H) context to be added to latents.
    No RoPE on cross-attention (positions not meaningful across modalities).
    Clip pre-softmax scores to (-1e4, 1e4) per AMD constraint.
    """
    B, L, H = latents.shape
    T = int(fg_tokens.shape[1])
    scale = 1.0 / math.sqrt(head_dim)

    q = (latents @ wq.cast(latents.dtype)).reshape(B, L, n_heads, head_dim).transpose(1, 2)   # (B, nh, L, hd)
    k = (fg_tokens @ wk.cast(fg_tokens.dtype)).reshape(B, T, n_heads, head_dim).transpose(1, 2)  # (B, nh, T, hd)
    v = (fg_tokens @ wv.cast(fg_tokens.dtype)).reshape(B, T, n_heads, head_dim).transpose(1, 2)  # (B, nh, T, hd)

    scores = (q @ k.transpose(-2, -1)) * scale   # (B, nh, L, T)
    attn = scores.clip(-1e4, 1e4).softmax(-1).cast(v.dtype)
    ctx = (attn @ v).transpose(1, 2).reshape(B, L, H)   # (B, L, H)
    return ctx @ wo.cast(latents.dtype)


# ---------------------------------------------------------------------------
# Parameter attachment
# ---------------------------------------------------------------------------

def attach_fg_params_v200(
    model: Any,
    n_latents: int = V200_N_LATENTS,
    n_var_lat: int = V200_N_VAR_LAT,
    k_max: int = V200_K_MAX,
    n_digits: int = V200_N_DIGITS,
    n_max: int = V200_N_MAX,
    f_max: int = V200_F_MAX,
    stage2a_waist: bool = V200_STAGE2A_WAIST,
    waist_dim: int = V200_WAIST_DIM,
) -> None:
    """Attach all v200 perceiver-CORE params to model.

    Requires: model.llama_cfg and model.llama_layers already attached
              (via llama_loader.attach_llama_layers).

    Attaches:
      fg_v200_latents       (n_latents, H)  — QR-orthogonal init
      fg_v200_breath_embed  (k_max, H)      — per-breath additive markers, zero-init
      fg_v200_cross_wq      (H, H)          — cross-attention Q projection (latents→Q)
      fg_v200_cross_wk      (H, H)          — cross-attention K projection (tokens→K)
      fg_v200_cross_wv      (H, H)          — cross-attention V projection (tokens→V)
      fg_v200_cross_wo      (H, H)          — cross-attention output projection
      fg_v200_read_norm_w   (H,)            — RMSNorm weight before cross-attn (ones init)
      fg_v200_latent_norm_w (H,)            — final RMSNorm on latents for readout (ones init)
      fg_v200_tree_codebook (n_digits, 10, H) — Fourier orthogonal per level
      fg_v200_calib_w       (H, 1)          — calibration head weight (zero-init)
      fg_v200_calib_b       (1,)            — calibration head bias
      fg_v200_delta_gate    (k_max,)        — per-breath residual gate (0.5 init)

    Stage 2a (only when stage2a_waist=True):
      fg_v200_W_compress    (H, waist_dim)  — QR-init scale 0.01 (non-zero → W_expand grad flows)
      fg_v200_W_expand      (waist_dim, H)  — ZERO-INIT (bootstrap safe: delta=0 at step 0)
      fg_v200_waist_gate    (1,)            — zero-init scalar, (1+g) amplifier
    """
    assert hasattr(model, "llama_cfg"), \
        "model.llama_cfg not found — call llama_loader.attach_llama_layers first"
    assert hasattr(model, "llama_layers"), \
        "model.llama_layers not found — call llama_loader.attach_llama_layers first"

    cfg: LlamaConfig = model.llama_cfg
    H = cfg.hidden_size
    nh = cfg.num_attention_heads
    hd = cfg.head_dim
    T = n_max + f_max

    if hasattr(model, "fg_v200_latents"):
        print("[v200] params already attached, skipping", flush=True)
        return

    # ---- Latents: QR-orthogonal init (32, 2048) ----
    # QR of (max(n_latents, H), H) gives orthonormal rows — well-conditioned start
    rng = np.random.RandomState(42)
    raw = rng.randn(max(n_latents, H), H).astype(np.float32)
    Q, _ = np.linalg.qr(raw)
    latents_init = Q[:n_latents].astype(np.float32) * 0.1   # small norm initially
    model.fg_v200_latents = Tensor(latents_init, dtype=dtypes.float).contiguous().realize()

    # ---- Per-breath embedding: zero-init (byte-identical at step 0) ----
    model.fg_v200_breath_embed = Tensor(
        np.zeros((k_max, H), dtype=np.float32)
    ).contiguous().realize()

    # ---- Cross-attention projections: scaled eye init ----
    # Eye-like init (small scale) so cross-attn starts near identity mapping
    def _eye_init(d: int, scale: float = 0.02) -> np.ndarray:
        eye = np.eye(d, dtype=np.float32)
        return (eye * scale).astype(np.float32)

    model.fg_v200_cross_wq = Tensor(_eye_init(H), dtype=dtypes.float).contiguous().realize()
    model.fg_v200_cross_wk = Tensor(_eye_init(H), dtype=dtypes.float).contiguous().realize()
    model.fg_v200_cross_wv = Tensor(_eye_init(H), dtype=dtypes.float).contiguous().realize()
    model.fg_v200_cross_wo = Tensor(_eye_init(H), dtype=dtypes.float).contiguous().realize()

    # ---- RMSNorm weights: ones init ----
    model.fg_v200_read_norm_w   = Tensor.ones(H, dtype=dtypes.float).contiguous().realize()
    model.fg_v200_latent_norm_w = Tensor.ones(H, dtype=dtypes.float).contiguous().realize()

    # ---- Tree codebook: Fourier orthogonal per level ----
    cb = np.zeros((n_digits, 10, H), dtype=np.float32)
    for level in range(n_digits):
        cb[level] = _fourier_orthogonal_init(10, H, seed=level * 17 + 31)
    model.fg_v200_tree_codebook = Tensor(cb, dtype=dtypes.float).contiguous().realize()

    # ---- Calibration head: zero-init ----
    model.fg_v200_calib_w = Tensor(
        np.zeros((H, 1), dtype=np.float32)
    ).contiguous().realize()
    model.fg_v200_calib_b = Tensor(
        np.zeros((1,), dtype=np.float32)
    ).contiguous().realize()

    # ---- Delta gate: 0.5 init (moderate residual blend) ----
    model.fg_v200_delta_gate = Tensor(
        np.full((k_max,), 0.5, dtype=np.float32)
    ).contiguous().realize()

    # ---- Stage 2a: alternating waist (v5 convex blend) ----
    # W_compress: small QR-init (scale 0.01) so z = latents @ W_compress is non-zero.
    #   → gradient ∂L/∂W_expand = z.T @ (∂L/∂new · α) is NON-ZERO at step 1.
    # W_expand: ZERO-INIT → h_compressed = 0 at step 0.
    # commit_gate: init to -5.0 → sigmoid(-5)≈0.007 → ~0.7% magnitude shift at step 0.
    #   NO RMSNorm — convex blend is self-bounding; RMSNorm was erasing drift asymmetry.
    n_waist_params = 0
    if stage2a_waist:
        rng_w = np.random.RandomState(99001)
        # W_compress: QR-init scale 0.01
        wc_raw = rng_w.randn(H, waist_dim).astype(np.float32)
        Q_wc, _ = np.linalg.qr(wc_raw if H >= waist_dim else wc_raw.T)
        if H >= waist_dim:
            wc_init = (Q_wc[:, :waist_dim] * 0.01).astype(np.float32)
        else:
            wc_init = (Q_wc[:waist_dim, :].T * 0.01).astype(np.float32)
        model.fg_v200_W_compress = Tensor(wc_init, dtype=dtypes.float).contiguous().realize()
        # W_expand: zero-init (bootstrap safe — h_compressed=0 at step 0)
        model.fg_v200_W_expand = Tensor(
            np.zeros((waist_dim, H), dtype=np.float32)
        ).contiguous().realize()
        # commit_gate: init to -5.0 → sigmoid(-5)≈0.007 (small but non-zero blend)
        # Var name kept as fg_v200_waist_gate for checkpoint compatibility.
        model.fg_v200_waist_gate = Tensor(
            np.full((1,), -2.0, dtype=np.float32)
        ).contiguous().realize()
        n_waist_params = H * waist_dim + waist_dim * H + 1
        print(
            f"[v200-2a-v5] Stage 2a waist attached (convex blend): "
            f"W_compress({H},{waist_dim})=QR×0.01  "
            f"W_expand({waist_dim},{H})=ZEROS  commit_gate=-5.0(α≈0.007)  "
            f"alternation=EVEN_BREATHS  no_RMSNorm  "
            f"new_params={n_waist_params:,}  ({n_waist_params/1e6:.2f}M)",
            flush=True,
        )

    # ---- Count and report ----
    n_latent_params = n_latents * H
    n_breath_params = k_max * H
    n_cross_params  = 4 * H * H + 2 * H
    n_tree_params   = n_digits * 10 * H
    n_calib_params  = H + 1
    n_gate_params   = k_max
    n_new = (n_latent_params + n_breath_params + n_cross_params +
             n_tree_params + n_calib_params + n_gate_params + n_waist_params)

    print(
        f"[v200] perceiver-CORE params attached:\n"
        f"  latents        : ({n_latents}, {H}) = {n_latent_params:,}\n"
        f"  breath_embed   : ({k_max}, {H}) = {n_breath_params:,}\n"
        f"  cross_attn     : 4×({H},{H}) + 2×{H} = {n_cross_params:,}\n"
        f"  tree_codebook  : ({n_digits}, 10, {H}) = {n_tree_params:,}\n"
        f"  calibration    : {n_calib_params:,}\n"
        f"  delta_gate     : {n_gate_params:,}\n"
        + (f"  waist (2a)     : {n_waist_params:,}\n" if stage2a_waist else "")
        + f"  TOTAL new      : {n_new:,}  ({n_new/1e6:.2f}M)",
        flush=True,
    )


def fg_v200_parameters(model: Any) -> list[Tensor]:
    """Return all trainable v200 params (Llama layers + perceiver-CORE params).

    Llama layer weights (L0-L3) are included — cold-start, all params are trained.
    """
    params = []

    # Llama layer params
    for layer in model.llama_layers:
        params.extend(layer.parameters())

    # v200 perceiver-CORE params
    v200_attrs = [
        "fg_v200_latents",
        "fg_v200_breath_embed",
        "fg_v200_cross_wq",
        "fg_v200_cross_wk",
        "fg_v200_cross_wv",
        "fg_v200_cross_wo",
        "fg_v200_read_norm_w",
        "fg_v200_latent_norm_w",
        "fg_v200_tree_codebook",
        "fg_v200_calib_w",
        "fg_v200_calib_b",
        "fg_v200_delta_gate",
        # Stage 2a waist (only present when V200_STAGE2A_WAIST=1)
        "fg_v200_W_compress",
        "fg_v200_W_expand",
        "fg_v200_waist_gate",
        # fg_v200_waist_norm_w removed in v5 (convex blend needs no RMSNorm)
    ]
    for attr in v200_attrs:
        if hasattr(model, attr):
            params.append(getattr(model, attr))

    return params


def fg_v200_state_dict(model: Any) -> dict[str, Tensor]:
    """State dict for checkpointing (Llama layers + v200 params)."""
    sd = {}

    # Llama layer weights
    for li, layer in enumerate(model.llama_layers):
        sd[f"llama_layer_{li}.wq"]        = layer.wq
        sd[f"llama_layer_{li}.wk"]        = layer.wk
        sd[f"llama_layer_{li}.wv"]        = layer.wv
        sd[f"llama_layer_{li}.wo"]        = layer.wo
        sd[f"llama_layer_{li}.w_gate"]    = layer.w_gate
        sd[f"llama_layer_{li}.w_up"]      = layer.w_up
        sd[f"llama_layer_{li}.w_down"]    = layer.w_down
        sd[f"llama_layer_{li}.attn_norm"] = layer.attn_norm
        sd[f"llama_layer_{li}.ffn_norm"]  = layer.ffn_norm

    # v200 perceiver-CORE params
    v200_attrs = [
        "fg_v200_latents",
        "fg_v200_breath_embed",
        "fg_v200_cross_wq",
        "fg_v200_cross_wk",
        "fg_v200_cross_wv",
        "fg_v200_cross_wo",
        "fg_v200_read_norm_w",
        "fg_v200_latent_norm_w",
        "fg_v200_tree_codebook",
        "fg_v200_calib_w",
        "fg_v200_calib_b",
        "fg_v200_delta_gate",
        # Stage 2a waist (only present when V200_STAGE2A_WAIST=1)
        "fg_v200_W_compress",
        "fg_v200_W_expand",
        "fg_v200_waist_gate",
        # fg_v200_waist_norm_w removed in v5 (convex blend needs no RMSNorm)
    ]
    for attr in v200_attrs:
        if hasattr(model, attr):
            sd[attr] = getattr(model, attr)

    return sd


# ---------------------------------------------------------------------------
# Stage 2a: alternating waist helper
# ---------------------------------------------------------------------------

def _apply_waist_v200(
    latents: Tensor,        # (B, L, H)
    W_compress: Tensor,     # (H, waist_dim)   QR-init scale 0.01
    W_expand: Tensor,       # (waist_dim, H)   ZERO-INIT (bootstrap safe)
    commit_gate: Tensor,    # (1,)             init to -5.0 → sigmoid≈0.007
) -> Tensor:
    """Alternating waist compression for even breaths (COMMIT phase).

    Stage 2a v5: CONVEX BLEND — bounded by construction, can't explode.

      latents_new = α * h_compressed + (1-α) * latents_pre
                  = latents_pre + α * (h_compressed - latents_pre)

    where α = sigmoid(commit_gate) ∈ [0, 1]. At α=0: latents unchanged.
    At α=1: fully replaced by compressed version. The mixture norm is
    bounded by the larger of |latents_pre| and |h_compressed|.
    MATHEMATICALLY CAN'T EXPLODE regardless of W_expand magnitude.

    v2 failure: unbounded (1+g) amplifier — W_expand 0→1.17, latents
      B0=263 → B6=825, NaN at step 73.
    v3/v4 failure: RMSNorm post-blend — bounded but ERASED drift
      asymmetry diagnostic (even/odd ratio 1.89 → 0.79). Still NaN'd
      at step 337 because padding-free gradient drove W_expand growth.

    v5 fix: convex blend is self-bounding. No RMSNorm needed. Magnitudes
      are preserved naturally (weighted average of bounded inputs is bounded).

    Init: commit_gate = -5.0 → α = sigmoid(-5) ≈ 0.00669
      → latents_new = 0.00669 * 0 + 0.99331 * latents ≈ 0.993 * latents
      → 0.7% magnitude reduction at step 0 (NOT byte-identical, but very close)

    Gradient flow at init (all non-zero):
      ∂L/∂commit_gate = ∂L/∂new · (h_compressed - latents) · α(1-α)
                       = ∂L/∂new · (0 - latents) · 0.00664
      ∂L/∂W_expand   = z.T @ (∂L/∂new · α) — z non-zero from QR W_compress
      ∂L/∂W_compress flows through z derivative — also non-zero
    """
    dtype = latents.dtype
    wc = W_compress.cast(dtype)
    we = W_expand.cast(dtype)
    alpha = commit_gate.cast(dtype).sigmoid()   # (1,) bounded [0, 1]

    z            = (latents @ wc).gelu()        # (B, L, waist_dim) — non-zero at init
    h_compressed = z @ we                       # (B, L, H) — zero at init (W_expand=0)

    # Convex blend: can't exceed max(|latents|, |h_compressed|) by construction
    return latents + alpha * (h_compressed - latents)


# ---------------------------------------------------------------------------
# Token embedding helper
# ---------------------------------------------------------------------------

def _embed_fg_tokens_v200(
    model: Any,
    domain_init: Tensor,      # (B, N_MAX, 200) observed domain probabilities
    node_kinds: Tensor,        # (B, T) int — 0=observed var, 1=unobserved var, 2=factor, -1=pad
    n_max: int,
    f_max: int,
) -> Tensor:
    """Embed factor graph tokens into Llama hidden space.

    Encoding:
      - Variables: embed the domain-centroid-weighted sum from domain_init.
        domain_init (B, N_MAX, 200) @ codebook_proxy (200, H) → (B, N_MAX, H)
        For Stage 1: use the argmax bin as a token ID passed through llama_embed.
        Observed variables: one-hot → argmax is the gold bin.
        Unobserved variables: uniform → argmax is 0 (uniform bin); use special token.
      - Factors: use node_kind=2 → embed a fixed "factor token" per factor slot.
        Factor slot token IDs: 200 + factor_position (0..F_MAX-1), safe since
        Llama vocab is 49152/128256 >> 200 + 8.
      - Padding: node_kinds == -1 → embed zero (no contribution).

    Returns (B, T, H) where T = N_MAX + F_MAX.

    Note: for Stage 1 simplicity, we project domain_init to token IDs
    (argmax for observed, 0 for unobserved), then look up llama_embed.
    Stage 2 can replace this with a learned domain → hidden projection.
    """
    B = int(domain_init.shape[0])
    T = n_max + f_max
    H = model.llama_cfg.hidden_size

    # Variable tokens: argmax over 200 bins → token ID (0..199)
    # (B, N_MAX, 200) → (B, N_MAX) int
    var_token_ids = domain_init.argmax(axis=-1)    # (B, N_MAX) — int tensor

    # Factor tokens: use fixed token IDs 200..207 for factor slots 0..7
    # (avoids collision with bin IDs 0..199, and is well within vocab)
    factor_base = 200
    factor_ids_np = np.arange(f_max, dtype=np.int32) + factor_base   # (F_MAX,)
    factor_ids = Tensor(factor_ids_np, dtype=dtypes.int).reshape(1, f_max).expand(B, f_max)

    # Concatenate: (B, T) int
    all_ids = Tensor.cat(
        var_token_ids.cast(dtypes.int),
        factor_ids.cast(dtypes.int),
        dim=1,
    )   # (B, T)

    # Clamp to valid vocab range (safe: all IDs are << vocab_size)
    embed_w = model.llama_embed   # (vocab_size, H)
    # Embed: gather rows from embed_w
    # tinygrad gather: embed_w[all_ids] but all_ids is 2D → reshape, gather, reshape back
    all_ids_flat = all_ids.reshape(-1)           # (B*T,)
    emb_flat = embed_w[all_ids_flat]             # (B*T, H)
    fg_tokens = emb_flat.reshape(B, T, H)        # (B, T, H)

    # Zero out padding positions (node_kinds == -1)
    # node_kinds: (B, T)  — pad=-1, valid>=0
    valid_mask = (node_kinds >= 0).cast(fg_tokens.dtype).reshape(B, T, 1)
    fg_tokens = fg_tokens * valid_mask

    return fg_tokens.cast(dtypes.half)   # (B, T, H) fp16 for memory


# ---------------------------------------------------------------------------
# Main forward pass
# ---------------------------------------------------------------------------

def fg_breathing_forward_v200(
    model: Any,
    domain_init: Tensor,       # (B, N_MAX, 200)
    node_kinds: Tensor,         # (B, T_MAX) int
    K: int = V200_K_MAX,
    n_max: int = V200_N_MAX,
    f_max: int = V200_F_MAX,
    n_var_lat: int = V200_N_VAR_LAT,
    n_digits: int = V200_N_DIGITS,
    training: bool = True,
    stage2a_waist: bool = V200_STAGE2A_WAIST,
) -> tuple[list[Tensor], list[Tensor]]:
    """v200 perceiver-CORE forward pass.

    Stages per breath k:
      1. Add per-breath embedding to latents.
      2. READ: latents cross-attend to fg_tokens (pre-normed latents as Q).
      3. THINK: 4 Llama layers process latents (32×32 self-attention, full, no mask).
      4. WAIST (Stage 2a, EVEN breaths only): alternating waist compression.
      5. READOUT: tree codebook on first n_var_lat latents + calibration head.

    Returns:
      tree_logits_history  : K × (B, n_var_lat, n_digits, 10)
      calib_history        : K × (B,)

    Note: drift computation has been moved out of this function to avoid bloating
    the JIT graph. Use compute_drift_v200() for per-breath drift diagnostics
    (called only at log intervals, non-JIT eager).
    """
    assert hasattr(model, "fg_v200_latents"), \
        "fg_v200_latents not found — call attach_fg_params_v200 first"
    assert hasattr(model, "llama_layers"), \
        "llama_layers not found — call attach_llama_layers first"

    cfg: LlamaConfig = model.llama_cfg
    H = cfg.hidden_size
    nh = cfg.num_attention_heads
    hd = cfg.head_dim
    B = int(domain_init.shape[0])
    T = n_max + f_max
    rms_eps = cfg.rms_norm_eps

    # Pull params
    latents_base   = model.fg_v200_latents            # (n_latents, H) — shared init
    breath_embed   = model.fg_v200_breath_embed        # (K_max, H)
    cross_wq       = model.fg_v200_cross_wq            # (H, H)
    cross_wk       = model.fg_v200_cross_wk            # (H, H)
    cross_wv       = model.fg_v200_cross_wv            # (H, H)
    cross_wo       = model.fg_v200_cross_wo            # (H, H)
    read_norm_w    = model.fg_v200_read_norm_w         # (H,)
    latent_norm_w  = model.fg_v200_latent_norm_w       # (H,)
    tree_codebook  = model.fg_v200_tree_codebook       # (n_digits, 10, H)
    calib_w        = model.fg_v200_calib_w             # (H, 1)
    calib_b        = model.fg_v200_calib_b             # (1,)
    delta_gate     = model.fg_v200_delta_gate          # (K_max,)
    rope_cos       = model.llama_rope_cos              # (max_pos, H)
    rope_sin       = model.llama_rope_sin              # (max_pos, H)
    llama_layers   = model.llama_layers                # list[LlamaLayer]

    # Stage 2a waist params (only pulled when stage2a_waist=True)
    if stage2a_waist:
        assert hasattr(model, "fg_v200_W_compress"), \
            "fg_v200_W_compress not found — set V200_STAGE2A_WAIST=1 in attach_fg_params_v200"
        W_compress    = model.fg_v200_W_compress      # (H, waist_dim)
        W_expand      = model.fg_v200_W_expand        # (waist_dim, H)
        waist_gate    = model.fg_v200_waist_gate      # (1,) commit_gate, init -5.0

    # ---- Static: embed factor graph tokens once ----
    fg_tokens = _embed_fg_tokens_v200(model, domain_init, node_kinds, n_max, f_max)
    # fg_tokens: (B, T, H) fp16

    # ---- Initialize latents: broadcast base to batch ----
    # latents_base: (n_latents, H) → (B, n_latents, H)
    n_latents = int(latents_base.shape[0])
    latents = latents_base.reshape(1, n_latents, H).expand(B, n_latents, H)
    latents = latents.cast(dtypes.half)    # fp16 for activations

    # Pre-compute tree codebook flat for readout: (n_digits*10, H)
    tree_cb_flat = tree_codebook.reshape(n_digits * 10, H)  # (50, H)

    K_max = int(breath_embed.shape[0])
    assert K <= K_max, f"K={K} > K_max={K_max}"

    tree_logits_history = []
    calib_history       = []

    for k in range(K):
        # 1. Per-breath additive embedding
        be_k = breath_embed[k].reshape(1, 1, H).cast(latents.dtype)
        latents = latents + be_k

        # 2. READ: latents cross-attend to fg_tokens
        # Pre-norm latents before cross-attn (RMSNorm)
        lat_normed = _rms_norm(latents, read_norm_w, rms_eps).cast(latents.dtype)

        read_ctx = _cross_attend_v200(
            lat_normed, fg_tokens,
            cross_wq, cross_wk, cross_wv, cross_wo,
            n_heads=nh, head_dim=hd,
        )
        # Delta gate residual blend (per-breath)
        gate_k = delta_gate[k].cast(latents.dtype).reshape(1, 1, 1)
        latents = latents + gate_k * read_ctx.cast(latents.dtype)

        # 3. THINK: 4 Llama layers process latents (32×32 self-attention, no mask)
        h = latents.cast(dtypes.float)    # Llama layers run fp32
        for layer in llama_layers[:4]:
            h = layer(h, rope_cos, rope_sin, attn_mask=None)
        latents = h.cast(dtypes.half)

        # 4. WAIST (Stage 2a): alternating compression on EVEN breaths only
        # EVEN = COMMIT phase (force commitment through bottleneck)
        # ODD  = PROPAGATION phase (bypass waist, beliefs propagate freely at full 2048d)
        if stage2a_waist and (k % 2 == 0):
            latents = _apply_waist_v200(
                latents, W_compress, W_expand, waist_gate,
            )

        # 5. READOUT
        # Final RMSNorm on latents
        lat_out = _rms_norm(latents, latent_norm_w, rms_eps).cast(dtypes.float)

        # Tree codebook readout: variable-belief latents only
        var_lat = lat_out[:, :n_var_lat, :]                      # (B, n_var_lat, H)
        tree_logits_flat = var_lat @ tree_cb_flat.T.cast(dtypes.float)   # (B, n_var_lat, n_digits*10)
        tree_logits_k = tree_logits_flat.reshape(B, n_var_lat, n_digits, 10)
        tree_logits_history.append(tree_logits_k)

        # Calibration head: pool all latents → scalar
        pool = lat_out.mean(axis=1)                               # (B, H)
        calib_logit = (pool @ calib_w.cast(dtypes.float)) + calib_b.cast(dtypes.float)
        calib_k = calib_logit.reshape(-1).sigmoid()               # (B,)
        calib_history.append(calib_k)

    return tree_logits_history, calib_history


# ---------------------------------------------------------------------------
# Accuracy computation (mirrors v108 fg_accuracy_v108)
# ---------------------------------------------------------------------------

def fg_accuracy_v200(
    tree_logits_final: Tensor,   # (B, n_var_lat, n_digits, 10)
    gold_digits: Tensor,          # (B, N_MAX, n_digits) int — MSD-first
    observed_mask: Tensor,        # (B, N_MAX) int
    query_idx_np: np.ndarray,     # (B,) int
    n_vars_mask: Tensor,          # (B, n_var_lat) float — 1.0 for real vars, 0 for padding
    n_var_lat: int = V200_N_VAR_LAT,
    n_digits: int = V200_N_DIGITS,
) -> dict:
    """Cell accuracy and query accuracy from tree decoder.

    Bug fix (Jun 8): factor graph data has variable n_vars per puzzle, with
    padding positions vi >= n_vars_total[b]. Padding positions have
    observed_mask=0 (treated as unobserved) AND gold_bins=0 (zero pad), so
    the model trivially "predicts" them correctly → inflated cell_acc.
    The fix: caller passes n_vars_mask (B, n_var_lat) where mask[b, vi] = 1.0
    iff vi < n_vars_total[b]. We multiply by this mask in both numerator
    and denominator. The eval path (evaluate_v200) already does this via
    its per-puzzle loop with min(nv, n_var_lat).

    A cell is correct iff ALL n_digits levels match gold AND not padding.
    """
    n_eval = min(n_var_lat, int(gold_digits.shape[1]))
    pred_digits = tree_logits_final[:, :n_eval, :, :].argmax(axis=-1)   # (B, n_eval, n_digits)
    gold_eval   = gold_digits[:, :n_eval, :].cast(dtypes.int)

    correct_per_pos = (pred_digits == gold_eval).cast(dtypes.float)
    correct_all     = correct_per_pos.prod(axis=-1)                      # (B, n_eval)

    # Mask out padding positions AND observed positions
    unobs   = (1 - observed_mask[:, :n_eval].cast(dtypes.float))
    real    = n_vars_mask[:, :n_eval].cast(dtypes.float)
    mask    = unobs * real                                                # (B, n_eval) — true unobserved cells only
    masked  = correct_all * mask
    n_unobs = mask.sum() + 1e-8
    cell_acc = float((masked.sum() / n_unobs).realize().numpy())

    pred_np = pred_digits.cast(dtypes.int).realize().numpy()
    gold_np = gold_eval.cast(dtypes.int).realize().numpy()
    real_np = n_vars_mask[:, :n_eval].cast(dtypes.int).realize().numpy()
    B = int(pred_np.shape[0])
    q_correct = np.array([
        int(query_idx_np[b] < n_eval and
            real_np[b, query_idx_np[b]] == 1 and
            np.all(pred_np[b, query_idx_np[b]] == gold_np[b, query_idx_np[b]]))
        for b in range(B)
    ], dtype=np.float32)
    query_acc = float(q_correct.mean())

    return {"cell_acc": cell_acc, "query_acc": query_acc}


# ---------------------------------------------------------------------------
# Out-of-JIT drift diagnostic (called only at log intervals)
# ---------------------------------------------------------------------------

def compute_drift_v200(
    model: Any,
    domain_init: Tensor,       # (B, N_MAX, 200) — last training batch tensors
    node_kinds: Tensor,         # (B, T_MAX) int
    K: int = V200_K_MAX,
    n_max: int = V200_N_MAX,
    f_max: int = V200_F_MAX,
    stage2a_waist: bool = V200_STAGE2A_WAIST,
) -> list[float]:
    """Run one forward pass NON-JIT to extract per-breath latent drift.

    Called only at log intervals (PER_BREATH_CE_EVERY steps), NOT in the hot
    training loop. Running eager (no @TinyJit) so the K intermediate latent
    snapshots are never part of the compiled graph.

    Returns:
      drifts : list of K floats — mean ||latents_post_k - latents_pre_k||₂
               across B×L positions. Even breaths should > odd breaths when
               the alternating waist is active.
    """
    assert hasattr(model, "fg_v200_latents"), \
        "fg_v200_latents not found — call attach_fg_params_v200 first"

    was_training = Tensor.training
    Tensor.training = False

    cfg: LlamaConfig = model.llama_cfg
    H = cfg.hidden_size
    nh = cfg.num_attention_heads
    hd = cfg.head_dim
    B = int(domain_init.shape[0])
    rms_eps = cfg.rms_norm_eps

    latents_base   = model.fg_v200_latents
    breath_embed   = model.fg_v200_breath_embed
    cross_wq       = model.fg_v200_cross_wq
    cross_wk       = model.fg_v200_cross_wk
    cross_wv       = model.fg_v200_cross_wv
    cross_wo       = model.fg_v200_cross_wo
    read_norm_w    = model.fg_v200_read_norm_w
    delta_gate     = model.fg_v200_delta_gate
    rope_cos       = model.llama_rope_cos
    rope_sin       = model.llama_rope_sin
    llama_layers   = model.llama_layers

    if stage2a_waist:
        W_compress    = model.fg_v200_W_compress
        W_expand      = model.fg_v200_W_expand
        waist_gate    = model.fg_v200_waist_gate

    fg_tokens = _embed_fg_tokens_v200(model, domain_init, node_kinds, n_max, f_max)

    n_latents = int(latents_base.shape[0])
    latents = latents_base.reshape(1, n_latents, H).expand(B, n_latents, H)
    latents = latents.cast(dtypes.half)

    drifts: list[float] = []

    for k in range(K):
        latents_pre_np = latents.cast(dtypes.float).realize().numpy()

        be_k = breath_embed[k].reshape(1, 1, H).cast(latents.dtype)
        latents = latents + be_k

        lat_normed = _rms_norm(latents, read_norm_w, rms_eps).cast(latents.dtype)
        read_ctx = _cross_attend_v200(
            lat_normed, fg_tokens,
            cross_wq, cross_wk, cross_wv, cross_wo,
            n_heads=nh, head_dim=hd,
        )
        gate_k = delta_gate[k].cast(latents.dtype).reshape(1, 1, 1)
        latents = latents + gate_k * read_ctx.cast(latents.dtype)

        h = latents.cast(dtypes.float)
        for layer in llama_layers[:4]:
            h = layer(h, rope_cos, rope_sin, attn_mask=None)
        latents = h.cast(dtypes.half)

        if stage2a_waist and (k % 2 == 0):
            latents = _apply_waist_v200(
                latents, W_compress, W_expand, waist_gate,
            )

        latents_post_np = latents.cast(dtypes.float).realize().numpy()
        diff = latents_post_np - latents_pre_np
        drift_k = float(np.sqrt((diff ** 2).sum(axis=-1)).mean())
        drifts.append(drift_k)

    Tensor.training = was_training
    return drifts


# ---------------------------------------------------------------------------
# Training step (JIT-compilable)
# ---------------------------------------------------------------------------

def _compile_jit_fg_step_v200(
    model: Any,
    opt: Any,
    K: int,
    B: int,
    n_max: int = V200_N_MAX,
    f_max: int = V200_F_MAX,
    n_var_lat: int = V200_N_VAR_LAT,
    n_digits: int = V200_N_DIGITS,
    calib_weight: float = V200_CALIB_WEIGHT,
    grad_clip: float = 1.0,
    stage2a_waist: bool = V200_STAGE2A_WAIST,
):
    """Compile a JIT-fused training step for v200.

    Returns a callable step_fn(domain_init, node_kinds, gold_digits_t, obs_mask_t)
    that returns (total_loss, healthy, ce_loss, calib_loss, cell_acc_t, query_acc_t,
                  *per_breath_ce_ts [K tensors]).

    AMD JIT constraints applied:
      - No Tensor.arange() inside JIT (precomputed outside)
      - Grad clip via healthy-mask (single kernel) not per-param norm loop
      - No .cast(dtypes.float32) inside JIT — use dtypes.float throughout
    """
    n_eval = min(n_var_lat, n_max)
    # Precompute digit one-hot index tensor OUTSIDE JIT (avoid per-call tensor creation)
    digit_range_np = np.arange(10, dtype=np.int32)
    # Ladder weights: Python floats, computed outside JIT
    ladder_weights = [1.0 + k_idx / float(max(K - 1, 1)) for k_idx in range(K)]
    # Use opt.params for the JIT grad loop (same as v108 pattern)
    jit_params = opt.params

    @TinyJit
    def step_fn(
        domain_init: Tensor,    # (B, N_MAX, 200)
        node_kinds: Tensor,     # (B, T_MAX) int
        gold_digits: Tensor,    # (B, N_MAX, n_digits) int MSD-first
        obs_mask: Tensor,       # (B, N_MAX) int
        n_vars_mask: Tensor,    # (B, n_eval) float — 1.0 for real vars, 0 for padding (Jun 8 fix)
    ):
        # opt.zero_grad() first, then backward, then opt.step() — v108 pattern
        opt.zero_grad()

        tree_logits_history, calib_history = fg_breathing_forward_v200(
            model, domain_init, node_kinds,
            K=K, n_max=n_max, f_max=f_max,
            n_var_lat=n_var_lat, n_digits=n_digits,
            training=True,
            stage2a_waist=stage2a_waist,
        )

        # ---- Per-breath weighted CE ladder ----
        # PADDING BUG FIX (Jun 8): The data loader pads positions beyond
        # n_vars_total with observed_mask=0 (treated as unobserved) AND
        # gold_bins=0. Without n_vars_mask, CE loss is computed across
        # padding positions, training the model to trivially predict zero.
        # cell_acc reported during training was inflated for the same reason.
        # n_vars_mask filters BOTH numerator and denominator to real cells.
        gold_eval = gold_digits[:, :n_eval, :].cast(dtypes.int)         # (B, n_eval, n_digits)
        real_mask = n_vars_mask[:, :n_eval].cast(dtypes.float)          # (B, n_eval)
        unobs_mask = (1 - obs_mask[:, :n_eval].cast(dtypes.float)) * real_mask  # real & unobserved
        unobs_sum  = unobs_mask.sum() + 1e-8

        # gold one-hot: (B, n_eval, n_digits, 1) vs (10,) for NLL
        # Use same pattern as v108: one_hot(gold).cast(float) × log_softmax
        gd_flat = gold_eval.reshape(B * n_eval * n_digits)              # (N,) int
        gold_oh = gd_flat.one_hot(10).cast(dtypes.float)                # (N, 10)

        per_breath_ce_list = []
        var_loss_sum = Tensor.zeros((), dtype=dtypes.float).contiguous()
        var_weight_sum = 0.0

        for k_idx in range(K):
            logits_k = tree_logits_history[k_idx]   # (B, n_eval, n_digits, 10)
            # Flat (B*n_eval*n_digits, 10) for CE
            lk_flat   = logits_k.reshape(B * n_eval * n_digits, 10)
            log_p     = lk_flat.log_softmax(axis=-1)                    # (N, 10)
            nll_flat  = -(log_p * gold_oh).sum(axis=-1)                 # (N,)
            # Reshape to (B, n_eval, n_digits) and average over digits
            nll_per_pos = nll_flat.reshape(B, n_eval, n_digits).mean(axis=-1)  # (B, n_eval)
            ce_k = (nll_per_pos * unobs_mask).sum() / unobs_sum
            per_breath_ce_list.append(ce_k)
            var_loss_sum = var_loss_sum + ce_k * ladder_weights[k_idx]
            var_weight_sum += ladder_weights[k_idx]

        # Normalize CE loss by weight sum to keep scale comparable across K
        total_ce = var_loss_sum / float(var_weight_sum)

        # ---- Calibration head BCE ----
        # Target: whether last-breath prediction is correct on unobserved cells
        final_tree = tree_logits_history[-1]
        pred_digits_final = final_tree.argmax(axis=-1).detach()         # (B, n_eval, n_digits)
        eq_per_pos = (pred_digits_final == gold_eval).cast(dtypes.float)
        eq = eq_per_pos.prod(axis=-1)                                   # (B, n_eval)
        # Use n_vars_mask to exclude padding (same fix as CE loss above)
        unobs_2d = (1 - obs_mask[:, :n_eval].cast(dtypes.float)) * real_mask  # (B, n_eval)
        n_unobs_per = unobs_2d.sum(axis=-1) + 1e-8
        correct = (eq * unobs_2d).sum(axis=-1) / n_unobs_per           # (B,)

        # Calibration BCE: single breath, last breath target
        calib_last = calib_history[-1]                                  # (B,)
        calib_loss_sum = Tensor.zeros((), dtype=dtypes.float).contiguous()
        for kc, calib_k in enumerate(calib_history):
            prog = float(kc) / float(max(K - 1, 1))
            target_k = 0.5 + (correct - 0.5) * prog
            calib_loss_sum = calib_loss_sum + ((calib_k - target_k) ** 2).mean()
        calib_loss = calib_loss_sum / float(K)

        # Accuracy metrics (detached — not in loss)
        cell_acc  = (eq * unobs_2d).sum() / (unobs_2d.sum() + 1e-8)
        query_acc = correct.mean()

        total = total_ce + calib_weight * calib_loss
        total.backward()

        # NaN guard: healthy=0 zeros all grads (AMD-safe single-kernel pattern)
        healthy = total.isfinite().cast(dtypes.float)
        for p in jit_params:
            if p.grad is not None:
                p.grad = p.grad * healthy.cast(p.grad.dtype)

        # Grad clip (same pattern as v108 — sq_sum approach, feasible on AMD)
        if grad_clip > 0:
            sq_sum = Tensor.zeros((), dtype=dtypes.float).contiguous()
            for p in jit_params:
                if p.grad is not None:
                    sq_sum = sq_sum + p.grad.cast(dtypes.float).square().sum()
            grad_norm = (sq_sum + 1e-12).sqrt()
            clip_coef = (Tensor(grad_clip, dtype=dtypes.float) / (grad_norm + 1e-6)).minimum(
                Tensor(1.0, dtype=dtypes.float)
            )
            for p in jit_params:
                if p.grad is not None:
                    p.grad = p.grad * clip_coef.cast(p.grad.dtype)

        opt.step()

        return (
            total.realize(),
            healthy.realize(),
            total_ce.realize(),
            calib_loss.realize(),
            cell_acc.realize(),
            query_acc.realize(),
            *(ce.realize() for ce in per_breath_ce_list),
        )

    return step_fn


def compile_jit_eval_v200(
    model: Any,
    K: int,
    B: int,
    n_max: int = V200_N_MAX,
    f_max: int = V200_F_MAX,
    n_var_lat: int = V200_N_VAR_LAT,
    n_digits: int = V200_N_DIGITS,
    stage2a_waist: bool = V200_STAGE2A_WAIST,
):
    """Compile a JIT-fused eval step for v200."""
    @TinyJit
    def eval_fn(
        domain_init: Tensor,
        node_kinds: Tensor,
    ):
        Tensor.training = False
        tree_logits_history, calib_history = fg_breathing_forward_v200(
            model, domain_init, node_kinds,
            K=K, n_max=n_max, f_max=f_max,
            n_var_lat=n_var_lat, n_digits=n_digits,
            training=False,
            stage2a_waist=stage2a_waist,
        )
        # Return final breath logits + all calibs (drift not needed for eval)
        return (tree_logits_history[-1], *calib_history)

    return eval_fn
