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

def _rms_norm_detached(x: Tensor, weight: Tensor, eps: float) -> Tensor:
    """RMSNorm with a DETACHED divisor — #237.5 substrate restoration (§1A.E.14).

    Forward IDENTICAL to _rms_norm. Backward difference: the divisor is a
    constant w.r.t. autograd, so pre-norm amplitude is no longer a flat
    direction of the loss (standard RMSNorm has ∂out/∂scale = 0 exactly —
    the #237 trapdoor: amplitude free to grow, backward taxed by 1/scale,
    organ starved to fp16 zero by step 600). With the detached scale the
    loss SEES amplitude locally, closing the flat direction before it can
    race. Applied at the three SEAM norms only (breath / read_ctx / blend);
    inside-operation pre-norms and Llama internals keep standard RMSNorm
    (bound the seams, not the organ — on both passes).
    """
    rms = (x.float().pow(2).mean(axis=-1, keepdim=True) + eps).sqrt().detach()
    return (x.float() / rms * weight.float()).cast(x.dtype)


def _nb_read_attend(
    lat_q: Tensor,      # (B, L, H) — pre-normed query source (same as fg READ's Q)
    N: Tensor,          # (B, S, H) — notebook slots written so far (S = k)
    nb_wq: Tensor,      # (H, H)
    nb_wo: Tensor,      # (H, H)
    n_heads: int,
    head_dim: int,
) -> "tuple[Tensor, Tensor]":
    """WRITE operator's READ-BACK attention (#238, §2 WRITE spec).

    Slots serve as K and V directly (they were projected through W_write at
    write time — re-projecting at read would be redundant params on a new
    pathway). Q gets its own projection. Pointer attention over S ≤ K−1 = 7
    keys — inside the attention-bootstrap principle's task-gradient-safe
    support (≤32-way); no auxiliary supervision needed.

    Returns (ctx (B, L, H), attn (B, nh, L, S)).
    """
    B, L, H = lat_q.shape
    S = int(N.shape[1])
    scale = 1.0 / math.sqrt(head_dim)
    q = (lat_q @ nb_wq.cast(lat_q.dtype)).reshape(B, L, n_heads, head_dim).transpose(1, 2)
    k = N.cast(lat_q.dtype).reshape(B, S, n_heads, head_dim).transpose(1, 2)
    scores = (q @ k.transpose(-2, -1)) * scale     # (B, nh, L, S)
    attn = scores.clip(-1e4, 1e4).softmax(-1).cast(k.dtype)
    ctx = (attn @ k).transpose(1, 2).reshape(B, L, H)
    return ctx @ nb_wo.cast(lat_q.dtype), attn


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
# Per-latent structural topology mask (§2, #237, Jun 11)
# "Diversity must be structural, not learned" — the project's oldest hard law.
# ---------------------------------------------------------------------------

def _build_topology_mask_1a(
    n_latents: int = 32,
    t_max: int = 24,
    n_max: int = 16,
    f_max: int = 8,
) -> np.ndarray:
    """Build partition 1a topology mask: (L=32, T=24) binary float32.

    Partition 1a (first-wire, per §2 spec):
      latents 0..23   — per-token:    mask[l, t] = (l == t)     support=1
      latents 24..27  — per-op-type:  each latent reads tokens of one op type
                                      assignment: token index mod 4 == op_idx  support≈6
      latents 28..31  — global:       all tokens visible                        support=24

    Random-init verification (§7):
      Per-token group  entropy ≈ 0      (support=1, log(1)=0)
      Per-op-type group entropy ≤ log(6)≈1.79  (support≈6 per op)
      Global group     entropy ≤ log(24)≈3.18  (support=24)
      All-32 mean      strictly < log(24)=3.178

    Returns np.ndarray of shape (L, T) dtype float32.
    """
    L = n_latents   # 32
    T = t_max       # 24 = n_max(16) + f_max(8)
    mask = np.zeros((L, T), dtype=np.float32)

    # --- Partition A: per-token latents (0..T-1 = 0..23) ---
    # One latent per fg token; latent l reads ONLY token l.
    # Support = 1 → entropy = 0 at random init (deterministic assignment).
    n_per_token = min(T, L)   # 24 (one per fg token)
    for l in range(n_per_token):
        mask[l, l] = 1.0

    # --- Partition B: per-op-type latents (24..27) ---
    # 4 latents, one per arithmetic op (ADD/SUB/MUL/DIV).
    # Token assignment: deterministic by token_index mod 4 (so each op-type latent
    # reads ≈ T/4 ≈ 6 tokens). This is structural, not learned; replaces the full
    # L×T cross-attention that gave every latent identical input.
    # In production, op_type annotation from the factor graph should route here;
    # for Stage 1C the mod-assignment is sufficient to test the per-group entropy.
    n_op_type = 4
    op_start  = n_per_token   # 24
    for l_off in range(n_op_type):
        l = op_start + l_off
        if l >= L:
            break
        for t in range(T):
            if t % n_op_type == l_off:
                mask[l, t] = 1.0

    # --- Partition C: global latents (28..31) ---
    # These 4 latents read all tokens — integration / boundary detection.
    global_start = op_start + n_op_type   # 28
    for l in range(global_start, L):
        mask[l, :] = 1.0

    return mask   # (L=32, T=24) float32


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
    topology_mask: "Tensor | None" = None,   # (L, T) binary float — #237 §2 structural mask
    return_weights: bool = False,            # #237 §7 instrumentation — eager-only
) -> "Tensor | tuple[Tensor, Tensor]":
    """Cross-attention: Q from latents, K/V from fg_tokens.

    Returns (B, L, H) context to be added to latents.
    No RoPE on cross-attention (positions not meaningful across modalities).
    Clip pre-softmax scores to (-1e4, 1e4) per AMD constraint.

    topology_mask: (L, T) binary float32 mask (#237, §2).
      Applied BEFORE softmax: masked positions receive score - 1e4 (→ ~0 after softmax).
      mask=1.0 → attend;  mask=0.0 → blocked.
      Broadcasts over (B, n_heads, L, T).
      If None: full L×T attention (Stage 1B behavior, pre-#237).

    return_weights: when True, returns (out, attn) where attn is the post-softmax
      (B, n_heads, L, T) weights — used by #237 per-mask-family entropy (§7).
      Default False keeps the single-return signature for all JIT call sites.
    """
    B, L, H = latents.shape
    T = int(fg_tokens.shape[1])
    scale = 1.0 / math.sqrt(head_dim)

    q = (latents @ wq.cast(latents.dtype)).reshape(B, L, n_heads, head_dim).transpose(1, 2)   # (B, nh, L, hd)
    k = (fg_tokens @ wk.cast(fg_tokens.dtype)).reshape(B, T, n_heads, head_dim).transpose(1, 2)  # (B, nh, T, hd)
    v = (fg_tokens @ wv.cast(fg_tokens.dtype)).reshape(B, T, n_heads, head_dim).transpose(1, 2)  # (B, nh, T, hd)

    scores = (q @ k.transpose(-2, -1)) * scale   # (B, nh, L, T)

    if topology_mask is not None:
        # mask shape (L, T) → reshape to (1, 1, L, T) for broadcast over (B, nh, L, T)
        # masked positions: (mask - 1) * 1e4 = -1e4; attended positions: (1 - 1)*1e4 = 0
        mask_bias = (topology_mask.reshape(1, 1, L, T) - 1.0).cast(scores.dtype) * 1e4
        scores = scores + mask_bias

    attn = scores.clip(-1e4, 1e4).softmax(-1).cast(v.dtype)
    ctx = (attn @ v).transpose(1, 2).reshape(B, L, H)   # (B, L, H)
    out = ctx @ wo.cast(latents.dtype)
    if return_weights:
        return out, attn
    return out


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
      fg_v200_breath_norm_w (H,)            — RMSNorm weight at breath boundary (ones init) ← §2 4th norm, Jun 11 #234
      fg_v200_read_norm_w   (H,)            — RMSNorm weight before cross-attn Q (ones init)
      fg_v200_read_ctx_norm_w (H,)          — RMSNorm weight on read_ctx before residual add (ones init) ← §1A.E.4 #235
      fg_v200_alpha_read    (1,)            — learnable scale on α·RMSNorm(read_ctx), init=1.0 ← §1A.E.4 #235
      fg_v200_commit_norm_w (H,)            — RMSNorm weight before waist input on even breaths (ones init)
      fg_v200_blend_norm_w  (H,)            — RMSNorm weight at Seam 3 (pre-blend, ones init) ← §2 6th norm, Jun 11 #236
      fg_v200_latent_norm_w (H,)            — RMSNorm weight before tree codebook readout (ones init)
      fg_v200_tree_codebook (n_digits, 10, H) — Fourier orthogonal per level
      fg_v200_calib_w       (H, 1)          — calibration head weight (zero-init)
      fg_v200_calib_b       (1,)            — calibration head bias
      fg_v200_delta_gate    (k_max,)        — per-breath residual gate, init=-2.0 (sigmoid→0.119) ← §2 spec-restore #234

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
        # #237 review fix: the early-return must still attach the topology mask
        # (deterministic non-trainable buffer, not in state_dict) — otherwise a
        # double-attach path silently runs unmasked full L×T while reporting a
        # mask1a arch_version.
        if not hasattr(model, "fg_v200_latent_topology_mask"):
            model.fg_v200_latent_topology_mask = Tensor(
                _build_topology_mask_1a(n_latents=n_latents, t_max=T,
                                        n_max=n_max, f_max=f_max),
                dtype=dtypes.float,
            ).contiguous().realize()
            print("[v200] params already attached; topology mask was missing — attached", flush=True)
        # #238 review fix: same backfill treatment for the WRITE params
        # (deterministic inits; missing on pre-#238-attached models)
        if not hasattr(model, "fg_v200_nb_gate"):
            def _eye02(d):
                return (np.eye(d, dtype=np.float32) * 0.02).astype(np.float32)
            model.fg_v200_nb_W_write     = Tensor(_eye02(H), dtype=dtypes.float).contiguous().realize()
            model.fg_v200_nb_wq          = Tensor(_eye02(H), dtype=dtypes.float).contiguous().realize()
            model.fg_v200_nb_wo          = Tensor(_eye02(H), dtype=dtypes.float).contiguous().realize()
            model.fg_v200_nb_write_norm_w = Tensor.ones(H, dtype=dtypes.float).contiguous().realize()
            model.fg_v200_nb_read_norm_w  = Tensor.ones(H, dtype=dtypes.float).contiguous().realize()
            model.fg_v200_nb_gate        = Tensor(np.zeros((1,), dtype=np.float32)).contiguous().realize()
            print("[v200] params already attached; WRITE params were missing — attached (gate=0)", flush=True)
        else:
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

    # ---- RMSNorm weights: ones init (§2, six norms after #233/#234/#235/#236) ----
    # fg_v200_breath_norm_w: 4th RMSNorm at breath boundary (spec-restore #234, Jun 11)
    # Bounds inter-breath residual accumulation (z oscillated 0.77→19160 in #233 without it).
    # fg_v200_read_ctx_norm_w: RMSNorm on read_ctx before residual add (#235, Jun 11)
    #   READ-dominance cell (§1A.E.4): rdr 173 at breath 0 — read_ctx magnitude is fixed by
    #   W_{q,k,v,o} and token embeddings, not z input. α·RMSNorm(read_ctx) bounds it.
    # fg_v200_alpha_read: learnable scalar init=1.0 (NOT zero-init — READ is the information
    #   inlet, not auxiliary; α=0 creates v118-v121 bootstrap bottleneck per §1A.E.8).
    # fg_v200_blend_norm_w: 6th RMSNorm — Seam 3 pre-blend norm (#236, Jun 11)
    #   Applied BEFORE delta_gate convex blend (not after — pre-blend keeps gate semantics
    #   interpretable: both inputs to the convex mix are unit-scale, gate is a fraction of
    #   the difference, not a product of scale-mismatched tensors).
    #   norm-after-blend would launder post-THINK magnitude mismatch and make gate's
    #   effective behavior scale-dependent (brief §2, locked Jun 11 evening).
    model.fg_v200_breath_norm_w    = Tensor.ones(H, dtype=dtypes.float).contiguous().realize()
    model.fg_v200_read_norm_w      = Tensor.ones(H, dtype=dtypes.float).contiguous().realize()
    model.fg_v200_read_ctx_norm_w  = Tensor.ones(H, dtype=dtypes.float).contiguous().realize()
    model.fg_v200_alpha_read       = Tensor.ones(1, dtype=dtypes.float).contiguous().realize()
    model.fg_v200_commit_norm_w    = Tensor.ones(H, dtype=dtypes.float).contiguous().realize()
    model.fg_v200_blend_norm_w     = Tensor.ones(H, dtype=dtypes.float).contiguous().realize()   # Seam 3, #236
    model.fg_v200_latent_norm_w    = Tensor.ones(H, dtype=dtypes.float).contiguous().realize()

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

    # ---- Delta gate: -2.0 init (sigmoid→0.119, spec-restore #234 Jun 11) ----
    # §2 spec: gate_k = sigmoid(delta_gate[k]); init=-2.0 per documented Pythia cold-start
    # finding (gate=-5.0 α=0.007 starved gradient; gate=-2.0 α=0.119 = 240× faster growth).
    model.fg_v200_delta_gate = Tensor(
        np.full((k_max,), -2.0, dtype=np.float32)
    ).contiguous().realize()

    # ---- WRITE operator (#238, §2 Jun 12): shared K-slot notebook ----
    # Fixed address (breath index), write-once-per-slot (by construction:
    # slots are appended, never reassigned), read-many (all latents attend).
    # g_nb ZERO-init: read-back is AUXILIARY at init (the architecture is
    # complete without it — §2 WRITE spec's outlet-vs-inlet paragraph);
    # exact 0.0 multiplier → step-0 forward byte-identical to #237.5.
    model.fg_v200_nb_W_write     = Tensor(_eye_init(H), dtype=dtypes.float).contiguous().realize()
    model.fg_v200_nb_wq          = Tensor(_eye_init(H), dtype=dtypes.float).contiguous().realize()
    model.fg_v200_nb_wo          = Tensor(_eye_init(H), dtype=dtypes.float).contiguous().realize()
    model.fg_v200_nb_write_norm_w = Tensor.ones(H, dtype=dtypes.float).contiguous().realize()
    model.fg_v200_nb_read_norm_w  = Tensor.ones(H, dtype=dtypes.float).contiguous().realize()
    model.fg_v200_nb_gate        = Tensor(np.zeros((1,), dtype=np.float32)).contiguous().realize()

    # ---- Per-latent topology mask (#237, §2 Jun 11) ----
    # Binary (L=32, T=24) mask encoding structural diversity per the project's oldest law:
    # "diversity must be structural, not learned." Partition 1a:
    #   latents 0..23:  per-token (each reads one fg token; support=1)
    #   latents 24..27: per-op-type (each reads tokens of one op type; support≈6)
    #   latents 28..31: global (reads all 24 tokens; support=24)
    # Applied in _cross_attend_v200() BEFORE softmax (masked positions → -1e4 → ~0 attn).
    # NOT trainable — structural mask is a config constant, not a learned parameter.
    # Stored as a non-trainable buffer on the model object.
    mask_np = _build_topology_mask_1a(
        n_latents=n_latents, t_max=T,
        n_max=n_max, f_max=f_max,
    )
    model.fg_v200_latent_topology_mask = Tensor(
        mask_np, dtype=dtypes.float
    ).contiguous().realize()

    # ---- Stage 2a: alternating waist (v5 convex blend) ----
    # W_compress: small QR-init (scale 0.01) so z = latents @ W_compress is non-zero.
    #   → gradient ∂L/∂W_expand = z.T @ (∂L/∂new · α) is NON-ZERO at step 1.
    # W_expand: ZERO-INIT → h_compressed = 0 at step 0.
    # commit_gate: init to -2.0 → sigmoid(-2)≈0.119 (v200 v6; -5.0 was the starved config).
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
        # commit_gate: init to -2.0 → sigmoid(-2)≈0.119 (v200 v6, commit a8d28c3)
        # Var name kept as fg_v200_waist_gate for checkpoint compatibility.
        model.fg_v200_waist_gate = Tensor(
            np.full((1,), -2.0, dtype=np.float32)
        ).contiguous().realize()
        n_waist_params = H * waist_dim + waist_dim * H + 1
        print(
            f"[v200-2a-v5] Stage 2a waist attached (convex blend): "
            f"W_compress({H},{waist_dim})=QR×0.01  "
            f"W_expand({waist_dim},{H})=ZEROS  commit_gate=-2.0(α≈0.119)  "
            f"alternation=EVEN_BREATHS  no_RMSNorm  "
            f"new_params={n_waist_params:,}  ({n_waist_params/1e6:.2f}M)",
            flush=True,
        )

    # ---- Count and report ----
    n_latent_params = n_latents * H
    n_breath_params = k_max * H
    # 4 proj + 6 RMSNorm weights (breath/read/read_ctx/commit/blend/latent) + alpha_read 1
    # #236: blend_norm added as Seam 3 (6th RMSNorm, +H params)
    n_cross_params  = 4 * H * H + 6 * H + 1
    n_tree_params   = n_digits * 10 * H
    n_calib_params  = H + 1
    n_gate_params   = k_max
    # topology mask: NOT trainable, stored as a buffer (not counted in trainable param total)
    n_new = (n_latent_params + n_breath_params + n_cross_params +
             n_tree_params + n_calib_params + n_gate_params + n_waist_params)

    n_mask_elems = n_latents * T

    print(
        f"[v200] perceiver-CORE params attached:\n"
        f"  latents        : ({n_latents}, {H}) = {n_latent_params:,}\n"
        f"  breath_embed   : ({k_max}, {H}) = {n_breath_params:,}\n"
        f"  cross_attn     : 4×({H},{H}) + 6×{H} (breath/read/read_ctx/commit/blend/latent RMSNorm) + 1 (alpha_read) = {n_cross_params:,}\n"
        f"  tree_codebook  : ({n_digits}, 10, {H}) = {n_tree_params:,}\n"
        f"  calibration    : {n_calib_params:,}\n"
        f"  delta_gate     : {n_gate_params:,}\n"
        + (f"  waist (2a)     : {n_waist_params:,}\n" if stage2a_waist else "")
        + f"  topology_mask  : ({n_latents},{T}) = {n_mask_elems:,} floats  [NOT trainable — structural buffer, #237]\n"
        + f"  TOTAL trainable: {n_new:,}  ({n_new/1e6:.2f}M)",
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
        "fg_v200_breath_norm_w",    # added Jun 11 #234 (§2 norm_breath — 4th RMSNorm at breath boundary)
        "fg_v200_read_norm_w",
        "fg_v200_read_ctx_norm_w",  # added Jun 11 #235 (§1A.E.4 READ-dominance fix)
        "fg_v200_alpha_read",       # added Jun 11 #235 (scalar α on α·RMSNorm(read_ctx), init=1.0)
        "fg_v200_commit_norm_w",    # added Jun 11 (§2 norm_commit)
        "fg_v200_blend_norm_w",     # added Jun 11 #236 (§2 Seam 3 — 6th RMSNorm, pre-blend)
        "fg_v200_latent_norm_w",
        "fg_v200_tree_codebook",
        "fg_v200_calib_w",
        "fg_v200_calib_b",
        "fg_v200_delta_gate",
        # WRITE operator (#238, §2): shared K-slot notebook
        "fg_v200_nb_W_write",
        "fg_v200_nb_wq",
        "fg_v200_nb_wo",
        "fg_v200_nb_write_norm_w",
        "fg_v200_nb_read_norm_w",
        "fg_v200_nb_gate",
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
        "fg_v200_breath_norm_w",    # added Jun 11 #234 (§2 norm_breath — 4th RMSNorm at breath boundary)
        "fg_v200_read_norm_w",
        "fg_v200_read_ctx_norm_w",  # added Jun 11 #235 (§1A.E.4 READ-dominance fix)
        "fg_v200_alpha_read",       # added Jun 11 #235 (scalar α on α·RMSNorm(read_ctx), init=1.0)
        "fg_v200_commit_norm_w",    # added Jun 11 (§2 norm_commit)
        "fg_v200_blend_norm_w",     # added Jun 11 #236 (§2 Seam 3 — 6th RMSNorm, pre-blend)
        "fg_v200_latent_norm_w",
        "fg_v200_tree_codebook",
        "fg_v200_calib_w",
        "fg_v200_calib_b",
        "fg_v200_delta_gate",
        # WRITE operator (#238, §2): shared K-slot notebook
        "fg_v200_nb_W_write",
        "fg_v200_nb_wq",
        "fg_v200_nb_wo",
        "fg_v200_nb_write_norm_w",
        "fg_v200_nb_read_norm_w",
        "fg_v200_nb_gate",
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
    commit_gate: Tensor,    # (1,)             init to -2.0 → sigmoid≈0.119 (v200 v6)
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

    Init: commit_gate = -2.0 → α = sigmoid(-2) ≈ 0.119 (v200 v6, commit a8d28c3;
      the original -5.0 / α≈0.007 starved the waist's training signal)
      → latents_new ≈ 0.881 * latents at step 0 (W_expand=0 → h_compressed=0)

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
    taps: "dict | None" = None,
    carrier_dim_mask: "Tensor | None" = None,
    carrier_mask_site: str = "boundary",
) -> tuple[list[Tensor], list[Tensor]]:
    """v200 perceiver-CORE forward pass.

    carrier_dim_mask (#237 §1A.E.13 carrier-projection ablation — EAGER ONLY,
    None in all JIT paths): optional (H,) float mask (1s with 0s at the shared
    carrier dims) multiplied into the latent state at the site(s) selected by
    carrier_mask_site. The placement embeds WHICH blackboard reading is being
    tested (E.13):
      "boundary"   — at latent init + immediately after the Seam-1 breath-
                     boundary norm each breath. Kills the carrier's
                     CROSS-BREATH PERSISTENCE (memory reading); within-breath
                     use stays free (blend may reintroduce; next boundary
                     clears).
      "post_think" — immediately after THINK output each breath, BEFORE
                     COMMIT/blend (init left unprojected). Kills the carrier's
                     WITHIN-BREATH BROADCAST (bus reading): COMMIT, Seam-3,
                     the gate blend, and the readout never see this breath's
                     THINK-written carrier content. Cross-breath persistence
                     via z_pre stays free. (The waist's own h_compressed may
                     re-write the dims post-projection — that is the waist's
                     writing, not THINK's broadcast; documented limitation.)
      "both"       — both sites + init.

    Stages per breath k:
      1. Add per-breath embedding to latents.
      2. READ: latents cross-attend to fg_tokens (pre-normed latents as Q).
      3. THINK: 4 Llama layers process latents (32×32 self-attention, full, no mask).
      4. WAIST (Stage 2a, EVEN breaths only): alternating waist compression.
      4.5. SEAM 3: pre-blend RMSNorm on post-THINK/COMMIT state (#236).
      5. GATE: delta_gate convex blend of pre-breath and post-THINK state;
         then WRITE (#238): settled state pooled → notebook slot k.
      6. READOUT: tree codebook on first n_var_lat latents + calibration head.

    Returns:
      tree_logits_history  : K × (B, n_var_lat, n_digits, 10)
      calib_history        : K × (B,)

    taps (#237, §7 single-forward instrumentation — EAGER ONLY, must be None in
    JIT paths; all JIT call sites pass nothing): pass fg_v200_empty_taps() and
    the forward fills it in place:
      "z_init"          : Tensor (B, L, H) fp32 — latent state before breath 0
      "wb_post_norm"    : K × Tensor (B, L, H) — post Seam-1 norm_breath   (§1A.E.9 ckpt 1)
      "wb_post_read"    : K × Tensor — post READ residual add              (§1A.E.9 ckpt 2)
      "wb_post_think"   : K × Tensor — post Llama L0-L3                    (§1A.E.9 ckpt 3)
      "wb_post_norm_blend": K × Tensor — post Seam-3 norm_blend            (§1A.E.9 ckpt 4)
      "wb_post_blend"   : K × Tensor — post delta_gate convex blend        (§1A.E.9 ckpt 5;
                          this is the per-breath snapshot point used for JSD/C6/E.4)
      "xattn_weights"   : K × Tensor (B, nh, L, T) — READ post-softmax weights
      "sa_weights"      : K × [4 × Tensor (B, nh, L, L)] — THINK per-layer weights
      "waist_contrib"   : (even breaths) × Tensor (B, L, H) — waist module output minus
                          its input = α·(h_compressed − z_w), captured at the waist
                          output BEFORE norm_blend (the C5 recalibrated metric site,
                          §1A.B.5 "pre-norm waist contribution")
      "waist_breaths"   : list of breath indices k where the waist fired
    This is the single code path for all #237 probes — within-breath trajectory,
    concentration drift, per-family entropy, per-latent THINK entropy, C5 — so
    probe forwards cannot drift from the trained forward (brief §2 single-forward).

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
    breath_norm_w  = model.fg_v200_breath_norm_w       # (H,)  — §2 4th RMSNorm at breath boundary, #234
    read_norm_w    = model.fg_v200_read_norm_w         # (H,)
    read_ctx_norm_w = model.fg_v200_read_ctx_norm_w   # (H,)  — §1A.E.4 READ-dominance fix, #235
    alpha_read     = model.fg_v200_alpha_read          # (1,)  — learnable scalar init=1.0, #235
    commit_norm_w  = model.fg_v200_commit_norm_w       # (H,)  — added Jun 11 (§2 norm_commit)
    blend_norm_w   = model.fg_v200_blend_norm_w        # (H,)  — added Jun 11 #236 (§2 Seam 3, pre-blend)
    latent_norm_w  = model.fg_v200_latent_norm_w       # (H,)
    tree_codebook  = model.fg_v200_tree_codebook       # (n_digits, 10, H)
    calib_w        = model.fg_v200_calib_w             # (H, 1)
    calib_b        = model.fg_v200_calib_b             # (1,)
    delta_gate     = model.fg_v200_delta_gate          # (K_max,) — stored pre-sigmoid; init=-2.0
    rope_cos       = model.llama_rope_cos              # (max_pos, H)
    rope_sin       = model.llama_rope_sin              # (max_pos, H)
    llama_layers   = model.llama_layers                # list[LlamaLayer]
    # Per-latent topology mask (#237, §2): structural diversity in READ phase.
    # None-fallback preserves pre-#237 behavior if called without mask on model.
    topology_mask  = getattr(model, "fg_v200_latent_topology_mask", None)  # (L, T) or None
    # WRITE operator (#238, §2): shared K-slot notebook params
    nb_W_write      = model.fg_v200_nb_W_write       # (H, H)
    nb_wq           = model.fg_v200_nb_wq            # (H, H)
    nb_wo           = model.fg_v200_nb_wo            # (H, H)
    nb_write_norm_w = model.fg_v200_nb_write_norm_w  # (H,)
    nb_read_norm_w  = model.fg_v200_nb_read_norm_w   # (H,)
    nb_gate         = model.fg_v200_nb_gate          # (1,) ZERO-init (auxiliary read-back)

    # Stage 2a waist params (only pulled when stage2a_waist=True)
    if stage2a_waist:
        assert hasattr(model, "fg_v200_W_compress"), \
            "fg_v200_W_compress not found — set V200_STAGE2A_WAIST=1 in attach_fg_params_v200"
        W_compress    = model.fg_v200_W_compress      # (H, waist_dim)
        W_expand      = model.fg_v200_W_expand        # (waist_dim, H)
        waist_gate    = model.fg_v200_waist_gate      # (1,) commit_gate, init -2.0

    # ---- Static: embed factor graph tokens once ----
    fg_tokens = _embed_fg_tokens_v200(model, domain_init, node_kinds, n_max, f_max).cast(dtypes.float)   # fp32 chain (#237.5)
    # fg_tokens: (B, T, H) fp32

    # ---- Initialize latents: broadcast base to batch ----
    # latents_base: (n_latents, H) → (B, n_latents, H)
    n_latents = int(latents_base.shape[0])
    latents = latents_base.reshape(1, n_latents, H).expand(B, n_latents, H)
    latents = latents.cast(dtypes.float)   # fp32 chain (#237.5; latent loop is tiny — 32×2048)

    if carrier_dim_mask is not None:
        assert carrier_mask_site in ("boundary", "post_think", "both"), carrier_mask_site
        if carrier_mask_site in ("boundary", "both"):
            latents = latents * carrier_dim_mask.cast(latents.dtype).reshape(1, 1, H)

    if taps is not None:
        taps["z_init"] = latents.cast(dtypes.float).realize()

    # Pre-compute tree codebook flat for readout: (n_digits*10, H)
    tree_cb_flat = tree_codebook.reshape(n_digits * 10, H)  # (50, H)

    K_max = int(breath_embed.shape[0])
    assert K <= K_max, f"K={K} > K_max={K_max}"

    tree_logits_history = []
    calib_history       = []
    notebook: list = []   # WRITE slots (#238) — appended once per breath, never reassigned

    for k in range(K):
        # 0. Breath boundary RMSNorm — §2 4th norm (spec-restore #234, Jun 11)
        # Bounds inter-breath residual accumulation. z_pre captured POST-norm so
        # the gate convex blend mixes at bounded scale (prevents pre/post scale mismatch).
        latents = _rms_norm_detached(latents, breath_norm_w, rms_eps).cast(latents.dtype)   # Seam 1, detached (#237.5)
        if carrier_dim_mask is not None and carrier_mask_site in ("boundary", "both"):
            # §1A.E.13 carrier projection, boundary site: clear the shared-
            # carrier dims at the breath boundary (cross-breath persistence /
            # memory reading; within-breath use stays free)
            latents = latents * carrier_dim_mask.cast(latents.dtype).reshape(1, 1, H)
        latents_pre_breath = latents    # ← captured POST-norm (§2 GATE blends normalized pre)

        if taps is not None:
            taps["wb_post_norm"].append(latents.cast(dtypes.float).realize())

        # 1. Per-breath additive embedding
        be_k = breath_embed[k].reshape(1, 1, H).cast(latents.dtype)
        latents = latents + be_k

        # 2. READ: latents cross-attend to fg_tokens (pure residual addition, no gate here)
        # Pre-norm latents before cross-attn (RMSNorm)
        lat_normed = _rms_norm(latents, read_norm_w, rms_eps).cast(latents.dtype)

        if taps is not None:
            read_ctx, xattn_w = _cross_attend_v200(
                lat_normed, fg_tokens,
                cross_wq, cross_wk, cross_wv, cross_wo,
                n_heads=nh, head_dim=hd,
                topology_mask=topology_mask,
                return_weights=True,
            )
            taps["xattn_weights"].append(xattn_w.cast(dtypes.float).realize())
        else:
            read_ctx = _cross_attend_v200(
                lat_normed, fg_tokens,
                cross_wq, cross_wk, cross_wv, cross_wo,
                n_heads=nh, head_dim=hd,
                topology_mask=topology_mask,   # §2 structural mask (#237); None = pre-#237 full-L×T
            )
        # §1A.E.4 READ-dominance fix (#235, Jun 11):
        # read_ctx scale is fixed by W_{q,k,v,o} + token embeddings, independent of z.
        # At breath 0: rdr=173 (read_ctx per-elem 14 >> z per-elem ~1).
        # α·RMSNorm(read_ctx) bounds it to per-elem ~α, with α init=1.0 (NOT zero-init
        # — READ is the information inlet; zero-init would bootstrap-shut it per §1A.E.8).
        read_ctx_normed = _rms_norm_detached(read_ctx.cast(dtypes.float), read_ctx_norm_w, rms_eps).cast(latents.dtype)   # Seam 2, detached (#237.5)
        latents = latents + alpha_read.cast(latents.dtype) * read_ctx_normed

        # WRITE operator READ-BACK (#238): attend to slots 0..k-1 (causal over
        # breaths). Q reuses lat_normed (one normalization, two reads — fg and
        # notebook are parallel reads from the same pre-add query state).
        # g_nb zero-init → exact 0 contribution at step 0 (byte-identity to
        # #237.5; auxiliary path per §2 WRITE spec's gate paragraph).
        if k >= 1:
            N_slots = Tensor.stack(*notebook, dim=1)   # (B, k, H)
            nb_ctx, nb_attn = _nb_read_attend(lat_normed, N_slots, nb_wq, nb_wo,
                                              n_heads=nh, head_dim=hd)
            nb_ctx_n = _rms_norm_detached(nb_ctx.cast(dtypes.float), nb_read_norm_w,
                                          rms_eps).cast(latents.dtype)
            latents = latents + nb_gate.cast(latents.dtype).reshape(1, 1, 1) * nb_ctx_n
            if taps is not None:
                taps["nb_attn"].append(nb_attn.cast(dtypes.float).realize())

        if taps is not None:
            taps["wb_post_read"].append(latents.cast(dtypes.float).realize())

        # 3. THINK: 4 Llama layers process latents (32×32 self-attention, no mask)
        h = latents.cast(dtypes.float)    # Llama layers run fp32
        if taps is not None:
            sa_k = []
            for layer in llama_layers[:4]:
                h, sa_w = layer.forward_return_weights(h, rope_cos, rope_sin, attn_mask=None)
                sa_k.append(sa_w.cast(dtypes.float).realize())
            taps["sa_weights"].append(sa_k)
        else:
            for layer in llama_layers[:4]:
                h = layer(h, rope_cos, rope_sin, attn_mask=None)
        latents = h   # fp32 inter-breath chain (#237.5) — no half cast, no underflow cliff

        if carrier_dim_mask is not None and carrier_mask_site in ("post_think", "both"):
            # §1A.E.13 carrier projection, post-THINK site: clear THINK's
            # carrier-dim broadcast before COMMIT/Seam-3/blend/readout see it
            # (within-breath bus reading)
            latents = latents * carrier_dim_mask.cast(latents.dtype).reshape(1, 1, H)

        if taps is not None:
            taps["wb_post_think"].append(latents.cast(dtypes.float).realize())

        # 4. WAIST (Stage 2a): alternating compression on EVEN breaths only
        # EVEN = COMMIT phase (force commitment through bottleneck)
        # ODD  = PROPAGATION phase (bypass waist, beliefs propagate freely at full 2048d)
        # Pre-norm waist input (§2, norm_commit, added Jun 11)
        if stage2a_waist and (k % 2 == 0):
            latents_w = _rms_norm(latents, commit_norm_w, rms_eps).cast(latents.dtype)
            latents = _apply_waist_v200(
                latents_w, W_compress, W_expand, waist_gate,
            )
            if taps is not None:
                # C5 recalibrated site (§1A.B.5): waist contribution α·(h_compressed − z_w),
                # read at the waist module output BEFORE norm_blend / any normalization.
                taps["waist_contrib"].append((latents - latents_w).cast(dtypes.float).realize())
                taps["waist_breaths"].append(k)

        # 4.5. SEAM 3: pre-blend RMSNorm (§2, norm_blend, Seam 3, #236, Jun 11)
        # Applied BEFORE the convex gate blend — not after. Pre-blend keeps gate semantics
        # interpretable: both inputs to the convex mix are ~unit-scale (z_pre ~1 from
        # breath-start RMSNorm, post-THINK now ~1 from norm_blend), so gate_k is a
        # fraction of the UNIT-SCALE difference rather than a fraction of a 45× attractor.
        # norm-after-blend would launder the mismatch silently and make gate's effective
        # behavior scale-dependent (brief §2, locked Jun 11 evening).
        latents = _rms_norm_detached(latents, blend_norm_w, rms_eps).cast(latents.dtype)   # Seam 3, detached (#237.5)

        if taps is not None:
            taps["wb_post_norm_blend"].append(latents.cast(dtypes.float).realize())

        # 5. GATE: §2 convex blend (spec-restore #234)
        # gate_k = sigmoid(delta_gate[k]); init=-2.0 → alpha≈0.119 at step 0.
        # Blends normalized pre-breath state with Seam-3-normed post-THINK/COMMIT state.
        # Both inputs now ~unit-scale; gate_k is a pure mixing coefficient.
        gate_k = delta_gate[k].sigmoid().cast(latents.dtype).reshape(1, 1, 1)
        latents = latents_pre_breath + gate_k * (latents - latents_pre_breath)

        if taps is not None:
            taps["wb_post_blend"].append(latents.cast(dtypes.float).realize())

        # WRITE (#238): commit the SETTLED state (post-gate-blend) to slot k.
        # Write-once by construction: the list is appended, never reassigned —
        # future breaths cannot overwrite slot k. The write passes a detached
        # seam norm (workspace meets memory = a seam). No gate on the write
        # side (writing to memory nobody reads yet is free; the gate is on
        # the read-back).
        pooled_settled = latents.mean(axis=1).cast(dtypes.float)        # (B, H)
        slot_k = _rms_norm_detached(pooled_settled, nb_write_norm_w, rms_eps) \
                 @ nb_W_write.cast(dtypes.float)
        notebook.append(slot_k)
        if taps is not None and k == K - 1:
            taps["nb_slots"] = [s.realize() for s in notebook]

        # 6. READOUT
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
    breath_norm_w  = model.fg_v200_breath_norm_w   # §2 4th RMSNorm, #234
    read_norm_w    = model.fg_v200_read_norm_w
    read_ctx_norm_w = model.fg_v200_read_ctx_norm_w  # §1A.E.4 READ-dominance fix, #235
    alpha_read     = model.fg_v200_alpha_read         # scalar α, init=1.0, #235
    commit_norm_w  = model.fg_v200_commit_norm_w   # added Jun 11
    blend_norm_w   = model.fg_v200_blend_norm_w    # added Jun 11 #236 (§2 Seam 3, pre-blend)
    delta_gate     = model.fg_v200_delta_gate      # pre-sigmoid; init=-2.0
    rope_cos       = model.llama_rope_cos
    rope_sin       = model.llama_rope_sin
    llama_layers   = model.llama_layers
    topology_mask  = getattr(model, "fg_v200_latent_topology_mask", None)   # #237 §2
    # WRITE operator (#238) — mirror of the canonical forward
    nb_W_write      = model.fg_v200_nb_W_write
    nb_wq           = model.fg_v200_nb_wq
    nb_wo           = model.fg_v200_nb_wo
    nb_write_norm_w = model.fg_v200_nb_write_norm_w
    nb_read_norm_w  = model.fg_v200_nb_read_norm_w
    nb_gate         = model.fg_v200_nb_gate

    if stage2a_waist:
        W_compress    = model.fg_v200_W_compress
        W_expand      = model.fg_v200_W_expand
        waist_gate    = model.fg_v200_waist_gate

    fg_tokens = _embed_fg_tokens_v200(model, domain_init, node_kinds, n_max, f_max).cast(dtypes.float)   # fp32 chain (#237.5)

    n_latents = int(latents_base.shape[0])
    latents = latents_base.reshape(1, n_latents, H).expand(B, n_latents, H)
    latents = latents.cast(dtypes.float)   # fp32 chain (#237.5)

    drifts: list[float] = []
    notebook: list = []   # WRITE slots (#238) — mirror of the canonical forward

    for k in range(K):
        latents_pre_np = latents.cast(dtypes.float).realize().numpy()

        # Breath boundary RMSNorm (§2 4th norm, spec-restore #234)
        latents = _rms_norm_detached(latents, breath_norm_w, rms_eps).cast(latents.dtype)   # Seam 1, detached (#237.5)
        latents_pre_breath = latents    # post-norm pre-breath (for convex gate blend)

        be_k = breath_embed[k].reshape(1, 1, H).cast(latents.dtype)
        latents = latents + be_k

        lat_normed = _rms_norm(latents, read_norm_w, rms_eps).cast(latents.dtype)
        read_ctx = _cross_attend_v200(
            lat_normed, fg_tokens,
            cross_wq, cross_wk, cross_wv, cross_wo,
            n_heads=nh, head_dim=hd,
            topology_mask=topology_mask,   # §2 structural mask (#237)
        )
        # §1A.E.4 READ-dominance fix (#235): α·RMSNorm(read_ctx)
        read_ctx_normed = _rms_norm_detached(read_ctx.cast(dtypes.float), read_ctx_norm_w, rms_eps).cast(latents.dtype)   # Seam 2, detached (#237.5)
        latents = latents + alpha_read.cast(latents.dtype) * read_ctx_normed

        # WRITE operator READ-BACK (#238) — mirror of the canonical forward
        if k >= 1:
            N_slots = Tensor.stack(*notebook, dim=1)
            nb_ctx, _nb_attn = _nb_read_attend(lat_normed, N_slots, nb_wq, nb_wo,
                                               n_heads=nh, head_dim=hd)
            nb_ctx_n = _rms_norm_detached(nb_ctx.cast(dtypes.float), nb_read_norm_w,
                                          rms_eps).cast(latents.dtype)
            latents = latents + nb_gate.cast(latents.dtype).reshape(1, 1, 1) * nb_ctx_n

        h = latents.cast(dtypes.float)
        for layer in llama_layers[:4]:
            h = layer(h, rope_cos, rope_sin, attn_mask=None)
        latents = h   # fp32 inter-breath chain (#237.5) — no half cast, no underflow cliff

        if stage2a_waist and (k % 2 == 0):
            latents_w_drift = _rms_norm(latents, commit_norm_w, rms_eps).cast(latents.dtype)
            latents = _apply_waist_v200(
                latents_w_drift, W_compress, W_expand, waist_gate,
            )

        # Seam 3: pre-blend RMSNorm (§2, norm_blend, #236, Jun 11)
        latents = _rms_norm_detached(latents, blend_norm_w, rms_eps).cast(latents.dtype)   # Seam 3, detached (#237.5)

        # GATE: §2 convex blend (spec-restore #234)
        gate_k = delta_gate[k].sigmoid().cast(latents.dtype).reshape(1, 1, 1)
        latents = latents_pre_breath + gate_k * (latents - latents_pre_breath)

        # WRITE (#238) — mirror of the canonical forward
        pooled_settled = latents.mean(axis=1).cast(dtypes.float)
        slot_k = _rms_norm_detached(pooled_settled, nb_write_norm_w, rms_eps) \
                 @ nb_W_write.cast(dtypes.float)
        notebook.append(slot_k)

        latents_post_np = latents.cast(dtypes.float).realize().numpy()
        diff = latents_post_np - latents_pre_np
        drift_k = float(np.sqrt((diff ** 2).sum(axis=-1)).mean())
        drifts.append(drift_k)

    Tensor.training = was_training
    return drifts


# ---------------------------------------------------------------------------
# Canonical JSD metric helpers — used by both training driver and reference
# curve generation. §7 method_sha discipline: both caller sites import these
# functions from here, so the same code path is guaranteed.
# ---------------------------------------------------------------------------

def _collect_latent_snapshots(
    model: Any,
    domain_init: Tensor,
    node_kinds: Tensor,
    K: int,
    n_max: int = V200_N_MAX,
    f_max: int = V200_F_MAX,
    stage2a_waist: bool = V200_STAGE2A_WAIST,
) -> list:
    """Run fg_breathing_forward_v200 and collect (K+1) latent snapshots.

    Returns list of K+1 np.ndarray of shape (B, L, H), float32.
    Snapshot 0 = latent init before breath 0.
    Snapshot k+1 = latent state after breath k (post-delta-gate blend).

    REWRITTEN #237: thin wrapper over the tapped canonical forward
    (fg_breathing_forward_v200 with taps). Snapshot points are identical to the
    pre-#237 implementation (init state + post-delta-gate-blend per breath); the
    breath-loop copy that used to live here is gone, so this function can no
    longer drift from the trained forward (brief §2 single-forward).

    Used by: training driver (JSD at step 200) + reference curve generation.
    §7 method_sha identity: compute_latent_jsd_from_snapshots operates on the
    output of this function; both are defined in factor_graph_v200.py so the
    same implementation is used everywhere.
    """
    was_training = Tensor.training
    Tensor.training = False

    taps = fg_v200_empty_taps()
    fg_breathing_forward_v200(
        model, domain_init, node_kinds, K=K, n_max=n_max, f_max=f_max,
        training=False, stage2a_waist=stage2a_waist, taps=taps,
    )

    Tensor.training = was_training

    snapshots = [np.asarray(taps["z_init"].numpy(), dtype=np.float32).copy()]
    for t in taps["wb_post_blend"]:
        snapshots.append(np.asarray(t.numpy(), dtype=np.float32).copy())
    return snapshots


# ---------------------------------------------------------------------------
# #237 §7 instrumentation: taps constructor, per-family / per-latent entropy,
# step-0 structural-mask verification
# ---------------------------------------------------------------------------

# Partition 1a family rows on the L axis (§2 first partition).
# Order matters: drives per-group assertion + per-group entropy logging.
V200_MASK_FAMILIES: "list[tuple[str, int, int]]" = [
    ("per_token", 0, 24),
    ("per_op",    24, 28),
    ("global",    28, 32),
]


def fg_v200_empty_taps() -> dict:
    """Empty taps dict for fg_breathing_forward_v200(taps=...). Eager-only."""
    return {
        "z_init": None,
        "wb_post_norm": [],
        "wb_post_read": [],
        "wb_post_think": [],
        "wb_post_norm_blend": [],
        "wb_post_blend": [],
        "xattn_weights": [],
        "sa_weights": [],
        "waist_contrib": [],
        "waist_breaths": [],
        "nb_attn": [],     # WRITE read-back attention per breath k>=1 (#238)
        "nb_slots": None,  # final K slot contents (#238)
    }


def _entropy_rows(w: np.ndarray) -> np.ndarray:
    """Entropy (nats) along the last axis of a probability array. Safe at w=0."""
    return -(w * np.log(w + 1e-12)).sum(axis=-1)


def xattn_entropy_per_family_per_breath(
    taps: dict,
    families: "list[tuple[str, int, int]] | None" = None,
) -> dict:
    """Per-mask-family cross-attn entropy per breath (§7, E.10 read 2).

    For each breath: per-query entropy over T (B, nh, L), then mean over batch,
    heads, and the family's latent rows. Per-group, never cross-group mean
    (§7 discipline); the all-latent mean is reported separately for the
    step-0 'mean strictly below log(24)' clause only.

    Returns {family: (K,) np.ndarray} plus "all_mean": (K,) and
    "per_latent": (K, L) (mean over B and heads, kept per query latent).
    """
    fams = families if families is not None else V200_MASK_FAMILIES
    K = len(taps["xattn_weights"])
    out: dict = {name: np.zeros(K) for name, _, _ in fams}
    L = int(taps["xattn_weights"][0].shape[2])
    out["all_mean"] = np.zeros(K)
    out["per_latent"] = np.zeros((K, L))
    for k in range(K):
        w = np.asarray(taps["xattn_weights"][k].numpy(), dtype=np.float64)  # (B, nh, L, T)
        h = _entropy_rows(w)                  # (B, nh, L)
        h_lat = h.mean(axis=(0, 1))           # (L,) per-query-latent
        out["per_latent"][k] = h_lat
        out["all_mean"][k] = float(h_lat.mean())
        for name, lo, hi in fams:
            out[name][k] = float(h_lat[lo:hi].mean())
    return out


def sa_per_latent_entropy_per_breath(taps: dict, exclude_self: bool = True) -> dict:
    """Per-latent THINK attention entropy (§7, added for #237; routes #238).

    For each breath, each query latent: entropy of its self-attention
    distribution inside Llama L0-L3. With exclude_self=True (the §7 / message-
    passing-memo definition) the diagonal is removed and rows renormalized, so
    the distribution is over the OTHER L-1 latents — random-init reference
    log(31) ≈ 3.434 for L=32.

    Aggregation: mean over batch, heads, and the 4 layers → (K, L) per-latent;
    headline "mean" (K,) and "std" (K,) are across the 32 query latents.
    Uniform low-std at training end → mean-field consensus mixing → #238 =
    THINK quotient-graph mask. Differentiated → self-organized selectivity.
    """
    K = len(taps["sa_weights"])
    if K == 0:
        return {"per_latent": np.zeros((0, 0)), "mean": np.zeros(0), "std": np.zeros(0)}
    L = int(taps["sa_weights"][0][0].shape[2])
    per_latent = np.zeros((K, L))
    for k in range(K):
        layer_h = []
        for w_t in taps["sa_weights"][k]:
            w = np.asarray(w_t.numpy(), dtype=np.float64)   # (B, nh, L, L)
            if exclude_self:
                eye = np.eye(L, dtype=np.float64).reshape(1, 1, L, L)
                w = w * (1.0 - eye)
                row_sum = w.sum(axis=-1, keepdims=True)
                w = w / (row_sum + 1e-12)
            h = _entropy_rows(w)               # (B, nh, L)
            layer_h.append(h.mean(axis=(0, 1)))  # (L,)
        per_latent[k] = np.stack(layer_h, axis=0).mean(axis=0)
    return {
        "per_latent": per_latent,                 # (K, L)
        "mean": per_latent.mean(axis=1),          # (K,) across latents
        "std": per_latent.std(axis=1),            # (K,) across latents
        "ref_log31": float(np.log(L - 1)) if exclude_self else float(np.log(L)),
    }


def verify_topology_mask_step0(
    ent: dict,
    mask_np: np.ndarray,
    eps: float = 0.01,
    families: "list[tuple[str, int, int]] | None" = None,
) -> dict:
    """§7 structural-mask verification at random init — the #237 launch gate.

    ent: the dict returned by xattn_entropy_per_family_per_breath (family name →
    per-breath entropy sequence, plus "all_mean"), computed on a random-init
    tapped forward.

    Per-group assertion, never mean-over-all-latents (§7): for each mask family,
    mean cross-attn entropy at breath 0 must be < log(family_support) + eps,
    where family_support = max row-sum of the mask over the family's rows
    (per-token: 1 → log(1)=0; per-op: 6 → log(6)≈1.79; global: 24 → log(24)≈3.18).
    Plus the E.10 clause: mean across ALL latents strictly below log(24)=3.178.

    Returns a result dict; raises AssertionError when the mask isn't wired
    ("fix and re-run", §1A.E.10 — not interpret).
    """
    fams = families if families is not None else V200_MASK_FAMILIES
    # V200_MASK_FAMILIES hardcodes partition-1a row ranges for L=32, T=24; any
    # other sizing silently slices wrong rows (review finding) — refuse it.
    if families is None:
        assert mask_np.shape == (32, 24), (
            f"V200_MASK_FAMILIES assumes mask shape (32, 24); got {mask_np.shape}. "
            "Pass explicit `families` for non-default partitions."
        )
    T = mask_np.shape[1]
    result = {"families": {}, "eps": eps}
    failures = []
    for name, lo, hi in fams:
        support = int(mask_np[lo:hi].sum(axis=1).max())
        bound = float(np.log(max(support, 1)))
        h0 = float(ent[name][0])                # breath 0
        h_all = [float(v) for v in ent[name]]   # all breaths, logged
        ok = h0 < bound + eps
        result["families"][name] = {
            "support": support, "log_support": bound,
            "entropy_breath0": h0, "entropy_per_breath": h_all, "pass": ok,
        }
        if not ok:
            failures.append(f"{name}: H={h0:.4f} >= log({support})+{eps}={bound + eps:.4f}")
    mean0 = float(ent["all_mean"][0])
    mean_bound = float(np.log(T))
    mean_ok = mean0 < mean_bound
    result["all_latent_mean_breath0"] = mean0
    result["all_latent_mean_bound_logT"] = mean_bound
    result["all_latent_mean_pass"] = mean_ok
    if not mean_ok:
        failures.append(f"all-latent mean: H={mean0:.4f} >= log(T)={mean_bound:.4f}")
    result["pass"] = len(failures) == 0
    assert result["pass"], (
        "STEP-0 MASK VERIFICATION FAILED — mask isn't wired (§1A.E.10: fix and "
        "re-run, do not train): " + "; ".join(failures)
    )
    return result


def compute_latent_jsd_from_snapshots(snapshots: list) -> list:
    """Compute JSD between consecutive latent snapshots.

    Uses the corrected pairwise inter-position cosine fingerprint method
    (same as FactorGraphV200.compute_latent_jsd_per_breath).
    Scale-invariant: L2 normalization before cosine computation.

    §7 method_sha identity: this function IS the reference implementation.
    Reference curves and training measurements both call this function from
    factor_graph_v200.py. The git SHA of this file == metric_sha.

    Args:
      snapshots: list of K+1 np.ndarray of shape (B, L, H)

    Returns:
      list of K floats — JSD(snapshot[k], snapshot[k+1]) for k=0..K-1
    """
    if len(snapshots) < 2:
        return []

    def _to_fingerprint(z_np: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(z_np, axis=-1, keepdims=True)
        z_n = z_np / (norms + 1e-8)                          # (B, L, H)
        gram = np.einsum('bld,bmd->blm', z_n, z_n)           # (B, L, L)
        L = z_np.shape[1]
        idx = np.triu_indices(L, k=1)
        fp = gram[:, idx[0], idx[1]]                         # (B, L*(L-1)//2)
        e = np.exp(fp - fp.max(axis=-1, keepdims=True))
        return e / (e.sum(axis=-1, keepdims=True) + 1e-8)    # (B, 496)

    def _jsd(p: np.ndarray, q: np.ndarray) -> float:
        eps = 1e-8
        p = np.clip(p, eps, 1.0); p = p / p.sum(axis=-1, keepdims=True)
        q = np.clip(q, eps, 1.0); q = q / q.sum(axis=-1, keepdims=True)
        m = 0.5 * (p + q)
        return float((0.5 * ((p * np.log(p / (m + eps))).sum(-1) +
                              (q * np.log(q / (m + eps))).sum(-1))).mean())

    return [_jsd(_to_fingerprint(snapshots[i]), _to_fingerprint(snapshots[i + 1]))
            for i in range(len(snapshots) - 1)]


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

        # NaN guard — #237.5 where-gated (§1A.E.14): multiply-gating passes
        # NaN through (NaN × 0 = NaN; #237's cascade poisoned Adam moments
        # straight through it). where() actually SELECTS: cond False → exact 0
        # regardless of NaN in the gradient. Still single-kernel-per-param,
        # still no per-param isnan (AM-safe).
        healthy_b = total.isfinite()
        healthy = healthy_b.cast(dtypes.float)
        for p in jit_params:
            if p.grad is not None:
                p.grad = healthy_b.where(p.grad, Tensor.zeros_like(p.grad))

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


# ===========================================================================
# Stage 1B class-based API (perceiver-core architecture per docs/v200_brief.md §15)
# ===========================================================================
#
# The functional API above (fg_breathing_forward_v200, attach_fg_params_v200,
# etc.) is the Stage 1 training-loop implementation. The class-based API below
# is the Stage 1B architecture proper — a self-contained module that wraps the
# backbone loader and exposes the canonical v200 interface.
#
# Both coexist in this file; the training driver (scripts/v200_train.py) uses
# the functional API; the arch smoke (scripts/v200_arch_smoke.py) and all
# Stage 1B+ tests use the class-based API.

import json
import math as _math
from typing import Any as _Any, Dict as _Dict, List as _List, Optional as _Optional, Tuple as _Tuple

from tinygrad.nn import Linear as _Linear, RMSNorm as _RMSNorm

from mycelium.llama_base import LlamaBase as _LlamaBase


# ---------------------------------------------------------------------------
# Architecture constants (Stage 1B)
# ---------------------------------------------------------------------------

_N_LATENTS    = 32     # L
_HIDDEN_DIM   = 2048   # H
_WAIST_DIM    = 512    # 4× compression
_PYTHIA_DIM   = 1024   # IB centroids are in Pythia space
_K_DEFAULT    = 8      # default breath count
_N_DIGITS     = 5      # digit positions in tree codebook
_N_DIGIT_VALS = 10     # digits 0-9
_XATTN_HEADS  = 16     # cross-attention head count
_XATTN_HEAD_DIM = _HIDDEN_DIM // _XATTN_HEADS   # 128

_PROJECT_ROOT_1B = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_IB_CENTROIDS_PATH = os.path.join(
    _PROJECT_ROOT_1B, ".cache", "ib_centroids_gsm8k_partial.npz"
)


# ---------------------------------------------------------------------------
# LatentInit (Stage 1B)
# ---------------------------------------------------------------------------

class LatentInit:
    """Init 32 latents from IB centroids (Pythia 1024d → 2048d) + jitter.

    Projection: learnable Linear(1024 → 2048).
    Jitter: 0.01 × randn for symmetry-breaking (near-duplicate pairs at
      cos_sim > 0.97 in native Pythia space need distinct starting points).
    """

    def __init__(self, centroids_path: str = _IB_CENTROIDS_PATH,
                 target_dim: int = _HIDDEN_DIM, jitter_std: float = 0.01):
        self.target_dim  = target_dim
        self.jitter_std  = jitter_std

        if os.path.exists(centroids_path):
            d = np.load(centroids_path)
            keys = list(d.keys())
            raw = d["centroids"].astype(np.float32) if "centroids" in keys \
                  else d[keys[0]].astype(np.float32)
            if raw.ndim == 1:
                raw = raw.reshape(-1, _PYTHIA_DIM)
            if raw.shape[1] != _PYTHIA_DIM:
                raise ValueError(
                    f"Expected IB centroids dim {_PYTHIA_DIM}, got {raw.shape[1]}"
                )
            n = raw.shape[0]
            if n < _N_LATENTS:
                pad = np.random.randn(_N_LATENTS - n, _PYTHIA_DIM).astype(np.float32)
                pad /= (np.linalg.norm(pad, axis=-1, keepdims=True) + 1e-8)
                raw = np.concatenate([raw, pad], axis=0)
            centroids_np = raw[:_N_LATENTS]
        else:
            print(f"[v200-1b] WARNING: IB centroids not found at {centroids_path}. "
                  "Using random init.")
            centroids_np = np.random.randn(_N_LATENTS, _PYTHIA_DIM).astype(np.float32)
            centroids_np /= (np.linalg.norm(centroids_np, axis=-1, keepdims=True) + 1e-8)

        self.proj = _Linear(_PYTHIA_DIM, target_dim)
        self._centroids_raw = Tensor(centroids_np)

    def init(self, batch_size: int, training: bool = True) -> Tensor:
        """Return (B, L=32, H) latent init tensor."""
        base = self._centroids_raw @ self.proj.weight.T
        if self.proj.bias is not None:
            base = base + self.proj.bias
        base = base.reshape(1, _N_LATENTS, self.target_dim)
        base = base.expand(batch_size, _N_LATENTS, self.target_dim)
        if training and self.jitter_std > 0.0:
            noise = Tensor.randn(batch_size, _N_LATENTS, self.target_dim) * self.jitter_std
            return base + noise
        return base


# ---------------------------------------------------------------------------
# CrossAttention (Stage 1B)
# ---------------------------------------------------------------------------

class CrossAttention:
    """Multi-head cross-attention: Q from latents, K/V from fg tokens.

    Q: (B, L=32, H=2048) — latents
    K/V: (B, T=24, H=2048) — static fg token embeddings
    Output: (B, L=32, H=2048)

    Full L×T attention (no topology routing; that's v1.1 row 1).
    No RoPE on cross-attention (latents have no positional semantics).
    """

    def __init__(self, d: int = _HIDDEN_DIM, n_heads: int = _XATTN_HEADS):
        self.d        = d
        self.n_heads  = n_heads
        self.head_dim = d // n_heads
        self.wq = _Linear(d, d)
        self.wk = _Linear(d, d)
        self.wv = _Linear(d, d)
        self.wo = _Linear(d, d)

    def forward(self, q_src: Tensor, kv_src: Tensor,
                return_weights: bool = False
                ) -> _Tuple[Tensor, _Optional[Tensor]]:
        """Cross-attend: q_src (B, L, H), kv_src (B, T, H) → out (B, L, H)."""
        B, L, H = q_src.shape
        _, T, _  = kv_src.shape
        nh = self.n_heads
        hd = self.head_dim

        q = self.wq(q_src).reshape(B, L, nh, hd).transpose(1, 2)   # (B, nh, L, hd)
        k = self.wk(kv_src).reshape(B, T, nh, hd).transpose(1, 2)   # (B, nh, T, hd)
        v = self.wv(kv_src).reshape(B, T, nh, hd).transpose(1, 2)   # (B, nh, T, hd)

        scale  = 1.0 / _math.sqrt(hd)
        scores = (q @ k.transpose(-2, -1)) * scale   # (B, nh, L, T)
        scores = scores.clip(-1e4, 1e4)
        attn_w = scores.softmax(-1)

        out = (attn_w @ v).transpose(1, 2).reshape(B, L, H)   # (B, L, H)
        out = self.wo(out)

        return (out, attn_w) if return_weights else (out, None)


# ---------------------------------------------------------------------------
# Waist (Stage 1B)
# ---------------------------------------------------------------------------

class Waist:
    """2048 → 512 → 2048 latent waist (COMMIT phase, even breaths).

    Zero-init up_proj → identity residual bypass at step 0.
    """

    def __init__(self, d_in: int = _HIDDEN_DIM, d_waist: int = _WAIST_DIM,
                 d_out: int = _HIDDEN_DIM):
        self.down_proj = _Linear(d_in, d_waist)
        self.waist_ln  = _RMSNorm(d_waist)
        self.up_proj   = _Linear(d_waist, d_out)
        self.up_proj.weight.assign(Tensor.zeros_like(self.up_proj.weight)).realize()
        if self.up_proj.bias is not None:
            self.up_proj.bias.assign(Tensor.zeros_like(self.up_proj.bias)).realize()

    def forward(self, z: Tensor) -> Tensor:
        h = self.down_proj(z)
        h = self.waist_ln(h)
        h = h.gelu()
        return z + self.up_proj(h)


# ---------------------------------------------------------------------------
# Tree Codebook Readout (Stage 1B)
# ---------------------------------------------------------------------------

class TreeCodebookReadout:
    """Mean-pool latents → MLP → (B, n_digits, n_vals).

    Option (a) from brief §15 pitfall 6: simple mean-pool then MLP.
    """

    def __init__(self, d: int = _HIDDEN_DIM, n_digits: int = _N_DIGITS,
                 n_vals: int = _N_DIGIT_VALS):
        self.n_digits = n_digits
        self.n_vals   = n_vals
        hidden = d // 2
        self.fc1 = _Linear(d, hidden)
        self.fc2 = _Linear(hidden, n_digits * n_vals)

    def forward(self, z: Tensor) -> Tensor:
        pooled = z.mean(axis=1)
        h = self.fc1(pooled).gelu()
        out = self.fc2(h)
        return out.reshape(out.shape[0], self.n_digits, self.n_vals)


# ---------------------------------------------------------------------------
# Calibration Head (Stage 1B)
# ---------------------------------------------------------------------------

class CalibHead:
    """Mean-pool latents → 2-layer MLP → sigmoid scalar per breath."""

    def __init__(self, d: int = _HIDDEN_DIM):
        mid = d // 4
        self.fc1 = _Linear(d, mid)
        self.fc2 = _Linear(mid, 1)

    def forward(self, z: Tensor) -> Tensor:
        pooled = z.mean(axis=1)
        h = self.fc1(pooled).gelu()
        out = self.fc2(h)
        return out.reshape(out.shape[0]).sigmoid()


# ---------------------------------------------------------------------------
# V200Config (Stage 1B)
# ---------------------------------------------------------------------------

class V200Config:
    """Configuration dataclass for FactorGraphV200."""

    def __init__(self,
                 base_model_id: _Optional[str] = None,
                 n_latents: int = _N_LATENTS,
                 hidden_dim: int = _HIDDEN_DIM,
                 waist_dim: int = _WAIST_DIM,
                 xattn_heads: int = _XATTN_HEADS,
                 k_max: int = _K_DEFAULT,
                 n_tokens: int = 24,
                 n_digits: int = _N_DIGITS,
                 n_digit_vals: int = _N_DIGIT_VALS,
                 jitter_std: float = 0.01,
                 centroids_path: str = _IB_CENTROIDS_PATH):
        self.base_model_id  = base_model_id
        self.n_latents      = n_latents
        self.hidden_dim     = hidden_dim
        self.waist_dim      = waist_dim
        self.xattn_heads    = xattn_heads
        self.k_max          = k_max
        self.n_tokens       = n_tokens
        self.n_digits       = n_digits
        self.n_digit_vals   = n_digit_vals
        self.jitter_std     = jitter_std
        self.centroids_path = centroids_path


# ---------------------------------------------------------------------------
# FactorGraphV200 (Stage 1B main architecture class)
# ---------------------------------------------------------------------------

class FactorGraphV200:
    """v200 Perceiver-Core Breathing Transformer (Stage 1B class-based API).

    Architecture per breath k:
      READ:    read_ctx = cross_attend(Q=latents, K=fg_tokens, V=fg_tokens)
               latents  = latents + read_ctx
      THINK:   for layer in backbone L0-L3:
                   latents = layer(latents)
      COMMIT:  if k % 2 == 0: latents = waist(latents)
      (delta_gate blend at each breath end)
      WRITE:   deferred to v201

    Key design:
      - 32 latents at H=2048 are the PRIMARY state
      - Factor graph tokens are STATIC (embedded once, never iterated)
      - IB-anchored init: 32 Pythia centroids projected 1024→2048 + 0.01·randn jitter
      - Conservative wv: each backbone layer's own wv (per brief §4)
      - Full L×T cross-attention (no topology routing at Stage 1B)

    Stage 1B does NOT run training. Verify:
      - K=8 forward on random tokens: no NaN, correct shapes
      - Persistence/provenance/instrumentation hooks fire correctly
    """

    # One-time stale-architecture warning flag (#237 review; see forward()).
    _stale_arch_warned = False

    def __init__(self, config: _Optional[V200Config] = None):
        if config is None:
            config = V200Config()
        self.config = config

        print("[v200-1b] Loading backbone...")
        self.backbone = _LlamaBase(weights_path=config.base_model_id)
        print(f"[v200-1b] Backbone: H={self.backbone.hidden_size} "
              f"GQA={self.backbone.is_gqa} "
              f"n_heads={self.backbone.n_heads} n_kv_heads={self.backbone.n_kv_heads}")

        assert self.backbone.hidden_size == config.hidden_dim, (
            f"Backbone hidden_size {self.backbone.hidden_size} != "
            f"config.hidden_dim {config.hidden_dim}"
        )

        self.latent_init = LatentInit(
            centroids_path=config.centroids_path,
            target_dim=config.hidden_dim,
            jitter_std=config.jitter_std,
        )
        self.read_xattn = CrossAttention(d=config.hidden_dim, n_heads=config.xattn_heads)
        self.waist       = Waist(d_in=config.hidden_dim, d_waist=config.waist_dim,
                                 d_out=config.hidden_dim)
        self.tree_readout = TreeCodebookReadout(d=config.hidden_dim,
                                                n_digits=config.n_digits,
                                                n_vals=config.n_digit_vals)
        self.calib_head   = CalibHead(d=config.hidden_dim)

        # Four RMSNorms on the latent loop (§2, four required after #233/#234).
        # Without per-breath normalization the latent residual accumulates unboundedly
        # across K breaths (‖z‖ 0.87 → 5345 at K=4, 6000×), making cos≈1 arithmetic
        # and softmax-JSD collapse arithmetic — not dynamics findings.
        # norm_breath added Jun 11 #234 (spec-restore): bounds inter-breath accumulation.
        # norm_read_ctx added Jun 11 #235 (§1A.E.4): bounds READ contribution before residual add.
        # Each has learnable gain init=1.0 (Llama RMSNorm convention).
        self.norm_breath   = _RMSNorm(config.hidden_dim)  # breath boundary pre-norm (§2 4th norm, #234)
        self.norm_read     = _RMSNorm(config.hidden_dim)  # pre-norm on Q before cross-attn
        self.norm_read_ctx = _RMSNorm(config.hidden_dim)  # norm on read_ctx before residual add (#235)
        self.norm_commit   = _RMSNorm(config.hidden_dim)  # pre-norm on waist input (even breaths)
        self.norm_readout  = _RMSNorm(config.hidden_dim)  # pre-norm before tree codebook readout

        # α scalar for READ residual: init=1.0 (NOT zero-init — READ is the information inlet
        # per §1A.E.8; zero-init creates bootstrap bottleneck mirroring v118-v121 failure).
        # Shape (1,) so it broadcasts over (B, L, H) without per-breath differentiation.
        # If this drifts large (>5) at step 200, that's the strain signal for §10 row 5.
        self.alpha_read = Tensor.ones(1)

        # Per-breath marker: (K_max, H) zero-init (identity at step 0)
        self.breath_embed = Tensor.zeros(config.k_max, config.hidden_dim)

        # Per-breath delta_gate: (K_max,) init=-2.0 (sigmoid→0.119, spec-restore #234 Jun 11)
        # §2: gate_k = sigmoid(delta_gate[k]); init=-2.0 per documented Pythia cold-start finding.
        # gate=-5.0 (α=0.007) starved gradient; gate=-2.0 (α=0.119) = 240× faster growth.
        self.delta_gate = Tensor.full((config.k_max,), -2.0)

        print(f"[v200-1b] FactorGraphV200 ready: "
              f"L={config.n_latents} H={config.hidden_dim} "
              f"waist={config.waist_dim} K_max={config.k_max} "
              f"xattn_heads={config.xattn_heads}")

    # ------------------------------------------------------------------
    # Static embedding
    # ------------------------------------------------------------------

    def embed_fg_tokens(self, fg_tokens: Tensor) -> Tensor:
        """Embed factor graph tokens via backbone embed + L0-L3 + ln_f.

        fg_tokens: (B, T) int32
        Returns:   (B, T, H=2048) — static reference for all K breaths
        """
        return self.backbone.forward(fg_tokens)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, fg_tokens: Tensor, K: _Optional[int] = None,
                training: bool = True,
                return_taps: bool = False
                ) -> _Tuple[Tensor, _List[_Dict[str, _Any]]]:
        """Forward through K breaths of READ/THINK/COMMIT.

        fg_tokens: (B, T) int32 factor-graph token IDs
        K:         number of breaths (≤ config.k_max)
        training:  if True, adds jitter to latent init
        return_taps: if True, collect per-breath traces

        Returns:
          tree_logits: (B, n_digits=5, n_digit_vals=10)
          traces:      list of K dicts (empty list if return_taps=False)

        Trace dict keys per breath k:
          k, z, z_pre, read_ctx, read_ctx_norm, xattn_weights,
          think_taps, delta_gate_val, calib, tree_logits, self_attn_resid_norm

        STALE-ARCHITECTURE WARNING (#237 review): this class implements the
        Stage-1B architecture — full L×T cross-attention (NO §2 topology mask)
        and 5 RMSNorms (NO Seam-3 norm_blend from #236). The canonical trained
        forward is the module-level fg_breathing_forward_v200 (+ taps for
        instrumentation). Do NOT use this class to measure anything under a
        mask1a-era arch_version — it measures a different architecture.
        """
        if not FactorGraphV200._stale_arch_warned:
            print(
                "[v200 WARNING] FactorGraphV200.forward is the STALE Stage-1B "
                "architecture (no §2 topology mask, no Seam-3 norm_blend). "
                "Canonical forward: fg_breathing_forward_v200. Do not measure "
                "mask1a-era runs through this class.",
                flush=True,
            )
            FactorGraphV200._stale_arch_warned = True
        if K is None:
            K = self.config.k_max
        assert K <= self.config.k_max, (
            f"K={K} > config.k_max={self.config.k_max}"
        )

        B, T = fg_tokens.shape

        # Static embedding (once per forward)
        fg_emb = self.embed_fg_tokens(fg_tokens)   # (B, T, H)

        # Initialize latents
        z = self.latent_init.init(B, training=training)   # (B, L, H)

        traces: _List[_Dict[str, _Any]] = []

        for k in range(K):
            # Breath boundary RMSNorm — §2 4th norm (spec-restore #234, Jun 11)
            # Bounds inter-breath residual accumulation before adding breath embed.
            # z_pre_breath captured AFTER norm so the gate blend mixes normalized
            # pre-breath state with post-THINK state (both at bounded scale).
            z = self.norm_breath(z)
            z_pre_breath = z    # ← captured POST-norm: prevents pre/post scale mismatch in gate

            # Per-breath marker
            z = z + self.breath_embed[k]

            # READ — pre-norm Q before cross-attention (§2, norm_read, Jun 11)
            z_q = self.norm_read(z)
            if return_taps:
                read_ctx, xattn_weights = self.read_xattn.forward(
                    q_src=z_q, kv_src=fg_emb, return_weights=True
                )
            else:
                read_ctx, xattn_weights = self.read_xattn.forward(
                    q_src=z_q, kv_src=fg_emb, return_weights=False
                )
            # §1A.E.4 READ-dominance fix (#235, Jun 11): α·RMSNorm(read_ctx)
            # Bounds read_ctx to per-elem ~1 before residual add; α init=1.0
            # (NOT zero-init per §1A.E.8 — READ is the information inlet).
            z = z + self.alpha_read * self.norm_read_ctx(read_ctx)

            # THINK — collect per-layer residual taps + self-attn weights in one pass
            z_pre_think = z
            if return_taps:
                z, think_taps, self_attn_weights_per_layer = \
                    self.backbone.forward_latents_with_taps_and_attn_weights(z)
            else:
                z = self.backbone.forward_latents(z)
                think_taps = None
                self_attn_weights_per_layer = None

            # COMMIT (even breaths only) — pre-norm waist input (§2, norm_commit, Jun 11)
            z_pre_waist = z
            waist_applied_this_breath = (k % 2 == 0)
            if waist_applied_this_breath:
                z_w = self.norm_commit(z)
                z = self.waist.forward(z_w)
            z_post_waist = z

            # Delta gate blend — §2: gate_k = sigmoid(delta_gate[k]) (spec-restore #234)
            gate_k = self.delta_gate[k].sigmoid()
            z = z_pre_breath + gate_k * (z - z_pre_breath)

            if return_taps:
                read_ctx_arr = read_ctx.float().numpy()
                read_ctx_norm = float(
                    np.linalg.norm(read_ctx_arr.reshape(B, -1), axis=-1).mean()
                )
                if think_taps is not None:
                    last_tap = think_taps[-1]
                    pre_arr  = last_tap["pre_ln_resid"].float().numpy()
                    post_arr = last_tap["post_mlp_resid"].float().numpy()
                    sa_resid_norm = float(
                        np.linalg.norm((post_arr - pre_arr).reshape(B, -1), axis=-1).mean()
                    )
                else:
                    sa_resid_norm = 0.0

                # Waist delta and applied-flag for alternation verification.
                # NOTE: at random init, up_proj=zeros so waist_delta=0 even on
                # even breaths. The code-path distinction is captured by
                # waist_applied (bool: did the waist.forward() call happen?),
                # not by delta magnitude. Post-training, even-breath delta will
                # be nonzero; at init it is zero by design (bootstrap-safe init).
                waist_delta = float(
                    np.linalg.norm(
                        (z_post_waist.float().numpy() - z_pre_waist.float().numpy())
                        .reshape(B, -1), axis=-1
                    ).mean()
                )

                traces.append({
                    "k":                          k,
                    "z":                          z,
                    "z_pre":                      z_pre_breath,
                    "read_ctx":                   read_ctx,
                    "read_ctx_norm":              read_ctx_norm,
                    "xattn_weights":              xattn_weights,
                    "think_taps":                 think_taps,
                    "self_attn_weights_per_layer": self_attn_weights_per_layer,
                    "delta_gate_val":             float(gate_k.float().numpy()),
                    "calib":                      self.calib_head.forward(z),
                    # per-breath readout uses norm_readout (§2, norm_readout, Jun 11)
                    "tree_logits":                self.tree_readout.forward(self.norm_readout(z)),
                    "self_attn_resid_norm":       sa_resid_norm,
                    "waist_delta":                waist_delta,
                    "waist_applied":              waist_applied_this_breath,
                })

        # Final pre-norm before tree codebook readout (§2, norm_readout, Jun 11)
        tree_logits = self.tree_readout.forward(self.norm_readout(z))
        return tree_logits, traces

    # ------------------------------------------------------------------
    # Instrumentation (§7, ε=5% convention)
    # ------------------------------------------------------------------

    def compute_latent_jsd_per_breath(
        self, traces: _List[_Dict[str, _Any]]
    ) -> _List[float]:
        """JSD between consecutive breath latent distributions (K-1 values).

        CORRECTED (2026-06-11): The original implementation applied softmax over
        the H=2048 hidden dimension, then averaged over L=32 latent positions.
        BUG: at large ||z|| (trained model), softmax over 2048 dims saturates to a
        near-degenerate peak on one or two dimensions — scale-insensitive. Two latent
        tensors that differ dramatically in content but share the same dominant
        dimension produce JSD ≈ 0. This caused the metric to report [0.69, 0, 0, 0]
        and was misread as "fixed-point collapse by breath 1."

        FIX: Replace hidden-dim softmax with pairwise inter-position cosine
        similarity fingerprints. For each breath snapshot, compute the L2-normalised
        pairwise cosine similarity among the L=32 latent positions (L*(L-1)/2 = 496
        scalars). Treat those 496 cosines as a raw signal, apply softmax over 496
        (not 2048) to get a distribution, then compute JSD between consecutive
        breaths' distributions.

        Scale-invariance: L2 normalisation before cosine computation means ||z||
        does not affect the result.

        Reference curve compatibility: the reference at
        .cache/v200_smoke/reference_curves/latent_jsd_random_init.npz was generated
        with the OLD (broken) metric. It MUST be re-generated under this corrected
        metric before any apples-to-apples Gate B comparison is valid. Flag raised in
        step200_eval_corrected.json.
        """
        if len(traces) < 2:
            return []

        def _to_fingerprint(z_tensor: Tensor) -> np.ndarray:
            # z_tensor: tinygrad Tensor of shape (B, L, H)
            z_np = z_tensor.float().numpy()  # (B, L, H)
            # L2-normalise each latent position vector
            norms = np.linalg.norm(z_np, axis=-1, keepdims=True)  # (B, L, 1)
            z_n = z_np / (norms + 1e-8)                           # (B, L, H)
            # Pairwise cosine similarity: Gram matrix (B, L, L)
            gram = np.einsum('bld,bmd->blm', z_n, z_n)
            # Extract upper triangle: L*(L-1)//2 scalars per batch element
            L = z_np.shape[1]
            idx = np.triu_indices(L, k=1)
            fp = gram[:, idx[0], idx[1]]  # (B, L*(L-1)//2)
            # Softmax over 496 pairs → distribution capturing inter-position topology
            e = np.exp(fp - fp.max(axis=-1, keepdims=True))
            return e / (e.sum(axis=-1, keepdims=True) + 1e-8)     # (B, 496)

        def _jsd(p: np.ndarray, q: np.ndarray) -> float:
            eps = 1e-8
            p = np.clip(p, eps, 1.0); p = p / p.sum(axis=-1, keepdims=True)
            q = np.clip(q, eps, 1.0); q = q / q.sum(axis=-1, keepdims=True)
            m = 0.5 * (p + q)
            return float((0.5 * ((p * np.log(p/(m+eps))).sum(-1) +
                                  (q * np.log(q/(m+eps))).sum(-1))).mean())

        return [_jsd(_to_fingerprint(traces[i]["z"]), _to_fingerprint(traces[i+1]["z"]))
                for i in range(len(traces) - 1)]

    def compute_energy_channel(
        self, traces: _List[_Dict[str, _Any]]
    ) -> np.ndarray:
        """Per-latent ‖Δz_j‖ per breath → (K, L) array."""
        if not traces:
            return np.zeros((0, self.config.n_latents))
        K = len(traces)
        L = self.config.n_latents
        energy = np.zeros((K, L))
        for k, trace in enumerate(traces):
            z_post = trace["z"].float().numpy()
            z_pre  = trace["z_pre"].float().numpy()
            delta  = z_post - z_pre
            energy[k] = np.linalg.norm(delta, axis=-1).mean(axis=0)
        return energy

    def compute_xattn_head_group_entropy(
        self, traces: _List[_Dict[str, _Any]]
    ) -> np.ndarray:
        """Per-head cross-attention entropy per breath → (K, n_heads) array."""
        if not traces or traces[0]["xattn_weights"] is None:
            return np.zeros((len(traces), self.config.xattn_heads))
        K  = len(traces)
        nh = self.config.xattn_heads
        entropies = np.zeros((K, nh))
        for k, trace in enumerate(traces):
            w = trace["xattn_weights"]
            if w is None:
                continue
            w_np = np.clip(w.float().numpy(), 1e-8, 1.0)
            h = -(w_np * np.log(w_np)).sum(axis=-1)   # (B, n_heads, L)
            entropies[k] = h.mean(axis=(0, 2))
        return entropies

    def compute_self_attn_layer_jsd(
        self, traces: _List[_Dict[str, _Any]]
    ) -> np.ndarray:
        """JSD between adjacent self-attn outputs across breaths → (K-1, 4)."""
        if len(traces) < 2 or traces[0]["think_taps"] is None:
            return np.zeros((max(0, len(traces) - 1), 4))

        def _to_dist(arr: np.ndarray) -> np.ndarray:
            e = np.exp(arr - arr.max(axis=-1, keepdims=True))
            s = e / (e.sum(axis=-1, keepdims=True) + 1e-8)
            return s.mean(axis=1)

        def _jsd(p: np.ndarray, q: np.ndarray) -> float:
            eps = 1e-8
            p = np.clip(p, eps, 1.0); p /= p.sum(axis=-1, keepdims=True)
            q = np.clip(q, eps, 1.0); q /= q.sum(axis=-1, keepdims=True)
            m = 0.5 * (p + q)
            return float((0.5 * ((p * np.log(p/(m+eps))).sum(-1) +
                                  (q * np.log(q/(m+eps))).sum(-1))).mean())

        K = len(traces)
        result = np.zeros((K - 1, 4))
        for k in range(K - 1):
            taps_k   = traces[k]["think_taps"]
            taps_kp1 = traces[k + 1]["think_taps"]
            for l in range(4):
                a = taps_k[l]["post_attn_resid"].float().numpy()
                b = taps_kp1[l]["post_attn_resid"].float().numpy()
                result[k, l] = _jsd(_to_dist(a), _to_dist(b))
        return result

    def compute_xattn_entropy_per_breath(
        self, traces: _List[_Dict[str, _Any]]
    ) -> np.ndarray:
        """Attention-weight entropy of cross-attention per breath → (K, n_heads) nats.

        Same as compute_xattn_head_group_entropy but renamed for the like-units
        Gate B clause (§8). Both names are aliases; the computation is identical.

        H(w) = -sum_i w_i * log(w_i), w over T=24 fg tokens per latent position.
        At random init: H ≈ log(T) = log(24) ≈ 3.18 nats (fully diffuse).
        Units: nats (natural log).
        """
        return self.compute_xattn_head_group_entropy(traces)

    def compute_self_attn_entropy_per_breath(
        self, traces: _List[_Dict[str, _Any]]
    ) -> np.ndarray:
        """Attention-weight entropy of self-attention per breath → (K, 4, n_sa_heads) nats.

        For each THINK-phase self-attn layer (4 layers) and each head,
        computes entropy of the attention distribution over L=32 latent keys.

        H(w) = -sum_i w_i * log(w_i), w over L=32 latent positions.
        At random init: H ≈ log(L) = log(32) ≈ 3.47 nats (fully diffuse).
        Units: nats (natural log).

        Returns (K, 4, n_heads) where n_heads is the backbone's Q head count.
        If self_attn_weights_per_layer is not in traces, returns zeros.
        """
        if not traces or traces[0].get("self_attn_weights_per_layer") is None:
            return np.zeros((len(traces), 4, self.backbone.n_heads))

        K = len(traces)
        n_sa_heads = self.backbone.n_heads
        entropies = np.zeros((K, 4, n_sa_heads))

        for k, trace in enumerate(traces):
            layers_w = trace.get("self_attn_weights_per_layer")
            if layers_w is None:
                continue
            for li, attn_w in enumerate(layers_w):
                if attn_w is None:
                    continue
                # attn_w: (B, n_heads, L, L) — attention weight over L latent keys
                w_np = np.clip(attn_w.float().numpy(), 1e-8, 1.0)
                # H = -sum_i w_i * log(w_i) over last dim (keys)
                h = -(w_np * np.log(w_np)).sum(axis=-1)   # (B, n_heads, L)
                # Mean over batch and query positions
                entropies[k, li] = h.mean(axis=(0, 2))    # (n_heads,)

        return entropies

    def compute_inter_position_cosine_mean_removed_per_breath(
        self, traces: _List[_Dict[str, _Any]]
    ) -> np.ndarray:
        """Mean-removed inter-position cosine per breath → (K,) array.

        For each breath k:
          1. Subtract the across-position mean from each latent position:
             z_centered[b, l, :] = z[b, l, :] - z[b, :, :].mean(axis=0)
          2. L2-normalise each centered latent.
          3. Compute mean pairwise cosine of the L=32 centered latent vectors.

        Interpretation:
          1.0 = all centered latents are identical (position-collapse)
          0.0 = orthogonal (good diversity)

        Mean-removing before cosine removes the shared-additive dominance
        artifact: when diffuse cross-attention adds the same mean-of-tokens
        vector to all latents, raw inter-position cosine → 1 by arithmetic
        (cos(a+C, b+C) → 1 when ‖C‖ ≫ ‖a‖,‖b‖). Mean-removed cosine is
        the component that pure arithmetic can't fake. This is the metric
        for §1A.E.4 position-collapse disambiguation (added Jun 11).

        Added Jun 11 per §7 and §1A.E.4.
        """
        if not traces:
            return np.zeros(0)

        results = []
        for trace in traces:
            z_tensor = trace["z"]
            z_np = z_tensor.float().numpy()        # (B, L, H)
            B, L, H_dim = z_np.shape

            # Subtract across-position mean (shared additive component)
            z_mean = z_np.mean(axis=1, keepdims=True)  # (B, 1, H)
            z_centered = z_np - z_mean                  # (B, L, H)

            # L2-normalise each centered latent
            norms = np.linalg.norm(z_centered, axis=-1, keepdims=True)  # (B, L, 1)
            z_n = z_centered / (norms + 1e-8)                            # (B, L, H)

            # Gram matrix of normalised centered latents
            gram = np.einsum('bld,bmd->blm', z_n, z_n)  # (B, L, L)

            # Mean over upper-triangle (excluding diagonal)
            idx = np.triu_indices(L, k=1)
            upper = gram[:, idx[0], idx[1]]              # (B, L*(L-1)/2)
            results.append(float(upper.mean()))

        return np.array(results, dtype=np.float32)

    def compute_read_dominance_ratio_per_breath(
        self, traces: _List[_Dict[str, _Any]]
    ) -> np.ndarray:
        """‖read_ctx‖ / ‖z_pre_breath‖ per breath → (K,) array.

        For each breath k: mean over B and L of the ratio
          ‖read_ctx[b, l, :]‖ / (‖z_pre[b, l, :]‖ + ε)

        Interpretation:
          < 1  = READ contribution smaller than pre-breath state (healthy)
          1-5  = READ dominant but bounded (acceptable)
          > 10 = READ operator numerically dominates (substrate fix didn't contain it)

        This is the ‖read_ctx‖/‖z_pre_breath‖ metric for §1A.E.4 cell 3
        (READ dominance). Added Jun 11 per §7 and §1A.E.4.
        """
        if not traces:
            return np.zeros(0)

        results = []
        for trace in traces:
            read_ctx_t = trace.get("read_ctx")
            z_pre_t    = trace.get("z_pre")
            if read_ctx_t is None or z_pre_t is None:
                results.append(float('nan'))
                continue

            rc_np  = read_ctx_t.float().numpy()     # (B, L, H)
            zp_np  = z_pre_t.float().numpy()        # (B, L, H)

            rc_norm = np.linalg.norm(rc_np,  axis=-1)   # (B, L)
            zp_norm = np.linalg.norm(zp_np,  axis=-1)   # (B, L)

            ratio = rc_norm / (zp_norm + 1e-8)          # (B, L)
            results.append(float(ratio.mean()))

        return np.array(results, dtype=np.float32)

    def check_waist_alternation(
        self, traces: _List[_Dict[str, _Any]]
    ) -> _Dict[str, _Any]:
        """Verify waist CODE PATH fires on even breaths only.

        The check uses the "waist_applied" boolean flag (True = waist.forward()
        was called), NOT the numerical delta. At random init, up_proj is
        zero-initialized so waist_delta=0 on even breaths too (bootstrap-safe).
        The code-path distinction is what matters: even breaths call
        waist.forward(), odd breaths skip it. Post-training, even-breath
        delta grows as up_proj learns; at init delta=0 by design.

        Returns dict with:
          "even_applied":  list of waist_applied bools for even-k breaths
          "odd_applied":   list of waist_applied bools for odd-k breaths
          "even_deltas":   list of waist_delta floats for even-k breaths
          "odd_deltas":    list of waist_delta floats for odd-k breaths
          "fires_correctly": bool — even all True AND odd all False
          "verdict": "YES" / "NO" / "UNKNOWN"
          "note":    explanation (esp. re: zero delta at random init)
        """
        if not traces:
            return {
                "even_applied": [], "odd_applied": [],
                "even_deltas": [], "odd_deltas": [],
                "fires_correctly": False, "verdict": "UNKNOWN",
                "note": "no traces"
            }

        # Prefer waist_applied flag (code-path check) over delta (numerical)
        if "waist_applied" in traces[0]:
            even_applied = [t["waist_applied"] for t in traces if t["k"] % 2 == 0]
            odd_applied  = [t["waist_applied"] for t in traces if t["k"] % 2 != 0]
            even_deltas  = [t.get("waist_delta", 0.0) for t in traces if t["k"] % 2 == 0]
            odd_deltas   = [t.get("waist_delta", 0.0) for t in traces if t["k"] % 2 != 0]

            even_fires = all(applied for applied in even_applied) if even_applied else False
            odd_skips  = all(not applied for applied in odd_applied) if odd_applied else True
            fires_correctly = even_fires and odd_skips
            verdict = "YES" if fires_correctly else "NO"

            note = (
                "waist_applied flag used (code-path check). "
                "At random init, up_proj=zeros → waist_delta=0 on even breaths too; "
                "delta grows post-training. Code path (not delta) is the check."
            )
            return {
                "even_applied":    even_applied,
                "odd_applied":     odd_applied,
                "even_deltas":     even_deltas,
                "odd_deltas":      odd_deltas,
                "fires_correctly": fires_correctly,
                "verdict":         verdict,
                "note":            note,
            }

        # Fallback: no waist_applied flag in traces (old-style traces)
        if "waist_delta" not in traces[0]:
            return {
                "even_applied": [], "odd_applied": [],
                "even_deltas": [], "odd_deltas": [],
                "fires_correctly": False, "verdict": "UNKNOWN",
                "note": "neither waist_applied nor waist_delta in traces"
            }

        even_deltas = [t["waist_delta"] for t in traces if t["k"] % 2 == 0]
        odd_deltas  = [t["waist_delta"] for t in traces if t["k"] % 2 != 0]
        even_nonzero = all(d > 1e-6 for d in even_deltas) if even_deltas else False
        odd_zero     = all(d < 1e-6 for d in odd_deltas)  if odd_deltas  else True
        fires_correctly = even_nonzero and odd_zero
        verdict = "YES" if fires_correctly else "NO"

        return {
            "even_applied": [True]*len(even_deltas),
            "odd_applied":  [False]*len(odd_deltas),
            "even_deltas":  even_deltas,
            "odd_deltas":   odd_deltas,
            "fires_correctly": fires_correctly,
            "verdict": verdict,
            "note": "delta-magnitude fallback (waist_applied not in traces)",
        }

    def print_freeze_table(self, traces: _List[_Dict[str, _Any]],
                           half_k: _Optional[int] = None) -> None:
        """Print freeze-breath table in the §7 format."""
        if not traces:
            print("[v200-1b] No traces to print.")
            return
        K = len(traces)
        if half_k is None:
            half_k = K // 2

        print(f"\n{'='*60}")
        print(f"v200 Stage 1B Freeze-Breath Table  (ε=5%  half_K={half_k})")
        print(f"{'='*60}")

        jsd_list = self.compute_latent_jsd_per_breath(traces)
        if jsd_list:
            eps_jsd = 0.05 * jsd_list[0] if jsd_list[0] > 0 else 1e-8
            freeze = next((i+1 for i, v in enumerate(jsd_list) if v <= eps_jsd), K)
            status = "MOVING" if freeze >= half_k else "FROZEN"
            print(f"\nLatent JSD:  ε={eps_jsd:.2e}  freeze={freeze}  [{status}]")
            print(f"  raw: [{' '.join(f'{v:.4f}' for v in jsd_list)}]")

        energy = self.compute_energy_channel(traces)
        if energy.shape[0] > 0:
            me = energy.mean(axis=1)
            eps_e = 0.05 * me[0] if me[0] > 0 else 1e-8
            freeze_e = next((k for k, v in enumerate(me) if v <= eps_e), K)
            status_e = "MOVING" if freeze_e >= half_k else "FROZEN"
            print(f"\n‖Δz‖ mean over L:  ε={eps_e:.2e}  freeze={freeze_e}  [{status_e}]")
            print(f"  raw: [{' '.join(f'{v:.4f}' for v in me)}]")

        xattn_ent = self.compute_xattn_entropy_per_breath(traces)
        if xattn_ent.shape[0] > 0 and xattn_ent.any():
            me_x = xattn_ent.mean(axis=1)
            print(f"\nCross-Attn Entropy (nats, mean over heads):")
            print(f"  raw: [{' '.join(f'{v:.4f}' for v in me_x)}]")
            print(f"  reference random-init: log(24) = {np.log(24):.4f} nats")

        sa_ent = self.compute_self_attn_entropy_per_breath(traces)
        if sa_ent.shape[0] > 0 and sa_ent.any():
            # (K, 4, n_heads) → mean over layers and heads
            me_sa = sa_ent.mean(axis=(1, 2))
            print(f"\nSelf-Attn Entropy (nats, mean over 4 layers × heads):")
            print(f"  raw: [{' '.join(f'{v:.4f}' for v in me_sa)}]")
            print(f"  reference random-init: log(32) = {np.log(32):.4f} nats")

            if xattn_ent.shape[0] > 0 and xattn_ent.any():
                mean_cross = float(xattn_ent.mean())
                mean_self  = float(sa_ent.mean())
                diff = abs(mean_cross - mean_self)
                print(f"\nLike-units distinguishability (Gate B §8):")
                print(f"  Cross-attn entropy mean (nats): {mean_cross:.4f}")
                print(f"  Self-attn entropy mean  (nats): {mean_self:.4f}")
                print(f"  |cross - self| = {diff:.4f}")

        # §1A.E.4 position-collapse disambiguation metrics (added Jun 11)
        ipc_mr = self.compute_inter_position_cosine_mean_removed_per_breath(traces)
        if ipc_mr.shape[0] > 0:
            print(f"\nInter-position cosine (mean-removed) per breath [§1A.E.4]:")
            print(f"  raw: [{' '.join(f'{v:.6f}' for v in ipc_mr)}]")
            print(f"  (1.0=collapse, 0.0=orthogonal; measures diversity after removing shared additive)")

        rdr = self.compute_read_dominance_ratio_per_breath(traces)
        if rdr.shape[0] > 0:
            print(f"\n‖read_ctx‖/‖z_pre‖ (READ dominance ratio) per breath [§1A.E.4]:")
            print(f"  raw: [{' '.join(f'{v:.4f}' for v in rdr)}]")
            print(f"  (<1=healthy, 1-5=dominant-but-bounded, >10=needs gated residual)")

        # Secondary diagnostic: self-attn JSD (between-breath, NOT the Gate B signal)
        sa_jsd = self.compute_self_attn_layer_jsd(traces)
        if sa_jsd.shape[0] > 0:
            print(f"\nSelf-Attn Layer JSD (secondary; between-breath, K-1 pairs × 4 layers):")
            print(f"  {'Pair':>6}  " + "  ".join(f"L{l}     " for l in range(4)))
            for ki in range(sa_jsd.shape[0]):
                row = "  ".join(f"{v:.4f}" for v in sa_jsd[ki])
                print(f"  {ki}→{ki+1}  {row}")

        print(f"{'='*60}\n")

    # ------------------------------------------------------------------
    # Persistence bundle (§5) — dry-run at Stage 1B
    # ------------------------------------------------------------------

    def save_persistence_bundle(
        self,
        traces: _List[_Dict[str, _Any]],
        step: int,
        output_dir: str,
        dry_run: bool = True,
        b_sample: int = 2,
    ) -> _Dict[str, _Any]:
        """Return (or write) persistence bundle metadata per brief §5.

        dry_run=True (Stage 1B): returns expected shapes/paths without writing.
        dry_run=False (Stage 1C+): writes z_k, xattn_weights, scalars, etc.
        """
        from mycelium.provenance import make_provenance

        if not os.path.isabs(output_dir):
            raise ValueError(
                f"save_persistence_bundle: output_dir must be absolute, got {output_dir!r}"
            )

        bundle_meta: _Dict[str, _Any] = {}

        for trace in traces:
            k = trace["k"]
            prefix = f"step{step:06d}_breath{k:02d}"

            z_np = trace["z"].float().numpy()[:b_sample]
            path_z = os.path.join(output_dir, f"{prefix}_z.npz")
            prov = make_provenance(
                metric=f"latent_z_breath{k}",
                units="bf16",
                shape=list(z_np.shape),
                ckpt="cold-start" if step == 0 else f"step{step}",
                split="smoke",
                seed=42,
                step=step,
                env_vars={
                    "K_MAX": str(self.config.k_max),
                    "N_LATENTS": str(self.config.n_latents),
                    "HIDDEN_DIM": str(self.config.hidden_dim),
                },
                output_path=path_z,
                key="data",
            )
            bundle_meta[f"breath{k}_z"] = {
                "shape": list(z_np.shape), "dtype": "bf16", "path": path_z
            }
            bundle_meta[f"breath{k}_z_provenance"] = prov

            if trace.get("xattn_weights") is not None:
                w_np = trace["xattn_weights"].float().numpy()[:b_sample]
                bundle_meta[f"breath{k}_xattn_weights"] = {
                    "shape": list(w_np.shape), "dtype": "float32",
                    "path": os.path.join(output_dir, f"{prefix}_xattn_weights.npz"),
                }

            bundle_meta[f"breath{k}_scalars"] = {
                "delta_gate":    trace.get("delta_gate_val"),
                "read_ctx_norm": trace.get("read_ctx_norm"),
                "sa_resid_norm": trace.get("self_attn_resid_norm"),
            }

            if trace.get("calib") is not None:
                calib_np = trace["calib"].float().numpy()[:b_sample]
                bundle_meta[f"breath{k}_calib"] = {
                    "shape": list(calib_np.shape),
                    "path":  os.path.join(output_dir, f"{prefix}_calib.npz"),
                }

            if trace.get("tree_logits") is not None:
                tl_np = trace["tree_logits"].float().numpy()[:b_sample]
                bundle_meta[f"breath{k}_tree_logits"] = {
                    "shape": list(tl_np.shape),
                    "path":  os.path.join(output_dir, f"{prefix}_tree_logits.npz"),
                }

        if not dry_run:
            from mycelium.provenance import write_with_provenance
            os.makedirs(output_dir, exist_ok=True)
            for trace in traces:
                k   = trace["k"]
                z_np = trace["z"].float().numpy()[:b_sample].astype(np.float16)
                prov = bundle_meta[f"breath{k}_z_provenance"]
                path_z = bundle_meta[f"breath{k}_z"]["path"]
                write_with_provenance({"data": z_np}, path_z, prov)

        return bundle_meta
