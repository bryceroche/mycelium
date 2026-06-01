"""v105.1.2 factor graph breathing transformer — full-stack architecture.

Combines four validated components on top of v105.1 (digit RoPE):

  1. Digit-by-digit codebook  (10 entries × 1024d, shared across positions)
     — from v105
  2. RoPE on digit positions  — from v105.1 (the correlation fix)
  3. Projection waist 1024 → 512 → 1024 with LoRA-style init  — from v101
     LoRA guarantee: W_expand zero-initialized → waist has zero contribution
     at step 0 → step-0 forward equals v105.1 backbone behavior exactly.
  4. IB semantic codebook  (32 entries × 1024d)  — from v104
     Loaded from .cache/ib_centroids_gsm8k_partial.npz (falls back to QR random).
     delta_gate_quant zero-initialized → codebook has zero contribution at step 0.
  5. Per-NUMBER factor auxiliary loss (joint log-likelihood, NOT per-digit average)
     factor_aux = mean_over_factors(-sum_p log P(gold_digit_p | hidden))
     This preserves per-digit supervision granularity while training on joint number
     coherence — combined with RoPE enforces digit correlation through BOTH
     architecture and loss.

Architecture change from v105.1:
  After the 4 transformer layers each breath, two residual paths are applied before
  the delta-gate update:
    a) IB codebook projection (same pattern as v104):
         scores   = h @ codebook.T / temperature    (B, T, 32)
         weights  = scores.clip(-1e4, 1e4).softmax(-1)
         recon    = weights @ codebook               (B, T, H)
         h = h + delta_gate_quant[k] * (recon - h)  # zero at init
    b) Waist compression (same pattern as v101):
         waist_h  = GELU(h @ W_compress)             (B, T, 512)
         quantize = waist_h @ W_expand               (B, T, H)   # zero at init
         h = h + quantize                            # zero at init

LoRA init verification:
  At step 0:
    delta_gate_quant = 0 → IB codebook term = 0 → h unchanged
    W_expand = 0 → waist term = 0 → h unchanged
    delta_gate = ones (inherited) → full delta from v105.1 is passed through
  Therefore: step-0 forward == v105.1 backbone

Loss structure:
  total = var_loss + fw * factor_aux_loss + aw * calib_loss + ew * energy_loss

  var_loss:         per-breath weighted CE on unobserved variable digit positions
  factor_aux_loss:  per-NUMBER joint log-likelihood on factor result nodes
                    (option c from spec: -sum_p log P(gold_digit_p | factor_hidden))
  calib_loss:       MSE calibration head vs progressive accuracy target
  energy_loss:      expected-value constraint energy (differentiable, from v105)

Env var gates:
  V105_1_2_TASK=1               — enable v105.1.2 forward path
  V105_1_2_K_MAX=8              — iterative-prefill breaths
  V105_1_2_N_DIGITS=5           — digit positions per variable (covers 0..99999)
  V105_1_2_N_MAX=16             — max variable nodes
  V105_1_2_F_MAX=8              — max factor nodes
  V105_1_2_WAIST=512            — projection waist dimension
  V105_1_2_CODEBOOK_N=32        — IB semantic codebook entries
  V105_1_2_IB_CENTROIDS=.cache/ib_centroids_gsm8k_partial.npz
  V105_1_2_ENERGY_WEIGHT=0.01   — constraint energy weight
  V105_1_2_CALIB_WEIGHT=0.05    — calibration loss weight
  V105_1_2_FACTOR_AUX_WEIGHT=0.5 — per-NUMBER aux loss weight
  V105_1_2_ROPE_BASE=10000      — digit RoPE base theta (same as v105.1)
  V105_1_2_IB_INIT=1            — use IB centroids (vs random QR fallback)
  V105_1_2_WAIST_LORA_INIT=1   — zero-init W_expand (recommended)
"""
from __future__ import annotations

import math
import os
import time as _time
from typing import Any

import numpy as np
from tinygrad import Tensor, dtypes
from tinygrad.engine.jit import TinyJit

from mycelium.config import Config

# Re-use helpers from v105 / v105.1.
from mycelium.factor_graph_v105 import (
    OP_ADD, OP_SUB, OP_MUL, OP_DIV,
    value_to_digits, digits_to_value,
    _expected_value_v105, constraint_energy_v105,
    fg_accuracy_v105,
)
from mycelium.factor_graph_v105_1 import (
    _precompute_digit_rope,
    apply_rope_digit_tg,
)
# IB centroid loader reused from v104.
from mycelium.factor_graph_v104 import load_ib_centroids

__all__ = [
    "OP_ADD", "OP_SUB", "OP_MUL", "OP_DIV",
    "value_to_digits", "digits_to_value",
]

# ---------------------------------------------------------------------------
# Env vars
# ---------------------------------------------------------------------------

V105_1_2_TASK             = int(os.environ.get("V105_1_2_TASK",             "0")) > 0
V105_1_2_K_MAX            = int(os.environ.get("V105_1_2_K_MAX",            "8"))
V105_1_2_N_DIGITS         = int(os.environ.get("V105_1_2_N_DIGITS",         "5"))
V105_1_2_N_MAX            = int(os.environ.get("V105_1_2_N_MAX",            "16"))
V105_1_2_F_MAX            = int(os.environ.get("V105_1_2_F_MAX",             "8"))
V105_1_2_N_HEADS          = 16   # fixed: Pythia-410M
V105_1_2_WAIST            = int(os.environ.get("V105_1_2_WAIST",            "512"))
V105_1_2_CODEBOOK_N       = int(os.environ.get("V105_1_2_CODEBOOK_N",       "32"))
V105_1_2_IB_CENTROIDS     = os.environ.get(
    "V105_1_2_IB_CENTROIDS", ".cache/ib_centroids_gsm8k_partial.npz"
)
V105_1_2_ENERGY_WEIGHT    = float(os.environ.get("V105_1_2_ENERGY_WEIGHT",   "0.01"))
V105_1_2_CALIB_WEIGHT     = float(os.environ.get("V105_1_2_CALIB_WEIGHT",    "0.05"))
V105_1_2_FACTOR_AUX_WEIGHT = float(os.environ.get("V105_1_2_FACTOR_AUX_WEIGHT", "0.5"))
V105_1_2_ROPE_BASE        = float(os.environ.get("V105_1_2_ROPE_BASE",       "10000.0"))
V105_1_2_IB_INIT          = int(os.environ.get("V105_1_2_IB_INIT",           "1")) > 0
V105_1_2_WAIST_LORA_INIT  = int(os.environ.get("V105_1_2_WAIST_LORA_INIT",   "1")) > 0
V105_1_2_FOURIER_INIT     = int(os.environ.get("V105_1_2_FOURIER_INIT",      "0")) > 0
# Drop per-digit CE entirely, use per-NUMBER MSE on reconstructed value as the
# sole variable supervision. Tests whether "partial credit" of per-digit CE is
# the trap holding upper positions in constant-predictor collapse.
V105_1_2_NUMBER_MSE_ONLY  = int(os.environ.get("V105_1_2_NUMBER_MSE_ONLY",   "0")) > 0


def _fourier_digit_codebook(n_digits: int, hidden: int) -> np.ndarray:
    """Cyclic-Fourier init for digit_codebook.

    Maps each digit d to phases on a circle: phase_d = 2π·d/n_digits.
    The codebook fills hidden dims with [cos(d·freq·phase), sin(d·freq·phase)]
    pairs cycling through the Nyquist-limited frequencies 1..n_digits/2.

    Motivation: digits form a cyclic group Z/n_digits. Random init forces
    the model to discover this structure via gradient (the "grokking"
    pathway, ~10K-100K steps for tiny modular-arithmetic transformers).
    Structured init provides the cyclic geometry directly — same intuition
    as v98's aligned init for state_embed.

    Each row is L2-normalized to unit norm (matches the QR-orthogonal init
    scale this replaces).
    """
    cb = np.zeros((n_digits, hidden), dtype=np.float32)
    n_unique_freqs = max(n_digits // 2, 1)  # 5 for n_digits=10
    n_pairs = hidden // 2
    for k in range(n_pairs):
        freq = (k % n_unique_freqs) + 1
        for d in range(n_digits):
            phase = 2.0 * np.pi * d * freq / n_digits
            cb[d, 2 * k]     = np.cos(phase)
            cb[d, 2 * k + 1] = np.sin(phase)
    norms = np.linalg.norm(cb, axis=1, keepdims=True)
    cb = cb / (norms + 1e-8)
    return cb.astype(np.float32)


# ---------------------------------------------------------------------------
# Component 1+2: Embedding with digit-axis RoPE  (same as v105.1)
# ---------------------------------------------------------------------------

def embed_factor_graph_v105_1_2(
    digit_init: Tensor,       # (B, N_MAX, N_DIGITS, 10) — one-hot / uniform
    node_kinds: Tensor,        # (B, T_MAX) int
    var_pos_embed: Tensor,     # (N_MAX, H)
    factor_pos_embed: Tensor,  # (F_MAX, H)
    node_kind_embed: Tensor,   # (3, H)
    digit_codebook: Tensor,    # (10, H)
    digit_rope_cos: Tensor,    # (N_DIGITS, H)
    digit_rope_sin: Tensor,    # (N_DIGITS, H)
    n_max: int,
    n_digits: int,
    f_max: int,
) -> Tensor:
    """Build (B, T_MAX, H) hidden states with digit-axis RoPE.

    Component 1: digit_codebook (10, H) — shared across all positions.
    Component 2: apply_rope_digit_tg — each position p gets rotated by p*freq.

    raw[b, v, p] = digit_codebook[digit_value] + var_pos_embed[v]
    embed[b, v, p] = RoPE(raw[b, v, p], p)

    No additive digit_pos_embed — digit position is encoded geometrically via RoPE.
    """
    B = int(digit_init.shape[0])
    H = int(var_pos_embed.shape[1])

    # Variable digit tokens: (B, N_MAX, N_DIGITS, H)
    di_cb = digit_init.cast(digit_codebook.dtype)           # (B, N_MAX, N_DIGITS, 10)
    var_digit_state = di_cb @ digit_codebook                # (B, N_MAX, N_DIGITS, H)
    vpe = var_pos_embed.reshape(1, n_max, 1, H).cast(var_digit_state.dtype)
    raw = var_digit_state + vpe.expand(B, n_max, n_digits, H)  # (B, N_MAX, N_DIGITS, H)

    # Apply digit RoPE (Component 2)
    var_digit_h = apply_rope_digit_tg(
        raw, digit_rope_cos, digit_rope_sin, n_digits=n_digits, hidden=H
    )  # (B, N_MAX, N_DIGITS, H)

    var_tokens = var_digit_h.reshape(B, n_max * n_digits, H)

    # Factor positions (unchanged)
    factor_pos = factor_pos_embed.reshape(1, f_max, H).cast(var_tokens.dtype).expand(B, f_max, H)
    x = var_tokens.cat(factor_pos, dim=1)  # (B, T, H)

    # Node-kind embedding
    nk_clamped = node_kinds.clip(0, 2)
    nk_oh      = nk_clamped.one_hot(3).cast(x.dtype)
    nk_emb     = nk_oh @ node_kind_embed.cast(x.dtype)
    x = x + nk_emb

    return x


# ---------------------------------------------------------------------------
# One transformer layer (identical to v105.1)
# ---------------------------------------------------------------------------

def fg_layer_forward_v105_1_2(
    layer: Any,
    x: Tensor,          # (B, T_MAX, H)
    attn_bias: Tensor,  # (B, N_HEADS, T_MAX, T_MAX)
) -> Tensor:
    """Run one BreathingLayer with per-head factor-graph attention mask."""
    cfg = layer.cfg
    B, S, H = x.shape
    n_heads  = cfg.n_heads
    head_dim = cfg.head_dim

    from mycelium.breathing import _layernorm
    attn_in = _layernorm(x, layer.shared.in_ln_g, layer.shared.in_ln_b, cfg.layer_norm_eps)
    mlp_in  = _layernorm(x, layer.shared.post_ln_g, layer.shared.post_ln_b, cfg.layer_norm_eps)
    attn_in_dt = attn_in.cast(x.dtype) if attn_in.dtype != x.dtype else attn_in
    mlp_in_dt  = mlp_in.cast(x.dtype) if mlp_in.dtype != x.dtype else mlp_in

    q = (attn_in_dt @ layer.wq + layer.bq).reshape(B, S, n_heads, head_dim).transpose(1, 2)
    k = (attn_in_dt @ layer.wk + layer.bk).reshape(B, S, n_heads, head_dim).transpose(1, 2)
    v = (attn_in_dt @ layer.shared.wv + layer.shared.bv).reshape(B, S, n_heads, head_dim).transpose(1, 2)

    scale  = 1.0 / math.sqrt(head_dim)
    scores = q @ k.transpose(-2, -1) * scale
    scores = scores + attn_bias.cast(scores.dtype)
    attn   = scores.clip(-1e4, 1e4).softmax(-1)
    ctx    = (attn @ v).transpose(1, 2).reshape(B, S, H)
    attn_out = ctx @ layer.shared.wo + layer.shared.bo

    ff      = (mlp_in_dt @ layer.w_in + layer.b_in).gelu()
    ffn_out = ff @ layer.shared.w_out + layer.shared.b_out

    return x + attn_out + ffn_out


# ---------------------------------------------------------------------------
# Component 3: Projection waist 1024 → 512 → 1024 (LoRA-style)
# ---------------------------------------------------------------------------

def apply_projection_waist(
    h: Tensor,         # (B, T, H)
    W_compress: Tensor,  # (H, waist)
    W_expand: Tensor,    # (waist, H)   — zero-initialized
    b_compress: Tensor,  # (waist,)
    b_expand: Tensor,    # (H,)
) -> Tensor:
    """Projection waist with LoRA-style zero init.

    At init: W_expand = 0 → quantize = 0 → output = h (byte-identical to no-waist).
    After unlock: h → 512d compressed → GELU → 1024d correction (added as residual).

    This is the 'Quantize' step from v101 (forces commitment to 512d compression).
    """
    wc = W_compress.cast(h.dtype)
    bc = b_compress.reshape(1, 1, -1).cast(h.dtype)
    we = W_expand.cast(h.dtype)
    be = b_expand.reshape(1, 1, -1).cast(h.dtype)

    waist_h  = (h @ wc + bc).gelu()   # (B, T, waist)
    quantize = waist_h @ we + be       # (B, T, H) — zero at init (W_expand=0)
    return h + quantize                # residual; = h at init


# ---------------------------------------------------------------------------
# Component 4: IB semantic codebook soft projection (LoRA-style gate)
# ---------------------------------------------------------------------------

def apply_ib_codebook(
    h: Tensor,              # (B, T, H)
    codebook: Tensor,        # (N_CODE, H)
    temperature: Tensor,     # () scalar
    delta_gate_quant_k: Tensor,  # () — zero at init
) -> Tensor:
    """IB soft codebook projection with zero-init gate.

    At init: delta_gate_quant = 0 → h_quant = h (byte-identical to no-codebook).
    After unlock: soft nearest-codebook reconstruction is blended in as residual.

    Architecture is identical to v104's per-breath codebook step.
    """
    cb  = codebook.cast(h.dtype)
    tmp = temperature.cast(h.dtype)

    scores   = h @ cb.T / tmp.reshape(1, 1, 1)     # (B, T, N_CODE)
    weights  = scores.clip(-1e4, 1e4).softmax(-1)  # (B, T, N_CODE)
    recon    = weights @ cb                          # (B, T, H)
    quantize = recon - h                            # delta: toward codebook
    gate_k   = delta_gate_quant_k.cast(h.dtype).reshape(1, 1, 1)
    return h + gate_k * quantize                    # = h at init


# ---------------------------------------------------------------------------
# Iterative prefill loop — full stack
# ---------------------------------------------------------------------------

def fg_breathing_forward_v105_1_2(
    model: Any,
    digit_init: Tensor,       # (B, N_MAX, N_DIGITS, 10)
    node_kinds: Tensor,        # (B, T_MAX) int
    staging_mask: Tensor,      # (B, K_MAX, T_MAX, T_MAX)
    head_op_mask: Tensor,      # (B, N_HEADS, T_MAX, T_MAX)
    K: int,
    n_max: int = V105_1_2_N_MAX,
    f_max: int = V105_1_2_F_MAX,
    n_digits: int = V105_1_2_N_DIGITS,
) -> tuple[list[Tensor], list[Tensor], list[Tensor]]:
    """K iterative-prefill breaths: digit RoPE + IB codebook + projection waist.

    Per breath k:
      1. Breath embedding added.
      2. Combined staging + op mask built.
      3. 4 transformer layers run.
      4. IB codebook soft projection  (Component 4 — zero effect at step 0).
      5. Projection waist compression  (Component 3 — zero effect at step 0).
      6. Delta gate residual update.
      7. Readout: digit logits (via codebook dot-product), factor logits, calib.
    """
    assert hasattr(model, "fg_v105_1_2_digit_codebook"), \
        "model has no v105.1.2 params; was attach_fg_params_v105_1_2 called?"

    # Component 1+2 params
    digit_codebook   = model.fg_v105_1_2_digit_codebook    # (10, H)
    digit_rope_cos   = model.fg_v105_1_2_digit_rope_cos    # (N_DIGITS, H)
    digit_rope_sin   = model.fg_v105_1_2_digit_rope_sin    # (N_DIGITS, H)
    var_pos_embed    = model.fg_v105_1_2_var_pos_embed      # (N_MAX, H)
    factor_pos_embed = model.fg_v105_1_2_factor_pos_embed  # (F_MAX, H)
    node_kind_embed  = model.fg_v105_1_2_node_kind_embed   # (3, H)
    breath_embed     = model.fg_v105_1_2_breath_embed      # (K_max, H)
    delta_gate       = model.fg_v105_1_2_delta_gate        # (K_max,)
    calib_head_w     = model.fg_v105_1_2_calib_head_w      # (H, 1)
    calib_head_b     = model.fg_v105_1_2_calib_head_b      # (1,)

    # Component 3: projection waist params
    W_compress = model.fg_v105_1_2_W_compress  # (H, waist)
    b_compress = model.fg_v105_1_2_b_compress  # (waist,)
    W_expand   = model.fg_v105_1_2_W_expand    # (waist, H) — zero at init
    b_expand   = model.fg_v105_1_2_b_expand    # (H,)

    # Component 4: IB semantic codebook params
    ib_codebook      = model.fg_v105_1_2_ib_codebook       # (N_CODE, H)
    delta_gate_quant = model.fg_v105_1_2_delta_gate_quant  # (K_max,) — zero at init
    ib_temperature   = model.fg_v105_1_2_ib_temperature    # ()

    B = int(digit_init.shape[0])
    T = n_max * n_digits + f_max
    H = int(var_pos_embed.shape[1])

    # Initial embedding: digit codebook (1) + RoPE on digit positions (2)
    x = embed_factor_graph_v105_1_2(
        digit_init, node_kinds,
        var_pos_embed, factor_pos_embed, node_kind_embed,
        digit_codebook, digit_rope_cos, digit_rope_sin,
        n_max=n_max, n_digits=n_digits, f_max=f_max,
    )
    x = x.cast(dtypes.half) if x.dtype != dtypes.half else x

    layers = list(model.block.layers)
    K_max  = int(breath_embed.shape[0])
    assert K <= K_max, f"K={K} exceeds K_max={K_max}"

    from mycelium.breathing import _layernorm

    digit_logits_history  = []
    factor_logits_history = []
    calib_history         = []

    for k in range(K):
        # 1. Breath embedding
        be_k  = breath_embed[k].reshape(1, 1, -1).cast(x.dtype)
        x_in  = x + be_k
        x_pre = x

        # 2. Combined mask (B, N_HEADS, T, T)
        stk   = staging_mask[:, k, :, :]     # (B, T, T)
        stk_h = stk.reshape(B, 1, T, T).expand(B, V105_1_2_N_HEADS, T, T)
        combined = stk_h.cast(x.dtype) + head_op_mask.cast(x.dtype)

        # 3. Four transformer layers
        h = x_in
        for layer in layers[:4]:
            h = fg_layer_forward_v105_1_2(layer, h, combined)

        # 4. IB semantic codebook soft projection  (Component 4)
        #    Zero effect at step 0 (delta_gate_quant initialized to zeros).
        h = apply_ib_codebook(h, ib_codebook, ib_temperature, delta_gate_quant[k])

        # 5. Projection waist compression  (Component 3)
        #    Zero effect at step 0 (W_expand initialized to zeros).
        h = apply_projection_waist(h, W_compress, W_expand, b_compress, b_expand)

        # 6. Delta gate residual update
        gate_k = delta_gate[k].cast(h.dtype).reshape(1, 1, 1)
        delta  = h - x_pre
        x      = x_pre + gate_k * delta

        # 7a. Variable digit logits
        x_ln = _layernorm(x, model.ln_f_g, model.ln_f_b, model.cfg.layer_norm_eps).cast(dtypes.float)
        n_var_tokens = n_max * n_digits
        var_tokens   = x_ln[:, :n_var_tokens, :]
        var_tokens_r = var_tokens.reshape(B, n_max, n_digits, -1)
        cb_fp        = digit_codebook.cast(dtypes.float)
        digit_logits_k = var_tokens_r @ cb_fp.T              # (B, N_MAX, N_DIGITS, 10)
        digit_logits_history.append(digit_logits_k)

        # 7b. Factor digit logits (for per-NUMBER aux loss)
        fac_tokens   = x_ln[:, n_var_tokens:n_var_tokens + f_max, :]
        fac_tokens_r = fac_tokens.reshape(B, f_max, 1, -1).expand(
            B, f_max, n_digits, int(x_ln.shape[-1])
        )
        fac_logits_k = fac_tokens_r @ cb_fp.T               # (B, F_MAX, N_DIGITS, 10)
        factor_logits_history.append(fac_logits_k)

        # 7c. Calibration head
        pool        = x_ln.mean(axis=1)
        calib_logit = pool @ calib_head_w.cast(dtypes.float) + calib_head_b.cast(dtypes.float)
        calib_k     = calib_logit.reshape(-1).sigmoid()
        calib_history.append(calib_k)

    return digit_logits_history, factor_logits_history, calib_history


# ---------------------------------------------------------------------------
# Parameter attachment
# ---------------------------------------------------------------------------

def attach_fg_params_v105_1_2(
    model: Any,
    hidden: int,
    n_digits: int = V105_1_2_N_DIGITS,
    n_max: int = V105_1_2_N_MAX,
    f_max: int = V105_1_2_F_MAX,
    k_max: int | None = None,
    waist: int | None = None,
    n_code: int | None = None,
    rope_base: float | None = None,
    ib_centroids_path: str | None = None,
    ib_init: bool | None = None,
    waist_lora_init: bool | None = None,
) -> None:
    """Attach v105.1.2 params to model.

    Components:
      1+2  digit_codebook (10, hidden) + frozen digit_rope_cos/sin tables
      3    projection waist: W_compress (hidden, waist_d), W_expand (waist_d, hidden)
           W_expand zero-initialized (LoRA-style) → zero waist effect at step 0
      4    IB semantic codebook (n_code, hidden) + delta_gate_quant (K_max,)
           delta_gate_quant zero-initialized → zero codebook effect at step 0

    LoRA guarantee: step-0 forward is byte-identical to v105.1 backbone
    (both the waist and the IB codebook contribute zero to h at step 0).
    """
    if k_max is None:
        k_max = V105_1_2_K_MAX
    if waist is None:
        waist = V105_1_2_WAIST
    if n_code is None:
        n_code = V105_1_2_CODEBOOK_N
    if rope_base is None:
        rope_base = V105_1_2_ROPE_BASE
    if ib_centroids_path is None:
        ib_centroids_path = V105_1_2_IB_CENTROIDS
    if ib_init is None:
        ib_init = V105_1_2_IB_INIT
    if waist_lora_init is None:
        waist_lora_init = V105_1_2_WAIST_LORA_INIT

    rng = np.random.RandomState(20013)

    # -----------------------------------------------------------------------
    # Components 1+2: digit codebook + frozen RoPE tables
    # -----------------------------------------------------------------------
    if V105_1_2_FOURIER_INIT:
        # Cyclic-Fourier init: digits as phases on the unit circle, giving the
        # model the cyclic group structure (Z/10Z) for free instead of forcing
        # discovery via gradient. Same scale as QR init (rows L2-normalized).
        dc = _fourier_digit_codebook(n_digits=10, hidden=hidden)
    else:
        raw_dc = rng.randn(max(hidden, 10), hidden).astype(np.float32)
        q_dc, _ = np.linalg.qr(raw_dc)
        dc = q_dc[:10].astype(np.float32) * 1.0
    model.fg_v105_1_2_digit_codebook = Tensor(dc, dtype=dtypes.float).contiguous()

    # Right-aligned RoPE for MSD-first layout: array index i gets RoPE position
    # (n_digits - 1 - i).  So the ones digit (array index N-1) ALWAYS has RoPE
    # position 0 (no rotation), regardless of n_digits.  The most-significant
    # digit (array index 0) gets the maximum rotation (position N-1).
    #
    # Implementation: reverse the precomputed tables along the n_digits axis.
    # _precompute_digit_rope returns rows ordered by position [0, 1, ..., N-1];
    # after reverse, row i holds position (N-1-i)'s rotation, which is exactly
    # what apply_rope_digit_tg picks up when indexing by array index i.
    cos_t, sin_t = _precompute_digit_rope(n_digits, hidden, base=rope_base)
    cos_t = cos_t[::-1, :].copy()   # reverse along position axis
    sin_t = sin_t[::-1, :].copy()
    model.fg_v105_1_2_digit_rope_cos = Tensor(cos_t, dtype=dtypes.float).contiguous()
    model.fg_v105_1_2_digit_rope_sin = Tensor(sin_t, dtype=dtypes.float).contiguous()

    # -----------------------------------------------------------------------
    # Position / kind embeddings
    # -----------------------------------------------------------------------
    vp = rng.randn(n_max, hidden).astype(np.float32) * 0.02
    model.fg_v105_1_2_var_pos_embed = Tensor(vp, dtype=dtypes.float).contiguous()

    fp_emb = rng.randn(f_max, hidden).astype(np.float32) * 0.02
    model.fg_v105_1_2_factor_pos_embed = Tensor(fp_emb, dtype=dtypes.float).contiguous()

    nk = rng.randn(3, hidden).astype(np.float32) * 0.02
    model.fg_v105_1_2_node_kind_embed = Tensor(nk, dtype=dtypes.float).contiguous()

    # -----------------------------------------------------------------------
    # Calibration head
    # -----------------------------------------------------------------------
    cw = (rng.randn(hidden, 1) * 0.02).astype(np.float32)
    model.fg_v105_1_2_calib_head_w = Tensor(cw, dtype=dtypes.float).contiguous()
    model.fg_v105_1_2_calib_head_b = Tensor.zeros((1,), dtype=dtypes.float).contiguous()

    # -----------------------------------------------------------------------
    # Per-breath embeddings + delta gate
    # -----------------------------------------------------------------------
    rng_be = np.random.RandomState(20014)
    raw_be = rng_be.randn(max(k_max, hidden), hidden).astype(np.float32)
    q_be, _ = np.linalg.qr(raw_be)
    be = q_be[:k_max].astype(np.float32) * 0.5
    model.fg_v105_1_2_breath_embed = Tensor(be, dtype=dtypes.float).contiguous()
    model.fg_v105_1_2_delta_gate   = Tensor.ones((k_max,), dtype=dtypes.float).contiguous()

    # -----------------------------------------------------------------------
    # Component 3: Projection waist  (LoRA-style)
    # -----------------------------------------------------------------------
    W_c = (rng.randn(hidden, waist) * 0.02).astype(np.float32)
    model.fg_v105_1_2_W_compress = Tensor(W_c, dtype=dtypes.float).contiguous()
    model.fg_v105_1_2_b_compress = Tensor.zeros((waist,), dtype=dtypes.float).contiguous()

    if waist_lora_init:
        # Zero-initialized → waist has zero contribution at step 0
        model.fg_v105_1_2_W_expand = Tensor.zeros((waist, hidden), dtype=dtypes.float).contiguous()
    else:
        We = (rng.randn(waist, hidden) * 0.02).astype(np.float32)
        model.fg_v105_1_2_W_expand = Tensor(We, dtype=dtypes.float).contiguous()
    model.fg_v105_1_2_b_expand = Tensor.zeros((hidden,), dtype=dtypes.float).contiguous()

    # -----------------------------------------------------------------------
    # Component 4: IB semantic codebook
    # -----------------------------------------------------------------------
    if ib_init:
        cb_np = load_ib_centroids(ib_centroids_path, n_code=n_code, hidden=hidden)
    else:
        raw_cb = rng.randn(max(hidden, n_code), hidden).astype(np.float32)
        q_cb, _ = np.linalg.qr(raw_cb)
        cb_np = q_cb[:n_code].astype(np.float32) * 0.5
        print(f"[v105.1.2] IB init disabled; random QR codebook ({n_code}, {hidden})")

    model.fg_v105_1_2_ib_codebook = Tensor(cb_np, dtype=dtypes.float).contiguous()
    # Zero-initialized gate → IB codebook has zero contribution at step 0
    model.fg_v105_1_2_delta_gate_quant = Tensor.zeros((k_max,), dtype=dtypes.float).contiguous()
    model.fg_v105_1_2_ib_temperature   = Tensor(
        np.array([1.0], dtype=np.float32), dtype=dtypes.float
    ).contiguous()

    T = n_max * n_digits + f_max
    print(
        f"[v105.1.2] params attached:\n"
        f"  digit_codebook=(10,{hidden}) init={'FOURIER' if V105_1_2_FOURIER_INIT else 'QR-random'}, "
        f"digit_rope (N_DIGITS={n_digits}, H={hidden}, base={rope_base:.0f}) [FROZEN]\n"
        f"  loss_mode={'NUMBER_MSE_ONLY' if V105_1_2_NUMBER_MSE_ONLY else 'per-digit CE'}\n"
        f"  var_pos_embed=({n_max},{hidden}), factor_pos_embed=({f_max},{hidden})\n"
        f"  waist=({hidden}→{waist}→{hidden}), W_expand={'ZEROS' if waist_lora_init else 'random'}\n"
        f"  ib_codebook=({n_code},{hidden}), "
        f"delta_gate_quant=ZEROS, ib_init={ib_init}\n"
        f"  T={T} (N_MAX*N_DIGITS+F_MAX={n_max}*{n_digits}+{f_max}), K_max={k_max}",
        flush=True,
    )


def fg_v105_1_2_parameters(model: Any) -> list[Tensor]:
    """Trainable v105.1.2 factor-graph-specific params.

    Deliberately excludes frozen tables:
      - digit_rope_cos, digit_rope_sin  (precomputed, like breathing.py RoPE)

    Includes all learnable params across the four components.
    """
    return [
        # Components 1+2
        model.fg_v105_1_2_digit_codebook,
        model.fg_v105_1_2_var_pos_embed,
        model.fg_v105_1_2_factor_pos_embed,
        model.fg_v105_1_2_node_kind_embed,
        model.fg_v105_1_2_calib_head_w,
        model.fg_v105_1_2_calib_head_b,
        model.fg_v105_1_2_breath_embed,
        model.fg_v105_1_2_delta_gate,
        # Component 3: projection waist
        model.fg_v105_1_2_W_compress,
        model.fg_v105_1_2_b_compress,
        model.fg_v105_1_2_W_expand,
        model.fg_v105_1_2_b_expand,
        # Component 4: IB codebook
        model.fg_v105_1_2_ib_codebook,
        model.fg_v105_1_2_delta_gate_quant,
        model.fg_v105_1_2_ib_temperature,
    ]


def fg_v105_1_2_state_dict(model: Any) -> dict[str, Tensor]:
    """Full state dict (frozen RoPE tables included for checkpoint self-containment)."""
    return {
        # Components 1+2
        "fg_v105_1_2.digit_codebook":   model.fg_v105_1_2_digit_codebook,
        "fg_v105_1_2.digit_rope_cos":   model.fg_v105_1_2_digit_rope_cos,   # frozen
        "fg_v105_1_2.digit_rope_sin":   model.fg_v105_1_2_digit_rope_sin,   # frozen
        "fg_v105_1_2.var_pos_embed":    model.fg_v105_1_2_var_pos_embed,
        "fg_v105_1_2.factor_pos_embed": model.fg_v105_1_2_factor_pos_embed,
        "fg_v105_1_2.node_kind_embed":  model.fg_v105_1_2_node_kind_embed,
        "fg_v105_1_2.calib_head_w":     model.fg_v105_1_2_calib_head_w,
        "fg_v105_1_2.calib_head_b":     model.fg_v105_1_2_calib_head_b,
        "fg_v105_1_2.breath_embed":     model.fg_v105_1_2_breath_embed,
        "fg_v105_1_2.delta_gate":       model.fg_v105_1_2_delta_gate,
        # Component 3
        "fg_v105_1_2.W_compress":       model.fg_v105_1_2_W_compress,
        "fg_v105_1_2.b_compress":       model.fg_v105_1_2_b_compress,
        "fg_v105_1_2.W_expand":         model.fg_v105_1_2_W_expand,
        "fg_v105_1_2.b_expand":         model.fg_v105_1_2_b_expand,
        # Component 4
        "fg_v105_1_2.ib_codebook":          model.fg_v105_1_2_ib_codebook,
        "fg_v105_1_2.delta_gate_quant":     model.fg_v105_1_2_delta_gate_quant,
        "fg_v105_1_2.ib_temperature":       model.fg_v105_1_2_ib_temperature,
    }


# ---------------------------------------------------------------------------
# Warm-start from v104 backbone checkpoint
# ---------------------------------------------------------------------------

def load_ckpt_v105_1_2(model: Any, path: str) -> None:
    """Load backbone (shared.*, phase*.*, ln_f.*) from any v104-family checkpoint.

    v105.1.2-specific params (digit_codebook, waist, ib_codebook, RoPE tables,
    breath_embed, etc.) are NOT loaded — they are fresh-initialized so the warm-start
    preserves LoRA/zero-gate guarantees.
    """
    from tinygrad.nn.state import safe_load

    sd = safe_load(path)

    backbone_keys = [
        ("ln_f.g", model.ln_f_g),
        ("ln_f.b", model.ln_f_b),
    ]
    sw = model.block.shared
    for a in ("wv", "bv", "wo", "bo", "w_out", "b_out",
              "in_ln_g", "in_ln_b", "post_ln_g", "post_ln_b"):
        backbone_keys.append((f"shared.{a}", getattr(sw, a)))
    for i, layer in enumerate(model.block.layers):
        for a in ("wq", "bq", "wk", "bk", "w_in", "b_in"):
            backbone_keys.append((f"phase{i}.{a}", getattr(layer, a)))

    # Also copy v104's IB codebook if present in the checkpoint
    if "fg_v104.codebook" in sd and hasattr(model, "fg_v105_1_2_ib_codebook"):
        src = sd["fg_v104.codebook"]
        dst = model.fg_v105_1_2_ib_codebook
        if src.shape == dst.shape:
            dst.assign(src.to(dst.device).cast(dst.dtype)).realize()
            print(f"  copied fg_v104.codebook → fg_v105_1_2.ib_codebook", flush=True)

    loaded  = []
    missing = []
    for name, dst in backbone_keys:
        if name not in sd:
            missing.append(name)
            continue
        src = sd[name].to(dst.device).realize()
        if src.shape != dst.shape:
            try:
                src = src.reshape(dst.shape)
            except Exception:
                missing.append(f"{name}(shape mismatch)")
                continue
        if src.dtype != dst.dtype:
            src = src.cast(dst.dtype)
        dst.assign(src).realize()
        loaded.append(name)

    print(
        f"  loaded {len(loaded)}/{len(backbone_keys)} backbone keys "
        f"from {os.path.basename(path)}",
        flush=True,
    )
    if missing:
        print(f"  missing {len(missing)} keys: {missing[:5]}", flush=True)


# ---------------------------------------------------------------------------
# JIT training step
# ---------------------------------------------------------------------------

_JIT_V105_1_2_CACHE: dict = {}


def _compile_jit_fg_step_v105_1_2(
    model: Any,
    opt: Any,
    K: int,
    B: int,
    factor_aux_weight: float = V105_1_2_FACTOR_AUX_WEIGHT,
    calib_weight: float = V105_1_2_CALIB_WEIGHT,
    energy_weight: float = V105_1_2_ENERGY_WEIGHT,
    n_max: int = V105_1_2_N_MAX,
    f_max: int = V105_1_2_F_MAX,
    n_digits: int = V105_1_2_N_DIGITS,
    grad_clip: float = 1.0,
):
    """Compile a TinyJit'd train step for v105.1.2.

    Loss structure:
      var_loss         — per-breath weighted CE on unobserved variable digits
      factor_aux_loss  — per-NUMBER joint log-likelihood (Component 5 from spec)
      calib_loss       — MSE calibration head
      energy_loss      — expected-value constraint energy (differentiable)

    Per-NUMBER aux (option c from spec):
      factor_aux = mean_over_factors(mean_over_valid(-sum_p log P(gold_digit_p)))
      This is the JOINT log-likelihood of the number's digit sequence.
      Unlike the per-digit average (used in the base v105 for variables), this
      explicitly models P(d_0, d_1, ..., d_{N-1}) = prod_p P(d_p) via summed
      log-probs — encouraging coherent multi-digit predictions.
    """
    n_code = int(model.fg_v105_1_2_ib_codebook.shape[0])
    key = ("v105_1_2", id(model), id(opt), int(K), int(B),
           float(factor_aux_weight), float(calib_weight), float(energy_weight),
           int(n_max), int(f_max), int(n_digits), float(grad_clip), int(n_code))
    if key in _JIT_V105_1_2_CACHE:
        return _JIT_V105_1_2_CACHE[key]

    aw     = float(calib_weight)
    fw     = float(factor_aux_weight)
    ew     = float(energy_weight)
    gc     = float(grad_clip)
    params = opt.params

    print(
        f"[JIT] compile v105.1.2 fg step: K={K} B={B} n_digits={n_digits} "
        f"T={n_max * n_digits + f_max} aw={aw} fw={fw} ew={ew} gc={gc} "
        f"n_code={n_code}...",
        flush=True,
    )

    @TinyJit
    def _step(
        digit_init: Tensor,
        node_kinds: Tensor,
        staging_mask: Tensor,
        head_op_mask: Tensor,
        gold_digits: Tensor,
        observed_mask: Tensor,
        factor_gold_dg: Tensor,
        factor_valid: Tensor,
        factor_types: Tensor,
        factor_args: Tensor,
        digit_valid_mask: Tensor,           # (B, N_MAX, N_DIGITS) float  (NEW)
        factor_digit_valid_mask: Tensor,    # (B, F_MAX, N_DIGITS) float  (NEW)
    ):
        opt.zero_grad()

        digit_logits_history, factor_logits_history, calib_history = \
            fg_breathing_forward_v105_1_2(
                model, digit_init, node_kinds, staging_mask, head_op_mask,
                K=K, n_max=n_max, f_max=f_max, n_digits=n_digits,
            )

        # --- Main loss on unobserved variables ---
        # MSD layout: valid positions are the TRAILING n_actual_digits per variable.
        # The leading positions are leading-zero padding above the most-significant
        # digit and must NOT contribute to the loss.
        unobs_float   = (1 - observed_mask.cast(dtypes.float))         # (B, N_MAX)
        unobs_dg      = unobs_float.reshape(B, n_max, 1).expand(B, n_max, n_digits)
        combined_mask = unobs_dg * digit_valid_mask                     # (B, N_MAX, N_DIGITS)
        n_active      = combined_mask.sum() + 1e-8

        gold_flat   = gold_digits.cast(dtypes.int).reshape(B * n_max * n_digits)
        cmask_flat  = combined_mask.reshape(B * n_max * n_digits)

        var_loss_sum   = Tensor.zeros((), dtype=dtypes.float).contiguous()
        var_weight_sum = 0.0
        per_breath_ce_t: list[Tensor] = []

        # MSD place values for number reconstruction: [10^(N-1), ..., 10^0]
        _place_values_np_var = [float(10 ** (n_digits - 1 - i)) for i in range(n_digits)]
        _place_values_t_var  = Tensor(_place_values_np_var, dtype=dtypes.float).reshape(1, 1, n_digits)
        _digit_vals_t_var    = Tensor([float(i) for i in range(10)], dtype=dtypes.float)

        if V105_1_2_NUMBER_MSE_ONLY:
            # ============================================================
            # NUMBER-ONLY LOSS PATH (V105_1_2_NUMBER_MSE_ONLY=1):
            # Drop per-digit CE entirely. Use ONLY per-NUMBER relative MSE
            # on the reconstructed value from digit logits. This removes the
            # "predict modal digit" partial-credit attractor — the model only
            # gets credit when the whole number matches.
            # ============================================================
            # Gold variable numbers (mask invalid positions; they're 0 anyway)
            _var_gold_float    = gold_digits.cast(dtypes.float)         # (B, N_MAX, N_DIGITS)
            _var_gold_masked   = _var_gold_float * digit_valid_mask
            var_gold_numbers   = (_var_gold_masked * _place_values_t_var).sum(axis=-1)  # (B, N_MAX)
            var_rel_denom      = var_gold_numbers.abs() + 1.0                            # (B, N_MAX)

            # Per-variable mask: unobserved AND real (non-padding)
            is_real_var_loss   = (digit_valid_mask.sum(axis=-1) > 0).cast(dtypes.float)  # (B, N_MAX)
            unobs_real_var     = unobs_float * is_real_var_loss                          # (B, N_MAX)
            n_unobs_real_var   = unobs_real_var.sum() + 1e-8

            for k, dig_logits in enumerate(digit_logits_history):
                weight_k    = 1.0 + float(k) / float(max(K - 1, 1))
                probs_k     = dig_logits.softmax(axis=-1)               # (B, N_MAX, N_DIGITS, 10)
                exp_digit_k = (probs_k * _digit_vals_t_var.reshape(1, 1, 1, 10)).sum(axis=-1)
                exp_digit_m = exp_digit_k * digit_valid_mask             # mask invalid positions
                pred_number = (exp_digit_m * _place_values_t_var).sum(axis=-1)            # (B, N_MAX)
                # Relative MSE with cold-start clip (same recipe as factor_aux)
                rel_err  = ((pred_number - var_gold_numbers) / var_rel_denom).clip(-5.0, 5.0)
                sq_err   = rel_err * rel_err
                sq_err_m = sq_err * unobs_real_var.cast(sq_err.dtype)
                ce_k     = sq_err_m.sum() / n_unobs_real_var

                per_breath_ce_t.append(ce_k)
                var_loss_sum   = var_loss_sum + ce_k * weight_k
                var_weight_sum += weight_k
        else:
            # ============================================================
            # PER-DIGIT CE PATH (default, original v105.1.2 v2 behavior)
            # ============================================================
            for k, dig_logits in enumerate(digit_logits_history):
                weight_k    = 1.0 + float(k) / float(max(K - 1, 1))
                logits_flat = dig_logits.reshape(B * n_max * n_digits, 10)
                log_probs   = logits_flat.log_softmax(axis=-1)
                gold_oh     = gold_flat.one_hot(10).cast(log_probs.dtype)
                nll         = -(log_probs * gold_oh).sum(axis=-1)
                masked_nll  = nll * cmask_flat.cast(nll.dtype)
                ce_k        = masked_nll.sum() / n_active
                per_breath_ce_t.append(ce_k)
                var_loss_sum   = var_loss_sum + ce_k * weight_k
                var_weight_sum += weight_k

        var_loss = var_loss_sum / float(var_weight_sum)

        # --- Per-NUMBER factor auxiliary loss (relative MSE) ---
        # MSD-first place values: index 0 = 10^(N-1), ..., index N-1 = 10^0.
        # Mask out invalid (leading-zero padding) digit positions so the model
        # doesn't get rewarded/penalized for guessing on positions it shouldn't
        # care about.  Gold digits at invalid positions are 0 by construction,
        # but the explicit mask matches the prediction-side masking below.
        n_valid_fac  = factor_valid.sum() + 1e-8

        # MSD place values: [10^(N-1), 10^(N-2), ..., 10^0]
        place_values_np = [float(10 ** (n_digits - 1 - i)) for i in range(n_digits)]
        place_values_t  = Tensor(place_values_np, dtype=dtypes.float).reshape(1, 1, n_digits)

        digit_vals_t    = Tensor([float(i) for i in range(10)], dtype=dtypes.float)

        # Reconstruct gold numbers, masking padding positions (they're 0 anyway).
        gold_dg_float   = factor_gold_dg.cast(dtypes.float)                       # (B, F, D)
        gold_dg_masked  = gold_dg_float * factor_digit_valid_mask                  # zero invalid
        gold_numbers    = (gold_dg_masked * place_values_t).sum(axis=-1)          # (B, F)
        rel_denom       = gold_numbers.abs() + 1.0                                 # (B, F)

        factor_aux_sum   = Tensor.zeros((), dtype=dtypes.float).contiguous()
        factor_aux_w_sum = 0.0

        for k_aux, fac_logits_k in enumerate(factor_logits_history):
            w_k_aux = 1.0 + float(k_aux) / float(max(K - 1, 1))

            fac_probs   = fac_logits_k.softmax(axis=-1)                            # (B,F,D,10)
            exp_digit   = (fac_probs * digit_vals_t.reshape(1, 1, 1, 10)).sum(axis=-1)
            # Mask out padding positions in the prediction — force the model to
            # contribute zero at invalid positions of the gold number.
            exp_digit_m = exp_digit * factor_digit_valid_mask
            pred_number = (exp_digit_m * place_values_t).sum(axis=-1)

            # Relative MSE with cold-start clip on the ERROR (not the loss):
            # clip(-5,5) bounds gradient at init without vanishing it.
            rel_err  = ((pred_number - gold_numbers) / rel_denom).clip(-5.0, 5.0)
            sq_err   = rel_err * rel_err
            sq_err_m = sq_err * factor_valid.cast(sq_err.dtype)
            fac_ce_k = sq_err_m.sum() / n_valid_fac

            factor_aux_sum   = factor_aux_sum + fac_ce_k * w_k_aux
            factor_aux_w_sum += w_k_aux

        factor_aux_loss = factor_aux_sum / float(factor_aux_w_sum)

        # --- Constraint energy (MSD place values, same as v105) ---
        final_dig_logits = digit_logits_history[-1]
        energy_loss = constraint_energy_v105(
            final_dig_logits, factor_types, factor_args,
            n_max=n_max, f_max=f_max, n_digits=n_digits,
        )

        # --- Calibration ---
        # Variable considered "correct" if all VALID digits match (invalid positions
        # — leading-zero padding above the most-significant digit — are ignored).
        # Padding rows (variables that don't exist; digit_valid_mask all zeros)
        # would otherwise auto-pass; exclude them via is_real_var.
        final_pred_dg = digit_logits_history[-1].argmax(axis=-1).detach()           # (B,N,D)
        dg_eq    = (final_pred_dg == gold_digits.cast(dtypes.int)).cast(dtypes.float)
        dg_match_or_invalid = dg_eq * digit_valid_mask + (1.0 - digit_valid_mask)
        var_eq   = dg_match_or_invalid.min(axis=-1)                                 # (B,N)
        is_real_var = (digit_valid_mask.sum(axis=-1) > 0).cast(dtypes.float)        # (B,N)
        unobs_real  = unobs_float * is_real_var                                     # (B,N)
        eq_unobs    = var_eq * unobs_real
        n_unobs_per = unobs_real.sum(axis=-1) + 1e-8
        correct     = eq_unobs.sum(axis=-1) / n_unobs_per

        calib_loss_sum = Tensor.zeros((), dtype=dtypes.float).contiguous()
        for k, calib_k in enumerate(calib_history):
            prog       = float(k) / float(max(K - 1, 1))
            target_k   = 0.5 + (correct - 0.5) * prog
            calib_loss_sum = calib_loss_sum + ((calib_k - target_k) ** 2).mean()
        calib_loss = calib_loss_sum / float(K)

        # --- Metrics ---
        cell_acc  = (eq_unobs.sum() / (unobs_real.sum() + 1e-8)).detach()
        query_acc = correct.mean().detach()

        # --- Total loss ---
        total_ce = var_loss + fw * factor_aux_loss + aw * calib_loss + ew * energy_loss
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
            grad_norm  = (sq_sum + 1e-12).sqrt()
            clip_coef  = (Tensor(gc, dtype=dtypes.float) / (grad_norm + 1e-6)).minimum(
                Tensor(1.0, dtype=dtypes.float)
            )
            for p in params:
                if p.grad is not None:
                    p.grad = p.grad * clip_coef.cast(p.grad.dtype)

        opt.step()

        return (
            total_ce.realize(),
            healthy.realize(),
            var_loss.realize(),
            factor_aux_loss.realize(),
            calib_loss.realize(),
            energy_loss.realize(),
            cell_acc.realize(),
            query_acc.realize(),
            *(ce.realize() for ce in per_breath_ce_t),
        )

    _JIT_V105_1_2_CACHE[key] = _step
    print(
        f"[JIT] v105.1.2 fg step ready (cache={len(_JIT_V105_1_2_CACHE)}); "
        f"first call compiles...",
        flush=True,
    )
    return _step


def _compile_jit_fg_eval_v105_1_2(
    model: Any,
    K: int,
    B: int,
    n_max: int = V105_1_2_N_MAX,
    f_max: int = V105_1_2_F_MAX,
    n_digits: int = V105_1_2_N_DIGITS,
):
    """Compile a TinyJit'd eval step (forward only, no gradient).

    Takes digit_valid_mask so cell_acc treats invalid (leading-zero padding)
    positions as automatically correct (consistent with the train loss).
    """
    key = ("eval_v105_1_2", id(model), int(K), int(B), int(n_max), int(f_max), int(n_digits))
    if key in _JIT_V105_1_2_CACHE:
        return _JIT_V105_1_2_CACHE[key]

    print(f"[JIT] compile v105.1.2 fg eval: K={K} B={B} n_digits={n_digits}...", flush=True)

    @TinyJit
    def _eval(
        digit_init: Tensor,
        node_kinds: Tensor,
        staging_mask: Tensor,
        head_op_mask: Tensor,
        gold_digits: Tensor,
        observed_mask: Tensor,
        digit_valid_mask: Tensor,
    ):
        digit_logits_history, _, _ = fg_breathing_forward_v105_1_2(
            model, digit_init, node_kinds, staging_mask, head_op_mask,
            K=K, n_max=n_max, f_max=f_max, n_digits=n_digits,
        )
        final_logits  = digit_logits_history[-1]
        pred_dg       = final_logits.argmax(axis=-1)
        dg_eq         = (pred_dg == gold_digits.cast(pred_dg.dtype)).cast(dtypes.float)
        # Variable correct: all VALID digits match (invalid digits ignored).
        # Exclude padding rows (digit_valid_mask all zeros) so they don't auto-pass.
        dg_or_invalid = dg_eq * digit_valid_mask + (1.0 - digit_valid_mask)
        var_eq        = dg_or_invalid.min(axis=-1)
        is_real_var   = (digit_valid_mask.sum(axis=-1) > 0).cast(dtypes.float)
        unobs_real    = (1 - observed_mask.cast(dtypes.float)) * is_real_var
        cell_acc      = (var_eq * unobs_real).sum() / (unobs_real.sum() + 1e-8)
        return pred_dg.realize(), cell_acc.realize()

    _JIT_V105_1_2_CACHE[key] = _eval
    print(f"[JIT] v105.1.2 eval ready (cache={len(_JIT_V105_1_2_CACHE)})", flush=True)
    return _eval
