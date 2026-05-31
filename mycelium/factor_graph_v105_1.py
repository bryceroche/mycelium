"""v105.1 factor graph breathing transformer — digit RoPE architecture.

Same as v105 EXCEPT: the additive digit_pos_embed is replaced with RoPE rotation
applied to the digit dimension.  Each of the N_DIGITS positions within a variable
gets a rotary rotation angle of p × freq(dim_pair) — making digit positions
RELATIONALLY AWARE via attention rather than merely additively distinct.

Motivation:
  v105 failed at eval (0% val cell_acc) despite 93% per-digit train accuracy.
  Root cause: per-digit predictions were architecturally independent — no mechanism
  enforced that digits 0-4 within a variable combined into a coherent number.
  The 5-digit positional one-hots (digit_pos_embed) added an additive offset that
  gradient descent could learn to ignore.

  RoPE rotation makes digit position INTRINSIC to the embedding geometry: a query
  at digit-position p can attend to key at digit-position p' only through the
  rotated inner product, which favours small |p - p'| (standard RoPE relative-position
  property).  Carry propagation (digit p attending to digit p+1) becomes naturally
  representable once RoPE encodes the relative digit offset.

Architecture change from v105:
  embed[v, p] = rotate(digit_codebook[digit_d] + var_pos_embed[v], digit_pos=p)

  # Old (v105):
  embed[v, p] = digit_codebook[d] + digit_pos_embed[p] + var_pos_embed[v]

  # New (v105.1):
  raw = digit_codebook[d] + var_pos_embed[v]   # NO additive digit pos
  embed[v, p] = apply_rope_digit(raw, p)        # full-hidden RoPE by digit pos

RoPE details:
  - base_theta = 10000.0 (standard).
  - Full hidden-dim rotation (all H dimensions in H/2 pairs).
  - Only applied to variable digit positions; factor nodes are unchanged.
  - The rotation happens BEFORE the transformer layers (in the embedding step).

Diagnostic added:
  At eval, compute attention weight from digit-pos-3 → digit-pos-4 (carry direction)
  averaged over all variables, heads, and problems.  If RoPE is doing its job,
  adjacent-digit attention should be systematically higher than non-adjacent.

Everything else is identical to v105 (same sequence layout, same losses, same JIT
structure, same env-var gating).

Env var gates (same as v105 plus one new):
  V105_TASK=1
  V105_K_MAX=8
  V105_N_DIGITS=5
  V105_N_MAX=16
  V105_F_MAX=8
  V105_ENERGY_WEIGHT=0.01
  V105_CALIB_WEIGHT=0.05
  V105_FACTOR_AUX_WEIGHT=0.5
  V105_1_ROPE_BASE=10000    — digit RoPE base theta (default 10000)
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

# Re-export shared utilities from v105 so the training script only needs one import.
from mycelium.factor_graph_v105 import (
    OP_ADD, OP_SUB, OP_MUL, OP_DIV,
    value_to_digits, digits_to_value,
    _expected_value_v105, constraint_energy_v105,
    fg_accuracy_v105,
)

# Op index constants (same as v100/v105)
__all__ = [
    "OP_ADD", "OP_SUB", "OP_MUL", "OP_DIV",
    "value_to_digits", "digits_to_value",
]

V105_TASK               = int(os.environ.get("V105_TASK", "0")) > 0
V105_K_MAX              = int(os.environ.get("V105_K_MAX",              "8"))
V105_N_DIGITS           = int(os.environ.get("V105_N_DIGITS",           "5"))
V105_N_MAX              = int(os.environ.get("V105_N_MAX",              "16"))
V105_F_MAX              = int(os.environ.get("V105_F_MAX",               "8"))
V105_N_HEADS            = 16    # fixed: Pythia-410M
V105_ENERGY_WEIGHT      = float(os.environ.get("V105_ENERGY_WEIGHT",    "0.01"))
V105_CALIB_WEIGHT       = float(os.environ.get("V105_CALIB_WEIGHT",     "0.05"))
V105_FACTOR_AUX_WEIGHT  = float(os.environ.get("V105_FACTOR_AUX_WEIGHT","0.5"))

# v105.1-specific: digit RoPE base theta.
V105_1_ROPE_BASE = float(os.environ.get("V105_1_ROPE_BASE", "10000.0"))


# ---------------------------------------------------------------------------
# Digit-axis RoPE
# ---------------------------------------------------------------------------

def _precompute_digit_rope(n_digits: int, hidden: int, base: float = 10000.0) -> tuple[np.ndarray, np.ndarray]:
    """Precompute cos/sin rotation matrices for digit positions.

    Standard RoPE construction over all H dimensions:
      freq_i = base^(-2i / H)  for i in 0 .. H//2-1
      angle_p_i = p * freq_i
      cos_table[p, 2i]   = cos(angle_p_i)
      cos_table[p, 2i+1] = cos(angle_p_i)
      sin_table[p, 2i]   = sin(angle_p_i)   (with sign flip for the rotated pair)
      sin_table[p, 2i+1] = sin(angle_p_i)

    Returns:
      cos_table : (n_digits, hidden) float32
      sin_table : (n_digits, hidden) float32

    The rotation for dimension pair (2i, 2i+1) at digit position p is:
      [cos(p*freq_i)  -sin(p*freq_i)] [x_2i  ]
      [sin(p*freq_i)   cos(p*freq_i)] [x_2i+1]

    Stored in the interleaved format matching standard RoPE _rotate:
      cos_table[p] broadcasts as x * cos + rotate_half(x) * sin
    where rotate_half(x)[2i] = -x[2i+1], rotate_half(x)[2i+1] = x[2i].
    """
    half = hidden // 2
    # freq_i = base^(-2i/hidden) for i in 0..half-1
    inv_freq = np.array(
        [1.0 / (base ** (2 * i / hidden)) for i in range(half)], dtype=np.float64
    )  # (half,)
    positions = np.arange(n_digits, dtype=np.float64)  # (n_digits,)
    # angles[p, i] = p * inv_freq[i]
    angles = np.outer(positions, inv_freq)  # (n_digits, half)

    cos_half = np.cos(angles)  # (n_digits, half)
    sin_half = np.sin(angles)  # (n_digits, half)

    # Interleave: repeat each freq for even/odd index pairs.
    # cos_table[:, 2i] = cos_table[:, 2i+1] = cos_half[:, i]
    cos_table = np.repeat(cos_half, 2, axis=-1).astype(np.float32)  # (n_digits, hidden)
    sin_table = np.repeat(sin_half, 2, axis=-1).astype(np.float32)  # (n_digits, hidden)

    return cos_table, sin_table


def _rotate_half_np(x: np.ndarray) -> np.ndarray:
    """rotate_half on the last dimension (numpy, for precomputed tables only)."""
    x1 = x[..., ::2]   # even indices
    x2 = x[..., 1::2]  # odd indices
    out = np.empty_like(x)
    out[..., ::2]  = -x2
    out[..., 1::2] =  x1
    return out


def apply_rope_digit_tg(raw: Tensor, cos_table: Tensor, sin_table: Tensor,
                         n_digits: int, hidden: int) -> Tensor:
    """Apply digit-axis RoPE rotation to raw embeddings.

    raw:         (..., n_digits, hidden)  — pre-rotation embeddings
    cos_table:   (n_digits, hidden)       — precomputed cos values per digit pos
    sin_table:   (n_digits, hidden)       — precomputed sin values per digit pos

    Returns: (..., n_digits, hidden) — rotated embeddings.

    Implementation: x * cos + rotate_half(x) * sin
    where rotate_half swaps pairs (2i, 2i+1) with a sign flip on the first:
      rotate_half(x)[..., 2i]   = -x[..., 2i+1]
      rotate_half(x)[..., 2i+1] =  x[..., 2i  ]
    """
    # rotate_half via slice-and-concat (no in-place ops; tinygrad JIT safe)
    x_even = raw[..., ::2]    # (..., n_digits, hidden//2)
    x_odd  = raw[..., 1::2]   # (..., n_digits, hidden//2)
    # Interleave: [-x_odd, x_even] in the pair-wise sense
    # Stack along last dim then reshape: each pair (2i, 2i+1) → (-x_odd_i, x_even_i)
    neg_x_odd = -x_odd        # (..., n_digits, hidden//2)
    # Interleave by stacking and reshaping: (..., n_digits, 2, hidden//2) → (..., n_digits, hidden)
    stacked = Tensor.stack(neg_x_odd, x_even, dim=-1)  # (..., n_digits, hidden//2, 2)
    rot_half = stacked.reshape(*raw.shape[:-1], hidden)  # (..., n_digits, hidden)

    # Broadcast cos/sin tables (n_digits, hidden) over batch dims
    ndim_extra = len(raw.shape) - 2  # number of batch dimensions before (n_digits, hidden)
    c = cos_table
    s = sin_table
    for _ in range(ndim_extra):
        c = c.reshape(1, *c.shape)
        s = s.reshape(1, *s.shape)

    return raw * c.cast(raw.dtype) + rot_half * s.cast(raw.dtype)


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def embed_factor_graph_v105_1(
    digit_init: Tensor,       # (B, N_MAX, N_DIGITS, 10) — observed: one-hot; unobs: uniform
    node_kinds: Tensor,        # (B, T_MAX) int
    var_pos_embed: Tensor,     # (N_MAX, H)
    factor_pos_embed: Tensor,  # (F_MAX, H)
    node_kind_embed: Tensor,   # (3, H)
    digit_codebook: Tensor,    # (10, H)
    digit_rope_cos: Tensor,    # (N_DIGITS, H) — precomputed digit RoPE cos table
    digit_rope_sin: Tensor,    # (N_DIGITS, H) — precomputed digit RoPE sin table
    n_max: int,
    n_digits: int,
    f_max: int,
) -> Tensor:
    """Build (B, T_MAX, H) hidden states with digit-axis RoPE.

    Layout (identical to v105):
      positions [v*n_digits : v*n_digits+n_digits]  ← variable v's digit positions
      positions [n_max*n_digits : n_max*n_digits+f_max]  ← factor nodes

    Key difference from v105:
      Instead of: var_digit_h = var_digit_state + digit_pos_embed + var_pos_embed
      We use:     raw = var_digit_state + var_pos_embed
                  var_digit_h = apply_rope_digit(raw, digit_pos)

    The RoPE rotation is applied to the (N_MAX, N_DIGITS, H) block before flattening,
    making each digit position p carry a geometrically distinct representation that
    is rotated by p × freq relative to p=0.  Attention between digit positions then
    naturally reflects their relative offset (standard RoPE relative-position property).
    """
    B = int(digit_init.shape[0])
    H = int(var_pos_embed.shape[1])
    T = n_max * n_digits + f_max

    # --- Variable digit positions ---
    # digit_init: (B, N_MAX, N_DIGITS, 10)
    # digit_codebook: (10, H)
    di_cb = digit_init.cast(digit_codebook.dtype)       # (B, N_MAX, N_DIGITS, 10)
    var_digit_state = di_cb @ digit_codebook             # (B, N_MAX, N_DIGITS, H)

    # Per-variable position embedding broadcast over digit dimension (NO digit_pos_embed added)
    vpe = var_pos_embed.reshape(1, n_max, 1, H).cast(var_digit_state.dtype)
    raw = var_digit_state + vpe.expand(B, n_max, n_digits, H)  # (B, N_MAX, N_DIGITS, H)

    # Apply digit RoPE: rotate raw by digit position p along the n_digits axis.
    # digit_rope_cos/sin: (N_DIGITS, H) — broadcast over (B, N_MAX) batch dims.
    var_digit_h = apply_rope_digit_tg(raw, digit_rope_cos, digit_rope_sin,
                                       n_digits=n_digits, hidden=H)  # (B, N_MAX, N_DIGITS, H)

    # Flatten to (B, N_MAX * N_DIGITS, H)
    var_tokens = var_digit_h.reshape(B, n_max * n_digits, H)

    # --- Factor positions (unchanged from v105) ---
    factor_pos = factor_pos_embed.reshape(1, f_max, H).cast(var_tokens.dtype).expand(B, f_max, H)

    # Concatenate
    x = var_tokens.cat(factor_pos, dim=1)  # (B, T, H)

    # Node-kind embedding
    nk_clamped = node_kinds.clip(0, 2)
    nk_oh      = nk_clamped.one_hot(3).cast(x.dtype)
    nk_emb     = nk_oh @ node_kind_embed.cast(x.dtype)
    x = x + nk_emb

    return x


# ---------------------------------------------------------------------------
# One transformer layer (identical to v105)
# ---------------------------------------------------------------------------

def fg_layer_forward_v105_1(
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
# Iterative prefill loop
# ---------------------------------------------------------------------------

def fg_breathing_forward_v105_1(
    model: Any,
    digit_init: Tensor,       # (B, N_MAX, N_DIGITS, 10)
    node_kinds: Tensor,        # (B, T_MAX) int
    staging_mask: Tensor,      # (B, K_MAX, T_MAX, T_MAX)
    head_op_mask: Tensor,      # (B, N_HEADS, T_MAX, T_MAX)
    K: int,
    n_max: int = V105_N_MAX,
    f_max: int = V105_F_MAX,
    n_digits: int = V105_N_DIGITS,
) -> tuple[list[Tensor], list[Tensor], list[Tensor]]:
    """Run K iterative-prefill breaths (identical structure to v105 except embedding)."""
    assert hasattr(model, "fg_v105_1_digit_codebook"), \
        "model has no v105.1 params; was attach_fg_params_v105_1 called?"

    digit_codebook   = model.fg_v105_1_digit_codebook    # (10, H)
    digit_rope_cos   = model.fg_v105_1_digit_rope_cos    # (N_DIGITS, H)
    digit_rope_sin   = model.fg_v105_1_digit_rope_sin    # (N_DIGITS, H)
    var_pos_embed    = model.fg_v105_1_var_pos_embed      # (N_MAX, H)
    factor_pos_embed = model.fg_v105_1_factor_pos_embed  # (F_MAX, H)
    node_kind_embed  = model.fg_v105_1_node_kind_embed   # (3, H)
    breath_embed     = model.fg_v105_1_breath_embed      # (K_max, H)
    delta_gate       = model.fg_v105_1_delta_gate        # (K_max,)
    calib_head_w     = model.fg_v105_1_calib_head_w      # (H, 1)
    calib_head_b     = model.fg_v105_1_calib_head_b      # (1,)

    B = int(digit_init.shape[0])
    T = n_max * n_digits + f_max
    H = int(var_pos_embed.shape[1])

    x = embed_factor_graph_v105_1(
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
        stk_h = stk.reshape(B, 1, T, T).expand(B, V105_N_HEADS, T, T)
        combined = stk_h.cast(x.dtype) + head_op_mask.cast(x.dtype)

        # 3. Four transformer layers
        h = x_in
        for layer in layers[:4]:
            h = fg_layer_forward_v105_1(layer, h, combined)

        # 4. Delta gate
        gate_k = delta_gate[k].cast(h.dtype).reshape(1, 1, 1)
        delta  = h - x_pre
        x      = x_pre + gate_k * delta

        # 5a. Variable digit logits
        x_ln = _layernorm(x, model.ln_f_g, model.ln_f_b, model.cfg.layer_norm_eps).cast(dtypes.float)
        n_var_tokens = n_max * n_digits
        var_tokens   = x_ln[:, :n_var_tokens, :]
        var_tokens_r = var_tokens.reshape(B, n_max, n_digits, -1)
        cb_fp        = digit_codebook.cast(dtypes.float)
        digit_logits_k = var_tokens_r @ cb_fp.T                # (B, N_MAX, N_DIGITS, 10)
        digit_logits_history.append(digit_logits_k)

        # 5b. Factor digit logits
        fac_tokens   = x_ln[:, n_var_tokens:n_var_tokens + f_max, :]
        fac_tokens_r = fac_tokens.reshape(B, f_max, 1, -1).expand(B, f_max, n_digits, int(x_ln.shape[-1]))
        fac_logits_k = fac_tokens_r @ cb_fp.T
        factor_logits_history.append(fac_logits_k)

        # 5c. Calibration
        pool        = x_ln.mean(axis=1)
        calib_logit = pool @ calib_head_w.cast(dtypes.float) + calib_head_b.cast(dtypes.float)
        calib_k     = calib_logit.reshape(-1).sigmoid()
        calib_history.append(calib_k)

    return digit_logits_history, factor_logits_history, calib_history


# ---------------------------------------------------------------------------
# Model parameter attachment
# ---------------------------------------------------------------------------

def attach_fg_params_v105_1(
    model: Any,
    hidden: int,
    n_digits: int = V105_N_DIGITS,
    n_max: int = V105_N_MAX,
    f_max: int = V105_F_MAX,
    k_max: int | None = None,
    rope_base: float | None = None,
) -> None:
    """Allocate v105.1 digit-RoPE factor-graph params on `model`.

    Key difference from v105: no digit_pos_embed (learnable sinusoidal).
    Instead stores precomputed digit_rope_cos / digit_rope_sin tables (frozen).

    Initialization:
      digit_codebook   (10, hidden):    random QR × 1.0 — orthonormal rows
      digit_rope_cos   (N_DIGITS, H):   precomputed (frozen, not a gradient param)
      digit_rope_sin   (N_DIGITS, H):   precomputed (frozen, not a gradient param)
      var_pos_embed    (N_MAX, H):      randn 0.02
      factor_pos_embed (F_MAX, H):      randn 0.02
      node_kind_embed  (3, H):          randn 0.02
      breath_embed     (K_max, H):      QR-orthonormal × 0.5
      delta_gate       (K_max,):        ones
      calib_head_w     (H, 1):          randn 0.02
      calib_head_b     (1,):            zeros

    The RoPE tables are precomputed once and stored as non-gradient tensors
    (they are NOT in fg_v105_1_parameters so the optimizer never updates them).
    This matches how breathing.py's RoPE stores its cos/sin tables.
    """
    if k_max is None:
        k_max = V105_K_MAX
    if rope_base is None:
        rope_base = V105_1_ROPE_BASE

    rng = np.random.RandomState(20011)

    # --- Digit codebook (10, hidden): orthonormal rows ---
    raw_dc = rng.randn(max(hidden, 10), hidden).astype(np.float32)
    q_dc, _ = np.linalg.qr(raw_dc)
    dc = q_dc[:10].astype(np.float32) * 1.0
    model.fg_v105_1_digit_codebook = Tensor(dc, dtype=dtypes.float).contiguous()

    # --- Digit RoPE tables (N_DIGITS, H): FROZEN, not a gradient parameter ---
    cos_t, sin_t = _precompute_digit_rope(n_digits, hidden, base=rope_base)
    model.fg_v105_1_digit_rope_cos = Tensor(cos_t, dtype=dtypes.float).contiguous()
    model.fg_v105_1_digit_rope_sin = Tensor(sin_t, dtype=dtypes.float).contiguous()

    # --- Variable position embedding (N_MAX, hidden) ---
    vp = rng.randn(n_max, hidden).astype(np.float32) * 0.02
    model.fg_v105_1_var_pos_embed = Tensor(vp, dtype=dtypes.float).contiguous()

    # --- Factor position embedding (F_MAX, hidden) ---
    fp_emb = rng.randn(f_max, hidden).astype(np.float32) * 0.02
    model.fg_v105_1_factor_pos_embed = Tensor(fp_emb, dtype=dtypes.float).contiguous()

    # --- Node-kind embedding (3, hidden) ---
    nk = rng.randn(3, hidden).astype(np.float32) * 0.02
    model.fg_v105_1_node_kind_embed = Tensor(nk, dtype=dtypes.float).contiguous()

    # --- Calibration head ---
    cw = (rng.randn(hidden, 1) * 0.02).astype(np.float32)
    model.fg_v105_1_calib_head_w = Tensor(cw, dtype=dtypes.float).contiguous()
    model.fg_v105_1_calib_head_b = Tensor.zeros((1,), dtype=dtypes.float).contiguous()

    # --- Per-breath embedding: QR-orthonormal at scale 0.5 ---
    rng_be = np.random.RandomState(20012)
    raw_be = rng_be.randn(max(k_max, hidden), hidden).astype(np.float32)
    q_be, _ = np.linalg.qr(raw_be)
    be = q_be[:k_max].astype(np.float32) * 0.5
    model.fg_v105_1_breath_embed = Tensor(be, dtype=dtypes.float).contiguous()

    # --- Per-breath delta gate: init 1.0 ---
    model.fg_v105_1_delta_gate = Tensor.ones((k_max,), dtype=dtypes.float).contiguous()

    print(
        f"[v105.1] params attached: digit_codebook=(10,{hidden}), "
        f"digit_rope (N_DIGITS={n_digits}, H={hidden}, base={rope_base:.0f}) [FROZEN], "
        f"var_pos_embed=({n_max},{hidden}), factor_pos_embed=({f_max},{hidden}), "
        f"breath_embed=({k_max},{hidden}), "
        f"T={n_max * n_digits + f_max}",
        flush=True,
    )


def fg_v105_1_parameters(model: Any) -> list[Tensor]:
    """Trainable v105.1 factor-graph-specific params.

    NOTE: digit_rope_cos and digit_rope_sin are deliberately EXCLUDED.
    They are precomputed frozen tables (like breathing.py's rope.cos/rope.sin).
    Updating them via gradient would undo the RoPE inductive bias.
    """
    return [
        model.fg_v105_1_digit_codebook,
        # digit_rope_cos/sin are FROZEN — not here
        model.fg_v105_1_var_pos_embed,
        model.fg_v105_1_factor_pos_embed,
        model.fg_v105_1_node_kind_embed,
        model.fg_v105_1_calib_head_w,
        model.fg_v105_1_calib_head_b,
        model.fg_v105_1_breath_embed,
        model.fg_v105_1_delta_gate,
    ]


def fg_v105_1_state_dict(model: Any) -> dict[str, Tensor]:
    """State dict: includes frozen RoPE tables so checkpoint is self-contained."""
    return {
        "fg_v105_1.digit_codebook":    model.fg_v105_1_digit_codebook,
        "fg_v105_1.digit_rope_cos":    model.fg_v105_1_digit_rope_cos,
        "fg_v105_1.digit_rope_sin":    model.fg_v105_1_digit_rope_sin,
        "fg_v105_1.var_pos_embed":     model.fg_v105_1_var_pos_embed,
        "fg_v105_1.factor_pos_embed":  model.fg_v105_1_factor_pos_embed,
        "fg_v105_1.node_kind_embed":   model.fg_v105_1_node_kind_embed,
        "fg_v105_1.calib_head_w":      model.fg_v105_1_calib_head_w,
        "fg_v105_1.calib_head_b":      model.fg_v105_1_calib_head_b,
        "fg_v105_1.breath_embed":      model.fg_v105_1_breath_embed,
        "fg_v105_1.delta_gate":        model.fg_v105_1_delta_gate,
    }


# ---------------------------------------------------------------------------
# JIT training step
# ---------------------------------------------------------------------------

_JIT_V105_1_CACHE: dict = {}


def _compile_jit_fg_step_v105_1(
    model: Any,
    opt: Any,
    K: int,
    B: int,
    factor_aux_weight: float = V105_FACTOR_AUX_WEIGHT,
    calib_weight: float = V105_CALIB_WEIGHT,
    energy_weight: float = V105_ENERGY_WEIGHT,
    n_max: int = V105_N_MAX,
    f_max: int = V105_F_MAX,
    n_digits: int = V105_N_DIGITS,
    grad_clip: float = 1.0,
):
    """Compile a TinyJit'd train step for v105.1 (identical structure to v105)."""
    key = ("v105_1", id(model), id(opt), int(K), int(B), float(factor_aux_weight),
           float(calib_weight), float(energy_weight), int(n_max), int(f_max),
           int(n_digits), float(grad_clip))
    if key in _JIT_V105_1_CACHE:
        return _JIT_V105_1_CACHE[key]

    aw     = float(calib_weight)
    fw     = float(factor_aux_weight)
    ew     = float(energy_weight)
    gc     = float(grad_clip)
    params = opt.params

    print(
        f"[JIT] compile v105.1 fg step: K={K} B={B} n_digits={n_digits} "
        f"T={n_max * n_digits + f_max} aw={aw} fw={fw} ew={ew} gc={gc}...",
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
    ):
        opt.zero_grad()

        digit_logits_history, factor_logits_history, calib_history = \
            fg_breathing_forward_v105_1(
                model, digit_init, node_kinds, staging_mask, head_op_mask,
                K=K, n_max=n_max, f_max=f_max, n_digits=n_digits,
            )

        # --- Main CE on unobserved variable digit positions ---
        unobs_float   = (1 - observed_mask.cast(dtypes.float))
        unobs_dg      = unobs_float.reshape(B, n_max, 1).expand(B, n_max, n_digits)
        n_unobs_dg    = unobs_dg.sum() + 1e-8
        gold_flat     = gold_digits.cast(dtypes.int).reshape(B * n_max * n_digits)
        unobs_flat    = unobs_dg.reshape(B * n_max * n_digits)

        var_loss_sum  = Tensor.zeros((), dtype=dtypes.float).contiguous()
        var_weight_sum = 0.0
        per_breath_ce_t: list[Tensor] = []

        for k, dig_logits in enumerate(digit_logits_history):
            weight_k    = 1.0 + float(k) / float(max(K - 1, 1))
            logits_flat = dig_logits.reshape(B * n_max * n_digits, 10)
            log_probs   = logits_flat.log_softmax(axis=-1)
            gold_oh     = gold_flat.one_hot(10).cast(log_probs.dtype)
            nll         = -(log_probs * gold_oh).sum(axis=-1)
            masked_nll  = nll * unobs_flat.cast(nll.dtype)
            ce_k        = masked_nll.sum() / n_unobs_dg
            per_breath_ce_t.append(ce_k)
            var_loss_sum   = var_loss_sum + ce_k * weight_k
            var_weight_sum += weight_k

        var_loss = var_loss_sum / float(var_weight_sum)

        # --- Factor-execute auxiliary loss ---
        n_valid_fac  = factor_valid.sum() + 1e-8
        fgd_flat     = factor_gold_dg.cast(dtypes.int).reshape(B * f_max * n_digits)
        valid_dg     = factor_valid.reshape(B, f_max, 1).expand(B, f_max, n_digits)
        valid_flat   = valid_dg.reshape(B * f_max * n_digits)
        n_valid_dg   = valid_flat.sum() + 1e-8
        gold_fac_oh  = fgd_flat.one_hot(10).cast(dtypes.float)

        factor_aux_sum   = Tensor.zeros((), dtype=dtypes.float).contiguous()
        factor_aux_w_sum = 0.0

        for k_aux, fac_logits_k in enumerate(factor_logits_history):
            w_k_aux    = 1.0 + float(k_aux) / float(max(K - 1, 1))
            fac_flat   = fac_logits_k.reshape(B * f_max * n_digits, 10)
            fac_lp     = fac_flat.log_softmax(axis=-1)
            fac_nll    = -(fac_lp * gold_fac_oh).sum(axis=-1)
            fac_masked = fac_nll * valid_flat.cast(fac_nll.dtype)
            fac_ce_k   = fac_masked.sum() / n_valid_dg
            factor_aux_sum   = factor_aux_sum + fac_ce_k * w_k_aux
            factor_aux_w_sum += w_k_aux

        factor_aux_loss = factor_aux_sum / float(factor_aux_w_sum)

        # --- Constraint energy ---
        final_dig_logits = digit_logits_history[-1]
        energy_loss = constraint_energy_v105(
            final_dig_logits, factor_types, factor_args,
            n_max=n_max, f_max=f_max, n_digits=n_digits,
        )

        # --- Calibration ---
        final_pred_dg = digit_logits_history[-1].argmax(axis=-1).detach()
        dg_eq   = (final_pred_dg == gold_digits.cast(dtypes.int)).cast(dtypes.float)
        var_eq  = dg_eq.min(axis=-1)
        unobs_2d = unobs_float
        eq_unobs = var_eq * unobs_2d
        n_unobs_per = unobs_2d.sum(axis=-1) + 1e-8
        correct = eq_unobs.sum(axis=-1) / n_unobs_per

        calib_loss_sum = Tensor.zeros((), dtype=dtypes.float).contiguous()
        for k, calib_k in enumerate(calib_history):
            prog       = float(k) / float(max(K - 1, 1))
            target_k   = 0.5 + (correct - 0.5) * prog
            calib_loss_sum = calib_loss_sum + ((calib_k - target_k) ** 2).mean()
        calib_loss = calib_loss_sum / float(K)

        # --- Metrics ---
        cell_acc  = (eq_unobs.sum() / (unobs_2d.sum() + 1e-8)).detach()
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

    _JIT_V105_1_CACHE[key] = _step
    print(f"[JIT] v105.1 fg step ready (cache={len(_JIT_V105_1_CACHE)}); first call compiles...",
          flush=True)
    return _step


def _compile_jit_fg_eval_v105_1(
    model: Any,
    K: int,
    B: int,
    n_max: int = V105_N_MAX,
    f_max: int = V105_F_MAX,
    n_digits: int = V105_N_DIGITS,
):
    """Compile a TinyJit'd eval step (forward only)."""
    key = ("eval_v105_1", id(model), int(K), int(B), int(n_max), int(f_max), int(n_digits))
    if key in _JIT_V105_1_CACHE:
        return _JIT_V105_1_CACHE[key]

    print(f"[JIT] compile v105.1 fg eval: K={K} B={B} n_digits={n_digits}...", flush=True)

    @TinyJit
    def _eval(
        digit_init: Tensor,
        node_kinds: Tensor,
        staging_mask: Tensor,
        head_op_mask: Tensor,
        gold_digits: Tensor,
        observed_mask: Tensor,
    ):
        digit_logits_history, _, _ = fg_breathing_forward_v105_1(
            model, digit_init, node_kinds, staging_mask, head_op_mask,
            K=K, n_max=n_max, f_max=f_max, n_digits=n_digits,
        )
        final_logits  = digit_logits_history[-1]
        pred_dg       = final_logits.argmax(axis=-1)
        dg_eq         = (pred_dg == gold_digits.cast(pred_dg.dtype)).cast(dtypes.float)
        var_eq        = dg_eq.min(axis=-1)
        unobs         = (1 - observed_mask.cast(dtypes.float))
        cell_acc      = (var_eq * unobs).sum() / (unobs.sum() + 1e-8)
        return pred_dg.realize(), cell_acc.realize()

    _JIT_V105_1_CACHE[key] = _eval
    print(f"[JIT] v105.1 eval ready (cache={len(_JIT_V105_1_CACHE)})", flush=True)
    return _eval
