"""v200-Sudoku — v98 constraint-propagation paradigm on a choice of backbone.

Supports two backbones selected via the V200_SUDOKU_BASE env var:

  V200_SUDOKU_BASE=smollm2_1_7b  (default)
    SmolLM2-1.7B L0-L3: hidden 2048d, 32 heads × 64 head_dim, SwiGLU + RMSNorm
    Mask partition: 10+10+10+2 = 32 heads
    RoPE: yes (theta=130000, applied by LlamaLayer; π rotation added on top)
    Backbone params: ~268M (L0-L3)

  V200_SUDOKU_BASE=pythia_410m
    Pythia-410M L0-L3: hidden 1024d, 16 heads × 64 head_dim, GELU + LayerNorm
    Mask partition: 5+5+5+1 = 16 heads  (exact v98 partition)
    RoPE: NONE (Pythia uses partial RoPE in its standard path, but the sudoku
          forward skips it — cells are positionally addressed by the learned
          (81, H) embedding, not by sequential token position. Same as v98.)
          The π-cycled per-breath Q rotation is the only per-breath angular signal.
    Backbone params: ~50M (L0-L3)

v98 components carried forward IDENTICALLY across both paths:
  - K = 20 breaths
  - Per-breath additive embedding  (K_max × H, QR-orthonormal init scale 0.5)
  - π-cycled per-breath Q rotation (k·π/K_max pairwise)
  - Structured attention masks (row/col/box/global, head count adapts to base)
  - Per-breath delta_gate (K,) — init 1.0, convex residual blend
  - Per-breath weighted CE: weight_k = 1 + k/(K-1)
  - Constraint energy loss on final breath
  - Calibration head: scalar P(correct|state) per breath
  - 9-way digit codebook readout (9 × H, orthonormal × 0.1 scale)
  - NO waist (deferred to a separate experiment)

SmolLM2 adaptation decisions:
  - RMSNorm (not LayerNorm): The final-breath readout uses a dedicated
    sudoku_v200_final_norm (shape H) as the projection RMSNorm.
  - No per-layer Q/K biases (LlamaLayer uses bias-free projections).

Pythia adaptation decisions:
  - Parallel-residual: y = x + Attn(in_LN(x)) + FFN(post_LN(x)) — two separate
    LayerNorm inputs (in_ln and post_ln), one additive output.
  - LayerNorm for final readout (uses model.pythia_ln_f_g / ln_f_b).
  - No RoPE in the sudoku forward (matches v98 exactly).

Env vars:
  V200_SUDOKU_BASE=smollm2_1_7b    backbone selection (default)
  V200_SUDOKU_TASK=1               enable this forward path
  V200_SUDOKU_K_MAX=20             number of iterative-prefill breaths
  V200_SUDOKU_CONSTRAINT_WEIGHT=0.3
  V200_SUDOKU_CALIB_WEIGHT=0.1
  V200_SUDOKU_BREATH_EMBED_SCALE=0.5
"""
from __future__ import annotations

import math
import os
import time as _time
from typing import Any

import numpy as np
from tinygrad import Tensor, dtypes
from tinygrad.engine.jit import TinyJit

from mycelium.llama_loader import (
    LlamaConfig,
    LlamaLayer,
    SMOLLM2_1_7B_CFG,
    _rms_norm,
    _rotate_half,
    _apply_rope,
    attach_llama_layers,
    load_llama_weights,
)


V200_SUDOKU_TASK               = int(os.environ.get("V200_SUDOKU_TASK", "0")) > 0
V200_SUDOKU_K_MAX              = int(os.environ.get("V200_SUDOKU_K_MAX", "20"))
V200_SUDOKU_CONSTRAINT_WEIGHT  = float(os.environ.get("V200_SUDOKU_CONSTRAINT_WEIGHT", "0.3"))
V200_SUDOKU_CALIB_WEIGHT       = float(os.environ.get("V200_SUDOKU_CALIB_WEIGHT", "0.1"))
V200_SUDOKU_BASE               = os.environ.get("V200_SUDOKU_BASE", "smollm2_1_7b").strip()


# ---------------------------------------------------------------------------
# Structured attention masks (32 heads, 10+10+10+2 partition)
# ---------------------------------------------------------------------------

def _build_sudoku_masks_v200(n_heads: int = 32) -> Tensor:
    """Return (n_heads, 81, 81) float additive bias Tensor.

    Mask 0.0 = allowed; -1e4 = blocked.  Self-attention always allowed.

    Head partition for n_heads=32 (2× the v98 16-head split):
      heads  0-9  (10): row constraints
      heads 10-19 (10): col constraints
      heads 20-29 (10): box constraints
      heads 30-31  (2): global (full 81×81)

    For other head counts, the formula scales proportionally and the
    remainder is assigned to global. Callers should use n_heads=32 for SmolLM2.
    """
    rows_idx  = np.array([i // 9 for i in range(81)])
    cols_idx  = np.array([i %  9 for i in range(81)])
    boxes_idx = np.array([(i // 9 // 3) * 3 + (i % 9 // 3) for i in range(81)])

    same_row = (rows_idx[:, None] == rows_idx[None, :]).astype(np.float32)
    same_col = (cols_idx[:, None] == cols_idx[None, :]).astype(np.float32)
    same_box = (boxes_idx[:, None] == boxes_idx[None, :]).astype(np.float32)
    eye      = np.eye(81, dtype=np.float32)
    full     = np.ones((81, 81), dtype=np.float32)

    # 10+10+10+2 split for 32 heads
    n_row    = max(1, n_heads * 10 // 32)
    n_col    = max(1, n_heads * 10 // 32)
    n_box    = max(1, n_heads * 10 // 32)
    n_global = max(1, n_heads - n_row - n_col - n_box)
    assigned = n_row + n_col + n_box + n_global
    if assigned != n_heads:
        n_global += (n_heads - assigned)

    mask_np = np.zeros((n_heads, 81, 81), dtype=np.float32)
    h = 0
    for _ in range(n_row):
        mask_np[h] = np.maximum(same_row, eye)
        h += 1
    for _ in range(n_col):
        mask_np[h] = np.maximum(same_col, eye)
        h += 1
    for _ in range(n_box):
        mask_np[h] = np.maximum(same_box, eye)
        h += 1
    for _ in range(n_global):
        mask_np[h] = full
        h += 1

    # Convert to additive bias: 1.0→0.0, 0.0→-1e4
    bias_np = (1.0 - mask_np) * (-1e4)
    return Tensor(bias_np, dtype=dtypes.float).contiguous()


# ---------------------------------------------------------------------------
# Position features (same structural prior as v98, scaled to 2048d)
# ---------------------------------------------------------------------------

def _build_position_features_v200(hidden: int) -> Tensor:
    """(81, hidden) position embedding with row/col/box structural priors.

    Layout:
      [0..8]   = row one-hot
      [9..17]  = col one-hot
      [18..26] = box one-hot
      [27..]   = randn(0.02) learned tail
    """
    pos = np.zeros((81, hidden), dtype=np.float32)
    for i in range(81):
        r, c = i // 9, i % 9
        b = (r // 3) * 3 + (c // 3)
        if hidden >= 9:
            pos[i, r] = 1.0
        if hidden >= 18:
            pos[i, 9 + c] = 1.0
        if hidden >= 27:
            pos[i, 18 + b] = 1.0
    rng = np.random.RandomState(200)   # seed 200 for v200 variant
    if hidden > 27:
        pos[:, 27:] = rng.randn(81, hidden - 27).astype(np.float32) * 0.02
    return Tensor(pos, dtype=dtypes.float).contiguous()


# ---------------------------------------------------------------------------
# Cell embedding
# ---------------------------------------------------------------------------

def embed_sudoku_v200(input_cells: Tensor,
                      state_embed: Tensor,
                      position_embed: Tensor) -> Tensor:
    """Convert (B, 81) int cell states → (B, 81, H) embedding.

    state_embed:    (10, H) — 0=unknown, 1..9=given digits
    position_embed: (81, H) — structural position priors

    Returns sum of state + position (same as v98).
    """
    B = int(input_cells.shape[0])
    one_hot = input_cells.one_hot(10).cast(state_embed.dtype)    # (B, 81, 10)
    state   = one_hot @ state_embed                               # (B, 81, H)
    pos = position_embed.reshape(1, 81, -1).cast(state.dtype).expand(B, 81, -1)
    return state + pos


# ---------------------------------------------------------------------------
# Per-breath Q rotation (π-cycled, same as v109pi but on LlamaLayer Q)
# ---------------------------------------------------------------------------

def _rotate_q_pi_v200(q: Tensor, cos_k: float, sin_k: float) -> Tensor:
    """Pairwise rotation of Q by a single (cos, sin) across head_dim.

    q shape: (B, n_heads, S, head_dim)
    Applied AFTER SmolLM2's standard RoPE so both mechanisms are additive
    in angle-space. At k=0: cos=1, sin=0 → identity (safe warm-start).

    The rotation is:
      q_rot[..., 2i]   = cos · q[..., 2i]   - sin · q[..., 2i+1]
      q_rot[..., 2i+1] = sin · q[..., 2i]   + cos · q[..., 2i+1]
    """
    hd = q.shape[-1]
    assert hd % 2 == 0, f"head_dim must be even, got {hd}"
    n_pairs = hd // 2

    q_pairs = q.reshape(*q.shape[:-1], n_pairs, 2)
    q_even  = q_pairs[..., 0]
    q_odd   = q_pairs[..., 1]

    cos_t = Tensor([cos_k], dtype=q.dtype).reshape(*([1] * (q.ndim - 1)))
    sin_t = Tensor([sin_k], dtype=q.dtype).reshape(*([1] * (q.ndim - 1)))

    new_even = cos_t * q_even - sin_t * q_odd
    new_odd  = sin_t * q_even + cos_t * q_odd

    out_pairs = new_even.unsqueeze(-1).cat(new_odd.unsqueeze(-1), dim=-1)
    return out_pairs.reshape(*q.shape)


# ---------------------------------------------------------------------------
# Pythia-410M single-layer forward with sudoku mask + π rotation
# (parallel-residual structure: y = x + Attn(in_LN(x)) + FFN(post_LN(x)))
# ---------------------------------------------------------------------------

def sudoku_pythia_layer_forward_v200(
    layer: Any,
    x: Tensor,
    attn_bias: Tensor,
    q_rot_cos: float = 1.0,
    q_rot_sin: float = 0.0,
) -> Tensor:
    """Run one PythiaLayer forward with sudoku-style bidirectional attention.

    Differences from PythiaLayer.__call__:
      - attn_bias (n_heads, 81, 81) injected as additive pre-softmax bias.
      - No causal mask (constraint propagation is bidirectional).
      - No RoPE (cells addressed by learned position embedding, not token position).
      - π-cycled Q rotation applied directly to Q.
      - Parallel-residual: two separate LN inputs (in_ln, post_ln); outputs summed.

    layer:     a PythiaLayer with attributes:
                 wq, bq, wk, bk, wv, bv — (H, H) / (H,) attention weights+biases
                 wo, bo                  — output projection
                 w_in, b_in, w_out, b_out — FFN weights
                 in_ln_g, in_ln_b        — input LayerNorm (for attention branch)
                 post_ln_g, post_ln_b    — post-attn LayerNorm (for FFN branch)
                 cfg                     — Config with n_heads, head_dim, ffn, layer_norm_eps
    x:         (B, 81, H) fp32 residual stream
    attn_bias: (n_heads, 81, 81) additive bias (precomputed, frozen)
    q_rot_cos: scalar float — per-breath π rotation cosine
    q_rot_sin: scalar float — per-breath π rotation sine

    Returns: (B, 81, H) updated residual stream.
    """
    from mycelium.breathing import _layernorm

    cfg = layer.cfg
    B, S, H = x.shape
    nh  = cfg.n_heads
    hd  = cfg.head_dim

    # Two separate LayerNorm inputs (Pythia parallel-residual)
    attn_in = _layernorm(x, layer.in_ln_g, layer.in_ln_b, cfg.layer_norm_eps).cast(x.dtype)
    mlp_in  = _layernorm(x, layer.post_ln_g, layer.post_ln_b, cfg.layer_norm_eps).cast(x.dtype)

    # Q, K, V projections with biases (Pythia has per-layer Q/K biases)
    q = (attn_in @ layer.wq + layer.bq).reshape(B, S, nh, hd).transpose(1, 2)  # (B, nh, 81, hd)
    k = (attn_in @ layer.wk + layer.bk).reshape(B, S, nh, hd).transpose(1, 2)
    v = (attn_in @ layer.wv + layer.bv).reshape(B, S, nh, hd).transpose(1, 2)

    # NO standard RoPE (sudoku cells addressed by learned position embedding)
    # π-cycled per-breath Q rotation (the only angular signal per breath)
    if not (abs(q_rot_cos - 1.0) < 1e-9 and abs(q_rot_sin) < 1e-9):
        q = _rotate_q_pi_v200(q, q_rot_cos, q_rot_sin)

    scale  = 1.0 / math.sqrt(hd)
    scores = (q @ k.transpose(-2, -1)) * scale                            # (B, nh, 81, 81)

    # Structured attention bias (broadcastable: (1, nh, 81, 81))
    scores = scores + attn_bias.cast(scores.dtype).reshape(1, nh, S, S)
    attn   = scores.clip(-1e4, 1e4).softmax(-1).cast(v.dtype)
    ctx    = (attn @ v).transpose(1, 2).reshape(B, S, H)
    attn_out = ctx @ layer.wo + layer.bo

    # GELU FFN
    ff      = (mlp_in @ layer.w_in + layer.b_in).gelu()
    ffn_out = ff @ layer.w_out + layer.b_out

    # Parallel-residual: both branches add to x simultaneously
    return x + attn_out + ffn_out


# ---------------------------------------------------------------------------
# SmolLM2 single-layer forward with sudoku mask + π rotation (no causal, no KV cache)
# ---------------------------------------------------------------------------

def sudoku_layer_forward_v200(
    layer: LlamaLayer,
    x: Tensor,
    attn_bias: Tensor,
    rope_cos: Tensor,
    rope_sin: Tensor,
    q_rot_cos: float = 1.0,
    q_rot_sin: float = 0.0,
) -> Tensor:
    """Run one LlamaLayer forward with sudoku-style bidirectional attention.

    Differences from LlamaLayer.__call__:
      - attn_bias (n_heads, 81, 81) injected as additive pre-softmax bias
        (the structured constraint mask from _build_sudoku_masks_v200).
      - No causal mask (constraint propagation is bidirectional).
      - π-cycled Q rotation applied after RoPE.
      - Sequence length S is always 81 (the 81 sudoku cells).
      - RoPE is applied with seq positions 0..80 (cells are "positionally
        indexed" by their cell index 0-80 in addition to the learned embed).

    x:         (B, 81, H) fp32 residual stream
    attn_bias: (n_heads, 81, 81) additive bias (precomputed, frozen)
    rope_cos:  (max_pos, head_dim) precomputed cosines
    rope_sin:  (max_pos, head_dim) precomputed sines
    q_rot_cos: scalar float — per-breath π rotation cosine
    q_rot_sin: scalar float — per-breath π rotation sine

    Returns: (B, 81, H) updated residual stream.
    """
    cfg = layer.cfg
    B, S, H = x.shape
    nh  = cfg.num_attention_heads
    hd  = cfg.head_dim
    nkv = cfg.num_key_value_heads

    # Pre-attention RMSNorm
    h = _rms_norm(x, layer.attn_norm, cfg.rms_norm_eps).cast(x.dtype)

    # Q, K, V projections (no bias in Llama)
    q = (h @ layer.wq.cast(x.dtype)).reshape(B, S, nh,  hd).transpose(1, 2)  # (B, nh, S, hd)
    k = (h @ layer.wk.cast(x.dtype)).reshape(B, S, nkv, hd).transpose(1, 2)  # (B, nkv, S, hd)
    v = (h @ layer.wv.cast(x.dtype)).reshape(B, S, nkv, hd).transpose(1, 2)  # (B, nkv, S, hd)

    # Apply SmolLM2's standard RoPE (theta=130000, positions 0..S-1)
    q, k = _apply_rope(q, k, rope_cos, rope_sin, S)

    # π-cycled per-breath Q rotation applied on top of RoPE
    if not (abs(q_rot_cos - 1.0) < 1e-9 and abs(q_rot_sin) < 1e-9):
        q = _rotate_q_pi_v200(q, q_rot_cos, q_rot_sin)

    # GQA repeat KV heads if needed (SmolLM2 has no GQA, n_rep=1; kept for correctness)
    if cfg.n_rep > 1:
        k = k.repeat((1, 1, cfg.n_rep, 1)).reshape(B, nh, S, hd)
        v = v.repeat((1, 1, cfg.n_rep, 1)).reshape(B, nh, S, hd)

    scale  = 1.0 / math.sqrt(hd)
    scores = (q @ k.transpose(-2, -1)) * scale                          # (B, nh, 81, 81)

    # Structured attention bias (broadcastable: (1, nh, 81, 81))
    scores = scores + attn_bias.cast(scores.dtype).reshape(1, nh, S, S)
    attn   = scores.clip(-1e4, 1e4).softmax(-1).cast(v.dtype)
    ctx    = (attn @ v).transpose(1, 2).reshape(B, S, H)
    attn_out = ctx @ layer.wo.cast(x.dtype)

    x = x + attn_out

    # Pre-FFN RMSNorm
    h2 = _rms_norm(x, layer.ffn_norm, cfg.rms_norm_eps).cast(x.dtype)

    # SwiGLU FFN
    gate    = (h2 @ layer.w_gate.cast(x.dtype)).silu()
    up      = (h2 @ layer.w_up.cast(x.dtype))
    ffn_out = (gate * up) @ layer.w_down.cast(x.dtype)

    return x + ffn_out


# ---------------------------------------------------------------------------
# Pythia-410M iterative prefill loop — K=20 breathing forward
# ---------------------------------------------------------------------------

def sudoku_breathing_forward_v200_pythia(
    model: Any,
    input_cells: Tensor,
    K: int = 20,
) -> tuple[list[Tensor], list[Tensor]]:
    """Run K breaths of constraint propagation (Pythia-410M backbone).

    Per-breath structure (each of K iterations):
      1. Add per-breath additive embedding
      2. Run 4 shared PythiaLayer passes with sudoku mask + π-cycled Q rotation
         (NO standard RoPE — matches v98 exactly)
      3. Apply learnable per-breath delta gate (convex residual blend)
      4. Readout: cell logits (B, 81, 9) via LayerNorm + digit codebook + calibration (B,)

    model must have the following attributes set by attach_sudoku_v200_pythia_params:
      pythia_layers              : list[PythiaLayer] (4 layers, L0-L3)
      pythia_ln_f_g              : (H,) final LayerNorm weight
      pythia_ln_f_b              : (H,) final LayerNorm bias
      pythia_cfg                 : Config (n_heads, head_dim, ffn, layer_norm_eps)
      sudoku_v200_state_embed    : (10, H)
      sudoku_v200_position_embed : (81, H)
      sudoku_v200_digit_codebook : (9, H)
      sudoku_v200_calib_head_w   : (H, 1)
      sudoku_v200_calib_head_b   : (1,)
      sudoku_v200_breath_embed   : (K_max, H)
      sudoku_v200_delta_gate     : (K_max,)
      sudoku_v200_attn_bias      : (n_heads, 81, 81) — frozen additive bias

    Returns:
      cell_logits_history : list of K (B, 81, 9) fp32 Tensors
      calib_history       : list of K (B,) fp32 Tensors (sigmoid'd confidence)
    """
    from mycelium.breathing import _layernorm

    assert hasattr(model, "pythia_layers"), \
        "model lacks pythia_layers — did you call attach_sudoku_v200_pythia_params()?"

    cfg    = model.pythia_cfg
    H      = cfg.hidden

    state_embed    = model.sudoku_v200_state_embed      # (10, H)
    position_embed = model.sudoku_v200_position_embed   # (81, H)
    attn_bias      = model.sudoku_v200_attn_bias        # (n_heads, 81, 81)
    breath_embed   = model.sudoku_v200_breath_embed     # (K_max, H)
    delta_gate     = model.sudoku_v200_delta_gate       # (K_max,)
    digit_codebook = model.sudoku_v200_digit_codebook   # (9, H)
    calib_head_w   = model.sudoku_v200_calib_head_w     # (H, 1)
    calib_head_b   = model.sudoku_v200_calib_head_b     # (1,)
    ln_f_g         = model.pythia_ln_f_g                # (H,)
    ln_f_b         = model.pythia_ln_f_b                # (H,)

    K_max = int(breath_embed.shape[0])
    assert K <= K_max, f"K={K} exceeds K_max={K_max}"

    layers = model.pythia_layers[:4]
    assert len(layers) == 4, f"expected 4 Pythia layers, got {len(layers)}"

    # Initial embedding — fp32 throughout
    x = embed_sudoku_v200(input_cells, state_embed, position_embed)
    x = x.cast(dtypes.float)

    # Precompute per-breath π rotation angles (constant floats — JIT-static)
    K_max_f = float(K_max)
    breath_cos = [math.cos(k * math.pi / K_max_f) for k in range(K_max)]
    breath_sin = [math.sin(k * math.pi / K_max_f) for k in range(K_max)]

    cell_logits_history: list[Tensor] = []
    calib_history:       list[Tensor] = []

    for k in range(K):
        # 1. Per-breath additive embedding
        be_k = breath_embed[k].reshape(1, 1, -1).cast(x.dtype)
        x_in = x + be_k

        # 2. Capture pre-layer state for gated delta
        x_pre = x

        # 3. 4 shared PythiaLayer passes with sudoku constraints + π rotation
        h = x_in
        cos_k = breath_cos[k]
        sin_k = breath_sin[k]
        for layer in layers:
            h = sudoku_pythia_layer_forward_v200(
                layer, h, attn_bias, cos_k, sin_k
            )

        # 4. Learnable per-breath delta gate (convex residual blend)
        gate_k = delta_gate[k].cast(h.dtype).reshape(1, 1, 1)
        delta  = h - x_pre
        x = x_pre + gate_k * delta

        # 5. Per-breath readout via LayerNorm + digit codebook
        x_ln = _layernorm(x, ln_f_g, ln_f_b, cfg.layer_norm_eps).cast(dtypes.float)
        cell_logits_k = x_ln @ digit_codebook.T.cast(dtypes.float)        # (B, 81, 9)
        cell_logits_history.append(cell_logits_k)

        # 6. Calibration: mean-pool 81 cells → scalar sigmoid
        pool = x_ln.mean(axis=1)                                           # (B, H)
        calib_logit = pool @ calib_head_w.cast(dtypes.float) + calib_head_b.cast(dtypes.float)
        calib_k = calib_logit.reshape(-1).sigmoid()                        # (B,)
        calib_history.append(calib_k)

    return cell_logits_history, calib_history


# ---------------------------------------------------------------------------
# Dispatcher: selects Pythia or SmolLM2 forward based on model.base_model attr
# ---------------------------------------------------------------------------

def sudoku_breathing_forward_v200_dispatch(
    model: Any,
    input_cells: Tensor,
    K: int = 20,
) -> tuple[list[Tensor], list[Tensor]]:
    """Dispatch to the correct backbone forward based on model.base_model."""
    if getattr(model, "base_model", "smollm2_1_7b") == "pythia_410m":
        return sudoku_breathing_forward_v200_pythia(model, input_cells, K=K)
    return sudoku_breathing_forward_v200(model, input_cells, K=K)


# ---------------------------------------------------------------------------
# SmolLM2-1.7B iterative prefill loop — K=20 breathing forward
# ---------------------------------------------------------------------------

def sudoku_breathing_forward_v200(
    model: Any,
    input_cells: Tensor,
    K: int = 20,
) -> tuple[list[Tensor], list[Tensor]]:
    """Run K breaths of constraint propagation on (B, 81) input cells.

    Per-breath structure (each of K iterations):
      1. Add per-breath additive embedding (tells model "I'm on breath k")
      2. Run 4 shared LlamaLayer passes with sudoku mask + π-cycled Q rotation
      3. Apply learnable per-breath delta gate (convex residual blend)
      4. Readout: cell logits (B, 81, 9) via digit codebook + calibration (B,)

    model must have the following attributes set by attach_sudoku_v200_params:
      llama_layers        : list[LlamaLayer] (4 layers, L0-L3)
      llama_rope_cos      : (max_pos, head_dim) Tensor
      llama_rope_sin      : (max_pos, head_dim) Tensor
      sudoku_v200_state_embed    : (10, H)
      sudoku_v200_position_embed : (81, H)
      sudoku_v200_digit_codebook : (9, H)
      sudoku_v200_calib_head_w   : (H, 1)
      sudoku_v200_calib_head_b   : (1,)
      sudoku_v200_breath_embed   : (K_max, H)
      sudoku_v200_delta_gate     : (K_max,)
      sudoku_v200_attn_bias      : (n_heads, 81, 81) — frozen additive bias
      sudoku_v200_final_norm     : (H,)              — RMSNorm weight for readout

    Returns:
      cell_logits_history : list of K (B, 81, 9) fp32 Tensors
      calib_history       : list of K (B,) fp32 Tensors (sigmoid'd confidence)
    """
    assert hasattr(model, "sudoku_v200_state_embed"), \
        "model lacks sudoku_v200 params — did you call attach_sudoku_v200_params()?"

    cfg   = model.llama_cfg
    H     = cfg.hidden_size
    n_heads = cfg.num_attention_heads
    head_dim = cfg.head_dim
    rms_eps  = cfg.rms_norm_eps

    state_embed    = model.sudoku_v200_state_embed      # (10, H)
    position_embed = model.sudoku_v200_position_embed   # (81, H)
    attn_bias      = model.sudoku_v200_attn_bias        # (n_heads, 81, 81)
    breath_embed   = model.sudoku_v200_breath_embed     # (K_max, H)
    delta_gate     = model.sudoku_v200_delta_gate       # (K_max,)
    digit_codebook = model.sudoku_v200_digit_codebook   # (9, H)
    calib_head_w   = model.sudoku_v200_calib_head_w     # (H, 1)
    calib_head_b   = model.sudoku_v200_calib_head_b     # (1,)
    final_norm_w   = model.sudoku_v200_final_norm       # (H,)
    rope_cos       = model.llama_rope_cos               # (max_pos, head_dim)
    rope_sin       = model.llama_rope_sin               # (max_pos, head_dim)

    K_max = int(breath_embed.shape[0])
    assert K <= K_max, f"K={K} exceeds K_max={K_max}"

    layers = model.llama_layers[:4]
    assert len(layers) == 4, f"expected 4 Llama layers, got {len(layers)}"

    # Initial embedding — fp32 throughout (LlamaLayer runs fp32 internally)
    x = embed_sudoku_v200(input_cells, state_embed, position_embed)
    x = x.cast(dtypes.float)

    # Precompute per-breath π rotation angles (constant floats — won't appear
    # in the JIT graph's dynamic shape, which keeps the K=20 unrolled graph static)
    K_max_f = float(K_max)
    breath_cos = [math.cos(k * math.pi / K_max_f) for k in range(K_max)]
    breath_sin = [math.sin(k * math.pi / K_max_f) for k in range(K_max)]

    cell_logits_history: list[Tensor] = []
    calib_history:       list[Tensor] = []

    for k in range(K):
        # 1. Per-breath additive embedding
        be_k = breath_embed[k].reshape(1, 1, -1).cast(x.dtype)
        x_in = x + be_k

        # 2. Capture pre-layer state for gated delta
        x_pre = x

        # 3. 4 shared LlamaLayer passes with sudoku constraints + π rotation
        h = x_in
        cos_k = breath_cos[k]
        sin_k = breath_sin[k]
        for layer in layers:
            h = sudoku_layer_forward_v200(
                layer, h, attn_bias, rope_cos, rope_sin, cos_k, sin_k
            )

        # 4. Learnable per-breath delta gate (convex residual blend)
        gate_k = delta_gate[k].cast(h.dtype).reshape(1, 1, 1)
        delta  = h - x_pre
        x = x_pre + gate_k * delta

        # 5. Per-breath readout via RMSNorm + digit codebook
        x_normed = _rms_norm(x, final_norm_w, rms_eps).cast(dtypes.float)
        cell_logits_k = x_normed @ digit_codebook.T.cast(dtypes.float)   # (B, 81, 9)
        cell_logits_history.append(cell_logits_k)

        # 6. Calibration: mean-pool 81 cells → scalar sigmoid
        pool = x_normed.mean(axis=1)                                      # (B, H)
        calib_logit = pool @ calib_head_w.cast(dtypes.float) + calib_head_b.cast(dtypes.float)
        calib_k = calib_logit.reshape(-1).sigmoid()                       # (B,)
        calib_history.append(calib_k)

    return cell_logits_history, calib_history


# ---------------------------------------------------------------------------
# Loss functions (identical to v98; copied here to avoid v98 import dependency)
# ---------------------------------------------------------------------------

def sudoku_constraint_energy_v200(probs: Tensor) -> Tensor:
    """Soft row/col/box AllDifferent violation energy. probs: (B, 81, 9) → (B,)."""
    B = int(probs.shape[0])
    probs_grid = probs.reshape(B, 9, 9, 9)

    row_sums = probs_grid.sum(axis=2)
    row_violation = ((row_sums - 1.0) ** 2).sum(axis=(1, 2))

    col_sums = probs_grid.sum(axis=1)
    col_violation = ((col_sums - 1.0) ** 2).sum(axis=(1, 2))

    probs_box = probs_grid.reshape(B, 3, 3, 3, 3, 9)
    probs_box_perm = probs_box.permute(0, 1, 3, 2, 4, 5)
    box_sums = probs_box_perm.reshape(B, 9, 9, 9).sum(axis=2)
    box_violation = ((box_sums - 1.0) ** 2).sum(axis=(1, 2))

    return row_violation + col_violation + box_violation


def sudoku_accuracy_v200(
    cell_logits_final: Tensor,
    gold_solution: Tensor,
) -> tuple[float, float]:
    """Return (cell_acc, puzzle_acc) on a batch.

    cell_logits_final: (B, 81, 9)
    gold_solution:     (B, 81) digits 1..9
    """
    pred = cell_logits_final.argmax(axis=-1) + 1
    eq   = (pred == gold_solution).cast(dtypes.float)
    cell_acc   = float(eq.mean().realize().numpy())
    puzzle_acc = float(eq.prod(axis=-1).mean().realize().numpy())
    return cell_acc, puzzle_acc


# ---------------------------------------------------------------------------
# Model param attachment
# ---------------------------------------------------------------------------

def attach_sudoku_v200_params(
    model: Any,
    n_heads: int = 32,
    k_max: int | None = None,
) -> None:
    """Allocate sudoku-v200-specific params on `model`.

    Assumes model.llama_layers (list of 4 LlamaLayer) and model.llama_cfg
    are already set by attach_llama_layers(). This function only adds the
    sudoku-specific tensors that sit on top of the shared backbone.

    Attributes added:
      sudoku_v200_state_embed    : (10, H)
      sudoku_v200_position_embed : (81, H)
      sudoku_v200_digit_codebook : (9, H)
      sudoku_v200_calib_head_w   : (H, 1)
      sudoku_v200_calib_head_b   : (1,)
      sudoku_v200_breath_embed   : (K_max, H)
      sudoku_v200_delta_gate     : (K_max,)
      sudoku_v200_attn_bias      : (n_heads, 81, 81) — frozen
      sudoku_v200_final_norm     : (H,) — RMSNorm weight for readout (trained)
    """
    if k_max is None:
        k_max = V200_SUDOKU_K_MAX

    cfg = model.llama_cfg
    H   = cfg.hidden_size
    rms_eps = cfg.rms_norm_eps

    # Position embedding (structural priors + learned tail)
    model.sudoku_v200_position_embed = _build_position_features_v200(H)

    # Digit codebook — orthonormal rows × 0.1 scale (same aligned-init as v98)
    rng_cb  = np.random.RandomState(20098)
    raw_cb  = rng_cb.randn(max(H, 9), H).astype(np.float32)
    q_cb, _ = np.linalg.qr(raw_cb)
    cb_unit = q_cb[:9].astype(np.float32)
    model.sudoku_v200_digit_codebook = Tensor(
        cb_unit * 0.1, dtype=dtypes.float
    ).contiguous()

    # State embedding — 10 rows (0=unknown, 1..9=given digit)
    # rows 1..9 aligned with codebook at scale 1.0 → immediate strong logit for given digits
    state_np = np.zeros((10, H), dtype=np.float32)
    state_np[0] = np.random.RandomState(20097).randn(H).astype(np.float32) * 0.02
    state_np[1:10] = cb_unit  # scale 1.0 vs codebook 0.1 → 10× stronger logit for given cells
    model.sudoku_v200_state_embed = Tensor(state_np, dtype=dtypes.float).contiguous()

    # Calibration head (small randn weight, zero bias)
    cw = (np.random.RandomState(20099).randn(H, 1) * 0.02).astype(np.float32)
    model.sudoku_v200_calib_head_w = Tensor(cw, dtype=dtypes.float).contiguous()
    model.sudoku_v200_calib_head_b = Tensor.zeros((1,), dtype=dtypes.float).contiguous()

    # Per-breath additive embedding — QR-orthonormal, scale 0.5 (v98 default)
    breath_scale = float(os.environ.get("V200_SUDOKU_BREATH_EMBED_SCALE", "0.5"))
    rng_be = np.random.RandomState(20100)
    raw_be = rng_be.randn(max(k_max, H), H).astype(np.float32)
    q_be, _ = np.linalg.qr(raw_be)
    be = q_be[:k_max].astype(np.float32) * breath_scale
    model.sudoku_v200_breath_embed = Tensor(be, dtype=dtypes.float).contiguous()

    # Per-breath delta gate (init 1.0 — identity residual blend at step 0)
    model.sudoku_v200_delta_gate = Tensor.ones((k_max,), dtype=dtypes.float).contiguous()

    # Structured attention bias (FROZEN — structural inductive bias, not trained)
    model.sudoku_v200_attn_bias = _build_sudoku_masks_v200(n_heads).contiguous()

    # Final RMSNorm for readout (init ones — same as SmolLM2 norm init)
    model.sudoku_v200_final_norm = Tensor.ones((H,), dtype=dtypes.float).contiguous()

    n_sudoku = (
        10 * H +       # state_embed
        81 * H +       # position_embed
        9  * H +       # digit_codebook
        H  + 1 +       # calib_head_w + b
        k_max * H +    # breath_embed
        k_max +        # delta_gate
        H              # final_norm
    )
    print(
        f"[sudoku_v200] attached sudoku params: H={H} K_max={k_max} n_heads={n_heads}\n"
        f"  mask partition: 10 row + 10 col + 10 box + 2 global (= {n_heads} heads)\n"
        f"  sudoku-specific params: {n_sudoku/1e6:.3f}M",
        flush=True,
    )


def sudoku_v200_parameters(model: Any) -> list[Tensor]:
    """Trainable sudoku-v200 params (excludes the frozen attn_bias).

    Called by collect_sudoku_v200_params() which adds these to the backbone
    params for the AdamW optimizer.
    """
    return [
        model.sudoku_v200_state_embed,
        model.sudoku_v200_position_embed,
        model.sudoku_v200_digit_codebook,
        model.sudoku_v200_calib_head_w,
        model.sudoku_v200_calib_head_b,
        model.sudoku_v200_breath_embed,
        model.sudoku_v200_delta_gate,
        model.sudoku_v200_final_norm,
    ]


def sudoku_v200_state_dict(model: Any) -> dict[str, Tensor]:
    """State dict entries for sudoku_v200 params (excluding frozen attn_bias)."""
    return {
        "sudoku_v200.state_embed":    model.sudoku_v200_state_embed,
        "sudoku_v200.position_embed": model.sudoku_v200_position_embed,
        "sudoku_v200.digit_codebook": model.sudoku_v200_digit_codebook,
        "sudoku_v200.calib_head_w":   model.sudoku_v200_calib_head_w,
        "sudoku_v200.calib_head_b":   model.sudoku_v200_calib_head_b,
        "sudoku_v200.breath_embed":   model.sudoku_v200_breath_embed,
        "sudoku_v200.delta_gate":     model.sudoku_v200_delta_gate,
        "sudoku_v200.final_norm":     model.sudoku_v200_final_norm,
    }


# ---------------------------------------------------------------------------
# Pythia-410M param attachment
# ---------------------------------------------------------------------------

def attach_sudoku_v200_pythia_params(
    model: Any,
    n_heads: int = 16,
    k_max: int | None = None,
) -> None:
    """Allocate sudoku-v200 Pythia-specific params on `model`.

    Assumes model.pythia_layers (list of 4 PythiaLayer), model.pythia_ln_f_g,
    model.pythia_ln_f_b, and model.pythia_cfg are already set by
    _attach_pythia_layers_v200(). This function only adds the sudoku-specific
    tensors on top of the shared backbone.

    The readout uses the Pythia final LayerNorm (pythia_ln_f_g / pythia_ln_f_b)
    directly — no separate sudoku_v200_final_norm is needed for the Pythia path.
    sudoku_v200_final_norm is still allocated as a unit vector to keep
    sudoku_v200_parameters() / state_dict() compatible across both paths, but
    it is NOT used in the Pythia forward path.

    Attributes added:
      sudoku_v200_state_embed    : (10, H)
      sudoku_v200_position_embed : (81, H)
      sudoku_v200_digit_codebook : (9, H)
      sudoku_v200_calib_head_w   : (H, 1)
      sudoku_v200_calib_head_b   : (1,)
      sudoku_v200_breath_embed   : (K_max, H)
      sudoku_v200_delta_gate     : (K_max,)
      sudoku_v200_attn_bias      : (n_heads, 81, 81) — frozen
      sudoku_v200_final_norm     : (H,) — stub (init ones, not used in forward)
    """
    if k_max is None:
        k_max = V200_SUDOKU_K_MAX

    cfg = model.pythia_cfg
    H   = cfg.hidden

    # Position embedding (structural priors + learned tail)
    model.sudoku_v200_position_embed = _build_position_features_v200(H)

    # Digit codebook — orthonormal rows × 0.1 scale (same aligned-init as v98)
    rng_cb  = np.random.RandomState(20098)
    raw_cb  = rng_cb.randn(max(H, 9), H).astype(np.float32)
    q_cb, _ = np.linalg.qr(raw_cb)
    cb_unit = q_cb[:9].astype(np.float32)
    model.sudoku_v200_digit_codebook = Tensor(
        cb_unit * 0.1, dtype=dtypes.float
    ).contiguous()

    # State embedding — 10 rows (0=unknown, 1..9=given digit), aligned with codebook
    state_np = np.zeros((10, H), dtype=np.float32)
    state_np[0] = np.random.RandomState(20097).randn(H).astype(np.float32) * 0.02
    state_np[1:10] = cb_unit
    model.sudoku_v200_state_embed = Tensor(state_np, dtype=dtypes.float).contiguous()

    # Calibration head (small randn weight, zero bias)
    cw = (np.random.RandomState(20099).randn(H, 1) * 0.02).astype(np.float32)
    model.sudoku_v200_calib_head_w = Tensor(cw, dtype=dtypes.float).contiguous()
    model.sudoku_v200_calib_head_b = Tensor.zeros((1,), dtype=dtypes.float).contiguous()

    # Per-breath additive embedding — QR-orthonormal, scale 0.5
    breath_scale = float(os.environ.get("V200_SUDOKU_BREATH_EMBED_SCALE", "0.5"))
    rng_be = np.random.RandomState(20100)
    raw_be = rng_be.randn(max(k_max, H), H).astype(np.float32)
    q_be, _ = np.linalg.qr(raw_be)
    be = q_be[:k_max].astype(np.float32) * breath_scale
    model.sudoku_v200_breath_embed = Tensor(be, dtype=dtypes.float).contiguous()

    # Per-breath delta gate (init 1.0 — identity residual blend at step 0)
    model.sudoku_v200_delta_gate = Tensor.ones((k_max,), dtype=dtypes.float).contiguous()

    # Structured attention bias (FROZEN — 5+5+5+1 = 16 heads for Pythia)
    model.sudoku_v200_attn_bias = _build_sudoku_masks_v200(n_heads).contiguous()

    # Stub final_norm (not used in Pythia forward; kept for state_dict compat)
    model.sudoku_v200_final_norm = Tensor.ones((H,), dtype=dtypes.float).contiguous()

    n_sudoku = (
        10 * H +       # state_embed
        81 * H +       # position_embed
        9  * H +       # digit_codebook
        H  + 1 +       # calib_head_w + b
        k_max * H +    # breath_embed
        k_max +        # delta_gate
        H              # final_norm stub
    )
    n_row = max(1, n_heads * 5 // 16)
    n_col = n_row
    n_box = n_row
    n_global = n_heads - n_row - n_col - n_box
    print(
        f"[sudoku_v200_pythia] attached sudoku params: H={H} K_max={k_max} n_heads={n_heads}\n"
        f"  mask partition: {n_row} row + {n_col} col + {n_box} box + {n_global} global"
        f" (= {n_heads} heads)\n"
        f"  sudoku-specific params: {n_sudoku/1e6:.3f}M",
        flush=True,
    )


def _attach_pythia_layers_v200(model: Any, n_layers: int = 4) -> None:
    """Load Pythia-410M L0-L3 weights onto `model`.

    Sets:
      model.pythia_layers    : list of 4 PythiaLayer (loaded from .cache/pythia-410m)
      model.pythia_ln_f_g    : (H,) final LayerNorm weight
      model.pythia_ln_f_b    : (H,) final LayerNorm bias
      model.pythia_cfg       : Config
    """
    from mycelium.config import Config
    from mycelium.loader import load_pythia_baseline

    cfg = Config()
    stack = load_pythia_baseline(cfg, n_layers=n_layers)
    model.pythia_layers = stack.layers
    model.pythia_ln_f_g = stack.ln_f_g
    model.pythia_ln_f_b = stack.ln_f_b
    model.pythia_cfg    = cfg


# ---------------------------------------------------------------------------
# JIT'd training step
# ---------------------------------------------------------------------------

_JIT_V200_SUDOKU_CACHE: dict = {}


def _compile_jit_sudoku_step_v200(
    model: Any,
    opt: Any,
    K: int,
    B: int,
    constraint_weight: float,
    calib_weight: float,
    grad_clip: float = 0.0,
):
    """Compile and return a TinyJit'd train step for sudoku_v200.

    Inputs (stable shapes; pass realized Tensors):
      input_cells   : (B, 81) int  — cell states 0=unknown, 1..9=given
      gold_solution : (B, 81) int  — gold digits 1..9

    Returns tuple of realized scalars:
      total, healthy, cell_ce, energy, calib, train_cell_acc, train_puzzle_acc,
      per_breath_ce[0], ..., per_breath_ce[K-1]

    AMD/JIT safety:
      - .cast(dtypes.float) (not dtypes.float32) inside JIT graph
      - .clip(-1e4, 1e4) before softmax (in sudoku_layer_forward_v200)
      - single isfinite() check for NaN-skip (not per-param isnan())
    """
    key = (
        "v200_sudoku", id(model), id(opt),
        int(K), int(B),
        float(constraint_weight), float(calib_weight), float(grad_clip),
    )
    if key in _JIT_V200_SUDOKU_CACHE:
        return _JIT_V200_SUDOKU_CACHE[key]

    cw     = float(constraint_weight)
    aw     = float(calib_weight)
    gc_val = float(grad_clip)
    params = opt.params

    print(
        f"[JIT] compile sudoku_v200 step: K={K} B={B} cw={cw} aw={aw} clip={gc_val}…",
        flush=True,
    )

    @TinyJit
    def _step(input_cells: Tensor, gold_solution: Tensor):
        opt.zero_grad()

        cell_logits_history, calib_history = sudoku_breathing_forward_v200_dispatch(
            model, input_cells, K=K
        )

        gold_idx  = gold_solution - 1
        gold_flat = gold_idx.reshape(B * 81)

        # Per-breath weighted CE ladder (linear ramp 1.0 → 2.0)
        cell_loss_sum = Tensor.zeros((), dtype=dtypes.float).contiguous()
        weight_sum    = 0.0
        per_breath_ce_losses = []
        for k, logits in enumerate(cell_logits_history):
            weight_k = 1.0 + (float(k) / float(K - 1)) if K > 1 else 1.0
            ce_k = logits.reshape(B * 81, 9).sparse_categorical_crossentropy(
                gold_flat, reduction="mean"
            )
            per_breath_ce_losses.append(ce_k)
            cell_loss_sum = cell_loss_sum + ce_k * weight_k
            weight_sum   += weight_k
        cell_loss = cell_loss_sum / float(weight_sum)

        # Constraint energy on final breath
        final_probs   = cell_logits_history[-1].softmax(axis=-1)
        energy_per_b  = sudoku_constraint_energy_v200(final_probs)
        energy        = energy_per_b.mean()

        # Calibration (detached argmax target)
        final_argmax = (cell_logits_history[-1].argmax(axis=-1) + 1).detach()
        eq           = (final_argmax == gold_solution).cast(dtypes.float)
        correct      = eq.prod(axis=-1)

        calib_loss_sum = Tensor.zeros((), dtype=dtypes.float).contiguous()
        for k, calib_k in enumerate(calib_history):
            progression = float(k) / float(K - 1) if K > 1 else 1.0
            target_k    = 0.5 + (correct - 0.5) * progression
            calib_loss_sum = calib_loss_sum + ((calib_k - target_k) ** 2).mean()
        calib_loss = calib_loss_sum / float(K)

        train_cell_acc   = eq.mean().detach()
        train_puzzle_acc = eq.prod(axis=-1).mean().detach()

        total = cell_loss + cw * energy + aw * calib_loss
        total.backward()

        # NaN-skip: single isfinite() kernel (AMD JIT safe — no per-param isnan())
        healthy = total.isfinite().cast(dtypes.float)
        for p in params:
            if p.grad is not None:
                p.grad = p.grad * healthy.cast(p.grad.dtype)

        # Optional global grad norm clipping
        if gc_val > 0:
            sq_sum = Tensor.zeros((), dtype=dtypes.float).contiguous()
            for p in params:
                if p.grad is not None:
                    sq_sum = sq_sum + p.grad.cast(dtypes.float).square().sum()
            grad_norm  = (sq_sum + 1e-12).sqrt()
            clip_coef  = (Tensor(gc_val, dtype=dtypes.float) / (grad_norm + 1e-6))
            clip_coef  = clip_coef.minimum(Tensor(1.0, dtype=dtypes.float))
            for p in params:
                if p.grad is not None:
                    p.grad = p.grad * clip_coef.cast(p.grad.dtype)

        opt.step()

        return (
            total.realize(),
            healthy.realize(),
            cell_loss.realize(),
            energy.realize(),
            calib_loss.realize(),
            train_cell_acc.realize(),
            train_puzzle_acc.realize(),
            *(ce.realize() for ce in per_breath_ce_losses),
        )

    _JIT_V200_SUDOKU_CACHE[key] = _step
    print(
        f"[JIT] sudoku_v200 step ready (cache={len(_JIT_V200_SUDOKU_CACHE)}); "
        f"first call will compile (K=20, expect ~5-10 min on AMD)…",
        flush=True,
    )
    return _step


def _compile_jit_sudoku_eval_v200(model: Any, K: int, B: int):
    """Compile a TinyJit'd eval step (forward-only, no backward).

    Returns (eq_mask (B,81) fp32, cell_acc scalar, puzzle_acc scalar).
    """
    key = ("v200_sudoku_eval", id(model), int(K), int(B))
    if key in _JIT_V200_SUDOKU_CACHE:
        return _JIT_V200_SUDOKU_CACHE[key]

    print(f"[JIT] compile sudoku_v200 eval: K={K} B={B}…", flush=True)

    @TinyJit
    def _eval(input_cells: Tensor, gold_solution: Tensor):
        cell_logits_history, _ = sudoku_breathing_forward_v200_dispatch(model, input_cells, K=K)
        final_logits = cell_logits_history[-1]
        pred         = final_logits.argmax(axis=-1) + 1
        eq           = (pred == gold_solution).cast(dtypes.float)
        cell_acc     = eq.mean()
        puzzle_acc   = eq.prod(axis=-1).mean()
        return eq.realize(), cell_acc.realize(), puzzle_acc.realize()

    _JIT_V200_SUDOKU_CACHE[key] = _eval
    print(f"[JIT] sudoku_v200 eval ready (cache={len(_JIT_V200_SUDOKU_CACHE)})", flush=True)
    return _eval


# ---------------------------------------------------------------------------
# Model wrapper — supports SmolLM2-1.7B and Pythia-410M backbones
# ---------------------------------------------------------------------------

class SudokuV200Model:
    """Minimal model shell that holds a backbone + sudoku params.

    Backbone is selected by `base_model` parameter (or V200_SUDOKU_BASE env var):
      "smollm2_1_7b" (default) — SmolLM2-1.7B L0-L3, H=2048, 32 heads
      "pythia_410m"             — Pythia-410M L0-L3, H=1024, 16 heads

    No dependency on BreathingTransformer. Weights are attached by the
    backbone-specific helper, then sudoku params are added on top.
    """

    def __init__(
        self,
        cfg: LlamaConfig | None = None,
        n_layers: int = 4,
        k_max: int | None = None,
        n_heads: int | None = None,
        base_model: str | None = None,
    ):
        self.base_model = (base_model or V200_SUDOKU_BASE).strip()
        if self.base_model not in ("smollm2_1_7b", "pythia_410m"):
            raise ValueError(
                f"Unknown V200_SUDOKU_BASE={self.base_model!r}. "
                "Choose 'smollm2_1_7b' or 'pythia_410m'."
            )
        self._n_layers = n_layers
        self._k_max = k_max or V200_SUDOKU_K_MAX

        if self.base_model == "pythia_410m":
            # Pythia-410M: H=1024, 16 heads, 5+5+5+1 mask partition
            self._n_heads_mask = n_heads if n_heads is not None else 16
            self.llama_cfg_arg = None  # not used
        else:
            # SmolLM2-1.7B: H=2048, 32 heads, 10+10+10+2 mask partition
            self.llama_cfg_arg = cfg or SMOLLM2_1_7B_CFG
            self._n_heads_mask = n_heads if n_heads is not None else 32

    def load(self, sd=None):
        """Load backbone weights and attach sudoku params."""
        if self.base_model == "pythia_410m":
            _attach_pythia_layers_v200(self, n_layers=self._n_layers)
            attach_sudoku_v200_pythia_params(
                self, n_heads=self._n_heads_mask, k_max=self._k_max
            )
        else:
            attach_llama_layers(
                self,
                n_layers=self._n_layers,
                sd=sd,
                cfg=self.llama_cfg_arg,
                layer_offset=0,
            )
            attach_sudoku_v200_params(self, n_heads=self._n_heads_mask, k_max=self._k_max)
        return self

    def parameters(self) -> list[Tensor]:
        """All trainable parameters: backbone layers + sudoku-specific."""
        params: list[Tensor] = []
        if self.base_model == "pythia_410m":
            for layer in self.pythia_layers:
                params.extend(layer.parameters())
            # Include Pythia final LN (used for readout)
            params.append(self.pythia_ln_f_g)
            params.append(self.pythia_ln_f_b)
            # Sudoku-specific params — EXCLUDE final_norm stub (not used in Pythia forward;
            # Pythia readout uses pythia_ln_f_g/b directly).
            params.extend([
                self.sudoku_v200_state_embed,
                self.sudoku_v200_position_embed,
                self.sudoku_v200_digit_codebook,
                self.sudoku_v200_calib_head_w,
                self.sudoku_v200_calib_head_b,
                self.sudoku_v200_breath_embed,
                self.sudoku_v200_delta_gate,
            ])
        else:
            for layer in self.llama_layers:
                params.extend(layer.parameters())
            params.extend(sudoku_v200_parameters(self))
        return params

    def state_dict(self) -> dict[str, Tensor]:
        """Compact state dict: backbone L0-L3 weights + sudoku params."""
        sd: dict[str, Tensor] = {}
        if self.base_model == "pythia_410m":
            for i, layer in enumerate(self.pythia_layers):
                prefix = f"pythia.layers.{i}"
                sd[f"{prefix}.wq"]       = layer.wq
                sd[f"{prefix}.bq"]       = layer.bq
                sd[f"{prefix}.wk"]       = layer.wk
                sd[f"{prefix}.bk"]       = layer.bk
                sd[f"{prefix}.wv"]       = layer.wv
                sd[f"{prefix}.bv"]       = layer.bv
                sd[f"{prefix}.wo"]       = layer.wo
                sd[f"{prefix}.bo"]       = layer.bo
                sd[f"{prefix}.w_in"]     = layer.w_in
                sd[f"{prefix}.b_in"]     = layer.b_in
                sd[f"{prefix}.w_out"]    = layer.w_out
                sd[f"{prefix}.b_out"]    = layer.b_out
                sd[f"{prefix}.in_ln_g"]  = layer.in_ln_g
                sd[f"{prefix}.in_ln_b"]  = layer.in_ln_b
                sd[f"{prefix}.post_ln_g"] = layer.post_ln_g
                sd[f"{prefix}.post_ln_b"] = layer.post_ln_b
            sd["pythia.ln_f_g"] = self.pythia_ln_f_g
            sd["pythia.ln_f_b"] = self.pythia_ln_f_b
        else:
            for i, layer in enumerate(self.llama_layers):
                prefix = f"llama.layers.{i}"
                sd[f"{prefix}.wq"]        = layer.wq
                sd[f"{prefix}.wk"]        = layer.wk
                sd[f"{prefix}.wv"]        = layer.wv
                sd[f"{prefix}.wo"]        = layer.wo
                sd[f"{prefix}.w_gate"]    = layer.w_gate
                sd[f"{prefix}.w_up"]      = layer.w_up
                sd[f"{prefix}.w_down"]    = layer.w_down
                sd[f"{prefix}.attn_norm"] = layer.attn_norm
                sd[f"{prefix}.ffn_norm"]  = layer.ffn_norm
        sd.update(sudoku_v200_state_dict(self))
        return sd

    def load_ckpt(self, path: str) -> None:
        """Resume from a v200-sudoku checkpoint (partial-row safe for breath params)."""
        from tinygrad.nn.state import safe_load
        raw = safe_load(path)
        targets = self.state_dict()
        missing, partial = [], []
        for name, dst in targets.items():
            if name not in raw:
                missing.append(name)
                continue
            src = raw[name].to(dst.device).realize()
            if src.shape != dst.shape:
                # Partial-row copy for per-breath tensors (K-expansion safe)
                if (name in ("sudoku_v200.breath_embed", "sudoku_v200.delta_gate")
                        and src.ndim == dst.ndim
                        and (src.ndim == 1 or src.shape[1:] == dst.shape[1:])
                        and src.shape[0] <= dst.shape[0]):
                    k_old = int(src.shape[0])
                    cur = dst.numpy()
                    cur[:k_old] = src.cast(dst.dtype).numpy()[:k_old]
                    dst.assign(Tensor(cur, dtype=dst.dtype).contiguous()).realize()
                    partial.append(f"{name} ({k_old}/{dst.shape[0]} rows)")
                    continue
                try:
                    src = src.reshape(dst.shape)
                except Exception:
                    missing.append(f"{name}(shape mismatch)")
                    continue
            if src.dtype != dst.dtype:
                src = src.cast(dst.dtype)
            dst.assign(src).realize()
        if missing:
            print(f"  ckpt missing {len(missing)} keys: {missing[:5]}")
        if partial:
            print(f"  ckpt partial-row load: {partial}")
