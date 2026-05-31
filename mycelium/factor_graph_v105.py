"""v105 factor graph breathing transformer — digit-by-digit codebook architecture.

Numbers are factor graphs too: the value 1234 is a 4-digit tree with place-value
weights (1000, 100, 10, 1).  Instead of a 100-way domain codebook (values 0-99),
v105 uses a 10-way digit codebook (0-9) shared across all digit positions.

Key architectural differences from v104:
  - No domain codebook (100-way).  Replaced by:
      digit_codebook     : (10, H) — one row per digit 0-9, shared across positions
      digit_pos_embed    : (N_DIGITS, H) — learnable per-place-value embeddings
  - Sequence layout:
      T = N_VARS × N_DIGITS + N_FACTORS
      positions 0 .. N_VARS*N_DIGITS-1  : variable digit positions
      positions N_VARS*N_DIGITS ..       : factor nodes (unchanged)
  - Variable embedding:
      embed[v, p] = digit_codebook[digit_value] + digit_pos_embed[p] + var_pos_embed[v]
  - Output decoding:
      digit_logits[v, p] = hidden[v*N_DIGITS+p] @ digit_codebook.T  →  (10,)
      predicted_value[v] = Σ_p argmax(digit_logits[v, p]) × 10^(N-1-p)
  - Constraint energy (expected-value form, differentiable):
      E_factor = |E[result] - op(E[arg1], E[arg2])|²
      where E[v] = Σ_p (Σ_d d * softmax(digit_logits[v,p])[d]) * 10^(N-1-p)
  - MUL/DIV handled via expected-value approximation (no digit-by-digit convolution).

Attention masks:
  - Within a variable: all N_DIGITS positions of variable v attend to each other
    (full within-variable block).
  - Cross variable↔factor: all N_DIGITS positions of variable v attend to factor f
    IFF v is in f's arg list; factor attends to all digit positions of its args.
  - Per-head op specialization: same 4-group scheme as v100 (heads 0-3=ADD, etc.)

Initialization:
  digit_codebook    : random orthonormal × 0.5  (small, will learn)
  digit_pos_embed   : sinusoidal positional encoding × 0.5 (place-value semantics)
  var_pos_embed     : same as v100 (randn 0.02)
  factor_pos_embed  : same as v100 (randn 0.02)
  breath_embed      : same as v100 (QR-orthonormal × 0.5)
  delta_gate        : ones (same as v100)

NO warm-start from v100 — different architecture, different parameter shapes.

Env var gates:
  V105_TASK=1                   — enable v105 forward path
  V105_K_MAX=8                  — number of iterative-prefill breaths (8 ≤ JIT limit)
  V105_N_DIGITS=5               — digit positions per variable (covers 0..99999)
  V105_N_MAX=16                 — max variable nodes
  V105_F_MAX=8                  — max factor nodes
  V105_ENERGY_WEIGHT=0.01       — expected-value energy loss weight
  V105_CALIB_WEIGHT=0.05        — calibration loss weight
  V105_FACTOR_AUX_WEIGHT=0.5    — factor node auxiliary loss weight
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

# Op index constants (same as v100)
OP_ADD = 0
OP_SUB = 1
OP_MUL = 2
OP_DIV = 3

V105_TASK               = int(os.environ.get("V105_TASK", "0")) > 0
V105_K_MAX              = int(os.environ.get("V105_K_MAX",              "8"))
V105_N_DIGITS           = int(os.environ.get("V105_N_DIGITS",           "5"))
V105_N_MAX              = int(os.environ.get("V105_N_MAX",              "16"))
V105_F_MAX              = int(os.environ.get("V105_F_MAX",               "8"))
V105_N_HEADS            = 16    # fixed: Pythia-410M
V105_ENERGY_WEIGHT      = float(os.environ.get("V105_ENERGY_WEIGHT",    "0.01"))
V105_CALIB_WEIGHT       = float(os.environ.get("V105_CALIB_WEIGHT",     "0.05"))
V105_FACTOR_AUX_WEIGHT  = float(os.environ.get("V105_FACTOR_AUX_WEIGHT","0.5"))

# Sequence length derived constants
# V105_T = V105_N_MAX * V105_N_DIGITS + V105_F_MAX  (computed per-init)
# Note: this is NOT a module-level constant because N_DIGITS/N_MAX/F_MAX are env-driven


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _make_digit_pos_embed(n_digits: int, hidden: int, scale: float = 0.5) -> np.ndarray:
    """Sinusoidal positional encoding for digit positions.

    Position p=0 is the MOST significant digit (10^(N-1)), p=N-1 is the least
    significant (10^0 = ones).  We use standard sin/cos alternation.

    Returns (n_digits, hidden) float32.
    """
    pe = np.zeros((n_digits, hidden), dtype=np.float32)
    position = np.arange(n_digits, dtype=np.float32).reshape(-1, 1)
    div_term = np.exp(
        np.arange(0, hidden, 2, dtype=np.float32) * -(math.log(10000.0) / hidden)
    )
    pe[:, 0::2] = np.sin(position * div_term)
    if hidden > 1:
        pe[:, 1::2] = np.cos(position * div_term[:hidden // 2])
    return pe * scale


def value_to_digits(value: int, n_digits: int) -> list[int]:
    """Decompose an integer value into n_digits digit positions (MSD-first).

    E.g. value=1234, n_digits=5 → [0, 1, 2, 3, 4]
    Clamps negative values to 0, clamps overflow (value >= 10^n_digits) to 9s.
    """
    v = max(0, int(round(value)))
    max_val = 10 ** n_digits - 1
    v = min(v, max_val)
    digits = []
    for p in range(n_digits - 1, -1, -1):
        place = 10 ** p
        d = v // place
        v = v % place
        digits.append(d)
    return digits  # index 0 = most significant


def digits_to_value(digits: list[int], n_digits: int) -> int:
    """Reconstruct integer from MSD-first digit list."""
    v = 0
    for p, d in enumerate(digits):
        v += d * (10 ** (n_digits - 1 - p))
    return v


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def embed_factor_graph_v105(
    digit_init: Tensor,       # (B, N_MAX, N_DIGITS, 10) — observed: one-hot; unobs: uniform
    node_kinds: Tensor,        # (B, T_MAX) int  — 0=obs_var, 1=unobs_var, 2=factor, -1=pad
                               #   T_MAX = N_MAX*N_DIGITS + F_MAX
    var_pos_embed: Tensor,     # (N_MAX, H)
    factor_pos_embed: Tensor,  # (F_MAX, H)
    node_kind_embed: Tensor,   # (3, H)
    digit_codebook: Tensor,    # (10, H)
    digit_pos_embed: Tensor,   # (N_DIGITS, H)
    n_max: int,
    n_digits: int,
    f_max: int,
) -> Tensor:
    """Build (B, T_MAX, H) hidden states.

    Layout:
      positions [v*n_digits : v*n_digits+n_digits]  ← variable v's digit positions
      positions [n_max*n_digits : n_max*n_digits+f_max]  ← factor nodes
    """
    B = int(digit_init.shape[0])
    H = int(var_pos_embed.shape[1])
    T = n_max * n_digits + f_max

    # --- Variable digit positions ---
    # digit_init: (B, N_MAX, N_DIGITS, 10)
    # digit_codebook: (10, H)
    # digit state: (B, N_MAX, N_DIGITS, 10) @ (10, H) → (B, N_MAX, N_DIGITS, H)
    di_cb = digit_init.cast(digit_codebook.dtype)      # (B, N_MAX, N_DIGITS, 10)
    var_digit_state = di_cb @ digit_codebook            # (B, N_MAX, N_DIGITS, H)

    # Per-digit-position embedding: (1, 1, N_DIGITS, H)
    dpe = digit_pos_embed.reshape(1, 1, n_digits, H).cast(var_digit_state.dtype)
    var_digit_h = var_digit_state + dpe.expand(B, n_max, n_digits, H)

    # Per-variable position embedding: (1, N_MAX, 1, H) broadcast over digits
    vpe = var_pos_embed.reshape(1, n_max, 1, H).cast(var_digit_h.dtype)
    var_digit_h = var_digit_h + vpe.expand(B, n_max, n_digits, H)

    # Flatten to (B, N_MAX * N_DIGITS, H)
    var_tokens = var_digit_h.reshape(B, n_max * n_digits, H)

    # --- Factor positions ---
    factor_pos = factor_pos_embed.reshape(1, f_max, H).cast(var_tokens.dtype).expand(B, f_max, H)

    # Concatenate
    x = var_tokens.cat(factor_pos, dim=1)  # (B, T, H)

    # Node-kind embedding (clamp padding to 0)
    nk_clamped = node_kinds.clip(0, 2)
    nk_oh      = nk_clamped.one_hot(3).cast(x.dtype)  # (B, T, 3)
    nk_emb     = nk_oh @ node_kind_embed.cast(x.dtype)
    x = x + nk_emb

    return x


# ---------------------------------------------------------------------------
# One transformer layer with per-head factor-graph attention mask
# ---------------------------------------------------------------------------

def fg_layer_forward_v105(
    layer: Any,
    x: Tensor,          # (B, T_MAX, H)
    attn_bias: Tensor,  # (B, N_HEADS, T_MAX, T_MAX)
) -> Tensor:
    """Run one BreathingLayer with per-head factor-graph attention (same as v100)."""
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
    scores = q @ k.transpose(-2, -1) * scale  # (B, n_heads, S, S)
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

def fg_breathing_forward_v105(
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
    """Run K iterative-prefill breaths on a batch of factor graphs.

    Per breath k:
      1. Add breath embedding.
      2. Build combined (B, N_HEADS, T, T) mask.
      3. Run 4 transformer layers.
      4. Learnable delta gate.
      5. Readout: digit_logits per variable (B, N_MAX, N_DIGITS, 10),
                  factor_digit_logits (B, F_MAX, N_DIGITS, 10) — projected via digit_codebook,
                  calib scalar.

    Returns:
      digit_logits_history   : K × (B, N_MAX, N_DIGITS, 10)
      factor_logits_history  : K × (B, F_MAX, N_DIGITS, 10)  — for factor-aux loss
      calib_history          : K × (B,)
    """
    assert hasattr(model, "fg_v105_digit_codebook"), \
        "model has no v105 params; was attach_fg_params_v105 called?"

    digit_codebook   = model.fg_v105_digit_codebook    # (10, H)
    digit_pos_embed  = model.fg_v105_digit_pos_embed   # (N_DIGITS, H)
    var_pos_embed    = model.fg_v105_var_pos_embed      # (N_MAX, H)
    factor_pos_embed = model.fg_v105_factor_pos_embed  # (F_MAX, H)
    node_kind_embed  = model.fg_v105_node_kind_embed   # (3, H)
    breath_embed     = model.fg_v105_breath_embed      # (K_max, H)
    delta_gate       = model.fg_v105_delta_gate        # (K_max,)
    calib_head_w     = model.fg_v105_calib_head_w      # (H, 1)
    calib_head_b     = model.fg_v105_calib_head_b      # (1,)

    B = int(digit_init.shape[0])
    T = n_max * n_digits + f_max

    x = embed_factor_graph_v105(
        digit_init, node_kinds,
        var_pos_embed, factor_pos_embed, node_kind_embed,
        digit_codebook, digit_pos_embed,
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
            h = fg_layer_forward_v105(layer, h, combined)

        # 4. Delta gate
        gate_k = delta_gate[k].cast(h.dtype).reshape(1, 1, 1)
        delta  = h - x_pre
        x      = x_pre + gate_k * delta

        # 5a. Variable digit logits: (B, N_MAX*N_DIGITS, H) → (B, N_MAX, N_DIGITS, 10)
        x_ln = _layernorm(x, model.ln_f_g, model.ln_f_b, model.cfg.layer_norm_eps).cast(dtypes.float)
        n_var_tokens = n_max * n_digits
        var_tokens   = x_ln[:, :n_var_tokens, :]              # (B, N_MAX*N_DIGITS, H)
        var_tokens_r = var_tokens.reshape(B, n_max, n_digits, -1)  # (B, N_MAX, N_DIGITS, H)
        cb_fp        = digit_codebook.cast(dtypes.float)       # (10, H)
        digit_logits_k = var_tokens_r @ cb_fp.T               # (B, N_MAX, N_DIGITS, 10)
        digit_logits_history.append(digit_logits_k)

        # 5b. Factor digit logits: (B, F_MAX, H) → (B, F_MAX, N_DIGITS, 10)
        #     Factor positions are point tokens (not digit-expanded); we tile to N_DIGITS
        #     and share the same digit_codebook projection.
        fac_tokens   = x_ln[:, n_var_tokens:n_var_tokens + f_max, :]   # (B, F_MAX, H)
        fac_tokens_r = fac_tokens.reshape(B, f_max, 1, -1).expand(B, f_max, n_digits, int(x_ln.shape[-1]))
        fac_logits_k = fac_tokens_r @ cb_fp.T                 # (B, F_MAX, N_DIGITS, 10)
        factor_logits_history.append(fac_logits_k)

        # 5c. Calibration
        pool        = x_ln.mean(axis=1)                        # (B, H)
        calib_logit = pool @ calib_head_w.cast(dtypes.float) + calib_head_b.cast(dtypes.float)
        calib_k     = calib_logit.reshape(-1).sigmoid()
        calib_history.append(calib_k)

    return digit_logits_history, factor_logits_history, calib_history


# ---------------------------------------------------------------------------
# Accuracy
# ---------------------------------------------------------------------------

def fg_accuracy_v105(
    digit_logits_final: Tensor,  # (B, N_MAX, N_DIGITS, 10)
    gold_digits: Tensor,          # (B, N_MAX, N_DIGITS) int — target digits
    observed_mask: Tensor,        # (B, N_MAX) int
    query_idx_np: np.ndarray,     # (B,) int
    n_digits: int = V105_N_DIGITS,
) -> dict:
    """Return accuracy stats: query_acc, unobserved_cell_acc (per-variable), digit_acc."""
    B   = int(digit_logits_final.shape[0])
    N   = int(digit_logits_final.shape[1])

    # Per-digit argmax → predicted digit sequence (B, N_MAX, N_DIGITS)
    pred_digits = digit_logits_final.argmax(axis=-1)  # (B, N_MAX, N_DIGITS)

    # Variable correct: ALL digits match gold
    dig_eq  = (pred_digits == gold_digits.cast(pred_digits.dtype)).cast(dtypes.float)  # (B, N, D)
    var_eq  = (dig_eq.min(axis=-1) > 0.5).cast(dtypes.float)                           # (B, N)
    unobs   = (1 - observed_mask.cast(dtypes.float))                                    # (B, N)

    cell_acc  = float(((var_eq * unobs).sum() / (unobs.sum() + 1e-8)).realize().numpy())
    digit_acc = float(((dig_eq * unobs.reshape(B, N, 1).expand(B, N, n_digits)).sum()
                       / (unobs.sum() * n_digits + 1e-8)).realize().numpy())

    pred_np = pred_digits.cast(dtypes.int).realize().numpy()  # (B, N, D)
    gold_np = gold_digits.cast(dtypes.int).realize().numpy()
    q_corr  = np.array([
        int(np.all(pred_np[b, query_idx_np[b]] == gold_np[b, query_idx_np[b]]))
        for b in range(B)
    ], dtype=np.float32)
    query_acc = float(q_corr.mean())

    return {"cell_acc": cell_acc, "digit_acc": digit_acc, "query_acc": query_acc}


# ---------------------------------------------------------------------------
# Model parameter attachment
# ---------------------------------------------------------------------------

def attach_fg_params_v105(
    model: Any,
    hidden: int,
    n_digits: int = V105_N_DIGITS,
    n_max: int = V105_N_MAX,
    f_max: int = V105_F_MAX,
    k_max: int | None = None,
) -> None:
    """Allocate v105 digit-codebook factor-graph params on `model`.

    Initialization:
      digit_codebook   (10, hidden):    random QR × 0.5 — orthonormal rows
      digit_pos_embed  (N_DIGITS, H):   sinusoidal × 0.5 — place-value semantics
      var_pos_embed    (N_MAX, H):      randn 0.02
      factor_pos_embed (F_MAX, H):      randn 0.02
      node_kind_embed  (3, H):          randn 0.02
      breath_embed     (K_max, H):      QR-orthonormal × 0.5
      delta_gate       (K_max,):        ones (full gating at init — pass-through)
      calib_head_w     (H, 1):          randn 0.02
      calib_head_b     (1,):            zeros
    """
    if k_max is None:
        k_max = V105_K_MAX

    rng = np.random.RandomState(20001)

    # --- Digit codebook (10, hidden): orthonormal rows at scale 1.0 ---
    # Scale 1.0 (vs positional embeds at 0.02) ensures initial embedding is
    # dominated by the digit identity: embed[v,p] ≈ digit_codebook[d] at init,
    # so readout logit[d] is large and correct (aligned init principle from v100).
    raw_dc = rng.randn(max(hidden, 10), hidden).astype(np.float32)
    q_dc, _ = np.linalg.qr(raw_dc)
    dc = q_dc[:10].astype(np.float32) * 1.0
    model.fg_v105_digit_codebook = Tensor(dc, dtype=dtypes.float).contiguous()

    # --- Digit position embedding (N_DIGITS, hidden): sinusoidal × 0.02 ---
    # Small scale so positional info doesn't swamp the digit identity signal.
    dpe = _make_digit_pos_embed(n_digits, hidden, scale=0.02)
    model.fg_v105_digit_pos_embed = Tensor(dpe, dtype=dtypes.float).contiguous()

    # --- Variable position embedding (N_MAX, hidden) ---
    vp = rng.randn(n_max, hidden).astype(np.float32) * 0.02
    model.fg_v105_var_pos_embed = Tensor(vp, dtype=dtypes.float).contiguous()

    # --- Factor position embedding (F_MAX, hidden) ---
    fp_emb = rng.randn(f_max, hidden).astype(np.float32) * 0.02
    model.fg_v105_factor_pos_embed = Tensor(fp_emb, dtype=dtypes.float).contiguous()

    # --- Node-kind embedding (3, hidden) ---
    nk = rng.randn(3, hidden).astype(np.float32) * 0.02
    model.fg_v105_node_kind_embed = Tensor(nk, dtype=dtypes.float).contiguous()

    # --- Calibration head ---
    cw = (rng.randn(hidden, 1) * 0.02).astype(np.float32)
    model.fg_v105_calib_head_w = Tensor(cw, dtype=dtypes.float).contiguous()
    model.fg_v105_calib_head_b = Tensor.zeros((1,), dtype=dtypes.float).contiguous()

    # --- Per-breath embedding: QR-orthonormal at scale 0.5 ---
    rng_be = np.random.RandomState(20002)
    raw_be = rng_be.randn(max(k_max, hidden), hidden).astype(np.float32)
    q_be, _ = np.linalg.qr(raw_be)
    be = q_be[:k_max].astype(np.float32) * 0.5
    model.fg_v105_breath_embed = Tensor(be, dtype=dtypes.float).contiguous()

    # --- Per-breath delta gate: init 1.0 ---
    model.fg_v105_delta_gate = Tensor.ones((k_max,), dtype=dtypes.float).contiguous()

    print(
        f"[v105] params attached: digit_codebook=(10,{hidden}), "
        f"digit_pos_embed=({n_digits},{hidden}), "
        f"var_pos_embed=({n_max},{hidden}), factor_pos_embed=({f_max},{hidden}), "
        f"breath_embed=({k_max},{hidden}), "
        f"T={n_max * n_digits + f_max} (was {n_max + f_max} in v100)",
        flush=True,
    )


def fg_v105_parameters(model: Any) -> list[Tensor]:
    """Trainable v105 factor-graph-specific params."""
    return [
        model.fg_v105_digit_codebook,
        model.fg_v105_digit_pos_embed,
        model.fg_v105_var_pos_embed,
        model.fg_v105_factor_pos_embed,
        model.fg_v105_node_kind_embed,
        model.fg_v105_calib_head_w,
        model.fg_v105_calib_head_b,
        model.fg_v105_breath_embed,
        model.fg_v105_delta_gate,
    ]


def fg_v105_state_dict(model: Any) -> dict[str, Tensor]:
    return {
        "fg_v105.digit_codebook":    model.fg_v105_digit_codebook,
        "fg_v105.digit_pos_embed":   model.fg_v105_digit_pos_embed,
        "fg_v105.var_pos_embed":     model.fg_v105_var_pos_embed,
        "fg_v105.factor_pos_embed":  model.fg_v105_factor_pos_embed,
        "fg_v105.node_kind_embed":   model.fg_v105_node_kind_embed,
        "fg_v105.calib_head_w":      model.fg_v105_calib_head_w,
        "fg_v105.calib_head_b":      model.fg_v105_calib_head_b,
        "fg_v105.breath_embed":      model.fg_v105_breath_embed,
        "fg_v105.delta_gate":        model.fg_v105_delta_gate,
    }


# ---------------------------------------------------------------------------
# Expected-value energy (differentiable constraint check)
# ---------------------------------------------------------------------------

def _expected_value_v105(digit_logits: Tensor, n_digits: int) -> Tensor:
    """Compute expected integer value from per-digit logit distributions.

    digit_logits: (..., N_DIGITS, 10)
    Returns: (...,) float — expected value = Σ_p E[digit_p] × 10^(N-1-p)
    """
    probs   = digit_logits.softmax(-1)  # (..., N_DIGITS, 10)
    d_vals  = Tensor(np.arange(10, dtype=np.float32)).cast(probs.dtype)  # (10,)
    exp_dig = (probs * d_vals).sum(axis=-1)  # (..., N_DIGITS)
    place_vals = Tensor(
        np.array([10 ** (n_digits - 1 - p) for p in range(n_digits)], dtype=np.float32)
    ).cast(exp_dig.dtype)  # (N_DIGITS,)
    return (exp_dig * place_vals).sum(axis=-1)  # (...)


def constraint_energy_v105(
    digit_logits_final: Tensor,   # (B, N_MAX, N_DIGITS, 10)
    factor_types: Tensor,          # (B, F_MAX) int
    factor_args: Tensor,           # (B, F_MAX, 3) int
    n_max: int = V105_N_MAX,
    f_max: int = V105_F_MAX,
    n_digits: int = V105_N_DIGITS,
) -> Tensor:
    """Expected-value constraint energy (differentiable, inside JIT).

    For each factor: E_factor = (E[result] - op(E[arg1], E[arg2]))^2
    Aggregated over all valid factors, normalized by count.

    MUL approximation: E[a*b] ≈ E[a] * E[b]  (independence assumption)
    DIV approximation: E[a/b] ≈ E[a] / (E[b] + ε)

    Both approximations are correct in expectation when arg distributions are
    narrow (peaky) — which they should be at convergence.
    """
    B = int(digit_logits_final.shape[0])

    # Compute expected value for all variables: (B, N_MAX)
    ev = _expected_value_v105(digit_logits_final, n_digits)  # (B, N_MAX)

    # Compute expected value factor-wise by gathering arg/result expected values.
    # We use gather-via-one-hot to avoid Python loops inside the JIT graph.
    # factor_args: (B, F_MAX, 3) — [arg1_idx, arg2_idx, result_idx]
    fa_clamped = factor_args.cast(dtypes.int).clip(0, n_max - 1)  # (B, F_MAX, 3)
    fa_oh = fa_clamped.reshape(B, f_max * 3).one_hot(n_max)       # (B, F_MAX*3, N_MAX)
    ev_bc = ev.reshape(B, 1, n_max).cast(dtypes.float)             # (B, 1, N_MAX)
    gathered = (fa_oh.cast(dtypes.float) * ev_bc).sum(axis=-1)    # (B, F_MAX*3)
    gathered_r = gathered.reshape(B, f_max, 3)                     # (B, F_MAX, 3)
    ev_arg1   = gathered_r[:, :, 0]   # (B, F_MAX)
    ev_arg2   = gathered_r[:, :, 1]   # (B, F_MAX)
    ev_result = gathered_r[:, :, 2]   # (B, F_MAX)

    # Expected result per op (vectorized over all factor slots)
    # ADD:  expected = ev_arg1 + ev_arg2
    # SUB:  expected = ev_arg1 - ev_arg2
    # MUL:  expected ≈ ev_arg1 * ev_arg2
    # DIV:  expected ≈ ev_arg1 / (ev_arg2 + eps)
    ev_add = ev_arg1 + ev_arg2                                   # (B, F_MAX)
    ev_sub = ev_arg1 - ev_arg2
    ev_mul = ev_arg1 * ev_arg2
    ev_div = ev_arg1 / (ev_arg2.abs() + 1.0)                    # +1 avoids div-by-zero

    # Select expected result via op-type one-hot mask
    # factor_types: (B, F_MAX) int in {-1,0,1,2,3}; clamp -1 → 0 (padding handled by valid mask)
    ft_clamped = factor_types.cast(dtypes.int).clip(0, 3)        # (B, F_MAX)
    ft_oh      = ft_clamped.one_hot(4).cast(dtypes.float)        # (B, F_MAX, 4)
    ev_expected_stack = ev_add.reshape(B, f_max, 1).cat(
        ev_sub.reshape(B, f_max, 1),
        ev_mul.reshape(B, f_max, 1),
        ev_div.reshape(B, f_max, 1),
        dim=-1,
    )  # (B, F_MAX, 4)
    ev_expected = (ft_oh * ev_expected_stack).sum(axis=-1)       # (B, F_MAX)

    # Valid factor mask: factor_types >= 0 AND all arg/result indices < n_max
    valid = (factor_types >= 0).cast(dtypes.float)               # (B, F_MAX)

    # Energy: squared residual, masked.
    # Use relative error instead of absolute squared error to avoid overflow for
    # large GSM8K values (e.g. expected_result=50000 → residual²=2.5e9 per factor).
    # relative_err = |residual| / (|ev_expected| + 1) so values ≤ 1 map to ≤ 1.
    residual    = ev_result - ev_expected                        # (B, F_MAX)
    rel_err     = residual.abs() / (ev_expected.abs() + 1.0)   # dimensionless
    # Clip to [0, 10] to guard against any remaining extreme cases
    rel_err_clipped = rel_err.clip(0.0, 10.0)
    energy      = rel_err_clipped * valid
    n_valid     = valid.sum() + 1e-8
    return energy.sum() / n_valid


# ---------------------------------------------------------------------------
# JIT training step
# ---------------------------------------------------------------------------

_JIT_V105_CACHE: dict = {}


def _compile_jit_fg_step_v105(
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
    """Compile a TinyJit'd train step for v105.

    Inputs to JIT:
      digit_init     : (B, N_MAX, N_DIGITS, 10) fp32 — digit one-hots or uniforms
      node_kinds     : (B, T_MAX) int
      staging_mask   : (B, K_MAX, T_MAX, T_MAX) fp32
      head_op_mask   : (B, N_HEADS, T_MAX, T_MAX) fp32
      gold_digits    : (B, N_MAX, N_DIGITS) int
      observed_mask  : (B, N_MAX) int
      factor_gold_dg : (B, F_MAX, N_DIGITS) int — digit decomp of gold result per factor
      factor_valid   : (B, F_MAX) float — 1=real factor, 0=pad
      factor_types   : (B, F_MAX) int — for energy computation
      factor_args    : (B, F_MAX, 3) int — for energy computation

    Returns:
      total, healthy, var_ce, factor_aux, calib, energy, cell_acc, query_acc,
      *pb_ce_0..K-1
    """
    key = ("v105", id(model), id(opt), int(K), int(B), float(factor_aux_weight),
           float(calib_weight), float(energy_weight), int(n_max), int(f_max),
           int(n_digits), float(grad_clip))
    if key in _JIT_V105_CACHE:
        return _JIT_V105_CACHE[key]

    aw     = float(calib_weight)
    fw     = float(factor_aux_weight)
    ew     = float(energy_weight)
    gc     = float(grad_clip)
    params = opt.params

    print(
        f"[JIT] compile v105 fg step: K={K} B={B} n_digits={n_digits} "
        f"T={n_max * n_digits + f_max} aw={aw} fw={fw} ew={ew} gc={gc}...",
        flush=True,
    )

    @TinyJit
    def _step(
        digit_init: Tensor,      # (B, N_MAX, N_DIGITS, 10)
        node_kinds: Tensor,       # (B, T_MAX) int
        staging_mask: Tensor,     # (B, K_MAX, T_MAX, T_MAX)
        head_op_mask: Tensor,     # (B, N_HEADS, T_MAX, T_MAX)
        gold_digits: Tensor,      # (B, N_MAX, N_DIGITS) int
        observed_mask: Tensor,    # (B, N_MAX) int
        factor_gold_dg: Tensor,   # (B, F_MAX, N_DIGITS) int
        factor_valid: Tensor,     # (B, F_MAX) float
        factor_types: Tensor,     # (B, F_MAX) int
        factor_args: Tensor,      # (B, F_MAX, 3) int
    ):
        opt.zero_grad()

        digit_logits_history, factor_logits_history, calib_history = \
            fg_breathing_forward_v105(
                model, digit_init, node_kinds, staging_mask, head_op_mask,
                K=K, n_max=n_max, f_max=f_max, n_digits=n_digits,
            )

        # --- Main CE on unobserved variable digit positions ---
        # gold_digits: (B, N_MAX, N_DIGITS) int
        # unobs: (B, N_MAX) float
        unobs_float   = (1 - observed_mask.cast(dtypes.float))            # (B, N_MAX)
        # Expand unobs to digit level: (B, N_MAX, N_DIGITS)
        unobs_dg      = unobs_float.reshape(B, n_max, 1).expand(B, n_max, n_digits)
        n_unobs_dg    = unobs_dg.sum() + 1e-8

        # Flatten for CE: (B*N_MAX*N_DIGITS,)
        gold_flat     = gold_digits.cast(dtypes.int).reshape(B * n_max * n_digits)
        unobs_flat    = unobs_dg.reshape(B * n_max * n_digits)

        var_loss_sum  = Tensor.zeros((), dtype=dtypes.float).contiguous()
        var_weight_sum = 0.0
        per_breath_ce_t: list[Tensor] = []

        for k, dig_logits in enumerate(digit_logits_history):
            weight_k    = 1.0 + float(k) / float(max(K - 1, 1))
            logits_flat = dig_logits.reshape(B * n_max * n_digits, 10)
            log_probs   = logits_flat.log_softmax(axis=-1)               # (B*N*D, 10)
            gold_oh     = gold_flat.one_hot(10).cast(log_probs.dtype)
            nll         = -(log_probs * gold_oh).sum(axis=-1)            # (B*N*D,)
            masked_nll  = nll * unobs_flat.cast(nll.dtype)
            ce_k        = masked_nll.sum() / n_unobs_dg
            per_breath_ce_t.append(ce_k)
            var_loss_sum   = var_loss_sum + ce_k * weight_k
            var_weight_sum += weight_k

        var_loss = var_loss_sum / float(var_weight_sum)

        # --- Factor-execute auxiliary loss on factor digit positions ---
        # factor_gold_dg: (B, F_MAX, N_DIGITS) int
        # factor_valid:   (B, F_MAX) float
        n_valid_fac  = factor_valid.sum() + 1e-8
        # Flatten: (B*F_MAX*N_DIGITS,)
        fgd_flat     = factor_gold_dg.cast(dtypes.int).reshape(B * f_max * n_digits)
        # valid expanded to digit level: (B*F_MAX*N_DIGITS,)
        valid_dg     = factor_valid.reshape(B, f_max, 1).expand(B, f_max, n_digits)
        valid_flat   = valid_dg.reshape(B * f_max * n_digits)
        n_valid_dg   = valid_flat.sum() + 1e-8
        gold_fac_oh  = fgd_flat.one_hot(10).cast(dtypes.float)           # (B*F*D, 10)

        factor_aux_sum   = Tensor.zeros((), dtype=dtypes.float).contiguous()
        factor_aux_w_sum = 0.0

        for k_aux, fac_logits_k in enumerate(factor_logits_history):
            w_k_aux    = 1.0 + float(k_aux) / float(max(K - 1, 1))
            fac_flat   = fac_logits_k.reshape(B * f_max * n_digits, 10)
            fac_lp     = fac_flat.log_softmax(axis=-1)
            fac_nll    = -(fac_lp * gold_fac_oh).sum(axis=-1)           # (B*F*D,)
            fac_masked = fac_nll * valid_flat.cast(fac_nll.dtype)
            fac_ce_k   = fac_masked.sum() / n_valid_dg
            factor_aux_sum   = factor_aux_sum + fac_ce_k * w_k_aux
            factor_aux_w_sum += w_k_aux

        factor_aux_loss = factor_aux_sum / float(factor_aux_w_sum)

        # --- Constraint energy (expected-value, differentiable) ---
        final_dig_logits = digit_logits_history[-1]  # (B, N_MAX, N_DIGITS, 10)
        energy_loss = constraint_energy_v105(
            final_dig_logits, factor_types, factor_args,
            n_max=n_max, f_max=f_max, n_digits=n_digits,
        )

        # --- Calibration ---
        # Variable correct if ALL digits match (MSD-first)
        final_pred_dg = digit_logits_history[-1].argmax(axis=-1).detach()  # (B, N_MAX, N_DIGITS)
        dg_eq   = (final_pred_dg == gold_digits.cast(dtypes.int)).cast(dtypes.float)  # (B, N, D)
        var_eq  = dg_eq.min(axis=-1)                                                   # (B, N)
        unobs_2d = unobs_float
        eq_unobs = var_eq * unobs_2d
        n_unobs_per = unobs_2d.sum(axis=-1) + 1e-8
        correct = eq_unobs.sum(axis=-1) / n_unobs_per                                  # (B,)

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

    _JIT_V105_CACHE[key] = _step
    print(f"[JIT] v105 fg step ready (cache={len(_JIT_V105_CACHE)}); first call compiles...",
          flush=True)
    return _step


def _compile_jit_fg_eval_v105(
    model: Any,
    K: int,
    B: int,
    n_max: int = V105_N_MAX,
    f_max: int = V105_F_MAX,
    n_digits: int = V105_N_DIGITS,
):
    """Compile a TinyJit'd eval step (forward only)."""
    key = ("eval_v105", id(model), int(K), int(B), int(n_max), int(f_max), int(n_digits))
    if key in _JIT_V105_CACHE:
        return _JIT_V105_CACHE[key]

    print(f"[JIT] compile v105 fg eval: K={K} B={B} n_digits={n_digits}...", flush=True)

    @TinyJit
    def _eval(
        digit_init: Tensor,
        node_kinds: Tensor,
        staging_mask: Tensor,
        head_op_mask: Tensor,
        gold_digits: Tensor,
        observed_mask: Tensor,
    ):
        digit_logits_history, _, _ = fg_breathing_forward_v105(
            model, digit_init, node_kinds, staging_mask, head_op_mask,
            K=K, n_max=n_max, f_max=f_max, n_digits=n_digits,
        )
        final_logits  = digit_logits_history[-1]                              # (B, N, D, 10)
        pred_dg       = final_logits.argmax(axis=-1)                          # (B, N, D)
        dg_eq         = (pred_dg == gold_digits.cast(pred_dg.dtype)).cast(dtypes.float)
        var_eq        = dg_eq.min(axis=-1)                                    # (B, N)
        unobs         = (1 - observed_mask.cast(dtypes.float))
        cell_acc      = (var_eq * unobs).sum() / (unobs.sum() + 1e-8)
        return pred_dg.realize(), cell_acc.realize()

    _JIT_V105_CACHE[key] = _eval
    print(f"[JIT] v105 eval ready (cache={len(_JIT_V105_CACHE)})", flush=True)
    return _eval
