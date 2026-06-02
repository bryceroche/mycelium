"""v105.4 factor graph breathing transformer — v105.3 + hierarchical codebooks.

Extends v105.3 (LSD-first array + digit RoPE + projection waist + IB codebook
+ AR digit decoding + delta_gate + calibration) with four targeted additions:

  ADDITION 1 — MAGNITUDE HEAD (4-way per-cell classification)
    For each variable cell, classify "how many digits is this number?":
      class 0 : 1-digit  (value < 10)
      class 1 : 2-digit  (value < 100)
      class 2 : 3-digit  (value < 1000)
      class 3 : 4+ digit (value >= 1000, capped)
    Magnitude probs weight a (4, hidden) centroid table to produce a
    magnitude_embed that is added to each digit's pre-codebook hidden state.

  ADDITION 2 — PER-POSITION DIGIT CODEBOOKS
    Was (10, hidden) shared; now (n_digits, 10, hidden) — one codebook per
    digit position.  Hidden vector for "digit d at position p" can differ
    across positions; the AR loop and reconstruction MSE both use the
    position-specific table.

  ADDITION 3 — HIERARCHICAL IB ATTENTION
    Two-level softmax: family attention (4 ops: ADD/SUB/MUL/DIV) gates leaf
    attention (32 IB leaves). family_centroids initialized from the mean of
    IB centroids within each family; leaf_to_family loaded from the IB tree.

  ADDITION 4 — SOFT MAGNITUDE-DERIVED VALID MASK FOR FACTOR_AUX
    The factor_aux loss uses (gold * soft) per-digit mask, where soft is
    derived from magnitude_softmax → class_to_valid (4, n_digits) mapping.
    Provides gradient on the magnitude head from per-NUMBER reconstruction.
    var_loss still uses the clean gold mask (preserves crisp per-digit CE).

Loss structure:
  total = var_loss
        + 0.3 * magnitude_loss        (NEW)
        + 1.0 * factor_aux_loss
        + 0.05 * calib_loss
        + 0.01 * energy_loss

Env var gates:
  V105_4_TASK=1                — enable v105.4 forward path
  V105_4_K_MAX=8               — iterative-prefill breaths
  V105_4_N_DIGITS=5
  V105_4_N_MAX=16
  V105_4_F_MAX=8
  V105_4_WAIST=512
  V105_4_CODEBOOK_N=32         — IB leaf codebook entries
  V105_4_N_FAMILIES=4          — fixed at 4 (add/sub/mul/div)
  V105_4_IB_CENTROIDS=.cache/ib_centroids_gsm8k_partial.npz
  V105_4_IB_TREE=.cache/ib_tree_gsm8k_partial.json
  V105_4_ENERGY_WEIGHT=0.01
  V105_4_CALIB_WEIGHT=0.05
  V105_4_FACTOR_AUX_WEIGHT=1.0
  V105_4_MAGNITUDE_WEIGHT=0.3  — α for magnitude_loss
  V105_4_ROPE_BASE=10000
  V105_4_IB_INIT=1
  V105_4_WAIST_LORA_INIT=1
  V105_4_NUMBER_MSE_ONLY=0
  V105_4_AR_DIGITS=1
  V105_4_AR_COND_SCALE=0.5
  V105_4_AR_MSD_FIRST=0
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
)
# LSD-first encoding helpers come from the v105.3 data module.
from mycelium.factor_graph_data_v105_4 import (
    value_to_digits_lsd, digits_to_value_lsd,
)
from mycelium.factor_graph_v105_1 import (
    _precompute_digit_rope,
    apply_rope_digit_tg,
)
# IB centroid loader reused from v104.
from mycelium.factor_graph_v104 import load_ib_centroids

__all__ = [
    "OP_ADD", "OP_SUB", "OP_MUL", "OP_DIV",
    "value_to_digits_lsd", "digits_to_value_lsd",
]

# ---------------------------------------------------------------------------
# Env vars
# ---------------------------------------------------------------------------

V105_4_TASK             = int(os.environ.get("V105_4_TASK",             "0")) > 0
V105_4_K_MAX            = int(os.environ.get("V105_4_K_MAX",            "8"))
V105_4_N_DIGITS         = int(os.environ.get("V105_4_N_DIGITS",         "5"))
V105_4_N_MAX            = int(os.environ.get("V105_4_N_MAX",            "16"))
V105_4_F_MAX            = int(os.environ.get("V105_4_F_MAX",             "8"))
V105_4_N_HEADS          = 16   # fixed: Pythia-410M
V105_4_WAIST            = int(os.environ.get("V105_4_WAIST",            "512"))
V105_4_CODEBOOK_N       = int(os.environ.get("V105_4_CODEBOOK_N",       "32"))
V105_4_N_FAMILIES       = 4    # fixed at 4: ADD/SUB/MUL/DIV
V105_4_N_MAGNITUDE      = 4    # fixed at 4: 1/2/3/4+-digit classes
V105_4_IB_CENTROIDS     = os.environ.get(
    "V105_4_IB_CENTROIDS", ".cache/ib_centroids_gsm8k_partial.npz"
)
V105_4_IB_TREE          = os.environ.get(
    "V105_4_IB_TREE", ".cache/ib_tree_gsm8k_partial.json"
)
V105_4_ENERGY_WEIGHT    = float(os.environ.get("V105_4_ENERGY_WEIGHT",   "0.01"))
V105_4_CALIB_WEIGHT     = float(os.environ.get("V105_4_CALIB_WEIGHT",    "0.05"))
V105_4_FACTOR_AUX_WEIGHT = float(os.environ.get("V105_4_FACTOR_AUX_WEIGHT", "1.0"))
V105_4_MAGNITUDE_WEIGHT = float(os.environ.get("V105_4_MAGNITUDE_WEIGHT", "0.3"))
V105_4_ROPE_BASE        = float(os.environ.get("V105_4_ROPE_BASE",       "10000.0"))
V105_4_IB_INIT          = int(os.environ.get("V105_4_IB_INIT",           "1")) > 0
V105_4_WAIST_LORA_INIT  = int(os.environ.get("V105_4_WAIST_LORA_INIT",   "1")) > 0
V105_4_FOURIER_INIT     = int(os.environ.get("V105_4_FOURIER_INIT",      "0")) > 0
# Drop per-digit CE entirely, use per-NUMBER MSE on reconstructed value as the
# sole variable supervision.
V105_4_NUMBER_MSE_ONLY  = int(os.environ.get("V105_4_NUMBER_MSE_ONLY",   "0")) > 0
# Autoregressive digit decoding. Each digit's logits condition on the soft
# (softmax) predictions of all previously committed digit positions.
V105_4_AR_DIGITS        = int(os.environ.get("V105_4_AR_DIGITS",         "0")) > 0
# Scale of the soft prediction's embedding contribution when conditioning the
# next digit's hidden state. Smaller = milder conditioning, larger = stronger.
V105_4_AR_COND_SCALE    = float(os.environ.get("V105_4_AR_COND_SCALE",   "0.5"))
# AR iteration direction (in LSD-first array layout):
#   0 = LSD-first (default; array index 0 → N-1 — ones first, condition upward)
#   1 = MSD-first (array index N-1 → 0 — ten-thousands first, condition downward)
V105_4_AR_MSD_FIRST     = int(os.environ.get("V105_4_AR_MSD_FIRST",      "0")) > 0


# ---------------------------------------------------------------------------
# IB tree loader: build leaf_to_family mapping (32 → 4) from JSON
# ---------------------------------------------------------------------------

_OP_NAME_TO_IDX = {"ADD": 0, "SUB": 1, "MUL": 2, "DIV": 3}


def _load_leaf_to_family(
    path: str, n_code: int = 32, n_families: int = 4
) -> tuple[np.ndarray, list[str]]:
    """Load leaf_to_family int array (n_code,) from IB tree JSON.

    The JSON has a 'leaves' list; each leaf has an 'op' field in {ADD,SUB,MUL,DIV}.
    Leaf order in the JSON matches the npz ordering and thus the codebook index.

    Returns:
      leaf_to_family : (n_code,) int array, values in [0, n_families)
      leaf_ids       : list of leaf id strings (for diagnostic)
    """
    import json
    leaf_to_family = np.zeros(n_code, dtype=np.int32)
    leaf_ids: list[str] = []

    if not os.path.exists(path):
        # Fall back to uniform 8 per family.
        print(
            f"[v105.4] IB tree JSON missing at {path}; falling back to "
            f"uniform 8-per-family leaf_to_family assignment.",
            flush=True,
        )
        per_fam = max(1, n_code // n_families)
        for i in range(n_code):
            leaf_to_family[i] = min(i // per_fam, n_families - 1)
        return leaf_to_family, [f"FAKE.{i}" for i in range(n_code)]

    try:
        with open(path) as f:
            tree = json.load(f)
        leaves = tree.get("leaves", [])
        if len(leaves) < n_code:
            print(
                f"[v105.4] IB tree has {len(leaves)} leaves < n_code={n_code}; "
                f"falling back to uniform 8-per-family.",
                flush=True,
            )
            per_fam = max(1, n_code // n_families)
            for i in range(n_code):
                leaf_to_family[i] = min(i // per_fam, n_families - 1)
            return leaf_to_family, [f"FAKE.{i}" for i in range(n_code)]
        for i in range(n_code):
            leaf = leaves[i]
            op = str(leaf.get("op", "ADD")).upper()
            leaf_to_family[i] = _OP_NAME_TO_IDX.get(op, 0)
            leaf_ids.append(str(leaf.get("leaf_id", f"L{i}")))
    except Exception as e:
        print(
            f"[v105.4] IB tree JSON load failed ({e}); falling back to "
            f"uniform 8-per-family.",
            flush=True,
        )
        per_fam = max(1, n_code // n_families)
        for i in range(n_code):
            leaf_to_family[i] = min(i // per_fam, n_families - 1)
        leaf_ids = [f"FAKE.{i}" for i in range(n_code)]

    return leaf_to_family, leaf_ids


def _fourier_digit_codebook(n_digits: int, hidden: int) -> np.ndarray:
    """Cyclic-Fourier init for digit_codebook.

    Maps each digit d to phases on a circle: phase_d = 2π·d/n_digits.
    The codebook fills hidden dims with [cos(d·freq·phase), sin(d·freq·phase)]
    pairs cycling through the Nyquist-limited frequencies 1..n_digits/2.
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
# Component 1+2: Embedding with digit-axis RoPE (LSD layout — no reversal)
# ---------------------------------------------------------------------------

def embed_factor_graph_v105_4(
    digit_init: Tensor,       # (B, N_MAX, N_DIGITS, 10) — one-hot / uniform
    node_kinds: Tensor,        # (B, T_MAX) int
    var_pos_embed: Tensor,     # (N_MAX, H)
    factor_pos_embed: Tensor,  # (F_MAX, H)
    node_kind_embed: Tensor,   # (3, H)
    digit_codebook: Tensor,    # (N_DIGITS, 10, H) — per-position codebooks (v105.4)
    digit_rope_cos: Tensor,    # (N_DIGITS, H)
    digit_rope_sin: Tensor,    # (N_DIGITS, H)
    n_max: int,
    n_digits: int,
    f_max: int,
) -> Tensor:
    """Build (B, T_MAX, H) hidden states with digit-axis RoPE.

    v105.4: digit_codebook is now (n_digits, 10, H) — PER-POSITION.

    Component 1: digit_codebook[p] (10, H) — distinct codebook per position p.
    Component 2: apply_rope_digit_tg — each position p gets rotated by p*freq.

    LSD layout: array index 0 = ones digit ⇒ RoPE position 0 = no rotation.
                array index N-1 = most significant ⇒ maximum rotation.

    raw[b, v, p] = digit_codebook[p, digit_value] + var_pos_embed[v]
    embed[b, v, p] = RoPE(raw[b, v, p], p)
    """
    B = int(digit_init.shape[0])
    H = int(var_pos_embed.shape[1])

    # Per-position codebook contraction: digit_init has shape (B, N_MAX, N_DIGITS, 10)
    # and digit_codebook is (N_DIGITS, 10, H).  We contract on dim 10 per position p.
    # Reshape both to align the digit-position axis for einsum-like behavior.
    di_cb = digit_init.cast(digit_codebook.dtype)             # (B, N_MAX, N_DIGITS, 10)
    # Broadcast multiply on (B, N_MAX, N_DIGITS, 10, H):
    cb_bcast = digit_codebook.reshape(1, 1, n_digits, 10, H)  # (1, 1, N_DIGITS, 10, H)
    di_bcast = di_cb.reshape(B, n_max, n_digits, 10, 1)        # (B, N_MAX, N_DIGITS, 10, 1)
    var_digit_state = (di_bcast * cb_bcast).sum(axis=3)        # (B, N_MAX, N_DIGITS, H)

    vpe = var_pos_embed.reshape(1, n_max, 1, H).cast(var_digit_state.dtype)
    raw = var_digit_state + vpe.expand(B, n_max, n_digits, H)  # (B, N_MAX, N_DIGITS, H)

    # Apply digit RoPE (Component 2) — LSD layout, no reversal
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
# One transformer layer (identical to v105.1.2)
# ---------------------------------------------------------------------------

def fg_layer_forward_v105_4(
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
    """
    cb  = codebook.cast(h.dtype)
    tmp = temperature.cast(h.dtype)

    scores   = h @ cb.T / tmp.reshape(1, 1, 1)     # (B, T, N_CODE)
    weights  = scores.clip(-1e4, 1e4).softmax(-1)  # (B, T, N_CODE)
    recon    = weights @ cb                          # (B, T, H)
    quantize = recon - h                            # delta: toward codebook
    gate_k   = delta_gate_quant_k.cast(h.dtype).reshape(1, 1, 1)
    return h + gate_k * quantize                    # = h at init


def apply_hierarchical_ib_codebook(
    h: Tensor,                       # (B, T, H)
    leaf_codebook: Tensor,            # (N_CODE, H)
    family_centroids: Tensor,         # (N_FAMILIES, H)
    leaf_to_family_oh: Tensor,        # (N_CODE, N_FAMILIES) float — frozen one-hot
    temperature: Tensor,              # () scalar
    delta_gate_quant_k: Tensor,       # () — zero at init
) -> Tensor:
    """Hierarchical IB soft codebook with family gating (v105.4 addition 3).

    Two-level soft attention:
      family_weights = softmax(h @ family_centroids^T)   (B, T, F)
      leaf_weights_flat = softmax(h @ leaf_codebook^T)   (B, T, N)
      # Multiply each leaf weight by its family weight, then renormalize:
      fam_for_each_leaf = family_weights @ leaf_to_family_oh^T  (B, T, N)
      hierarchical_weights = leaf_weights_flat * fam_for_each_leaf
      hierarchical_weights /= sum + 1e-8
      ib_context = hierarchical_weights @ leaf_codebook  (B, T, H)
      h_new = h + delta_gate_quant_k * (ib_context - h)

    At init: delta_gate_quant_k = 0 → output = h (byte-identical to v105.3 step-0).
    """
    cb     = leaf_codebook.cast(h.dtype)            # (N, H)
    famcb  = family_centroids.cast(h.dtype)         # (F, H)
    l2f_oh = leaf_to_family_oh.cast(h.dtype)        # (N, F)
    tmp    = temperature.cast(h.dtype)

    # 1. Family attention
    fam_scores  = h @ famcb.T / tmp.reshape(1, 1, 1)              # (B, T, F)
    fam_weights = fam_scores.clip(-1e4, 1e4).softmax(-1)          # (B, T, F)

    # 2. Leaf attention (flat)
    leaf_scores  = h @ cb.T / tmp.reshape(1, 1, 1)                # (B, T, N)
    leaf_w_flat  = leaf_scores.clip(-1e4, 1e4).softmax(-1)        # (B, T, N)

    # 3. Reweight leaves by their family weight (gather via one-hot matmul).
    # family_weights: (B, T, F), l2f_oh: (N, F) → fam_for_each_leaf: (B, T, N)
    fam_for_each_leaf = fam_weights @ l2f_oh.T                    # (B, T, N)

    hierarchical = leaf_w_flat * fam_for_each_leaf
    norm         = hierarchical.sum(axis=-1, keepdim=True) + 1e-8
    hierarchical = hierarchical / norm

    # 4. Apply
    ib_context = hierarchical @ cb                                # (B, T, H)
    quantize   = ib_context - h
    gate_k     = delta_gate_quant_k.cast(h.dtype).reshape(1, 1, 1)
    return h + gate_k * quantize                                  # = h at init


# ---------------------------------------------------------------------------
# Constraint energy (LSD place values)
# ---------------------------------------------------------------------------

def _expected_value_v105_4(digit_logits: Tensor, n_digits: int) -> Tensor:
    """Compute expected integer value from per-digit logit distributions (LSD layout).

    digit_logits: (..., N_DIGITS, 10)
    Returns: (...,) float — expected value = Σ_p E[digit_p] × 10^p

    LSD place values: array index p has place 10^p (ones at idx 0, ten-thousands at idx N-1).
    """
    probs   = digit_logits.softmax(-1)  # (..., N_DIGITS, 10)
    d_vals  = Tensor(np.arange(10, dtype=np.float32)).cast(probs.dtype)  # (10,)
    exp_dig = (probs * d_vals).sum(axis=-1)  # (..., N_DIGITS)
    place_vals = Tensor(
        np.array([10 ** p for p in range(n_digits)], dtype=np.float32)
    ).cast(exp_dig.dtype)  # (N_DIGITS,) — LSD place values
    return (exp_dig * place_vals).sum(axis=-1)  # (...)


def constraint_energy_v105_4(
    digit_logits_final: Tensor,   # (B, N_MAX, N_DIGITS, 10)
    factor_types: Tensor,          # (B, F_MAX) int
    factor_args: Tensor,           # (B, F_MAX, 3) int
    n_max: int = V105_4_N_MAX,
    f_max: int = V105_4_F_MAX,
    n_digits: int = V105_4_N_DIGITS,
) -> Tensor:
    """Expected-value constraint energy (differentiable, inside JIT) — LSD place values.

    Same formula as constraint_energy_v105 but with LSD-first place values.
    For each factor: E_factor = (E[result] - op(E[arg1], E[arg2]))^2, aggregated
    over valid factors and normalized.
    """
    B = int(digit_logits_final.shape[0])

    # Compute expected value for all variables: (B, N_MAX) — LSD-aware
    ev = _expected_value_v105_4(digit_logits_final, n_digits)  # (B, N_MAX)

    # Gather arg/result expected values via one-hot.
    fa_clamped = factor_args.cast(dtypes.int).clip(0, n_max - 1)  # (B, F_MAX, 3)
    fa_oh = fa_clamped.reshape(B, f_max * 3).one_hot(n_max)       # (B, F_MAX*3, N_MAX)
    ev_bc = ev.reshape(B, 1, n_max).cast(dtypes.float)             # (B, 1, N_MAX)
    gathered = (fa_oh.cast(dtypes.float) * ev_bc).sum(axis=-1)    # (B, F_MAX*3)
    gathered_r = gathered.reshape(B, f_max, 3)                     # (B, F_MAX, 3)
    ev_arg1   = gathered_r[:, :, 0]
    ev_arg2   = gathered_r[:, :, 1]
    ev_result = gathered_r[:, :, 2]

    ev_add = ev_arg1 + ev_arg2
    ev_sub = ev_arg1 - ev_arg2
    ev_mul = ev_arg1 * ev_arg2
    ev_div = ev_arg1 / (ev_arg2.abs() + 1.0)

    ft_clamped = factor_types.cast(dtypes.int).clip(0, 3)
    ft_oh      = ft_clamped.one_hot(4).cast(dtypes.float)
    ev_expected_stack = ev_add.reshape(B, f_max, 1).cat(
        ev_sub.reshape(B, f_max, 1),
        ev_mul.reshape(B, f_max, 1),
        ev_div.reshape(B, f_max, 1),
        dim=-1,
    )
    ev_expected = (ft_oh * ev_expected_stack).sum(axis=-1)

    valid = (factor_types >= 0).cast(dtypes.float)
    residual    = ev_result - ev_expected
    rel_err     = residual.abs() / (ev_expected.abs() + 1.0)
    rel_err_clipped = rel_err.clip(0.0, 10.0)
    energy      = rel_err_clipped * valid
    n_valid     = valid.sum() + 1e-8
    return energy.sum() / n_valid


# ---------------------------------------------------------------------------
# Iterative prefill loop — full stack
# ---------------------------------------------------------------------------

def fg_breathing_forward_v105_4(
    model: Any,
    digit_init: Tensor,       # (B, N_MAX, N_DIGITS, 10)
    node_kinds: Tensor,        # (B, T_MAX) int
    staging_mask: Tensor,      # (B, K_MAX, T_MAX, T_MAX)
    head_op_mask: Tensor,      # (B, N_HEADS, T_MAX, T_MAX)
    K: int,
    n_max: int = V105_4_N_MAX,
    f_max: int = V105_4_F_MAX,
    n_digits: int = V105_4_N_DIGITS,
) -> tuple[list[Tensor], list[Tensor], list[Tensor], list[Tensor]]:
    """K iterative-prefill breaths with v105.4 additions.

    Per breath k:
      1. Breath embedding added.
      2. Combined staging + op mask built.
      3. 4 transformer layers run.
      4. Hierarchical IB codebook soft projection (Component 4 v105.4)
         — family attention gates leaf attention.
      5. Projection waist compression  (Component 3 — zero effect at step 0).
      6. Delta gate residual update.
      7. Per-cell magnitude head: 4-way "n_digits" classification, used to
         construct a magnitude_embed added to each digit-position hidden state.
      8. Readout: per-position digit codebook → AR digit logits; factor logits;
         calibration; magnitude logits.

    Returns:
      digit_logits_history     : list[K] of (B, N_MAX, N_DIGITS, 10)
      factor_logits_history    : list[K] of (B, F_MAX, N_DIGITS, 10)
      calib_history            : list[K] of (B,)
      magnitude_logits_history : list[K] of (B, N_MAX, N_MAGNITUDE=4)
    """
    assert hasattr(model, "fg_v105_4_digit_codebook"), \
        "model has no v105.4 params; was attach_fg_params_v105_4 called?"

    # Component 1+2 params
    # v105.4: digit_codebook now has shape (N_DIGITS, 10, H) — per-position.
    digit_codebook   = model.fg_v105_4_digit_codebook    # (N_DIGITS, 10, H)
    digit_rope_cos   = model.fg_v105_4_digit_rope_cos    # (N_DIGITS, H)
    digit_rope_sin   = model.fg_v105_4_digit_rope_sin    # (N_DIGITS, H)
    var_pos_embed    = model.fg_v105_4_var_pos_embed      # (N_MAX, H)
    factor_pos_embed = model.fg_v105_4_factor_pos_embed  # (F_MAX, H)
    node_kind_embed  = model.fg_v105_4_node_kind_embed   # (3, H)
    breath_embed     = model.fg_v105_4_breath_embed      # (K_max, H)
    delta_gate       = model.fg_v105_4_delta_gate        # (K_max,)
    calib_head_w     = model.fg_v105_4_calib_head_w      # (H, 1)
    calib_head_b     = model.fg_v105_4_calib_head_b      # (1,)

    # Component 3: projection waist params
    W_compress = model.fg_v105_4_W_compress  # (H, waist)
    b_compress = model.fg_v105_4_b_compress  # (waist,)
    W_expand   = model.fg_v105_4_W_expand    # (waist, H) — zero at init
    b_expand   = model.fg_v105_4_b_expand    # (H,)

    # Component 4 (v105.4): hierarchical IB codebook
    ib_codebook       = model.fg_v105_4_ib_codebook        # (N_CODE, H)
    family_centroids  = model.fg_v105_4_family_centroids    # (N_FAMILIES, H)
    leaf_to_family_oh = model.fg_v105_4_leaf_to_family_oh   # (N_CODE, N_FAMILIES) frozen
    delta_gate_quant  = model.fg_v105_4_delta_gate_quant   # (K_max,) — zero at init
    ib_temperature    = model.fg_v105_4_ib_temperature     # ()

    # NEW v105.4 — magnitude head
    magnitude_head_w   = model.fg_v105_4_magnitude_head_w    # (H, N_MAG)
    magnitude_head_b   = model.fg_v105_4_magnitude_head_b    # (N_MAG,)
    magnitude_centroids = model.fg_v105_4_magnitude_centroids # (N_MAG, H)

    B = int(digit_init.shape[0])
    T = n_max * n_digits + f_max
    H = int(var_pos_embed.shape[1])

    # Initial embedding: per-position digit codebook + RoPE on digit positions
    x = embed_factor_graph_v105_4(
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

    digit_logits_history     = []
    factor_logits_history    = []
    calib_history            = []
    magnitude_logits_history = []

    for k in range(K):
        # 1. Breath embedding
        be_k  = breath_embed[k].reshape(1, 1, -1).cast(x.dtype)
        x_in  = x + be_k
        x_pre = x

        # 2. Combined mask (B, N_HEADS, T, T)
        stk   = staging_mask[:, k, :, :]     # (B, T, T)
        stk_h = stk.reshape(B, 1, T, T).expand(B, V105_4_N_HEADS, T, T)
        combined = stk_h.cast(x.dtype) + head_op_mask.cast(x.dtype)

        # 3. Four transformer layers
        h = x_in
        for layer in layers[:4]:
            h = fg_layer_forward_v105_4(layer, h, combined)

        # 4. Hierarchical IB semantic codebook soft projection (v105.4 addition 3)
        h = apply_hierarchical_ib_codebook(
            h, ib_codebook, family_centroids, leaf_to_family_oh,
            ib_temperature, delta_gate_quant[k],
        )

        # 5. Projection waist compression  (Component 3)
        h = apply_projection_waist(h, W_compress, W_expand, b_compress, b_expand)

        # 6. Delta gate residual update
        gate_k = delta_gate[k].cast(h.dtype).reshape(1, 1, 1)
        delta  = h - x_pre
        x      = x_pre + gate_k * delta

        # 7. Per-cell magnitude head (NEW v105.4)
        x_ln = _layernorm(x, model.ln_f_g, model.ln_f_b, model.cfg.layer_norm_eps).cast(dtypes.float)
        n_var_tokens = n_max * n_digits
        var_tokens   = x_ln[:, :n_var_tokens, :]
        var_tokens_r = var_tokens.reshape(B, n_max, n_digits, -1)   # (B, N_MAX, N_DIGITS, H)

        # cell_hidden = mean over the 5 digit positions for that cell.
        cell_hidden = var_tokens_r.mean(axis=2)                      # (B, N_MAX, H)
        mh_w = magnitude_head_w.cast(dtypes.float)
        mh_b = magnitude_head_b.cast(dtypes.float)
        magnitude_logits = cell_hidden @ mh_w + mh_b.reshape(1, 1, -1)  # (B, N_MAX, N_MAG)
        magnitude_logits_history.append(magnitude_logits)

        magnitude_probs  = magnitude_logits.softmax(axis=-1)          # (B, N_MAX, N_MAG)
        mc = magnitude_centroids.cast(dtypes.float)                   # (N_MAG, H)
        magnitude_embed_cell = magnitude_probs @ mc                    # (B, N_MAX, H)
        # Broadcast magnitude_embed across all digit positions of each cell.
        magnitude_embed_dg = magnitude_embed_cell.reshape(
            B, n_max, 1, -1
        ).expand(B, n_max, n_digits, int(var_tokens_r.shape[-1]))     # (B, N_MAX, N_DIGITS, H)

        var_tokens_r = var_tokens_r + magnitude_embed_dg               # add magnitude_embed

        # 8a. Per-position digit codebook readout (v105.4 addition 2).
        # digit_codebook: (N_DIGITS, 10, H). For AR, per-iter codebook is digit_codebook[p].
        cb_fp_all = digit_codebook.cast(dtypes.float)  # (N_DIGITS, 10, H)

        if V105_4_AR_DIGITS:
            ar_logits_list: list[Tensor] = [None] * n_digits  # type: ignore
            cond_accum = Tensor.zeros(
                (B, n_max, int(x_ln.shape[-1])), dtype=dtypes.float
            ).contiguous()
            ar_cond_scale_t = Tensor(
                np.array([float(V105_4_AR_COND_SCALE)], dtype=np.float32),
                dtype=dtypes.float,
            ).reshape(1, 1, 1)

            if V105_4_AR_MSD_FIRST:
                ar_iter = range(n_digits - 1, -1, -1)
            else:
                ar_iter = range(n_digits)   # LSD-first default

            for p in ar_iter:
                cb_p = cb_fp_all[p]                                    # (10, H)
                pos_hidden = var_tokens_r[:, :, p, :] + cond_accum     # (B, N_MAX, H)
                pos_logits = pos_hidden @ cb_p.T                       # (B, N_MAX, 10)
                ar_logits_list[p] = pos_logits
                pos_probs = pos_logits.softmax(axis=-1)                 # (B, N_MAX, 10)
                pos_embed = pos_probs @ cb_p                           # (B, N_MAX, H)
                cond_accum = cond_accum + pos_embed * ar_cond_scale_t.cast(pos_embed.dtype)

            digit_logits_k = Tensor.stack(*ar_logits_list, dim=2)      # (B, N_MAX, N_DIGITS, 10)
        else:
            # Parallel — but with per-position codebook each iteration is independent.
            parallel_logits_list: list[Tensor] = []
            for p in range(n_digits):
                cb_p = cb_fp_all[p]                                    # (10, H)
                pos_hidden = var_tokens_r[:, :, p, :]                  # (B, N_MAX, H)
                pos_logits = pos_hidden @ cb_p.T                       # (B, N_MAX, 10)
                parallel_logits_list.append(pos_logits)
            digit_logits_k = Tensor.stack(*parallel_logits_list, dim=2)

        digit_logits_history.append(digit_logits_k)

        # 8b. Factor digit logits — also use per-position codebook.
        fac_tokens   = x_ln[:, n_var_tokens:n_var_tokens + f_max, :]
        # We don't add magnitude_embed for factor cells (factors are not variables
        # — they carry result digits but their cell_hidden derivation differs).
        fac_logits_list: list[Tensor] = []
        for p in range(n_digits):
            cb_p = cb_fp_all[p]                                        # (10, H)
            fac_logits_p = fac_tokens @ cb_p.T                          # (B, F_MAX, 10)
            fac_logits_list.append(fac_logits_p)
        fac_logits_k = Tensor.stack(*fac_logits_list, dim=2)           # (B, F_MAX, N_DIGITS, 10)
        factor_logits_history.append(fac_logits_k)

        # 8c. Calibration head
        pool        = x_ln.mean(axis=1)
        calib_logit = pool @ calib_head_w.cast(dtypes.float) + calib_head_b.cast(dtypes.float)
        calib_k     = calib_logit.reshape(-1).sigmoid()
        calib_history.append(calib_k)

    return digit_logits_history, factor_logits_history, calib_history, magnitude_logits_history


# ---------------------------------------------------------------------------
# Parameter attachment
# ---------------------------------------------------------------------------

def attach_fg_params_v105_4(
    model: Any,
    hidden: int,
    n_digits: int = V105_4_N_DIGITS,
    n_max: int = V105_4_N_MAX,
    f_max: int = V105_4_F_MAX,
    k_max: int | None = None,
    waist: int | None = None,
    n_code: int | None = None,
    rope_base: float | None = None,
    ib_centroids_path: str | None = None,
    ib_tree_path: str | None = None,
    ib_init: bool | None = None,
    waist_lora_init: bool | None = None,
    n_families: int = V105_4_N_FAMILIES,
    n_magnitude: int = V105_4_N_MAGNITUDE,
) -> None:
    """Attach v105.4 params to model.

    v105.4 additions vs v105.3:
      • per-position digit_codebook (n_digits, 10, hidden)
      • family_centroids (n_families, hidden) initialized from IB tree means
      • leaf_to_family_oh (n_code, n_families) frozen one-hot lookup
      • magnitude_head_w (hidden, n_magnitude)
      • magnitude_head_b (n_magnitude,) — zeros
      • magnitude_centroids (n_magnitude, hidden) — small random init

    All v105.4 additions are wired so that initial behavior reduces to v105.3
    where possible. The hierarchical IB and magnitude head produce non-zero
    output at step 0 (cannot trivially zero-out since they're additive in
    the residual stream), but their effect is small because:
      - delta_gate_quant = 0  → hierarchical IB has zero residual contribution
      - magnitude_centroids ≈ 0.02 std → magnitude_embed is small
    """
    if k_max is None:
        k_max = V105_4_K_MAX
    if waist is None:
        waist = V105_4_WAIST
    if n_code is None:
        n_code = V105_4_CODEBOOK_N
    if rope_base is None:
        rope_base = V105_4_ROPE_BASE
    if ib_centroids_path is None:
        ib_centroids_path = V105_4_IB_CENTROIDS
    if ib_tree_path is None:
        ib_tree_path = V105_4_IB_TREE
    if ib_init is None:
        ib_init = V105_4_IB_INIT
    if waist_lora_init is None:
        waist_lora_init = V105_4_WAIST_LORA_INIT

    rng = np.random.RandomState(20013)

    # -----------------------------------------------------------------------
    # Components 1+2: PER-POSITION digit codebook (v105.4 addition 2)
    # + frozen RoPE tables (LSD layout)
    # -----------------------------------------------------------------------
    # Build n_digits independent QR-orthogonal codebooks, each (10, hidden).
    dc_per_pos = np.zeros((n_digits, 10, hidden), dtype=np.float32)
    for p in range(n_digits):
        if V105_4_FOURIER_INIT:
            dc_per_pos[p] = _fourier_digit_codebook(n_digits=10, hidden=hidden)
        else:
            rng_p = np.random.RandomState(20013 + p)
            raw_dc = rng_p.randn(max(hidden, 10), hidden).astype(np.float32)
            q_dc, _ = np.linalg.qr(raw_dc)
            dc_per_pos[p] = q_dc[:10].astype(np.float32) * 1.0
    model.fg_v105_4_digit_codebook = Tensor(dc_per_pos, dtype=dtypes.float).contiguous()

    # LSD-first array layout: array index i ↔ RoPE position i naturally.
    cos_t, sin_t = _precompute_digit_rope(n_digits, hidden, base=rope_base)
    model.fg_v105_4_digit_rope_cos = Tensor(cos_t, dtype=dtypes.float).contiguous()
    model.fg_v105_4_digit_rope_sin = Tensor(sin_t, dtype=dtypes.float).contiguous()

    # -----------------------------------------------------------------------
    # Position / kind embeddings
    # -----------------------------------------------------------------------
    vp = rng.randn(n_max, hidden).astype(np.float32) * 0.02
    model.fg_v105_4_var_pos_embed = Tensor(vp, dtype=dtypes.float).contiguous()

    fp_emb = rng.randn(f_max, hidden).astype(np.float32) * 0.02
    model.fg_v105_4_factor_pos_embed = Tensor(fp_emb, dtype=dtypes.float).contiguous()

    nk = rng.randn(3, hidden).astype(np.float32) * 0.02
    model.fg_v105_4_node_kind_embed = Tensor(nk, dtype=dtypes.float).contiguous()

    # -----------------------------------------------------------------------
    # Calibration head
    # -----------------------------------------------------------------------
    cw = (rng.randn(hidden, 1) * 0.02).astype(np.float32)
    model.fg_v105_4_calib_head_w = Tensor(cw, dtype=dtypes.float).contiguous()
    model.fg_v105_4_calib_head_b = Tensor.zeros((1,), dtype=dtypes.float).contiguous()

    # -----------------------------------------------------------------------
    # Per-breath embeddings + delta gate
    # -----------------------------------------------------------------------
    rng_be = np.random.RandomState(20014)
    raw_be = rng_be.randn(max(k_max, hidden), hidden).astype(np.float32)
    q_be, _ = np.linalg.qr(raw_be)
    be = q_be[:k_max].astype(np.float32) * 0.5
    model.fg_v105_4_breath_embed = Tensor(be, dtype=dtypes.float).contiguous()
    model.fg_v105_4_delta_gate   = Tensor.ones((k_max,), dtype=dtypes.float).contiguous()

    # -----------------------------------------------------------------------
    # Component 3: Projection waist  (LoRA-style)
    # -----------------------------------------------------------------------
    W_c = (rng.randn(hidden, waist) * 0.02).astype(np.float32)
    model.fg_v105_4_W_compress = Tensor(W_c, dtype=dtypes.float).contiguous()
    model.fg_v105_4_b_compress = Tensor.zeros((waist,), dtype=dtypes.float).contiguous()

    if waist_lora_init:
        model.fg_v105_4_W_expand = Tensor.zeros((waist, hidden), dtype=dtypes.float).contiguous()
    else:
        We = (rng.randn(waist, hidden) * 0.02).astype(np.float32)
        model.fg_v105_4_W_expand = Tensor(We, dtype=dtypes.float).contiguous()
    model.fg_v105_4_b_expand = Tensor.zeros((hidden,), dtype=dtypes.float).contiguous()

    # -----------------------------------------------------------------------
    # Component 4 (v105.4): hierarchical IB semantic codebook
    # -----------------------------------------------------------------------
    if ib_init:
        cb_np = load_ib_centroids(ib_centroids_path, n_code=n_code, hidden=hidden)
    else:
        raw_cb = rng.randn(max(hidden, n_code), hidden).astype(np.float32)
        q_cb, _ = np.linalg.qr(raw_cb)
        cb_np = q_cb[:n_code].astype(np.float32) * 0.5
        print(f"[v105.4] IB init disabled; random QR codebook ({n_code}, {hidden})")

    model.fg_v105_4_ib_codebook = Tensor(cb_np, dtype=dtypes.float).contiguous()
    model.fg_v105_4_delta_gate_quant = Tensor.zeros((k_max,), dtype=dtypes.float).contiguous()
    model.fg_v105_4_ib_temperature   = Tensor(
        np.array([1.0], dtype=np.float32), dtype=dtypes.float
    ).contiguous()

    # Build leaf_to_family from IB tree JSON.
    l2f_np, leaf_ids = _load_leaf_to_family(
        ib_tree_path, n_code=n_code, n_families=n_families
    )
    # Frozen int → frozen one-hot float for matmul-style gather.
    l2f_oh = np.zeros((n_code, n_families), dtype=np.float32)
    for i in range(n_code):
        l2f_oh[i, int(l2f_np[i])] = 1.0
    model.fg_v105_4_leaf_to_family    = Tensor(l2f_np, dtype=dtypes.int).contiguous()
    model.fg_v105_4_leaf_to_family_oh = Tensor(l2f_oh, dtype=dtypes.float).contiguous()

    # family_centroids = mean of IB centroids in each family.
    family_centroids = np.zeros((n_families, hidden), dtype=np.float32)
    counts = np.zeros((n_families,), dtype=np.float32)
    for i in range(n_code):
        fam = int(l2f_np[i])
        family_centroids[fam] += cb_np[i]
        counts[fam] += 1.0
    for f in range(n_families):
        if counts[f] > 0:
            family_centroids[f] /= counts[f]
        else:
            family_centroids[f] = rng.randn(hidden).astype(np.float32) * 0.02
    model.fg_v105_4_family_centroids = Tensor(family_centroids, dtype=dtypes.float).contiguous()

    # -----------------------------------------------------------------------
    # NEW v105.4 — magnitude head + magnitude_centroids
    # -----------------------------------------------------------------------
    mh_w = (rng.randn(hidden, n_magnitude) * 0.02).astype(np.float32)
    model.fg_v105_4_magnitude_head_w = Tensor(mh_w, dtype=dtypes.float).contiguous()
    model.fg_v105_4_magnitude_head_b = Tensor.zeros((n_magnitude,), dtype=dtypes.float).contiguous()
    mag_c = (rng.randn(n_magnitude, hidden) * 0.02).astype(np.float32)
    model.fg_v105_4_magnitude_centroids = Tensor(mag_c, dtype=dtypes.float).contiguous()

    T = n_max * n_digits + f_max
    print(
        f"[v105.4] params attached:\n"
        f"  digit_codebook=(N_DIGITS={n_digits}, 10, {hidden}) PER-POSITION "
        f"init={'FOURIER' if V105_4_FOURIER_INIT else 'QR-random'}, "
        f"digit_rope (N_DIGITS={n_digits}, H={hidden}, base={rope_base:.0f}) [FROZEN]\n"
        f"  LSD layout: idx 0=ones (RoPE pos 0)\n"
        f"  loss_mode={'NUMBER_MSE_ONLY' if V105_4_NUMBER_MSE_ONLY else 'per-digit CE'}\n"
        f"  ar_digits={V105_4_AR_DIGITS} (cond_scale={V105_4_AR_COND_SCALE if V105_4_AR_DIGITS else 'N/A'}, "
        f"dir={'MSD-first' if V105_4_AR_MSD_FIRST else 'LSD-first'})\n"
        f"  var_pos_embed=({n_max},{hidden}), factor_pos_embed=({f_max},{hidden})\n"
        f"  waist=({hidden}→{waist}→{hidden}), W_expand={'ZEROS' if waist_lora_init else 'random'}\n"
        f"  ib_codebook=({n_code},{hidden}), family_centroids=({n_families},{hidden})\n"
        f"  leaf_to_family[:8]={l2f_np[:8].tolist()}\n"
        f"  leaf_ids[:4]={leaf_ids[:4]}\n"
        f"  magnitude_head=({hidden},{n_magnitude}), centroids=({n_magnitude},{hidden})\n"
        f"  delta_gate_quant=ZEROS, ib_init={ib_init}\n"
        f"  T={T} (N_MAX*N_DIGITS+F_MAX={n_max}*{n_digits}+{f_max}), K_max={k_max}",
        flush=True,
    )


def fg_v105_4_parameters(model: Any) -> list[Tensor]:
    """Trainable v105.4 factor-graph-specific params.

    Deliberately excludes frozen tables:
      - digit_rope_cos, digit_rope_sin  (precomputed)
      - leaf_to_family, leaf_to_family_oh  (frozen mapping)

    Includes all learnable params across the four components + magnitude head.
    """
    return [
        # Components 1+2
        model.fg_v105_4_digit_codebook,     # (N_DIGITS, 10, H) per-position
        model.fg_v105_4_var_pos_embed,
        model.fg_v105_4_factor_pos_embed,
        model.fg_v105_4_node_kind_embed,
        model.fg_v105_4_calib_head_w,
        model.fg_v105_4_calib_head_b,
        model.fg_v105_4_breath_embed,
        model.fg_v105_4_delta_gate,
        # Component 3: projection waist
        model.fg_v105_4_W_compress,
        model.fg_v105_4_b_compress,
        model.fg_v105_4_W_expand,
        model.fg_v105_4_b_expand,
        # Component 4 (v105.4): hierarchical IB codebook
        model.fg_v105_4_ib_codebook,
        model.fg_v105_4_family_centroids,
        model.fg_v105_4_delta_gate_quant,
        model.fg_v105_4_ib_temperature,
        # NEW v105.4: magnitude head + centroids
        model.fg_v105_4_magnitude_head_w,
        model.fg_v105_4_magnitude_head_b,
        model.fg_v105_4_magnitude_centroids,
    ]


def fg_v105_4_state_dict(model: Any) -> dict[str, Tensor]:
    """Full state dict (frozen tables included for checkpoint self-containment)."""
    return {
        # Components 1+2
        "fg_v105_4.digit_codebook":   model.fg_v105_4_digit_codebook,  # (N_DIGITS, 10, H)
        "fg_v105_4.digit_rope_cos":   model.fg_v105_4_digit_rope_cos,   # frozen
        "fg_v105_4.digit_rope_sin":   model.fg_v105_4_digit_rope_sin,   # frozen
        "fg_v105_4.var_pos_embed":    model.fg_v105_4_var_pos_embed,
        "fg_v105_4.factor_pos_embed": model.fg_v105_4_factor_pos_embed,
        "fg_v105_4.node_kind_embed":  model.fg_v105_4_node_kind_embed,
        "fg_v105_4.calib_head_w":     model.fg_v105_4_calib_head_w,
        "fg_v105_4.calib_head_b":     model.fg_v105_4_calib_head_b,
        "fg_v105_4.breath_embed":     model.fg_v105_4_breath_embed,
        "fg_v105_4.delta_gate":       model.fg_v105_4_delta_gate,
        # Component 3
        "fg_v105_4.W_compress":       model.fg_v105_4_W_compress,
        "fg_v105_4.b_compress":       model.fg_v105_4_b_compress,
        "fg_v105_4.W_expand":         model.fg_v105_4_W_expand,
        "fg_v105_4.b_expand":         model.fg_v105_4_b_expand,
        # Component 4 (v105.4): hierarchical IB
        "fg_v105_4.ib_codebook":          model.fg_v105_4_ib_codebook,
        "fg_v105_4.family_centroids":     model.fg_v105_4_family_centroids,
        "fg_v105_4.leaf_to_family":       model.fg_v105_4_leaf_to_family,       # frozen
        "fg_v105_4.leaf_to_family_oh":    model.fg_v105_4_leaf_to_family_oh,    # frozen
        "fg_v105_4.delta_gate_quant":     model.fg_v105_4_delta_gate_quant,
        "fg_v105_4.ib_temperature":       model.fg_v105_4_ib_temperature,
        # NEW v105.4: magnitude head + centroids
        "fg_v105_4.magnitude_head_w":     model.fg_v105_4_magnitude_head_w,
        "fg_v105_4.magnitude_head_b":     model.fg_v105_4_magnitude_head_b,
        "fg_v105_4.magnitude_centroids":  model.fg_v105_4_magnitude_centroids,
    }


# ---------------------------------------------------------------------------
# Warm-start from v104 backbone checkpoint (or v105.1.2)
# ---------------------------------------------------------------------------

def load_ckpt_v105_4(model: Any, path: str) -> None:
    """Load backbone (shared.*, phase*.*, ln_f.*) from any v104-family checkpoint.

    v105.3-specific params (digit_codebook, waist, ib_codebook, RoPE tables,
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
    if "fg_v104.codebook" in sd and hasattr(model, "fg_v105_4_ib_codebook"):
        src = sd["fg_v104.codebook"]
        dst = model.fg_v105_4_ib_codebook
        if src.shape == dst.shape:
            dst.assign(src.to(dst.device).cast(dst.dtype)).realize()
            print(f"  copied fg_v104.codebook → fg_v105_4.ib_codebook", flush=True)

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

_JIT_V105_4_CACHE: dict = {}


def _compile_jit_fg_step_v105_4(
    model: Any,
    opt: Any,
    K: int,
    B: int,
    factor_aux_weight: float = V105_4_FACTOR_AUX_WEIGHT,
    calib_weight: float = V105_4_CALIB_WEIGHT,
    energy_weight: float = V105_4_ENERGY_WEIGHT,
    magnitude_weight: float = V105_4_MAGNITUDE_WEIGHT,
    n_max: int = V105_4_N_MAX,
    f_max: int = V105_4_F_MAX,
    n_digits: int = V105_4_N_DIGITS,
    n_magnitude: int = V105_4_N_MAGNITUDE,
    grad_clip: float = 1.0,
):
    """Compile a TinyJit'd train step for v105.4.

    Loss structure (additions marked NEW):
      var_loss         — per-breath weighted CE on unobserved variable digits
      magnitude_loss   — NEW: per-breath weighted CE on 4-way magnitude class
      factor_aux_loss  — per-NUMBER relative MSE on reconstructed value
                         multiplied by SOFT-magnitude-derived valid mask
      calib_loss       — MSE calibration head
      energy_loss      — expected-value constraint energy (LSD place values)
    """
    n_code = int(model.fg_v105_4_ib_codebook.shape[0])
    key = ("v105_4", id(model), id(opt), int(K), int(B),
           float(factor_aux_weight), float(calib_weight), float(energy_weight),
           float(magnitude_weight),
           int(n_max), int(f_max), int(n_digits), int(n_magnitude),
           float(grad_clip), int(n_code))
    if key in _JIT_V105_4_CACHE:
        return _JIT_V105_4_CACHE[key]

    aw     = float(calib_weight)
    fw     = float(factor_aux_weight)
    ew     = float(energy_weight)
    mw     = float(magnitude_weight)
    gc     = float(grad_clip)
    params = opt.params

    print(
        f"[JIT] compile v105.4 fg step: K={K} B={B} n_digits={n_digits} "
        f"T={n_max * n_digits + f_max} aw={aw} fw={fw} ew={ew} mw={mw} gc={gc} "
        f"n_code={n_code}...",
        flush=True,
    )

    # Build the (4, n_digits) class_to_valid mapping for the soft mask:
    #   class 0 (1-digit):  [1, 0, 0, 0, 0]
    #   class 1 (2-digit):  [1, 1, 0, 0, 0]
    #   class 2 (3-digit):  [1, 1, 1, 0, 0]
    #   class 3 (4-digit):  [1, 1, 1, 1, 0]
    c2v_np = np.zeros((n_magnitude, n_digits), dtype=np.float32)
    for cls in range(n_magnitude):
        n_valid_for_cls = min(cls + 1, n_digits)
        c2v_np[cls, :n_valid_for_cls] = 1.0
    class_to_valid_t = Tensor(c2v_np, dtype=dtypes.float).contiguous()

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
        digit_valid_mask: Tensor,           # (B, N_MAX, N_DIGITS) float
        factor_digit_valid_mask: Tensor,    # (B, F_MAX, N_DIGITS) float
        magnitude_target: Tensor,           # (B, N_MAX) int  — gold magnitude class
    ):
        opt.zero_grad()

        (digit_logits_history, factor_logits_history,
         calib_history, magnitude_logits_history) = \
            fg_breathing_forward_v105_4(
                model, digit_init, node_kinds, staging_mask, head_op_mask,
                K=K, n_max=n_max, f_max=f_max, n_digits=n_digits,
            )

        # --- Main loss on unobserved variables ---
        # LSD layout: valid positions are the LEADING n_actual_digits per variable.
        # The trailing positions are leading-zero padding above the most-significant
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

        # LSD place values for number reconstruction: [10^0, 10^1, ..., 10^(N-1)]
        _place_values_np_var = [float(10 ** i) for i in range(n_digits)]
        _place_values_t_var  = Tensor(_place_values_np_var, dtype=dtypes.float).reshape(1, 1, n_digits)
        _digit_vals_t_var    = Tensor([float(i) for i in range(10)], dtype=dtypes.float)

        if V105_4_NUMBER_MSE_ONLY:
            # ============================================================
            # NUMBER-ONLY LOSS PATH (V105_4_NUMBER_MSE_ONLY=1)
            # ============================================================
            _var_gold_float    = gold_digits.cast(dtypes.float)         # (B, N_MAX, N_DIGITS)
            _var_gold_masked   = _var_gold_float * digit_valid_mask
            var_gold_numbers   = (_var_gold_masked * _place_values_t_var).sum(axis=-1)  # (B, N_MAX)
            var_rel_denom      = var_gold_numbers.abs() + 1.0                            # (B, N_MAX)

            is_real_var_loss   = (digit_valid_mask.sum(axis=-1) > 0).cast(dtypes.float)  # (B, N_MAX)
            unobs_real_var     = unobs_float * is_real_var_loss                          # (B, N_MAX)
            n_unobs_real_var   = unobs_real_var.sum() + 1e-8

            for k, dig_logits in enumerate(digit_logits_history):
                weight_k    = 1.0 + float(k) / float(max(K - 1, 1))
                probs_k     = dig_logits.softmax(axis=-1)               # (B, N_MAX, N_DIGITS, 10)
                exp_digit_k = (probs_k * _digit_vals_t_var.reshape(1, 1, 1, 10)).sum(axis=-1)
                exp_digit_m = exp_digit_k * digit_valid_mask             # mask invalid positions
                pred_number = (exp_digit_m * _place_values_t_var).sum(axis=-1)            # (B, N_MAX)
                rel_err  = ((pred_number - var_gold_numbers) / var_rel_denom).clip(-5.0, 5.0)
                sq_err   = rel_err * rel_err
                sq_err_m = sq_err * unobs_real_var.cast(sq_err.dtype)
                ce_k     = sq_err_m.sum() / n_unobs_real_var

                per_breath_ce_t.append(ce_k)
                var_loss_sum   = var_loss_sum + ce_k * weight_k
                var_weight_sum += weight_k
        else:
            # ============================================================
            # PER-DIGIT CE PATH (default behavior)
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

        # --- Magnitude loss (NEW v105.4) — per-breath weighted 4-way CE ---
        is_real_var = (digit_valid_mask.sum(axis=-1) > 0).cast(dtypes.float)    # (B,N)
        unobs_real  = unobs_float * is_real_var                                  # (B,N)
        n_unobs_real = unobs_real.sum() + 1e-8

        mag_target_int = magnitude_target.cast(dtypes.int).clip(0, n_magnitude - 1)
        mag_target_flat = mag_target_int.reshape(B * n_max)

        mag_loss_sum   = Tensor.zeros((), dtype=dtypes.float).contiguous()
        mag_weight_sum = 0.0
        for k, mag_logits_k in enumerate(magnitude_logits_history):
            weight_k    = 1.0 + float(k) / float(max(K - 1, 1))
            mlogits_flat = mag_logits_k.reshape(B * n_max, n_magnitude)
            log_probs    = mlogits_flat.log_softmax(axis=-1)
            tgt_oh       = mag_target_flat.one_hot(n_magnitude).cast(log_probs.dtype)
            nll          = -(log_probs * tgt_oh).sum(axis=-1)            # (B*N,)
            unobs_flat   = unobs_real.reshape(B * n_max)
            masked_nll   = nll * unobs_flat.cast(nll.dtype)
            ce_k         = masked_nll.sum() / n_unobs_real
            mag_loss_sum = mag_loss_sum + ce_k * weight_k
            mag_weight_sum += weight_k
        magnitude_loss = mag_loss_sum / float(mag_weight_sum)

        # --- Per-NUMBER factor auxiliary loss (relative MSE) ---
        # LSD-first place values: index 0 = 10^0, ..., index N-1 = 10^(N-1).
        # v105.4 addition 4: multiply factor_digit_valid_mask by a soft mask
        # derived from the FINAL-breath magnitude prediction over the RESULT
        # variable of each factor. The resulting mask is per-factor.
        n_valid_fac  = factor_valid.sum() + 1e-8

        # LSD place values: [10^0, 10^1, ..., 10^(N-1)]
        place_values_np = [float(10 ** i) for i in range(n_digits)]
        place_values_t  = Tensor(place_values_np, dtype=dtypes.float).reshape(1, 1, n_digits)

        digit_vals_t    = Tensor([float(i) for i in range(10)], dtype=dtypes.float)

        gold_dg_float   = factor_gold_dg.cast(dtypes.float)                       # (B, F, D)
        gold_dg_masked  = gold_dg_float * factor_digit_valid_mask                  # zero invalid
        gold_numbers    = (gold_dg_masked * place_values_t).sum(axis=-1)          # (B, F)
        rel_denom       = gold_numbers.abs() + 1.0                                 # (B, F)

        # Build per-factor SOFT valid mask from final-breath magnitude over the
        # result variable.  factor_args[:, :, 2] is the result idx; gather the
        # magnitude_softmax of the corresponding variable, then mix with class_to_valid.
        final_mag_logits = magnitude_logits_history[-1]                           # (B, N_MAX, N_MAG)
        final_mag_probs  = final_mag_logits.softmax(axis=-1)                       # (B, N_MAX, N_MAG)
        res_idx          = factor_args[:, :, 2].cast(dtypes.int).clip(0, n_max - 1) # (B, F_MAX)
        # Gather via one-hot: (B, F_MAX, N_MAX) @ (B, N_MAX, N_MAG) → (B, F_MAX, N_MAG)
        res_oh = res_idx.one_hot(n_max).cast(dtypes.float)                        # (B, F_MAX, N_MAX)
        fac_mag_probs = res_oh @ final_mag_probs                                   # (B, F_MAX, N_MAG)
        # class_to_valid: (N_MAG, N_DIGITS)
        fac_soft_valid = fac_mag_probs @ class_to_valid_t.cast(fac_mag_probs.dtype) # (B, F_MAX, N_DIGITS)
        # Combine soft + gold mask (multiplicative) — gradient on magnitude head
        # comes through fac_soft_valid; gold mask still enforces ground truth.
        fac_combined_mask = factor_digit_valid_mask * fac_soft_valid               # (B, F_MAX, N_DIGITS)

        factor_aux_sum   = Tensor.zeros((), dtype=dtypes.float).contiguous()
        factor_aux_w_sum = 0.0

        for k_aux, fac_logits_k in enumerate(factor_logits_history):
            w_k_aux = 1.0 + float(k_aux) / float(max(K - 1, 1))

            fac_probs   = fac_logits_k.softmax(axis=-1)                            # (B,F,D,10)
            exp_digit   = (fac_probs * digit_vals_t.reshape(1, 1, 1, 10)).sum(axis=-1)
            exp_digit_m = exp_digit * fac_combined_mask                            # SOFT-masked
            pred_number = (exp_digit_m * place_values_t).sum(axis=-1)

            rel_err  = ((pred_number - gold_numbers) / rel_denom).clip(-5.0, 5.0)
            sq_err   = rel_err * rel_err
            sq_err_m = sq_err * factor_valid.cast(sq_err.dtype)
            fac_ce_k = sq_err_m.sum() / n_valid_fac

            factor_aux_sum   = factor_aux_sum + fac_ce_k * w_k_aux
            factor_aux_w_sum += w_k_aux

        factor_aux_loss = factor_aux_sum / float(factor_aux_w_sum)

        # --- Constraint energy (LSD place values) ---
        final_dig_logits = digit_logits_history[-1]
        energy_loss = constraint_energy_v105_4(
            final_dig_logits, factor_types, factor_args,
            n_max=n_max, f_max=f_max, n_digits=n_digits,
        )

        # --- Calibration ---
        final_pred_dg = digit_logits_history[-1].argmax(axis=-1).detach()           # (B,N,D)
        dg_eq    = (final_pred_dg == gold_digits.cast(dtypes.int)).cast(dtypes.float)
        dg_match_or_invalid = dg_eq * digit_valid_mask + (1.0 - digit_valid_mask)
        var_eq   = dg_match_or_invalid.min(axis=-1)                                 # (B,N)
        eq_unobs    = var_eq * unobs_real
        n_unobs_per = unobs_real.sum(axis=-1) + 1e-8
        correct     = eq_unobs.sum(axis=-1) / n_unobs_per

        calib_loss_sum = Tensor.zeros((), dtype=dtypes.float).contiguous()
        for k, calib_k in enumerate(calib_history):
            prog       = float(k) / float(max(K - 1, 1))
            target_k   = 0.5 + (correct - 0.5) * prog
            calib_loss_sum = calib_loss_sum + ((calib_k - target_k) ** 2).mean()
        calib_loss = calib_loss_sum / float(K)

        # --- Magnitude accuracy diagnostic ---
        final_mag_pred = magnitude_logits_history[-1].argmax(axis=-1).detach()      # (B, N_MAX)
        mag_eq         = (final_mag_pred == magnitude_target.cast(dtypes.int)).cast(dtypes.float)
        mag_eq_unobs   = mag_eq * unobs_real
        mag_acc        = (mag_eq_unobs.sum() / n_unobs_real).detach()

        # --- Metrics ---
        cell_acc  = (eq_unobs.sum() / (unobs_real.sum() + 1e-8)).detach()
        query_acc = correct.mean().detach()

        # --- Total loss ---
        total_ce = (
            var_loss
            + mw * magnitude_loss
            + fw * factor_aux_loss
            + aw * calib_loss
            + ew * energy_loss
        )
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
            magnitude_loss.realize(),
            mag_acc.realize(),
            cell_acc.realize(),
            query_acc.realize(),
            *(ce.realize() for ce in per_breath_ce_t),
        )

    _JIT_V105_4_CACHE[key] = _step
    print(
        f"[JIT] v105.4 fg step ready (cache={len(_JIT_V105_4_CACHE)}); "
        f"first call compiles...",
        flush=True,
    )
    return _step


def _compile_jit_fg_eval_v105_4(
    model: Any,
    K: int,
    B: int,
    n_max: int = V105_4_N_MAX,
    f_max: int = V105_4_F_MAX,
    n_digits: int = V105_4_N_DIGITS,
):
    """Compile a TinyJit'd eval step (forward only, no gradient).

    Takes digit_valid_mask so cell_acc treats invalid (leading-zero padding)
    positions as automatically correct (consistent with the train loss).
    """
    key = ("eval_v105_4", id(model), int(K), int(B), int(n_max), int(f_max), int(n_digits))
    if key in _JIT_V105_4_CACHE:
        return _JIT_V105_4_CACHE[key]

    print(f"[JIT] compile v105.4 fg eval: K={K} B={B} n_digits={n_digits}...", flush=True)

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
        digit_logits_history, _, _, _ = fg_breathing_forward_v105_4(
            model, digit_init, node_kinds, staging_mask, head_op_mask,
            K=K, n_max=n_max, f_max=f_max, n_digits=n_digits,
        )
        final_logits  = digit_logits_history[-1]
        pred_dg       = final_logits.argmax(axis=-1)
        dg_eq         = (pred_dg == gold_digits.cast(pred_dg.dtype)).cast(dtypes.float)
        dg_or_invalid = dg_eq * digit_valid_mask + (1.0 - digit_valid_mask)
        var_eq        = dg_or_invalid.min(axis=-1)
        is_real_var   = (digit_valid_mask.sum(axis=-1) > 0).cast(dtypes.float)
        unobs_real    = (1 - observed_mask.cast(dtypes.float)) * is_real_var
        cell_acc      = (var_eq * unobs_real).sum() / (unobs_real.sum() + 1e-8)
        return pred_dg.realize(), cell_acc.realize()

    _JIT_V105_4_CACHE[key] = _eval
    print(f"[JIT] v105.4 eval ready (cache={len(_JIT_V105_4_CACHE)})", flush=True)
    return _eval
