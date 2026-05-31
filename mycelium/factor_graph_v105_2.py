"""v105.2 factor graph breathing transformer — LSD-first digit encoding + valid mask.

Identical to v105.1.2 EXCEPT:

  1. LSD-FIRST DIGIT ENCODING
     - Position 0 = ones place (ALWAYS valid for any non-negative integer).
     - Position p = 10^p place.
     - Place values used in factor_aux and constraint_energy are [1, 10, 100, ...]
       (instead of v105.1.2's MSD-first [10000, 1000, ..., 1]).

  2. DIGIT VALID MASK
     - var_loss and factor_aux both mask out positions that aren't part of the
       number's natural representation (i.e. leading-zero padding above the
       most-significant digit).  Example for value=7 with n_digits=5:
       valid_mask = [1, 0, 0, 0, 0]  (only the ones place contributes to loss).
     - This eliminates the v105.1.2 collapse mode where the model learned
       "always predict 0" for the 3 trivially-zero MSB positions in [0,99] data.

  3. NEW CONSTRAINT ENERGY (LSD-first place values)
     - constraint_energy_v105_2 uses LSD place values [10^0, 10^1, ...].

Everything else (4 phase layers, K=8, RoPE on digit positions, IB codebook,
projection waist, delta_gate zero-init for stages) is preserved.

LoRA invariant: step-0 forward equals v105.1 backbone behavior (both the waist
and the IB codebook contribute zero to h at step 0).  Warm-start from v104
checkpoint copies backbone + codebook.

Env var gates:
  V105_2_TASK=1                — enable v105.2 forward path
  V105_2_K_MAX=8               — iterative-prefill breaths
  V105_2_N_DIGITS=5            — digit positions per variable
  V105_2_N_MAX=16              — max variable nodes
  V105_2_F_MAX=8               — max factor nodes
  V105_2_WAIST=512             — projection waist dimension
  V105_2_CODEBOOK_N=32         — IB semantic codebook entries
  V105_2_IB_CENTROIDS=.cache/ib_centroids_gsm8k_partial.npz
  V105_2_ENERGY_WEIGHT=0.01    — constraint energy weight
  V105_2_CALIB_WEIGHT=0.05     — calibration loss weight
  V105_2_FACTOR_AUX_WEIGHT=1.0 — per-NUMBER aux loss weight
  V105_2_ROPE_BASE=10000       — digit RoPE base theta
  V105_2_IB_INIT=1             — use IB centroids
  V105_2_WAIST_LORA_INIT=1     — zero-init W_expand
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

# Re-use op-id constants and RoPE helpers from prior versions.
from mycelium.factor_graph_v105 import (
    OP_ADD, OP_SUB, OP_MUL, OP_DIV,
)
from mycelium.factor_graph_v105_1 import (
    _precompute_digit_rope,
    apply_rope_digit_tg,
)
# IB centroid loader reused from v104.
from mycelium.factor_graph_v104 import load_ib_centroids

# LSD-first digit utilities (the only data-side change in v105.2).
from mycelium.factor_graph_data_v105_2 import (
    value_to_digits_lsd, digits_to_value_lsd,
)


__all__ = [
    "OP_ADD", "OP_SUB", "OP_MUL", "OP_DIV",
    "value_to_digits_lsd", "digits_to_value_lsd",
]

# ---------------------------------------------------------------------------
# Env vars
# ---------------------------------------------------------------------------

V105_2_TASK             = int(os.environ.get("V105_2_TASK",             "0")) > 0
V105_2_K_MAX            = int(os.environ.get("V105_2_K_MAX",            "8"))
V105_2_N_DIGITS         = int(os.environ.get("V105_2_N_DIGITS",         "5"))
V105_2_N_MAX            = int(os.environ.get("V105_2_N_MAX",            "16"))
V105_2_F_MAX            = int(os.environ.get("V105_2_F_MAX",             "8"))
V105_2_N_HEADS          = 16   # fixed: Pythia-410M
V105_2_WAIST            = int(os.environ.get("V105_2_WAIST",            "512"))
V105_2_CODEBOOK_N       = int(os.environ.get("V105_2_CODEBOOK_N",       "32"))
V105_2_IB_CENTROIDS     = os.environ.get(
    "V105_2_IB_CENTROIDS", ".cache/ib_centroids_gsm8k_partial.npz"
)
V105_2_ENERGY_WEIGHT    = float(os.environ.get("V105_2_ENERGY_WEIGHT",   "0.01"))
V105_2_CALIB_WEIGHT     = float(os.environ.get("V105_2_CALIB_WEIGHT",    "0.05"))
V105_2_FACTOR_AUX_WEIGHT = float(os.environ.get("V105_2_FACTOR_AUX_WEIGHT", "1.0"))
V105_2_ROPE_BASE        = float(os.environ.get("V105_2_ROPE_BASE",       "10000.0"))
V105_2_IB_INIT          = int(os.environ.get("V105_2_IB_INIT",           "1")) > 0
V105_2_WAIST_LORA_INIT  = int(os.environ.get("V105_2_WAIST_LORA_INIT",   "1")) > 0


# ---------------------------------------------------------------------------
# LSD-first expected-value energy (mirrors _expected_value_v105 but LSD-place-values)
# ---------------------------------------------------------------------------

def _expected_value_v105_2(digit_logits: Tensor, n_digits: int) -> Tensor:
    """Expected integer value from per-digit logit distributions, LSD-first.

    digit_logits: (..., N_DIGITS, 10)
    Returns: (...,) float — expected value = Σ_p E[digit_p] × 10^p
    (Note: place value at index p is 10^p in LSD-first encoding.)
    """
    probs   = digit_logits.softmax(-1)
    d_vals  = Tensor(np.arange(10, dtype=np.float32)).cast(probs.dtype)
    exp_dig = (probs * d_vals).sum(axis=-1)
    place_vals_np = np.array([10 ** p for p in range(n_digits)], dtype=np.float32)  # LSD
    place_vals = Tensor(place_vals_np).cast(exp_dig.dtype)
    return (exp_dig * place_vals).sum(axis=-1)


def constraint_energy_v105_2(
    digit_logits_final: Tensor,   # (B, N_MAX, N_DIGITS, 10)
    factor_types: Tensor,          # (B, F_MAX) int
    factor_args: Tensor,           # (B, F_MAX, 3) int
    n_max: int = V105_2_N_MAX,
    f_max: int = V105_2_F_MAX,
    n_digits: int = V105_2_N_DIGITS,
) -> Tensor:
    """Expected-value constraint energy for v105.2 (LSD-first place values).

    Mathematically equivalent to v105's constraint energy — value = Σ d × 10^p
    works the same regardless of digit ordering convention, but we use the
    LSD-first expected-value helper to keep semantics consistent.
    """
    B = int(digit_logits_final.shape[0])

    ev = _expected_value_v105_2(digit_logits_final, n_digits)
    fa_clamped = factor_args.cast(dtypes.int).clip(0, n_max - 1)
    fa_oh = fa_clamped.reshape(B, f_max * 3).one_hot(n_max)
    ev_bc = ev.reshape(B, 1, n_max).cast(dtypes.float)
    gathered = (fa_oh.cast(dtypes.float) * ev_bc).sum(axis=-1)
    gathered_r = gathered.reshape(B, f_max, 3)
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

    valid    = (factor_types >= 0).cast(dtypes.float)
    residual = ev_result - ev_expected
    rel_err  = residual.abs() / (ev_expected.abs() + 1.0)
    rel_err_clipped = rel_err.clip(0.0, 10.0)
    energy   = rel_err_clipped * valid
    n_valid  = valid.sum() + 1e-8
    return energy.sum() / n_valid


# ---------------------------------------------------------------------------
# Embedding (same as v105.1.2 — RoPE depends on position index, not semantic)
# ---------------------------------------------------------------------------

def embed_factor_graph_v105_2(
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

    Identical to v105.1.2 embedding — RoPE encodes position index, not the
    semantic meaning, so LSD-first vs MSD-first is transparent at this layer.
    Position 0 still gets rotated by 0 × freq (i.e. identity rotation).
    """
    B = int(digit_init.shape[0])
    H = int(var_pos_embed.shape[1])

    di_cb = digit_init.cast(digit_codebook.dtype)
    var_digit_state = di_cb @ digit_codebook
    vpe = var_pos_embed.reshape(1, n_max, 1, H).cast(var_digit_state.dtype)
    raw = var_digit_state + vpe.expand(B, n_max, n_digits, H)

    var_digit_h = apply_rope_digit_tg(
        raw, digit_rope_cos, digit_rope_sin, n_digits=n_digits, hidden=H
    )

    var_tokens = var_digit_h.reshape(B, n_max * n_digits, H)

    factor_pos = factor_pos_embed.reshape(1, f_max, H).cast(var_tokens.dtype).expand(B, f_max, H)
    x = var_tokens.cat(factor_pos, dim=1)

    nk_clamped = node_kinds.clip(0, 2)
    nk_oh      = nk_clamped.one_hot(3).cast(x.dtype)
    nk_emb     = nk_oh @ node_kind_embed.cast(x.dtype)
    x = x + nk_emb

    return x


# ---------------------------------------------------------------------------
# One transformer layer (identical to v105.1.2)
# ---------------------------------------------------------------------------

def fg_layer_forward_v105_2(
    layer: Any,
    x: Tensor,
    attn_bias: Tensor,
) -> Tensor:
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
# Projection waist + IB codebook (identical to v105.1.2)
# ---------------------------------------------------------------------------

def apply_projection_waist_v105_2(
    h: Tensor,
    W_compress: Tensor,
    W_expand: Tensor,
    b_compress: Tensor,
    b_expand: Tensor,
) -> Tensor:
    wc = W_compress.cast(h.dtype)
    bc = b_compress.reshape(1, 1, -1).cast(h.dtype)
    we = W_expand.cast(h.dtype)
    be = b_expand.reshape(1, 1, -1).cast(h.dtype)

    waist_h  = (h @ wc + bc).gelu()
    quantize = waist_h @ we + be
    return h + quantize


def apply_ib_codebook_v105_2(
    h: Tensor,
    codebook: Tensor,
    temperature: Tensor,
    delta_gate_quant_k: Tensor,
) -> Tensor:
    cb  = codebook.cast(h.dtype)
    tmp = temperature.cast(h.dtype)

    scores   = h @ cb.T / tmp.reshape(1, 1, 1)
    weights  = scores.clip(-1e4, 1e4).softmax(-1)
    recon    = weights @ cb
    quantize = recon - h
    gate_k   = delta_gate_quant_k.cast(h.dtype).reshape(1, 1, 1)
    return h + gate_k * quantize


# ---------------------------------------------------------------------------
# Iterative prefill loop
# ---------------------------------------------------------------------------

def fg_breathing_forward_v105_2(
    model: Any,
    digit_init: Tensor,       # (B, N_MAX, N_DIGITS, 10)
    node_kinds: Tensor,        # (B, T_MAX) int
    staging_mask: Tensor,      # (B, K_MAX, T_MAX, T_MAX)
    head_op_mask: Tensor,      # (B, N_HEADS, T_MAX, T_MAX)
    K: int,
    n_max: int = V105_2_N_MAX,
    f_max: int = V105_2_F_MAX,
    n_digits: int = V105_2_N_DIGITS,
) -> tuple[list[Tensor], list[Tensor], list[Tensor]]:
    """K iterative-prefill breaths — identical control flow to v105.1.2.

    Per-breath: embed → mask → 4 layers → IB codebook → waist → delta-gate →
    digit/factor logits + calibration.
    """
    assert hasattr(model, "fg_v105_2_digit_codebook"), \
        "model has no v105.2 params; was attach_fg_params_v105_2 called?"

    digit_codebook   = model.fg_v105_2_digit_codebook
    digit_rope_cos   = model.fg_v105_2_digit_rope_cos
    digit_rope_sin   = model.fg_v105_2_digit_rope_sin
    var_pos_embed    = model.fg_v105_2_var_pos_embed
    factor_pos_embed = model.fg_v105_2_factor_pos_embed
    node_kind_embed  = model.fg_v105_2_node_kind_embed
    breath_embed     = model.fg_v105_2_breath_embed
    delta_gate       = model.fg_v105_2_delta_gate
    calib_head_w     = model.fg_v105_2_calib_head_w
    calib_head_b     = model.fg_v105_2_calib_head_b

    W_compress = model.fg_v105_2_W_compress
    b_compress = model.fg_v105_2_b_compress
    W_expand   = model.fg_v105_2_W_expand
    b_expand   = model.fg_v105_2_b_expand

    ib_codebook      = model.fg_v105_2_ib_codebook
    delta_gate_quant = model.fg_v105_2_delta_gate_quant
    ib_temperature   = model.fg_v105_2_ib_temperature

    B = int(digit_init.shape[0])
    T = n_max * n_digits + f_max
    H = int(var_pos_embed.shape[1])

    x = embed_factor_graph_v105_2(
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
        be_k  = breath_embed[k].reshape(1, 1, -1).cast(x.dtype)
        x_in  = x + be_k
        x_pre = x

        stk   = staging_mask[:, k, :, :]
        stk_h = stk.reshape(B, 1, T, T).expand(B, V105_2_N_HEADS, T, T)
        combined = stk_h.cast(x.dtype) + head_op_mask.cast(x.dtype)

        h = x_in
        for layer in layers[:4]:
            h = fg_layer_forward_v105_2(layer, h, combined)

        h = apply_ib_codebook_v105_2(h, ib_codebook, ib_temperature, delta_gate_quant[k])
        h = apply_projection_waist_v105_2(h, W_compress, W_expand, b_compress, b_expand)

        gate_k = delta_gate[k].cast(h.dtype).reshape(1, 1, 1)
        delta  = h - x_pre
        x      = x_pre + gate_k * delta

        x_ln = _layernorm(x, model.ln_f_g, model.ln_f_b, model.cfg.layer_norm_eps).cast(dtypes.float)
        n_var_tokens = n_max * n_digits
        var_tokens   = x_ln[:, :n_var_tokens, :]
        var_tokens_r = var_tokens.reshape(B, n_max, n_digits, -1)
        cb_fp        = digit_codebook.cast(dtypes.float)
        digit_logits_k = var_tokens_r @ cb_fp.T
        digit_logits_history.append(digit_logits_k)

        fac_tokens   = x_ln[:, n_var_tokens:n_var_tokens + f_max, :]
        fac_tokens_r = fac_tokens.reshape(B, f_max, 1, -1).expand(
            B, f_max, n_digits, int(x_ln.shape[-1])
        )
        fac_logits_k = fac_tokens_r @ cb_fp.T
        factor_logits_history.append(fac_logits_k)

        pool        = x_ln.mean(axis=1)
        calib_logit = pool @ calib_head_w.cast(dtypes.float) + calib_head_b.cast(dtypes.float)
        calib_k     = calib_logit.reshape(-1).sigmoid()
        calib_history.append(calib_k)

    return digit_logits_history, factor_logits_history, calib_history


# ---------------------------------------------------------------------------
# Parameter attachment
# ---------------------------------------------------------------------------

def attach_fg_params_v105_2(
    model: Any,
    hidden: int,
    n_digits: int = V105_2_N_DIGITS,
    n_max: int = V105_2_N_MAX,
    f_max: int = V105_2_F_MAX,
    k_max: int | None = None,
    waist: int | None = None,
    n_code: int | None = None,
    rope_base: float | None = None,
    ib_centroids_path: str | None = None,
    ib_init: bool | None = None,
    waist_lora_init: bool | None = None,
) -> None:
    """Attach v105.2 params to model. Identical layout to v105.1.2.

    LoRA invariant: step-0 forward is byte-identical to v105.1 backbone
    (waist W_expand = 0 and delta_gate_quant = 0 means both new residual paths
    are pass-throughs at init).
    """
    if k_max is None:
        k_max = V105_2_K_MAX
    if waist is None:
        waist = V105_2_WAIST
    if n_code is None:
        n_code = V105_2_CODEBOOK_N
    if rope_base is None:
        rope_base = V105_2_ROPE_BASE
    if ib_centroids_path is None:
        ib_centroids_path = V105_2_IB_CENTROIDS
    if ib_init is None:
        ib_init = V105_2_IB_INIT
    if waist_lora_init is None:
        waist_lora_init = V105_2_WAIST_LORA_INIT

    rng = np.random.RandomState(20013)

    # Digit codebook + frozen RoPE tables
    raw_dc = rng.randn(max(hidden, 10), hidden).astype(np.float32)
    q_dc, _ = np.linalg.qr(raw_dc)
    dc = q_dc[:10].astype(np.float32) * 1.0
    model.fg_v105_2_digit_codebook = Tensor(dc, dtype=dtypes.float).contiguous()

    cos_t, sin_t = _precompute_digit_rope(n_digits, hidden, base=rope_base)
    model.fg_v105_2_digit_rope_cos = Tensor(cos_t, dtype=dtypes.float).contiguous()
    model.fg_v105_2_digit_rope_sin = Tensor(sin_t, dtype=dtypes.float).contiguous()

    # Position / kind embeddings
    vp = rng.randn(n_max, hidden).astype(np.float32) * 0.02
    model.fg_v105_2_var_pos_embed = Tensor(vp, dtype=dtypes.float).contiguous()

    fp_emb = rng.randn(f_max, hidden).astype(np.float32) * 0.02
    model.fg_v105_2_factor_pos_embed = Tensor(fp_emb, dtype=dtypes.float).contiguous()

    nk = rng.randn(3, hidden).astype(np.float32) * 0.02
    model.fg_v105_2_node_kind_embed = Tensor(nk, dtype=dtypes.float).contiguous()

    # Calibration head
    cw = (rng.randn(hidden, 1) * 0.02).astype(np.float32)
    model.fg_v105_2_calib_head_w = Tensor(cw, dtype=dtypes.float).contiguous()
    model.fg_v105_2_calib_head_b = Tensor.zeros((1,), dtype=dtypes.float).contiguous()

    # Per-breath embeddings + delta gate
    rng_be = np.random.RandomState(20014)
    raw_be = rng_be.randn(max(k_max, hidden), hidden).astype(np.float32)
    q_be, _ = np.linalg.qr(raw_be)
    be = q_be[:k_max].astype(np.float32) * 0.5
    model.fg_v105_2_breath_embed = Tensor(be, dtype=dtypes.float).contiguous()
    model.fg_v105_2_delta_gate   = Tensor.ones((k_max,), dtype=dtypes.float).contiguous()

    # Projection waist (LoRA-style)
    W_c = (rng.randn(hidden, waist) * 0.02).astype(np.float32)
    model.fg_v105_2_W_compress = Tensor(W_c, dtype=dtypes.float).contiguous()
    model.fg_v105_2_b_compress = Tensor.zeros((waist,), dtype=dtypes.float).contiguous()

    if waist_lora_init:
        model.fg_v105_2_W_expand = Tensor.zeros((waist, hidden), dtype=dtypes.float).contiguous()
    else:
        We = (rng.randn(waist, hidden) * 0.02).astype(np.float32)
        model.fg_v105_2_W_expand = Tensor(We, dtype=dtypes.float).contiguous()
    model.fg_v105_2_b_expand = Tensor.zeros((hidden,), dtype=dtypes.float).contiguous()

    # IB semantic codebook
    if ib_init:
        cb_np = load_ib_centroids(ib_centroids_path, n_code=n_code, hidden=hidden)
    else:
        raw_cb = rng.randn(max(hidden, n_code), hidden).astype(np.float32)
        q_cb, _ = np.linalg.qr(raw_cb)
        cb_np = q_cb[:n_code].astype(np.float32) * 0.5
        print(f"[v105.2] IB init disabled; random QR codebook ({n_code}, {hidden})")

    model.fg_v105_2_ib_codebook = Tensor(cb_np, dtype=dtypes.float).contiguous()
    model.fg_v105_2_delta_gate_quant = Tensor.zeros((k_max,), dtype=dtypes.float).contiguous()
    model.fg_v105_2_ib_temperature   = Tensor(
        np.array([1.0], dtype=np.float32), dtype=dtypes.float
    ).contiguous()

    T = n_max * n_digits + f_max
    print(
        f"[v105.2] params attached (LSD-first + valid_mask):\n"
        f"  digit_codebook=(10,{hidden}), "
        f"digit_rope (N_DIGITS={n_digits}, H={hidden}, base={rope_base:.0f}) [FROZEN]\n"
        f"  var_pos_embed=({n_max},{hidden}), factor_pos_embed=({f_max},{hidden})\n"
        f"  waist=({hidden}→{waist}→{hidden}), W_expand={'ZEROS' if waist_lora_init else 'random'}\n"
        f"  ib_codebook=({n_code},{hidden}), "
        f"delta_gate_quant=ZEROS, ib_init={ib_init}\n"
        f"  T={T} (N_MAX*N_DIGITS+F_MAX={n_max}*{n_digits}+{f_max}), K_max={k_max}",
        flush=True,
    )


def fg_v105_2_parameters(model: Any) -> list[Tensor]:
    """Trainable v105.2 factor-graph-specific params (frozen RoPE tables excluded)."""
    return [
        model.fg_v105_2_digit_codebook,
        model.fg_v105_2_var_pos_embed,
        model.fg_v105_2_factor_pos_embed,
        model.fg_v105_2_node_kind_embed,
        model.fg_v105_2_calib_head_w,
        model.fg_v105_2_calib_head_b,
        model.fg_v105_2_breath_embed,
        model.fg_v105_2_delta_gate,
        model.fg_v105_2_W_compress,
        model.fg_v105_2_b_compress,
        model.fg_v105_2_W_expand,
        model.fg_v105_2_b_expand,
        model.fg_v105_2_ib_codebook,
        model.fg_v105_2_delta_gate_quant,
        model.fg_v105_2_ib_temperature,
    ]


def fg_v105_2_state_dict(model: Any) -> dict[str, Tensor]:
    """Full state dict (includes frozen RoPE tables for checkpoint self-containment)."""
    return {
        "fg_v105_2.digit_codebook":   model.fg_v105_2_digit_codebook,
        "fg_v105_2.digit_rope_cos":   model.fg_v105_2_digit_rope_cos,
        "fg_v105_2.digit_rope_sin":   model.fg_v105_2_digit_rope_sin,
        "fg_v105_2.var_pos_embed":    model.fg_v105_2_var_pos_embed,
        "fg_v105_2.factor_pos_embed": model.fg_v105_2_factor_pos_embed,
        "fg_v105_2.node_kind_embed":  model.fg_v105_2_node_kind_embed,
        "fg_v105_2.calib_head_w":     model.fg_v105_2_calib_head_w,
        "fg_v105_2.calib_head_b":     model.fg_v105_2_calib_head_b,
        "fg_v105_2.breath_embed":     model.fg_v105_2_breath_embed,
        "fg_v105_2.delta_gate":       model.fg_v105_2_delta_gate,
        "fg_v105_2.W_compress":       model.fg_v105_2_W_compress,
        "fg_v105_2.b_compress":       model.fg_v105_2_b_compress,
        "fg_v105_2.W_expand":         model.fg_v105_2_W_expand,
        "fg_v105_2.b_expand":         model.fg_v105_2_b_expand,
        "fg_v105_2.ib_codebook":          model.fg_v105_2_ib_codebook,
        "fg_v105_2.delta_gate_quant":     model.fg_v105_2_delta_gate_quant,
        "fg_v105_2.ib_temperature":       model.fg_v105_2_ib_temperature,
    }


# ---------------------------------------------------------------------------
# Warm-start from v104 / v105.x backbone checkpoint
# ---------------------------------------------------------------------------

def load_ckpt_v105_2(model: Any, path: str) -> None:
    """Load backbone (shared.*, phase*.*, ln_f.*) from any v104-family checkpoint.

    v105.2-specific params (digit_codebook, waist, ib_codebook, RoPE tables,
    breath_embed, etc.) are NOT loaded — they are fresh-initialized so the warm
    start preserves LoRA/zero-gate guarantees.

    If the source checkpoint has fg_v104.codebook, it's copied into
    fg_v105_2.ib_codebook (same shape).
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

    if "fg_v104.codebook" in sd and hasattr(model, "fg_v105_2_ib_codebook"):
        src = sd["fg_v104.codebook"]
        dst = model.fg_v105_2_ib_codebook
        if src.shape == dst.shape:
            dst.assign(src.to(dst.device).cast(dst.dtype)).realize()
            print(f"  copied fg_v104.codebook → fg_v105_2.ib_codebook", flush=True)

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
# JIT training step (LSD-first + digit_valid_mask)
# ---------------------------------------------------------------------------

_JIT_V105_2_CACHE: dict = {}


def _compile_jit_fg_step_v105_2(
    model: Any,
    opt: Any,
    K: int,
    B: int,
    factor_aux_weight: float = V105_2_FACTOR_AUX_WEIGHT,
    calib_weight: float = V105_2_CALIB_WEIGHT,
    energy_weight: float = V105_2_ENERGY_WEIGHT,
    n_max: int = V105_2_N_MAX,
    f_max: int = V105_2_F_MAX,
    n_digits: int = V105_2_N_DIGITS,
    grad_clip: float = 1.0,
):
    """Compile a TinyJit'd train step for v105.2.

    NEW inputs (vs v105.1.2):
      digit_valid_mask        : (B, N_MAX, N_DIGITS) float — 1 for natural digits
      factor_digit_valid_mask : (B, F_MAX, N_DIGITS) float — 1 for natural digits

    Loss structure (unchanged form, but masks are different):
      var_loss   — per-breath weighted CE on (unobserved AND digit-valid) positions
      factor_aux — relative MSE on reconstructed numbers using LSD-first place values
      calib_loss — MSE calibration head
      energy_loss — expected-value constraint energy (LSD-first)
    """
    n_code = int(model.fg_v105_2_ib_codebook.shape[0])
    key = ("v105_2", id(model), id(opt), int(K), int(B),
           float(factor_aux_weight), float(calib_weight), float(energy_weight),
           int(n_max), int(f_max), int(n_digits), float(grad_clip), int(n_code))
    if key in _JIT_V105_2_CACHE:
        return _JIT_V105_2_CACHE[key]

    aw     = float(calib_weight)
    fw     = float(factor_aux_weight)
    ew     = float(energy_weight)
    gc     = float(grad_clip)
    params = opt.params

    print(
        f"[JIT] compile v105.2 fg step (LSD-first + valid_mask): K={K} B={B} "
        f"n_digits={n_digits} T={n_max * n_digits + f_max} "
        f"aw={aw} fw={fw} ew={ew} gc={gc} n_code={n_code}...",
        flush=True,
    )

    @TinyJit
    def _step(
        digit_init: Tensor,                # (B, N_MAX, N_DIGITS, 10)
        node_kinds: Tensor,                 # (B, T_MAX) int
        staging_mask: Tensor,
        head_op_mask: Tensor,
        gold_digits: Tensor,                # (B, N_MAX, N_DIGITS) int
        observed_mask: Tensor,              # (B, N_MAX) int
        factor_gold_dg: Tensor,             # (B, F_MAX, N_DIGITS) int
        factor_valid: Tensor,               # (B, F_MAX) float
        factor_types: Tensor,
        factor_args: Tensor,
        digit_valid_mask: Tensor,           # (B, N_MAX, N_DIGITS) float  (NEW)
        factor_digit_valid_mask: Tensor,    # (B, F_MAX, N_DIGITS) float  (NEW)
    ):
        opt.zero_grad()

        digit_logits_history, factor_logits_history, calib_history = \
            fg_breathing_forward_v105_2(
                model, digit_init, node_kinds, staging_mask, head_op_mask,
                K=K, n_max=n_max, f_max=f_max, n_digits=n_digits,
            )

        # --- Main CE on (unobserved AND digit-valid) variable digit positions ---
        unobs_float   = (1 - observed_mask.cast(dtypes.float))         # (B, N_MAX)
        unobs_dg      = unobs_float.reshape(B, n_max, 1).expand(B, n_max, n_digits)
        # Combined mask: position is BOTH unobserved AND part of the number.
        combined_mask = unobs_dg * digit_valid_mask                     # (B, N_MAX, N_DIGITS)
        n_active      = combined_mask.sum() + 1e-8

        gold_flat   = gold_digits.cast(dtypes.int).reshape(B * n_max * n_digits)
        cmask_flat  = combined_mask.reshape(B * n_max * n_digits)

        var_loss_sum  = Tensor.zeros((), dtype=dtypes.float).contiguous()
        var_weight_sum = 0.0
        per_breath_ce_t: list[Tensor] = []

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

        # --- Per-NUMBER factor aux (relative MSE) with LSD-first place values ---
        # LSD-first place values: [10^0, 10^1, ..., 10^(N-1)]
        n_valid_fac = factor_valid.sum() + 1e-8

        place_values_np = [float(10 ** i) for i in range(n_digits)]  # LSD!
        place_values_t  = Tensor(place_values_np, dtype=dtypes.float).reshape(1, 1, n_digits)
        digit_vals_t    = Tensor([float(i) for i in range(10)], dtype=dtypes.float)

        # Reconstruct gold numbers, masking out padding positions.
        # (Gold digits at invalid positions are already 0, but the explicit mask
        # is cheap and matches the prediction-side masking below for symmetry.)
        gold_dg_float   = factor_gold_dg.cast(dtypes.float)                       # (B, F, D)
        gold_dg_masked  = gold_dg_float * factor_digit_valid_mask                  # zero invalid
        gold_numbers    = (gold_dg_masked * place_values_t).sum(axis=-1)          # (B, F)
        rel_denom       = gold_numbers.abs() + 1.0

        factor_aux_sum   = Tensor.zeros((), dtype=dtypes.float).contiguous()
        factor_aux_w_sum = 0.0

        for k_aux, fac_logits_k in enumerate(factor_logits_history):
            w_k_aux = 1.0 + float(k_aux) / float(max(K - 1, 1))

            fac_probs   = fac_logits_k.softmax(axis=-1)                            # (B,F,D,10)
            exp_digit   = (fac_probs * digit_vals_t.reshape(1, 1, 1, 10)).sum(axis=-1)
            # Mask out padding positions in the prediction (force the model to
            # predict 0 contribution at invalid positions — they aren't part of
            # the number).
            exp_digit_m = exp_digit * factor_digit_valid_mask
            pred_number = (exp_digit_m * place_values_t).sum(axis=-1)

            rel_err  = ((pred_number - gold_numbers) / rel_denom).clip(-5.0, 5.0)
            sq_err   = rel_err * rel_err
            sq_err_m = sq_err * factor_valid.cast(sq_err.dtype)
            fac_ce_k = sq_err_m.sum() / n_valid_fac

            factor_aux_sum   = factor_aux_sum + fac_ce_k * w_k_aux
            factor_aux_w_sum += w_k_aux

        factor_aux_loss = factor_aux_sum / float(factor_aux_w_sum)

        # --- Constraint energy (LSD-first) ---
        final_dig_logits = digit_logits_history[-1]
        energy_loss = constraint_energy_v105_2(
            final_dig_logits, factor_types, factor_args,
            n_max=n_max, f_max=f_max, n_digits=n_digits,
        )

        # --- Calibration ---
        # Variable considered "correct" if all VALID digits match (invalid positions
        # are ignored — their gold is 0 by construction and prediction is unconstrained).
        # A variable counts toward the metric only if it's both unobserved AND real
        # (real = has at least one valid digit position; padding rows have
        # digit_valid_mask == 0 everywhere and would otherwise auto-pass).
        final_pred_dg = digit_logits_history[-1].argmax(axis=-1).detach()           # (B,N,D)
        dg_eq    = (final_pred_dg == gold_digits.cast(dtypes.int)).cast(dtypes.float)
        dg_match_or_invalid = dg_eq * digit_valid_mask + (1.0 - digit_valid_mask)
        var_eq   = dg_match_or_invalid.min(axis=-1)                                 # (B,N)
        # is_real_var: 1 if variable uses at least one digit position, else 0
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

    _JIT_V105_2_CACHE[key] = _step
    print(
        f"[JIT] v105.2 fg step ready (cache={len(_JIT_V105_2_CACHE)}); "
        f"first call compiles...",
        flush=True,
    )
    return _step


def _compile_jit_fg_eval_v105_2(
    model: Any,
    K: int,
    B: int,
    n_max: int = V105_2_N_MAX,
    f_max: int = V105_2_F_MAX,
    n_digits: int = V105_2_N_DIGITS,
):
    """Compile a TinyJit'd eval step (forward only).

    NOTE: eval takes digit_valid_mask too so that cell_acc treats padding
    positions as automatically correct (consistent with the train loss).
    """
    key = ("eval_v105_2", id(model), int(K), int(B), int(n_max), int(f_max), int(n_digits))
    if key in _JIT_V105_2_CACHE:
        return _JIT_V105_2_CACHE[key]

    print(f"[JIT] compile v105.2 fg eval: K={K} B={B} n_digits={n_digits}...", flush=True)

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
        digit_logits_history, _, _ = fg_breathing_forward_v105_2(
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

    _JIT_V105_2_CACHE[key] = _eval
    print(f"[JIT] v105.2 eval ready (cache={len(_JIT_V105_2_CACHE)})", flush=True)
    return _eval
