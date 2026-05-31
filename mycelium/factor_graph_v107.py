"""v107 factor graph breathing transformer — hybrid 200-bin codebook architecture.

Architectural pivot after v105's digit-by-digit prediction collapsed to 0% val
accuracy: per-digit predictions were 93% accurate independently but UNCORRELATED
across digit positions, so full-number accuracy was 0%.

Key architectural insight:
  PREDICTION: model outputs a 200-way softmax over a HYBRID codebook (one number,
              fully correlated). The 200 bins cover [0, 9999] with 100 linear +
              50 log-spaced [100,999] + 50 log-spaced [1000,9999].
  SEARCH: MCTS navigates the digit decomposition of the 200 bins (tractable
          10-way branching per digit position). The model never predicts
          independent digits; digits are only DERIVED from the predicted bin
          for MCTS navigation.

Two distinct codebooks:
  domain_codebook  (200, H): value vocabulary; the model's prediction target.
                              Hybrid spacing: 100 linear + 50 log + 50 log.
                              Random orthonormal init.
  semantic_codebook (32, H): operator-role vocabulary; per-breath residual
                              correction. Init from IB centroids (same as v104).

This is structurally identical to v104 with the following changes:
  - domain_codebook: 100→200 bins (hybrid spacing)
  - domain_init:     (B, N_MAX, 200) one-hot / uniform (vs 100-way in v104)
  - All loss/accuracy: 200-way CE (vs 100-way)
  - Energy: operates on bin expected values (linear weighted sum over bin_values)

Everything else is preserved from v104: same transformer, same v100 params,
same IB semantic codebook mechanism.

Env var gates:
  V107_TASK=1                   — enable v107 forward path
  V107_BIN_COUNT=200            — must be 200
  V107_K_MAX=10                 — number of iterative-prefill breaths
  V107_ENERGY_WEIGHT=0.01       — expected-value energy loss weight
  V107_CALIB_WEIGHT=0.05        — calibration loss weight
  V107_FACTOR_AUX_WEIGHT=0.5    — factor-execute auxiliary loss weight
  V107_N_MAX=16                 — max variable nodes
  V107_F_MAX=8                  — max factor nodes
  V107_CODEBOOK_N=32            — semantic codebook size (must match IB centroids=32)
  V107_IB_CENTROIDS=.cache/ib_centroids_gsm8k_partial.npz
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

# Re-export unchanged pieces from v100/v104
from mycelium.factor_graph_v100 import (
    embed_factor_graph_v100_aligned,
    fg_layer_forward_v100,
    V100_N_HEADS,
    OP_ADD, OP_SUB, OP_MUL, OP_DIV,
)
from mycelium.factor_graph_v104 import (
    load_ib_centroids,
)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

V107_TASK               = int(os.environ.get("V107_TASK", "0")) > 0
V107_BIN_COUNT          = int(os.environ.get("V107_BIN_COUNT",         "200"))
V107_K_MAX              = int(os.environ.get("V107_K_MAX",              "10"))
V107_ENERGY_WEIGHT      = float(os.environ.get("V107_ENERGY_WEIGHT",    "0.01"))
V107_CALIB_WEIGHT       = float(os.environ.get("V107_CALIB_WEIGHT",     "0.05"))
V107_FACTOR_AUX_WEIGHT  = float(os.environ.get("V107_FACTOR_AUX_WEIGHT","0.5"))
V107_N_MAX              = int(os.environ.get("V107_N_MAX",              "16"))
V107_F_MAX              = int(os.environ.get("V107_F_MAX",               "8"))
V107_T_MAX              = V107_N_MAX + V107_F_MAX
V107_N_HEADS            = 16   # fixed: Pythia-410M
V107_CODEBOOK_N         = int(os.environ.get("V107_CODEBOOK_N",         "32"))
V107_IB_CENTROIDS       = os.environ.get(
    "V107_IB_CENTROIDS", ".cache/ib_centroids_gsm8k_partial.npz"
)


# ---------------------------------------------------------------------------
# Hybrid 200-bin domain codebook definition
# ---------------------------------------------------------------------------

def build_bin_values() -> np.ndarray:
    """Construct the 200 representative integer values for the hybrid codebook.

    Layout:
      bins   0-99:  linear, one bin per integer 0..99
      bins 100-149: log-spaced 100..999 (50 bins)
      bins 150-199: log-spaced 1000..9999 (50 bins)

    Returns (200,) int64 array. All values are unique.
    """
    bins_linear = np.arange(100, dtype=np.int64)
    bins_log1   = np.round(np.geomspace(100, 999,  50)).astype(np.int64)
    bins_log2   = np.round(np.geomspace(1000, 9999, 50)).astype(np.int64)
    bv = np.concatenate([bins_linear, bins_log1, bins_log2])
    assert len(bv) == 200, f"Expected 200 bins, got {len(bv)}"
    assert len(np.unique(bv)) == 200, "Bin values must be unique"
    return bv


# Module-level cache so it's computed once
_BIN_VALUES: np.ndarray | None = None


def get_bin_values() -> np.ndarray:
    """Return cached (200,) int64 array of bin representative values."""
    global _BIN_VALUES
    if _BIN_VALUES is None:
        _BIN_VALUES = build_bin_values()
    return _BIN_VALUES


def nearest_bin(value: int, bin_values: np.ndarray) -> int:
    """Return the index of the bin whose representative value is closest to `value`.

    For values in [0,99] (linear region) this is exact: nearest_bin(v) == v.
    For values outside [0,9999] the nearest boundary bin is returned.
    """
    return int(np.argmin(np.abs(bin_values.astype(np.float64) - float(value))))


# ---------------------------------------------------------------------------
# MCTS utility functions (digit decomposition of bins)
# ---------------------------------------------------------------------------

def bin_to_digits(bin_idx: int, n_digits: int = 5) -> list[int]:
    """Convert a bin index to its representative integer value's digits (MSD-first).

    Args:
      bin_idx:  Index in [0, 200).
      n_digits: Total digit positions (padded with leading zeros).

    Returns list of ints, length n_digits, MSD-first.
    E.g. bin_idx=134 (repr=494), n_digits=5 → [0, 0, 4, 9, 4]
    """
    bv = get_bin_values()
    val = int(bv[bin_idx])
    val = max(0, min(val, 10 ** n_digits - 1))
    digits = []
    for p in range(n_digits - 1, -1, -1):
        place = 10 ** p
        d = val // place
        val = val % place
        digits.append(d)
    return digits  # index 0 = most significant


def digits_to_bin_constraint(
    digit_clamps: list[tuple[int, int]],
    n_digits: int = 5,
) -> list[int]:
    """Return all bin indices whose representative value is consistent with digit clamps.

    Args:
      digit_clamps: List of (position, digit_value) constraints.
                    position 0 = MSD, position n_digits-1 = LSD.
      n_digits:     Total digit positions.

    Returns sorted list of valid bin indices (may be empty).
    """
    bv = get_bin_values()
    valid = []
    for i, val in enumerate(bv):
        digits = bin_to_digits(i, n_digits)
        ok = True
        for pos, dval in digit_clamps:
            if digits[pos] != dval:
                ok = False
                break
        if ok:
            valid.append(i)
    return valid


# ---------------------------------------------------------------------------
# Iterative prefill loop (v107)
# ---------------------------------------------------------------------------

def fg_breathing_forward_v107(
    model: Any,
    domain_init: Tensor,     # (B, N_MAX, 200) — one-hot observed, uniform unobserved
    node_kinds: Tensor,       # (B, T_MAX) int
    staging_mask: Tensor,     # (B, K_MAX, T_MAX, T_MAX)
    head_op_mask: Tensor,     # (B, N_HEADS, T_MAX, T_MAX)
    K: int,
    n_max: int = V107_N_MAX,
    f_max: int = V107_F_MAX,
) -> tuple[list[Tensor], list[Tensor], list[Tensor]]:
    """Run K iterative-prefill breaths on a batch of factor graphs.

    Identical forward path to v104 EXCEPT:
      - domain_codebook is (200, H) instead of (100, H)
      - domain_init is (B, N_MAX, 200) instead of (B, N_MAX, 100)
      - var_state_embed is (200, H)

    Per breath k:
      1. Add breath embedding.
      2. Build combined (B, N_HEADS, T, T) mask.
      3. Run 4 transformer layers.
      4. IB-anchored semantic codebook compression in 1024d residual space.
      5. Learnable delta gate.
      6. Readout: var_logits (B, N_MAX, 200), factor_logits (B, F_MAX, 200), calib.
    """
    assert hasattr(model, "fg_v107_domain_codebook"), \
        "model has no v107 params; was attach_fg_params_v107 called?"

    domain_codebook  = model.fg_v107_domain_codebook    # (200, H)
    var_state_embed  = model.fg_v107_var_state_embed     # (200, H)
    var_pos_embed    = model.fg_v107_var_pos_embed       # (N_MAX, H)
    factor_pos_embed = model.fg_v107_factor_pos_embed   # (F_MAX, H)
    node_kind_embed  = model.fg_v107_node_kind_embed    # (3, H)
    breath_embed     = model.fg_v107_breath_embed       # (K_max, H)
    delta_gate       = model.fg_v107_delta_gate         # (K_max,)
    calib_head_w     = model.fg_v107_calib_head_w       # (H, 1)
    calib_head_b     = model.fg_v107_calib_head_b       # (1,)

    # IB semantic codebook params
    semantic_codebook    = model.fg_v107_semantic_codebook     # (N_CODE, H)
    delta_gate_quant     = model.fg_v107_delta_gate_quant      # (K_max,)
    temperature          = model.fg_v107_temperature           # ()

    B = int(domain_init.shape[0])
    T = n_max + f_max

    # Initial embedding: (B, N_MAX, 200) @ (200, H) → (B, N_MAX, H)
    # Use embed_factor_graph_v100_aligned but with the 200d codebook.
    # The function signature accepts generic shapes, so this works directly.
    x = embed_factor_graph_v100_aligned(
        domain_init, node_kinds,
        var_pos_embed, factor_pos_embed, node_kind_embed,
        domain_codebook, var_state_embed,
    )
    x = x.cast(dtypes.half) if x.dtype != dtypes.half else x

    layers = list(model.block.layers)
    K_max  = int(breath_embed.shape[0])
    assert K <= K_max, f"K={K} > K_max={K_max}"

    from mycelium.breathing import _layernorm

    var_logits_history    = []
    factor_logits_history = []
    calib_history         = []

    for k in range(K):
        be_k  = breath_embed[k].reshape(1, 1, -1).cast(x.dtype)
        x_in  = x + be_k
        x_pre = x

        # Combined (B, N_HEADS, T, T) mask for breath k
        stk      = staging_mask[:, k, :, :]   # (B, T, T)
        stk_h    = stk.reshape(B, 1, T, T).expand(B, V107_N_HEADS, T, T)
        combined = stk_h.cast(x.dtype) + head_op_mask.cast(x.dtype)

        # Four transformer layers (shared across breaths)
        h = x_in
        for layer in layers[:4]:
            h = fg_layer_forward_v100(layer, h, combined)

        # IB semantic codebook compression (same as v104, in 1024d residual space)
        cb  = semantic_codebook.cast(h.dtype)
        tmp = temperature.cast(h.dtype)
        scores   = h @ cb.T / tmp.reshape(1, 1, 1)          # (B, T, N_CODE)
        weights  = scores.clip(-1e4, 1e4).softmax(-1)        # (B, T, N_CODE)
        recon    = weights @ cb                              # (B, T, H)
        quantize = recon - h
        gate_quant_k = delta_gate_quant[k].cast(h.dtype).reshape(1, 1, 1)
        h_quant = h + gate_quant_k * quantize               # = h at init (gate=0)

        # Learnable delta gate over h_quant
        gate_k = delta_gate[k].cast(h_quant.dtype).reshape(1, 1, 1)
        delta  = h_quant - x_pre
        x      = x_pre + gate_k * delta

        # Readout: (B, T, H) → layernorm → project via 200-way domain_codebook
        x_ln  = _layernorm(x, model.ln_f_g, model.ln_f_b, model.cfg.layer_norm_eps).cast(dtypes.float)

        var_x = x_ln[:, :n_max, :]                                     # (B, N_MAX, H)
        var_logits_k  = var_x @ domain_codebook.T.cast(dtypes.float)  # (B, N_MAX, 200)
        var_logits_history.append(var_logits_k)

        fac_x = x_ln[:, n_max:n_max + f_max, :]                       # (B, F_MAX, H)
        fac_logits_k  = fac_x @ domain_codebook.T.cast(dtypes.float)  # (B, F_MAX, 200)
        factor_logits_history.append(fac_logits_k)

        pool        = x_ln.mean(axis=1)                                # (B, H)
        calib_logit = pool @ calib_head_w.cast(dtypes.float) + calib_head_b.cast(dtypes.float)
        calib_k     = calib_logit.reshape(-1).sigmoid()
        calib_history.append(calib_k)

    return var_logits_history, factor_logits_history, calib_history


# ---------------------------------------------------------------------------
# Energy (expected-value form, operates on bin_values in linear space)
# ---------------------------------------------------------------------------

def constraint_energy_v107(
    var_logits_final: Tensor,   # (B, N_MAX, 200)
    factor_types: Tensor,        # (B, F_MAX) int
    factor_args: Tensor,         # (B, F_MAX, 3) int
    bin_values_t: Tensor,        # (200,) float — representative integer values per bin
    n_max: int = V107_N_MAX,
    f_max: int = V107_F_MAX,
) -> Tensor:
    """Expected-value constraint energy (differentiable, inside JIT).

    For each factor: E[result] - op(E[arg1], E[arg2])
    where E[v] = Σ_bin P(bin) * bin_value[bin]  (linear expected value)

    Relative error normalization: |residual| / (|expected| + 1)
    so large GSM8K values don't dominate.
    """
    B = int(var_logits_final.shape[0])

    # Compute expected value for all variables: (B, N_MAX)
    probs   = var_logits_final.softmax(axis=-1)  # (B, N_MAX, 200)
    bv      = bin_values_t.cast(probs.dtype)      # (200,)
    ev      = (probs * bv).sum(axis=-1)           # (B, N_MAX)

    # Gather arg/result expected values via one-hot (no Python loops in JIT)
    fa_clamped = factor_args.cast(dtypes.int).clip(0, n_max - 1)   # (B, F_MAX, 3)
    fa_oh      = fa_clamped.reshape(B, f_max * 3).one_hot(n_max)   # (B, F_MAX*3, N_MAX)
    ev_bc      = ev.reshape(B, 1, n_max).cast(dtypes.float)         # (B, 1, N_MAX)
    gathered   = (fa_oh.cast(dtypes.float) * ev_bc).sum(axis=-1)   # (B, F_MAX*3)
    gathered_r = gathered.reshape(B, f_max, 3)                      # (B, F_MAX, 3)
    ev_arg1    = gathered_r[:, :, 0]    # (B, F_MAX)
    ev_arg2    = gathered_r[:, :, 1]
    ev_result  = gathered_r[:, :, 2]

    # Expected result per op
    ev_add = ev_arg1 + ev_arg2
    ev_sub = ev_arg1 - ev_arg2
    ev_mul = ev_arg1 * ev_arg2
    ev_div = ev_arg1 / (ev_arg2.abs() + 1.0)

    ft_clamped        = factor_types.cast(dtypes.int).clip(0, 3)  # (B, F_MAX)
    ft_oh             = ft_clamped.one_hot(4).cast(dtypes.float)   # (B, F_MAX, 4)
    ev_expected_stack = ev_add.reshape(B, f_max, 1).cat(
        ev_sub.reshape(B, f_max, 1),
        ev_mul.reshape(B, f_max, 1),
        ev_div.reshape(B, f_max, 1),
        dim=-1,
    )
    ev_expected = (ft_oh * ev_expected_stack).sum(axis=-1)         # (B, F_MAX)

    valid     = (factor_types >= 0).cast(dtypes.float)
    residual  = ev_result - ev_expected
    rel_err   = residual.abs() / (ev_expected.abs() + 1.0)
    rel_err_c = rel_err.clip(0.0, 10.0)
    energy    = rel_err_c * valid
    n_valid   = valid.sum() + 1e-8
    return energy.sum() / n_valid


# ---------------------------------------------------------------------------
# Accuracy
# ---------------------------------------------------------------------------

def fg_accuracy_v107(
    var_logits_final: Tensor,  # (B, N_MAX, 200)
    gold_bins: Tensor,          # (B, N_MAX) int — bin index of gold value
    observed_mask: Tensor,      # (B, N_MAX) int
    query_idx_np: np.ndarray,   # (B,) int
) -> dict:
    """Return accuracy stats: query_acc, unobserved_cell_acc."""
    B       = int(var_logits_final.shape[0])
    pred    = var_logits_final.argmax(axis=-1)                            # (B, N_MAX)
    correct = (pred == gold_bins.cast(dtypes.int)).cast(dtypes.float)    # (B, N_MAX)
    unobs   = (1 - observed_mask.cast(dtypes.float))

    masked   = correct * unobs
    n_unobs  = unobs.sum() + 1e-8
    cell_acc = float((masked.sum() / n_unobs).realize().numpy())

    pred_np  = pred.cast(dtypes.int).realize().numpy()
    gold_np  = gold_bins.cast(dtypes.int).realize().numpy()
    q_correct = np.array([
        int(pred_np[b, query_idx_np[b]] == gold_np[b, query_idx_np[b]])
        for b in range(B)
    ], dtype=np.float32)
    query_acc = float(q_correct.mean())

    return {"cell_acc": cell_acc, "query_acc": query_acc}


# ---------------------------------------------------------------------------
# Parameter attachment
# ---------------------------------------------------------------------------

def attach_fg_params_v107(
    model: Any,
    hidden: int,
    n_max: int = V107_N_MAX,
    f_max: int = V107_F_MAX,
    k_max: int | None = None,
    n_code: int = V107_CODEBOOK_N,
    ib_centroids_path: str = V107_IB_CENTROIDS,
) -> None:
    """Attach v107 factor-graph parameters to `model`.

    Domain codebook (200, hidden): random orthonormal × 0.1.
    var_state_embed (200, hidden): ALIGNED with domain_codebook at init
                                   (same principle as v100 for 100-way).
    IB semantic codebook (n_code, hidden): loaded from npz; delta_gate_quant=0.
    All other params (pos_embed, breath_embed, etc.): same init as v100.

    Attributes added (prefixed fg_v107_):
      fg_v107_domain_codebook    (200, hidden)
      fg_v107_var_state_embed    (200, hidden)   — aligned with domain_codebook
      fg_v107_var_pos_embed      (N_MAX, hidden)
      fg_v107_factor_pos_embed   (F_MAX, hidden)
      fg_v107_node_kind_embed    (3, hidden)
      fg_v107_breath_embed       (K_max, hidden)
      fg_v107_delta_gate         (K_max,)
      fg_v107_calib_head_w       (hidden, 1)
      fg_v107_calib_head_b       (1,)
      fg_v107_semantic_codebook  (n_code, hidden)  — IB centroids
      fg_v107_delta_gate_quant   (K_max,)          — zero at init
      fg_v107_temperature        ()                — 1.0 at init
    """
    if k_max is None:
        k_max = V107_K_MAX

    rng_cb = np.random.RandomState(30001)

    # --- Domain codebook (200, hidden): orthonormal rows × 0.1 ---
    # Ensure we generate at least max(hidden, 200) rows for QR
    raw_cb = rng_cb.randn(max(hidden, 200), hidden).astype(np.float32)
    q_cb, _ = np.linalg.qr(raw_cb)
    cb_unit  = q_cb[:200].astype(np.float32)   # (200, hidden) orthonormal
    model.fg_v107_domain_codebook = Tensor(cb_unit * 0.1, dtype=dtypes.float).contiguous()

    # --- var_state_embed (200, hidden): ALIGNED with domain_codebook ---
    model.fg_v107_var_state_embed = Tensor(cb_unit.copy(), dtype=dtypes.float).contiguous()

    # --- Variable position embedding ---
    vp = rng_cb.randn(n_max, hidden).astype(np.float32) * 0.02
    model.fg_v107_var_pos_embed = Tensor(vp, dtype=dtypes.float).contiguous()

    # --- Factor position embedding ---
    fp_emb = rng_cb.randn(f_max, hidden).astype(np.float32) * 0.02
    model.fg_v107_factor_pos_embed = Tensor(fp_emb, dtype=dtypes.float).contiguous()

    # --- Node-kind embedding ---
    nk = rng_cb.randn(3, hidden).astype(np.float32) * 0.02
    model.fg_v107_node_kind_embed = Tensor(nk, dtype=dtypes.float).contiguous()

    # --- Calibration head ---
    cw = (rng_cb.randn(hidden, 1) * 0.02).astype(np.float32)
    model.fg_v107_calib_head_w = Tensor(cw, dtype=dtypes.float).contiguous()
    model.fg_v107_calib_head_b = Tensor.zeros((1,), dtype=dtypes.float).contiguous()

    # --- Per-breath embedding: QR-orthonormal × 0.5 ---
    breath_scale = float(os.environ.get("V107_BREATH_EMBED_SCALE", "0.5"))
    rng_be = np.random.RandomState(30002)
    raw_be = rng_be.randn(max(k_max, hidden), hidden).astype(np.float32)
    q_be, _ = np.linalg.qr(raw_be)
    be = q_be[:k_max].astype(np.float32) * breath_scale
    model.fg_v107_breath_embed = Tensor(be, dtype=dtypes.float).contiguous()

    # --- Per-breath delta gate: init 1.0 ---
    model.fg_v107_delta_gate = Tensor.ones((k_max,), dtype=dtypes.float).contiguous()

    # --- IB semantic codebook (same as v104) ---
    cb_np = load_ib_centroids(ib_centroids_path, n_code=n_code, hidden=hidden)
    model.fg_v107_semantic_codebook = Tensor(cb_np, dtype=dtypes.float).contiguous()

    # delta_gate_quant=0 → h_quant = h at init → byte-identical warm-start
    model.fg_v107_delta_gate_quant = Tensor.zeros((k_max,), dtype=dtypes.float).contiguous()
    model.fg_v107_temperature = Tensor(
        np.array([1.0], dtype=np.float32), dtype=dtypes.float
    ).contiguous()

    bv = get_bin_values()
    print(
        f"[v107] params attached: domain_codebook=(200,{hidden}), "
        f"semantic_codebook=({n_code},{hidden}), "
        f"breath_embed=({k_max},{hidden}), T={n_max+f_max}  "
        f"bin_range=[{bv[0]},{bv[99]}|{bv[100]},{bv[149]}|{bv[150]},{bv[199]}]",
        flush=True,
    )


def fg_v107_parameters(model: Any) -> list[Tensor]:
    """All trainable v107 factor-graph-specific params."""
    return [
        model.fg_v107_domain_codebook,
        model.fg_v107_var_state_embed,
        model.fg_v107_var_pos_embed,
        model.fg_v107_factor_pos_embed,
        model.fg_v107_node_kind_embed,
        model.fg_v107_calib_head_w,
        model.fg_v107_calib_head_b,
        model.fg_v107_breath_embed,
        model.fg_v107_delta_gate,
        model.fg_v107_semantic_codebook,
        model.fg_v107_delta_gate_quant,
        model.fg_v107_temperature,
    ]


def fg_v107_state_dict(model: Any) -> dict[str, Tensor]:
    """State dict entries for v107 factor-graph params."""
    return {
        "fg_v107.domain_codebook":    model.fg_v107_domain_codebook,
        "fg_v107.var_state_embed":    model.fg_v107_var_state_embed,
        "fg_v107.var_pos_embed":      model.fg_v107_var_pos_embed,
        "fg_v107.factor_pos_embed":   model.fg_v107_factor_pos_embed,
        "fg_v107.node_kind_embed":    model.fg_v107_node_kind_embed,
        "fg_v107.calib_head_w":       model.fg_v107_calib_head_w,
        "fg_v107.calib_head_b":       model.fg_v107_calib_head_b,
        "fg_v107.breath_embed":       model.fg_v107_breath_embed,
        "fg_v107.delta_gate":         model.fg_v107_delta_gate,
        "fg_v107.semantic_codebook":  model.fg_v107_semantic_codebook,
        "fg_v107.delta_gate_quant":   model.fg_v107_delta_gate_quant,
        "fg_v107.temperature":        model.fg_v107_temperature,
    }


# ---------------------------------------------------------------------------
# JIT training step
# ---------------------------------------------------------------------------

_JIT_V107_CACHE: dict = {}


def _compile_jit_fg_step_v107(
    model: Any,
    opt: Any,
    K: int,
    B: int,
    factor_aux_weight: float = V107_FACTOR_AUX_WEIGHT,
    calib_weight: float = V107_CALIB_WEIGHT,
    energy_weight: float = V107_ENERGY_WEIGHT,
    n_max: int = V107_N_MAX,
    f_max: int = V107_F_MAX,
    grad_clip: float = 1.0,
):
    """Compile a TinyJit'd training step for v107.

    Inputs to JIT:
      domain_init    : (B, N_MAX, 200) fp32 — one-hot observed, uniform unobs
      node_kinds     : (B, T_MAX) int
      staging_mask   : (B, K_MAX, T_MAX, T_MAX) fp32
      head_op_mask   : (B, N_HEADS, T_MAX, T_MAX) fp32
      gold_bins      : (B, N_MAX) int — bin index of gold value per variable
      observed_mask  : (B, N_MAX) int
      factor_gold_bin: (B, F_MAX) int — bin index of gold result per factor
      factor_valid   : (B, F_MAX) float — 1=real factor, 0=pad
      factor_types   : (B, F_MAX) int — for energy
      factor_args    : (B, F_MAX, 3) int — for energy
      bin_values_t   : (200,) float — representative integer values (constant)

    Returns:
      total, healthy, var_ce, factor_aux, calib, energy, cell_acc, query_acc,
      *pb_ce_0..K-1
    """
    n_bins = int(model.fg_v107_domain_codebook.shape[0])
    assert n_bins == 200, f"Expected 200 bins, got {n_bins}"

    key = ("v107", id(model), id(opt), int(K), int(B), float(factor_aux_weight),
           float(calib_weight), float(energy_weight), int(n_max), int(f_max),
           float(grad_clip), int(n_bins))
    if key in _JIT_V107_CACHE:
        return _JIT_V107_CACHE[key]

    aw     = float(calib_weight)
    fw     = float(factor_aux_weight)
    ew     = float(energy_weight)
    gc     = float(grad_clip)
    params = opt.params

    print(
        f"[JIT] compile v107 fg step: K={K} B={B} n_bins={n_bins} "
        f"aw={aw} fw={fw} ew={ew} gc={gc}...",
        flush=True,
    )

    @TinyJit
    def _step(
        domain_init: Tensor,      # (B, N_MAX, 200)
        node_kinds: Tensor,        # (B, T_MAX) int
        staging_mask: Tensor,      # (B, K_MAX, T_MAX, T_MAX)
        head_op_mask: Tensor,      # (B, N_HEADS, T_MAX, T_MAX)
        gold_bins: Tensor,         # (B, N_MAX) int
        observed_mask: Tensor,     # (B, N_MAX) int
        factor_gold_bin: Tensor,   # (B, F_MAX) int
        factor_valid: Tensor,      # (B, F_MAX) float
        factor_types: Tensor,      # (B, F_MAX) int
        factor_args: Tensor,       # (B, F_MAX, 3) int
        bin_values_t: Tensor,      # (200,) float
    ):
        opt.zero_grad()

        var_logits_history, factor_logits_history, calib_history = \
            fg_breathing_forward_v107(
                model, domain_init, node_kinds, staging_mask, head_op_mask,
                K=K, n_max=n_max, f_max=f_max,
            )

        # --- Main CE on unobserved variables (200-way) ---
        gold_flat    = gold_bins.cast(dtypes.int).reshape(B * n_max)
        unobs_float  = (1 - observed_mask.cast(dtypes.float)).reshape(B * n_max)
        n_unobs_sum  = unobs_float.sum() + 1e-8

        var_loss_sum   = Tensor.zeros((), dtype=dtypes.float).contiguous()
        var_weight_sum = 0.0
        per_breath_ce_t: list[Tensor] = []

        for k, logits in enumerate(var_logits_history):
            weight_k    = 1.0 + float(k) / float(max(K - 1, 1))
            logits_flat = logits.reshape(B * n_max, 200)
            log_probs   = logits_flat.log_softmax(axis=-1)
            gold_oh     = gold_flat.one_hot(200).cast(log_probs.dtype)
            nll         = -(log_probs * gold_oh).sum(axis=-1)
            masked_nll  = nll * unobs_float.cast(nll.dtype)
            ce_k        = masked_nll.sum() / n_unobs_sum
            per_breath_ce_t.append(ce_k)
            var_loss_sum   = var_loss_sum + ce_k * weight_k
            var_weight_sum += weight_k
        var_loss = var_loss_sum / float(var_weight_sum)

        # --- Factor-execute auxiliary loss (200-way, vectorized) ---
        n_valid_factors  = factor_valid.cast(dtypes.float).sum() + 1e-8
        gold_fac_flat    = factor_gold_bin.cast(dtypes.int).reshape(B * f_max)
        gold_fac_oh      = gold_fac_flat.one_hot(200).cast(dtypes.float)
        valid_flat       = factor_valid.cast(dtypes.float).reshape(B * f_max)

        factor_aux_sum   = Tensor.zeros((), dtype=dtypes.float).contiguous()
        factor_aux_w_sum = 0.0

        for k_aux, fac_logits_k in enumerate(factor_logits_history):
            w_k_aux   = 1.0 + float(k_aux) / float(max(K - 1, 1))
            fac_flat  = fac_logits_k.reshape(B * f_max, 200)
            fac_lp    = fac_flat.log_softmax(axis=-1)
            fac_nll   = -(fac_lp * gold_fac_oh).sum(axis=-1)
            fac_masked= fac_nll * valid_flat
            fac_ce_k  = fac_masked.sum() / n_valid_factors
            factor_aux_sum   = factor_aux_sum + fac_ce_k * w_k_aux
            factor_aux_w_sum += w_k_aux
        factor_aux_loss = factor_aux_sum / float(factor_aux_w_sum)

        # --- Expected-value constraint energy ---
        energy_loss = constraint_energy_v107(
            var_logits_history[-1], factor_types, factor_args, bin_values_t,
            n_max=n_max, f_max=f_max,
        )

        # --- Calibration ---
        final_argmax = var_logits_history[-1].argmax(axis=-1).detach()  # (B, N_MAX)
        eq           = (final_argmax == gold_bins.cast(dtypes.int)).cast(dtypes.float)
        unobs_2d     = (1 - observed_mask.cast(dtypes.float))
        eq_unobs     = eq * unobs_2d
        n_unobs_per  = unobs_2d.sum(axis=-1) + 1e-8
        correct      = eq_unobs.sum(axis=-1) / n_unobs_per

        calib_loss_sum = Tensor.zeros((), dtype=dtypes.float).contiguous()
        for k, calib_k in enumerate(calib_history):
            prog       = float(k) / float(max(K - 1, 1))
            target_k   = 0.5 + (correct - 0.5) * prog
            calib_loss_sum = calib_loss_sum + ((calib_k - target_k) ** 2).mean()
        calib_loss = calib_loss_sum / float(K)

        # --- Metrics ---
        cell_acc  = (eq_unobs.sum() / (unobs_2d.sum() + 1e-8)).detach()
        query_acc = correct.mean().detach()

        # --- Total ---
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
            grad_norm = (sq_sum + 1e-12).sqrt()
            clip_coef = (Tensor(gc, dtype=dtypes.float) / (grad_norm + 1e-6)).minimum(
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

    _JIT_V107_CACHE[key] = _step
    print(f"[JIT] v107 fg step ready (cache={len(_JIT_V107_CACHE)}); first call compiles...",
          flush=True)
    return _step


def _compile_jit_fg_eval_v107(
    model: Any,
    K: int,
    B: int,
    n_max: int = V107_N_MAX,
    f_max: int = V107_F_MAX,
):
    """Compile a TinyJit'd eval step (forward only, 200-way argmax)."""
    key = ("eval_v107", id(model), int(K), int(B), int(n_max), int(f_max))
    if key in _JIT_V107_CACHE:
        return _JIT_V107_CACHE[key]

    print(f"[JIT] compile v107 fg eval: K={K} B={B}...", flush=True)

    @TinyJit
    def _eval(
        domain_init: Tensor,
        node_kinds: Tensor,
        staging_mask: Tensor,
        head_op_mask: Tensor,
        gold_bins: Tensor,
        observed_mask: Tensor,
    ):
        var_logits_history, _, _ = fg_breathing_forward_v107(
            model, domain_init, node_kinds, staging_mask, head_op_mask,
            K=K, n_max=n_max, f_max=f_max,
        )
        final_logits = var_logits_history[-1]             # (B, N_MAX, 200)
        pred     = final_logits.argmax(axis=-1)           # (B, N_MAX)
        eq       = (pred == gold_bins.cast(dtypes.int)).cast(dtypes.float)
        unobs    = (1 - observed_mask.cast(dtypes.float))
        cell_acc = (eq * unobs).sum() / (unobs.sum() + 1e-8)
        return pred.realize(), cell_acc.realize()

    _JIT_V107_CACHE[key] = _eval
    print(f"[JIT] v107 eval ready (cache={len(_JIT_V107_CACHE)})", flush=True)
    return _eval
