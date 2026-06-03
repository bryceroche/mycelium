"""v105.5 factor graph data loader — v105.3 + magnitude_target.

Extends v105.3 with one addition: magnitude_target per cell — a 4-way
classification of "how many digits does this value use?":
  class 0:  value in [0, 9]      (1 digit)
  class 1:  value in [10, 99]    (2 digits)
  class 2:  value in [100, 999]  (3 digits)
  class 3:  value >= 1000        (4+ digits — cap at 3)

Layout is otherwise identical to v105.3:
  LSD-first: index 0 = ones, index N-1 = most significant.
  digit_valid_mask : (B, N_MAX, N_DIGITS) — leading-trues semantics.
"""
from __future__ import annotations

import json
import math
import os
import random
from typing import Iterator

import numpy as np
from tinygrad import Tensor, dtypes

from mycelium.factor_graph_data import (
    load_jsonl,
    index_by_difficulty,
    DIFFICULTIES,
    OP_MAP,
)
# Constants are shared with v105 (same N_MAX/F_MAX/K_MAX/N_HEADS/N_DIGITS layout).
from mycelium.factor_graph_v105 import (
    V105_N_MAX, V105_F_MAX, V105_K_MAX, V105_N_DIGITS, V105_N_HEADS,
)

# Import bipartite adjacency builder from v99 (still valid geometry)
from mycelium.factor_graph import build_factor_graph_masks_np
# v107 bin-mapping utilities for v105.8 per-NUMBER readout.
# build_bin_values(): default 200-bin hybrid scheme (100 linear + 50 log + 50 log).
# nearest_bin(value, bv): map an integer value to the nearest bin index.
from mycelium.factor_graph_v107 import build_bin_values, nearest_bin


# v105.8 — per-NUMBER bin mapping (uses v107's hybrid 200-bin scheme by default).
# Env var V105_8_N_NUMBER_BINS controls the number of bins; when 200 we use
# v107's exact scheme. When != 200 we synthesize a simpler linear scheme on
# [0, n_bins) (the unit-bin region of v107 when n_bins <= 100).
V105_8_N_NUMBER_BINS = int(os.environ.get("V105_8_N_NUMBER_BINS", "200"))


_V105_8_BIN_VALUES: np.ndarray | None = None


def _v105_8_bin_values() -> np.ndarray:
    """Cached (n_bins,) int64 array of representative bin values for v105.8."""
    global _V105_8_BIN_VALUES
    if _V105_8_BIN_VALUES is None:
        n_bins = int(V105_8_N_NUMBER_BINS)
        if n_bins == 200:
            # Use v107's hybrid 200-bin scheme exactly (consistent with v107
            # eval semantics — bin 0..99 = integer 0..99, bin 100..149 log
            # [100,999], bin 150..199 log [1000,9999]).
            _V105_8_BIN_VALUES = build_bin_values()
        else:
            # Fallback: linear bins 0..n_bins-1. Suitable when n_bins <= 100.
            _V105_8_BIN_VALUES = np.arange(n_bins, dtype=np.int64)
    return _V105_8_BIN_VALUES


def value_to_number_bin(value: int) -> int:
    """Map an integer value to the v105.8 bin index using the cached scheme."""
    bv = _v105_8_bin_values()
    return int(nearest_bin(max(0, int(value)), bv))

# Gated experimental feature: lateral attention bias on same-position digits
# across operands of a factor. Adds attention edges digit(var_A, p) ↔
# digit(var_B, p) for all operand+result variables of each factor, per digit
# position p. Default off for back-compat.
V105_5_LATERAL_ATTN = int(os.environ.get("V105_5_LATERAL_ATTN", "0")) > 0

# Mean-field-collapse fix (post-Jun 1 linear probe diagnostic):
# The breathing's all-to-all within-variable attention drives digit positions
# toward a shared representation (cos similarity 0.99+ at every position).
# This env var controls within-variable digit-to-digit attention:
#   0 = ALL allowed (default, original — known to cause averaging collapse)
#   1 = HARD block — no within-variable digit attention at all
#       (forces communication through factors only; may break carry composition)
#   2 = SOFT (adjacent-only) — pos p ↔ pos p±1 allowed, far positions blocked
#       (carry can flow sequentially through adjacent positions; far-position
#       averaging is blocked — implements the "traveling wave" via attention mask)
V105_5_BLOCK_WITHIN_VAR = int(os.environ.get("V105_5_BLOCK_WITHIN_VAR", "0"))

# Import DAG depth from v100 data (same algorithm)
from mycelium.factor_graph_data_v100 import compute_var_depth


# ---------------------------------------------------------------------------
# LSD-first digit utilities + validity mask
# ---------------------------------------------------------------------------

def value_to_digits_lsd(value: int, n_digits: int) -> list[int]:
    """LSD-first: digits[0] = ones, digits[1] = tens, ..., digits[N-1] = most significant.

    E.g. value=1234, n_digits=5 → [4, 3, 2, 1, 0]
    Clamps negative values to 0, clamps overflow (value >= 10^n_digits) to 9s.
    """
    v = max(0, int(round(value)))
    max_val = 10 ** n_digits - 1
    v = min(v, max_val)
    digits = []
    for _p in range(n_digits):
        digits.append(v % 10)
        v = v // 10
    return digits  # index 0 = ones digit


def digits_to_value_lsd(digits: list[int]) -> int:
    """Reconstruct integer from LSD-first digit list."""
    return sum(int(d) * (10 ** i) for i, d in enumerate(digits))


def n_actual_digits(value: int, n_digits: int) -> int:
    """How many positions does this value actually use?

    value=0   → 1   (still uses one digit position to encode "0")
    value=7   → 1
    value=42  → 2
    value=1234→ 4
    value=99999 → 5

    Capped at n_digits (so larger values are treated as fully using all positions).
    """
    v = max(0, int(round(value)))
    if v == 0:
        return 1
    n = int(math.floor(math.log10(v))) + 1
    return min(n, n_digits)


def build_digit_valid_mask(value: int, n_digits: int) -> np.ndarray:
    """Returns an (n_digits,) float32 mask: 1 for positions used by the value, 0 else.

    LSD-first layout ⇒ valid positions are the LEADING n_actual positions:
        positions [0, n_actual)
    For value=42, n_actual=2, valid = [1, 1, 0, 0, 0].
    """
    n = n_actual_digits(value, n_digits)
    m = np.zeros(n_digits, dtype=np.float32)
    m[:n] = 1.0
    return m


def gold_magnitude_class(value: int) -> int:
    """4-way magnitude classification — how many digits is this value?

      class 0 : value < 10        (1 digit)
      class 1 : value < 100       (2 digits)
      class 2 : value < 1000      (3 digits)
      class 3 : value >= 1000     (4+ digits, capped here)
    """
    v = max(0, int(round(value)))
    if v < 10:
        return 0
    if v < 100:
        return 1
    if v < 1000:
        return 2
    return 3


# ---------------------------------------------------------------------------
# Digit-level mask construction (identical to v105 / v105.1.2 — only digit ORDER differs)
# ---------------------------------------------------------------------------

def build_staging_and_head_masks_v105_5_np(
    factor_types_np: np.ndarray,   # (F_MAX,) int, -1=padding
    factor_args_np: np.ndarray,    # (F_MAX, 3) int
    var_depth_np: np.ndarray,      # (N_MAX,) int, -1=padding
    n_vars: int,
    n_factors: int,
    n_max: int = V105_N_MAX,
    f_max: int = V105_F_MAX,
    k_max: int = V105_K_MAX,
    n_heads: int = V105_N_HEADS,
    n_digits: int = V105_N_DIGITS,
) -> tuple[np.ndarray, np.ndarray]:
    """Build digit-expanded staging and head-op masks for ONE problem.

    Same logic as v105/v105.1.2 (digit positions of a variable form a within-variable
    attention block; cross-attention to factors follows the bipartite topology).
    The semantic ORDER of digits (LSD vs MSD) doesn't affect this geometry.

    Returns:
      staging  : (k_max, T, T) float32  0=allow, -1e4=blocked
                 T = n_max * n_digits + f_max
      head_ops : (n_heads, T, T) float32
    """
    t_max     = n_max * n_digits + f_max
    n_var_tok = n_max * n_digits

    # 1. Variable-level bipartite adjacency
    bipartite_batch = build_factor_graph_masks_np(
        factor_types_np[np.newaxis],
        factor_args_np[np.newaxis],
        np.array([n_vars], dtype=np.int32),
        np.array([n_factors], dtype=np.int32),
        n_max=n_max,
        f_max=f_max,
    )
    bipartite_v100 = bipartite_batch[0]

    # 2. Expand to digit-level adjacency
    bipartite_full = np.full((t_max, t_max), -1e4, dtype=np.float32)

    # a) Within-variable: digit positions of same variable
    #    Gated by V105_5_BLOCK_WITHIN_VAR:
    #      0 = all-to-all (default, original)
    #      1 = blocked entirely (no within-variable attention)
    #      2 = adjacent-only (pos p ↔ pos p±1; far positions blocked)
    for vi in range(n_max):
        start = vi * n_digits
        end   = start + n_digits
        if V105_5_BLOCK_WITHIN_VAR == 0:
            # Original: all 5 positions attend to each other
            bipartite_full[start:end, start:end] = 0.0
        elif V105_5_BLOCK_WITHIN_VAR == 1:
            # Hard block: only diagonal (self-attention) within variable
            for p in range(n_digits):
                bipartite_full[start + p, start + p] = 0.0
        elif V105_5_BLOCK_WITHIN_VAR == 2:
            # Soft (adjacent-only): pos p ↔ pos p±1 allowed
            for p1 in range(n_digits):
                for p2 in range(n_digits):
                    if abs(p1 - p2) <= 1:
                        bipartite_full[start + p1, start + p2] = 0.0

    # b) Factor self-attention
    for fi in range(f_max):
        fpos = n_var_tok + fi
        bipartite_full[fpos, fpos] = 0.0

    # c) Variable↔factor cross-attention
    for vi in range(n_max):
        for fi in range(n_factors):
            fpos_v100 = n_max + fi
            if bipartite_v100[vi, fpos_v100] >= -1e3:
                vstart = vi * n_digits
                vend   = vstart + n_digits
                fpos_full = n_var_tok + fi
                bipartite_full[vstart:vend, fpos_full] = 0.0
                bipartite_full[fpos_full, vstart:vend] = 0.0

    # 3. Staging masks
    factor_result_depth = np.full(f_max, -1, dtype=np.int32)
    for fi in range(n_factors):
        ft = int(factor_types_np[fi])
        if ft < 0:
            continue
        res_idx = int(factor_args_np[fi, 2])
        if 0 <= res_idx < n_max and var_depth_np[res_idx] >= 0:
            factor_result_depth[fi] = int(var_depth_np[res_idx])

    staging = np.full((k_max, t_max, t_max), -1e4, dtype=np.float32)
    pos_visible = np.zeros((t_max, k_max), dtype=bool)

    for vi in range(n_max):
        vd = int(var_depth_np[vi])
        if vd < 0:
            continue
        for k in range(k_max):
            visible = (vd == 0) or (vd > 0 and vd <= k + 1)
            if visible:
                start = vi * n_digits
                pos_visible[start:start + n_digits, k] = True

    for fi in range(n_factors):
        frd = int(factor_result_depth[fi])
        ft  = int(factor_types_np[fi])
        if ft < 0 or frd < 0:
            continue
        fpos = n_var_tok + fi
        for k in range(k_max):
            if frd == 1 or (frd > 0 and frd <= k + 1):
                pos_visible[fpos, k] = True

    for k in range(k_max):
        pv = pos_visible[:, k]
        both_visible = pv[:, np.newaxis] & pv[np.newaxis, :]
        staging[k] = np.where(both_visible, bipartite_full, -1e4)

    # 4. Head-op masks
    heads_per_op = n_heads // 4
    op_adj = np.full((4, t_max, t_max), -1e4, dtype=np.float32)

    diag_idx = np.arange(t_max)
    op_adj[:, diag_idx, diag_idx] = 0.0

    for fi in range(n_factors):
        ft = int(factor_types_np[fi])
        if ft < 0 or ft >= 4:
            continue
        fpos = n_var_tok + fi
        for vi_raw in factor_args_np[fi]:
            vi = int(vi_raw)
            if 0 <= vi < n_vars:
                vstart = vi * n_digits
                vend   = vstart + n_digits
                op_adj[ft, vstart:vend, fpos] = 0.0
                op_adj[ft, fpos, vstart:vend] = 0.0
        for vi_raw in factor_args_np[fi]:
            vi = int(vi_raw)
            if 0 <= vi < n_vars:
                vstart = vi * n_digits
                vend   = vstart + n_digits
                # Within-variable digit attention (same gating as bipartite_full)
                if V105_5_BLOCK_WITHIN_VAR == 0:
                    op_adj[ft, vstart:vend, vstart:vend] = 0.0
                elif V105_5_BLOCK_WITHIN_VAR == 1:
                    # Hard block: diagonal only
                    for p in range(n_digits):
                        op_adj[ft, vstart + p, vstart + p] = 0.0
                elif V105_5_BLOCK_WITHIN_VAR == 2:
                    # Soft (adjacent-only)
                    for p1 in range(n_digits):
                        for p2 in range(n_digits):
                            if abs(p1 - p2) <= 1:
                                op_adj[ft, vstart + p1, vstart + p2] = 0.0

        # Lateral attention bias: same-position digits ACROSS operands of a
        # factor attend to each other (the per-position arithmetic pathway).
        # Without this, the model can only learn "digit p of var A talks to
        # digit p of var B" through indirect routes via the factor node.
        # Gated by V105_5_LATERAL_ATTN env var (default off for back-compat).
        if V105_5_LATERAL_ATTN:
            for vi_a_raw in factor_args_np[fi]:
                vi_a = int(vi_a_raw)
                if not (0 <= vi_a < n_vars):
                    continue
                for vi_b_raw in factor_args_np[fi]:
                    vi_b = int(vi_b_raw)
                    if not (0 <= vi_b < n_vars):
                        continue
                    for p in range(n_digits):
                        t_a = vi_a * n_digits + p
                        t_b = vi_b * n_digits + p
                        op_adj[ft, t_a, t_b] = 0.0

    head_ops = np.repeat(op_adj[:, np.newaxis, :, :], heads_per_op, axis=1).reshape(n_heads, t_max, t_max)

    return staging, head_ops


# ---------------------------------------------------------------------------
# Batch construction
# ---------------------------------------------------------------------------

def _execute_fg_v105_5(rec: dict) -> list[float] | None:
    """Execute a factor graph to completion, returning values for all variables.

    Same as v105.1.2's executor — no range restriction. Returns None on failure.
    """
    ft_list  = rec.get("factor_types", [])
    fa_list  = rec.get("factor_args", [])
    obs_vals = list(rec.get("observed_values", []))

    vals = list(obs_vals)
    ops  = {
        "add": lambda a, b: a + b,
        "sub": lambda a, b: a - b,
        "mul": lambda a, b: a * b,
        "div": lambda a, b: a / b if b != 0 else None,
    }

    changed = True
    iters   = 0
    while changed and iters < 30:
        changed = False
        iters  += 1
        for ft, fa in zip(ft_list, fa_list):
            a1, a2, res = int(fa[0]), int(fa[1]), int(fa[2])
            if (a1 < len(vals) and a2 < len(vals) and res < len(vals)
                    and vals[a1] is not None and vals[a2] is not None
                    and vals[res] is None):
                try:
                    v = ops.get(ft, lambda a, b: None)(vals[a1], vals[a2])
                    if v is None:
                        continue
                    vals[res] = float(v)
                    changed = True
                except Exception:
                    pass

    if any(v is None for v in vals):
        return None
    return vals


def convert_gsm8k_record_v105_5(
    rec: dict,
    n_digits: int = V105_N_DIGITS,
    n_max: int = V105_N_MAX,
    f_max: int = V105_F_MAX,
) -> dict | None:
    """Convert a GSM8K factor graph record to v105.3 format.

    Identical to v105.1.2's converter — the record itself stores integer values;
    the LSD-first encoding happens at batch-build time.
    """
    vals = _execute_fg_v105_5(rec)
    if vals is None:
        return None

    try:
        int_vals = [int(round(v)) for v in vals]
    except Exception:
        return None
    if any(abs(int_vals[i] - vals[i]) > 0.01 for i in range(len(vals))):
        return None

    if any(v < 0 for v in int_vals):
        return None

    ft_list  = rec.get("factor_types", [])
    fa_list  = rec.get("factor_args",  [])
    obs_mask = rec.get("observed_mask", [])
    obs_vals = rec.get("observed_values", [])
    n_vars   = rec.get("n_vars", 0)
    n_facts  = rec.get("n_factors", 0)
    qi       = int(rec.get("query_idx", 0))

    obs_vals_int = [int(round(obs_vals[i])) if obs_mask[i] == 1 else int_vals[i]
                    for i in range(len(obs_mask))]

    return {
        "gold_values":     int_vals,
        "observed_mask":   obs_mask,
        "observed_values": obs_vals_int,
        "factor_types":    ft_list,
        "factor_args":     fa_list,
        "n_factors":       n_facts,
        "query_idx":       qi,
        "difficulty":      "gsm8k",
    }


def _records_to_batch_v105_5(
    picks: list[dict],
    n_max: int = V105_N_MAX,
    f_max: int = V105_F_MAX,
    k_max: int = V105_K_MAX,
    n_heads: int = V105_N_HEADS,
    n_digits: int = V105_N_DIGITS,
) -> dict:
    """Convert a list of records to numpy arrays for one v105.5 batch.

    Fields vs v105.3:
      magnitude_target : (B, N_MAX) int32 — 4-way magnitude class (gold_magnitude_class)
                         Padded positions = 0 (will be masked out via digit_valid_mask
                         and is_real_var downstream).

    Other fields unchanged.
    """
    B = len(picks)
    T = n_max * n_digits + f_max

    digit_init     = np.full((B, n_max, n_digits, 10), 1.0 / 10.0, dtype=np.float32)
    node_kinds     = np.full((B, T), -1, dtype=np.int32)
    gold_digits_np = np.zeros((B, n_max, n_digits), dtype=np.int32)
    obs_mask       = np.zeros((B, n_max), dtype=np.int32)
    factor_types_np = np.full((B, f_max), -1, dtype=np.int32)
    factor_args_np  = np.full((B, f_max, 3), -1, dtype=np.int32)
    query_idx_np   = np.zeros((B,), dtype=np.int32)
    n_vars_np      = np.zeros((B,), dtype=np.int32)
    n_factors_np   = np.zeros((B,), dtype=np.int32)
    var_depth_np   = np.full((B, n_max), -1, dtype=np.int32)
    staging_masks  = np.full((B, k_max, T, T), -1e4, dtype=np.float32)
    head_op_masks  = np.full((B, n_heads, T, T), -1e4, dtype=np.float32)
    factor_gold_dg = np.zeros((B, f_max, n_digits), dtype=np.int32)
    factor_valid   = np.zeros((B, f_max), dtype=np.float32)

    # Per-position validity masks (LSD-flipped semantics).
    digit_valid_mask        = np.zeros((B, n_max, n_digits), dtype=np.float32)
    factor_digit_valid_mask = np.zeros((B, f_max, n_digits), dtype=np.float32)

    # NEW in v105.5: 4-way magnitude target per variable cell.
    magnitude_target = np.zeros((B, n_max), dtype=np.int32)

    # NEW in v105.8: per-NUMBER bin target per variable cell.
    # Padded positions stay 0 (masked out via unobs_real downstream).
    number_bin_target = np.zeros((B, n_max), dtype=np.int32)

    n_var_tok = n_max * n_digits

    for b, rec in enumerate(picks):
        gold_vals    = rec["gold_values"]
        obs_mask_rec = rec["observed_mask"]
        obs_vals     = rec["observed_values"]
        ft_list      = rec["factor_types"]
        fa_list      = rec["factor_args"]
        n_total      = len(gold_vals)
        nf           = rec["n_factors"]

        n_vars_np[b]    = min(n_total, n_max)
        n_factors_np[b] = min(nf, f_max)
        query_idx_np[b] = int(rec["query_idx"])

        # Fill variable positions (LSD-first)
        for vi in range(min(n_total, n_max)):
            gv = max(0, int(round(gold_vals[vi])))
            digs = value_to_digits_lsd(gv, n_digits)   # LSD-first
            gold_digits_np[b, vi] = digs

            # Validity mask is derived from the GOLD value (always known to the data loader)
            digit_valid_mask[b, vi] = build_digit_valid_mask(gv, n_digits)

            # Magnitude target (LSD-first independent — depends only on value)
            magnitude_target[b, vi] = gold_magnitude_class(gv)

            # v105.8 — per-NUMBER bin target (uses v107's hybrid scheme by default)
            number_bin_target[b, vi] = value_to_number_bin(gv)

            tok_base = vi * n_digits
            if obs_mask_rec[vi] == 1:
                obs_val = max(
                    0,
                    int(round(obs_vals[vi])) if obs_vals[vi] is not None else gv,
                )
                obs_digs = value_to_digits_lsd(obs_val, n_digits)   # LSD-first
                for p in range(n_digits):
                    digit_init[b, vi, p, :] = 0.0
                    digit_init[b, vi, p, obs_digs[p]] = 1.0
                    node_kinds[b, tok_base + p] = 0
                obs_mask[b, vi] = 1
            else:
                for p in range(n_digits):
                    node_kinds[b, tok_base + p] = 1
                obs_mask[b, vi] = 0

        # Fill factor type/args (LSD-first gold digits)
        for fi in range(min(nf, f_max)):
            ft_str = ft_list[fi]
            ft_int = OP_MAP.get(ft_str, -1)
            fa     = fa_list[fi]
            factor_types_np[b, fi] = ft_int
            for j in range(3):
                factor_args_np[b, fi, j] = int(fa[j])
            fpos = n_var_tok + fi
            node_kinds[b, fpos] = 2

            r_idx = int(fa[2])
            if 0 <= r_idx < n_total:
                r_val = max(0, int(round(gold_vals[r_idx])))
                factor_gold_dg[b, fi] = value_to_digits_lsd(r_val, n_digits)   # LSD-first
                factor_valid[b, fi]   = 1.0
                factor_digit_valid_mask[b, fi] = build_digit_valid_mask(r_val, n_digits)

        # DAG depths
        depth_dict = compute_var_depth(
            obs_mask_rec,
            [fa_list[fi] for fi in range(min(nf, f_max))],
            n_total,
        )
        for vi in range(min(n_total, n_max)):
            var_depth_np[b, vi] = depth_dict.get(vi, -1)

        staging_b, head_op_b = build_staging_and_head_masks_v105_5_np(
            factor_types_np[b],
            factor_args_np[b],
            var_depth_np[b],
            n_vars=int(n_vars_np[b]),
            n_factors=int(n_factors_np[b]),
            n_max=n_max, f_max=f_max, k_max=k_max, n_heads=n_heads, n_digits=n_digits,
        )
        staging_masks[b] = staging_b
        head_op_masks[b] = head_op_b

    return {
        "digit_init":              digit_init,
        "node_kinds":              node_kinds,
        "gold_digits":             gold_digits_np,
        "observed_mask":           obs_mask,
        "factor_types":            factor_types_np,
        "factor_args":             factor_args_np,
        "factor_gold_dg":          factor_gold_dg,
        "factor_valid":            factor_valid,
        "query_idx":               query_idx_np,
        "n_vars_total":            n_vars_np,
        "n_factors":               n_factors_np,
        "var_depth":               var_depth_np,
        "staging_mask":            staging_masks,
        "head_op_mask":            head_op_masks,
        "digit_valid_mask":        digit_valid_mask,
        "factor_digit_valid_mask": factor_digit_valid_mask,
        "magnitude_target":        magnitude_target,
        "number_bin_target":       number_bin_target,
    }


def batch_to_tensors_v105_5(batch_np: dict) -> dict:
    """Convert v105.5 numpy batch dict to tinygrad Tensors (realized)."""
    return {
        "digit_init":              Tensor(batch_np["digit_init"],              dtype=dtypes.float).contiguous().realize(),
        "node_kinds":              Tensor(batch_np["node_kinds"],              dtype=dtypes.int).contiguous().realize(),
        "gold_digits":             Tensor(batch_np["gold_digits"],             dtype=dtypes.int).contiguous().realize(),
        "observed_mask":           Tensor(batch_np["observed_mask"],           dtype=dtypes.int).contiguous().realize(),
        "factor_types":            Tensor(batch_np["factor_types"],            dtype=dtypes.int).contiguous().realize(),
        "factor_args":             Tensor(batch_np["factor_args"],             dtype=dtypes.int).contiguous().realize(),
        "factor_gold_dg":          Tensor(batch_np["factor_gold_dg"],          dtype=dtypes.int).contiguous().realize(),
        "factor_valid":            Tensor(batch_np["factor_valid"],            dtype=dtypes.float).contiguous().realize(),
        "staging_mask":            Tensor(batch_np["staging_mask"],            dtype=dtypes.float).contiguous().realize(),
        "head_op_mask":            Tensor(batch_np["head_op_mask"],            dtype=dtypes.float).contiguous().realize(),
        "digit_valid_mask":        Tensor(batch_np["digit_valid_mask"],        dtype=dtypes.float).contiguous().realize(),
        "factor_digit_valid_mask": Tensor(batch_np["factor_digit_valid_mask"], dtype=dtypes.float).contiguous().realize(),
        "magnitude_target":        Tensor(batch_np["magnitude_target"],        dtype=dtypes.int).contiguous().realize(),
        "number_bin_target":       Tensor(batch_np["number_bin_target"],       dtype=dtypes.int).contiguous().realize(),
        # numpy-only
        "query_idx":      batch_np["query_idx"],
        "n_vars_total":   batch_np["n_vars_total"],
        "n_factors":      batch_np["n_factors"],
    }


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def load_gsm8k_records_v105_5(
    path: str,
    n_digits: int = V105_N_DIGITS,
    n_max: int = V105_N_MAX,
    f_max: int = V105_F_MAX,
) -> list[dict]:
    """Load and convert GSM8K factor graph records to v105.3 format."""
    if not path or not os.path.exists(path):
        return []
    records = []
    n_loaded = 0
    with open(path) as f:
        for line in f:
            rec = json.loads(line.strip())
            n_loaded += 1
            converted = convert_gsm8k_record_v105_5(rec, n_digits=n_digits, n_max=n_max, f_max=f_max)
            if converted is not None:
                records.append(converted)
    pct = 100.0 * len(records) / max(n_loaded, 1)
    print(
        f"[fg_data_v105_5] GSM8K: loaded {n_loaded}, kept {len(records)} ({pct:.1f}%) "
        f"(LSD-first, no range filter; n_digits={n_digits})",
        flush=True,
    )
    return records


class FactorGraphLoaderV105_5:
    """Iterates factor-graph records in v105.3 format (LSD-first + valid mask)."""

    def __init__(
        self,
        path: str,
        batch_size: int = 8,
        difficulty_filter: str | None = None,
        curriculum: bool = False,
        curriculum_anneal_steps: int = 1000,
        n_max: int = V105_N_MAX,
        f_max: int = V105_F_MAX,
        k_max: int = V105_K_MAX,
        n_heads: int = V105_N_HEADS,
        n_digits: int = V105_N_DIGITS,
        seed: int = 0,
    ):
        self.batch_size     = batch_size
        self.difficulty_filter = difficulty_filter
        self.curriculum     = curriculum
        self.curriculum_anneal_steps = curriculum_anneal_steps
        self.n_max          = n_max
        self.f_max          = f_max
        self.k_max          = k_max
        self.n_heads        = n_heads
        self.n_digits       = n_digits
        self.rng            = random.Random(seed)

        records = load_jsonl(path)
        if difficulty_filter is not None:
            records = [r for r in records if r.get("difficulty") == difficulty_filter]
            assert records, f"No records match difficulty={difficulty_filter} in {path}"

        self.records = records
        self.by_diff = index_by_difficulty(records)
        self.active_diffs = [d for d in DIFFICULTIES if len(self.by_diff.get(d, [])) > 0]
        print(
            f"[fg_data_v105_5] loaded {len(records)} from {os.path.basename(path)}; "
            f"n_digits={n_digits} (LSD-first) T={n_max*n_digits+f_max}  "
            f"by difficulty: {[(d, len(self.by_diff.get(d,[]))) for d in self.active_diffs]}",
            flush=True,
        )

    def _step_difficulty_weights(self, step: int) -> dict[str, float]:
        if not self.curriculum or "easy" not in self.active_diffs:
            return {d: 1.0 / len(self.active_diffs) for d in self.active_diffs}
        progress = min(1.0, step / max(1.0, float(self.curriculum_anneal_steps)))
        N = len(self.active_diffs)
        weights: dict[str, float] = {}
        for d in self.active_diffs:
            if d == "easy":
                weights[d] = (1.0 - progress) + progress * (1.0 / N)
            else:
                weights[d] = progress * (1.0 / N)
        s = sum(weights.values())
        return {d: w / s for d, w in weights.items()}

    def sample_batch(self, step: int = 0) -> dict:
        if self.difficulty_filter is not None:
            picks = self.rng.choices(self.records, k=self.batch_size)
        else:
            weights = self._step_difficulty_weights(step)
            picks = []
            for _ in range(self.batch_size):
                diff = self.rng.choices(
                    self.active_diffs,
                    weights=[weights[d] for d in self.active_diffs],
                    k=1,
                )[0]
                picks.append(self.rng.choice(self.by_diff[diff]))

        batch_np = _records_to_batch_v105_5(
            picks, self.n_max, self.f_max, self.k_max, self.n_heads, self.n_digits,
        )
        batch_t = batch_to_tensors_v105_5(batch_np)
        batch_t["picks"] = picks
        return batch_t

    def iter_eval(self, batch_size: int | None = None) -> Iterator[dict]:
        bs = batch_size or self.batch_size
        n  = len(self.records)
        for start in range(0, n, bs):
            batch_recs = self.records[start : start + bs]
            while len(batch_recs) < bs:
                batch_recs.append(self.records[0])
            batch_np = _records_to_batch_v105_5(
                batch_recs, self.n_max, self.f_max, self.k_max, self.n_heads, self.n_digits,
            )
            batch_t = batch_to_tensors_v105_5(batch_np)
            batch_t["picks"] = batch_recs
            yield batch_t

    def __len__(self) -> int:
        return len(self.records)


class DualDataLoaderV105_5:
    """Samples from synthetic + GSM8K records at configurable ratio (v105.3 format)."""

    def __init__(
        self,
        synth_loader: FactorGraphLoaderV105_5,
        gsm8k_records: list[dict],
        gsm8k_ratio: float = 0.5,
        n_max: int = V105_N_MAX,
        f_max: int = V105_F_MAX,
        k_max: int = V105_K_MAX,
        n_heads: int = V105_N_HEADS,
        n_digits: int = V105_N_DIGITS,
        seed: int = 42,
    ):
        self.synth_loader  = synth_loader
        self.gsm8k_records = gsm8k_records
        self.gsm8k_ratio   = gsm8k_ratio if gsm8k_records else 0.0
        self.n_max    = n_max
        self.f_max    = f_max
        self.k_max    = k_max
        self.n_heads  = n_heads
        self.n_digits = n_digits
        self.rng      = random.Random(seed)
        print(
            f"[fg_data_v105_5] DualLoader: synth={len(synth_loader)} gsm8k={len(gsm8k_records)} "
            f"ratio={self.gsm8k_ratio:.2f}",
            flush=True,
        )

    def sample_batch(self, step: int = 0) -> dict:
        synth_batch  = self.synth_loader.sample_batch(step=step)
        synth_picks  = synth_batch["picks"]
        B            = len(synth_picks)

        if self.gsm8k_ratio <= 0 or not self.gsm8k_records:
            return synth_batch

        n_gsm8k = sum(1 for _ in range(B) if self.rng.random() < self.gsm8k_ratio)
        n_synth = B - n_gsm8k

        mixed: list[dict] = []
        if n_synth > 0:
            mixed += self.rng.choices(synth_picks, k=n_synth)
        if n_gsm8k > 0:
            mixed += self.rng.choices(self.gsm8k_records, k=n_gsm8k)
        self.rng.shuffle(mixed)

        batch_np = _records_to_batch_v105_5(mixed, self.n_max, self.f_max, self.k_max,
                                            self.n_heads, self.n_digits)
        batch_t  = batch_to_tensors_v105_5(batch_np)
        batch_t["picks"] = mixed
        return batch_t
