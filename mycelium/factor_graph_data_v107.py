"""v107 factor graph data loader — hybrid 200-bin codebook.

Extends the v100 data loader (staging + head-op masks) with the following changes:
  NOTE (v114): if env V114_MIRROR_AT_K is set to a positive integer, the staging
  mask for breaths >= mirror_at_k is built with reversed topological ordering
  (deepest node first, expanding back toward inputs). This implements the
  bidirectional "compute forward / verify backward" hypothesis.

  1. VALUE RANGE: No [0,99] filtering. The 200-bin hybrid codebook handles [0,9999].
     Values above 9999 are clamped to the nearest boundary bin (bin 199 = 9999).
     The 4261 GSM8K records in .cache/gsm8k_factor_graphs_train.jsonl are
     included in full (only ~103 records exceed 9999; those get clamped to bin 199).

  2. BIN ASSIGNMENT:
     gold_bins: (B, N_MAX) int32 — bin index (0..199) for each variable's gold value.
       - For values in [0,99]: bin_idx == value (linear region, exact match).
       - For values in [100,9999]: nearest log-spaced bin.
       - For values > 9999: bin 199 (clamped).
     domain_init: (B, N_MAX, 200) float32
       - Observed: one-hot at the gold bin index.
       - Unobserved: uniform 1/200.

  3. FACTOR GOLD BINS:
     factor_gold_bin: (B, F_MAX) int32 — bin index of the gold result per factor.
     factor_valid: (B, F_MAX) float32 — 1=real factor with valid result, 0=pad.

  4. TRAINING DATA:
     50K synthetic factor graphs (values in [0,99], exact bin = value).
     4261 GSM8K factor graphs (full value range, bin assignment via nearest_bin).
     50/50 mix during training (DualDataLoaderV107).

  All mask construction (staging + head-op) is unchanged from v100.
"""
from __future__ import annotations

import json
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
from mycelium.factor_graph_data_v100 import (
    compute_var_depth,
    build_staging_and_head_masks_np,
    batch_to_tensors_v100,
)
from mycelium.factor_graph_v107 import (
    V107_N_MAX, V107_F_MAX, V107_K_MAX, V107_N_HEADS,
    get_bin_values, nearest_bin,
)

# ---------------------------------------------------------------------------
# v114: mirror staging mask (bidirectional: forward then backward)
# ---------------------------------------------------------------------------

_V114_MIRROR_AT_K = int(os.environ.get("V114_MIRROR_AT_K", "0"))


def build_staging_and_head_masks_mirror_np(
    factor_types_np: np.ndarray,    # (F_MAX,) int, -1=padding
    factor_args_np: np.ndarray,     # (F_MAX, 3) int
    var_depth_np: np.ndarray,       # (N_MAX,) int  (-1 = padding)
    n_vars: int,
    n_factors: int,
    n_max: int,
    f_max: int,
    k_max: int,
    n_heads: int,
    mirror_at_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build staging mask with mirror transform for breaths k >= mirror_at_k.

    Breaths 0..mirror_at_k-1: forward staging (depth <= k+1 visible).
    Breaths mirror_at_k..k_max-1: reversed — deepest node first, expanding
    back toward observed inputs as k increases past mirror_at_k.

    Head-op masks are unchanged (op-type assignment is direction-agnostic).
    Returns:
      staging  : (k_max, t_max, t_max) float32
      head_ops : (n_heads, t_max, t_max) float32
    """
    # Delegate to forward helper for the shared head-op mask and bipartite graph
    staging_fwd, head_ops = build_staging_and_head_masks_np(
        factor_types_np, factor_args_np, var_depth_np,
        n_vars=n_vars, n_factors=n_factors,
        n_max=n_max, f_max=f_max, k_max=k_max, n_heads=n_heads,
    )

    if mirror_at_k <= 0:
        return staging_fwd, head_ops

    # Compute max depth of valid variables
    valid_depths = var_depth_np[:n_vars]
    valid_depths = valid_depths[valid_depths >= 0]
    if len(valid_depths) == 0:
        return staging_fwd, head_ops
    max_depth = int(valid_depths.max())

    # Build factor result depths (same logic as forward helper)
    factor_result_depth = np.full(f_max, -1, dtype=np.int32)
    for fi in range(n_factors):
        ft = int(factor_types_np[fi])
        if ft < 0:
            continue
        res_idx = int(factor_args_np[fi, 2])
        if 0 <= res_idx < n_max and var_depth_np[res_idx] >= 0:
            factor_result_depth[fi] = int(var_depth_np[res_idx])

    # Need bipartite mask for combining visibility — extract from staging_fwd[k_max-1]
    # (at last breath, everything visible → bipartite is the full allowed mask)
    bipartite = staging_fwd[k_max - 1]  # (t_max, t_max)

    t_max = n_max + f_max
    staging = staging_fwd.copy()

    for k in range(mirror_at_k, k_max):
        reverse_progress = k - mirror_at_k  # 0 at mirror_at_k, grows with k

        # Reversed variable visibility: deepest nodes are visible first.
        # At reverse_progress=0: only nodes with depth == max_depth are visible.
        # At reverse_progress=p: nodes with depth >= max_depth - p are visible.
        # Observed nodes (depth 0) are always visible (anchor inputs).
        pv = np.zeros(t_max, dtype=bool)
        for vi in range(n_vars):
            d = int(var_depth_np[vi])
            if d < 0:
                continue
            if d == 0:  # observed input — always visible
                pv[vi] = True
            else:
                threshold = max_depth - reverse_progress
                pv[vi] = (d >= threshold)

        # Reversed factor visibility: factor visible if its result var is visible
        for fi in range(n_factors):
            fd = factor_result_depth[fi]
            if fd < 0:
                continue
            fpos = n_max + fi
            if fd == 0:
                pv[fpos] = True
            else:
                threshold = max_depth - reverse_progress
                pv[fpos] = (fd >= threshold)

        both_visible = pv[:, np.newaxis] & pv[np.newaxis, :]  # (t_max, t_max)
        staging[k] = np.where(both_visible, bipartite, -1e4)

    return staging, head_ops

# ---------------------------------------------------------------------------
# GSM8K record conversion (no [0,99] filter; bin-assign all values)
# ---------------------------------------------------------------------------

def _execute_fg_v107(rec: dict) -> list[float] | None:
    """Execute a factor graph to completion. No value range restriction."""
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


def convert_gsm8k_record_v107(rec: dict) -> dict | None:
    """Convert a GSM8K factor graph record to v107 format.

    Returns None if:
      - Factor graph cannot be executed.
      - Any value is non-integer.
      - Any value is negative.

    Values > 9999 are KEPT (clamped to bin 199 at batch time, not filtered out).
    This retains all ~4261 GSM8K records.
    """
    vals = _execute_fg_v107(rec)
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
    n_facts  = rec.get("n_factors", 0)
    qi       = int(rec.get("query_idx", 0))

    obs_vals_int = [
        int(round(obs_vals[i])) if obs_mask[i] == 1 else int_vals[i]
        for i in range(len(obs_mask))
    ]

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


def load_gsm8k_records_v107(path: str) -> list[dict]:
    """Load and convert GSM8K factor graph records to v107 format (no range filter)."""
    if not path or not os.path.exists(path):
        return []
    records = []
    n_loaded = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            n_loaded += 1
            converted = convert_gsm8k_record_v107(rec)
            if converted is not None:
                records.append(converted)
    pct = 100.0 * len(records) / max(n_loaded, 1)
    print(
        f"[fg_data_v107] GSM8K: loaded {n_loaded}, kept {len(records)} ({pct:.1f}%) "
        "(no range filter; values > 9999 clamped to bin 199 at batch time)",
        flush=True,
    )
    return records


# ---------------------------------------------------------------------------
# Batch construction
# ---------------------------------------------------------------------------

def _records_to_batch_v107(
    picks: list[dict],
    n_max: int = V107_N_MAX,
    f_max: int = V107_F_MAX,
    k_max: int = V107_K_MAX,
    n_heads: int = V107_N_HEADS,
) -> dict:
    """Convert a list of records to numpy arrays for one v107 batch.

    Key differences from v100:
      - domain_init:     (B, N_MAX, 200) instead of (B, N_MAX, 100)
      - gold_bins:       (B, N_MAX) int — bin index per variable
      - factor_gold_bin: (B, F_MAX) int — bin index of factor result
      - factor_valid:    (B, F_MAX) float
      - gold_values:     (B, N_MAX) int — raw integer values (for energy / diagnostics)

    Staging mask and head-op mask: identical to v100 (same geometry).
    """
    B     = len(picks)
    t_max = n_max + f_max
    bv    = get_bin_values()   # (200,) int64

    domain_init_np   = np.full((B, n_max, 200), 1.0 / 200.0, dtype=np.float32)
    node_kinds_np    = np.full((B, t_max), -1, dtype=np.int32)
    gold_bins_np     = np.zeros((B, n_max), dtype=np.int32)
    gold_values_np   = np.zeros((B, n_max), dtype=np.int32)   # raw integer (diagnostics)
    obs_mask_np      = np.zeros((B, n_max), dtype=np.int32)
    factor_types_np  = np.full((B, f_max), -1, dtype=np.int32)
    factor_args_np   = np.full((B, f_max, 3), -1, dtype=np.int32)
    factor_gold_bin  = np.zeros((B, f_max), dtype=np.int32)
    factor_valid_np  = np.zeros((B, f_max), dtype=np.float32)
    query_idx_np     = np.zeros((B,), dtype=np.int32)
    n_vars_np        = np.zeros((B,), dtype=np.int32)
    n_factors_np     = np.zeros((B,), dtype=np.int32)
    var_depth_np     = np.full((B, n_max), -1, dtype=np.int32)
    staging_masks    = np.full((B, k_max, t_max, t_max), -1e4, dtype=np.float32)
    head_op_masks    = np.full((B, n_heads, t_max, t_max), -1e4, dtype=np.float32)

    for b, rec in enumerate(picks):
        gold_vals    = rec["gold_values"]
        obs_mask_rec = rec["observed_mask"]
        obs_vals     = rec["observed_values"]
        ft_list      = rec["factor_types"]
        fa_list      = rec["factor_args"]
        n_total      = len(gold_vals)
        nf           = rec["n_factors"]

        n_vars_np[b]   = min(n_total, n_max)
        n_factors_np[b] = min(nf, f_max)
        query_idx_np[b] = int(rec["query_idx"])

        for vi in range(min(n_total, n_max)):
            gv = int(round(gold_vals[vi]))
            gv = max(0, gv)
            gold_values_np[b, vi] = gv
            bi = nearest_bin(gv, bv)
            gold_bins_np[b, vi] = bi

            if obs_mask_rec[vi] == 1:
                ov = int(round(obs_vals[vi])) if obs_vals[vi] is not None else gv
                ov = max(0, ov)
                ov_bi = nearest_bin(ov, bv)
                domain_init_np[b, vi, :] = 0.0
                domain_init_np[b, vi, ov_bi] = 1.0
                obs_mask_np[b, vi] = 1
                node_kinds_np[b, vi] = 0
            else:
                # uniform 1/200 (already set at init above)
                obs_mask_np[b, vi] = 0
                node_kinds_np[b, vi] = 1

        for fi in range(min(nf, f_max)):
            ft_str = ft_list[fi]
            ft_int = OP_MAP.get(ft_str, -1)
            fa     = fa_list[fi]
            factor_types_np[b, fi] = ft_int
            for j in range(3):
                factor_args_np[b, fi, j] = int(fa[j])
            fpos = n_max + fi
            node_kinds_np[b, fpos] = 2

            r_idx = int(fa[2])
            if 0 <= r_idx < n_total:
                rv = int(round(gold_vals[r_idx]))
                rv = max(0, rv)
                factor_gold_bin[b, fi] = nearest_bin(rv, bv)
                factor_valid_np[b, fi] = 1.0

        depth_dict = compute_var_depth(
            obs_mask_rec,
            [fa_list[fi] for fi in range(min(nf, f_max))],
            n_total,
        )
        for vi in range(min(n_total, n_max)):
            var_depth_np[b, vi] = depth_dict.get(vi, -1)

        if _V114_MIRROR_AT_K > 0:
            staging_b, head_op_b = build_staging_and_head_masks_mirror_np(
                factor_types_np[b],
                factor_args_np[b],
                var_depth_np[b],
                n_vars=int(n_vars_np[b]),
                n_factors=int(n_factors_np[b]),
                n_max=n_max,
                f_max=f_max,
                k_max=k_max,
                n_heads=n_heads,
                mirror_at_k=_V114_MIRROR_AT_K,
            )
        else:
            staging_b, head_op_b = build_staging_and_head_masks_np(
                factor_types_np[b],
                factor_args_np[b],
                var_depth_np[b],
                n_vars=int(n_vars_np[b]),
                n_factors=int(n_factors_np[b]),
                n_max=n_max,
                f_max=f_max,
                k_max=k_max,
                n_heads=n_heads,
            )
        staging_masks[b] = staging_b
        head_op_masks[b] = head_op_b

    return {
        "domain_init":    domain_init_np,
        "node_kinds":     node_kinds_np,
        "gold_bins":      gold_bins_np,
        "gold_values":    gold_values_np,
        "observed_mask":  obs_mask_np,
        "factor_types":   factor_types_np,
        "factor_args":    factor_args_np,
        "factor_gold_bin": factor_gold_bin,
        "factor_valid":   factor_valid_np,
        "query_idx":      query_idx_np,
        "n_vars_total":   n_vars_np,
        "n_factors":      n_factors_np,
        "var_depth":      var_depth_np,
        "staging_mask":   staging_masks,
        "head_op_mask":   head_op_masks,
    }


def batch_to_tensors_v107(batch_np: dict) -> dict:
    """Convert v107 numpy batch dict to tinygrad Tensors (realized)."""
    return {
        "domain_init":    Tensor(batch_np["domain_init"],    dtype=dtypes.float).contiguous().realize(),
        "node_kinds":     Tensor(batch_np["node_kinds"],     dtype=dtypes.int).contiguous().realize(),
        "gold_bins":      Tensor(batch_np["gold_bins"],      dtype=dtypes.int).contiguous().realize(),
        "gold_values":    Tensor(batch_np["gold_values"],    dtype=dtypes.int).contiguous().realize(),
        "observed_mask":  Tensor(batch_np["observed_mask"],  dtype=dtypes.int).contiguous().realize(),
        "factor_types":   Tensor(batch_np["factor_types"],   dtype=dtypes.int).contiguous().realize(),
        "factor_args":    Tensor(batch_np["factor_args"],    dtype=dtypes.int).contiguous().realize(),
        "factor_gold_bin":Tensor(batch_np["factor_gold_bin"],dtype=dtypes.int).contiguous().realize(),
        "factor_valid":   Tensor(batch_np["factor_valid"],   dtype=dtypes.float).contiguous().realize(),
        "staging_mask":   Tensor(batch_np["staging_mask"],   dtype=dtypes.float).contiguous().realize(),
        "head_op_mask":   Tensor(batch_np["head_op_mask"],   dtype=dtypes.float).contiguous().realize(),
        # numpy-only
        "query_idx":     batch_np["query_idx"],
        "n_vars_total":  batch_np["n_vars_total"],
        "n_factors":     batch_np["n_factors"],
    }


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

class FactorGraphLoaderV107:
    """Iterates factor-graph records in v107 format (200-bin hybrid codebook)."""

    def __init__(
        self,
        path: str,
        batch_size: int = 8,
        difficulty_filter: str | None = None,
        curriculum: bool = False,
        curriculum_anneal_steps: int = 1000,
        n_max: int = V107_N_MAX,
        f_max: int = V107_F_MAX,
        k_max: int = V107_K_MAX,
        n_heads: int = V107_N_HEADS,
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
        self.rng            = random.Random(seed)

        records = load_jsonl(path)
        if difficulty_filter is not None:
            records = [r for r in records if r.get("difficulty") == difficulty_filter]
            assert records, f"No records match difficulty={difficulty_filter} in {path}"

        self.records     = records
        self.by_diff     = index_by_difficulty(records)
        self.active_diffs = [d for d in DIFFICULTIES if len(self.by_diff.get(d, [])) > 0]
        print(
            f"[fg_data_v107] loaded {len(records)} from {os.path.basename(path)}; "
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

        batch_np = _records_to_batch_v107(picks, self.n_max, self.f_max, self.k_max, self.n_heads)
        batch_t  = batch_to_tensors_v107(batch_np)
        batch_t["picks"] = picks
        return batch_t

    def iter_eval(self, batch_size: int | None = None) -> Iterator[dict]:
        bs = batch_size or self.batch_size
        n  = len(self.records)
        for start in range(0, n, bs):
            batch_recs = self.records[start : start + bs]
            while len(batch_recs) < bs:
                batch_recs.append(self.records[0])
            batch_np = _records_to_batch_v107(
                batch_recs, self.n_max, self.f_max, self.k_max, self.n_heads
            )
            batch_t = batch_to_tensors_v107(batch_np)
            batch_t["picks"] = batch_recs
            yield batch_t

    def __len__(self) -> int:
        return len(self.records)


class DualDataLoaderV107:
    """Samples 50/50 (configurable) from synthetic + GSM8K records (v107 format)."""

    def __init__(
        self,
        synth_loader: FactorGraphLoaderV107,
        gsm8k_records: list[dict],
        gsm8k_ratio: float = 0.5,
        n_max: int = V107_N_MAX,
        f_max: int = V107_F_MAX,
        k_max: int = V107_K_MAX,
        n_heads: int = V107_N_HEADS,
        seed: int = 42,
    ):
        self.synth_loader  = synth_loader
        self.gsm8k_records = gsm8k_records
        self.gsm8k_ratio   = gsm8k_ratio if gsm8k_records else 0.0
        self.n_max         = n_max
        self.f_max         = f_max
        self.k_max         = k_max
        self.n_heads       = n_heads
        self.rng           = random.Random(seed)
        print(
            f"[fg_data_v107] DualLoader: synth={len(synth_loader)} "
            f"gsm8k={len(gsm8k_records)} ratio={self.gsm8k_ratio:.2f}",
            flush=True,
        )

    def sample_batch(self, step: int = 0) -> dict:
        synth_batch = self.synth_loader.sample_batch(step=step)
        synth_picks = synth_batch["picks"]
        B           = len(synth_picks)

        if self.gsm8k_ratio <= 0 or not self.gsm8k_records:
            return synth_batch

        n_gsm8k = sum(1 for _ in range(B) if self.rng.random() < self.gsm8k_ratio)
        n_synth  = B - n_gsm8k

        mixed: list[dict] = []
        if n_synth > 0:
            mixed += self.rng.choices(synth_picks, k=n_synth)
        if n_gsm8k > 0:
            mixed += self.rng.choices(self.gsm8k_records, k=n_gsm8k)
        self.rng.shuffle(mixed)

        batch_np = _records_to_batch_v107(mixed, self.n_max, self.f_max, self.k_max, self.n_heads)
        batch_t  = batch_to_tensors_v107(batch_np)
        batch_t["picks"] = mixed
        return batch_t
