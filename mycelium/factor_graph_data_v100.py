"""v100 factor graph data loader — adds DAG depth + topological staging masks.

Extends the v99 loader with three new per-batch fields:
  var_depth       : (B, N_MAX) int32 — DAG depth of each variable (0 = observed leaf,
                     depth+1 = result of a factor whose deepest arg has depth)
  staging_mask_t  : (B, K_MAX, T_MAX, T_MAX) float32 — topological staging mask
                     per breath. staging_mask_t[b, k, i, j] = bipartite[b, i, j]
                     restricted to positions visible at breath k.
  head_op_mask_t  : (B, N_HEADS, T_MAX, T_MAX) float32 — per-head op-type mask.
                     heads 0-3 = ADD, 4-7 = SUB, 8-11 = MUL, 12-15 = DIV.

Both masks are built in NumPy outside JIT (per v99 pattern) and passed as realized
Tensors — stable shapes, trivial memory cost.

Visibility rule for breath k (0-indexed):
  - Observed leaves (depth 0) are always visible.
  - A variable node at depth d is visible when k >= d - 1 (i.e., depth-d nodes
    open up on breath d-1, so breath 0 sees only depth-0 and depth-1 results).
  - A factor node is visible if its RESULT variable is visible at breath k.
  - Concretely: visible if result_depth <= k + 1 (observed always has depth 0,
    so result_depth=1 → visible at k=0).

This implements the "earn visibility by waiting for predecessor breaths" discipline
described in the v100 spec.
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

# Import the bipartite mask builder from v99 for reuse
from mycelium.factor_graph import build_factor_graph_masks_np

# Env defaults (can be overridden by V100_* env vars in launchers)
V100_N_MAX   = int(os.environ.get("V100_N_MAX",   "16"))
V100_F_MAX   = int(os.environ.get("V100_F_MAX",   "8"))
V100_K_MAX   = int(os.environ.get("V100_K_MAX",   "10"))
V100_T_MAX   = V100_N_MAX + V100_F_MAX
V100_N_HEADS = 16   # fixed: Pythia-410M has 16 attention heads


# ---------------------------------------------------------------------------
# DAG depth computation
# ---------------------------------------------------------------------------

def compute_var_depth(
    obs_mask_rec: list[int],
    factor_args: list[list[int]],
    n_total_vars: int,
) -> dict[int, int]:
    """Compute DAG depth for each variable index in a single problem record.

    depth[i] = 0  for observed leaf variables (obs_mask[i] == 1)
    depth[i] = max(depth[arg1], depth[arg2]) + 1  for result variables

    Returns a dict {var_idx: depth}. Variables unreachable from leaves
    (shouldn't happen in well-formed data) get depth = max_possible + 1.
    """
    n_total = n_total_vars
    depth: dict[int, int] = {}

    # Observed leaves: depth 0
    for i in range(n_total):
        if i < len(obs_mask_rec) and obs_mask_rec[i] == 1:
            depth[i] = 0

    # Iterative fixed-point (topological order)
    changed = True
    while changed:
        changed = False
        for fa in factor_args:
            if len(fa) < 3:
                continue
            a1, a2, res = int(fa[0]), int(fa[1]), int(fa[2])
            if a1 in depth and a2 in depth and res not in depth:
                depth[res] = max(depth[a1], depth[a2]) + 1
                changed = True

    # Anything still unresolved (e.g., padding or disconnected) → -1
    for i in range(n_total):
        if i not in depth:
            depth[i] = -1

    return depth


# ---------------------------------------------------------------------------
# Per-problem mask construction
# ---------------------------------------------------------------------------

def build_staging_and_head_masks_np(
    factor_types_np: np.ndarray,   # (F_MAX,) int, -1=padding
    factor_args_np: np.ndarray,    # (F_MAX, 3) int
    var_depth_np: np.ndarray,      # (N_MAX,) int  (-1 = padding)
    n_vars: int,
    n_factors: int,
    n_max: int = V100_N_MAX,
    f_max: int = V100_F_MAX,
    k_max: int = V100_K_MAX,
    n_heads: int = V100_N_HEADS,
) -> tuple[np.ndarray, np.ndarray]:
    """Build staging mask (K_MAX, T, T) and head-op mask (N_HEADS, T, T) for ONE problem.

    Returns:
      staging  : (k_max, t_max, t_max) float32  0=allow, -1e4=blocked
      head_ops : (n_heads, t_max, t_max) float32 — 0=allow, -1e4=blocked
                 heads 0-3 → ADD, 4-7 → SUB, 8-11 → MUL, 12-15 → DIV
    """
    t_max = n_max + f_max
    # Bipartite adjacency (shared across heads and breaths): (T, T) float
    # Build single-problem bipartite mask via v99 helper (wraps in batch dim 1)
    bipartite_batch = build_factor_graph_masks_np(
        factor_types_np[np.newaxis],   # (1, F_MAX)
        factor_args_np[np.newaxis],    # (1, F_MAX, 3)
        np.array([n_vars], dtype=np.int32),
        np.array([n_factors], dtype=np.int32),
        n_max=n_max,
        f_max=f_max,
    )
    bipartite = bipartite_batch[0]  # (T, T) — 0.0=allow, -1e4=blocked

    # --- Staging masks --------------------------------------------------
    # Compute the depth of each RESULT variable so we know when its factor
    # becomes visible.  Factor fi is visible at breath k iff its result
    # variable has depth <= k + 1.
    #  - depth 1 result → visible at breath 0 (k >= 0)
    #  - depth 2 result → visible at breath 1 (k >= 1)
    #  - depth d result → visible at breath d-1 (k >= d-1)

    # Build factor result depths
    factor_result_depth = np.full(f_max, -1, dtype=np.int32)
    for fi in range(n_factors):
        ft = int(factor_types_np[fi])
        if ft < 0:
            continue
        res_idx = int(factor_args_np[fi, 2])
        if 0 <= res_idx < n_max and var_depth_np[res_idx] >= 0:
            factor_result_depth[fi] = int(var_depth_np[res_idx])

    staging = np.full((k_max, t_max, t_max), -1e4, dtype=np.float32)

    # Vectorize staging mask construction over k: compute visibility in numpy
    # var_depth_np[0:n_vars] holds depths; vars beyond n_vars are padding → depth -1
    k_range = np.arange(k_max, dtype=np.int32)  # (k_max,)

    # Variable visibility: (n_max, k_max) — True if var vi visible at breath k
    # depth==0 → always visible; depth d>0 → visible when k >= d-1
    vd = var_depth_np[:n_max].reshape(n_max, 1)  # (n_max, 1)
    var_visible_mat = ((vd == 0) | ((vd > 0) & (vd <= k_range[np.newaxis, :] + 1)))
    # padding positions: depth -1 → never visible
    var_visible_mat[var_depth_np[:n_max] < 0, :] = False

    # Factor visibility: (f_max, k_max)
    frd = factor_result_depth[:n_factors].reshape(n_factors, 1)   # (n_factors, 1)
    factor_valid_mask = (factor_types_np[:n_factors] >= 0).reshape(n_factors, 1)
    fac_visible_mat_real = ((frd == 1) | ((frd > 0) & (frd <= k_range[np.newaxis, :] + 1))) & factor_valid_mask
    fac_visible_mat = np.zeros((f_max, k_max), dtype=bool)
    fac_visible_mat[:n_factors, :] = fac_visible_mat_real

    # Combine into pos_visible (t_max, k_max)
    pos_visible_mat = np.zeros((t_max, k_max), dtype=bool)
    pos_visible_mat[:n_max, :] = var_visible_mat
    pos_visible_mat[n_max:n_max + f_max, :] = fac_visible_mat

    # For each k: staging[k] = bipartite where both row and col are visible, else -1e4
    # pos_visible_k: (t_max,) boolean → outer product gives (t_max, t_max) both-visible
    for k in range(k_max):
        pv = pos_visible_mat[:, k]  # (t_max,)
        both_visible = pv[:, np.newaxis] & pv[np.newaxis, :]  # (t_max, t_max)
        staging[k] = np.where(both_visible, bipartite, -1e4)

    # --- Head-op masks --------------------------------------------------
    # 4 groups of 4 heads: ADD (0-3), SUB (4-7), MUL (8-11), DIV (12-15)
    # Head group h_grp (0=ADD,1=SUB,2=MUL,3=DIV) attends ONLY along edges
    # of its op type.  Self-attention is always allowed.

    # Build per-op bipartite adjacency: (4, T, T)
    op_adj = np.full((4, t_max, t_max), -1e4, dtype=np.float32)

    # Diagonal (self-attention) always allowed for all ops
    eye_idx = np.arange(t_max)
    op_adj[:, eye_idx, eye_idx] = 0.0

    # Fill variable↔factor edges per op (only over real factors)
    for fi in range(n_factors):
        ft = int(factor_types_np[fi])
        if ft < 0 or ft >= 4:
            continue
        fpos = n_max + fi
        for vi in factor_args_np[fi]:
            vi = int(vi)
            if 0 <= vi < n_vars:
                op_adj[ft, vi, fpos] = 0.0
                op_adj[ft, fpos, vi] = 0.0

    # Assign to heads: 4 heads per op — use fancy indexing (no Python loop over heads)
    heads_per_op = n_heads // 4  # = 4
    # head_ops[op*4 : op*4+4] = op_adj[op] for op in 0..3
    # Reshape op_adj from (4, T, T) → (4, 1, T, T), repeat → (4, 4, T, T), flatten → (16, T, T)
    head_ops = np.repeat(op_adj[:, np.newaxis, :, :], heads_per_op, axis=1).reshape(n_heads, t_max, t_max)

    return staging, head_ops


# ---------------------------------------------------------------------------
# Batch construction
# ---------------------------------------------------------------------------

def _records_to_batch_v100(
    picks: list[dict],
    n_max: int = V100_N_MAX,
    f_max: int = V100_F_MAX,
    k_max: int = V100_K_MAX,
    n_heads: int = V100_N_HEADS,
) -> dict:
    """Convert a list of records to numpy arrays for one v100 batch.

    Returns dict with all v99 fields PLUS:
      var_depth     : (B, N_MAX) int32
      staging_mask  : (B, K_MAX, T_MAX, T_MAX) float32
      head_op_mask  : (B, N_HEADS, T_MAX, T_MAX) float32
    """
    B = len(picks)
    t_max = n_max + f_max

    domain_init      = np.zeros((B, n_max, 100), dtype=np.float32)
    node_kinds       = np.full((B, t_max), -1, dtype=np.int32)
    gold_np          = np.zeros((B, n_max), dtype=np.int32)
    obs_mask         = np.zeros((B, n_max), dtype=np.int32)
    factor_types_np  = np.full((B, f_max), -1, dtype=np.int32)
    factor_args_np   = np.full((B, f_max, 3), -1, dtype=np.int32)
    query_idx_np     = np.zeros((B,), dtype=np.int32)
    n_vars_total_np  = np.zeros((B,), dtype=np.int32)
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

        n_vars_total_np[b] = min(n_total, n_max)
        n_factors_np[b]    = min(nf, f_max)
        query_idx_np[b]    = int(rec["query_idx"])

        # Fill variable positions
        for vi in range(min(n_total, n_max)):
            gv = int(gold_vals[vi])
            gold_np[b, vi] = gv

            if obs_mask_rec[vi] == 1:
                ov = int(obs_vals[vi]) if obs_vals[vi] is not None else gv
                domain_init[b, vi, ov] = 1.0
                obs_mask[b, vi] = 1
                node_kinds[b, vi] = 0
            else:
                domain_init[b, vi, :] = 1.0 / 100.0
                obs_mask[b, vi] = 0
                node_kinds[b, vi] = 1

        # Fill factor type/args arrays
        for fi in range(min(nf, f_max)):
            ft_str = ft_list[fi]
            ft_int = OP_MAP.get(ft_str, -1)
            fa     = fa_list[fi]
            factor_types_np[b, fi] = ft_int
            for j in range(3):
                factor_args_np[b, fi, j] = int(fa[j])
            fpos = n_max + fi
            node_kinds[b, fpos] = 2

        # Compute DAG depths
        depth_dict = compute_var_depth(
            obs_mask_rec,
            [fa_list[fi] for fi in range(min(nf, f_max))],
            n_total,
        )
        for vi in range(min(n_total, n_max)):
            var_depth_np[b, vi] = depth_dict.get(vi, -1)

        # Build staging mask and head-op mask for this problem
        staging_b, head_op_b = build_staging_and_head_masks_np(
            factor_types_np[b],
            factor_args_np[b],
            var_depth_np[b],
            n_vars=int(n_vars_total_np[b]),
            n_factors=int(n_factors_np[b]),
            n_max=n_max,
            f_max=f_max,
            k_max=k_max,
            n_heads=n_heads,
        )
        staging_masks[b] = staging_b
        head_op_masks[b] = head_op_b

    return {
        "domain_init":   domain_init,
        "node_kinds":    node_kinds,
        "gold_values":   gold_np,
        "observed_mask": obs_mask,
        "factor_types":  factor_types_np,
        "factor_args":   factor_args_np,
        "query_idx":     query_idx_np,
        "n_vars_total":  n_vars_total_np,
        "n_factors":     n_factors_np,
        "var_depth":     var_depth_np,
        "staging_mask":  staging_masks,
        "head_op_mask":  head_op_masks,
    }


def batch_to_tensors_v100(batch_np: dict) -> dict:
    """Convert v100 numpy batch dict to tinygrad Tensors (realized)."""
    return {
        "domain_init":   Tensor(batch_np["domain_init"],   dtype=dtypes.float).contiguous().realize(),
        "node_kinds":    Tensor(batch_np["node_kinds"],    dtype=dtypes.int).contiguous().realize(),
        "gold_values":   Tensor(batch_np["gold_values"],   dtype=dtypes.int).contiguous().realize(),
        "observed_mask": Tensor(batch_np["observed_mask"], dtype=dtypes.int).contiguous().realize(),
        "factor_types":  Tensor(batch_np["factor_types"],  dtype=dtypes.int).contiguous().realize(),
        "factor_args":   Tensor(batch_np["factor_args"],   dtype=dtypes.int).contiguous().realize(),
        "var_depth":     Tensor(batch_np["var_depth"],     dtype=dtypes.int).contiguous().realize(),
        "staging_mask":  Tensor(batch_np["staging_mask"],  dtype=dtypes.float).contiguous().realize(),
        "head_op_mask":  Tensor(batch_np["head_op_mask"],  dtype=dtypes.float).contiguous().realize(),
        # numpy-only fields (used for Python-side indexing / diagnostics)
        "query_idx":    batch_np["query_idx"],
        "n_vars_total": batch_np["n_vars_total"],
        "n_factors":    batch_np["n_factors"],
    }


class FactorGraphLoaderV100:
    """Iterates factor-graph records in v100 format (staging + head-op masks).

    Supports:
      - difficulty_filter: only sample from one band (smoke mode)
      - curriculum: start easy-only, anneal to uniform over anneal_steps
    """

    def __init__(
        self,
        path: str,
        batch_size: int = 8,
        difficulty_filter: str | None = None,
        curriculum: bool = False,
        curriculum_anneal_steps: int = 1000,
        n_max: int = V100_N_MAX,
        f_max: int = V100_F_MAX,
        k_max: int = V100_K_MAX,
        n_heads: int = V100_N_HEADS,
        seed: int = 0,
    ):
        self.path = path
        self.batch_size = batch_size
        self.difficulty_filter = difficulty_filter
        self.curriculum = curriculum
        self.curriculum_anneal_steps = curriculum_anneal_steps
        self.n_max = n_max
        self.f_max = f_max
        self.k_max = k_max
        self.n_heads = n_heads
        self.rng = random.Random(seed)

        records = load_jsonl(path)
        if difficulty_filter is not None:
            records = [r for r in records if r.get("difficulty") == difficulty_filter]
            assert records, f"No records match difficulty={difficulty_filter} in {path}"

        self.records = records
        self.by_diff = index_by_difficulty(records)
        self.active_diffs = [d for d in DIFFICULTIES if len(self.by_diff[d]) > 0]
        print(
            f"[fg_data_v100] loaded {len(records)} from {os.path.basename(path)}; "
            f"by difficulty: {[(d, len(self.by_diff[d])) for d in self.active_diffs]}",
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
        """Return a batch dict with realized Tensors."""
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

        batch_np = _records_to_batch_v100(picks, self.n_max, self.f_max, self.k_max, self.n_heads)
        batch_t = batch_to_tensors_v100(batch_np)
        batch_t["picks"] = picks
        return batch_t

    def iter_eval(self, batch_size: int | None = None) -> Iterator[dict]:
        """Iterate through ALL records in order in fixed-size batches.
        Pads the last batch with repeats so all batches are full."""
        bs = batch_size or self.batch_size
        n = len(self.records)
        for start in range(0, n, bs):
            batch_recs = self.records[start : start + bs]
            while len(batch_recs) < bs:
                batch_recs.append(self.records[0])
            batch_np = _records_to_batch_v100(batch_recs, self.n_max, self.f_max, self.k_max, self.n_heads)
            batch_t = batch_to_tensors_v100(batch_np)
            batch_t["picks"] = batch_recs
            yield batch_t

    def __len__(self) -> int:
        return len(self.records)
