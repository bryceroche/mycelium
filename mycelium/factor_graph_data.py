"""v99 factor graph data loader.

Streams JSONL records from the factor graph dataset (built by
scripts/build_factor_graph_data.py). Supports curriculum sampling and
fixed-shape tensor batching for JIT-stable training.

Each record schema:
  n_vars      : int — number of LEAF variables (observed)
  n_factors   : int — number of factors (= number of result vars)
  factor_types: list[str] — "add"/"sub"/"mul"/"div"
  factor_args : list[[arg1_idx, arg2_idx, result_idx]] — into gold_values indexing
  observed_mask : list[int] 0/1 — 1=observed leaf, 0=unobserved result
  observed_values : list[int|None] — value if observed, else None
  gold_values  : list[int] — full solution (len = n_vars + n_factors total)
  query_idx    : int — the target variable to predict
  difficulty   : "easy"/"medium"/"hard"

Output tensors per batch:
  domain_init   : (B, N_MAX, 100) float32 — one-hot at observed value; uniform (1/100) at unobserved
  node_kinds    : (B, T_MAX) int32         — 0=observed, 1=unobserved, 2=factor, -1=padding
  gold_values_t : (B, N_MAX) int32         — values 0..99 at all var positions
  observed_mask_t: (B, N_MAX) int32        — 1=observed, 0=unobserved (padding positions = 0)
  factor_types_t : (B, F_MAX) int32        — 0=add,1=sub,2=mul,3=div; -1=padding
  factor_args_t  : (B, F_MAX, 3) int32     — [arg1_idx, arg2_idx, result_idx]; -1 for padding
  query_idx_t    : (B,) int32              — queried variable index
  n_vars_total_t : (B,) int32              — total vars (leaves + results) = len(gold_values)
  n_factors_t    : (B,) int32             — number of actual factors
  attn_bias_t    : (B, T_MAX, T_MAX) float32 — precomputed adjacency mask
  kv_bias_t      : (B, T_MAX, hidden) float32 — op-type additive K/V bias (built lazily)

The attn_bias and kv_bias tensors are precomputed in NumPy and then wrapped
as Tensors — they're re-computed per batch (dynamic per-problem).
"""
import json
import os
import random
from typing import Iterator

import numpy as np
from tinygrad import Tensor, dtypes


DIFFICULTIES = ["easy", "medium", "hard"]
OP_MAP = {"add": 0, "sub": 1, "mul": 2, "div": 3}


def load_jsonl(path: str) -> list[dict]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def index_by_difficulty(records: list[dict]) -> dict[str, list[dict]]:
    idx: dict[str, list[dict]] = {d: [] for d in DIFFICULTIES}
    for r in records:
        d = r.get("difficulty", "easy")
        if d not in idx:
            idx[d] = []
        idx[d].append(r)
    return idx


def _records_to_batch(
    picks: list[dict],
    n_max: int,
    f_max: int,
) -> dict:
    """Convert a list of records to numpy arrays for one batch.

    n_max  = V99_N_MAX (max total variable nodes = leaves + results)
    f_max  = V99_F_MAX (max factor nodes)
    t_max  = n_max + f_max
    """
    B = len(picks)
    t_max = n_max + f_max

    domain_init = np.zeros((B, n_max, 100), dtype=np.float32)
    node_kinds = np.full((B, t_max), -1, dtype=np.int32)
    gold_np = np.zeros((B, n_max), dtype=np.int32)
    obs_mask = np.zeros((B, n_max), dtype=np.int32)
    factor_types_np = np.full((B, f_max), -1, dtype=np.int32)
    factor_args_np = np.full((B, f_max, 3), -1, dtype=np.int32)
    query_idx_np = np.zeros((B,), dtype=np.int32)
    n_vars_total_np = np.zeros((B,), dtype=np.int32)
    n_factors_np = np.zeros((B,), dtype=np.int32)

    for b, rec in enumerate(picks):
        gold_vals = rec["gold_values"]       # list[int], len = n_total_vars
        obs_mask_rec = rec["observed_mask"]  # list[int], len = n_total_vars
        obs_vals = rec["observed_values"]    # list[int|None], len = n_total_vars
        ft_list = rec["factor_types"]        # list[str]
        fa_list = rec["factor_args"]         # list[[a, b, r]]
        n_total = len(gold_vals)             # = n_leaves + n_factors
        nf = rec["n_factors"]

        n_vars_total_np[b] = min(n_total, n_max)
        n_factors_np[b] = min(nf, f_max)
        query_idx_np[b] = int(rec["query_idx"])

        # Fill variable positions (0 .. n_total-1, capped at n_max)
        for vi in range(min(n_total, n_max)):
            gv = int(gold_vals[vi])
            gold_np[b, vi] = gv

            if obs_mask_rec[vi] == 1:
                # Observed: one-hot at the observed value
                ov = int(obs_vals[vi]) if obs_vals[vi] is not None else gv
                domain_init[b, vi, ov] = 1.0
                obs_mask[b, vi] = 1
                node_kinds[b, vi] = 0  # observed
            else:
                # Unobserved: uniform distribution over [0, 99]
                domain_init[b, vi, :] = 1.0 / 100.0
                obs_mask[b, vi] = 0
                node_kinds[b, vi] = 1  # unobserved

        # Fill factor positions (n_max .. n_max + nf - 1)
        for fi in range(min(nf, f_max)):
            ft_str = ft_list[fi]
            ft_int = OP_MAP.get(ft_str, -1)
            fa = fa_list[fi]  # [arg1_idx, arg2_idx, result_idx]
            factor_types_np[b, fi] = ft_int
            for j in range(3):
                factor_args_np[b, fi, j] = int(fa[j])
            fpos = n_max + fi
            node_kinds[b, fpos] = 2  # factor node

    # Build attention bias: (B, T_MAX, T_MAX)
    from mycelium.factor_graph import build_factor_graph_masks_np
    attn_bias = build_factor_graph_masks_np(
        factor_types_np, factor_args_np, n_vars_total_np, n_factors_np,
        n_max=n_max, f_max=f_max,
    )

    return {
        "domain_init":    domain_init,
        "node_kinds":     node_kinds,
        "gold_values":    gold_np,
        "observed_mask":  obs_mask,
        "factor_types":   factor_types_np,
        "factor_args":    factor_args_np,
        "query_idx":      query_idx_np,
        "n_vars_total":   n_vars_total_np,
        "n_factors":      n_factors_np,
        "attn_bias":      attn_bias,
    }


def batch_to_tensors(batch_np: dict) -> dict:
    """Convert numpy batch dict to tinygrad Tensors (realized)."""
    return {
        "domain_init":   Tensor(batch_np["domain_init"],   dtype=dtypes.float).contiguous().realize(),
        "node_kinds":    Tensor(batch_np["node_kinds"],    dtype=dtypes.int).contiguous().realize(),
        "gold_values":   Tensor(batch_np["gold_values"],   dtype=dtypes.int).contiguous().realize(),
        "observed_mask": Tensor(batch_np["observed_mask"], dtype=dtypes.int).contiguous().realize(),
        "factor_types":  Tensor(batch_np["factor_types"],  dtype=dtypes.int).contiguous().realize(),
        "factor_args":   Tensor(batch_np["factor_args"],   dtype=dtypes.int).contiguous().realize(),
        "query_idx":     batch_np["query_idx"],    # kept as numpy (Python-only indexing)
        "n_vars_total":  batch_np["n_vars_total"],
        "n_factors":     batch_np["n_factors"],
        "attn_bias":     Tensor(batch_np["attn_bias"], dtype=dtypes.float).contiguous().realize(),
    }


class FactorGraphLoader:
    """Iterates factor-graph records in batches. Supports:
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
        n_max: int = 16,
        f_max: int = 8,
        seed: int = 0,
    ):
        self.path = path
        self.batch_size = batch_size
        self.difficulty_filter = difficulty_filter
        self.curriculum = curriculum
        self.curriculum_anneal_steps = curriculum_anneal_steps
        self.n_max = n_max
        self.f_max = f_max
        self.rng = random.Random(seed)

        records = load_jsonl(path)
        if difficulty_filter is not None:
            records = [r for r in records if r.get("difficulty") == difficulty_filter]
            assert records, f"No records match difficulty={difficulty_filter} in {path}"

        self.records = records
        self.by_diff = index_by_difficulty(records)
        self.active_diffs = [d for d in DIFFICULTIES if len(self.by_diff[d]) > 0]
        print(f"[fg_data] loaded {len(records)} from {os.path.basename(path)}; "
              f"by difficulty: {[(d, len(self.by_diff[d])) for d in self.active_diffs]}",
              flush=True)

    def _step_difficulty_weights(self, step: int) -> dict[str, float]:
        if not self.curriculum or "easy" not in self.active_diffs:
            return {d: 1.0 / len(self.active_diffs) for d in self.active_diffs}
        progress = min(1.0, step / max(1.0, float(self.curriculum_anneal_steps)))
        N = len(self.active_diffs)
        weights = {}
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

        batch_np = _records_to_batch(picks, self.n_max, self.f_max)
        batch_t = batch_to_tensors(batch_np)
        batch_t["picks"] = picks  # keep metadata for logging
        return batch_t

    def iter_eval(self, batch_size: int | None = None) -> Iterator[dict]:
        """Iterate through ALL records in order, in fixed-size batches.
        Pads the last batch with repeats so all batches are full."""
        bs = batch_size or self.batch_size
        n = len(self.records)
        for start in range(0, n, bs):
            batch_recs = self.records[start:start + bs]
            while len(batch_recs) < bs:
                batch_recs.append(self.records[0])
            batch_np = _records_to_batch(batch_recs, self.n_max, self.f_max)
            batch_t = batch_to_tensors(batch_np)
            batch_t["picks"] = batch_recs
            yield batch_t

    def __len__(self):
        return len(self.records)
