"""kenken_dual_data.py — DUAL-VIEW KenKen factor graph (primal cells + dual variables).

THE EXPERIMENT (Bryce's multi-view / channeling idea, framed as a GENERALITY
mechanism, not a KenKen trick): augment the KenKen factor graph with a DUAL
viewpoint and CHANNELING constraints, and ask whether the extra cross-view data
exchange helps the learned-BP deducer.

  PRIMAL view  (positions 0..48): "what number goes in cell (r,c)?"  — the usual
                KenKen cells, values 1..N, with row/col all-different + cage factors.
  DUAL view    (positions 49..97): "which COLUMN does value v occupy in row r?"
                dual variable D[v,r] (laid out at 49 + (v-1)*7 + r), VALUE = the
                column 1..N where value v sits in row r of the solution.
  CHANNELING   (one factor per row): connects row r's 7 cells <-> row r's 7 dual
                vars, the bipartite mask across which the engine must learn
                cell(r,c)=v  <=>  D[v,r]=c.

Factor types (n_factor_types=6): 0 primal-row, 1 primal-col, 2 cage,
  3 dual-value-alldiff (D[v,.] a permutation), 4 dual-row-alldiff (D[.,r] a
  permutation), 5 channeling. s_max = 2*49 = 98.

The PRIMAL half reproduces mycelium.kenken_data.encode_puzzle EXACTLY (so the
primal-cell solve is directly comparable to the h=1024 baseline 0.796 — that is
what lets us skip a separate control train). The cage verification inlet is built
at 49 by the oracle (untouched) and zero-padded to 98 in the trainer.

Mirrors kenken_data.py structure (encode -> Batch -> Loader).
"""
from __future__ import annotations

import os
import random
from typing import Iterator

import numpy as np
from tinygrad import Tensor, dtypes

from mycelium.kenken_data import (
    N_CELLS, N_MAX, encode_puzzle, load_jsonl,
)

S_DUAL = 2 * N_CELLS          # 98 = 49 primal cells + 49 dual variables
DUAL_OFFSET = N_CELLS         # dual variables start at position 49


def _dual_pos(v: int, r: int) -> int:
    """Position of dual variable D[v, r] (v in 1..N_MAX, r in 0..N_MAX-1).

    Fixed 7x7 layout (N_MAX spacing) so the index map is identical across N; for
    N<7 the v>N / r>=N slots are simply never marked valid (masked like padding cells).
    """
    return DUAL_OFFSET + (v - 1) * N_MAX + r


def encode_dual_puzzle(rec: dict, n_cages_max: int) -> dict:
    """One puzzle -> dual-view per-puzzle numpy arrays (98-wide; membership pre-built)."""
    p = encode_puzzle(rec, n_cages_max)          # primal 49-wide (exact baseline encoding)
    N = int(rec["N"])
    solution = rec["solution"]                   # NxN nested list (values 1..N)

    # ---- 98-wide per-cell arrays: primal in [0,49), dual in [49,98) ----
    input_cells = np.zeros((S_DUAL,), dtype=np.int32)
    gold = np.zeros((S_DUAL,), dtype=np.int32)
    cell_valid = np.zeros((S_DUAL,), dtype=np.float32)
    value_domain_mask = np.zeros((S_DUAL, N_MAX), dtype=np.float32)

    input_cells[:N_CELLS] = p["input_cells"]
    gold[:N_CELLS] = p["gold"]
    cell_valid[:N_CELLS] = p["cell_valid"]
    value_domain_mask[:N_CELLS] = p["value_domain_mask"]

    # ---- dual variables: D[v,r] = column (1..N) where value v sits in row r ----
    for r in range(N):
        row = solution[r]
        for v in range(1, N + 1):
            # column c (0-indexed) with solution[r][c] == v
            c0 = next(cc for cc in range(N) if int(row[cc]) == v)
            pos = _dual_pos(v, r)
            cell_valid[pos] = 1.0
            gold[pos] = c0 + 1                    # 1..N column index (codebook is 1-indexed)
            value_domain_mask[pos, :N] = 1.0      # legal "values" = columns 1..N
    # dual variables are never observed -> input_cells stays 0 there -> all supervised.

    # ---- factor membership (L, 98) + latent_type (L,) ----
    cell_cage_id = p["cell_cage_id"]              # (49,) -1 = pad cell
    cage_members: dict[int, list[int]] = {}
    for f in range(N_CELLS):
        ci = int(cell_cage_id[f])
        if ci >= 0:
            cage_members.setdefault(ci, []).append(f)

    factors: list[tuple[list[int], int]] = []
    # type 0: primal rows (all 7; padding cells masked by cell_valid downstream)
    for r in range(N_MAX):
        factors.append(([r * N_MAX + c for c in range(N_MAX)], 0))
    # type 1: primal cols
    for c in range(N_MAX):
        factors.append(([r * N_MAX + c for r in range(N_MAX)], 1))
    # type 2: cages (pad to n_cages_max with empty membership rows)
    for ci in range(n_cages_max):
        factors.append((cage_members.get(ci, []), 2))
    # type 3: dual VALUE all-different — D[v, 0..6] a permutation (v once per column)
    for v in range(1, N_MAX + 1):
        factors.append(([_dual_pos(v, r) for r in range(N_MAX)], 3))
    # type 4: dual ROW all-different — D[0..6, r] a permutation (row r a permutation)
    for r in range(N_MAX):
        factors.append(([_dual_pos(v, r) for v in range(1, N_MAX + 1)], 4))
    # type 5: CHANNELING — per row r: row r's cells <-> row r's dual vars (bipartite)
    for r in range(N_MAX):
        cells = [r * N_MAX + c for c in range(N_MAX)]
        duals = [_dual_pos(v, r) for v in range(1, N_MAX + 1)]
        factors.append((cells + duals, 5))

    L = len(factors)                              # 7+7+n_cages_max+7+7+7 = 35 + n_cages_max
    membership = np.zeros((L, S_DUAL), dtype=np.float32)
    latent_type = np.zeros((L,), dtype=np.int32)
    for li, (members, t) in enumerate(factors):
        for m in members:
            membership[li, m] = 1.0
        latent_type[li] = t

    return {
        "input_cells": input_cells,
        "gold": gold,
        "cell_valid": cell_valid,
        "value_domain_mask": value_domain_mask,
        "membership": membership,
        "latent_type": latent_type,
        # cage features for the verification inlet (built at 49, padded to 98 in trainer)
        "cage_op": p["cage_op"],
        "cage_target": p["cage_target"],
        "cage_size": p["cage_size"],
        "cell_cage_id": p["cell_cage_id"],        # (49,) primal cage ids for the inlet
        # metadata
        "deduction_depth": p["deduction_depth"],
        "N": N,
        "n_givens": p["n_givens"],
        "band": p["band"],
    }


class KenKenDualBatch:
    """Realized dual-view batch (FactorGraphBatch fields + cage features for the inlet)."""

    def __init__(self, d: dict):
        self.input_cells: Tensor = d["input_cells"]
        self.gold: Tensor = d["gold"]
        self.cell_valid: Tensor = d["cell_valid"]
        self.value_domain_mask: Tensor = d["value_domain_mask"]
        self.membership: Tensor = d["membership"]
        self.latent_type: Tensor = d["latent_type"]
        self.cage_op: Tensor = d["cage_op"]
        self.cage_target: Tensor = d["cage_target"]
        self.cage_size: Tensor = d["cage_size"]
        self.cell_cage_id: Tensor = d["cell_cage_id"]      # (B, 49) primal, for the inlet
        # python-side metadata
        self.deduction_depth: list[int] = d["deduction_depth"]
        self.N: list[int] = d["N"]
        self.n_givens: list[int] = d["n_givens"]
        self.band: list[str] = d["band"]


class KenKenDualLoader:
    """Iterates dual-view KenKen puzzles (mirrors KenKenLoader; membership pre-built)."""

    def __init__(self, path: str, batch_size: int = 8, seed: int = 0,
                 n_cages_max: int | None = None):
        self.path = path
        self.batch_size = batch_size
        self.rng = random.Random(seed)
        records = load_jsonl(path)
        assert records, f"No records loaded from {path}"
        self.records = records
        corpus_max = max(len(r["cages"]) for r in records)
        self.n_cages_max = int(n_cages_max) if n_cages_max is not None else corpus_max
        assert self.n_cages_max >= corpus_max, (
            f"n_cages_max={self.n_cages_max} < corpus max cages {corpus_max}")
        by_N: dict[int, int] = {}
        for r in records:
            by_N[r["N"]] = by_N.get(r["N"], 0) + 1
        print(f"[kenken_dual_data] loaded {len(records)} from {os.path.basename(path)}; "
              f"by N: {sorted(by_N.items())}; n_cages_max={self.n_cages_max}; "
              f"s_max={S_DUAL} L={35 + self.n_cages_max}", flush=True)

    def _stack(self, picks: list[dict]) -> KenKenDualBatch:
        encs = [encode_dual_puzzle(r, self.n_cages_max) for r in picks]

        def stack_int(key):
            return Tensor(np.stack([e[key] for e in encs]).astype(np.int32),
                          dtype=dtypes.int).contiguous().realize()

        def stack_f(key):
            return Tensor(np.stack([e[key] for e in encs]).astype(np.float32),
                          dtype=dtypes.float).contiguous().realize()

        d = {
            "input_cells": stack_int("input_cells"),
            "gold": stack_int("gold"),
            "cell_valid": stack_f("cell_valid"),
            "value_domain_mask": stack_f("value_domain_mask"),
            "membership": stack_f("membership"),
            "latent_type": stack_int("latent_type"),
            "cage_op": stack_int("cage_op"),
            "cage_target": stack_int("cage_target"),
            "cage_size": stack_int("cage_size"),
            "cell_cage_id": stack_int("cell_cage_id"),
            "deduction_depth": [e["deduction_depth"] for e in encs],
            "N": [e["N"] for e in encs],
            "n_givens": [e["n_givens"] for e in encs],
            "band": [e["band"] for e in encs],
        }
        return KenKenDualBatch(d)

    def sample_batch(self) -> KenKenDualBatch:
        picks = self.rng.choices(self.records, k=self.batch_size)
        return self._stack(picks)

    def iter_eval(self, batch_size: int | None = None) -> Iterator[KenKenDualBatch]:
        bs = batch_size or self.batch_size
        n = len(self.records)
        for start in range(0, n, bs):
            batch = list(self.records[start:start + bs])
            while len(batch) < bs:
                batch.append(self.records[0])
            yield self._stack(batch)

    def __len__(self):
        return len(self.records)
