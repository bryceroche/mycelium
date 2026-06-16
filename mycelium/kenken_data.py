"""KenKen data loader for the KenKen scaffold (mirrors sudoku_data.py).

Streams the uniqueness-verified KenKen corpus emitted by
`scripts/build_kenken_data.py` (.cache/kenken_{train,test}.jsonl). Each row:

  {N, cages:[[ [r,c],... ],...], clues:[[op,target],...],
   solution:[[...]], n_givens, deduction_depth, unique:true}

The size-1 "given" cages are the OBSERVED cells (their target IS the value).

This loader's job is to flatten each variable-N puzzle onto the FIXED N_max=7
grid (49 cells) the model expects, and to emit — per batch — every tensor the
KenKen forward needs that is PER-INSTANCE:

  input_cells        (B, 49) int   — 0=unknown, 1..7 given digit (given cages only)
  gold_solution      (B, 49) int   — gold cell digits 1..7 (padding cells = 0)
  cell_valid         (B, 49) float — 1.0 if cell is inside the puzzle's NxN, else 0
  cage_mask          (B, 49, 49) f — symmetric cage-membership clique (+ self)
  cell_cage_id       (B, 49) int    — which cage each cell belongs to (-1 = pad)
  cage_op            (B, n_cages_max) int   — op-type id per cage (verification inlet)
  cage_target        (B, n_cages_max) int   — log-bucket id of the cage target
  cage_size          (B, n_cages_max) int   — cell count per cage (verification inlet)
  cage_cell_count_per_cell (B, 49) int      — size of the cage each cell is in
  value_domain_mask  (B, 49, N_max) float   — 1.0 for legal values 1..N, else 0
  deduction_depth    list[int]              — per-puzzle (Property-2 x-axis)

CRITICAL design pins (carried from the data-builder + survey memo):
  - The cage mask is SYMMETRIC MEMBERSHIP only (cell i ~ cell j iff same cage).
    op-type is NEVER a mask channel — it feeds ONLY the verification inlet.
  - Padding cells (outside NxN) attend to nothing but self; excluded everywhere.
  - value-domain: an N-board admits values 1..N; values N+1..N_max are masked
    in the readout.
"""
from __future__ import annotations

import json
import math
import os
import random
from typing import Iterator

import numpy as np
from tinygrad import Tensor, dtypes


N_MAX = 7                       # pad every board to 7x7
N_CELLS = N_MAX * N_MAX         # 49
OP_VOCAB = ["given", "add", "sub", "mul", "div"]   # index 0..4 (verification inlet)
OP_TO_ID = {op: i for i, op in enumerate(OP_VOCAB)}
N_OPS = len(OP_VOCAB)

# Log-magnitude bucketed target encoding. A mul of 3 cells (each up to 7) can
# reach 7*7*7 = 343, far larger than a single cell magnitude (≤7). We bucket the
# target value into log-spaced buckets over [1, ~1000] so the encoded target
# lives in a bounded, codebook-compatible representation space rather than as a
# raw scalar (the magnitude-mismatch that the substrate LN-fix addressed).
TARGET_BUCKETS = 32
TARGET_MAX = 1000.0


def target_to_bucket(t: int) -> int:
    """Map an integer cage target to a log-spaced bucket id in [0, TARGET_BUCKETS-1]."""
    t = max(1, int(t))
    # log scale: bucket = floor( (log t / log TARGET_MAX) * (B-1) ), clamped.
    frac = math.log(float(t)) / math.log(TARGET_MAX)
    b = int(frac * (TARGET_BUCKETS - 1))
    return max(0, min(TARGET_BUCKETS - 1, b))


def load_jsonl(path: str) -> list[dict]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _rc_to_flat(r: int, c: int) -> int:
    """(row, col) on the N_max grid → flat cell index in [0, 49)."""
    return r * N_MAX + c


def encode_puzzle(rec: dict, n_cages_max: int) -> dict:
    """Flatten ONE puzzle record onto the fixed N_max grid → per-puzzle numpy arrays.

    Returns a dict of numpy arrays (no batch axis) — the loader stacks them.
    """
    N = int(rec["N"])
    cages = rec["cages"]          # list of list of [r,c]
    clues = rec["clues"]          # list of [op, target]
    solution = rec["solution"]    # NxN nested list

    input_cells = np.zeros((N_CELLS,), dtype=np.int32)
    gold = np.zeros((N_CELLS,), dtype=np.int32)
    cell_valid = np.zeros((N_CELLS,), dtype=np.float32)
    cage_mask = np.zeros((N_CELLS, N_CELLS), dtype=np.float32)
    cell_cage_id = np.full((N_CELLS,), -1, dtype=np.int32)
    cage_cell_count_per_cell = np.zeros((N_CELLS,), dtype=np.int32)
    value_domain_mask = np.zeros((N_CELLS, N_MAX), dtype=np.float32)

    # Per-cage arrays (verification inlet). Pad up to n_cages_max with zeros; a
    # cage_size of 0 marks a padding (non-existent) cage so the forward can mask it.
    cage_op = np.zeros((n_cages_max,), dtype=np.int32)       # default op id 0 (given) — masked by size 0
    cage_target = np.zeros((n_cages_max,), dtype=np.int32)   # bucket id
    cage_size = np.zeros((n_cages_max,), dtype=np.int32)     # 0 = padding cage

    # Mark valid cells + gold + value-domain.
    for r in range(N):
        for c in range(N):
            f = _rc_to_flat(r, c)
            cell_valid[f] = 1.0
            gold[f] = int(solution[r][c])
            value_domain_mask[f, :N] = 1.0     # values 1..N legal (cols 0..N-1 = digits 1..N)

    # Build cages: membership clique + per-cage features + given inputs.
    for ci, (cage, clue) in enumerate(zip(cages, clues)):
        op, target = clue[0], clue[1]
        members = [_rc_to_flat(int(r), int(c)) for (r, c) in cage]
        for f in members:
            cell_cage_id[f] = ci
            cage_cell_count_per_cell[f] = len(members)
        # symmetric membership clique (+ self via the i==j diagonal entries)
        for i in members:
            for j in members:
                cage_mask[i, j] = 1.0
        if ci < n_cages_max:
            cage_op[ci] = OP_TO_ID.get(op, OP_TO_ID["add"])
            cage_target[ci] = target_to_bucket(int(target))
            cage_size[ci] = len(members)
        # "given" cages are the observed cells: target IS the cell value.
        if op == "given":
            input_cells[members[0]] = int(target)

    return {
        "input_cells": input_cells,
        "gold": gold,
        "cell_valid": cell_valid,
        "cage_mask": cage_mask,
        "cell_cage_id": cell_cage_id,
        "cage_cell_count_per_cell": cage_cell_count_per_cell,
        "value_domain_mask": value_domain_mask,
        "cage_op": cage_op,
        "cage_target": cage_target,
        "cage_size": cage_size,
        "deduction_depth": int(rec.get("deduction_depth", 0)),
        "N": N,
        "n_givens": int(rec.get("n_givens", 0)),
        # Curriculum band by ACTUAL givens density (g40~easy/shallow .. g10~hard).
        # Backward-compat: old struct/instance corpora have no 'band' field; fall
        # back to a single 'all' group so existing runs are unaffected.
        "band": str(rec.get("band", "all")),
    }


class KenKenBatch:
    """A realized batch of KenKen tensors (all on the default device)."""

    def __init__(self, d: dict):
        self.input_cells: Tensor = d["input_cells"]
        self.gold: Tensor = d["gold"]
        self.cell_valid: Tensor = d["cell_valid"]
        self.cage_mask: Tensor = d["cage_mask"]
        self.cell_cage_id: Tensor = d["cell_cage_id"]
        self.cage_cell_count_per_cell: Tensor = d["cage_cell_count_per_cell"]
        self.value_domain_mask: Tensor = d["value_domain_mask"]
        self.cage_op: Tensor = d["cage_op"]
        self.cage_target: Tensor = d["cage_target"]
        self.cage_size: Tensor = d["cage_size"]
        # python-side metadata (NOT tensors)
        self.deduction_depth: list[int] = d["deduction_depth"]
        self.N: list[int] = d["N"]
        self.n_givens: list[int] = d["n_givens"]
        # Curriculum band per puzzle ('g40'|'g30'|'g20'|'g10', or 'all' for
        # band-less legacy corpora). Eval-only metadata (never read in the JIT
        # training body); drives per-band eval aggregation + Property-2 tagging.
        self.band: list[str] = d["band"]


class KenKenLoader:
    """Iterates KenKen puzzles in batches, flattening to the fixed N_max grid.

    n_cages_max is fixed across the loader (the max #cages over the whole corpus,
    rounded up) so the per-cage verification tensors have a stable shape — required
    for the JIT graph topology to be static (mirrors sudoku's fixed shapes).
    """

    def __init__(self, path: str, batch_size: int = 8, seed: int = 0,
                 n_cages_max: int | None = None):
        self.path = path
        self.batch_size = batch_size
        self.rng = random.Random(seed)
        records = load_jsonl(path)
        assert records, f"No records loaded from {path}"
        self.records = records
        # Fixed n_cages_max: max #cages in the corpus (so every per-cage tensor
        # has the same shape across batches).
        corpus_max = max(len(r["cages"]) for r in records)
        self.n_cages_max = int(n_cages_max) if n_cages_max is not None else corpus_max
        assert self.n_cages_max >= corpus_max, (
            f"n_cages_max={self.n_cages_max} < corpus max cages {corpus_max}"
        )
        by_N: dict[int, int] = {}
        for r in records:
            by_N[r["N"]] = by_N.get(r["N"], 0) + 1
        print(f"[kenken_data] loaded {len(records)} from {os.path.basename(path)}; "
              f"by N: {sorted(by_N.items())}; n_cages_max={self.n_cages_max}",
              flush=True)

    def _stack(self, picks: list[dict]) -> KenKenBatch:
        encs = [encode_puzzle(r, self.n_cages_max) for r in picks]
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
            "cage_mask": stack_f("cage_mask"),
            "cell_cage_id": stack_int("cell_cage_id"),
            "cage_cell_count_per_cell": stack_int("cage_cell_count_per_cell"),
            "value_domain_mask": stack_f("value_domain_mask"),
            "cage_op": stack_int("cage_op"),
            "cage_target": stack_int("cage_target"),
            "cage_size": stack_int("cage_size"),
            "deduction_depth": [e["deduction_depth"] for e in encs],
            "N": [e["N"] for e in encs],
            "n_givens": [e["n_givens"] for e in encs],
            "band": [e["band"] for e in encs],
        }
        return KenKenBatch(d)

    def sample_batch(self) -> KenKenBatch:
        picks = self.rng.choices(self.records, k=self.batch_size)
        return self._stack(picks)

    def iter_eval(self, batch_size: int | None = None) -> Iterator[KenKenBatch]:
        """Iterate all records in order, padding the last batch with repeats."""
        bs = batch_size or self.batch_size
        n = len(self.records)
        for start in range(0, n, bs):
            batch = list(self.records[start:start + bs])
            while len(batch) < bs:
                batch.append(self.records[0])
            yield self._stack(batch)

    def __len__(self):
        return len(self.records)
