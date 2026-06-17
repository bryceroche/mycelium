"""Perceiver-Poincaré data adapter (BRICK-1).

Wraps the validated KenKenLoader (mycelium/kenken_data.py — the curriculum batches;
cages = the factor graph) and adds the GENERIC constraint-membership tensors the
perceiver needs: one LATENT per CONSTRAINT (row / col / cage), derived from the
factor graph WITHOUT hardcoding the roles into the geometry.

THE KEY: build_constraint_membership turns the raw factor-graph structure (the
49-cell grid's rows + cols, plus the per-instance cages from batch.cell_cage_id)
into a UNIFORM (L, 49) membership table — a constraint is just a SET of cells. The
perceiver module never branches on "row"/"col"/"cage" for its routing geometry;
it reads this membership table generically. The only role information that leaks
out is latent_type (row/col/cage/global), used ONLY for (a) the learned per-type
latent hidden init and (b) the per_constraint cell-field selection in the Tier-2
fallback path — NEVER for the anchor / routing geometry.

Latent layout per puzzle (padded to L_max for a static JIT graph):
  rows  : N_MAX row constraints   (cells of row r = the N valid cells in that row)
  cols  : N_MAX col constraints
  cages : up to n_cages_max cage constraints (from cell_cage_id)
  global: PERCEIVER_N_GLOBAL latents (no membership; anchored at the origin)
A constraint with NO valid cells (e.g. a row entirely in the padding region for an
N<7 board, or a padding cage) is marked latent_valid=0 and excluded everywhere.
"""
from __future__ import annotations

import numpy as np
from tinygrad import Tensor, dtypes

from mycelium.kenken_data import KenKenLoader, N_MAX, N_CELLS, KenKenBatch
from mycelium.perceiver_poincare import PERCEIVER_N_GLOBAL


# latent type ids (NOT used for routing geometry — see module docstring).
LTYPE_ROW = 0
LTYPE_COL = 1
LTYPE_CAGE = 2
LTYPE_GLOBAL = 3


def latent_capacity(n_cages_max: int, n_global: int = PERCEIVER_N_GLOBAL) -> int:
    """L_max = N_MAX rows + N_MAX cols + n_cages_max cages + n_global globals."""
    return 2 * N_MAX + int(n_cages_max) + int(n_global)


def build_constraint_membership(cell_cage_id_np: np.ndarray, cell_valid_np: np.ndarray,
                                n_cages_max: int, n_global: int = PERCEIVER_N_GLOBAL):
    """Build the generic constraint-membership tensors from the factor graph.

    cell_cage_id_np: (B, 49) int — cell's cage id (-1 = padding cell). The cages
                     ARE the per-instance factor graph (from batch.cell_cage_id).
    cell_valid_np:   (B, 49) float — 1.0 valid cell / 0.0 padding.

    Returns numpy arrays:
      membership (B, L, 49) f — 1.0 if cell j is in latent l's constraint cell-set.
      latent_valid (B, L) f   — 1.0 if the latent's constraint has >=1 valid cell
                                (globals are always valid).
      latent_type (B, L) int  — LTYPE_{ROW,COL,CAGE,GLOBAL} (NOT routing geometry).
      cell_relation_id (B, 49, 3) int — per cell, its [row_group, col_group, cage_id]
                                indices (only read by the per_constraint cell field;
                                row/col group = the cell's row/col index 0..6).

    GENERIC by construction: a constraint = a set of cells. Rows/cols come from the
    grid geometry (cell i is in row i//7 and col i%7); cages come from cell_cage_id.
    The membership table is uniform — the perceiver reads it without knowing which
    relation produced each latent.
    """
    Bn = int(cell_cage_id_np.shape[0])
    L = latent_capacity(n_cages_max, n_global)
    membership = np.zeros((Bn, L, N_CELLS), dtype=np.float32)
    latent_valid = np.zeros((Bn, L), dtype=np.float32)
    latent_type = np.full((Bn, L), LTYPE_GLOBAL, dtype=np.int32)
    cell_relation_id = np.zeros((Bn, N_CELLS, 3), dtype=np.int32)

    rows_idx = np.array([i // N_MAX for i in range(N_CELLS)], dtype=np.int64)
    cols_idx = np.array([i % N_MAX for i in range(N_CELLS)], dtype=np.int64)

    for b in range(Bn):
        valid = cell_valid_np[b] > 0.5                                # (49,)
        # per-cell relation ids (row group, col group, cage id) — for the per_constraint
        # cell field selection; row/col group = the cell's row/col index.
        cell_relation_id[b, :, 0] = rows_idx
        cell_relation_id[b, :, 1] = cols_idx
        cid = cell_cage_id_np[b].copy()
        cid[cid < 0] = 0
        cell_relation_id[b, :, 2] = cid

        li = 0
        # ROW constraints: cells in row r (valid only).
        for r in range(N_MAX):
            cells = np.where((rows_idx == r) & valid)[0]
            if len(cells) > 0:
                membership[b, li, cells] = 1.0
                latent_valid[b, li] = 1.0
            latent_type[b, li] = LTYPE_ROW
            li += 1
        # COL constraints.
        for c in range(N_MAX):
            cells = np.where((cols_idx == c) & valid)[0]
            if len(cells) > 0:
                membership[b, li, cells] = 1.0
                latent_valid[b, li] = 1.0
            latent_type[b, li] = LTYPE_COL
            li += 1
        # CAGE constraints (the per-instance factor graph).
        for cg in range(n_cages_max):
            cells = np.where((cell_cage_id_np[b] == cg) & valid)[0]
            if len(cells) > 0:
                membership[b, li, cells] = 1.0
                latent_valid[b, li] = 1.0
            latent_type[b, li] = LTYPE_CAGE
            li += 1
        # GLOBAL latents: no membership; always valid (widest horizon).
        for _g in range(n_global):
            latent_valid[b, li] = 1.0
            latent_type[b, li] = LTYPE_GLOBAL
            li += 1
        assert li == L, f"latent fill {li} != L {L}"

    return membership, latent_valid, latent_type, cell_relation_id


class PerceiverBatch:
    """A KenKenBatch + the perceiver constraint-membership tensors."""

    def __init__(self, kk: KenKenBatch, n_cages_max: int,
                 n_global: int = PERCEIVER_N_GLOBAL):
        # passthrough the KenKen tensors the readout / loss / instrument need.
        self.input_cells = kk.input_cells
        self.gold = kk.gold
        self.cell_valid = kk.cell_valid
        self.value_domain_mask = kk.value_domain_mask
        self.cell_cage_id = kk.cell_cage_id
        # python-side metadata (eval-only).
        self.deduction_depth = kk.deduction_depth
        self.N = kk.N
        self.n_givens = kk.n_givens
        self.band = kk.band

        cid_np = kk.cell_cage_id.realize().numpy().astype(np.int32)
        cv_np = kk.cell_valid.realize().numpy().astype(np.float32)
        mem, lv, lt, crid = build_constraint_membership(
            cid_np, cv_np, n_cages_max, n_global)
        self.latent_membership = Tensor(mem, dtype=dtypes.float).contiguous().realize()
        self.latent_valid = Tensor(lv, dtype=dtypes.float).contiguous().realize()
        self.latent_type = Tensor(lt, dtype=dtypes.int).contiguous().realize()
        self.cell_relation_id = Tensor(crid, dtype=dtypes.int).contiguous().realize()


class PerceiverLoader:
    """Wraps a KenKenLoader, yielding PerceiverBatch (KenKen tensors + membership)."""

    def __init__(self, path: str, batch_size: int = 8, seed: int = 0,
                 n_cages_max: int | None = None,
                 n_global: int = PERCEIVER_N_GLOBAL):
        self.kk = KenKenLoader(path, batch_size=batch_size, seed=seed,
                               n_cages_max=n_cages_max)
        self.n_cages_max = self.kk.n_cages_max
        self.n_global = n_global
        self.batch_size = batch_size

    def sample_batch(self) -> PerceiverBatch:
        return PerceiverBatch(self.kk.sample_batch(), self.n_cages_max, self.n_global)

    def iter_eval(self, batch_size: int | None = None):
        for kk in self.kk.iter_eval(batch_size=batch_size):
            yield PerceiverBatch(kk, self.n_cages_max, self.n_global)

    def __len__(self):
        return len(self.kk)
