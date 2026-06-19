"""graph_coloring_data.py — the graph-coloring GENERALITY testbed.

The SECOND factor graph for the general executor (mycelium/factor_graph_engine.py
+ mycelium/factor_masks.py), built to prove "one engine, any factor graph". KenKen
was the first; graph k-coloring is the second, structurally maximally different:
KenKen's factors are large symmetric cliques over a fixed 49-cell grid; coloring's
factors are tiny binary not-equal constraints over an arbitrary graph. If the SAME
engine solves both, the generality claim is real.

THE ENCODING (the key decisions)
--------------------------------
* VARIABLE NODES  = graph VERTICES.  Padded to s_max; cell_valid=1 for the n real
  vertices, 0 for the s_max-n padding slots.  A vertex's "value" is its color.
* FACTOR NODES    = EDGES.  Each edge (u,v) is ONE factor whose membership row has
  EXACTLY TWO 1s, at columns u and v.  membership shape (B, n_edges_max, s_max).
  The constraint is binary "not-equal" (a proper coloring gives adjacent vertices
  different colors).
* FACTOR TYPE     = ONE relation for all edges => n_factor_types T = 1.
  latent_type = 0 for every real edge factor; the GLOBAL sentinel handles the
  global head.  A padding edge row (all zeros) is marked latent_type = global
  sentinel so it contributes NOTHING to the type-0 adjacency.

  *** T=1 membership encodes the adjacency matrix EXACTLY. ***
  The mask builder (factor_masks.build_factor_attn_bias) computes, for type t,
  the per-type members m_t = membership * (latent_type == t) and then
        A_t[b,i,j] = (m_t^T @ m_t)[b,i,j] > 0.
  With T=1 and each edge-row carrying 1s at exactly (u,v):
        (m_0^T @ m_0)[i,j] = number of edges incident to BOTH i and j
                           = 1   iff (i,j) is an edge (or i==j on an endpoint)
                           = 0   otherwise.
  So A_0 > 0 is exactly the graph adjacency matrix (the diagonal is additionally
  forced for every valid vertex by the builder's self-edge fix). Thus the type-0
  heads implement "each vertex attends to its graph neighbours (+ itself)" with NO
  coloring-specific code in the engine. The single global head attends to all valid
  vertices. This is verified in the CPU smoke (see _smoke()).

* VALUE SPACE     N = k colors.  value_domain_mask = all k colors legal for every
  real vertex (coloring imposes no per-vertex domain restriction; the not-equal
  edges do all the constraining).  Padding vertices: all-zero domain.
* INPUT CELLS     = all-uncolored (state 0) — coloring is fully latent, no givens.
  (The engine's loss therefore supervises every valid vertex: supervise =
  cell_valid * (1 - (input_cells>0)) = cell_valid.)
* GOLD            = a proper k-coloring, stored as values 1..k (the engine does
  gold_idx = gold - 1, so color index c in 0..k-1 is stored as c+1). Padding = 0.

COLOR-PERMUTATION SYMMETRY (the load-bearing correctness pin)
-------------------------------------------------------------
Graph coloring is invariant to relabeling colors: if (c_0,...,c_{n-1}) is proper,
so is (pi(c_0),...,pi(c_{n-1})) for any permutation pi of the k colors. A fixed-gold
cross-entropy would punish the model for a CORRECT coloring that merely uses a
different (but equivalent) labeling. We break that symmetry with CANONICAL labeling
= COLOR-BY-FIRST-USE in a FIXED vertex order (vertex 0, 1, ..., n-1):
    - the first vertex (in index order) gets color 0;
    - each subsequent vertex keeps its color if it has appeared, else is assigned
      the next unused canonical color id (1, 2, ...).
This maps every member of a permutation-equivalence class to ONE representative, so
gold is deterministic and the fixed-gold CE is well-posed. (Stored +1, so canonical
color 0 -> gold value 1.) See _canonicalize_coloring().

GENERATOR + SOLVER
------------------
* Random graphs: G(n,p) (Erdos-Renyi) and random d-regular, n <= s_max, density
  controllable.  A difficulty curriculum by edge-density (edges/n) and an explicit
  "band" tag gives the convergence instrument an x-axis (analogous to KenKen's
  givens bands).
* Ground truth: a DSATUR-ordered backtracking k-coloring solver. Only graphs that
  ARE k-colorable are kept (the solver returns a proper coloring; non-colorable
  graphs at the chosen k are discarded).  We additionally record a coarse
  "deduction_depth" proxy (DSATUR backtrack count bucketed) for Property-2.

LOADER
------
GraphColoringLoader yields GraphColoringBatch objects that satisfy the engine's
FactorGraphBatch contract (mycelium.factor_graph_engine.FactorGraphBatch). Mirrors
the shape/dtype discipline of mycelium/kenken_data.py and the perceiver loader:
numpy intermediates -> Tensor(..., dtype=...).contiguous().realize(); fixed
s_max / n_edges_max for a static JIT graph topology; train/test split; batch_size;
seed.

SUBSTRATE
---------
GPU-free: numpy/python data generation; tinygrad Tensors only to pack the batch
(matching kenken_data._stack). No dtypes.float32 Tensor literal baked into a JIT
graph (data tensors are wrapped from numpy with explicit dtype, the engine's job).
"""
from __future__ import annotations

import os
import random
from typing import Iterator

import numpy as np
from tinygrad import Tensor, dtypes


# ---------------------------------------------------------------------------
# Difficulty bands (the convergence-instrument x-axis).
# Bands are ordered easy -> hard by edge density (edges/n). Mirrors KenKen's
# g40..g10 givens bands: a coarse difficulty label per instance.
# ---------------------------------------------------------------------------
BANDS = ["d10", "d15", "d20", "d25"]   # target edges/n ~ 1.0, 1.5, 2.0, 2.5


# ---------------------------------------------------------------------------
# Canonical labeling (color-permutation symmetry breaker)
# ---------------------------------------------------------------------------

def _canonicalize_coloring(coloring: list[int]) -> list[int]:
    """Color-by-first-use canonicalization in fixed vertex order.

    coloring[i] = some color id for vertex i (any labeling of a proper coloring).
    Returns a NEW list where colors are renamed by order of first appearance:
    the first vertex's color -> 0, the next new color -> 1, etc.

    This collapses every member of a color-permutation equivalence class to one
    deterministic representative, so a fixed-gold CE is well-posed (graph coloring
    is invariant to relabeling colors; see module docstring).
    """
    remap: dict[int, int] = {}
    out: list[int] = []
    nxt = 0
    for c in coloring:
        if c not in remap:
            remap[c] = nxt
            nxt += 1
        out.append(remap[c])
    return out


# ---------------------------------------------------------------------------
# Graph generation
# ---------------------------------------------------------------------------

def _gen_gnp(n: int, p: float, rng: random.Random) -> list[tuple[int, int]]:
    """Erdos-Renyi G(n,p): each undirected edge present independently w.p. p."""
    edges: list[tuple[int, int]] = []
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                edges.append((u, v))
    return edges


def _gen_regular(n: int, d: int, rng: random.Random,
                 max_tries: int = 200) -> list[tuple[int, int]] | None:
    """Random d-regular graph via the pairing (configuration) model with rejection.

    Returns an edge list (simple graph) or None if no simple graph was sampled
    within max_tries (caller falls back to G(n,p)). Requires n*d even and d < n.
    """
    if d <= 0 or d >= n or (n * d) % 2 != 0:
        return None
    for _ in range(max_tries):
        stubs: list[int] = []
        for v in range(n):
            stubs.extend([v] * d)
        rng.shuffle(stubs)
        ok = True
        seen: set[tuple[int, int]] = set()
        for i in range(0, len(stubs), 2):
            a, b = stubs[i], stubs[i + 1]
            if a == b:                       # self-loop -> reject
                ok = False
                break
            e = (a, b) if a < b else (b, a)
            if e in seen:                    # multi-edge -> reject
                ok = False
                break
            seen.add(e)
        if ok:
            return sorted(seen)
    return None


# ---------------------------------------------------------------------------
# DSATUR-ordered backtracking k-coloring solver (ground truth)
# ---------------------------------------------------------------------------

def _solve_k_coloring(n: int, adj: list[set[int]], k: int):
    """DSATUR-ordered backtracking k-coloring.

    Returns (coloring, backtracks) where coloring is a length-n list of color ids
    in 0..k-1 (NOT yet canonicalized), or (None, backtracks) if the graph is not
    k-colorable. `backtracks` is the number of failed-branch unwinds — a coarse
    search-difficulty proxy used to bucket deduction_depth.

    DSATUR ordering: at each step color the uncolored vertex with the highest
    saturation (number of distinct colors among its neighbours), ties broken by
    degree then index. This is the standard strong heuristic; it also makes the
    backtrack count a meaningful difficulty signal.
    """
    color = [-1] * n
    deg = [len(adj[v]) for v in range(n)]
    backtracks = [0]

    def _pick_vertex() -> int:
        best = -1
        best_key = None
        for v in range(n):
            if color[v] != -1:
                continue
            sat = len({color[w] for w in adj[v] if color[w] != -1})
            key = (sat, deg[v], -v)          # max sat, then max deg, then min index
            if best_key is None or key > best_key:
                best_key = key
                best = v
        return best

    def _bt(num_colored: int) -> bool:
        if num_colored == n:
            return True
        v = _pick_vertex()
        used = {color[w] for w in adj[v] if color[w] != -1}
        for c in range(k):
            if c in used:
                continue
            color[v] = c
            if _bt(num_colored + 1):
                return True
            color[v] = -1
            backtracks[0] += 1
        return False

    ok = _bt(0)
    if not ok:
        return None, backtracks[0]
    return list(color), backtracks[0]


def _depth_bucket(backtracks: int) -> int:
    """Coarse deduction-depth proxy from DSATUR backtrack count (Property-2 x-axis).

    0 backtracks -> depth 0 (greedy-solvable); then log-ish buckets.
    """
    if backtracks <= 0:
        return 0
    if backtracks <= 2:
        return 1
    if backtracks <= 8:
        return 2
    if backtracks <= 32:
        return 3
    return 4


# ---------------------------------------------------------------------------
# One instance
# ---------------------------------------------------------------------------

def generate_instance(rng: random.Random, s_max: int, k_colors: int,
                      band: str, regular_frac: float = 0.4,
                      max_resample: int = 50) -> dict | None:
    """Generate ONE k-colorable graph-coloring instance in the given band.

    band controls target edge density (edges/n). Returns a dict of python/numpy
    fields (no batch axis), or None if generation failed (caller resamples).

    Fields:
      n            : int                       — number of real vertices (<= s_max)
      edges        : list[(u,v)]               — undirected edge list (0-based)
      coloring     : list[int] (len n)         — CANONICAL proper k-coloring, 0..k-1
      band         : str
      deduction_depth : int                    — DSATUR backtrack bucket (0..4)
      n_edges      : int
    """
    # density target by band (edges/n)
    dens = {"d10": 1.0, "d15": 1.5, "d20": 2.0, "d25": 2.5}.get(band, 1.5)

    for _ in range(max_resample):
        # node count: 4..s_max (need >= a few nodes for a meaningful graph).
        n = rng.randint(max(4, min(8, s_max)), s_max)

        use_regular = (rng.random() < regular_frac)
        edges: list[tuple[int, int]] | None = None
        if use_regular:
            # degree ~ 2*density, clamped even & < n.
            d = int(round(2.0 * dens))
            d = max(1, min(d, n - 1))
            if (n * d) % 2 != 0:
                d -= 1
            if d >= 1:
                edges = _gen_regular(n, d, rng)
        if edges is None:
            # G(n,p) with p tuned so expected edges/n ~ dens  =>  p ~ 2*dens/(n-1).
            p = min(0.95, 2.0 * dens / max(1, n - 1))
            edges = _gen_gnp(n, p, rng)

        if len(edges) == 0:
            continue                          # empty graph is degenerate; resample

        # adjacency sets
        adj: list[set[int]] = [set() for _ in range(n)]
        for (u, v) in edges:
            adj[u].add(v)
            adj[v].add(u)

        coloring, backtracks = _solve_k_coloring(n, adj, k_colors)
        if coloring is None:
            continue                          # not k-colorable -> discard

        coloring = _canonicalize_coloring(coloring)
        return {
            "n": n,
            "edges": edges,
            "coloring": coloring,
            "band": band,
            "deduction_depth": _depth_bucket(backtracks),
            "n_edges": len(edges),
        }
    return None


def generate_corpus(n_instances: int, s_max: int, k_colors: int,
                    seed: int = 0, bands: list[str] | None = None,
                    regular_frac: float = 0.4) -> list[dict]:
    """Generate a balanced curriculum corpus across difficulty bands.

    Returns a list of instance dicts (see generate_instance). Bands are sampled
    round-robin so the corpus is balanced; each instance is k-colorable.
    """
    bands = bands or BANDS
    rng = random.Random(seed)
    out: list[dict] = []
    bi = 0
    attempts = 0
    max_attempts = n_instances * 200 + 1000
    while len(out) < n_instances and attempts < max_attempts:
        band = bands[bi % len(bands)]
        bi += 1
        attempts += 1
        inst = generate_instance(rng, s_max, k_colors, band,
                                 regular_frac=regular_frac)
        if inst is not None:
            out.append(inst)
    if len(out) < n_instances:
        raise RuntimeError(
            f"generate_corpus produced {len(out)}/{n_instances} after "
            f"{attempts} attempts (k={k_colors} too small for the densities?)")
    return out


# ---------------------------------------------------------------------------
# Encode one instance -> per-vertex / per-edge numpy arrays (no batch axis)
# ---------------------------------------------------------------------------

# latent_type values: 0 = the single edge relation; the GLOBAL sentinel marks
# padding edge rows so they drop out of the type-0 adjacency. The sentinel must
# equal n_factor_types (= 1) — factor_masks treats any latent_type not in 0..T-1
# as non-type (the global head uses an all-valid mask, not membership), and the
# engine's spec.n_factor_types is 1, so 1 is the global sentinel.
LTYPE_EDGE = 0
LTYPE_GLOBAL = 1                              # == n_factor_types (T) for coloring


def encode_instance(inst: dict, s_max: int, n_edges_max: int,
                    k_colors: int) -> dict:
    """Flatten ONE coloring instance onto fixed (s_max, n_edges_max) -> numpy arrays.

    Mirrors kenken_data.encode_puzzle: fixed shapes for a static JIT topology.
    """
    n = int(inst["n"])
    edges = inst["edges"]
    coloring = inst["coloring"]               # canonical, 0..k-1, length n
    assert n <= s_max, f"n={n} exceeds s_max={s_max}"
    assert len(edges) <= n_edges_max, \
        f"n_edges={len(edges)} exceeds n_edges_max={n_edges_max}"

    # --- per-vertex (variable node) arrays ---
    input_cells = np.zeros((s_max,), dtype=np.int32)            # all uncolored
    gold = np.zeros((s_max,), dtype=np.int32)                   # 0 on padding
    cell_valid = np.zeros((s_max,), dtype=np.float32)
    value_domain_mask = np.zeros((s_max, k_colors), dtype=np.float32)
    for v in range(n):
        cell_valid[v] = 1.0
        gold[v] = int(coloring[v]) + 1                          # store color c as c+1
        value_domain_mask[v, :] = 1.0                          # all k colors legal

    # --- per-edge (factor node) arrays ---
    # membership (n_edges_max, s_max): edge row e has exactly two 1s at (u,v).
    membership = np.zeros((n_edges_max, s_max), dtype=np.float32)
    latent_type = np.full((n_edges_max,), LTYPE_GLOBAL, dtype=np.int32)  # pad = global sentinel
    for e, (u, v) in enumerate(edges):
        membership[e, u] = 1.0
        membership[e, v] = 1.0
        latent_type[e] = LTYPE_EDGE                            # real edge -> type 0

    return {
        "input_cells": input_cells,
        "gold": gold,
        "cell_valid": cell_valid,
        "value_domain_mask": value_domain_mask,
        "membership": membership,
        "latent_type": latent_type,
        "deduction_depth": int(inst.get("deduction_depth", 0)),
        "n": n,
        "n_edges": int(inst.get("n_edges", len(edges))),
        "band": str(inst.get("band", "all")),
    }


# ---------------------------------------------------------------------------
# Batch (satisfies the engine's FactorGraphBatch contract)
# ---------------------------------------------------------------------------

class GraphColoringBatch:
    """A realized batch of graph-coloring tensors.

    Satisfies mycelium.factor_graph_engine.FactorGraphBatch (same tensor attrs +
    shapes/dtypes), so factor_breathing_forward / factor_loss / factor_accuracy
    consume it directly. No factor_inlet (coloring has no arithmetic to verify ->
    spec.has_factor_inlet=False).

    Tensor attributes
    -----------------
    input_cells       Tensor (B, s_max)         int   — all 0 (uncolored).
    cell_valid        Tensor (B, s_max)         float — 1 real vertex / 0 padding.
    value_domain_mask Tensor (B, s_max, k)      float — 1 for all k colors / 0 pad.
    gold              Tensor (B, s_max)         int   — color+1 (1..k) / 0 padding.
    membership        Tensor (B, n_edges_max, s_max) float — 2 ones per real edge.
    latent_type       Tensor (B, n_edges_max)   int   — 0 edge / 1 (=T) pad sentinel.

    Python-side metadata (NOT tensors; eval-only)
    ---------------------------------------------
    deduction_depth   list[int]  — DSATUR backtrack bucket (Property-2 x-axis).
    n                 list[int]  — real vertex count per instance.
    n_edges           list[int]  — edge count per instance.
    band              list[str]  — difficulty band.
    """

    def __init__(self, d: dict):
        self.input_cells: Tensor = d["input_cells"]
        self.cell_valid: Tensor = d["cell_valid"]
        self.value_domain_mask: Tensor = d["value_domain_mask"]
        self.gold: Tensor = d["gold"]
        self.membership: Tensor = d["membership"]
        self.latent_type: Tensor = d["latent_type"]
        # No verification inlet for coloring.
        self.factor_inlet = None
        # python-side metadata
        self.deduction_depth: list[int] = d["deduction_depth"]
        self.n: list[int] = d["n"]
        self.n_edges: list[int] = d["n_edges"]
        self.band: list[str] = d["band"]


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

class GraphColoringLoader:
    """Generates a coloring corpus once, splits train/test, yields batches.

    Fixed shapes across the loader (s_max vertices, n_edges_max edge rows) so the
    JIT graph topology is static — n_edges_max is the max edge count over the whole
    corpus (rounded up), exactly mirroring kenken_data's fixed n_cages_max.

    Parameters
    ----------
    n_instances  : total corpus size to generate (before the split).
    s_max        : fixed sequence length (== FactorGraphSpec.s_max). DEFAULT 49 —
                   the engine reuses kenken.py:kenken_layer_forward, which hard-
                   asserts S == 49 (N_CELLS). The data module is general on s_max
                   (the smoke runs at 12); production pads vertices to 49 exactly
                   as KenKen pads its 7x7 board to 49 cells. Use s_max=49 to drive
                   the real factor_breathing_forward.
    k_colors     : number of colors (== FactorGraphSpec.n_values, N).
    batch_size   : batch size.
    seed         : RNG seed (corpus + sampler).
    test_frac    : held-out fraction (by INSTANCE; coloring has no structural-leak
                   notion as sharp as KenKen's, but the split is disjoint).
    n_edges_max  : override the computed max-edge cap (must be >= corpus max).
    bands        : difficulty bands to sample.
    regular_frac : fraction of instances drawn as random d-regular (rest G(n,p)).
    """

    def __init__(self, n_instances: int = 4000, s_max: int = 49,
                 k_colors: int = 3, batch_size: int = 8, seed: int = 0,
                 test_frac: float = 0.15, n_edges_max: int | None = None,
                 bands: list[str] | None = None, regular_frac: float = 0.4):
        self.s_max = int(s_max)
        self.k_colors = int(k_colors)
        self.batch_size = int(batch_size)
        self.rng = random.Random(seed)

        corpus = generate_corpus(n_instances, s_max, k_colors, seed=seed,
                                 bands=bands, regular_frac=regular_frac)
        # shuffle then split by instance (disjoint train/test).
        split_rng = random.Random(seed + 1)
        split_rng.shuffle(corpus)
        n_test = max(1, int(round(test_frac * len(corpus))))
        self.test_records = corpus[:n_test]
        self.train_records = corpus[n_test:]
        assert self.train_records and self.test_records, \
            "split produced an empty train or test set; raise n_instances"

        corpus_max_edges = max(r["n_edges"] for r in corpus)
        self.n_edges_max = (int(n_edges_max) if n_edges_max is not None
                            else corpus_max_edges)
        assert self.n_edges_max >= corpus_max_edges, (
            f"n_edges_max={self.n_edges_max} < corpus max edges {corpus_max_edges}")

        by_band: dict[str, int] = {}
        for r in corpus:
            by_band[r["band"]] = by_band.get(r["band"], 0) + 1
        print(f"[graph_coloring_data] generated {len(corpus)} instances "
              f"(train {len(self.train_records)} / test {len(self.test_records)}); "
              f"k={self.k_colors} s_max={self.s_max} n_edges_max={self.n_edges_max}; "
              f"by band: {sorted(by_band.items())}", flush=True)

    # -- packing --------------------------------------------------------------

    def _stack(self, picks: list[dict]) -> GraphColoringBatch:
        encs = [encode_instance(r, self.s_max, self.n_edges_max, self.k_colors)
                for r in picks]

        def stack_int(key):
            return Tensor(np.stack([e[key] for e in encs]).astype(np.int32),
                          dtype=dtypes.int).contiguous().realize()

        def stack_f(key):
            return Tensor(np.stack([e[key] for e in encs]).astype(np.float32),
                          dtype=dtypes.float).contiguous().realize()

        d = {
            "input_cells":       stack_int("input_cells"),
            "cell_valid":        stack_f("cell_valid"),
            "value_domain_mask": stack_f("value_domain_mask"),
            "gold":              stack_int("gold"),
            "membership":        stack_f("membership"),
            "latent_type":       stack_int("latent_type"),
            "deduction_depth":   [e["deduction_depth"] for e in encs],
            "n":                 [e["n"] for e in encs],
            "n_edges":           [e["n_edges"] for e in encs],
            "band":              [e["band"] for e in encs],
        }
        return GraphColoringBatch(d)

    # -- sampling -------------------------------------------------------------

    def sample_batch(self) -> GraphColoringBatch:
        picks = self.rng.choices(self.train_records, k=self.batch_size)
        return self._stack(picks)

    def iter_eval(self, batch_size: int | None = None
                  ) -> Iterator[GraphColoringBatch]:
        """Iterate the TEST set in order, padding the last batch with repeats."""
        bs = batch_size or self.batch_size
        recs = self.test_records
        n = len(recs)
        for start in range(0, n, bs):
            batch = list(recs[start:start + bs])
            while len(batch) < bs:
                batch.append(recs[0])
            yield self._stack(batch)

    def iter_eval_train(self, batch_size: int | None = None
                        ) -> Iterator[GraphColoringBatch]:
        """Iterate the TRAIN set in order (for train-set diagnostics)."""
        bs = batch_size or self.batch_size
        recs = self.train_records
        n = len(recs)
        for start in range(0, n, bs):
            batch = list(recs[start:start + bs])
            while len(batch) < bs:
                batch.append(recs[0])
            yield self._stack(batch)

    def __len__(self):
        return len(self.train_records)


# ---------------------------------------------------------------------------
# CPU smoke (GPU-free): verify the T=1 adjacency identity + the engine contract.
# ---------------------------------------------------------------------------

def _smoke() -> None:
    """Self-check: T=1 membership reproduces the adjacency matrix, canonical
    labeling is deterministic, and batches satisfy the engine contract."""
    # 1. canonical labeling determinism / first-use rule.
    assert _canonicalize_coloring([2, 2, 5, 0, 5]) == [0, 0, 1, 2, 1], \
        _canonicalize_coloring([2, 2, 5, 0, 5])
    assert _canonicalize_coloring([7, 3, 7]) == [0, 1, 0]

    # 2. small loader, then verify A_0 == adjacency for one instance.
    loader = GraphColoringLoader(n_instances=40, s_max=12, k_colors=3,
                                 batch_size=4, seed=7)
    batch = loader.sample_batch()
    S = loader.s_max

    # shapes / dtypes
    assert batch.input_cells.shape == (4, S)
    assert batch.cell_valid.shape == (4, S)
    assert batch.value_domain_mask.shape == (4, S, 3)
    assert batch.gold.shape == (4, S)
    assert batch.membership.shape == (4, loader.n_edges_max, S)
    assert batch.latent_type.shape == (4, loader.n_edges_max)
    assert batch.factor_inlet is None
    # input is fully uncolored
    assert int(batch.input_cells.sum().numpy()) == 0
    # gold values in {0 (pad)} U {1..k}
    g = batch.gold.numpy()
    assert g.min() >= 0 and g.max() <= 3

    # 3. T=1 adjacency identity: A_0 = (m_0^T @ m_0 > 0) == graph adjacency.
    mem = batch.membership.numpy()                  # (B, E, S)
    lt = batch.latent_type.numpy()                  # (B, E)
    cv = batch.cell_valid.numpy()                   # (B, S)
    for b in range(mem.shape[0]):
        m0 = mem[b] * (lt[b] == LTYPE_EDGE).astype(np.float32)[:, None]  # (E,S)
        A0 = (m0.T @ m0) > 0                         # (S,S) co-occurrence
        # reconstruct adjacency directly from edge rows.
        adj = np.zeros((S, S), dtype=bool)
        for e in range(m0.shape[0]):
            idx = np.where(mem[b, e] > 0)[0]
            if lt[b, e] != LTYPE_EDGE:
                continue
            assert len(idx) == 2, f"edge row must have exactly 2 ones, got {len(idx)}"
            u, v = int(idx[0]), int(idx[1])
            adj[u, v] = True
            adj[v, u] = True
        # off-diagonal A0 must equal adjacency exactly.
        off = ~np.eye(S, dtype=bool)
        assert np.array_equal(A0 & off, adj & off), \
            f"A0 off-diagonal != adjacency for batch elem {b}"
        # diagonal of A0 is 1 exactly for endpoints (vertices with >=1 edge); the
        # builder's self-edge fix forces the rest, so this is consistent.
        # Every valid vertex with an incident edge appears on the diagonal.
        deg = adj.sum(axis=1)
        for vtx in range(S):
            if cv[b, vtx] > 0.5 and deg[vtx] > 0:
                assert A0[vtx, vtx], f"endpoint vertex {vtx} missing from A0 diagonal"

    # 4. eval iterator yields full batches.
    n_eval = sum(1 for _ in loader.iter_eval(batch_size=4))
    assert n_eval >= 1

    print("[graph_coloring_data] _smoke OK: "
          "canonical labeling + T=1 adjacency identity + engine contract verified.")


if __name__ == "__main__":
    _smoke()
