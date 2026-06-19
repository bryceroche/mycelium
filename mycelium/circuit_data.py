"""circuit_data.py — the nested Boolean-circuit DAG testbed (B3).

The HIERARCHICAL factor graph for the general executor (mycelium/factor_graph_engine.py
+ mycelium/factor_masks.py). KenKen was first (large symmetric cliques on a fixed 49-
cell grid); graph k-coloring was second (tiny binary not-equal constraints on an
arbitrary FLAT graph). Both are FLAT — their factors have no notion of abstraction
LEVEL. This third testbed is the missing piece: a LAYERED Boolean circuit, where a
node's true deduction depth (longest path from any leaf) IS its abstraction level.
This is the testbed where the Tier-1/2 radial-depth claim (radial position =
abstraction level; deduction-depth <-> radial traversal) can actually express —
KenKen's lateral cliques mute it (CLAUDE.md §5: "reserve the radial-depth verdict
for a DAG"). If the SAME engine solves circuits, AND the per-node depth label `lvl`
correlates with the convergence instrument (against the matched depth-shuffle NULL),
the hierarchy claim has a clean home.

THE TASK
--------
A leveled Boolean circuit:
  * level 0      = LEAF inputs (primary inputs).  GIVEN (their bit is observed).
  * levels 1..D  = GATE nodes.  DEDUCED (their value is computed, not given).
Each gate has a type in {AND, OR, NOT} (NOT fan-in 1; AND/OR fan-in 2).  XOR is an
OPT-IN stress companion (it is NOT linearly separable, so AND/OR/NOT is the PRIMARY
corpus and XOR a harder stress band).  A gate's operands are STRICTLY LOWER-LEVEL
(the leveled generator guarantees a DAG by construction — no cycles possible).

PER-NODE DEPTH LABEL.  lvl(v) = the LONGEST path (in edges) from any leaf to v.
Leaves have lvl 0.  A gate's lvl = 1 + max(lvl(operand)).  This is the TRUE
deduction depth — how many serial gate-evaluations a bottom-up solver needs before v
is determined.  It is NOT the same as the assigned generation level when a gate
happens to draw both operands from far-below levels (then the assigned level can
exceed the longest-path depth); we ALWAYS recompute lvl by longest-path on the final
operand wiring so the label is exact.

PER-INSTANCE DEPTH.  circuit_depth D = max over nodes of lvl(v).  Bucketed into
bands D2/D3/D4/D5 (the difficulty x-axis, mirroring coloring's density BANDS and
KenKen's givens bands).  Caps: D <= 5 and total nodes <= 49 (the engine reuses
kenken.py:kenken_layer_forward, which hard-asserts S == 49).

GROUND TRUTH
------------
Deterministic TOPOLOGICAL forward-eval: take the leaf bits, then evaluate gates
level-by-level (every operand is strictly lower so it is already known).  The result
is UNIQUE — no canonicalization needed (unlike coloring's color-permutation
symmetry; a Boolean circuit on fixed inputs has exactly one output assignment).

THE ENCODING (FactorGraphBatch contract)
-----------------------------------------
* VARIABLE NODES = circuit nodes (leaves + gates).  Padded to s_max; cell_valid=1
  for the n real nodes, 0 for the s_max-n padding slots.  A node's "value" is its
  Boolean bit.
* VALUE SPACE   N = 2.   false -> gold 1, true -> gold 2  (engine does gold_idx =
  gold-1, so bit b in {0,1} is stored as gold b+1).  value_domain_mask = both
  values legal for every valid node (no per-node domain restriction).
* INPUT CELLS   leaves are GIVEN: input_cells[leaf] = bit + 1 (1 or 2).  Gates are
  UNKNOWN: input_cells[gate] = 0.  (The engine's loss supervises only valid AND
  non-given cells: supervise = cell_valid * (1 - (input_cells>0)) -> exactly the
  gate nodes.  Leaves are observed, never supervised.)
* GOLD          topo-eval bit + 1 (1=false, 2=true) for every valid node; pad = 0.
* FACTOR per GATE.  Each gate g is ONE factor whose membership row has 1s at
  {g} UNION {g's operands}.  Cardinality 3 for AND/OR (g + 2 operands),
  2 for NOT (g + 1 operand), 3 for XOR.  membership shape (B, n_gates_max, s_max).
* FACTOR TYPE   latent_type[g] = the gate-type index (AND=0, OR=1, NOT=2 [, XOR=3]).
  n_factor_types T = 3 (or 4 with XOR).  Padding-gate rows (all-zero membership) get
  the GLOBAL sentinel (== T) so they drop out of every type-t adjacency.

  *** Membership encodes the gate's local constraint CLIQUE. ***
  The mask builder (factor_masks.build_factor_attn_bias) computes per type t:
      m_t = membership * (latent_type == t)
      A_t[b,i,j] = (m_t^T @ m_t)[b,i,j] > 0.
  For type-t gate g with members {g, op0, op1}, the symmetric outer product
  m_g^T @ m_g lights up EVERY pair within {g, op0, op1}.  So A_t is the union of all
  type-t gate cliques: "a node attends to the other members of any same-type gate it
  participates in" — a gate attends to its operands AND its operands attend to the
  gate (bidirectional — the engine does DEDUCTION/constraint propagation, not a
  directed forward pass; do NOT encode direction in the mask).  The single global
  head attends to all valid nodes.  Verified in _smoke().

  *** Where the hierarchy lives. ***  The mask is DIRECTIONLESS and LEVEL-BLIND —
  it carries only "who shares a gate".  The HIERARCHY (cell ∈ cage ∈ board analogue:
  leaf -> gate -> gate-of-gate) is carried by (a) the per-node `lvl` labels (the
  Property-2 x-axis the convergence instrument reads) and (b) the gate-TYPE field
  (latent_type, the per-head split).  This is exactly the spec discipline: hierarchy
  is in the coordinate/label fields, not the mask direction (CLAUDE.md §2 verification
  inlet rule, §6 "hierarchy carried by lvl labels + gate-type fields").

MATCHED FLAT CONTROL (the rho NULL, matched by construction)
------------------------------------------------------------
From the SAME circuit instances we expose a DEPTH-SHUFFLED variant: identical nodes,
factors, masks, gold — but the per-node `lvl` labels randomly PERMUTED within each
instance.  This is the matched NULL for the Property-2 read: every structural/learned
quantity is byte-identical, ONLY the depth axis is destroyed.  If the convergence
instrument correlates with the TRUE lvl but not the shuffled lvl, the correlation is
about depth, not some confound.  We expose the true `lvl` AND a shuffle utility
(shuffle_lvl); the loader can emit either via shuffle_depth=True.  (A separately-
generated flattened-CSP control is a possible later add; the depth-shuffle is the
primary, matched-by-construction NULL.)

LOADER
------
CircuitLoader yields CircuitBatch objects satisfying the engine's FactorGraphBatch
contract (mycelium.factor_graph_engine.FactorGraphBatch).  Mirrors graph_coloring_data
exactly: numpy intermediates -> Tensor(...).contiguous().realize(); fixed
s_max / n_gates_max for a static JIT topology; train/test split; batch_size; seed.

SUBSTRATE
---------
GPU-free: numpy/python data generation; tinygrad Tensors only to pack the batch
(matching graph_coloring_data._stack).  No dtypes.float32 Tensor literal baked into a
JIT graph (data tensors wrap numpy with explicit dtype, the engine's job).
"""
from __future__ import annotations

import random
from typing import Iterator

import numpy as np
from tinygrad import Tensor, dtypes


# ---------------------------------------------------------------------------
# Difficulty bands (the convergence-instrument x-axis).
# Bands are ordered easy -> hard by circuit_depth D.  Mirrors coloring's d10..d25
# density bands and KenKen's g40..g10 givens bands: a coarse difficulty label by
# the depth of the deepest gate.  D2/D3/D4/D5 == "target max depth".
# ---------------------------------------------------------------------------
BANDS = ["D2", "D3", "D4", "D5"]
_BAND_TARGET_D = {"D2": 2, "D3": 3, "D4": 4, "D5": 5}

# Gate type catalog.  AND/OR/NOT is the PRIMARY corpus (XOR opt-in).
# The type INDEX is its position in the active gate_types tuple, so AND=0, OR=1,
# NOT=2 for the default; XOR=3 when enabled.  This index IS latent_type[g].
_FANIN = {"AND": 2, "OR": 2, "NOT": 1, "XOR": 2}


# ---------------------------------------------------------------------------
# Gate evaluation (pure python, deterministic)
# ---------------------------------------------------------------------------

def _eval_gate(gtype: str, operands: list[int]) -> int:
    """Evaluate one gate given its operand BITS (0/1).  Returns a bit (0/1)."""
    if gtype == "AND":
        return int(operands[0] & operands[1])
    if gtype == "OR":
        return int(operands[0] | operands[1])
    if gtype == "NOT":
        return int(1 - operands[0])
    if gtype == "XOR":
        return int(operands[0] ^ operands[1])
    raise ValueError(f"unknown gate type {gtype!r}")


# ---------------------------------------------------------------------------
# Longest-path depth (the TRUE per-node deduction depth label).
# ---------------------------------------------------------------------------

def _longest_path_lvl(n: int, operands: list[list[int]]) -> list[int]:
    """lvl(v) = longest path (in edges) from any leaf to v.

    operands[v] is the list of operand node-ids for node v (empty for a leaf).
    Because every operand is a STRICTLY LOWER node-id (leveled generation), a single
    forward pass in node-id order is a valid topological order.  Leaves -> 0; a gate
    -> 1 + max(lvl(operand)).
    """
    lvl = [0] * n
    for v in range(n):
        if operands[v]:
            lvl[v] = 1 + max(lvl[o] for o in operands[v])
        else:
            lvl[v] = 0
    return lvl


# ---------------------------------------------------------------------------
# Depth-shuffle control (the matched rho NULL)
# ---------------------------------------------------------------------------

def shuffle_lvl(lvl: list[int], n: int, rng: random.Random) -> list[int]:
    """Randomly permute the per-node depth labels WITHIN one instance (the NULL).

    Returns a NEW length-n list that is a random permutation of the first n entries
    of lvl (the real nodes); padding entries (>= n) are not touched by the loader.
    Everything else about the instance (nodes, factors, masks, gold) is unchanged —
    this destroys ONLY the node<->depth association, giving a matched-by-construction
    NULL for the Property-2 correlation.
    """
    perm = list(lvl[:n])
    rng.shuffle(perm)
    return perm


# ---------------------------------------------------------------------------
# Difficulty band from circuit depth
# ---------------------------------------------------------------------------

def _band_for_depth(D: int) -> str:
    """Bucket a circuit_depth D (max lvl) into a difficulty band D2..D5."""
    if D <= 2:
        return "D2"
    if D == 3:
        return "D3"
    if D == 4:
        return "D4"
    return "D5"


# ---------------------------------------------------------------------------
# One instance
# ---------------------------------------------------------------------------

def generate_instance(rng: random.Random, s_max: int,
                      band: str, gate_types: tuple[str, ...] = ("AND", "OR", "NOT"),
                      n_leaves_range: tuple[int, int] = (3, 8),
                      gates_per_level_range: tuple[int, int] = (1, 4),
                      max_resample: int = 80) -> dict | None:
    """Generate ONE leveled Boolean-circuit instance in the given depth band.

    band -> target circuit depth D (D2..D5).  Returns a dict of python/numpy fields
    (no batch axis), or None if generation failed (caller resamples).

    Generation:
      * level 0: n_leaves random leaf inputs, each a random bit.
      * for assigned level 1..target_D:
          add a few gates; each gate picks a type from gate_types, then operands
          from the pool of ALL strictly-lower nodes (so the DAG is guaranteed and a
          gate may reach back several levels — which is exactly why lvl is the
          LONGEST path, recomputed at the end, not the assigned level).
      * forward-eval bottom-up for the gold bits.
      * recompute lvl = longest-path; D = max lvl.
    Resampled until total nodes <= s_max, D == target (so bands are honest), and at
    least one gate exists.

    Fields:
      n            : int                     — number of real nodes (<= s_max).
      n_leaves     : int                     — leaf count (the GIVEN nodes).
      n_gates      : int                     — gate count (the DEDUCED nodes).
      operands     : list[list[int]] len n   — operand node-ids per node ([] for leaf).
      gtypes       : list[str|None] len n    — gate type per node (None for leaf).
      gtype_idx    : list[int] len n         — gate-type INDEX (-1 for leaf).
      bits         : list[int] len n         — topo-eval bit per node (0/1).
      leaf_bits    : list[int]               — the leaf input bits (subset of bits).
      lvl          : list[int] len n         — LONGEST-path depth per node.
      circuit_depth: int                     — max lvl (== target D).
      band         : str
    """
    target_D = _BAND_TARGET_D.get(band, 3)
    type_index = {gt: i for i, gt in enumerate(gate_types)}
    has_binary = any(_FANIN[gt] == 2 for gt in gate_types)
    if not has_binary:
        # NOT alone can build a depth chain, but no fan-in-2 gate => a path graph.
        # Allowed, but we still need >=2 nodes per gate level.
        pass

    for _ in range(max_resample):
        n_leaves = rng.randint(n_leaves_range[0], n_leaves_range[1])
        # node 0..n_leaves-1 are leaves.
        operands: list[list[int]] = [[] for _ in range(n_leaves)]
        gtypes: list[str | None] = [None] * n_leaves
        leaf_bits = [rng.randint(0, 1) for _ in range(n_leaves)]
        bits: list[int] = list(leaf_bits)        # gate bits appended as we add gates

        ok = True
        # assigned levels 1..target_D — each adds >=1 gate drawing from lower nodes.
        for assigned_lvl in range(1, target_D + 1):
            n_gates_here = rng.randint(gates_per_level_range[0],
                                       gates_per_level_range[1])
            # The FIRST gate of the deepest level MUST reach the previous level so the
            # longest path actually attains target_D.  We enforce: at least one gate at
            # assigned_lvl draws an operand whose lvl == assigned_lvl-1.  Easiest robust
            # rule: force one operand of the first gate to be the most-recently-added
            # node (which is at level assigned_lvl-1 by construction of the loop).
            for gi in range(n_gates_here):
                if len(operands) >= s_max:
                    break
                gt = gate_types[rng.randrange(len(gate_types))]
                fanin = _FANIN[gt]
                pool = list(range(len(operands)))   # all strictly-lower nodes
                if len(pool) < fanin:
                    ok = False
                    break
                if gi == 0 and assigned_lvl >= 1:
                    # guarantee depth progression: first operand = last node added.
                    deepest = pool[-1]
                    rest_pool = [p for p in pool if p != deepest]
                    rng.shuffle(rest_pool)
                    ops = [deepest] + rest_pool[:fanin - 1]
                    rng.shuffle(ops)
                else:
                    ops = rng.sample(pool, fanin)
                v = len(operands)
                operands.append(list(ops))
                gtypes.append(gt)
                bits.append(_eval_gate(gt, [bits[o] for o in ops]))
            if not ok:
                break
        if not ok:
            continue

        n = len(operands)
        if n > s_max:
            continue
        if n == n_leaves:
            continue                                # no gates -> degenerate, resample

        lvl = _longest_path_lvl(n, operands)
        D = max(lvl)
        if D != target_D:
            continue                                # band must be honest, resample

        gtype_idx = [type_index[g] if g is not None else -1 for g in gtypes]
        n_gates = n - n_leaves
        return {
            "n": n,
            "n_leaves": n_leaves,
            "n_gates": n_gates,
            "operands": operands,
            "gtypes": gtypes,
            "gtype_idx": gtype_idx,
            "bits": bits,
            "leaf_bits": leaf_bits,
            "lvl": lvl,
            "circuit_depth": D,
            "band": band,
        }
    return None


def generate_corpus(n_instances: int, s_max: int, seed: int = 0,
                    bands: list[str] | None = None,
                    gate_types: tuple[str, ...] = ("AND", "OR", "NOT"),
                    ) -> list[dict]:
    """Generate a balanced curriculum corpus across depth bands.

    Returns a list of instance dicts (see generate_instance).  Bands are sampled
    round-robin so the corpus is balanced across depth; every instance evaluates to
    a unique gold by topological forward-eval.
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
        inst = generate_instance(rng, s_max, band, gate_types=gate_types)
        if inst is not None:
            out.append(inst)
    if len(out) < n_instances:
        raise RuntimeError(
            f"generate_corpus produced {len(out)}/{n_instances} after "
            f"{attempts} attempts (s_max={s_max} too small for the bands?)")
    return out


# ---------------------------------------------------------------------------
# latent_type sentinel: padding gate rows get the GLOBAL sentinel (== T) so they
# drop out of every type-t adjacency.  factor_masks treats any latent_type not in
# 0..T-1 as non-type (the global head uses an all-valid mask, not membership).
# ---------------------------------------------------------------------------


def encode_instance(inst: dict, s_max: int, n_gates_max: int,
                    n_factor_types: int) -> dict:
    """Flatten ONE circuit instance onto fixed (s_max, n_gates_max) -> numpy arrays.

    Mirrors graph_coloring_data.encode_instance: fixed shapes for a static JIT
    topology.  n_factor_types T = len(gate_types); padding gate rows get latent_type
    = T (the global sentinel).
    """
    n = int(inst["n"])
    n_leaves = int(inst["n_leaves"])
    operands = inst["operands"]
    gtypes = inst["gtypes"]
    gtype_idx = inst["gtype_idx"]
    bits = inst["bits"]
    lvl = inst["lvl"]
    T = int(n_factor_types)
    sentinel = T                                              # global-head sentinel
    assert n <= s_max, f"n={n} exceeds s_max={s_max}"

    # --- per-node (variable node) arrays ---
    input_cells = np.zeros((s_max,), dtype=np.int32)         # gates unknown (0)
    gold = np.zeros((s_max,), dtype=np.int32)                # 0 on padding
    cell_valid = np.zeros((s_max,), dtype=np.float32)
    value_domain_mask = np.zeros((s_max, 2), dtype=np.float32)  # N=2 (false/true)
    lvl_arr = np.full((s_max,), -1, dtype=np.int32)          # -1 on padding
    is_leaf = np.zeros((s_max,), dtype=np.float32)
    for v in range(n):
        cell_valid[v] = 1.0
        gold[v] = int(bits[v]) + 1                            # bit b -> gold b+1
        value_domain_mask[v, :] = 1.0                        # both values legal
        lvl_arr[v] = int(lvl[v])
        if gtypes[v] is None:                                # leaf -> GIVEN
            input_cells[v] = int(bits[v]) + 1                # observed leaf bit
            is_leaf[v] = 1.0

    # --- per-gate (factor node) arrays ---
    # membership (n_gates_max, s_max): gate row g has 1s at {g} U {operands}.
    membership = np.zeros((n_gates_max, s_max), dtype=np.float32)
    latent_type = np.full((n_gates_max,), sentinel, dtype=np.int32)  # pad = sentinel
    g_row = 0
    for v in range(n):
        if gtypes[v] is None:
            continue                                          # leaves are not factors
        assert g_row < n_gates_max, \
            f"n_gates exceeds n_gates_max={n_gates_max}"
        membership[g_row, v] = 1.0                            # the gate node itself
        for o in operands[v]:
            membership[g_row, o] = 1.0                        # its operands
        latent_type[g_row] = int(gtype_idx[v])               # gate-type index
        g_row += 1

    return {
        "input_cells": input_cells,
        "gold": gold,
        "cell_valid": cell_valid,
        "value_domain_mask": value_domain_mask,
        "membership": membership,
        "latent_type": latent_type,
        "lvl": lvl_arr,                                       # (s_max,) per-node depth
        "is_leaf": is_leaf,                                   # (s_max,) leaf indicator
        "circuit_depth": int(inst.get("circuit_depth", max(0, *([0] + list(lvl[:n]))))),
        "n": n,
        "n_leaves": n_leaves,
        "n_gates": int(inst.get("n_gates", n - n_leaves)),
        "band": str(inst.get("band", "all")),
    }


# ---------------------------------------------------------------------------
# Batch (satisfies the engine's FactorGraphBatch contract)
# ---------------------------------------------------------------------------

class CircuitBatch:
    """A realized batch of Boolean-circuit tensors.

    Satisfies mycelium.factor_graph_engine.FactorGraphBatch (same tensor attrs +
    shapes/dtypes), so factor_breathing_forward / factor_loss / factor_accuracy
    consume it directly.  No factor_inlet (a circuit has no arithmetic to verify ->
    spec.has_factor_inlet=False).

    Tensor attributes
    -----------------
    input_cells       Tensor (B, s_max)         int   — leaf bit+1 (given), 0 gate.
    cell_valid        Tensor (B, s_max)         float — 1 real node / 0 padding.
    value_domain_mask Tensor (B, s_max, 2)      float — both bits legal / 0 pad.
    gold              Tensor (B, s_max)         int   — bit+1 (1=false,2=true)/0 pad.
    membership        Tensor (B, n_gates_max, s_max) float — gate U operands per gate.
    latent_type       Tensor (B, n_gates_max)   int   — gate-type idx / T (=sentinel) pad.

    Python-side metadata (NOT tensors; eval/metric-only)
    ----------------------------------------------------
    deduction_depth   list[int]  — per-instance circuit_depth D (Property-2 x-axis;
                                   the engine's FactorGraphBatch reads this attr name).
    lvl               np.ndarray (B, s_max) int — PER-NODE longest-path depth
                                   (-1 on padding).  The fine-grained depth axis; if
                                   shuffle_depth=True the real-node entries are
                                   permuted within each instance (the matched NULL).
    is_leaf           np.ndarray (B, s_max) float — 1 leaf / 0 gate-or-pad.
    circuit_depth     list[int]  — per-instance max depth (== deduction_depth).
    band              list[str]  — difficulty band (D2..D5).
    n                 list[int]  — real node count per instance.
    n_leaves          list[int]  — leaf count per instance.
    n_gates           list[int]  — gate count per instance.
    depth_shuffled    bool       — whether `lvl` is the NULL (permuted) variant.
    """

    def __init__(self, d: dict):
        self.input_cells: Tensor = d["input_cells"]
        self.cell_valid: Tensor = d["cell_valid"]
        self.value_domain_mask: Tensor = d["value_domain_mask"]
        self.gold: Tensor = d["gold"]
        self.membership: Tensor = d["membership"]
        self.latent_type: Tensor = d["latent_type"]
        # No verification inlet for a Boolean circuit.
        self.factor_inlet = None
        # python-side metadata
        self.circuit_depth: list[int] = d["circuit_depth"]
        # engine's FactorGraphBatch reads `deduction_depth`; for circuits that IS the
        # per-instance circuit depth D (the coarse Property-2 x-axis).
        self.deduction_depth: list[int] = list(d["circuit_depth"])
        self.lvl: np.ndarray = d["lvl"]                 # (B, s_max) per-node depth
        self.is_leaf: np.ndarray = d["is_leaf"]         # (B, s_max) leaf indicator
        self.band: list[str] = d["band"]
        self.n: list[int] = d["n"]
        self.n_leaves: list[int] = d["n_leaves"]
        self.n_gates: list[int] = d["n_gates"]
        self.depth_shuffled: bool = bool(d.get("depth_shuffled", False))


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

class CircuitLoader:
    """Generates a circuit corpus once, splits train/test, yields batches.

    Fixed shapes across the loader (s_max nodes, n_gates_max gate rows) so the JIT
    graph topology is static — n_gates_max is the max gate count over the whole
    corpus, mirroring graph_coloring_data's fixed n_edges_max.

    Parameters
    ----------
    n_instances  : total corpus size to generate (before the split).
    s_max        : fixed sequence length (== FactorGraphSpec.s_max). DEFAULT 49 —
                   the engine reuses kenken.py:kenken_layer_forward which hard-asserts
                   S == 49.  The data module is general on s_max (the smoke runs at
                   16); production pads circuit nodes to 49 exactly as KenKen pads its
                   7x7 board.  Use s_max=49 to drive factor_breathing_forward.
    n_values     : number of cell values (== FactorGraphSpec.n_values).  Boolean
                   circuits are N=2 (false/true).  Asserts n_values==2.
    batch_size   : batch size.
    seed         : RNG seed (corpus + sampler + depth-shuffle).
    test_frac    : held-out fraction (by instance; disjoint train/test).
    n_gates_max  : override the computed max-gate cap (must be >= corpus max).
    bands        : difficulty bands to sample (default D2..D5).
    gate_types   : active gate types — AND/OR/NOT default; add 'XOR' for the stress
                   companion.  n_factor_types T = len(gate_types) (3 default, 4 w/XOR).
    shuffle_depth: if True, the emitted `lvl` metadata is the DEPTH-SHUFFLED NULL
                   (per-node depths permuted within each instance; nodes/factors/
                   masks/gold unchanged).  Default False (true labels).
    """

    def __init__(self, n_instances: int = 8000, s_max: int = 49,
                 n_values: int = 2, batch_size: int = 8, seed: int = 0,
                 test_frac: float = 0.15, n_gates_max: int | None = None,
                 bands: list[str] | None = None,
                 gate_types: tuple[str, ...] = ("AND", "OR", "NOT"),
                 shuffle_depth: bool = False):
        assert int(n_values) == 2, \
            f"Boolean circuits are N=2 (false/true); got n_values={n_values}"
        self.s_max = int(s_max)
        self.n_values = int(n_values)
        self.batch_size = int(batch_size)
        self.gate_types = tuple(gate_types)
        self.n_factor_types = len(self.gate_types)
        self.shuffle_depth = bool(shuffle_depth)
        self.rng = random.Random(seed)
        self._shuf_rng = random.Random(seed + 9973)        # independent shuffle RNG

        corpus = generate_corpus(n_instances, s_max, seed=seed, bands=bands,
                                 gate_types=self.gate_types)
        # shuffle then split by instance (disjoint train/test).
        split_rng = random.Random(seed + 1)
        split_rng.shuffle(corpus)
        n_test = max(1, int(round(test_frac * len(corpus))))
        self.test_records = corpus[:n_test]
        self.train_records = corpus[n_test:]
        assert self.train_records and self.test_records, \
            "split produced an empty train or test set; raise n_instances"

        corpus_max_gates = max(r["n_gates"] for r in corpus)
        self.n_gates_max = (int(n_gates_max) if n_gates_max is not None
                            else corpus_max_gates)
        assert self.n_gates_max >= corpus_max_gates, (
            f"n_gates_max={self.n_gates_max} < corpus max gates {corpus_max_gates}")

        by_band: dict[str, int] = {}
        for r in corpus:
            by_band[r["band"]] = by_band.get(r["band"], 0) + 1
        print(f"[circuit_data] generated {len(corpus)} instances "
              f"(train {len(self.train_records)} / test {len(self.test_records)}); "
              f"N={self.n_values} s_max={self.s_max} n_gates_max={self.n_gates_max} "
              f"types={self.gate_types} shuffle_depth={self.shuffle_depth}; "
              f"by band: {sorted(by_band.items())}", flush=True)

    # -- packing --------------------------------------------------------------

    def _stack(self, picks: list[dict]) -> CircuitBatch:
        encs = [encode_instance(r, self.s_max, self.n_gates_max,
                                self.n_factor_types) for r in picks]

        # Per-node depth labels (B, s_max); optionally depth-shuffled (the NULL).
        lvl_rows = []
        for e in encs:
            lvl_v = e["lvl"].copy()                          # (s_max,) -1 on pad
            if self.shuffle_depth:
                n = int(e["n"])
                # permute the real-node entries in place; padding (-1) untouched.
                real = lvl_v[:n].tolist()
                self._shuf_rng.shuffle(real)
                lvl_v[:n] = np.array(real, dtype=np.int32)
            lvl_rows.append(lvl_v)
        lvl_stack = np.stack(lvl_rows).astype(np.int32)      # (B, s_max)
        is_leaf_stack = np.stack([e["is_leaf"] for e in encs]).astype(np.float32)

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
            "lvl":               lvl_stack,                  # numpy metadata
            "is_leaf":           is_leaf_stack,              # numpy metadata
            "circuit_depth":     [e["circuit_depth"] for e in encs],
            "n":                 [e["n"] for e in encs],
            "n_leaves":          [e["n_leaves"] for e in encs],
            "n_gates":           [e["n_gates"] for e in encs],
            "band":              [e["band"] for e in encs],
            "depth_shuffled":    self.shuffle_depth,
        }
        return CircuitBatch(d)

    # -- sampling -------------------------------------------------------------

    def sample_batch(self) -> CircuitBatch:
        picks = self.rng.choices(self.train_records, k=self.batch_size)
        return self._stack(picks)

    def iter_eval(self, batch_size: int | None = None) -> Iterator[CircuitBatch]:
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
                        ) -> Iterator[CircuitBatch]:
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
# CPU smoke (GPU-free): verify gate cliques + topo gold + longest-path lvl +
# the engine contract.
# ---------------------------------------------------------------------------

def _smoke() -> None:
    """Self-check: gate membership cardinality, m_t^T@m_t cliques, gold matches an
    independent re-eval, lvl is longest-path, and batches satisfy the contract."""
    # 1. longest-path lvl on a tiny hand circuit.
    #    nodes 0,1 leaves; 2 = AND(0,1) lvl1; 3 = NOT(2) lvl2; 4 = OR(0,3) lvl3.
    ops = [[], [], [0, 1], [2], [0, 3]]
    lvl = _longest_path_lvl(5, ops)
    assert lvl == [0, 0, 1, 2, 3], lvl

    # 2. gate eval truth table.
    assert _eval_gate("AND", [1, 1]) == 1 and _eval_gate("AND", [1, 0]) == 0
    assert _eval_gate("OR", [0, 0]) == 0 and _eval_gate("OR", [0, 1]) == 1
    assert _eval_gate("NOT", [0]) == 1 and _eval_gate("NOT", [1]) == 0
    assert _eval_gate("XOR", [1, 1]) == 0 and _eval_gate("XOR", [1, 0]) == 1

    # 3. small loader, then verify the engine contract + clique reconstruction.
    loader = CircuitLoader(n_instances=80, s_max=16, n_values=2,
                           batch_size=4, seed=7)
    batch = loader.sample_batch()
    S = loader.s_max
    T = loader.n_factor_types                         # 3 (AND/OR/NOT)

    # shapes / dtypes
    assert batch.input_cells.shape == (4, S)
    assert batch.cell_valid.shape == (4, S)
    assert batch.value_domain_mask.shape == (4, S, 2)
    assert batch.gold.shape == (4, S)
    assert batch.membership.shape == (4, loader.n_gates_max, S)
    assert batch.latent_type.shape == (4, loader.n_gates_max)
    assert batch.factor_inlet is None
    assert batch.lvl.shape == (4, S)
    assert batch.is_leaf.shape == (4, S)
    # gold values in {0 (pad)} U {1,2}; leaf input_cells in {1,2}, gates 0.
    g = batch.gold.numpy()
    assert g.min() >= 0 and g.max() <= 2
    ic = batch.input_cells.numpy()
    assert ic.min() >= 0 and ic.max() <= 2

    mem = batch.membership.numpy()                    # (B, G, S)
    lt = batch.latent_type.numpy()                    # (B, G)
    cv = batch.cell_valid.numpy()                     # (B, S)
    leaf = batch.is_leaf                              # (B, S) numpy metadata
    lvl_b = batch.lvl                                 # (B, S) numpy metadata
    sentinel = T

    for b in range(mem.shape[0]):
        n = int(batch.n[b])
        n_gates = int(batch.n_gates[b])

        # --- gate membership cardinality (3 for AND/OR, 2 for NOT) ---
        real_rows = [g for g in range(mem.shape[1]) if lt[b, g] != sentinel]
        assert len(real_rows) == n_gates, \
            f"real gate rows {len(real_rows)} != n_gates {n_gates}"
        for grow in real_rows:
            card = int((mem[b, grow] > 0).sum())
            gtype_idx = int(lt[b, grow])
            gtype = loader.gate_types[gtype_idx]
            expect = 1 + _FANIN[gtype]                # gate node + fan-in operands
            assert card == expect, \
                f"gate row {grow} type {gtype}: card {card} != {expect}"

        # --- m_t^T @ m_t > 0 reconstructs per-type gate cliques ---
        for t in range(T):
            m_t = mem[b] * (lt[b] == t).astype(np.float32)[:, None]   # (G,S)
            A_t = (m_t.T @ m_t) > 0                                   # (S,S)
            # reconstruct the type-t clique union directly from member sets.
            expect_adj = np.zeros((S, S), dtype=bool)
            for grow in range(mem.shape[1]):
                if lt[b, grow] != t:
                    continue
                idx = np.where(mem[b, grow] > 0)[0]
                for ii in idx:
                    for jj in idx:
                        expect_adj[ii, jj] = True
                assert len(idx) == 1 + _FANIN[loader.gate_types[t]], \
                    f"type-{t} gate row {grow} wrong cardinality"
            assert np.array_equal(A_t, expect_adj), \
                f"A_{t} != type-{t} clique union for batch elem {b}"

        # --- pad gate rows are the global sentinel (drop from every type) ---
        for grow in range(mem.shape[1]):
            if grow not in real_rows:
                assert lt[b, grow] == sentinel, \
                    f"pad gate row {grow} latent_type {lt[b, grow]} != {sentinel}"
                assert int((mem[b, grow] > 0).sum()) == 0, \
                    f"pad gate row {grow} has membership"

        # --- gold matches an INDEPENDENT topological re-eval ---
        # rebuild operands per gate node from the membership rows + the gate node id.
        # The gate node is the member with input_cells==0 that is NOT a leaf; but we
        # stored gates as the row's HIGHEST node id (gates are added after operands,
        # so the gate node-id > all its operand ids).  Recover it that way.
        bits = [None] * n
        for v in range(n):
            if leaf[b, v] > 0.5:
                bits[v] = int(ic[b, v]) - 1            # leaf bit from input_cells
        # operand recovery: for each real gate row, the gate node = max member id.
        gate_rows_by_node: dict[int, tuple[str, list[int]]] = {}
        for grow in real_rows:
            idx = sorted(int(x) for x in np.where(mem[b, grow] > 0)[0])
            gate_node = idx[-1]                         # gate added after operands
            operand_ids = idx[:-1]
            gtype = loader.gate_types[int(lt[b, grow])]
            gate_rows_by_node[gate_node] = (gtype, operand_ids)
        # evaluate in node-id order (topological: operands have lower ids).
        for v in range(n):
            if bits[v] is not None:
                continue
            assert v in gate_rows_by_node, f"node {v} neither leaf nor gate"
            gtype, operand_ids = gate_rows_by_node[v]
            bits[v] = _eval_gate(gtype, [bits[o] for o in operand_ids])
        gold_b = batch.gold.numpy()[b]
        for v in range(n):
            assert gold_b[v] == bits[v] + 1, \
                f"gold[{v}]={gold_b[v]} != reeval {bits[v]+1}"

        # --- lvl is the longest path (recompute from recovered operands) ---
        operands_rebuilt: list[list[int]] = [[] for _ in range(n)]
        for gate_node, (_, operand_ids) in gate_rows_by_node.items():
            operands_rebuilt[gate_node] = list(operand_ids)
        lvl_re = _longest_path_lvl(n, operands_rebuilt)
        for v in range(n):
            assert int(lvl_b[b, v]) == lvl_re[v], \
                f"lvl[{v}]={lvl_b[b, v]} != longest-path {lvl_re[v]}"
        # padding lvl is -1.
        for v in range(n, S):
            assert int(lvl_b[b, v]) == -1, f"pad lvl[{v}] != -1"

        # --- supervised set == gates (valid AND not given) ---
        observed = (ic[b] > 0).astype(np.float32)
        supervise = cv[b] * (1.0 - observed)
        n_sup = int(supervise.sum())
        assert n_sup == n_gates, \
            f"supervised {n_sup} != n_gates {n_gates} (leaves must be GIVEN)"

    # 4. depth-shuffle control: identical tensors, ONLY lvl permuted within instance.
    loader_null = CircuitLoader(n_instances=80, s_max=16, n_values=2,
                                batch_size=4, seed=7, shuffle_depth=True)
    b_true = loader.iter_eval(batch_size=4).__next__()
    b_null = loader_null.iter_eval(batch_size=4).__next__()
    # structural tensors byte-identical (same corpus/seed/split).
    assert np.array_equal(b_true.gold.numpy(), b_null.gold.numpy())
    assert np.array_equal(b_true.membership.numpy(), b_null.membership.numpy())
    assert np.array_equal(b_true.latent_type.numpy(), b_null.latent_type.numpy())
    assert b_null.depth_shuffled and not b_true.depth_shuffled
    # the NULL lvl is a within-instance permutation of the true lvl (same multiset).
    for b in range(b_true.lvl.shape[0]):
        n = int(b_true.n[b])
        assert sorted(b_true.lvl[b, :n].tolist()) == sorted(b_null.lvl[b, :n].tolist())
        # the deepest depth is preserved (multiset), circuit_depth unchanged.
        assert b_true.circuit_depth[b] == b_null.circuit_depth[b]

    # 5. standalone shuffle_lvl utility is a within-instance permutation.
    lvl_demo = [0, 0, 1, 2, 3]
    perm = shuffle_lvl(lvl_demo, 5, random.Random(0))
    assert sorted(perm) == sorted(lvl_demo) and len(perm) == 5

    # 6. eval iterator yields full batches; deduction_depth mirrors circuit_depth.
    n_eval = sum(1 for _ in loader.iter_eval(batch_size=4))
    assert n_eval >= 1
    one = loader.iter_eval(batch_size=4).__next__()
    assert one.deduction_depth == one.circuit_depth

    print("[circuit_data] _smoke OK: gate cliques + topo gold + longest-path lvl "
          "+ depth-shuffle NULL + engine contract verified.")


if __name__ == "__main__":
    _smoke()
