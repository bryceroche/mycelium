"""csp_coloring_legacy.py — the LEGACY coloring-specific reference (frozen).

This is a byte-for-byte preserved copy of the pre-refactor mycelium/csp_search.py
(coloring-only AC-3 / DSATUR / LCV + verifier + skeleton). It is kept REACHABLE ONLY
as the PARITY REFERENCE for the Phase-0 behavior-preservation gate: the new
predicate-driven core (mycelium/csp_core.py + csp_registry.py + csp_domains.py) is
proven equal to THIS module on a fixed coloring fixture set (see
scripts/test_csp_parity.py). It is NOT on the live path; new code should import the
general core, not this file.

ORIGINAL HEADER (verbatim):
===========================
csp_search.py — a PURE-SYMBOLIC, GPU-FREE branch-and-propagate skeleton for
graph k-coloring, plus the EXACT verifier and the symbolic plug-ins.

PATH B — "policy-free verifier-driven branch-and-propagate".
============================================================
A viability gate (scripts/analyze_search_guidance_gate.py on fg_coloring_k16_final)
established three facts about the trained factor-graph deducer on the search-hard
band (depth>=3):
  - G1 FAIL: the learned policy is CONFIDENTLY WRONG at the vertices it gets wrong
    (puts BELOW-chance mass on gold). => do NOT use the policy as a search prior/value.
  - G2 PASS (AUC 0.69): per-vertex ENTROPY localises WHERE the deducer is unsure.
  - G4 PASS + MONOTONIC: CLAMPing a few vertices to a commitment + re-deducing
    IMPROVES the rest. => the deducer PROPAGATES partial commitments productively.
So the search SUBSTRATE works (localise + propagate); only the learned guidance is
bad. Path B uses the deducer ONLY as a propagator + ambiguity-detector and uses the
CSP's EXACT verifier as the value/pruning function — graph coloring is verifiable,
so NO learned value is needed.

THE DESIGN: one backtracking skeleton, pluggable components (fair by construction).
----------------------------------------------------------------------------------
A SINGLE depth-first branch-and-propagate skeleton, `backtrack_search`, parameterised
by three pluggable callables:
    propagate_fn(state) -> CSPState            # tighten domains / fix vertices
    varorder_fn(state)  -> int (unresolved v)  # pick the next branch vertex
    valorder_fn(state, v) -> list[int]         # ordered candidate colors for v
Plus FIXED, shared, EXACT logic used by ALL configs:
    - the verifier: is_proper_partial / is_complete_proper (no monochromatic edge).
    - forward-check prune: a branch is dead if any unassigned vertex has an EMPTY
      remaining domain, or any assigned edge is violated.
    - budget: a cap on decision-nodes expanded; on exhaustion return the BEST-EFFORT
      (most-complete proper partial seen) + status='budget'.
The ONLY thing that differs between a baseline and the neural version is which
plug-ins are passed. That is what makes the baselines fair-by-construction.

This module is the PURE-SYMBOLIC half: the skeleton, the verifier, the symbolic
plug-ins (AC-3, DSATUR, LCV + random/no-op controls), and the complete symbolic
ceiling solver `solve_symbolic`. The neural plug-ins live in
scripts/search_coloring.py and are INJECTED into this skeleton (so the search logic
is byte-identical between symbolic and neural configs). See the INTERFACE section.

GRAPH / COLOR CONVENTIONS (must match mycelium/graph_coloring_data.py)
---------------------------------------------------------------------
* Vertices are 0..n_vertices-1.
* `edges` is an iterable of (u, v) undirected pairs, 0-indexed, u != v. Self-loops
  are rejected (a self-loop makes the graph trivially non-colorable; we treat it as
  an error rather than silently dropping it).
* Colors used INSIDE this module are 0-indexed, 0..k-1. (graph_coloring_data stores
  gold as color+1; the neural wrapper in scripts/search_coloring.py is responsible
  for the +1/-1 conversion at the boundary — this module stays 0-indexed throughout.)
* An ASSIGNMENT is a dict {vertex: color} OR a list of length n_vertices with -1 for
  unassigned. The verifier accepts either form (see _as_color_lookup).

GPU-FREE: pure python + numpy. ast.parse clean, CPU import clean, CPU selftest under
SELFTEST_ONLY=1.
"""

from __future__ import annotations

import ast
import os
import random
import sys
from dataclasses import dataclass, field, replace
from typing import Callable, Iterable, Optional, Sequence, Union

UNASSIGNED = -1

# An assignment may be a dict {v: color} or a list[int] with -1 = unassigned.
Assignment = Union[dict, list]
Edges = Sequence[tuple]


# ===========================================================================
# Edge normalisation
# ===========================================================================

def normalize_edges(edges: Edges, n_vertices: int) -> list[tuple[int, int]]:
    """Validate + canonicalise an edge list to a sorted, deduplicated list of
    (u, v) with u < v, both in [0, n_vertices). Rejects self-loops and
    out-of-range endpoints (these are caller bugs, not solver inputs)."""
    out: set[tuple[int, int]] = set()
    for e in edges:
        u, v = int(e[0]), int(e[1])
        if u == v:
            raise ValueError(f"self-loop edge ({u},{v}) is not a valid coloring constraint")
        if not (0 <= u < n_vertices) or not (0 <= v < n_vertices):
            raise ValueError(f"edge ({u},{v}) endpoint out of range for n_vertices={n_vertices}")
        out.add((u, v) if u < v else (v, u))
    return sorted(out)


def build_adjacency(edges: Edges, n_vertices: int) -> list[set[int]]:
    """Adjacency sets adj[v] = {neighbours of v}. Used by AC-3, DSATUR, LCV,
    forward-check, and the verifier."""
    adj: list[set[int]] = [set() for _ in range(n_vertices)]
    for u, v in normalize_edges(edges, n_vertices):
        adj[u].add(v)
        adj[v].add(u)
    return adj


def edges_from_membership(membership_np, latent_type_np, b: int,
                          ltype_edge: int = 0):
    """Recover the 0-indexed edge list for batch element b from the engine's
    (B, n_edges_max, s_max) membership + (B, n_edges_max) latent_type tensors.

    A real edge row (latent_type==ltype_edge) has EXACTLY TWO 1s, at columns u,v.
    Padding rows (latent_type != ltype_edge) are skipped. This is the bridge the
    neural wrapper uses to feed graph structure into this symbolic module.
    """
    import numpy as np
    mem = np.asarray(membership_np)
    lt = np.asarray(latent_type_np)
    edges: list[tuple[int, int]] = []
    n_edges = mem.shape[1]
    for e in range(n_edges):
        if int(lt[b, e]) != ltype_edge:
            continue
        idx = np.where(mem[b, e] > 0.0)[0]
        if len(idx) == 2:
            u, v = int(idx[0]), int(idx[1])
            edges.append((u, v) if u < v else (v, u))
    return edges


# ===========================================================================
# THE EXACT VERIFIER (the arbiter — soundness lives here)
# ===========================================================================

def _as_color_lookup(assignment: Assignment) -> dict:
    """Normalise assignment (dict OR list-with-(-1)) to a dict {v: color} that
    contains ONLY assigned vertices (color != UNASSIGNED)."""
    if isinstance(assignment, dict):
        return {int(v): int(c) for v, c in assignment.items() if int(c) != UNASSIGNED}
    return {i: int(c) for i, c in enumerate(assignment) if int(c) != UNASSIGNED}


def is_proper_partial(assignment: Assignment, edges: Edges,
                      n_vertices: Optional[int] = None) -> bool:
    """True iff NO edge is monochromatic AMONG THE ASSIGNED vertices.

    A partial assignment is 'proper' if every edge whose BOTH endpoints are
    assigned has differently-colored endpoints. Unassigned endpoints impose no
    constraint yet. This is the soundness backstop: the skeleton calls it before
    ever extending, so a wrong neural auto-commit that violates an edge is caught.
    """
    color = _as_color_lookup(assignment)
    if n_vertices is None:
        n_vertices = (max(color) + 1) if color else 0
    for u, v in normalize_edges(edges, max(n_vertices, (max(color) + 1) if color else 0)):
        cu = color.get(u, UNASSIGNED)
        cv = color.get(v, UNASSIGNED)
        if cu != UNASSIGNED and cv != UNASSIGNED and cu == cv:
            return False
    return True


def is_complete_proper(assignment: Assignment, edges: Edges,
                       n_vertices: int, k: Optional[int] = None) -> bool:
    """True iff EVERY vertex 0..n_vertices-1 is assigned a color in [0,k) AND no
    edge is monochromatic. This is the success arbiter: SOLVED <=> this returns
    True. Frame-invariant (any proper k-coloring passes; canonical gold is NOT
    required). If k is given, also checks all colors are in [0,k)."""
    color = _as_color_lookup(assignment)
    if len(color) != n_vertices:
        return False
    for v in range(n_vertices):
        if v not in color:
            return False
        if k is not None and not (0 <= color[v] < k):
            return False
    return is_proper_partial(color, edges, n_vertices)


# ===========================================================================
# SEARCH STATE
# ===========================================================================

@dataclass
class CSPState:
    """The mutable-by-copy state threaded through the search.

    Fields:
      n_vertices : int
      k          : int                         number of colors
      adj        : list[set[int]]              adjacency (immutable; shared)
      colors     : list[int]                   length n; -1 = unassigned, else 0..k-1
      domains    : list[set[int]]              per-vertex remaining legal colors
                                               (for assigned v, domains[v] == {colors[v]})
      n_assigned : int                         count of assigned vertices
    `meta` is a free-form dict for plug-ins (e.g. the neural propagator stashing
    beliefs/entropy). The skeleton never reads it.
    """
    n_vertices: int
    k: int
    adj: list = field(default_factory=list)
    colors: list = field(default_factory=list)
    domains: list = field(default_factory=list)
    n_assigned: int = 0
    meta: dict = field(default_factory=dict)

    def copy(self) -> "CSPState":
        """Deep-enough copy for branching: colors/domains are copied, adj is shared
        (it is immutable graph structure)."""
        return CSPState(
            n_vertices=self.n_vertices,
            k=self.k,
            adj=self.adj,                                   # shared, immutable
            colors=list(self.colors),
            domains=[set(d) for d in self.domains],
            n_assigned=self.n_assigned,
            meta=dict(self.meta),
        )

    def assigned(self) -> dict:
        """{v: color} of currently assigned vertices."""
        return {v: c for v, c in enumerate(self.colors) if c != UNASSIGNED}

    def unassigned_vertices(self) -> list[int]:
        return [v for v in range(self.n_vertices) if self.colors[v] == UNASSIGNED]


def make_initial_state(n_vertices: int, edges: Edges, k: int) -> CSPState:
    """Fresh root state: nothing assigned, every vertex's domain = all k colors."""
    adj = build_adjacency(edges, n_vertices)
    return CSPState(
        n_vertices=n_vertices,
        k=k,
        adj=adj,
        colors=[UNASSIGNED] * n_vertices,
        domains=[set(range(k)) for _ in range(n_vertices)],
        n_assigned=0,
        meta={},
    )


def assign_vertex(state: CSPState, v: int, c: int) -> CSPState:
    """Return a NEW state with vertex v assigned color c and c pruned from the
    domains of its still-unassigned neighbours (unit forward-checking). Does NOT
    validate properness — the caller checks via is_proper_partial / empty-domain.
    """
    s = state.copy()
    s.colors[v] = c
    s.domains[v] = {c}
    s.n_assigned += 1
    for w in s.adj[v]:
        if s.colors[w] == UNASSIGNED and c in s.domains[w]:
            s.domains[w].discard(c)
    return s


def has_empty_domain(state: CSPState) -> bool:
    """Forward-check dead-end test: any UNASSIGNED vertex with no legal color left."""
    for v in range(state.n_vertices):
        if state.colors[v] == UNASSIGNED and len(state.domains[v]) == 0:
            return True
    return False


# ===========================================================================
# SYMBOLIC PLUG-INS — propagators
# ===========================================================================

def noop_propagate(state: CSPState) -> CSPState:
    """The 'no propagation' control: return the state unchanged (only the implicit
    unit forward-check from assign_vertex is in effect)."""
    return state


def ac3_propagate(state: CSPState) -> CSPState:
    """AC-3 arc-consistency for the not-equal binary CSP, plus unit propagation.

    For not-equal constraints, an arc (x -> y) prunes value a from D(x) only if a
    is the SOLE remaining value of y (i.e. y is forced to a, so x cannot be a).
    We additionally propagate singletons: when a vertex's domain collapses to one
    value, that value is removed from neighbours (and the collapse is committed as
    an assignment). Returns a NEW state (the input is not mutated). On an
    inconsistency (empty domain) it returns a state with an empty domain so the
    caller's forward-check prunes it.
    """
    s = state.copy()
    # queue of arcs (xi, xj): revise D(xi) against xj
    from collections import deque
    queue = deque()
    for u in range(s.n_vertices):
        for w in s.adj[u]:
            queue.append((u, w))

    def _revise(xi: int, xj: int) -> bool:
        # not-equal: a in D(xi) is unsupported iff D(xj) == {a}
        removed = False
        if len(s.domains[xj]) == 1:
            (a,) = tuple(s.domains[xj])
            if a in s.domains[xi]:
                s.domains[xi].discard(a)
                removed = True
        return removed

    while queue:
        xi, xj = queue.popleft()
        if _revise(xi, xj):
            if len(s.domains[xi]) == 0:
                return s                                    # inconsistent; caller prunes
            for xk in s.adj[xi]:
                if xk != xj:
                    queue.append((xk, xi))

    # commit any newly-forced singletons into the assignment (keeps colors/n_assigned
    # consistent with the tightened domains so the verifier + var-orderers see them).
    for v in range(s.n_vertices):
        if s.colors[v] == UNASSIGNED and len(s.domains[v]) == 1:
            (a,) = tuple(s.domains[v])
            s.colors[v] = a
            s.n_assigned += 1
    return s


# ===========================================================================
# SYMBOLIC PLUG-INS — variable orderers (pick the next UNRESOLVED vertex)
# ===========================================================================

def dsatur_varorder(state: CSPState) -> int:
    """DSATUR: choose the unassigned vertex with the highest saturation degree
    (number of DISTINCT colors among its assigned neighbours), ties broken by
    higher graph degree, then lower index. Returns -1 if all assigned."""
    best = -1
    best_key = None
    for v in range(state.n_vertices):
        if state.colors[v] != UNASSIGNED:
            continue
        sat = len({state.colors[w] for w in state.adj[v] if state.colors[w] != UNASSIGNED})
        key = (sat, len(state.adj[v]), -v)        # max sat, max degree, min index
        if best_key is None or key > best_key:
            best_key = key
            best = v
    return best


def random_varorder(seed: int) -> Callable:
    """Return a seeded random variable-orderer (the 'no good heuristic' control).
    Each returned callable carries its own RNG so a run is reproducible."""
    rng = random.Random(seed)

    def _pick(state: CSPState) -> int:
        unassigned = state.unassigned_vertices()
        if not unassigned:
            return -1
        return rng.choice(unassigned)

    return _pick


# ===========================================================================
# SYMBOLIC PLUG-INS — value orderers (ordered candidate colors for v)
# ===========================================================================

def lcv_valorder(state: CSPState, v: int) -> list[int]:
    """Least-Constraining-Value: order v's legal colors so the color that removes
    the FEWEST options from unassigned neighbours comes first. Only colors in
    v's current domain are returned (forward-check + AC-3 already pruned)."""
    legal = sorted(state.domains[v])
    nbrs = [w for w in state.adj[v] if state.colors[w] == UNASSIGNED]

    def _constraint(c: int) -> int:
        # number of neighbour-domain options this color would remove
        return sum(1 for w in nbrs if c in state.domains[w])

    return sorted(legal, key=lambda c: (_constraint(c), c))


def random_valorder(seed: int) -> Callable:
    """Return a seeded random value-orderer (the arbitrary-order control)."""
    rng = random.Random(seed)

    def _order(state: CSPState, v: int) -> list[int]:
        legal = sorted(state.domains[v])
        rng.shuffle(legal)
        return legal

    return _order


# ===========================================================================
# THE PLUGGABLE DFS SKELETON (shared by ALL configs — fairness lives here)
# ===========================================================================

def backtrack_search(
    n_vertices: int,
    edges: Edges,
    k: int,
    propagate_fn: Callable[[CSPState], CSPState],
    varorder_fn: Callable[[CSPState], int],
    valorder_fn: Callable[[CSPState, int], list],
    budget: int = 100,
    seed: int = 0,
    can_certify_unsat: bool = True,
    fallback_fn: Optional[Callable[[], dict]] = None,
) -> dict:
    """Depth-first branch-and-propagate for graph k-coloring.

    Identical search logic for every config; ONLY the three plug-ins differ.

    Args:
      n_vertices, edges, k : the instance (edges 0-indexed (u,v), colors 0..k-1).
      propagate_fn(state) -> state         : tighten domains/fix vertices.
      varorder_fn(state)  -> int           : choose an UNASSIGNED vertex (-1 if none).
      valorder_fn(state,v)-> list[int]     : ordered candidate colors for v
                                             (returned colors should be in D(v)).
      budget : max DECISION-NODES expanded (a decision-node = one varorder pick +
               its value loop). On exhaustion -> status='budget', best-effort return.
      seed   : reserved for any stochastic plug-in coordination (the random plug-ins
               carry their own RNG; this is here for signature stability).
      can_certify_unsat : whether THIS propagator is allowed to certify 'unsat'.
               TRUE for SOUND symbolic propagators (AC-3, no-op): a tree exhausted
               with no trusted commits is a real unsat certificate. FALSE for neural
               propagators that AUTO-COMMIT high-confidence vertices: a confidently
               WRONG commit can manufacture a conflict (empty domain / improper
               partial) at/near the root and dead-end the tree with nothing on the
               decision stack to backtrack — a FALSE unsat on a SOLVABLE instance
               (the G1-failure fairness bug). When FALSE, any would-be 'unsat'
               (root-dead OR exhausted tree) is NEVER certified; instead `fallback_fn`
               is invoked (FAIRNESS + SOUNDNESS, see below) and its decisions/
               backtracks are MERGED with the wasted ones spent here, so the failed
               neural propagation is counted HONESTLY.
      fallback_fn : () -> dict, a complete-search fallback to run when this config is
               NOT allowed to certify unsat and the neural search dead-ends. Conven-
               tionally a B1-style full enumeration (noop propagate, no trusted neural
               commits). Its result REPLACES the would-be unsat; the wasted neural
               decisions/backtracks are added on top. If None, the would-be unsat is
               downgraded to status='budget' (never a false certificate). Soundness is
               unchanged: the fallback still only returns verifier-proper solutions.

    Returns dict:
      status        : 'solved' | 'budget' | 'unsat'
      assignment    : list[int] length n (-1 = unassigned) — the SOLUTION if solved,
                      else the best-effort most-complete proper partial seen.
      decisions     : int  decision-nodes expanded
      backtracks    : int  failed-branch unwinds
      best_partial  : list[int]  the most-complete proper partial encountered
      n_assigned_best : int  how many vertices the best_partial assigns

    Soundness: SOLVED is returned ONLY when is_complete_proper passes on the exact
    verifier. A wrong propagation/auto-commit that violates an edge is caught by the
    properness check before extension and is backtracked. 'unsat' is certified ONLY
    by exhaustive symbolic enumeration (can_certify_unsat=True) or AC-3 proving an
    empty domain — neural auto-commits are advisory/revertible and can only ACCELERATE
    or fall back, never certify non-colorability.
    """
    edges_n = normalize_edges(edges, n_vertices)
    root = make_initial_state(n_vertices, edges_n, k)
    root = propagate_fn(root)                              # propagate at the root too

    counters = {"decisions": 0, "backtracks": 0}
    best = {"colors": list(root.colors), "n_assigned": _count_assigned(root.colors)}

    def _track_best(state: CSPState):
        na = state.n_assigned
        if na > best["n_assigned"] and is_proper_partial(state.colors, edges_n, n_vertices):
            best["n_assigned"] = na
            best["colors"] = list(state.colors)

    def _resolve_deadend(reason: str) -> dict:
        """Build the return dict for a tree that dead-ended with NO solution found
        (root conflict OR exhausted tree, budget NOT hit). FAIR + SOUND fallback:

        * can_certify_unsat=True (symbolic): this IS a real unsat certificate.
        * can_certify_unsat=False (neural): a neural auto-commit may have manufactured
          the conflict on a SOLVABLE instance — we MUST NOT certify unsat. Re-run via
          fallback_fn (complete enumeration, auto-commit disabled). The fallback's
          status/assignment REPLACE ours; the wasted neural decisions/backtracks are
          ADDED so the failed propagation is counted honestly (Fix 2 forward cost is
          tracked separately by the deduce_fn wrapper). If no fallback is supplied,
          downgrade to 'budget' — never a false certificate.
        """
        if can_certify_unsat:
            return {
                "status": "unsat",
                "assignment": best["colors"],
                "decisions": counters["decisions"],
                "backtracks": counters["backtracks"],
                "best_partial": best["colors"],
                "n_assigned_best": best["n_assigned"],
            }
        if fallback_fn is not None:
            fb = fallback_fn()
            # merge: the neural search's wasted work still counts.
            fb = dict(fb)
            fb["decisions"] = fb.get("decisions", 0) + counters["decisions"]
            fb["backtracks"] = fb.get("backtracks", 0) + counters["backtracks"]
            fb["fellback"] = True                          # provenance flag
            fb["fallback_reason"] = reason
            return fb
        # no fallback available: refuse to certify unsat (NEVER a false certificate).
        return {
            "status": "budget",
            "assignment": best["colors"],
            "decisions": counters["decisions"],
            "backtracks": counters["backtracks"],
            "best_partial": best["colors"],
            "n_assigned_best": best["n_assigned"],
        }

    # Quick root sanity: an immediate empty domain or improper root => dead-end.
    # For symbolic propagators this is a genuine unsat; for neural propagators a
    # confidently-wrong root auto-commit could have manufactured it -> fall back.
    if has_empty_domain(root) or not is_proper_partial(root.colors, edges_n, n_vertices):
        return _resolve_deadend("root_conflict")

    _track_best(root)
    solution_box = {"colors": None}

    def _dfs(state: CSPState) -> bool:
        # success: complete + proper (exact verifier).
        if is_complete_proper(state.colors, edges_n, n_vertices, k):
            solution_box["colors"] = list(state.colors)
            return True

        if counters["decisions"] >= budget:
            return False                                  # budget exhausted; unwind

        v = varorder_fn(state)
        if v == UNASSIGNED or v < 0:
            # nothing to branch on but not complete => dead (shouldn't normally hit).
            return False

        counters["decisions"] += 1
        candidates = valorder_fn(state, v)
        for c in candidates:
            if c not in state.domains[v]:
                continue                                  # respect domain pruning
            child = assign_vertex(state, v, c)
            # forward-check + properness prune BEFORE recursing (soundness backstop).
            if not is_proper_partial(child.colors, edges_n, n_vertices):
                counters["backtracks"] += 1
                continue
            child = propagate_fn(child)
            if has_empty_domain(child):
                counters["backtracks"] += 1
                continue
            if not is_proper_partial(child.colors, edges_n, n_vertices):
                counters["backtracks"] += 1
                continue
            _track_best(child)
            if _dfs(child):
                return True
            counters["backtracks"] += 1
            if counters["decisions"] >= budget:
                return False
        return False

    solved = _dfs(root)

    if solved:
        return {
            "status": "solved",
            "assignment": solution_box["colors"],
            "decisions": counters["decisions"],
            "backtracks": counters["backtracks"],
            "best_partial": solution_box["colors"],
            "n_assigned_best": n_vertices,
        }

    if counters["decisions"] >= budget:
        return {
            "status": "budget",
            "assignment": best["colors"],
            "decisions": counters["decisions"],
            "backtracks": counters["backtracks"],
            "best_partial": best["colors"],
            "n_assigned_best": best["n_assigned"],
        }

    # tree exhausted under budget with no solution: dead-end -> resolve (cert or fall back).
    return _resolve_deadend("tree_exhausted")


def _count_assigned(colors: list) -> int:
    return sum(1 for c in colors if c != UNASSIGNED)


# ===========================================================================
# THE SYMBOLIC CEILING SOLVER (B3) = {AC-3 propagate, DSATUR varorder, LCV valorder}
# ===========================================================================

def solve_symbolic(
    n_vertices: int,
    edges: Edges,
    k: int,
    budget: int = 100000,
    seed: int = 0,
) -> dict:
    """The complete symbolic solver = the B3 CEILING / no-neural control.
    Plugs {ac3_propagate, dsatur_varorder, lcv_valorder} into backtrack_search.

    With a large budget this is a complete solver: status='solved' returns a proper
    k-coloring; status='unsat' is a CERTIFICATE that the graph is not k-colorable
    (the whole tree was exhausted without exceeding budget). status='budget' means
    the budget cut search short (increase budget for a definitive answer)."""
    return backtrack_search(
        n_vertices=n_vertices,
        edges=edges,
        k=k,
        propagate_fn=ac3_propagate,
        varorder_fn=dsatur_varorder,
        valorder_fn=lcv_valorder,
        budget=budget,
        seed=seed,
    )


# ===========================================================================
# PUBLIC INTERFACE (the injection contract for scripts/search_coloring.py)
# ===========================================================================
# Verifier (the arbiter):
#   is_proper_partial(assignment, edges, n_vertices=None) -> bool
#   is_complete_proper(assignment, edges, n_vertices, k=None) -> bool
#
# Skeleton (shared by ALL configs):
#   backtrack_search(n_vertices, edges, k, propagate_fn, varorder_fn,
#                    valorder_fn, budget=100, seed=0,
#                    can_certify_unsat=True, fallback_fn=None) -> dict
#       returns {status, assignment, decisions, backtracks, best_partial,
#                n_assigned_best}  (+ fellback/fallback_reason if a neural config
#                dead-ended and re-ran the complete fallback).
#       can_certify_unsat: SOUND symbolic propagators (AC-3, no-op) => True (a tree
#         exhausted with no trusted commits is a real unsat certificate). NEURAL
#         propagators that auto-commit => False: a confidently-WRONG commit can dead-
#         end a SOLVABLE instance, so 'unsat' is NEVER certified; instead fallback_fn
#         (a complete B1-style enumeration) runs and its decisions/backtracks are
#         merged with the wasted neural ones. This is the FAIRNESS fix: neural prop
#         can only ACCELERATE or gracefully degrade to complete search, never produce
#         a false unsat that AC-3 would not.
#
# State helpers (the plug-in API surface — neural plug-ins build on these):
#   make_initial_state(n_vertices, edges, k) -> CSPState
#   assign_vertex(state, v, c) -> CSPState
#   has_empty_domain(state) -> bool
#   CSPState fields: n_vertices, k, adj, colors(list, -1=unassigned), domains
#       (list[set]), n_assigned, meta(dict for plug-in scratch); methods .copy(),
#       .assigned()->dict, .unassigned_vertices()->list[int].
#
# Symbolic plug-ins (the baselines):
#   propagate: ac3_propagate(state) -> CSPState ; noop_propagate(state) -> CSPState
#   varorder : dsatur_varorder(state) -> int ; random_varorder(seed) -> callable
#   valorder : lcv_valorder(state, v) -> list[int] ; random_valorder(seed) -> callable
#
# Ceiling:
#   solve_symbolic(n_vertices, edges, k, budget=100000, seed=0) -> dict
#
# Bridge:
#   edges_from_membership(membership_np, latent_type_np, b, ltype_edge=0) -> list
#   normalize_edges(edges, n_vertices) -> list[(u,v)]
#   build_adjacency(edges, n_vertices) -> list[set[int]]
#
# NEURAL PLUG-IN CONTRACT (implemented in scripts/search_coloring.py):
#   - neural propagate_fn(state): CLAMP state.assigned() into the deducer input
#     (input_cells + value_domain_mask), run ONE forward, read per-vertex softmax;
#     AUTO-COMMIT any unassigned vertex whose top-color prob > threshold AND is
#     constraint-consistent (in domains[v]); return assign_vertex(...)-chained state.
#     MUST stay sound: never auto-commit a color absent from domains[v]; the
#     verifier is the backstop. Stash beliefs/entropy in state.meta for varorder.
#     ADVISORY/REVERTIBLE: auto-commits accelerate but are NEVER allowed to certify
#     unsat -> the neural config calls backtrack_search with can_certify_unsat=False
#     and a fallback_fn that re-runs COMPLETE search (noop propagate) on a dead-end.
#   - neural varorder_fn(state): return the highest-ENTROPY UNASSIGNED vertex (G2),
#     reading entropy the propagator stashed in state.meta (recompute if absent).
#   - neural valorder_fn(state, v): deducer policy descending, filtered to D(v).
# ===========================================================================


# ===========================================================================
# CPU SELFTEST (run under SELFTEST_ONLY=1)
# ===========================================================================

def _selftest() -> bool:
    import numpy as np  # noqa: F401  (proves numpy import is clean on CPU)
    all_ok = True

    def _check(name: str, cond: bool):
        nonlocal all_ok
        status = "PASS" if cond else "FAIL"
        if not cond:
            all_ok = False
        print(f"[selftest] {status}: {name}", flush=True)

    # --- TEST 1: verifier rejects an improper coloring, accepts a proper one ---
    # triangle 0-1-2 (K3): needs 3 colors.
    tri_edges = [(0, 1), (1, 2), (0, 2)]
    improper = {0: 0, 1: 0, 2: 1}                  # edge (0,1) monochromatic
    proper = {0: 0, 1: 1, 2: 2}
    _check("verifier rejects improper triangle (partial-proper)",
           is_proper_partial(improper, tri_edges, 3) is False)
    _check("verifier accepts proper triangle (partial-proper)",
           is_proper_partial(proper, tri_edges, 3) is True)
    _check("verifier rejects improper triangle (complete)",
           is_complete_proper(improper, tri_edges, 3, k=3) is False)
    _check("verifier accepts proper triangle (complete)",
           is_complete_proper(proper, tri_edges, 3, k=3) is True)
    # incomplete assignment is NOT complete-proper even though it's partial-proper.
    _check("verifier: incomplete is not complete_proper",
           is_complete_proper({0: 0, 1: 1}, tri_edges, 3, k=3) is False)
    # partial-proper allows unassigned endpoints.
    _check("verifier: partial with unassigned endpoint is proper",
           is_proper_partial({0: 0}, tri_edges, 3) is True)
    # list form parity.
    _check("verifier: list form matches dict form (proper)",
           is_complete_proper([0, 1, 2], tri_edges, 3, k=3) is True)
    _check("verifier: list form matches dict form (improper)",
           is_proper_partial([0, 0, 1], tri_edges, 3) is False)

    # --- TEST 2 (adversarial soundness): feed an improper coloring as if 'solved' ---
    # The skeleton must NEVER call it solved. We verify the arbiter directly AND that
    # a search returning 'solved' always yields a verifiably proper coloring.
    adversarial_improper = {0: 1, 1: 1, 2: 2}      # (0,1) monochromatic
    _check("adversarial: improper coloring rejected by arbiter",
           is_complete_proper(adversarial_improper, tri_edges, 3, k=3) is False)

    # --- TEST 3: solve_symbolic solves a known 3-colorable instance ---
    # 5-cycle C5: 3-colorable (odd cycle needs 3 colors). Use 2 colors -> unsat;
    # 3 colors -> solvable.
    c5_edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]
    res_c5 = solve_symbolic(5, c5_edges, k=3, budget=100000)
    _check("solve_symbolic: C5 3-colorable -> solved",
           res_c5["status"] == "solved")
    _check("solve_symbolic: C5 solution verifies proper",
           is_complete_proper(res_c5["assignment"], c5_edges, 5, k=3) is True)

    # --- TEST 4: solve_symbolic reports unsat on a non-3-colorable instance (K4) ---
    k4_edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    res_k4 = solve_symbolic(4, k4_edges, k=3, budget=100000)
    _check("solve_symbolic: K4 NOT 3-colorable -> unsat",
           res_k4["status"] == "unsat")
    _check("solve_symbolic: K4 IS 4-colorable -> solved",
           solve_symbolic(4, k4_edges, k=4, budget=100000)["status"] == "solved")
    # C5 with 2 colors must be unsat (odd cycle).
    _check("solve_symbolic: C5 2-coloring -> unsat",
           solve_symbolic(5, c5_edges, k=2, budget=100000)["status"] == "unsat")

    # --- TEST 5: the skeleton respects the budget ---
    # Build a harder graph and run with a tiny budget; decisions must not exceed it.
    rng = random.Random(7)
    n_big = 25
    big_edges = []
    for u in range(n_big):
        for w in range(u + 1, n_big):
            if rng.random() < 0.45:
                big_edges.append((u, w))
    tight = 3
    res_budget = backtrack_search(
        n_big, big_edges, k=tight,
        propagate_fn=noop_propagate,
        varorder_fn=dsatur_varorder,
        valorder_fn=lcv_valorder,
        budget=5,
    )
    _check("skeleton: decisions <= budget",
           res_budget["decisions"] <= 5)
    _check("skeleton: budget exhaustion status is 'budget' (or solved/unsat fast)",
           res_budget["status"] in ("budget", "solved", "unsat"))
    # best-effort partial must itself be proper (never returns an improper partial).
    _check("skeleton: best_partial is proper",
           is_proper_partial(res_budget["best_partial"], big_edges, n_big) is True)

    # --- TEST 6: AC-3 propagation collapses the tree vs no-op (fewer decisions) ---
    # On C5/3 both solve; with a real graph AC-3 should not INCREASE decisions.
    noop_res = backtrack_search(5, c5_edges, 3, noop_propagate, dsatur_varorder,
                                lcv_valorder, budget=100000)
    ac3_res = backtrack_search(5, c5_edges, 3, ac3_propagate, dsatur_varorder,
                               lcv_valorder, budget=100000)
    _check("ac3: both solve C5", noop_res["status"] == "solved" and ac3_res["status"] == "solved")
    _check("ac3: solutions verify",
           is_complete_proper(noop_res["assignment"], c5_edges, 5, k=3) and
           is_complete_proper(ac3_res["assignment"], c5_edges, 5, k=3))

    # --- TEST 7: random plug-ins drive the skeleton + are reproducible ---
    rvar = random_varorder(123)
    rval = random_valorder(123)
    res_rand = backtrack_search(5, c5_edges, 3, noop_propagate, rvar, rval, budget=100000)
    _check("random plug-ins: solve C5", res_rand["status"] == "solved")
    _check("random plug-ins: solution verifies",
           is_complete_proper(res_rand["assignment"], c5_edges, 5, k=3) is True)
    # same seed -> same first pick (reproducibility of the orderer)
    s0 = make_initial_state(5, c5_edges, 3)
    a = random_varorder(999)(s0)
    b = random_varorder(999)(s0)
    _check("random plug-ins: same seed -> same pick", a == b)

    # --- TEST 8: MOCK neural plug-ins drive the skeleton end-to-end (injection) ---
    # This proves the injection contract WITHOUT a GPU/ckpt: a stand-in propagator
    # returns synthetic per-vertex softmax and auto-commits confident vertices; a
    # mock entropy-varorder reads stashed beliefs; a mock policy-valorder orders by
    # synthetic logits. SOUNDNESS: auto-commit filtered through domains[v].
    def _mock_beliefs(state: CSPState):
        # synthetic, deterministic softmax: prefer color (v % k) strongly.
        beliefs = {}
        for v in state.unassigned_vertices():
            probs = [0.1] * state.k
            top = v % state.k
            probs[top] = 1.0 - 0.1 * (state.k - 1)
            beliefs[v] = probs
        return beliefs

    _C5 = normalize_edges(c5_edges, 5)

    def mock_propagate(state: CSPState) -> CSPState:
        s = state.copy()
        beliefs = _mock_beliefs(s)
        # auto-commit any unassigned vertex with top-prob > 0.85 that is consistent.
        # SOUNDNESS: only commit if the color is in D(v) AND keeps the partial proper.
        for v in list(beliefs.keys()):
            if s.colors[v] != UNASSIGNED:
                continue
            probs = beliefs[v]
            top = max(range(s.k), key=lambda c: probs[c])
            if probs[top] > 0.85 and top in s.domains[v]:
                cand = assign_vertex(s, v, top)
                if is_proper_partial(cand.colors, _C5, s.n_vertices):
                    s = cand
        s.meta["beliefs"] = _mock_beliefs(s)
        return s

    def mock_entropy_varorder(state: CSPState) -> int:
        import math
        beliefs = state.meta.get("beliefs") or _mock_beliefs(state)
        best_v, best_ent = -1, -1.0
        for v in state.unassigned_vertices():
            p = beliefs.get(v, [1.0 / state.k] * state.k)
            ent = -sum(pi * math.log(pi + 1e-12) for pi in p)
            if ent > best_ent:
                best_ent, best_v = ent, v
        return best_v

    def mock_policy_valorder(state: CSPState, v: int) -> list[int]:
        beliefs = state.meta.get("beliefs") or _mock_beliefs(state)
        p = beliefs.get(v, [1.0 / state.k] * state.k)
        order = sorted(range(state.k), key=lambda c: -p[c])
        return [c for c in order if c in state.domains[v]]

    res_mock = backtrack_search(5, c5_edges, 3, mock_propagate,
                                mock_entropy_varorder, mock_policy_valorder,
                                budget=100000)
    _check("mock neural plug-ins: drive skeleton to solve C5",
           res_mock["status"] == "solved")
    _check("mock neural plug-ins: solution verifies proper",
           is_complete_proper(res_mock["assignment"], c5_edges, 5, k=3) is True)

    # mock propagator must NEVER produce an improper 'solved' (soundness under bad
    # auto-commit): force a propagator that auto-commits ALL vertices to color 0
    # (deliberately improper) and confirm the skeleton does NOT report it solved.
    def bad_propagate(state: CSPState) -> CSPState:
        s = state.copy()
        for v in s.unassigned_vertices():
            if 0 in s.domains[v]:
                cand = assign_vertex(s, v, 0)
                # bad_propagate intentionally SKIPS the properness guard to test the
                # skeleton's own backstop.
                s = cand
        return s

    res_bad = backtrack_search(5, c5_edges, 3, bad_propagate,
                               dsatur_varorder, lcv_valorder, budget=1000)
    # all-color-0 on C5 is improper => skeleton must NOT report solved with it.
    if res_bad["status"] == "solved":
        bad_ok = is_complete_proper(res_bad["assignment"], c5_edges, 5, k=3)
    else:
        bad_ok = True   # not reported solved => sound
    _check("soundness: bad all-0 propagator never yields improper 'solved'", bad_ok)

    # --- TEST 9 (FIX 1 FAIRNESS): a confidently-WRONG neural-style propagator on a
    # SOLVABLE instance must NOT return a false 'unsat'. It must fall back to complete
    # search and SOLVE. The bug being guarded: an auto-commit at/near the root creates
    # a conflict before any real decision -> dead-end with nothing to backtrack ->
    # false 'unsat'. With can_certify_unsat=False + a complete fallback_fn, the neural
    # config gracefully degrades to B1-style enumeration and solves.
    # C5/3 is solvable. The wrong propagator slams vertex 0 AND vertex 1 to color 0
    # (a monochromatic edge (0,1)), manufacturing an improper-root conflict.
    _C5 = normalize_edges(c5_edges, 5)

    def wrong_root_commit_propagate(state: CSPState) -> CSPState:
        """Confidently-wrong: at the root, force the (0,1) edge monochromatic. This
        skips the properness guard ON PURPOSE to simulate a deducer that auto-commits
        a conflicting pair high-confidence (the G1 failure mode)."""
        if state.n_assigned == 0:
            s = state.copy()
            for v in (0, 1):
                s.colors[v] = 0
                s.domains[v] = {0}
            s.n_assigned = 2
            return s
        return state

    def _complete_fallback():
        # complete search, NO trusted neural commits (auto-commit disabled),
        # allowed to certify unsat. This is the B1-style graceful degradation.
        return backtrack_search(
            5, c5_edges, 3,
            propagate_fn=noop_propagate,
            varorder_fn=dsatur_varorder,
            valorder_fn=lcv_valorder,
            budget=100000,
            can_certify_unsat=True,
        )

    # WITHOUT the fix (can_certify_unsat=True, no fallback) -> FALSE unsat.
    res_falseunsat = backtrack_search(
        5, c5_edges, 3, wrong_root_commit_propagate,
        dsatur_varorder, lcv_valorder, budget=100000,
        can_certify_unsat=True, fallback_fn=None)
    _check("fix1: WITHOUT guard, wrong root-commit yields (the bug) false 'unsat'",
           res_falseunsat["status"] == "unsat")

    # WITH the fix (can_certify_unsat=False + complete fallback) -> SOLVED via fallback.
    res_fixed = backtrack_search(
        5, c5_edges, 3, wrong_root_commit_propagate,
        dsatur_varorder, lcv_valorder, budget=100000,
        can_certify_unsat=False, fallback_fn=_complete_fallback)
    _check("fix1: WITH guard, wrong root-commit FALLS BACK and SOLVES (no false unsat)",
           res_fixed["status"] == "solved")
    _check("fix1: fallback solution verifies proper",
           is_complete_proper(res_fixed["assignment"], _C5, 5, k=3) is True)
    _check("fix1: fallback is flagged as fellback (provenance)",
           res_fixed.get("fellback") is True)

    # A neural config WITHOUT a fallback must still NEVER certify unsat -> 'budget'.
    res_nofallback = backtrack_search(
        5, c5_edges, 3, wrong_root_commit_propagate,
        dsatur_varorder, lcv_valorder, budget=100000,
        can_certify_unsat=False, fallback_fn=None)
    _check("fix1: neural config with no fallback downgrades unsat -> 'budget' (never false cert)",
           res_nofallback["status"] == "budget")

    # The fallback must NOT understate cost: merged decisions >= the fallback's own.
    fb_alone = _complete_fallback()
    _check("fix1: fallback merges wasted decisions honestly (merged >= fallback-alone)",
           res_fixed["decisions"] >= fb_alone["decisions"])

    print(f"[selftest] {'ALL PASS' if all_ok else 'SOME FAILED'}", flush=True)
    return all_ok


def _ast_parse_ok() -> bool:
    with open(os.path.abspath(__file__)) as f:
        src = f.read()
    try:
        ast.parse(src)
        return True
    except SyntaxError as e:
        print(f"[ast.parse] FAILED: {e}", flush=True)
        return False


if __name__ == "__main__":
    parse_ok = _ast_parse_ok()
    print(f"[ast.parse] ok={parse_ok}", flush=True)
    if not parse_ok:
        sys.exit(1)
    if os.environ.get("SELFTEST_ONLY", "0") == "1":
        ok = _selftest()
        sys.exit(0 if ok else 1)
