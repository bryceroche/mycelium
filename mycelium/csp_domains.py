"""csp_domains.py — the DOMAIN CONTENT (the ONLY file with domain knowledge).

This is the single place a domain's constraint semantics live (spec §2.2, §2.5, §6.1):
  * the per-FACTOR-TYPE predicate(s) — the entire domain surface, and
  * a thin BRIDGE that reads (membership, latent_type, value_domain_mask, params) out
    of the engine's tensors into a general csp_core.Problem.

Phase 0 ships COLORING ONLY. KenKen and circuit predicates/bridges are present as
DOCUMENTED STUBS / TODOs per the spec's leak table (§5) — the registry entry shapes
are sketched so Phase 2/4 is a small table addition, but they raise NotImplementedError
until their grounding (build_kenken_data.cage_ok / circuit_data._eval_gate) and the
all-different specialized propagator (L-ALLDIFF) are wired.

CONVENTIONS (must match the coloring data + the legacy csp_search module):
  * Values used inside the core are 0-indexed (a coloring color is 0..k-1).
  * The engine's membership/gold encoding uses color+1 (0 = pad/unknown); the bridge
    consumes membership (2-ones rows for edges) and value_domain_mask; the +1/-1 offset
    is handled at the deduce_fn boundary in the driver, not here.
"""

from __future__ import annotations

import ast
import os
import sys
from typing import Optional

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mycelium.csp_core import (
    UNASSIGNED,
    Consistency,
    Factor,
    Problem,
)
from mycelium.csp_registry import new_registry, register


# ===========================================================================
# COLORING — the not-equal relation (the ONLY domain shipped in Phase 0)
# ===========================================================================
# Factor-type id for a coloring edge. The coloring loader emits latent_type==0 for a
# real edge row and a different value for padding (the legacy bridge's ltype_edge=0).
LTYPE_EDGE = 0


def not_equal_pred(_ftype, _params, member_values) -> Consistency:
    """Coloring's binary not-equal predicate: scope=(u, v), params=None.

    Three-valued (the L-MONO-respecting contract): a hole leaves the pair extendable
    (UNVIOLATED); two equal assigned values are dead (VIOLATED); two distinct assigned
    values satisfy the factor (SAT). This IS the `u != v` the coloring verifier encodes
    — the single source of constraint truth, exposed as a registry predicate.
    """
    a, b = member_values
    if a == UNASSIGNED or b == UNASSIGNED:
        return Consistency.UNVIOLATED
    return Consistency.SAT if a != b else Consistency.VIOLATED


def coloring_registry() -> dict:
    """A registry holding ONLY the coloring not-equal type (arity 2).

    `check_alphabet` makes registration ENFORCE the L-MONO hole-monotonicity contract
    (not_equal is monotone for any alphabet; (0,1,2) is a representative sample) -- so the
    partial-soundness gate can never be wired to a non-monotone predicate unnoticed."""
    reg = new_registry()
    register(reg, LTYPE_EDGE, not_equal_pred, name="not_equal", arity_hint=2,
             check_alphabet=(0, 1, 2))
    return reg


def problem_from_coloring(n_vertices: int, edges, k: int,
                          registry: Optional[dict] = None) -> Problem:
    """Bridge: build a general Problem from a coloring instance (n_vertices, edges, k).

    Every real vertex starts with the full k-color alphabet (coloring is the degenerate
    "every variable starts at the full alphabet" case). Each edge becomes a not-equal
    Factor over its 2-vertex scope. var_factors is the GENERAL incidence replacing the
    coloring-only adjacency.

    `edges` is an iterable of (u, v) pairs (0-indexed, u != v). Duplicates / orientation
    are de-duplicated here so the factor list matches the legacy adjacency exactly.
    """
    reg = registry if registry is not None else coloring_registry()
    # De-duplicate + canonicalize edges (u < v), reject self-loops / out-of-range.
    norm = set()
    for e in edges:
        u, v = int(e[0]), int(e[1])
        if u == v:
            raise ValueError(f"self-loop edge ({u},{v}) is not a valid constraint")
        if not (0 <= u < n_vertices) or not (0 <= v < n_vertices):
            raise ValueError(
                f"edge ({u},{v}) endpoint out of range for n_vertices={n_vertices}")
        norm.add((u, v) if u < v else (v, u))

    factors = []
    var_factors = [[] for _ in range(n_vertices)]
    for (u, v) in sorted(norm):
        fi = len(factors)
        factors.append(Factor(ftype=LTYPE_EDGE, scope=(u, v), params=None))
        var_factors[u].append(fi)
        var_factors[v].append(fi)

    domains0 = [set(range(k)) for _ in range(n_vertices)]
    return Problem(
        n_vars=n_vertices,
        domains0=domains0,
        factors=factors,
        var_factors=var_factors,
        registry=reg,
    )


def edges_from_membership(membership_np, latent_type_np, b: int,
                          ltype_edge: int = LTYPE_EDGE):
    """Recover the 0-indexed edge list for batch element b from the engine's
    (B, n_edges_max, s_max) membership + (B, n_edges_max) latent_type tensors.

    A real edge row (latent_type == ltype_edge) has EXACTLY TWO 1s at columns u, v.
    Padding rows are skipped. This is the not-equal case of the general
    factorgraph_from_membership (other domains pass their own factor_params / scopes).
    """
    import numpy as np
    mem = np.asarray(membership_np)
    lt = np.asarray(latent_type_np)
    edges = []
    n_edges = mem.shape[1]
    for e in range(n_edges):
        if int(lt[b, e]) != ltype_edge:
            continue
        idx = np.where(mem[b, e] > 0.0)[0]
        if len(idx) == 2:
            u, v = int(idx[0]), int(idx[1])
            edges.append((u, v) if u < v else (v, u))
    return edges


def problem_from_coloring_membership(membership_np, latent_type_np,
                                     value_domain_mask_np, cell_valid_np, b: int,
                                     k: int, registry: Optional[dict] = None) -> Problem:
    """Bridge straight from the engine's tensors for batch element b (the path the GPU
    driver uses). Recovers the real-vertex count from cell_valid, the edges from
    membership/latent_type, and per-vertex domains from value_domain_mask (defaulting
    to the full k-alphabet for any real vertex with no explicit mask)."""
    import numpy as np
    cv = np.asarray(cell_valid_np)
    n_vertices = int((cv[b] > 0.5).sum()) if cv.ndim == 2 else int((cv > 0.5).sum())
    edges = edges_from_membership(membership_np, latent_type_np, b)
    prob = problem_from_coloring(n_vertices, edges, k, registry=registry)
    # If a value_domain_mask is given, intersect the initial domains with it (givens).
    if value_domain_mask_np is not None:
        vdm = np.asarray(value_domain_mask_np)
        row = vdm[b] if vdm.ndim == 3 else vdm
        domains0 = []
        for v in range(n_vertices):
            legal = {c for c in range(k) if row[v, c] > 0.5}
            domains0.append(legal if legal else set(range(k)))
        prob = Problem(
            n_vars=prob.n_vars, domains0=domains0, factors=prob.factors,
            var_factors=prob.var_factors, registry=prob.registry,
        )
    return prob


# ===========================================================================
# COLORING — thin wrappers preserving the legacy verifier surface
# ===========================================================================
# These keep scripts/legacy code that asks "is this coloring proper?" working through
# the general core, so the coloring selftest reads identically. They are coloring-named
# CONVENIENCE shims (this is the coloring-content file); the GENERAL arbiters are
# csp_core.verify_complete / is_consistent_partial.

def is_proper_partial_coloring(assignment, n_vertices: int, edges, k: int) -> bool:
    """True iff no edge is monochromatic AMONG ASSIGNED vertices (general partial gate,
    specialized to coloring). `assignment` is a list[int] (len n_vertices, -1=unassigned)
    or a dict {v: color}."""
    from mycelium.csp_core import is_consistent_partial, make_initial_state
    prob = problem_from_coloring(n_vertices, edges, k)
    st = make_initial_state(prob)
    st.values = _as_value_list(assignment, n_vertices)
    st.domains = [({st.values[v]} if st.values[v] != UNASSIGNED else set(range(k)))
                  for v in range(n_vertices)]
    st.n_assigned = sum(1 for c in st.values if c != UNASSIGNED)
    return is_consistent_partial(st)


def is_complete_proper_coloring(assignment, n_vertices: int, edges, k: int) -> bool:
    """True iff every vertex is assigned a color in [0,k) AND no edge is monochromatic
    (general success arbiter, specialized to coloring)."""
    from mycelium.csp_core import verify_complete, make_initial_state
    vals = _as_value_list(assignment, n_vertices)
    if any(c == UNASSIGNED for c in vals):
        return False
    if any(not (0 <= c < k) for c in vals):
        return False
    prob = problem_from_coloring(n_vertices, edges, k)
    st = make_initial_state(prob)
    st.values = vals
    st.domains = [{vals[v]} for v in range(n_vertices)]
    st.n_assigned = n_vertices
    return verify_complete(st)


def _as_value_list(assignment, n_vertices: int) -> list:
    if isinstance(assignment, dict):
        out = [UNASSIGNED] * n_vertices
        for v, c in assignment.items():
            if 0 <= int(v) < n_vertices and int(c) != UNASSIGNED:
                out[int(v)] = int(c)
        return out
    return [int(c) for c in assignment[:n_vertices]] + \
           [UNASSIGNED] * max(0, n_vertices - len(assignment))


# ===========================================================================
# KENKEN — DOCUMENTED STUB (Phase 2, the FIRST generality test; L-ALLDIFF/L-PARAM)
# ===========================================================================
# Predicates REUSE build_kenken_data.cage_ok (single source of constraint truth).
# Shipping requires: (1) the all-different specialized_propagator (L-ALLDIFF: a 7^7
# n-ary all-diff blows the arity_cap -> forward-check only until a Hall-interval /
# value-occurrence propagator is registered), and (2) a bridge that pulls the EXACT
# integer cage target via cage_op/cage_target side-tensors (L-PARAM), NOT the lossy
# inlet bucket. Left as TODO per the Phase-0 leak table.

def all_diff_pred(_ftype, _params, member_values) -> Consistency:
    """KenKen row/col / Sudoku unit: n-ary all-different, no params (spec §2.2). SAT
    when complete with all-distinct; VIOLATED on any duplicate among assigned; else
    UNVIOLATED. Provided (correct + monotone) so Phase 2 is a registration, but the
    row/col n-ary instance needs the L-ALLDIFF specialized propagator for GAC strength."""
    seen = [v for v in member_values if v != UNASSIGNED]
    if len(seen) != len(set(seen)):
        return Consistency.VIOLATED
    return Consistency.SAT if UNASSIGNED not in member_values else Consistency.UNVIOLATED


def cage_pred(_ftype, params, member_values) -> Consistency:  # noqa: D401
    """KenKen cage: params=(op_id, target_INT); REUSE build_kenken_data.cage_ok.
    STUB — wiring the op_id<->op-name map + the bridge's exact-target plumbing is
    Phase 2 (L-PARAM)."""
    raise NotImplementedError(
        "KenKen cage_pred is a Phase-2 stub (reuse build_kenken_data.cage_ok + "
        "exact-target bridge; see docs/general_factor_graph_search.md §6.2 Phase 2).")


def problem_from_kenken(*args, **kwargs) -> Problem:
    """STUB bridge for KenKen (Phase 2). Will read cages/ops/targets from the engine's
    side-tensors (membership for cage scope; cage_op/cage_target for exact params) and
    register {ROW: all_diff, COL: all_diff, CAGE: cage} + the all-diff specialized
    propagator (L-ALLDIFF)."""
    raise NotImplementedError(
        "problem_from_kenken is a Phase-2 stub (see the leak table L-ALLDIFF/L-PARAM).")


# ===========================================================================
# CIRCUIT — DOCUMENTED STUB (Phase 4, the DAG testbed; L-ASYM)
# ===========================================================================
# Predicate REUSES circuit_data._eval_gate; scope is ORDERED (element 0 = output) and
# the bridge supplies ordered scope from the data generator's side-tensors, NOT from a
# symmetric membership clique (L-ASYM). Left as TODO per the Phase-0 leak table.

def gate_pred(_ftype, params, member_values) -> Consistency:  # noqa: D401
    """Circuit gate: scope=(out, *operands), params=gate_type; REUSE
    circuit_data._eval_gate. STUB — Phase 4 (the hierarchical/DAG testbed; L-ASYM
    resolved at the bridge: ordered scope, element 0 = output)."""
    raise NotImplementedError(
        "circuit gate_pred is a Phase-4 stub (reuse circuit_data._eval_gate + ordered "
        "scope bridge; see docs/general_factor_graph_search.md §6.2 Phase 4).")


def problem_from_circuit(*args, **kwargs) -> Problem:
    """STUB bridge for circuits (Phase 4). Will register gate types and build ordered
    scopes (element 0 = output) from the generator's side-tensors (L-ASYM)."""
    raise NotImplementedError(
        "problem_from_circuit is a Phase-4 stub (see the leak table L-ASYM).")


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
    sys.exit(0 if parse_ok else 1)
