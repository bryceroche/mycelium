"""csp_domains.py — the DOMAIN CONTENT (the ONLY file with domain knowledge).

This is the single place a domain's constraint semantics live (spec §2.2, §2.5, §6.1):
  * the per-FACTOR-TYPE predicate(s) — the entire domain surface, and
  * a thin BRIDGE that reads (membership, latent_type, value_domain_mask, params) out
    of the engine's tensors into a general csp_core.Problem.

COLORING (Phase 0) and KenKen (Phase 2) are SHIPPED + VALIDATED — KenKen via cage_pred +
problem_from_kenken + the L-ALLDIFF all-different specialized propagator, added here with
ZERO csp_core/csp_registry edits (the one-trick-pony guarantee). The CIRCUIT predicate/
bridge (Phase 4) remains a DOCUMENTED STUB per the spec's leak table (§5, L-ASYM): its
registry-entry shape is sketched but it raises NotImplementedError until the ordered-scope
bridge (gate output vs operands) and circuit_data._eval_gate grounding are wired.

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

# KenKen reuses build_kenken_data.cage_ok as the SINGLE source of cage constraint truth
# (Phase 2 must not reinvent the verifier). scripts/ is added to sys.path above for the
# package-relative + script-relative import to both resolve.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
from build_kenken_data import cage_ok  # noqa: E402


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
# KENKEN — Phase 2, the FIRST generality test (L-ALLDIFF / L-PARAM / L-ASYM)
# ===========================================================================
# A domain STRUCTURALLY UNLIKE coloring: param-carrying n-ary arithmetic cages
# (op=target factors) PLUS row/col ALL-DIFFERENT over up to 7 cells. Shipped with
# ZERO csp_core / csp_registry edits — the ONLY extension points used are the
# predicate seam + the specialized_propagator slot that already exists in csp_core.
#
#   * cage_pred (L-PARAM): params=(op_id, target_INT) — the EXACT integer target,
#     never the lossy verification-inlet bucket. REUSES build_kenken_data.cage_ok as
#     the single source of constraint truth on COMPLETE tuples; extends it to PARTIAL
#     tuples (holes) with a per-op REACHABILITY check that is HOLE-MONOTONE
#     (assigning a hole never flips VIOLATED -> not-VIOLATED), so is_consistent_partial
#     stays a sound prune (csp_core §5 L-MONO).
#   * cage_pred (L-ASYM): subtraction / division cages are ORDER-INDEPENDENT — a 2-cell
#     cage's relation holds "one way or the other". cage_ok already checks both orders
#     internally (abs() for sub; both divisor orders for div), so the Factor.scope may
#     be ANY order of the cage cells; the predicate is COMMUTATIVE. No ordered-scope
#     convention is needed for KenKen cages (unlike a circuit gate's output element 0).
#   * l_alldiff_propagator (L-ALLDIFF): a 7-cell all-different has 7^7 ~ 823k tuples in
#     its domain product, blowing gac_propagate's arity_cap (20000) -> generic GAC
#     SKIPS it (forward-check only). This SOUND specialized propagator (naked-singles +
#     Hall intervals ALWAYS; Hall value-occurrence/hidden-single ONLY in the permutation
#     regime, len(scope)==|value universe|) restores all-different GAC strength. It is
#     registered into the FactorType.specialized_propagator slot for ROW/COL, which
#     problem_from_kenken always builds as full n-cell rows/cols over 1..n (a true
#     permutation), so the hidden-single rule is always live there.

# Op-id <-> op-name map. MUST match build_kenken_data / kenken_data.OP_VOCAB
# (["given", "add", "sub", "mul", "div"], indices 0..4) so cage_pred can call the
# string-keyed cage_ok oracle from the integer op_id carried in factor.params.
CAGE_OPS = {0: "given", 1: "add", 2: "sub", 3: "mul", 4: "div"}
OP_TO_ID = {name: oid for oid, name in CAGE_OPS.items()}

# Factor-type ids for the KenKen relations (distinct from the coloring LTYPE_EDGE=0).
LTYPE_ROW = 1     # row all-different (n-ary, specialized propagator)
LTYPE_COL = 2     # col all-different (n-ary, specialized propagator)
LTYPE_CAGE = 3    # arithmetic cage (cage_pred, (op_id, target) params)


def all_diff_pred(_ftype, _params, member_values) -> Consistency:
    """KenKen row/col / Sudoku unit: n-ary all-different, no params (spec §2.2). SAT
    when complete with all-distinct; VIOLATED on any duplicate among ASSIGNED values;
    else UNVIOLATED (still extendable).

    HOLE-MONOTONE (L-MONO): a duplicate among assigned values cannot be undone by
    filling a hole, so a VIOLATED partial stays VIOLATED — the partial-soundness gate
    pruning it is sound. The n-ary GAC strength comes from l_alldiff_propagator (this
    predicate alone, behind the arity_cap, would only forward-check)."""
    seen = [v for v in member_values if v != UNASSIGNED]
    if len(seen) != len(set(seen)):
        return Consistency.VIOLATED
    return Consistency.SAT if UNASSIGNED not in member_values else Consistency.UNVIOLATED


def _cage_partial_reachable(op: str, target: int, assigned, n_holes: int) -> bool:
    """Three-valued PARTIAL extendability for a cage with holes (the L-MONO-respecting
    core of cage_pred). `assigned` = the already-fixed values (in any order, the cage is
    commutative); `n_holes` = remaining unassigned cells (each can take a value >= 1).

    Returns True if SOME completion of the holes can still satisfy op == target (the
    partial is UNVIOLATED), False if NO completion can (VIOLATED). Values are >= 1
    (KenKen domains are 1..n), which is what makes the monotone bounds below valid:
      * add : each hole adds >= 1, so partial_sum only grows. VIOLATED iff partial_sum +
              n_holes > target (the minimum each hole can add is 1, so even the smallest
              completion overshoots; this subsumes partial_sum > target since n_holes>=1
              here, each hole contributing >= 1).
      * mul : each hole multiplies by >= 1, so partial_product only grows (never shrinks
              below itself). VIOLATED iff partial_product does not DIVIDE target (a
              future factor v contributes target/partial only if it is an integer), OR
              partial_product > target. (target/partial >= 1 is reachable by holes
              each >= 1, so divisibility + not-overshoot is the exact monotone test.)
      * sub/div (2-cell, exactly one cell assigned): one hole. ANY relation with a free
              second cell of value >= 1 is still extendable in general, so UNVIOLATED
              (the complete-tuple cage_ok check fires once the hole is filled). This is
              the documented MINIMUM partial check for binary ops (kept deliberately
              weak to stay trivially sound + monotone; the all-diff + row/col factors do
              the heavy pruning around 2-cell cages).
      * given: a 1-cell cage with a hole is just an unfixed given -> UNVIOLATED until
              assigned, then cage_ok checks asg[0]==target.
    MONOTONICITY: every False (VIOLATED) verdict here is derived from a bound that can
    only get TIGHTER as holes are filled (sum/product grow; a fixed binary value is
    checked exactly by cage_ok), so a VIOLATED partial never becomes non-VIOLATED."""
    if op == "given":
        return True  # unfixed given: extendable until its single cell is assigned
    if op == "add":
        s = sum(assigned)
        # minimum completion adds n_holes*1; if even that overshoots, dead.
        if s + n_holes > target:
            return False
        return True
    if op == "mul":
        p = 1
        for x in assigned:
            p *= x
        if p > target:
            return False
        if target % p != 0:
            return False
        return True
    if op in ("sub", "div"):
        # 2-cell cage, exactly one cell assigned (one hole): generically extendable.
        return True
    return False


def cage_pred(_ftype, params, member_values) -> Consistency:
    """KenKen arithmetic cage predicate (params = (op_id, target_INT); L-PARAM exact
    integer target). THREE-VALUED + HOLE-MONOTONE + COMMUTATIVE (L-ASYM):

      * COMPLETE tuple (no holes): SAT iff build_kenken_data.cage_ok(op, target, vals)
        — the SINGLE source of constraint truth — else VIOLATED.
      * PARTIAL tuple (holes): UNVIOLATED if still reachable to op == target
        (_cage_partial_reachable), else VIOLATED.

    L-ASYM: subtraction / division are order-independent; cage_ok checks both orders
    internally, so member_values may be in ANY scope order — the predicate is
    commutative and needs no ordered-scope convention (unlike a circuit gate)."""
    op_id, target = params
    op = CAGE_OPS[op_id]
    assigned = [v for v in member_values if v != UNASSIGNED]
    n_holes = len(member_values) - len(assigned)
    # cage_ok's sub/div branch assumes a 2-cell cage (a, b = asg); 'given' a 1-cell cage.
    # Guard the arity so a malformed/representative call (e.g. the registration-time
    # monotonicity sampler feeding arity-3 sub) is VIOLATED, never a crash — and stays
    # hole-monotone (a wrong-arity cage can never be satisfied, at any hole-filling).
    arity = len(member_values)
    if (op in ("sub", "div") and arity != 2) or (op == "given" and arity != 1):
        return Consistency.VIOLATED
    if n_holes == 0:
        # COMPLETE: defer to the oracle (single source of truth).
        return Consistency.SAT if cage_ok(op, target, tuple(member_values)) \
            else Consistency.VIOLATED
    # PARTIAL: monotone reachability.
    return (Consistency.UNVIOLATED if _cage_partial_reachable(op, target, assigned, n_holes)
            else Consistency.VIOLATED)


def l_alldiff_propagator(state, factor):
    """L-ALLDIFF: a SOUND all-different GAC propagator for an n-ary all-different factor
    (csp_core FactorType.specialized_propagator slot). Generic GAC skips a 7-cell
    all-different (7^7 > arity_cap), forward-checking only; this restores all-different
    arc-consistency strength WITHOUT enumerating the 7^7 product.

    Three prunings, iterated to a fixpoint:
      1. NAKED SINGLES (SOUND for ANY all-different scope): a member fixed to a singleton
         value `a` removes `a` from every OTHER member's domain (a can appear in at most
         one cell).
      2. HALL VALUE-OCCURRENCE / HIDDEN SINGLE (SOUND ONLY IN THE PERMUTATION REGIME):
         a value `a` legal in EXACTLY ONE member's domain is FORCED into that member (its
         domain collapses to {a}); then (1) propagates it. This is sound ONLY when EVERY
         value in the scope's value universe MUST be placed — i.e. len(scope) == |union
         of member domains| (a bijection). OFF-permutation (len(scope) < |union|) not
         every value need appear, so a single-home value is NOT forced and this rule
         would LOSE solutions (e.g. 2 cells [{1,2},{2,3}]: value 1 has a single home in
         cell0 but (2,3) is a valid completion, so forcing cell0={1} is unsound). The
         guard below fires Rule 2 ONLY when len(scope) == |union| (the permutation
         regime, which is exactly what KenKen's full n-cell rows/cols are). When
         len(scope) > |union| the constraint is already dead by pigeonhole (no all-diff
         completion exists); Rule 2 is skipped there too and the dead branch is caught by
         all_diff_pred at completion / the search's empty-domain + verify gates.
      3. HALL INTERVALS (SOUND for ANY all-different scope): if K members have domains all
         contained in a value-set of size exactly K (a "Hall set" — those K values are
         fully consumed by those K members), remove those K values from every member
         OUTSIDE the set.

    SOUNDNESS (non-negotiable): every removed value provably participates in NO valid
    all-different completion on this scope — (1) a fixed value can't repeat; (2) IN THE
    PERMUTATION REGIME a value with a single home must go there (guarded); (3) a Hall
    set's values are exhausted by its members, so an outside member taking one would
    leave < K values for K members (pigeonhole, no completion). Returns a NEW state
    (gac_propagate copies before calling); an emptied domain is left for the caller's
    empty-domain prune.
    """
    s = state.copy()
    scope = list(factor.scope)
    changed = True
    while changed:
        changed = False
        # (1) NAKED SINGLES: a singleton domain removes its value from co-members.
        singles = {}
        for u in scope:
            if len(s.domains[u]) == 1:
                singles[u] = next(iter(s.domains[u]))
        for u in scope:
            for w, a in singles.items():
                if w != u and a in s.domains[u]:
                    s.domains[u] = s.domains[u] - {a}
                    changed = True
                    if not s.domains[u]:
                        return s
        # (2) HALL VALUE-OCCURRENCE: a value with a single possible home is forced.
        # SCOPE GUARD: sound ONLY in the permutation regime (len(scope)==|union|), where
        # every value in the union MUST be placed by pigeonhole so a single-home value is
        # genuinely forced. Off-permutation (len(scope)<|union|) not all values appear ->
        # skip (would lose solutions); over-determined (len(scope)>|union|) is dead and
        # also skipped (caught downstream by all_diff_pred / empty-domain).
        all_vals = set()
        for u in scope:
            all_vals |= s.domains[u]
        if len(scope) == len(all_vals):
            for a in all_vals:
                homes = [u for u in scope if a in s.domains[u]]
                if len(homes) == 1 and len(s.domains[homes[0]]) > 1:
                    s.domains[homes[0]] = {a}
                    changed = True
        # (3) HALL INTERVALS: K members whose domains union to exactly K values consume
        # those K values; remove them from members OUTSIDE the Hall set. Restrict to
        # value-sets actually realized as a member's domain (sound + cheap; covers the
        # binding cases — naked pairs/triples — without enumerating all subsets).
        candidate_sets = {frozenset(s.domains[u]) for u in scope if s.domains[u]}
        for vs in candidate_sets:
            ksize = len(vs)
            if ksize == 0:
                continue
            members_in = [u for u in scope if s.domains[u] and s.domains[u] <= vs]
            if len(members_in) == ksize:
                # Hall set: these ksize values are fully consumed by these ksize members.
                inset = set(members_in)
                for u in scope:
                    if u in inset:
                        continue
                    overlap = s.domains[u] & vs
                    if overlap:
                        s.domains[u] = s.domains[u] - vs
                        changed = True
                        if not s.domains[u]:
                            return s
    return s


def kenken_registry(n: int) -> dict:
    """Registry for KenKen: ROW + COL all-different (with the L-ALLDIFF specialized
    propagator) + CAGE cage_pred. arity_hint=n on ROW/COL feeds the GAC cost guard
    (so n-ary rows route to the specialized propagator past the arity_cap); arity_hint
    on CAGE feeds the L-MONO monotonicity check at registration.

    The L-MONO hole-monotonicity contract is ENFORCED at registration for all three
    types (check_alphabet = 1..n, the KenKen value alphabet) so the partial-soundness
    gate can never be wired to a non-monotone predicate unnoticed. cage_pred is checked
    with a representative ('add', n) param (the worst case for the monotone sum bound)."""
    reg = new_registry()
    register(reg, LTYPE_ROW, all_diff_pred, name="row_all_diff",
             specialized_propagator=l_alldiff_propagator, arity_hint=n,
             check_alphabet=tuple(range(1, n + 1)))
    register(reg, LTYPE_COL, all_diff_pred, name="col_all_diff",
             specialized_propagator=l_alldiff_propagator, arity_hint=n,
             check_alphabet=tuple(range(1, n + 1)))
    # CAGE: monotonicity is op-dependent; 'add' with a small target is the tightest
    # monotone bound, the representative for the registration-time L-MONO check.
    register(reg, LTYPE_CAGE, cage_pred, name="cage", arity_hint=4,
             check_alphabet=tuple(range(1, n + 1)),
             representative_params=(OP_TO_ID["add"], n))
    return reg


def _cell_id(r: int, c: int, n: int) -> int:
    """Row-major flat variable id for grid cell (r, c) on an n x n board."""
    return r * n + c


def problem_from_kenken(n: int, cages, clues, registry: Optional[dict] = None) -> Problem:
    """Bridge: build a general csp_core.Problem from a KenKen instance.

    Args (the build_kenken_data corpus interface):
      n      : board size (grid is n x n; values are 1..n).
      cages  : list of cages, each a list of (r, c) cells (lists or tuples).
      clues  : list of (op_str, target_int) paired 1:1 with `cages` (EXACT integer
               target — L-PARAM; NOT the lossy inlet bucket).

    Variables = the n*n cells (row-major flat id r*n+c; 0-indexed). domains0 = {1..n}
    for every cell EXCEPT 1-cell 'given' cages, whose single cell is pre-fixed to a
    singleton {target} (flows through make_initial_state's given handling).

    Factors:
      * ROW  : one all_diff (LTYPE_ROW) over each row's n cells.
      * COL  : one all_diff (LTYPE_COL) over each col's n cells.
      * CAGE : one cage_pred (LTYPE_CAGE) per NON-given cage, params=(op_id, target).
               (1-cell givens become singleton domains0, not factors — the value is
               fixed, no relation to enforce. A 'given' could also be a degenerate
               cage_pred, but pinning the domain is the cleaner, search-cheaper form.)

    L-ASYM: cage scopes are emitted in cage-cell order and the predicate is commutative
    (sub/div handled both ways inside cage_ok) — no ordered-scope convention needed.
    """
    reg = registry if registry is not None else kenken_registry(n)
    n_vars = n * n
    domains0 = [set(range(1, n + 1)) for _ in range(n_vars)]
    factors = []

    # CAGE factors + given singletons.
    for cage, clue in zip(cages, clues):
        op, target = clue[0], int(clue[1])
        cells = [(int(r), int(c)) for (r, c) in cage]
        if op == "given":
            (r, c) = cells[0]
            domains0[_cell_id(r, c, n)] = {target}
            continue
        scope = tuple(_cell_id(r, c, n) for (r, c) in cells)
        factors.append(Factor(ftype=LTYPE_CAGE, scope=scope,
                              params=(OP_TO_ID[op], target)))

    # ROW + COL all-different factors.
    for r in range(n):
        scope = tuple(_cell_id(r, c, n) for c in range(n))
        factors.append(Factor(ftype=LTYPE_ROW, scope=scope, params=None))
    for c in range(n):
        scope = tuple(_cell_id(r, c, n) for r in range(n))
        factors.append(Factor(ftype=LTYPE_COL, scope=scope, params=None))

    var_factors = [[] for _ in range(n_vars)]
    for fi, f in enumerate(factors):
        for u in f.scope:
            var_factors[u].append(fi)

    return Problem(
        n_vars=n_vars,
        domains0=domains0,
        factors=factors,
        var_factors=var_factors,
        registry=reg,
    )


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
