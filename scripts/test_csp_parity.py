"""test_csp_parity.py — the PHASE-0 PARITY GATE (behavior-preservation proof).

This is the load-bearing selftest for the Phase-0 refactor (docs/general_factor_graph_
search.md §6.2 Phase 0 + the parity-gate requirement). It asserts the NEW
predicate-driven core (mycelium/csp_core.py + csp_registry.py + csp_domains.py) is
EQUAL to the OLD coloring-specific module (mycelium/csp_coloring_legacy.py, the frozen
pre-refactor csp_search.py) on a fixed fixture set of coloring instances WITH REAL
BACKTRACKING, on every behavior the spec names:

  1. generic gac_propagate on not-equal  ==  old ac3_propagate
       identical per-vertex DOMAIN PRUNING + identical FORCED-SINGLETON commits.
  2. mrv_varorder on not-equal           ==  old dsatur_varorder
       identical PICK at every forward-checked state along a real search path
       (L-TIE: same order/decision-count on fixtures, NOT a byte-identical tie-stream).
  3. lcv_valorder                          ==  old lcv_valorder (identical value order).
  4. verify_complete                       ==  old is_complete_proper (all-SAT arbiter).
     is_consistent_partial                 ==  old is_proper_partial (soundness gate).
  5. solve_symbolic (new core)             ==  old solve_symbolic
       identical solved/unsat status + identical decisions + identical backtracks.

The fixtures are NON-TRIVIAL: K4/3 (9 backtracks), random dense graphs that force
real backtracking, odd cycles (unsat), the 3-prism. A tautology fixture set would not
exercise the search, so the parity claim would be vacuous; these do.

Run:  SELFTEST_ONLY=1 .venv/bin/python3 scripts/test_csp_parity.py
GPU-FREE: pure python + numpy. ast.parse clean, CPU import clean.
"""

from __future__ import annotations

import ast
import os
import random
import sys

_THIS = os.path.abspath(__file__)
sys.path.insert(0, os.path.dirname(os.path.dirname(_THIS)))

from mycelium import csp_coloring_legacy as L  # noqa: E402
from mycelium.csp_core import (  # noqa: E402
    UNASSIGNED,
    assign_var,
    gac_propagate,
    is_consistent_partial,
    lcv_valorder,
    make_initial_state,
    mrv_varorder,
    solve_symbolic,
    verify_complete,
)
from mycelium.csp_domains import (  # noqa: E402
    is_complete_proper_coloring,
    is_proper_partial_coloring,
    problem_from_coloring,
)


# ===========================================================================
# FIXTURES — coloring instances WITH REAL backtracking
# ===========================================================================

def _fixtures():
    fx = []
    fx.append(("C5/3", 5, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)], 3))
    fx.append(("C5/2", 5, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)], 2))
    k4 = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    fx.append(("K4/3", 4, k4, 3))         # unsat, 9 backtracks
    fx.append(("K4/4", 4, k4, 4))
    tri_a = [(0, 1), (1, 2), (0, 2)]
    tri_b = [(3, 4), (4, 5), (3, 5)]
    rungs = [(0, 3), (1, 4), (2, 5)]
    fx.append(("3prism/3", 6, tri_a + tri_b + rungs, 3))
    fx.append(("3prism/2", 6, tri_a + tri_b + rungs, 2))
    # random dense graphs (force real backtracking at k=3 and k=4)
    for s in range(10):
        rng = random.Random(1000 + s)
        n = 12
        e = []
        for u in range(n):
            for w in range(u + 1, n):
                if rng.random() < 0.45:
                    e.append((u, w))
        fx.append((f"rand{s}/3", n, e, 3))
        fx.append((f"rand{s}/4", n, e, 4))
    return fx


# ===========================================================================
# BRIDGES between the two state representations (legacy CSPState <-> new CSPState)
# ===========================================================================

def _new_state_from_assignment(n, edges, k, assignment):
    """A new-core CSPState at a given (partial) assignment, with domains forward-checked
    from scratch by replaying assign_var (so it matches a real search node)."""
    prob = problem_from_coloring(n, edges, k)
    st = make_initial_state(prob)
    for v, c in sorted(assignment.items()):
        st = assign_var(st, v, c)
    return st, prob


def _legacy_state_from_assignment(n, edges, k, assignment):
    edges_n = L.normalize_edges(edges, n)
    st = L.make_initial_state(n, edges_n, k)
    for v, c in sorted(assignment.items()):
        st = L.assign_vertex(st, v, c)
    return st


# ===========================================================================
# THE PARITY CHECKS
# ===========================================================================

def _check_verifier_parity(check, n, edges, k):
    """verify_complete == is_complete_proper; is_consistent_partial == is_proper_partial
    on a spread of complete + partial + improper assignments."""
    edges_n = L.normalize_edges(edges, n)
    rng = random.Random(7)
    ok = True
    for _ in range(40):
        # random partial-or-complete assignment (possibly improper)
        asg = {}
        for v in range(n):
            r = rng.random()
            if r < 0.7:
                asg[v] = rng.randrange(k)
        # legacy verdicts
        old_partial = L.is_proper_partial(asg, edges_n, n)
        old_complete = L.is_complete_proper(asg, edges_n, n, k=k)
        # new verdicts via the coloring wrappers (which call the GENERAL arbiters)
        new_partial = is_proper_partial_coloring(asg, n, edges, k)
        new_complete = is_complete_proper_coloring(asg, n, edges, k)
        if old_partial != new_partial or old_complete != new_complete:
            ok = False
    return ok


def _check_gac_vs_ac3(check, n, edges, k):
    """gac_propagate(not_equal) == ac3_propagate, from a spread of partial roots.

    On a LIVE (non-conflict) branch: identical per-vertex domains AND identical
    forced-singleton commits — the equivalence the search relies on (this is what makes
    solve_symbolic byte-identical).

    On a DEAD branch (some domain empties): BOTH must detect the conflict (deadness
    agreement — the load-bearing soundness property), but the RESIDUAL domains are
    allowed to differ. They differ only because each propagator early-returns the moment
    it hits the first empty domain, and the queue-drain ORDER differs; the caller
    discards a dead node's residual domains, so this never affects search. Comparing
    residual domains on a dead branch would test an implementation-internal ordering, not
    behavior — so we assert deadness agreement there, not domain identity.
    """
    edges_n = L.normalize_edges(edges, n)
    rng = random.Random(11)
    ok = True
    for _ in range(30):
        asg = {}
        for v in range(n):
            if rng.random() < 0.3:
                asg[v] = rng.randrange(k)
        if not L.is_proper_partial(asg, edges_n, n):
            continue
        old_s = _legacy_state_from_assignment(n, edges, k, asg)
        old_s = L.ac3_propagate(old_s)
        new_s, _ = _new_state_from_assignment(n, edges, k, asg)
        new_s = gac_propagate(new_s)
        old_dead = any(len(old_s.domains[v]) == 0 for v in range(n))
        new_dead = any(len(new_s.domains[v]) == 0 for v in range(n))
        # SOUNDNESS: both must agree the branch is dead-or-alive.
        if old_dead != new_dead:
            ok = False
            continue
        if old_dead:
            continue  # dead branch: residual domains discarded by the caller
        # LIVE branch: byte-identical domains + commits.
        for v in range(n):
            if old_s.domains[v] != new_s.domains[v]:
                ok = False
            if old_s.colors[v] != new_s.values[v]:
                ok = False
    return ok


def _check_mrv_vs_dsatur(check, n, edges, k):
    """mrv_varorder == dsatur_varorder DECISION sequence along a real greedy search
    path: at each forward-checked node, both must pick the SAME variable, then we extend
    by the SAME value and re-check (L-TIE: same order/decisions, not a tie-stream)."""
    edges_n = L.normalize_edges(edges, n)
    # build matched states and walk a path picking the var both agree on, value = min
    old_s = L.make_initial_state(n, edges_n, k)
    old_s = L.ac3_propagate(old_s)
    new_s, _ = _new_state_from_assignment(n, edges, k, {})
    new_s = gac_propagate(new_s)
    ok = True
    for _ in range(n):
        ov = L.dsatur_varorder(old_s)
        nv = mrv_varorder(new_s)
        if ov != nv:
            ok = False
            break
        if ov == UNASSIGNED or ov < 0:
            break
        # also assert the value ORDER agrees at this node (LCV parity)
        ovals = L.lcv_valorder(old_s, ov)
        nvals = lcv_valorder(new_s, nv)
        if ovals != nvals:
            ok = False
            break
        if not ovals:
            break
        c = ovals[0]
        # advance BOTH the same way (assign + propagate), staying on a proper path
        cand_old = L.ac3_propagate(L.assign_vertex(old_s, ov, c))
        cand_new = gac_propagate(assign_var(new_s, nv, c))
        if L.has_empty_domain(cand_old):
            break
        old_s, new_s = cand_old, cand_new
    return ok


def _check_solve_symbolic_parity(check, n, edges, k):
    """solve_symbolic (new) == solve_symbolic (old): identical status + decisions +
    backtracks (the integrated proof — everything above composed through the skeleton)."""
    old_r = L.solve_symbolic(n, edges, k, budget=200000)
    new_r = solve_symbolic(problem_from_coloring(n, edges, k), budget=200000)
    same = (
        old_r["status"] == new_r["status"]
        and old_r["decisions"] == new_r["decisions"]
        and old_r["backtracks"] == new_r["backtracks"]
    )
    # plus: a 'solved' result from the new core must verify proper
    if new_r["status"] == "solved":
        same = same and is_complete_proper_coloring(new_r["assignment"], n, edges, k)
    return same, old_r, new_r


def _selftest() -> bool:
    all_ok = True

    def _check(name, cond):
        nonlocal all_ok
        if not cond:
            all_ok = False
        print(f"[parity] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    fx = _fixtures()
    n_bt = 0     # count fixtures that actually backtracked (non-triviality witness)

    for (name, n, edges, k) in fx:
        _check(f"{name}: verifier parity (verify_complete==is_complete_proper, "
               f"partial gate==is_proper_partial)",
               _check_verifier_parity(_check, n, edges, k))
        _check(f"{name}: gac_propagate(not_equal)==ac3_propagate (domains + commits)",
               _check_gac_vs_ac3(_check, n, edges, k))
        _check(f"{name}: mrv==dsatur pick + lcv order along a real path",
               _check_mrv_vs_dsatur(_check, n, edges, k))
        same, old_r, new_r = _check_solve_symbolic_parity(_check, n, edges, k)
        _check(f"{name}: solve_symbolic parity old{(old_r['status'], old_r['decisions'], old_r['backtracks'])} "
               f"== new{(new_r['status'], new_r['decisions'], new_r['backtracks'])}",
               same)
        if new_r["backtracks"] > 0:
            n_bt += 1

    # NON-TRIVIALITY GUARD: the fixture set MUST exercise real backtracking, else the
    # parity claim is vacuous. Require several fixtures with >0 backtracks.
    _check(f"non-triviality: >= 5 fixtures actually backtracked (got {n_bt})",
           n_bt >= 5)

    print(f"[parity] {'ALL PASS' if all_ok else 'SOME FAILED'}", flush=True)
    return all_ok


def _ast_parse_ok() -> bool:
    with open(_THIS) as f:
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
    print("set SELFTEST_ONLY=1 to run the parity gate.", flush=True)
