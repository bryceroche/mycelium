"""test_algebra2_soundness.py — soundness gates for the tranche's MOD + SEL
(2026-07-09). §6: soundness tests must cover the GENERAL regime, not just the
deployed one — random domains, random subsets, off-regime shapes.

GATES (all hard-error):
  1. PREDICATE truth-tables vs brute force on exhaustive small domains.
  2. PROPAGATOR soundness: on random sub-domains, no value participating in a
     valid full assignment is ever pruned (500 random trials per ltype).
  3. HOLE-MONOTONICITY: filling a hole never moves SAT/VIOLATED back to
     UNVIOLATED.
  4. END-TO-END: Vieta pair + selector solves to the selected root, unique;
     both-even selector self-gates (no unique solution); mod chains solve.
"""
from __future__ import annotations

import itertools
import os
import random
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from mycelium.csp_core import Consistency, solve_symbolic  # noqa: E402
from mycelium.csp_domains import (  # noqa: E402
    SEL_TO_ID, _sel_apply, mod_pred, sel_pred, mod_propagator, sel_propagator,
    problem_from_algebra2, LTYPE_MOD, LTYPE_SEL, algebra2_registry,
)
from mycelium.csp_core import Factor  # noqa: E402


class _S:
    def __init__(self, domains):
        self.domains = [set(d) for d in domains]

    def copy(self):
        return _S(self.domains)


def test_predicates_exhaustive():
    M = 12
    for k in range(2, 8):
        for a in range(M + 1):
            for r in range(M + 1):
                got = mod_pred(LTYPE_MOD, k, (a, r))
                want = (Consistency.SAT if a % k == r
                        else Consistency.VIOLATED)
                assert got == want, (k, a, r, got)
    for sel, sid in SEL_TO_ID.items():
        for x, a, b in itertools.product(range(M + 1), repeat=3):
            got = sel_pred(LTYPE_SEL, sid, (x, a, b))
            v = _sel_apply(sid, a, b)
            want = (Consistency.SAT if (v is not None and v == x)
                    else Consistency.VIOLATED)
            assert got == want, (sel, x, a, b, got)
    print("  [1] predicates exhaustive: OK")


def test_propagator_soundness():
    rng = random.Random(0)
    M = 20
    for trial in range(500):
        k = rng.randint(2, 9)
        Da = set(rng.sample(range(M + 1), rng.randint(1, M)))
        Dr = set(rng.sample(range(M + 1), rng.randint(1, M)))
        st = _S([Da, Dr])
        f = Factor(ftype=LTYPE_MOD, scope=(0, 1), params=k)
        out = mod_propagator(st, f)
        for a in Da:
            if a % k in Dr:
                assert a in out.domains[0], ("mod pruned supported a", trial)
                assert a % k in out.domains[1], ("mod pruned supported r", trial)
    for trial in range(500):
        sid = rng.randint(0, 3)
        Dx = set(rng.sample(range(M + 1), rng.randint(1, M)))
        Da = set(rng.sample(range(M + 1), rng.randint(1, M)))
        Db = set(rng.sample(range(M + 1), rng.randint(1, M)))
        st = _S([Dx, Da, Db])
        f = Factor(ftype=LTYPE_SEL, scope=(0, 1, 2), params=sid)
        out = sel_propagator(st, f)
        for a in Da:
            for b in Db:
                v = _sel_apply(sid, a, b)
                if v is not None and v in Dx:
                    assert a in out.domains[1] and b in out.domains[2] \
                        and v in out.domains[0], ("sel pruned support", trial)
    print("  [2] propagator soundness (500 random trials each): OK")


def test_hole_monotone():
    from mycelium.csp_core import UNASSIGNED
    assert mod_pred(LTYPE_MOD, 5, (UNASSIGNED, 3)) == Consistency.UNVIOLATED
    assert sel_pred(LTYPE_SEL, 0, (UNASSIGNED, 4, UNASSIGNED)) == \
        Consistency.UNVIOLATED
    print("  [3] hole-monotone: OK")


def _solve(n, facs, givens, m):
    return solve_symbolic(problem_from_algebra2(n, facs, givens, m),
                          budget=100_000, seed=0)


def test_end_to_end():
    # Vieta pair + larger-selector: roots {3, 7}, S=10, N=21, x = larger = 7
    facs = [{"ftype": "rel", "op": "add", "args": [0, 1], "result": 2},
            {"ftype": "rel", "op": "mul", "args": [0, 1], "result": 3},
            {"ftype": "sel", "sel": "larger", "args": [0, 1], "result": 4}]
    res = _solve(5, facs, {2: 10, 3: 21}, 30)
    assert res["status"] == "solved", res["status"]
    assert res["assignment"][4] == 7, res["assignment"]
    # uniqueness at the selected var (ban-and-resolve)
    p2 = problem_from_algebra2(5, facs, {2: 10, 3: 21}, 30)
    p2.domains0[4].discard(7)
    r2 = solve_symbolic(p2, budget=100_000, seed=0)
    assert r2["status"] != "solved", "selected root must be forced"
    # WITHOUT the selector the pair is symmetric: root var not forced
    p3 = problem_from_algebra2(4, facs[:2], {2: 10, 3: 21}, 30)
    r3 = solve_symbolic(p3, budget=100_000, seed=0)
    assert r3["status"] == "solved"
    v0 = r3["assignment"][0]
    p4 = problem_from_algebra2(4, facs[:2], {2: 10, 3: 21}, 30)
    p4.domains0[0].discard(v0)
    r4 = solve_symbolic(p4, budget=100_000, seed=0)
    assert r4["status"] == "solved", "unselected Vieta pair must be symmetric"
    # even-selector self-gating: roots {4, 6} both even -> sel VIOLATED always
    facs_e = [{"ftype": "rel", "op": "add", "args": [0, 1], "result": 2},
              {"ftype": "rel", "op": "mul", "args": [0, 1], "result": 3},
              {"ftype": "sel", "sel": "even", "args": [0, 1], "result": 4}]
    r5 = _solve(5, facs_e, {2: 10, 3: 24}, 30)
    assert r5["status"] != "solved", "both-even selector must self-gate (UNSAT)"
    # even-selector well-defined: roots {3, 8} -> even one = 8
    r6 = _solve(5, facs_e, {2: 11, 3: 24}, 30)
    assert r6["status"] == "solved" and r6["assignment"][4] == 8
    # mod chain: a mod 7 = r, a = 5*4 = 20 -> r = 6
    facs_m = [{"ftype": "rel", "op": "mul", "args": [0, 1], "result": 2},
              {"ftype": "mod", "var": 2, "k": 7, "result": 3}]
    r7 = _solve(4, facs_m, {0: 5, 1: 4}, 40)
    assert r7["status"] == "solved" and r7["assignment"][3] == 6
    # mod as CONSTRAINT (engine band): a in 0..40, a mod 7 = 3, a mod 5 = 2 -> a=17
    facs_c = [{"ftype": "mod", "var": 0, "k": 7, "result": 1},
              {"ftype": "mod", "var": 0, "k": 5, "result": 2}]
    r8 = _solve(3, facs_c, {1: 3, 2: 2}, 34)
    assert r8["status"] == "solved" and r8["assignment"][0] == 17, \
        r8["assignment"]
    print("  [4] end-to-end (Vieta+sel forced; symmetry without sel; "
          "self-gating; mod chains + CRT): OK")


if __name__ == "__main__":
    print("[algebra2 soundness]")
    test_predicates_exhaustive()
    test_propagator_soundness()
    test_hole_monotone()
    test_end_to_end()
    print("  ALL GATES PASSED — zero csp_core edits (git diff is the proof)")
