"""algebra_bridge_smoke.py — the REGISTRY-EXTENSION BRICK's symbolic half, proven on
hand-written systems BEFORE any generator exists (the math expansion's first object,
2026-07-07).

FOUR CLAIMS, each asserted:
  (1) BRIDGE: op(a,b)=r relations with unknowns on both sides solve through the
      UNCHANGED search tier (predicate + bridge only — zero csp_core edits, 7th
      domain through the same seam).
  (2) CALCULATOR BAND: a triangular (forward-DAG) system solves at 0 decisions —
      functional propagation alone (consistent with the Job-B gate's GSM8K result).
  (3) ENGINE BAND BY CONSTRUCTION: a COUPLED system (x+y=c1, x-y=c2) needs >0
      decisions — pairwise arc-consistency keeps locally-supported values that only
      JOINT reasoning eliminates. This is the generator dial the unknowns corpus was
      chosen for: coupling depth targets the engine band directly.
  (4) UNIQUENESS IS CHECKABLE (the KenKen property transplanted): ban-and-resolve
      proves the hand systems unique -> gold + equivalence stay FREE in this domain.

USAGE:  .venv/bin/python3 scripts/algebra_bridge_smoke.py
"""
from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from mycelium.csp_domains import problem_from_algebra          # noqa: E402
from mycelium.csp_core import solve_symbolic                    # noqa: E402


def solve(n_vars, relations, givens, m, budget=100_000):
    res = solve_symbolic(problem_from_algebra(n_vars, relations, givens, m),
                         budget=budget, seed=0)
    asg = res["assignment"] if res["status"] == "solved" else None
    return res["status"], asg, res.get("decisions", -1)


def is_unique(n_vars, relations, givens, m, solution):
    """Ban-and-resolve: any SAT re-solve after banning one solved value = 2nd solution."""
    for v in range(n_vars):
        if v in givens:
            continue
        prob = problem_from_algebra(n_vars, relations, givens, m)
        prob.domains0[v].discard(solution[v])
        if not prob.domains0[v]:
            continue
        res = solve_symbolic(prob, budget=100_000, seed=0)
        if res["status"] == "solved":
            return False
    return True


def main() -> None:
    ok = True

    def check(name, cond):
        nonlocal ok
        print(f"  [{'OK' if cond else 'FAIL'}] {name}")
        ok = ok and bool(cond)

    # (1)+(2) TRIANGULAR: c3=3, c4=4 given; z = c3+c4; w = z*c4  -> forward DAG
    # vars: 0=c3 1=c4 2=z 3=w
    st, asg, dec = solve(4, [("add", 0, 1, 2), ("mul", 2, 1, 3)], {0: 3, 1: 4}, m=40)
    check(f"triangular solves (z=7, w=28): {st}, decisions={dec}",
          st == "solved" and asg[2] == 7 and asg[3] == 28)
    check("triangular is CALCULATOR BAND (0 decisions)", dec == 0)

    # (3) COUPLED: x+y=5, x-y=1 -> x=3, y=2. AC keeps x=2 (supported per-constraint
    # separately) — only joint reasoning/search eliminates it.
    # vars: 0=x 1=y 2=c5 3=c1
    st, asg, dec = solve(4, [("add", 0, 1, 2), ("sub", 0, 1, 3)], {2: 5, 3: 1}, m=10)
    check(f"coupled solves (x=3, y=2): {st}, x={asg[0] if asg else '-'}, "
          f"y={asg[1] if asg else '-'}, decisions={dec}",
          st == "solved" and asg[0] == 3 and asg[1] == 2)
    check(f"coupled is ENGINE BAND (>0 decisions): {dec}", dec > 0)
    check("coupled system is UNIQUE (ban-and-resolve)",
          is_unique(4, [("add", 0, 1, 2), ("sub", 0, 1, 3)], {2: 5, 3: 1}, 10,
                    {0: 3, 1: 2}))

    # (3b) THE DIAL: a chain of coupled pairs — decisions grow with coupling count.
    # pairs (x_i + y_i = s_i, x_i - y_i = d_i), all independent -> decisions stack.
    rels, givens, nv = [], {}, 0
    for i in range(4):
        x, y, sv, dv = nv, nv + 1, nv + 2, nv + 3
        rels += [("add", x, y, sv), ("sub", x, y, dv)]
        givens[sv] = 6 + i
        givens[dv] = 2 - (i % 2)
        nv += 4
    st, asg, dec4 = solve(nv, rels, givens, m=12)
    check(f"4-pair chain solves; decisions={dec4} (the generator DIAL: more coupling"
          f" -> deeper into the engine band)", st == "solved" and dec4 > 0)

    # (4) DIV exactness: 12 / 4 = 3 exact; 13 / 4 has NO integer solution.
    st, asg, _ = solve(3, [("div", 0, 1, 2)], {0: 12, 1: 4}, m=20)
    check("exact div solves (12/4=3)", st == "solved" and asg[2] == 3)
    st2, _a, _d = solve(3, [("div", 0, 1, 2)], {0: 13, 1: 4}, m=20)
    check(f"inexact div is UNSAT (13/4 over integers): {st2}", st2 != "solved")

    print(f"[smoke] {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
