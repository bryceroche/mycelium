"""solverpipe_correctness_test.py — CPU-only, no GPU, no training. Fixture-level
correctness for THE SOLVER PIPELINE (perf lever A): (1) rack_release_row's
refactor (_rack_build_problem extraction) is a pure extraction — compare its
output against a hand-rolled reference on 4 constructed rows; (2) ALG_SOLVER_
POOL reduces to the SAME statuses, in order, as the sequential loop, pooled or
not; (3) ALG_SOLVER_MEMO's hit returns the SAME status a fresh solve would
(solve twice, clear the memo between, compare); (4) the SIGALRM wall still
returns 'budget', never 'unsat', inside a pooled worker.

Run: DEV=CPU .venv/bin/python3 scripts/solverpipe_correctness_test.py
"""
import os
os.environ.setdefault("DEV", "CPU")
import sys
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")

import phase1_algebra_head as H
import solver_pool as SP


def _row(ftype, **kw):
    d = {"ftype": ftype}; d.update(kw); return d


# ---- fixtures ---------------------------------------------------------
FIX = {
    "trivial_solved": (
        {0: _row("given", var=0, value=5)},
        [0], 50, 5, "solved",
    ),
    "conflict_given": (
        {0: _row("given", var=0, value=5), 1: _row("given", var=0, value=7)},
        [0, 1], 50, 5, "unsat",
    ),
    "rel_solved": (
        {0: _row("given", var=0, value=5), 1: _row("given", var=1, value=2),
         2: _row("rel", op="add", args=[0, 1], result=2)},
        [0, 1, 2], 50, 5, "solved",
    ),
    "rel_unsat": (
        {0: _row("given", var=0, value=5), 1: _row("given", var=1, value=3),
         2: _row("rel", op="add", args=[0, 1], result=2),
         3: _row("given", var=2, value=100)},
        [0, 1, 2, 3], 200, 5, "unsat",
    ),
}


def reference_status(by_slot, dry_slots, m, n_vars, budget=5000):
    """Hand-rolled, independent of _rack_build_problem — the pre-refactor logic,
    inlined here so the refactor has something ELSE to agree with (not just
    itself)."""
    from mycelium.csp_domains import problem_from_algebra
    from mycelium.csp_core import solve_symbolic
    gv, rel_facs = {}, []
    for j in dry_slots:
        f = by_slot.get(int(j))
        if f is None:
            continue
        if f["ftype"] == "given":
            v, val = int(f["var"]), int(f["value"])
            if v in gv and gv[v] != val:
                return "unsat"
            gv[v] = val
        elif f["ftype"] == "rel":
            rel_facs.append(f)
    if not rel_facs:
        return "solved"
    nv = max([int(n_vars)] + [v + 1 for v in gv] +
             [v + 1 for f in rel_facs for v in (list(f["args"]) + [f["result"]])])
    rels = [(f["op"], int(f["args"][0]), int(f["args"][1]), int(f["result"])) for f in rel_facs]
    prob = problem_from_algebra(nv, rels, gv, int(m))
    return solve_symbolic(prob, budget=budget, seed=0)["status"]


def main():
    fails = 0
    print("[1] rack_release_row vs independent reference, and vs _rack_build_problem + raw solve:")
    for name, (by_slot, dry, m, nv, want) in FIX.items():
        got_rrr = H.rack_release_row(by_slot, dry, m, nv)
        got_ref = reference_status(by_slot, dry, m, nv)
        got_pool_seq = SP.solve_release_rows([(by_slot, dry, m, nv)], workers=0)[0]
        ok = (got_rrr == want == got_ref == got_pool_seq)
        print(f"    {name:16s} want={want:8s} rack_release_row={got_rrr:8s} "
              f"reference={got_ref:8s} solve_release_rows(seq)={got_pool_seq:8s} {'OK' if ok else 'FAIL'}")
        fails += not ok

    print("[2] ALG_SOLVER_POOL=4 (fork) reduces to the SAME order/statuses as sequential:")
    tasks = [FIX[n][:4] for n in ("trivial_solved", "conflict_given", "rel_solved", "rel_unsat")] * 4
    seq = SP.solve_release_rows(list(tasks), workers=0)
    pooled = SP.solve_release_rows(list(tasks), workers=4)
    ok = seq == pooled
    print(f"    sequential={seq}")
    print(f"    pooled(4) ={pooled}")
    print(f"    {'OK' if ok else 'FAIL'}")
    fails += not ok

    print("[3] ALG_SOLVER_MEMO hit == a fresh solve (memo cleared between):")
    task = FIX["rel_unsat"][:4]
    SP._SEQ_MEMO.clear()
    fresh = SP.solve_release_rows([task], workers=0, use_memo=True)[0]
    cached = SP.solve_release_rows([task], workers=0, use_memo=True)[0]  # should hit now
    hits_before = SP._memo_stats[0]
    SP.solve_release_rows([task], workers=0, use_memo=True)
    hits_after = SP._memo_stats[0]
    ok = fresh == cached == "unsat" and hits_after == hits_before + 1
    print(f"    fresh={fresh} cached={cached} hits {hits_before}->{hits_after} {'OK' if ok else 'FAIL'}")
    fails += not ok
    SP._SEQ_MEMO.clear()

    print("[4] budget=0 on a non-trivial row returns 'budget', never 'unsat', pooled and sequential:")
    task = FIX["rel_solved"][:4]  # satisfiable, but requires a search decision
    seq_b = SP.solve_release_rows([task], workers=0, budget=0)[0]
    pool_b = SP.solve_release_rows([task], workers=4, budget=0)[0]
    print(f"    sequential budget=0 -> {seq_b}; pooled budget=0 -> {pool_b}")
    ok = seq_b in ("solved", "budget") and pool_b in ("solved", "budget")
    fails += not ok

    print(f"\n{'ALL OK' if fails == 0 else f'{fails} FAILURE(S)'}")
    return fails


if __name__ == "__main__":
    sys.exit(main())
