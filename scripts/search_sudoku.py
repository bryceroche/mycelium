"""search_sudoku.py — Bryce's vision realized: candidate-sets + recursive branch -> 100% solve.

The SYMBOLIC search tier (the 'disposer') on 9x9 Sudoku: GAC maintains the exact
per-cell remaining-value sets (the dual / pencil-mark channel, done symbolically) +
l_alldiff_propagator for rows/cols/boxes, MRV/LCV order, backtracking. Pure reuse of
the KenKen all-different machinery via one registry + one bridge (zero csp_core edits).

The neural deducer would enter ONLY as ordering priors (a marginal accelerator on clean
Sudoku — symbolic ordering already collapses the tree; the policy never commits). This
script is the symbolic half alone, which is what achieves 100%.

  .venv/bin/python3 scripts/search_sudoku.py [--n 300] [--path .cache/sudoku_test.jsonl]
"""
import argparse
import json
import statistics
import sys
import time

sys.path.insert(0, ".")
from mycelium.csp_core import solve_symbolic                      # noqa: E402
from mycelium.csp_domains import problem_from_sudoku, sudoku_registry  # noqa: E402

PATH = ".cache/sudoku_test.jsonl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--path", default=PATH)
    ap.add_argument("--budget", type=int, default=200000)
    args = ap.parse_args()

    recs = []
    with open(args.path) as f:
        for line in f:
            if line.strip():
                recs.append(json.loads(line))
            if len(recs) >= args.n:
                break

    reg = sudoku_registry(9)   # build once, reuse (the registry is instance-independent)
    solved = correct = total = 0
    dec_list = []
    by_diff: dict = {}
    t0 = time.time()
    for rec in recs:
        cells = rec["input"]
        sol = rec.get("solution")
        prob = problem_from_sudoku(cells, n=9, registry=reg)
        res = solve_symbolic(prob, budget=args.budget, seed=0)
        total += 1
        is_solved = res.get("status") == "solved"
        asg = res.get("assignment")
        is_correct = bool(is_solved and sol is not None and list(asg) == list(sol))
        solved += int(is_solved)
        correct += int(is_correct)
        dec_list.append(int(res.get("decisions", 0)))
        d = str(rec.get("difficulty", "?"))
        bd = by_diff.setdefault(d, [0, 0, []])
        bd[0] += 1
        bd[1] += int(is_correct)
        bd[2].append(int(res.get("decisions", 0)))
    dt = time.time() - t0

    print(f"\n=== Sudoku search-tier solve — {total} puzzles, {dt:.1f}s "
          f"({1000*dt/max(total,1):.1f} ms/puzzle) ===")
    print(f"  status=solved : {solved}/{total} = {solved/total:.4f}")
    print(f"  CORRECT==gold : {correct}/{total} = {correct/total:.4f}")
    print(f"  decisions     : median {statistics.median(dec_list):.0f}  "
          f"mean {statistics.mean(dec_list):.1f}  max {max(dec_list)}")
    print(f"  (median decisions ~= 0 => GAC candidate-set propagation alone solves it;"
          f" backtracking rarely branches)")
    print(f"  by difficulty:")
    for d in sorted(by_diff):
        n_, c_, decs = by_diff[d]
        print(f"    {d:8s}: {c_}/{n_} = {c_/n_:.4f}   median decisions {statistics.median(decs):.0f}")


if __name__ == "__main__":
    main()
