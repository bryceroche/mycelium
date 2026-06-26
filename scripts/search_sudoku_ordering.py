"""search_sudoku_ordering.py — does NEURAL value-ordering help symbolic Sudoku search?

Tests the hybrid on the HARDEST known Sudoku (famous backtracking-stress puzzles), plus
the deepest-search puzzles in our test set. For each, compares DECISION COUNTS under:
  * symbolic   : LCV value-ordering (pure search tier; the baseline)
  * oracle     : policy_valorder with a PERFECT policy (one-hot at the solution) — the
                 UPPER BOUND on what ANY neural policy could buy (no neural net needed).
  * random     : policy_valorder with a random policy — the control.

LOGIC: if even the ORACLE ordering barely cuts symbolic's decision count, then no neural
policy can help on Sudoku (the symbolic 'second jaws' are already too good) -> Sudoku is
not the prey, and we look for bigger game. If the oracle slashes a large symbolic
decision count, there IS a gap a good policy could close -> worth a real deducer.

  .venv/bin/python3 scripts/search_sudoku_ordering.py
"""
import json
import random
import statistics
import sys

sys.path.insert(0, ".")
from mycelium.csp_core import (                                   # noqa: E402
    backtrack_search, gac_propagate, mrv_varorder, lcv_valorder, policy_valorder,
)
from mycelium.csp_domains import problem_from_sudoku, sudoku_registry  # noqa: E402

# Famous hardest-known Sudoku (0 = blank). These stress backtracking solvers.
HARD = {
    "AI_Escargot":    "100007090030020008009600500005300900010080002600004000300000010040000007007000300",
    "Inkala_2010":    "005300000800000020070010500400005300010070006003200080060500009004000030000009700",
    "Inkala_2012":    "800000000003600000070090200050007000000045700000100030001000068008500010090000400",
    "Platinum_Blonde":"000000012000000003002300400001800005060070800000009000008500000900040500470006000",
    "Golden_Nugget":  "000000039000001005003050800008090006070002000100400000009080050020000600400700000",
    "Easter_Monster": "100000002090400050006000700050903000000070000000850040700000600030009080002000001",
}
BUDGET = 2_000_000


def parse(s):
    return [int(ch) for ch in s]


def solve(cells, reg, valorder):
    prob = problem_from_sudoku(cells, n=9, registry=reg)
    return backtrack_search(prob, gac_propagate, mrv_varorder, valorder, budget=BUDGET)


def oracle_policy(solution):
    # policy[v][a] = 1.0 at the solution value, else 0 (a in 1..9; index 0 unused).
    return [[1.0 if a == solution[v] else 0.0 for a in range(10)] for v in range(81)]


def random_policy(seed):
    rng = random.Random(seed)
    return [[rng.random() for _ in range(10)] for _ in range(81)]


def run_one(name, cells, reg):
    # 1) symbolic (LCV) — get the solution + baseline decisions
    r_sym = solve(cells, reg, lcv_valorder)
    sol = r_sym.get("assignment")
    d_sym = r_sym.get("decisions", -1)
    st = r_sym.get("status")
    if st != "solved":
        print(f"  {name:16s}: symbolic status={st} decisions={d_sym} (skip ordering)")
        return None
    # 2) oracle policy-first (upper bound)
    r_or = solve(cells, reg, policy_valorder(oracle_policy(sol)))
    d_or = r_or.get("decisions", -1)
    # 3) random policy-first (control)
    r_rd = solve(cells, reg, policy_valorder(random_policy(1234)))
    d_rd = r_rd.get("decisions", -1)
    print(f"  {name:16s}: symbolic={d_sym:6d}   oracle={d_or:6d}   random={d_rd:6d}  decisions")
    return (d_sym, d_or, d_rd)


def main():
    reg = sudoku_registry(9)
    print("=== HARDEST-KNOWN Sudoku — decisions by value-ordering ===")
    rows = []
    for name, s in HARD.items():
        r = run_one(name, parse(s), reg)
        if r:
            rows.append(r)

    # Also: the deepest-search puzzles in our own test set (top by symbolic decisions).
    print("\n=== deepest-search puzzles in .cache/sudoku_test.jsonl (scan 5000) ===")
    recs = []
    with open(".cache/sudoku_test.jsonl") as f:
        for line in f:
            if line.strip():
                recs.append(json.loads(line))
    scored = []
    for rec in recs:
        r = solve(rec["input"], reg, lcv_valorder)
        scored.append((r.get("decisions", 0), rec))
    scored.sort(key=lambda x: -x[0])
    print(f"  max symbolic decisions in test set: {scored[0][0]}  "
          f"(median {statistics.median([s for s, _ in scored])})")
    for d_sym0, rec in scored[:3]:
        r = run_one(f"testset_d{d_sym0}", rec["input"], reg)
        if r:
            rows.append(r)

    if rows:
        print("\n=== VERDICT ===")
        sym = [r[0] for r in rows]
        orc = [r[1] for r in rows]
        print(f"  symbolic decisions: median {statistics.median(sym):.0f}  max {max(sym)}")
        print(f"  oracle   decisions: median {statistics.median(orc):.0f}  max {max(orc)}")
        gap = max(sym)
        print(f"  -> even the HARDEST needed {gap} symbolic decisions; oracle ordering "
              f"cuts to ~{max(orc)}.")
        if gap < 200:
            print("  => Sudoku is TOO EASY for neural ordering (symbolic search barely "
                  "branches; no gap to close). The prey is bigger/softer CSPs.")
        else:
            print("  => There IS a real symbolic-search gap; a good policy could close it "
                  "-> a Sudoku deducer's ordering is worth testing.")


if __name__ == "__main__":
    main()
