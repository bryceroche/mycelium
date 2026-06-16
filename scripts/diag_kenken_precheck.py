"""KenKen precheck (tripwire 18) — is the deduction natively ITERATIVE, or one-pass?

THE GATE before any KenKen build. The loop's home is problems where a cell's
relevance depends on OTHERS' resolved status (native iterative relevance), so
you must PROPAGATE — not one-pass-solvable (the v99 flat-K / GSM8K-calculator
failure). Measures, per generated puzzle, how many ROUNDS of constraint
propagation (alternating cage-arithmetic and Latin row/col AllDiff) a pure
deduction solver needs to reach fixpoint, and whether propagation ALONE solves
it (loop-only) or leaves ambiguity (needs search = loop+search, harder regime).

PRE-REGISTERED BANDS (pinned before running):
  DEEP / loop-only (BUILD): median rounds-to-fixpoint >= 3 AND >= 60% of
    puzzles solved by propagation alone (over-determined enough that BP
    settles, like Sudoku). KenKen has the native iterative relevance.
  SHALLOW / one-pass (calculator risk): median rounds <= 2 OR a large mass at
    rounds==1 — the loop would be a calculator (flat-K). Move to larger boards
    / tighter cage designs before training.
  SEARCH-NEEDED (loop+search): propagation alone solves < 60% — BP smears,
    needs a search layer the loop-only architecture lacks (MCTS gated). Harder.

Pure Python, no GPU. Usage:
  .venv/bin/python scripts/diag_kenken_precheck.py
"""
from __future__ import annotations
import itertools, json, os, random
from collections import Counter

OUT = ".cache/v200_smoke/kenken_precheck.json"
SEED = 42
PUZZLES_PER_N = 200
N_SIZES = [4, 5, 6, 7]


def latin_square(n, rng):
    base = list(range(1, n + 1))
    rows = [base[i:] + base[:i] for i in range(n)]
    rng.shuffle(rows)
    cols = list(range(n)); rng.shuffle(cols)
    sq = [[rows[r][c] for c in cols] for r in range(n)]
    # random relabel of symbols
    perm = base[:]; rng.shuffle(perm); m = {i + 1: perm[i] for i in range(n)}
    return [[m[sq[r][c]] for c in range(n)] for r in range(n)]


def make_cages(n, rng):
    """Partition the grid into contiguous cages of size 1-4 (KenKen-like)."""
    cells = [(r, c) for r in range(n) for c in range(n)]
    rng.shuffle(cells)
    assigned = {}
    cages = []
    for start in cells:
        if start in assigned:
            continue
        target = rng.choices([1, 2, 3, 4], weights=[1, 4, 3, 2])[0]
        cage = [start]; assigned[start] = True
        while len(cage) < target:
            # grow to a random unassigned orthogonal neighbour
            frontier = []
            for (r, c) in cage:
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nb = (r + dr, c + dc)
                    if 0 <= nb[0] < n and 0 <= nb[1] < n and nb not in assigned:
                        frontier.append(nb)
            if not frontier:
                break
            pick = rng.choice(frontier); cage.append(pick); assigned[pick] = True
        cages.append(cage)
    return cages


def cage_clue(cage, sol, rng):
    vals = [sol[r][c] for (r, c) in cage]
    if len(cage) == 1:
        return ("given", vals[0])
    if len(cage) == 2:
        a, b = sorted(vals, reverse=True)
        choices = [("add", a + b), ("mul", a * b)]
        if a - b >= 0:
            choices.append(("sub", a - b))
        if b != 0 and a % b == 0:
            choices.append(("div", a // b))
        return rng.choice(choices)
    # size>=3: add or mul
    if rng.random() < 0.5:
        return ("add", sum(vals))
    p = 1
    for v in vals:
        p *= v
    return ("mul", p)


def cage_consistent(typ, target, assignment):
    if typ == "given":
        return assignment[0] == target
    if typ == "add":
        return sum(assignment) == target
    if typ == "mul":
        p = 1
        for v in assignment:
            p *= v
        return p == target
    if typ == "sub":  # 2-cell, order-free
        return abs(assignment[0] - assignment[1]) == target
    if typ == "div":
        a, b = assignment
        return (b != 0 and a == b * target) or (a != 0 and b == a * target)
    return False


def propagate(n, cages, clues, max_rounds=50):
    """Pure constraint propagation. Returns (rounds_to_fixpoint, solved, domains)."""
    dom = {(r, c): set(range(1, n + 1)) for r in range(n) for c in range(n)}
    for cage, (typ, tgt) in zip(cages, clues):
        if typ == "given":
            dom[cage[0]] = {tgt}
    rounds = 0
    while rounds < max_rounds:
        rounds += 1
        changed = False
        # Latin AllDiff: naked singles remove from row/col peers
        for (r, c), d in list(dom.items()):
            if len(d) == 1:
                v = next(iter(d))
                for cc in range(n):
                    if cc != c and v in dom[(r, cc)]:
                        dom[(r, cc)].discard(v); changed = True
                for rr in range(n):
                    if rr != r and v in dom[(rr, c)]:
                        dom[(rr, c)].discard(v); changed = True
        # Latin hidden singles: a value with one home in a row/col
        for line in ([[(r, c) for c in range(n)] for r in range(n)] +
                     [[(r, c) for r in range(n)] for c in range(n)]):
            for v in range(1, n + 1):
                homes = [cell for cell in line if v in dom[cell]]
                if len(homes) == 1 and len(dom[homes[0]]) > 1:
                    dom[homes[0]] = {v}; changed = True
        # Cage feasibility: prune values not in any consistent cage assignment
        for cage, (typ, tgt) in zip(cages, clues):
            doms = [sorted(dom[cell]) for cell in cage]
            support = [set() for _ in cage]
            # cap enumeration to keep it cheap
            if 1:
                for combo in itertools.product(*doms):
                    # AllDiff within cage for cells sharing a row or col
                    ok = True
                    for i in range(len(cage)):
                        for j in range(i + 1, len(cage)):
                            if combo[i] == combo[j] and (cage[i][0] == cage[j][0] or cage[i][1] == cage[j][1]):
                                ok = False; break
                        if not ok:
                            break
                    if ok and cage_consistent(typ, tgt, combo):
                        for i, v in enumerate(combo):
                            support[i].add(v)
            for i, cell in enumerate(cage):
                if dom[cell] != support[i] and support[i]:
                    if dom[cell] - support[i]:
                        dom[cell] = dom[cell] & support[i]; changed = True
        if not changed:
            break
    solved = all(len(d) == 1 for d in dom.values())
    return rounds, solved, dom


def main():
    rng = random.Random(SEED)
    results = {}
    for n in N_SIZES:
        rounds_list, solved_list = [], []
        for _ in range(PUZZLES_PER_N):
            sol = latin_square(n, rng)
            cages = make_cages(n, rng)
            clues = [cage_clue(cage, sol, rng) for cage in cages]
            # avoid trivial all-given by capping size-1 cages
            r, solved, _ = propagate(n, cages, clues)
            rounds_list.append(r); solved_list.append(solved)
        rl = sorted(rounds_list)
        med = rl[len(rl) // 2]
        solve_rate = sum(solved_list) / len(solved_list)
        round_hist = dict(sorted(Counter(rounds_list).items()))
        # rounds among SOLVED puzzles (the meaningful deduction depth)
        solved_rounds = sorted([r for r, s in zip(rounds_list, solved_list) if s])
        med_solved = solved_rounds[len(solved_rounds) // 2] if solved_rounds else None
        results[n] = {"median_rounds": med, "median_rounds_solved": med_solved,
                      "prop_solve_rate": solve_rate, "round_hist": round_hist,
                      "frac_round1": round_hist.get(1, 0) / len(rounds_list),
                      "frac_ge3": sum(v for k, v in round_hist.items() if k >= 3) / len(rounds_list)}
        print(f"N={n}: median_rounds={med} (solved-only={med_solved}) prop_solve_rate={solve_rate:.2f} "
              f"frac_round1={results[n]['frac_round1']:.2f} frac_ge3rounds={results[n]['frac_ge3']:.2f} "
              f"hist={round_hist}", flush=True)

    # binding read on the largest board (deepest deduction)
    big = results[N_SIZES[-1]]
    if big["median_rounds"] >= 3 and big["prop_solve_rate"] >= 0.60:
        verdict = "DEEP / loop-only — BUILD (native iterative relevance present)"
    elif big["prop_solve_rate"] < 0.60:
        verdict = "SEARCH-NEEDED — propagation alone leaves ambiguity (loop+search, MCTS gated)"
    else:
        verdict = "SHALLOW / one-pass — calculator risk (move to larger boards / tighter cages)"
    print(f"\nBINDING (N={N_SIZES[-1]}): median_rounds={big['median_rounds']} "
          f"prop_solve_rate={big['prop_solve_rate']:.2f} -> {verdict}")
    print("BANDS: DEEP = med_rounds>=3 AND prop_solve>=0.60; SEARCH-NEEDED = prop_solve<0.60; else SHALLOW")

    out = {"metric": "kenken_precheck_deduction_depth", "seed": SEED,
           "puzzles_per_n": PUZZLES_PER_N, "by_N": results, "verdict": verdict,
           "bands_pre_committed": "DEEP med_rounds>=3 & prop_solve>=0.60; SEARCH-NEEDED prop_solve<0.60; else SHALLOW"}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
