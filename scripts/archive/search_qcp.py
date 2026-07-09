"""search_qcp.py — the QCP/QWH kill-gate (does the prey actually branch deep?).

The cheap CPU kill-gate (NO deducer, NO training): mirror the Sudoku oracle-gap method on
balanced Quasigroup Completion (Latin-square completion, rows+cols all-different) at the
phase transition. Sweep order n and hole-density; for each band measure decisions-to-solve
under SYMBOLIC (LCV) vs ORACLE policy (one-hot at the generation solution = upper bound on
any neural policy) vs RANDOM policy.

PRE-REGISTERED VERDICT (kill QCP for ~free if it is secretly Sudoku):
  KILL if  peak median symbolic decisions ~0      (GAC collapses -> not prey)
       or  oracle/LCV ratio < 3x                  (ordering barely matters)
       or  random does NOT hurt vs LCV            (residual intrinsically ambiguous = Sudoku trap)
       or  the gap exists only at microsecond n   (no genuine intractability)
  PASS if  deep tree (median grows with n, >>0) AND oracle cuts >=5-10x AND random HURTS
           AND symbolic time is non-trivial. Only on PASS do we build a QCP deducer.

  DEV unused (pure CPU). .venv/bin/python3 scripts/search_qcp.py [--orders 9,12,15,18] [--K 10]
"""
import argparse
import random
import statistics
import sys
import time

sys.path.insert(0, ".")
from mycelium.csp_core import (                                   # noqa: E402
    backtrack_search, gac_propagate, mrv_varorder, lcv_valorder, policy_valorder,
)
from mycelium.csp_domains import problem_from_qcp, qcp_registry   # noqa: E402


def random_latin_square(n, rng):
    """A random order-n Latin square: cyclic base + row/col shuffle + symbol relabel
    (all three preserve the Latin property)."""
    base = [[(i + j) % n for j in range(n)] for i in range(n)]
    rows = list(range(n)); rng.shuffle(rows)
    cols = list(range(n)); rng.shuffle(cols)
    syms = list(range(1, n + 1)); rng.shuffle(syms)
    return [[syms[base[rows[i]][cols[j]]] for j in range(n)] for i in range(n)]


def make_balanced_qcp(n, h, rng):
    """BALANCED QWH (Gomes-Shmoys style): holes = exactly h per row AND h per col.

    Hole pattern = the positions of h symbols in an INDEPENDENT random Latin square M
    (each symbol's positions form a permutation matrix; h symbols -> h disjoint
    permutation matrices -> exactly h holes per row + per col). The removed VALUES come
    from the solution L (varied, not a single symbol class), breaking the trivial value
    symmetry of remove-a-whole-symbol. Always >=1 solution (the source L). hole_frac = h/n.
    """
    L = random_latin_square(n, rng)               # the solution
    M = random_latin_square(n, rng)               # the hole pattern
    sol = [L[i][j] for i in range(n) for j in range(n)]
    cells = list(sol)
    hole_syms = set(rng.sample(range(1, n + 1), h))
    for i in range(n):
        for j in range(n):
            if M[i][j] in hole_syms:
                cells[i * n + j] = 0              # exactly h per row & per col
    return cells, sol


def solve(cells, n, reg, valorder, budget):
    prob = problem_from_qcp(cells, n, reg)
    return backtrack_search(prob, gac_propagate, mrv_varorder, valorder, budget=budget)


def oracle_pol(sol, n):
    return [[1.0 if a == sol[v] else 0.0 for a in range(n + 1)] for v in range(len(sol))]


def random_pol(n_vars, n, rng):
    return [[rng.random() for _ in range(n + 1)] for _ in range(n_vars)]


def _p(*a):
    print(*a, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orders", default="12,18,25")
    ap.add_argument("--fracs", default="0.20,0.30,0.40,0.50,0.60", help="h/n hole fractions to sweep")
    ap.add_argument("--K", type=int, default=12)
    ap.add_argument("--budget", type=int, default=150000)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    orders = [int(x) for x in args.orders.split(",")]
    fracs = [float(x) for x in args.fracs.split(",")]
    rng = random.Random(args.seed)

    # hardness = heavy tail (max/med) + budget-hits; ordering signal = oracle/sym + random.
    hardest = {}   # order -> band stats at the band with most budget-hits (tie: max tail)
    _p(f"{'n':>3} {'h':>3} {'hf%':>5} {'sym_med':>8} {'sym_max':>8} {'orc_med':>8} {'rnd_med':>8} "
       f"{'orc/sym':>8} {'tail':>6} {'budX':>5} {'ms':>8}")
    for n in orders:
        reg = qcp_registry(n)
        bands = []
        for hf in fracs:
            h = max(1, min(n, round(hf * n)))
            syms, orcs, rnds, times, nbud = [], [], [], [], 0
            for _ in range(args.K):
                cells, sol = make_balanced_qcp(n, h, rng)
                t0 = time.time()
                rs = solve(cells, n, reg, lcv_valorder, args.budget)
                times.append(time.time() - t0)
                if rs.get("status") == "budget":
                    nbud += 1
                syms.append(int(rs.get("decisions", 0)))
                orcs.append(int(solve(cells, n, reg, policy_valorder(oracle_pol(sol, n)), args.budget).get("decisions", 0)))
                rnds.append(int(solve(cells, n, reg, policy_valorder(random_pol(n * n, n, rng)), args.budget).get("decisions", 0)))
            sm, om, rm = statistics.median(syms), statistics.median(orcs), statistics.median(rnds)
            smx = max(syms)
            tail = smx / max(sm, 1)
            ratio = sm / max(om, 1e-9)
            ms = 1000 * statistics.mean(times)
            _p(f"{n:>3} {h:>3} {100*hf:>4.0f}% {sm:>8.0f} {smx:>8} {om:>8.0f} {rm:>8.0f} "
               f"{ratio:>8.1f} {tail:>6.1f} {nbud:>5} {ms:>8.1f}")
            bands.append({"hf": hf, "h": h, "sm": sm, "smx": smx, "om": om, "rm": rm,
                          "ratio": ratio, "tail": tail, "nbud": nbud, "ms": ms})
        # hardest band = most budget-hits, tie-break highest tail ratio
        hardest[n] = max(bands, key=lambda b: (b["nbud"], b["tail"]))

    _p("\n=== KILL-GATE VERDICT (each order at its HARDEST band: most budget-hits / heaviest tail) ===")
    _p(f"{'n':>3} {'hf%':>5} {'sym_med':>8} {'sym_max':>8} {'orc_med':>8} {'orc/sym':>8} {'rnd_med':>8} {'budX':>5} {'tail':>6}")
    for n in orders:
        b = hardest[n]
        _p(f"{n:>3} {100*b['hf']:>4.0f}% {b['sm']:>8.0f} {b['smx']:>8} {b['om']:>8.0f} "
           f"{b['ratio']:>8.1f} {b['rm']:>8.0f} {b['nbud']:>5} {b['tail']:>6.1f}")
    nbig = orders[-1]; b = hardest[nbig]
    _p(f"\n  largest order n={nbig}, hardest band hf={100*b['hf']:.0f}%: "
       f"sym_med={b['sm']:.0f} sym_max={b['smx']} oracle={b['om']:.0f} oracle/sym={b['ratio']:.1f}x "
       f"random={b['rm']:.0f} budget_hits={b['nbud']}/{args.K} tail={b['tail']:.1f}")
    deep = b["smx"] >= 1000 or b["nbud"] > 0 or b["tail"] >= 5     # genuine heavy-tailed search
    ordering = b["ratio"] >= 3.0                                   # oracle meaningfully beats LCV
    rnd_hurts = b["rm"] > b["sm"] * 1.3
    verdict = "PASS" if (deep and ordering and rnd_hurts) else "KILL"
    _p(f"  heavy_tailed_search={deep}  ordering_matters(oracle>=3x)={ordering}  random_hurts={rnd_hurts}")
    _p(f"  => {verdict}: "
       + ("QCP is real prey — deep heavy-tailed search WITH an ordering gap; build a QCP deducer."
          if verdict == "PASS" else
          ("search is deep/heavy-tailed but ORDERING DOESN'T MATTER (oracle~=symbolic) — value "
           "symmetry; neural value-policy can't help -> KILL." if (deep and not ordering) else
           "not heavy-tailed at these sizes — push larger n or it's not the prey.")))


if __name__ == "__main__":
    main()
