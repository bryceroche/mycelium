"""test_kenken_parity.py — the PHASE-2 KENKEN CORRECTNESS GATES (CPU, GPU-free).

Phase 2 ports KenKen to the GENERAL predicate-driven CSP core (mycelium/csp_core.py +
csp_registry.py, shipped Phase 0) with ZERO edits to those two modules — KenKen lives
ONLY in mycelium/csp_domains.py (cage_pred + all_diff_pred + l_alldiff_propagator +
problem_from_kenken + kenken_registry) + this test + scripts/search_kenken.py. The
specialized_propagator dispatch slot the all-different propagator plugs into ALREADY
EXISTS in csp_core; the predicate seam is the only other extension point. This file is
the proof the port is CORRECT and SOUND. It runs three gates (spec §6.2):

  GATE-1  CAGE GAC PARITY.  Generic gac_propagate on cage_pred (for cages under the
          arity cap) == build_kenken_data.propagate's per-cage support-intersection,
          on a fixture set of restricted-domain states:
            * LIVE cage (every cell keeps support): byte-identical domain pruning.
            * DEAD cage (some cell loses all support): GAC EMPTIES the domain (detects
              the conflict) while propagate leaves the stale domain — GAC is sound +
              STRONGER. We assert deadness DETECTION there, not residual identity (the
              SAME live/dead distinction the coloring gac-vs-ac3 gate makes — the cage
              relation is compared on live branches; deadness is a soundness property).

  GATE-2  ALL-DIFFERENT SOUNDNESS.  l_alldiff_propagator vs a BRUTE-FORCE oracle: every
          value it prunes is absent from ALL valid all-different completions; it NEVER
          prunes a value present in SOME completion. Exercised in BOTH regimes:
            * PERMUTATION (len(scope)==|value universe|, e.g. n cells over 1..n): the
              KenKen ROW/COL case, where naked-singles + Hall value-occurrence/hidden-
              single + Hall intervals are ALL sound.
            * NON-PERMUTATION (len(scope) < |value universe|, e.g. 2-3 cells over 1..5):
              not every value need appear, so hidden-single is UNSOUND and is GUARDED
              OFF; only naked-singles + Hall intervals fire. This regime is what catches
              the old unguarded Rule 2 (it would prune solution-bearing values here).
          The propagator is therefore a SOUND all-different GAC propagator for ANY scope
          (naked-singles + Hall-intervals always; value-occurrence/hidden-single only in
          the permutation regime). Plus: it actually PRUNES (non-vacuous) and cage_pred is
          HOLE-MONOTONE at registration.

  GATE-3  GENERALITY.  Symbolic search (B1 no-prop+MRV+LCV; B3 GAC+all-diff+MRV+LCV)
          SOLVES a fixture set of unique KenKen instances across difficulty bands,
          VERIFIED by the EXACT cage_ok + all-different verifier (NOT the generator's
          own solver output, NOT gold-match). B3's propagation collapses the tree
          (fewer decisions than B1).

Run:  SELFTEST_ONLY=1 .venv/bin/python3 scripts/test_kenken_parity.py
GPU-FREE: pure python + numpy-optional. ast.parse clean, CPU import clean.
"""

from __future__ import annotations

import ast
import itertools
import os
import random
import sys

_THIS = os.path.abspath(__file__)
sys.path.insert(0, os.path.dirname(os.path.dirname(_THIS)))
sys.path.insert(0, os.path.dirname(_THIS))   # scripts/ for build_kenken_data

import build_kenken_data as bk  # noqa: E402
from mycelium.csp_core import (  # noqa: E402
    UNASSIGNED,
    Consistency,
    Factor,
    Problem,
    assert_hole_monotone,
    backtrack_search,
    gac_propagate,
    lcv_valorder,
    make_initial_state,
    mrv_varorder,
    noop_propagate,
    solve_symbolic,
)
from mycelium.csp_domains import (  # noqa: E402
    CAGE_OPS,
    LTYPE_CAGE,
    OP_TO_ID,
    _cell_id,
    cage_pred,
    kenken_registry,
    l_alldiff_propagator,
    problem_from_kenken,
)


# ===========================================================================
# THE EXACT KENKEN VERIFIER (cage_ok + all-different) — the success arbiter
# ===========================================================================
# GATE-3 success = THIS verifier passes, NOT the generator's solver output and NOT
# gold-match. Any assignment satisfying every cage's cage_ok AND row/col all-different
# is a proper KenKen solution (uniqueness is the corpus's separate guarantee).

def verify_kenken_solution(n, cages, clues, assignment) -> bool:
    """assignment: list[n*n] of values (row-major). True iff every row + col is a
    permutation of 1..n AND every cage satisfies cage_ok(op, target, vals)."""
    if any(a == UNASSIGNED for a in assignment[: n * n]):
        return False
    grid = [[assignment[_cell_id(r, c, n)] for c in range(n)] for r in range(n)]
    full = set(range(1, n + 1))
    for r in range(n):
        if set(grid[r]) != full:
            return False
    for c in range(n):
        if set(grid[r][c] for r in range(n)) != full:
            return False
    for cage, clue in zip(cages, clues):
        op, tgt = clue[0], int(clue[1])
        vals = tuple(grid[r][c] for (r, c) in cage)
        if not bk.cage_ok(op, tgt, vals):
            return False
    return True


def _cages_clues(rec):
    """Normalize a build_kenken_data record's cages (lists of [r,c]) to tuples."""
    cages = [[tuple(cell) for cell in cage] for cage in rec["cages"]]
    clues = [tuple(cl) for cl in rec["clues"]]
    return cages, clues


# ===========================================================================
# GATE-1 — CAGE GAC PARITY vs build_kenken_data.propagate
# ===========================================================================

def _ref_cage_support(n, cage, clue, dom):
    """build_kenken_data.propagate's per-cage support-intersection (the cage RELATION,
    no interleaved line-propagation, no same-line all-different check — that strengthening
    belongs to the ROW/COL all-different factors, not the cage). Returns (out, live):
      out[cell]  = dom[cell] & support[cell]      (the cage's own pruning)
      live       = True iff EVERY cell keeps >=1 support (propagate only prunes on
                   non-empty support; an empty-support cell is a conflict GAC must catch).
    None if the cage blows the arity cap (propagate skips it; so does generic GAC)."""
    typ, tgt = clue
    doms = [sorted(dom[cell]) for cell in cage]
    if bk._prod([len(d) for d in doms]) > 20000:
        return None
    support = [set() for _ in cage]
    for combo in itertools.product(*doms):
        if bk.cage_ok(typ, tgt, combo):
            for i, v in enumerate(combo):
                support[i].add(v)
    live = all(len(s) > 0 for s in support)
    out = {cell: (dom[cell] & support[i]) for i, cell in enumerate(cage)}
    return out, live


def _gac_one_cage(n, cage, clue, base):
    """Generic gac_propagate over a Problem with ONLY this cage's cage_pred factor,
    starting from per-cell domains `base`. Returns (out_by_cell, dead)."""
    typ, tgt = clue
    from mycelium.csp_registry import new_registry, register
    reg = new_registry()
    register(reg, LTYPE_CAGE, cage_pred, name="cage", arity_hint=4)
    cellids = [_cell_id(r, c, n) for (r, c) in cage]
    factors = [Factor(ftype=LTYPE_CAGE, scope=tuple(cellids),
                      params=(OP_TO_ID[typ], tgt))]
    nv = n * n
    var_factors = [[] for _ in range(nv)]
    for fi, f in enumerate(factors):
        for u in f.scope:
            var_factors[u].append(fi)
    prob = Problem(n_vars=nv, domains0=[set(range(1, n + 1)) for _ in range(nv)],
                   factors=factors, var_factors=var_factors, registry=reg)
    st = make_initial_state(prob)
    for r in range(n):
        for c in range(n):
            st.domains[_cell_id(r, c, n)] = set(base[(r, c)])
    st.values = [UNASSIGNED] * nv
    st.n_assigned = 0
    out = gac_propagate(st)
    dead = any(len(out.domains[cid]) == 0 for cid in cellids)
    return {cell: out.domains[_cell_id(*cell, n)] for cell in cage}, dead


def _gate1_cage_parity(seed=3, n_puzzles=120):
    """Returns (live_checked, live_mism, dead_total, dead_detected)."""
    rng = random.Random(seed)
    live_checked = live_mism = dead_total = dead_detected = 0
    made = 0
    while made < n_puzzles:
        n = rng.choice([5, 6, 7])
        rec = bk.gen_unique_puzzle_banded(n, rng, rng.choice(["g40", "g20", "g10"]),
                                          tries=20)
        if rec is None:
            continue
        made += 1
        cages, clues = _cages_clues(rec)
        # a random mid-search restricted-domain state (each cell maybe pruned)
        base = {(r, c): set(range(1, n + 1)) for r in range(n) for c in range(n)}
        for cell in base:
            if rng.random() < 0.3:
                keep = {v for v in range(1, n + 1) if rng.random() < 0.7} or \
                       {rng.choice(range(1, n + 1))}
                base[cell] = keep
        for cage, clue in zip(cages, clues):
            if clue[0] == "given" or len(cage) > 4:
                continue
            r = _ref_cage_support(n, cage, clue, base)
            if r is None:
                continue
            ref, live = r
            gac, dead = _gac_one_cage(n, cage, clue, base)
            if live:
                live_checked += 1
                if any(gac[cell] != ref[cell] for cell in cage):
                    live_mism += 1
            else:
                dead_total += 1
                if dead:
                    dead_detected += 1
    return live_checked, live_mism, dead_total, dead_detected


# ===========================================================================
# GATE-2 — ALL-DIFFERENT SOUNDNESS vs a brute-force oracle (BOTH regimes)
# ===========================================================================

def _brute_alldiff_keep(domains):
    """Per-position set of values appearing in SOME valid all-different completion of the
    given per-cell domains (the soundness oracle). domains: list of sets."""
    k = len(domains)
    keep = [set() for _ in range(k)]
    for combo in itertools.product(*[sorted(d) for d in domains]):
        if len(set(combo)) == k:
            for i, v in enumerate(combo):
                keep[i].add(v)
    return keep


def _run_alldiff_propagator(domains):
    """Run l_alldiff_propagator on a single all-different scope with the given per-cell
    `domains` (list of sets). The value universe is sized to cover every value present so
    the registry's check_alphabet spans it. Returns the propagated per-cell domains."""
    from mycelium.csp_domains import LTYPE_ROW
    n = len(domains)
    universe = max((max(d) for d in domains if d), default=n)
    reg = kenken_registry(max(universe, n))
    f = Factor(ftype=LTYPE_ROW, scope=tuple(range(n)), params=None)
    prob = Problem(n_vars=n, domains0=[set(d) for d in domains], factors=[f],
                   var_factors=[[0]] * n, registry=reg)
    st = make_initial_state(prob)
    st.values = [UNASSIGNED] * n
    st.domains = [set(d) for d in domains]
    st.n_assigned = 0
    return l_alldiff_propagator(st, f).domains


def _gate2_one(domains):
    """Soundness of one propagator run vs the brute oracle. Returns (unsound, pruned):
    unsound == 1 iff ANY pruned value is present in SOME valid completion."""
    out = _run_alldiff_propagator(domains)
    keep = _brute_alldiff_keep(domains)
    pruned = 0
    for i in range(len(domains)):
        removed = domains[i] - out[i]
        if removed & keep[i]:
            return 1, pruned
        pruned += len(removed)
    return 0, pruned


def _gate2_alldiff_soundness(seed=0, perm_trials=5000, nonperm_trials=5000):
    """Soundness vs the brute oracle in BOTH regimes (returns a stats dict).

      * PERMUTATION: n cells over exactly 1..n (len(scope)==|value universe|) — the
        KenKen ROW/COL case where every value MUST be placed (hidden-single is sound).
      * NON-PERMUTATION: 2-3 cells with domains drawn from a LARGER value set (1..5),
        so len(scope) < |value universe| — not every value need appear, the regime where
        the OLD unguarded Rule 2 LOST solutions. The guard must leave 0 unsound prunes.

    Both must show 0 unsound prunes; both must witness non-vacuity (pruned > 0)."""
    rng = random.Random(seed)
    stats = {"perm_cases": 0, "perm_unsound": 0, "perm_pruned": 0,
             "nonperm_cases": 0, "nonperm_unsound": 0, "nonperm_pruned": 0}
    # --- PERMUTATION regime: n cells over exactly 1..n ---
    for _ in range(perm_trials):
        n = rng.randint(3, 5)
        vals = list(range(1, n + 1))
        domains = [({v for v in vals if rng.random() < 0.6} or {rng.choice(vals)})
                   for _ in range(n)]
        u, p = _gate2_one(domains)
        stats["perm_cases"] += 1
        stats["perm_unsound"] += u
        if not u:
            stats["perm_pruned"] += p
    # --- NON-PERMUTATION regime: 2-3 cells, domains drawn from a LARGER set 1..5 ---
    big = list(range(1, 6))  # value universe 1..5
    for _ in range(nonperm_trials):
        n = rng.randint(2, 3)                       # len(scope) < |universe| (=5)
        domains = [({v for v in big if rng.random() < 0.55} or {rng.choice(big)})
                   for _ in range(n)]
        u, p = _gate2_one(domains)
        stats["nonperm_cases"] += 1
        stats["nonperm_unsound"] += u
        if not u:
            stats["nonperm_pruned"] += p
    return stats


# ===========================================================================
# GATE-3 — GENERALITY: B1 + B3 SOLVE KenKen, verified by the exact verifier
# ===========================================================================

def _solve_b1(prob, budget):
    return backtrack_search(prob, propagate_fn=noop_propagate,
                            varorder_fn=mrv_varorder, valorder_fn=lcv_valorder,
                            budget=budget, can_certify_unsat=True)


def _solve_b3(prob, budget):
    return solve_symbolic(prob, budget=budget)


def _gate3_generality(seed=7, per_band=3, budget=200000):
    """Returns per-band stats dicts + totals for B1 and B3."""
    rng = random.Random(seed)
    by_band = {}
    for band in bk.BAND_ORDER:
        cnt = 0
        rows = {"b1_solved": 0, "b1_dec": [], "b3_solved": 0, "b3_dec": [], "n": 0}
        while cnt < per_band:
            n = rng.choice([5, 6, 7])
            rec = bk.gen_unique_puzzle_banded(n, rng, band, tries=40)
            if rec is None:
                continue
            cnt += 1
            cages, clues = _cages_clues(rec)
            rows["n"] += 1
            r3 = _solve_b3(problem_from_kenken(n, cages, clues), budget)
            ok3 = (r3["status"] == "solved"
                   and verify_kenken_solution(n, cages, clues, r3["assignment"]))
            rows["b3_solved"] += int(ok3)
            if ok3:
                rows["b3_dec"].append(r3["decisions"])
            r1 = _solve_b1(problem_from_kenken(n, cages, clues), budget)
            ok1 = (r1["status"] == "solved"
                   and verify_kenken_solution(n, cages, clues, r1["assignment"]))
            rows["b1_solved"] += int(ok1)
            if ok1:
                rows["b1_dec"].append(r1["decisions"])
        by_band[band] = rows
    return by_band


# ===========================================================================
# THE SELFTEST DRIVER
# ===========================================================================

def _selftest() -> bool:
    all_ok = True

    def _check(name, cond):
        nonlocal all_ok
        if not cond:
            all_ok = False
        print(f"[kenken-parity] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    # --- L-ASYM documentation check: cage_pred is COMMUTATIVE for sub/div (both
    # orders give the SAME verdict — no ordered-scope convention needed). ---
    sub_p = (OP_TO_ID["sub"], 2)
    div_p = (OP_TO_ID["div"], 3)
    _check("L-ASYM: sub cage_pred commutative (|5-3|==|3-5|==2)",
           cage_pred(LTYPE_CAGE, sub_p, (5, 3)) == cage_pred(LTYPE_CAGE, sub_p, (3, 5))
           == Consistency.SAT)
    _check("L-ASYM: div cage_pred commutative (6/2==2/6-as-quotient-3? -> 6,2 SAT)",
           cage_pred(LTYPE_CAGE, div_p, (6, 2)) == cage_pred(LTYPE_CAGE, div_p, (2, 6))
           == Consistency.SAT)

    # --- L-PARAM: cage_pred uses the EXACT integer target via cage_ok ---
    _check("L-PARAM: add complete SAT 2+4==6",
           cage_pred(LTYPE_CAGE, (OP_TO_ID["add"], 6), (2, 4)) == Consistency.SAT)
    _check("L-PARAM: add complete VIOLATED 2+3!=6",
           cage_pred(LTYPE_CAGE, (OP_TO_ID["add"], 6), (2, 3)) == Consistency.VIOLATED)
    _check("L-PARAM: mul complete SAT 3*4==12",
           cage_pred(LTYPE_CAGE, (OP_TO_ID["mul"], 12), (3, 4)) == Consistency.SAT)

    # --- partial three-valued logic (the propagation-strength tier) ---
    _check("partial add extendable (2+?<=6 reachable -> UNVIOLATED)",
           cage_pred(LTYPE_CAGE, (OP_TO_ID["add"], 6), (2, UNASSIGNED))
           == Consistency.UNVIOLATED)
    _check("partial add dead (5+1+1=7>6 -> VIOLATED)",
           cage_pred(LTYPE_CAGE, (OP_TO_ID["add"], 6),
                     (5, UNASSIGNED, UNASSIGNED)) == Consistency.VIOLATED)
    _check("partial mul non-divisor dead (5 does not divide 12 -> VIOLATED)",
           cage_pred(LTYPE_CAGE, (OP_TO_ID["mul"], 12), (5, UNASSIGNED))
           == Consistency.VIOLATED)
    _check("partial mul overshoot dead (20>12 -> VIOLATED)",
           cage_pred(LTYPE_CAGE, (OP_TO_ID["mul"], 12), (20, UNASSIGNED))
           == Consistency.VIOLATED)
    _check("partial mul divisor live (3 divides 12 -> UNVIOLATED)",
           cage_pred(LTYPE_CAGE, (OP_TO_ID["mul"], 12), (3, UNASSIGNED))
           == Consistency.UNVIOLATED)

    # --- L-MONO: cage_pred hole-monotone for every op at op-valid arities ---
    mono_ok = True
    for op in ("add", "sub", "mul", "div"):
        arities = (2,) if op in ("sub", "div") else (2, 3, 4)
        for tgt in range(1, 16):
            for arity in arities:
                if not assert_hole_monotone(LTYPE_CAGE, (OP_TO_ID[op], tgt), cage_pred,
                                            tuple(range(1, 8)), arity, samples=300,
                                            seed=__import__("zlib").crc32(repr((op, tgt, arity)).encode()) % 9973):
                    mono_ok = False
    _check("L-MONO: cage_pred hole-monotone (all ops x targets x valid arities)", mono_ok)
    # wrong-arity sub/div degrade to VIOLATED + stay monotone (no crash on the sampler)
    _check("cage_pred wrong-arity sub -> VIOLATED + monotone",
           cage_pred(LTYPE_CAGE, (OP_TO_ID["sub"], 2), (3, 4, 5)) == Consistency.VIOLATED
           and assert_hole_monotone(LTYPE_CAGE, (OP_TO_ID["sub"], 2), cage_pred,
                                    tuple(range(1, 8)), 3, samples=200))

    # --- registration enforces L-MONO for the whole KenKen registry (raises if not) ---
    reg_ok = True
    try:
        for n in (5, 6, 7):
            kenken_registry(n)
    except ValueError:
        reg_ok = False
    _check("kenken_registry registers (L-MONO enforced at registration, no raise)",
           reg_ok)

    # =====================================================================
    # GATE-1 — cage GAC parity vs build_kenken_data.propagate
    # =====================================================================
    live_checked, live_mism, dead_total, dead_detected = _gate1_cage_parity()
    _check(f"GATE-1: live-cage GAC == propagate cage support "
           f"({live_checked} checked, {live_mism} mismatch)",
           live_mism == 0 and live_checked >= 200)
    _check(f"GATE-1: GAC detects every dead cage propagate leaves stale "
           f"({dead_detected}/{dead_total} emptied)",
           dead_detected == dead_total)

    # =====================================================================
    # GATE-2 — all-different propagator soundness vs brute-force oracle
    # (BOTH the permutation AND the non-permutation regime)
    # =====================================================================
    g2 = _gate2_alldiff_soundness()
    _check(f"GATE-2 [permutation]: l_alldiff SOUND vs brute oracle "
           f"({g2['perm_cases']} cases, {g2['perm_unsound']} unsound prunes)",
           g2["perm_unsound"] == 0 and g2["perm_cases"] >= 1000)
    _check(f"GATE-2 [permutation]: NON-VACUOUS (pruned {g2['perm_pruned']} values)",
           g2["perm_pruned"] > 0)
    _check(f"GATE-2 [non-permutation]: l_alldiff SOUND vs brute oracle "
           f"(len(scope)<|universe|; {g2['nonperm_cases']} cases, "
           f"{g2['nonperm_unsound']} unsound prunes)",
           g2["nonperm_unsound"] == 0 and g2["nonperm_cases"] >= 1000)
    _check(f"GATE-2 [non-permutation]: NON-VACUOUS (pruned {g2['nonperm_pruned']} "
           f"values via naked-singles/Hall-intervals)", g2["nonperm_pruned"] > 0)
    # THE OLD-BUG CATCHER: the verified counterexample. 2-cell all-different over
    # [{1,2},{2,3}] (universe {1,2,3}, len(scope)=2 < 3 -> NON-permutation). Valid
    # completions are (1,2),(1,3),(2,3) -> oracle keeps cell0={1,2}, cell1={2,3}. The OLD
    # unguarded Rule 2 forced value 1 into cell0 (single home) and 3 into cell1 -> [{1},
    # {2}], LOSING solutions. The guard must leave BOTH domains untouched (no prune).
    cex = [{1, 2}, {2, 3}]
    cex_out = _run_alldiff_propagator(cex)
    cex_keep = _brute_alldiff_keep(cex)
    _check("GATE-2 [non-permutation]: oracle keeps {1,2},{2,3} (sanity)",
           cex_keep == [{1, 2}, {2, 3}])
    _check("GATE-2 [old-bug catcher]: guarded Rule 2 does NOT prune {1,2},{2,3} "
           f"(got {cex_out[0]},{cex_out[1]}; old bug -> {{1}},{{2}})",
           cex_out[0] == {1, 2} and cex_out[1] == {2, 3})
    # a hand case (PERMUTATION): naked pair {1,2},{1,2} forces the third cell off 1 and 2
    from mycelium.csp_domains import LTYPE_ROW
    reg = kenken_registry(3)
    f = Factor(ftype=LTYPE_ROW, scope=(0, 1, 2), params=None)
    prob = Problem(n_vars=3, domains0=[{1, 2}, {1, 2}, {1, 2, 3}], factors=[f],
                   var_factors=[[0], [0], [0]], registry=reg)
    st = make_initial_state(prob)
    st.values = [UNASSIGNED] * 3
    st.domains = [{1, 2}, {1, 2}, {1, 2, 3}]
    st.n_assigned = 0
    out = l_alldiff_propagator(st, f)
    _check("GATE-2: naked-pair Hall prune {1,2},{1,2} -> third cell = {3}",
           out.domains[2] == {3})

    # =====================================================================
    # GATE-3 — generality: B1 + B3 SOLVE KenKen (exact verifier), B3 collapses tree
    # =====================================================================
    by_band = _gate3_generality()
    tot_n = sum(r["n"] for r in by_band.values())
    tot_b1 = sum(r["b1_solved"] for r in by_band.values())
    tot_b3 = sum(r["b3_solved"] for r in by_band.values())
    all_b1_dec = [d for r in by_band.values() for d in r["b1_dec"]]
    all_b3_dec = [d for r in by_band.values() for d in r["b3_dec"]]
    print("\n  GATE-3 per-band solve (verified by exact cage_ok + all-different):",
          flush=True)
    print(f"  {'band':<6} {'n':>3}  {'B1 solved':>10} {'B1 dec':>8}  "
          f"{'B3 solved':>10} {'B3 dec':>8}", flush=True)
    for band in bk.BAND_ORDER:
        r = by_band[band]
        b1d = (sum(r["b1_dec"]) / len(r["b1_dec"])) if r["b1_dec"] else float("nan")
        b3d = (sum(r["b3_dec"]) / len(r["b3_dec"])) if r["b3_dec"] else float("nan")
        print(f"  {band:<6} {r['n']:>3}  {r['b1_solved']:>10} {b1d:>8.1f}  "
              f"{r['b3_solved']:>10} {b3d:>8.1f}", flush=True)
    _check(f"GATE-3: B1 solves all unique instances ({tot_b1}/{tot_n})",
           tot_b1 == tot_n and tot_n >= 8)
    _check(f"GATE-3: B3 solves all unique instances ({tot_b3}/{tot_n})",
           tot_b3 == tot_n)
    mean_b1 = sum(all_b1_dec) / len(all_b1_dec) if all_b1_dec else float("nan")
    mean_b3 = sum(all_b3_dec) / len(all_b3_dec) if all_b3_dec else float("nan")
    _check(f"GATE-3: B3 (GAC+all-diff) collapses the tree vs B1 "
           f"(mean dec {mean_b3:.1f} <= {mean_b1:.1f})",
           mean_b3 <= mean_b1)

    print(f"\n[kenken-parity] {'ALL PASS' if all_ok else 'SOME FAILED'}", flush=True)
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
    print("set SELFTEST_ONLY=1 to run the KenKen parity + soundness + generality gates.",
          flush=True)
