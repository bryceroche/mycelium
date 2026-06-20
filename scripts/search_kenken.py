"""search_kenken.py — the KENKEN SYMBOLIC GENERALITY EVAL (Phase 2; CPU-only).

The Mycelium general search tier (mycelium/csp_core.py + csp_registry.py, shipped
Phase 0) was built to be domain-agnostic: a problem is just typed factors + a
three-valued predicate per factor type. Phase 0 validated it on graph coloring (the
not-equal relation). Phase 2 is the REAL generality test — KenKen is STRUCTURALLY
UNLIKE coloring: param-carrying n-ary arithmetic cages (op=target factors) PLUS
row/col ALL-DIFFERENT over up to 7 cells. The claim under test (the "one-trick-pony
guarantee"): the SAME core solves KenKen with ZERO csp_core / csp_registry edits, by
adding ONLY a predicate set + a bridge + the all-different specialized propagator, all
in mycelium/csp_domains.py.

SCOPE — SYMBOLIC ONLY (no neural ordering). There is no KenKen factor-graph deducer
checkpoint (fg_ckpts has coloring + circuit only), so the neural-ordering arm (the
search_coloring "B2") is GATED on training a KenKen deducer (a FUTURE step) and is NOT
run here. Phase 2 = does SYMBOLIC systematic search SOLVE KenKen through the new
predicate + bridge? Two configs, both pure general-core functions (no KenKen-specific
search code):

  B1  no-propagation  : noop_propagate + mrv_varorder + lcv_valorder        [search-only]
  B3  full symbolic   : gac_propagate (cage GAC + the L-ALLDIFF all-different
                        specialized propagator) + mrv_varorder + lcv_valorder [ceiling]

METRICS (per difficulty band g40/g30/g20/g10, easy->hard by givens density), success =
the EXACT KenKen verifier (cage_ok + all-different) passes — NOT the generator's solver
output, NOT gold-match:
  * solve_rate@budget    : fraction solved within the decision budget.
  * decisions_to_solve   : mean/median decision-nodes on solved instances (the
                           cross-config work unit — does GAC + all-different collapse
                           the tree the way it does for coloring?).
  * backtracks           : mean backtracks per instance.
B3's propagation should solve more / with FEWER decisions than B1, widening on the
harder (lower-givens, deeper) bands — the generality payoff.

GPU-FREE: pure python + numpy-optional. SELFTEST_ONLY=1 runs B1 + B3 on a small fixed
fixture set on CPU (no corpus generation cost), proving the wiring solves + verifies.

RUN COMMANDS (CPU)
------------------
  SELFTEST_ONLY=1 .venv/bin/python3 scripts/search_kenken.py        # CPU smoke
  KENKEN_PER_BAND=20 KENKEN_BUDGET=200000 .venv/bin/python3 scripts/search_kenken.py
"""

from __future__ import annotations

import ast
import math
import os
import random
import sys

_THIS = os.path.abspath(__file__)
sys.path.insert(0, os.path.dirname(os.path.dirname(_THIS)))
sys.path.insert(0, os.path.dirname(_THIS))   # scripts/ for build_kenken_data

import build_kenken_data as bk  # noqa: E402
from mycelium.csp_core import (  # noqa: E402
    backtrack_search,
    lcv_valorder,
    mrv_varorder,
    noop_propagate,
    solve_symbolic,
)
from mycelium.csp_domains import problem_from_kenken  # noqa: E402

# The exact KenKen verifier (cage_ok + all-different) lives in the parity test — reuse it
# as the SINGLE success arbiter so the eval and the gate agree on "solved".
from test_kenken_parity import verify_kenken_solution, _cages_clues  # noqa: E402


CONFIGS = ["B1", "B3"]
CONFIG_DESC = {
    "B1": "search, NO propagation (noop) + MRV + LCV                    [search-only]",
    "B3": "full symbolic: GAC (cage + L-ALLDIFF all-different) + MRV+LCV [ceiling]",
}


# ===========================================================================
# RUN ONE CONFIG ON ONE INSTANCE (both configs share the ONE general skeleton)
# ===========================================================================

def run_config_on_instance(config, n, cages, clues, budget, seed=0):
    """Run ONE config on ONE KenKen instance. Returns the skeleton dict augmented with
    'solved' (bool) per the EXACT cage_ok + all-different verifier. Both configs share
    backtrack_search; only the propagate plug-in differs (fairness by construction)."""
    prob = problem_from_kenken(n, cages, clues)
    if config == "B3":
        res = solve_symbolic(prob, budget=budget, seed=seed)
    elif config == "B1":
        res = backtrack_search(
            prob, propagate_fn=noop_propagate, varorder_fn=mrv_varorder,
            valorder_fn=lcv_valorder, budget=budget, seed=seed,
            can_certify_unsat=True,        # B1 is complete -> may certify unsat
        )
    else:
        raise ValueError(f"unknown config {config!r}")
    res["solved"] = (res["status"] == "solved"
                     and verify_kenken_solution(n, cages, clues, res["assignment"]))
    return res


# ===========================================================================
# METRICS AGGREGATION (mirrors search_coloring._new_agg / _summarise_agg)
# ===========================================================================

def _new_agg():
    return {"n": 0, "n_solved": 0, "decisions_solved": [], "backtracks_all": []}


def _summarise_agg(agg):
    import numpy as np
    n, ns = agg["n"], agg["n_solved"]
    dec, bt = agg["decisions_solved"], agg["backtracks_all"]
    return {
        "n": n,
        "solve_rate": (ns / n) if n else 0.0,
        "n_solved": ns,
        "decisions_mean": (float(np.mean(dec)) if dec else float("nan")),
        "decisions_median": (float(np.median(dec)) if dec else float("nan")),
        "backtracks_mean": (float(np.mean(bt)) if bt else float("nan")),
    }


def _print_band_table(results_by_band):
    """results_by_band[band][config] = summarised agg dict. solve_rate +
    decisions_to_solve + backtracks per band per config."""
    print(f"\n{'=' * 78}", flush=True)
    print("  KENKEN SYMBOLIC GENERALITY GRID — by givens-density band (easy -> hard)",
          flush=True)
    print("  success = EXACT KenKen verifier (cage_ok + all-different), NOT gold-match",
          flush=True)
    print(f"{'=' * 78}", flush=True)

    bands = [b for b in bk.BAND_ORDER if b in results_by_band]

    def _block(title, key, fmt):
        print(f"\n  {title}", flush=True)
        hdr = f"  {'band':<6} " + " ".join(f"{c:>10}" for c in CONFIGS)
        print(hdr, flush=True)
        print("  " + "-" * (len(hdr) - 2), flush=True)
        for band in bands:
            cells = []
            for c in CONFIGS:
                s = results_by_band[band].get(c)
                if s is None:
                    cells.append(f"{'-':>10}")
                else:
                    val = s[key]
                    cells.append(f"{'-':>10}" if (isinstance(val, float)
                                 and math.isnan(val)) else fmt.format(val))
            print(f"  {band:<6} " + " ".join(cells), flush=True)

    _block("solve_rate @ budget", "solve_rate", "{:10.3f}")
    _block("decisions_to_solve (mean on solved)", "decisions_mean", "{:10.1f}")
    _block("decisions_to_solve (median on solved)", "decisions_median", "{:10.1f}")
    _block("backtracks (mean per instance)", "backtracks_mean", "{:10.1f}")
    print("", flush=True)


# ===========================================================================
# CPU CORPUS DRIVER (no GPU, no checkpoint — pure symbolic search)
# ===========================================================================

def run_eval():
    """Generate a small banded corpus on the fly (uniqueness-verified by the generator)
    and run B1 + B3 over it, reporting per-band solve_rate + decisions + backtracks."""
    per_band = int(os.environ.get("KENKEN_PER_BAND", "20"))
    budget = int(os.environ.get("KENKEN_BUDGET", "200000"))
    seed = int(os.environ.get("KENKEN_SEED", "42"))
    sizes = [int(x) for x in os.environ.get("KENKEN_SIZES", "5,6,7").split(",")]

    print(f"KenKen symbolic generality eval: per_band={per_band} budget={budget} "
          f"sizes={sizes} seed={seed}", flush=True)
    for c in CONFIGS:
        print(f"  {c}: {CONFIG_DESC[c]}", flush=True)

    rng = random.Random(seed)
    results_by_band = {}
    for band in bk.BAND_ORDER:
        aggs = {c: _new_agg() for c in CONFIGS}
        made = 0
        while made < per_band:
            n = rng.choice(sizes)
            rec = bk.gen_unique_puzzle_banded(n, rng, band, tries=40)
            if rec is None:
                continue
            made += 1
            cages, clues = _cages_clues(rec)
            for c in CONFIGS:
                res = run_config_on_instance(c, n, cages, clues, budget, seed=seed)
                aggs[c]["n"] += 1
                if res["solved"]:
                    aggs[c]["n_solved"] += 1
                    aggs[c]["decisions_solved"].append(res["decisions"])
                aggs[c]["backtracks_all"].append(res["backtracks"])
        results_by_band[band] = {c: _summarise_agg(aggs[c]) for c in CONFIGS}
        # progress line per band
        b1, b3 = results_by_band[band]["B1"], results_by_band[band]["B3"]
        print(f"  [{band}] B1 solve {b1['solve_rate']:.2f} dec {b1['decisions_mean']:.1f}"
              f" | B3 solve {b3['solve_rate']:.2f} dec {b3['decisions_mean']:.1f}",
              flush=True)

    _print_band_table(results_by_band)
    return results_by_band


# ===========================================================================
# SELFTEST — CPU smoke on a small fixed fixture set
# ===========================================================================

def _fixtures():
    """A few uniqueness-verified KenKen instances spanning N + bands, generated with a
    fixed seed so the smoke is deterministic + fast (no large corpus)."""
    rng = random.Random(123)
    fx = []
    plan = [(5, "g40"), (5, "g10"), (6, "g30"), (6, "g10"), (7, "g20"), (7, "g10")]
    for (n, band) in plan:
        rec = None
        for _ in range(200):
            rec = bk.gen_unique_puzzle_banded(n, rng, band, tries=40)
            if rec is not None:
                break
        if rec is None:
            continue
        cages, clues = _cages_clues(rec)
        fx.append((f"N{n}-{band}", n, cages, clues, rec.get("deduction_depth")))
    return fx


def _selftest() -> bool:
    all_ok = True

    def _check(name, cond):
        nonlocal all_ok
        if not cond:
            all_ok = False
        print(f"[selftest] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    fx = _fixtures()
    _check(f"generated >= 5 uniqueness-verified fixtures (got {len(fx)})", len(fx) >= 5)

    budget = 200000
    tot_b1 = tot_b3 = 0
    dec_b1 = dec_b3 = 0
    n_inst = 0
    for (name, n, cages, clues, dd) in fx:
        r1 = run_config_on_instance("B1", n, cages, clues, budget, seed=1)
        r3 = run_config_on_instance("B3", n, cages, clues, budget, seed=1)
        n_inst += 1
        tot_b1 += int(r1["solved"])
        tot_b3 += int(r3["solved"])
        dec_b1 += r1["decisions"] if r1["solved"] else 0
        dec_b3 += r3["decisions"] if r3["solved"] else 0
        _check(f"{name} (dd={dd}): B1 SOLVES + verifies (exact cage_ok+all-diff)",
               r1["solved"] is True)
        _check(f"{name} (dd={dd}): B3 SOLVES + verifies (exact cage_ok+all-diff)",
               r3["solved"] is True)
        # B3 must not need MORE decisions than B1 (propagation collapses the tree).
        _check(f"{name}: B3 decisions <= B1 decisions "
               f"({r3['decisions']} <= {r1['decisions']})",
               r3["decisions"] <= r1["decisions"])

    _check(f"ALL fixtures solved by B1 ({tot_b1}/{n_inst})", tot_b1 == n_inst)
    _check(f"ALL fixtures solved by B3 ({tot_b3}/{n_inst})", tot_b3 == n_inst)
    _check(f"B3 total decisions <= B1 total decisions ({dec_b3} <= {dec_b1})",
           dec_b3 <= dec_b1)

    # SOUNDNESS (direct): the exact verifier rejects a hand-broken solution.
    (name, n, cages, clues, dd) = fx[0]
    r3 = run_config_on_instance("B3", n, cages, clues, budget, seed=1)
    broken = list(r3["assignment"])
    # break a row's all-different: copy cell 0's value into cell 1 (same row 0).
    broken[1] = broken[0]
    _check("SOUNDNESS: exact verifier rejects a hand-broken (duplicate-in-row) solution",
           verify_kenken_solution(n, cages, clues, broken) is False)
    # and rejects an incomplete assignment.
    incomplete = list(r3["assignment"])
    incomplete[0] = -1
    _check("SOUNDNESS: exact verifier rejects an incomplete assignment",
           verify_kenken_solution(n, cages, clues, incomplete) is False)

    # METRICS aggregation + table print run cleanly.
    agg = _new_agg()
    agg["n"] = 3
    agg["n_solved"] = 2
    agg["decisions_solved"] = [4, 7]
    agg["backtracks_all"] = [1, 0, 3]
    summ = _summarise_agg(agg)
    _check("metrics: solve_rate computed", abs(summ["solve_rate"] - 2 / 3) < 1e-9)
    _check("metrics: decisions_median computed",
           abs(summ["decisions_median"] - 5.5) < 1e-9)
    try:
        _print_band_table({"g40": {"B1": summ, "B3": summ}})
        table_ok = True
    except Exception as e:  # noqa: BLE001
        print(f"  table print raised: {e}", flush=True)
        table_ok = False
    _check("metrics: band table prints cleanly", table_ok)

    print(f"[selftest] {'ALL PASS' if all_ok else 'SOME FAILED'}", flush=True)
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
    run_eval()
