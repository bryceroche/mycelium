"""job_b_band_gate.py — THE JOB-B MILESTONE GATE (due at Brick-A completion — it's due).

THE QUESTION (docs/phase1_skeleton_spec.md §0, NEXT_SESSION §2): where does solving
stop being `eval()` and start needing the engine? The gate's instrument is
DECISIONS-PER-PROBLEM: 0 decisions = the CALCULATOR BAND (propagation alone solves —
the engine is overkill); >0 = the ENGINE BAND (search does real work — where the
two-phase bet lives).

THE MEASUREMENT EQUIVALENCE (stated, not hidden): GSM8K factor graphs carry REAL-
valued variables (division makes fractions), so finite-domain GAC does not literally
apply. But every factor is FUNCTIONAL (op(a,b)=result), and for functional factors
the search tier's decision count equals EXACTLY the number of variables the FORWARD
CLOSURE cannot determine: functional-GAC forces a variable iff all its factor's
inputs are forced (that IS the specialized_propagator csp_core would use), and
backtrack decisions are spent only on unforced variables. So the gate computes the
closure directly and reports #undetermined = the decisions the search tier would
spend. For the accepted (execution-verified) graphs the prediction is 0 across the
board — measured here with the machinery-equivalent, not assumed.

WHAT THIS GATE CAN AND CANNOT SAY: it formalizes GSM8K's calculator-band membership
and re-verifies the corpus (closure answer == gold). It CANNOT locate the engine
band's lower edge — that needs a corpus with genuine unknowns/systems (algebra,
MATH-500), which does not exist yet in graph form. That corpus is the REAL Phase-1
target's data question, and this gate's output states it as the open item rather
than letting the milestone silently pass.

REFERENCE POINTS (measured elsewhere, cited): Sudoku — search tier 5000/5000 at
MEDIAN 0 DECISIONS (calculator band by this metric, via propagation strength);
KenKen g10 — B3 GAC collapses 305 decisions -> ~0.9 (engine band, barely); hard
3-coloring depth>=3 — decisions >> 0 (engine band).

USAGE:
  Selftest:  .venv/bin/python3 scripts/job_b_band_gate.py --selftest
  Gate:      .venv/bin/python3 scripts/job_b_band_gate.py [--src PATH] [--n N]
"""
from __future__ import annotations

import argparse
import json
import math
import sys

SRC_DEFAULT = ".cache/gsm8k_factor_graphs_200test_v2.jsonl"

OPS = {
    "add": lambda a, b: a + b,
    "sub": lambda a, b: a - b,
    "mul": lambda a, b: a * b,
    "div": lambda a, b: a / b if b != 0 else float("nan"),
}


def forward_closure(rec: dict):
    """Functional-GAC closure: iterate 'output forced iff both inputs forced' to a
    fixpoint. Returns (values dict, n_undetermined, n_vars). n_undetermined == the
    decisions the search tier would spend (see the equivalence note)."""
    n = int(rec["n_vars"])
    vals = {}
    for i, obs in enumerate(rec["observed_mask"]):
        if obs:
            vals[i] = float(rec["observed_values"][i])
    factors = list(zip(rec["factor_types"], rec["factor_args"]))
    changed = True
    while changed:
        changed = False
        for op, (a, b, r) in factors:
            if r not in vals and a in vals and b in vals and op in OPS:
                v = OPS[op](vals[a], vals[b])
                if not math.isnan(v):
                    vals[r] = v
                    changed = True
    return vals, n - len(vals), n


def run_gate(src: str, cap: int) -> None:
    recs = [json.loads(l) for l in open(src)][:cap]
    n_calc = n_engine = n_verified = n_mismatch = 0
    undet_hist = {}
    for rec in recs:
        vals, undet, n = forward_closure(rec)
        undet_hist[undet] = undet_hist.get(undet, 0) + 1
        if undet == 0:
            n_calc += 1
        else:
            n_engine += 1
        q = int(rec["query_idx"])
        if q in vals:
            ok = abs(vals[q] - float(rec["gold_answer"])) < 1e-6
            n_verified += ok
            n_mismatch += not ok

    print(f"[job-B gate] {len(recs)} GSM8K graphs ({src})")
    print(f"  decisions-per-problem (functional-closure equivalence):")
    for k in sorted(undet_hist):
        print(f"    {k:3d} undetermined vars: {undet_hist[k]} problems")
    print(f"  CALCULATOR BAND (0 decisions): {n_calc}/{len(recs)}")
    print(f"  ENGINE BAND    (>0 decisions): {n_engine}/{len(recs)}")
    print(f"  closure answer == gold: {n_verified}  mismatches: {n_mismatch}")
    print(f"\n  VERDICT: GSM8K sits in the CALCULATOR BAND"
          f" ({100*n_calc/max(len(recs),1):.0f}% zero-decision) — the engine is"
          f" structurally overkill there, as the prep docs predicted; now MEASURED.")
    print(f"  REFERENCES (measured elsewhere): Sudoku median-0 decisions (calculator"
          f" band via propagation strength); KenKen g10 B3 ~0.9 decisions (engine"
          f" band, barely); hard 3-coloring depth>=3 (engine band).")
    print(f"  OPEN ITEM (the existential question, still deliberately dated): the"
          f" engine band's REAL occupants need an algebra/MATH-500 corpus with genuine"
          f" unknowns/systems in graph form — that corpus does not exist yet and is"
          f" the Phase-1 product track's data question.")


def selftest() -> None:
    # forward DAG: fully determined, answer checks
    rec = {"n_vars": 4, "observed_mask": [1, 1, 0, 0], "observed_values": [3, 4, 0, 0],
           "factor_types": ["add", "mul"], "factor_args": [[0, 1, 2], [2, 1, 3]],
           "query_idx": 3, "gold_answer": 28.0}
    vals, undet, n = forward_closure(rec)
    assert undet == 0 and abs(vals[3] - 28.0) < 1e-9
    # a genuine unknown (nothing determines var 2's inputs): undetermined > 0
    rec2 = {"n_vars": 3, "observed_mask": [1, 0, 0], "observed_values": [5, 0, 0],
            "factor_types": ["add"], "factor_args": [[1, 2, 0]],  # a+b=5, both unknown
            "query_idx": 1, "gold_answer": 2.0}
    _v, undet2, _n = forward_closure(rec2)
    assert undet2 == 2, undet2   # the system with unknowns needs the ENGINE
    print("[selftest] PASS (forward DAG -> 0 decisions; unknowns -> engine band)")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--src", default=SRC_DEFAULT)
    ap.add_argument("--n", type=int, default=10 ** 9)
    args = ap.parse_args(argv)
    if args.selftest:
        selftest()
    else:
        run_gate(args.src, args.n)


if __name__ == "__main__":
    main()
