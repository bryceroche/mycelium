"""grade_equivalence.py — THE EQUIVALENCE UPGRADE: one uniform honest metric across
every row of the stack-at-convergence table (spec registration, 2026-07-08).

THE INCONSISTENCY IT FIXES: the composed-stack rows require the answer FORCED
(unique at the query variable — solution-set equivalence); the one-shot ANSWER
(802/1500) does not. This script grades the converged parser's one-shot output on
bigtest with the UNIFORM metric and taxonomizes the correctness boundary:

  FORCED-CORRECT       : solved, answer == gold, unique at query (the honest unit)
  LUCKY-UNFORCED       : answer == gold but a second query-value exists (solver
                         happened to pick right — NOT creditable)
  RIGHT-ASKED-WRONG-GRAPH: forced-correct AND the graph mismatches gold factor-wise
                         (wrong-but-equivalent-where-asked — creditable, counted,
                         and reported separately: the class MATH-500 grading will
                         force a policy on)

OUTPUT: the honest one-shot number, the corrected END-TO-END baseline
(one-shot-forced + repaired-forced 226), and the boundary taxonomy.

USAGE: DEV=AMD ALG_TEST=.cache/algebra_nl_bigtest.jsonl ALG_TEST_NAME=bigtest \
           .venv/bin/python3 scripts/grade_equivalence.py
"""
from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "scripts"))

import numpy as np

from phase1_algebra_head import (  # noqa: E402
    L_FAC, build_params, forward, load_alg, decode, ALG_CKPT,
)


def main():
    from tinygrad import Tensor, dtypes
    from tinygrad.nn.state import safe_load
    from mycelium.csp_domains import problem_from_algebra
    from mycelium.csp_core import solve_symbolic

    samples, states, tokmask, gold, sent = load_alg("test")
    p = build_params(0)
    sd = safe_load(ALG_CKPT)
    for k in p:
        p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
    n = len(samples)

    forced_ok = lucky = raw_ok = right_asked_wrong_graph = 0
    for s0 in range(0, n, 8):
        sl = np.arange(s0, min(s0 + 8, n))
        pad = 8 - len(sl)
        sl_p = np.concatenate([sl, sl[:1].repeat(pad)]) if pad else sl
        out = forward(p, Tensor(states[sl_p].astype(np.float32), dtype=dtypes.float),
                      Tensor(tokmask[sl_p].astype(np.float32), dtype=dtypes.float),
                      Tensor(sent[sl_p].astype(np.int32), dtype=dtypes.int))
        o = {k: out[k].realize().numpy() for k in
             ("pres", "ftype", "op", "islit", "dig", "args", "res", "query")}
        for bi, i in enumerate(sl):
            i = int(i)
            smp = samples[i]
            facs, q_pred = decode({k: o[k][bi] for k in o})
            rels = [(f["op"], f["args"][0], f["args"][1], f["result"])
                    for f in facs if f["ftype"] == "rel"]
            gv = {f["var"]: f["value"] for f in facs if f["ftype"] == "given"}
            gold_ans = smp["solution"][smp["query_var"]]
            try:
                nv = max([smp["n_vars"]] + [v + 1 for f in facs for v in
                         ((list(f["args"]) + [f["result"]]) if f["ftype"] == "rel"
                          else [f["var"]])] + [q_pred + 1])
                res = solve_symbolic(problem_from_algebra(nv, rels, gv, smp["m"]),
                                     budget=200_000, seed=0)
                if res["status"] != "solved":
                    continue
                sol = [int(res["assignment"][v]) for v in range(nv)]
                if not (q_pred < len(sol) and sol[q_pred] == gold_ans):
                    continue
                raw_ok += 1
                # the forcing check: unique at the query variable?
                p2 = problem_from_algebra(nv, rels, gv, smp["m"])
                p2.domains0[q_pred].discard(sol[q_pred])
                forced = True
                if p2.domains0[q_pred]:
                    r2 = solve_symbolic(p2, budget=100_000, seed=0)
                    forced = r2["status"] != "solved"
                if not forced:
                    lucky += 1
                    continue
                forced_ok += 1
                # graph-match check (the equivalence-class boundary)
                gset = set()
                for f in smp["factors"]:
                    gset.add(("rel", f["op"], tuple(sorted(f["args"])), f["result"])
                             if f["ftype"] == "rel"
                             else ("given", f["var"], f["value"]))
                pset = set()
                for f in facs:
                    pset.add(("rel", f["op"], tuple(f["args"]), f["result"])
                             if f["ftype"] == "rel"
                             else ("given", f["var"], f["value"]))
                if pset != gset:
                    right_asked_wrong_graph += 1
            except Exception:
                continue

    print(f"[equivalence] n={n} (converged parser, one-shot)")
    print(f"  raw ANSWER (the old metric)         : {raw_ok}")
    print(f"  LUCKY-UNFORCED (removed)            : {lucky}")
    print(f"  FORCED-CORRECT (the honest unit)    : {forced_ok}")
    print(f"  ...of which right-asked-wrong-graph : {right_asked_wrong_graph}"
          f"  (equivalent-where-asked — creditable, policy-relevant for MATH-500)")
    print(f"\n  CORRECTED END-TO-END BASELINE: {forced_ok} one-shot-forced"
          f" + 226 repaired-forced = {forced_ok + 226}/{n}"
          f" = {(forced_ok + 226) / n:.3f}")
    print(f"  (multi-round's asymptote frame starts HERE.)")


if __name__ == "__main__":
    main()
