"""repair_replace_swap.py — ARM C: candidate-restriction repair (spec
registration, 2026-07-09). Replace-and-solve, the sibling of withhold-and-solve.

THE MECHANISM (the routing-verdict fix): the L4 states — and the raw text —
contain the correct given values; the trained pointer misroutes. So don't steer
the pointer: SHRINK THE SHELF. The value INVENTORY is lexed symbolically from
the input text (digit literals, 1.0-reliability; the neural probe earns its
place later on content lexing can't reach). Moves over emitted given slots:
  REPLACE — one slot's value <- one inventory value (withhold could never fix a
            misbound given: removal loses the constraint and un-forces the
            system; replace keeps the constraint, corrected)
  SWAP    — two slots exchange values (the coordinated two-slot fix a parallel
            marginal decoder cannot emit)
The SOLVER disposes: a move is a candidate fix iff the system becomes
forced-solved (unique at query). ACCEPTANCE (pinned): collect all passing
moves; accept iff all passing answers AGREE (ties-to-one-answer accepted);
distinct answers = ambiguous = reject. Fully gold-free.

DEPLOYMENT HONESTY: repair fires only on VISIBLE failures (unsolved/unforced).
Survivors whose original parse is forced-WRONG are deployment-invisible
(gold-only failures) — counted and reported, not repaired.

REGISTERED (before firing):
  - Mine: recovery on the 460 concentrated in low-m given_value survivors;
    60-120 of 460; ambiguity-rejections <10% of otherwise-fixable (coupled
    systems are tight — wrong substitutions mostly go UNSAT).
  - Relay (polarity-flipped with the mechanism): this arm recovers the BULK of
    the 396's convertible fraction; the marker-token beacon adds little or
    nothing on top (fifth bootstrap sighting: pointers don't re-aim under
    conditioning, and a beacon is conditioning through the input). If the
    beacon DOES later add recovery beyond this floor, that is the interesting
    result.
  - Soundness gate: accepted-answer==gold fraction ~1.0 (forced-unique
    acceptance should not admit luck; measure it).
  v0 scope: single move only (given_value class); multi-move + add-given +
  neural reads for args/ops = v1, gated on this result.

USAGE: DEV=AMD ALG_TEST=.cache/algebra_nl_bigtest.jsonl ALG_TEST_NAME=bigtest \
           .venv/bin/python3 scripts/repair_replace_swap.py
"""
from __future__ import annotations

import os
import re
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "scripts"))

import numpy as np

from phase1_algebra_head import (  # noqa: E402
    build_params, forward, load_alg, decode, ALG_CKPT,
)


def solve_forced(facs, q_pred, smp):
    """Gold-free: returns forced answer at q_pred, or None. v2-aware
    (2026-07-09): routes through problem_from_algebra2, which handles
    rel/given identically and adds mod/sel — one seam, all callers upgraded."""
    from mycelium.csp_domains import problem_from_algebra2
    from mycelium.csp_core import solve_symbolic
    gv = {f["var"]: f["value"] for f in facs if f["ftype"] == "given"}

    def fv(f):
        if f["ftype"] in ("rel", "sel"):
            return list(f["args"]) + [f["result"]]
        if f["ftype"] == "mod":
            return [f["var"], f["result"]]
        return [f["var"]]
    try:
        nv = max([smp["n_vars"]] + [v + 1 for f in facs for v in fv(f)]
                 + [q_pred + 1])
        res = solve_symbolic(problem_from_algebra2(nv, facs, gv, smp["m"]),
                             budget=200_000, seed=0)
        if res["status"] != "solved":
            return None
        sol = [int(res["assignment"][v]) for v in range(nv)]
        if q_pred >= len(sol):
            return None
        p2 = problem_from_algebra2(nv, facs, gv, smp["m"])
        p2.domains0[q_pred].discard(sol[q_pred])
        if p2.domains0[q_pred]:
            r2 = solve_symbolic(p2, budget=100_000, seed=0)
            if r2["status"] == "solved":
                return None
        return sol[q_pred]
    except Exception:
        return None


def main():
    from tinygrad import Tensor, dtypes
    from tinygrad.nn.state import safe_load

    prof = np.load(".cache/survivor_profile_bigtest.npz")
    surv460 = sorted(int(i) for i, s in zip(prof["idx"], prof["status"])
                     if s == 2)
    mult = {int(i): int(m) for i, m in zip(prof["idx"], prof["mult"])}
    orc = set(int(i) for i in
              np.load(".cache/oracle_recovered_bigtest.npz")["recovered"])

    samples, states, tokmask, gold, sent = load_alg("test")
    p = build_params(0)
    sd = safe_load(ALG_CKPT)
    for k in p:
        p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()

    parses = {}
    for s0 in range(0, len(surv460), 8):
        sl = np.array(surv460[s0:s0 + 8])
        pad = 8 - len(sl)
        sl_p = np.concatenate([sl, sl[:1].repeat(pad)]) if pad else sl
        out = forward(p, Tensor(states[sl_p].astype(np.float32),
                                dtype=dtypes.float),
                      Tensor(tokmask[sl_p].astype(np.float32),
                             dtype=dtypes.float),
                      Tensor(sent[sl_p].astype(np.int32), dtype=dtypes.int))
        o = {k: out[k].realize().numpy() for k in
             ("pres", "ftype", "op", "islit", "dig", "args", "res", "query")}
        for bi, i in enumerate(sl):
            parses[int(i)] = decode({k: o[k][bi] for k in o})

    invisible = 0
    repaired, ambiguous, none_pass = [], 0, 0
    lucky = 0
    for i in surv460:
        smp = samples[i]
        facs, q_pred = parses[i]
        gold_ans = smp["solution"][smp["query_var"]]
        # deployment visibility: forced-wrong originals look like successes
        orig = solve_forced(facs, q_pred, smp)
        if orig is not None:
            invisible += 1
            continue
        inventory = sorted(set(int(x) for x in re.findall(r"\d+", smp["text"])
                               if int(x) < 10 ** 3))
        gslots = [j for j, f in enumerate(facs) if f["ftype"] == "given"]
        answers = set()
        for j in gslots:                           # REPLACE
            cur = facs[j]["value"]
            for v in inventory:
                if v == cur:
                    continue
                facs[j]["value"] = v
                a = solve_forced(facs, q_pred, smp)
                if a is not None:
                    answers.add(a)
                facs[j]["value"] = cur
        for xi in range(len(gslots)):              # SWAP
            for yi in range(xi + 1, len(gslots)):
                ja, jb = gslots[xi], gslots[yi]
                va, vb = facs[ja]["value"], facs[jb]["value"]
                if va == vb:
                    continue
                facs[ja]["value"], facs[jb]["value"] = vb, va
                a = solve_forced(facs, q_pred, smp)
                if a is not None:
                    answers.add(a)
                facs[ja]["value"], facs[jb]["value"] = va, vb
        if not answers:
            none_pass += 1
        elif len(answers) > 1:
            ambiguous += 1
        else:
            ans = answers.pop()
            repaired.append(i)
            if ans != gold_ans:
                lucky += 1

    n_rep = len(repaired)
    print(f"[replace+swap] survivors={len(surv460)} | deployment-invisible "
          f"(forced-wrong) = {invisible}")
    print(f"  REPAIRED (accepted, unambiguous) : {n_rep}")
    print(f"  ...accepted-but-wrong (luck gate): {lucky}")
    print(f"  ambiguous (rejected)             : {ambiguous}")
    print(f"  no passing move                  : {none_pass}")
    in396 = sum(1 for i in repaired if i not in orc)
    print(f"\n  split: in-hard-396 = {in396} | in-oracle-64 = {n_rep - in396}")
    m_rep = [mult[i] for i in repaired]
    if m_rep:
        print(f"  multiplicity of repaired: mean {np.mean(m_rep):.2f} | "
              f"m<=2 share {np.mean([m <= 2 for m in m_rep]):.2f}")
    correct_new = n_rep - lucky
    print(f"\n  END-TO-END: 1040 + {correct_new} = {1040 + correct_new}/1500 "
          f"= {(1040 + correct_new) / 1500:.3f}")
    print(f"  (oracle NEURAL ceiling was 64/460 = 13.9% — this arm is symbolic,"
          f" deployable, zero-training)")


if __name__ == "__main__":
    main()
