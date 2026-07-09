"""math500_anchor.py — THE EXTERNAL ANCHOR (2026-07-10): the gen-4 lattice
reads MATH-500 itself. Different IN KIND from every prior eval: foreign text,
no gold graphs, no forcing probe against gold (answer-match only; 0.6%
luck-inflation margin from the equivalence study), template-untaught
phrasing, coverage partial BY DESIGN.

THE REGISTERED CLAIM STRUCTURE (the headline is honesty, not accuracy):
  P1 raw answer-rate lands wherever coverage puts it (band-sweep ceiling);
  P2 CERTIFIED precision holds near its bound on problems nobody generated
     (zero certifications = zero-numerator report, either way honest);
  P3 abstention correlates with out-of-coverage presence (non-integer
     answers = the registered out-of-coverage control, EXPECTED to abstain).
SCOPE, declared: integer solve domain capped at 300 (propagator cost);
answers compared as integers; non-integer-answer problems are the control
stratum. Lattice: one-shot/withhold forced + 5-view sentence-permutation
vote; certify=5/5, answer=majority>=3 else stack, abstain otherwise.
"""
import json, os, sys
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np
from collections import Counter
os.environ.setdefault("ALG2", "1"); os.environ.setdefault("ALG_FTYPES", "6")
from phase1_algebra_head import (T_ALG, L_FAC, build_params, forward, decode,
                                 sent_indices, TOKENIZER_JSON)
from tta_views import permuted_view
from beacon_closing_arm import recompute_states
from math500_band_sweep import answer_type
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load
from tokenizers import Tokenizer

M_CAP = 300

def solve_forced_m(facs, q_pred, m):
    from mycelium.csp_domains import problem_from_algebra3
    from mycelium.csp_core import solve_symbolic
    gv = {f["var"]: f["value"] for f in facs if f["ftype"] == "given"}
    def fv(f):
        if f["ftype"] in ("rel", "sel"): return list(f["args"]) + [f["result"]]
        if f["ftype"] == "pct": return list(f["args"])
        if f["ftype"] in ("mod", "fdiv"): return [f["var"], f["result"]]
        return [f["var"]]
    try:
        nv = max([1] + [v + 1 for f in facs for v in fv(f)] + [q_pred + 1])
        if nv > 24: return None
        prob = problem_from_algebra3(nv, facs, gv, m)
        res = solve_symbolic(prob, budget=20_000, seed=0)
        if res["status"] != "solved": return None
        sol = [int(res["assignment"][v]) for v in range(nv)]
        p2 = problem_from_algebra3(nv, facs, gv, m)
        p2.domains0[q_pred].discard(sol[q_pred])
        if p2.domains0[q_pred]:
            r2 = solve_symbolic(p2, budget=10_000, seed=0)
            if r2["status"] == "solved": return None
        return sol[q_pred]
    except Exception:
        return None

rows = [json.loads(l) for l in open(".cache/math500_test.jsonl")]
tok = Tokenizer.from_file(TOKENIZER_JSON)
p = build_params(0)
sd = safe_load(".cache/phase1_algebra4_head.safetensors")
assert set(sd.keys()) == set(p.keys())
for k in p:
    p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()

def gold_int(a):
    try: return int(a.strip())
    except Exception: return None

def parse_texts(texts):
    n = len(texts)
    ids = np.zeros((n, T_ALG), np.int32); msk = np.zeros((n, T_ALG), np.float32)
    snt = np.zeros((n, T_ALG), np.int32); ok = np.zeros(n, bool)
    for i, t in enumerate(texts):
        e = tok.encode(t)
        if len(e.ids) > T_ALG: continue
        ok[i] = True
        ids[i, :len(e.ids)] = e.ids; msk[i, :len(e.ids)] = 1.0
        snt[i] = sent_indices(t, list(e.offsets), msk[i])
    sts = recompute_states(ids)
    answers = [None] * n
    for s0 in range(0, n, 8):
        sl = np.arange(s0, min(s0 + 8, n))
        out = forward(p, Tensor(sts[sl].astype(np.float32), dtype=dtypes.float),
                      Tensor(msk[sl].astype(np.float32), dtype=dtypes.float),
                      Tensor(snt[sl].astype(np.int32), dtype=dtypes.int))
        keys = ("pres", "ftype", "op", "islit", "dig", "args", "res",
                "query") + (("sel",) if "sel" in out else ())
        o = {k: out[k].realize().numpy() for k in keys}
        for bi, i in enumerate(sl):
            i = int(i)
            if not ok[i]: continue
            facs, q = decode({k: o[k][bi] for k in o})
            nums = [f["value"] for f in facs if f["ftype"] == "given"]
            m = min(M_CAP, max([60] + [2 * v for v in nums]))
            answers[i] = solve_forced_m(facs, q, m)
    return answers

texts0 = [r["problem"] for r in rows]
views = [parse_texts(texts0)]
print(f"[anchor] view 0 forced: {sum(a is not None for a in views[0])}/500")
for k in range(1, 5):
    views.append(parse_texts([permuted_view(t, 1000 * k + i)
                              for i, t in enumerate(texts0)]))
    print(f"[anchor] view {k} forced: {sum(a is not None for a in views[k])}/500")

strata = {}
outcomes = []
for i, r in enumerate(rows):
    at = answer_type(r["answer"])
    stratum = "integer" if at == "integer" else "non-integer"
    g = gold_int(r["answer"])
    votes = [views[k][i] for k in range(5) if views[k][i] is not None]
    top, cnt = (Counter(votes).most_common(1)[0] if votes else (None, 0))
    if cnt >= 5: decision, ans = "certify", top
    elif cnt >= 3: decision, ans = "answer", top
    elif views[0][i] is not None: decision, ans = "answer", views[0][i]
    else: decision, ans = "abstain", None
    st = strata.setdefault(stratum, Counter())
    st["n"] += 1
    st[decision] += 1
    if decision != "abstain":
        st["correct"] += int(g is not None and ans == g)
        if decision == "certify":
            st["cert_correct"] += int(g is not None and ans == g)
    outcomes.append({"i": i, "subject": r["subject"], "level": r["level"],
                     "stratum": stratum, "decision": decision,
                     "ans": ans, "gold": g})
json.dump(outcomes, open(".cache/math500_anchor_outcomes.json", "w"))
print("\n=== THE EXTERNAL ANCHOR (gen-4 lattice on MATH-500) ===")
for s, st in strata.items():
    n = st["n"]; ans_n = st["answer"] + st["certify"]
    print(f"  {s:12s}: n={n} | certified {st['certify']} "
          f"(correct {st['cert_correct']}) | answered {ans_n} "
          f"(correct {st['correct']}) | abstained {st['abstain']} "
          f"({st['abstain']/n:.1%})")
print("  P2 read: certified precision = "
      + (f"{strata['integer']['cert_correct']}/{strata['integer']['certify']}"
         if strata.get('integer', {}).get('certify') else "zero-numerator (no certifications on foreign text)"))
print("  [saved] .cache/math500_anchor_outcomes.json (for the abstention hand-label)")
