"""book5_tranche7.py — BOOK 5, tranche 7 (2026-07-22): [23]'s
add-voicing retry leads; fresh stock idx 34-57. FIRST TRANCHE UNDER
THE M-DIAL RULE ([45]: FRAC_OF crown THROUGH a 432 intermediate at
m=500 — the dial and the crown composing) and the negative-fold rule
([51]: signed sum voiced as unsigned difference). 11 primes + 3
crowns ([35] OP_APPLY k1=1 'a plus 2 times b' — a dialect probe;
[56] OP_APPLY 3x+4y; [45] FRAC_OF+dial) + 7 certificates ([48]
recharges the spent radical family's zener, honestly metered).
Gate = crown_reader_v4 (FTYPES=8); crowns bank on PRIME TWINS.
"""
import json, sys, os
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np
from collections import Counter
os.environ.setdefault("ALG2", "1"); os.environ.setdefault("ALG_FTYPES", "8")
os.environ.setdefault("ALG_DUP", "1")
from phase1_algebra_head import T_ALG, build_params, forward, decode, sent_indices, TOKENIZER_JSON
from beacon_closing_arm import recompute_states
from tta_views import permuted_view
from tta_alg2_dials import solve2
from mycelium.macros import expand_graph, MACRO_GRAMMAR_VERSION
from hash_audit_iso import canon
from tokenizers import Tokenizer
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load

L = json.load(open(".cache/book5_lanes.json"))
BY = {l["idx"]: l for l in L}
D = "Consider the numbers "

PAGES = [
 # [23] RETRY (t6: right answer at 1/5) — the 'exceeds' sub re-voiced as add
 (23, 300, D+"a, b, c, d, e. a is 24. b is 16. b plus c equals a. d is 3. c plus d equals e. What is e?", True, "v2"),
 # [36] h(2): f(2)=9, g=sqrt(9)-2=1, f(1)=7 — full chain from givens 2 and 5
 (36, 300, D+"a, b, c, d, e, f, g, h, i. a is 2. b is 5. a plus a equals c. c plus b equals d. e times e equals d. f is 2. e exceeds g by f. g plus g equals h. h plus b equals i. What is i?", True, "v1"),
 # [38] a=6, b=18 from the system: ab = 108
 (38, 300, D+"a, b, c. a is 6. b is 18. a times b equals c. What is c?", True, "v1"),
 # [39] max height at t=1: 32+15-16 = 31, sub voiced as add
 (39, 300, D+"a, b, c, d, e. a is 32. b is 15. a plus b equals c. e is 16. d plus e equals c. What is d?", True, "v1"),
 # [40] 12x-18 = -x-5 -> 13x = 13 -> x = 1
 (40, 300, D+"a, b, c. a is 13. b is 13. a times c equals b. What is c?", True, "v1"),
 # [43] geometric product: x/(x-1) = 4/3 -> 3c = 4(c-1), solver derives c=4
 (43, 300, D+"a, b, c, d, e, f. a is 3. b is 4. e is 1. c exceeds d by e. b times d equals f. a times c equals f. What is c?", True, "v1"),
 # [46] 2*2^x = 32 -> 2^x = 16 -> x = 4 (f*f = 16)
 (46, 300, D+"a, b, c, d, e, f. a is 6. b is 26. a plus b equals c. d is 2. d times e equals c. f times f equals e. What is f?", True, "v1"),
 # [51] sum of x with 1<(x-2)^2<25: positives 4+5+6=15, negative tail 3 folds unsigned
 (51, 300, D+"a, b, c, d, e, f, g. a is 4. b is 5. c is 6. a plus b equals d. d plus c equals e. f is 3. f plus g equals e. What is g?", True, "v1"),
 # [53] (x+4)(y+5)=15, max y: y+5 = 15 -> y = 10
 (53, 300, D+"a, b, c. a is 15. b is 5. c plus b equals a. What is c?", True, "v1"),
 # [55] g(25) values 13 and 7 from x=+-3: 9+3+1 and 9-3+1, sum 20
 (55, 300, D+"a, b, c, d, e, f, g, h. a is 9. b is 3. c is 1. a plus b equals d. d plus c equals e. f plus b equals a. f plus c equals g. e plus g equals h. What is h?", True, "v1"),
 # [57] xy(x+y): c+3c=4 -> c=xy=1, d=3c = the answer
 (57, 300, D+"a, b, c, d. a is 4. b is 3. b times c equals d. c plus d equals a. What is d?", True, "v1"),
]
PAGES = [p for p in PAGES if not p[4].startswith("SKIP")]

MACRO_PAGES = [
 # [35] (a+b+c)^2 = 39 + 2*21 = 81 -> 9: OP_APPLY k1=1 dialect probe
 (35, 300,
  D+"a, b, c, d. a is 39. b is 21. a plus 2 times b equals c. d times d equals c. What is d?",
  D+"a, b, c, d, e, f. a is 39. b is 21. c is 2. c times b equals d. a plus d equals e. f times f equals e. What is f?",
  [{"ftype": "given", "var": 0, "value": 39, "spans": []},
   {"ftype": "given", "var": 1, "value": 21, "spans": []},
   {"ftype": "macro", "name": "OP_APPLY", "op": "add", "k1": 1, "x": 0,
    "k2": 2, "y": 1, "result": 2},
   {"ftype": "rel", "op": "mul", "args": [3, 3], "result": 2, "spans": []}],
  3),
 # [45] four consecutive evens summing 420: largest = (420+12)/4 = 108:
 # FRAC_OF through a 432 intermediate — THE M-DIAL AND THE CROWN COMPOSING
 (45, 500,
  D+"a, b, c, d. a is 420. b is 12. a plus b equals c. When c is divided by 4, the quotient is d. What is d?",
  D+"a, b, c, d, e. a is 420. b is 12. a plus b equals c. When c is divided by 4, the quotient is d and the remainder is e. What is d?",
  [{"ftype": "given", "var": 0, "value": 420, "spans": []},
   {"ftype": "given", "var": 1, "value": 12, "spans": []},
   {"ftype": "rel", "op": "add", "args": [0, 1], "result": 2, "spans": []},
   {"ftype": "macro", "name": "FRAC_OF", "a": 1, "k": 4, "x": 2, "result": 3}],
  3),
 # [56] x^3 coefficient: 3*3 + 4*5 = 29, then e + 3 = 29 -> 26
 (56, 300,
  D+"a, b, c, d, e. a is 3. b is 5. 3 times a plus 4 times b equals c. d is 3. e plus d equals c. What is e?",
  D+"a, b, c, d, e, f, g, h, i. a is 3. b is 5. c is 3. c times a equals d. e is 4. e times b equals f. d plus f equals g. h is 3. i plus h equals g. What is i?",
  [{"ftype": "given", "var": 0, "value": 3, "spans": []},
   {"ftype": "given", "var": 1, "value": 5, "spans": []},
   {"ftype": "macro", "name": "OP_APPLY", "op": "add", "k1": 3, "x": 0,
    "k2": 4, "y": 1, "result": 2},
   {"ftype": "given", "var": 3, "value": 3, "spans": []},
   {"ftype": "rel", "op": "add", "args": [4, 3], "result": 2, "spans": []}],
  4),
]

REGISTRY = [
 (34, "geometric-ratio-system"),
 (41, "inverse-function-match"),
 (44, "line-intersection-count"),
 (47, "factor-pair-count"),
 (48, "radical-rationalize"),        # recharges the SPENT family — metered
 (50, "collatz-domain-count"),
 (52, "degree-arithmetic"),
]

tok = Tokenizer.from_file(TOKENIZER_JSON)
p = build_params(0)
CKPT = os.environ.get("GATE_CKPT", ".cache/crown_reader_v4.safetensors")
sd = safe_load(CKPT)
assert set(sd.keys()) == set(p.keys())
for k in p:
    p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
print(f"[b5t7] gate = {CKPT} (FTYPES=8) | pages {len(PAGES)} + {len(MACRO_PAGES)} macro | registry {len(REGISTRY)}")


def parse_batch(texts):
    n = len(texts)
    N = ((n + 7) // 8) * 8
    ids = np.zeros((N, T_ALG), np.int32); msk = np.zeros((N, T_ALG), np.float32)
    snt = np.zeros((N, T_ALG), np.int32)
    for i, t in enumerate(texts):
        e = tok.encode(t)
        Ln = min(len(e.ids), T_ALG)
        ids[i, :Ln] = e.ids[:Ln]; msk[i, :Ln] = 1.0
        snt[i] = sent_indices(t, list(e.offsets), msk[i])
    st = recompute_states(ids)
    res = []
    for s0 in range(0, N, 8):
        out = forward(p, Tensor(st[s0:s0+8].astype(np.float32), dtype=dtypes.float),
                      Tensor(msk[s0:s0+8].astype(np.float32), dtype=dtypes.float),
                      Tensor(snt[s0:s0+8].astype(np.int32), dtype=dtypes.int))
        keys = [k for k in ("pres", "ftype", "op", "islit", "dig", "dig2", "args",
                            "res", "query", "sel", "dup", "y") if k in out]
        o = {k: out[k].realize().numpy() for k in keys}
        for bi in range(8):
            if s0 + bi < n:
                res.append(decode({k: o[k][bi] for k in o}))
    return res


def gate_dialect(text, m, answer, seed0):
    texts = [text] + [permuted_view(text, seed0 + k) for k in range(1, 5)]
    votes, best = [], None
    for facs, q in parse_batch(texts):
        a = solve2(facs, q, {"n_vars": 24, "m": m})
        if a is not None:
            votes.append(a)
            if a == answer and best is None:
                best = (facs, q)
    top, cnt = (Counter(votes).most_common(1)[0] if votes else (None, 0))
    return (cnt >= 3 and top == answer), votes, best


banked, missed = [], []
for li, m, dia, fs, ver in PAGES:
    x = BY[li]
    ok, votes, best = gate_dialect(dia, m, x["answer"], 630000 + 100 * li)
    if ok and best:
        facs, q = best
        banked.append(dict(text=x["problem"], factors=facs, query_var=q,
                           n_vars=24, m=m, decisions=[], mentions=[],
                           solution=[0] * 24,
                           gen=dict(src_idx=x["src_idx"], book=5, tranche=7,
                                    floor="prime", fs=fs, dialect=dia,
                                    gate="5view-vote+key", generation="16")))
        print(f"  [page {li}] BANKED (votes {votes})")
    else:
        missed.append((li, votes))
        print(f"  [page {li}] MISS (votes {votes})")

macro_banked = 0
for li, m, mdia, pdia, mfacs, q in MACRO_PAGES:
    x = BY[li]
    pfacs, nv = expand_graph([dict(f) for f in mfacs], 24)
    used = sorted({v for f in pfacs for v in (list(f.get("args", [])) +
                   [f[k] for k in ("result", "var") if k in f])})
    rm = {v: i for i, v in enumerate(used)}
    pf2 = []
    for f in pfacs:
        f = dict(f)
        if "args" in f:
            f["args"] = [rm[v] for v in f["args"]]
        for kk in ("result", "var"):
            if kk in f:
                f[kk] = rm[f[kk]]
        pf2.append(f)
    q_p = rm[q]
    a = solve2(pf2, q_p, {"n_vars": 24, "m": m})
    assert a == x["answer"], (li, a, x["answer"])
    dg_m, _ = canon({"factors": mfacs, "n_vars": 24, "query_var": q})
    dg_p, _ = canon({"factors": pf2, "n_vars": 24, "query_var": q_p})
    assert dg_m == dg_p, (li, dg_m, dg_p)
    ok, votes, best = gate_dialect(pdia, m, x["answer"], 640000 + 100 * li)
    if ok:
        for floor, facs_, dia_, qv in (("macro", mfacs, mdia, q),
                                       ("prime", pf2, pdia, q_p)):
            banked.append(dict(text=x["problem"], factors=facs_, query_var=qv,
                               n_vars=24, m=m, decisions=[], mentions=[],
                               solution=[0] * 24,
                               gen=dict(src_idx=x["src_idx"], book=5, tranche=7,
                                        floor=floor, fs=True, dialect=dia_,
                                        knot=dg_m, grammar=MACRO_GRAMMAR_VERSION,
                                        gate="5view-vote+key(prime-twin)",
                                        generation="16")))
        macro_banked += 1
        print(f"  [MACRO {li}] BANKED both floors, one knot {dg_m[:12]} (votes {votes})")
    else:
        missed.append((li, votes))
        print(f"  [MACRO {li}] prime-twin MISS (votes {votes})")

certs = [dict(lane_idx=li, family=fam, raw=BY[li]["problem"],
              answer=BY[li]["answer"], src_idx=BY[li]["src_idx"],
              book=5, tranche=7) for li, fam in REGISTRY]

with open(".cache/book5_prose_pairs.jsonl", "a") as f:
    for b in banked:
        f.write(json.dumps(b) + "\n")
json.dump(certs, open(".cache/book5_organ_registry_t7.json", "w"))
print(f"\n[b5t7] banked rows: {len(banked)} ({macro_banked} macro pairs, "
      f"grammar {MACRO_GRAMMAR_VERSION}) | missed: {[m_[0] for m_ in missed]} | "
      f"certificates: {len(certs)}")
