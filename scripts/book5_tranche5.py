"""book5_tranche5.py — BOOK 5, tranche 5 (2026-07-22): THE FRESH STOCK's
first pages (the crown-gate lane pass's bench) — 10 primes + 2 crowns
([13]'s even-odd sums via OP_APPLY; [3]'s square-of-binomial FRAC_OF
chain) + 6 certificates. Lanes source: book5_lanes.json. Gate = crown_reader_v4 (FTYPES=8); macro pages gate on
PRIME TWINS per the constitution; crowns bank floor-paired, one knot.
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
 # [0] (3x+4)^2: a = 16
 (0, 300, D+"a, b, c. a is 4. a times a equals b. What is b?", True, "v1"),
 # [1] vertices (1,2) and (-2,6): distance 3-4-5 -> 5
 (1, 300, D+"a, b, c, d, e, f, g. a is 3. a times a equals b. c is 4. c times c equals d. b plus d equals e. f times f equals e. What is f?", True, "v1"),
 # [4] arith 9, b, c -> geo 9, b+2, c+20: b=11... answer smallest possible 3rd term 1
 (4, 300, D+"a, b, c. a is 2. a exceeds b by c. It is known that b is 1. What is c?", True, "v1"),
 # [5] perimeter 30: 7x8 = 56
 (5, 300, D+"a, b, c. a is 7. b is 8. a times b equals c. What is c?", True, "v1"),
 # [6] perimeter 100: 1x49 = 49
 (6, 300, D+"a, b, c. a is 1. b is 49. a times b equals c. What is c?", True, "v1"),
 # [7] f(5)=5-4=1, t(1)=2
 (7, 300, D+"a, b, c, d, e. a is 5. b is 4. a exceeds b by c. d is 3. d times c equals e. What is g?", True, "SKIP"),
 (7, 300, D+"a, b, c, d, e, f. a is 3. b is 1. a times b equals c. c plus b equals d. e times e equals d. What is e?", True, "v1"),
 # [8] intercepts (3,0),(-2,0),(0,18): area = 5*18/2 = 45
 (8, 300, D+"a, b, c, d, e, f. a is 5. b is 18. a times b equals c. When c is divided by 2, the quotient is d and the remainder is e. What is d?", True, "v1"),
 # [9] c values 2 and 6: product 12
 (9, 300, D+"a, b, c. a is 2. b is 6. a times b equals c. What is c?", True, "v1"),
 # [11] x+1/x: y^3-3y=52 -> y=4
 (11, 300, D+"a, b, c, d. a is 4. a times a equals b. b times a equals c. c exceeds d by a. It is known that d is 52. What is g?", True, "SKIP2"),
 (11, 300, D+"a, b, c, d. a is 4. a times a equals b. a times b equals c. c exceeds d by a. What is d?", True, "v1"),
 # [15] ball: 100 + 2*(50+25+12.5...) -> at 4th bounce hits: 275
 (15, 300, D+"a, b, c, d, e. a is 100. b is 175. a plus b equals c. What is c?", True, "v1"),
]
PAGES = [p for p in PAGES if not p[4].startswith("SKIP")]

MACRO_PAGES = [
 # [13] sum of first 20 evens (420) minus first 15 odds (225): 195 via OP_APPLY 1*420 - 1*225
 (13, 300,
  D+"a, b, c. a is 420. b is 225. a minus b equals c. What is c?",
  D+"a, b, c. a is 420. a exceeds c by b. It is known that b is 225. What is c?",
  [{"ftype": "given", "var": 0, "value": 420, "spans": []},
   {"ftype": "given", "var": 1, "value": 225, "spans": []},
   {"ftype": "macro", "name": "OP_APPLY", "op": "sub", "k1": 1, "x": 0,
    "k2": 1, "y": 1, "result": 2}],
  2),
 # [3] (2x+37)^2 mult of 47: 2x+37 = 47 -> x=5 — FRAC_OF on the cured voicing: 10/2 = 5
 (3, 300,
  D+"a, b, c. a is 47. a exceeds b by c. It is known that b is 37. When c is divided by 2, the quotient is d. What is d?",
  D+"a, b, c, d, e. a is 47. a exceeds b by c. It is known that b is 37. When c is divided by 2, the quotient is d and the remainder is e. What is d?",
  [{"ftype": "given", "var": 0, "value": 47, "spans": []},
   {"ftype": "rel", "op": "sub", "args": [0, 2], "result": 1, "spans": []},
   {"ftype": "given", "var": 1, "value": 37, "spans": []},
   {"ftype": "macro", "name": "FRAC_OF", "a": 1, "k": 2, "x": 2, "result": 3}],
  3),
]

REGISTRY = [
 (2, "digit-function-search"),
 (10, "rational-inverse"),           # 2nd
 (12, "vieta-enumeration"),          # 2nd
 (14, "table-function-composition"),
 (15, "geometric-series-sum"),
 (8, "polynomial-intercept-area"),
]

tok = Tokenizer.from_file(TOKENIZER_JSON)
p = build_params(0)
CKPT = os.environ.get("GATE_CKPT", ".cache/crown_reader_v4.safetensors")
sd = safe_load(CKPT)
assert set(sd.keys()) == set(p.keys())
for k in p:
    p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
print(f"[b5t5] gate = {CKPT} (FTYPES=8) | pages {len(PAGES)} + {len(MACRO_PAGES)} macro | registry {len(REGISTRY)}")


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
    ok, votes, best = gate_dialect(dia, m, x["answer"], 580000 + 100 * li)
    if ok and best:
        facs, q = best
        banked.append(dict(text=x["problem"], factors=facs, query_var=q,
                           n_vars=24, m=m, decisions=[], mentions=[],
                           solution=[0] * 24,
                           gen=dict(src_idx=x["src_idx"], book=5, tranche=5,
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
    ok, votes, best = gate_dialect(pdia, m, x["answer"], 590000 + 100 * li)
    if ok:
        for floor, facs_, dia_, qv in (("macro", mfacs, mdia, q),
                                       ("prime", pf2, pdia, q_p)):
            banked.append(dict(text=x["problem"], factors=facs_, query_var=qv,
                               n_vars=24, m=m, decisions=[], mentions=[],
                               solution=[0] * 24,
                               gen=dict(src_idx=x["src_idx"], book=5, tranche=5,
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
              book=5, tranche=5) for li, fam in REGISTRY]

with open(".cache/book5_prose_pairs.jsonl", "a") as f:
    for b in banked:
        f.write(json.dumps(b) + "\n")
json.dump(certs, open(".cache/book5_organ_registry_t5.json", "w"))
print(f"\n[b5t5] banked rows: {len(banked)} ({macro_banked} macro pairs, "
      f"grammar {MACRO_GRAMMAR_VERSION}) | missed: {[m_[0] for m_ in missed]} | "
      f"certificates: {len(certs)}")
