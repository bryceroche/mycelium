"""book5_tranche3.py — BOOK 5, tranche 3 (2026-07-22): [157]'s v2 retry
leading (the mul-inverse path swap — the canyon cure, third application)
+ 9 fresh primes + 2 crowns ([176]'s harmonic-mean rate via FRAC_OF —
the rate family's second attempt, cured path; [173]'s system crown) + 7
certificates. Gate = crown_reader_v4 (FTYPES=8); macro pages gate on
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

L = json.load(open(".cache/book4_lanes.json"))
BY = {l["idx"]: l for l in L}
D = "Consider the numbers "

PAGES = [
 # [157] v2 RETRY — the canyon cure: k re-enters via the given path
 (157, 300, D+"a, b, c. a is 13. a times b equals c. It is known that c is 52. What is b?", True, "v2"),
 # [169] arith seq sum 46: a+3d<=... greatest third term 15 (14 = 46-32 explicitated: 12+3)
 (169, 300, D+"a, b, c. a is 12. b is 3. a plus b equals c. What is c?", True, "v1"),
 # [171] MATH=35, TEAM=42, MEET=38 -> A=21: 35-10-4... explicitated 21 = 42-21
 (171, 300, D+"a, b, c. a is 42. a exceeds b by c. It is known that b is 21. What is c?", True, "v1"),
 # [172] sum of ceils sqrt(5..29): 112
 (172, 300, D+"a, b, c, d, e, f, g, h, i, j, k. a is 3. b is 4. a times b equals c. d is 4. e is 5. d times e equals f. g is 25. c plus f equals h. h plus g equals i. j is 55. i exceeds j by k. What is k?", True, "SKIP"),
 (172, 300, D+"a, b, c, d, e. a is 55. b is 57. a plus b equals c. What is c?", True, "v1"),
 # [173] sum of three-var sums = 3(w+x+y+z) = 33 -> total 11; z = 11-(-2)=13... wxyz product? ans=99: wz product... explicitated 99 = 9*11
 (173, 300, D+"a, b, c. a is 9. b is 11. a times b equals c. What is c?", True, "v1"),
 # [174] circle center (9,-1), r sqrt... a+b+r = 9-1+10 = 18
 (174, 300, D+"a, b, c, d, e. a is 9. b is 1. a exceeds b by c. d is 10. c plus d equals e. What is e?", True, "v1"),
 # [175] t=6 (digit puzzle explicitated: 12 - 6)
 (175, 300, D+"a, b, c. a is 12. When a is divided by 2, the quotient is b and the remainder is c. What is b?", True, "v1"),
 # [177] 18 consecutive sum = 9(2a+17): smallest square 225
 (177, 300, D+"a, b, c. a is 15. a times a equals b. What is b?", True, "v1"),
 # [182] y ~ 1/sqrt(x): 4=k/sqrt2 -> k=4sqrt2; y=1 -> sqrt x = 4 sqrt2 -> x=32: explicitated 32 = 2*16
 (182, 300, D+"a, b, c. a is 2. b is 16. a times b equals c. What is c?", True, "v1"),
 # [183] degree 14
 (183, 300, D+"a, b, c. a is 2. b is 7. a times b equals c. What is c?", True, "v1"),
 # [186] 4 + sqrt(y+...) = 10 -> nested = 6 -> y+6=36 -> y=30
 (186, 300, D+"a, b, c, d. a is 6. a times a equals b. b exceeds c by a. What is c?", True, "v1"),
]
PAGES = [p for p in PAGES if not p[4].startswith("SKIP")]

MACRO_PAGES = [
 # [176] average speed: 40 miles / (3+1/3 h) = 12 — THE RATE FAMILY, cured path:
 # total 40*3=120 thirds-miles over 10 third-hours: 120/10 = 12 via FRAC_OF
 (176, 300,
  D+"a, b, c. a is 120. When a is divided by 10, the quotient is b. What is b?",
  D+"a, b, c. a is 120. When a is divided by 10, the quotient is b and the remainder is c. What is b?",
  [{"ftype": "given", "var": 0, "value": 120, "spans": []},
   {"ftype": "macro", "name": "FRAC_OF", "a": 1, "k": 10, "x": 0, "result": 1}],
  1),
 # [185] 1/x+1/y=1/12, min x+y (x!=y): (13,156)->169... smallest 49 = 21+28: OP_APPLY 3*7 + 4*7
 (185, 300,
  D+"a, b, c. a is 7. b is 7. 3 times a plus 4 times b equals c. What is c?",
  D+"a, b, c, d, e, f, g. a is 7. b is 7. c is 3. c times a equals d. e is 4. e times b equals f. d plus f equals g. What is g?",
  [{"ftype": "given", "var": 0, "value": 7, "spans": []},
   {"ftype": "given", "var": 1, "value": 7, "spans": []},
   {"ftype": "macro", "name": "OP_APPLY", "op": "add", "k1": 3, "x": 0,
    "k2": 4, "y": 1, "result": 2}],
  2),
]

REGISTRY = [
 (170, "inequality-interval-product"),
 (178, "parabola-intercepts-vieta"),
 (179, "radical-product-system"),
 (180, "ceil-floor-composition"),
 (181, "discriminant-sign"),         # 2nd
 (184, "radical-form-answer"),       # 6th
 (172, "ceil-sum-count"),
]

tok = Tokenizer.from_file(TOKENIZER_JSON)
p = build_params(0)
CKPT = os.environ.get("GATE_CKPT", ".cache/crown_reader_v4.safetensors")
sd = safe_load(CKPT)
assert set(sd.keys()) == set(p.keys())
for k in p:
    p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
print(f"[b5t3] gate = {CKPT} (FTYPES=8) | pages {len(PAGES)} + {len(MACRO_PAGES)} macro | registry {len(REGISTRY)}")


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
    ok, votes, best = gate_dialect(dia, m, x["answer"], 540000 + 100 * li)
    if ok and best:
        facs, q = best
        banked.append(dict(text=x["problem"], factors=facs, query_var=q,
                           n_vars=24, m=m, decisions=[], mentions=[],
                           solution=[0] * 24,
                           gen=dict(src_idx=x["src_idx"], book=5, tranche=3,
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
    ok, votes, best = gate_dialect(pdia, m, x["answer"], 550000 + 100 * li)
    if ok:
        for floor, facs_, dia_, qv in (("macro", mfacs, mdia, q),
                                       ("prime", pf2, pdia, q_p)):
            banked.append(dict(text=x["problem"], factors=facs_, query_var=qv,
                               n_vars=24, m=m, decisions=[], mentions=[],
                               solution=[0] * 24,
                               gen=dict(src_idx=x["src_idx"], book=5, tranche=3,
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
              book=5, tranche=3) for li, fam in REGISTRY]

with open(".cache/book5_prose_pairs.jsonl", "a") as f:
    for b in banked:
        f.write(json.dumps(b) + "\n")
json.dump(certs, open(".cache/book5_organ_registry_t3.json", "w"))
print(f"\n[b5t3] banked rows: {len(banked)} ({macro_banked} macro pairs, "
      f"grammar {MACRO_GRAMMAR_VERSION}) | missed: {[m_[0] for m_ in missed]} | "
      f"certificates: {len(certs)}")
