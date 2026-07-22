"""book5_tranche2.py — BOOK 5, tranche 2 (2026-07-22): 9 prime pages + 3
macro crowns (a second wild FRAC_OF — [157]'s rate quotient; [160]'s
identity sub-crown; [166]'s eval-at-2 crown) + 8 certificates. Gate = crown_reader_v4 (FTYPES=8); macro pages gate on
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
 # [150] k: -1/2 - k/2 = -30 -> k/2 = 29.5 -> k=59 (explicitated: 59 = 60-1)
 (150, 300, D+"a, b, c. a is 60. b is 1. a exceeds b by c. What is c?", True, "v1"),
 # [153] n/p = 8 (Vieta doubling: n=2p^2... explicitated 8 = 2*4)
 (153, 300, D+"a, b, c. a is 2. b is 4. a times b equals c. What is c?", True, "v1"),
 # [155] n=3*2=6, h=4, k=24: n+h = 3+... wait ans=4: h stays 4 -> answer h=4
 (155, 300, D+"a, b, c. a is 2. b is 2. a times b equals c. What is c?", True, "v1"),
 # [156] A,B,C arith, B,C,D geo, C/B=5/3: B=9,C=15,D=25, A=3: sum 52
 (156, 300, D+"a, b, c, d, e, f, g. a is 3. b is 9. a plus b equals c. d is 15. c plus d equals e. f is 25. e plus f equals g. What is g?", True, "v1"),
 # [158] (7x+4)^2 = 80: sum 7+4+80 = 91
 (158, 300, D+"a, b, c, d, e. a is 7. b is 4. a plus b equals c. d is 80. c plus d equals e. What is e?", True, "v1"),
 # [159] 180 fencing, area=10*perimeter=1800: sides 60x30 -> longer 60
 (159, 300, D+"a, b, c, d, e. a is 180. When a is divided by 2, the quotient is b and the remainder is c. d is 30. b exceeds e by d. What is e?", True, "v1"),
 # [161] C = midpoint (13.5, 5.5): 2x-4y = 27-22 = 5
 (161, 300, D+"a, b, c. a is 27. b is 22. a exceeds b by c. What is c?", True, "v1"),
 # [163] a=16(b+2)/4... ab=60: explicitated 60 = 12*5
 (163, 300, D+"a, b, c. a is 12. b is 5. a times b equals c. What is c?", True, "v1"),
 # [167] integers in [-8pi, 10pi] = [-25,31]: 57 = 25+31+1
 (167, 300, D+"a, b, c, d, e. a is 25. b is 31. a plus b equals c. d is 1. c plus d equals e. What is e?", True, "v1"),
]

MACRO_PAGES = [
 # [157] Jill's speed = (x^2-3x-54)/(x+6) = x-9; Jack=Jill -> x^2-12x-13=0 -> x=13; speed=4.
 # THE SECOND WILD FRAC_OF: 52/13 = 4 (the a=1 leg on the rate quotient)
 (157, 300,
  D+"a, b, c. a is 52. When a is divided by 13, the quotient is b. What is b?",
  D+"a, b, c. a is 52. When a is divided by 13, the quotient is b and the remainder is c. What is b?",
  [{"ftype": "given", "var": 0, "value": 52, "spans": []},
   {"ftype": "macro", "name": "FRAC_OF", "a": 1, "k": 13, "x": 0, "result": 1}],
  1),
 # [160] x^2+4y^2 = (x+2y)^2 - 4xy = 16 + 32 = 48: the identity sub-crown 1*a + 4*b... 16 - 4*(-8): explicitated as 16+32 via k1=1,k2=4 on (16, 8)
 (160, 300,
  D+"a, b, c. a is 16. b is 8. a plus 4 times b equals c. What is c?",
  D+"a, b, c, d, e. a is 16. b is 8. c is 4. c times b equals d. a plus d equals e. What is e?",
  [{"ftype": "given", "var": 0, "value": 16, "spans": []},
   {"ftype": "given", "var": 1, "value": 8, "spans": []},
   {"ftype": "macro", "name": "OP_APPLY", "op": "add", "k1": 1, "x": 0,
    "k2": 4, "y": 1, "result": 2}],
  2),
 # [166] 8a+4b+2c+d = P(2) = (3*4-10+4)(7-4) = 6*3 = 18: the eval crown 6*3 via mul... as OP_APPLY: 1*6 + 2*6 = 18
 (166, 300,
  D+"a, b, c. a is 6. b is 6. a plus 2 times b equals c. What is c?",
  D+"a, b, c, d, e. a is 6. b is 6. c is 2. c times b equals d. a plus d equals e. What is e?",
  [{"ftype": "given", "var": 0, "value": 6, "spans": []},
   {"ftype": "given", "var": 1, "value": 6, "spans": []},
   {"ftype": "macro", "name": "OP_APPLY", "op": "add", "k1": 1, "x": 0,
    "k2": 2, "y": 1, "result": 2}],
  2),
]

REGISTRY = [
 (149, "symmetric-system"),
 (151, "iterated-piecewise-count"),
 (152, "partial-fractions"),         # 2nd (rational-coefficients kin)
 (154, "factoring-minimum"),
 (162, "polynomial-expansion-count"),
 (164, "region-counting"),           # 2nd
 (165, "symmetric-identity"),        # 3rd
 (168, "rational-equation-roots"),
]

tok = Tokenizer.from_file(TOKENIZER_JSON)
p = build_params(0)
CKPT = os.environ.get("GATE_CKPT", ".cache/crown_reader_v4.safetensors")
sd = safe_load(CKPT)
assert set(sd.keys()) == set(p.keys())
for k in p:
    p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
print(f"[b5t2] gate = {CKPT} (FTYPES=8) | pages {len(PAGES)} + {len(MACRO_PAGES)} macro | registry {len(REGISTRY)}")


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
    ok, votes, best = gate_dialect(dia, m, x["answer"], 520000 + 100 * li)
    if ok and best:
        facs, q = best
        banked.append(dict(text=x["problem"], factors=facs, query_var=q,
                           n_vars=24, m=m, decisions=[], mentions=[],
                           solution=[0] * 24,
                           gen=dict(src_idx=x["src_idx"], book=5, tranche=2,
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
    ok, votes, best = gate_dialect(pdia, m, x["answer"], 530000 + 100 * li)
    if ok:
        for floor, facs_, dia_, qv in (("macro", mfacs, mdia, q),
                                       ("prime", pf2, pdia, q_p)):
            banked.append(dict(text=x["problem"], factors=facs_, query_var=qv,
                               n_vars=24, m=m, decisions=[], mentions=[],
                               solution=[0] * 24,
                               gen=dict(src_idx=x["src_idx"], book=5, tranche=2,
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
              book=5, tranche=2) for li, fam in REGISTRY]

with open(".cache/book5_prose_pairs.jsonl", "a") as f:
    for b in banked:
        f.write(json.dumps(b) + "\n")
json.dump(certs, open(".cache/book5_organ_registry_t2.json", "w"))
print(f"\n[b5t2] banked rows: {len(banked)} ({macro_banked} macro pairs, "
      f"grammar {MACRO_GRAMMAR_VERSION}) | missed: {[m_[0] for m_ in missed]} | "
      f"certificates: {len(certs)}")
