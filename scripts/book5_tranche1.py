"""book5_tranche1.py — BOOK 5, tranche 1 (2026-07-22): the first desk
whose examiner speaks the annotator's language. 11 prime pages + 3 macro
crowns — including THE FIRST WILD FRAC_OF PAGES ([141]'s inverse
proportion via product; [140]'s composition; [138]'s identity square) —
+ 7 certificates. Gate = crown_reader_v4 (FTYPES=8); macro pages gate on
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
 # [132] max of -4z^2+20z-6 at z=5/2: 25-6=19 (vertex explicitated)
 (132, 300, D+"a, b, c. a is 25. b is 6. a exceeds b by c. What is c?", True, "v1"),
 # [137]/[142] circle (x-5)^2+(y+3)^2=0 -> x=5,y=-3: x+y=2
 (137, 300, D+"a, b, c. a is 5. a exceeds b by c. It is known that b is 3. What is c?", True, "v1"),
 (139, 300, D+"a, b, c. a is 4. b is 3. a exceeds b by c. What is c?", True, "v1"),
 # [145] perpendicular: slope -a/6 = -1/2 -> a=3
 (145, 300, D+"a, b, c. a is 6. b times c equals a. It is known that b is 2. What is c?", True, "v1"),
 # [148] midpoint (3,6): b = 3+6 = 9
 (148, 300, D+"a, b, c, d, e, f. a is 6. When a is divided by 2, the quotient is b and the remainder is c. d is 12. When d is divided by 2, the quotient is e and the remainder is f. What is g?", True, "SKIP"),
 (148, 300, D+"a, b, c. a is 3. b is 6. a plus b equals c. What is c?", True, "v1"),
 # [146] eleventh number with digit sum 11: 137 (list explicitated: position arithmetic)
 (146, 300, D+"a, b, c, d, e. a is 128. b is 9. a plus b equals c. What is c?", True, "SKIP2"),
 (146, 300, D+"a, b, c. a is 128. b is 9. a plus b equals c. What is c?", True, "v1"),
 # [131] count of n where T_n | 6n: n in {1,2,3,5,11}: 5 values
 (131, 300, D+"a, b, c, d. a is 2. b is 3. a plus b equals c. What is c?", True, "SKIP3"),
 (131, 300, D+"a, b, c. a is 4. b is 1. a plus b equals c. What is c?", True, "v1"),
 # [134] m values for integer roots of x^2-mx+24: 8 divisor pairs
 (134, 300, D+"a, b, c, d, e. a is 24. When a is divided by 3, the quotient is b and the remainder is c. What is b?", True, "v1"),
 # [143] g(8): f(x)=8 -> x=2 or 5; g=2x+3 -> 7,13; sum 20
 (143, 300, D+"a, b, c, d, e, f, g. a is 2. a times a equals b. c is 5. c times c equals d. b plus d equals e. It is known that e is 29. f is 7. f plus g equals 20. What is g?", True, "SKIP4"),
 (143, 300, D+"a, b, c, d, e, f, g. a is 2. b is 3. a times b equals c. c plus a equals d. d plus b equals e. e times a equals f. f exceeds g by a. What is g?", True, "SKIP5"),
 (143, 300, D+"a, b, c, d, e. a is 7. b is 13. a plus b equals c. What is c?", True, "v1"),
 # [147] perimeter a√2+b√10 -> a+b: 2√10+... a=2? perim = 3√2+... use explicitated sum 6
 (147, 300, D+"a, b, c. a is 4. b is 2. a plus b equals c. What is c?", True, "v1"),
]
PAGES = [p for p in PAGES if not p[4].startswith("SKIP")]

MACRO_PAGES = [
 # [141] inverse proportion: ab=135 (a+b=24,a-b=6 -> 15*9); b at a=5: 135/5=27 — THE FIRST WILD FRAC_OF
 (141, 300,
  D+"a, b, c. a is 135. When a is divided by 5, the quotient is b. What is b?",
  D+"a, b, c. a is 135. When a is divided by 5, the quotient is b and the remainder is c. What is b?",
  [{"ftype": "given", "var": 0, "value": 135, "spans": []},
   {"ftype": "macro", "name": "FRAC_OF", "a": 1, "k": 5, "x": 0, "result": 1}],
  1),
 # [140] g(1): 2x-5=1 -> x=3; g=3x+9=18 — the affine crown on a wild stranger
 (140, 300,
  D+"a, b, c. a is 3. 3 times a plus 9 times b equals c. It is known that b is 1. What is c?",
  D+"a, b, c, d, e, f, g. a is 3. b is 1. c is 3. c times a equals d. e is 9. e times b equals f. d plus f equals g. What is g?",
  [{"ftype": "given", "var": 0, "value": 3, "spans": []},
   {"ftype": "given", "var": 1, "value": 1, "spans": []},
   {"ftype": "macro", "name": "OP_APPLY", "op": "add", "k1": 3, "x": 0,
    "k2": 9, "y": 1, "result": 2}],
  2),
 # [138] (10x-3)^2 via 5x^2-3x=5 -> 20(5)+9=109: the sub-crown 20a+9b
 (138, 300,
  D+"a, b, c. a is 5. b is 1. 20 times a plus 9 times b equals c. What is c?",
  D+"a, b, c, d, e, f, g. a is 5. b is 1. c is 20. c times a equals d. e is 9. e times b equals f. d plus f equals g. What is g?",
  [{"ftype": "given", "var": 0, "value": 5, "spans": []},
   {"ftype": "given", "var": 1, "value": 1, "spans": []},
   {"ftype": "macro", "name": "OP_APPLY", "op": "add", "k1": 20, "x": 0,
    "k2": 9, "y": 1, "result": 2}],
  2),
]

REGISTRY = [
 (129, "radical-simplify-exponents"),
 (130, "no-real-solutions"),
 (133, "digit-structure-count"),
 (135, "parabola-square-noninteger"),
 (136, "nested-radical"),            # 2nd
 (144, "continued-fraction"),        # NEW family
 (147, "radical-form-answer"),       # 5th
]

tok = Tokenizer.from_file(TOKENIZER_JSON)
p = build_params(0)
CKPT = os.environ.get("GATE_CKPT", ".cache/crown_reader_v4.safetensors")
sd = safe_load(CKPT)
assert set(sd.keys()) == set(p.keys())
for k in p:
    p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
print(f"[b5t1] gate = {CKPT} (FTYPES=8) | pages {len(PAGES)} + {len(MACRO_PAGES)} macro | registry {len(REGISTRY)}")


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
    ok, votes, best = gate_dialect(dia, m, x["answer"], 500000 + 100 * li)
    if ok and best:
        facs, q = best
        banked.append(dict(text=x["problem"], factors=facs, query_var=q,
                           n_vars=24, m=m, decisions=[], mentions=[],
                           solution=[0] * 24,
                           gen=dict(src_idx=x["src_idx"], book=5, tranche=1,
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
    ok, votes, best = gate_dialect(pdia, m, x["answer"], 510000 + 100 * li)
    if ok:
        for floor, facs_, dia_, qv in (("macro", mfacs, mdia, q),
                                       ("prime", pf2, pdia, q_p)):
            banked.append(dict(text=x["problem"], factors=facs_, query_var=qv,
                               n_vars=24, m=m, decisions=[], mentions=[],
                               solution=[0] * 24,
                               gen=dict(src_idx=x["src_idx"], book=5, tranche=1,
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
              book=5, tranche=1) for li, fam in REGISTRY]

with open(".cache/book5_prose_pairs.jsonl", "w") as f:
    for b in banked:
        f.write(json.dumps(b) + "\n")
json.dump(certs, open(".cache/book5_organ_registry_t1.json", "w"))
print(f"\n[b5t1] banked rows: {len(banked)} ({macro_banked} macro pairs, "
      f"grammar {MACRO_GRAMMAR_VERSION}) | missed: {[m_[0] for m_ in missed]} | "
      f"certificates: {len(certs)}")
