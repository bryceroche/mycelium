"""book4_tranche4.py — BOOK 4, tranche 4 (2026-07-17): 7 prime pages +
1 macro crown ([51]: (x-y)^2 = (x+y)^2 - 4xy — a textbook identity
wearing the k1=1 sub-crown) + 12 registry certificates (incl. 2nd/3rd
family certificates: piecewise, value-range, radical-form). Same gate,
same law. One cap casualty logged: [68]'s 27a+10b=600 crown dies at the
value cap (600 > 300) — a beautiful crown lost to domain, certificated.
"""
import json, sys, os
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np
from collections import Counter
os.environ.setdefault("ALG2", "1"); os.environ.setdefault("ALG_FTYPES", "6")
os.environ.setdefault("ALG_DUP", "1")
from phase1_algebra_head import T_ALG, L_FAC, build_params, forward, decode, sent_indices, TOKENIZER_JSON
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
 (55, 300, D+"a, b, c, d, e. a times a equals b. c is 1. b plus c equals d. e is 2. e times a equals d. What is a?", True, "v1"),
 (63, 300, D+"a, b, c, d, e, f, g, h, i. b is 3. b times a equals c. It is known that c is 3. d is 2. d times a equals e. f is 5. f times a equals g. e plus g equals h. h exceeds i by b. What is i?", True, "v1"),
 (65, 300, D+"a, b, c, d, e. a is 49. b is 96. a plus b equals c. d is 3. c plus d equals e. What is e?", True, "v1"),
 (69, 300, D+"a, b, c, d, e. a is 7. b is 3. a plus b equals c. d is 5. d times e equals c. What is e?", True, "v1"),
 (73, 300, D+"a, b, c, d, e, f, g, h, i, j. a is 56. When a is divided by 4, the quotient is b and the remainder is c. d is 3. a times d equals e. When e is divided by 7, the quotient is f and the remainder is g. b plus f equals h. When h is divided by 2, the quotient is i and the remainder is j. What is i?", True, "v1"),
 (74, 300, D+"a, b, c, d, e. a is 5. a times b equals c. It is known that c is 20. d is 1. b plus d equals e. What is e?", True, "v1"),
 (79, 300, D+"a, b, c, d. a times a equals b. It is known that b is 64. c is 3. a times c equals d. What is d?", True, "v1"),
]

MACRO_PAGES = [
 # [51] (x+y)^2=45, xy=10 -> (x-y)^2 = 45 - 4*10 = 5 (identity explicitated)
 (51, 300,
  D+"a, b, c. a is 45. b is 10. a minus 4 times b equals c. What is c?",
  D+"a, b, c, d, e. a is 45. b is 10. c is 4. c times b equals d. a exceeds d by e. What is e?",
  [{"ftype": "given", "var": 0, "value": 45, "spans": []},
   {"ftype": "given", "var": 1, "value": 10, "spans": []},
   {"ftype": "macro", "name": "OP_APPLY", "op": "sub", "k1": 1, "x": 0,
    "k2": 4, "y": 1, "result": 2}],
  2),
]

REGISTRY = [
 (52, "lattice-counting"),
 (53, "rounding-sum"),
 (58, "functional-identity"),
 (60, "rational-inverse"),
 (61, "piecewise-inverse"),          # 2nd
 (62, "value-range"),                # 3rd
 (67, "region-counting"),
 (68, "value-range-diophantine"),    # the cap casualty (27a+10b=600)
 (71, "exponent-cases"),
 (76, "factoring-diophantine"),
 (77, "rational-coefficients"),
 (78, "radical-form-answer"),        # 2nd
]

tok = Tokenizer.from_file(TOKENIZER_JSON)
p = build_params(0)
CKPT = os.environ.get("GATE_CKPT", ".cache/phase1_gen14_head.safetensors")
sd = safe_load(CKPT)
assert set(sd.keys()) == set(p.keys())
for k in p:
    p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
print(f"[b4t4] gate = {CKPT} | pages {len(PAGES)} + {len(MACRO_PAGES)} macro | registry {len(REGISTRY)}")


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
        keys = ("pres", "ftype", "op", "islit", "dig", "args", "res", "query") + \
            (("sel",) if "sel" in out else ()) + (("dup",) if "dup" in out else ())
        o = {k: out[k].realize().numpy() for k in keys}
        for bi in range(8):
            if s0 + bi < n:
                res.append(decode({k: o[k][bi] for k in o}))
    return res


def gate_dialect(text, m, answer, seed0):
    texts = [text] + [permuted_view(text, seed0 + k) for k in range(1, 5)]
    parses = parse_batch(texts)
    votes, best = [], None
    for facs, q in parses:
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
    ok, votes, best = gate_dialect(dia, m, x["answer"], 200000 + 100 * li)
    if ok and best:
        facs, q = best
        banked.append(dict(text=x["problem"], factors=facs, query_var=q,
                           n_vars=24, m=m, decisions=[], mentions=[],
                           solution=[0] * 24,
                           gen=dict(src_idx=x["src_idx"], book=4, tranche=4,
                                    floor="prime", fs=fs, ver=ver, dialect=dia,
                                    gate="5view-vote+key", generation="14")))
        print(f"  [page {li}] BANKED (votes {votes})")
    else:
        missed.append((li, votes))
        print(f"  [page {li}] MISS (votes {votes})")

macro_banked = 0
for li, m, mdia, pdia, mfacs, q in MACRO_PAGES:
    x = BY[li]
    pfacs, nv = expand_graph(mfacs, 24)
    a = solve2(pfacs, q, {"n_vars": 24, "m": m})
    assert a == x["answer"], (li, a, x["answer"])
    dg_m, _ = canon({"factors": mfacs, "n_vars": 24, "query_var": q})
    dg_p, _ = canon({"factors": pfacs, "n_vars": nv, "query_var": q})
    assert dg_m == dg_p, (li, dg_m, dg_p)
    ok, votes, best = gate_dialect(pdia, m, x["answer"], 210000 + 100 * li)
    if ok:
        for floor, facs_, dia_ in (("macro", mfacs, mdia), ("prime", pfacs, pdia)):
            banked.append(dict(text=x["problem"], factors=facs_, query_var=q,
                               n_vars=24, m=m, decisions=[], mentions=[],
                               solution=[0] * 24,
                               gen=dict(src_idx=x["src_idx"], book=4, tranche=4,
                                        floor=floor, fs=True, dialect=dia_,
                                        knot=dg_m, grammar=MACRO_GRAMMAR_VERSION,
                                        gate="5view-vote+key(prime-twin)",
                                        generation="14")))
        macro_banked += 1
        print(f"  [MACRO {li}] BANKED both floors, one knot {dg_m[:12]} (votes {votes})")
    else:
        missed.append((li, votes))
        print(f"  [MACRO {li}] prime-twin MISS (votes {votes})")

certs = [dict(lane_idx=li, family=fam, raw=BY[li]["problem"],
              answer=BY[li]["answer"], src_idx=BY[li]["src_idx"],
              book=4, tranche=4) for li, fam in REGISTRY]

with open(".cache/book4_prose_pairs.jsonl", "a") as f:
    for b in banked:
        f.write(json.dumps(b) + "\n")
json.dump(certs, open(".cache/book4_organ_registry_t4.json", "w"))
print(f"\n[b4t4] banked rows: {len(banked)} ({macro_banked} macro pairs) | "
      f"missed: {[m_[0] for m_ in missed]} | certificates: {len(certs)}")
