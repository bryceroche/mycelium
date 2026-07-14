"""book2_tranche4.py — BOOK 2, tranche 4 (2026-07-14): 25 hand dialects
through the gen-9b gate (5 views, vote>=3, key disposes); 9 organ-registry
certificates for annotation-impossible items. Stratification: 12 L2
repairs / 5 rate-family / 8 style-middle. Census pool untouched (fixture).
"""
import json, sys, os
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np
from collections import Counter
os.environ.setdefault("ALG2", "1"); os.environ.setdefault("ALG_FTYPES", "6")
os.environ.setdefault("ALG_DUP", "1")
from phase1_algebra_head import T_ALG, build_params, forward, decode, sent_indices, TOKENIZER_JSON
from beacon_closing_arm import recompute_states
from tta_views import permuted_view
from tta_alg2_dials import solve2
from tokenizers import Tokenizer
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load

L = json.load(open(".cache/book2_lanes.json"))
BY = {l["idx"]: l for l in L}
D = "Consider the numbers "
T1 = [
 (238, 300, D+"a, b, c, d, e, f, g. a times a equals c. b times b equals d. c exceeds d by e. It is known that e is 12. a plus b equals f. It is known that f is 6. a exceeds b by g. What is g?"),
 (271, 300, D+"a, b, c. a is 6. b is 205. The remainder when b is divided by 2 is c. It is known that c is 1. What is a?"),
 (298, 300, D+"a, b, c, d. a is 24. When a is divided by 4, the quotient is b and the remainder is c. b times b equals d. What is d?"),
 (304, 300, D+"a, b, c, d, e. a plus b equals c. a times b equals d. It is known that d is 3. Of a and b, the larger one is e. It is known that e is 3. What is c?"),
 (316, 300, D+"a, b, c, d, e, f. a is 4. b is 25. a times b equals c. d times d equals c. e is 2. d times e equals f. What is f?"),
 (322, 300, D+"a, b, c, d. a plus b equals c. It is known that c is 11. a exceeds b by d. It is known that d is 3. What is a?"),
 (333, 300, D+"a, b, c, d. a plus b equals c. It is known that c is 4. a times b equals d. It is known that d is 4. What is a?"),
 (354, 300, D+"a, b, c. a is 2. b is 2. a exceeds b by c. What is c?"),
 (373, 300, D+"a, b, c. a is 4. b is 4. a plus b equals c. What is c?"),
 (382, 300, D+"a, b, c, d. a is 144. b is 25. a plus b equals c. d times d equals c. What is d?"),
 (284, 300, D+"a, b, c, d, e. a plus b equals c. It is known that c is 31. a exceeds b by d. It is known that d is 3. Of a and b, the larger one is e. What is e?"),
 (285, 300, D+"a, b, c, d, e, f, g, h. a is 16. b is 9. a plus b equals c. d times d equals c. e is 2. d times e equals f. g is 6. f plus g equals h. What is h?"),
 (283, 300, D+"a, b. a times a equals b. It is known that b is 4. What is a?"),
 (282, 300, D+"a, b, c. a is 4. a times b equals c. It is known that c is 8. What is b?"),
]
REGISTRY = [(281, "exponent-laws"), (288, "logarithms"),
            (289, "fractional-exponents"), (286, "negative-rational-roots")]

tok = Tokenizer.from_file(TOKENIZER_JSON)
p = build_params(0)
sd = safe_load(os.environ.get("GATE_CKPT", ".cache/phase1_gen9b_head.safetensors"))
assert set(sd.keys()) == set(p.keys())
for k in p:
    p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()

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
        keys = ("pres","ftype","op","islit","dig","args","res","query") + \
            (("sel",) if "sel" in out else ()) + (("dup",) if "dup" in out else ())
        o = {k: out[k].realize().numpy() for k in keys}
        for bi in range(8):
            if s0 + bi < n:
                res.append(decode({k: o[k][bi] for k in o}))
    return res

banked = []
for ti, (li, m, dia) in enumerate(T1):
    row = BY[li]
    gold = row["answer"]
    texts = [dia] + [permuted_view(dia, 95000 + 100*ti + k) for k in range(1, 5)]
    votes = []
    for facs, q in parse_batch(texts):
        a = solve2(facs, q, {"n_vars": 24, "m": m})
        if a is not None:
            votes.append(a)
    top, cnt = (Counter(votes).most_common(1)[0] if votes else (None, 0))
    ok = cnt >= 3 and top == gold
    print(f"  [{li:3d}] gold {gold:>4} | votes {votes} -> "
          f"{'BANKS' if ok else 'refuses'}")
    if ok:
        banked.append(dict(lane_idx=li, raw=row["problem"], dialect=dia,
                           answer=gold, m=m, lane=row["lane"], book=2,
                           tranche=4, gate="lattice-vote+answer-key",
                           generation="9b"))
with open(".cache/book2.jsonl", "a") as f:
    for b in banked:
        f.write(json.dumps(b) + "\n")
reg = [dict(lane_idx=i, family=fam, raw=BY[i]["problem"], answer=BY[i]["answer"],
            certificate="annotation-impossible", book=2) for i, fam in REGISTRY]
json.dump(reg, open(".cache/book2_organ_registry_t4.json", "w"))
n_total = sum(1 for _ in open(".cache/book2.jsonl"))
print(f"\n[book2-t1] {len(banked)}/{len(T1)} dialects banked | book2 = {n_total} "
      f"entries (incl. {len(json.loads(open('.cache/book2_lane1.jsonl').read().splitlines()[0]) and [1,2,3,4])} lane-1) | "
      f"registry +{len(reg)}")
