"""book2_tranche1.py — BOOK 2, tranche 1 (2026-07-14): 25 hand dialects
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
T1 = [  # (lane-idx, m, dialect)
 (13, 300, D+"a, b, c, d, e, f, g. a plus b equals c. It is known that c is 32. d is 5. d times b equals e. e plus a equals f. It is known that f is 100. What is b?"),
 (15, 300, D+"a, b, c, d, e. a is 6. b is 3. a plus b equals c. a exceeds b by d. c times d equals e. What is e?"),
 (32, 999, D+"a, b, c. a is 729. b is 49. a exceeds b by c. What is c?"),
 (35, 300, D+"a, b, c. a is 60. When a is divided by 4, the quotient is b and the remainder is c. What is b?"),
 (46, 300, D+"a, b, c, d, e, f, g. a is 57. When a is divided by 19, the quotient is b and the remainder is c. d is 2. e is 4. e times b equals f. d plus f equals g. What is g?"),
 (49, 300, D+"a, b, c, d, e, f, g. a is 16. When a is divided by 2, the quotient is b and the remainder is c. d is 3. b times d equals e. When e is divided by 2, the quotient is f and the remainder is g. What is f?"),
 (62, 300, D+"a, b, c, d, e, f, g, h. c is 2. c times a equals d. e is 4. e times b equals f. d plus f equals g. It is known that g is 40. a plus b equals h. It is known that h is 15. What is a?"),
 (67, 300, D+"a, b, c, d, e, f. a plus b equals c. It is known that c is 26. When b is divided by 2, the quotient is d and the remainder is e. e is 0. a plus d equals f. It is known that f is 16. What is a?"),
 (69, 300, D+"a, b, c. a is 15. a times b equals c. It is known that c is 270. What is b?"),
 (101, 300, D+"a, b, c, d. a exceeds b by c. It is known that c is 24. a plus b equals d. It is known that d is 68. What is b?"),
 (105, 300, D+"a, b, c, d, e. a is 289. b is 85. a exceeds b by c. d is 34. d times e equals c. What is e?"),
 (119, 300, D+"a, b, c. a is 9. b is 4. a plus b equals c. What is c?"),
 (160, 300, D+"a, b, c. a is 120. When a is divided by 3, the quotient is b and the remainder is c. What is b?"),
 (216, 300, D+"a, b, c, d, e, f, g, h, i, j, k, l. c is 4. c times a equals d. e is 3. e times b equals f. d plus f equals g. It is known that g is 224. h is 2. h times a equals i. j is 5. j times b equals k. i plus k equals l. It is known that l is 154. What is b?"),
 (220, 999, D+"a, b, c, d, e. a is 80. a times b equals c. It is known that c is 400. d is 100. b times d equals e. What is e?"),
 (223, 300, D+"a, b, c, d, e, f. a is 4. b is 6. a times b equals c. d is 4. d times e equals c. e exceeds a by f. What is f?"),
 (294, 999, D+"a, b, c, d, e, f, g, h, i. c is 3. c times a equals d. e is 2. e times b equals f. d plus f equals g. It is known that g is 320. e times a equals h. h plus b equals i. It is known that i is 200. What is a?"),
 (1, 300, D+"a, b, c, d, e. a is 18. When a is divided by 2, the quotient is b and the remainder is c. d is 3. b times d equals e. What is e?"),
 (5, 300, D+"a, b, c, d, e, f, g, h. b exceeds a by c. It is known that c is 2. d is 9. d times a equals e. f is 5. f times b equals g. e plus g equals h. It is known that h is 108. What is b?"),
 (4, 300, D+"a, b, c, f, g, h, i, j. a is 3. a times b equals c. c exceeds f by g. It is known that g is 7. h is 5. h times b equals i. j is 2. j times f equals i. What is b?"),
 (2, 300, D+"a, b, c, d, g, h, k, j, m, l. a is 4. When c is divided by 4, the quotient is b and the remainder is d. It is known that b is 6. d is 0. g is 6. c exceeds g by h. k is 2. k times j equals l. m is 3. m times h equals l. What is j?"),
 (6, 300, D+"a, b, c, d, e, f, g, h. a is 2. b is 4. a times b equals c. c exceeds d by e. It is known that e is 8. c exceeds f by g. It is known that g is 4. d plus f equals h. What is h?"),
 (8, 300, D+"a, b, c, d, e. a is 256. b is 64. b times c equals a. d is 25. d times c equals e. What is e?"),
 (9, 300, D+"a, b, c, d, e, f, g. c is 8. When c is divided by 4, the quotient is d and the remainder is e. f is 4. d plus f equals g. What is g?"),
 (7, 300, D+"a, b, c, d, e. a is 3. b is 3. c is 3. a times b equals d. d times c equals e. What is e?"),
]
REGISTRY = [  # annotation-impossible: (idx, family)
 (74, "exponent-laws"), (83, "polynomial-expansion"), (54, "cube-roots"),
 (89, "sqrt-of-product"), (90, "negative-roots"), (314, "iterated-halving+compare"),
 (145, "negative-domain"), (0, "distance-formula-sqrt"), (3, "distance-formula-sqrt"),
]

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
                           tranche=1, gate="lattice-vote+answer-key",
                           generation="9b"))
with open(".cache/book2.jsonl", "w") as f:
    for b in [json.loads(l) for l in open(".cache/book2_lane1.jsonl")]:
        f.write(json.dumps(dict(b, tranche=1)) + "\n")
    for b in banked:
        f.write(json.dumps(b) + "\n")
reg = [dict(lane_idx=i, family=fam, raw=BY[i]["problem"], answer=BY[i]["answer"],
            certificate="annotation-impossible", book=2) for i, fam in REGISTRY]
json.dump(reg, open(".cache/book2_organ_registry.json", "w"))
n_total = sum(1 for _ in open(".cache/book2.jsonl"))
print(f"\n[book2-t1] {len(banked)}/{len(T1)} dialects banked | book2 = {n_total} "
      f"entries (incl. {len(json.loads(open('.cache/book2_lane1.jsonl').read().splitlines()[0]) and [1,2,3,4])} lane-1) | "
      f"registry +{len(reg)}")
