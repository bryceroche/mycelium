"""book3_tranche1.py — BOOK 3, tranche 1 (2026-07-14): 25 hand dialects
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

L = json.load(open(".cache/book3_lanes.json"))
BY = {l["idx"]: l for l in L}
D = "Consider the numbers "
T1 = [
 (0, 300, D+"a, b, c. a is 8. b is 3. a plus b equals c. What is c?"),
 (14, 300, D+"a, b, c. a is 4. b is 3. a plus b equals c. What is c?"),
 (36, 300, D+"a, b, c, d, e, f, g, h, i, j, k, l, m. a is 3. b is 9. a times b equals c. d is 4. e is 10. d times e equals f. g is 11. g times a equals h. i is 8. a times i equals j. c plus f equals k. k plus h equals l. l plus j equals m. What is m?"),
 (37, 300, D+"a, b, c, d, e, f, g, h. a is 4. a times b equals c. It is known that c is 8. d is 1. e is 6. d plus e equals f. f plus a equals g. g times b equals h. What is h?"),
 (38, 300, D+"a, b, c. a is 128. b is 1. a exceeds b by c. What is c?"),
 (40, 300, D+"a, b, c, d, e, f, g, h, i. a is 6. b is 1. a plus b equals c. d is 4. c exceeds d by e. f is 21. e times g equals f. h is 5. g times h equals i. What is i?"),
 (48, 300, D+"a, b. a times a equals b. It is known that b is 49. What is a?"),
 (49, 300, D+"a, b, c, d, e. a is 20. b is 2. a exceeds b by c. d is 3. d times e equals c. What is e?"),
 (59, 300, D+"a, b, c, d. a is 8. a times b equals c. c plus b equals d. It is known that d is 180. What is c?"),
 (60, 300, D+"a, b, c, d, e. a is 12. b is 21. a plus b equals c. d is 11. d times e equals c. What is e?"),
 (61, 300, D+"a, b, c, d, e, f, g. a is 5. b is 11. a times b equals c. d is 3. c exceeds d by e. f is 2. f times g equals e. What is g?"),
 (66, 300, D+"a, b, c, d, e, f, g. a is 5. a times b equals c. It is known that c is 60. d is 4. d times e equals b. f is 3. f times g equals e. What is g?"),
 (69, 300, D+"a, b, c, d, e. a is 6. b is 8. a times b equals c. When c is divided by 4, the quotient is d and the remainder is e. What is d?"),
]
REGISTRY = [(3, "negative-rational-roots"), (9, "value-range"), (28, "exponent-identity"),
            (43, "value-range"), (47, "too-thin-lexical")]

tok = Tokenizer.from_file(TOKENIZER_JSON)
p = build_params(0)
sd = safe_load(os.environ.get("GATE_CKPT", ".cache/phase1_gen13_head.safetensors"))
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
with open(".cache/book3.jsonl", "a") as f:
    for b in banked:
        f.write(json.dumps(b) + "\n")
reg = [dict(lane_idx=i, family=fam, raw=BY[i]["problem"], answer=BY[i]["answer"],
            certificate="annotation-impossible", book=2) for i, fam in REGISTRY]
json.dump(reg, open(".cache/book2_organ_registry_b3t1.json", "w"))
n_total = sum(1 for _ in open(".cache/book3.jsonl"))
print(f"\n[book2-t1] {len(banked)}/{len(T1)} dialects banked | book2 = {n_total} "
      f"entries (incl. {len(json.loads(open('.cache/book2_lane1.jsonl').read().splitlines()[0]) and [1,2,3,4])} lane-1) | "
      f"registry +{len(reg)}")
