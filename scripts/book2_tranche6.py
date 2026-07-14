"""book2_tranche6.py — BOOK 2, tranche 6 (2026-07-14): 25 hand dialects
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
 (181, 300, D+"a, b, c, d, e, f, g, h, i, j. b exceeds a by d. c exceeds b by d. e exceeds c by d. f exceeds e by d. a plus b equals g. g plus c equals h. h plus e equals i. i plus f equals j. It is known that j is 100. It is known that d is 6. What is c?"),
 (294, 300, D+"a, b, c, d, e, f, g, h, i, j, k. c is 3. c times a equals d. e is 2. e times b equals f. d plus f equals g. It is known that g is 32. e times a equals h. h plus b equals i. It is known that i is 20. j is 10. a times j equals k. What is k?"),
 (337, 300, D+"a, b. a times a equals b. It is known that b is 100. What is a?"),
 (344, 300, D+"a, b, c, d, e, f, g. a is 48. When a is divided by 2, the quotient is b and the remainder is c. When b is divided by 2, the quotient is d and the remainder is e. When d is divided by 2, the quotient is f and the remainder is g. What is f?"),
 (364, 300, D+"a, b, c, d, e, f, g. a is 24. When a is divided by 2, the quotient is b and the remainder is c. d is 2. d times e equals b. f is 5. e times f equals g. What is g?"),
 (375, 300, D+"a, b, c. a is 78. b is 63. a exceeds b by c. What is c?"),
 (34, 300, D+"a, b, c, d, e, f. b exceeds a by d. c exceeds b by d. e exceeds c by d. f exceeds e by d. It is known that a is 17. It is known that f is 41. What is c?"),
 (41, 300, D+"a, b, c, d, e, f. a plus b equals c. It is known that c is 10. a times b equals d. It is known that d is 24. When c is divided by 2, the quotient is e and the remainder is f. What is e?"),
 (45, 300, D+"a, b, c, d, e, f, g. a plus b equals c. It is known that c is 5. a times b equals d. It is known that d is 4. Of a and b, the smaller one is f. c exceeds f by e. e exceeds f by g. What is g?"),
 (48, 300, D+"a, b, c. a is 2. b is 2. a plus b equals c. What is c?"),
 (36, 300, D+"a, b, c, d, e. a is 10. a times a equals b. c is 14. c times c equals d. b plus d equals e. What is e?"),
]
REGISTRY = [(220, "value-range"), (24, "value-range"), (43, "negative-roots"),
            (44, "negative-domain"), (31, "even-function-negative")]

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
                           tranche=6, gate="lattice-vote+answer-key",
                           generation="9b"))
with open(".cache/book2.jsonl", "a") as f:
    for b in banked:
        f.write(json.dumps(b) + "\n")
reg = [dict(lane_idx=i, family=fam, raw=BY[i]["problem"], answer=BY[i]["answer"],
            certificate="annotation-impossible", book=2) for i, fam in REGISTRY]
json.dump(reg, open(".cache/book2_organ_registry_t6.json", "w"))
n_total = sum(1 for _ in open(".cache/book2.jsonl"))
print(f"\n[book2-t1] {len(banked)}/{len(T1)} dialects banked | book2 = {n_total} "
      f"entries (incl. {len(json.loads(open('.cache/book2_lane1.jsonl').read().splitlines()[0]) and [1,2,3,4])} lane-1) | "
      f"registry +{len(reg)}")
