"""book3_tranche18.py — BOOK 3, TRANCHE 18 (sonnet L3 round 3, sealed 15/15): THE FIRST OP-GRAIN TRANCHE
(2026-08-27, word given). 8 L2 rows, surgery = dialect translation +
THE OP-SPAN DUTY (per op factor, the raw-text cue phrase that realizes
it — located as substrings, banked as char offsets; partial coverage
allowed, honesty over completeness). Gate = the DEPLOYED stack (gen-41
manifest: g41_onemass_refold, FTYPES=8), 5 views, vote >= 3, the key
disposes. Census fixture untouched.
"""
import json, sys, os
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
os.environ.update({"DEV": "AMD", "ALG2": "1", "ALG_FTYPES": "8",
                   "ALG_DUP": "1", "ALG_HW": "512", "ALG_WIDE": "1"})
import numpy as np
from collections import Counter
from phase1_algebra_head import (T_ALG, build_params, forward, decode,
                                 sent_indices, TOKENIZER_JSON)
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
 (12, 300, D + "a, b. a is 55. When a is divided by 11, the quotient is b. What is b?"),
 (33, 300, D + "a, b, c. a is 3. b is 2. a times b equals c. What is c?"),
 (88, 300, D + "a, b, c, d, e. a is 23. b is 27. b exceeds a by c. d is 2. c times d equals e. What is e?"),
 (89, 300, D + "a, b, c, d. a is 28. b is 3. a times b equals c. When c is divided by 7, the quotient is d. What is d?"),
 (100, 300, D + "a, b, c, d, e. a is 6. b is 10. a plus b equals c. d is 12. c exceeds d by e. What is e?"),
 (102, 300, D + "a, b, c. a is 6. a times a equals b. When b is divided by 2, the quotient is c. What is c?"),
 (103, 300, D + "a, b, c, d. a is 100. When a is divided by 20, the quotient is b. c is 7. b times c equals d. What is d?"),
 (105, 300, D + "a, b. a is 78. When a is divided by 39, the quotient is b. What is b?"),
 (106, 300, D + "a, b, c, d, e. a is 4. b is 3. a times b equals c. d is 2. c times d equals e. What is e?"),
 (112, 300, D + "a, b, c, d. a is 28. When a is divided by 7, the quotient is b. c is 5. b times c equals d. What is d?"),
 (117, 300, D + "a, b, c, d. a is 40. When a is divided by 10, the quotient is b. c is 15. b times c equals d. What is d?"),
 (119, 300, D + "a, b, c. a is 3. b is 2. a times b equals c. What is c?"),
 (120, 300, D + "a, b, c, d. a is 200. When a is divided by 50, the quotient is b. c is 3. b times c equals d. What is d?"),
 (122, 300, D + "a, b, c, d. a is 15. When a is divided by 5, the quotient is b. c is 4. b times c equals d. What is d?"),
 (123, 300, D + "a, b, c, d. a is 60. When a is divided by 30, the quotient is b. c is 2. b times c equals d. What is d?"),
]
OP_SPANS = {   # per lane_idx: (op, cue substring in the RAW text)
 12: [("fr", "grandfather was 55 years old when Andrew was born")],
 33: [("mul", "how many different positive two-digit integers can be formed")],
 88: [("addf", "The average of the numbers 23 and $x$ is 27"), ("mul", "positive difference between 23 and $x$")],
 89: [("mul", "one of the canoes weighs a total of 28 pounds"), ("fr", "Seven identical bowling balls weigh the same as three identical canoes")],
 100: [("addf", "Only six slices have pepperoni, and exactly ten slices have mushrooms"), ("addf", "A 12-slice pizza was made with only pepperoni and mushroom toppings, and every slice has at least one topping")],
 102: [("fr", "area of triangle")],
 103: [("fr", "was discounted"), ("mul", "was reduced by")],
 105: [("fr", "have 78 marbles")],
 106: [("mul", "In how many orders")],
 112: [("fr", "Jimmy has $28$ oranges"), ("mul", "$7$ oranges weigh the same as $5$ apples")],
 117: [("fr", "40-foot tree next to her is casting a 10-foot shadow"), ("mul", "casting a 15-inch shadow at the same time")],
 119: [("mul", "in how many ways can these three be chosen to be the three officers")],
 120: [("fr", "the plane is carrying 200 passengers"), ("mul", "ten percent of those women are in first class")],
 122: [("fr", "a rate of $4 per five pounds"), ("mul", "how many dollars does it cost to buy 15 pounds")],
 123: [("fr", "A recipe for 30 cookies requires two cups of flour"), ("mul", "Eduardo wants to bake five dozen cookies")],
}

tok = Tokenizer.from_file(TOKENIZER_JSON)
p = build_params(0)
sd = safe_load(".cache/g41_onemass_refold.safetensors")
assert set(sd.keys()) == set(p.keys()), "gate ckpt/env mismatch (eval-load law)"
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
            (("sel",) if "sel" in out else ()) + (("dup",) if "dup" in out else ()) + \
            (("sgn",) if "sgn" in out else ())
        o = {k: out[k].realize().numpy() for k in keys}
        for bi in range(8):
            if s0 + bi < n:
                res.append(decode({k: o[k][bi] for k in o}))
    return res

banked = []
for ti, (li, m, dia) in enumerate(T1):
    row = BY[li]
    gold = row["answer"]
    raw = row.get("problem") or row.get("raw", "")
    texts = [dia] + [permuted_view(dia, 99998 + 100*ti + k) for k in range(1, 5)]
    votes = []
    for facs, q in parse_batch(texts):
        try:
            a = solve2(facs, q, {"n_vars": 24, "m": m})
        except Exception:
            a = None
        if a is not None:
            votes.append(a)
    top, cnt = (Counter(votes).most_common(1)[0] if votes else (None, 0))
    ok = cnt >= 3 and top == gold
    spans = []
    for op, cue in OP_SPANS.get(li, []):
        pos = raw.find(cue)
        if pos >= 0:
            spans.append({"op": op, "span": [pos, pos + len(cue)], "cue": cue})
        else:
            print(f"  [{li}] SPAN MISS: {cue!r}", flush=True)
    print(f"  [{li:3d}] gold {gold:>4} | votes {votes} | spans {len(spans)} -> "
          f"{'BANKS' if ok else 'refuses'}", flush=True)
    if ok:
        banked.append(dict(lane_idx=li, raw=raw, dialect=dia, answer=gold,
                           m=m, lane=row["lane"], book=3, tranche=18,
                           gate="5view-vote+answer-key", generation="41",
                           op_spans=spans))
with open(".cache/book3.jsonl", "a") as f:
    for b in banked:
        f.write(json.dumps(b) + "\n")
n_total = sum(1 for _ in open(".cache/book3.jsonl"))
print(f"[t8] banked {len(banked)}/8; book3 total {n_total}; "
      f"op-spans banked {sum(len(b['op_spans']) for b in banked)}", flush=True)
