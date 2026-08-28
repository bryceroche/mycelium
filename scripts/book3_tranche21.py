"""book3_tranche21.py — BOOK 3, TRANCHE 21 (sonnet L3 round 4, sealed 8/9): THE FIRST OP-GRAIN TRANCHE
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
 (180, 300, D + "a, b, c, d, e, f, g. a is 9. b is 12. a times b equals c. d is 2. e is 4. d times e equals f. c exceeds f by g. What is g?"),
 (188, 300, D + "a, b, c, d, e, f, g. a is 60. b is 9. c is 48. d is 5. b plus c equals e. e exceeds d by f. a exceeds f by g. What is g?"),
 (190, 300, D + "a, b, c, d, e. a is 5. b is 4. c is 3. a times b equals d. d times c equals e. What is e?"),
 (191, 300, D + "a, b, c, d, e. a is 16. b is 22. c is 13. b exceeds c by d. a exceeds d by e. What is e?"),
 (201, 300, D + "a, b, c, d. a is 120. When a is divided by 4, the quotient is b. c is 5. b times c equals d. What is d?"),
 (221, 300, D + "a, b, c, d, e. a is 87. b is 70. c is 100. a plus b equals d. d exceeds c by e. What is e?"),
 (222, 300, D + "a, b, c. a is 5. b is 4. a times b equals c. What is c?"),
 (227, 300, D + "a, b, c, d. a is 12. When a is divided by 2, the quotient is b. c is 11. b times c equals d. What is d?"),
 (231, 300, D + "a, b, c. a is 3. b is 4. a times b equals c. What is c?"),
 (234, 300, D + "a, b, c, d. a is 28. When a is divided by 2, the quotient is b. c is 5. b times c equals d. What is d?"),
]
OP_SPANS = {
 180: [("mul", "9 feet high and 12 feet long"), ("mul", "2-foot by 4-foot area"), ("addf", "will not have to paint")],
 188: [("addf", "9 dogs like watermelon, 48 dogs like salmon"), ("addf", "5 like both salmon and watermelon"), ("addf", "will not eat either")],
 190: [("mul", "1st-2nd-3rd place outcomes")],
 191: [("addf", "13 of the students who brought calculators are girls"), ("addf", "boys didn't bring their calculators to class")],
 201: [("fr", "$20\\%$")],
 221: [("addf", "$87$ indicated they liked Mozart and $70$ indicated they liked Bach"), ("addf", "minimum number of people surveyed who could have said they liked both")],
 222: [("mul", "cannot be the same person")],
 227: [("fr", "eleven lassis out of two mangoes"), ("mul", "twelve mangoes")],
 231: [("mul", "requires exactly one color and one pattern")],
 234: [("fr", "Two identical CDs"), ("mul", "five of these CDs")],
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
    texts = [dia] + [permuted_view(dia, 99996 + 100*ti + k) for k in range(1, 5)]
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
                           m=m, lane=row["lane"], book=3, tranche=21,
                           gate="5view-vote+answer-key", generation="41",
                           op_spans=spans))
with open(".cache/book3.jsonl", "a") as f:
    for b in banked:
        f.write(json.dumps(b) + "\n")
n_total = sum(1 for _ in open(".cache/book3.jsonl"))
print(f"[t8] banked {len(banked)}/8; book3 total {n_total}; "
      f"op-spans banked {sum(len(b['op_spans']) for b in banked)}", flush=True)
