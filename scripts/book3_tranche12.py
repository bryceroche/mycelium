"""book3_tranche12.py — BOOK 3, TRANCHE 12: THE FIRST OP-GRAIN TRANCHE
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
 (378, 300, D + "a, b, c, d, e. a is 10. b is 10. a plus b equals c. d is 18. c exceeds e by d. What is e?"),
 (233, 300, D + "a, b, c, d. a is 192. When a is divided by 6, the quotient is b. c is 8. b times c equals d. What is d?"),
 (245, 300, D + "a, b, c, d, e, f, g, h. a is 5. b is 6. a plus b equals c. d is 7. c plus d equals e. f is 180. When f is divided by 18, the quotient is g. g times a equals h. What is h?"),
 (259, 300, D + "a, b, c, d, e, f, g, h, i, j. a is 4. b is 6. a times b equals c. d is 8. e is 3. d times e equals f. c plus f equals g. h is 28. When h is divided by 2, the quotient is i. g exceeds j by i. What is j?"),
 (263, 300, D + "a, b, c, d, e. a is 2. b is 87. a times b equals c. e is 1. c exceeds d by e. What is d?"),
 (264, 300, D + "a, b, c, d, e, f, g, h, i. a is 5. a times a equals b. c is 3. d is 4. c times d equals e. b exceeds f by e. g is 3. g times g equals h. f plus h equals i. What is i?"),
 (238, 300, D + "a, b, c, d, e, f, g. a is 5. b is 4. a times b equals c. d is 3. c times d equals e. f is 2. e times f equals g. What is g?"),
 (265, 300, D + "a, b, c, d, e. a is 8. b is 6. a exceeds c by b. d is 10. c times d equals e. What is e?"),
]
OP_SPANS = {   # per lane_idx: (op, cue substring in the RAW text)
 378: [("addf", "at least one topping"), ("addf", "both")],
 233: [("fr", "on 6 gallons"), ("mul", "on 8 gallons")],
 245: [("fr", "5:6:7 ratio")],
 259: [("mul", "\\cdot"), ("fr", "\\div")],
 263: [("mul", "87^"), ("addf", "odd positive integer")],
 264: [("sq", "5^2"), ("mul", "3(4)"), ("sq", "3^2")],
 238: [("mul", "distinct arrangements")],
 265: [("addf", "minutes after"), ("mul", "10-mile")],
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
    texts = [dia] + [permuted_view(dia, 99850 + 100*ti + k) for k in range(1, 5)]
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
                           m=m, lane=row["lane"], book=3, tranche=12,
                           gate="5view-vote+answer-key", generation="41",
                           op_spans=spans))
with open(".cache/book3.jsonl", "a") as f:
    for b in banked:
        f.write(json.dumps(b) + "\n")
n_total = sum(1 for _ in open(".cache/book3.jsonl"))
print(f"[t8] banked {len(banked)}/8; book3 total {n_total}; "
      f"op-spans banked {sum(len(b['op_spans']) for b in banked)}", flush=True)
