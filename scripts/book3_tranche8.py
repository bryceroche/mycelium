"""book3_tranche8.py — BOOK 3, TRANCHE 8: THE FIRST OP-GRAIN TRANCHE
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
 (6, 300, D + "a, b, c, d. a is 9. When a is divided by 3, the quotient is b. c is 7. b times c equals d. What is d?"),
 (17, 300, D + "a, b, c, d. a plus b equals c. c is 180. a plus d equals b. When b is divided by 2, the quotient is d. What is b?"),
 (18, 300, D + "a, b, c, d, e, f. a is 14. b is 3. a plus b equals c. d is 4. c exceeds e by d. e plus b equals f. What is f?"),
 (52, 300, D + "a, b, c, d, e, f. a is 25. b is 20. a plus b equals c. When c is divided by 5, the quotient is d. e is 2. d times e equals f. What is f?"),
 (92, 300, D + "a, b, c, d, e. a is 8. b is 3. a times b equals c. d is 4. c times d equals e. What is e?"),
 (114, 300, D + "a, b, c, d, e, f. a is 3. b is 5. a plus b equals c. d is 240. When d is divided by 8, the quotient is e. b times e equals f. What is f?"),
 (148, 300, D + "a, b, c, d, e. a is 27. b is 2. b times c equals d. a plus d equals e. e is 39. What is c?"),
 (164, 300, D + "a, b, c, d. a is 3. b is 4. a times b equals c. When c is divided by 6, the quotient is d. What is d?"),
]
OP_SPANS = {   # per lane_idx: (op, cue substring in the RAW text)
 6:   [("fr", "increases by 9 units"), ("mul", "increases by 7 units")],
 17:  [("addf", "sum of"), ("addf", "subtract"), ("fr", "half of")],
 18:  [("addf", "three years younger than"), ("addf", "four years older than"),
       ("addf", "three years older than")],
 52:  [("addf", "add 20 more"), ("fr", "ratio of irises to roses"),
       ("mul", "keep the same ratio")],
 92:  [("mul", "combinations are possible")],
 114: [],
 148: [("mul", "twice"), ("addf", "increased by")],
 164: [("mul", "\\times"), ("fr", "\\dfrac")],
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
    texts = [dia] + [permuted_view(dia, 98000 + 100*ti + k) for k in range(1, 5)]
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
                           m=m, lane=row["lane"], book=3, tranche=8,
                           gate="5view-vote+answer-key", generation="41",
                           op_spans=spans))
with open(".cache/book3.jsonl", "a") as f:
    for b in banked:
        f.write(json.dumps(b) + "\n")
n_total = sum(1 for _ in open(".cache/book3.jsonl"))
print(f"[t8] banked {len(banked)}/8; book3 total {n_total}; "
      f"op-spans banked {sum(len(b['op_spans']) for b in banked)}", flush=True)
