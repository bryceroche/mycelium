"""book3_tranche10.py — BOOK 3, TRANCHE 10: THE FIRST OP-GRAIN TRANCHE
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
 (200, 300, D + "a, b. a is 108. When a is divided by 12, the quotient is b. What is b?"),
 (202, 300, D + "a, b, c, d, e, f, g. a is 5. a times a equals b. c is 4. b plus c equals d. e is 6. e times e equals f. g is 7. f exceeds d by g. What is d?"),
 (170, 300, D + "a, b, c, d, e, f, g, h, i, j, k. a is 5. a times a equals b. b times a equals c. d is 4. d times d equals e. e times d equals f. c plus f equals g. a times d equals h. b plus e equals i. i exceeds j by h. j times k equals g. What is k?"),
 (206, 300, D + "a, b, c, d, e, f. a is 14. b is 22. a plus b equals c. d is 36. c plus d equals e. When e is divided by 3, the quotient is f. What is f?"),
 (209, 300, D + "a, b, c, d. b is 3. a times b equals c. c plus a equals d. d is 8. What is a?"),
 (217, 300, D + "a, b, c, d, e, f, g, h. a is 2. b is 8. a times b equals c. d is 6. When d is divided by 3, the quotient is e. e plus c equals f. h is 10. f exceeds g by h. What is g?"),
 (220, 300, D + "a, b, c, d. a is 8. a times a equals b. c is 2. b plus c equals d. What is d?"),
 (229, 300, D + "a, b, c, d. a is 9. a times a equals b. d is 7. b exceeds c by d. What is c?"),
 (230, 300, D + "a, b, c, d. a is 3. a times b equals c. c plus a equals d. d is 9. What is b?"),
 (223, 300, D + "a, b, c, d, e, f. a is 14. When a is divided by 2, the quotient is b. c is 3. c times d equals e. e is 18. b plus d equals f. What is f?"),
 (228, 300, D + "a, b, c, d, e, f. a is 12. b is 10. a plus b equals c. c exceeds d by 5. e is 25. e exceeds f by d. What is f?"),
]
OP_SPANS = {   # per lane_idx: (op, cue substring in the RAW text)
 200: [("fr", "each side has length")],
 202: [("sq", "perfect square"), ("addf", "4 greater than"), ("addf", "7 less than")],
 170: [("sq", "a^2")],
 206: [("fr", "arithmetic mean")],
 209: [("fr", "\\frac {a} {3}")],
 217: [("fr", "\\div"), ("mul", "\\cdot")],
 220: [("sq", "\\sqrt"), ("addf", "x - 2")],
 229: [("sq", "\\sqrt"), ("addf", "x+ 7")],
 230: [("addf", "n + (n + 1)")],
 223: [("fr", "\\div"), ("mul", "\\cdot"), ("addf", "2+3+4+5")],
 228: [("addf", "drank both"), ("addf", "neither coffee nor tea")],
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
    texts = [dia] + [permuted_view(dia, 99500 + 100*ti + k) for k in range(1, 5)]
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
                           m=m, lane=row["lane"], book=3, tranche=10,
                           gate="5view-vote+answer-key", generation="41",
                           op_spans=spans))
with open(".cache/book3.jsonl", "a") as f:
    for b in banked:
        f.write(json.dumps(b) + "\n")
n_total = sum(1 for _ in open(".cache/book3.jsonl"))
print(f"[t8] banked {len(banked)}/8; book3 total {n_total}; "
      f"op-spans banked {sum(len(b['op_spans']) for b in banked)}", flush=True)
