"""book3_tranche19.py — BOOK 3, TRANCHE 19 (sonnet L3 round 4, sealed 8/9): THE FIRST OP-GRAIN TRANCHE
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
 (10, 300, D + "a, b, c. a is 4. b is 3. a plus b equals c. What is c?"),
 (32, 300, D + "a, b, c. a is 6. b is 2. a times b equals c. What is c?"),
 (57, 300, D + "a, b, c, d, e, f, g. a is 65. b is 43. a plus b equals c. d is 10. c exceeds e by d. f is 100. f exceeds g by e. What is g?"),
 (124, 300, D + "a, b, c. a is 3. b is 2. a times b equals c. What is c?"),
 (127, 300, D + "a, b, c. a is 24. b is 6. a times b equals c. What is c?"),
 (137, 300, D + "a, b, c. a is 6. b is 5. a plus b equals c. What is c?"),
 (144, 300, D + "a, b, c, d, e, f, g. a is 40. b is 18. c is 15. d is 12. a exceeds e by d. b plus c equals f. f exceeds g by e. What is g?"),
 (145, 300, D + "a, b. a is 72. When a is divided by 3, the quotient is b. What is b?"),
]
OP_SPANS = {
 10: [("addf", "x^2-4x-14=3x+16")],
 32: [("mul", "Bryce received 6 more raisins than Carter, and Carter received half the number of raisins Bryce received")],
 57: [("addf", "65 take mathematics, 43 take physics"), ("addf", "10 students take both mathematics and physics"), ("addf", "100 students in the science club")],
 124: [("mul", "three-digit area code contain a 9, 8, and 7")],
 127: [("mul", "$50\\%$ of the students were eliminated after the first round. Only $\\frac{1}{3}$ of the remaining students were still in the contest after the second round. If 24 students were still in the contest")],
 137: [("addf", "\\sqrt{a}+\\sqrt{b}=x")],
 144: [("addf", "class of 40 students"), ("addf", "12 said they did not like either"), ("addf", "18 said they liked apple pie, 15 said they liked chocolate cake")],
 145: [("fr", "$60\\%$ of the students selected soda while $20\\%$ selected milk. If 72 students selected soda")],
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
    texts = [dia] + [permuted_view(dia, 99999 + 100*ti + k) for k in range(1, 5)]
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
                           m=m, lane=row["lane"], book=3, tranche=19,
                           gate="5view-vote+answer-key", generation="41",
                           op_spans=spans))
with open(".cache/book3.jsonl", "a") as f:
    for b in banked:
        f.write(json.dumps(b) + "\n")
n_total = sum(1 for _ in open(".cache/book3.jsonl"))
print(f"[t8] banked {len(banked)}/8; book3 total {n_total}; "
      f"op-spans banked {sum(len(b['op_spans']) for b in banked)}", flush=True)
