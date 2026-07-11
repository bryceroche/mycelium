"""harvest_seed_gate3.py — SEED ROUND THREE (2026-07-11): template-exact
voice, in-range values, generation-indexed gold (gen=5). Midpoint seed
exercises FDIV on real prose — tranche 2's relation meeting the wild."""
import json, sys, os
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np
from collections import Counter
os.environ.setdefault("ALG2", "1"); os.environ.setdefault("ALG_FTYPES", "6")
from phase1_algebra_head import T_ALG, build_params, forward, decode, sent_indices, TOKENIZER_JSON
from beacon_closing_arm import recompute_states
from tta_views import permuted_view
from tta_alg2_dials import solve2
from tokenizers import Tokenizer
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load

SEEDS = [
 dict(src="rect garden: perim 60, len twice width, area", answer=200, m=250,
      dialect="Consider the numbers a, b, c, d, e, f, g, h. It is known "
              "that c is 30. The value of b is 2. The product of a and b is "
              "d. The sum of a and d is c. The product of d and a is e. "
              "f is 5. g is 4. The sum of f and g is h. What is e?"),
 dict(src="midpoint of (8,5),(2,-1): coord sum", answer=7, m=99,
      dialect="Consider the numbers a, b, c, d, e, f, g, h, i, j, k. "
              "a is 8. b is 2. The sum of a and b is c. When c is divided "
              "by 2, the quotient is d and the remainder is e. f is 5. "
              "The value of g is 1. f exceeds g by h. When h is divided by "
              "2, the quotient is i and the remainder is j. The sum of d "
              "and i is k. What is k?"),
]
tok = Tokenizer.from_file(TOKENIZER_JSON)
p = build_params(0)
sd = safe_load(".cache/phase1_bilingual_head.safetensors")
for k in p:
    p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()

def parse_batch(texts):
    n = len(texts)
    N = ((n + 7) // 8) * 8
    ids = np.zeros((N, T_ALG), np.int32); msk = np.zeros((N, T_ALG), np.float32)
    snt = np.zeros((N, T_ALG), np.int32)
    for i, t in enumerate(texts):
        e = tok.encode(t)
        ids[i, :len(e.ids)] = e.ids; msk[i, :len(e.ids)] = 1.0
        snt[i] = sent_indices(t, list(e.offsets), msk[i])
    st = recompute_states(ids)
    res = []
    for s0 in range(0, N, 8):
        out = forward(p, Tensor(st[s0:s0+8].astype(np.float32), dtype=dtypes.float),
                      Tensor(msk[s0:s0+8].astype(np.float32), dtype=dtypes.float),
                      Tensor(snt[s0:s0+8].astype(np.int32), dtype=dtypes.int))
        keys = ("pres","ftype","op","islit","dig","args","res","query") + \
            (("sel",) if "sel" in out else ())
        o = {k: out[k].realize().numpy() for k in keys}
        for bi in range(8):
            if s0 + bi < n:
                res.append(decode({k: o[k][bi] for k in o}))
    return res

banked = list(json.loads(l) for l in open(".cache/harvest_seed.jsonl"))
for si, s in enumerate(SEEDS):
    texts = [s["dialect"]] + [permuted_view(s["dialect"], 300*si+k) for k in range(1, 5)]
    parses = parse_batch(texts)
    votes = []
    for facs, q in parses:
        a = solve2(facs, q, {"n_vars": 12, "m": s["m"]})
        if a is not None:
            votes.append(a)
    top, cnt = (Counter(votes).most_common(1)[0] if votes else (None, 0))
    ok = cnt >= 3 and top == s["answer"]
    print(f"  [{si}] gold {s['answer']:4d} | votes {votes} | "
          f"{'BANKED' if ok else 'REJECTED'} — {s['src']}")
    if ok:
        banked.append(dict(src=s["src"], dialect=s["dialect"],
                           answer=s["answer"], gate="lattice-vote+answer-key",
                           generation=5))
with open(".cache/harvest_seed.jsonl", "w") as f:
    for b in banked:
        f.write(json.dumps(b) + "\n")
print(f"[seed-3] substrate now n={len(banked)}")
