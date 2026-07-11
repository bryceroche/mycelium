"""harvest_seed_gate2.py — SEED ROUND TWO (2026-07-11): in-shape dialect +
THE LATTICE AS GATE (5-view vote; a single unstable pointer gets outvoted).
Bank iff the vote's answer == the official answer key."""
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
 dict(src="sum 45 diff 3, lesser", answer=21, m=99,
      dialect="Consider the numbers a, b, c, d, e, f, g, h, i. The value of "
              "f is 7. g is 2. f times g equals h. a plus b equals c. It is "
              "known that c is 45. a exceeds b by d. The value of d is 3. "
              "Of a and b, the smaller one is e. h exceeds f by i. What is e?"),
 dict(src="rect garden area", answer=200, m=250,
      dialect="Consider the numbers a, b, c, d, e, f, g, h. The value of c "
              "is 30. b is 2. a times b equals d. a plus d equals c. d times "
              "a equals e. f is 5. g is 4. f plus g equals h. What is e?"),
 dict(src="f(4)=x^2-x", answer=12, m=99,
      dialect="Consider the numbers a, b, c, d, e, f. a is 4. The value of "
              "b is 16. b exceeds a by c. d is 3. e is 6. d plus e equals f. "
              "What is c?"),
 dict(src="Emily 30^2-29^2 helper", answer=59, m=999,
      dialect="Consider the numbers a, b, c, d, e, f. a is 900. b is 841. "
              "a exceeds b by c. d is 2. e is 8. d times e equals f. "
              "What is c?"),
 dict(src="diff of squares 17^2-15^2", answer=64, m=999,
      dialect="Consider the numbers a, b, c, d, e, f. a is 289. b is 225. "
              "a exceeds b by c. d is 5. e is 4. d plus e equals f. "
              "What is c?"),
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

banked = []
for si, s in enumerate(SEEDS):
    texts = [s["dialect"]] + [permuted_view(s["dialect"], 100*si+k) for k in range(1, 5)]
    parses = parse_batch(texts)
    votes = []
    for facs, q in parses:
        a = solve2(facs, q, {"n_vars": 12, "m": s["m"]})
        if a is not None:
            votes.append(a)
    top, cnt = (Counter(votes).most_common(1)[0] if votes else (None, 0))
    ok = cnt >= 3 and top == s["answer"]
    print(f"  [{si}] gold {s['answer']:4d} | votes {votes} | "
          f"{'BANKED (vote '+str(cnt)+'/5)' if ok else 'REJECTED'} — {s['src']}")
    if ok:
        banked.append(dict(src=s["src"], dialect=s["dialect"],
                           answer=s["answer"], gate="lattice-vote+answer-key"))
with open(".cache/harvest_seed.jsonl", "w") as f:
    for b in banked:
        f.write(json.dumps(b) + "\n")
print(f"[seed-2] {len(banked)}/{len(SEEDS)} banked through the lattice gate")
