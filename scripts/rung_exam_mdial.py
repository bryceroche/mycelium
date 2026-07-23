"""rung_exam_mdial.py — the rung test's empirical leg (2026-07-22).

Three value-range certificates re-annotated as desk pages with the
solver domain (m) raised per-page. GIVENS stay in the digit reader's
range (<=999, small where possible); INTERMEDIATES exceed 300 freely —
derived values never touch the digit path. Bars pinned in the ledger
before this ran: pass = vote >=3/5 at the key; thesis holds at >=2/3.
"""
import json, sys, os
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np
from collections import Counter
os.environ.setdefault("ALG2", "1"); os.environ.setdefault("ALG_FTYPES", "8")
os.environ.setdefault("ALG_DUP", "1")
from phase1_algebra_head import T_ALG, build_params, forward, decode, sent_indices, TOKENIZER_JSON
from beacon_closing_arm import recompute_states
from tta_views import permuted_view
from tta_alg2_dials import solve2
from tokenizers import Tokenizer
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load

D = "Consider the numbers "

# (name, m, dialect, official_answer)
EXAM = [
 # sqrt8 * sqrt50 = sqrt400 = 20; intermediate 400 > 300, givens 8/50 small
 ("sqrt-product", 500,
  D+"a, b, c, d. a is 8. b is 50. a times b equals c. d times d equals c. What is d?", 20),
 # x*24=7 star-op: 49*24=1176, +24=1200, /48 -> x=25; intermediates 1176/1200 >> 300
 ("star-op", 1500,
  D+"a, b, c, d, e, f. a is 49. b is 24. a times b equals c. c plus b equals d. e is 48. e times f equals d. What is f?", 25),
 # (235^2-221^2)/14 = (235+221)(235-221)/14 = 456*14/14 = 456; mag-3 GIVENS probe
 ("diff-squares", 7000,
  D+"a, b, c, d, e, f, g. a is 235. b is 221. a plus b equals c. a exceeds d by b. c times d equals e. f is 14. f times g equals e. What is g?", 456),
]

tok = Tokenizer.from_file(TOKENIZER_JSON)
p = build_params(0)
CKPT = os.environ.get("GATE_CKPT", ".cache/crown_reader_v4.safetensors")
sd = safe_load(CKPT)
assert set(sd.keys()) == set(p.keys())
for k in p:
    p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
print(f"[mdial] gate = {CKPT} (FTYPES=8) | pages {len(EXAM)}")


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
        keys = [k for k in ("pres", "ftype", "op", "islit", "dig", "dig2", "args",
                            "res", "query", "sel", "dup", "y") if k in out]
        o = {k: out[k].realize().numpy() for k in keys}
        for bi in range(8):
            if s0 + bi < n:
                res.append(decode({k: o[k][bi] for k in o}))
    return res


passes = 0
results = []
for name, m, dia, ans in EXAM:
    texts = [dia] + [permuted_view(dia, 620000 + hash(name) % 1000 + k) for k in range(1, 5)]
    votes = []
    for facs, q in parse_batch(texts):
        a = solve2(facs, q, {"n_vars": 24, "m": m})
        if a is not None:
            votes.append(a)
    top, cnt = (Counter(votes).most_common(1)[0] if votes else (None, 0))
    ok = cnt >= 3 and top == ans
    passes += ok
    results.append(dict(name=name, m=m, votes=votes, answer=ans, ok=bool(ok)))
    print(f"  [{name}] m={m} votes {votes} vs key {ans} -> {'PASS' if ok else 'FAIL'}")

thesis = passes >= 2
print(f"\n[mdial] {passes}/3 passed | THESIS {'HOLDS' if thesis else 'FAILS'}"
      " — the m-dial decouples solver range from reader range"
      if thesis else f"\n[mdial] {passes}/3 passed | THESIS FAILS")
json.dump(dict(results=results, passes=passes, thesis=bool(thesis)),
          open(".cache/rung_exam_mdial.json", "w"), indent=1)
print("[mdial] banked -> .cache/rung_exam_mdial.json")
