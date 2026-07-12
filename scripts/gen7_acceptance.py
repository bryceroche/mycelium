"""gen7_acceptance.py — GEN-7 acceptance probes + census re-run (2026-07-11).

The eight refused book-1 v1 dialects, re-gated under the gen-7 head
(predictions pinned in the ledger BEFORE training fired), then the
100-pool census replay (predicted ~unchanged: gen-7 teaches moves in
the dialect register; the style wall waits on reading-training).
"""
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
import re

PROBES = [
 dict(idx=46, gold=6, expect="BANK (paraphrase probe)",
      dialect="Consider the numbers a, b, c, d, e, f, g, h, i, j. a plus b "
              "equals c. c plus b equals d. d plus b equals e. a plus c "
              "equals f. f plus d equals g. g plus e equals h. It is known "
              "that h is 10. e plus b equals i. It is known that i is 5. "
              "i plus b equals j. What is j?"),
 dict(idx=71, gold=9, expect="BANK (fdiv)",
      dialect="Consider the numbers a, b, c, d. a is 6. When a is divided "
              "by 2, the quotient is b and the remainder is c. a plus b "
              "equals d. What is d?"),
 dict(idx=78, gold=16, expect="BANK (fdiv)",
      dialect="Consider the numbers a, b, c, d. a is 12. When a is divided "
              "by 3, the quotient is b and the remainder is c. a plus b "
              "equals d. What is d?"),
 dict(idx=7, gold=45, expect="BANK (fdiv)",
      dialect="Consider the numbers a, b, c, d, e, f, g. a is 6. When a is "
              "divided by 2, the quotient is b and the remainder is c. "
              "d is 60. b times d equals e. f is 4. f times g equals e. "
              "What is g?"),
 dict(idx=45, gold=168, expect="BANK (fdiv)",
      dialect="Consider the numbers a, b, c, d, e. a is 120. When a is "
              "divided by 5, the quotient is b and the remainder is c. "
              "d is 7. d times b equals e. What is e?"),
 dict(idx=85, gold=12, expect="BANK (repeated-arg mul)",
      dialect="Consider the numbers a, b, c. a times a equals b. a plus b "
              "equals c. It is known that c is 156. What is a?"),
 dict(idx=72, gold=90, expect="BANK (coupled)",
      dialect="Consider the numbers a, b, c, d, e, f, g. c is 2. c times a "
              "equals d. d plus b equals e. It is known that e is 210. "
              "c times b equals f. f plus a equals g. It is known that g "
              "is 240. What is b?"),
 dict(idx=56, gold=75, expect="BANK (ladder)",
      dialect="Consider the numbers a, b, c, d, e, f, g, h, i, j, k, l, m, "
              "n, o, p, q, r, s. a is 12. b is 11. c is 10. d is 9. e is 8. "
              "f is 7. g is 6. h is 5. i is 4. j is 3. a plus b equals k. "
              "k plus c equals l. l plus d equals m. m plus e equals n. "
              "n plus f equals o. o plus g equals p. p plus h equals q. "
              "q plus i equals r. r plus j equals s. What is s?"),
]

tok = Tokenizer.from_file(TOKENIZER_JSON)
p = build_params(0)
CKPT = os.environ.get("GATE_CKPT", ".cache/phase1_gen7_head.safetensors")
sd = safe_load(CKPT)
assert set(sd.keys()) == set(p.keys())
for k in p:
    p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
print(f"[acceptance] gate ckpt: {CKPT}")

def parse_batch(texts):
    n = len(texts)
    N = ((n + 7) // 8) * 8
    ids = np.zeros((N, T_ALG), np.int32); msk = np.zeros((N, T_ALG), np.float32)
    snt = np.zeros((N, T_ALG), np.int32)
    for i, t in enumerate(texts):
        e = tok.encode(t)
        L = min(len(e.ids), T_ALG)
        ids[i, :L] = e.ids[:L]; msk[i, :L] = 1.0
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

def gate(text, gold, seed0):
    texts = [text] + [permuted_view(text, seed0 + k) for k in range(1, 5)]
    votes = []
    for facs, q in parse_batch(texts):
        a = solve2(facs, q, {"n_vars": 24, "m": 300})
        if a is not None:
            votes.append(a)
    top, cnt = (Counter(votes).most_common(1)[0] if votes else (None, 0))
    return (cnt >= 3 and top == gold), votes

print("=== GEN-7 ACCEPTANCE PROBES (the eight refused v1 dialects) ===")
nb = 0
for pi, e in enumerate(PROBES):
    ok, votes = gate(e["dialect"], e["gold"], 30000 + 100 * pi)
    nb += ok
    print(f"  [{e['idx']:2d}] gold {e['gold']:>4} | votes {votes} -> "
          f"{'BANKS' if ok else 'refuses'} | predicted {e['expect']}")
print(f"  ACCEPTANCE: {nb}/{len(PROBES)} banked")

print("\n=== CENSUS REPLAY UNDER GEN-7 (predicted ~unchanged 65-72) ===")
h = [json.loads(l) for l in open(".cache/math_harvest_v0.jsonl")]
pool = [x for x in h if x["level"] in ("Level 1", "Level 2", "Level 3")
        and len(x["problem"]) < 300 and "asy]" not in x["problem"]
        and all(int(n) <= 300 for n in re.findall(r"\d+", x["problem"]))]
pool = pool[:100]
banked, near, knotted = 0, 0, 0
for xi, x in enumerate(pool):
    texts = [x["problem"]] + [permuted_view(x["problem"], 700*xi+k) for k in range(1, 5)]
    votes = []
    for facs, q in parse_batch(texts):
        a = solve2(facs, q, {"n_vars": 24, "m": 300})
        if a is not None:
            votes.append(a)
    top, cnt = (Counter(votes).most_common(1)[0] if votes else (None, 0))
    if cnt >= 3 and top == x["answer"]:
        banked += 1
    elif len(votes) >= 2:
        near += 1
    else:
        knotted += 1
print(f"  CENSUS gen-7: banked {banked} / near {near} / knotted {knotted}"
      f"  (gen-6 was 2/26/72)")
