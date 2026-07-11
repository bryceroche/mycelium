"""book1_v2_retry.py — BOOK 1, second annotation pass (2026-07-11).

ONE v2 per refused item, pre-declared, then the book closes (no gate
p-hacking: retries stay taxonomy-faithful; the ledger records attempts).

v1 finding being routed around: the FDIV register fails in hand-written
compositions (0/4 — and [78] voted a consistent WRONG 12, not a refusal).
Taxonomy clause applied: division of two KNOWN quantities is lexical
explicitation (6/2=3 is a fact about knowns) — so v2 states the quotient
as a literal and the fdiv register's weakness is priced for gen-7, not
laundered into ORGAN verdicts. [72]/[46] get surface rephrases only
(same relations). [56] (length wall) and [85] (repeated-arg grammar gap)
stand as ORGAN-B — no faithful alternative exists.
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

JOIN = {r["idx"]: r for r in json.load(open(".cache/census_mouth_join.json"))}

V2 = [
 dict(idx=71, tier="N", note="fdiv->lexical (6/2=3 known)",
      dialect="Consider the numbers a, b, c. a is 6. b is 3. a plus b "
              "equals c. What is c?"),
 dict(idx=78, tier="N", note="fdiv->lexical (12/3=4 known)",
      dialect="Consider the numbers a, b, c. a is 12. b is 4. a plus b "
              "equals c. What is c?"),
 dict(idx=7, tier="N", note="fdiv->lexical (6/2=3 known)",
      dialect="Consider the numbers a, b, c, d, e. a is 3. b is 60. a times "
              "b equals c. d is 4. d times e equals c. What is e?"),
 dict(idx=45, tier="S", note="fdiv->lexical (120/5=24 known)",
      dialect="Consider the numbers a, b, c. a is 24. b is 7. a times b "
              "equals c. What is c?"),
 dict(idx=72, tier="N", note="surface rephrase, same relations",
      dialect="Consider the numbers a, b, c, d, e, f, g. The value of c is "
              "2. The product of c and a is d. The sum of d and b is e. "
              "e is 210. The product of c and b is f. The sum of f and a "
              "is g. g is 240. What is b?"),
 dict(idx=46, tier="N", note="surface rephrase, same relations",
      dialect="Consider the numbers a, b, c, d, e, f, g, h, i, j. The sum "
              "of a and b is c. The sum of c and b is d. The sum of d and "
              "b is e. The sum of a and c is f. The sum of f and d is g. "
              "The sum of g and e is h. h is 10. The sum of e and b is i. "
              "The value of i is 5. The sum of i and b is j. What is j?"),
]

tok = Tokenizer.from_file(TOKENIZER_JSON)
p = build_params(0)
sd = safe_load(os.environ.get("GATE_CKPT", ".cache/phase1_gen6_head.safetensors"))
assert set(sd.keys()) == set(p.keys())
for k in p:
    p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()

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
            (("sel",) if "sel" in out else ())
        o = {k: out[k].realize().numpy() for k in keys}
        for bi in range(8):
            if s0 + bi < n:
                res.append(decode({k: o[k][bi] for k in o}))
    return res

banked = []
for bi, e in enumerate(V2):
    row = JOIN[e["idx"]]
    gold = row["answer"]
    texts = [e["dialect"]] + [permuted_view(e["dialect"], 20000 + 100*bi + k)
                              for k in range(1, 5)]
    votes = []
    for facs, q in parse_batch(texts):
        a = solve2(facs, q, {"n_vars": 24, "m": 300})
        if a is not None:
            votes.append(a)
    top, cnt = (Counter(votes).most_common(1)[0] if votes else (None, 0))
    ok = cnt >= 3 and top == gold
    verdict = (("STYLE CASUALTY" if row["census"] == "knotted"
                else "FRICTION RECOVERED") if ok
               else "ORGAN-B stands (v2 also refused)")
    print(f"  [{e['idx']:2d}] {e['tier']} gold {gold:>4} | v2 votes {votes} -> "
          f"{'BANKS' if ok else 'refuses'} | {verdict} | {e['note']}")
    if ok:
        banked.append(dict(idx=e["idx"], tier=e["tier"], answer=gold,
                           raw=row["problem"], dialect=e["dialect"],
                           residual=False, verdict=verdict, attempt=2,
                           note=e["note"], gate="lattice-vote+answer-key",
                           generation=6))

print(f"\n[book1-v2] {len(banked)}/{len(V2)} recovered on second annotation")
with open(".cache/book1.jsonl", "a") as f:
    for b in banked:
        f.write(json.dumps(b) + "\n")
n_total = sum(1 for _ in open(".cache/book1.jsonl"))
print(f"[book1] volume now {n_total} entries (.cache/book1.jsonl)")
