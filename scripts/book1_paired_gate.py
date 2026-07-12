"""book1_paired_gate.py — BOOK 1 (2026-07-11): the library's first volume.

n=18 stratified per the binding sampling law (registered in
docs/phase1_skeleton_spec.md before this ran). Each entry is a PAIR:
raw MATH-train prose + a hand-written faithful in-grammar dialect. Both
run through the gen-6 lattice gate (5 views, vote>=3, answer key
disposes). THE PAIR'S FATE ATTRIBUTES the census refusal:
  raw refuses + dialect banks  -> STYLE CASUALTY (books-recoverable)
  raw near    + dialect banks  -> FRICTION RECOVERED
  dialect refuses              -> ORGAN PATIENT type B
  residual=True (no faithful dialect exists) -> ORGAN PATIENT type A
Banked dialects enter the substrate (.cache/book1.jsonl) as
generation-indexed teacher demonstrations.

Lexical explicitation allowed (literal facts about KNOWN quantities);
supplying an unknown's value or a rewritten equation is forbidden —
except in residual entries, which are flagged and never counted as
style casualties.
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

BOOK = [
 # ---- TIER N (near-miss: parse carried, vote unstable) ----
 dict(idx=71, tier="N",
      dialect="Consider the numbers a, b, c, d. a is 6. When a is divided "
              "by 2, the quotient is b and the remainder is c. a plus b "
              "equals d. What is d?"),
 dict(idx=78, tier="N",
      dialect="Consider the numbers a, b, c, d. a is 12. When a is divided "
              "by 3, the quotient is b and the remainder is c. a plus b "
              "equals d. What is d?"),
 dict(idx=89, tier="N",
      dialect="Consider the numbers a, b, c, d, e, f, g. a is 2. b is 6. "
              "c is 10. c times a equals d. e is 4. b plus d equals f. "
              "f exceeds e by g. What is g?"),
 dict(idx=72, tier="N",
      dialect="Consider the numbers a, b, c, d, e, f, g. c is 2. c times a "
              "equals d. d plus b equals e. It is known that e is 210. "
              "c times b equals f. f plus a equals g. It is known that g "
              "is 240. What is b?"),
 dict(idx=46, tier="N",
      dialect="Consider the numbers a, b, c, d, e, f, g, h, i, j. a plus b "
              "equals c. c plus b equals d. d plus b equals e. a plus c "
              "equals f. f plus d equals g. g plus e equals h. It is known "
              "that h is 10. e plus b equals i. It is known that i is 5. "
              "i plus b equals j. What is j?"),
 dict(idx=7, tier="N",
      dialect="Consider the numbers a, b, c, d, e, f, g. a is 6. When a is "
              "divided by 2, the quotient is b and the remainder is c. "
              "d is 60. b times d equals e. f is 4. f times g equals e. "
              "What is g?"),
 # ---- TIER S (knotted, style-suspect) ----
 dict(idx=21, tier="S",  # THE SENTINEL — seed-2's banked dialect verbatim
      dialect="Consider the numbers a, b, c, d, e, f, g, h, i. The value of "
              "f is 7. g is 2. f times g equals h. a plus b equals c. It is "
              "known that c is 45. a exceeds b by d. The value of d is 3. "
              "Of a and b, the smaller one is e. h exceeds f by i. "
              "What is e?"),
 dict(idx=99, tier="S",
      dialect="Consider the numbers a, b, c, d, e. a exceeds b by c. It is "
              "known that c is 12. a times b equals d. It is known that d "
              "is 45. a plus b equals e. What is e?"),
 dict(idx=16, tier="S",
      dialect="Consider the numbers a, b, c. a is 4. b is 16. b exceeds a "
              "by c. What is c?"),
 dict(idx=57, tier="S",
      dialect="Consider the numbers a, b, c, d, e. a is 1. b is 3. c is 14. "
              "c times b equals d. a plus d equals e. What is e?"),
 dict(idx=28, tier="S",
      dialect="Consider the numbers a, b, c, d, e, f. a is 1. b is 10. "
              "b exceeds a by c. d is 20. d times c equals e. a plus e "
              "equals f. What is f?"),
 dict(idx=56, tier="S",
      dialect="Consider the numbers a, b, c, d, e, f, g, h, i, j, k, l, m, "
              "n, o, p, q, r, s. a is 12. b is 11. c is 10. d is 9. e is 8. "
              "f is 7. g is 6. h is 5. i is 4. j is 3. a plus b equals k. "
              "k plus c equals l. l plus d equals m. m plus e equals n. "
              "n plus f equals o. o plus g equals p. p plus h equals q. "
              "q plus i equals r. r plus j equals s. What is s?"),
 dict(idx=45, tier="S",
      dialect="Consider the numbers a, b, c, d, e. a is 120. When a is "
              "divided by 5, the quotient is b and the remainder is c. "
              "d is 7. d times b equals e. What is e?"),
 # ---- TIER O (knotted, organ-suspect) ----
 dict(idx=54, tier="O",  # integer factoring == the tranche-2 Vieta shape
      dialect="Consider the numbers a, b, c, d, e, f, g, h, i. a plus b "
              "equals c. It is known that c is 16. a times b equals d. It "
              "is known that d is 60. Of a and b, the smaller one is f. "
              "c exceeds f by e. g is 3. g times f equals h. h exceeds e "
              "by i. What is i?"),
 dict(idx=90, tier="O", residual=True,  # completing the square: TYPE A
      dialect="Consider the numbers a, b, c. a is 16. b is 1. a plus b "
              "equals c. What is c?"),
 dict(idx=51, tier="O",  # composition unwound to forward relations
      dialect="Consider the numbers a, b, c, d, e, f, g, h, i. a is 3. "
              "b is 4. a times b equals c. d is 2. d times e equals f. "
              "g is 7. f exceeds g by h. It is known that h is 7. "
              "c exceeds i by e. What is i?"),
 dict(idx=37, tier="O",  # sign-rewrite reduced to positive-form relations
      dialect="Consider the numbers a, b, c, d, e. a is 49. b is 28. "
              "a exceeds b by c. d is 7. d times e equals c. What is e?"),
 dict(idx=85, tier="O",  # repeated-arg mul probe (untrained shape)
      dialect="Consider the numbers a, b, c. a times a equals b. a plus b "
              "equals c. It is known that c is 156. What is a?"),
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

banked, table = [], []
for bi, e in enumerate(BOOK):
    row = JOIN[e["idx"]]
    gold = row["answer"]
    raw_ok, raw_votes = gate(row["problem"], gold, 700 * e["idx"])  # census seeds
    dia_ok, dia_votes = gate(e["dialect"], gold, 9000 + 100 * bi)
    if e.get("residual"):
        verdict = "ORGAN-A (annotation-impossible; residual only)"
    elif dia_ok:
        verdict = ("STYLE CASUALTY" if row["census"] == "knotted"
                   else "FRICTION RECOVERED")
    else:
        verdict = "ORGAN-B (faithful dialect refused)"
    table.append(dict(idx=e["idx"], tier=e["tier"], gold=gold,
                      census=row["census"], raw_ok=raw_ok, dia_ok=dia_ok,
                      verdict=verdict, raw_votes=raw_votes, dia_votes=dia_votes))
    print(f"  [{e['idx']:2d}] {e['tier']} gold {gold:>4} | raw {row['census']:7s} "
          f"votes {raw_votes} | dialect votes {dia_votes} -> "
          f"{'BANKS' if dia_ok else 'refuses'} | {verdict}")
    if dia_ok:
        banked.append(dict(idx=e["idx"], tier=e["tier"], answer=gold,
                           raw=row["problem"], dialect=e["dialect"],
                           residual=bool(e.get("residual")), verdict=verdict,
                           gate="lattice-vote+answer-key", generation=6))

print(f"\n=== BOOK 1: THE ATTRIBUTION TABLE (n={len(BOOK)}) ===")
for tier in "NSO":
    rows = [t for t in table if t["tier"] == tier]
    nb = sum(t["dia_ok"] for t in rows)
    print(f"  TIER {tier}: {nb}/{len(rows)} dialects bank | " +
          ", ".join(f"[{t['idx']}]{'B' if t['dia_ok'] else 'r'}" for t in rows))
v = Counter(t["verdict"].split(" (")[0] for t in table)
print(f"  VERDICTS: {dict(v)}")
subst = [b for b in banked if not b["residual"]]
print(f"  SUBSTRATE GROWTH: +{len(subst)} faithful entries "
      f"(+{len(banked)-len(subst)} residual, flagged)")
with open(".cache/book1.jsonl", "w") as f:
    for b in banked:
        f.write(json.dumps(b) + "\n")
json.dump(table, open(".cache/book1_attribution.json", "w"))
print(f"  [saved] .cache/book1.jsonl + .cache/book1_attribution.json")
