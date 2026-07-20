"""build_fire_corpora.py — THE FIRE'S CORPORA (2026-07-20). Four arms
from the charter, doses declared BOTH-NUMBERS per the dose law.

Base = mixed13 (38,220 rows — gen-14's register). Book dose = book-4
prime pages x10 reps (the proven rep count; share lands at ~1.5% on this
base — the gift recipe's reps preserved, share halved by the larger
base, declared not inflated). Mint dose = 2,000 floor-paired crowns x1.

  A  (prime control):     base + book4-prime x10
  B  (macro-only):        base + book4-noncrown-prime x10
                               + book4-macro x10 + mint-MACRO x1
  C1 (paired, SPREAD):    base + book4-prime x10 + mint BOTH floors x1
  C2 (paired, CONCENTRATED): phase1 = A's corpus (12k steps);
      phase2 = mint both floors + base sample sized so C2's total
      dose-row visits MATCH C1's (matched-everything, the #21(c) design).
"""
import json, random

random.seed(1500)
base = [json.loads(l) for l in open(".cache/algebra_mixed13_train.jsonl")]
b4 = [json.loads(l) for l in open(".cache/book4_prose_pairs.jsonl")]
mint = [json.loads(l) for l in open(".cache/macro_mint_pairs.jsonl")]


def row(text, factors, q, dec=1):
    return {"text": text, "factors": factors, "mentions": {}, "n_vars": 24,
            "query_var": q, "decisions": dec, "m": 300,
            "solution": [0] * 24}


def dialect_rows(pop, floor):
    out = []
    for r in pop:
        if r["gen"]["floor"] != floor:
            continue
        out.append(row(r["gen"]["dialect"], r["factors"], r["query_var"]))
    return out


b4_prime = dialect_rows(b4, "prime")                       # 60 pages
crown_src = {r["gen"]["src_idx"] for r in b4 if r["gen"]["floor"] == "macro"}
b4_prime_noncrown = [row(r["gen"]["dialect"], r["factors"], r["query_var"])
                     for r in b4 if r["gen"]["floor"] == "prime"
                     and r["gen"]["src_idx"] not in crown_src]
b4_macro = dialect_rows(b4, "macro")                       # 9 crowns
mint_macro = [row(r["macro"]["text"], r["macro"]["factors"], r["query_var"])
              for r in mint]
mint_prime = [row(r["prime"]["text"], r["prime"]["factors"],
                  r["prime"].get("query_var", r["query_var"]))
              for r in mint]


def emit(path, rows_):
    random.shuffle(rows_)
    with open(path, "w") as f:
        for r in rows_:
            f.write(json.dumps(r) + "\n")
    print(f"  {path}: {len(rows_)} rows")


A = base + b4_prime * 10
emit(".cache/fire_armA.jsonl", list(A))

B = base + b4_prime_noncrown * 10 + b4_macro * 10 + mint_macro
emit(".cache/fire_armB.jsonl", list(B))

C1 = base + b4_prime * 10 + mint_macro + mint_prime
emit(".cache/fire_armC1.jsonl", list(C1))

# C2 phase-2: match C1's mint-row visit count.
# C1 mint visits ~= STEPS*BATCH * (4000/len(C1)); C2p2 visits = 4000/len(P2)
# * P2STEPS*BATCH. With STEPS=16000, P2STEPS=4000: share2 = 4*share1.
share1 = 4000 / len(C1)
share2 = min(4 * share1, 0.5)
n_base2 = int(4000 / share2) - 4000
P2 = mint_macro + mint_prime + random.sample(base, n_base2)
emit(".cache/fire_armC2_phase2.jsonl", list(P2))
print(f"[corpora] doses declared: book4-prime 60 uniques x10 "
      f"({600/len(A):.2%} of A); mint 2,000 pairs x1 ({4000/len(C1):.2%} of C1); "
      f"C2 phase2 mint share {4000/len(P2):.2%} over 4k steps (visit-matched to C1)")
