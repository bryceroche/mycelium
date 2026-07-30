"""build_gen22_mix.py — THE GEN-22 DIET MIX (2026-07-30, on the word).

One lever: the fdiv-on-derived diet (corpus v3, 400 solver-verified mint
rows, quotas 100 control / 150 d1 / 150 d2, axes spanned) folded into
gen-21's promoted register. DOSE, both numbers per the dose law:
400 uniques x10 reps = 4,000 rows; share 4,000/82,400 = 4.85%
(the book-dose BAU shape: x10 reps, share in the regularizing zone).

Base order PRESERVED (gen21_mix rows 0..78399 unchanged) so
gen21_ration_idx.json stays valid — the diet block appends after,
shuffled within itself. Minimal delta: one lever, the diet."""
import json, random

base = open(".cache/gen21_mix.jsonl").read().splitlines()
diet = [json.loads(l) for l in open(".cache/diet_fdiv_derived_corpus.jsonl")]
assert len(base) == 78400 and len(diet) == 400
for r in diet:
    assert "src_idx" not in r.get("gen", {}), "pen row in diet corpus"

random.seed(2200)
block = [json.dumps(r) for r in diet for _ in range(10)]
random.shuffle(block)
with open(".cache/gen22_mix.jsonl", "w") as f:
    f.write("\n".join(base + block) + "\n")
print(f"[gen22 mix] base {len(base)} (order preserved) + diet {len(diet)}x10 "
      f"= {len(base)+len(block)} rows; diet share {len(block)/(len(base)+len(block)):.2%}")
