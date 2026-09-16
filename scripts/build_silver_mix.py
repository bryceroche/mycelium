"""build_silver_mix.py — THE SILVER DIET (2026-09-16): form_mix12 + the silver
stories + their resampled copies, at a DECLARED dose (the dose law: share-of-
mix AND reps-per-unique, both printed). Silver rows carry gen.silver and
gen.resample; the base rows are untouched. The split gets its own name
(never precompute a split anything reads).
usage: build_silver_mix.py silver_all.jsonl silver_x.jsonl out.jsonl [dup=2]"""
import json, sys
silver = [json.loads(l) for l in open(sys.argv[1])]; xs = [json.loads(l) for l in open(sys.argv[2])] if len(sys.argv) > 2 and sys.argv[2] != "-" else []
out = sys.argv[3]; dup = int(sys.argv[4]) if len(sys.argv) > 4 else 2
base = [l for l in open(".cache/form_mix12.jsonl")]
def norm(r):
    r = dict(r); r.setdefault("decisions", 0); r.setdefault("solution", []); r.setdefault("m", 10000); r.setdefault("n_vars", 0); r.setdefault("query_var", 0); r.setdefault("mentions", {})
    g = dict(r.get("gen") or {}); g.setdefault("src", "gsm8k"); r["gen"] = g; return r
add = [norm(r) for r in silver] + [norm(r) for r in xs]
with open(out, "w") as f:
    for l in base: f.write(l if l.endswith("\n") else l + "\n")
    for _ in range(dup):
        for r in add: f.write(json.dumps(r) + "\n")
n_base = len(base); n_add = len(add) * dup; uniq = len(silver)
print(f"[silver-mix] base {n_base} + silver ({len(silver)} stories + {len(xs)} resampled copies) x dup {dup} = {n_add} rows -> {out}: share-of-mix {n_add / (n_base + n_add):.2%}; reps-per-unique story = {(n_add / max(uniq, 1)):.1f} rows per story per epoch")
