"""gen17c_mint.py — 17c's fresh-uniques line (2026-07-23): 6,000 NEW
unique hundreds rows (not reps — the fresh-uniques law), deduped
against the gen-17 hundreds train AND held knots, same patterns and
gates (reuses gen17_mint's defs via exec, fresh seed). Mix = gen17_mix
(already carries 1x hundreds) + the fresh line, 1 rep/unique.
"""
import json, random, sys
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
from hash_audit_iso import canon

seen = set()
for path in (".cache/gen17_hundreds.jsonl", ".cache/gen17_hundreds_held.jsonl"):
    for ln in open(path):
        r = json.loads(ln)
        dg, _ = canon({"factors": r["factors"], "n_vars": 24,
                       "query_var": r["query_var"]})
        seen.add(dg)
print(f"[g17c] preloaded {len(seen)} existing hundreds knots for dedup", flush=True)

src = open("scripts/gen17_mint.py").read()
cut = src.index("seen = set()")
ns = {}
exec(src[:cut].replace("rng = random.Random(1700)",
                       "rng = random.Random(1701)"), ns)
fresh = ns["mint_hundreds"](6000, seen)
with open(".cache/gen17c_hundreds.jsonl", "w") as f:
    for r in fresh:
        f.write(json.dumps(r) + "\n")
print(f"[g17c] fresh hundreds uniques: {len(fresh)}", flush=True)

mix = [ln.rstrip("\n") for ln in open(".cache/gen17_mix.jsonl")]
for r in fresh:
    mix.append(json.dumps(r))
random.Random(1718).shuffle(mix)
with open(".cache/gen17c_mix.jsonl", "w") as f:
    for ln in mix:
        f.write(ln + "\n")
n = len(mix)
print(f"[g17c] MIX: {n} rows | hundreds uniques total 9000 "
      f"({9000/n:.1%}, 1 rep/unique — fresh over re-epochs)", flush=True)
