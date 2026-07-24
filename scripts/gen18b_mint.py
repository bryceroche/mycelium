
import json, random, sys
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
from hash_audit_iso import canon

seen = set()
for path in ("gen17_hundreds", "gen17_hundreds_held", "gen17c_hundreds",
             "gen18_hundreds"):
    for ln in open(f".cache/{path}.jsonl"):
        r = json.loads(ln)
        dg, _ = canon({"factors": r["factors"], "n_vars": 24,
                       "query_var": r["query_var"]})
        seen.add(dg)
print(f"[g18b] preloaded {len(seen)} hundreds knots", flush=True)

src17 = open("scripts/gen17_mint.py").read()
cut = src17.index("seen = set()")
ns = {}
exec(src17[:cut].replace("rng = random.Random(1700)",
                         "rng = random.Random(1850)"), ns)
src18 = open("scripts/gen18_mint.py").read()
cut18 = src18.index("# dedup against every prior minted line")
ns18 = {}
exec(src18[:cut18], ns18)
render_spanned = ns18["render_spanned"]

raw = ns["mint_hundreds"](2000, seen)
fresh = []
for r in raw:
    text, facs2, mentions = render_spanned(r["factors"], r["query_var"])
    fresh.append({**r, "text": text, "factors": facs2, "mentions": mentions})
with open(".cache/gen18b_hundreds.jsonl", "w") as f:
    for r in fresh:
        f.write(json.dumps(r) + "\n")
print(f"[g18b] fresh hundreds: {len(fresh)} (spans+mentions)", flush=True)

mix = [ln.rstrip("\n") for ln in open(".cache/gen18_mix.jsonl")]
for r in fresh:
    mix.append(json.dumps(r))
random.Random(1858).shuffle(mix)
with open(".cache/gen18b_mix.jsonl", "w") as f:
    for ln in mix:
        f.write(ln + "\n")
ration = [i for i, ln in enumerate(mix)
          if len(json.loads(ln).get("factors", [])) >= 14]
json.dump(ration, open(".cache/gen18b_ration_idx.json", "w"))
print(f"[g18b] MIX: {len(mix)} rows | hundreds total 3000 uniques (teaching dose) | "
      f"ration band: {len(ration)} ({len(ration)/len(mix):.1%})", flush=True)
