"""build_diet_v2.py — THE REBALANCED DIET (2026-09-17, word given). Composes a training split from named
sources at DECLARED shares, printing the dose law for each (share-of-mix AND reps-per-unique-story per
epoch). Sources: mint (form_mix12's non-gsm8k rows, SUBSAMPLED to its share), pen (the diet's 1,225
chain-lane stories, reps capped), silver sets (stories + their resampled copies; dup chosen to land the
share without exceeding the reps cap). Every silver row must be positional (gen.canonical) — asserted.
usage: build_diet_v2.py out.jsonl --mint 0.60 --pen 0.10 --reps-cap 8 --silver name=stories.jsonl[:copies.jsonl] ..."""
import json, sys, random, argparse, collections
ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--mint", type=float, default=0.60); ap.add_argument("--pen", type=float, default=0.10)
ap.add_argument("--reps-cap", type=float, default=8.0); ap.add_argument("--silver", action="append", default=[]); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--total", type=int, default=0); ap.add_argument("--wild-frac", type=float, default=0.0, help="fraction of the sampled MINT rows passed through wild_mint (worded numerals / distractor prefix)")
a = ap.parse_args(); rng = random.Random(a.seed)
mix = [json.loads(l) for l in open(".cache/form_mix12.jsonl")]
G = lambda r: r.get("gen") if isinstance(r.get("gen"), dict) else {}
pen = [r for r in mix if G(r).get("src") == "gsm8k"]; mint = [r for r in mix if G(r).get("src") != "gsm8k"]
pen_u = collections.defaultdict(list)
for r in pen: pen_u[r["text"]].append(r)
silver = {}
for spec in a.silver:
    name, paths = spec.split("=", 1); parts = paths.split(":"); S = [json.loads(l) for l in open(parts[0])]; C = [json.loads(l) for l in open(parts[1])] if len(parts) > 1 else []
    for r in S + C: assert G(r).get("canonical") == "positional", f"{name}: a non-positional row"
    silver[name] = (S, C)
total = a.total or len(mix)
n_mint = int(total * a.mint); n_pen = int(total * a.pen); n_silver_all = total - n_mint - n_pen
n_sources = len(silver); per_src = n_silver_all / max(n_sources, 1)
out = []; rng.shuffle(mint); mint_rows = mint[:n_mint]; n_wild = 0
if a.wild_frac > 0:
    sys.path.insert(0, "scripts"); from wild_mint import wild
    mint_rows = [wild(r, rng, do_words=rng.random() < 0.7, do_distractor=rng.random() < 0.7) if rng.random() < a.wild_frac else r for r in mint_rows]; n_wild = sum(1 for r in mint_rows if (G(r) or {}).get("wild"))
out += mint_rows; report = [f"mint {n_mint} rows ({n_mint/total:.1%}; {n_mint/len(mint):.0%} of the pool's {len(mint)}; wilded {n_wild} = {n_wild/max(n_mint,1):.0%})"]
reps_pen = min(a.reps_cap, n_pen / len(pen_u)); pen_rows = []
for t, rs in pen_u.items(): pen_rows += (rs * 20)[:max(1, round(reps_pen))]
out += pen_rows; report.append(f"pen {len(pen_rows)} rows ({len(pen_rows)/total:.1%}; {len(pen_u)} unique stories x {reps_pen:.1f} reps)")
for name, (S, C) in silver.items():
    per_story = (len(S) + len(C)) / len(S); dup = max(1, min(int(a.reps_cap // per_story), int(per_src // (len(S) + len(C))) or 1))
    rows = (S + C) * dup; out += rows; report.append(f"silver:{name} {len(rows)} rows ({len(rows)/total:.1%}; {len(S)} stories + {len(C)} copies x dup {dup} = {per_story*dup:.1f} reps/story)")
rng.shuffle(out)
with open(a.out, "w") as f:
    for r in out: f.write(json.dumps(r) + "\n")
print(f"[diet-v2] {len(out)} rows -> {a.out}\n  " + "\n  ".join(report))
