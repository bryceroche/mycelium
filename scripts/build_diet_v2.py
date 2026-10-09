"""build_diet_v2.py — THE REBALANCED DIET (2026-09-17, word given). Composes a training split from named
sources at DECLARED shares, printing the dose law for each (share-of-mix AND reps-per-unique-story per
epoch). Sources: mint (form_mix12's non-gsm8k rows, SUBSAMPLED to its share), pen (the diet's 1,225
chain-lane stories, reps capped), silver sets (stories + their resampled copies; dup chosen to land the
share without exceeding the reps cap). Every silver row must be positional (gen.canonical) — asserted.

THE INVERSE DOSE (2026-10-08, THE POLARITY CENSUS's registered form (b)): --inverse-frac target biases
the mint subsample (SAME n_mint as a plain --mint run) toward rows whose relations are inverse-form
(polarity_census.classify_row's exact method — gold res names an arg, not the result) until the
SELECTED mint rows' aggregate relation-slot inverse share hits the target. Mint candidates are split
into bucket A (inv > fwd relations on that row) and bucket B (the rest); x = the bucket-A share of
n_mint solves x*shareA + (1-x)*shareB >= target (shareA/shareB = each bucket's own aggregate inverse
share), both buckets independently shuffled so the selection stays diverse, not a handful of exotic
rows repeated. Prints the dose law's two legs for the mint portion specifically: share-of-mix (the
achieved inverse relation-slot share) and reps-per-unique (rows per distinct TEXT in the selection —
mint rows are drawn without duplication from the pool, so this leg is declared, not assumed, per the
dose law's "declare BOTH" rule, even though it is trivially ~1.0 by construction).
usage: build_diet_v2.py out.jsonl --mint 0.60 --pen 0.10 --reps-cap 8 --silver name=stories.jsonl[:copies.jsonl] ...
       build_diet_v2.py out.jsonl --mint 0.35 --inverse-frac 0.50 --wild-frac 0.5 ..."""
import json, sys, random, argparse, collections
sys.path.insert(0, "scripts")
from polarity_census import classify_row
ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--mint", type=float, default=0.60); ap.add_argument("--pen", type=float, default=0.10)
ap.add_argument("--reps-cap", type=float, default=8.0); ap.add_argument("--silver", action="append", default=[]); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--total", type=int, default=0); ap.add_argument("--wild-frac", type=float, default=0.0, help="fraction of the sampled MINT rows passed through wild_mint (worded numerals / distractor prefix)")
ap.add_argument("--inverse-frac", type=float, default=None, help="THE INVERSE DOSE: target aggregate inverse-relation-slot share of the selected MINT rows (default: unbiased random subsample, the plain --mint behavior)")
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
out = []; rng.shuffle(mint); n_wild = 0
dose_report = None
if a.inverse_frac is None:
    mint_rows = mint[:n_mint]
else:
    def _inv_fwd(r):
        c = classify_row(r["factors"])
        return sum(1 for v in c if v == "fwd"), sum(1 for v in c if v == "inv")
    _stats = [_inv_fwd(r) for r in mint]
    bucketA = [r for r, (nf, ni) in zip(mint, _stats) if ni > nf]
    bucketB = [r for r, (nf, ni) in zip(mint, _stats) if ni <= nf]
    statsA = [(nf, ni) for (nf, ni) in _stats if ni > nf]
    statsB = [(nf, ni) for (nf, ni) in _stats if ni <= nf]
    fA = sum(nf + ni for nf, ni in statsA) / max(len(statsA), 1)   # avg relation-slots/row, bucket A
    fB = sum(nf + ni for nf, ni in statsB) / max(len(statsB), 1)   # bucket B
    shareA = sum(ni for _, ni in statsA) / max(sum(nf + ni for nf, ni in statsA), 1)
    shareB = sum(ni for _, ni in statsB) / max(sum(nf + ni for nf, ni in statsB), 1)
    # THE AGGREGATE SHARE IS SLOT-WEIGHTED, NOT ROW-WEIGHTED (bucket A and B
    # have different average relation-slots/row, fA != fB) -- mixing by ROW
    # fraction x does NOT land on x*shareA + (1-x)*shareB; solve the SLOT-
    # weighted achieved share = target instead: x*fA*(shareA-target) =
    # (1-x)*fB*(target-shareB)  =>  x = R/(1+R).
    if a.inverse_frac <= shareB:
        x = 0.0
    elif a.inverse_frac >= shareA:
        x = 1.0
    else:
        R = (fB * (a.inverse_frac - shareB)) / (fA * (shareA - a.inverse_frac))
        x = R / (1.0 + R)
    nA = min(len(bucketA), round(x * n_mint)); nB = min(len(bucketB), n_mint - nA)
    nA = min(len(bucketA), n_mint - nB)   # re-clamp if bucketB ran short
    rng.shuffle(bucketA); rng.shuffle(bucketB)
    mint_rows = bucketA[:nA] + bucketB[:nB]
    rng.shuffle(mint_rows)
    _af, _ai = 0, 0
    for r in mint_rows:
        c = classify_row(r["factors"])
        _af += sum(1 for v in c if v == "fwd"); _ai += sum(1 for v in c if v == "inv")
    _achieved = _ai / max(_af + _ai, 1)
    _texts = collections.Counter(r["text"] for r in mint_rows)
    _reps_per_unique = len(mint_rows) / max(len(_texts), 1)
    dose_report = (f"THE INVERSE DOSE: target {a.inverse_frac:.3f}, achieved relation-slot inverse "
                    f"share {_achieved:.3f} (bucketA n={len(bucketA)} share {shareA:.3f} x={x:.3f} "
                    f"taken {nA}; bucketB n={len(bucketB)} share {shareB:.3f} taken {nB}) -- THE DOSE "
                    f"LAW's two legs: share-of-mix (inverse) {_achieved:.3f}; reps-per-unique "
                    f"{_reps_per_unique:.3f} ({len(mint_rows)} rows / {len(_texts)} distinct texts)")
if a.wild_frac > 0:
    sys.path.insert(0, "scripts"); from wild_mint import wild
    from tokenizers import Tokenizer; import re as _re
    _tj = _re.search(r'^TOKENIZER_JSON\s*=\s*(.+)$', open("scripts/phase1_algebra_head.py").read(), _re.M).group(1); _tok = Tokenizer.from_file(eval(_tj)); T_ALG = 256
    def _wild_fit(r):   # THE TOKEN BUDGET: a wilded row past T_ALG tokens is refused by the precompute (TRUNCATION) — keep the row plain instead
        w = wild(r, rng, do_words=rng.random() < 0.7, do_distractor=rng.random() < 0.7)
        return w if len(_tok.encode(w["text"]).ids) <= T_ALG else r
    mint_rows = [_wild_fit(r) if rng.random() < a.wild_frac else r for r in mint_rows]; n_wild = sum(1 for r in mint_rows if (G(r) or {}).get("wild"))
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
