"""constraint_density_meter.py — GUT #24, READ (a): THE CONSTRAINT-DENSITY
METER (2026-07-16). Zero-GPU, banked graphs only.

The meter: per-factor withhold-recoverability. Withhold one factor; if the
remainder still forces the gold answer uniquely at the query (solve2's own
uniqueness discipline), the withheld factor was REDUNDANT — the graph's
constraints overlap there. The per-graph recoverable fraction is the
'contradiction surface' the settling-loop economics depend on; the banked
prior is THIN (withhold-and-solve recovered 26% of plateau failures).

Reporting frame pinned before the run: distribution (median/quartiles) per
corpus; books (182 real-prose golds) vs bigtest sample (n=200, generated).
The macro-floor comparison is REGISTERED, not run — it waits for
floor-paired corpora (the prediction enters honestly empirical).
"""
import json, sys
import numpy as np

sys.path.insert(0, "scripts")
from tta_alg2_dials import solve2

BOOKS = [".cache/book1_prose_pairs.jsonl", ".cache/book2_prose_pairs.jsonl",
         ".cache/book3_prose_pairs.jsonl"]
BIGTEST = ".cache/algebra_nl_bigtest.jsonl"
N_BIG = 200


def meter(rows, tag):
    fracs, per_item = [], []
    for r in rows:
        facs, q, m = r["factors"], r["query_var"], r.get("m", 60)
        gold = solve2(facs, q, {"n_vars": 24, "m": m})
        if gold is None:
            continue                      # graph itself not uniquely solvable; skip, log
        rec = 0
        for i in range(len(facs)):
            sub = facs[:i] + facs[i + 1:]
            if solve2(sub, q, {"n_vars": 24, "m": m}) == gold:
                rec += 1
        fracs.append(rec / len(facs))
        per_item.append({"n_factors": len(facs), "recoverable": rec})
    f = np.array(fracs)
    print(f"  {tag}: n={len(f)} graphs (skipped {len(rows)-len(f)} non-unique)")
    print(f"    per-graph recoverable fraction: median {np.median(f):.3f}  "
          f"IQR [{np.percentile(f,25):.3f}, {np.percentile(f,75):.3f}]  mean {f.mean():.3f}")
    print(f"    graphs with ZERO redundancy: {(f==0).mean():.1%}   "
          f"with >=half redundant: {(f>=0.5).mean():.1%}")
    return {"tag": tag, "n": len(f), "median": float(np.median(f)),
            "q25": float(np.percentile(f, 25)), "q75": float(np.percentile(f, 75)),
            "mean": float(f.mean()), "zero_frac": float((f == 0).mean()),
            "half_frac": float((f >= 0.5).mean()), "items": per_item}


print("=== THE CONSTRAINT-DENSITY METER (per-factor withhold-recoverability) ===")
book_rows = [json.loads(l) for p in BOOKS for l in open(p)]
out = {"books": meter(book_rows, "BOOKS (182 real-prose golds)")}
big_rows = [json.loads(l) for l in open(BIGTEST)][:N_BIG]
out["bigtest"] = meter(big_rows, f"BIGTEST sample (n={N_BIG}, generated)")

json.dump(out, open(".cache/constraint_density_meter.json", "w"))
print("\n[meter] banked -> .cache/constraint_density_meter.json")
print("[meter] macro-floor comparison REGISTERED, awaiting floor-paired corpora")
