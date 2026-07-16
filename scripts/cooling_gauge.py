"""cooling_gauge.py — GUT #26, READ (b): THE COOLING GAUGE, ZERO POINT
(2026-07-16). The standing bet (§6.4: agreement-based detection decays as
its population hardens) gets its thermometer: per-item vote entropy
distributions for the certified / answered / surviving-error / abstain
populations, banked per generation. Two vintages exist today — that is a
zero point and an early point, NOT a curve; verdicts accrue per promotion.

Entropy pinned: H = -sum p ln p (nats) over the 5 views' forced answers,
None counted as its own outcome. Populations pinned to the continuity
audit's exit channels. Vintage law: gen-14 = the lattice gate artifact;
ARM D = TTA-era sentence permutations (ckpt NOT recorded in the artifact —
caveat at full strength; the standing series records ckpt going forward).
ARM O is EXCLUDED: gold re-renders consult the key (oracle arm).
"""
import json
import numpy as np
from collections import Counter

rows = [json.loads(l) for l in open(".cache/algebra_nl_bigtest.jsonl")]
gold = [r["solution"][r["query_var"]] for r in rows]
n = len(gold)


def H(votes):
    c = Counter("⊥" if v is None else v for v in votes)
    p = np.array(list(c.values()), float) / len(votes)
    return float(-(p * np.log(p)).sum())


def maj(votes):
    vs = [v for v in votes if v is not None]
    if not vs:
        return None, 0
    return Counter(vs).most_common(1)[0]


def portrait(vote_rows, tag):
    pops = {"certified(5/5)": [], "answered-correct": [],
            "answered-WRONG(surviving)": [], "vote-abstain": []}
    for i in range(n):
        v = vote_rows[i]
        t, c = maj(v)
        h = H(v)
        if c == 5:
            pops["certified(5/5)"].append(h)   # unanimity tier (pre-panel)
        elif c >= 3:
            pops["answered-correct" if t == gold[i]
                 else "answered-WRONG(surviving)"].append(h)
        else:
            pops["vote-abstain"].append(h)
    print(f"\n  {tag}")
    out = {}
    for k, hs in pops.items():
        a = np.array(hs)
        if len(a) == 0:
            print(f"    {k:26s} n=0")
            out[k] = {"n": 0}
        elif len(a) < 5:
            print(f"    {k:26s} n={len(a):4d}  values {[round(x,3) for x in hs]}")
            out[k] = {"n": len(a), "values": [float(x) for x in hs]}
        else:
            print(f"    {k:26s} n={len(a):4d}  mean H {a.mean():.3f}  "
                  f"median {np.median(a):.3f}  P90 {np.percentile(a,90):.3f}")
            out[k] = {"n": len(a), "mean": float(a.mean()),
                      "median": float(np.median(a)), "p90": float(np.percentile(a, 90))}
    return out


print("=== THE COOLING GAUGE — per-item vote-entropy portraits (nats) ===")
g14 = json.load(open(".cache/lattice_gate.json"))["bigtest"]

# ARM D STRUCK from the series (2026-07-16, the provenance law catching
# Code's own decode): tta_arm_D_bigtest.npz's view_forced is BOOLEAN
# (per-view forced-correctly flags), NOT per-view answers — entropy
# portraits cannot be reconstructed from it honestly (the `agree`
# fraction under-determines the outcome distribution). The early point
# is UNAVAILABLE; the series starts at gen-14 and accrues per promotion.
out = {"gen14": portrait(g14, "GEN-14 (lattice gate artifact — the zero point)"),
       "armD": "STRUCK — artifact carries boolean flags + agreement "
               "fraction, not answer distributions"}

json.dump(out, open(".cache/cooling_gauge.json", "w"))
print("\n[gauge] banked -> .cache/cooling_gauge.json — the standing battery's"
      " temperature-band column starts here; the curve accrues per promotion")
