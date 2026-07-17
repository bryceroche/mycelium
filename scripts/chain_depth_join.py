"""chain_depth_join.py — GUTS #29+#30, READ (a): THE CHAIN-DEPTH JOIN
(2026-07-17). Does failure correlate with DERIVATION DEPTH (the longest
factor-dependency chain, computed fresh from gold DAGs — NOT the bands
column, whose jurisdiction is solver decisions) at controlled size?

PINNED BEFORE PRINT: within factor-count strata, rank AUC(depth ->
not-certified) pooled across strata: >=0.60 = depth wall REAL; 0.50-0.60
= weak; <0.50 = inverted. Kill-only. THE CROWN-RECOVERY RIDER registers
(fires at crown mass, per the crown-count pin): if the wall is real,
crowned items with deep prime chains should recover disproportionately —
the tower as depth remedy, the recursion's second dividend path.
Constitution: the solver iterates, the head binds (Brick-P, twice) —
this measures the BINDING burden's depth ceiling, not deduction's.
"""
import json, sys
import numpy as np
from collections import Counter, defaultdict

sys.path.insert(0, "scripts")
from schema_miner import producer_of, inputs_of

rows = [json.loads(l) for l in open(".cache/algebra_nl_bigtest.jsonl")]
gold = [r["solution"][r["query_var"]] for r in rows]
gate = json.load(open(".cache/lattice_gate.json"))["bigtest"]
armb = json.load(open(".cache/lattice_armB.json"))["bigtest"]
c2x = json.load(open(".cache/lattice_cap2x.json"))["bigtest"]


def maj(v):
    vs = [x for x in v if x is not None]
    if not vs:
        return None, 0
    return Counter(vs).most_common(1)[0]


def depth(factors):
    """Longest chain in the factor-dependency DAG (factors on the path)."""
    prod = {}
    for i, f in enumerate(factors):
        pv = producer_of(f)
        if pv is not None and pv not in prod:
            prod[pv] = i
    memo = {}

    def d(i):
        if i in memo:
            return memo[i]
        memo[i] = 1 + max([d(prod[v]) for v in inputs_of(factors[i])
                           if v in prod] or [0])
        return memo[i]

    return max(d(i) for i in range(len(factors)))


def rank_auc(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if not len(a) or not len(b):
        return float("nan")
    allv = np.concatenate([a, b])
    order = np.argsort(allv, kind="mergesort")
    ranks = np.empty(len(order)); ranks[order] = np.arange(1, len(order) + 1)
    for v in np.unique(allv):
        m = allv == v
        ranks[m] = ranks[m].mean()
    u = ranks[:len(a)].sum() - len(a) * (len(a) + 1) / 2
    return u / (len(a) * len(b))


chan, dep, nfac = [], [], []
for i, r in enumerate(rows):
    gt, gc = maj(gate[i]); at, _ = maj(armb[i]); ct, _ = maj(c2x[i])
    if gc == 5 and at == gt and ct == gt:
        ch = "certify"
    elif gc >= 3:
        ch = "answer"
    else:
        ch = "abstain"
    chan.append(ch)
    dep.append(depth(r["factors"]))
    nfac.append(len(r["factors"]))
dep, nfac = np.array(dep), np.array(nfac)
cert = np.array([c == "certify" for c in chan])
print(f"[depth-join] bigtest n=1500 | depth range {dep.min()}-{dep.max()} | "
      f"factor-count range {nfac.min()}-{nfac.max()}")

print("\n=== WITHIN-STRATUM READ (factor-count controlled) ===")
strata = [(4, 6), (7, 8), (9, 10), (11, 24)]
aucs, ws = [], []
for lo, hi in strata:
    m = (nfac >= lo) & (nfac <= hi)
    if m.sum() < 30:
        continue
    d_bad = dep[m & ~cert]; d_good = dep[m & cert]
    auc = rank_auc(d_bad, d_good)          # P(bad deeper than good)
    aucs.append(auc); ws.append(m.sum())
    print(f"  factors {lo:2d}-{hi:2d}: n={m.sum():4d}  "
          f"certified {cert[m].mean():.1%}  depth mean cert {d_good.mean():.2f} "
          f"vs non-cert {d_bad.mean():.2f}  AUC {auc:.3f}")
pooled = float(np.average(aucs, weights=ws))
verdict = ("DEPTH WALL REAL" if pooled >= 0.60 else
           "WEAK" if pooled >= 0.50 else "INVERTED")
print(f"\n  POOLED within-stratum AUC(depth -> not-certified): {pooled:.3f} "
      f"=> {verdict}")

# raw (uncontrolled) context + the length-frontier scouts
print(f"\n  context (uncontrolled): depth-vs-noncert AUC "
      f"{rank_auc(dep[~cert], dep[cert]):.3f}; "
      f"factor-count-vs-noncert AUC {rank_auc(nfac[~cert], nfac[cert]):.3f}")
b4 = [json.loads(l) for l in open(".cache/book4_prose_pairs.jsonl")]
bd = [depth(r["factors"]) for r in b4 if r["gen"]["floor"] == "prime"]
print(f"  book-4 banked prime pages: depth mean {np.mean(bd):.2f} "
      f"max {max(bd)} (the misses [33]/[100] sat at 12-15 VARS — scouts, "
      f"not yet joinable: misses bank no gold)")

json.dump({"pooled_auc": pooled, "verdict": verdict,
           "strata": [{"range": s, "n": int(w)} for s, w in zip(strata, ws)]},
          open(".cache/chain_depth_join.json", "w"))
print("[depth-join] banked -> .cache/chain_depth_join.json | crown-recovery "
      "rider REGISTERED (fires at crown mass)")
