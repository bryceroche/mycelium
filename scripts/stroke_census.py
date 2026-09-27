"""THE STROKE CENSUS (2026-09-27, the word; the painter's thread): are wrong wild graphs too LONG (extra strokes) or
too SHORT (missing strokes)? Zero GPU. The top-1 masked decode per row (from the beam oracle's records, flips == [])
vs the row's GOLD factor list (wild_admitted_holdout.jsonl), split by the row's masked chain status (correct / wrong /
refused). Counts: total factors, givens, relations; the signed difference decoded - gold; and the share of rows with
extra / missing strokes per status. A READ, never a loss (the Goodhart fence; the consume-once credit's grave)."""
import json, pickle, sys, numpy as np
from collections import Counter, defaultdict
TAG = sys.argv[1] if len(sys.argv) > 1 else "PMS8_241"
rows = [json.loads(l) for l in open(".cache/wild_admitted_holdout.jsonl")]
recs = [r for r in pickle.load(open(f".cache/beam_oracle_{TAG}.pkl", "rb")) if not r["flips"]]
assert len(recs) == len(rows), (len(recs), len(rows))
by = defaultdict(list)
for r in recs:
    g = rows[r["row"]]["factors"]; p = r["parse"]
    gg = sum(1 for f in g if f["ftype"] == "given"); gr = len(g) - gg
    pg = sum(1 for f in p if f["ftype"] == "given"); pr = len(p) - pg
    by[r["top1_status"]].append((len(p) - len(g), pg - gg, pr - gr, len(p), len(g)))
out = []
P = lambda s: (print(s), out.append(s))
P(f"THE STROKE CENSUS — {TAG}, wild (311 rows), top-1 masked decode vs gold factors")
P(f"{'status':9s} {'n':>4s} {'dec':>6s} {'gold':>6s} {'d_total':>8s} {'d_given':>8s} {'d_rel':>7s} {'longer':>7s} {'exact':>6s} {'shorter':>8s}")
for st in ("correct", "wrong", "refused"):
    a = np.array(by[st]); n = len(a)
    if n == 0: continue
    P(f"{st:9s} {n:4d} {a[:,3].mean():6.2f} {a[:,4].mean():6.2f} {a[:,0].mean():+8.2f} {a[:,1].mean():+8.2f} {a[:,2].mean():+7.2f} "
      f"{(a[:,0]>0).mean():7.2f} {(a[:,0]==0).mean():6.2f} {(a[:,0]<0).mean():8.2f}")
a = np.array(by["wrong"]); P(f"wrong rows, d_total histogram: {dict(sorted(Counter(a[:,0].tolist()).items()))}")
a = np.array(by["correct"]); P(f"correct rows, d_total histogram: {dict(sorted(Counter(a[:,0].tolist()).items()))}")
a = np.array(by["refused"]); P(f"refused rows, d_total histogram: {dict(sorted(Counter(a[:,0].tolist()).items()))}")
open(f".cache/stroke_census_{TAG}.txt", "w").write("\n".join(out) + "\n")
