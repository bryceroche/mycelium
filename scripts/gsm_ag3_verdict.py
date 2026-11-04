"""AG3 part 5, THE VERDICT (ledger 2026-11-04, bars pinned): reads the per-seed view joins .cache/gsmb_agree_BT4M8_s{0,1}_BT4M8_s{0,1}_v1.json
(scripts/gsm_agree_read.py TAG TAG_v1: T0 = view 0, T1 = view 1; s0 = the view-0 perspective) and the item census, and prints:
  AG3-1  TEST ans AGREE(view 0, view 1) - BEAM-1(view 0) per seed: mean >= +0.010 and both > 0
  AG3-I  TEST shared-value rate across views (mean of the seeds' view-0 perspective) > 0.378
  AG3-C  TEST certificate precision >= 3 x BEAM-1 (view 0) for BOTH seeds; VOID if coverage < 0.02
         (certificate = the two views' BEAM-1 answers agree; precision = view-0 BEAM-1 right on those; the base is VIEW 0's BEAM-1, per the bar,
          not gsm_agree_read's mean of both views)
Beside (read only, not bars): AG3-x the same numbers on the NON-IDENTITY items only (identity views are the same text, so they agree trivially
and inflate I / C; counts from .cache/gsmb_data_v1.json).  -> .cache/gsm_ag3_verdict.txt (the AG3-* lines)
"""
import json, math, sys

import numpy as np

sys.path.insert(0, "scripts")
J = {s: json.load(open(f".cache/gsmb_agree_BT4M8_s{s}_BT4M8_s{s}_v1.json")) for s in (0, 1)}
t = {s: J[s]["TEST"] for s in J}
d = [t[s]["s0"]["diff"] for s in (0, 1)]; ag1 = np.mean(d) >= 0.010 and min(d) > 0
sh = [t[s]["s0"]["shared"] for s in (0, 1)]; agi = np.mean(sh) > 0.378
base = [t[s]["s0"]["beam1"] for s in (0, 1)]; prec = [t[s]["cert"]["precision"] for s in (0, 1)]; cov = [t[s]["cert"]["coverage"] for s in (0, 1)]
void = min(cov) < 0.02; agc = all(p >= 3 * b for p, b in zip(prec, base))
V = [f"AG3-1: TEST ans AGREE(view 0, view 1) - BEAM-1(view 0) {d[0]:+.4f} / {d[1]:+.4f} (mean {np.mean(d):+.4f}; bar +0.010, both > 0) -> {'PASS' if ag1 else 'FAIL'}",
     f"AG3-I: TEST shared-value rate across views {sh[0]:.3f} / {sh[1]:.3f} (mean {np.mean(sh):.3f}; bar > 0.378) -> {'PASS' if agi else 'FAIL'}",
     f"AG3-C: TEST certificate precision {prec[0]:.4f} / {prec[1]:.4f} vs 3 x BEAM-1(view 0) {3 * base[0]:.4f} / {3 * base[1]:.4f} "
     f"(coverage {cov[0]:.3f} / {cov[1]:.3f}; VOID below 0.02) -> {'VOID (coverage < 0.02)' if void else ('PASS' if agc else 'FAIL')}"]
# supplementary: non-identity items only (recomputed from the dumps)
Dv = json.load(open(".cache/gsmb_data_v1.json"))["TEST"]; ni = np.array([not it["ident"] for it in Dv])
key = lambda v: round(v * 1e6)
def pen(v, lam): return lam[0] * (abs(v - round(v)) > 1e-6) + lam[1] * (v < -1e-9)
def b1(r, lam):
    if not r["beam"]: return r["g_ok"], (None if r["g"] is None else key(r["g"]))
    i = int(np.argmax([lp - pen(v, lam) for lp, v, _ in r["beam"]])); return r["beam"][i][2], key(r["beam"][i][1])
X = []
for s in (0, 1):
    a, b = (json.load(open(f".cache/gsmb_agdump_BT4M8_s{s}{x}.json")) for x in ("", "_v1")); lam = a["lam"]
    A, B = a["TEST"], b["TEST"]; assert len(A) == len(B) == len(Dv)
    x0 = [b1(r, lam) for r in A]; x1 = [b1(r, lam) for r in B]
    cert = np.array([p[1] is not None and p[1] == q[1] for p, q in zip(x0, x1)]) & ni; right = np.array([p[0] for p in x0], float)
    X.append((ni.sum(), cert.sum() / ni.sum(), float(right[cert].mean()) if cert.any() else float("nan"), float(right[ni].mean())))
V.append("AG3-x (not a bar): non-identity TEST items only n %d | per seed certificate coverage %s precision %s vs BEAM-1(view 0) %s" %
         (X[0][0], "/".join(f"{x[1]:.3f}" for x in X), "/".join(f"{x[2]:.4f}" for x in X), "/".join(f"{x[3]:.4f}" for x in X)))
V.append(f"AG3-x identity views in TEST: {int((~ni).sum())}/{len(ni)} (same text twice: trivial agreement)")
print("\n".join(V)); open(".cache/gsm_ag3_verdict.txt", "w").write("\n".join(V) + "\n")
