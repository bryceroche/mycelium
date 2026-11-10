"""gsm_hop_census.py -- THE HOP CENSUS (ledger 2026-11-10 pin): oracle-plan pointer accuracy on TEST-U by gold hop depth and by
operand kind, stratified by step index k. Zero GPU: reads the banked .cache/gsm_hr_{ARM}.npz + the gold programs."""
import json, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gsm_bench_data as GD

K = 12


def depths(prog):
    d = []
    for o, A, B in prog:
        refs = A["s"] + B["s"]
        d.append(1 + max((d[r] for r in refs), default=0))
    return d


def strat(x, k, a, b):
    """sum_k w_k (mean x[a & k] - mean x[b & k]) / sum_k w_k"""
    num = den = 0.0
    for kk in np.unique(k):
        na, nb = (a & (k == kk)).sum(), (b & (k == kk)).sum()
        if na and nb:
            w = na * nb / (na + nb); num += w * (x[a & (k == kk)].mean() - x[b & (k == kk)].mean()); den += w
    return num / den if den else float("nan")


its = json.load(open(GD.OUT))["TEST"]
out = []
for arm in ("BIND_BIND_s0", "BINDT_BIND_s1"):
    h = np.load(f".cache/gsm_hr_{arm}.npz"); ptr, seen = h["ptr"], h["seen"]
    dep, k, ref = [], [], []
    for it in its:
        p = it["prog"]; m = min(len(p), K); dd = depths(p)[:m]
        dep += dd; k += list(range(m)); ref += [bool(A["s"] or B["s"]) for o, A, B in p[:m]]
    dep, k, ref = np.array(dep), np.array(k), np.array(ref); assert len(dep) == len(ptr)
    u = ~seen; dc = np.minimum(dep, 3)
    row = [f"{arm} TEST-U n {u.sum()} | ptr by depth: " + " ".join(f"d{d}{'+' if d == 3 else ''} {ptr[u & (dc == d)].mean():.3f} "
           f"(n {(u & (dc == d)).sum()})" for d in (1, 2, 3))]
    row.append(f"  by kind: step-ref {ptr[u & ref].mean():.3f} (n {(u & ref).sum()}) tokens-only {ptr[u & ~ref].mean():.3f} (n {(u & ~ref).sum()})")
    hd = strat(ptr, k, u & (dc == 3), u & (dc == 1)); rd = strat(ptr, k, u & ref, u & ~ref)
    row.append(f"  HOP (d3+ - d1 | k): {hd:+.4f} | REF (ref - tokens | k): {rd:+.4f}")
    row.append(f"  [seen, same] HOP {strat(ptr, k, seen & (dc == 3), seen & (dc == 1)):+.4f} REF {strat(ptr, k, seen & ref, seen & ~ref):+.4f}")
    out += row; out.append((arm, hd, rd))
txt = [r for r in out if isinstance(r, str)]; vals = [r for r in out if isinstance(r, tuple)]
txt.append(f"HOP-NAMED: {'NAMED' if all(v[1] <= -0.10 for v in vals) else 'none named'} | "
           f"REF-NAMED: {'NAMED' if all(v[2] <= -0.10 for v in vals) else 'none named'}")
print("\n".join(txt)); open(".cache/gsm_hop_census.txt", "w").write("\n".join(txt) + "\n")
