"""crossover_balanced_read.py — the balanced re-read under the pinned range
criterion (ledger 2026-07-25, b94fa11). AUC (Mann-Whitney, validated import)
for naked_single; macro-recall for cand_size. t95 counts ONLY on certified
dynamic range: rise >= 0.05 AND above-baseline throughout.
"""
import sys, json
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np
from dart_cluster_probe import auc_mann_whitney
from crossover_offceiling_read import ns, cs, reps, valid, giv, K, H, Ni

rng = np.random.default_rng(0); perm = rng.permutation(Ni)
fit_i, read_i = perm[:160], perm[160:]
m = valid & ~giv

def ridge_score(k, y):
    X = reps[k].astype(np.float32)
    Xf = X[fit_i][m[fit_i]]; yf = y[fit_i][m[fit_i]]
    Xr = X[read_i][m[read_i]]; yr = y[read_i][m[read_i]]
    mu = Xf.mean(0); Xf = Xf - mu; Xr = Xr - mu
    return Xf, yf, Xr, yr

auc_curve = []
for k in range(K):
    Xf, yf, Xr, yr = ridge_score(k, ns)
    w = np.linalg.solve(Xf.T @ Xf + 100.0*np.eye(H, dtype=np.float32),
                        Xf.T @ (2.0*yf - 1.0).astype(np.float32))
    s = Xr @ w
    auc_curve.append(float(auc_mann_whitney(s, yr == 1)))
print("[naked_single AUC] " + " ".join(f"{a:.3f}" for a in auc_curve))

mr_curve = []
for k in range(K):
    Xf, yf, Xr, yr = ridge_score(k, cs)
    cls = np.unique(yf)
    nmin = min((yf == c).sum() for c in cls)
    idx = np.concatenate([rng.choice(np.where(yf == c)[0], nmin, replace=False) for c in cls])
    Xb, yb = Xf[idx], yf[idx]
    Y = (yb[:, None] == cls[None, :]).astype(np.float32)
    W = np.linalg.solve(Xb.T @ Xb + 100.0*np.eye(H, dtype=np.float32), Xb.T @ Y)
    pred = cls[np.argmax(Xr @ W, 1)]
    rec = [float((pred[yr == c] == c).mean()) for c in cls if (yr == c).sum() > 0]
    mr_curve.append(float(np.mean(rec)))
print("[cand_size mrec  ] " + " ".join(f"{a:.3f}" for a in mr_curve))

def judge(c, base, name):
    c = np.array(c); rise = c.max() - c[0]; above = bool(np.all(c > base))
    ok = rise >= 0.05 and above
    t = int(np.argmax(c >= 0.95*c[-1])) if ok else None
    print(f"  {name}: rise {rise:+.3f} | above-baseline({base}) {above} | RANGE {'CERTIFIED, t95='+str(t) if ok else 'FAILED — contributes no t95'}")
    return t
t1 = judge(auc_curve, 0.5, "naked_single")
t2 = judge(mr_curve, 1/7, "cand_size")
print("  LEG:", "OPEN — MLP follow-on pre-authorized" if t1 is None and t2 is None else f"t95s {t1},{t2} vs solution 5")
json.dump({"auc": auc_curve, "mrec": mr_curve}, open(".cache/crossover_balanced_read.json","w"), indent=1)
