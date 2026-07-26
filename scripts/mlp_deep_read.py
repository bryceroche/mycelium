"""mlp_deep_read.py — the MLP read on the two floor-suspect curves
(2026-07-26; MLP fixtures ALL PASS first; three outcomes registered at
28861ff; rate prediction: rise centered breath 10-12)."""
import sys, json
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np
from dart_cluster_probe import auc_mann_whitney
from mlp_probe import mlp_scores
from crossover_depth_graded_read import fr, reps, valid, giv, K, H, Ni
d = np.load(".cache/crossover_capture_k16.npz")
gold = d["gold"].astype(int)
rng = np.random.default_rng(0); perm = rng.permutation(Ni)
fit_i, read_i = perm[:160], perm[160:]
base = valid & ~giv

def bal_fit(Xf, yf, cap=1500):
    cls = np.unique(yf)
    nmin = min(min((yf == c).sum() for c in cls), cap)
    idx = np.concatenate([rng.choice(np.where(yf == c)[0], nmin, replace=False) for c in cls])
    return Xf[idx], yf[idx], cls

res = {}
# family 1: d3-schema (newly-forced-at-round-3), binary AUC
m = base; y = (fr == 3).astype(int)
curve = []
for k in range(K):
    X = reps[k].astype(np.float32)
    Xf, yf = X[fit_i][m[fit_i]], y[fit_i][m[fit_i]]
    Xr, yr = X[read_i][m[read_i]], y[read_i][m[read_i]]
    mu = Xf.mean(0); sd = Xf.std(0) + 1e-6
    Xb, yb, cls = bal_fit((Xf-mu)/sd, yf)
    Y = (yb[:, None] == cls[None, :]).astype(np.float32)
    s = mlp_scores(Xb, Y, (Xr-mu)/sd, steps=500)
    curve.append(float(auc_mann_whitney(s[:, 1]-s[:, 0], yr == 1)))
print("[MLP d3-schema AUC ] " + " ".join(f"{a:.3f}" for a in curve))
res["d3_schema"] = curve
# family 2: d3plus-solution (gold at fr>=3 cells), macro-recall
m2 = base & (fr >= 3)
curve2 = []
for k in range(K):
    X = reps[k].astype(np.float32)
    Xf, yf = X[fit_i][m2[fit_i]], gold[fit_i][m2[fit_i]]
    Xr, yr = X[read_i][m2[read_i]], gold[read_i][m2[read_i]]
    mu = Xf.mean(0); sd = Xf.std(0) + 1e-6
    Xb, yb, cls = bal_fit((Xf-mu)/sd, yf)
    Y = (yb[:, None] == cls[None, :]).astype(np.float32)
    s = mlp_scores(Xb, Y, (Xr-mu)/sd, steps=500)
    pred = cls[np.argmax(s, 1)]
    rec = [float((pred[yr == c] == c).mean()) for c in cls if (yr == c).sum() > 0]
    curve2.append(float(np.mean(rec)))
print("[MLP d3+ solution  ] " + " ".join(f"{a:.3f}" for a in curve2))
res["d3plus_solution"] = curve2

for name, c, basel in (("d3_schema", curve, 0.5), ("d3plus_solution", curve2, 1/7)):
    c = np.array(c); rise = c.max()-c[0]; above = bool(np.all(c > basel))
    ok = rise >= 0.05 and above
    t = int(np.argmax(c >= 0.95*c[-1])) if ok else None
    knee = int(np.argmax(np.diff(c).cumsum() >= 0.5*(c.max()-c[0]))) if rise > 0 else None
    print(f"  {name}: rise {rise:+.3f} above {above} -> {'t95='+str(t)+' half-rise@'+str(knee) if ok else 'RANGE FAILED'}")
    res[name + "_verdict"] = {"rise": float(rise), "t95": t, "half_rise": knee}
json.dump(res, open(".cache/mlp_deep_read.json", "w"), indent=1)
