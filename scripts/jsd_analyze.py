"""jsd_analyze.py — Phase B (pure CPU): the JSD verdict on banked waist
features. Readable = union-right row indices from recruit_round1.json;
residue = the rest (anchors excluded from BOTH sides — the question is
about the frontier, not the candles)."""
import json
import numpy as np

feats = np.load('.cache/jsd_feats.npy')
rec = json.load(open('.cache/recruit_round1.json'))
resid = set(rec["residue_rows"])
reads = json.load(open('.cache/audition_R01.json'))["rows"]
tags = [r["tag"] for r in reads]
front = [i for i, t in enumerate(tags) if t in ("wv", "held", "cen")]
R = [i for i in front if i in resid]        # unreadable frontier
G = [i for i in front if i not in resid]    # readable frontier
print(f"[jsd] frontier: readable {len(G)} vs residue {len(R)}")

def jsd_dim(x, y, bins=24):
    lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
    if hi <= lo: return 0.0
    px, _ = np.histogram(x, bins=bins, range=(lo, hi), density=False)
    py, _ = np.histogram(y, bins=bins, range=(lo, hi), density=False)
    px = px / px.sum() + 1e-12; py = py / py.sum() + 1e-12
    m = 0.5 * (px + py)
    kl = lambda a, b: float((a * np.log2(a / b)).sum())
    return 0.5 * kl(px, m) + 0.5 * kl(py, m)

X, Y = feats[G], feats[R]
js = np.array([jsd_dim(X[:, d], Y[:, d]) for d in range(feats.shape[1])])
top = np.argsort(-js)[:16]
print(f"[jsd] top dims: {[(int(d), round(float(js[d]), 3)) for d in top[:8]]}")
# held-split AUC on the top subspace (simple Fisher direction)
rng = np.random.default_rng(7)
gi = rng.permutation(G); ri = rng.permutation(R)
gtr, gte = gi[:len(gi)//2], gi[len(gi)//2:]
rtr, rte = ri[:len(ri)//2], ri[len(ri)//2:]
A, B = feats[gtr][:, top], feats[rtr][:, top]
w = np.linalg.pinv(np.cov(np.vstack([A, B]).T) + 1e-3 * np.eye(len(top))) @ (A.mean(0) - B.mean(0))
sg = feats[gte][:, top] @ w; sr = feats[rte][:, top] @ w
auc = float(np.mean([1.0 if a > b else 0.5 if a == b else 0.0
                     for a in sg for b in sr]))
print(f"[jsd] held-split AUC (top-16 subspace): {auc:.3f}  "
      f"(bar >= 0.75 = the segmenter earns its build)")
# residue structure: k-means over top subspace
from numpy.linalg import norm
Z = feats[R][:, top]
Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-6)
best = None
for k in (2, 3, 4):
    c = Z[rng.choice(len(Z), k, replace=False)]
    for _ in range(25):
        d = ((Z[:, None] - c[None]) ** 2).sum(-1)
        a = d.argmin(1)
        c = np.array([Z[a == j].mean(0) if (a == j).any() else c[j]
                      for j in range(k)])
    inertia = float(((Z - c[a]) ** 2).sum())
    sizes = [int((a == j).sum()) for j in range(k)]
    print(f"[jsd] residue k={k}: sizes {sizes} inertia {inertia:.0f}")
print("[jsd] VERDICT: " + ("SUBSPACE EXISTS — segmenter earns its build"
      if auc >= 0.75 else "residue is diet-shaped — round 2 answers with candidates"))
