"""caricature_read.py — THE CARICATURE READ (2026-09-10, the word; the 25th
gut: "stereotypes are exaggerated — would exaggerating the centroids give
more separation?"). Factor-KIND recognition by nearest centroid on the
per-slot loop states (a dump from dump_polar_states.py: all[N,7,24,512]),
gold kinds = (ftype, op): given / rel_add / rel_mul. Leave-one-out over
slots, per breath, in five spaces:
  raw-euclid, raw-cos          the monitor's own readers
  caric-a (a in 2, 3)          centroids pushed from the grand mean:
                               c' = m + a (c - m), euclid read
  white-l (lambda in 0.1, 0.5) shrinkage-whitened by the within-kind
                               covariance (LDA's caricature: amplify what
                               separates, suppress what is shared)
Plus the centroid cosine matrix raw vs whitened at the last breath.
Env: CA_DUMP (npz), CA_TEST mint|wild. CPU only."""
import os, sys, collections
os.environ.setdefault("DEV", "CPU")
FX = os.environ.get("CA_TEST", "mint")
os.environ.update({"ALG2": "1", "ALG_FTYPES": "9", "ALG_DUP": "1", "ALG_HW": "512", "ALG_WIDE": "1",
                   "ALG_TEST": ".cache/algebra_nl_test.jsonl" if FX == "mint" else ".cache/wild_admitted_holdout.jsonl",
                   "ALG_TEST_NAME": "test23" if FX == "mint" else "wildhold"})
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np
from phase1_algebra_head import load_alg
vs, vst, vtk, vg, vse = load_alg("test")
z = np.load(os.environ["CA_DUMP"]); X = z["all"]            # (N, K, L, H)
N, K, L, H = X.shape
pres = vg["presence"][:N] > 0.5
ft = vg["ftype"][:N]; op = vg["op"][:N]
kind = np.where(ft == 1, 0, np.where(op == 0, 1, 2))          # given / rel_add / rel_mul
KN = ["given", "rel_add", "rel_mul"]
y = kind[pres]; n = len(y)
print(f"[caricature] {FX} dump={os.environ['CA_DUMP']} rows={N} present slots={n} kinds={dict(collections.Counter(KN[k] for k in y))}")

def loo_centroid_acc(Z, y, metric):
    """Leave-one-out nearest centroid: the centroid of the row's own kind is
    recomputed without the row (exact, vectorized)."""
    C = len(KN); sums = np.zeros((C, Z.shape[1])); cnt = np.zeros(C)
    for c in range(C):
        sums[c] = Z[y == c].sum(0); cnt[c] = (y == c).sum()
    cent = sums / cnt[:, None]                                # (C, H)
    own = (sums[y] - Z) / (cnt[y] - 1)[:, None]               # LOO own centroid
    cents = np.repeat(cent[None], len(Z), 0); cents[np.arange(len(Z)), y] = own
    if metric == "cos":
        Zn = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)
        cn = cents / (np.linalg.norm(cents, axis=2, keepdims=True) + 1e-9)
        s = np.einsum("nh,nch->nc", Zn, cn); pred = s.argmax(1)
    else:
        d = ((cents - Z[:, None, :]) ** 2).sum(-1); pred = d.argmin(1)
    return float((pred == y).mean()), cent

def caricature(Z, y, a):
    C = len(KN); m = Z.mean(0)
    sums = np.zeros((C, Z.shape[1])); cnt = np.zeros(C)
    for c in range(C):
        sums[c] = Z[y == c].sum(0); cnt[c] = (y == c).sum()
    cent = sums / cnt[:, None]; own = (sums[y] - Z) / (cnt[y] - 1)[:, None]
    cents = np.repeat(cent[None], len(Z), 0); cents[np.arange(len(Z)), y] = own
    cents = m + a * (cents - m)
    d = ((cents - Z[:, None, :]) ** 2).sum(-1)
    return float((d.argmin(1) == y).mean())

def whiten(Z, y, lam):
    C = len(KN); W = np.zeros((Z.shape[1], Z.shape[1]))
    for c in range(C):
        Zc = Z[y == c] - Z[y == c].mean(0); W += Zc.T @ Zc
    W /= len(Z); tr = np.trace(W) / Z.shape[1]
    S = (1 - lam) * W + lam * tr * np.eye(Z.shape[1])
    ev, U = np.linalg.eigh(S); Wm = U @ np.diag(ev ** -0.5) @ U.T
    return Z @ Wm

rows = []
for k in range(K):
    Z = X[:, k][pres].astype(np.float64)                       # (n, H)
    r_e, cent = loo_centroid_acc(Z, y, "euclid"); r_c, _ = loo_centroid_acc(Z, y, "cos")
    c2 = caricature(Z, y, 2.0); c3 = caricature(Z, y, 3.0)
    w1, _ = loo_centroid_acc(whiten(Z, y, 0.1), y, "euclid"); w5, _ = loo_centroid_acc(whiten(Z, y, 0.5), y, "euclid")
    rows.append((k, r_e, r_c, c2, c3, w1, w5))
    print(f"[caricature] {FX} b{k}: raw-euclid={r_e:.3f} raw-cos={r_c:.3f} caric-2={c2:.3f} caric-3={c3:.3f} white-0.1={w1:.3f} white-0.5={w5:.3f}")
    if k == K - 1:
        cn = cent / np.linalg.norm(cent, axis=1, keepdims=True)
        Zw = whiten(Z, y, 0.5); cw = np.stack([Zw[y == c].mean(0) for c in range(len(KN))]); cwn = cw / np.linalg.norm(cw, axis=1, keepdims=True)
        print(f"[caricature] {FX} b{k} centroid cosines raw: " + " ".join(f"{KN[i]}/{KN[j]}={cn[i]@cn[j]:.3f}" for i in range(3) for j in range(i + 1, 3)))
        print(f"[caricature] {FX} b{k} centroid cosines whitened(0.5): " + " ".join(f"{KN[i]}/{KN[j]}={cwn[i]@cwn[j]:.3f}" for i in range(3) for j in range(i + 1, 3)))
k, r_e, r_c, c2, c3, w1, w5 = rows[-1]
best = max(c2, c3, w1, w5); raw = max(r_e, r_c)
print(f"[caricature] {FX} VERDICT b{k}: raw best {raw:.3f} -> caricature/whitened best {best:.3f} (delta {best - raw:+.3f}; bar >= +0.050 on wild)")
