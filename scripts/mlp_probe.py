"""mlp_probe.py — the sharper ruler (2026-07-26): tiny numpy MLP probe
(1024 -> 64 ReLU -> out, plain GD, balanced subsample) + its known-signal
fixtures (REQUIRED to pass before any real read banks; null fixture
first-class per the scaled apparatus law)."""
import numpy as np

def mlp_scores(Xf, yf_onehot, Xr, seed=0, hid=64, steps=400, lr=0.05):
    rng = np.random.default_rng(seed)
    H = Xf.shape[1]; C = yf_onehot.shape[1]
    W1 = rng.normal(0, 1/np.sqrt(H), (H, hid)).astype(np.float32); b1 = np.zeros(hid, np.float32)
    W2 = np.zeros((hid, C), np.float32); b2 = np.zeros(C, np.float32)
    for _ in range(steps):
        h = np.maximum(Xf @ W1 + b1, 0)
        z = h @ W2 + b2
        z -= z.max(1, keepdims=True); p = np.exp(z); p /= p.sum(1, keepdims=True)
        g = (p - yf_onehot) / len(Xf)
        gW2 = h.T @ g; gb2 = g.sum(0)
        gh = (g @ W2.T) * (h > 0)
        gW1 = Xf.T @ gh; gb1 = gh.sum(0)
        W2 -= lr*gW2; b2 -= lr*gb2; W1 -= lr*gW1; b1 -= lr*gb1
    hr = np.maximum(Xr @ W1 + b1, 0)
    return hr @ W2 + b2

def fixtures():
    import sys; sys.path.insert(0, "scripts")
    from dart_cluster_probe import auc_mann_whitney
    fails = []
    rng = np.random.default_rng(2)
    # F1: XOR — nonlinear signal linear ridge CANNOT read, MLP must (AUC>0.85)
    X = rng.normal(size=(3000, 8)).astype(np.float32)
    y = (X[:, 0] * X[:, 1] > 0)
    Xf, Xr, yf, yr = X[:2000], X[2000:], y[:2000], y[2000:]
    w = np.linalg.solve(Xf.T@Xf + 10*np.eye(8, dtype=np.float32), Xf.T@(2.*yf-1).astype(np.float32))
    a_lin = auc_mann_whitney(Xr@w, yr)
    Y = np.stack([~yf, yf], 1).astype(np.float32)
    s = mlp_scores(Xf, Y, Xr, steps=600)
    a_mlp = auc_mann_whitney(s[:, 1]-s[:, 0], yr)
    print(f"[MLP-F1] XOR: linear AUC {a_lin:.3f} (must be <0.6) | MLP AUC {a_mlp:.3f} (must be >0.85)")
    (a_lin < 0.6 and a_mlp > 0.85) or fails.append(1)
    # F2: NULL — shuffled labels must print chance (the expressive-probe trap)
    ysh = rng.permutation(y)
    s2 = mlp_scores(Xf, np.stack([~ysh[:2000], ysh[:2000]], 1).astype(np.float32), Xr, steps=600)
    a_null = auc_mann_whitney(s2[:, 1]-s2[:, 0], ysh[2000:])
    print(f"[MLP-F2] null: shuffled AUC {a_null:.3f} (must be in [.42,.58])")
    (.42 < a_null < .58) or fails.append(2)
    # F3: linear signal — MLP must not be WORSE than ridge (AUC>0.9)
    w0 = rng.normal(size=8); yl = (X @ w0 > 0)
    s3 = mlp_scores(Xf, np.stack([~yl[:2000], yl[:2000]], 1).astype(np.float32), Xr, steps=600)
    a3 = auc_mann_whitney(s3[:, 1]-s3[:, 0], yl[2000:])
    print(f"[MLP-F3] linear: MLP AUC {a3:.3f} (must be >0.9)")
    a3 > 0.9 or fails.append(3)
    print("MLP METER FIXTURES:", "ALL PASS" if not fails else f"FAIL {fails}")
    return not fails

if __name__ == "__main__":
    import sys
    sys.exit(0 if fixtures() else 1)
