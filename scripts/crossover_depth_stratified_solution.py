"""crossover_depth_stratified_solution.py — the confound test (2026-07-26,
bars pinned at 95c4f94). Solution family stratified by forced-round depth;
balanced meter; late-rise >= 0.03 on deep stratum dissolves the inversion
and prints the DEPTH-AXIS hypothesis."""
import sys, json
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np
from crossover_depth_graded_read import fr, reps, valid, giv, K, H, Ni
d = np.load(".cache/crossover_capture_k16.npz")
gold = d["gold"].astype(int)
rng = np.random.default_rng(0); perm = rng.permutation(Ni)
fit_i, read_i = perm[:160], perm[160:]

def mrec_curve(mask):
    out = []
    for k in range(K):
        X = reps[k].astype(np.float32)
        Xf = X[fit_i][mask[fit_i]]; yf = gold[fit_i][mask[fit_i]]
        Xr = X[read_i][mask[read_i]]; yr = gold[read_i][mask[read_i]]
        mu = Xf.mean(0); Xf = Xf - mu; Xr = Xr - mu
        cls = np.unique(yf)
        nmin = min((yf == c).sum() for c in cls)
        idx = np.concatenate([rng.choice(np.where(yf == c)[0], nmin, replace=False) for c in cls])
        Y = (yf[idx][:, None] == cls[None, :]).astype(np.float32)
        W = np.linalg.solve(Xf[idx].T @ Xf[idx] + 100.0*np.eye(H, dtype=np.float32), Xf[idx].T @ Y)
        pred = cls[np.argmax(Xr @ W, 1)]
        rec = [float((pred[yr == c] == c).mean()) for c in cls if (yr == c).sum() > 0]
        out.append(float(np.mean(rec)))
    return out

base = valid & ~giv
strata = {"d1": base & (fr == 1), "d2": base & (fr == 2), "d3plus": base & (fr >= 3)}
res = {}
for name, m in strata.items():
    n_read = int(m[read_i].sum())
    if n_read < 100:
        print(f"[{name}] INSUFFICIENT-SUPPORT ({n_read})"); res[name] = {"support": n_read}; continue
    c = mrec_curve(m)
    late = c[15] - c[5]
    t = int(np.argmax(np.array(c) >= 0.95*c[-1]))
    print(f"[{name}] n_read {n_read} | " + " ".join(f"{a:.3f}" for a in c))
    print(f"        t95 {t} | late-rise c[15]-c[5] = {late:+.3f} (bar: >=0.03 dissolves inversion)")
    res[name] = {"support": n_read, "curve": c, "t95": t, "late_rise": float(late)}
deep = res.get("d3plus", {})
if "late_rise" in deep:
    print("\nVERDICT:", "INVERSION DISSOLVES — deep solution co-resolves late; DEPTH-AXIS PRINTS (resolution tracks depth regardless of family)"
          if deep["late_rise"] >= 0.03 else
          "INVERSION STANDS — deep solution asymptotes with the mean; schema-after-answer remains the strange fact")
json.dump(res, open(".cache/crossover_depth_stratified_solution.json", "w"), indent=1)
