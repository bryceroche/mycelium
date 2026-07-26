"""crossover_depth_graded_read.py — THE LEG'S CLOSING INSTRUMENT
(2026-07-25; meter fixtures ALL PASS first). Iterated propagation labels
newly-forced-at-round-d cells; probe AUC per breath per depth. Pinned:
ordering bar (t95 climbs with d, stays <5 -> seals C), linearity read
(t95~d, separate finding), min-support >=100 read positives, range
criterion (rise >=0.05 + above-baseline).
"""
import sys, json
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np
from itertools import product as iproduct
from dart_cluster_probe import auc_mann_whitney
from crossover_offceiling_read import recs, reps, valid, giv, K, H, Ni

MAXD = 4
def propagate_rounds(r):
    N = r["N"]
    cand = [[set(range(1, N+1)) for _ in range(N)] for _ in range(N)]
    forced_round = np.zeros((N, N), int)  # 0 = given/never; d = newly singleton at round d
    for cells, (op, tgt) in zip(r["cages"], r["clues"]):
        if op == "given":
            rr, cc = cells[0]; cand[rr][cc] = {tgt}; forced_round[rr][cc] = -1
    for d in range(1, MAXD + 1):
        newly = []
        # row/col elimination from all current singletons
        for rr in range(N):
            for cc in range(N):
                if len(cand[rr][cc]) == 1:
                    v = next(iter(cand[rr][cc]))
                    for k in range(N):
                        if k != cc: cand[rr][k].discard(v)
                        if k != rr: cand[k][cc].discard(v)
        # cage filtering
        for cells, (op, tgt) in zip(r["cages"], r["clues"]):
            if op == "given" or len(cells) > 4: continue
            for idx, (rr, cc) in enumerate(cells):
                others = [cand[r2][c2] for j,(r2,c2) in enumerate(cells) if j != idx]
                keep = set()
                for v in cand[rr][cc]:
                    for combo in iproduct(*others):
                        vals = [v] + list(combo)
                        if op == "add" and sum(vals) == tgt: keep.add(v); break
                        if op == "mul":
                            p = 1
                            for x in vals: p *= x
                            if p == tgt: keep.add(v); break
                        if op == "sub" and len(vals) == 2 and abs(vals[0]-vals[1]) == tgt: keep.add(v); break
                        if op == "div" and len(vals) == 2 and (vals[0] == tgt*vals[1] or vals[1] == tgt*vals[0]): keep.add(v); break
                cand[rr][cc] = keep
        for rr in range(N):
            for cc in range(N):
                if forced_round[rr][cc] == 0 and len(cand[rr][cc]) == 1:
                    forced_round[rr][cc] = d
    return forced_round

S = reps.shape[2]
fr = np.zeros((Ni, S), int)
for i, r in enumerate(recs):
    f = propagate_rounds(r); N = r["N"]
    for rr in range(N):
        for cc in range(N):
            fr[i, rr*7+cc] = f[rr][cc]

rng = np.random.default_rng(0); perm = rng.permutation(Ni)
fit_i, read_i = perm[:160], perm[160:]
m = valid & ~giv
results = {}
for d in range(1, MAXD + 1):
    y = (fr == d).astype(int)
    n_pos_read = int(y[read_i][m[read_i]].sum())
    if n_pos_read < 100:
        print(f"[depth {d}] INSUFFICIENT-SUPPORT ({n_pos_read} read positives < 100)")
        results[f"d{d}"] = {"support": n_pos_read, "verdict": "insufficient-support"}
        continue
    curve = []
    for k in range(K):
        X = reps[k].astype(np.float32)
        Xf = X[fit_i][m[fit_i]]; yf = y[fit_i][m[fit_i]]
        Xr = X[read_i][m[read_i]]; yr = y[read_i][m[read_i]]
        mu = Xf.mean(0)
        w = np.linalg.solve((Xf-mu).T@(Xf-mu) + 100.0*np.eye(H, dtype=np.float32),
                            (Xf-mu).T@(2.0*yf-1.0).astype(np.float32))
        curve.append(float(auc_mann_whitney((Xr-mu)@w, yr == 1)))
    c = np.array(curve); rise = c.max()-c[0]; above = bool(np.all(c > 0.5))
    ok = rise >= 0.05 and above
    t = int(np.argmax(c >= 0.95*c[-1])) if ok else None
    print(f"[depth {d}] n_read+ {n_pos_read} | " + " ".join(f"{a:.3f}" for a in curve))
    print(f"          rise {rise:+.3f} above {above} -> {'t95='+str(t) if ok else 'RANGE FAILED (flat-high edge if above-baseline)'}")
    results[f"d{d}"] = {"support": n_pos_read, "curve": curve, "rise": float(rise), "t95": t}
json.dump(results, open(".cache/crossover_depth_graded_read.json","w"), indent=1)
ts = [(int(k[1]), v["t95"]) for k, v in results.items() if v.get("t95") is not None]
print("\nbanked t95s by depth:", ts, "| solution t95: 5")
