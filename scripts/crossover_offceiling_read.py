"""crossover_offceiling_read.py — THE VERDICT-CRITICAL READ (2026-07-25).
Derived-schema family: depth-1 candidate propagation facts (naked-single
flag, candidate-set size) — content existing nowhere in the input.
Sidecar recovered via seed-0 determinism. Seal rule (pinned): derived-schema
t95 strictly < solution t95 (5) -> C SEALS; == or > -> B prints.
"""
import sys, json
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np
from itertools import product as iproduct
from diag_kenken_granularity_probe import _sample_balanced_records

recs = _sample_balanced_records(".cache/kenken_test_curriculum.jsonl",
                                ["g10","g20","g30","g40"], 240, [5,6,7], seed=0)
d = np.load(".cache/crossover_capture_k16.npz")
reps = d["reps"]; K, Ni, S, H = reps.shape
assert Ni == len(recs)
valid = d["cell_valid"].astype(bool); giv = d["is_given"].astype(bool)
gold_bank = d["gold"].astype(int)

def cands_depth1(r):
    N = r["N"]
    cand = [[set(range(1, N+1)) for _ in range(N)] for _ in range(N)]
    givens = []
    for cells, (op, tgt) in zip(r["cages"], r["clues"]):
        if op == "given":
            (rr, cc) = cells[0]; givens.append((rr, cc, tgt))
    for rr, cc, v in givens:
        cand[rr][cc] = {v}
        for k in range(N):
            if k != cc: cand[rr][k].discard(v)
            if k != rr: cand[k][cc].discard(v)
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
    return cand

ns = np.zeros((Ni, S), int); cs = np.zeros((Ni, S), int)
for i, r in enumerate(recs):
    N = r["N"]; cand = cands_depth1(r)
    for rr in range(N):
        for cc in range(N):
            s = rr * 7 + cc  # S=49 grid layout, N_MAX=7
            ns[i, s] = 1 if len(cand[rr][cc]) == 1 else 0
            cs[i, s] = min(len(cand[rr][cc]), 7)

# sanity: naked singles at non-given cells should match gold where flagged
chk = valid & ~giv & (ns == 1)
print(f"[sanity] naked-single cells: {chk.sum()} ({100*chk.sum()/max(1,(valid&~giv).sum()):.1f}% of deducible)")

rng = np.random.default_rng(0); perm = rng.permutation(Ni)
fit_i, read_i = perm[:160], perm[160:]
def probe_acc(k, y, m):
    X = reps[k].astype(np.float32)
    Xf = X[fit_i][m[fit_i]]; yf = y[fit_i][m[fit_i]]
    Xr = X[read_i][m[read_i]]; yr = y[read_i][m[read_i]]
    mu = Xf.mean(0); Xf = Xf - mu; Xr = Xr - mu
    cls = np.unique(np.concatenate([yf, yr]))
    Y = (yf[:, None] == cls[None, :]).astype(np.float32)
    W = np.linalg.solve(Xf.T @ Xf + 100.0*np.eye(H, dtype=np.float32), Xf.T @ Y)
    return float((cls[np.argmax(Xr @ W, 1)] == yr).mean())

m = valid & ~giv
curves = {}
for name, y in (("naked_single", ns), ("cand_size", cs)):
    curves[name] = [probe_acc(k, y, m) for k in range(K)]
    print(f"[{name:12}] " + " ".join(f"{a:.3f}" for a in curves[name]))
def t95(c):
    c = np.array(c); return int(np.argmax(c >= 0.95*c[-1]))
ts = {n: t95(c) for n, c in curves.items()}
print(f"\n  derived-schema t95: {ts} | solution t95 (banked): 5")
worst = max(ts.values())
print("  VERDICT LEG:", "C SEALS — derived schema resolves strictly inside solution" if worst < 5
      else ("B PRINTS — asymptotes arrive together" if worst == 5 else
            "UNCLASSIFIED — derived schema resolves AFTER solution (no signature predicted)"))
json.dump({"curves": curves, "t95": ts}, open(".cache/crossover_offceiling_read.json","w"), indent=1)
