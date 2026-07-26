"""crossover_probe_read.py — THE ALTITUDE-CROSSOVER PROBE'S READ (2026-07-25).
Zero-GPU: linear probes (ridge to one-hot, argmax) fit per (breath, family)
on the banked capture, by-instance fit/read split. Signatures A/B/C + the
fourth bin and the t95 operationalization are pre-registered in the ledger;
this script prints curves, t95, and the mechanical verdict.

Families (KenKen, the engine's home domain):
  SURFACE  : input_cells token identity (clue/given/blank at the cell)
  SCHEMA   : per-cell cage op (constraint type the cell lives under)
  SOLUTION : gold value at NON-GIVEN cells (the deduced content)

Monotone tolerance (stated, transparent): a family is monotone-improving
if no breath drops more than 0.02 absolute below its running max. Raw
curves print regardless; ambiguity -> UNCLASSIFIED per the fourth bin.
"""
import sys
sys.path.insert(0, ".")
import numpy as np
import json

d = np.load(".cache/crossover_capture_k16.npz")
reps = d["reps"]                      # (K, N, S, H) fp16
K, N, S, H = reps.shape
valid = d["cell_valid"].astype(bool)  # (N, S)
giv = d["is_given"].astype(bool)
rng = np.random.default_rng(0)
perm = rng.permutation(N)
fit_i, read_i = perm[:160], perm[160:]

FAMS = {
    "surface":  (d["input_cells"].astype(int), valid),
    "schema":   (d["cage_op"].astype(int),     valid),
    "solution": (d["gold"].astype(int),        valid & ~giv),
}

def probe_acc(k, y, m):
    X = reps[k].astype(np.float32)              # (N, S, H)
    Xf = X[fit_i][m[fit_i]]; yf = y[fit_i][m[fit_i]]
    Xr = X[read_i][m[read_i]]; yr = y[read_i][m[read_i]]
    mu = Xf.mean(0); Xf = Xf - mu; Xr = Xr - mu
    classes = np.unique(np.concatenate([yf, yr]))
    Y = (yf[:, None] == classes[None, :]).astype(np.float32)
    A = Xf.T @ Xf + 100.0 * np.eye(H, dtype=np.float32)
    W = np.linalg.solve(A, Xf.T @ Y)
    pred = classes[np.argmax(Xr @ W, axis=1)]
    return float((pred == yr).mean())

curves = {}
for fam, (y, m) in FAMS.items():
    curves[fam] = [probe_acc(k, y, m) for k in range(K)]
    print(f"[{fam:9}] " + " ".join(f"{a:.3f}" for a in curves[fam]), flush=True)

def t95(c):
    c = np.array(c); target = 0.95 * c[-1]
    return int(np.argmax(c >= target))

def monotone(c, tol=0.02):
    c = np.array(c); runmax = np.maximum.accumulate(c)
    return bool(np.all(c >= runmax - tol))

t = {f: t95(c) for f, c in curves.items()}
mono = {f: monotone(c) for f, c in curves.items()}
print("\n=== THE READ (signatures + t95 pre-registered; ledger 2026-07-25) ===")
print(f"  t95: surface={t['surface']} schema={t['schema']} solution={t['solution']}")
print(f"  monotone(tol .02): {mono}")

# rank crossover (A): schema decode DECLINES from an early peak while
# solution passes it — operationalized: schema's peak breath is early
# (< K/2), schema's final is >= 0.05 below its peak, and solution's
# final exceeds schema's final.
sc = np.array(curves["schema"]); so = np.array(curves["solution"])
crossover = (int(np.argmax(sc)) < K // 2 and sc[-1] <= sc.max() - 0.05
             and so[-1] > sc[-1])
ordering_C = t["surface"] <= t["schema"] < t["solution"]
all_mono = all(mono.values())

if crossover:
    verdict = "A — SEQUENTIAL LOWERING (rank crossover)"
elif all_mono and ordering_C:
    verdict = "C — PARALLEL LOWERING / DIFFUSION-LIKE (no crossover; asymptote ordered by altitude)"
elif all_mono and not ordering_C and t["surface"] == t["schema"] == t["solution"]:
    verdict = "B — FLAT CONVERGENCE (no ordering in asymptote arrival)"
elif all_mono and not ordering_C:
    verdict = "UNCLASSIFIED (fourth bin): monotone but asymptote ordering matches no signature"
else:
    verdict = "UNCLASSIFIED (fourth bin): non-monotone curve(s) no signature predicted"
print(f"  VERDICT: {verdict}")
print("  (scope rivet: binds the KenKen engine's breath dynamics; v200 transfer is a separate measurable question)")
json.dump({"curves": curves, "t95": t, "monotone": mono,
           "crossover": bool(crossover), "verdict": verdict},
          open(".cache/crossover_probe_read.json", "w"), indent=1)
print("[read] banked -> .cache/crossover_probe_read.json")
