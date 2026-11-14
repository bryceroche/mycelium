"""rel_dsl_gq2_verdict.py -- ROUTING QUEUE 2 verdict (ledger 2026-11-14 15:48), NOMLP cov 0.25, two seeds, the fixture frame (C0 VOID; >= +0.03
both seeds positive HELPS / KEEPS; <= -0.02 HURTS; else EQUIVALENT / CEILING). G4N{r} - additive (KEEPS; retention of G4's lift); BYPASS
iff G4N10 - G8N10 >= +0.03 both seeds positive; GKS = G8K3S - additive; DISCOVERS iff the last-breath true-neighbour share >= 3x chance
both seeds; the autopsy (derived-var routed mass on factors) beside. -> .cache/rel_dsl_gq2_verdict.txt"""
import json, os

import numpy as np

J = lambda t: json.load(open(f".cache/rel_dsl_{t}.json"))
C = "_depth2c0.25_s"
R = {"additive": [J(f"TGNOMLP{C}0"), J(f"TGNOMLP{C}1")], **{k: [J(f"TGNOMLP{k}{C}{s}") for s in (0, 1)] for k in ("G4", "G8N10", "G8N30", "G4N10", "G4N30", "G8K3", "G8K3S")}}
out = []
for k, X in R.items():
    bd = {d: np.mean([r["deep_by_depth"][d] for r in X]) for d in X[0]["deep_by_depth"]}
    out.append(f"  {k:9s} SHALLOW {X[0]['shallow']:.3f}/{X[1]['shallow']:.3f} MID {X[0]['mid']:.3f}/{X[1]['mid']:.3f} DEEP {X[0]['deep']:.3f}/{X[1]['deep']:.3f} | by depth "
               + " ".join(f"d{d} {v:.3f}" for d, v in bd.items()))


def diff(a, b):
    return [float(np.mean(np.array(x["deep_per"]) - np.array(y["deep_per"]))) for x, y in zip(R[a], R[b])]


def frame(a, b, win="HELPS"):
    d = diff(a, b); m = float(np.mean(d)); void = any(np.mean([r["shallow"] for r in R[k]]) < 0.95 for k in (a, b))
    ceil = min(np.mean([r["deep"] for r in R[a]]), np.mean([r["deep"] for r in R[b]])) >= 0.97
    v = "VOID (C0)" if void else win if m >= 0.03 and min(d) > 0 else "HURTS" if m <= -0.02 else "CEILING" if ceil else "EQUIVALENT"
    return m, f"{m:+.3f} (seeds {d[0]:+.3f} / {d[1]:+.3f}) -> {v}"


lift = float(np.mean(diff("G4", "additive")))
for r in (10, 30):
    m, t = frame(f"G4N{r}", "additive", "KEEPS"); out.append(f"G4N{r} - additive: {t} | retention of G4's +{lift:.3f}: {m / lift:.2f} | beside G4N{r} - G8N{r} {frame(f'G4N{r}', f'G8N{r}')[1]}")
d = diff("G4N10", "G8N10"); out.append(f"BYPASS (G4N10 - G8N10 >= +0.03 both seeds positive): {'YES' if np.mean(d) >= 0.03 and min(d) > 0 else 'NO'}")
out.append(f"GKS (G8K3S - additive): {frame('G8K3S', 'additive')[1]} | beside G8K3S - G8K3 {frame('G8K3S', 'G8K3')[1]}")
rat = []
for s in (0, 1):
    f = f".cache/spin_reads/TGNOMLPG8K3S{C}{s}_knn.json"
    if os.path.exists(f):
        c = json.load(open(f)); rat.append(c["ratio_last"])
        out.append(f"  KNN s{s}: true-neighbour share per breath " + " ".join(f"{p:.3f}" for p in c["precision"]) + f" | chance {c['chance']:.3f} | last / chance {c['ratio_last']:.2f}")
out.append(f"DISCOVERS (last-breath share >= 3x chance both seeds): {'YES' if len(rat) == 2 and min(rat) >= 3 else 'NO'}")
txt = "\n".join(out); print(txt); open(".cache/rel_dsl_gq2_verdict.txt", "w").write(txt + "\n")
