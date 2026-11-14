"""rel_dsl_gqueue_verdict.py -- THE ROUTING QUEUE verdict (ledger 2026-11-14 13:17), NOMLP cov 0.25, two seeds. Fixture frame per bar:
C0 SHALLOW seed mean < 0.95 -> VOID; >= +0.03 with both seeds positive -> HELPS (KEEPS for GN); <= -0.02 -> HURTS; else EQUIVALENT
(CEILING if both >= 0.97). GU = UG8 - U; GN10 / GN30 = G8N{r} - additive (+ retention of G8's lift); GK = G8K3 - additive; census bar
DISCOVERS iff the KNN picks' true-neighbour share at the last breath >= 3x chance on both seeds. -> .cache/rel_dsl_gqueue_verdict.txt"""
import json, os

import numpy as np

J = lambda t: json.load(open(f".cache/rel_dsl_{t}.json"))
C = "_depth2c0.25_s"
R = {"additive": [J(f"TGNOMLP{C}0"), J(f"TGNOMLP{C}1")], "U": [J(f"TGNOMLPU{C}0_step_tg"), J(f"TGNOMLPU{C}1_rs")],
     **{k: [J(f"TGNOMLP{k}{C}{s}") for s in (0, 1)] for k in ("G8", "UG8", "G8N10", "G8N30", "G8K3")}}
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


out.append(f"GU  (UG8 - U):           {frame('UG8', 'U')[1]}   | beside UG8 - G8 {frame('UG8', 'G8')[1]}")
lift = float(np.mean(diff("G8", "additive")))
for r in (10, 30):
    m, t = frame(f"G8N{r}", "additive", "KEEPS"); out.append(f"GN{r} (G8N{r} - additive): {t} | retention of G8's +{lift:.3f}: {m / lift:.2f}")
out.append(f"GK  (G8K3 - additive):   {frame('G8K3', 'additive')[1]}   | beside G8K3 - G8 {frame('G8K3', 'G8')[1]}")
rat = []
for s in (0, 1):
    f = f".cache/spin_reads/TGNOMLPG8K3{C}{s}_knn.json"
    if os.path.exists(f):
        c = json.load(open(f)); rat.append(c["ratio_last"])
        out.append(f"  KNN s{s}: true-neighbour share per breath " + " ".join(f"{p:.3f}" for p in c["precision"]) + f" | chance {c['chance']:.3f} | last / chance {c['ratio_last']:.2f}")
out.append(f"DISCOVERS (census bar, last-breath share >= 3x chance both seeds): {'YES' if len(rat) == 2 and min(rat) >= 3 else 'NO'}")
for k in ("UG8", "G8N10", "G8N30", "G8K3"):
    for s in (0, 1):
        f = f".cache/spin_reads/TGNOMLP{k}{C}{s}.json"
        if os.path.exists(f):
            c = json.load(open(f)); out.append(f"  COMM {k} s{s}: mass on the TRUE own factors {c['comm_right']:.3f} (uniform {c['comm_uniform']:.3f})")
txt = "\n".join(out); print(txt); open(".cache/rel_dsl_gqueue_verdict.txt", "w").write(txt + "\n")
