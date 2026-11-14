"""rel_dsl_graph_verdict.py -- GRAPH-ROUTED HEADS verdict (ledger 2026-11-14 09:35): G-8 / G-4 = G8 / G4 - additive per arm (NOMLP cov
0.25, NOTEBOOK cov 0.03125). Frame: C0 SHALLOW seed mean < 0.95 -> VOID; >= +0.03 both seeds positive HELPS; <= -0.02 HURTS; else
EQUIVALENT (CEILING if both arms >= 0.97). Beside: the COMM read on the new checkpoints (.cache/spin_reads), STEP-A angles. -> .cache/rel_dsl_graph_verdict.txt"""
import json, os

import numpy as np

J = lambda t: json.load(open(f".cache/rel_dsl_{t}.json"))
out = []


def frame(B, X):
    d = [float(np.mean(np.array(x["deep_per"]) - np.array(b["deep_per"]))) for b, x in zip(B, X)]
    se = [float(np.std(np.array(x["deep_per"]) - np.array(b["deep_per"]), ddof=1) / np.sqrt(len(b["deep_per"]))) for b, x in zip(B, X)]
    m = float(np.mean(d)); void = any(np.mean([r["shallow"] for r in R]) < 0.95 for R in (B, X))
    ceil = min(np.mean([r["deep"] for r in B]), np.mean([r["deep"] for r in X])) >= 0.97
    v = "VOID (C0)" if void else "HELPS" if m >= 0.03 and min(d) > 0 else "HURTS" if m <= -0.02 else "CEILING" if ceil else "EQUIVALENT"
    return f"DEEP {m:+.3f} (seeds {d[0]:+.3f} / {d[1]:+.3f}; paired SE {se[0]:.3f} / {se[1]:.3f}) -> {v}"


for arm, cov in (("NOMLP", "0.25"), ("NOTEBOOK", "0.03125")):
    A = [J(f"TG{arm}_depth2c{cov}_s{s}") for s in (0, 1)]
    out.append(f"== {arm} cov {cov}")
    G = {h: [J(f"TG{arm}G{h}_depth2c{cov}_s{s}") for s in (0, 1)] for h in (8, 4)}
    for tag, X in ((arm, A), (arm + "+G8", G[8]), (arm + "+G4", G[4])):
        bd = {d: np.mean([r["deep_by_depth"][d] for r in X]) for d in X[0]["deep_by_depth"]}
        out.append(f"  {tag:12s} SHALLOW {X[0]['shallow']:.3f}/{X[1]['shallow']:.3f} MID {X[0]['mid']:.3f}/{X[1]['mid']:.3f} DEEP {X[0]['deep']:.3f}/{X[1]['deep']:.3f} | by depth "
                   + " ".join(f"d{d} {v:.3f}" for d, v in bd.items()) + " | angles s0 " + " ".join(f"{a:.1f}" for a in X[0]["norm_census"]["angle"]))
    for h in (8, 4):
        for s in (0, 1):
            f = f".cache/spin_reads/TG{arm}G{h}_depth2c{cov}_s{s}.json"
            if os.path.exists(f):
                c = json.load(open(f)); out.append(f"  COMM G{h} s{s}: own-factor mass right {c['comm_right']:.3f} wrong {c['comm_wrong']} (uniform {c['comm_uniform']:.3f}; on vars {c['att_self_var']:.3f}) | right {c['n_pos']} wrong {c['n_neg']}")
        out.append(f"  G-{h} {arm}: {frame(A, G[h])}")
txt = "\n".join(out); print(txt); open(".cache/rel_dsl_graph_verdict.txt", "w").write(txt + "\n")
