"""rel_dsl_dial_verdict.py -- DIAL queue verdict (ledger 2026-11-13 19:09): U-H (NOTEBOOK+U - NOTEBOOK at the hard rung cov 0.03125),
R-U / R-A per arm (DIAL - U, DIAL - additive), R-C (AUROC of the derived vars' final log-radius for a correct pick >= 0.60 on all four
DIAL runs -> RADIUS = CONFIDENCE; any <= 0.55 -> NOT; else WEAK). Frame: C0 SHALLOW seed mean < 0.95 -> VOID; >= +0.03 both seeds
positive HELPS; <= -0.02 HURTS; else EQUIVALENT (CEILING if both arms >= 0.97). All U / R arms tinygrad-scored. -> .cache/rel_dsl_dial_verdict.txt"""
import json

import numpy as np

J = lambda t: json.load(open(f".cache/rel_dsl_{t}.json"))
out = []


def frame(B, X):
    d = [float(np.mean(np.array(x["deep_per"]) - np.array(b["deep_per"]))) for b, x in zip(B, X)]
    se = [float(np.std(np.array(x["deep_per"]) - np.array(b["deep_per"]), ddof=1) / np.sqrt(len(b["deep_per"]))) for b, x in zip(B, X)]
    m = float(np.mean(d)); void = [i for i, R in enumerate((B, X)) if np.mean([r["shallow"] for r in R]) < 0.95]
    ceil = min(np.mean([r["deep"] for r in B]), np.mean([r["deep"] for r in X])) >= 0.97
    v = ("VOID (C0)" if void else "HELPS" if m >= 0.03 and min(d) > 0 else "HURTS" if m <= -0.02 else "CEILING" if ceil else "EQUIVALENT")
    return f"DEEP {m:+.3f} (seeds {d[0]:+.3f} / {d[1]:+.3f}; paired SE {se[0]:.3f} / {se[1]:.3f}) -> {v}"


ARMS = {"NOMLP": ("0.25", [J("TGNOMLP_depth2c0.25_s0"), J("TGNOMLP_depth2c0.25_s1")],
                  [J("TGNOMLPU_depth2c0.25_s0_step_tg"), J("TGNOMLPU_depth2c0.25_s1_rs")]),
        "NOTEBOOK": ("0.03125", [J(f"TGNOTEBOOK_depth2c0.03125_s{s}") for s in (0, 1)], [J(f"TGNOTEBOOKU_depth2c0.03125_s{s}") for s in (0, 1)])}
aucs = []
for arm, (cov, A, U) in ARMS.items():
    R = [J(f"TG{arm}R_depth2c{cov}_s{s}") for s in (0, 1)]
    out.append(f"== {arm} cov {cov}")
    for tag, X in ((arm, A), (arm + "+U", U), (arm + "+R", R)):
        bd = {d: np.mean([r["deep_by_depth"][d] for r in X]) for d in X[0]["deep_by_depth"]}
        out.append(f"  {tag:12s} SHALLOW {X[0]['shallow']:.3f}/{X[1]['shallow']:.3f} MID {X[0]['mid']:.3f}/{X[1]['mid']:.3f} DEEP {X[0]['deep']:.3f}/{X[1]['deep']:.3f} | by depth "
                   + " ".join(f"d{d} {v:.3f}" for d, v in bd.items()))
    for s, r in enumerate(R):
        c = r["dial_census"]; aucs.append(c["auroc"])
        out.append(f"  R s{s}: sigma median per breath " + " ".join(f"{q[1]:.2f}" for q in c["sigma_q"]) + f" | final log-radius q05/25/50/75/95 "
                   + " ".join(f"{x:.2f}" for x in c["logr_q"]) + f" | gains " + " ".join(f"{g:.3f}" for g in c["gains"])
                   + f" | angles " + " ".join(f"{a:.1f}" for a in r["norm_census"]["angle"]) + f" | R-C AUROC {c['auroc']:.3f} (pos {c['n_pos']} neg {c['n_neg']})")
    if arm == "NOTEBOOK": out.append(f"  U-H NOTEBOOK+U - NOTEBOOK: {frame(A, U)}")
    out.append(f"  R-U {arm}: {frame(U, R)}")
    out.append(f"  R-A {arm}: {frame(A, R)}")
rc = "RADIUS = CONFIDENCE" if all(a >= 0.60 for a in aucs) else "NOT" if any(not (a > 0.55) for a in aucs) else "WEAK"
out.append(f"R-C: {rc} (AUROC " + " / ".join(f"{a:.3f}" for a in aucs) + ")")
txt = "\n".join(out); print(txt); open(".cache/rel_dsl_dial_verdict.txt", "w").write(txt + "\n")
