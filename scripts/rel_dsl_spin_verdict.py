"""rel_dsl_spin_verdict.py -- KEEP-SPINNING queue verdict (ledger 2026-11-14 09:00): Q (relative dial) and B (bounded dial) vs R (the
19:09 dial), U, additive, per arm (NOMLP cov 0.25, NOTEBOOK cov 0.03125); Q-C. Frame: C0 SHALLOW seed mean < 0.95 -> VOID; >= +0.03 both
seeds positive HELPS; <= -0.02 HURTS; else EQUIVALENT (CEILING if both arms >= 0.97). -> .cache/rel_dsl_spin_verdict.txt"""
import json

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


ARMS = {"NOMLP": ("0.25", [J("TGNOMLP_depth2c0.25_s0"), J("TGNOMLP_depth2c0.25_s1")], [J("TGNOMLPU_depth2c0.25_s0_step_tg"), J("TGNOMLPU_depth2c0.25_s1_rs")]),
        "NOTEBOOK": ("0.03125", [J(f"TGNOTEBOOK_depth2c0.03125_s{s}") for s in (0, 1)], [J(f"TGNOTEBOOKU_depth2c0.03125_s{s}") for s in (0, 1)])}
aucs = []
for arm, (cov, A, U) in ARMS.items():
    R, Q, Bd = ([J(f"TG{arm}{k}_depth2c{cov}_s{s}") for s in (0, 1)] for k in "RQB")
    out.append(f"== {arm} cov {cov}")
    for tag, X in ((arm, A), (arm + "+U", U), (arm + "+R", R), (arm + "+Q", Q), (arm + "+B", Bd)):
        bd = {d: np.mean([r["deep_by_depth"][d] for r in X]) for d in X[0]["deep_by_depth"]}
        out.append(f"  {tag:12s} SHALLOW {X[0]['shallow']:.3f}/{X[1]['shallow']:.3f} MID {X[0]['mid']:.3f}/{X[1]['mid']:.3f} DEEP {X[0]['deep']:.3f}/{X[1]['deep']:.3f} | by depth "
                   + " ".join(f"d{d} {v:.3f}" for d, v in bd.items()))
    for k, X in (("Q", Q), ("B", Bd)):
        for s, r in enumerate(X):
            c = r["dial_census"]
            if k == "Q" and c["n_neg"] >= 20: aucs.append(c["auroc"])
            out.append(f"  {k} s{s}: sigma median per breath " + " ".join(f"{q[1]:.2f}" for q in c["sigma_q"]) + " | final log-radius q05/50/95 "
                       + " ".join(f"{c['logr_q'][i]:.2f}" for i in (0, 2, 4)) + " | angles " + " ".join(f"{a:.1f}" for a in r["norm_census"]["angle"])
                       + f" | AUROC {c['auroc']:.3f} (pos {c['n_pos']} neg {c['n_neg']})")
    for name, B_, X in (("Q-R", R, Q), ("Q-U", U, Q), ("Q-A", A, Q), ("B-R", R, Bd)): out.append(f"  {name} {arm}: {frame(B_, X)}")
qc = ("NO QUALIFYING RUN" if not aucs else "RADIUS = CONFIDENCE" if all(a >= 0.60 for a in aucs) else "NOT" if any(not (a > 0.55) for a in aucs) else "WEAK")
out.append(f"Q-C: {qc} (AUROC on runs with >= 20 wrong picks: " + " / ".join(f"{a:.3f}" for a in aucs) + ")")
txt = "\n".join(out); print(txt); open(".cache/rel_dsl_spin_verdict.txt", "w").write(txt + "\n")
