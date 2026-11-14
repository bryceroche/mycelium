"""rel_dsl_su_verdict2.py -- (ledger 11-13 16:15) the STEP queue + the UNITARY re-read on CORRECTED scores (the tinygrad forward the
unit arms trained with). Witnesses: the _step reruns reproduce the banked per-item DEEP reads (torch-scored, pre-fix); the U s1 reruns'
deep_per_torch reproduce the banked; the control _step checkpoints' tinygrad rescore equals the banked control reads (scorer self-check).
Frames unchanged: UNITARY (12:15 / 10-29: C0 SHALLOW seed mean < 0.95 -> VOID; >= +0.03 both positive HELPS; <= -0.02 HURTS; else
EQUIVALENT / CEILING >= 0.97); STEP-A (control 9->10 / 1->2 <= 0.5 both arms -> COOLS); S-REC / S-PAR (15:12). -> .cache/rel_dsl_su_verdict2.txt"""
import json

import numpy as np

J = lambda t: json.load(open(f".cache/rel_dsl_{t}.json"))
ARMS = (("NOMLP", "0.25"), ("NOTEBOOK", "0.0625"))
out, ok = [], True
for arm, cov in ARMS:
    c = f"TG{arm}_depth2c{cov}_s"; u = f"TG{arm}U_depth2c{cov}_s"
    for name, a, b in ((c + "0_step", J(c + "0_step")["deep_per"], J(c + "0")["deep_per"]), (u + "0_step", J(u + "0_step")["deep_per"], J(u + "0")["deep_per"]),
                       (u + "1_rs", J(u + "1_rs")["deep_per_torch"], J(u + "1")["deep_per"]),
                       (c + "0_step (tinygrad rescore = banked control)", J(c + "0_step_tg")["deep_per"], J(c + "0")["deep_per"])):
        ok &= a == b; out.append(f"witness {name}: {'exact' if a == b else 'MISMATCH'}")
out.append(f"WITNESS: {'PASS' if ok else 'FAIL -- nothing below banks'}")
out.append(open(".cache/step_r.txt").read().strip().splitlines()[-1])
cool = []
for arm, cov in ARMS:
    for k in ("", "U"):
        a = J(f"TG{arm}{k}_depth2c{cov}_s0_step")["norm_census"]["angle"]; r = a[9] / a[1]
        if not k: cool.append(r <= 0.5)
        out.append(f"angles {arm}{k:1s} (deg, 0->1 .. 9->10): " + " ".join(f"{x:.2f}" for x in a) + f" | 9->10 / 1->2 {r:.3f}" + (" (beside)" if k else ""))
out.append(f"STEP-A: {'COOLS' if all(cool) else 'NO-COOL'}")


def frame(B, X, hi, lo, eq, ceil_ok):
    d = [float(np.mean(np.array(x["deep_per"]) - np.array(b["deep_per"]))) for b, x in zip(B, X)]
    se = [float(np.std(np.array(x["deep_per"]) - np.array(b["deep_per"]), ddof=1) / np.sqrt(len(b["deep_per"]))) for b, x in zip(B, X)]
    m = float(np.mean(d)); ceil = ceil_ok and min(np.mean([r["deep"] for r in B]), np.mean([r["deep"] for r in X])) >= 0.97
    v = hi if m >= 0.03 and min(d) > 0 else lo if m <= -0.02 else "CEILING" if ceil else eq
    return v, f"DEEP {m:+.3f} (seeds {d[0]:+.3f} / {d[1]:+.3f}; paired SE {se[0]:.3f} / {se[1]:.3f})"


for (arm, cov), un in zip(ARMS, ("U1", "U2")):
    A = [J(f"TG{arm}_depth2c{cov}_s{s}") for s in (0, 1)]
    U = [J(f"TG{arm}U_depth2c{cov}_s0_step_tg"), J(f"TG{arm}U_depth2c{cov}_s1_rs")]
    S = [J(f"TG{arm}S_depth2c{cov}_s{s}_tg") for s in (0, 1)]
    out.append(f"== {arm} cov {cov} (U and SU scored on the tinygrad forward)")
    for tag, R in ((arm, A), (arm + "+U", U), (arm + "+SU", S)):
        bd = {d: np.mean([r["deep_by_depth"][d] for r in R]) for d in R[0]["deep_by_depth"]}
        out.append(f"  {tag:12s} SHALLOW {R[0]['shallow']:.3f}/{R[1]['shallow']:.3f} MID {R[0]['mid']:.3f}/{R[1]['mid']:.3f} DEEP {R[0]['deep']:.3f}/{R[1]['deep']:.3f} | by depth "
                   + " ".join(f"d{d} {v:.3f}" for d, v in bd.items()))
    out.append(f"  SU s0 angles: " + " ".join(f"{x:.2f}" for x in J(f"TG{arm}S_depth2c{cov}_s0")["norm_census"]["angle"]))
    vu = [t for t, R in ((arm, A), (arm + "+U", U)) if np.mean([r["shallow"] for r in R]) < 0.95]
    v, txt = frame(A, U, "HELPS", "HURTS", "EQUIVALENT", True)
    out.append(f"  {un} (re-read) {arm}+U - {arm}: {txt} -> {'VOID (C0: ' + ', '.join(vu) + ')' if vu else v}")
    c0 = np.mean([r["shallow"] for r in S]) < 0.95
    for name, B, hi, lo, eq, co in (("S-REC", U, "RECOVERS", "HURTS", "NONE", False), ("S-PAR", A, "HELPS", "SHORT", "EQUIVALENT", True)):
        v, txt = frame(B, S, hi, lo, eq, co); out.append(f"  {name} {arm}: {txt} -> {'VOID (C0)' if c0 else v}")
txt = "\n".join(out); print(txt); open(".cache/rel_dsl_su_verdict2.txt", "w").write(txt + "\n")
