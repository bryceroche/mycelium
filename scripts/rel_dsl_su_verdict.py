"""rel_dsl_su_verdict.py -- STEP queue verdict (ledger 2026-11-13 15:12): the witness (the _step reruns reproduce the banked per-item
DEEP reads exactly, or nothing banks), STEP-R (.cache/step_r.txt), STEP-A (control angle(9->10) / angle(1->2) <= 0.5 on both arms ->
COOLS), SCHEDULED UNITARY: C0 (SU SHALLOW seed mean < 0.95 -> VOID); S-REC = SU - U (>= +0.03 both seeds positive -> RECOVERS; <= -0.02
-> HURTS; else NONE); S-PAR = SU - control (>= +0.03 both positive -> HELPS; <= -0.02 -> SHORT; else EQUIVALENT, CEILING if both arms
>= 0.97). -> .cache/rel_dsl_su_verdict.txt"""
import json

import numpy as np

J = lambda t: json.load(open(f".cache/rel_dsl_{t}.json"))
ARMS = (("NOMLP", "0.25"), ("NOTEBOOK", "0.0625"))
out, ok = [], True
for arm, cov in ARMS:
    for k in ("", "U"):
        t = f"TG{arm}{k}_depth2c{cov}_s0"; same = J(t + "_step")["deep_per"] == J(t)["deep_per"]; ok &= same
        out.append(f"witness {t}: DEEP {J(t + '_step')['deep']:.4f} vs banked {J(t)['deep']:.4f} -> {'exact' if same else 'MISMATCH'}")
out.append(f"WITNESS: {'PASS' if ok else 'FAIL -- nothing below banks'}")
out.append(open(".cache/step_r.txt").read().strip().splitlines()[-1])
cool = []
for arm, cov in ARMS:
    for k in ("", "U"):
        a = J(f"TG{arm}{k}_depth2c{cov}_s0_step")["norm_census"]["angle"]; r = a[9] / a[1]
        if not k: cool.append(r <= 0.5)
        out.append(f"angles {arm}{k:1s} (deg, breath 0->1 .. 9->10): " + " ".join(f"{x:.2f}" for x in a) + f" | 9->10 / 1->2 {r:.3f}" + ("" if not k else " (beside)"))
out.append(f"STEP-A: {'COOLS' if all(cool) else 'NO-COOL'}")
for arm, cov in ARMS:
    A, U, S = ([J(f"TG{arm}{k}_depth2c{cov}_s{s}") for s in (0, 1)] for k in ("", "U", "S"))
    out.append(f"== {arm} cov {cov}")
    for tag, R in ((arm, A), (arm + "+U", U), (arm + "+SU", S)):
        bd = {d: np.mean([r["deep_by_depth"][d] for r in R]) for d in R[0]["deep_by_depth"]}
        out.append(f"  {tag:12s} SHALLOW {R[0]['shallow']:.3f}/{R[1]['shallow']:.3f} MID {R[0]['mid']:.3f}/{R[1]['mid']:.3f} DEEP {R[0]['deep']:.3f}/{R[1]['deep']:.3f} | by depth "
                   + " ".join(f"d{d} {v:.3f}" for d, v in bd.items()))
    a = S[0]["norm_census"].get("angle")
    if a: out.append(f"  SU s0 angles: " + " ".join(f"{x:.2f}" for x in a))
    c0 = np.mean([r["shallow"] for r in S]) < 0.95
    for name, B, hi, lo in (("S-REC", U, "RECOVERS", "HURTS"), ("S-PAR", A, "HELPS", "SHORT")):
        d = [float(np.mean(np.array(s["deep_per"]) - np.array(b["deep_per"]))) for b, s in zip(B, S)]
        se = [float(np.std(np.array(s["deep_per"]) - np.array(b["deep_per"]), ddof=1) / np.sqrt(len(b["deep_per"]))) for b, s in zip(B, S)]
        m = float(np.mean(d)); ceil = name == "S-PAR" and min(np.mean([r["deep"] for r in B]), np.mean([r["deep"] for r in S])) >= 0.97
        v = "VOID (C0)" if c0 else hi if m >= 0.03 and min(d) > 0 else lo if m <= -0.02 else "CEILING" if ceil else ("EQUIVALENT" if name == "S-PAR" else "NONE")
        out.append(f"  {name} {arm}: DEEP {m:+.3f} (seeds {d[0]:+.3f} / {d[1]:+.3f}; paired SE {se[0]:.3f} / {se[1]:.3f}) -> {v}")
txt = "\n".join(out); print(txt); open(".cache/rel_dsl_su_verdict.txt", "w").write(txt + "\n")
