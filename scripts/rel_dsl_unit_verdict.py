"""rel_dsl_unit_verdict.py -- UNITARY BREATH verdict (ledger 2026-11-13 pin): sphere exp-map breaths vs additive, rel-DSL hard depth
split, tinygrad. U1 = NOMLP pair at cov 0.25; U2 = NOTEBOOK pair at cov 0.0625. Frame (10-29): C0 SHALLOW seed mean < 0.95 -> VOID;
DEEP diff >= +0.03 with both seed pairs positive -> HELPS; <= -0.02 -> HURTS; else EQUIVALENT (CEILING if both arms' DEEP seed means
>= 0.97). Beside: MID, by depth, paired SE, the norm census, the banked torch reads. -> .cache/rel_dsl_unit_verdict.txt"""
import json, os

import numpy as np

J = lambda t: json.load(open(f".cache/rel_dsl_{t}.json"))
out = []
for name, arm, cov, torch_tag in (("U1", "NOMLP", "0.25", None), ("U2", "NOTEBOOK", "0.0625", "NOTEBOOK_depth2c0.0625_s{}")):
    A = [J(f"TG{arm}_depth2c{cov}_s{s}") for s in (0, 1)]; U = [J(f"TG{arm}U_depth2c{cov}_s{s}") for s in (0, 1)]
    out.append(f"== {name}: {arm} cov {cov} (train {A[0]['n_train']} structures, depth <= 2) | RESULT deep {A[0]['result']['deep']:.3f}")
    for tag, R in ((arm, A), (arm + "+U", U)):
        bd = {d: np.mean([r["deep_by_depth"][d] for r in R]) for d in R[0]["deep_by_depth"]}
        out.append(f"  {tag:12s} SHALLOW {R[0]['shallow']:.3f}/{R[1]['shallow']:.3f} MID {R[0]['mid']:.3f}/{R[1]['mid']:.3f} DEEP {R[0]['deep']:.3f}/{R[1]['deep']:.3f} | by depth "
                   + " ".join(f"d{d} {v:.3f}" for d, v in bd.items()) + f" | norms(all) b1 {R[0]['norm_census']['all'][0]:.2f} -> b{len(R[0]['norm_census']['all'])} {R[0]['norm_census']['all'][-1]:.2f}")
    if torch_tag:
        T = [J(torch_tag.format(s)) for s in (0, 1)]; out.append(f"  (banked torch {arm}: DEEP {T[0]['deep']:.3f}/{T[1]['deep']:.3f}; no bar, G0's lesson)")
    void = [t for t, R in ((arm, A), (arm + "+U", U)) if np.mean([r["shallow"] for r in R]) < 0.95]
    d = [float(np.mean(np.array(u["deep_per"]) - np.array(a["deep_per"]))) for a, u in zip(A, U)]
    se = [float(np.std(np.array(u["deep_per"]) - np.array(a["deep_per"]), ddof=1) / np.sqrt(len(a["deep_per"]))) for a, u in zip(A, U)]
    m = float(np.mean(d)); ceil = min(np.mean([r["deep"] for r in A]), np.mean([r["deep"] for r in U])) >= 0.97
    v = ("VOID (C0: " + ", ".join(void) + ")" if void else "CEILING" if ceil else
         "HELPS: NORM-PRESERVING BREATHS CARRY DEPTH" if m >= 0.03 and min(d) > 0 else "HURTS" if m <= -0.02 else "EQUIVALENT")
    out.append(f"  DEEP {arm}+U - {arm} {m:+.3f} (seeds {d[0]:+.3f} / {d[1]:+.3f}; paired SE {se[0]:.3f} / {se[1]:.3f}) | MID "
               f"{np.mean([u['mid'] - a['mid'] for a, u in zip(A, U)]):+.3f}")
    out.append(f"{name}: {v} ({m:+.3f})")
txt = "\n".join(out); print(txt); open(".cache/rel_dsl_unit_verdict.txt", "w").write(txt + "\n")
