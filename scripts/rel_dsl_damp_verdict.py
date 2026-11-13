"""rel_dsl_damp_verdict.py -- DAMP + POLE verdicts (ledger 2026-11-13 12:15 pin). U3 = NOMLP+DAMP vs NOMLP (cov 0.25), U4 = NOTEBOOK+DAMP
vs NOTEBOOK (cov 0.0625); controls = the unitb tinygrad arms. Frame (10-29): C0 SHALLOW < 0.95 VOID; DEEP >= +0.03 with both seeds
positive HELPS; <= -0.02 HURTS; else EQUIVALENT (CEILING if both >= 0.97). WITNESS: the control reruns (_pole tags) must reproduce the
banked per-item DEEP exactly, else nothing banks. POLE: R^2 < 0.5 VOID; on NOMLP share |lambda| < 0.99 >= 0.5 CONTENT CONTRACTS, share
in [0.99, 1.01] >= 0.5 MARGINAL, else MIXED. -> .cache/rel_dsl_damp_verdict.txt"""
import json

import numpy as np

J = lambda t: json.load(open(f".cache/rel_dsl_{t}.json"))
out = []
w = [(t, J(t + "_pole"), J(t)) for t in ("TGNOMLP_depth2c0.25_s0", "TGNOMLPU_depth2c0.25_s0")]
ok = all(a["deep_per"] == b["deep_per"] and a["shallow_per"] == b["shallow_per"] for _, a, b in w)
out.append("WITNESS (control reruns reproduce the banked per-item reads): " + " | ".join(f"{t}: DEEP {a['deep']:.4f} vs {b['deep']:.4f}" for t, a, b in w)
           + f" -> {'PASS' if ok else 'FAIL: the DAMP build changed the control path; nothing banks'}")
out.append("tg-score agree on the control reruns (tinygrad scorer vs RB.score, DEEP per item): " + " ".join(str(a["tg_score_agree"]) for _, a, _ in w))
for name, arm, cov in (("U3", "NOMLP", "0.25"), ("U4", "NOTEBOOK", "0.0625")):
    A = [J(f"TG{arm}_depth2c{cov}_s{s}") for s in (0, 1)]; Dm = [J(f"TG{arm}D_depth2c{cov}_s{s}") for s in (0, 1)]
    out.append(f"== {name}: {arm} cov {cov}")
    for tag, R in ((arm, A), (arm + "+DAMP", Dm)):
        bd = {d: np.mean([r["deep_by_depth"][d] for r in R]) for d in R[0]["deep_by_depth"]}
        out.append(f"  {tag:14s} SHALLOW {R[0]['shallow']:.3f}/{R[1]['shallow']:.3f} MID {R[0]['mid']:.3f}/{R[1]['mid']:.3f} DEEP {R[0]['deep']:.3f}/{R[1]['deep']:.3f} | by depth "
                   + " ".join(f"d{d} {v:.3f}" for d, v in bd.items()) + f" | norms b1 {R[0]['norm_census']['all'][0]:.1f} -> {R[0]['norm_census']['all'][-1]:.1f}")
    for r in Dm: out.append(f"    learned poles s{r['seed']}: {r['pole']['learned']}")
    void = [t for t, R in ((arm, A), (arm + "+DAMP", Dm)) if np.mean([r["shallow"] for r in R]) < 0.95]
    d = [float(np.mean(np.array(u["deep_per"]) - np.array(a["deep_per"]))) for a, u in zip(A, Dm)]
    se = [float(np.std(np.array(u["deep_per"]) - np.array(a["deep_per"]), ddof=1) / np.sqrt(len(a["deep_per"]))) for a, u in zip(A, Dm)]
    m = float(np.mean(d)); ceil = min(np.mean([r["deep"] for r in A]), np.mean([r["deep"] for r in Dm])) >= 0.97
    v = ("VOID (C0: " + ", ".join(void) + ")" if void else "CEILING" if ceil else "HELPS: POLES INSIDE THE CIRCLE CARRY DEPTH" if m >= 0.03 and min(d) > 0
         else "HURTS" if m <= -0.02 else "EQUIVALENT")
    out.append(f"  DEEP {arm}+DAMP - {arm} {m:+.3f} (seeds {d[0]:+.3f} / {d[1]:+.3f}; paired SE {se[0]:.3f} / {se[1]:.3f})")
    out.append(f"{name}: {v} ({m:+.3f})")
out.append("== POLE (A from x_b+1 ~ x_b A, 256 DEEP items; |lambda| quartiles / shares inside < 0.99, on, outside > 1.01 / top 5 / R^2)")
P = [("NOMLP s0 (control)", w[0][1]), ("NOMLP+U s0", w[1][1])] + [(f"{a}+DAMP s{s}", J(f"TG{a}D_depth2c{c}_s{s}")) for a, c in (("NOMLP", "0.25"), ("NOTEBOOK", "0.0625")) for s in (0, 1)]
for t, r in P:
    p = r["pole"]; out.append(f"  {t:20s} q {'/'.join(f'{x:.3f}' for x in p['q'])} | inside {p['inside']:.3f} on {p['on']:.3f} outside {p['outside']:.3f} | top "
                              + " ".join(f"{x:.3f}" for x in p["top"]) + f" | R2 {p['r2']:.3f}")
p = w[0][1]["pole"]
pv = "VOID (R^2 < 0.5)" if p["r2"] < 0.5 else "CONTENT CONTRACTS" if p["inside"] >= 0.5 else "MARGINAL" if p["on"] >= 0.5 else "MIXED"
out.append(f"POLE-x (NOMLP control): {pv} (inside {p['inside']:.3f}, on {p['on']:.3f}, R^2 {p['r2']:.3f})")
txt = "\n".join(out); print(txt); open(".cache/rel_dsl_damp_verdict.txt", "w").write(txt + "\n")
