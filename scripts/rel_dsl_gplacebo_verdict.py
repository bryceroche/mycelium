"""rel_dsl_gplacebo_verdict.py -- GRAPH PLACEBO verdict (ledger 2026-11-14 11:51), NOMLP cov 0.25, seed means: C0 SHALLOW < 0.95 -> VOID;
EDGES iff G8P - additive <= +0.03; SPARSITY iff G8 - G8P < 0.03; else MIXED (quote the split). -> .cache/rel_dsl_gplacebo_verdict.txt"""
import json, os

import numpy as np

J = lambda t: json.load(open(f".cache/rel_dsl_{t}.json"))
R = {k: [J(f"TGNOMLP{k}_depth2c0.25_s{s}") for s in (0, 1)] for k in ("", "G8", "G8P")}
out = []
for k, X in R.items():
    bd = {d: np.mean([r["deep_by_depth"][d] for r in X]) for d in X[0]["deep_by_depth"]}
    out.append(f"  NOMLP{('+' + k) if k else '':6s} SHALLOW {X[0]['shallow']:.3f}/{X[1]['shallow']:.3f} MID {X[0]['mid']:.3f}/{X[1]['mid']:.3f} DEEP {X[0]['deep']:.3f}/{X[1]['deep']:.3f} | by depth "
               + " ".join(f"d{d} {v:.3f}" for d, v in bd.items()))
d = lambda a, b: [float(np.mean(np.array(x["deep_per"]) - np.array(y["deep_per"]))) for x, y in zip(R[a], R[b])]
pa, gp = d("G8P", ""), d("G8", "G8P")
for s in (0, 1):
    f = f".cache/spin_reads/TGNOMLPG8P_depth2c0.25_s{s}.json"
    if os.path.exists(f):
        c = json.load(open(f)); out.append(f"  COMM G8P s{s}: mass on the TRUE own factors {c['comm_right']:.3f} (uniform {c['comm_uniform']:.3f}; on vars {c['att_self_var']:.3f})")
void = any(np.mean([r["shallow"] for r in X]) < 0.95 for X in R.values())
v = "VOID (C0)" if void else "EDGES" if np.mean(pa) <= 0.03 else "SPARSITY" if np.mean(gp) < 0.03 else "MIXED"
out.append(f"G-P: G8P - additive {np.mean(pa):+.3f} (seeds {pa[0]:+.3f} / {pa[1]:+.3f}); G8 - G8P {np.mean(gp):+.3f} (seeds {gp[0]:+.3f} / {gp[1]:+.3f}) -> {v}")
txt = "\n".join(out); print(txt); open(".cache/rel_dsl_gplacebo_verdict.txt", "w").write(txt + "\n")
