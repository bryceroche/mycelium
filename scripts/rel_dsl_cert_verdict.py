"""rel_dsl_cert_verdict.py -- CERTIFIED-EDGE ROUTING verdict (ledger 2026-11-15), NOMLP cov 0.25, the unitary breath, two seeds, the fixture
frame (C0 VOID; >= +0.03 both seeds positive HELPS; <= -0.02 HURTS; else EQUIVALENT / CEILING). CERT = UG4C50 - U (half the true edges
certified, the rest MISSING; four routed heads, four free); retention = CERT / (UG4 - U); PARTIAL CARRIES iff CERT HELPS and retention >= 0.5;
FREE-NEEDED iff UG4C50 - UG8C50 >= +0.03 both seeds positive. -> .cache/rel_dsl_cert_verdict.txt"""
import json

import numpy as np

J = lambda t: json.load(open(f".cache/rel_dsl_{t}.json"))
C = "_depth2c0.25_s"
R = {"U": [J(f"TGNOMLPU{C}0_step_tg"), J(f"TGNOMLPU{C}1_rs")],   # the tinygrad-scored unitary control (the 17:20 rescore; the plain TGNOMLPU json is torch-scored, struck 11-13)
     **{k: [J(f"TGNOMLP{k}{C}{s}") for s in (0, 1)] for k in ("UG8", "UG4", "UG4C50", "UG8C50")}}
out = []
for k, X in R.items():
    bd = {d: np.mean([r["deep_by_depth"][d] for r in X]) for d in X[0]["deep_by_depth"]}
    out.append(f"  {k:7s} SHALLOW {X[0]['shallow']:.3f}/{X[1]['shallow']:.3f} MID {X[0]['mid']:.3f}/{X[1]['mid']:.3f} DEEP {X[0]['deep']:.3f}/{X[1]['deep']:.3f} | by depth "
               + " ".join(f"d{d} {v:.3f}" for d, v in bd.items()))


def diff(a, b):
    return [float(np.mean(np.array(x["deep_per"]) - np.array(y["deep_per"]))) for x, y in zip(R[a], R[b])]


def frame(a, b, win="HELPS"):
    d = diff(a, b); m = float(np.mean(d)); void = any(np.mean([r["shallow"] for r in R[k]]) < 0.95 for k in (a, b))
    ceil = min(np.mean([r["deep"] for r in R[a]]), np.mean([r["deep"] for r in R[b]])) >= 0.97
    v = "VOID (C0)" if void else win if m >= 0.03 and min(d) > 0 else "HURTS" if m <= -0.02 else "CEILING" if ceil else "EQUIVALENT"
    return m, v, f"{m:+.3f} (seeds {d[0]:+.3f} / {d[1]:+.3f}) -> {v}"


lift = frame("UG4", "U")[0]; m, v, t = frame("UG4C50", "U")
out.append(f"UG4 - U (the full-graph base): {frame('UG4', 'U')[2]} | beside UG4 - UG8 {frame('UG4', 'UG8')[2]}")
out.append(f"CERT (UG4C50 - U): {t} | retention of UG4's {lift:+.3f}: {m / lift:.2f}")
out.append(f"PARTIAL CARRIES (CERT HELPS and retention >= 0.5): {'YES' if v == 'HELPS' and m / lift >= 0.5 else 'NO'}")
d = diff("UG4C50", "UG8C50"); out.append(f"FREE-NEEDED (UG4C50 - UG8C50 >= +0.03 both seeds positive): {'YES' if np.mean(d) >= 0.03 and min(d) > 0 else 'NO'} "
                                         f"({np.mean(d):+.3f}; seeds {d[0]:+.3f} / {d[1]:+.3f}) | beside UG8C50 - U {frame('UG8C50', 'U')[2]}")
txt = "\n".join(out); print(txt); open(".cache/rel_dsl_cert_verdict.txt", "w").write(txt + "\n")
