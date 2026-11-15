"""rel_dsl_soft_verdict.py -- SOFT-BIAS + WITNESS-LOCK verdict (ledger 2026-11-15), NOMLP cov 0.25, the unitary breath, two seeds, the
fixture frame (C0 VOID; >= +0.03 both seeds positive HELPS; <= -0.02 HURTS; else EQUIVALENT / CEILING).
SOFT-SAFE iff UG8N10B4 - UG8N10 >= +0.03 both seeds positive; SOFT-KEEPS iff UG8B4 - U HELPS with retention >= 0.5 of UG8 - U;
LOCK-HELPS iff UG4W10 - UG4N10 >= +0.03 both seeds positive; WITNESS CARRIES iff UG4W10 - U HELPS with retention >= 0.5 of UG4 - U.
-> .cache/rel_dsl_soft_verdict.txt"""
import json

import numpy as np

J = lambda t: json.load(open(f".cache/rel_dsl_{t}.json"))
C = "_depth2c0.25_s"
R = {k: [J(f"TGNOMLP{k}{C}{s}") for s in (0, 1)] for k in ("U", "UG8", "UG4", "UG8B4", "UG8N10", "UG8N10B4", "UG4N10", "UG4W10")}
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
    return m, v, f"{m:+.3f} (seeds {d[0]:+.3f} / {d[1]:+.3f}) -> {v}"


def yes(a, b):
    d = diff(a, b); return "YES" if np.mean(d) >= 0.03 and min(d) > 0 else "NO", f"{np.mean(d):+.3f} (seeds {d[0]:+.3f} / {d[1]:+.3f})"


def keeps(a, base):
    lift = frame(base, "U")[0]; m, v, t = frame(a, "U")
    return f"{a} - U: {t} | retention of {base}'s {lift:+.3f}: {m / lift:.2f}", "YES" if v == "HELPS" and m / lift >= 0.5 else "NO"


y, t = yes("UG8N10B4", "UG8N10"); out.append(f"SOFT-SAFE (UG8N10B4 - UG8N10): {t} -> {y} | beside UG8N10B4 - U {frame('UG8N10B4', 'U')[2]} | UG8N10 - U {frame('UG8N10', 'U')[2]}")
t, y = keeps("UG8B4", "UG8"); out.append(f"SOFT-KEEPS: {t} -> {y}")
y, t = yes("UG4W10", "UG4N10"); out.append(f"LOCK-HELPS (UG4W10 - UG4N10): {t} -> {y} | beside UG4N10 - U {frame('UG4N10', 'U')[2]}")
t, y = keeps("UG4W10", "UG4"); out.append(f"WITNESS CARRIES: {t} -> {y}")
txt = "\n".join(out); print(txt); open(".cache/rel_dsl_soft_verdict.txt", "w").write(txt + "\n")
