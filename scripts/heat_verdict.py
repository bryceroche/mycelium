"""heat_verdict.py -- HEAT-ZOOM verdicts (ledger 2026-11-13 08:46 pin). STACK = BEAM4+CMOTIF TEST-U, 4 k-seeds x local / full.
HEAT-x: HEAT - banked STACK >= +0.020 both GAIN, >= -0.010 both PASS, else FAIL. HEAT-P: HEAT - HEAT-PLACEBO >= +0.020 both.
HEAT-B: HEAT-BIC - BIC >= +0.020 both RECOVERS, <= -0.010 either HURTS, else EQUIVALENT. -> .cache/heat_verdict.txt"""
import re

import numpy as np


def stack(src, suf):
    out, ts = [], []
    for k in ("", "_k1", "_k2", "_k3"):
        t = open(f".cache/gsm_ladder{'_local' if src == 'local' else ''}{k}_chain_beam4_cmotif1.0{suf}.txt").read()
        out.append(float(re.search(r"BEAM4\+CMOTIF : TEST-U ([\d.]+)", t).group(1)))
        m = re.search(r"ZOOM \w+: t\* ([\d.]+) \(grid edge (\w+)", t); ts.append(f"{m.group(1)}{'*' if m.group(2) == 'True' else ''}" if m else "-")
    return np.array(out), ts


L = []; R = {}
for name, suf in (("banked STACK", ""), ("HEAT", "_heat"), ("HEAT-PLACEBO", "_heatplacebo"), ("BIC", "_bic_law"), ("HEAT-BIC", "_bic_law_heat")):
    for src in ("local", "full"):
        R[name, src], ts = stack(src, suf)
        L.append(f"  {name:13s} {src:5s}: " + " / ".join(f"{x:.4f}" for x in R[name, src]) + f" | mean {R[name, src].mean():.4f}" + (f" | t* {' '.join(ts)}" if ts[0] != "-" else ""))
d = lambda a, b: [float((R[a, s] - R[b, s]).mean()) for s in ("local", "full")]
x = d("HEAT", "banked STACK"); p = d("HEAT", "HEAT-PLACEBO"); b = d("HEAT-BIC", "BIC")
L.append(f"HEAT-x (HEAT - banked STACK): local {x[0]:+.4f}, full {x[1]:+.4f} -> " + ("GAIN" if min(x) >= 0.020 else "PASS" if min(x) >= -0.010 else "FAIL"))
L.append(f"HEAT-P (HEAT - HEAT-PLACEBO): local {p[0]:+.4f}, full {p[1]:+.4f} -> " + ("PASS" if min(p) >= 0.020 else "FAIL"))
L.append(f"HEAT-B (HEAT-BIC - BIC): local {b[0]:+.4f}, full {b[1]:+.4f} -> " + ("RECOVERS" if min(b) >= 0.020 else "HURTS" if min(b) <= -0.010 else "EQUIVALENT"))
L.append("  (t* marked * = on the grid edge; the HEAT runs print two fits each -- chain off then chain pred; the file carries the STACK arm's)")
txt = "\n".join(L); print(txt); open(".cache/heat_verdict.txt", "w").write(txt + "\n")
