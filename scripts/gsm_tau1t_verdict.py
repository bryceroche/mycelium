"""gsm_tau1t_verdict.py -- TAU1T (ledger 2026-11-08 pin): the seed twin of FIT-TAU1 (5k, s0 banked by FS + s1 here).
FIT-TAU1 TWIN: ORIG step_all(TAU1) - ORIG step_all(E0) >= +0.050 on BOTH seeds -> the constant temperature is a fit lever (claimed).
CJ-TAU1: CJ(TAU1) - CJ(E0) named if >= +0.020 with the same sign on both seeds (the twin form of the CJ-DIFF frame).
UNSEEN-TAU1: TEST-U(TAU1) - TEST-U(E0) per seed (descriptive). -> .cache/gsm_tau1t_verdict.txt"""
import json, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gsm_cj_verdict import cj

ARMS = {0: ("FS_E0S_s0", "FS_TAU1_s0"), 1: ("TAU1T_E0S_s1", "TAU1T_TAU1_s1")}
out = []; R = {}
for s, (e, t) in ARMS.items():
    try:
        je, jt = (json.load(open(f".cache/gsmb_btc_{a}.json")) for a in (e, t)); de, dt = (np.load(f".cache/gsm_cj_{a}.npz") for a in (e, t))
    except FileNotFoundError as x: out.append(f"s{s}: MISSING {x.filename}"); continue
    if any(abs(d["ok"].mean() - j["TEST_step_all"]) > 0.003 for d, j in ((de, je), (dt, jt))): out.append(f"s{s}: CJ re-score VOID"); continue
    f = jt["ORIG_step_all"] - je["ORIG_step_all"]; c = cj(dt)[0] - cj(de)[0]; u = dt["ok"][~dt["seen"]].mean() - de["ok"][~de["seen"]].mean()
    R[s] = (f, c, u)
    out.append(f"s{s}: ORIG E0 {je['ORIG_step_all']:.4f} TAU1 {jt['ORIG_step_all']:.4f} (FIT {f:+.4f}) | TEST E0 {je['TEST_step_all']:.4f} "
               f"TAU1 {jt['TEST_step_all']:.4f} | TEST-U d {u:+.4f} | CJ E0 {cj(de)[0]:+.4f} TAU1 {cj(dt)[0]:+.4f} (d {c:+.4f})")
if len(R) == 2:
    out.append(f"FIT-TAU1 TWIN: {'PASS' if min(R[0][0], R[1][0]) >= 0.050 else 'FAIL'}")
    out.append(f"CJ-TAU1: {'NAMED' if min(R[0][1], R[1][1]) >= 0.020 or max(R[0][1], R[1][1]) <= -0.020 else 'none named'}")
txt = "\n".join(out); print(txt); open(".cache/gsm_tau1t_verdict.txt", "w").write(txt + "\n")
