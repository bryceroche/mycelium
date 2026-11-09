"""gsm_bindt_verdict.py -- BINDT (ledger 2026-11-09 pin): the seed twin of FIT-BIND at 5k (s0: BIND_BIND_s0 vs FS_E0S_s0, banked;
s1: BINDT_BIND_s1 vs TAU1T_E0S_s1). FIT-BIND TWIN: ORIG step_all (BIND - E0) >= +0.050 on BOTH seeds -> claimed.
CF-BIND TWIN: CFNUM and CFNAME step_all (BIND - E0) >= +0.020 on BOTH seeds -> claimed. TEST-U d, CJ d descriptive. -> .cache/gsm_bindt_verdict.txt"""
import json, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gsm_cj_verdict import cj

ARMS = {0: ("FS_E0S_s0", "BIND_BIND_s0"), 1: ("TAU1T_E0S_s1", "BINDT_BIND_s1")}
out = []; R = {}
for s, (e, t) in ARMS.items():
    try:
        je, jt = (json.load(open(f".cache/gsmb_btc_{a}.json")) for a in (e, t)); de, dt = (np.load(f".cache/gsm_cj_{a}.npz") for a in (e, t))
    except FileNotFoundError as x: out.append(f"s{s}: MISSING {x.filename}"); continue
    if any(abs(d["ok"].mean() - j["TEST_step_all"]) > 0.003 for d, j in ((de, je), (dt, jt))): out.append(f"s{s}: CJ re-score VOID"); continue
    d_ = {k: jt[f"{k}_step_all"] - je[f"{k}_step_all"] for k in ("ORIG", "CFNUM", "CFNAME", "TEST")}
    u = dt["ok"][~dt["seen"]].mean() - de["ok"][~de["seen"]].mean(); c = cj(dt)[0] - cj(de)[0]; R[s] = d_
    out.append(f"s{s}: ORIG {je['ORIG_step_all']:.4f} -> {jt['ORIG_step_all']:.4f} ({d_['ORIG']:+.4f}) | CFNUM {d_['CFNUM']:+.4f} CFNAME "
               f"{d_['CFNAME']:+.4f} | TEST {d_['TEST']:+.4f} TEST-U {u:+.4f} CJ {c:+.4f}")
if len(R) == 2:
    out.append(f"FIT-BIND TWIN: {'PASS' if min(R[0]['ORIG'], R[1]['ORIG']) >= 0.050 else 'FAIL'}")
    out.append(f"CF-BIND TWIN: {'PASS' if min(R[s][k] for s in R for k in ('CFNUM', 'CFNAME')) >= 0.020 else 'FAIL'}")
txt = "\n".join(out); print(txt); open(".cache/gsm_bindt_verdict.txt", "w").write(txt + "\n")
