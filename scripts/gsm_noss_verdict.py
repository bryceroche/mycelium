"""gsm_noss_verdict.py -- NOSS (ledger 2026-11-08 pin): BTC2 with no slot-self road vs BTC2_s0, read by the combination split.
NOSS-ROAD: CJ(NOSS_s0) <= CJ(BTC2_s0) - 0.040 -> the slot-self road is a template road (a seed twin before any claim).
NOSS-UNSEEN: TEST-U (unseen-skeleton step_all) - BTC2_s0's >= +0.020 -> a lead. Beside: TEST step, seen, ORIG/CF from the json.
-> .cache/gsm_noss_verdict.txt"""
import json, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gsm_cj_verdict import cj

R = {}
for a in ("BTC2_s0", "NOSS_s0"):
    d = np.load(f".cache/gsm_cj_{a}.npz"); assert abs(d["ok"].mean() - float(d["banked"])) <= 0.003, f"{a} VOID"
    c, _ = cj(d); j = json.load(open(f".cache/gsmb_btc_{a}.json"))
    R[a] = dict(cj=c, u=d["ok"][~d["seen"]].mean(), s=d["ok"][d["seen"]].mean(), t=d["ok"].mean(), orig=j["ORIG_step_all"],
                cfn=j["CFNUM_step_all"], cfm=j["CFNAME_step_all"], par=j["params"])
out = [f"{a}: TEST {r['t']:.4f} | seen {r['s']:.4f} unseen (TEST-U) {r['u']:.4f} | CJ {r['cj']:+.4f} | ORIG {r['orig']:.4f} CFNUM "
       f"{r['cfn']:.4f} CFNAME {r['cfm']:.4f} | params {r['par']}" for a, r in R.items()]
b, n = R["BTC2_s0"], R["NOSS_s0"]
out.append(f"NOSS-ROAD: CJ {n['cj']:+.4f} vs BTC2 {b['cj']:+.4f} - 0.040 = {b['cj'] - 0.040:+.4f} -> {'PASS' if n['cj'] <= b['cj'] - 0.040 else 'FAIL'}")
out.append(f"NOSS-UNSEEN: TEST-U {n['u']:.4f} - {b['u']:.4f} = {n['u'] - b['u']:+.4f} (>= +0.020) -> {'PASS' if n['u'] - b['u'] >= 0.020 else 'FAIL'}")
txt = "\n".join(out); print(txt); open(".cache/gsm_noss_verdict.txt", "w").write(txt + "\n")
