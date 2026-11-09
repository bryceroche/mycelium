"""gsm_brel_verdict.py -- BREL (ledger 2026-11-09 pin): binding on the T9 OFFSET gears only (BTC_BIND=1 BTC_BINDREL=1), 5k s0,
against BIND (BIND_BIND_s0: absolute position bound too) and E0 (FS_E0S_s0).
SEEN-BREL:   TEST seen-skeleton step_all - BIND's (0.1757) >= +0.020 -> the seen drop was absolute-position binding
FIT-BREL:    ORIG step_all - E0 (0.4858) >= +0.050 -> binding's fit survives without the absolute position
UNSEEN-BREL: TEST-U - BIND's (0.0931) >= +0.020. -> .cache/gsm_brel_verdict.txt"""
import json, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gsm_cj_verdict import cj

out = []
for a in ("FS_E0S_s0", "BIND_BIND_s0", "BREL_s0"):
    try: j = json.load(open(f".cache/gsmb_btc_{a}.json")); d = np.load(f".cache/gsm_cj_{a}.npz")
    except FileNotFoundError as x: out.append(f"{a}: MISSING {x.filename}"); continue
    if abs(d["ok"].mean() - j["TEST_step_all"]) > 0.003: out.append(f"{a}: CJ re-score VOID"); continue
    sn, un = d["ok"][d["seen"]].mean(), d["ok"][~d["seen"]].mean()
    out.append(f"{a:13} ORIG {j['ORIG_step_all']:.4f} | CFNUM {j['CFNUM_step_all']:.4f} CFNAME {j['CFNAME_step_all']:.4f} | TEST "
               f"{j['TEST_step_all']:.4f} seen {sn:.4f} TEST-U {un:.4f} CJ {cj(d)[0]:+.4f}")
    if a == "BREL_s0":
        out.append(f"  SEEN-BREL: {sn - 0.1757:+.4f} (>= +0.020) -> {'PASS' if sn - 0.1757 >= 0.020 else 'FAIL'}")
        out.append(f"  FIT-BREL: {j['ORIG_step_all'] - 0.4858:+.4f} (>= +0.050) -> {'PASS' if j['ORIG_step_all'] - 0.4858 >= 0.050 else 'FAIL'}")
        out.append(f"  UNSEEN-BREL: {un - 0.0931:+.4f} (>= +0.020) -> {'PASS' if un - 0.0931 >= 0.020 else 'FAIL'}")
txt = "\n".join(out); print(txt); open(".cache/gsm_brel_verdict.txt", "w").write(txt + "\n")
