"""gsm_sab_verdict.py -- SAB (ledger 2026-11-09 pin): the sabotage trick on BIND (BTC_BIND=1 BTC_SAB=0.25 / 0.5), 5k s0, against
BIND_BIND_s0 (the clean notebook).
TEST-x:   TEST step_all - BIND's (0.1267) >= +0.020 -> training on a corrupted notebook closes the gold-vs-own-commits gap
UNSEEN-x: TEST-U - BIND's (0.0931) >= +0.020
KEEP-x:   ORIG step_all - BIND's (0.6446) >= -0.050 -> the fit lever survives the sabotage. -> .cache/gsm_sab_verdict.txt"""
import json, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gsm_cj_verdict import cj

out = []
for a in ("BIND_BIND_s0", "SAB25_s0", "SAB50_s0"):
    try: j = json.load(open(f".cache/gsmb_btc_{a}.json")); d = np.load(f".cache/gsm_cj_{a}.npz")
    except FileNotFoundError as x: out.append(f"{a}: MISSING {x.filename}"); continue
    if abs(d["ok"].mean() - j["TEST_step_all"]) > 0.003: out.append(f"{a}: CJ re-score VOID"); continue
    sn, un, te, og = d["ok"][d["seen"]].mean(), d["ok"][~d["seen"]].mean(), j["TEST_step_all"], j["ORIG_step_all"]
    out.append(f"{a:13} ORIG {og:.4f} type {j['ORIG_type_ok']:.3f} | CFNUM {j['CFNUM_step_all']:.4f} CFNAME {j['CFNAME_step_all']:.4f} | "
               f"TEST {te:.4f} type {j['TEST_type_ok']:.3f} seen {sn:.4f} TEST-U {un:.4f} CJ {cj(d)[0]:+.4f}")
    if a != "BIND_BIND_s0":
        x = a.split("_")[0]
        out.append(f"  TEST-{x}: {te - 0.1267:+.4f} (>= +0.020) -> {'PASS' if te - 0.1267 >= 0.020 else 'FAIL'}")
        out.append(f"  UNSEEN-{x}: {un - 0.0931:+.4f} (>= +0.020) -> {'PASS' if un - 0.0931 >= 0.020 else 'FAIL'}")
        out.append(f"  KEEP-{x}: {og - 0.6446:+.4f} (>= -0.050) -> {'PASS' if og - 0.6446 >= -0.050 else 'FAIL'}")
txt = "\n".join(out); print(txt); open(".cache/gsm_sab_verdict.txt", "w").write(txt + "\n")
