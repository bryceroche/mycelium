"""gsm_macro_verdict.py -- MACRO1 (ledger 2026-11-09 pin): the op plan, coarse to fine (BTC_MACRO=1) on BIND (BTC_BIND=1), 5k s0, against
BIND_BIND_s0. MACRO = + the sabotage (BTC_SAB=0.25 types, BTC_SABOP=0.5 ops: NCM's decreasing rates); MACRO0 = no sabotage.
UNSEEN-x: TEST-U - BIND's (0.0931) >= +0.020 -> the plan level carries unseen skeletons
TEST-x:   TEST step_all - BIND's (0.1267) >= +0.020
KEEP-x:   ORIG step_all - BIND's (0.6446) >= -0.050
SAB-ATTR: TEST-U (MACRO - MACRO0), NAMED if >= +0.020 (descriptive). TEST step_op / type_ok descriptive. -> .cache/gsm_macro_verdict.txt"""
import json, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gsm_cj_verdict import cj

out = []; U = {}
for a in ("BIND_BIND_s0", "MACRO_s0", "MACRO0_s0"):
    try: j = json.load(open(f".cache/gsmb_btc_{a}.json")); d = np.load(f".cache/gsm_cj_{a}.npz")
    except FileNotFoundError as x: out.append(f"{a}: MISSING {x.filename}"); continue
    if abs(d["ok"].mean() - j["TEST_step_all"]) > 0.003: out.append(f"{a}: CJ re-score VOID"); continue
    sn, un, te, og = d["ok"][d["seen"]].mean(), d["ok"][~d["seen"]].mean(), j["TEST_step_all"], j["ORIG_step_all"]; U[a] = un
    out.append(f"{a:13} ORIG {og:.4f} op {j['ORIG_step_op']:.3f} type {j['ORIG_type_ok']:.3f} | CFNUM {j['CFNUM_step_all']:.4f} CFNAME "
               f"{j['CFNAME_step_all']:.4f} | TEST {te:.4f} op {j['TEST_step_op']:.3f} type {j['TEST_type_ok']:.3f} ptr {j['TEST_step_ptr']:.3f} "
               f"seen {sn:.4f} TEST-U {un:.4f} CJ {cj(d)[0]:+.4f}")
    if a != "BIND_BIND_s0":
        x = a.split("_")[0]
        out.append(f"  UNSEEN-{x}: {un - 0.0931:+.4f} (>= +0.020) -> {'PASS' if un - 0.0931 >= 0.020 else 'FAIL'}")
        out.append(f"  TEST-{x}: {te - 0.1267:+.4f} (>= +0.020) -> {'PASS' if te - 0.1267 >= 0.020 else 'FAIL'}")
        out.append(f"  KEEP-{x}: {og - 0.6446:+.4f} (>= -0.050) -> {'PASS' if og - 0.6446 >= -0.050 else 'FAIL'}")
if "MACRO_s0" in U and "MACRO0_s0" in U:
    dd = U["MACRO_s0"] - U["MACRO0_s0"]; out.append(f"SAB-ATTR: TEST-U {dd:+.4f} -> {'NAMED' if abs(dd) >= 0.020 else 'none named'}")
txt = "\n".join(out); print(txt); open(".cache/gsm_macro_verdict.txt", "w").write(txt + "\n")
