"""gsm_fs_verdict.py -- FS (ledger 2026-11-08 pin): the fit-speed screen at 5k s0. FIT-x: ORIG step_all - E0 (0.4858) >= +0.050.
Beside: ORIG op / type / ptr, TEST step, TEST-U + CJ (vs E0S, from .cache/gsm_cj_FS_{arm}_s0.npz), CF splits. -> .cache/gsm_fs_verdict.txt"""
import json, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gsm_cj_verdict import cj

E0 = json.load(open(".cache/gsmb_btc_BTL_E0_s0.json")); out = []
hd = "arm      ORIG step  op     type   ptr   | CFNUM  CFNAME | TEST   TEST-U  CJ"
out.append(hd)
def row(n, j, f):
    cu = ""
    if f and os.path.exists(f):
        d = np.load(f); cu = f"{d['ok'][~d['seen']].mean():.4f} {cj(d)[0]:+.4f}"
    return (f"{n:8} {j['ORIG_step_all']:.4f} {j['ORIG_step_op']:.3f} {j['ORIG_type_ok']:.3f} {j['ORIG_step_ptr']:.3f} | "
            f"{j['CFNUM_step_all']:.4f} {j['CFNAME_step_all']:.4f} | {j['TEST_step_all']:.4f} {cu}")
out.append(row("E0", E0, None))
for a in ("E0S", "TAU1", "ZOOM", "PSCALE", "SLOTK"):
    f = f".cache/gsmb_btc_FS_{a}_s0.json"
    if not os.path.exists(f): out.append(f"{a:8} MISSING"); continue
    j = json.load(open(f)); out.append(row(a, j, f".cache/gsm_cj_FS_{a}_s0.npz"))
    if a == "E0S":
        out.append(f"  determinism: E0S ORIG {j['ORIG_step_all']:.4f} vs E0 {E0['ORIG_step_all']:.4f} -> "
                   f"{'EQUAL' if abs(j['ORIG_step_all'] - E0['ORIG_step_all']) < 1e-9 else 'DIFFERS'}")
        continue
    d = j["ORIG_step_all"] - E0["ORIG_step_all"]
    out.append(f"  FIT-{a}: ORIG step {j['ORIG_step_all']:.4f} - E0 {E0['ORIG_step_all']:.4f} = {d:+.4f} (>= +0.050) -> {'PASS' if d >= 0.050 else 'FAIL'}")
txt = "\n".join(out); print(txt); open(".cache/gsm_fs_verdict.txt", "w").write(txt + "\n")
