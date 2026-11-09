"""gsm_bind_verdict.py -- BIND (ledger 2026-11-08 pin): multiplicative binding at 5k s0 against E0 (BTC2 at 5k s0).
FIT-x:    ORIG step_all - E0 (0.4858) >= +0.050 -> a fit lever (a seed twin before any claim)
UNSEEN-x: TEST-U (unseen-skeleton step_all) - E0S (0.0923) >= +0.020 -> transfer off the cross join
CJ-x:     CJ - E0S (+0.0782), NAMED only if >= 0.040 in size (descriptive; no bar)
Arms: BIND (BTC_BIND=1, the residual roads) | BINDP (BTC_BIND=2, + the pointer roads). -> .cache/gsm_bind_verdict.txt"""
import json, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gsm_cj_verdict import cj

E0 = json.load(open(".cache/gsmb_btc_BTL_E0_s0.json")); e0 = np.load(".cache/gsm_cj_FS_E0S_s0.npz")
U0, C0 = e0["ok"][~e0["seen"]].mean(), cj(e0)[0]
out = [f"E0: ORIG step {E0['ORIG_step_all']:.4f} | TEST {E0['TEST_step_all']:.4f} TEST-U {U0:.4f} CJ {C0:+.4f}"]
for a in ("BIND", "BINDP"):
    f = f".cache/gsmb_btc_BIND_{a}_s0.json"; g = f".cache/gsm_cj_BIND_{a}_s0.npz"
    if not (os.path.exists(f) and os.path.exists(g)): out.append(f"{a}: MISSING"); continue
    j = json.load(open(f)); d = np.load(g); u, c = d["ok"][~d["seen"]].mean(), cj(d)[0]
    if abs(d["ok"].mean() - j["TEST_step_all"]) > 0.003: out.append(f"{a}: CJ re-score VOID"); continue
    fd, ud, cd = j["ORIG_step_all"] - E0["ORIG_step_all"], u - U0, c - C0
    out.append(f"{a}: ORIG step {j['ORIG_step_all']:.4f} op {j['ORIG_step_op']:.3f} type {j['ORIG_type_ok']:.3f} ptr "
               f"{j['ORIG_step_ptr']:.3f} | CFNUM {j['CFNUM_step_all']:.4f} CFNAME {j['CFNAME_step_all']:.4f} | TEST "
               f"{j['TEST_step_all']:.4f} TEST-U {u:.4f} CJ {c:+.4f}")
    out.append(f"  FIT-{a}: {fd:+.4f} (>= +0.050) -> {'PASS' if fd >= 0.050 else 'FAIL'}")
    out.append(f"  UNSEEN-{a}: {ud:+.4f} (>= +0.020) -> {'PASS' if ud >= 0.020 else 'FAIL'}")
    out.append(f"  CJ-{a}: {cd:+.4f} -> {'NAMED' if abs(cd) >= 0.040 else 'none named'}")
txt = "\n".join(out); print(txt); open(".cache/gsm_bind_verdict.txt", "w").write(txt + "\n")
