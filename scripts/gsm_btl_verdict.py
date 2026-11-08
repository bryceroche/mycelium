"""gsm_btl_verdict.py -- THE BTL LADDER VERDICT (ledger 2026-11-08 BTL pin): five 5k-step BTC arms on seed 0 vs E0 (BTC2 at 5k) and
the old code at the same budget and seed (BT4RB8_5k_s0). Screen SCR-x: TEST step_all - E0 >= +0.020 (a lead -> seed twin, never a
claim alone). Beside: GAP-x (vs BT4RB8_5k_s0 >= -0.010), CF splits, ORIG, the TEST[:600] curve's peak - final. -> .cache/gsm_btl_verdict.txt"""
import json

ARMS = ("E0", "DROP", "WD", "TID", "SC")
A = {a: json.load(open(f".cache/gsmb_btc_BTL_{a}_s0.json")) for a in ARMS}
R = json.load(open(".cache/gsmb_breath_BT4RB8_5k_s0.json"))
SP = ("ORIG", "CFNUM", "CFNAME", "TEST")
L = ["BTL (5k, s0) | " + " ".join(f"{sp} step" for sp in SP) + " | TEST type | curve TEST[:600] peak-final | params"]
L.append(f"  BT4RB8_5k " + " ".join(f"{R[f'{sp}_step_all']:.4f}" for sp in SP) + f" | {R['TEST_type_ok']:.4f} | -- | {R['params']}")
for a, d in A.items():
    c = [x["TEST_step_all"] for x in d["curve"]]
    L.append(f"  {a:9s} " + " ".join(f"{d[f'{sp}_step_all']:.4f}" for sp in SP) + f" | {d['TEST_type_ok']:.4f} | {max(c) - c[-1]:+.3f} | {d['params']}")
e0 = A["E0"]["TEST_step_all"]
for a in ARMS[1:]:
    t = A[a]["TEST_step_all"]
    L.append(f"SCR-{a}: TEST step_all {t:.4f} - E0 {e0:.4f} = {t - e0:+.4f} (>= +0.020) -> {'PASS (lead: seed twin next)' if t - e0 >= 0.02 else 'FAIL'}"
             f" | beside GAP vs BT4RB8_5k {t - R['TEST_step_all']:+.4f} (>= -0.010 -> {'closed' if t - R['TEST_step_all'] >= -0.01 else 'open'})")
L.append(f"E0 GAP vs BT4RB8_5k {e0 - R['TEST_step_all']:+.4f}")
print("\n".join(L)); open(".cache/gsm_btl_verdict.txt", "w").write("\n".join(L) + "\n")
