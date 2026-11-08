"""gsm_btw_verdict.py -- THE BTW RUNG VERDICT (ledger 2026-11-08 BTW pin): the waist placement + the funnel tier, 5k steps, seed 0, vs E0
(BTL's BTC2-at-5k arm) and the old code at the same budget/seed (BT4RB8_5k_s0). Screen SCR-x: TEST step_all - E0 >= +0.020 (a lead ->
seed twin, never a claim alone). Beside: GAP-x vs BT4RB8_5k (>= -0.010 = closed), CF splits, ORIG, curve peak - final, params.
-> .cache/gsm_btw_verdict.txt"""
import json, os

ARMS = [a for a in ("LAST", "END", "FUN", "FUNW") if os.path.exists(f".cache/gsmb_btc_BTW_{a}_s0.json")]
A = {a: json.load(open(f".cache/gsmb_btc_BTW_{a}_s0.json")) for a in ARMS}
E = json.load(open(".cache/gsmb_btc_BTL_E0_s0.json")); R = json.load(open(".cache/gsmb_breath_BT4RB8_5k_s0.json"))
SP = ("ORIG", "CFNUM", "CFNAME", "TEST")
L = ["BTW (5k, s0) | " + " ".join(f"{sp} step" for sp in SP) + " | TEST type | curve TEST[:600] peak-final | params"]
L.append("  BT4RB8_5k " + " ".join(f"{R[f'{sp}_step_all']:.4f}" for sp in SP) + f" | {R['TEST_type_ok']:.4f} | -- | {R['params']}")
for a, d in [("E0", E)] + list(A.items()):
    c = [x["TEST_step_all"] for x in d["curve"]]
    L.append(f"  {a:9s} " + " ".join(f"{d[f'{sp}_step_all']:.4f}" for sp in SP) + f" | {d['TEST_type_ok']:.4f} | {max(c) - c[-1]:+.3f} | {d['params']}")
e0 = E["TEST_step_all"]
for a in ARMS:
    t = A[a]["TEST_step_all"]; g = t - R["TEST_step_all"]
    L.append(f"SCR-{a}: TEST step_all {t:.4f} - E0 {e0:.4f} = {t - e0:+.4f} (>= +0.020) -> {'PASS (lead: seed twin next)' if t - e0 >= 0.02 else 'FAIL'}"
             f" | beside GAP vs BT4RB8_5k {g:+.4f} ({'closed' if g >= -0.01 else 'open'})" + (f" | stacked: {A[a].get('stack')}" if a == "FUNW" else ""))
print("\n".join(L)); open(".cache/gsm_btw_verdict.txt", "w").write("\n".join(L) + "\n")
