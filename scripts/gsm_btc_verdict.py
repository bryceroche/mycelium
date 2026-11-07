"""gsm_btc_verdict.py -- THE BTC VERDICT (ledger 2026-11-07 BTC pin): the official-class GSM tier (gsm_btc.py) vs BT4RB8, two seeds.
Bars BTC-N (non-inferior), BTC-G (gain), BTC-CF. -> .cache/gsm_btc_verdict.txt"""
import json

A = [json.load(open(f".cache/gsmb_btc_BTC_s{s}.json")) for s in (0, 1)]
R = [json.load(open(f".cache/gsmb_breath_BT4RB8_s{s}.json")) for s in (0, 1)]
g = lambda d, k: d[k]
L = [f"params BTC {A[0]['params']} vs BT4RB8 {R[0]['params']} | steps {A[0]['steps']} / {R[0]['steps']}"]
for sp in ("ORIG", "CFNUM", "CFNAME", "TEST"):
    L.append(f"  {sp:6s} " + " | ".join(f"{m} BTC {A[0][f'{sp}_{m}']:.4f} / {A[1][f'{sp}_{m}']:.4f} vs {R[0][f'{sp}_{m}']:.4f} / {R[1][f'{sp}_{m}']:.4f}"
                                      for m in ("step_all", "type_ok", "step_ptr", "ans")))
d = [A[s]["TEST_step_all"] - R[s]["TEST_step_all"] for s in (0, 1)]
L.append(f"BTC-N TEST step_all BTC - BT4RB8 {d[0]:+.4f} / {d[1]:+.4f} (>= -0.020 both) -> {'PASS' if min(d) >= -0.02 else 'FAIL'}")
L.append(f"BTC-G TEST step_all BTC - BT4RB8 {d[0]:+.4f} / {d[1]:+.4f} (>= +0.020 both) -> {'PASS' if min(d) >= 0.02 else 'FAIL'}")
dc = [A[s][f"{cf}_step_all"] - R[s][f"{cf}_step_all"] for s in (0, 1) for cf in ("CFNUM", "CFNAME")]
L.append("BTC-CF CF step_all BTC - BT4RB8 (s0 CFNUM, s0 CFNAME, s1 CFNUM, s1 CFNAME) " + " / ".join(f"{x:+.4f}" for x in dc)
         + f" (>= -0.030 all) -> {'PASS' if min(dc) >= -0.03 else 'FAIL'}")
print("\n".join(L)); open(".cache/gsm_btc_verdict.txt", "w").write("\n".join(L) + "\n")
