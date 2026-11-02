"""gsm_cy_verdict.py -- ledger 2026-11-02 CY (THE CLOCK YARDSTICK): BT4 on the RoPE-free T9-NS reader's GSM8K states (BT4T9NS) vs BT4
on NMDB's: C0 fit / Y1 PARITY TEST step (mean >= -0.010) / Y2 BETTER TEST step (mean >= +0.010, both seeds > 0)."""
import json

J = lambda t: [json.load(open(f".cache/gsmb_breath_{t}_s{s}.json")) for s in (0, 1)]
two = lambda X, k: " / ".join(f"{x[k]:.3f}" for x in X)
Y, B4 = J("BT4T9NS"), J("BT4"); L = [f"BT4T9NS params {Y[0]['params']} vs BT4 {B4[0]['params']}"]
for k in ("ORIG_step_all", "TEST_type_ok", "TEST_step_op", "TEST_step_ptr", "TEST_step_all", "TEST_ans", "CFNUM_ans", "CFNAME_ans"):
    L.append(f"  {k:15s} BT4T9NS {two(Y, k)} | BT4 {two(B4, k)}")
d = [a["TEST_step_all"] - b["TEST_step_all"] for a, b in zip(Y, B4)]; m = sum(d) / 2
c0 = all(b["ORIG_step_all"] >= 0.90 for b in Y)
L += [f"C0 FIT: {two(Y, 'ORIG_step_all')} (bar 0.90 both) -> {'PASS' if c0 else 'VOID'}",
      f"Y1 PARITY TEST STEP: {d[0]:+.4f} / {d[1]:+.4f} (mean {m:+.4f}; bar >= -0.010) -> {'PASS' if m >= -0.010 else 'FAIL'}",
      f"Y2 BETTER TEST STEP: mean {m:+.4f} (bar +0.010, both > 0) -> {'PASS' if m >= 0.010 and min(d) > 0 else 'FAIL'}"]
print("\n".join(L)); open(".cache/gsm_cy_verdict.txt", "w").write("\n".join(L) + "\n")
