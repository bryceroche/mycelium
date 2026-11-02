"""gsm_bt5_verdict.py -- ledger 2026-11-01: BT5 (BT4 + the soft notebook) vs BT4: C0 fit / F1 TEST step (mean >= +0.010, both seeds
positive) / F2 TEST type (mean >= +0.010)."""
import json

J = lambda t: [json.load(open(f".cache/gsmb_breath_{t}_s{s}.json")) for s in (0, 1)]
two = lambda X, k: " / ".join(f"{x[k]:.3f}" for x in X)
B5, B4 = J("BT5"), J("BT4"); L = [f"BT5 params {B5[0]['params']} vs BT4 {B4[0]['params']}"]
for k in ("ORIG_step_all", "TEST_type_ok", "TEST_step_op", "TEST_step_ptr", "TEST_step_all", "TEST_ans", "CFNUM_ans", "CFNAME_ans"):
    L.append(f"  {k:15s} BT5 {two(B5, k)} | BT4 {two(B4, k)}")
d = lambda k: [a[k] - b[k] for a, b in zip(B5, B4)]
c0 = all(b["ORIG_step_all"] >= 0.90 for b in B5); f1 = d("TEST_step_all"); f2 = d("TEST_type_ok")
L += [f"C0 FIT: {two(B5, 'ORIG_step_all')} (bar 0.90 both) -> {'PASS' if c0 else 'VOID'}",
      f"F1 TEST STEP: {f1[0]:+.4f} / {f1[1]:+.4f} (mean {sum(f1) / 2:+.4f}; bar +0.010, both > 0) -> {'PASS' if sum(f1) / 2 >= 0.010 and min(f1) > 0 else 'FAIL'}",
      f"F2 TEST TYPE: {f2[0]:+.4f} / {f2[1]:+.4f} (mean {sum(f2) / 2:+.4f}; bar +0.010) -> {'PASS' if sum(f2) / 2 >= 0.010 else 'FAIL'}"]
print("\n".join(L)); open(".cache/gsm_bt5_verdict.txt", "w").write("\n".join(L) + "\n")
