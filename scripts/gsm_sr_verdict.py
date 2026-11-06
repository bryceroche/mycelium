"""gsm_sr_verdict.py -- ledger 2026-11-06 06:35, THE SHORT-RUN RULE: SR-1 |BT4RB8_5k - BT4RB8 (18k)| TEST step <= 0.010 on both seeds.
Usage: gsm_sr_verdict.py [ARM (default BT4RB8_5k)]  -> .cache/gsm_sr_verdict.txt"""
import json, sys

ARM = sys.argv[1] if len(sys.argv) > 1 else "BT4RB8_5k"
J = lambda t: [json.load(open(f".cache/gsmb_breath_{t}_s{s}.json")) for s in (0, 1)]
X, C = J(ARM), J("BT4RB8")
z = [a["TEST_step_all"] - b["TEST_step_all"] for a, b in zip(X, C)]
L = [f"SR-1: TEST_step_all {ARM} - BT4RB8 {z[0]:+.4f} / {z[1]:+.4f} (bar |d| <= 0.010 both) -> {'PASS' if max(map(abs, z)) <= 0.010 else 'FAIL'}"]
for k in ("TEST_step_op", "TEST_step_ptr", "TEST_type_ok", "TEST_ans", "ORIG_step_all", "CFNAME_step_all", "CFNUM_step_all"):
    L.append(f"  beside {k}: {ARM} " + " / ".join(f"{r[k]:.4f}" for r in X) + " vs BT4RB8 " + " / ".join(f"{r[k]:.4f}" for r in C))
print("\n".join(L)); open(".cache/gsm_sr_verdict.txt", "w").write("\n".join(L) + "\n")
