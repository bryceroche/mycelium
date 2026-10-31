"""gsm_bt3_verdict.py -- ledger 2026-10-31: BT3 (512 -> 64 waist, the per-cycle T6 clock, coarse-to-fine zoom) vs BT18S, the student's
states, 18k: C0 fit / Z1 the silhouette generalizes (TEST type) / Z2 TEST step; the oracle-type read on BT3's weights beside."""
import json

J = lambda f: json.load(open(f".cache/{f}.json"))
two = lambda X, k: " / ".join(f"{x[k]:.3f}" for x in X)
B3 = [J(f"gsmb_breath_BT3_s{s}") for s in (0, 1)]; B = [J(f"gsmb_breath_BT18S_s{s}") for s in (0, 1)]
O3 = [J(f"gsmb_breath_BT3O_s{s}") for s in (0, 1)]; S = [J(f"gsmb_trunk_STUTICK_s{s}") for s in (0, 1)]
L = [f"BT3 params {B3[0]['params']} vs BT18S {B[0]['params']}"]
for k in ("ORIG_step_all", "ORIG_type_ok", "TEST_type_ok", "TEST_step_op", "TEST_step_ptr", "TEST_step_all", "TEST_ans", "CFNUM_ans", "CFNAME_ans"):
    L.append(f"  {k:15s} BT3 {two(B3, k)} | BT18S {two(B, k)}" + (f" | STU-TICK {two(S, k)}" if k in S[0] else "") + f" | BT3 ORACLE {two(O3, k)}")
L.append("  ORIG step curve (every 3k): " + " | ".join(" ".join(f"{c['ORIG_step_all']:.2f}" for c in b["curve"][2::3]) for b in B3))
c0 = all(b["ORIG_step_all"] >= 0.90 for b in B3); v = "" if c0 else " [VOID: read only]"
z1 = [b["TEST_type_ok"] - a["TEST_type_ok"] for b, a in zip(B3, B)]
z2 = [b["TEST_step_all"] - a["TEST_step_all"] for b, a in zip(B3, B)]; m2 = sum(z2) / 2
L += [f"C0 FIT: ORIG step {two(B3, 'ORIG_step_all')} (bar 0.90 both) -> {'PASS' if c0 else 'VOID'}",
      f"Z1 THE SILHOUETTE GENERALIZES: TEST type {z1[0]:+.4f} / {z1[1]:+.4f} (bar +0.05 both) -> {'PASS' if all(x >= 0.05 for x in z1) else 'FAIL'}{v}",
      f"Z2 TEST STEP: {z2[0]:+.4f} / {z2[1]:+.4f} (mean {m2:+.4f}) -> "
      f"{'BETTER' if all(x >= 0.02 for x in z2) else ('EQUIVALENT' if m2 >= -0.010 else 'WORSE')}{v}"]
print("\n".join(L)); open(".cache/gsm_bt3_verdict.txt", "w").write("\n".join(L) + "\n")
