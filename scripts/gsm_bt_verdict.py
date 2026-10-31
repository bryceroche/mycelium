"""gsm_bt_verdict.py -- the BT arm's verdict (ledger 2026-10-30): C0 fit gate, D1 TEST step vs TICK, D2 TEST pointers vs TICK; reads."""
import json

B = [json.load(open(f".cache/gsmb_breath_BT_s{s}.json")) for s in (0, 1)]
T = [json.load(open(f".cache/gsmb_trunk_TICK_s{s}.json")) for s in (0, 1)]
f = lambda k: " / ".join(f"{b[k]:.3f}" for b in B) + "  (TICK " + " / ".join(f"{t[k]:.3f}" for t in T) + ")"
L = [f"BT params {B[0]['params']}; TICK D {T[0]['d']} params {T[0]['params']}"]
for k in ("ORIG_step_all", "ORIG_ans", "TEST_step_all", "TEST_step_op", "TEST_step_ptr", "TEST_ans", "CFNUM_ans", "CFNAME_ans", "TEST_num_for_slot"):
    L.append(f"  {k:18s} BT {f(k)}")
L.append("  TEST_type_ok       BT " + " / ".join(f"{b['TEST_type_ok']:.3f}" for b in B) + " | ORIG " + " / ".join(f"{b['ORIG_type_ok']:.3f}" for b in B))
L.append("  TEST commit share by cycle " + " | ".join(str(b["TEST_commit_share"]) for b in B))
c0 = all(b["ORIG_step_all"] >= 0.90 for b in B)
d = [b["TEST_step_all"] - t["TEST_step_all"] for b, t in zip(B, T)]; dm = sum(d) / 2
d1 = "BETTER" if all(x >= 0.020 for x in d) else ("EQUIVALENT-OR-BETTER" if dm >= -0.010 else "WORSE")
dp = [b["TEST_step_ptr"] - t["TEST_step_ptr"] for b, t in zip(B, T)]; d2 = all(x >= 0.03 for x in dp)
L += [f"C0 FIT GATE: ORIG step " + " / ".join(f"{b['ORIG_step_all']:.3f}" for b in B) + f" (bar 0.90 both) -> {'PASS' if c0 else 'VOID'}",
      f"D1 TEST STEP: BT - TICK {d[0]:+.4f} / {d[1]:+.4f} (mean {dm:+.4f}) -> {d1}" + ("" if c0 else " [VOID: read only]"),
      f"D2 SILHOUETTE FIXES THE KEYS: TEST pointers BT - TICK {dp[0]:+.4f} / {dp[1]:+.4f} (bar +0.03 both) -> {'PASS' if d2 else 'FAIL'}" + ("" if c0 else " [VOID: read only]")]
print("\n".join(L)); open(".cache/gsm_bt_verdict.txt", "w").write("\n".join(L) + "\n")
