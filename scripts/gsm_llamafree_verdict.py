"""gsm_llamafree_verdict.py -- THE LLAMA-FREE GSM8K BENCH (ledger 2026-10-31, Bryce: "retire llama, all in on the breathing transformer").
S0 THE SWAP: TICK on the MLP-free student's word states (STU-TICK) vs TICK on the banked Llama states. BT18S: the BT arm on the
student's states, 18k steps: C0 / D1 / D2 against STU-TICK (same input). ORACLE-TYPE read on BT18S's saved weights."""
import json, os

J = lambda f: json.load(open(f".cache/{f}.json"))
T = [J(f"gsmb_trunk_TICK_s{s}") for s in (0, 1)]; S = [J(f"gsmb_trunk_STUTICK_s{s}") for s in (0, 1)]
B = [J(f"gsmb_breath_BT18S_s{s}") for s in (0, 1)]
O = [J(f"gsmb_breath_BT18SO_s{s}") if os.path.exists(f".cache/gsmb_breath_BT18SO_s{s}.json") else None for s in (0, 1)]
two = lambda X, k: " / ".join(f"{x[k]:.3f}" for x in X)
mean = lambda X, k: sum(x[k] for x in X) / len(X)
L = ["S0 THE SWAP (TICK arm, 6k steps; Llama states vs the student's):"]
for k in ("ORIG_step_all", "TEST_step_all", "TEST_step_op", "TEST_step_ptr", "TEST_ans", "CFNUM_ans", "CFNAME_ans"):
    L.append(f"  {k:15s} STU {two(S, k)}  | LLAMA {two(T, k)}  | diff {mean(S, k) - mean(T, k):+.4f}")
fit = all(s["ORIG_step_all"] >= 0.90 for s in S); sw = mean(S, "TEST_step_all") - mean(T, "TEST_step_all")
L.append(f"S0: STU fit {two(S, 'ORIG_step_all')} (bar 0.90 both), TEST step diff {sw:+.4f} (bar >= -0.020) -> "
         f"{'PASS' if fit and sw >= -0.020 else ('VOID (no fit)' if not fit else 'FAIL')}")
L.append("BT18S (BT on the student's states, 18k steps) vs STU-TICK:")
for k in ("ORIG_step_all", "ORIG_ans", "TEST_step_all", "TEST_step_op", "TEST_step_ptr", "TEST_ans", "CFNUM_ans", "CFNAME_ans",
          "TEST_num_for_slot", "TEST_type_ok", "ORIG_type_ok"):
    L.append(f"  {k:18s} BT18S {two(B, k)}" + (f"  (STU-TICK {two(S, k)})" if k in S[0] else ""))
L.append("  curve ORIG step: " + " | ".join(" ".join(f"{c['ORIG_step_all']:.2f}" for c in b["curve"][2::3]) for b in B))
c0 = all(b["ORIG_step_all"] >= 0.90 for b in B)
d = [b["TEST_step_all"] - s["TEST_step_all"] for b, s in zip(B, S)]; dm = sum(d) / 2
d1 = "BETTER" if all(x >= 0.020 for x in d) else ("EQUIVALENT-OR-BETTER" if dm >= -0.010 else "WORSE")
dp = [b["TEST_step_ptr"] - s["TEST_step_ptr"] for b, s in zip(B, S)]
v = "" if c0 else " [VOID: read only]"
L += [f"C0 FIT GATE: ORIG step {two(B, 'ORIG_step_all')} (bar 0.90 both) -> {'PASS' if c0 else 'VOID'}",
      f"D1 TEST STEP: BT18S - STU-TICK {d[0]:+.4f} / {d[1]:+.4f} (mean {dm:+.4f}) -> {d1}{v}",
      f"D2 SILHOUETTE FIXES THE KEYS: TEST pointers {dp[0]:+.4f} / {dp[1]:+.4f} (bar +0.03 both) -> {'PASS' if all(x >= 0.03 for x in dp) else 'FAIL'}{v}"]
if all(O):
    L.append("ORACLE-TYPE READ (gold step type as the silhouette's class, BT18S weights): " + " | ".join(
        f"{k} {two(O, k)} (vs {two(B, k)})" for k in ("TEST_step_all", "TEST_step_ptr", "TEST_ans", "ORIG_step_all")))
print("\n".join(L)); open(".cache/gsm_llamafree_verdict.txt", "w").write("\n".join(L) + "\n")
