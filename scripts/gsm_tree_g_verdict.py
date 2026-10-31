"""gsm_tree_g_verdict.py -- ledger 2026-10-31: THE TREE ROAD (BT-TREE vs BT18S: C0 / R1 / R2) and THE GSM8K-TEXT FORM's G1 (G-TICK vs
Llama-TICK on the counterfactuals, TEST held; S0's STU-TICK beside)."""
import json

J = lambda f: json.load(open(f".cache/{f}.json"))
two = lambda X, k: " / ".join(f"{x[k]:.3f}" for x in X)
mean = lambda X, k: sum(x[k] for x in X) / len(X)
BT = [J(f"gsmb_breath_BTTREE_s{s}") for s in (0, 1)]; B = [J(f"gsmb_breath_BT18S_s{s}") for s in (0, 1)]
G = [J(f"gsmb_trunk_GTICK_s{s}") for s in (0, 1)]; T = [J(f"gsmb_trunk_TICK_s{s}") for s in (0, 1)]; S = [J(f"gsmb_trunk_STUTICK_s{s}") for s in (0, 1)]
L = ["THE TREE ROAD (BT-TREE vs BT18S, the student's states, 18k):"]
for k in ("ORIG_step_all", "TEST_step_all", "TEST_step_op", "TEST_step_ptr", "TEST_ans", "CFNUM_ans", "CFNAME_ans", "TEST_type_ok"):
    L.append(f"  {k:15s} BT-TREE {two(BT, k)} | BT18S {two(B, k)} | diff {mean(BT, k) - mean(B, k):+.4f}")
L.append("  ORIG step curve (every 3k): BT-TREE " + " | ".join(" ".join(f"{c['ORIG_step_all']:.2f}" for c in b["curve"][2::3]) for b in BT)
         + " ; BT18S " + " | ".join(" ".join(f"{c['ORIG_step_all']:.2f}" for c in b["curve"][2::3]) for b in B))
c0 = all(b["ORIG_step_all"] >= 0.90 for b in BT); v = "" if c0 else " [VOID: read only]"
r1 = [b["TEST_step_op"] - a["TEST_step_op"] for b, a in zip(BT, B)]; r2 = mean(BT, "TEST_step_all") - mean(B, "TEST_step_all")
L += [f"C0 FIT: BT-TREE ORIG step {two(BT, 'ORIG_step_all')} (bar 0.90 both) -> {'PASS' if c0 else 'VOID'}",
      f"R1 THE TREE NAMES THE OP: TEST op {r1[0]:+.4f} / {r1[1]:+.4f} (bar +0.02 both) -> {'PASS' if all(x >= 0.02 for x in r1) else 'FAIL'}{v}",
      f"R2 TEST STEP: mean {r2:+.4f} (bar +0.010) -> {'PASS' if r2 >= 0.010 else 'FAIL'}{v}", "THE GSM8K-TEXT FORM (G-TICK on NMDBG's native-head states):"]
for k in ("ORIG_step_all", "TEST_step_all", "TEST_ans", "CFNUM_ans", "CFNAME_ans", "CFNUM_step_all", "CFNAME_step_all"):
    L.append(f"  {k:15s} G {two(G, k)} | STU {two(S, k)} | LLAMA {two(T, k)} | G - LLAMA {mean(G, k) - mean(T, k):+.4f}")
g = {k: mean(G, k) - mean(T, k) for k in ("CFNUM_ans", "CFNAME_ans", "TEST_step_all")}
ok = g["CFNUM_ans"] >= -0.020 and g["CFNAME_ans"] >= -0.020 and g["TEST_step_all"] >= -0.020
L.append(f"G1 THE CF GAP CLOSES: CFNUM {g['CFNUM_ans']:+.4f}, CFNAME {g['CFNAME_ans']:+.4f} (bars >= -0.020), TEST step {g['TEST_step_all']:+.4f} "
         f"(held >= -0.020) -> {'PASS' if ok else 'FAIL'}")
print("\n".join(L)); open(".cache/gsm_tree_g_verdict.txt", "w").write("\n".join(L) + "\n")
