"""gsm_ub_verdict.py -- ledger 2026-11-05: UNIT-BIND (BT_UNIT=1) vs BT4RB8 (same engine, same RB8 states, same seeds): UB-0 fit, UB-1 the op
wall (TEST step_op), UB-2 TEST type, UB-3 TEST step (mean >= +0.010, both seeds > 0). With "P": UB-P vs the placebo UBP (BT_UNIT=2).
Usage: gsm_ub_verdict.py [ARM (default UB)] [P]  -> .cache/gsm_ub_verdict.txt (P: .cache/gsm_ubp_verdict.txt)"""
import json, sys

ARM = sys.argv[1] if len(sys.argv) > 1 else "UB"; PLAC = len(sys.argv) > 2 and sys.argv[2] == "P"
J = lambda t: [json.load(open(f".cache/gsmb_breath_{t}_s{s}.json")) for s in (0, 1)]
X, C = J(ARM), J("BT4RB8")
gain = lambda A, B, k: [a[k] - b[k] for a, b in zip(A, B)]


def bar(name, A, B, k, vs):
    z = gain(A, B, k); m = sum(z) / 2; ok = m >= 0.010 and min(z) > 0
    return f"{name}: {k} {ARM if A is X else ''} - {vs} {z[0]:+.4f} / {z[1]:+.4f} (mean {m:+.4f}; bar +0.010, both > 0) -> {'PASS' if ok else 'FAIL'}"


L = []
if PLAC:
    P = J(ARM + "P"); L.append(bar("UB-P", X, P, "TEST_step_op", ARM + "P"))
    for k in ("TEST_type_ok", "TEST_step_all", "ORIG_step_all"):
        L.append(f"  beside {k}: {ARM} " + " / ".join(f"{r[k]:.4f}" for r in X) + f" vs {ARM}P " + " / ".join(f"{r[k]:.4f}" for r in P))
else:
    fit = all(r["ORIG_step_all"] >= 0.90 for r in X)
    L.append(f"UB-0 FIT: {ARM} ORIG step " + " / ".join(f"{r['ORIG_step_all']:.4f}" for r in X) + f" (bar 0.90 both) -> {'PASS' if fit else 'FAIL'}")
    L += [bar("UB-1", X, C, "TEST_step_op", "BT4RB8"), bar("UB-2", X, C, "TEST_type_ok", "BT4RB8"), bar("UB-3", X, C, "TEST_step_all", "BT4RB8")]
    for k in ("ORIG_step_op", "CFNAME_step_all", "CFNUM_step_all", "TEST_step_ptr", "TEST_ans"):
        L.append(f"  beside {k}: {ARM} " + " / ".join(f"{r[k]:.4f}" for r in X) + " vs BT4RB8 " + " / ".join(f"{r[k]:.4f}" for r in C))
L.append("  [engine: tinygrad, both sides]")
print("\n".join(L)); open(f".cache/gsm_ub{'p' if PLAC else ''}_verdict.txt", "w").write("\n".join(L) + "\n")
