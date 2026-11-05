"""gsm_so_verdict.py -- ledger 2026-11-05: THE SEALED OP HEAD (BT_SEAL=1) vs BT4RB8 (same engine, same RB8 states, same seeds).
SO-S the seal holds (ORIG op <= 0.70 both), SO-P op-blind pointers keep their fit (ORIG ptr >= 0.90 both), SO-H the head reaches the
table (TEST op on GOLD operands >= 0.55 both), SO-1 TEST op / SO-2 TEST step (mean >= +0.010, both seeds > 0).
Usage: gsm_so_verdict.py [ARM (default SO)]  -> .cache/gsm_so_verdict.txt"""
import json, sys

ARM = sys.argv[1] if len(sys.argv) > 1 else "SO"
J = lambda t: [json.load(open(f".cache/gsmb_breath_{t}_s{s}.json")) for s in (0, 1)]
X, C = J(ARM), J("BT4RB8")
f2 = lambda R, k: " / ".join(f"{r[k]:.4f}" for r in R)


def each(name, k, ok, bar):
    v = all(ok(r[k]) for r in X)
    return f"{name}: {k} {ARM} {f2(X, k)} ({bar}) -> {'PASS' if v else 'FAIL'}"


def gain(name, k):
    z = [a[k] - b[k] for a, b in zip(X, C)]; m = sum(z) / 2; ok = m >= 0.010 and min(z) > 0
    return f"{name}: {k} {ARM} - BT4RB8 {z[0]:+.4f} / {z[1]:+.4f} (mean {m:+.4f}; bar +0.010, both > 0) -> {'PASS' if ok else 'FAIL'}"


L = [each("SO-S", "ORIG_step_op", lambda v: v <= 0.70, "bar <= 0.70 both"),
     each("SO-P", "ORIG_step_ptr", lambda v: v >= 0.90, "bar >= 0.90 both"),
     each("SO-H", "TEST_op_gold", lambda v: v >= 0.55, "bar >= 0.55 both"),
     gain("SO-1", "TEST_step_op"), gain("SO-2", "TEST_step_all")]
for k in ("TEST_step_ptr", "TEST_type_ok", "ORIG_step_all", "TEST_ans", "CFNAME_step_all", "CFNUM_step_all"):
    L.append(f"  beside {k}: {ARM} {f2(X, k)} vs BT4RB8 {f2(C, k)}")
L.append(f"  beside ORIG_op_gold: {ARM} {f2(X, 'ORIG_op_gold')} | [engine: tinygrad, both sides]")
print("\n".join(L)); open(".cache/gsm_so_verdict.txt", "w").write("\n".join(L) + "\n")
