"""gsm_bt4_verdict.py -- ledger 2026-11-01: BT4 (BT3 + the tree road + T6OFF) vs BT3 (C0 / B1 build parity / B2 gain); the size arms
R4 / B3 / H16 vs BT4 (S1: TEST step mean >= +0.010, both seeds positive); BT3N (BT18S + clock + zoom at BT18S size) vs BT18S (A1).
Reads whichever arms have finished; prints PENDING for the rest."""
import json, os

J = lambda t: [json.load(open(f".cache/gsmb_breath_{t}_s{s}.json")) for s in (0, 1)]
have = lambda t: all(os.path.exists(f".cache/gsmb_breath_{t}_s{s}.json") for s in (0, 1))
two = lambda X, k: " / ".join(f"{x[k]:.3f}" for x in X)
KS = ("ORIG_step_all", "TEST_type_ok", "TEST_step_op", "TEST_step_ptr", "TEST_step_all", "TEST_ans", "CFNUM_ans", "CFNAME_ans")
d = lambda A, B, k: [a[k] - b[k] for a, b in zip(A, B)]
L = []
B3, B18 = J("BT3"), J("BT18S")
if have("BT4"):
    B4 = J("BT4"); L.append(f"BT4 params {B4[0]['params']}")
    for k in KS: L.append(f"  {k:15s} BT4 {two(B4, k)} | BT3 {two(B3, k)}")
    c0 = all(b["ORIG_step_all"] >= 0.90 for b in B4); m = sum(d(B4, B3, "TEST_step_all")) / 2
    L += [f"C0 FIT (BT4): {two(B4, 'ORIG_step_all')} (bar 0.90 both) -> {'PASS' if c0 else 'VOID'}",
          f"B1 BUILD PARITY: TEST step mean {m:+.4f} vs BT3 (bar >= -0.010) -> {'PASS' if m >= -0.010 else 'FAIL'}",
          f"B2 GAIN: TEST step mean {m:+.4f} vs BT3 (bar >= +0.010) -> {'PASS' if m >= 0.010 else 'FAIL'}"]
    for t in ("BT4R4", "BT4B3", "BT4H16"):
        if not have(t): L.append(f"S1 {t}: PENDING"); continue
        X = J(t); L.append(f"{t} params {X[0]['params']}")
        for k in KS: L.append(f"  {k:15s} {t} {two(X, k)} | BT4 {two(B4, k)}")
        z = d(X, B4, "TEST_step_all"); m = sum(z) / 2; ok = m >= 0.010 and all(x > 0 for x in z)
        L.append(f"S1 {t}: C0 {two(X, 'ORIG_step_all')} | TEST step {z[0]:+.4f} / {z[1]:+.4f} (mean {m:+.4f}; bar +0.010, both > 0) -> "
                 f"{'PASS' if ok else 'FAIL'} | TEST type {sum(d(X, B4, 'TEST_type_ok')) / 2:+.4f}")
else: L.append("BT4: PENDING")
if have("BT3N"):
    N = J("BT3N")
    for k in KS: L.append(f"  {k:15s} BT3N {two(N, k)} | BT18S {two(B18, k)} | BT3 {two(B3, k)}")
    m = sum(d(N, B18, "TEST_step_all")) / 2; tot = sum(d(B3, B18, "TEST_step_all")) / 2
    L.append(f"A1 BT3N: TEST step mean {m:+.4f} vs BT18S (bar >= +0.010) -> {'PASS' if m >= 0.010 else 'FAIL'} | organs' share of BT3's {tot:+.4f}: {m / tot:.2f}")
else: L.append("A1 BT3N: PENDING")
print("\n".join(L)); open(".cache/gsm_bt4_verdict.txt", "w").write("\n".join(L) + "\n")
