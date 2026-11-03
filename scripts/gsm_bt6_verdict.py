"""BT6 verdict (ledger 2026-11-02 18:22): G1 on BT5 weights, G2 on BT4 weights.
Bar each: TEST step_all gated - scheduled, mean over seeds >= +0.010 with both seeds > 0."""
import json

lines = []
for bar, arm in (("G1", "BT5"), ("G2", "BT4")):
    R = [json.load(open(f".cache/gsmb_bt6_{arm}_s{s}.json")) for s in (0, 1)]
    d = [r["TEST"]["diff"] for r in R]; m = sum(d) / 2
    ok = m >= 0.010 and min(d) > 0
    lines.append(f"{bar} {arm}: rho0 {R[0]['rho0']} / {R[1]['rho0']} | TEST step diff {d[0]:+.4f} / {d[1]:+.4f} (mean {m:+.4f}; bar +0.010, both > 0) "
                 f"-> {'PASS' if ok else 'FAIL'} | ORIG diff {R[0]['ORIG']['diff']:+.4f} / {R[1]['ORIG']['diff']:+.4f}")
print("\n".join(lines)); open(".cache/gsm_bt6_verdict.txt", "w").write("\n".join(lines) + "\n")
