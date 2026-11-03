"""BT7b verdict (ledger 2026-11-03 05:43): C0, B7b (TEST step), B7bT (TEST type) on the BT7b weights; BP vs the BT6 gate, same weights.
Bars: C0 ORIG step (BT6 gate) >= 0.90 both seeds; B7b / B7bT mean over seeds >= +0.010 with both seeds > 0."""
import json

R = [json.load(open(f".cache/gsmb_bt7_BT7b_s{s}.json")) for s in (0, 1)]
B5 = [json.load(open(f".cache/gsmb_bt6_BT5_s{s}.json")) for s in (0, 1)]
c0 = [r["ORIG"]["bt6"]["step_all"] for r in R]
lines = [f"C0 FIT (BT6 gate): {c0[0]:.3f} / {c0[1]:.3f} (bar 0.90 both) -> {'PASS' if min(c0) >= 0.90 else 'FAIL'}"]
for bar, key in (("B7b", "diff_step"), ("B7bT", "diff_type")):
    d = [r["TEST"][key] for r in R]; m = sum(d) / 2
    lines.append(f"{bar}: TEST {key} BP - gate {d[0]:+.4f} / {d[1]:+.4f} (mean {m:+.4f}; bar +0.010, both > 0) -> "
                 f"{'PASS' if m >= 0.010 and min(d) > 0 else 'FAIL'} | ORIG {R[0]['ORIG'][key]:+.4f} / {R[1]['ORIG'][key]:+.4f}")
lines.append("read: TEST step BT6 gate, BT7b vs BT5 weights: " + " / ".join(f"{r['TEST']['bt6']['step_all']:.4f} vs {b['TEST']['gate']['step_all']:.4f}" for r, b in zip(R, B5))
             + " | TEST type: " + " / ".join(f"{r['TEST']['bt6']['type_ok']:.4f} vs {b['TEST']['gate']['type_ok']:.4f}" for r, b in zip(R, B5)))
print("\n".join(lines)); open(".cache/gsm_bt7b_verdict.txt", "w").write("\n".join(lines) + "\n")
