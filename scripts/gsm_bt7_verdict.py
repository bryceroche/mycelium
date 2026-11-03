"""BT7a verdict (ledger 2026-11-02 20:31): B7 (TEST step) and B7T (TEST type) on BT5 weights; BT4 read beside.
Bar each: BT7a - BT6 gate, mean over seeds >= +0.010 with both seeds > 0. Custody: the BT6 arm reproduces BT6's banked TEST step."""
import json

lines = []
for arm in ("BT5", "BT4"):
    R = [json.load(open(f".cache/gsmb_bt7_{arm}_s{s}.json")) for s in (0, 1)]
    B6 = [json.load(open(f".cache/gsmb_bt6_{arm}_s{s}.json")) for s in (0, 1)]
    cust = all(abs(r["TEST"]["bt6"]["step_all"] - b["TEST"]["gate"]["step_all"]) < 5e-5 for r, b in zip(R, B6))
    for bar, key in (("B7", "diff_step"), ("B7T", "diff_type")):
        d = [r["TEST"][key] for r in R]; m = sum(d) / 2; ok = m >= 0.010 and min(d) > 0
        lines.append(f"{bar}{'' if arm == 'BT5' else ' (read, BT4)'} {arm}: TEST {key} {d[0]:+.4f} / {d[1]:+.4f} (mean {m:+.4f}; bar +0.010, both > 0) -> "
                     f"{'PASS' if ok else 'FAIL'} | ORIG {R[0]['ORIG'][key]:+.4f} / {R[1]['ORIG'][key]:+.4f} | custody {'OK' if cust else 'MISMATCH'}")
print("\n".join(lines)); open(".cache/gsm_bt7_verdict.txt", "w").write("\n".join(lines) + "\n")
