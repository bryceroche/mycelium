#!/usr/bin/bash
# BT8 THE FACTORED SILHOUETTE (ledger 11-03; word given for the whole queue): after lf8m launched BT4M8 and lfbeam2read finished,
# apply scripts/patches/bt8_fact.py; BT4 config + BT_FACT=1, s0/s1 18k on the 8M-reader states (the control = BT4M8, same seeds/states).
set -eo pipefail; cd /home/bryce/mycelium; exec > .cache/run_lfbt8.log 2>&1
echo "== start $(date +%H:%M) =="; PY=.venv/bin/python3
while ! grep -q "BT4M8_s1 step 1000 \|BT4M8_s0 step 1000 " .cache/run_lf8m.log 2>/dev/null || systemctl --user is-active -q lfbeam2read; do grep -q "Traceback" .cache/run_lf8m.log 2>/dev/null && { echo "lf8m failed"; exit 1; }; sleep 120; done
$PY scripts/patches/bt8_fact.py
export RD_STATES=gsmb_mbr8_states.npy BT_ROLEF=gsmb_role_feats_mbr8.npy; B4="BT_D=512 BT_NH=8 BT_DB=64 BT_CYCLOCK=1 BT_ZOOM=1 BT_ROLE=1 BT_T6OFF=1"
for S in 0 1; do env $B4 BT_FACT=1 BT_STEPS=18000 BT_SAVE=1 BT_TAG=BT8_s$S BT_SEED=$S TOY_THREADS=3 $PY scripts/gsm_breath_reader.py 2>&1 | grep --line-buffered -v -i warn & done; wait
while systemctl --user is-active -q lf8m; do sleep 120; done
$PY - <<'PY'
import json
R = {a: [json.load(open(f".cache/gsmb_breath_{a}_s{s}.json")) for s in (0, 1)] for a in ("BT8", "BT4M8")}
L = []
f0 = [r["ORIG_step_all"] for r in R["BT8"]]
L.append(f"F0 FIT: BT8 ORIG step {f0[0]:.4f} / {f0[1]:.4f} (bar 0.90 both) -> {'PASS' if min(f0) >= 0.90 else 'FAIL'}")
for bar, m in (("F1", "TEST_type_ok"), ("F2", "TEST_step_all")):
    d = [a[m] - b[m] for a, b in zip(R["BT8"], R["BT4M8"])]; mn = sum(d) / 2
    L.append(f"{bar}: {m} BT8 - BT4M8 {d[0]:+.4f} / {d[1]:+.4f} (mean {mn:+.4f}; bar +0.010, both > 0) -> {'PASS' if mn >= 0.010 and min(d) > 0 else 'FAIL'}")
for m in ("ORIG_step_all", "CFNUM_step_all", "CFNAME_step_all", "TEST_step_op", "TEST_ans"):
    L.append(f"  beside {m}: BT8 " + " / ".join(f"{r[m]:.4f}" for r in R["BT8"]) + " vs BT4M8 " + " / ".join(f"{r[m]:.4f}" for r in R["BT4M8"]))
print("\n".join(L)); open(".cache/gsm_bt8_verdict.txt", "w").write("\n".join(L) + "\n")
PY
echo "== done $(date +%H:%M) =="
