#!/usr/bin/bash
# BT9 THE ID SKIP (ledger 11-03; word given): after lfbt8 finishes and lfw128 has launched its runs (both re-read gsm_breath_reader.py),
# apply scripts/patches/bt9_idskip.py; BT4 config + BT_IDSKIP=1, s0/s1 18k on the 8M-reader states; control BT4M8.
set -eo pipefail; cd /home/bryce/mycelium; exec > .cache/run_lfbt9.log 2>&1
echo "== start $(date +%H:%M) =="; PY=.venv/bin/python3
while systemctl --user is-active -q lfbt8 || ! grep -q "BTW128_s1" .cache/run_lfw128.log 2>/dev/null; do sleep 300; done
$PY scripts/patches/bt9_idskip.py
export RD_STATES=gsmb_mbr8_states.npy BT_ROLEF=gsmb_role_feats_mbr8.npy; B="BT_D=512 BT_NH=8 BT_DB=64 BT_CYCLOCK=1 BT_ZOOM=1 BT_ROLE=1 BT_T6OFF=1"
for S in 0 1; do env $B BT_IDSKIP=1 BT_STEPS=18000 BT_SAVE=1 BT_TAG=BT9_s$S BT_SEED=$S TOY_THREADS=3 $PY scripts/gsm_breath_reader.py 2>&1 | grep --line-buffered -v -i warn & done; wait
while systemctl --user is-active -q lf8m; do sleep 300; done
$PY - <<'PY'
import json
R = {a: [json.load(open(f".cache/gsmb_breath_{a}_s{s}.json")) for s in (0, 1)] for a in ("BT9", "BT4M8")}
f0 = [r["ORIG_step_all"] for r in R["BT9"]]
L = [f"B9-0 FIT: BT9 ORIG step {f0[0]:.4f} / {f0[1]:.4f} (bar 0.90 both) -> {'PASS' if min(f0) >= 0.90 else 'FAIL'}"]
for bar, m in (("B9-1", "TEST_type_ok"), ("B9-2", "TEST_step_all")):
    d = [a[m] - b[m] for a, b in zip(R["BT9"], R["BT4M8"])]; mn = sum(d) / 2
    L.append(f"{bar}: {m} BT9 - BT4M8 {d[0]:+.4f} / {d[1]:+.4f} (mean {mn:+.4f}; bar +0.010, both > 0) -> {'PASS' if mn >= 0.010 and min(d) > 0 else 'FAIL'}")
for m in ("CFNAME_step_all", "TEST_step_op", "TEST_ans"):
    L.append(f"  beside {m}: BT9 " + " / ".join(f"{r[m]:.4f}" for r in R["BT9"]) + " vs BT4M8 " + " / ".join(f"{r[m]:.4f}" for r in R["BT4M8"]))
print("\n".join(L)); open(".cache/gsm_bt9_verdict.txt", "w").write("\n".join(L) + "\n")
PY
echo "== done $(date +%H:%M) =="
