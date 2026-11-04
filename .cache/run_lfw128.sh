#!/usr/bin/bash
# W128 THE WIDER WAIST (ledger 11-03; word given): BT4 config with BT_DB=128 (the twelve hills' 512/128), s0/s1 18k on the 8M-reader
# states; control = BT4M8 (lf8m). Starts when lf8m (BT4M8) finishes, so the CPU holds BT8 + W128.
set -eo pipefail; cd /home/bryce/mycelium; exec > .cache/run_lfw128.log 2>&1
echo "== start $(date +%H:%M) =="; PY=.venv/bin/python3
while systemctl --user is-active -q lf8m; do sleep 300; done
export RD_STATES=gsmb_mbr8_states.npy BT_ROLEF=gsmb_role_feats_mbr8.npy; B="BT_D=512 BT_NH=8 BT_DB=128 BT_CYCLOCK=1 BT_ZOOM=1 BT_ROLE=1 BT_T6OFF=1"
for S in 0 1; do env $B BT_STEPS=18000 BT_SAVE=1 BT_TAG=BTW128_s$S BT_SEED=$S TOY_THREADS=3 $PY scripts/gsm_breath_reader.py 2>&1 | grep --line-buffered -v -i warn & done; wait
$PY - <<'PY'
import json
R = {a: [json.load(open(f".cache/gsmb_breath_{a}_s{s}.json")) for s in (0, 1)] for a in ("BTW128", "BT4M8")}
L = []
for bar, m in (("W1", "TEST_type_ok"), ("W2", "TEST_step_all")):
    d = [a[m] - b[m] for a, b in zip(R["BTW128"], R["BT4M8"])]; mn = sum(d) / 2
    L.append(f"{bar}: {m} BTW128 - BT4M8 {d[0]:+.4f} / {d[1]:+.4f} (mean {mn:+.4f}; bar +0.010, both > 0) -> {'PASS' if mn >= 0.010 and min(d) > 0 else 'FAIL'}")
for m in ("ORIG_step_all", "CFNAME_step_all", "TEST_step_op"):
    L.append(f"  beside {m}: BTW128 " + " / ".join(f"{r[m]:.4f}" for r in R["BTW128"]) + " vs BT4M8 " + " / ".join(f"{r[m]:.4f}" for r in R["BT4M8"]))
print("\n".join(L)); open(".cache/gsm_w128_verdict.txt", "w").write("\n".join(L) + "\n")
PY
echo "== done $(date +%H:%M) =="
