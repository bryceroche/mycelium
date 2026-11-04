#!/usr/bin/bash
# BT8b THE CHAINED BINDING (ledger 11-04; word given): BT_FACT=2 on tinygrad / the GPU (gsm_breath_tg.py TG_MODE=train, scored by the torch
# scorer), BT4 config on the 8M-reader states, s0 then s1 sequentially, each under the GPU lock; controls BT8 + BT4M8 (torch, banked).
set -eo pipefail; cd /home/bryce/mycelium; exec >> .cache/run_lfbt8b.log 2>&1
echo "== start $(date +%H:%M) =="; PY=.venv/bin/python3
export RD_STATES=gsmb_mbr8_states.npy BT_ROLEF=gsmb_role_feats_mbr8.npy; B="BT_D=512 BT_NH=8 BT_DB=64 BT_CYCLOCK=1 BT_ZOOM=1 BT_ROLE=1 BT_T6OFF=1"
for S in 0 1; do
  flock -w 36000 .cache/gpu.lock env $B BT_FACT=2 DEV=PCI+AMD TG_MODE=train BT_SEED=$S BT_STEPS=18000 BT_SAVE=1 BT_TAG=BT8b_s$S TOY_THREADS=4 \
    $PY scripts/gsm_breath_tg.py 2>&1 | grep --line-buffered -v -i warn
done
$PY - <<'PY'
import json
R = {a: [json.load(open(f".cache/gsmb_breath_{a}_s{s}.json")) for s in (0, 1)] for a in ("BT8b", "BT8", "BT4M8")}
f0 = [r["ORIG_step_all"] for r in R["BT8b"]]
L = [f"B8b-0 FIT: BT8b ORIG step {f0[0]:.4f} / {f0[1]:.4f} (bar 0.90 both) -> {'PASS' if min(f0) >= 0.90 else 'FAIL'}"]
for bar, ctl, m in (("B8b-C", "BT8", "TEST_type_ok"), ("B8b-1", "BT4M8", "TEST_type_ok"), ("B8b-2", "BT4M8", "TEST_step_all")):
    d = [a[m] - b[m] for a, b in zip(R["BT8b"], R[ctl])]; mn = sum(d) / 2
    L.append(f"{bar}: {m} BT8b - {ctl} {d[0]:+.4f} / {d[1]:+.4f} (mean {mn:+.4f}; bar +0.010, both > 0) -> {'PASS' if mn >= 0.010 and min(d) > 0 else 'FAIL'}")
for m in ("ORIG_step_all", "CFNAME_step_all", "TEST_step_op", "TEST_ans"):
    L.append(f"  beside {m}: " + " | ".join(f"{a} " + " / ".join(f"{r[m]:.4f}" for r in R[a]) for a in ("BT8b", "BT8", "BT4M8")))
print("\n".join(L)); open(".cache/gsm_bt8b_verdict.txt", "w").write("\n".join(L) + "\n")
PY
echo "== done $(date +%H:%M) =="
