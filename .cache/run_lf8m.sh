#!/usr/bin/bash
# ALL IN ON 8M (Bryce 11-03 "lets go all in on 8m"): the BT benches move to the 8M breathing reader's states (MBR, mbr2x4).
# GSM8K export (GPU, after lfbeam8) -> role feats -> BT4 s0/s1 18k on the new states (CY's config) -> P8 verdict vs BT4T9NS.
set -eo pipefail; cd /home/bryce/mycelium; exec > .cache/run_lf8m.log 2>&1
echo "== start $(date +%H:%M) =="; PY=.venv/bin/python3
while systemctl --user is-active -q lfbeam8; do sleep 60; done
A="DEV=PCI+AMD DR_FF=0 DR_D=768 DR_LAYERS=2 DR_DB=1 DR_T6=3 DR_T9NS=1 DR_BREATHS=4 DR_TAG=mbr2x4"
[ -e .cache/gsmb_mbr8_states.npy ] || flock -w 36000 .cache/gpu.lock env $A $PY scripts/distill_reader.py --gsm .cache/gsmb_mbr8_states.npy 2>&1 | grep -v -i warn
[ -e .cache/gsmb_role_feats_mbr8.npy ] || RF_STATES=gsmb_mbr8_states.npy RF_OUT=gsmb_role_feats_mbr8.npy $PY scripts/gsm_role_feats.py 2>&1 | grep -v -i warn
export RD_STATES=gsmb_mbr8_states.npy BT_ROLEF=gsmb_role_feats_mbr8.npy; B4="BT_D=512 BT_NH=8 BT_DB=64 BT_CYCLOCK=1 BT_ZOOM=1 BT_ROLE=1 BT_T6OFF=1"
echo "states $(date +%H:%M)"
for S in 0 1; do env $B4 BT_STEPS=18000 BT_SAVE=1 BT_TAG=BT4M8_s$S BT_SEED=$S TOY_THREADS=6 $PY scripts/gsm_breath_reader.py 2>&1 | grep --line-buffered -v -i warn & done; wait
$PY - <<'PY'
import re
f = lambda log, tag: [float(re.search(r"TEST ans [0-9.]+ step ([0-9.]+)", l).group(1)) for l in open(log) if f"{tag}_s" in l and "FINAL" in l]
new, old = f(".cache/run_lf8m.log", "BT4M8"), f(".cache/run_lfcy.log", "BT4T9NS")
assert len(new) == 2 and len(old) == 2, (new, old)
mn, mo = sum(new) / 2, sum(old) / 2
print(f"P8: BT4 TEST step on 8M-reader states {new[0]:.3f} / {new[1]:.3f} (mean {mn:.4f}) vs BT4T9NS {old[0]:.3f} / {old[1]:.3f} (mean {mo:.4f}); "
      f"diff {mn - mo:+.4f}; bar >= -0.010 -> {'PASS' if mn - mo >= -0.010 else 'FAIL'}")
PY
echo "== done $(date +%H:%M) =="
