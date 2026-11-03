#!/usr/bin/bash
# BT6 THE CONFIDENCE-GATED COLLAPSE (ledger 11-02 18:22; word given): zero-training read on BT5/BT4 s0/s1 weights -> gsm_bt6_verdict.py
set -eo pipefail; cd /home/bryce/mycelium; exec > .cache/run_lfbt6.log 2>&1
echo "== start $(date +%H:%M) =="; PY=.venv/bin/python3; export RD_STATES=gsmb_student_states.npy
B4="BT_D=512 BT_NH=8 BT_DB=64 BT_CYCLOCK=1 BT_ZOOM=1 BT_ROLE=1 BT_T6OFF=1"
for A in BT5 BT4; do
  B=$B4; [ $A = BT5 ] && B="$B4 BT_SOFT=1"
  for S in 0 1; do env $B BT_LOAD=.cache/gsmb_breath_${A}_s$S.pt TOY_THREADS=3 $PY scripts/gsm_bt6_read.py 2>&1 | grep --line-buffered -v -i warn & done; wait
done
$PY scripts/gsm_bt6_verdict.py
echo "== done $(date +%H:%M) =="
