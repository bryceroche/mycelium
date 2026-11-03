#!/usr/bin/bash
# BT7b BP POINTERS TRAINED TO COMPARE KINDS (ledger 11-03 05:43; word given): BT5 + BT_KINDCE, s0/s1 18k -> BT7a read on the weights -> verdict
set -eo pipefail; cd /home/bryce/mycelium; exec > .cache/run_lfbt7b.log 2>&1
echo "== start $(date +%H:%M) =="; PY=.venv/bin/python3; export RD_STATES=gsmb_student_states.npy
B="BT_D=512 BT_NH=8 BT_DB=64 BT_CYCLOCK=1 BT_ZOOM=1 BT_ROLE=1 BT_T6OFF=1 BT_SOFT=1 BT_KINDCE=1"
for S in 0 1; do env $B BT_STEPS=18000 BT_SAVE=1 BT_TAG=BT7b_s$S BT_SEED=$S TOY_THREADS=6 $PY scripts/gsm_breath_reader.py 2>&1 | grep --line-buffered -v -i warn & done; wait
for S in 0 1; do env $B BT_LOAD=.cache/gsmb_breath_BT7b_s$S.pt BT_KINDCE= TOY_THREADS=6 $PY scripts/gsm_bt7_read.py 2>&1 | grep --line-buffered -v -i warn & done; wait
$PY scripts/gsm_bt7b_verdict.py
echo "== done $(date +%H:%M) =="
