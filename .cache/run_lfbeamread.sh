#!/usr/bin/bash
# BEAM THE PARSE-SIDE BEAM + CIRCUIT BREAKER (ledger 11-03 13:38; word given): full read on BT4T9NS s0/s1 (T9-NS defaults)
set -eo pipefail; cd /home/bryce/mycelium; exec > .cache/run_lfbeamread.log 2>&1
echo "== start $(date +%H:%M) =="; PY=.venv/bin/python3
B="BT_D=512 BT_NH=8 BT_DB=64 BT_CYCLOCK=1 BT_ZOOM=1 BT_ROLE=1 BT_T6OFF=1"
for S in 0 1; do env $B BT_LOAD=.cache/gsmb_breath_BT4T9NS_s$S.pt TOY_THREADS=6 $PY scripts/gsm_beam_read.py 2>&1 | grep --line-buffered -v -i warn & done; wait
echo "== done $(date +%H:%M) =="
