#!/usr/bin/bash
# BT8 DIAG (ledger 11-04): per-factor accuracy of the committed type, BT8 vs BT4M8, both seeds (zero-training, CPU).
set -eo pipefail; cd /home/bryce/mycelium; exec >> .cache/run_lfbt8diag.log 2>&1
echo "== start $(date +%H:%M) =="; PY=.venv/bin/python3
export RD_STATES=gsmb_mbr8_states.npy BT_ROLEF=gsmb_role_feats_mbr8.npy; B="BT_D=512 BT_NH=8 BT_DB=64 BT_CYCLOCK=1 BT_ZOOM=1 BT_ROLE=1 BT_T6OFF=1"
for S in 0 1; do
  env $B BT_FACT=1 BT_TAG=BT8_s$S BT_LOAD=.cache/gsmb_breath_BT8_s$S.pt TOY_THREADS=2 $PY scripts/gsm_bt8_diag.py 2>&1 | grep --line-buffered -v -i warn &
  env $B BT_TAG=BT4M8_s$S BT_LOAD=.cache/gsmb_breath_BT4M8_s$S.pt TOY_THREADS=2 $PY scripts/gsm_bt8_diag.py 2>&1 | grep --line-buffered -v -i warn &
done; wait
echo "== done $(date +%H:%M) =="
