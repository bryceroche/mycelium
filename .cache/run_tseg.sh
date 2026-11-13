#!/usr/bin/bash
# TREE-SEG (ledger 11-12 queue, item 2): tree spans -> local texts -> RB8 local states (inference, GPU lock) -> STACK x 8
set -eo pipefail; cd /home/bryce/mycelium; exec >> .cache/run_tseg.log 2>&1; echo "== start $(date +%H:%M) =="; PY=.venv/bin/python3
A="DEV=PCI+AMD DR_FF=0 DR_D=768 DR_LAYERS=2 DR_DB=1 DR_T6=3 DR_T9NS=1 DR_BREATHS=4 DR_RB=1"
env DEV=CPU LADDER_SPANS=tree LADDER_DUMP=1 $PY scripts/gsm_ladder_read.py
flock -w 36000 .cache/gpu.lock env $A DR_TAG=rb8x4 DR_LOCAL_OUT=.cache/gsmb_local_rb8_tree.npy $PY scripts/distill_reader.py --local .cache/gsm_ladder_local_texts_tree.json
for st in local full; do pids=""; for k in 0 1 2 3; do
  LADDER_SPANS=tree LADDER_HYB=1 LADDER_CHAIN=1 LADDER_BEAM=4 LADDER_MOTIF=class LADDER_STATES=$st LADDER_SEED=$k DEV=CPU $PY scripts/gsm_ladder_read.py > /dev/null & pids="$pids $!"
done; for p in $pids; do wait $p; done; echo "tseg $st done $(date +%H:%M)"; done
echo "== done $(date +%H:%M) =="
