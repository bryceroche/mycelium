#!/usr/bin/bash
# BIC-TREE (ledger 11-12 queue, item 1): STACK on the x-means Tree + law-only admission; then the LAW attribution arm (KS + law)
set -eo pipefail; cd /home/bryce/mycelium; exec >> .cache/run_bic.log 2>&1; echo "== start $(date +%H:%M) =="
for st in local full; do pids=""; for k in 0 1 2 3; do
  LADDER_TREE=bic LADDER_ADMIT=law LADDER_HYB=1 LADDER_CHAIN=1 LADDER_BEAM=4 LADDER_MOTIF=class LADDER_STATES=$st LADDER_SEED=$k DEV=CPU .venv/bin/python3 scripts/gsm_ladder_read.py > /dev/null & pids="$pids $!"
done; for p in $pids; do wait $p; done; echo "bic $st done $(date +%H:%M)"; done
pids=""; for st in local full; do for k in 0 1 2 3; do
  LADDER_ADMIT=law LADDER_HYB=1 LADDER_CHAIN=1 LADDER_BEAM=4 LADDER_MOTIF=class LADDER_STATES=$st LADDER_SEED=$k DEV=CPU .venv/bin/python3 scripts/gsm_ladder_read.py > /dev/null & pids="$pids $!"
done; done; for p in $pids; do wait $p; done
echo "== done $(date +%H:%M) =="
