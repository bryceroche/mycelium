#!/bin/bash
# pool_round1.sh — round 1 of the reader bank: 24 sequential auditions.
set -o pipefail
cd /home/bryce/mycelium
for ID in R01 R02 R03 R04 R05 R06 R07 R08 R09 R10 R11 R12 \
          R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24; do
  echo "==== ROUND1 $ID START $(date +%H:%M) ===="
  if ./scripts/pool_run.sh $ID; then
    echo "==== ROUND1 $ID DONE $(date +%H:%M) ===="
  else
    echo "==== ROUND1 $ID FAILED (continuing) ===="
  fi
done
echo "==== ROUND 1 COMPLETE ===="
