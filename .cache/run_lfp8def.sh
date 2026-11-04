#!/bin/bash
# P8 DEFAULTS (ledger 11-04): apply scripts/patches/p8_defaults.py once lfbt9 has applied bt9_idskip.py and launched both BT9 seeds.
set -eo pipefail
cd /home/bryce/mycelium
echo "== start $(date +%H:%M) =="
until grep -q "\[bt\] BT9_s1 " .cache/run_lfbt9.log 2>/dev/null; do
  systemctl --user is-active -q lfbt9 || { echo "lfbt9 ended without launching BT9_s1; not applying"; exit 1; }
  sleep 120
done
.venv/bin/python3 scripts/patches/p8_defaults.py
echo "== done $(date +%H:%M) =="
