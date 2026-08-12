#!/bin/bash
set -eo pipefail
cd /home/bryce/mycelium
PY=.venv/bin/python3
for G in "2.0 0" "4.0 0" "4.0 4.0"; do
  set -- $G
  echo "=== GATE PRES>$1 ARGMARGIN>$2 ==="
  env XG_PRES=$1 XG_ARG=$2 XG_OUT=.cache/xg_p${1}_a${2}.json $PY scripts/crossgrain_smoke.py | grep xgrain
done
echo "== GATE SWEEP COMPLETE =="
