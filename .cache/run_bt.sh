#!/usr/bin/bash
# THE BT ARM (ledger 2026-10-30, word given): BT s0/s1 in parallel on CPU, then the verdict
set -eo pipefail; cd /home/bryce/mycelium; exec > .cache/run_bt.log 2>&1
echo "== start $(date +%H:%M) =="
for S in 0 1; do
  BT_SEED=$S TOY_THREADS=6 .venv/bin/python3 scripts/gsm_breath_reader.py 2>&1 | grep --line-buffered -v -i warn &
done
wait
.venv/bin/python3 scripts/gsm_bt_verdict.py
echo "== done $(date +%H:%M) =="
