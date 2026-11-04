#!/usr/bin/bash
# BT GPU PORT TIMING SMOKE (ledger 11-03): scripts/gsm_breath_tg.py TG_MODE=time on the GPU, BT4 config, BATCH 64, 300 steps from the
# torch seed-0 init (no checkpoint saved; a full run needs Bryce's word). Queues behind RB8 on .cache/gpu.lock.
set -eo pipefail; cd /home/bryce/mycelium; exec > .cache/run_lftgtime.log 2>&1
echo "== start $(date +%H:%M) =="; PY=.venv/bin/python3
B="BT_D=512 BT_NH=8 BT_DB=64 BT_CYCLOCK=1 BT_ZOOM=1 BT_ROLE=1 BT_T6OFF=1"
flock -w 36000 .cache/gpu.lock env $B DEV=PCI+AMD TG_MODE=time TG_STEPS=300 TOY_THREADS=3 $PY scripts/gsm_breath_tg.py 2>&1 | grep -v -i warn
echo "torch CPU reference: ~1.1 s/step at 6 threads (BT7b log, 1000 steps / ~1100 s incl. evals)"
echo "== done $(date +%H:%M) =="
