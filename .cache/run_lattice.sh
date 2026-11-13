#!/usr/bin/bash
# TIER-LATTICE (ledger 11-12 queue, item 3): views -> RB8 pooled states (inference, GPU lock) -> measure (CPU)
set -eo pipefail; cd /home/bryce/mycelium; exec >> .cache/run_lattice.log 2>&1
echo "== start $(date +%H:%M) =="; PY=.venv/bin/python3
A="DEV=PCI+AMD DR_FF=0 DR_D=768 DR_LAYERS=2 DR_DB=1 DR_T6=3 DR_T9NS=1 DR_BREATHS=4 DR_RB=1"
flock -w 36000 .cache/gpu.lock env $A DR_TAG=rb8x4 DR_LOCAL_OUT=.cache/tier_lattice_states.npy $PY scripts/distill_reader.py --local .cache/tier_lattice_views.json
env DEV=CPU $PY scripts/tier_lattice_read.py --measure .cache/tier_lattice_states.npy
echo "== done $(date +%H:%M) =="
