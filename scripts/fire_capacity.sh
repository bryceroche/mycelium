#!/bin/bash
# fire_capacity.sh — THE CAPACITY FIRE (2026-07-24): ALG_HW=1024 cold
# start via the quench rite; gen-14 recipe at full strength; states
# reused (trunk-side, H_W-invariant); ration per the proven clause.
set -eo pipefail
cd /home/bryce/mycelium
export DEV=AMD ALG2=1 ALG_FTYPES=8 ALG_DUP=1 ALG_HW=1024
PY=.venv/bin/python3
MIX=.cache/gen18b_mix.jsonl
echo "=== CAPACITY: 32k cold at H_W=1024 (quench rite) ==="
ALG_TRAIN=$MIX ALG_TRAIN_NAME=g18b ALG_CKPT=.cache/g19_w1024.safetensors STEPS=32000 LR=3e-4 BATCH=8 SEED=1024 SNAP_EVERY=2000 RATION_FILE=.cache/gen18b_ration_idx.json RATION_W=1.5 $PY scripts/phase1_algebra_head.py --train
echo "=== THE CAPACITY FIRE IS BURNED — g19_w1024 banked ==="
