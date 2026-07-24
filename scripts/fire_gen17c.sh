#!/bin/bash
set -eo pipefail
cd /home/bryce/mycelium
export DEV=AMD ALG2=1 ALG_FTYPES=8 ALG_DUP=1
PY=.venv/bin/python3
echo "=== G17C 1/3: precompute ==="
ALG_TRAIN=.cache/gen17c_mix.jsonl ALG_TRAIN_NAME=g17c PRECOMPUTE_ONLY=g17c $PY scripts/phase1_algebra_head.py --precompute
echo "=== G17C 2/3: SGDR 1/2 (4k warm from armR) ==="
ALG_TRAIN=.cache/gen17c_mix.jsonl ALG_TRAIN_NAME=g17c WARM_FROM=.cache/g17_armR.safetensors ALG_CKPT=.cache/g17c.safetensors STEPS=4000 LR=1e-4 BATCH=8 SEED=23 $PY scripts/phase1_algebra_head.py --train
echo "=== G17C 3/3: SGDR 2/2 (RESUME, SNAP_EVERY=500) ==="
ALG_TRAIN=.cache/gen17c_mix.jsonl ALG_TRAIN_NAME=g17c RESUME=1 ALG_CKPT=.cache/g17c.safetensors STEPS=4000 LR=1e-4 BATCH=8 SEED=24 SNAP_EVERY=500 $PY scripts/phase1_algebra_head.py --train
echo "=== THE DOSE IS BURNED — g17c + snapshots banked ==="
