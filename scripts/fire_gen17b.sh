#!/bin/bash
# fire_gen17b.sh — THE GEN-17B CONTINUATION (2026-07-23, the kill's
# priced dose): 2x4k SGDR from g17_armR on the reps-raised mix.
set -eo pipefail
cd /home/bryce/mycelium
export DEV=AMD ALG2=1 ALG_FTYPES=8 ALG_DUP=1
PY=.venv/bin/python3

echo "=== G17B 1/3: precompute mix states ==="
ALG_TRAIN=.cache/gen17b_mix.jsonl ALG_TRAIN_NAME=g17b PRECOMPUTE_ONLY=g17b $PY scripts/phase1_algebra_head.py --precompute

echo "=== G17B 2/3: SGDR segment 1/2 (4k, warm from g17_armR) ==="
ALG_TRAIN=.cache/gen17b_mix.jsonl ALG_TRAIN_NAME=g17b WARM_FROM=.cache/g17_armR.safetensors ALG_CKPT=.cache/g17b.safetensors STEPS=4000 LR=1e-4 BATCH=8 SEED=21 $PY scripts/phase1_algebra_head.py --train

echo "=== G17B 3/3: SGDR segment 2/2 (RESUME, fresh cosine) ==="
ALG_TRAIN=.cache/gen17b_mix.jsonl ALG_TRAIN_NAME=g17b RESUME=1 ALG_CKPT=.cache/g17b.safetensors STEPS=4000 LR=1e-4 BATCH=8 SEED=22 $PY scripts/phase1_algebra_head.py --train

echo "=== THE DOSE IS BURNED — g17b banked; the battery speaks next ==="
