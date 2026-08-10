#!/bin/bash
set -eo pipefail
cd /home/bryce/mycelium
export DEV=AMD ALG2=1 ALG_FTYPES=8 ALG_DUP=1 ALG_HW=512 ALG_WIDE=1 ALG_ALLOW_PEN_TRAIN=1
export ALG_TEST=.cache/algebra_nl_test.jsonl ALG_TEST_NAME=test23
export RATION_FILE=.cache/size_sliver_idx.json RATION_W=8
PY=.venv/bin/python3
echo "=== DOSE [size8x]: 4k from g23v5, ration 8x hot-phase ==="
env WARM_FROM=.cache/g23v5.safetensors ALG_TRAIN=.cache/size_mix.jsonl ALG_TRAIN_NAME=size \
    ALG_CKPT=.cache/g35_size8x.safetensors STEPS=4000 LR=1e-4 BATCH=8 SEED=127 SNAP_EVERY=0 \
    $PY scripts/phase1_algebra_head.py --train | tail -2
echo "=== DOSE BURNED ==="
