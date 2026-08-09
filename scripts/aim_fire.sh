#!/bin/bash
set -eo pipefail
cd /home/bryce/mycelium
export DEV=AMD ALG2=1 ALG_FTYPES=8 ALG_DUP=1 ALG_HW=512 ALG_WIDE=1 ALG_ALLOW_PEN_TRAIN=1
export ALG_TEST=.cache/algebra_nl_test.jsonl ALG_TEST_NAME=test23
PY=.venv/bin/python3
SEED=119
for ARM in actl aim; do
  SEED=$((SEED+1))
  if [ "$ARM" = "aim" ]; then MIX=.cache/aim_mix.jsonl; NAME=aim; else MIX=.cache/gen23_mix.jsonl; NAME=gen23; fi
  echo "=== AIM [$ARM]: 4k from g23 ==="
  env WARM_FROM=.cache/g23.safetensors ALG_TRAIN=$MIX ALG_TRAIN_NAME=$NAME \
      ALG_CKPT=.cache/g31_aim_${ARM}.safetensors STEPS=4000 LR=1e-4 BATCH=8 SEED=${SEED} SNAP_EVERY=0 \
      $PY scripts/phase1_algebra_head.py --train | tail -2
done
echo "=== AIM BURNED ==="
