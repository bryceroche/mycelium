#!/bin/bash
set -eo pipefail
cd /home/bryce/mycelium
export DEV=AMD ALG2=1 ALG_FTYPES=8 ALG_DUP=1 ALG_HW=512 ALG_WIDE=1 ALG_ALLOW_PEN_TRAIN=1
export ALG_TEST=.cache/algebra_nl_test.jsonl ALG_TEST_NAME=test23
export ALG_FREEZE_DUP=1
PY=.venv/bin/python3
SEED=123
for ARM in bctl breath; do
  SEED=$((SEED+1))
  if [ "$ARM" = "breath" ]; then export ALG_BREATH=3; else export ALG_BREATH=1; fi
  echo "=== BREATH [$ARM]: 4k from g23v5 (h_dup frozen) ==="
  env WARM_FROM=.cache/g23v5.safetensors ALG_TRAIN=.cache/gen23_mix.jsonl ALG_TRAIN_NAME=gen23 \
      ALG_CKPT=.cache/g33_breath_${ARM}.safetensors STEPS=4000 LR=1e-4 BATCH=8 SEED=${SEED} SNAP_EVERY=0 \
      $PY scripts/phase1_algebra_head.py --train | tail -2
done
echo "=== BREATH BURNED ==="
