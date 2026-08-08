#!/bin/bash
set -eo pipefail
cd /home/bryce/mycelium
export DEV=AMD ALG2=1 ALG_FTYPES=8 ALG_DUP=1 ALG_HW=512 ALG_WIDE=1 ALG_ALLOW_PEN_TRAIN=1
export ALG_TEST=.cache/algebra_nl_test.jsonl ALG_TEST_NAME=test23

PY=.venv/bin/python3
SEED=117
for ARM in ictl install; do
  SEED=$((SEED+1))
  if [ "$ARM" = "install" ]; then
    export ALG_DUPPTR=1
  else
    export ALG_DUPPTR=0
  fi
  echo "=== PHASE2 [$ARM]: 4k from g23 ==="
  env WARM_FROM=.cache/g23.safetensors ALG_TRAIN=.cache/augfire_vdup19_mix.jsonl \
      ALG_TRAIN_NAME=vdup19 ALG_CKPT=.cache/g30_inst_${ARM}.safetensors \
      STEPS=4000 LR=1e-4 BATCH=8 SEED=${SEED} SNAP_EVERY=0 \
      $PY scripts/phase1_algebra_head.py --train | tail -2
done
echo "=== PHASE2 BURNED ==="
