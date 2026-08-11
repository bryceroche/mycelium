#!/bin/bash
set -eo pipefail
cd /home/bryce/mycelium
export DEV=AMD ALG2=1 ALG_FTYPES=8 ALG_DUP=1 ALG_HW=512 ALG_WIDE=1 ALG_ALLOW_PEN_TRAIN=1
PY=.venv/bin/python3
for ARM in H B; do
  if [ "$ARM" = "H" ]; then R2=.cache/ration42_t3_idx.json; else R2=.cache/ration46_t3_idx.json; fi
  echo "=== JITTER ARM-$ARM (seed 227) ==="
  env ALG_TEST=.cache/algebra_nl_test.jsonl ALG_TEST_NAME=test23 \
      WARM_FROM=.cache/g23v5.safetensors ALG_TRAIN=.cache/form_mix3.jsonl ALG_TRAIN_NAME=form3 \
      ALG_CKPT=.cache/g47_jit${ARM}.safetensors ALG_FREEZE_DUP=1 STEPS=4000 LR=1e-4 BATCH=8 SEED=227 SNAP_EVERY=0 \
      RATION_FILE=.cache/ration41_idx.json RATION_W=8 \
      RATION_FILE2=$R2 RATION_W2=3 \
      $PY scripts/phase1_algebra_head.py --train | tail -1
  echo "=== ARM-$ARM burned; rite + wall read ==="
  env CK=.cache/g23v5.safetensors OUT_JSON=.cache/calib_c47.json $PY scripts/calibrated_rite.py | grep headroom
  env CK=.cache/g47_jit${ARM}.safetensors OUT_JSON=.cache/calib_j${ARM}.json CK_OUT=.cache/g47_jit${ARM}_refold.safetensors $PY scripts/calibrated_rite.py | grep -E "headroom|fold-ready"
  env CK=.cache/g47_jit${ARM}_refold.safetensors $PY scripts/dup_axis_scan2.py | grep "^\[scan\]"
  env ND=0 OPAUT_NAME=jit${ARM} OPAUT_CK=.cache/g47_jit${ARM}_refold.safetensors $PY scripts/op_autopsy.py 2>/dev/null | grep -E "gold=mul|op-miss"
done
echo "== JITTER DISCRIMINATOR COMPLETE =="
