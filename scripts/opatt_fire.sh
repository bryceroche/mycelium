#!/bin/bash
# opatt_fire.sh — DUP STAGING CURE, ARM 1 (2026-08-07; registration
# 797666887c04 predates the design). Two arms from g23 on gen23_mix,
# gentle continuation; only delta = the operand-attention target.
set -eo pipefail
cd /home/bryce/mycelium
export DEV=AMD ALG2=1 ALG_FTYPES=8 ALG_DUP=1 ALG_HW=512 ALG_WIDE=1 ALG_ALLOW_PEN_TRAIN=1
export ALG_TEST=.cache/algebra_nl_test.jsonl ALG_TEST_NAME=test23
PY=.venv/bin/python3
SEED=107
for ARM in ctl opatt; do
  SEED=$((SEED+1))
  if [ "$ARM" = "opatt" ]; then export ALG_OPATT=1 OPATT_GOLD=.cache/opatt_gold.npy; else export ALG_OPATT=0; fi
  echo "=== OPATT [$ARM]: 4k from g23 ==="
  env WARM_FROM=.cache/g23.safetensors ALG_TRAIN=.cache/gen23_mix.jsonl \
      ALG_TRAIN_NAME=gen23 ALG_CKPT=.cache/g26_opatt_${ARM}.safetensors \
      STEPS=4000 LR=1e-4 BATCH=8 SEED=${SEED} SNAP_EVERY=0 \
      $PY scripts/phase1_algebra_head.py --train | tail -2
done
echo "=== OPATT BURNED — probes + timing re-run next ==="
