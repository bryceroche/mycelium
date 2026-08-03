#!/bin/bash
# gen23_fire.sh — THE GEN-23 FIRE (2026-08-03, the word given; the
# ANSWER_SPACE_SPEC's generation: E1+E3 behind ALG_WIDE, padwarm from
# g22, dose 600x10=6.8% per the declaration; bars B1(-5 pinned)/B2/B3).
set -eo pipefail
cd /home/bryce/mycelium
export DEV=AMD ALG2=1 ALG_FTYPES=8 ALG_DUP=1 ALG_HW=512 ALG_WIDE=1 ALG_ALLOW_PEN_TRAIN=1
export ALG_TEST=.cache/algebra_nl_test.jsonl ALG_TEST_NAME=test23
PY=.venv/bin/python3

echo "=== GEN-23: prep (mix + states + gold + padwarm) ==="
$PY scripts/gen23_fire_prep.py

echo "=== GEN-23: fire (4x4k from padwarm, LR 1e-4) ==="
for seg in 1 2 3 4; do
  if [ $seg -eq 1 ]; then W="WARM_FROM=.cache/g23_padwarm_init.safetensors"; else W="RESUME=1"; fi
  env $W ALG_TRAIN=.cache/gen23_mix.jsonl ALG_TRAIN_NAME=gen23 \
      ALG_CKPT=.cache/g23.safetensors STEPS=4000 LR=1e-4 BATCH=8 \
      SEED=230${seg} SNAP_EVERY=2000 $PY scripts/phase1_algebra_head.py --train
done
echo "=== GEN-23 IS BURNED — battery next; g22 REMAINS THE GATE until a verdict script prints PROMOTED ==="
