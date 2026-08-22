#!/bin/bash
set -eo pipefail
cd /home/bryce/mycelium
export DEV=AMD ALG2=1 ALG_DUP=1 ALG_HW=512 ALG_WIDE=1 ALG_ALLOW_PEN_TRAIN=1
PY=.venv/bin/python3
echo "== ASSEMBLE form22 (THE JOINT SCALE DIET) =="
env ALG_FTYPES=8 ALG_TRUNK_LORA=1 $PY scripts/form_assemble22.py
echo "== G63: THE JOINT SCALE FIRE =="
env ALG_FTYPES=8 ALG_TRUNK_LORA=1 ALG_LORA_SCALE=8.0 RESUME=1 ALG_TRAIN=.cache/form_mix22.jsonl ALG_TRAIN_NAME=form22 \
    ALG_TEST=.cache/algebra_nl_test.jsonl ALG_TEST_NAME=test23 \
    ALG_CKPT=.cache/g63_bridge.safetensors ALG_FREEZE_DUP=1 STEPS=20000 LR=1e-5 BATCH=8 SEED=153 SNAP_EVERY=2000 \
    $PY scripts/phase1_algebra_head.py --train | tail -1
echo "== HONEST JOINT RING (bigtest through adapters) =="
env ALG_FTYPES=8 ALG_TRUNK_LORA=1 ALG_CKPT=.cache/g63_bridge.safetensors ALG_TEST=.cache/algebra_nl_bigtest.jsonl ALG_TEST_NAME=bigtest $PY scripts/phase1_algebra_head.py --eval | grep TOTAL
echo "== G63 COMPLETE =="
