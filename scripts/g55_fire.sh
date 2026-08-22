#!/bin/bash
set -eo pipefail
cd /home/bryce/mycelium
export DEV=AMD ALG2=1 ALG_DUP=1 ALG_HW=512 ALG_WIDE=1 ALG_ALLOW_PEN_TRAIN=1
PY=.venv/bin/python3
echo "== ASSEMBLE form15 (mixture: deeds->base, surgery+rescue->diet) =="
env ALG_FTYPES=8 DIET_REPS=${DIET_REPS:-3} $PY scripts/form_assemble15.py
echo "== G55 BRIDGE: gentle continuation on form_mix15 =="
env ALG_FTYPES=8 RESUME=1 ALG_TRAIN=.cache/form_mix15.jsonl ALG_TRAIN_NAME=form15 \
    ALG_TEST=.cache/algebra_nl_test.jsonl ALG_TEST_NAME=test23 \
    ALG_CKPT=.cache/g55_bridge.safetensors ALG_FREEZE_DUP=1 STEPS=4000 LR=2.5e-5 BATCH=8 SEED=145 SNAP_EVERY=250 \
    $PY scripts/phase1_algebra_head.py --train | tail -1
echo "== RING (bigtest straight) =="
env ALG_FTYPES=8 ALG_CKPT=.cache/g55_bridge.safetensors ALG_TEST=.cache/algebra_nl_bigtest.jsonl ALG_TEST_NAME=bigtest $PY scripts/phase1_algebra_head.py --eval | grep TOTAL
echo "== G55 COMPLETE =="
