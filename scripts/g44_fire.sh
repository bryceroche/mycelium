#!/bin/bash
set -eo pipefail
cd /home/bryce/mycelium
export DEV=AMD ALG2=1 ALG_DUP=1 ALG_HW=512 ALG_WIDE=1 ALG_ALLOW_PEN_TRAIN=1
PY=.venv/bin/python3
echo "== ASSEMBLE form10 =="
env ALG_FTYPES=8 $PY scripts/form_assemble10.py
echo "== G42 BRIDGE: gentle continuation on form_mix10 =="
env ALG_FTYPES=8 RESUME=1 ALG_TRAIN=.cache/form_mix10.jsonl ALG_TRAIN_NAME=form10 \
    ALG_TEST=.cache/algebra_nl_test.jsonl ALG_TEST_NAME=test23 \
    ALG_CKPT=.cache/g44_bridge.safetensors ALG_FREEZE_DUP=1 STEPS=4000 LR=2.5e-5 BATCH=8 SEED=134 SNAP_EVERY=0 \
    $PY scripts/phase1_algebra_head.py --train | tail -1
echo "== RING (bigtest straight) =="
env ALG_FTYPES=8 ALG_CKPT=.cache/g44_bridge.safetensors ALG_TEST=.cache/algebra_nl_bigtest.jsonl ALG_TEST_NAME=bigtest $PY scripts/phase1_algebra_head.py --eval | grep TOTAL
echo "== G42 COMPLETE =="
