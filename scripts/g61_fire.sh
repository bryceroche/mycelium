#!/bin/bash
set -eo pipefail
cd /home/bryce/mycelium
export DEV=AMD ALG2=1 ALG_DUP=1 ALG_HW=512 ALG_WIDE=1 ALG_ALLOW_PEN_TRAIN=1
PY=.venv/bin/python3
echo "== ASSEMBLE form20 (corpus 143, fixed wild-val excluded) =="
env ALG_FTYPES=8 DIET_REPS=1 $PY scripts/form_assemble20.py
echo "== G61 BRIDGE: gentle continuation on form_mix20 =="
env ALG_FTYPES=8 ALG_TRUNK_LORA=1 ALG_LORA_SCALE=8.0 RESUME=1 ALG_TRAIN=.cache/form_mix20.jsonl ALG_TRAIN_NAME=form20 \
    ALG_TEST=.cache/algebra_nl_test.jsonl ALG_TEST_NAME=test23 \
    ALG_CKPT=.cache/g61_bridge.safetensors ALG_FREEZE_DUP=1 STEPS=6000 LR=1e-5 BATCH=8 SEED=151 SNAP_EVERY=500 \
    $PY scripts/phase1_algebra_head.py --train | tail -1
echo "== RING (bigtest straight) =="
env ALG_FTYPES=8 ALG_TRUNK_LORA=1 ALG_CKPT=.cache/g61_bridge.safetensors ALG_TEST=.cache/algebra_nl_bigtest.jsonl ALG_TEST_NAME=bigtest $PY scripts/phase1_algebra_head.py --eval | grep TOTAL
echo "== G61 COMPLETE =="
