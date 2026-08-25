#!/bin/bash
set -eo pipefail
cd /home/bryce/mycelium
PY=.venv/bin/python3
echo "== LEVER 3 FIRE: gsb227_opc (the op-multiset count head) =="
env DEV=AMD ALG2=1 ALG_FTYPES=9 ALG_DUP=1 ALG_HW=512 ALG_WIDE=1 \
    ALG_BREATH=7 ALG_NOTEBOOK=1 ALG_SIXWAVE=1 ALG_OPCOUNT=${OPC_MODE:-1} ALG_ALLOW_PEN_TRAIN=1 \
    WARM_FROM=.cache/gsb227_real.safetensors \
    ALG_TRAIN=.cache/form_mix8.jsonl ALG_TRAIN_NAME=form8 \
    ALG_TEST=.cache/algebra_nl_test.jsonl ALG_TEST_NAME=test23 \
    ALG_CKPT=.cache/${OPC_CKPT:-gsb227_opc}.safetensors STEPS=${STEPS:-4000} LR=1e-5 BATCH=8 SEED=${OPC_SEED:-170} SNAP_EVERY=0 \
    $PY scripts/phase1_algebra_head.py --train | tail -3
echo "== OPC FIRE COMPLETE =="
