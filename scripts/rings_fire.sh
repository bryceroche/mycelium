#!/bin/bash
set -eo pipefail
cd /home/bryce/mycelium
PY=.venv/bin/python3
echo "== V2.5 FIRE: gsb227_rings (wake the revoke port — organ-2) =="
env DEV=AMD ALG2=1 ALG_FTYPES=9 ALG_DUP=1 ALG_HW=512 ALG_WIDE=1 \
    ALG_BREATH=7 ALG_NOTEBOOK=1 ALG_SIXWAVE=1 ALG_RINGS=1 ALG_XOUT=1 \
    ALG_ALLOW_PEN_TRAIN=1 \
    WARM_FROM=.cache/gsb227_real.safetensors \
    ALG_TRAIN=.cache/form_mix8.jsonl ALG_TRAIN_NAME=form8 \
    ALG_TEST=.cache/algebra_nl_test.jsonl ALG_TEST_NAME=test23 \
    ALG_CKPT=.cache/gsb227_rings.safetensors STEPS=${STEPS:-4000} LR=${LR:-1e-5} BATCH=8 SEED=172 SNAP_EVERY=0 \
    $PY scripts/phase1_algebra_head.py --train | tail -3
echo "== RINGS FIRE COMPLETE =="
