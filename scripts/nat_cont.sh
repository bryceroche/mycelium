#!/bin/bash
set -eo pipefail
cd /home/bryce/mycelium
export DEV=AMD ALG2=1 ALG_DUP=1 ALG_HW=512 ALG_WIDE=1 ALG_ALLOW_PEN_TRAIN=1
PY=.venv/bin/python3
echo "== TWIN CONTINUATION (+10000) =="
env ALG_FTYPES=9 RESUME=1 ALG_TRAIN=.cache/form_mix8.jsonl ALG_TRAIN_NAME=form8 \
    ALG_TEST=.cache/algebra_nl_test.jsonl ALG_TEST_NAME=test23 \
    ALG_CKPT=.cache/gnat_twin.safetensors ALG_FREEZE_DUP=1 STEPS=10000 LR=1e-4 BATCH=8 SEED=128 SNAP_EVERY=0 \
    RATION_FILE=.cache/ration41_idx.json RATION_W=8 RATION_FILE2=.cache/ration60_t3_idx.json RATION_W2=3 \
    $PY scripts/phase1_algebra_head.py --train | tail -2
echo "== NATIVE CONTINUATION (+10000) =="
env ALG_FTYPES=9 ALG_SIXWAVE=1 ALG_BREATH=3 BREATH_NORM=1 RESUME=1 \
    ALG_TRAIN=.cache/form_mix8.jsonl ALG_TRAIN_NAME=form8 \
    ALG_TEST=.cache/algebra_nl_test.jsonl ALG_TEST_NAME=test23 \
    ALG_CKPT=.cache/gnat_native.safetensors ALG_FREEZE_DUP=1 STEPS=10000 LR=1e-4 BATCH=8 SEED=128 SNAP_EVERY=0 \
    RATION_FILE=.cache/ration41_idx.json RATION_W=8 RATION_FILE2=.cache/ration60_t3_idx.json RATION_W2=3 \
    $PY scripts/phase1_algebra_head.py --train | tail -2
echo "== SHEET 2: same bars + engagement column =="
env ALG_FTYPES=9 ALG_SIXWAVE=1 ALG_BREATH=3 BREATH_NORM=1 ALG_CKPT=.cache/gnat_native.safetensors ALG_TEST=.cache/algebra_nl_test.jsonl ALG_TEST_NAME=test23 $PY scripts/phase1_algebra_head.py --eval | grep TOTAL
env ALG_FTYPES=9 ALG_CKPT=.cache/gnat_twin.safetensors ALG_TEST=.cache/algebra_nl_test.jsonl ALG_TEST_NAME=test23 $PY scripts/phase1_algebra_head.py --eval | grep TOTAL
env ALG_FTYPES=9 ALG_SIXWAVE=1 ALG_BREATH=3 BREATH_NORM=1 OV_CK=.cache/gnat_native.safetensors ALG_TEST=.cache/algebra_nl_bigtest.jsonl ALG_TEST_NAME=bigtest $PY scripts/sw_overlap.py
env ALG_FTYPES=9 OV_CK=.cache/gnat_twin.safetensors ALG_TEST=.cache/algebra_nl_bigtest.jsonl ALG_TEST_NAME=bigtest $PY scripts/sw_overlap.py
env ALG_FTYPES=9 ALG_SIXWAVE=1 ALG_BREATH=3 BREATH_NORM=1 CK=.cache/gnat_native.safetensors $PY scripts/dup_axis_scan2.py | grep "^\[scan\]"
env ALG_FTYPES=9 CK=.cache/gnat_twin.safetensors $PY scripts/dup_axis_scan2.py | grep "^\[scan\]"
echo "== CONTINUATION COMPLETE =="
