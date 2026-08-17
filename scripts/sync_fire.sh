#!/bin/bash
set -eo pipefail
cd /home/bryce/mycelium
export DEV=AMD ALG2=1 ALG_DUP=1 ALG_HW=512 ALG_WIDE=1 ALG_ALLOW_PEN_TRAIN=1
PY=.venv/bin/python3
echo "== ASSEMBLE form8 (VALATT+LSENT gold restage) =="
env ALG_FTYPES=9 ALG_SYNC=1 $PY scripts/form_assemble8.py
echo "== V2 ARM (letter-key + deepsup, from birth) =="
env ALG_FTYPES=9 ALG_SIXWAVE=1 ALG_BREATH=3 BREATH_NORM=1 ALG_SYNC=1 \
    ALG_TRAIN=.cache/form_mix8.jsonl ALG_TRAIN_NAME=form8 \
    ALG_TEST=.cache/algebra_nl_test.jsonl ALG_TEST_NAME=test23 \
    ALG_CKPT=.cache/gsync_arm.safetensors ALG_FREEZE_DUP=1 STEPS=4000 LR=1e-4 BATCH=8 SEED=127 SNAP_EVERY=0 \
    RATION_FILE=.cache/ration41_idx.json RATION_W=8 RATION_FILE2=.cache/ration60_t3_idx.json RATION_W2=3 \
    $PY scripts/phase1_algebra_head.py --train | tail -2
echo "== V2 CONTINUATION (+10000) =="
env ALG_FTYPES=9 ALG_SIXWAVE=1 ALG_BREATH=3 BREATH_NORM=1 ALG_SYNC=1 RESUME=1 \
    ALG_TRAIN=.cache/form_mix8.jsonl ALG_TRAIN_NAME=form8 \
    ALG_TEST=.cache/algebra_nl_test.jsonl ALG_TEST_NAME=test23 \
    ALG_CKPT=.cache/gsync_arm.safetensors ALG_FREEZE_DUP=1 STEPS=10000 LR=1e-4 BATCH=8 SEED=128 SNAP_EVERY=0 \
    RATION_FILE=.cache/ration41_idx.json RATION_W=8 RATION_FILE2=.cache/ration60_t3_idx.json RATION_W2=3 \
    $PY scripts/phase1_algebra_head.py --train | tail -2
echo "== V2 SHEET (frozen bars) =="
env ALG_FTYPES=9 ALG_SIXWAVE=1 ALG_BREATH=3 BREATH_NORM=1 ALG_SYNC=1 ALG_CKPT=.cache/gsync_arm.safetensors ALG_TEST=.cache/algebra_nl_test.jsonl ALG_TEST_NAME=test23 $PY scripts/phase1_algebra_head.py --eval | grep TOTAL
env ALG_FTYPES=9 ALG_SIXWAVE=1 ALG_BREATH=3 BREATH_NORM=1 ALG_SYNC=1 CK=.cache/gsync_arm.safetensors $PY scripts/dup_axis_scan2.py | grep "^\[scan\]"
env ALG_FTYPES=9 ALG_SIXWAVE=1 ALG_BREATH=3 BREATH_NORM=1 ALG_SYNC=1 OV_CK=.cache/gsync_arm.safetensors ALG_TEST=.cache/algebra_nl_bigtest.jsonl ALG_TEST_NAME=bigtest $PY scripts/sw_overlap.py
echo "== V2 FIRE COMPLETE =="
echo "== SCRAMBLED-CLOCK ARM (the placebo, to maturity) =="
env ALG_FTYPES=9 ALG_SYNC=1 SYNC_SCRAMBLE=1 ALG_BREATH=3 BREATH_NORM=1 \
    ALG_TRAIN=.cache/form_mix8.jsonl ALG_TRAIN_NAME=form8 \
    ALG_TEST=.cache/algebra_nl_test.jsonl ALG_TEST_NAME=test23 \
    ALG_CKPT=.cache/gsyncscr_arm.safetensors ALG_FREEZE_DUP=1 STEPS=4000 LR=1e-4 BATCH=8 SEED=127 SNAP_EVERY=0 \
    RATION_FILE=.cache/ration41_idx.json RATION_W=8 RATION_FILE2=.cache/ration60_t3_idx.json RATION_W2=3 \
    .venv/bin/python3 scripts/phase1_algebra_head.py --train | tail -1
env ALG_FTYPES=9 ALG_SYNC=1 SYNC_SCRAMBLE=1 ALG_BREATH=3 BREATH_NORM=1 RESUME=1 \
    ALG_TRAIN=.cache/form_mix8.jsonl ALG_TRAIN_NAME=form8 \
    ALG_TEST=.cache/algebra_nl_test.jsonl ALG_TEST_NAME=test23 \
    ALG_CKPT=.cache/gsyncscr_arm.safetensors ALG_FREEZE_DUP=1 STEPS=10000 LR=1e-4 BATCH=8 SEED=128 SNAP_EVERY=0 \
    RATION_FILE=.cache/ration41_idx.json RATION_W=8 RATION_FILE2=.cache/ration60_t3_idx.json RATION_W2=3 \
    .venv/bin/python3 scripts/phase1_algebra_head.py --train | tail -1
env ALG_FTYPES=9 ALG_SYNC=1 SYNC_SCRAMBLE=1 ALG_BREATH=3 BREATH_NORM=1 ALG_CKPT=.cache/gsyncscr_arm.safetensors ALG_TEST=.cache/algebra_nl_test.jsonl ALG_TEST_NAME=test23 .venv/bin/python3 scripts/phase1_algebra_head.py --eval | grep TOTAL
env ALG_FTYPES=9 ALG_SYNC=1 SYNC_SCRAMBLE=1 ALG_BREATH=3 BREATH_NORM=1 CK=.cache/gsyncscr_arm.safetensors .venv/bin/python3 scripts/dup_axis_scan2.py | grep "^\[scan\]"
echo "== SYNC + CLOCK-PLACEBO COMPLETE =="
