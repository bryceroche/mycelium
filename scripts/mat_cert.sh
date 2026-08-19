#!/bin/bash
set -eo pipefail
cd /home/bryce/mycelium
export DEV=AMD ALG2=1 ALG_DUP=1 ALG_HW=512 ALG_WIDE=1 ALG_ALLOW_PEN_TRAIN=1
PY=.venv/bin/python3
echo "== gsb227_real +20k to 34k =="
env ALG_FTYPES=9 ALG_SEPHASE=1 ALG_SEPHASE_B=1 ALG_SIXWAVE=1 ALG_BREATH=7 BREATH_NORM=1 ALG_NOTEBOOK=1 RESUME=1 \
    ALG_TRAIN=.cache/form_mix8.jsonl ALG_TRAIN_NAME=form8 \
    ALG_TEST=.cache/algebra_nl_test.jsonl ALG_TEST_NAME=test23 \
    ALG_CKPT=.cache/gsb227_real.safetensors ALG_FREEZE_DUP=1 STEPS=20000 LR=1e-4 BATCH=8 SEED=229 SNAP_EVERY=0 \
    RATION_FILE=.cache/ration41_idx.json RATION_W=8 RATION_FILE2=.cache/ration60_t3_idx.json RATION_W2=3 \
    $PY scripts/phase1_algebra_head.py --train | tail -1
echo "== 34k val =="
env ALG_FTYPES=9 ALG_SEPHASE=1 ALG_SEPHASE_B=1 ALG_SIXWAVE=1 ALG_BREATH=7 BREATH_NORM=1 ALG_NOTEBOOK=1 ALG_CKPT=.cache/gsb227_real.safetensors ALG_TEST=.cache/algebra_nl_test.jsonl ALG_TEST_NAME=test23 $PY scripts/phase1_algebra_head.py --eval | grep TOTAL
echo "== BIGTEST CERTIFICATION =="
env ALG_FTYPES=9 ALG_SEPHASE=1 ALG_SEPHASE_B=1 ALG_SIXWAVE=1 ALG_BREATH=7 BREATH_NORM=1 ALG_NOTEBOOK=1 ALG_CKPT=.cache/gsb227_real.safetensors ALG_TEST=.cache/algebra_nl_bigtest.jsonl ALG_TEST_NAME=bigtest $PY scripts/phase1_algebra_head.py --eval | grep TOTAL
echo "== MATURITY CERT COMPLETE =="
