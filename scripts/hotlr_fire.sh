#!/bin/bash
set -eo pipefail
cd /home/bryce/mycelium
export DEV=AMD ALG2=1 ALG_FTYPES=8 ALG_DUP=1 ALG_HW=512 ALG_WIDE=1 ALG_ALLOW_PEN_TRAIN=1
PY=.venv/bin/python3
echo "=== DOOR #49: single-pass @ LR 1.5e-4 (the confound isolated) ==="
env ALG_TEST=.cache/algebra_nl_test.jsonl ALG_TEST_NAME=test23 \
    WARM_FROM=.cache/g23v5.safetensors ALG_TRAIN=.cache/form_mix3.jsonl ALG_TRAIN_NAME=form3 \
    ALG_CKPT=.cache/g49_hotlr.safetensors ALG_FREEZE_DUP=1 STEPS=4000 LR=1.5e-4 BATCH=8 SEED=127 SNAP_EVERY=0 \
    RATION_FILE=.cache/ration41_idx.json RATION_W=8 \
    RATION_FILE2=.cache/dup_only_idx.json RATION_W2=3 \
    $PY scripts/phase1_algebra_head.py --train | tail -1
echo "=== BURNED; rite + reads ==="
env CK=.cache/g23v5.safetensors OUT_JSON=.cache/calib_c49.json $PY scripts/calibrated_rite.py | grep headroom
env CK=.cache/g49_hotlr.safetensors OUT_JSON=.cache/calib_g49.json CK_OUT=.cache/g49_hotlr_refold.safetensors $PY scripts/calibrated_rite.py | grep -E "headroom|fold-ready"
echo "== scan (nd0 wall + ND2 THE REFERENT) =="
env CK=.cache/g49_hotlr_refold.safetensors $PY scripts/dup_axis_scan2.py | grep "^\[scan\]"
echo "== op autopsy nd0 =="
env ND=0 OPAUT_NAME=g49 OPAUT_CK=.cache/g49_hotlr_refold.safetensors $PY scripts/op_autopsy.py 2>/dev/null | grep -E "gold=mul|op-miss"
echo "== P-233 (the ordinal-lean read) =="
env CENSUS_CKPT=.cache/g49_hotlr_refold.safetensors CENSUS_OUT=.cache/miss_census_g49.json $PY scripts/miss_census.py 2>/dev/null | grep -E "census"
env P233_NEW=.cache/miss_census_g49.json $PY scripts/p233_read.py
echo "== ring (reported) =="
env ALG_CKPT=.cache/g49_hotlr_refold.safetensors ALG_TEST=.cache/algebra_nl_bigtest.jsonl ALG_TEST_NAME=bigtest $PY scripts/phase1_algebra_head.py --eval | grep TOTAL
env ALG_CKPT=.cache/g49_hotlr_refold.safetensors ALG_TEST=.cache/algebra4_nl_test.jsonl ALG_TEST_NAME=alg4test $PY scripts/phase1_algebra_head.py --eval | grep TOTAL
env CK=.cache/g49_hotlr_refold.safetensors $PY scripts/aim_battery.py | grep sentinel
env WFF_CKPT=.cache/g49_hotlr_refold.safetensors WFF_OUT=.cache/wff_g49.json $PY scripts/wild_frontier_fixture.py | grep -E "DEPLOYED|STRESS|LIE"
echo "== DOOR #49 COMPLETE =="
