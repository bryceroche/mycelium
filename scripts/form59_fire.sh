#!/bin/bash
set -eo pipefail
cd /home/bryce/mycelium
export DEV=AMD ALG2=1 ALG_DUP=1 ALG_HW=512 ALG_WIDE=1 ALG_ALLOW_PEN_TRAIN=1
PY=.venv/bin/python3
echo "=== ASSEMBLE form7 ==="
env ALG_FTYPES=9 $PY scripts/form_assemble7.py
echo "=== DOOR #59: THE EMISSION FIRE (ftype 9-wide, pad-warm) ==="
env ALG_FTYPES=9 ALG_TEST=.cache/algebra_nl_test.jsonl ALG_TEST_NAME=test23 \
    WARM_FROM=.cache/g58_inband.safetensors ALG_TRAIN=.cache/form_mix7.jsonl ALG_TRAIN_NAME=form7 \
    ALG_CKPT=.cache/g59_dv.safetensors ALG_FREEZE_DUP=1 STEPS=4000 LR=1e-4 BATCH=8 SEED=127 SNAP_EVERY=0 \
    RATION_FILE=.cache/ration41_idx.json RATION_W=8 \
    RATION_FILE2=.cache/ration59_t3_idx.json RATION_W2=3 \
    $PY scripts/phase1_algebra_head.py --train | tail -2
echo "=== BURNED; EMISSION READ FIRST (held-out cascade prose) ==="
env ALG_FTYPES=9 env ALG_FTYPES=9 SEQ_READ=1 INBAND=1 DIVERSE_VALS=1 EM_CK=.cache/g59_dv.safetensors $PY scripts/chain_emission_read.py
echo "== rite (control at 8; arm at 9) =="
env CK=.cache/g23v5.safetensors OUT_JSON=.cache/calib_c59.json $PY scripts/calibrated_rite.py | grep headroom
env ALG_FTYPES=9 CK=.cache/g59_dv.safetensors OUT_JSON=.cache/calib_g59.json CK_OUT=.cache/g59_dv_refold.safetensors $PY scripts/calibrated_rite.py | grep -E "headroom|fold-ready"
env ALG_FTYPES=9 CK=.cache/g59_dv_refold.safetensors $PY scripts/dup_axis_scan2.py | grep "^\[scan\]"
echo "== THE GAP READ (the 233, reported) =="
env ALG_FTYPES=9 CENSUS_CKPT=.cache/g59_dv_refold.safetensors CENSUS_OUT=.cache/mc_g59.json $PY scripts/miss_census.py 2>/dev/null | grep census
env P233_NEW=.cache/mc_g59.json $PY scripts/p233_read.py
echo "== ring =="
env ALG_FTYPES=9 ALG_CKPT=.cache/g59_dv_refold.safetensors ALG_TEST=.cache/algebra_nl_bigtest.jsonl ALG_TEST_NAME=bigtest $PY scripts/phase1_algebra_head.py --eval | grep TOTAL
env ALG_FTYPES=9 ALG_CKPT=.cache/g59_dv_refold.safetensors ALG_TEST=.cache/algebra4_nl_test.jsonl ALG_TEST_NAME=alg4test $PY scripts/phase1_algebra_head.py --eval | grep TOTAL
env ALG_FTYPES=9 CK=.cache/g59_dv_refold.safetensors $PY scripts/aim_battery.py | grep sentinel
env ALG_FTYPES=9 WFF_CKPT=.cache/g59_dv_refold.safetensors WFF_OUT=.cache/wff_g59.json $PY scripts/wild_frontier_fixture.py | grep -E "DEPLOYED|LIE"
echo "== DOOR #59 COMPLETE =="
