#!/bin/bash
set -eo pipefail
cd /home/bryce/mycelium
export DEV=AMD ALG2=1 ALG_FTYPES=8 ALG_DUP=1 ALG_HW=512 ALG_WIDE=1 ALG_ALLOW_PEN_TRAIN=1
export ALG_TEST=.cache/algebra_nl_test.jsonl ALG_TEST_NAME=test23
export RATION_FILE=.cache/size_sliver_idx.json RATION_W=8
export RATION_FILE2=.cache/dup_rehearsal_idx.json RATION_W2=3
PY=.venv/bin/python3
echo "=== DOOR #36 [freeze8x]: 4k from g23v5, ration 8x + dup rehearsal 3x, h_dup FROZEN ==="
env WARM_FROM=.cache/g23v5.safetensors ALG_TRAIN=.cache/size_mix.jsonl ALG_TRAIN_NAME=size \
    ALG_CKPT=.cache/g36_freeze8x.safetensors ALG_FREEZE_DUP=1 STEPS=4000 LR=1e-4 BATCH=8 SEED=127 SNAP_EVERY=0 \
    $PY scripts/phase1_algebra_head.py --train | tail -2
echo "=== BURNED; THE RITE RIDES ==="
export -n ALG_TEST ALG_TEST_NAME
echo "== RITE: refold on g36's own waist =="
env WILD_EXCLUDE=0 CK=.cache/g36_freeze8x.safetensors CK_OUT=.cache/g36_freeze8x_refold.safetensors $PY scripts/refold_rite.py
echo "== refolded: dup scan =="
env CK=.cache/g36_freeze8x_refold.safetensors $PY scripts/dup_axis_scan2.py | grep "^\[scan\]"
echo "== refolded: bigtest =="
env ALG_CKPT=.cache/g36_freeze8x_refold.safetensors ALG_TEST=.cache/algebra_nl_bigtest.jsonl ALG_TEST_NAME=bigtest $PY scripts/phase1_algebra_head.py --eval | grep TOTAL
echo "== refolded: alg4test =="
env ALG_CKPT=.cache/g36_freeze8x_refold.safetensors ALG_TEST=.cache/algebra4_nl_test.jsonl ALG_TEST_NAME=alg4test $PY scripts/phase1_algebra_head.py --eval | grep TOTAL
echo "== refolded: sentinels =="
env CK=.cache/g36_freeze8x_refold.safetensors $PY scripts/aim_battery.py | grep sentinel
echo "== refolded: THE FRONTIER =="
env WFF_CKPT=.cache/g36_freeze8x_refold.safetensors WFF_OUT=.cache/wff_g36refold.json $PY scripts/wild_frontier_fixture.py | grep -E "DEPLOYED|STRESS|LIE"
echo "== DOOR #36 BATTERY COMPLETE =="
