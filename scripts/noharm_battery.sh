#!/bin/bash
set -eo pipefail
cd /home/bryce/mycelium
export ALG2=1 ALG_FTYPES=8 ALG_HW=512 ALG_DUP=1 ALG_WIDE=1
PY=.venv/bin/python3
echo "== RITE: refold on g35_size8x's own waist =="
env CK=.cache/g35_size8x.safetensors CK_OUT=.cache/g35_size8x_refold.safetensors $PY scripts/refold_rite.py
echo "== refolded: dup scan (the model's own head) =="
env CK=.cache/g35_size8x_refold.safetensors $PY scripts/dup_axis_scan2.py | grep "^\[scan\]"
echo "== refolded: bigtest =="
env ALG_CKPT=.cache/g35_size8x_refold.safetensors ALG_TEST=.cache/algebra_nl_bigtest.jsonl ALG_TEST_NAME=bigtest $PY scripts/phase1_algebra_head.py --eval | grep TOTAL
echo "== refolded: alg4test =="
env ALG_CKPT=.cache/g35_size8x_refold.safetensors ALG_TEST=.cache/algebra4_nl_test.jsonl ALG_TEST_NAME=alg4test $PY scripts/phase1_algebra_head.py --eval | grep TOTAL
echo "== refolded: sentinels =="
env CK=.cache/g35_size8x_refold.safetensors $PY scripts/aim_battery.py | grep sentinel
echo "== refolded: THE FRONTIER (deployed-chain shape) =="
env WFF_CKPT=.cache/g35_size8x_refold.safetensors WFF_OUT=.cache/wff_g35refold.json $PY scripts/wild_frontier_fixture.py | grep -E "DEPLOYED|STRESS|LIE|mouth"
echo "== NO-HARM BATTERY COMPLETE =="
