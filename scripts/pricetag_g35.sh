#!/bin/bash
set -eo pipefail
cd /home/bryce/mycelium
export ALG2=1 ALG_FTYPES=8 ALG_HW=512 ALG_DUP=1 ALG_WIDE=1
PY=.venv/bin/python3
CK=.cache/g35_size8x.safetensors
echo "== g35_size8x (trained head, NO fold): dup scan =="
env CK=$CK $PY scripts/dup_axis_scan2.py | grep "^\[scan\]"
echo "== alg4test =="
env ALG_CKPT=$CK ALG_TEST=.cache/algebra4_nl_test.jsonl ALG_TEST_NAME=alg4test $PY scripts/phase1_algebra_head.py --eval | grep TOTAL
echo "== sentinels =="
env CK=$CK $PY scripts/aim_battery.py | grep sentinel
echo "== PRICE TAG COMPLETE =="
