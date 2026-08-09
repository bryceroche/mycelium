#!/bin/bash
set -eo pipefail
cd /home/bryce/mycelium
export ALG2=1 ALG_FTYPES=8 ALG_HW=512 ALG_DUP=1 ALG_WIDE=1
PY=.venv/bin/python3
for CK in g23 g23v5_candidate; do
  echo "== $CK: dup scan =="
  env CK=.cache/${CK}.safetensors $PY scripts/dup_axis_scan2.py | grep "^\[scan\]"
  echo "== $CK: bigtest =="
  env ALG_CKPT=.cache/${CK}.safetensors ALG_TEST=.cache/algebra_nl_bigtest.jsonl ALG_TEST_NAME=bigtest $PY scripts/phase1_algebra_head.py --eval | grep TOTAL
  echo "== $CK: alg4test =="
  env ALG_CKPT=.cache/${CK}.safetensors ALG_TEST=.cache/algebra4_nl_test.jsonl ALG_TEST_NAME=alg4test $PY scripts/phase1_algebra_head.py --eval | grep TOTAL
done
echo "== candidate: sentinels =="
env CK=.cache/g23v5_candidate.safetensors $PY scripts/aim_battery.py | grep sentinel
