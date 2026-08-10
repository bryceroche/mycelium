#!/bin/bash
set -eo pipefail
cd /home/bryce/mycelium
export ALG2=1 ALG_FTYPES=8 ALG_HW=512 ALG_DUP=1 ALG_WIDE=1
PY=.venv/bin/python3
for A in tctl tail; do
  echo "== $A: bigtest =="
  env ALG_CKPT=.cache/g32_tail_${A}.safetensors ALG_TEST=.cache/algebra_nl_bigtest.jsonl ALG_TEST_NAME=bigtest $PY scripts/phase1_algebra_head.py --eval | grep TOTAL
  echo "== $A: alg4test =="
  env ALG_CKPT=.cache/g32_tail_${A}.safetensors ALG_TEST=.cache/algebra4_nl_test.jsonl ALG_TEST_NAME=alg4test $PY scripts/phase1_algebra_head.py --eval | grep TOTAL
done
echo "== tail arm: dup cells =="
env CK=.cache/g32_tail_tail.safetensors $PY scripts/dup_axis_scan2.py | grep "^\[scan\]"
echo "== tail arm: miss census re-run =="
env ALG_CKPT=.cache/g32_tail_tail.safetensors $PY scripts/miss_census.py | grep -E "census|families"
