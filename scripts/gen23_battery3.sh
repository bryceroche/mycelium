#!/bin/bash
set -eo pipefail
cd /home/bryce/mycelium
export DEV=AMD ALG2=1 ALG_FTYPES=8 ALG_DUP=1 ALG_HW=512 ALG_WIDE=1
PY=.venv/bin/python3
: > .cache/gen23_battery3.log
for F in "bigtest .cache/algebra_nl_bigtest.jsonl" "alg4test .cache/algebra4_nl_test.jsonl" "alg2test .cache/algebra2_nl_test.jsonl" "vtest .cache/algv_test_verbose.jsonl" "dagtest .cache/dag_test.jsonl" "dag7btest .cache/dag7b_test.jsonl" "dag8test .cache/dag8_test.jsonl"; do
  set -- $F
  echo "=== eval $1 ===" | tee -a .cache/gen23_battery3.log
  env ALG_CKPT=.cache/g23.safetensors ALG_TEST=$2 ALG_TEST_NAME=$1 $PY scripts/phase1_algebra_head.py --eval >> .cache/gen23_battery3.log 2>&1
  tail -2 .cache/gen23_battery3.log
done
echo "=== THE VERDICT HOLDS THE PEN ==="
$PY scripts/gen23_verdict.py
