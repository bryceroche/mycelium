#!/bin/bash
set -eo pipefail
cd /home/bryce/mycelium
export ALG2=1 ALG_FTYPES=8 ALG_HW=512 ALG_DUP=1 ALG_WIDE=1
PY=.venv/bin/python3
echo "== scan dctl (plain) =="
env ALG_DUPPTR=0 CK=.cache/g29_dp_dctl.safetensors $PY scripts/dup_axis_scan5.py | grep "^\[scan\]"
echo "== scan dupptr (structural) =="
env ALG_DUPPTR=1 CK=.cache/g29_dp_dupptr.safetensors $PY scripts/dup_axis_scan5.py | grep "^\[scan\]"
echo "== timing (dupptr) =="
env ALG_DUPPTR=1 CK=.cache/g29_dp_dupptr.safetensors $PY scripts/dup_timing5.py | tail -1
for A in dctl dupptr; do
  echo "== bigtest $A =="
  if [ "$A" = "dupptr" ]; then export ALG_DUPPTR=1; else export ALG_DUPPTR=0; fi
  env ALG_CKPT=.cache/g29_dp_${A}.safetensors ALG_TEST=.cache/algebra_nl_bigtest.jsonl ALG_TEST_NAME=bigtest $PY scripts/phase1_algebra_head.py --eval | grep TOTAL
done
