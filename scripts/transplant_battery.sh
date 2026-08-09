#!/bin/bash
set -eo pipefail
cd /home/bryce/mycelium
export ALG2=1 ALG_FTYPES=8 ALG_HW=512 ALG_DUP=1 ALG_WIDE=1 ALG_DUPPTR=1
PY=.venv/bin/python3
echo "== scan transplant =="
env CK=.cache/g30_transplant.safetensors $PY scripts/dup_axis_scan5.py | grep "^\[scan\]"
echo "== bigtest transplant =="
env ALG_CKPT=.cache/g30_transplant.safetensors ALG_TEST=.cache/algebra_nl_bigtest.jsonl ALG_TEST_NAME=bigtest $PY scripts/phase1_algebra_head.py --eval | grep TOTAL
