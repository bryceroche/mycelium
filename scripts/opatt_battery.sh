#!/bin/bash
set -eo pipefail
cd /home/bryce/mycelium
export ALG2=1 ALG_FTYPES=8 ALG_HW=512 ALG_DUP=1 ALG_WIDE=1
PY=.venv/bin/python3
for CK in g26_opatt_ctl g26_opatt_opatt; do
  echo "== scan $CK =="
  env CK=.cache/${CK}.safetensors $PY scripts/dup_axis_scan2.py | grep "^\[scan\]"
done
echo "== timing re-run (opatt arm) =="
env CK=.cache/g26_opatt_opatt.safetensors $PY scripts/dup_timing2.py | tail -1
for CK in g26_opatt_ctl g26_opatt_opatt; do
  echo "== bigtest $CK =="
  env ALG_CKPT=.cache/${CK}.safetensors ALG_TEST=.cache/algebra_nl_bigtest.jsonl ALG_TEST_NAME=bigtest $PY scripts/phase1_algebra_head.py --eval | grep TOTAL
done
