#!/bin/bash
set -eo pipefail
cd /home/bryce/mycelium
export ALG2=1 ALG_FTYPES=8 ALG_HW=512 ALG_DUP=1 ALG_WIDE=1 ALG_RINGS=1 ALG_BREATH=3
PY=.venv/bin/python3
echo "== scan rctl (rings, no exit) =="
env ALG_BEXIT=0 CK=.cache/g26_bexit_rctl.safetensors $PY scripts/dup_axis_scan3.py | grep "^\[scan\]"
echo "== scan bexit (rings + exit) =="
env ALG_BEXIT=1 CK=.cache/g26_bexit_bexit.safetensors $PY scripts/dup_axis_scan3.py | grep "^\[scan\]"
echo "== timing (bexit) =="
env ALG_BEXIT=1 CK=.cache/g26_bexit_bexit.safetensors $PY scripts/dup_timing3.py | tail -1
for A in rctl bexit; do
  echo "== bigtest $A =="
  if [ "$A" = "bexit" ]; then export ALG_BEXIT=1; else export ALG_BEXIT=0; fi
  env ALG_CKPT=.cache/g26_bexit_${A}.safetensors ALG_TEST=.cache/algebra_nl_bigtest.jsonl ALG_TEST_NAME=bigtest $PY scripts/phase1_algebra_head.py --eval | grep TOTAL
done
