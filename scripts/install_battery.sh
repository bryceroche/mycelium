#!/bin/bash
set -eo pipefail
cd /home/bryce/mycelium
export ALG2=1 ALG_FTYPES=8 ALG_HW=512 ALG_DUP=1 ALG_WIDE=1
PY=.venv/bin/python3
echo "== scan ictl (vdup19) =="
env ALG_DUPPTR=0 CK=.cache/g30_inst_ictl.safetensors $PY scripts/dup_axis_scan5.py | grep "^\[scan\]"
echo "== scan install (assembly) =="
env ALG_DUPPTR=1 CK=.cache/g30_inst_install.safetensors $PY scripts/dup_axis_scan5.py | grep "^\[scan\]"
echo "== timing (install) =="
env ALG_DUPPTR=1 CK=.cache/g30_inst_install.safetensors $PY scripts/dup_timing5.py | tail -1
for A in ictl install; do
  echo "== bigtest $A =="
  if [ "$A" = "install" ]; then export ALG_DUPPTR=1; else export ALG_DUPPTR=0; fi
  env ALG_CKPT=.cache/g30_inst_${A}.safetensors ALG_TEST=.cache/algebra_nl_bigtest.jsonl ALG_TEST_NAME=bigtest $PY scripts/phase1_algebra_head.py --eval | grep TOTAL
done
