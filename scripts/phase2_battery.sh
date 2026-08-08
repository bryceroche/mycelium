#!/bin/bash
set -eo pipefail
cd /home/bryce/mycelium
export ALG2=1 ALG_FTYPES=8 ALG_HW=512 ALG_DUP=1 ALG_WIDE=1 ALG_RINGS=1 ALG_BREATH=3
PY=.venv/bin/python3
echo "== scan c2ctl (rings+exit) =="
env ALG_BEXIT=1 ALG_CLOCK=0 CK=.cache/g28_p2_c2ctl.safetensors $PY scripts/dup_axis_scan4.py | grep "^\[scan\]"
echo "== scan cure (gate+win+exit) =="
env ALG_BEXIT=1 ALG_CLOCK=1 ALG_CLOCK_FLOOR=0.3 CK=.cache/g28_p2_cure.safetensors $PY scripts/dup_axis_scan4.py | grep "^\[scan\]"
echo "== timing (cure) =="
env ALG_BEXIT=1 ALG_CLOCK=1 ALG_CLOCK_FLOOR=0.3 CK=.cache/g28_p2_cure.safetensors $PY scripts/dup_timing4.py | tail -1
for A in c2ctl cure; do
  echo "== bigtest $A =="
  if [ "$A" = "cure" ]; then export ALG_BEXIT=1 ALG_CLOCK=1 ALG_CLOCK_FLOOR=0.3; else export ALG_BEXIT=1 ALG_CLOCK=0; fi
  env ALG_CKPT=.cache/g28_p2_${A}.safetensors ALG_TEST=.cache/algebra_nl_bigtest.jsonl ALG_TEST_NAME=bigtest $PY scripts/phase1_algebra_head.py --eval | grep TOTAL
done
