#!/bin/bash
set -eo pipefail
cd /home/bryce/mycelium
export ALG2=1 ALG_FTYPES=8 ALG_HW=512 ALG_DUP=1 ALG_WIDE=1 ALG_RINGS=1 ALG_BREATH=3
PY=.venv/bin/python3
echo "== scan cctl (rings+exit) =="
env ALG_BEXIT=1 ALG_CLOCK=0 CK=.cache/g27_clock_cctl.safetensors $PY scripts/dup_axis_scan4.py | grep "^\[scan\]"
echo "== scan clock (rings+exit+gate) =="
env ALG_BEXIT=1 ALG_CLOCK=1 CK=.cache/g27_clock_clock.safetensors $PY scripts/dup_axis_scan4.py | grep "^\[scan\]"
echo "== timing (clock) =="
env ALG_BEXIT=1 ALG_CLOCK=1 CK=.cache/g27_clock_clock.safetensors $PY scripts/dup_timing4.py | tail -1
for A in cctl clock; do
  echo "== bigtest $A =="
  if [ "$A" = "clock" ]; then export ALG_BEXIT=1 ALG_CLOCK=1; else export ALG_BEXIT=1 ALG_CLOCK=0; fi
  env ALG_CKPT=.cache/g27_clock_${A}.safetensors ALG_TEST=.cache/algebra_nl_bigtest.jsonl ALG_TEST_NAME=bigtest $PY scripts/phase1_algebra_head.py --eval | grep TOTAL
done
