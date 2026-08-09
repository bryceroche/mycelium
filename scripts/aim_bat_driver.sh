#!/bin/bash
set -eo pipefail
cd /home/bryce/mycelium
export ALG2=1 ALG_FTYPES=8 ALG_HW=512 ALG_DUP=1 ALG_WIDE=1
for A in actl aim; do
  echo "== $A =="
  env CK=.cache/g31_aim_${A}.safetensors .venv/bin/python3 scripts/aim_battery.py
  env ALG_CKPT=.cache/g31_aim_${A}.safetensors ALG_TEST=.cache/algebra_nl_bigtest.jsonl ALG_TEST_NAME=bigtest .venv/bin/python3 scripts/phase1_algebra_head.py --eval | grep TOTAL
done
