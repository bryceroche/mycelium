#!/bin/bash
set -eo pipefail
cd /home/bryce/mycelium
export DEV=AMD ALG2=1 ALG_FTYPES=8 ALG_DUP=1 ALG_HW=512 ALG_WIDE=1
export ALG_CKPT=.cache/g41_onemass_refold.safetensors
echo "== RECERT 1: bigtest deploy battery (fixed seam) =="
env ALG_TEST=.cache/algebra_nl_bigtest.jsonl ALG_TEST_NAME=bigtest .venv/bin/python3 scripts/deploy_battery.py
echo "== RECERT 2: MATH-500 true chain (fixed seam) =="
env ALG_TEST=.cache/algebra_nl_bigtest.jsonl ALG_TEST_NAME=bigtest .venv/bin/python3 scripts/math500_mouth.py
echo "== RECERT COMPLETE =="
