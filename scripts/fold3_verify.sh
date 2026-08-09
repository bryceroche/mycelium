#!/bin/bash
set -eo pipefail
cd /home/bryce/mycelium
export ALG2=1 ALG_FTYPES=8 ALG_HW=512 ALG_DUP=1 ALG_WIDE=1
env CK=.cache/g31_fold3.safetensors .venv/bin/python3 scripts/dup_axis_scan2.py | grep "^\[scan\]"
env ALG_CKPT=.cache/g31_fold3.safetensors ALG_TEST=.cache/algebra_nl_bigtest.jsonl ALG_TEST_NAME=bigtest .venv/bin/python3 scripts/phase1_algebra_head.py --eval | grep TOTAL
