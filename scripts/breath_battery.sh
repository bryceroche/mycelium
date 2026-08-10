#!/bin/bash
set -eo pipefail
cd /home/bryce/mycelium
export ALG2=1 ALG_FTYPES=8 ALG_HW=512 ALG_DUP=1 ALG_WIDE=1
PY=.venv/bin/python3
echo "== bctl: bigtest =="
env ALG_BREATH=1 ALG_CKPT=.cache/g33_breath_bctl.safetensors ALG_TEST=.cache/algebra_nl_bigtest.jsonl ALG_TEST_NAME=bigtest $PY scripts/phase1_algebra_head.py --eval | grep TOTAL
echo "== breath: bigtest (K=3) =="
env ALG_BREATH=3 ALG_CKPT=.cache/g33_breath_breath.safetensors ALG_TEST=.cache/algebra_nl_bigtest.jsonl ALG_TEST_NAME=bigtest $PY scripts/phase1_algebra_head.py --eval | grep TOTAL
echo "== bctl: dup cells =="
env ALG_BREATH=1 CK=.cache/g33_breath_bctl.safetensors $PY scripts/dup_axis_scan2.py | grep "^\[scan\]"
echo "== breath: dup cells (K=3, frozen fold on breath waist) =="
env ALG_BREATH=3 ALG_RINGS=0 ALG_BEXIT=0 ALG_CLOCK=0 CK=.cache/g33_breath_breath.safetensors $PY scripts/dup_axis_scan3.py | grep "^\[scan\]"
echo "== alg4 both =="
env ALG_BREATH=1 ALG_CKPT=.cache/g33_breath_bctl.safetensors ALG_TEST=.cache/algebra4_nl_test.jsonl ALG_TEST_NAME=alg4test $PY scripts/phase1_algebra_head.py --eval | grep TOTAL
env ALG_BREATH=3 ALG_CKPT=.cache/g33_breath_breath.safetensors ALG_TEST=.cache/algebra4_nl_test.jsonl ALG_TEST_NAME=alg4test $PY scripts/phase1_algebra_head.py --eval | grep TOTAL
