#!/bin/bash
set -eo pipefail
cd /home/bryce/mycelium
export DEV=AMD ALG2=1 ALG_FTYPES=8 ALG_DUP=1 ALG_HW=512 ALG_WIDE=1
PY=.venv/bin/python3
echo "=== split read (per-arm) ==="
$PY scripts/gen23_split_read.py
echo "=== B2 bigtest (sequential — device free) ==="
env ALG_CKPT=.cache/g23.safetensors ALG_TEST=.cache/algebra_nl_bigtest.jsonl ALG_TEST_NAME=bigtest $PY scripts/phase1_algebra_head.py --eval | tail -8
echo "=== battery2 complete ==="
