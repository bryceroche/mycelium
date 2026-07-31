#!/bin/bash
# dup_verdict_pass.sh — the neighbor reads (2026-07-31): the displacement
# discriminator, the fire's one surviving readable call. Six ckpts x
# bigtest+alg4 in one session (comparable conditions).
set -eo pipefail
cd /home/bryce/mycelium
export DEV=AMD ALG2=1 ALG_FTYPES=8 ALG_DUP=1
PY=.venv/bin/python3
echo "=== NEIGHBOR READS (g22 bars: bigtest 1226, alg4 398) ==="
for ARM in dry_d02 dry_d05 dry_d12 wet_d02 wet_d05 wet_d12; do
  CKPT=.cache/g23_${ARM}.safetensors
  echo "--- $ARM ---"
  env ALG_CKPT=$CKPT ALG_TEST=.cache/algebra_nl_bigtest.jsonl ALG_TEST_NAME=bigtest $PY scripts/phase1_algebra_head.py --eval 2>&1 | grep TOTAL | tail -1
  env ALG_CKPT=$CKPT ALG_TEST=.cache/algebra4_nl_test.jsonl ALG_TEST_NAME=alg4test $PY scripts/phase1_algebra_head.py --eval 2>&1 | grep TOTAL | tail -1
done
echo "=== NEIGHBOR READS COMPLETE ==="
