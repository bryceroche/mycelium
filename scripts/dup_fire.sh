#!/bin/bash
# dup_fire.sh — THE SIX-ARM CROSS (2026-07-31, the word given).
# Design: docs/DUP_DIET_DESIGN.md. Prep (mint/assembly/sentinels) runs
# first and gates everything; then six gentle continuations from g22;
# then per-arm after-reads (2b population + settle, bigtest+alg4
# neighbors, wild fixture STRESS). No-mid-fire binds all arms:
# completion or pre-registered kill, nothing stirred.
set -eo pipefail
cd /home/bryce/mycelium
export DEV=AMD ALG2=1 ALG_FTYPES=8 ALG_DUP=1 ALG_ALLOW_PEN_TRAIN=1
PY=.venv/bin/python3
echo "=== DUP 1/3: prep (mint, WET fold, assembly, sentinels) ==="
$PY scripts/dup_fire_prep.py
SEED=90
for ARM in dry_d02 dry_d05 dry_d12 wet_d02 wet_d05 wet_d12; do
  SEED=$((SEED+1))
  MIX=.cache/dupfire_${ARM}_mix.jsonl
  CKPT=.cache/g23_${ARM}.safetensors
  echo "=== DUP 2/3 [$ARM]: fire (4x4k from g22, seed base ${SEED}) ==="
  for seg in 1 2 3 4; do
    if [ $seg -eq 1 ]; then W="WARM_FROM=.cache/g22.safetensors"; else W="RESUME=1"; fi
    env $W ALG_TRAIN=$MIX ALG_TRAIN_NAME=g23${ARM} ALG_CKPT=$CKPT STEPS=4000 LR=1e-4 BATCH=8 SEED=${SEED}${seg} SNAP_EVERY=2000 $PY scripts/phase1_algebra_head.py --train
  done
  echo "=== DUP 3/3 [$ARM]: after-reads ==="
  R2B_CKPT=$CKPT $PY scripts/bench_rung2b.py 2>&1 | tail -6
  cp .cache/bench_rung2b.json .cache/dupfire_${ARM}_2b.json
  env ALG_CKPT=$CKPT ALG_TEST=.cache/algebra_nl_bigtest.jsonl ALG_TEST_NAME=bigtest $PY scripts/phase1_algebra_head.py --eval 2>&1 | tail -1
  env ALG_CKPT=$CKPT ALG_TEST=.cache/algebra4_nl_test.jsonl ALG_TEST_NAME=alg4test $PY scripts/phase1_algebra_head.py --eval 2>&1 | tail -1
  WFF_CKPT=$CKPT WFF_OUT=.cache/dupfire_${ARM}_wff.json $PY scripts/wild_frontier_fixture.py 2>&1 | tail -3
done
echo "=== THE CROSS IS BURNED — six arms banked; the verdict reads next (g22 REMAINS THE GATE) ==="
