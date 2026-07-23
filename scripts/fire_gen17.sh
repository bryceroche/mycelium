#!/bin/bash
# fire_gen17.sh — THE GEN-17 FIRE (2026-07-23, lit on Bryce's word; the
# zener-convened review's charter). Two arms, matched 16k budget, warm
# from crown_reader_v4; every ckpt built alongside gen-16, nothing touched.
# Arm F = single 16k cosine (incumbent recipe verbatim).
# Arm R = 4x4k RESUME segments (SGDR warm restarts, flat mix — the one
#         lever isolated; band-timing deliberately unasked).
set -eo pipefail
cd /home/bryce/mycelium
export DEV=AMD ALG2=1 ALG_FTYPES=8 ALG_DUP=1
PY=.venv/bin/python3
G16=.cache/crown_reader_v4.safetensors

echo "=== G17 1/6: precompute mix states ==="
ALG_TRAIN=.cache/gen17_mix.jsonl ALG_TRAIN_NAME=g17 PRECOMPUTE_ONLY=g17 $PY scripts/phase1_algebra_head.py --precompute

echo "=== G17 2/6: ARM F — flat 16k (incumbent recipe) ==="
ALG_TRAIN=.cache/gen17_mix.jsonl ALG_TRAIN_NAME=g17 WARM_FROM=$G16 ALG_CKPT=.cache/g17_armF.safetensors STEPS=16000 LR=1e-4 BATCH=8 SEED=17 $PY scripts/phase1_algebra_head.py --train

echo "=== G17 3/6: ARM R — SGDR segment 1/4 (4k, warm from gen-16) ==="
ALG_TRAIN=.cache/gen17_mix.jsonl ALG_TRAIN_NAME=g17 WARM_FROM=$G16 ALG_CKPT=.cache/g17_armR.safetensors STEPS=4000 LR=1e-4 BATCH=8 SEED=17 $PY scripts/phase1_algebra_head.py --train

echo "=== G17 4/6: ARM R — SGDR segment 2/4 (RESUME, fresh cosine) ==="
ALG_TRAIN=.cache/gen17_mix.jsonl ALG_TRAIN_NAME=g17 RESUME=1 ALG_CKPT=.cache/g17_armR.safetensors STEPS=4000 LR=1e-4 BATCH=8 SEED=18 $PY scripts/phase1_algebra_head.py --train

echo "=== G17 5/6: ARM R — SGDR segment 3/4 ==="
ALG_TRAIN=.cache/gen17_mix.jsonl ALG_TRAIN_NAME=g17 RESUME=1 ALG_CKPT=.cache/g17_armR.safetensors STEPS=4000 LR=1e-4 BATCH=8 SEED=19 $PY scripts/phase1_algebra_head.py --train

echo "=== G17 6/6: ARM R — SGDR segment 4/4 ==="
ALG_TRAIN=.cache/gen17_mix.jsonl ALG_TRAIN_NAME=g17 RESUME=1 ALG_CKPT=.cache/g17_armR.safetensors STEPS=4000 LR=1e-4 BATCH=8 SEED=20 $PY scripts/phase1_algebra_head.py --train

echo "=== THE FIRE IS BURNED — two arms banked; the battery speaks next ==="
