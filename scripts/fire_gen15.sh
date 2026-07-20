#!/bin/bash
# fire_gen15.sh — THE TRAINING FIRE (2026-07-20, lit on Bryce's word).
# Four arms, sequential; every ckpt built alongside gen-14, nothing touched.
set -eo pipefail
cd /home/bryce/mycelium
export DEV=AMD ALG2=1 ALG_FTYPES=7 ALG_DUP=1
PY=.venv/bin/python3
G14=.cache/phase1_gen14_head.safetensors

echo "=== FIRE 1/9: precompute arm A ==="
ALG_TRAIN=.cache/fire_armA.jsonl ALG_TRAIN_NAME=fireA PRECOMPUTE_ONLY=fireA $PY scripts/phase1_algebra_head.py --precompute
echo "=== FIRE 2/9: train arm A (prime control, 16k warm gen-14) ==="
ALG_TRAIN=.cache/fire_armA.jsonl ALG_TRAIN_NAME=fireA WARM_FROM=$G14 ALG_CKPT=.cache/fire_armA.safetensors STEPS=16000 LR=1e-4 BATCH=8 SEED=15 $PY scripts/phase1_algebra_head.py --train
echo "=== FIRE 3/9: precompute arm B ==="
ALG_TRAIN=.cache/fire_armB.jsonl ALG_TRAIN_NAME=fireB PRECOMPUTE_ONLY=fireB $PY scripts/phase1_algebra_head.py --precompute
echo "=== FIRE 4/9: train arm B (macro-only) ==="
ALG_TRAIN=.cache/fire_armB.jsonl ALG_TRAIN_NAME=fireB WARM_FROM=$G14 ALG_CKPT=.cache/fire_armB.safetensors STEPS=16000 LR=1e-4 BATCH=8 SEED=15 $PY scripts/phase1_algebra_head.py --train
echo "=== FIRE 5/9: precompute arm C1 ==="
ALG_TRAIN=.cache/fire_armC1.jsonl ALG_TRAIN_NAME=fireC1 PRECOMPUTE_ONLY=fireC1 $PY scripts/phase1_algebra_head.py --precompute
echo "=== FIRE 6/9: train arm C1 (paired SPREAD — both channels' lean) ==="
ALG_TRAIN=.cache/fire_armC1.jsonl ALG_TRAIN_NAME=fireC1 WARM_FROM=$G14 ALG_CKPT=.cache/fire_armC1.safetensors STEPS=16000 LR=1e-4 BATCH=8 SEED=15 $PY scripts/phase1_algebra_head.py --train
echo "=== FIRE 7/9: train arm C2 phase-1 (12k on arm-A states) ==="
ALG_TRAIN=.cache/fire_armA.jsonl ALG_TRAIN_NAME=fireA WARM_FROM=$G14 ALG_CKPT=.cache/fire_armC2.safetensors STEPS=12000 LR=1e-4 BATCH=8 SEED=15 $PY scripts/phase1_algebra_head.py --train
echo "=== FIRE 8/9: precompute C2 phase-2 (the concentrated dose) ==="
ALG_TRAIN=.cache/fire_armC2_phase2.jsonl ALG_TRAIN_NAME=fireC2p2 PRECOMPUTE_ONLY=fireC2p2 $PY scripts/phase1_algebra_head.py --precompute
echo "=== FIRE 9/9: train C2 phase-2 (4k concentrated, RESUME) ==="
ALG_TRAIN=.cache/fire_armC2_phase2.jsonl ALG_TRAIN_NAME=fireC2p2 RESUME=1 ALG_CKPT=.cache/fire_armC2.safetensors STEPS=4000 LR=1e-4 BATCH=8 SEED=15 $PY scripts/phase1_algebra_head.py --train
echo "=== THE FIRE IS BURNED — four arms banked; the battery speaks next ==="
