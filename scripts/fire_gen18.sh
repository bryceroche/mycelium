#!/bin/bash
# fire_gen18.sh — THE GEN-18 FIRE (2026-07-24, on the word). Two ration
# arms, one mix, one precompute; SGDR 4x4k with snapshots on ALL
# segments; arm B adds hot-phase ration (RATION_FILE + RATION_W).
set -eo pipefail
cd /home/bryce/mycelium
export DEV=AMD ALG2=1 ALG_FTYPES=8 ALG_DUP=1
PY=.venv/bin/python3
G16=.cache/crown_reader_v4.safetensors
MIX=.cache/gen18_mix.jsonl

echo "=== G18 1/9: precompute (shared) ==="
ALG_TRAIN=$MIX ALG_TRAIN_NAME=g18 PRECOMPUTE_ONLY=g18 $PY scripts/phase1_algebra_head.py --precompute

for seg in 1 2 3 4; do
  echo "=== G18 armA seg $seg/4 ==="
  if [ $seg -eq 1 ]; then W="WARM_FROM=$G16"; else W="RESUME=1"; fi
  env $W ALG_TRAIN=$MIX ALG_TRAIN_NAME=g18 ALG_CKPT=.cache/g18_armA.safetensors STEPS=4000 LR=1e-4 BATCH=8 SEED=3$seg SNAP_EVERY=500 $PY scripts/phase1_algebra_head.py --train
  mv .cache/g18_armA_s500.safetensors .cache/g18_armA_seg${seg}_s500.safetensors 2>/dev/null || true
  for st in 1000 1500 2000 2500 3000 3500 4000; do
    mv .cache/g18_armA_s${st}.safetensors .cache/g18_armA_seg${seg}_s${st}.safetensors 2>/dev/null || true
  done
done

for seg in 1 2 3 4; do
  echo "=== G18 armB seg $seg/4 (RATION hot-phase) ==="
  if [ $seg -eq 1 ]; then W="WARM_FROM=$G16"; else W="RESUME=1"; fi
  env $W ALG_TRAIN=$MIX ALG_TRAIN_NAME=g18 ALG_CKPT=.cache/g18_armB.safetensors STEPS=4000 LR=1e-4 BATCH=8 SEED=4$seg SNAP_EVERY=500 RATION_FILE=.cache/gen18_ration_idx.json RATION_W=1.5 $PY scripts/phase1_algebra_head.py --train
  for st in 500 1000 1500 2000 2500 3000 3500 4000; do
    mv .cache/g18_armB_s${st}.safetensors .cache/g18_armB_seg${seg}_s${st}.safetensors 2>/dev/null || true
  done
done
echo "=== THE FIRE IS BURNED — two ration arms + full snapshot trajectories banked ==="
