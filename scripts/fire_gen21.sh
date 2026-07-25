#!/bin/bash
# fire_gen21.sh — THE GEN-21 FIRE (the two-book fire) (2026-07-24, on the word): the
# budget-shift thesis — spans from birth + book 5 complete, ration at
# the interpolated middle, every bar unchanged.
set -eo pipefail
cd /home/bryce/mycelium
export DEV=AMD ALG2=1 ALG_FTYPES=8 ALG_DUP=1
PY=.venv/bin/python3
MIX=.cache/gen21_mix.jsonl
echo "=== G21 1/5: precompute ==="
ALG_TRAIN=$MIX ALG_TRAIN_NAME=g21 PRECOMPUTE_ONLY=g21 $PY scripts/phase1_algebra_head.py --precompute
for seg in 1 2 3 4; do
  echo "=== G21 seg $seg/4 (RATION_W=1.75 hot-phase) ==="
  if [ $seg -eq 1 ]; then W="WARM_FROM=.cache/g20.safetensors"; else W="RESUME=1"; fi
  env $W ALG_TRAIN=$MIX ALG_TRAIN_NAME=g21 ALG_CKPT=.cache/g21.safetensors STEPS=4000 LR=1e-4 BATCH=8 SEED=7$seg SNAP_EVERY=500 RATION_FILE=.cache/gen21_ration_idx.json RATION_W=1.75 $PY scripts/phase1_algebra_head.py --train
  for st in 500 1000 1500 2000 2500 3000 3500 4000; do
    mv .cache/g21_s${st}.safetensors .cache/g21_seg${seg}_s${st}.safetensors 2>/dev/null || true
  done
done
echo "=== THE FIRE IS BURNED — g20 banked ==="
