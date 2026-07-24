#!/bin/bash
set -eo pipefail
cd /home/bryce/mycelium
export DEV=AMD ALG2=1 ALG_FTYPES=8 ALG_DUP=1
PY=.venv/bin/python3
MIX=.cache/gen18b_mix.jsonl
for seg in 1 2 3 4; do
  echo "=== G18C seg $seg/4 (RATION_W=2.0) ==="
  if [ $seg -eq 1 ]; then W="WARM_FROM=.cache/crown_reader_v4.safetensors"; else W="RESUME=1"; fi
  env $W ALG_TRAIN=$MIX ALG_TRAIN_NAME=g18b ALG_CKPT=.cache/g18c.safetensors STEPS=4000 LR=1e-4 BATCH=8 SEED=6$seg SNAP_EVERY=500 RATION_FILE=.cache/gen18b_ration_idx.json RATION_W=2.0 $PY scripts/phase1_algebra_head.py --train
  for st in 500 1000 1500 2000 2500 3000 3500 4000; do
    mv .cache/g18c_s${st}.safetensors .cache/g18c_seg${seg}_s${st}.safetensors 2>/dev/null || true
  done
done
echo "=== THE KNOB IS TURNED — g18c banked ==="
