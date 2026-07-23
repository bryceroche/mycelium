#!/bin/bash
set -eo pipefail
cd /home/bryce/mycelium
export DEV=AMD ALG2=1 ALG_FTYPES=8 ALG_DUP=1
PY=.venv/bin/python3
echo "=== CANVAS 1/2: dose burn (FAT_W=4, 6k from crown_v4) ==="
ALG_TRAIN=.cache/crown_v4.jsonl ALG_TRAIN_NAME=crownv4 WARM_FROM=.cache/crown_reader_v4.safetensors ALG_CKPT=.cache/canvas_dose.safetensors STEPS=6000 LR=1e-4 BATCH=8 SEED=25 FAT_W=4 $PY scripts/phase1_algebra_head.py --train
echo "=== CANVAS 2/2: re-photograph + guard ==="
FID_CKPT=.cache/canvas_dose.safetensors FID_OUT=.cache/routing_fidelity_dosed.json $PY scripts/routing_fidelity.py
ALG_CKPT=.cache/canvas_dose.safetensors ALG_TEST=.cache/algebra_nl_bigtest.jsonl ALG_TEST_NAME=bigtest $PY scripts/phase1_algebra_head.py --eval | grep TOTAL
echo "=== THE CANVAS DOSE IS MEASURED ==="
