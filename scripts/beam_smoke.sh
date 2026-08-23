#!/bin/bash
# beam_smoke.sh — BEAM/TC kernel-tuning smoke (2026-08-23, word given).
# Three 1000-step fires, identical recipe/seed, BEAM={0,2,4}. Bars:
# steps/s ratio; test23 eval TOTAL within +/-8 of control (kernels change,
# math must not); capture time reported. Runs AFTER G65 frees the GPU.
set -eo pipefail
cd /home/bryce/mycelium
PY=.venv/bin/python3
export DEV=AMD ALG2=1 ALG_DUP=1 ALG_HW=512 ALG_WIDE=1 ALG_ALLOW_PEN_TRAIN=1
for B in 0 2 4; do
  CK=.cache/beam_smoke_$B.safetensors
  $PY - << PYEOF2
import numpy as np, sys
sys.path.insert(0,'.')
from tinygrad.nn.state import safe_load, safe_save
from tinygrad import Tensor
sd=safe_load('.cache/g55_bridge.safetensors')
out={k:v.to("CPU").realize() for k,v in sd.items()}
rng=np.random.RandomState(160+7000)
for li in range(4):
    for nm,din in (("wq",2048),("wo",2048),("wdown",8192)):
        out[f"lora{li}_{nm}_A"]=Tensor((rng.randn(din,16)*0.01).astype(np.float32))
        out[f"lora{li}_{nm}_B"]=Tensor(np.zeros((16,2048),np.float32))
safe_save(out,"$CK")
PYEOF2
  echo "== BEAM=$B: 1000 steps =="
  T0=$(date +%s)
  env BEAM=$B ALG_FTYPES=8 ALG_TRUNK_LORA=1 ALG_LORA_SCALE=8.0 RESUME=1 \
      ALG_TRAIN=.cache/form_mix22.jsonl ALG_TRAIN_NAME=form22 \
      ALG_TEST=.cache/algebra_nl_test.jsonl ALG_TEST_NAME=test23 \
      ALG_CKPT=$CK ALG_FREEZE_DUP=1 STEPS=1000 LR=1e-5 BATCH=8 SEED=160 SNAP_EVERY=0 \
      $PY scripts/phase1_algebra_head.py --train | tail -1
  T1=$(date +%s)
  echo "== BEAM=$B WALL $((T1-T0))s =="
  env ALG_FTYPES=8 ALG_TRUNK_LORA=1 ALG_CKPT=$CK \
      ALG_TEST=.cache/algebra_nl_test.jsonl ALG_TEST_NAME=test23 \
      $PY scripts/phase1_algebra_head.py --eval | grep TOTAL
done
echo "== BEAM SMOKE COMPLETE =="
