#!/bin/bash
# LEG A PILOT (2026-08-26, word given): opc heads onto three
# register-repaired readers (R01 g55/bal, R17 g41/chain/rope23,
# R20 g47/pert_c/r4wod). Continuation from each pool ckpt (bank ckpts
# untouched), own diet (g_opc injected), 4k steps. Then the LoRA-aware
# opc read on the 143 golds. BAR: op-only exact > 14/143 on ANY pilot.
set -eo pipefail
cd /home/bryce/mycelium
PY=.venv/bin/python3

run_pilot () {
  ID=$1; rank=$2; span=$3; proj=$4; ropeoff=$5; objwptr=$6; objwdig=$7; seed=$8; mix=$9
  NAME=$(basename $mix .jsonl | sed 's/form_mix/form/')
  echo "== LEG A PILOT $ID (mix $NAME) =="
  env DEV=AMD ALG2=1 ALG_FTYPES=8 ALG_DUP=1 ALG_HW=512 ALG_WIDE=1 \
      ALG_TRUNK_LORA=1 ALG_LORA_R=$rank ALG_LORA_SPAN=$span ALG_LORA_PROJ=$proj \
      ALG_ROPE_OFF=$ropeoff ALG_LORA_SCALE=8.0 \
      ALG_STRAW=1 STRAW_HUMAN=3.0 OBJW_PTR=$objwptr OBJW_DIG=$objwdig \
      ALG_OPCOUNT=1 ALG_ALLOW_PEN_TRAIN=1 ALG_FREEZE_DUP=1 \
      WARM_FROM=.cache/pool_${ID}.safetensors \
      ALG_TRAIN=$mix ALG_TRAIN_NAME=$NAME \
      ALG_TEST=.cache/algebra_nl_test.jsonl ALG_TEST_NAME=test23 \
      ALG_CKPT=.cache/pool_${ID}_opc.safetensors \
      STEPS=${STEPS:-4000} LR=1e-5 BATCH=8 SEED=$((seed+300)) SNAP_EVERY=0 \
      $PY scripts/phase1_algebra_head.py --train 2>&1 | tail -1
  env DEV=AMD ALG2=1 ALG_FTYPES=8 ALG_DUP=1 ALG_HW=512 ALG_WIDE=1 \
      ALG_TRUNK_LORA=1 ALG_LORA_R=$rank ALG_LORA_SPAN=$span ALG_LORA_PROJ=$proj \
      ALG_ROPE_OFF=$ropeoff ALG_LORA_SCALE=8.0 ALG_OPCOUNT=1 \
      OPCL_CKPT=pool_${ID}_opc \
      ALG_TEST=.cache/algebra_nl_bigtest.jsonl ALG_TEST_NAME=bigtest \
      $PY scripts/opc_lora_read.py 2>&1 | grep opcl
}

run_pilot R01 16 0123 all 0  1.0 1.0 200 .cache/form_mix23_bal.jsonl
run_pilot R17 8  0123 wq  23 1.0 2.0 216 .cache/form_mix23_chain.jsonl
run_pilot R20 4  0123 wod 0  1.0 1.0 219 .cache/form_mix23_pert_c.jsonl
echo "== LEG A PILOT COMPLETE =="
