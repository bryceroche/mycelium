#!/bin/bash
# hire_continue.sh — the 16 hires continue-train 5k->10k (their full depth).
set -o pipefail
cd /home/bryce/mycelium
PY=.venv/bin/python3
for ID in $($PY -c "import json; print(' '.join(json.load(open('.cache/recruit_round1.json'))['hires']))"); do
  if [ -f ".cache/pool_${ID}.cont" ]; then echo "==== CONT $ID done, skip ===="; continue; fi
  CFG=$($PY -c "
import json
r=[x for x in json.load(open('docs/reader_pool.json')) if x['id']=='$ID'][0]
print(' '.join(f'{k}={v}' for k,v in r.items()))")
  eval "export $CFG"
  export DEV=AMD ALG2=1 ALG_DUP=1 ALG_HW=512 ALG_WIDE=1 ALG_ALLOW_PEN_TRAIN=1
  CSUF=""
  if [ "$incanon" = "1" ]; then CSUF="_c"; fi
  MIX=.cache/form_mix23_${diet}${CSUF}.jsonl
  NAME=$(basename $MIX .jsonl | sed 's/form_mix/form/')
  NBENV=""
  if [ "$nb" = "1" ]; then NBENV="ALG_NOTEBOOK=1 ALG_BREATH=3"; fi
  echo "==== CONT $ID START $(date +%H:%M) ===="
  if env ALG_FTYPES=8 ALG_TRUNK_LORA=1 ALG_LORA_R=$rank ALG_LORA_SPAN=$span ALG_LORA_PROJ=$proj \
      ALG_ROPE_OFF=$ropeoff $NBENV \
      ALG_STRAW=1 STRAW_HUMAN=$strawh OBJW_PTR=$objwptr OBJW_DIG=$objwdig ALG_LORA_SCALE=8.0 \
      RESUME=1 ALG_TRAIN=$MIX ALG_TRAIN_NAME=$NAME \
      ALG_TEST=.cache/algebra_nl_test.jsonl ALG_TEST_NAME=test23 \
      ALG_CKPT=.cache/pool_${ID}.safetensors ALG_FREEZE_DUP=1 STEPS=5000 LR=1e-5 BATCH=8 SEED=$((seed+100)) SNAP_EVERY=0 \
      $PY scripts/phase1_algebra_head.py --train | tail -1; then
    touch ".cache/pool_${ID}.cont"
    echo "==== CONT $ID DONE $(date +%H:%M) ===="
  else
    echo "==== CONT $ID FAILED (continuing) ===="
  fi
done
echo "==== HIRE CONTINUATION COMPLETE ===="
