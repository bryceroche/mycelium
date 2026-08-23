#!/bin/bash
# pool_run.sh <id> — audition one reader-pool candidate (5k steps).
# Reads its row from docs/reader_pool.json; word-gated per round.
set -eo pipefail
cd /home/bryce/mycelium
ID=$1
CFG=$(.venv/bin/python3 -c "
import json,sys
rows=json.load(open('docs/reader_pool.json'))
r=[x for x in rows if x['id']=='$ID'][0]
print(' '.join(f'{k}={v}' for k,v in r.items()))")
echo "== POOL $ID :: $CFG =="
eval "export $CFG"
export DEV=AMD ALG2=1 ALG_DUP=1 ALG_HW=512 ALG_WIDE=1 ALG_ALLOW_PEN_TRAIN=1
MIX=.cache/form_mix23_${diet}$( [ "$incanon" = "1" ] && echo _c ).jsonl
if [ ! -f "$MIX" ]; then
  env ALG_FTYPES=8 ALG_TRUNK_LORA=1 DIET_SKEW=$diet IN_CANON=$incanon MIX_OUT=$MIX \
      .venv/bin/python3 scripts/form_assemble23.py
fi
NAME=$(basename $MIX .jsonl | sed 's/form_mix/form/')
CK=.cache/pool_${ID}.safetensors
NBENV2=""
[ "$nb" = "1" ] && NBENV2="ALG_NOTEBOOK=1 ALG_BREATH=3"
env DEV=CPU ALG2=1 ALG_FTYPES=8 ALG_DUP=1 ALG_HW=512 ALG_WIDE=1 ALG_TRUNK_LORA=1 \
    ALG_LORA_R=$rank ALG_LORA_SPAN=$span ALG_LORA_PROJ=$proj $NBENV2 \
    .venv/bin/python3 - << PYEOF2
import sys; sys.path.insert(0,'.'); sys.path.insert(0,'scripts')
from phase1_algebra_head import build_params
from tinygrad.nn.state import safe_load, safe_save
p=build_params(int("$seed"))
sd=safe_load(".cache/${lineage}.safetensors")
out={}
for k,v in p.items():
    out[k]=sd[k].to("CPU").realize() if k in sd and not k.startswith("lora") else v.to("CPU").realize()
safe_save(out,"$CK")
print(f"[pool $ID] seed ckpt: head={sum(1 for k in out if not k.startswith('lora'))} lora={sum(1 for k in out if k.startswith('lora'))}")
PYEOF2
NBENV=""
[ "$nb" = "1" ] && NBENV="ALG_NOTEBOOK=1 ALG_BREATH=3"
env ALG_FTYPES=8 ALG_TRUNK_LORA=1 ALG_LORA_R=$rank ALG_LORA_SPAN=$span ALG_LORA_PROJ=$proj \
    ALG_ROPE_OFF=$ropeoff $NBENV \
    ALG_STRAW=1 STRAW_HUMAN=$strawh OBJW_PTR=$objwptr OBJW_DIG=$objwdig ALG_LORA_SCALE=8.0 \
    RESUME=1 ALG_TRAIN=$MIX ALG_TRAIN_NAME=$NAME \
    ALG_TEST=.cache/algebra_nl_test.jsonl ALG_TEST_NAME=test23 \
    ALG_CKPT=$CK ALG_FREEZE_DUP=1 STEPS=5000 LR=1e-5 BATCH=8 SEED=$seed SNAP_EVERY=2500 \
    .venv/bin/python3 scripts/phase1_algebra_head.py --train | tail -1
echo "== POOL $ID COMPLETE =="
