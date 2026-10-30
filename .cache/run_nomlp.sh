#!/usr/bin/bash
# THE MLP-FREE STUDENT (ledger 2026-10-30, word given; the patience charter): NM (attention only) and NMDB (+ the reader DB row),
# D 768 x 8 layers, 60k steps, snapshot at 30k -> export wildhold at 30k and 60k -> masked wild read on CLK3_241's FROZEN head
set -eo pipefail; cd /home/bryce/mycelium; exec > .cache/run_nomlp.log 2>&1
while systemctl --user is-active -q dbbuild; do sleep 30; done
test -f .cache/readerdb_formpm35cBP.npy && test -f .cache/readerdb_wildholdBP.npy
echo "== start $(date +%H:%M) =="
PY=.venv/bin/python3; A="DEV=PCI+AMD DR_FF=0 DR_D=768 DR_LAYERS=8"
flock -w 36000 .cache/gpu.lock env $A DR_DB=1 DR_STEPS=21 DR_TAG=smokeNMDB $PY scripts/distill_reader.py; rm -f .cache/distill_reader_smokeNMDB.*
FAM=$(grep -oP '^FAM="\K[^"]+' .cache/rack_chain_SEVCLK.sh); SURF8=$(grep -oP '^SURF8="\K[^"]+' .cache/rack_chain_SEVCLK.sh)
for ARM in NM NMDB; do
  DB=0; [ "$ARM" = NMDB ] && DB=1; TAG=$(echo $ARM | tr A-Z a-z)768x8
  flock -w 36000 .cache/gpu.lock env $A DR_DB=$DB DR_STEPS=60000 DR_SAVE_AT=30000 DR_TAG=$TAG $PY scripts/distill_reader.py
  for S in 30 60; do
    CK=.cache/distill_reader_$TAG.safetensors; [ $S = 30 ] && CK=.cache/distill_reader_${TAG}_s30000.safetensors
    flock -w 36000 .cache/gpu.lock env $A DR_DB=$DB DR_TAG=$TAG DR_CKPT=$CK $PY scripts/distill_reader.py --export .cache/wild_admitted_holdout.jsonl --src wildholdBP --out wildhold$ARM$S
    flock -w 36000 .cache/gpu.lock env $FAM ALG_TEST=.cache/wild_admitted_holdout.jsonl ALG_TEST_NAME=wildhold$ARM$S $SURF8 ALG_LOOP_STOP=3 ALG_SORT=4 ALG_CLIMB=1 ALG_JIT_READ=1 LV_LEGAL=num LV_FIELDS=1 LV_CKPT=.cache/sharp_CLK3_241.safetensors LV_PER_SLOT=.cache/ps_nomlp_$ARM$S.npz $PY scripts/loop_val.py > .cache/read_nomlp_$ARM$S.log 2>&1
    echo "$ARM@${S}k masked wild FROZEN CLK3_241 head: $(grep -aoE 'fac-exact=[0-9.]+' .cache/read_nomlp_$ARM$S.log | tail -1) | $(grep -a '^\[fields\]' .cache/read_nomlp_$ARM$S.log | cut -c1-200)"
    echo "  vs TEACHER: $($PY scripts/paired_read.py .cache/ps_sevclk_open_CLK3_241.npz .cache/ps_nomlp_$ARM$S.npz teacher $ARM$S)"
    echo "  vs SwiGLU student: $($PY scripts/paired_read.py .cache/ps_distill_s6x512.npz .cache/ps_nomlp_$ARM$S.npz swiglu $ARM$S)"
  done
done
echo "== done $(date +%H:%M) =="
