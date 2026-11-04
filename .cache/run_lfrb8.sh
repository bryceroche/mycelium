#!/usr/bin/bash
# RB8 THE READING BREATHS DESCEND THE TREE (ledger 11-03; word given for the whole queue): after lf8m's GSM8K export, apply
# scripts/patches/rb8.py, smoke, 30k on the 8M rig, wild export, frozen CLK3_241 read, paired reads vs MBR / MBRC.
set -eo pipefail; cd /home/bryce/mycelium; exec > .cache/run_lfrb8.log 2>&1
echo "== start $(date +%H:%M) =="; PY=.venv/bin/python3
while ! grep -q "^\[gsm\]" .cache/run_lf8m.log 2>/dev/null || systemctl --user is-active -q lfbeam8; do grep -q "Traceback" .cache/run_lf8m.log 2>/dev/null && { echo "lf8m failed"; exit 1; }; sleep 60; done
while pgrep -f 'scripts/distill_reader.py' >/dev/null; do sleep 30; done
$PY scripts/patches/rb8.py
A="DEV=PCI+AMD DR_FF=0 DR_D=768 DR_LAYERS=2 DR_DB=1 DR_T6=3 DR_T9NS=1 DR_BREATHS=4 DR_RB=1"
FAM2=$(grep -oP '^FAM="\K[^"]+' .cache/rack_chain_SEVCLK.sh); SURF2=$(grep -oP '^SURF8="\K[^"]+' .cache/rack_chain_SEVCLK.sh)
TAG=rb8x4; OUT=RB8
flock -w 36000 .cache/gpu.lock env $A DR_TAG=smk$TAG DR_STEPS=21 $PY scripts/distill_reader.py
rm -f .cache/distill_reader_smk$TAG.*
flock -w 36000 .cache/gpu.lock env $A DR_TAG=$TAG DR_STEPS=30000 $PY scripts/distill_reader.py
flock -w 36000 .cache/gpu.lock env $A DR_TAG=$TAG $PY scripts/distill_reader.py --export .cache/wild_admitted_holdout.jsonl --src wildholdBP --out wildhold$OUT
flock -w 36000 .cache/gpu.lock env $FAM2 ALG_TEST=.cache/wild_admitted_holdout.jsonl ALG_TEST_NAME=wildhold$OUT $SURF2 ALG_LOOP_STOP=3 ALG_SORT=4 ALG_CLIMB=1 ALG_JIT_READ=1 LV_LEGAL=num LV_FIELDS=1 LV_CKPT=.cache/sharp_CLK3_241.safetensors LV_PER_SLOT=.cache/ps_nomlp_$OUT.npz $PY scripts/loop_val.py > .cache/read_nomlp_$OUT.log 2>&1
echo "masked wild $OUT frozen CLK3_241: $(grep -aoE 'fac-exact=[0-9.]+' .cache/read_nomlp_$OUT.log | tail -1) | $(grep -a '^\[fields\]' .cache/read_nomlp_$OUT.log | cut -c1-200)"
for C in MBR MBRC MBR22; do echo "  PAIRED $C vs $OUT: $($PY scripts/paired_read.py .cache/ps_nomlp_$C.npz .cache/ps_nomlp_$OUT.npz $C $OUT)"; done
echo "== done $(date +%H:%M) =="
