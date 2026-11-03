#!/usr/bin/bash
# MBR-C (ledger 11-02, the param-matched control: no breath loop). From MBR THE BREATHING READER (Bryce 11-02 "lets do this"): after lfq5's last arm and CY's GSM8K export, patch distill_reader.py
# (breathing_reader + val_alias_fix), smoke, 30k, wild export, frozen CLK3_241 read, paired reads.
set -eo pipefail; cd /home/bryce/mycelium; exec > .cache/run_lfmbrc.log 2>&1
echo "== start $(date +%H:%M) =="



PY=.venv/bin/python3
A="DEV=PCI+AMD DR_FF=0 DR_D=768 DR_LAYERS=2 DR_DB=1 DR_T6=3 DR_T9NS=1 DR_BREATHS=0"
FAM2=$(grep -oP '^FAM="\K[^"]+' .cache/rack_chain_SEVCLK.sh); SURF2=$(grep -oP '^SURF8="\K[^"]+' .cache/rack_chain_SEVCLK.sh)
TAG=mbrc2x1; OUT=MBRC
flock -w 36000 .cache/gpu.lock env $A DR_TAG=smk$TAG DR_STEPS=21 $PY scripts/distill_reader.py
rm -f .cache/distill_reader_smk$TAG.*
flock -w 36000 .cache/gpu.lock env $A DR_TAG=$TAG DR_STEPS=30000 $PY scripts/distill_reader.py
flock -w 36000 .cache/gpu.lock env $A DR_TAG=$TAG $PY scripts/distill_reader.py --export .cache/wild_admitted_holdout.jsonl --src wildholdBP --out wildhold$OUT
flock -w 36000 .cache/gpu.lock env $FAM2 ALG_TEST=.cache/wild_admitted_holdout.jsonl ALG_TEST_NAME=wildhold$OUT $SURF2 ALG_LOOP_STOP=3 ALG_SORT=4 ALG_CLIMB=1 ALG_JIT_READ=1 LV_LEGAL=num LV_FIELDS=1 LV_CKPT=.cache/sharp_CLK3_241.safetensors LV_PER_SLOT=.cache/ps_nomlp_$OUT.npz $PY scripts/loop_val.py > .cache/read_nomlp_$OUT.log 2>&1
echo "masked wild $OUT frozen CLK3_241: $(grep -aoE 'fac-exact=[0-9.]+' .cache/read_nomlp_$OUT.log | tail -1) | $(grep -a '^\[fields\]' .cache/read_nomlp_$OUT.log | cut -c1-200)"
for C in MBR NMDBT9NS NMDBT9NSR1; do echo "  PAIRED $C vs $OUT: $($PY scripts/paired_read.py .cache/ps_nomlp_$C.npz .cache/ps_nomlp_$OUT.npz $C $OUT)"; done
echo "== done $(date +%H:%M) =="
