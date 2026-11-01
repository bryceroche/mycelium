#!/usr/bin/bash
# HY (Bryce 11-01 "word given!"; ledger 09:42): the MLP-free DB student with RoPE kept on planes [0:24) and T6 x the SENTENCE on
# RoPE's 8 idle planes [24:32) (DR_T6=2) -- smoke, 30k (NMDB30's budget), wild export, frozen CLK3_241 swap, paired vs NMDB30 + teacher.
set -eo pipefail; cd /home/bryce/mycelium; exec > .cache/run_lfhy.log 2>&1
echo "== start $(date +%H:%M) =="
PY=.venv/bin/python3; A="DEV=PCI+AMD DR_FF=0 DR_D=768 DR_LAYERS=8 DR_DB=1 DR_T6=2"
flock -w 36000 .cache/gpu.lock env $A DR_TAG=smokeHY DR_STEPS=21 $PY scripts/distill_reader.py
flock -w 36000 .cache/gpu.lock env $A DR_TAG=nmdbhy768x8 DR_STEPS=30000 $PY scripts/distill_reader.py
flock -w 36000 .cache/gpu.lock env $A DR_TAG=nmdbhy768x8 $PY scripts/distill_reader.py --export .cache/wild_admitted_holdout.jsonl --src wildholdBP --out wildholdNMDBHY
FAM2=$(grep -oP '^FAM="\K[^"]+' .cache/rack_chain_SEVCLK.sh); SURF2=$(grep -oP '^SURF8="\K[^"]+' .cache/rack_chain_SEVCLK.sh)
flock -w 36000 .cache/gpu.lock env $FAM2 ALG_TEST=.cache/wild_admitted_holdout.jsonl ALG_TEST_NAME=wildholdNMDBHY $SURF2 ALG_LOOP_STOP=3 ALG_SORT=4 ALG_CLIMB=1 ALG_JIT_READ=1 LV_LEGAL=num LV_FIELDS=1 LV_CKPT=.cache/sharp_CLK3_241.safetensors LV_PER_SLOT=.cache/ps_nomlp_NMDBHY.npz $PY scripts/loop_val.py > .cache/read_nomlp_NMDBHY.log 2>&1
echo "masked wild NMDBHY frozen CLK3_241: $(grep -aoE 'fac-exact=[0-9.]+' .cache/read_nomlp_NMDBHY.log | tail -1) | $(grep -a '^\[fields\]' .cache/read_nomlp_NMDBHY.log | cut -c1-200)"
echo "  PAIRED NMDB30 vs NMDBHY: $($PY scripts/paired_read.py .cache/ps_nomlp_NMDB30.npz .cache/ps_nomlp_NMDBHY.npz NMDB30 NMDBHY)"
echo "  PAIRED NMDBT6 vs NMDBHY: $($PY scripts/paired_read.py .cache/ps_nomlp_NMDBT6.npz .cache/ps_nomlp_NMDBHY.npz NMDBT6 NMDBHY)"
echo "  PAIRED teacher vs NMDBHY: $($PY scripts/paired_read.py .cache/ps_sevclk_open_CLK3_241.npz .cache/ps_nomlp_NMDBHY.npz teacher NMDBHY)"
echo "== done $(date +%H:%M) =="
