#!/usr/bin/bash
# RB8 TWIN (ledger 11-04; word given for the whole queue): seed 1 of RB8 and of MBR on the 8M rig (DR_SEED=1, both with BEAM=2 PARALLEL=0),
# wild export, frozen CLK3_241 read, paired reads; RB-T verdict over both seed pairs (seed 0 = the banked RB8 / MBR).
set -eo pipefail; cd /home/bryce/mycelium; exec >> .cache/run_lftwin.log 2>&1
echo "== start $(date +%H:%M) =="; PY=.venv/bin/python3
FAM2=$(grep -oP '^FAM="\K[^"]+' .cache/rack_chain_SEVCLK.sh); SURF2=$(grep -oP '^SURF8="\K[^"]+' .cache/rack_chain_SEVCLK.sh)
A0="DEV=PCI+AMD DR_FF=0 DR_D=768 DR_LAYERS=2 DR_DB=1 DR_T6=3 DR_T9NS=1 DR_BREATHS=4 DR_SEED=1"
for ARM in RB8 MBR; do
  if [ $ARM = RB8 ]; then A="$A0 DR_RB=1"; TAG=rb8x4r1; else A="$A0"; TAG=mbr2x4r1; fi; OUT=${ARM}R1
  flock -w 36000 .cache/gpu.lock env $A BEAM=2 PARALLEL=0 DR_TAG=$TAG DR_STEPS=30000 $PY scripts/distill_reader.py
  flock -w 36000 .cache/gpu.lock env $A DR_TAG=$TAG $PY scripts/distill_reader.py --export .cache/wild_admitted_holdout.jsonl --src wildholdBP --out wildhold$OUT
  flock -w 36000 .cache/gpu.lock env $FAM2 ALG_TEST=.cache/wild_admitted_holdout.jsonl ALG_TEST_NAME=wildhold$OUT $SURF2 ALG_LOOP_STOP=3 ALG_SORT=4 ALG_CLIMB=1 ALG_JIT_READ=1 LV_LEGAL=num LV_FIELDS=1 LV_CKPT=.cache/sharp_CLK3_241.safetensors LV_PER_SLOT=.cache/ps_nomlp_$OUT.npz $PY scripts/loop_val.py
  echo "masked wild $OUT frozen CLK3_241: $(grep -aoE 'fac-exact=[0-9.]+' .cache/read_nomlp_$OUT.log | tail -1)"
done
P0=$($PY scripts/paired_read.py .cache/ps_nomlp_MBR.npz .cache/ps_nomlp_RB8.npz MBR RB8); P1=$($PY scripts/paired_read.py .cache/ps_nomlp_MBRR1.npz .cache/ps_nomlp_RB8R1.npz MBRR1 RB8R1)
echo "  seed 0: $P0"; echo "  seed 1: $P1"
for C in MBR22 RB8; do echo "  beside $C vs RB8R1: $($PY scripts/paired_read.py .cache/ps_nomlp_$C.npz .cache/ps_nomlp_RB8R1.npz $C RB8R1)"; done
$PY - "$P0" "$P1" <<'PY'
import re, sys
d = [-float(re.search(r"diff ([+-][0-9.]+)", s).group(1)) for s in sys.argv[1:]]   # paired_read prints A - B = MBR - RB8
m = sum(d) / 2
print(f"RB-T: RB8 - MBR wild {d[0]:+.4f} / {d[1]:+.4f} (mean {m:+.4f}; bar mean >= +0.010 and both > 0) -> {'PASS' if m >= 0.010 and min(d) > 0 else 'FAIL'}")
PY
echo "== done $(date +%H:%M) =="
