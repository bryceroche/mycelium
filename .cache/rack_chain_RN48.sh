#!/usr/bin/bash
# THE TWO-ARM DIET RUN, RE-READ AT STEP 48000 (2026-10-17). The RN chain read each arm's BEST-BY-VAL checkpoint, and the val
# split is the lettered mint (test23BP), which these mint-free arms never see: the 'best' was noise (RLK3 restored step
# 4000 at val 0.0164; RNK3 step 36000). SRK3_241's read IS its step 48000 (mint val rose monotonically to 0.8010). So both
# arms are re-read from their s48000 snapshots: the same step as the baseline. Inference only. Bars as pinned 07:02.
set -eo pipefail; cd /home/bryce/mycelium; exec > .cache/rack_chain_RN48.log 2>&1
.venv/bin/python3 scripts/contracts/preflight.py "$0" || exit 1
FAM="DEV=PCI+AMD ALG2=1 ALG_FTYPES=9 ALG_DUP=1 ALG_HW=512 ALG_WIDE=1 ALG_BREATH=7 ALG_NOTEBOOK=1 ALG_SIXWAVE=1 NB_PERSLOT=1 ALG_BINDBUS=7 ALG_BIND_D=512 BIND_CODES=.cache/bindbus_codes512.npz ALG_BUSGARAGE=2 ALG_SHELF_CIRCLE=2 ALG_ALTMASK=1 ALG_ALT21=1 ALG_ALT2=1 ALG_MASKHEAD=1 ALG_FED=1 ALG_POLAR=1 ALG_POLAR_D=128 ALG_POLAR_EM=0.1 ALG_POLAR_D_INIT=.cache/polar_waist_init_d128u.npz ALG_PRUNE=pforms,s4,fednl0,lane2 ALG_SLOT_ALL=1 ALG_STELLAR=2 ALG_CLOCK_CANON=1 SC_EVAL=0"
SURF8="ALG_ROUTER=2 R_GAIN_INIT=1.0 ALG_FREEZE=r_gain ALG_ROUTER_PTR=0.0 ALG_SPAN_ALL=1 ALG_SPAN_ARGS=1 ALG_SPAN_OP=1 ALG_SPAN_RCUE=1 ALG_SPAN_ARCUE=1 ALG_PTR_SURF=role:add:2.0 BIND_CODES=.cache/bindbus_codes512r.npz"
X="$SURF8 ALG_LOOP_STOP=3 ALG_SORT=4"
W="ALG_TEST=.cache/wild_admitted_holdout.jsonl ALG_TEST_NAME=wildholdBP"; M="ALG_TEST=.cache/algebra_nl_test.jsonl ALG_TEST_NAME=test23BP"; S="ALG_TEST=.cache/silver460_test.jsonl ALG_TEST_NAME=silver460BP"
PR() { echo "paired $1 $2 vs $3: $(.venv/bin/python3 -c "import sys,runpy; sys.argv=['paired_read.py','$4','$5','$3','$2']; runpy.run_path('scripts/paired_read.py', run_name='__main__')" | tail -1)"; }
for B in RLK3_241 RNK3_241; do C=${B}s48; K=.cache/sharp_${B}_s48000.safetensors
  echo "== $C (= $B at step 48000) ($(date +%H:%M)) =="
  for Y in "wild:$W" "mint:$M" "silver460:$S"; do F=${Y%%:*}; V=${Y#*:}
    flock -w 36000 .cache/gpu.lock env $FAM $V $X ALG_JIT_READ=1 LV_FIELDS=1 LV_DUMP=.cache/dump_open_${F}_$C.pkl LV_CKPT=$K LV_PER_SLOT=.cache/ps_open_${F}_$C.npz .venv/bin/python3 scripts/loop_val.py > .cache/read_open_${F}_$C.log 2>&1; echo "open $F $C: $(grep -aoE 'fac-exact=[0-9.]+' .cache/read_open_${F}_$C.log | tail -1) | $(grep -a '^\[fields\]' .cache/read_open_${F}_$C.log | cut -c1-170)"
  done
  flock -w 36000 .cache/gpu.lock env $FAM $W $X ALG_JIT_READ=1 LV_LEGAL=num LV_FIELDS=1 LV_DUMP=.cache/dump_wild_$C.pkl LV_CKPT=$K LV_PER_SLOT=.cache/ps_legal_wild_$C.npz .venv/bin/python3 scripts/loop_val.py > .cache/read_legal_wild_$C.log 2>&1; echo "masked wild $C: $(grep -aoE 'fac-exact=[0-9.]+' .cache/read_legal_wild_$C.log | tail -1) | $(grep -a '^\[fields\]' .cache/read_legal_wild_$C.log | cut -c1-200)"
  cp .cache/dump_wild_$C.pkl .cache/dump_wildPB0_$C.pkl
  for v in 1 2 3 4; do
    flock -w 36000 .cache/gpu.lock env $FAM ALG_TEST=.cache/wild_PB_v$v.jsonl ALG_TEST_NAME=wildPB$v $X ALG_JIT_READ=1 LV_LEGAL=num LV_FIELDS=1 LV_DUMP=.cache/dump_wildPB${v}_$C.pkl LV_CKPT=$K LV_PER_SLOT=.cache/ps_legal_wildPB${v}_$C.npz .venv/bin/python3 scripts/loop_val.py > .cache/read_legal_wildPB${v}_$C.log 2>&1 || { echo "FAILED read view $v $C"; exit 1; }
    echo "masked wild view $v $C: $(grep -aoE 'fac-exact=[0-9.]+' .cache/read_legal_wildPB${v}_$C.log | tail -1)"
  done
  .venv/bin/python3 scripts/picker/photobooth_vote_read.py $C | tee .cache/photobooth_vote_read_$C.txt | tail -8
  PR "masked wild (MINT ATTRIBUTION)" $C SRK3_241 .cache/ps_legal_wild_SRK3_241.npz .cache/ps_legal_wild_$C.npz
done
PR "masked wild (THE AUGMENTATION BAR)" RNK3_241s48 RLK3_241s48 .cache/ps_legal_wild_RLK3_241s48.npz .cache/ps_legal_wild_RNK3_241s48.npz
echo "== WHOLE PROBLEM ($(date +%H:%M)) =="
SBC_OUT=.cache/step_blame_census_RN48.txt .venv/bin/python3 scripts/picker/step_blame_census.py SRK3_241 RLK3_241s48 RNK3_241s48 > .cache/step_blame_census_RN48.log 2>&1; grep -aE "^== |STEP CREDIT|GOLD CEIL" .cache/step_blame_census_RN48.log
echo "RN48 CHAIN COMPLETE ($(date +%H:%M))"
