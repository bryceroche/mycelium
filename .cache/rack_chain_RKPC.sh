#!/usr/bin/bash
# THE RACK ON THE PLAIN CHASSIS (2026-10-06): RKPC_241 = PMS8_241's recipe (SURF8, no hierarchical state) + ALG_ALT3=1 ALG_CERT=2.0 (the control) from scratch, 48k. Bars: RKP vs RKPC paired masked >= +0.015; rows >= 16 (the rack's record 17 on HS needs a second body); mint vs ALT3_241's 0.661 (the consults are free on this chassis).
# BARS (ledger 10-05 'WORD GIVEN: THE RACK', pinned before the build): RK vs RKC paired masked wild >= +0.015 (a CLAIM needs +0.020 or the twin); rows >= +2; mint >= -0.01; digits not down; MECHANISM: right slots' token hit >= 0.76 flat; THE WALL moves >= +0.03 from 0.526; DRY CENSUS on wild: share committed + precision >= 0.90. KILL: masked < RKC - 0.010 or dry precision < 0.80.
BODY=PMS8_241; TAU=0
set -eo pipefail; cd /home/bryce/mycelium; exec > .cache/rack_chain_RKPC.log 2>&1
.venv/bin/python3 scripts/contracts/preflight.py "$0" || exit 1   # THE PRE-FLIGHT CONTRACT
FAM="DEV=PCI+AMD ALG2=1 ALG_FTYPES=9 ALG_DUP=1 ALG_HW=512 ALG_WIDE=1 ALG_BREATH=7 ALG_NOTEBOOK=1 ALG_SIXWAVE=1 NB_PERSLOT=1 ALG_BINDBUS=7 ALG_BIND_D=512 BIND_CODES=.cache/bindbus_codes512.npz ALG_BUSGARAGE=2 ALG_SHELF_CIRCLE=2 ALG_ALTMASK=1 ALG_ALT21=1 ALG_ALT2=1 ALG_MASKHEAD=1 ALG_FED=1 ALG_POLAR=1 ALG_POLAR_D=128 ALG_POLAR_EM=0.1 ALG_POLAR_D_INIT=.cache/polar_waist_init_d128u.npz ALG_PRUNE=pforms,s4,fednl0,lane2 ALG_SLOT_ALL=1 ALG_STELLAR=2 ALG_CLOCK_CANON=1 SC_EVAL=0"
TR="ALG_ALLOW_PEN_TRAIN=1 ALG_TRAIN=.cache/form_mix_pm35c.jsonl ALG_TRAIN_NAME=formpm35c ALG_TEST=.cache/algebra_nl_test.jsonl ALG_TEST_NAME=test23 ALG_MASKPREP_CACHE=1 ALG_MASKPREP_JIT=1 ALG_FACTS_POOL=1 ALG_FACTS_WORKERS=6 ALG_MASKPREP_IGNORE=ALG_MASK_COOK,ALG_MASK_COOK_SKEL,ALG_TOK_COOK,ALG_TOK_COOK_S3,ALG_TG_INIT,ALG_BAL_COOK,ALG_NB2,ALG_ALT5,ALG_PRUNE,ALG_MASKPREP_JIT,ALG_MASKPREP_B,ALG_FACTS_POOL,ALG_FACTS_FLUSH,ALG_JIT_VAL LR=1e-4 SEED=241 SNAP_EVERY=100000"
SURF8="${SURF8:-ALG_ROUTER=2 R_GAIN_INIT=1.0 ALG_FREEZE=r_gain ALG_ROUTER_PTR=0.0 ALG_SPAN_ALL=1 ALG_SPAN_ARGS=1 ALG_SPAN_OP=1 ALG_SPAN_RCUE=1 ALG_SPAN_ARCUE=1 ALG_PTR_SURF=role:add:2.0 BIND_CODES=.cache/bindbus_codes512r.npz}"
W="ALG_TEST=.cache/wild_admitted_holdout.jsonl ALG_TEST_NAME=wildhold"; M="ALG_TEST=.cache/algebra_nl_test.jsonl ALG_TEST_NAME=test23"; S="ALG_TEST=.cache/silver460_test.jsonl ALG_TEST_NAME=silver460"
grep -aq "gate unset: step     0 loss=5.2995" .cache/rack_merge_gate.log || { echo "ABORT: the post-merge gate did not certify the head bit-identical unset"; exit 1; }
grep -a "gate rack:" .cache/rack_merge_gate.log | tail -1 | grep -aq "loss=[0-9]" || { echo "ABORT: the rack config does not run on the fixture"; exit 1; }
for ARM in "RKPC_241:$SURF8 ALG_ALT3=1 ALG_CERT=2.0 SNAP_EVERY=8000"; do C=${ARM%%:*}; X=${ARM#*:}
  echo "== ARM $C ($X; random init; form_mix_pm35; 48k, B=8) ($(date +%H:%M)) =="
  flock -w 36000 .cache/gpu.lock env $FAM $TR $X BATCH=8 STEPS=48000 ALG_CKPT=.cache/sharp_$C.safetensors PYTHONUNBUFFERED=1 .venv/bin/python3 -u scripts/phase1_algebra_head.py --train > .cache/sharp_$C.log 2>&1 || { echo "FAILED $C: $(grep -aE 'Error|assert' .cache/sharp_$C.log | tail -1)"; exit 1; }
  grep -aE "router|xcorr\]|anchor|span-all|step +(0|24000|47999) " .cache/sharp_$C.log | tail -8 | cut -c1-200 || true
  for Y in "wild:$W" "mint:$M" "silver460:$S"; do F=${Y%%:*}; V=${Y#*:}
    flock -w 36000 .cache/gpu.lock env $FAM $V $X ALG_JIT_READ=1 LV_FIELDS=1 LV_DUMP=.cache/dump_${F}_$C.pkl LV_CKPT=.cache/sharp_$C.safetensors LV_PER_SLOT=.cache/ps_open_${F}_$C.npz .venv/bin/python3 scripts/loop_val.py > .cache/read_open_${F}_$C.log 2>&1; echo "open $F $C: $(grep -aoE 'fac-exact=[0-9.]+' .cache/read_open_${F}_$C.log | tail -1) | $(grep -a '^\[fields\]' .cache/read_open_${F}_$C.log | cut -c1-170)"
  done
  flock -w 36000 .cache/gpu.lock env $FAM $W $X ALG_JIT_READ=1 LV_LEGAL=num LV_FIELDS=1 LV_CKPT=.cache/sharp_$C.safetensors LV_PER_SLOT=.cache/ps_legal_wild_$C.npz .venv/bin/python3 scripts/loop_val.py > .cache/read_legal_wild_$C.log 2>&1; echo "masked wild $C: $(grep -aoE 'fac-exact=[0-9.]+' .cache/read_legal_wild_$C.log | tail -1)"
  echo "matched $C: $(.venv/bin/python3 scripts/matched_read.py .cache/dump_wild_$C.pkl | cut -c1-200)"
  for MK in 0 1; do flock -w 36000 .cache/gpu.lock env $FAM $W $X CA_CKPT=.cache/sharp_$C.safetensors CA_MASK=$MK .venv/bin/python3 scripts/chain_acc.py 2>&1 | grep -a "chain-acc\]" | tail -1 || true; done
done
for C in RKPC_241; do X="$SURF8 ALG_ALT3=1 ALG_CERT=2.0"   # RKPC is the rack-OFF control: never add ALG_RACK here (audit 2026-10-06: the old "[ $C = RK_241 ]" guard was stale copy-paste from an earlier two-arm template and could never match this loop's own C, which is RKPC_241 -- harmless today only by coincidence, so it is removed rather than "fixed" to compare against RKPC_241, which would silently turn the control on)
  echo "paired masked wild $C vs $BODY (the body): $(.venv/bin/python3 scripts/paired_read.py .cache/ps_legal_wild_$BODY.npz .cache/ps_legal_wild_$C.npz $BODY $C | tail -1)"
  echo "paired masked wild $C vs RKC_241 (the same arm on the hierarchical body): $(.venv/bin/python3 scripts/paired_read.py .cache/ps_legal_wild_RKC_241.npz .cache/ps_legal_wild_$C.npz RKC_241 $C | tail -1)"
  echo "paired masked wild $C vs PMS8_241: $(.venv/bin/python3 scripts/paired_read.py .cache/ps_legal_wild_PMS8_241.npz .cache/ps_legal_wild_$C.npz PMS8_241 $C | tail -1)"
  $(.venv/bin/python3 scripts/contracts/golden.py command .cache/sharp_$C.safetensors --tag $C 2>/dev/null | grep -v "^#" | head -1 | sed "s|ALG_FTYPES=9|ALG_FTYPES=9 $X|")
  .venv/bin/python3 scripts/contracts/golden.py check .cache/golden24_rows_$C.json --tag $C || echo "golden: see above"
  flock -w 36000 .cache/gpu.lock env $FAM $W $X CB_MODE=collect CKPT=.cache/sharp_$C.safetensors .venv/bin/python3 scripts/clock_band_probe.py > .cache/clock_band_collect_$C.log 2>&1 || echo "band-probe collect FAILED $C"
  flock -w 36000 .cache/gpu.lock env $FAM $W $X MS_CKPT=.cache/sharp_$C.safetensors MS_MODE=collect .venv/bin/python3 scripts/membrane_scale.py > .cache/membrane_scale_collect_$C.log 2>&1 || echo "membrane collect FAILED $C"
  flock -w 36000 .cache/gpu.lock env $FAM $W $X MS_MODE=report MS_CKPT=.cache/sharp_$C.safetensors .venv/bin/python3 scripts/membrane_scale.py > .cache/membrane_scale_report_$C.log 2>&1 || echo "FAILED report $C"
  grep -A4 "argmax band share, WRONG" .cache/membrane_scale_$C.txt | grep other_sent | sed "s/^/WALL $C /" || true; grep -A4 "argmax band share, RIGHT" .cache/membrane_scale_$C.txt | grep "  token" | sed "s/^/HIT $C /" || true
  [ -f scripts/rack_dryness_census.py ] && .venv/bin/python3 scripts/rack_dryness_census.py .cache/dump_wild_$C.pkl --tag $C 2>&1 | tail -6 | sed "s/^/DRY $C /"
done
echo "paired masked wild RKP_241 vs RKPC_241 (THE BAR: >= +0.015 paired; claim +0.020): $(.venv/bin/python3 scripts/paired_read.py .cache/ps_legal_wild_RKPC_241.npz .cache/ps_legal_wild_RKP_241.npz RKPC_241 RKP_241 | tail -1)"
.venv/bin/python3 scripts/reads.py ingest | tail -1; echo "RACK RKPC CHAIN COMPLETE ($(date +%H:%M))"
