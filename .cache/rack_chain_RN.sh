#!/usr/bin/bash
# THE TWO-ARM DIET RUN (2026-10-17; Bryce relayed Opus: "yes, the two-arm run: GSM8K only, and GSM8K plus name swaps ... Fable
# has the word"). SRK3_241's recipe exactly (K3 + BP states + the sorting room, ORIGINAL positional frame, 48k, B=8, seed 241);
# ONE change per arm:
#   RLK3_241  THE REAL-ONLY DIET: form_mix_real (30,533 rows) = form_mix_pm35c minus every lettered (mint / pen-dialect) row
#             and minus the leaked wild text (scripts/build_real_diet.py). vs SRK3_241 = THE MINT ATTRIBUTION.
#   RNK3_241  + NAME VIEWS: form_mix_realPB (47,346) = form_mix_real + 16,813 photobooth name views (PB_WIDE=1, one view per
#             name-bearing row; honesty-checked). vs RLK3_241 = THE NAME AUGMENTATION.
# DOSE (declared both ways): A 30,533 rows x 12.6 reps; B 47,346 rows x 8.1 reps, views = 35.5% of the mix; a name-bearing
#   problem is seen 8.1x as itself + 8.1x renamed, a name-free row 8.1x (vs 12.6x in A). EQUAL CARD: the augmentation pays
#   its way out of the same 384k samples (declared confound: B's originals get 36% fewer reps).
# STATES: sliced from formpm35cBP, never recomputed (scripts/diet_stage.py: text / tokmask / sent / gold bit-identical gates);
#   the views' states by precompute_masked --mask bidirpad, gated by photobooth_stage.py.
# BARS: pinned in the ledger 2026-10-17 'THE TWO-ARM DIET RUN', before firing.
set -eo pipefail; cd /home/bryce/mycelium; exec > .cache/rack_chain_RN.log 2>&1
.venv/bin/python3 scripts/contracts/preflight.py "$0" || exit 1   # THE PRE-FLIGHT CONTRACT
FAM="DEV=PCI+AMD ALG2=1 ALG_FTYPES=9 ALG_DUP=1 ALG_HW=512 ALG_WIDE=1 ALG_BREATH=7 ALG_NOTEBOOK=1 ALG_SIXWAVE=1 NB_PERSLOT=1 ALG_BINDBUS=7 ALG_BIND_D=512 BIND_CODES=.cache/bindbus_codes512.npz ALG_BUSGARAGE=2 ALG_SHELF_CIRCLE=2 ALG_ALTMASK=1 ALG_ALT21=1 ALG_ALT2=1 ALG_MASKHEAD=1 ALG_FED=1 ALG_POLAR=1 ALG_POLAR_D=128 ALG_POLAR_EM=0.1 ALG_POLAR_D_INIT=.cache/polar_waist_init_d128u.npz ALG_PRUNE=pforms,s4,fednl0,lane2 ALG_SLOT_ALL=1 ALG_STELLAR=2 ALG_CLOCK_CANON=1 SC_EVAL=0"
TRC="ALG_ALLOW_PEN_TRAIN=1 ALG_TEST=.cache/algebra_nl_test.jsonl ALG_TEST_NAME=test23BP ALG_MASKPREP_CACHE=1 ALG_MASKPREP_JIT=1 ALG_FACTS_POOL=1 ALG_FACTS_WORKERS=6 ALG_MASKPREP_IGNORE=ALG_MASK_COOK,ALG_MASK_COOK_SKEL,ALG_TOK_COOK,ALG_TOK_COOK_S3,ALG_TG_INIT,ALG_BAL_COOK,ALG_NB2,ALG_ALT5,ALG_PRUNE,ALG_MASKPREP_JIT,ALG_MASKPREP_B,ALG_FACTS_POOL,ALG_FACTS_FLUSH,ALG_JIT_VAL LR=1e-4 SEED=241 SNAP_EVERY=100000"
SURF8="ALG_ROUTER=2 R_GAIN_INIT=1.0 ALG_FREEZE=r_gain ALG_ROUTER_PTR=0.0 ALG_SPAN_ALL=1 ALG_SPAN_ARGS=1 ALG_SPAN_OP=1 ALG_SPAN_RCUE=1 ALG_SPAN_ARCUE=1 ALG_PTR_SURF=role:add:2.0 BIND_CODES=.cache/bindbus_codes512r.npz"
FAMCPU="${FAM/DEV=PCI+AMD/DEV=CPU}"
X="$SURF8 ALG_LOOP_STOP=3 ALG_SORT=4 SNAP_EVERY=8000"
W="ALG_TEST=.cache/wild_admitted_holdout.jsonl ALG_TEST_NAME=wildholdBP"; M="ALG_TEST=.cache/algebra_nl_test.jsonl ALG_TEST_NAME=test23BP"; S="ALG_TEST=.cache/silver460_test.jsonl ALG_TEST_NAME=silver460BP"
echo "== STAGE: arm A split + the views' source split (CPU, idempotent) ($(date +%H:%M)) =="
[ -f .cache/phase1_alg_states_realPBsrcBP_states.npy ] || bash .cache/stage_real.sh
[ -f .cache/phase1_alg_states_realBP_states.npy ] || { echo "ABORT: realBP not staged"; exit 1; }
[ -f .cache/phase1_alg_states_realPBsrcBP_states.npy ] || { echo "ABORT: realPBsrcBP not staged"; exit 1; }
echo "== STAGE: the name views' BP states (GPU precompute) + the photobooth gate ($(date +%H:%M)) =="
[ -f .cache/phase1_alg_states_realPBvBP_states.npy ] || flock -w 36000 .cache/gpu.lock env DEV=PCI+AMD .venv/bin/python3 scripts/precompute_masked.py --split realPBsrcBP --jsonl .cache/form_real_PBviews.jsonl --mask bidirpad --out realPBvBP > .cache/rn_precompute_views.log 2>&1
[ -f .cache/phase1_alg_states_realPBvBP_states.npy ] || { echo "ABORT: the views' precompute wrote no states"; exit 1; }
tail -1 .cache/rn_precompute_views.log
env $FAMCPU $SURF8 .venv/bin/python3 scripts/photobooth_stage.py realPBsrcBP .cache/form_real_PBviews.jsonl realPBvBP | tail -1
echo "== STAGE: arm B split (CPU) ($(date +%H:%M)) =="
[ -f .cache/phase1_alg_states_realPBBP.npz ] || env $FAMCPU $SURF8 .venv/bin/python3 scripts/diet_stage.py realPBBP .cache/form_mix_realPB.jsonl realBP@.cache/form_mix_real.jsonl realPBvBP@.cache/form_real_PBviews.jsonl
[ -f .cache/phase1_alg_states_realPBBP_states.npy ] || { echo "ABORT: realPBBP not staged"; exit 1; }
# THE HEAD GATE (CPU): today's head reproduces the k3 gate line bit-identically on tiny64
GC="ALG2=1 ALG_FTYPES=9 ALG_DUP=1 ALG_HW=512 ALG_WIDE=1 ALG_BREATH=7 ALG_NOTEBOOK=1 ALG_SIXWAVE=1 NB_PERSLOT=1 ALG_BINDBUS=7 ALG_BIND_D=512 BIND_CODES=.cache/bindbus_codes512.npz ALG_BUSGARAGE=2 ALG_SHELF_CIRCLE=2 ALG_ALTMASK=1 ALG_ALT21=1 ALG_ALT2=1 ALG_MASKHEAD=1 ALG_FED=1 ALG_POLAR=1 ALG_POLAR_D=128 ALG_POLAR_EM=0.1 ALG_POLAR_D_INIT=.cache/polar_waist_init_d128u.npz ALG_PRUNE=pforms,s4,fednl0,lane2 ALG_SLOT_ALL=1 ALG_STELLAR=2 ALG_CLOCK_CANON=1 SC_EVAL=0 ALG_ALLOW_PEN_TRAIN=1 ALG_TRAIN=.cache/form_tiny64.jsonl ALG_TRAIN_NAME=tiny64 ALG_TEST=.cache/test_tiny64.jsonl ALG_TEST_NAME=testtiny64 ALG_MASKPREP_CACHE=1 ALG_MASKPREP_IGNORE=ALG_MASK_COOK,ALG_MASK_COOK_SKEL,ALG_TOK_COOK,ALG_TOK_COOK_S3,ALG_TG_INIT,ALG_BAL_COOK,ALG_NB2,ALG_ALT5,ALG_PRUNE,ALG_MASKPREP_JIT,ALG_MASKPREP_B,ALG_FACTS_POOL,ALG_FACTS_FLUSH,ALG_JIT_VAL LR=1e-4 SEED=242 WARM_FROM=.cache/sharp_balV242.safetensors"
echo "== HEAD GATE (CPU, k3 on tiny64) ($(date +%H:%M)) =="
env DEV=CPU $GC $SURF8 ALG_LOOP_STOP=3 BATCH=2 STEPS=2 ALG_CKPT=.cache/sharp_rngate.safetensors .venv/bin/python3 scripts/phase1_algebra_head.py --train > .cache/rn_headgate.log 2>&1
grep -aoE 'step +[01] loss=[0-9.]+' .cache/rn_headgate.log | tr '\n' ';' | grep -aq "loss=6.6232;step     1 loss=1.1444" || { echo "ABORT: the head does not reproduce the k3 gate line (6.6232/1.1444): $(grep -aoE 'step +[01] loss=[0-9.]+' .cache/rn_headgate.log | tr '\n' ';')"; exit 1; }
echo "HEAD GATE: PASS (k3 6.6232/1.1444 bit-identical)"; rm -f .cache/sharp_rngate.safetensors
PR() { echo "paired $1 $2 vs $3: $(.venv/bin/python3 -c "import sys,runpy; sys.argv=['paired_read.py','$4','$5','$3','$2']; runpy.run_path('scripts/paired_read.py', run_name='__main__')" | tail -1)"; }
for ARM in "RLK3_241:ALG_TRAIN=.cache/form_mix_real.jsonl ALG_TRAIN_NAME=realBP" "RNK3_241:ALG_TRAIN=.cache/form_mix_realPB.jsonl ALG_TRAIN_NAME=realPBBP"; do C=${ARM%%:*}; D=${ARM#*:}
  echo "== GPU SMOKE $C (BATCH=2 STEPS=2) ($(date +%H:%M)) =="
  if ! flock -w 36000 .cache/gpu.lock env $FAM $TRC $D $X BATCH=2 STEPS=2 ALG_CKPT=.cache/sharp_rngpusmoke.safetensors .venv/bin/python3 scripts/phase1_algebra_head.py --train > .cache/rngpusmoke_$C.log 2>&1; then
    echo "ABORT: the GPU smoke crashes: $(grep -aE 'Error|Segmentation|assert' .cache/rngpusmoke_$C.log | tail -1)"; rm -f .cache/sharp_rngpusmoke.safetensors; exit 1
  fi
  echo "GPU SMOKE: PASS -- $(grep -aoE 'step +[01] loss=[0-9.]+' .cache/rngpusmoke_$C.log | tr '\n' ';')"; rm -f .cache/sharp_rngpusmoke.safetensors
  if grep -aq "sha-fence\] UNSTAMPED" .cache/rngpusmoke_$C.log; then echo "ABORT: $C's train split read UNSTAMPED"; exit 1; fi
  echo "== ARM $C ($D; random init; BP states; 48k, B=8) ($(date +%H:%M)) =="
  flock -w 36000 .cache/gpu.lock env $FAM $TRC $D $X BATCH=8 STEPS=48000 ALG_CKPT=.cache/sharp_$C.safetensors PYTHONUNBUFFERED=1 .venv/bin/python3 -u scripts/phase1_algebra_head.py --train > .cache/sharp_$C.log 2>&1 || { echo "FAILED $C: $(grep -aE 'Error|assert' .cache/sharp_$C.log | tail -1)"; exit 1; }
  grep -aE "step +(0|24000|47999) |steady" .cache/sharp_$C.log | tail -4 | cut -c1-200 || true
  for Y in "wild:$W" "mint:$M" "silver460:$S"; do F=${Y%%:*}; V=${Y#*:}
    flock -w 36000 .cache/gpu.lock env $FAM $V $X ALG_JIT_READ=1 LV_FIELDS=1 LV_DUMP=.cache/dump_open_${F}_$C.pkl LV_CKPT=.cache/sharp_$C.safetensors LV_PER_SLOT=.cache/ps_open_${F}_$C.npz .venv/bin/python3 scripts/loop_val.py > .cache/read_open_${F}_$C.log 2>&1; echo "open $F $C: $(grep -aoE 'fac-exact=[0-9.]+' .cache/read_open_${F}_$C.log | tail -1) | $(grep -a '^\[fields\]' .cache/read_open_${F}_$C.log | cut -c1-170)"
  done
  # the masked wild read IS view 0 of the name-swap vote read (same split, same env): one read, two names
  flock -w 36000 .cache/gpu.lock env $FAM $W $X ALG_JIT_READ=1 LV_LEGAL=num LV_FIELDS=1 LV_DUMP=.cache/dump_wild_$C.pkl LV_CKPT=.cache/sharp_$C.safetensors LV_PER_SLOT=.cache/ps_legal_wild_$C.npz .venv/bin/python3 scripts/loop_val.py > .cache/read_legal_wild_$C.log 2>&1; echo "masked wild $C: $(grep -aoE 'fac-exact=[0-9.]+' .cache/read_legal_wild_$C.log | tail -1) | $(grep -a '^\[fields\]' .cache/read_legal_wild_$C.log | cut -c1-200)"
  cp .cache/dump_wild_$C.pkl .cache/dump_wildPB0_$C.pkl
  for v in 1 2 3 4; do
    flock -w 36000 .cache/gpu.lock env $FAM ALG_TEST=.cache/wild_PB_v$v.jsonl ALG_TEST_NAME=wildPB$v $X ALG_JIT_READ=1 LV_LEGAL=num LV_FIELDS=1 LV_DUMP=.cache/dump_wildPB${v}_$C.pkl LV_CKPT=.cache/sharp_$C.safetensors LV_PER_SLOT=.cache/ps_legal_wildPB${v}_$C.npz .venv/bin/python3 scripts/loop_val.py > .cache/read_legal_wildPB${v}_$C.log 2>&1 || { echo "FAILED read view $v $C"; exit 1; }
    echo "masked wild view $v $C: $(grep -aoE 'fac-exact=[0-9.]+' .cache/read_legal_wildPB${v}_$C.log | tail -1)"
  done
  .venv/bin/python3 scripts/picker/photobooth_vote_read.py $C | tee .cache/photobooth_vote_read_$C.txt | tail -12
  PR "masked wild (MINT ATTRIBUTION)" $C SRK3_241 .cache/ps_legal_wild_SRK3_241.npz .cache/ps_legal_wild_$C.npz
  PR "open mint (mint leaves the diet)" $C SRK3_241 .cache/ps_open_mint_SRK3_241.npz .cache/ps_open_mint_$C.npz
  if [ "$C" = RNK3_241 ]; then PR "masked wild (THE AUGMENTATION BAR)" RNK3_241 RLK3_241 .cache/ps_legal_wild_RLK3_241.npz .cache/ps_legal_wild_RNK3_241.npz; fi
done
echo "== WHOLE PROBLEM (the scoreboard law) ($(date +%H:%M)) =="
SBC_OUT=.cache/step_blame_census_RN.txt .venv/bin/python3 scripts/picker/step_blame_census.py SRK3_241 RLK3_241 RNK3_241 > .cache/step_blame_census_RN.log 2>&1; grep -aE "^== |STEP CREDIT|GOLD CEIL" .cache/step_blame_census_RN.log
.venv/bin/python3 scripts/reads.py ingest | tail -1 || true; echo "RN CHAIN COMPLETE ($(date +%H:%M))"
