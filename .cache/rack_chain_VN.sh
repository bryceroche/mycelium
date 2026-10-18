#!/usr/bin/bash
# VIEWS, NOT REPS (2026-10-18; Bryce's cross-join ruling: train on the unique rows, varied by the photobooth, not on copies;
# word given for the queue). SRK3/RLK3's recipe exactly (K3 + BP states + the sorting room, ORIGINAL positional frame, 48k,
# B=8); ONE change vs RLK3: form_mix_realVN = form_mix_real with every EXACT REPEAT (8,094 rows: a text's 2nd..kth
# occurrence) replaced by photobooth view v=o of that row (PB_WIDE=1, the audited photobooth: listed names only, hyphen-aware;
# a row without a swappable name keeps its copy). DOSE (both ways): rows 30,533 = form_mix_real; reps 12.6 = form_mix_real.
# ARMS: RLK3_242 (the baseline's seed twin) | VNK3_241 | VNK3_242. Baseline RLK3_241 is banked (read at s48000).
# READS at the s48000 snapshot (THE CHECKPOINT FENCE, 10-17: these diets exclude the val split's format, so best-by-val is
# not lawful; the final step is read). PR prints ARM - BASELINE and ABORTS on a crash (the 10-18 audit).
# BARS: pinned in the ledger 2026-10-18 'VIEWS, NOT REPS', before firing.
set -eo pipefail; cd /home/bryce/mycelium; exec > .cache/rack_chain_VN.log 2>&1
.venv/bin/python3 scripts/contracts/preflight.py "$0" || exit 1   # THE PRE-FLIGHT CONTRACT
FAM="DEV=PCI+AMD ALG2=1 ALG_FTYPES=9 ALG_DUP=1 ALG_HW=512 ALG_WIDE=1 ALG_BREATH=7 ALG_NOTEBOOK=1 ALG_SIXWAVE=1 NB_PERSLOT=1 ALG_BINDBUS=7 ALG_BIND_D=512 BIND_CODES=.cache/bindbus_codes512.npz ALG_BUSGARAGE=2 ALG_SHELF_CIRCLE=2 ALG_ALTMASK=1 ALG_ALT21=1 ALG_ALT2=1 ALG_MASKHEAD=1 ALG_FED=1 ALG_POLAR=1 ALG_POLAR_D=128 ALG_POLAR_EM=0.1 ALG_POLAR_D_INIT=.cache/polar_waist_init_d128u.npz ALG_PRUNE=pforms,s4,fednl0,lane2 ALG_SLOT_ALL=1 ALG_STELLAR=2 ALG_CLOCK_CANON=1 SC_EVAL=0"
TRC="ALG_ALLOW_PEN_TRAIN=1 ALG_TEST=.cache/algebra_nl_test.jsonl ALG_TEST_NAME=test23BP ALG_MASKPREP_CACHE=1 ALG_MASKPREP_JIT=1 ALG_FACTS_POOL=1 ALG_FACTS_WORKERS=6 ALG_MASKPREP_IGNORE=ALG_MASK_COOK,ALG_MASK_COOK_SKEL,ALG_TOK_COOK,ALG_TOK_COOK_S3,ALG_TG_INIT,ALG_BAL_COOK,ALG_NB2,ALG_ALT5,ALG_PRUNE,ALG_MASKPREP_JIT,ALG_MASKPREP_B,ALG_FACTS_POOL,ALG_FACTS_FLUSH,ALG_JIT_VAL LR=1e-4 SNAP_EVERY=100000"
SURF8="ALG_ROUTER=2 R_GAIN_INIT=1.0 ALG_FREEZE=r_gain ALG_ROUTER_PTR=0.0 ALG_SPAN_ALL=1 ALG_SPAN_ARGS=1 ALG_SPAN_OP=1 ALG_SPAN_RCUE=1 ALG_SPAN_ARCUE=1 ALG_PTR_SURF=role:add:2.0 BIND_CODES=.cache/bindbus_codes512r.npz"
FAMCPU="${FAM/DEV=PCI+AMD/DEV=CPU}"
X="$SURF8 ALG_LOOP_STOP=3 ALG_SORT=4 SNAP_EVERY=8000"
W="ALG_TEST=.cache/wild_admitted_holdout.jsonl ALG_TEST_NAME=wildholdBP"
PR() { local o; o=$(.venv/bin/python3 scripts/paired_read.py "$4" "$5" "$2" "$3") || { echo "ABORT: paired read '$1' crashed"; exit 1; }
       echo "paired $1 ($2 - $3): $(echo "$o" | tail -1)"; }
echo "== STAGE: the views' source split (CPU) ($(date +%H:%M)) =="
[ -f .cache/phase1_alg_states_realVNsrcBP_states.npy ] || env $FAMCPU $SURF8 .venv/bin/python3 scripts/diet_stage.py realVNsrcBP .cache/form_real_VNsrc.jsonl realBP@.cache/form_mix_real.jsonl@.cache/form_real_VNsrc.idx.npy
[ -f .cache/phase1_alg_states_realVNsrcBP_states.npy ] || { echo "ABORT: realVNsrcBP not staged"; exit 1; }
echo "== STAGE: the views' BP states (GPU precompute) + the photobooth gate ($(date +%H:%M)) =="
[ -f .cache/phase1_alg_states_realVNvBP_states.npy ] || flock -w 36000 .cache/gpu.lock env DEV=PCI+AMD .venv/bin/python3 scripts/precompute_masked.py --split realVNsrcBP --jsonl .cache/form_real_VNviews.jsonl --mask bidirpad --out realVNvBP > .cache/vn_precompute_views.log 2>&1
[ -f .cache/phase1_alg_states_realVNvBP_states.npy ] || { echo "ABORT: the views' precompute wrote no states"; exit 1; }
tail -1 .cache/vn_precompute_views.log
env $FAMCPU $SURF8 .venv/bin/python3 scripts/photobooth_stage.py realVNsrcBP .cache/form_real_VNviews.jsonl realVNvBP | tail -1
echo "== STAGE: the views diet (CPU) ($(date +%H:%M)) =="
[ -f .cache/phase1_alg_states_realVNBP.npz ] || env $FAMCPU $SURF8 .venv/bin/python3 scripts/diet_stage.py realVNBP .cache/form_mix_realVN.jsonl realBP@.cache/form_mix_real.jsonl@.cache/form_real_VNkeep.idx.npy realVNvBP@.cache/form_real_VNviews.jsonl
[ -f .cache/phase1_alg_states_realVNBP_states.npy ] || { echo "ABORT: realVNBP not staged"; exit 1; }
# THE HEAD GATE (CPU): today's head reproduces the k3 gate line bit-identically on tiny64
GC="ALG2=1 ALG_FTYPES=9 ALG_DUP=1 ALG_HW=512 ALG_WIDE=1 ALG_BREATH=7 ALG_NOTEBOOK=1 ALG_SIXWAVE=1 NB_PERSLOT=1 ALG_BINDBUS=7 ALG_BIND_D=512 BIND_CODES=.cache/bindbus_codes512.npz ALG_BUSGARAGE=2 ALG_SHELF_CIRCLE=2 ALG_ALTMASK=1 ALG_ALT21=1 ALG_ALT2=1 ALG_MASKHEAD=1 ALG_FED=1 ALG_POLAR=1 ALG_POLAR_D=128 ALG_POLAR_EM=0.1 ALG_POLAR_D_INIT=.cache/polar_waist_init_d128u.npz ALG_PRUNE=pforms,s4,fednl0,lane2 ALG_SLOT_ALL=1 ALG_STELLAR=2 ALG_CLOCK_CANON=1 SC_EVAL=0 ALG_ALLOW_PEN_TRAIN=1 ALG_TRAIN=.cache/form_tiny64.jsonl ALG_TRAIN_NAME=tiny64 ALG_TEST=.cache/test_tiny64.jsonl ALG_TEST_NAME=testtiny64 ALG_MASKPREP_CACHE=1 ALG_MASKPREP_IGNORE=ALG_MASK_COOK,ALG_MASK_COOK_SKEL,ALG_TOK_COOK,ALG_TOK_COOK_S3,ALG_TG_INIT,ALG_BAL_COOK,ALG_NB2,ALG_ALT5,ALG_PRUNE,ALG_MASKPREP_JIT,ALG_MASKPREP_B,ALG_FACTS_POOL,ALG_FACTS_FLUSH,ALG_JIT_VAL LR=1e-4 SEED=242 WARM_FROM=.cache/sharp_balV242.safetensors"
echo "== HEAD GATE (CPU, k3 on tiny64) ($(date +%H:%M)) =="
env DEV=CPU $GC $SURF8 ALG_LOOP_STOP=3 BATCH=2 STEPS=2 ALG_CKPT=.cache/sharp_vngate.safetensors .venv/bin/python3 scripts/phase1_algebra_head.py --train > .cache/vn_headgate.log 2>&1
grep -aoE 'step +[01] loss=[0-9.]+' .cache/vn_headgate.log | tr '\n' ';' | grep -aq "loss=6.6232;step     1 loss=1.1444" || { echo "ABORT: the head does not reproduce the k3 gate line (6.6232/1.1444): $(grep -aoE 'step +[01] loss=[0-9.]+' .cache/vn_headgate.log | tr '\n' ';')"; exit 1; }
echo "HEAD GATE: PASS (k3 6.6232/1.1444 bit-identical)"; rm -f .cache/sharp_vngate.safetensors
for ARM in "RLK3_242:SEED=242 ALG_TRAIN=.cache/form_mix_real.jsonl ALG_TRAIN_NAME=realBP" "VNK3_241:SEED=241 ALG_TRAIN=.cache/form_mix_realVN.jsonl ALG_TRAIN_NAME=realVNBP" "VNK3_242:SEED=242 ALG_TRAIN=.cache/form_mix_realVN.jsonl ALG_TRAIN_NAME=realVNBP"; do C=${ARM%%:*}; D=${ARM#*:}
  echo "== GPU SMOKE $C (BATCH=2 STEPS=2) ($(date +%H:%M)) =="
  if ! flock -w 36000 .cache/gpu.lock env $FAM $TRC $D $X BATCH=2 STEPS=2 ALG_CKPT=.cache/sharp_vngpusmoke.safetensors .venv/bin/python3 scripts/phase1_algebra_head.py --train > .cache/vngpusmoke_$C.log 2>&1; then
    echo "ABORT: the GPU smoke crashes: $(grep -aE 'Error|Segmentation|assert' .cache/vngpusmoke_$C.log | tail -1)"; rm -f .cache/sharp_vngpusmoke*.safetensors; exit 1
  fi
  echo "GPU SMOKE: PASS -- $(grep -aoE 'step +[01] loss=[0-9.]+' .cache/vngpusmoke_$C.log | tr '\n' ';')"; rm -f .cache/sharp_vngpusmoke*.safetensors
  if grep -aq "sha-fence\] UNSTAMPED" .cache/vngpusmoke_$C.log; then echo "ABORT: $C's train split read UNSTAMPED"; exit 1; fi
  echo "== ARM $C ($D; random init; BP states; 48k, B=8) ($(date +%H:%M)) =="
  flock -w 36000 .cache/gpu.lock env $FAM $TRC $D $X BATCH=8 STEPS=48000 ALG_CKPT=.cache/sharp_$C.safetensors PYTHONUNBUFFERED=1 .venv/bin/python3 -u scripts/phase1_algebra_head.py --train > .cache/sharp_$C.log 2>&1 || { echo "FAILED $C: $(grep -aE 'Error|assert' .cache/sharp_$C.log | tail -1)"; exit 1; }
  grep -aE "step +(0|24000|47999) |steady" .cache/sharp_$C.log | tail -4 | cut -c1-200 || true
  K=.cache/sharp_${C}_s48000.safetensors; [ -f $K ] || { echo "ABORT: no s48000 snapshot for $C"; exit 1; }
  T=${C}s48
  flock -w 36000 .cache/gpu.lock env $FAM $W $X ALG_JIT_READ=1 LV_FIELDS=1 LV_DUMP=.cache/dump_open_wild_$T.pkl LV_CKPT=$K LV_PER_SLOT=.cache/ps_open_wild_$T.npz .venv/bin/python3 scripts/loop_val.py > .cache/read_open_wild_$T.log 2>&1 || { echo "FAILED open read $T"; exit 1; }
  echo "open wild $T: $(grep -aoE 'fac-exact=[0-9.]+' .cache/read_open_wild_$T.log | tail -1) | $(grep -a '^\[fields\]' .cache/read_open_wild_$T.log | cut -c1-170)"
  flock -w 36000 .cache/gpu.lock env $FAM $W $X ALG_JIT_READ=1 LV_LEGAL=num LV_FIELDS=1 LV_DUMP=.cache/dump_wild_$T.pkl LV_CKPT=$K LV_PER_SLOT=.cache/ps_legal_wild_$T.npz .venv/bin/python3 scripts/loop_val.py > .cache/read_legal_wild_$T.log 2>&1 || { echo "FAILED masked read $T"; exit 1; }
  echo "masked wild $T: $(grep -aoE 'fac-exact=[0-9.]+' .cache/read_legal_wild_$T.log | tail -1) | $(grep -a '^\[fields\]' .cache/read_legal_wild_$T.log | cut -c1-200)"
  cp .cache/dump_wild_$T.pkl .cache/dump_wildPB0_$T.pkl
  for v in 1 2 3 4; do
    flock -w 36000 .cache/gpu.lock env $FAM ALG_TEST=.cache/wild_PB_v$v.jsonl ALG_TEST_NAME=wildPB$v $X ALG_JIT_READ=1 LV_LEGAL=num LV_FIELDS=1 LV_DUMP=.cache/dump_wildPB${v}_$T.pkl LV_CKPT=$K LV_PER_SLOT=.cache/ps_legal_wildPB${v}_$T.npz .venv/bin/python3 scripts/loop_val.py > .cache/read_legal_wildPB${v}_$T.log 2>&1 || { echo "FAILED read view $v $T"; exit 1; }
    echo "masked wild view $v $T: $(grep -aoE 'fac-exact=[0-9.]+' .cache/read_legal_wildPB${v}_$T.log | tail -1)"
  done
  .venv/bin/python3 scripts/picker/photobooth_vote_read.py $T | tee .cache/photobooth_vote_read_$T.txt | tail -8
done
echo "== THE BARS ($(date +%H:%M)) =="
PR "masked wild, seed 241 (VIEWS vs COPIES)" VNK3_241s48 RLK3_241s48 .cache/ps_legal_wild_VNK3_241s48.npz .cache/ps_legal_wild_RLK3_241s48.npz
PR "masked wild, seed 242 (VIEWS vs COPIES)" VNK3_242s48 RLK3_242s48 .cache/ps_legal_wild_VNK3_242s48.npz .cache/ps_legal_wild_RLK3_242s48.npz
PR "masked wild, the baseline's seed spread (no bar)" RLK3_242s48 RLK3_241s48 .cache/ps_legal_wild_RLK3_242s48.npz .cache/ps_legal_wild_RLK3_241s48.npz
for T in RLK3_241s48 RLK3_242s48 VNK3_241s48 VNK3_242s48; do echo "invariance $T: $(grep -aiE 'invarian' .cache/photobooth_vote_read_$T.txt | head -2 | tr '\n' ' ')"; done
echo "== WHOLE PROBLEM (the scoreboard law) ($(date +%H:%M)) =="
SBC_OUT=.cache/step_blame_census_VN.txt .venv/bin/python3 scripts/picker/step_blame_census.py RLK3_241s48 RLK3_242s48 VNK3_241s48 VNK3_242s48 > .cache/step_blame_census_VN.log 2>&1; grep -aE "^== |STEP CREDIT|GOLD CEIL" .cache/step_blame_census_VN.log
echo "VN CHAIN COMPLETE ($(date +%H:%M))"
