#!/usr/bin/bash
# THE READING-ORDER RUN (2026-10-16; Bryce: "word given brother -- Salute!"). ONE ARM: ROK3_241 = SRK3_241's recipe exactly
# (K3 + BP states + the sorting room, 48k, B=8, seed 241) on the READING-ORDER gold (scripts/reading_order.py: givens in
# text-mention order as a reading block, then the planning block; lettered mint rows untouched). States are SHARED with
# the BP splits (text-only; symlinks); gold restaged by scripts/reading_order_restage.py (gate: the banked BP gold
# reproduces bit-identically from the original jsonl before the RO gold is built).
# THE FRAME CHANGES: the positional read on wildholdRO is a NEW frame (not comparable to SRK3's). Cross-frame reads are
# the MATCHED read and the reading/planning read (both order-invariant); mint (lettered, unchanged) pairs directly.
# KNOWN CONFOUND (declared): the diet now carries two layouts (prose: reading block then planning block; mint: span-sorted
# interleaved), separable by surface (a declared letter list).
# BARS (pinned in the ledger 2026-10-16, 'THE READING ORDER, STEPS 1-3', before measurement), vs SRK3_241:
#   READING per-given (reading_planning_read.py) >= SRK3's 0.776 + 0.03 = 0.806;
#   THE ORDER COIN-FLIP GONE: the machine's k-th given = the text-order value >= 0.80 on the reordered wild rows;
#   NO CONTENT LOSS: matched wild fac-exact >= 0.4569 - 0.010 = 0.4469; KILL: matched wild < 0.4469.
#   (not a bar, watched: open mint paired vs SRK3_241; planning on relation slots, NOT expected to move.)
set -eo pipefail; cd /home/bryce/mycelium; exec > .cache/rack_chain_ROQ.log 2>&1
.venv/bin/python3 scripts/contracts/preflight.py "$0" || exit 1   # THE PRE-FLIGHT CONTRACT
FAM="DEV=PCI+AMD ALG2=1 ALG_FTYPES=9 ALG_DUP=1 ALG_HW=512 ALG_WIDE=1 ALG_BREATH=7 ALG_NOTEBOOK=1 ALG_SIXWAVE=1 NB_PERSLOT=1 ALG_BINDBUS=7 ALG_BIND_D=512 BIND_CODES=.cache/bindbus_codes512.npz ALG_BUSGARAGE=2 ALG_SHELF_CIRCLE=2 ALG_ALTMASK=1 ALG_ALT21=1 ALG_ALT2=1 ALG_MASKHEAD=1 ALG_FED=1 ALG_POLAR=1 ALG_POLAR_D=128 ALG_POLAR_EM=0.1 ALG_POLAR_D_INIT=.cache/polar_waist_init_d128u.npz ALG_PRUNE=pforms,s4,fednl0,lane2 ALG_SLOT_ALL=1 ALG_STELLAR=2 ALG_CLOCK_CANON=1 SC_EVAL=0"
TR="ALG_ALLOW_PEN_TRAIN=1 ALG_TRAIN=.cache/form_mix_pm35c_RO.jsonl ALG_TRAIN_NAME=formpm35cRO ALG_TEST=.cache/algebra_nl_test.jsonl ALG_TEST_NAME=test23BP ALG_MASKPREP_CACHE=1 ALG_MASKPREP_JIT=1 ALG_FACTS_POOL=1 ALG_FACTS_WORKERS=6 ALG_MASKPREP_IGNORE=ALG_MASK_COOK,ALG_MASK_COOK_SKEL,ALG_TOK_COOK,ALG_TOK_COOK_S3,ALG_TG_INIT,ALG_BAL_COOK,ALG_NB2,ALG_ALT5,ALG_PRUNE,ALG_MASKPREP_JIT,ALG_MASKPREP_B,ALG_FACTS_POOL,ALG_FACTS_FLUSH,ALG_JIT_VAL LR=1e-4 SEED=241 SNAP_EVERY=100000"
SURF8="ALG_ROUTER=2 R_GAIN_INIT=1.0 ALG_FREEZE=r_gain ALG_ROUTER_PTR=0.0 ALG_SPAN_ALL=1 ALG_SPAN_ARGS=1 ALG_SPAN_OP=1 ALG_SPAN_RCUE=1 ALG_SPAN_ARCUE=1 ALG_PTR_SURF=role:add:2.0 BIND_CODES=.cache/bindbus_codes512r.npz"
LSX="ALG_LOOP_STOP=3"
W="ALG_TEST=.cache/wild_admitted_holdout_RO.jsonl ALG_TEST_NAME=wildholdRO"; M="ALG_TEST=.cache/algebra_nl_test.jsonl ALG_TEST_NAME=test23BP"; S="ALG_TEST=.cache/silver460_test_RO.jsonl ALG_TEST_NAME=silver460RO"
# THE HEAD GATE (CPU): today's head (the loader's 'reading' line landed) reproduces the k3 gate line bit-identically on tiny64
GC="ALG2=1 ALG_FTYPES=9 ALG_DUP=1 ALG_HW=512 ALG_WIDE=1 ALG_BREATH=7 ALG_NOTEBOOK=1 ALG_SIXWAVE=1 NB_PERSLOT=1 ALG_BINDBUS=7 ALG_BIND_D=512 BIND_CODES=.cache/bindbus_codes512.npz ALG_BUSGARAGE=2 ALG_SHELF_CIRCLE=2 ALG_ALTMASK=1 ALG_ALT21=1 ALG_ALT2=1 ALG_MASKHEAD=1 ALG_FED=1 ALG_POLAR=1 ALG_POLAR_D=128 ALG_POLAR_EM=0.1 ALG_POLAR_D_INIT=.cache/polar_waist_init_d128u.npz ALG_PRUNE=pforms,s4,fednl0,lane2 ALG_SLOT_ALL=1 ALG_STELLAR=2 ALG_CLOCK_CANON=1 SC_EVAL=0 ALG_ALLOW_PEN_TRAIN=1 ALG_TRAIN=.cache/form_tiny64.jsonl ALG_TRAIN_NAME=tiny64 ALG_TEST=.cache/test_tiny64.jsonl ALG_TEST_NAME=testtiny64 ALG_MASKPREP_CACHE=1 ALG_MASKPREP_IGNORE=ALG_MASK_COOK,ALG_MASK_COOK_SKEL,ALG_TOK_COOK,ALG_TOK_COOK_S3,ALG_TG_INIT,ALG_BAL_COOK,ALG_NB2,ALG_ALT5,ALG_PRUNE,ALG_MASKPREP_JIT,ALG_MASKPREP_B,ALG_FACTS_POOL,ALG_FACTS_FLUSH,ALG_JIT_VAL LR=1e-4 SEED=242 WARM_FROM=.cache/sharp_balV242.safetensors"
echo "== HEAD GATE (CPU, k3 on tiny64) ($(date +%H:%M)) =="
env DEV=CPU $GC $SURF8 ALG_LOOP_STOP=3 BATCH=2 STEPS=2 ALG_CKPT=.cache/sharp_roqgate.safetensors .venv/bin/python3 scripts/phase1_algebra_head.py --train > .cache/roq_headgate.log 2>&1
grep -aoE 'step +[01] loss=[0-9.]+' .cache/roq_headgate.log | tr '\n' ';' | grep -aq "loss=6.6232;step     1 loss=1.1444" || { echo "ABORT: the head does not reproduce the k3 gate line (6.6232/1.1444): $(grep -aoE 'step +[01] loss=[0-9.]+' .cache/roq_headgate.log | tr '\n' ';')"; exit 1; }
echo "HEAD GATE: PASS (k3 6.6232/1.1444 bit-identical)"; rm -f .cache/sharp_roqgate.safetensors
echo "== PRE-ARM GPU SMOKE (RO gold, the arm's config, BATCH=2 STEPS=2) ($(date +%H:%M)) =="
if ! flock -w 36000 .cache/gpu.lock env $FAM $TR $SURF8 $LSX ALG_SORT=4 SNAP_EVERY=4000 BATCH=2 STEPS=2 ALG_CKPT=.cache/sharp_roqgpusmoke.safetensors .venv/bin/python3 scripts/phase1_algebra_head.py --train > .cache/roqgpusmoke.log 2>&1; then
  echo "ABORT: the GPU smoke crashes: $(grep -aE 'Error|Segmentation|assert' .cache/roqgpusmoke.log | tail -1)"; rm -f .cache/sharp_roqgpusmoke.safetensors; exit 1
fi
echo "GPU SMOKE: PASS -- $(grep -aoE 'step +[01] loss=[0-9.]+' .cache/roqgpusmoke.log | tr '\n' ';')"; rm -f .cache/sharp_roqgpusmoke.safetensors
grep -aq "sha-fence\] UNSTAMPED" .cache/roqgpusmoke.log && { echo "ABORT: the RO train split read UNSTAMPED"; exit 1; } || true
C=ROK3_241; X="$SURF8 $LSX ALG_SORT=4 SNAP_EVERY=8000"
echo "== ARM $C ($X; random init; form_mix_pm35c_RO on BP states; 48k, B=8) ($(date +%H:%M)) =="
flock -w 36000 .cache/gpu.lock env $FAM $TR $X BATCH=8 STEPS=48000 ALG_CKPT=.cache/sharp_$C.safetensors PYTHONUNBUFFERED=1 .venv/bin/python3 -u scripts/phase1_algebra_head.py --train > .cache/sharp_$C.log 2>&1 || { echo "FAILED $C: $(grep -aE 'Error|assert' .cache/sharp_$C.log | tail -1)"; exit 1; }
grep -aE "step +(0|24000|47999) |steady" .cache/sharp_$C.log | tail -4 | cut -c1-200 || true
for Y in "wild:$W" "mint:$M" "silver460:$S"; do F=${Y%%:*}; V=${Y#*:}
  flock -w 36000 .cache/gpu.lock env $FAM $V $X ALG_JIT_READ=1 LV_FIELDS=1 LV_DUMP=.cache/dump_${F}_$C.pkl LV_CKPT=.cache/sharp_$C.safetensors LV_PER_SLOT=.cache/ps_open_${F}_$C.npz .venv/bin/python3 scripts/loop_val.py > .cache/read_open_${F}_$C.log 2>&1; echo "open $F $C (RO frame for wild/silver): $(grep -aoE 'fac-exact=[0-9.]+' .cache/read_open_${F}_$C.log | tail -1) | $(grep -a '^\[fields\]' .cache/read_open_${F}_$C.log | cut -c1-170)"
done
flock -w 36000 .cache/gpu.lock env $FAM $W $X ALG_JIT_READ=1 LV_LEGAL=num LV_FIELDS=1 LV_DUMP=.cache/dump_wild_$C.pkl LV_CKPT=.cache/sharp_$C.safetensors LV_PER_SLOT=.cache/ps_legal_wild_$C.npz .venv/bin/python3 scripts/loop_val.py > .cache/read_legal_wild_$C.log 2>&1; echo "masked wild $C (RO frame): $(grep -aoE 'fac-exact=[0-9.]+' .cache/read_legal_wild_$C.log | tail -1) | $(grep -a '^\[fields\]' .cache/read_legal_wild_$C.log | cut -c1-200)"
echo "== THE BARS' READS ($(date +%H:%M)) =="
.venv/bin/python3 scripts/matched_read.py .cache/dump_wild_$C.pkl .cache/dump_wild_SRK3_241.pkl | tail -2
.venv/bin/python3 scripts/picker/reading_planning_read.py $C SRK3_241 | tee .cache/reading_planning_read_$C.txt
echo "paired open mint $C vs SRK3_241 (same frame): $(.venv/bin/python3 -c "import sys,runpy; sys.argv=['paired_read.py','.cache/ps_open_mint_SRK3_241.npz','.cache/ps_open_mint_$C.npz','SRK3_241','$C']; runpy.run_path('scripts/paired_read.py', run_name='__main__')" | tail -1)"
.venv/bin/python3 scripts/reads.py ingest | tail -1 || true; echo "ROQ CHAIN COMPLETE ($(date +%H:%M))"
