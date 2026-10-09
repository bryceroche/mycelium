#!/usr/bin/bash
# THE DIRECTION BIT, plain chassis (2026-10-08): DIR_241 = PMS8_241's exact recipe (SURF8, no
# hierarchical state, form_mix_pm35c, random init, 48k, B=8, SEED=241) + ALG_DIR=1 -- the direction
# bit's own BCE term + its res-pointer structural mask (soft-by-p at train, hard -1e4 at read).
# WRITTEN, NOT FIRED (dirbit worktree task; gen-weights may be imported by a running chain).
# BARS (ledger 2026-10-08 "THE POLARITY CENSUS", registered forms (a)/(b)): paired masked wild
# DIR_241 vs PMS8_241 >= +0.015 (a CLAIM needs +0.020 or the twin); THE MECHANISM (polarity_census.py
# re-run against DIR_241's own dump): inverse-form res accuracy >= 0.60 (from 0.25), forward res
# NOT DOWN. digits/rows for information.
BODY=PMS8_241
set -eo pipefail; cd /home/bryce/mycelium; exec > .cache/rack_chain_DIR.log 2>&1
.venv/bin/python3 scripts/contracts/preflight.py "$0" || exit 1   # THE PRE-FLIGHT CONTRACT
FAM="DEV=PCI+AMD ALG2=1 ALG_FTYPES=9 ALG_DUP=1 ALG_HW=512 ALG_WIDE=1 ALG_BREATH=7 ALG_NOTEBOOK=1 ALG_SIXWAVE=1 NB_PERSLOT=1 ALG_BINDBUS=7 ALG_BIND_D=512 BIND_CODES=.cache/bindbus_codes512.npz ALG_BUSGARAGE=2 ALG_SHELF_CIRCLE=2 ALG_ALTMASK=1 ALG_ALT21=1 ALG_ALT2=1 ALG_MASKHEAD=1 ALG_FED=1 ALG_POLAR=1 ALG_POLAR_D=128 ALG_POLAR_EM=0.1 ALG_POLAR_D_INIT=.cache/polar_waist_init_d128u.npz ALG_PRUNE=pforms,s4,fednl0,lane2 ALG_SLOT_ALL=1 ALG_STELLAR=2 ALG_CLOCK_CANON=1 SC_EVAL=0"
TR="ALG_ALLOW_PEN_TRAIN=1 ALG_TRAIN=.cache/form_mix_pm35c.jsonl ALG_TRAIN_NAME=formpm35c ALG_TEST=.cache/algebra_nl_test.jsonl ALG_TEST_NAME=test23 ALG_MASKPREP_CACHE=1 ALG_MASKPREP_JIT=1 ALG_FACTS_POOL=1 ALG_FACTS_WORKERS=6 ALG_MASKPREP_IGNORE=ALG_MASK_COOK,ALG_MASK_COOK_SKEL,ALG_TOK_COOK,ALG_TOK_COOK_S3,ALG_TG_INIT,ALG_BAL_COOK,ALG_NB2,ALG_ALT5,ALG_PRUNE,ALG_MASKPREP_JIT,ALG_MASKPREP_B,ALG_FACTS_POOL,ALG_FACTS_FLUSH,ALG_JIT_VAL LR=1e-4 SEED=241 SNAP_EVERY=100000"
SURF8="${SURF8:-ALG_ROUTER=2 R_GAIN_INIT=1.0 ALG_FREEZE=r_gain ALG_ROUTER_PTR=0.0 ALG_SPAN_ALL=1 ALG_SPAN_ARGS=1 ALG_SPAN_OP=1 ALG_SPAN_RCUE=1 ALG_SPAN_ARCUE=1 ALG_PTR_SURF=role:add:2.0 BIND_CODES=.cache/bindbus_codes512r.npz}"
DIRX="ALG_DIR=1"
W="ALG_TEST=.cache/wild_admitted_holdout.jsonl ALG_TEST_NAME=wildhold"; M="ALG_TEST=.cache/algebra_nl_test.jsonl ALG_TEST_NAME=test23"; S="ALG_TEST=.cache/silver460_test.jsonl ALG_TEST_NAME=silver460"
# gate check: self-heal the post-merge gate (.cache/dirbit_merge_gate.log) if missing -- the
# rack_chain_RK3X.sh precedent, pointed at the main tree directly (this chain itself runs there,
# post-merge; dirbit_gate.sh is the wt9-only pre-merge CPU gate this one's config strings descend
# from).
if [ ! -f .cache/dirbit_merge_gate.log ]; then
  echo "[self-heal] .cache/dirbit_merge_gate.log missing -- re-running the dirbit gate against the main tree"
  sed 's#cd /home/bryce/mycelium-wt9#cd /home/bryce/mycelium#; s#PYTHONPATH=/home/bryce/mycelium-wt9#PYTHONPATH=/home/bryce/mycelium#g; s#dirbit_gate\.log#dirbit_merge_gate.log#; s#dirbit_gate_#dirbit_merge_gate_#g' .cache/dirbit_gate.sh > .cache/dirbit_merge_gate.sh
  bash .cache/dirbit_merge_gate.sh || { echo "ABORT: the self-healed post-merge gate failed to run"; exit 1; }
fi
grep -aq "gate unset: step     0 loss=5.2995" .cache/dirbit_merge_gate.log || { echo "ABORT: the post-merge gate did not certify the head bit-identical unset"; exit 1; }
grep -a "gate dir:" .cache/dirbit_merge_gate.log | tail -1 | grep -aq "loss=[0-9]" || { echo "ABORT: the dir config does not run on the fixture"; exit 1; }
for ARM in "DIR_241:$SURF8 $DIRX SNAP_EVERY=8000"; do C=${ARM%%:*}; X=${ARM#*:}
  echo "== ARM $C ($X; random init; form_mix_pm35c; 48k, B=8) ($(date +%H:%M)) =="
  flock -w 36000 .cache/gpu.lock env $FAM $TR $X BATCH=8 STEPS=48000 ALG_CKPT=.cache/sharp_$C.safetensors PYTHONUNBUFFERED=1 .venv/bin/python3 -u scripts/phase1_algebra_head.py --train > .cache/sharp_$C.log 2>&1 || { echo "FAILED $C: $(grep -aE 'Error|assert' .cache/sharp_$C.log | tail -1)"; exit 1; }
  grep -aE "router|xcorr\]|anchor|span-all|step +(0|24000|47999) " .cache/sharp_$C.log | tail -8 | cut -c1-200 || true
  for Y in "wild:$W" "mint:$M" "silver460:$S"; do F=${Y%%:*}; V=${Y#*:}
    flock -w 36000 .cache/gpu.lock env $FAM $V $X ALG_JIT_READ=1 LV_FIELDS=1 LV_DUMP=.cache/dump_${F}_$C.pkl LV_CKPT=.cache/sharp_$C.safetensors LV_PER_SLOT=.cache/ps_open_${F}_$C.npz .venv/bin/python3 scripts/loop_val.py > .cache/read_open_${F}_$C.log 2>&1; echo "open $F $C: $(grep -aoE 'fac-exact=[0-9.]+' .cache/read_open_${F}_$C.log | tail -1) | $(grep -a '^\[fields\]' .cache/read_open_${F}_$C.log | cut -c1-170)"
  done
  flock -w 36000 .cache/gpu.lock env $FAM $W $X ALG_JIT_READ=1 LV_LEGAL=num LV_FIELDS=1 LV_DUMP=.cache/dump_wild_$C.pkl LV_CKPT=.cache/sharp_$C.safetensors LV_PER_SLOT=.cache/ps_legal_wild_$C.npz .venv/bin/python3 scripts/loop_val.py > .cache/read_legal_wild_$C.log 2>&1; echo "masked wild $C: $(grep -aoE 'fac-exact=[0-9.]+' .cache/read_legal_wild_$C.log | tail -1) | $(grep -a '^\[fields\]' .cache/read_legal_wild_$C.log | cut -c1-200)"
  echo "matched $C: $(.venv/bin/python3 scripts/matched_read.py .cache/dump_wild_$C.pkl | cut -c1-200)"
  for MK in 0 1; do flock -w 36000 .cache/gpu.lock env $FAM $W $X CA_CKPT=.cache/sharp_$C.safetensors CA_MASK=$MK .venv/bin/python3 scripts/chain_acc.py 2>&1 | grep -a "chain-acc\]" | tail -1 || true; done
  echo "paired masked wild $C vs $BODY: $(.venv/bin/python3 scripts/paired_read.py .cache/ps_legal_wild_$BODY.npz .cache/ps_legal_wild_$C.npz $BODY $C | tail -1)"
  # THE MECHANISM READ: polarity_census.py's own forward/inverse classification + per-head
  # accuracy join, re-pointed at this arm's own dump (its BODIES/MAIN/OUT are module globals,
  # not a CLI surface -- overridden in-process rather than editing the campaign artifact).
  .venv/bin/python3 -c "
import sys; sys.argv=['polarity_census']
sys.path.insert(0, 'scripts')
import polarity_census as PC
PC.BODIES = ['$C']; PC.MAIN = '$C'; PC.OUT = '.cache/polarity_census_$C.txt'
PC.main()
" > .cache/polarity_census_$C.log 2>&1 || echo "polarity-census FAILED $C: $(tail -1 .cache/polarity_census_$C.log)"
  grep -aE "INVERSE FORMS FAIL|res .* vs|^  2\. THE HEAD" -A2 .cache/polarity_census_$C.txt 2>/dev/null | head -12 | sed "s/^/MECH $C /" || true
done
.venv/bin/python3 scripts/reads.py ingest | tail -1; echo "DIR CHAIN COMPLETE ($(date +%H:%M))"
