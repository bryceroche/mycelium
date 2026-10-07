#!/usr/bin/bash
# THE RACK (2026-10-05, word given): RKX_241 = EY_241's recipe (HS_241 body, TAU 0,) + ALG_ALT3=1 ALG_CERT=2.0 (the facts road) + ALG_RACK=1 ALG_RACK_TESTS=given_unique ALG_RACK_FREEZE=leaf (the per-slot dryness test at the consults; freeze + hard token claim + one-way glass), and RKC_241 = the same WITHOUT ALG_RACK (the honest control: the consults' own cost). From scratch, 48k, seed 241.
# BARS, THE ARM (ledger 2026-10-06 17:02, pinned before any fire): open mint >= 0.586 (HS_241's —
# the consults' -0.046 cost on the hierarchical body removed); masked wild >= 0.3432 - 0.005 (RK_241's
# own number, within one paired SE); rows >= 15; THE HIT (membrane_rack.py) >= 0.72; THE WALL <= 0.49.
# RKX vs RKXC paired is the rack's own bar, unchanged from RK vs RKC (>= +0.015, claim +0.020/twin).
# THE SCHEDULE COLLISION FIX (2026-10-06): RKX_241 = RK_241's exact recipe + ALG_HIER_DAMP=3,5,0
# (the chosen knob — settle breaths moved to AFTER each consult's evidence first arrives; built on
# worktree mycelium-wt7, branch listen; needs NO new flag, conductor(kb)'s existing arithmetic).
# REGISTERED, NOT FIRED (ledger 2026-10-06; the diagnosis + listen_gate.sh land first).
BODY=HS_241; TAU=0
set -eo pipefail; cd /home/bryce/mycelium; exec > .cache/rack_chain_RKX.log 2>&1
# audit 2026-10-06: .cache/listen_merge_gate.log exists only when queue_listen.sh performs the merge itself (its "listen already merged" branch
# produces no post-merge gate) — and the preflight below re-greps that log, so a hand-merged tree would abort this chain at 05:00 with nobody
# watching. Self-heal: derive and run the SAME post-merge gate queue_listen.sh would have (the rule: the arm trains what was gated).
if [ ! -s .cache/listen_merge_gate.log ]; then echo "NOTE: no .cache/listen_merge_gate.log — deriving and running the post-merge gate now ($(date +%H:%M))"
  sed -e 's#/home/bryce/mycelium-wt7#/home/bryce/mycelium#g' -e 's#listen_gate.log#listen_merge_gate.log#' -e 's#listen_gate_\$N#listen_merge_gate_$N#g' .cache/listen_gate.sh > .cache/listen_merge_gate.sh
  /usr/bin/bash .cache/listen_merge_gate.sh || true; echo "post-merge gate (run by $0): $(grep -a '^gate' .cache/listen_merge_gate.log | tr '\n' ' ' | cut -c1-700)"
fi
.venv/bin/python3 scripts/contracts/preflight.py "$0" || exit 1   # THE PRE-FLIGHT CONTRACT
FAM="DEV=PCI+AMD ALG2=1 ALG_FTYPES=9 ALG_DUP=1 ALG_HW=512 ALG_WIDE=1 ALG_BREATH=7 ALG_NOTEBOOK=1 ALG_SIXWAVE=1 NB_PERSLOT=1 ALG_BINDBUS=7 ALG_BIND_D=512 BIND_CODES=.cache/bindbus_codes512.npz ALG_BUSGARAGE=2 ALG_SHELF_CIRCLE=2 ALG_ALTMASK=1 ALG_ALT21=1 ALG_ALT2=1 ALG_MASKHEAD=1 ALG_FED=1 ALG_POLAR=1 ALG_POLAR_D=128 ALG_POLAR_EM=0.1 ALG_POLAR_D_INIT=.cache/polar_waist_init_d128u.npz ALG_PRUNE=pforms,s4,fednl0,lane2 ALG_SLOT_ALL=1 ALG_STELLAR=2 ALG_CLOCK_CANON=1 SC_EVAL=0"
TR="ALG_ALLOW_PEN_TRAIN=1 ALG_TRAIN=.cache/form_mix_pm35c.jsonl ALG_TRAIN_NAME=formpm35c ALG_TEST=.cache/algebra_nl_test.jsonl ALG_TEST_NAME=test23 ALG_MASKPREP_CACHE=1 ALG_MASKPREP_JIT=1 ALG_FACTS_POOL=1 ALG_FACTS_WORKERS=6 ALG_MASKPREP_IGNORE=ALG_MASK_COOK,ALG_MASK_COOK_SKEL,ALG_TOK_COOK,ALG_TOK_COOK_S3,ALG_TG_INIT,ALG_BAL_COOK,ALG_NB2,ALG_ALT5,ALG_PRUNE,ALG_MASKPREP_JIT,ALG_MASKPREP_B,ALG_FACTS_POOL,ALG_FACTS_FLUSH,ALG_JIT_VAL LR=1e-4 SEED=241 SNAP_EVERY=100000"
SURF8="${SURF8:-ALG_ROUTER=2 R_GAIN_INIT=1.0 ALG_FREEZE=r_gain ALG_ROUTER_PTR=0.0 ALG_SPAN_ALL=1 ALG_SPAN_ARGS=1 ALG_SPAN_OP=1 ALG_SPAN_RCUE=1 ALG_SPAN_ARCUE=1 ALG_PTR_SURF=role:add:2.0 BIND_CODES=.cache/bindbus_codes512r.npz}"
W="ALG_TEST=.cache/wild_admitted_holdout.jsonl ALG_TEST_NAME=wildhold"; M="ALG_TEST=.cache/algebra_nl_test.jsonl ALG_TEST_NAME=test23"; S="ALG_TEST=.cache/silver460_test.jsonl ALG_TEST_NAME=silver460"
grep -aq "gate unset: step     0 loss=5.2995" .cache/listen_merge_gate.log || { echo "ABORT: the listen gate did not certify the head bit-identical unset"; exit 1; }
grep -a "gate listenrack:" .cache/listen_merge_gate.log | tail -1 | grep -aq "loss=[0-9]" || { echo "ABORT: the listenrack config (RACK + ALG_HIER_DAMP=3,5,0's sibling, listenrack) does not run on the fixture"; exit 1; }
grep -a "gate rkarm:" .cache/listen_merge_gate.log | tail -1 | grep -aq "loss=10.8373" || { echo "ABORT: rkarm did not reproduce 10.8373 (the knob-unset reference) on this worktree"; exit 1; }
for ARM in "RKX_241:$SURF8 ALG_HIER_READ=1 ALG_HIER_WAIST=1 ALG_HIER_DAMP=3,5,0 ALG_HIER_TAU=$TAU ALG_ALT3=1 ALG_CERT=2.0 ALG_RACK=1 ALG_RACK_TESTS=given_unique ALG_RACK_FREEZE=leaf SNAP_EVERY=8000"; do C=${ARM%%:*}; X=${ARM#*:}
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
for C in RKX_241; do X="$SURF8 ALG_HIER_READ=1 ALG_HIER_WAIST=1 ALG_HIER_DAMP=3,5,0 ALG_HIER_TAU=$TAU ALG_ALT3=1 ALG_CERT=2.0"; [ $C = RKX_241 ] && X="$X ALG_RACK=1 ALG_RACK_TESTS=given_unique ALG_RACK_FREEZE=leaf"
  echo "paired masked wild $C vs $BODY (the body): $(.venv/bin/python3 scripts/paired_read.py .cache/ps_legal_wild_$BODY.npz .cache/ps_legal_wild_$C.npz $BODY $C | tail -1)"
  echo "paired masked wild $C vs EY_241 (the eyes): $(.venv/bin/python3 scripts/paired_read.py .cache/ps_legal_wild_EY_241.npz .cache/ps_legal_wild_$C.npz EY_241 $C | tail -1)"
  echo "paired masked wild $C vs PMS8_241: $(.venv/bin/python3 scripts/paired_read.py .cache/ps_legal_wild_PMS8_241.npz .cache/ps_legal_wild_$C.npz PMS8_241 $C | tail -1)"
  CMD=$(.venv/bin/python3 scripts/contracts/golden.py command .cache/sharp_$C.safetensors --tag $C 2>/dev/null | grep -v "^#" | head -1 | sed "s|ALG_FTYPES=9|ALG_FTYPES=9 $X|") || true
  if [ -n "$CMD" ]; then eval "$CMD" || echo "golden command FAILED $C"; else echo "golden command EMPTY $C (golden.py command printed nothing)"; fi   # audit 2026-10-06: the bare $(...) form executed golden.py's line by word-splitting, so its "> .cache/golden24_check_$C.log 2>&1" became literal argv and the log was never written (rack_tail.sh's eval form is the one that worked at 19:54); an empty expansion also aborted the chain under set -e
  .venv/bin/python3 scripts/contracts/golden.py check .cache/golden24_rows_$C.json --tag $C || echo "golden: see above"
  flock -w 36000 .cache/gpu.lock env $FAM $W $X CB_MODE=collect CKPT=.cache/sharp_$C.safetensors .venv/bin/python3 scripts/clock_band_probe.py > .cache/clock_band_collect_$C.log 2>&1 || echo "band-probe collect FAILED $C"
  env $FAM $W $X DEV=CPU .venv/bin/python3 scripts/membrane_rack.py .cache/sharp_$C.safetensors --tag $C > .cache/membrane_rack_$C.log 2>&1 || echo "membrane-rack collect FAILED $C"   # audit 2026-10-06: the rack-aware ALT3+CERT+RACK read cycle (CPU, ~7 min; the ledger's own 14:35/20:11 instrument). membrane_scale.py's collect threads none of facts3/cert3/rack3: it asserts at the head under ALG_RACK=1 (rack_tail.log 19:54) and on the consult-only controls silently measures the ALT2 cycle, not the trained composition.
  grep -A8 "argmax band share, WRONG" .cache/membrane_rack_$C.txt | grep other_sent | sed "s/^/WALL $C /" || true; grep -A8 "argmax band share, RIGHT" .cache/membrane_rack_$C.txt | grep "  token" | sed "s/^/HIT $C /" || true   # audit 2026-10-06: other_sent sits 7 lines below the header (-A4 never reached it: the WALL bar printed NOTHING, silently)
  [ -f scripts/rack_dryness_census.py ] && .venv/bin/python3 scripts/rack_dryness_census.py .cache/dump_wild_$C.pkl --tag $C 2>&1 | tail -6 | sed "s/^/DRY $C /"
done
.venv/bin/python3 scripts/reads.py ingest | tail -1; echo "RACK RKX CHAIN COMPLETE ($(date +%H:%M))"
