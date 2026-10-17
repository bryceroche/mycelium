#!/usr/bin/bash
# THE NAME-SWAP VOTE READ (2026-10-16; Bryce: "agreed and word given"; registered + bars pinned in the ledger 19:20). INFERENCE
# ONLY, no training: SRK3_241 reads the wild original (view 0) and 4 photobooth name-swap views (scripts/photobooth.py;
# honesty check passed). Each view: BP states via precompute_masked --mask bidirpad (its tokmask assert = gate 1), then
# photobooth_stage.py (tokmask + sentence indices + every gold array bit-identical to wildholdBP; npz stamped). Reads masked
# (LV_LEGAL=num), SRK3's own read env. Verdict: scripts/picker/photobooth_vote_read.py SRK3_241 (the pinned frame).
set -eo pipefail; cd /home/bryce/mycelium; exec > .cache/rack_chain_PB.log 2>&1
.venv/bin/python3 scripts/contracts/preflight.py "$0" || exit 1   # THE PRE-FLIGHT CONTRACT
FAM="DEV=PCI+AMD ALG2=1 ALG_FTYPES=9 ALG_DUP=1 ALG_HW=512 ALG_WIDE=1 ALG_BREATH=7 ALG_NOTEBOOK=1 ALG_SIXWAVE=1 NB_PERSLOT=1 ALG_BINDBUS=7 ALG_BIND_D=512 BIND_CODES=.cache/bindbus_codes512.npz ALG_BUSGARAGE=2 ALG_SHELF_CIRCLE=2 ALG_ALTMASK=1 ALG_ALT21=1 ALG_ALT2=1 ALG_MASKHEAD=1 ALG_FED=1 ALG_POLAR=1 ALG_POLAR_D=128 ALG_POLAR_EM=0.1 ALG_POLAR_D_INIT=.cache/polar_waist_init_d128u.npz ALG_PRUNE=pforms,s4,fednl0,lane2 ALG_SLOT_ALL=1 ALG_STELLAR=2 ALG_CLOCK_CANON=1 SC_EVAL=0"
SURF8="ALG_ROUTER=2 R_GAIN_INIT=1.0 ALG_FREEZE=r_gain ALG_ROUTER_PTR=0.0 ALG_SPAN_ALL=1 ALG_SPAN_ARGS=1 ALG_SPAN_OP=1 ALG_SPAN_RCUE=1 ALG_SPAN_ARCUE=1 ALG_PTR_SURF=role:add:2.0 BIND_CODES=.cache/bindbus_codes512r.npz"
X="$SURF8 ALG_LOOP_STOP=3 ALG_SORT=4"
FAMCPU="${FAM/DEV=PCI+AMD/DEV=CPU}"
echo "== STAGE VIEW 1 ($(date +%H:%M)) =="
[ -f .cache/phase1_alg_states_wildPB1.npz ] || flock -w 36000 .cache/gpu.lock env DEV=PCI+AMD .venv/bin/python3 scripts/precompute_masked.py --split wildholdBP --jsonl .cache/wild_PB_v1.jsonl --mask bidirpad --out wildPB1 > .cache/pb_precompute_v1.log 2>&1
[ -f .cache/phase1_alg_states_wildPB1_states.npy ] || { echo "ABORT: precompute view 1 wrote no states"; exit 1; }
tail -1 .cache/pb_precompute_v1.log
env $FAMCPU $SURF8 .venv/bin/python3 scripts/photobooth_stage.py wildholdBP .cache/wild_PB_v1.jsonl wildPB1 | tail -1
echo "== STAGE VIEW 2 ($(date +%H:%M)) =="
[ -f .cache/phase1_alg_states_wildPB2.npz ] || flock -w 36000 .cache/gpu.lock env DEV=PCI+AMD .venv/bin/python3 scripts/precompute_masked.py --split wildholdBP --jsonl .cache/wild_PB_v2.jsonl --mask bidirpad --out wildPB2 > .cache/pb_precompute_v2.log 2>&1
[ -f .cache/phase1_alg_states_wildPB2_states.npy ] || { echo "ABORT: precompute view 2 wrote no states"; exit 1; }
tail -1 .cache/pb_precompute_v2.log
env $FAMCPU $SURF8 .venv/bin/python3 scripts/photobooth_stage.py wildholdBP .cache/wild_PB_v2.jsonl wildPB2 | tail -1
echo "== STAGE VIEW 3 ($(date +%H:%M)) =="
[ -f .cache/phase1_alg_states_wildPB3.npz ] || flock -w 36000 .cache/gpu.lock env DEV=PCI+AMD .venv/bin/python3 scripts/precompute_masked.py --split wildholdBP --jsonl .cache/wild_PB_v3.jsonl --mask bidirpad --out wildPB3 > .cache/pb_precompute_v3.log 2>&1
[ -f .cache/phase1_alg_states_wildPB3_states.npy ] || { echo "ABORT: precompute view 3 wrote no states"; exit 1; }
tail -1 .cache/pb_precompute_v3.log
env $FAMCPU $SURF8 .venv/bin/python3 scripts/photobooth_stage.py wildholdBP .cache/wild_PB_v3.jsonl wildPB3 | tail -1
echo "== STAGE VIEW 4 ($(date +%H:%M)) =="
[ -f .cache/phase1_alg_states_wildPB4.npz ] || flock -w 36000 .cache/gpu.lock env DEV=PCI+AMD .venv/bin/python3 scripts/precompute_masked.py --split wildholdBP --jsonl .cache/wild_PB_v4.jsonl --mask bidirpad --out wildPB4 > .cache/pb_precompute_v4.log 2>&1
[ -f .cache/phase1_alg_states_wildPB4_states.npy ] || { echo "ABORT: precompute view 4 wrote no states"; exit 1; }
tail -1 .cache/pb_precompute_v4.log
env $FAMCPU $SURF8 .venv/bin/python3 scripts/photobooth_stage.py wildholdBP .cache/wild_PB_v4.jsonl wildPB4 | tail -1
flock -w 36000 .cache/gpu.lock env $FAM ALG_TEST=.cache/wild_admitted_holdout.jsonl ALG_TEST_NAME=wildholdBP $X ALG_JIT_READ=1 LV_LEGAL=num LV_FIELDS=1 LV_DUMP=.cache/dump_wildPB0_SRK3_241.pkl LV_CKPT=.cache/sharp_SRK3_241.safetensors LV_PER_SLOT=.cache/ps_legal_wildPB0_SRK3_241.npz .venv/bin/python3 scripts/loop_val.py > .cache/read_legal_wildPB0_SRK3_241.log 2>&1 || { echo "FAILED read view 0"; exit 1; }
echo "masked wild view 0 SRK3_241: $(grep -aoE 'fac-exact=[0-9.]+' .cache/read_legal_wildPB0_SRK3_241.log | tail -1)"
if ! grep -aq "fac-exact=0.3774" .cache/read_legal_wildPB0_SRK3_241.log; then echo "ABORT: view 0 does not reproduce SRK3_241's banked masked wild 0.3774 (the in-chain read just above wrote this log)"; exit 1; fi
echo "VIEW-0 GATE: PASS (0.3774 reproduced)"
flock -w 36000 .cache/gpu.lock env $FAM ALG_TEST=.cache/wild_PB_v1.jsonl ALG_TEST_NAME=wildPB1 $X ALG_JIT_READ=1 LV_LEGAL=num LV_FIELDS=1 LV_DUMP=.cache/dump_wildPB1_SRK3_241.pkl LV_CKPT=.cache/sharp_SRK3_241.safetensors LV_PER_SLOT=.cache/ps_legal_wildPB1_SRK3_241.npz .venv/bin/python3 scripts/loop_val.py > .cache/read_legal_wildPB1_SRK3_241.log 2>&1 || { echo "FAILED read view 1"; exit 1; }
echo "masked wild view 1 SRK3_241: $(grep -aoE 'fac-exact=[0-9.]+' .cache/read_legal_wildPB1_SRK3_241.log | tail -1)"
flock -w 36000 .cache/gpu.lock env $FAM ALG_TEST=.cache/wild_PB_v2.jsonl ALG_TEST_NAME=wildPB2 $X ALG_JIT_READ=1 LV_LEGAL=num LV_FIELDS=1 LV_DUMP=.cache/dump_wildPB2_SRK3_241.pkl LV_CKPT=.cache/sharp_SRK3_241.safetensors LV_PER_SLOT=.cache/ps_legal_wildPB2_SRK3_241.npz .venv/bin/python3 scripts/loop_val.py > .cache/read_legal_wildPB2_SRK3_241.log 2>&1 || { echo "FAILED read view 2"; exit 1; }
echo "masked wild view 2 SRK3_241: $(grep -aoE 'fac-exact=[0-9.]+' .cache/read_legal_wildPB2_SRK3_241.log | tail -1)"
flock -w 36000 .cache/gpu.lock env $FAM ALG_TEST=.cache/wild_PB_v3.jsonl ALG_TEST_NAME=wildPB3 $X ALG_JIT_READ=1 LV_LEGAL=num LV_FIELDS=1 LV_DUMP=.cache/dump_wildPB3_SRK3_241.pkl LV_CKPT=.cache/sharp_SRK3_241.safetensors LV_PER_SLOT=.cache/ps_legal_wildPB3_SRK3_241.npz .venv/bin/python3 scripts/loop_val.py > .cache/read_legal_wildPB3_SRK3_241.log 2>&1 || { echo "FAILED read view 3"; exit 1; }
echo "masked wild view 3 SRK3_241: $(grep -aoE 'fac-exact=[0-9.]+' .cache/read_legal_wildPB3_SRK3_241.log | tail -1)"
flock -w 36000 .cache/gpu.lock env $FAM ALG_TEST=.cache/wild_PB_v4.jsonl ALG_TEST_NAME=wildPB4 $X ALG_JIT_READ=1 LV_LEGAL=num LV_FIELDS=1 LV_DUMP=.cache/dump_wildPB4_SRK3_241.pkl LV_CKPT=.cache/sharp_SRK3_241.safetensors LV_PER_SLOT=.cache/ps_legal_wildPB4_SRK3_241.npz .venv/bin/python3 scripts/loop_val.py > .cache/read_legal_wildPB4_SRK3_241.log 2>&1 || { echo "FAILED read view 4"; exit 1; }
echo "masked wild view 4 SRK3_241: $(grep -aoE 'fac-exact=[0-9.]+' .cache/read_legal_wildPB4_SRK3_241.log | tail -1)"
.venv/bin/python3 scripts/picker/photobooth_vote_read.py SRK3_241 | tee .cache/photobooth_vote_read_SRK3_241.txt
echo "PB CHAIN COMPLETE ($(date +%H:%M))"
