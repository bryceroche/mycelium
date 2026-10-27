#!/usr/bin/bash
# THE WHOLE-WORD BACKSTOP ARM (Bryce 10-27 "word given"; ledger 10-27): export wildhold through the 48,170-row table; HCS read; verdict.
set -eo pipefail; cd /home/bryce/mycelium; exec > .cache/run_queue3.log 2>&1
while systemctl --user is-active -q queue2; do sleep 60; done          # after the bus arm: no GPU beside a CPU run
echo "== start $(date +%H:%M) =="
FAM=$(grep -oP '^FAM="\K[^"]+' .cache/rack_chain_CLQ.sh); SURF8=$(grep -oP '^SURF8="\K[^"]+' .cache/rack_chain_CLQ.sh)
X="$SURF8 ALG_LOOP_STOP=3 ALG_SORT=4 ALG_CLIMB=1"
[ -e .cache/phase1_alg_states_wildholdSTUW_states.npy ] || flock -w 36000 .cache/gpu.lock env DEV=PCI+AMD DR_TAG=s6x512 DR_VOCAB=prune DR_VOCAB_K=.cache/reader_vocab_kw.npz .venv/bin/python3 scripts/distill_reader.py --export .cache/wild_admitted_holdout.jsonl --src wildholdBP --out wildholdSTUW 2>&1 | grep -v -i warn
flock -w 36000 .cache/gpu.lock env $FAM ALG_TEST=.cache/wild_admitted_holdout.jsonl ALG_TEST_NAME=wildholdSTUW $X ALG_JIT_READ=1 LV_LEGAL=num LV_FIELDS=1 \
  LV_CKPT=.cache/sharp_HCS.safetensors LV_PER_SLOT=.cache/ps_shrink_STUW.npz .venv/bin/python3 scripts/loop_val.py > .cache/read_shrink_STUW.log 2>&1
echo "masked wild HCS on STUW: $(grep -aoE 'fac-exact=[0-9.]+' .cache/read_shrink_STUW.log | tail -1) | $(grep -a '^\[fields\]' .cache/read_shrink_STUW.log | cut -c1-200)"
.venv/bin/python3 scripts/reader_shrink_w_verdict.py
echo "== done $(date +%H:%M) =="
