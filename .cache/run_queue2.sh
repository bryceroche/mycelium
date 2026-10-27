#!/usr/bin/bash
# THE QUEUE (Bryce 10-27: "word given for the whole queue"; ledger 10-27 THE SHRUNK READER + THE BUS ON REAL TEXT).
# (1) GPU: the shrunk reader -- export wildhold through the FOLDED table and the PRUNED table (reader_vocab.py's K + fallback
#     pieces); masked wild reads with the HCS head; paired vs HCS's own read. (2) CPU alone: eq_slot BUS arm, seeds 0,1, STU.
set -eo pipefail; cd /home/bryce/mycelium; exec > .cache/run_queue2.log 2>&1
while systemctl --user is-active -q eqslot; do sleep 60; done
echo "== start $(date +%H:%M) =="
FAM=$(grep -oP '^FAM="\K[^"]+' .cache/rack_chain_CLQ.sh); SURF8=$(grep -oP '^SURF8="\K[^"]+' .cache/rack_chain_CLQ.sh)
X="$SURF8 ALG_LOOP_STOP=3 ALG_SORT=4 ALG_CLIMB=1"
for M in "fold:STUF" "prune:STUP"; do V=${M%%:*}; S=${M#*:}
  [ -e .cache/phase1_alg_states_wildhold${S}_states.npy ] || flock -w 36000 .cache/gpu.lock env DEV=PCI+AMD DR_TAG=s6x512 DR_VOCAB=$V .venv/bin/python3 scripts/distill_reader.py --export .cache/wild_admitted_holdout.jsonl --src wildholdBP --out wildhold$S 2>&1 | grep -v -i warn
  flock -w 36000 .cache/gpu.lock env $FAM ALG_TEST=.cache/wild_admitted_holdout.jsonl ALG_TEST_NAME=wildhold$S $X ALG_JIT_READ=1 LV_LEGAL=num LV_FIELDS=1 \
    LV_CKPT=.cache/sharp_HCS.safetensors LV_PER_SLOT=.cache/ps_shrink_$S.npz .venv/bin/python3 scripts/loop_val.py > .cache/read_shrink_$S.log 2>&1
  echo "masked wild HCS on $S: $(grep -aoE 'fac-exact=[0-9.]+' .cache/read_shrink_$S.log | tail -1) | $(grep -a '^\[fields\]' .cache/read_shrink_$S.log | cut -c1-200)"
  echo "PAIRED $S vs STU: $(.venv/bin/python3 scripts/paired_read.py .cache/ps_headcont_HCS.npz .cache/ps_shrink_$S.npz STU $S)"
done
.venv/bin/python3 scripts/reader_shrink_verdict.py
for S in 0 1; do EQ_ARM=BUS EQ_SEED=$S EQ_READER=STU TOY_THREADS=16 .venv/bin/python3 scripts/eq_slot_breath.py 2>&1 | grep -v -i warn; done
.venv/bin/python3 scripts/eq_slot_bus_verdict.py
echo "== done $(date +%H:%M) =="
