#!/usr/bin/bash
# (B2) HEAD CONTINUATION ON THE STUDENT'S STATES (Bryce 10-27: "word given for everything"; ledger 10-27 HEAD CONTINUATION).
# Export the s6x512 student over the training mix + test23 -> two MATCHED gentle continuations from CLK3_241 (same env,
# seed, steps, LR; only the states differ): HCT = teacher states (the control: what extra training alone buys), HCS =
# student states. Masked wild read each on its own states; paired reads.
# BARS (pinned before measurement): PRIMARY HCS >= 0.3593 (as registered 10-26 19:27). PARITY HCS - HCT >= -0.020 paired
# (the student replaces the trunk at inference); below that = a real student deficit the head cannot absorb.
set -eo pipefail; cd /home/bryce/mycelium; exec > .cache/run_headcont.log 2>&1
echo "== start $(date +%H:%M) =="
for J in ".cache/form_mix_pm35c.jsonl:formpm35cBP:formpm35cSTU" ".cache/algebra_nl_test.jsonl:test23BP:test23STU"; do IFS=: read -r F S O <<< "$J"
  [ -e .cache/phase1_alg_states_${O}_states.npy ] && { echo "export $O exists, kept"; continue; }
  flock -w 36000 .cache/gpu.lock env DEV=PCI+AMD DR_TAG=s6x512 .venv/bin/python3 scripts/distill_reader.py --export $F --src $S --out $O
done
FAM=$(grep -oP '^FAM="\K[^"]+' .cache/rack_chain_CLQ.sh); SURF8=$(grep -oP '^SURF8="\K[^"]+' .cache/rack_chain_CLQ.sh)
TRB=$(grep -oP '^TR="\K[^"]+' .cache/rack_chain_CLQ.sh | sed -e 's/ALG_TRAIN_NAME=formpm35cBP//' -e 's/ALG_TEST_NAME=test23BP//' -e 's/LR=1e-4//')
X="$SURF8 ALG_LOOP_STOP=3 ALG_SORT=4 ALG_CLIMB=1"
for ARM in "HCT:BP" "HCS:STU"; do C=${ARM%%:*}; V=${ARM#*:}
  echo "== ARM $C (warm CLK3_241, states ${V}, 12k B=8 LR 5e-5) ($(date +%H:%M)) =="
  flock -w 36000 .cache/gpu.lock env $FAM $TRB $X ALG_TRAIN_NAME=formpm35c$V ALG_TEST_NAME=test23$V LR=5e-5 BATCH=8 STEPS=12000 SNAP_EVERY=100000 \
    WARM_FROM=.cache/sharp_CLK3_241.safetensors ALG_CKPT=.cache/sharp_$C.safetensors PYTHONUNBUFFERED=1 .venv/bin/python3 -u scripts/phase1_algebra_head.py --train > .cache/sharp_$C.log 2>&1 \
    || { echo "FAILED $C: $(grep -aE 'Error|assert' .cache/sharp_$C.log | tail -1)"; exit 1; }
  grep -aE "WARM from|step +(0|6000|11999) |restored BEST" .cache/sharp_$C.log | cut -c1-200 || true
  flock -w 36000 .cache/gpu.lock env $FAM ALG_TEST=.cache/wild_admitted_holdout.jsonl ALG_TEST_NAME=wildhold$V $X ALG_JIT_READ=1 LV_LEGAL=num LV_FIELDS=1 \
    LV_CKPT=.cache/sharp_$C.safetensors LV_PER_SLOT=.cache/ps_headcont_$C.npz .venv/bin/python3 scripts/loop_val.py > .cache/read_headcont_$C.log 2>&1
  echo "masked wild $C: $(grep -aoE 'fac-exact=[0-9.]+' .cache/read_headcont_$C.log | tail -1) | $(grep -a '^\[fields\]' .cache/read_headcont_$C.log | cut -c1-200)"
done
echo "PAIRED HCS vs HCT (PARITY): $(.venv/bin/python3 scripts/paired_read.py .cache/ps_headcont_HCT.npz .cache/ps_headcont_HCS.npz HCT HCS)"
echo "PAIRED HCS vs frozen student: $(.venv/bin/python3 scripts/paired_read.py .cache/ps_distill_s6x512.npz .cache/ps_headcont_HCS.npz frozen HCS)"
echo "PAIRED HCT vs CLK3_241: $(.venv/bin/python3 scripts/paired_read.py .cache/ps_sevclk_open_CLK3_241.npz .cache/ps_headcont_HCT.npz CLK3 HCT)"
echo "== done $(date +%H:%M) =="
