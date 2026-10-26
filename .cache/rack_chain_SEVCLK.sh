#!/usr/bin/bash
# THE CLOCK SEVER READS (2026-10-26; Bryce: "word given for the sever read -- Salute!"; ledger "THE CLOCK SEVER READS").
# Eval only. Bodies SRK3_241, CLK3_241; arms OPEN / SW0 (sw_g = 0 checkpoint copy) / CLOCKBAND (ALG_SEVER=clockband).
# Lines per sever, both bodies: PASSENGER open-sever < 0.010; LOAD-BEARING >= 0.020; else BETWEEN.
set -eo pipefail; cd /home/bryce/mycelium; exec > .cache/rack_chain_SEVCLK.log 2>&1
while systemctl --user is-active -q t3cd; do sleep 60; done          # CPU toys run alone
echo "== start $(date +%H:%M) =="
FAM="DEV=PCI+AMD ALG2=1 ALG_FTYPES=9 ALG_DUP=1 ALG_HW=512 ALG_WIDE=1 ALG_BREATH=7 ALG_NOTEBOOK=1 ALG_SIXWAVE=1 NB_PERSLOT=1 ALG_BINDBUS=7 ALG_BIND_D=512 BIND_CODES=.cache/bindbus_codes512.npz ALG_BUSGARAGE=2 ALG_SHELF_CIRCLE=2 ALG_ALTMASK=1 ALG_ALT21=1 ALG_ALT2=1 ALG_MASKHEAD=1 ALG_FED=1 ALG_POLAR=1 ALG_POLAR_D=128 ALG_POLAR_EM=0.1 ALG_POLAR_D_INIT=.cache/polar_waist_init_d128u.npz ALG_PRUNE=pforms,s4,fednl0,lane2 ALG_SLOT_ALL=1 ALG_STELLAR=2 ALG_CLOCK_CANON=1 SC_EVAL=0"
SURF8="ALG_ROUTER=2 R_GAIN_INIT=1.0 ALG_FREEZE=r_gain ALG_ROUTER_PTR=0.0 ALG_SPAN_ALL=1 ALG_SPAN_ARGS=1 ALG_SPAN_OP=1 ALG_SPAN_RCUE=1 ALG_SPAN_ARCUE=1 ALG_PTR_SURF=role:add:2.0 BIND_CODES=.cache/bindbus_codes512r.npz"
LSX="ALG_LOOP_STOP=3"
W="ALG_TEST=.cache/wild_admitted_holdout.jsonl ALG_TEST_NAME=wildholdBP"
RL() { # RL TAG CKPT ENV : one masked wild read
  flock -w 36000 .cache/gpu.lock env $FAM $W $3 ALG_JIT_READ=1 LV_LEGAL=num LV_FIELDS=1 LV_CKPT=$2 LV_PER_SLOT=.cache/ps_sevclk_$1.npz .venv/bin/python3 scripts/loop_val.py > .cache/read_sevclk_$1.log 2>&1
  echo "masked wild $1: $(grep -aoE 'fac-exact=[0-9.]+' .cache/read_sevclk_$1.log | tail -1) | $(grep -a '^\[fields\]' .cache/read_sevclk_$1.log | cut -c1-200) [$(date +%H:%M)]"
}
for ARM in "SRK3_241:$SURF8 $LSX ALG_SORT=4" "CLK3_241:$SURF8 $LSX ALG_SORT=4 ALG_CLIMB=1"; do C=${ARM%%:*}; X=${ARM#*:}
  RL open_$C .cache/sharp_$C.safetensors "$X"
  RL sw0_$C .cache/sharp_${C}_sw0.safetensors "$X"
  RL clockband_$C .cache/sharp_$C.safetensors "$X ALG_SEVER=clockband"
  echo "PAIRED $C open vs banked 10-15: $(.venv/bin/python3 scripts/paired_read.py .cache/ps_sevclk_open_$C.npz .cache/ps_legal_wild_$C.npz today banked)"
  echo "PAIRED $C SW0: $(.venv/bin/python3 scripts/paired_read.py .cache/ps_sevclk_open_$C.npz .cache/ps_sevclk_sw0_$C.npz open sw0)"
  echo "PAIRED $C CLOCKBAND: $(.venv/bin/python3 scripts/paired_read.py .cache/ps_sevclk_open_$C.npz .cache/ps_sevclk_clockband_$C.npz open clockband)"
done
echo "== done $(date +%H:%M) =="
