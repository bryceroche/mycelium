#!/usr/bin/bash
set -eo pipefail
cd /home/bryce/mycelium

echo "== eq gate: pre-dumps A/B/C (unpatched module) =="
env EQ_CFG=A EQ_OUT=.cache/eqref21_pre_A.npz .venv/bin/python3 scripts/eq_check.py
env EQ_CFG=B EQ_OUT=.cache/eqref21_pre_B.npz .venv/bin/python3 scripts/eq_check.py
env EQ_CFG=C EQ_OUT=.cache/eqref21_pre_C.npz .venv/bin/python3 scripts/eq_check.py

echo "== apply the alt21 staged patch =="
.venv/bin/python3 scripts/apply_alternator_v21.py

echo "== eq gate: post-dumps A/B/C (patched module, ALG_ALT21 unset) =="
env EQ_CFG=A EQ_OUT=.cache/eqref21_post_A.npz .venv/bin/python3 scripts/eq_check.py
env EQ_CFG=B EQ_OUT=.cache/eqref21_post_B.npz .venv/bin/python3 scripts/eq_check.py
env EQ_CFG=C EQ_OUT=.cache/eqref21_post_C.npz .venv/bin/python3 scripts/eq_check.py
.venv/bin/python3 .cache/eq_compare.py .cache/eqref21_pre_A.npz .cache/eqref21_post_A.npz
.venv/bin/python3 .cache/eq_compare.py .cache/eqref21_pre_B.npz .cache/eqref21_post_B.npz
.venv/bin/python3 .cache/eq_compare.py .cache/eqref21_pre_C.npz .cache/eqref21_post_C.npz

echo "== alt21 smoke (300 steps, mechanics + step-time) =="
env DEV=AMD ALG2=1 ALG_FTYPES=9 ALG_DUP=1 ALG_HW=512 ALG_WIDE=1 ALG_BREATH=7 ALG_NOTEBOOK=1 ALG_SIXWAVE=1 NB_PERSLOT=1 ALG_BINDBUS=7 ALG_BIND_D=512 BIND_CODES=.cache/bindbus_codes512.npz ALG_ALLOW_PEN_TRAIN=1 ALG_TRAIN=.cache/form_mix11.jsonl ALG_TRAIN_NAME=form11 ALG_TEST=.cache/algebra_nl_test.jsonl ALG_TEST_NAME=test23 BATCH=8 SNAP_EVERY=0 ALG_BUSGARAGE=2 ALG_SHELF_CIRCLE=2 ALG_ALTMASK=1 SC_EVAL=0 ALG_ALT2=1 ALG_ALT21=1 WARM_FROM=.cache/sharp_bind14a.safetensors ALG_CKPT=.cache/sharp_alt21smoke.safetensors STEPS=300 SEED=239 .venv/bin/python3 scripts/phase1_algebra_head.py --train > .cache/alt21_smoke.log 2>&1
tail -3 .cache/alt21_smoke.log

echo "== THE CANDIDATE: 4-layer step + alternation (seed 241, 50k, from noise) =="
env DEV=AMD ALG2=1 ALG_FTYPES=9 ALG_DUP=1 ALG_HW=512 ALG_WIDE=1 ALG_BREATH=7 ALG_NOTEBOOK=1 ALG_SIXWAVE=1 NB_PERSLOT=1 ALG_BINDBUS=7 ALG_BIND_D=512 BIND_CODES=.cache/bindbus_codes512.npz ALG_ALLOW_PEN_TRAIN=1 ALG_TRAIN=.cache/form_mix12.jsonl ALG_TRAIN_NAME=form12 ALG_TEST=.cache/algebra_nl_test.jsonl ALG_TEST_NAME=test23 BATCH=8 SNAP_EVERY=0 ALG_BUSGARAGE=2 ALG_SHELF_CIRCLE=2 ALG_ALTMASK=1 SC_EVAL=0 ALG_ALT2=1 ALG_ALT21=1 ALG_CKPT=.cache/sharp_alt21cand241.safetensors STEPS=50000 SEED=241 .venv/bin/python3 scripts/phase1_algebra_head.py --train > .cache/alt21_cand241.log 2>&1

echo "== read fleet: loop_val test23 + wildhold, bind_read =="
{
env DEV=AMD ALG2=1 ALG_FTYPES=9 ALG_DUP=1 ALG_HW=512 ALG_WIDE=1 ALG_BREATH=7 ALG_NOTEBOOK=1 ALG_SIXWAVE=1 NB_PERSLOT=1 ALG_BINDBUS=7 ALG_BIND_D=512 BIND_CODES=.cache/bindbus_codes512.npz ALG_TEST=.cache/algebra_nl_test.jsonl ALG_TEST_NAME=test23 ALG_BUSGARAGE=2 ALG_SHELF_CIRCLE=2 ALG_ALTMASK=1 SC_EVAL=0 ALG_ALT2=1 ALG_ALT21=1 LV_CKPT=.cache/sharp_alt21cand241.safetensors .venv/bin/python3 scripts/loop_val.py
env DEV=AMD ALG2=1 ALG_FTYPES=9 ALG_DUP=1 ALG_HW=512 ALG_WIDE=1 ALG_BREATH=7 ALG_NOTEBOOK=1 ALG_SIXWAVE=1 NB_PERSLOT=1 ALG_BINDBUS=7 ALG_BIND_D=512 BIND_CODES=.cache/bindbus_codes512.npz ALG_TEST=.cache/wild_admitted_holdout.jsonl ALG_TEST_NAME=wildhold ALG_BUSGARAGE=2 ALG_SHELF_CIRCLE=2 ALG_ALTMASK=1 SC_EVAL=0 ALG_ALT2=1 ALG_ALT21=1 LV_CKPT=.cache/sharp_alt21cand241.safetensors .venv/bin/python3 scripts/loop_val.py
env DEV=AMD ALG2=1 ALG_FTYPES=9 ALG_DUP=1 ALG_HW=512 ALG_WIDE=1 ALG_BREATH=7 ALG_NOTEBOOK=1 ALG_SIXWAVE=1 NB_PERSLOT=1 ALG_BINDBUS=7 ALG_BIND_D=512 BIND_CODES=.cache/bindbus_codes512.npz ALG_ALLOW_PEN_TRAIN=1 ALG_TRAIN=.cache/form_mix12.jsonl ALG_TRAIN_NAME=form12 ALG_TEST=.cache/algebra_nl_test.jsonl ALG_TEST_NAME=test23 BATCH=8 SNAP_EVERY=0 ALG_BUSGARAGE=2 ALG_SHELF_CIRCLE=2 ALG_ALTMASK=1 SC_EVAL=0 ALG_ALT2=1 ALG_ALT21=1 BR_V=7 BR_D=512 BR_CKPT=.cache/sharp_alt21cand241.safetensors .venv/bin/python3 scripts/bind_read.py 2>&1 | grep "bind gold"
} >> .cache/alt21_reads.log 2>&1
cat .cache/alt21_reads.log
echo "ALT21 CHAIN COMPLETE"
