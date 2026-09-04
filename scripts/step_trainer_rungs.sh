#!/usr/bin/bash
# step_trainer_rungs.sh — THE FINAL BOSS BRING-UP LADDER (2026-09-03).
# alternator_v21_training_spec.md S4, walked without shortcuts. NOT FIRED
# at delivery (hold-for-the-word law); every line literal, zero variables.
# Precondition: apply_step_trainer.py is APPLIED (done at delivery;
# factoring verified bitwise on CPU) and no other GPU process is live
# (AM driver is single-process). Run rungs IN ORDER; each rung's failure
# stops the ladder (set -e). Blackbird fence: rung-2/3 early-step reads
# are leak gauges, never verdicts — judgment at cruise on rung 4.
set -eo pipefail
cd /home/bryce/mycelium

echo "== rung 0: CPU-only re-verification (no GPU; safe while anything runs elsewhere is NOT true — AM is single-process, run alone anyway) =="
.venv/bin/python3 scripts/step_trainer.py --selftest
.venv/bin/python3 scripts/step_engine_read.py --selftest
env ST_JIT=0 .venv/bin/python3 scripts/step_trainer.py --cpuprobe
env ST_JIT=1 .venv/bin/python3 scripts/step_trainer.py --cpuprobe

echo "== rung 1: forward equivalence on the champion — step-partitioned vs fused, np.array_equal per key, FAIL LOUDLY on any diff (ST_EQ_TOL=1e-5 is the pinned bar; the printout carries the numbers if a re-pin must be argued) =="
env DEV=AMD ALG2=1 ALG_FTYPES=9 ALG_DUP=1 ALG_HW=512 ALG_WIDE=1 ALG_BREATH=7 ALG_NOTEBOOK=1 ALG_SIXWAVE=1 NB_PERSLOT=1 ALG_BINDBUS=7 ALG_BIND_D=512 BIND_CODES=.cache/bindbus_codes512.npz ALG_ALLOW_PEN_TRAIN=1 ALG_TRAIN=.cache/form_mix12.jsonl ALG_TRAIN_NAME=form12 ALG_BUSGARAGE=2 ALG_SHELF_CIRCLE=2 ALG_ALTMASK=1 SC_EVAL=0 ALG_ALT21=1 ALG_ALT2=1 ST_CKPT=.cache/sharp_port242.safetensors ST_EQN=8 ST_PING=0 ST_JIT=0 .venv/bin/python3 scripts/step_trainer.py --eqfwd
env DEV=AMD ALG2=1 ALG_FTYPES=9 ALG_DUP=1 ALG_HW=512 ALG_WIDE=1 ALG_BREATH=7 ALG_NOTEBOOK=1 ALG_SIXWAVE=1 NB_PERSLOT=1 ALG_BINDBUS=7 ALG_BIND_D=512 BIND_CODES=.cache/bindbus_codes512.npz ALG_ALLOW_PEN_TRAIN=1 ALG_TRAIN=.cache/form_mix12.jsonl ALG_TRAIN_NAME=form12 ALG_BUSGARAGE=2 ALG_SHELF_CIRCLE=2 ALG_ALTMASK=1 SC_EVAL=0 ALG_ALT21=1 ALG_ALT2=1 ST_CKPT=.cache/sharp_port242.safetensors ST_EQN=8 ST_PING=0 ST_JIT=1 .venv/bin/python3 scripts/step_trainer.py --eqfwd

echo "== rung 1b: backward equivalence — reverse-walk grads vs one fused training step, printed then asserted (1e-4 relative, 1e-6-of-scale dust floor, pinned) =="
env DEV=AMD ALG2=1 ALG_FTYPES=9 ALG_DUP=1 ALG_HW=512 ALG_WIDE=1 ALG_BREATH=7 ALG_NOTEBOOK=1 ALG_SIXWAVE=1 NB_PERSLOT=1 ALG_BINDBUS=7 ALG_BIND_D=512 BIND_CODES=.cache/bindbus_codes512.npz ALG_ALLOW_PEN_TRAIN=1 ALG_TRAIN=.cache/form_mix12.jsonl ALG_TRAIN_NAME=form12 ALG_BUSGARAGE=2 ALG_SHELF_CIRCLE=2 ALG_ALTMASK=1 SC_EVAL=0 ALG_ALT21=1 ALG_ALT2=1 ST_CKPT=.cache/sharp_port242.safetensors ST_EQN=8 ST_PING=0 ST_JIT=0 .venv/bin/python3 scripts/step_trainer.py --eqbwd
env DEV=AMD ALG2=1 ALG_FTYPES=9 ALG_DUP=1 ALG_HW=512 ALG_WIDE=1 ALG_BREATH=7 ALG_NOTEBOOK=1 ALG_SIXWAVE=1 NB_PERSLOT=1 ALG_BINDBUS=7 ALG_BIND_D=512 BIND_CODES=.cache/bindbus_codes512.npz ALG_ALLOW_PEN_TRAIN=1 ALG_TRAIN=.cache/form_mix12.jsonl ALG_TRAIN_NAME=form12 ALG_BUSGARAGE=2 ALG_SHELF_CIRCLE=2 ALG_ALTMASK=1 SC_EVAL=0 ALG_ALT21=1 ALG_ALT2=1 ST_CKPT=.cache/sharp_port242.safetensors ST_EQN=8 ST_PING=0 ST_JIT=1 .venv/bin/python3 scripts/step_trainer.py --eqbwd

echo "== rung 2: 100-step smoke, pings LIVE — step-time + NaN guard + per-breath fact rate (expect the Blackbird profile: near-zero at breath 1, rising by 5-6; a lean early gauge is DESIGN, not defect) =="
env DEV=AMD ALG2=1 ALG_FTYPES=9 ALG_DUP=1 ALG_HW=512 ALG_WIDE=1 ALG_BREATH=7 ALG_NOTEBOOK=1 ALG_SIXWAVE=1 NB_PERSLOT=1 ALG_BINDBUS=7 ALG_BIND_D=512 BIND_CODES=.cache/bindbus_codes512.npz ALG_ALLOW_PEN_TRAIN=1 ALG_TRAIN=.cache/form_mix12.jsonl ALG_TRAIN_NAME=form12 ALG_BUSGARAGE=2 ALG_SHELF_CIRCLE=2 ALG_ALTMASK=1 SC_EVAL=0 ALG_ALT21=1 ALG_ALT2=1 ST_CKPT=.cache/sharp_port242.safetensors ST_STEPS=100 ST_LR=1e-4 ST_BATCH=32 ST_PING=1 ST_JIT=1 ST_LOG_EVERY=10 SEED=242 ST_OUT=.cache/step_smoke.safetensors .venv/bin/python3 scripts/step_trainer.py --train

echo "== rung 3: THE TWIN — 6k steps from the same warm source, single-bit delta = the ping schedule (per-step live facts vs frozen fact_0); gentle LR per the continuation law; snapshots for external selection =="
env DEV=AMD ALG2=1 ALG_FTYPES=9 ALG_DUP=1 ALG_HW=512 ALG_WIDE=1 ALG_BREATH=7 ALG_NOTEBOOK=1 ALG_SIXWAVE=1 NB_PERSLOT=1 ALG_BINDBUS=7 ALG_BIND_D=512 BIND_CODES=.cache/bindbus_codes512.npz ALG_ALLOW_PEN_TRAIN=1 ALG_TRAIN=.cache/form_mix12.jsonl ALG_TRAIN_NAME=form12 ALG_BUSGARAGE=2 ALG_SHELF_CIRCLE=2 ALG_ALTMASK=1 SC_EVAL=0 ALG_ALT21=1 ALG_ALT2=1 ST_CKPT=.cache/sharp_port242.safetensors ST_STEPS=6000 ST_LR=1e-4 ST_BATCH=32 ST_PING=1 ST_JIT=1 ST_SNAP_EVERY=2000 SEED=242 ST_OUT=.cache/step_twin_ping242.safetensors .venv/bin/python3 scripts/step_trainer.py --train
env DEV=AMD ALG2=1 ALG_FTYPES=9 ALG_DUP=1 ALG_HW=512 ALG_WIDE=1 ALG_BREATH=7 ALG_NOTEBOOK=1 ALG_SIXWAVE=1 NB_PERSLOT=1 ALG_BINDBUS=7 ALG_BIND_D=512 BIND_CODES=.cache/bindbus_codes512.npz ALG_ALLOW_PEN_TRAIN=1 ALG_TRAIN=.cache/form_mix12.jsonl ALG_TRAIN_NAME=form12 ALG_BUSGARAGE=2 ALG_SHELF_CIRCLE=2 ALG_ALTMASK=1 SC_EVAL=0 ALG_ALT21=1 ALG_ALT2=1 ST_CKPT=.cache/sharp_port242.safetensors ST_STEPS=6000 ST_LR=1e-4 ST_BATCH=32 ST_PING=0 ST_JIT=1 ST_SNAP_EVERY=2000 SEED=242 ST_OUT=.cache/step_twin_frozen242.safetensors .venv/bin/python3 scripts/step_trainer.py --train

echo "== rung 4: reads on both twins — engine (R=1/R=3, wild holdout + mint) + loop_val; env lines verbatim from the port-arm read discipline (reads carry the TRAINED env, always) =="
env DEV=AMD ALG2=1 ALG_FTYPES=9 ALG_DUP=1 ALG_HW=512 ALG_WIDE=1 ALG_BREATH=7 ALG_NOTEBOOK=1 ALG_SIXWAVE=1 NB_PERSLOT=1 ALG_BINDBUS=7 ALG_BIND_D=512 BIND_CODES=.cache/bindbus_codes512.npz ALG_BUSGARAGE=2 ALG_SHELF_CIRCLE=2 ALG_ALTMASK=1 SC_EVAL=0 ALG_ALT21=1 ALG_ALT2=1 ALG_TEST=.cache/wild_admitted_holdout.jsonl ALG_TEST_NAME=wildhold SE_CKPT=.cache/step_twin_ping242.safetensors SE_R=1 .venv/bin/python3 scripts/step_engine_read.py
env DEV=AMD ALG2=1 ALG_FTYPES=9 ALG_DUP=1 ALG_HW=512 ALG_WIDE=1 ALG_BREATH=7 ALG_NOTEBOOK=1 ALG_SIXWAVE=1 NB_PERSLOT=1 ALG_BINDBUS=7 ALG_BIND_D=512 BIND_CODES=.cache/bindbus_codes512.npz ALG_BUSGARAGE=2 ALG_SHELF_CIRCLE=2 ALG_ALTMASK=1 SC_EVAL=0 ALG_ALT21=1 ALG_ALT2=1 ALG_TEST=.cache/wild_admitted_holdout.jsonl ALG_TEST_NAME=wildhold SE_CKPT=.cache/step_twin_ping242.safetensors SE_R=3 .venv/bin/python3 scripts/step_engine_read.py
env DEV=AMD ALG2=1 ALG_FTYPES=9 ALG_DUP=1 ALG_HW=512 ALG_WIDE=1 ALG_BREATH=7 ALG_NOTEBOOK=1 ALG_SIXWAVE=1 NB_PERSLOT=1 ALG_BINDBUS=7 ALG_BIND_D=512 BIND_CODES=.cache/bindbus_codes512.npz ALG_BUSGARAGE=2 ALG_SHELF_CIRCLE=2 ALG_ALTMASK=1 SC_EVAL=0 ALG_ALT21=1 ALG_ALT2=1 ALG_TEST=.cache/wild_admitted_holdout.jsonl ALG_TEST_NAME=wildhold SE_CKPT=.cache/step_twin_frozen242.safetensors SE_R=1 .venv/bin/python3 scripts/step_engine_read.py
env DEV=AMD ALG2=1 ALG_FTYPES=9 ALG_DUP=1 ALG_HW=512 ALG_WIDE=1 ALG_BREATH=7 ALG_NOTEBOOK=1 ALG_SIXWAVE=1 NB_PERSLOT=1 ALG_BINDBUS=7 ALG_BIND_D=512 BIND_CODES=.cache/bindbus_codes512.npz ALG_BUSGARAGE=2 ALG_SHELF_CIRCLE=2 ALG_ALTMASK=1 SC_EVAL=0 ALG_ALT21=1 ALG_ALT2=1 ALG_TEST=.cache/wild_admitted_holdout.jsonl ALG_TEST_NAME=wildhold SE_CKPT=.cache/step_twin_frozen242.safetensors SE_R=3 .venv/bin/python3 scripts/step_engine_read.py
env DEV=AMD ALG2=1 ALG_FTYPES=9 ALG_DUP=1 ALG_HW=512 ALG_WIDE=1 ALG_BREATH=7 ALG_NOTEBOOK=1 ALG_SIXWAVE=1 NB_PERSLOT=1 ALG_BINDBUS=7 ALG_BIND_D=512 BIND_CODES=.cache/bindbus_codes512.npz ALG_BUSGARAGE=2 ALG_SHELF_CIRCLE=2 ALG_ALTMASK=1 SC_EVAL=0 ALG_ALT21=1 ALG_ALT2=1 ALG_TEST=.cache/wild_admitted_holdout.jsonl ALG_TEST_NAME=wildhold LV_CKPT=.cache/step_twin_ping242.safetensors .venv/bin/python3 scripts/loop_val.py
env DEV=AMD ALG2=1 ALG_FTYPES=9 ALG_DUP=1 ALG_HW=512 ALG_WIDE=1 ALG_BREATH=7 ALG_NOTEBOOK=1 ALG_SIXWAVE=1 NB_PERSLOT=1 ALG_BINDBUS=7 ALG_BIND_D=512 BIND_CODES=.cache/bindbus_codes512.npz ALG_BUSGARAGE=2 ALG_SHELF_CIRCLE=2 ALG_ALTMASK=1 SC_EVAL=0 ALG_ALT21=1 ALG_ALT2=1 ALG_TEST=.cache/wild_admitted_holdout.jsonl ALG_TEST_NAME=wildhold LV_CKPT=.cache/step_twin_frozen242.safetensors .venv/bin/python3 scripts/loop_val.py

echo "LADDER COMPLETE — 2-seed law still governs any promotion claim (seed 241 twin owed before a wild record is banked)"
