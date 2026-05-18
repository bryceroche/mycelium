#!/bin/bash
# v45 take 3 → 200-step continuation. Warm-start from v45_reg_take3_step1000
# (the new project champion at 96/94/93 on L4_MIXED A=1/4/8).
#
# Same reg stack as take 3 — tests whether v45 keeps climbing past step 1000
# or starts to overfit. Eval at step 100 and 200.
set -e
cd "$(dirname "$0")/.."

# Reg stack (matches v45 take 3)
export STOCH_DEPTH_P=0.10
export LABEL_SMOOTHING=0.1
export WEIGHT_DECAY=0.05

# v24c-era architecture (matches the trained checkpoint)
export PER_HEAD_PITCH=1
export SINE_TEMP=1
export SINE_TEMP_MAX=2.0
export SINE_TEMP_MIN=0.7
export CONSTANT_RADIUS=1
export BREATH_TIME_EMBED=1
export BREATH_TIME_INIT_SCALE=0.0
export CROSS_BREATH_HANDOFF=1
export ABLATE_BREATH_ROTATION=1
export NOTEBOOK_V24=1
export NOTEBOOK_DUAL=1
export NOTEBOOK_POOL_MODE=attn
export NOTEBOOK_INIT_SCALE=0.02

# Training
export DEV=PCI+AMD
export LEVEL=L4_MIXED
export SPACE_DIGITS=1
export BATCH=16
export FIXED_LEN=96
export STEPS=200
export LR=3e-5
export TRAIN_LOOPS=1,2,4,8
export EVAL_LOOPS=1,4,8
export ACC_EVAL_EVERY=100
export CKPT_EVERY=100
export LOOKUP_AUX_WEIGHT=0.1
export USE_JIT=1
export CKPT_LABEL=v45_continue
export RESUME_FROM=.cache/l4_mixed_ckpts/v45_reg_take3_step1000.safetensors

/home/bryce/mycelium/.venv/bin/python scripts/l3_train.py 2>&1 | tee .cache/l4_mixed_v45_continue.log
