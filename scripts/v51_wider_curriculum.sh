#!/bin/bash
# v51 — WIDER mixed curriculum (4 levels) on top of validated v45 reg stack.
#
# Adds L4.7 (4-step word problems including rates, profit/loss, multi-person
# tracking, multi-purchase) to the v49 mix. L4.7 was BUILT in the codebase
# but never wired into the training-time dispatcher — adding it gives the
# model curriculum coverage closer to GSM8K's variety without needing real
# GSM8K data (which we use sparingly via mixed exposure).
#
# Pure curriculum experiment — NO architectural additions on top of v45.
# No B-field, no codebook, no quadrature. The hypothesis is the bottleneck
# was OP / DISTRIBUTION COVERAGE all along, and the v45 reg stack will do
# its job once the model sees a wider range of operations.
#
# Warm-start from v46b step 750 (L4.5 champion). Eval on GSM8K test set.

set -e
cd "$(dirname "$0")/.."

# v51 wider curriculum (the test)
export MIXED_LEVELS=L4_MIXED,L4.5,L4.7,GSM8K_SPACED

# v45 reg stack
export STOCH_DEPTH_P=0.10
export LABEL_SMOOTHING=0.1
export WEIGHT_DECAY=0.05

# v24c-era architecture
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

# All architectural extensions OFF (clean curriculum-only baseline)
export BFIELD_WAIST=0
export WAIST_CODEBOOK_N=0
export QUADRATURE_HEADS=0
export ACROSS_LAYER_PITCH_TARGET=0
export PER_LAYER_OFFSETS_RADIANS=""

# Training
export DEV=PCI+AMD
export LEVEL=GSM8K_SPACED
export SPACE_DIGITS=1
export BATCH=4
export FIXED_LEN=320              # max across levels (GSM8K longest)
export STEPS=3000
export LR=3e-5
export TRAIN_LOOPS=1,2,4
export EVAL_LOOPS=1,2,4
export ACC_EVAL_EVERY=500
export CKPT_EVERY=500
export NUM_EVAL=100
export NUM_PROBLEMS=20000
export EVAL_BATCH=16
export EVAL_CACHE_LEN=400
export LOOKUP_AUX_WEIGHT=0.1
export USE_JIT=1
export CKPT_LABEL=v51_wider
export RESUME_FROM=.cache/l4_5_ckpts/v46b_control_l4_5_step750.safetensors

/home/bryce/mycelium/.venv/bin/python scripts/l3_train.py 2>&1 | tee .cache/v51_wider_curriculum.log
