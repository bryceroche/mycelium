#!/bin/bash
# v55 — v54 + waist codebook ON.
#
# Same architecture as v54 (K=2 inner breaths, REPLACE notebook, WaistController
# decoding at the waist) PLUS the 64-entry codebook injected at the waist before
# the controller reads it. Tests whether op-discriminative structure at the
# waist (per-(head, op)) helps the controller decode partial answers better.
#
# Codebook init: keys randn × 0.02, values ZERO (graceful — zero contribution
# at step 0, gradient grows them). Same regime as v53.
#
# Warm-start from v46b step 750. L4 only (K=2 uniform). 500 steps.

set -e
cd "$(dirname "$0")/.."

# v54 architecture
export CONTROLLER_DECODE=1
export CONTROLLER_N_LAYERS=1
export PER_BREATH_DECODE=1
export BFIELD_WAIST=512
export BFIELD_END_OF_BREATH=1
export BFIELD_ENFORCED=0
export BFIELD_ALPHA=1.0

# v55 addition: waist codebook ON
export WAIST_CODEBOOK_N=64                  # 16 heads × 4 ops
export WAIST_CODEBOOK_INJECT_WEIGHT=1.0

# REPLACE-only notebook
export NOTEBOOK_V24=1
export NOTEBOOK_ACCUMULATE_ENABLED=0
export NOTEBOOK_DUAL=1
export NOTEBOOK_POOL_MODE=attn
export NOTEBOOK_INIT_SCALE=0.02

# Reg stack
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

# Everything else off
export QUADRATURE_HEADS=0
export ACROSS_LAYER_PITCH_TARGET=0
export PER_LAYER_OFFSETS_RADIANS=""

# Training
export DEV=PCI+AMD
export LEVEL=L4
export SPACE_DIGITS=1
export BATCH=8
export FIXED_LEN=96
export STEPS=500
export LR=3e-5
export TRAIN_LOOPS=2
export EVAL_LOOPS=1,2
export ACC_EVAL_EVERY=125
export CKPT_EVERY=125
export NUM_EVAL=100
export NUM_PROBLEMS=20000
export EVAL_BATCH=32
export EVAL_CACHE_LEN=136
export LOOKUP_AUX_WEIGHT=0.1
export USE_JIT=0
export CKPT_LABEL=v55_controller_codebook
export RESUME_FROM=.cache/l4_5_ckpts/v46b_control_l4_5_step750.safetensors

/home/bryce/mycelium/.venv/bin/python scripts/l3_train.py 2>&1 | tee .cache/v55_controller_codebook.log
