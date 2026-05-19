#!/bin/bash
# v57 — WaistController + codebook on L4.7 (K=4 inner breaths).
#
# Scaling test for the rep-space-thinking paradigm. Question: does the
# segmented–aligned gap keep growing with K? Data so far:
#   K=2 (v55, L4):   aligned 89, segmented 90  →  +1
#   K=3 (v56, L4.5): aligned 75, segmented 89  →  +14
# If K=4 lands at +20 or more, the K-axis scales with complexity → green
# light for Haiku-curated GSM8K step targets. If K=4 plateaus at ~+14, we've
# hit the WaistController capacity ceiling and should widen it first
# (CONTROLLER_N_LAYERS=2 or wider hidden) before going to GSM8K.
#
# L4.7 = 4-step word problems (linear 4-op, buy/sell/profit, multi-person,
# rate problems). Uniform K=4 per problem.
#
# Warm-start from v56 step 500. JIT'd per-breath path (USE_JIT=1).
# 500 steps. Aligned + segmented evals after training.

set -e
cd "$(dirname "$0")/.."

# v55/v56 architecture (carry forward)
export CONTROLLER_DECODE=1
export CONTROLLER_N_LAYERS=1
export PER_BREATH_DECODE=1
export BFIELD_WAIST=512
export BFIELD_END_OF_BREATH=1
export BFIELD_ENFORCED=0
export BFIELD_ALPHA=1.0
export WAIST_CODEBOOK_N=64
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

# Training — L4.7 only (K=4 uniform)
export DEV=PCI+AMD
export LEVEL=L4.7
export SPACE_DIGITS=1
export BATCH=2                              # K=4 + FIXED_LEN=200; B=4 OOM'd (memviol)
export FIXED_LEN=200                        # matches DEFAULT_FIXED_LEN["L4.7"]
export STEPS=500
export LR=3e-5
export TRAIN_LOOPS=4
export EVAL_LOOPS=1,2,3,4
export ACC_EVAL_EVERY=125
export CKPT_EVERY=125
export NUM_EVAL=100
export NUM_PROBLEMS=20000
export EVAL_BATCH=4                          # paired tighter for FIXED_LEN=200
export EVAL_CACHE_LEN=208
export LOOKUP_AUX_WEIGHT=0.1
export USE_JIT=1                             # JIT'd per_breath path (built today)
export CKPT_LABEL=v57_controller_codebook_l4_7
export RESUME_FROM=.cache/l4_5_ckpts/v56_controller_codebook_l4_5_step500.safetensors

/home/bryce/mycelium/.venv/bin/python scripts/l3_train.py 2>&1 | tee .cache/v57_controller_codebook_l4_7.log
