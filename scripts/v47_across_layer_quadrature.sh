#!/bin/bash
# v47 — ACROSS-LAYER QUADRATURE on L4.5.
#
# Per-layer pitch step ramped from v23a base (π/64 ≈ 2.8°) to target π/2 (90°)
# over 500 training steps. With n_phases=4 and target π/2, the layers settle
# at offsets {0, π/2, π, 3π/2} — full circle spread across the 4 breath layers.
# All 16 heads WITHIN each layer keep the same offset, so W_O isn't broken
# (lesson from v46 take 1 where within-layer mismatch caused -35 collapse).
#
# Phase-shift sweep on v46b step 750 showed:
# - Uniform shifts up to π/4 cost <3 points (model has phase tolerance)
# - Within-layer shift π/2 catastrophic (-35)
# - Cross-layer π/2 untested; this run probes that gap
#
# Warm-start from v46b control step 750 (L4.5 champion, 92/92/88). Reg stack
# inherited from v45 (validated). If quadrature lifts depth-spread further
# (target: A=8 past 88), it confirms cross-layer phase diversity is useful.
# If it collapses, we know the model can't bridge a π/2 phase gap between
# adjacent layers even with ramp — quadrature is fundamentally incompatible
# with iterative breathing geometry.

set -e
cd "$(dirname "$0")/.."

# v47 architectural change (the test)
export ACROSS_LAYER_PITCH_TARGET=1.5707963   # π/2
export ACROSS_LAYER_PITCH_RAMP_STEPS=500

# Make sure within-layer quadrature is OFF
export QUADRATURE_HEADS=0
export QUADRATURE_RAMP_STEPS=0

# Reg stack inherited from v45
export STOCH_DEPTH_P=0.10
export LABEL_SMOOTHING=0.1
export WEIGHT_DECAY=0.05

# v24c-era architecture (matches v46b warm-start)
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

# Training — L4.5, warm-start from v46b control step 750
export DEV=PCI+AMD
export LEVEL=L4.5
export SPACE_DIGITS=1
export BATCH=8
export FIXED_LEN=160
export STEPS=1000
export LR=3e-5
export TRAIN_LOOPS=1,2,4,8
export EVAL_LOOPS=1,4,8
export ACC_EVAL_EVERY=250
export CKPT_EVERY=250
export LOOKUP_AUX_WEIGHT=0.1
export USE_JIT=1
export CKPT_LABEL=v47_across_layer
export RESUME_FROM=.cache/l4_5_ckpts/v46b_control_l4_5_step750.safetensors

/home/bryce/mycelium/.venv/bin/python scripts/l3_train.py 2>&1 | tee .cache/l4_5_v47_across_layer.log
