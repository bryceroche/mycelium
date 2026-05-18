#!/bin/bash
# v46 — HYBRID-HEADS QUADRATURE. Warm-start from v45 take 3 step 1000
# (96/94/93 on L4_MIXED). Same reg stack + add QUADRATURE_HEADS=1.
#
# Quadrature splits the 16 heads in each layer:
#   heads 0..7    keep PER_HEAD_PITCH offset l*π/64 (the "I" channel)
#   heads 8..15   add π/2 → offset l*π/64 + π/2     (the "Q" channel)
#
# Geometric idea: when "I heads" are at rotation angle θ, "Q heads" are at
# θ + π/2 — quadrature pair forms a constant-magnitude rotating field
# (circular polarization analogy) instead of linear oscillation that goes
# through zero-crossings. Cheapest possible test of the photon analogy:
# zero added params, zero added compute, just a different angular geometry.
#
# Decision criteria (step 125 / 250 eval):
#   - acc on A=8 matches or beats v45's 93 → quadrature is working; consider
#     dual-stack version (item §8 #2/#10/#15 of CLAUDE.md).
#   - acc drops 5+ pts → quadrature destabilizes; abandon this direction.
#   - neutral → maybe needs more training; try longer warm-start.
set -e
cd "$(dirname "$0")/.."

# v46 architecture change (the test)
export QUADRATURE_HEADS=1

# Inherit v45's validated reg stack
export STOCH_DEPTH_P=0.10
export LABEL_SMOOTHING=0.1
export WEIGHT_DECAY=0.05

# v24c-era architecture (matches the warm-start ckpt)
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
export STEPS=500
export LR=3e-5
export TRAIN_LOOPS=1,2,4,8
export EVAL_LOOPS=1,4,8
export ACC_EVAL_EVERY=125
export CKPT_EVERY=125
export LOOKUP_AUX_WEIGHT=0.1
export USE_JIT=1
export CKPT_LABEL=v46_quadrature
export RESUME_FROM=.cache/l4_mixed_ckpts/v45_reg_take3_step1000.safetensors

/home/bryce/mycelium/.venv/bin/python scripts/l3_train.py 2>&1 | tee .cache/l4_mixed_v46_quadrature.log
