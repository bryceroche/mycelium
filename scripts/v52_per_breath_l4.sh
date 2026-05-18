#!/bin/bash
# v52 Stage 1 — per-breath waist supervision on L4 (K=2 fixed).
#
# Drops outer cycles entirely. K-step problem trains with n_loops=K breaths.
# Each breath's end-of-breath output (post-waist, after L3 + bfield_waist) is
# decoded via ln_f + embed_out and supervised against THAT step's tokens.
#
# Architectural intent: enforce depth-helps by construction. A=1 cannot solve
# a 2-step problem because there's only 1 breath. K-step problem REQUIRES K
# breaths. The 92/92/88 (≈ A=1 ≈ A=8) plateau pattern of all prior runs
# should break here.
#
# Validation criteria at step 500:
#   PASS: breath 1 val CE < 0.5 on step-1 tokens AND breath 2 val CE < 0.5
#         on step-2 tokens → waist is encoding decodable intermediates.
#   PARTIAL: breath 1 OK but breath 2 fails → model collapsing to one breath;
#            REPLACE notebook needed (Stage 2).
#   FAIL: both > 1.5 → waist=512 too tight, try waist=1024 or no compression.
#
# Warm-start from v46b step 750.

set -e
cd "$(dirname "$0")/.."

# v52 Stage 1 architectural changes
export PER_BREATH_DECODE=1
export BFIELD_WAIST=512               # NEW: 2× compression (was 256 in v38)
export BFIELD_END_OF_BREATH=1         # waist at end of each breath
export BFIELD_ENFORCED=0              # residual mode, proj_up zero-init for warm-start safety
export BFIELD_ALPHA=1.0

# Codebook OFF for Stage 1 (Stage 3 adds it)
export WAIST_CODEBOOK_N=0
export WAIST_CODEBOOK_INJECT_WEIGHT=0

# Reg stack (validated)
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

# All other architectural extensions OFF
export QUADRATURE_HEADS=0
export ACROSS_LAYER_PITCH_TARGET=0
export PER_LAYER_OFFSETS_RADIANS=""

# Training — L4 only (K=2 uniform), n_loops sampling disabled (K determined by data)
export DEV=PCI+AMD
export LEVEL=L4
export SPACE_DIGITS=1
export BATCH=8                        # smaller — non-JIT path, plus longer reg path
export FIXED_LEN=96
export STEPS=500
export LR=3e-5
# TRAIN_LOOPS unused by per_breath_train_step (n_loops = K from data)
export TRAIN_LOOPS=2
export EVAL_LOOPS=1,2
export ACC_EVAL_EVERY=125
export CKPT_EVERY=125
export NUM_EVAL=100
export NUM_PROBLEMS=20000
export EVAL_BATCH=32
export EVAL_CACHE_LEN=136
export LOOKUP_AUX_WEIGHT=0.1
export USE_JIT=0                      # JIT not implemented for per_breath_train_step (Stage 1 MVP)
export CKPT_LABEL=v52_per_breath
export RESUME_FROM=.cache/l4_5_ckpts/v46b_control_l4_5_step750.safetensors

/home/bryce/mycelium/.venv/bin/python scripts/l3_train.py 2>&1 | tee .cache/v52_per_breath.log
