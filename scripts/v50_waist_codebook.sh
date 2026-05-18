#!/bin/bash
# v50 — LEARNABLE CODEBOOK at the IB waist (item #1/#4 MVP).
#
# Architecture: B-field IB bottleneck (1024 → 256 → 1024) between layers L1
# and L2, augmented with N=256 learnable (key, value) pairs at the compressed
# state. After compression, the model queries the codebook via dot-product
# attention and adds a weighted-sum of values to the compressed state before
# GELU. Values init zero (graceful warm-start); gradient shapes both keys
# and values via the main CE.
#
# Hypothesis: giving the model 256 discrete attractor entries at its bottleneck
# helps it commit to op-specific representations. Particularly relevant for
# GSM8K where the model needs to identify operations (add/sub/mul/div/percent/
# ratio/etc) and dispatch to them — instead of compressing to a continuous 256d.
#
# Mixed curriculum (L4_MIXED + L4.5 + GSM8K_SPACED) — same as v49.
# Warm-start from v46b step 750 (L4.5 champion). bfield_proj_up is zero-init,
# so the B-field waist starts as identity (matches v46b exactly at step 0).
# Codebook values are zero-init for the same reason.

set -e
cd "$(dirname "$0")/.."

# v50 architectural additions
export BFIELD_WAIST=256                 # enable IB bottleneck (1024 → 256 → 1024)
export BFIELD_ENFORCED=0                # residual mode (zero-init proj_up + alpha=1 means identity at step 0)
export BFIELD_ALPHA=1.0
export BFIELD_END_OF_BREATH=0           # waist between L1-L2 (v38's tested placement)
export WAIST_CODEBOOK_N=256             # N learnable codebook entries
export WAIST_CODEBOOK_INJECT_WEIGHT=1.0

# Mixed curriculum (same as v49)
export MIXED_LEVELS=L4_MIXED,L4.5,GSM8K_SPACED

# Reg stack (validated)
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

# Quadrature off
export QUADRATURE_HEADS=0
export ACROSS_LAYER_PITCH_TARGET=0
export PER_LAYER_OFFSETS_RADIANS=""

# Training
export DEV=PCI+AMD
export LEVEL=GSM8K_SPACED
export SPACE_DIGITS=1
export BATCH=4
export FIXED_LEN=320
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
export CKPT_LABEL=v50_codebook_take2
export RESUME_FROM=.cache/l4_5_ckpts/v46b_control_l4_5_step750.safetensors

# Take 2: codebook values changed from zero-init to random-small (0.02). Take 1
# matched v49 exactly because zero-init values had ~3e-7 per-step gradient
# growth — bootstrap too slow to differentiate from v49 in ~3000 steps.
/home/bryce/mycelium/.venv/bin/python scripts/l3_train.py 2>&1 | tee .cache/v50_waist_codebook_take2.log
