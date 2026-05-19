#!/bin/bash
# v53 — Stage 2: REPLACE notebook + waist codebook + per-breath supervision.
#
# K-step problem trains with n_loops=K inner breaths (no outer cycles). The
# REPLACE notebook breaks the symmetry that v52 Stage 1 hit: breath k+1 reads
# breath k's COMMITTED waist state via the notebook, so the breaths must
# differentiate. Per-breath supervision at the waist (via ln_f + embed_out)
# trains each breath to produce decodable step-k content.
#
# Architecture:
#   - 1 forward pass through 4 layers per breath
#   - BFIELD_WAIST=512 at end of each breath (after L3)
#   - 64-entry learnable codebook injected at waist (keys=0.02 random,
#     values=ZERO — bootstrap from zero contribution, graceful warm-start)
#   - REPLACE notebook (NOTEBOOK_V24=1 + ACCUMULATE_OFF + DUAL=1): breath k
#     writes its committed waist state; breath k+1 reads it as input context
#   - Per-breath supervision via PER_BREATH_DECODE=1: each breath decoded
#     via ln_f → embed_out, CE against step-k tokens
#
# Validation criterion at step 250:
#   PASS: pb_ce[0] and pb_ce[1] DIFFERENTIATE significantly (e.g. > 0.2 apart)
#         → REPLACE notebook is breaking the symmetry, breaths specialize.
#   FAIL: pb_ce[0] ≈ pb_ce[1] → REPLACE alone isn't enough; need more
#         architectural force or different supervision.
#
# Eval is INFORMATIONAL only (current cached_generate_batch reads integrated
# rep, which is incoherent under per-breath specialization — Stage 3 fixes
# this with per-breath / final-waist decode at inference).
#
# Warm-start from v46b step 750. L4 only (K=2 uniform). 500 steps.

set -e
cd "$(dirname "$0")/.."

# v53 Stage 2 architectural additions
export PER_BREATH_DECODE=1
export BFIELD_WAIST=512
export BFIELD_END_OF_BREATH=1
export BFIELD_ENFORCED=0
export BFIELD_ALPHA=1.0

# Codebook ON, N=64 (conceptually 16 heads × 4 ops), flat structure
export WAIST_CODEBOOK_N=64
export WAIST_CODEBOOK_INJECT_WEIGHT=1.0

# REPLACE notebook ONLY (no accumulate)
export NOTEBOOK_V24=1
export NOTEBOOK_ACCUMULATE_ENABLED=0
export NOTEBOOK_DUAL=1
export NOTEBOOK_POOL_MODE=attn
export NOTEBOOK_INIT_SCALE=0.02

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

# All other architectural extensions OFF
export QUADRATURE_HEADS=0
export ACROSS_LAYER_PITCH_TARGET=0
export PER_LAYER_OFFSETS_RADIANS=""

# Training — L4 only (K=2 uniform)
export DEV=PCI+AMD
export LEVEL=L4
export SPACE_DIGITS=1
export BATCH=8
export FIXED_LEN=96
export STEPS=500
export LR=3e-5
export TRAIN_LOOPS=2                  # K=2 breaths (unused by per_breath_train_step; just for logging)
export EVAL_LOOPS=1,2
export ACC_EVAL_EVERY=125
export CKPT_EVERY=125
export NUM_EVAL=100
export NUM_PROBLEMS=20000
export EVAL_BATCH=32
export EVAL_CACHE_LEN=136
export LOOKUP_AUX_WEIGHT=0.1
export USE_JIT=0                       # per_breath_train_step is non-JIT (Stage 2 MVP)
export CKPT_LABEL=v53_replace_codebook
export RESUME_FROM=.cache/l4_5_ckpts/v46b_control_l4_5_step750.safetensors

/home/bryce/mycelium/.venv/bin/python scripts/l3_train.py 2>&1 | tee .cache/v53_replace_waist_codebook.log
