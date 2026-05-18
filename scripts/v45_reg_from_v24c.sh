#!/bin/bash
# v45 — REGULARIZATION pass on top of v24c (the 96/94/91 champion).
#
# Goal: validate that the new regularization mechanisms lift the v24c warm-start
# past its overfitting wall at step 1000. v24c peaks at step 500 and regresses
# by step 1000 — if reg works, step 750/1000 should match-or-beat step 500.
#
# Same architecture as v24c (no DOUBLED_LAYERS, no MIXED_LEVELS) — those are
# v44's experiments and confound the regularization signal. Stack them only
# after v45 confirms reg lifts.
#
# Three regularization knobs added in 2026-05-17 pass:
#   STOCH_DEPTH_P=0.15  — drop ~1.2 of 8 breaths per training step (ResNet-style
#                         scaling: kept breaths weighted 1/(1-p), output is an
#                         unbiased estimator of the no-drop mean).
#   LABEL_SMOOTHING=0.1 — applied to the main answer-CE (training only; eval CE
#                         stays unsmoothed for comparability).
#   WEIGHT_DECAY=0.05   — bumped from 0.01 default. AdamW typical range for
#                         this scale.
set -e
cd "$(dirname "$0")/.."

# v45 regularization knobs (the test)
# STOCH_DEPTH_P=0.10 (down from 0.15 after take-1 collapse — see below).
# At p=0.15 over-scaling factor 1/(1-p)=1.176 was meaningful distortion; 0.10
# gives 1.11, much milder. l3_train.py now also guarantees >=1 active breath
# kept per step (no catastrophic all-dropped events) and skips SD at n_loops=1.
export STOCH_DEPTH_P=0.10
export LABEL_SMOOTHING=0.1
export WEIGHT_DECAY=0.05

# Match v24c architecture exactly (no DOUBLED_LAYERS / MIXED_LEVELS — those
# are v44's experiments and would confound this run).
export PER_HEAD_PITCH=1
export SINE_TEMP=1
export SINE_TEMP_MAX=2.0
export SINE_TEMP_MIN=0.7
export CONSTANT_RADIUS=1
export BREATH_TIME_EMBED=1
export BREATH_TIME_INIT_SCALE=0.0
export CROSS_BREATH_HANDOFF=1
export ABLATE_BREATH_ROTATION=1

# v24c dual notebook
export NOTEBOOK_V24=1
export NOTEBOOK_DUAL=1
export NOTEBOOK_POOL_MODE=attn
export NOTEBOOK_INIT_SCALE=0.02

# Training — warm-start from the v24c step-500 champion
export DEV=PCI+AMD
export LEVEL=L4_MIXED
export SPACE_DIGITS=1
export BATCH=16
export FIXED_LEN=96
export STEPS=1000              # 500 more steps from v24c step 500
export LR=3e-5
export TRAIN_LOOPS=1,2,4,8
export EVAL_LOOPS=1,4,8
export ACC_EVAL_EVERY=125      # eval every 125 steps → 8 eval points to catch overfitting onset
export CKPT_EVERY=125
export LOOKUP_AUX_WEIGHT=0.1
export USE_JIT=1
export CKPT_LABEL=v45_reg_take3
export RESUME_FROM=.cache/l4_mixed_ckpts/l4_mixed_v24c_dual_notebook_step500.safetensors

# Take 3: launched after the STAGE2_NOTEBOOK inference-path bug fix. Takes 1
# and 2 collapsed because cached_generate_batch's Stage 2 was updating the
# notebook per generated token — train/eval mismatch — putting the notebook
# OOD by ~240 extra updates per problem. Even the v24c step-500 ckpt with
# zero training gave 0% acc + garbage output. With STAGE2_NOTEBOOK=0 default,
# v24c step 500 evaluates to 96/94/91 (matches its training-time numbers).
# This take is the actual test of whether the reg stack lifts the model.

/home/bryce/mycelium/.venv/bin/python scripts/l3_train.py 2>&1 | tee .cache/l4_mixed_v45_reg_take3.log
