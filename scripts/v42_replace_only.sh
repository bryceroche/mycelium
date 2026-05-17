#!/bin/bash
# v42 — Test REPLACE-only notebook (no accumulate).
#
# v24c achieved 96/94/91 on L4_MIXED with DUAL notebook (REPLACE + ACCUMULATE).
# The original ablation suggested REPLACE was the load-bearing path; this run
# tests that directly by continuing v24c training with ACCUMULATE disabled.
#
# Hypothesis: if REPLACE alone preserves the 96% accuracy, the ACCUMULATE notebook
# was decorative and we can simplify the architecture. If accuracy degrades, both
# notebooks contribute distinct memory channels.
#
# Setup: warm-start from v24c step 500, continue training on L4_MIXED for 500
# more steps. All other env vars match v24c training exactly.
set -e
cd "$(dirname "$0")/.."

# v24c training geometry (must match for warm-start coherence)
export SINE_TEMP=1
export SINE_TEMP_MAX=2.0
export SINE_TEMP_MIN=0.7
export ABLATE_BREATH_ROTATION=1
export BREATH_TIME_EMBED=1
export BREATH_TIME_INIT_SCALE=0.0
export CROSS_BREATH_HANDOFF=1
export CONSTANT_RADIUS=1
export PER_HEAD_PITCH=1

# REPLACE-only notebook (the test variable)
export NOTEBOOK_V24=1
export NOTEBOOK_DUAL=1
export NOTEBOOK_POOL_MODE=attn
export NOTEBOOK_INIT_SCALE=0.02
export NOTEBOOK_ACCUMULATE_ENABLED=0   # ← the change

# v24c did NOT have LOOKUP_VALUE_INJECT (predates v28 prototype retrieval)
export LOOKUP_VALUE_INJECT=0

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
export CKPT_LABEL=v42_replace_only
export RESUME_FROM=/home/bryce/mycelium/.cache/l4_mixed_ckpts/l4_mixed_v24c_dual_notebook_step500.safetensors

/home/bryce/mycelium/.venv/bin/python scripts/l3_train.py
