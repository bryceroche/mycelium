#!/bin/bash
# v65 — per-breath prompt refresh + #### boundary aux loss.
#
# Two targeted fixes for the two diagnosed dominant failure modes:
#   1. PROMPT_REFRESH_ALPHA=0.1 — skip connection from raw prompt_emb into every
#      breath's input. Fixes entity confabulation (rename diagnostic showed entity
#      info DESTROYED in waist compression — 0/20 grounded on renamed prompts).
#   2. BOUNDARY_AUX_WEIGHT=0.1 — BCE head per breath predicting "is next token
#      ####?". Fixes segment-shortfall (76% of failures: model emits ~2 segments
#      regardless of K=2..6).
#
# Warm-start: v60-take-2 step 6000 (2.0% on GSM8K). boundary_head missing in ckpt
# → default init. PROMPT_REFRESH affects forward at step 0 by α × prompt_emb
# (small perturbation). Both fixes co-exist with rest of v60-take-2 architecture.

set -e
cd "$(dirname "$0")/.."

if [ ! -f ".cache/gsm8k_steps_ckpts/v60_take2_gsm8k_steps_step6000.safetensors" ]; then
    echo "ERROR: v60-take-2 step 6000 ckpt not found."
    exit 1
fi

# v60-take-2 architecture (no DAG, no K-pos)
export NOTEBOOK_DAG=0
export CONTROLLER_DECODE=1
export CONTROLLER_N_LAYERS=2
export PER_BREATH_DECODE=1
export BFIELD_WAIST=512
export BFIELD_END_OF_BREATH=1
export BFIELD_ENFORCED=0
export BFIELD_ALPHA=1.0
export WAIST_CODEBOOK_N=64
export WAIST_CODEBOOK_INJECT_WEIGHT=1.0
export NOTEBOOK_V24=1
export NOTEBOOK_ACCUMULATE_ENABLED=0
export NOTEBOOK_DUAL=1
export NOTEBOOK_POOL_MODE=attn
export NOTEBOOK_INIT_SCALE=0.02
export STOCH_DEPTH_P=0.10
export LABEL_SMOOTHING=0.1
export WEIGHT_DECAY=0.05
export PER_HEAD_PITCH=1
export SINE_TEMP=1
export SINE_TEMP_MAX=2.0
export SINE_TEMP_MIN=0.7
export CONSTANT_RADIUS=1
export BREATH_TIME_EMBED=1
export BREATH_TIME_INIT_SCALE=0.0
export CROSS_BREATH_HANDOFF=1
export ABLATE_BREATH_ROTATION=1
export QUADRATURE_HEADS=0

# v65 — THE TWO FIXES
export PROMPT_REFRESH_ALPHA=0.1   # per-breath skip from raw prompt_emb
export BOUNDARY_AUX_WEIGHT=0.1    # BCE on #### prediction per breath

# Training
export DEV=PCI+AMD
export LEVEL=GSM8K_STEPS
export GSM8K_STEPS_PATH=.cache/gsm8k_steps_v1_train.jsonl
export GSM8K_STEPS_MIN_K=2
export GSM8K_STEPS_MAX_K=6
export SPACE_DIGITS=1
export BATCH=2
export FIXED_LEN=320
export STEPS=3000
export LR=3e-5
export TRAIN_LOOPS=4
export EVAL_LOOPS=1,2,3,4
export ACC_EVAL_EVERY=10000
export SKIP_FINAL_ACC=1
export CKPT_EVERY=500
export NUM_EVAL=100
export NUM_PROBLEMS=20000
export EVAL_BATCH=4
export EVAL_CACHE_LEN=328
export LOOKUP_AUX_WEIGHT=0.1
export USE_JIT=1
export USE_KV_CACHE=1
export CKPT_LABEL=v65_refresh_boundary
export RESUME_FROM=.cache/gsm8k_steps_ckpts/v60_take2_gsm8k_steps_step6000.safetensors

/home/bryce/mycelium/.venv/bin/python scripts/l3_train.py 2>&1 | tee .cache/v65_refresh_boundary.log
