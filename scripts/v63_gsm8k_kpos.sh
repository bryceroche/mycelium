#!/bin/bash
# v63 — K-position embedding + prompt-dropout. Warm-start v60-take-2 step 6000.
#
# Architecture delta vs v60-take-2:
#   - k_pos_embed (max_loops, max_loops, waist_dim) added to waist before
#     WaistController reads it. Tells controller "you're at breath k of K total".
#     Zero-init → byte-identical forward at step 0 → warm-start safe.
#   - Prompt-dropout: every step, with probability CONTROLLER_PROMPT_DROPOUT,
#     zero the prompt embeddings passed to the WaistController. Forces controller
#     to be robust when prompt is missing — enables CFG at inference later.
#
# Hypothesis: K-pos fixes the segment-shortfall (76% of test data caps at ~2
# segments regardless of K=2..6). Prompt-dropout enables future CFG inference
# to reduce name confabulation (55% of failures).

set -e
cd "$(dirname "$0")/.."

if [ ! -f ".cache/gsm8k_steps_ckpts/v60_take2_gsm8k_steps_step6000.safetensors" ]; then
    echo "ERROR: v60-take-2 step 6000 ckpt not found."
    exit 1
fi

# v55..v60 architecture (carry forward; no DAG)
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
export ACROSS_LAYER_PITCH_TARGET=0
export PER_LAYER_OFFSETS_RADIANS=""

# v63 — prompt dropout for CFG-readiness
export CONTROLLER_PROMPT_DROPOUT=0.1   # 10% of steps zero the prompt

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
export CKPT_LABEL=v63_gsm8k_kpos
export RESUME_FROM=.cache/gsm8k_steps_ckpts/v60_take2_gsm8k_steps_step6000.safetensors

/home/bryce/mycelium/.venv/bin/python scripts/l3_train.py 2>&1 | tee .cache/v63_gsm8k_kpos.log
