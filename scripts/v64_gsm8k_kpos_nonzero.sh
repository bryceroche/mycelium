#!/bin/bash
# v64 — K-position embedding with NON-ZERO init (vs v63's zero init).
#
# v63 diagnosis: zero-init k_pos_embed had a dead-bootstrap problem.
# Mean loss diff across 2300 (step, K) pairs vs v61 was +0.0025 — pure noise.
# The embed contributes 0 when value is 0, so gradient is 0, so it stays 0.
#
# v64 fix: K_POS_INIT_SCALE=0.02 — randn × 0.02 gives non-zero starting point,
# loss now depends on k_pos_embed, gradient flows. Slight perturbation from
# v60-take-2 warm-start at step 0 (small relative to typical waist magnitudes).

set -e
cd "$(dirname "$0")/.."

if [ ! -f ".cache/gsm8k_steps_ckpts/v60_take2_gsm8k_steps_step6000.safetensors" ]; then
    echo "ERROR: v60-take-2 step 6000 ckpt not found."
    exit 1
fi

# Same as v63 except K_POS_INIT_SCALE
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

# v64 — non-zero K-pos init (the key change vs v63)
export K_POS_INIT_SCALE=0.02
export CONTROLLER_PROMPT_DROPOUT=0.1

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
export CKPT_LABEL=v64_kpos_nonzero
export RESUME_FROM=.cache/gsm8k_steps_ckpts/v60_take2_gsm8k_steps_step6000.safetensors

/home/bryce/mycelium/.venv/bin/python scripts/l3_train.py 2>&1 | tee .cache/v64_kpos_nonzero.log
