#!/bin/bash
# Run diagnostic dump on v60-take-2 step 6000 (our best baseline) to categorize failures.

set -e
cd "$(dirname "$0")/.."

# v60-take-2 architecture flags (NOTEBOOK_DAG=0 — original baseline)
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

export DEV=PCI+AMD
export LEVEL=GSM8K_STEPS
export GSM8K_STEPS_PATH=.cache/gsm8k_steps_v1_test.jsonl
export GSM8K_STEPS_MIN_K=2
export GSM8K_STEPS_MAX_K=6
export SPACE_DIGITS=1
export NUM_EVAL=100
export BATCH=2
export FIXED_LEN=400
export MAX_NEW=120
export USE_KV_CACHE=1
export CKPT=.cache/gsm8k_steps_ckpts/v60_take2_gsm8k_steps_step6000.safetensors
export DIAG_OUT=/tmp/v60_take2_diag.jsonl

/home/bryce/mycelium/.venv/bin/python scripts/diagnose_gsm8k_failures.py
