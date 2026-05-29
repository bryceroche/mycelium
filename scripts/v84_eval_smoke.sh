#!/bin/bash
# v84 smoke eval — DAG parse rate + accuracy at step 500.
#
# Run this AFTER scripts/v84_smoke_train.sh completes. Uses the same env
# config the training had so the model + masks match.
set -e
cd "$(dirname "$0")/.."

CKPT="${CKPT:-.cache/gsm8k_steps_ckpts/v84_smoke_step500.safetensors}"
if [ ! -f "$CKPT" ]; then
    echo "ERROR: ckpt not found at $CKPT — did smoke run finish?"
    exit 1
fi

# Match the v84 smoke architecture flags so the model loads + runs the same
# forward as training. (eval_v84_dag.py uses these via import-time env reads.)
export V77_DAG_TRAINING=1
export BREATH_EMBED_ORTHO_INIT=2.0
export PER_BREATH_TEMP=1
export BREATH_NORM_OSC=1
export MAX_STEP_BASE=2.0
export MAX_STEP_MIN=0.1
export NOTEBOOK_ACCUMULATE_ENABLED=1
export NOTEBOOK_NO_DETACH=1
export V78_HEAD_CODEBOOK=1
export V78_HEAD_CODEBOOK_N=32
export CONTROLLER_N_LAYERS=4
export WAIST_ATTN_SUPERVISION=1
export WAIST_ATTN_AUX_WEIGHT=0.5
export V79_CAUSAL_MASKS=1
export SCHED_SAMPLE_RATE=0.0
export MULTI_HEAD_WAIST=0
export V81_MAIN_ATTN_MASK=1
export V82_PARALLEL_DIFFUSION=1
export BFIELD_WAIST_SCHEDULE="64,256,384,512,512"
export V83_ANYTIME_SUPERVISION=1
export V83_GRADUATION=1
export NOTEBOOK_DAG=0
export CONTROLLER_DECODE=1
export PER_BREATH_DECODE=1
export BFIELD_WAIST=512
export BFIELD_END_OF_BREATH=1
export BFIELD_ENFORCED=0
export BFIELD_ALPHA=1.0
export WAIST_CODEBOOK_N=64
export WAIST_CODEBOOK_INJECT_WEIGHT=1.0
export NOTEBOOK_V24=1
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
export PROMPT_REFRESH_ALPHA=0.1
export BOUNDARY_AUX_WEIGHT=0.0
export PER_BREATH_FULL_ANSWER=0
export V77_N_LAYERS=5

export DEV=PCI+AMD
export CKPT="$CKPT"
export V77_TEST_PATH="${V77_TEST_PATH:-.cache/gsm8k_steps_v84_test.jsonl}"
export NUM_EVAL="${NUM_EVAL:-60}"
export K="${K:-5}"
export FIXED_LEN="${FIXED_LEN:-256}"
export BATCH="${BATCH:-4}"
export MAX_NEW="${MAX_NEW:-80}"
export GSM8K_STEPS_MIN_K=2
export GSM8K_STEPS_MAX_K=6

echo "=== v84 DAG eval — ckpt $CKPT ==="
/home/bryce/mycelium/.venv/bin/python scripts/eval_v84_dag.py 2>&1 | tee .cache/v84_smoke_eval.log
