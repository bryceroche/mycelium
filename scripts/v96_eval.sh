#!/bin/bash
# v96 EVAL — eval_v77_dag.py wrapped with v96 env so the consolidation table fires.
#
# Pass CKPT=path/to/ckpt.safetensors as env var (or first arg).

set -e
cd "$(dirname "$0")/.."

if [ -n "$1" ]; then
    export CKPT="$1"
fi
if [ -z "$CKPT" ]; then
    echo "USAGE: $0 <path-to-ckpt> | CKPT=path $0"
    exit 1
fi
if [ ! -f "$CKPT" ]; then
    echo "ERROR: ckpt not found: $CKPT"
    exit 1
fi

export V77_TEST_PATH="${V77_TEST_PATH:-.cache/gsm8k_steps_v80_test.jsonl}"
export NUM_EVAL="${NUM_EVAL:-60}"
export K="${K:-7}"
export FIXED_LEN="${FIXED_LEN:-256}"
export BATCH="${BATCH:-4}"
export MAX_NEW="${MAX_NEW:-120}"
export GSM8K_STEPS_MIN_K=2
export GSM8K_STEPS_MAX_K=6

# v96 ON — table reading at the final breath.
export V96_CONSOLIDATION=1

# Match v96 training env (AR masking, no slot, no v95 aux).
export V77_DAG_TRAINING=1
export V77_N_LAYERS=7
export V85_QUERYABLE=0
export V79_CAUSAL_MASKS=1
export V81_MAIN_ATTN_MASK=0
export MULTI_HEAD_WAIST=0
export V82_PARALLEL_DIFFUSION=0
export V83_ANYTIME_SUPERVISION=0
export V91_SIMPLIFIED_ARGS=0

# v77b knobs
export BREATH_EMBED_ORTHO_INIT=2.0
export PER_BREATH_TEMP=1
export BREATH_NORM_OSC=1

# v78 — N=32 codebook (matches v96 training)
export MAX_STEP_BASE=2.0
export MAX_STEP_MIN=0.1
export NOTEBOOK_ACCUMULATE_ENABLED=1
export NOTEBOOK_NO_DETACH=1
export V78_HEAD_CODEBOOK=1
export V78_HEAD_CODEBOOK_N=32

export WAIST_ATTN_SUPERVISION=0
export V95_OPERAND_AUX=0
export CONTROLLER_N_LAYERS=4

# v66 architecture (matches v96 smoke)
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

export DEV=PCI+AMD
export USE_JIT=1
export USE_KV_CACHE=1
export SPACE_DIGITS=1

/home/bryce/mycelium/.venv/bin/python -u scripts/eval_v77_dag.py 2>&1
