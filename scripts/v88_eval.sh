#!/bin/bash
# v88 eval — eval_v85_dag.py on the v88 step 300 ckpt.
set -e
cd "$(dirname "$0")/.."

V88_CKPT="${V88_CKPT:-.cache/gsm8k_steps_ckpts/v88_smoke_step300.safetensors}"
NUM_EVAL="${NUM_EVAL:-60}"

# Same env needed for model construction (matches v88_smoke_train.sh).
export V85_QUERYABLE=1
export V85_K_MAX=10
export V85_N_MAX=20
export V85_TYPES_N=32
export V86_ARGS_CROSS_ATTN=1
export V86_ACTIVE_POS_WEIGHT=5.0
# (slot_pos_embed values are in the ckpt; init scale just informs eager
# fallback init. Setting the scale flag does no harm.)
export V87_SLOT_POS_INIT_SCALE=0.5
export V87_REINIT_SLOT_POS=0
export V88_REINIT_KV_PROJ=0
export V88_KV_PROJ_INIT_SCALE=0.02

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
export WAIST_ATTN_SUPERVISION=0
export WAIST_ATTN_AUX_WEIGHT=0.0

export V79_CAUSAL_MASKS=1
export SCHED_SAMPLE_RATE=0.0
export MULTI_HEAD_WAIST=0
export V81_MAIN_ATTN_MASK=1

export V82_PARALLEL_DIFFUSION=0
export V83_ANYTIME_SUPERVISION=0
export V83_GRADUATION=1

export BFIELD_WAIST_SCHEDULE="64,256,384,512,512"

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

export DEV=PCI+AMD
export CKPT="$V88_CKPT"
export K=5
export FIXED_LEN=256
export NUM_EVAL="$NUM_EVAL"
export GSM8K_STEPS_MIN_K=2
export GSM8K_STEPS_MAX_K=6
export V77_TEST_PATH=".cache/gsm8k_steps_v85_train.jsonl"

/home/bryce/mycelium/.venv/bin/python -u scripts/eval_v85_dag.py 2>&1 | tee .cache/v88_smoke_eval.log
