#!/bin/bash
# v85 SMOKE — differentiable queryable structures.
#
# v85 is the strategic pivot after v82-v84 hit 23% parse / 0% accuracy. The
# blocker was OPERAND BINDING: the model didn't bind args to actual prompt
# numbers. v85 fixes this by making the prompt's numbers, verbs, and our ops/
# types into queryable differentiable tensors that attention heads natively
# query.
#
# Architecture:
#   - K_max=10 DAG slots, each with: ops (4-way), types (32-way), 2 args
#     (pointer over N_max + K_max), is_active (sigmoid)
#   - Slots fill in PARALLEL per breath (not AR)
#   - All breaths emit the SAME structure (no per-breath specialization)
#   - Soft commit: argmax only at final inference for SymPy execution
#
# Data: .cache/gsm8k_steps_v85_train.jsonl (Haiku-annotated structured records).
# Warm-start: v83_smoke_grad_step400 (the 20% parse ckpt) — many params are
# NEW (slot decoder, codebooks); they start from init.

set -e
cd "$(dirname "$0")/.."

V85_INIT="${V85_INIT:-.cache/gsm8k_steps_ckpts/v83_smoke_grad_step400.safetensors}"
if [ ! -f "$V85_INIT" ]; then
    echo "ERROR: warm-start ckpt not found at $V85_INIT"
    exit 1
fi
V85_DATA="${V85_DATA:-.cache/gsm8k_steps_v85_train.jsonl}"
if [ ! -f "$V85_DATA" ]; then
    echo "ERROR: v85 train data not found at $V85_DATA"
    exit 1
fi

# ---- V77 path off (v85 has its own loader + train step) ----
export V77_DAG_TRAINING=0

# ---- v85 knobs ----
export V85_QUERYABLE=1
export V85_K_MAX=10
export V85_N_MAX=20
export V85_TYPES_N=32

# ---- v77b knobs ----
export BREATH_EMBED_ORTHO_INIT=2.0
export PER_BREATH_TEMP=1
export BREATH_NORM_OSC=1

# ---- v78 knobs ----
export MAX_STEP_BASE=2.0
export MAX_STEP_MIN=0.1
export NOTEBOOK_ACCUMULATE_ENABLED=1
export NOTEBOOK_NO_DETACH=1
export V78_HEAD_CODEBOOK=1
export V78_HEAD_CODEBOOK_N=32

# ---- v78b knobs (waist attn supervision NOT used in v85; keep off) ----
export CONTROLLER_N_LAYERS=4
export WAIST_ATTN_SUPERVISION=0
export WAIST_ATTN_AUX_WEIGHT=0.0

# ---- v79/v81 knobs (4 masks required for clean train/eval) ----
export V79_CAUSAL_MASKS=1
export SCHED_SAMPLE_RATE=0.0
export MULTI_HEAD_WAIST=0
export V81_MAIN_ATTN_MASK=1

# ---- v82/v83 knobs (v85 does NOT need these; bypass) ----
export V82_PARALLEL_DIFFUSION=0
export V83_ANYTIME_SUPERVISION=0
export V83_GRADUATION=0   # v85 uses uniform per-breath weighting (same task per breath)

# ---- v83 waist schedule (KEEP — load-bearing) ----
export BFIELD_WAIST_SCHEDULE="${BFIELD_WAIST_SCHEDULE:-64,256,384,512,512}"

# ---- v66 architecture (inherited verbatim) ----
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
export BOUNDARY_POS_WEIGHT=5.0
export PER_BREATH_FULL_ANSWER=0

# ---- LR decay ----
export LR_DECAY_TO_ZERO=1

# ---- Training config — SMOKE (500 steps, K=5 breaths) ----
export DEV=PCI+AMD
export LEVEL=GSM8K_STEPS
export GSM8K_STEPS_PATH="$V85_DATA"
export V77_TEST_PATH="$V85_DATA"
export GSM8K_STEPS_MIN_K=2
export GSM8K_STEPS_MAX_K=6
export SPACE_DIGITS=1
export BATCH="${BATCH:-4}"
export FIXED_LEN="${FIXED_LEN:-224}"
export STEPS="${STEPS:-500}"
export LR=3e-5
# K=5 breaths — all breaths emit the same target structure (coarse-to-fine
# refinement of the slot mixtures).
export TRAIN_LOOPS="${TRAIN_LOOPS:-5}"
export EVAL_LOOPS="${EVAL_LOOPS:-5}"
export V77_N_LAYERS="${V77_N_LAYERS:-5}"
export ACC_EVAL_EVERY=10000
export SKIP_FINAL_ACC=1
export CKPT_EVERY="${CKPT_EVERY:-50}"
export NUM_EVAL=20
export NUM_PROBLEMS=20000
export EVAL_BATCH=4
export EVAL_CACHE_LEN="${EVAL_CACHE_LEN:-232}"
export LOOKUP_AUX_WEIGHT=0.0
export USE_JIT=1
export USE_KV_CACHE=1
export CKPT_LABEL="${CKPT_LABEL:-v85_smoke}"
export RESUME_FROM="$V85_INIT"

/home/bryce/mycelium/.venv/bin/python -u scripts/l3_train.py 2>&1 | tee .cache/v85_smoke_train.log
