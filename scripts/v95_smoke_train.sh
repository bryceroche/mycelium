#!/bin/bash
# v95 SMOKE — operand-position attention supervision on top of v80 AR paradigm.
#
# Hypothesis: v80_prod_step400 hits 78% DAG parse / 1.7% accuracy. Ceiling is
# CONTENT BINDING (model picks wrong operands; emits undefined variables).
# v95 adds a direct supervision signal: at each AR output position emitting a
# multi-digit number token (e.g. `Ġ50` in `x0 = 50 / 60`), force the
# WaistController's cross-attention at that output position to peak at the
# digit-spaced prompt position of the matching number.
#
# Implementation: extends the v78b WAIST_ATTN_SUPERVISION machinery — same stash,
# same JIT signature, same aux loss formula — with whole-number-token targets
# (v78b covers single-digit tokens; v95 covers multi-digit number tokens).
#
# V78_HEAD_CODEBOOK_N=12 to match v80_prod_step400 ckpt (DO NOT use 32).
#
# Success bar (smoke):
#   - Accuracy > 3% at any of step 100/200/300 (breakthrough from 0-1.7%)
#   - DAG executable rate > 20% (was 13%)
#   - Undefined-variable failures drop below 50% (was 65%)

set -e
cd "$(dirname "$0")/.."

V95_INIT="${V95_INIT:-.cache/gsm8k_steps_ckpts/v80_prod_step400.safetensors}"
if [ ! -f "$V95_INIT" ]; then
    echo "ERROR: v80 prod warm-start ckpt not found at $V95_INIT"
    exit 1
fi
V95_DATA="${V95_DATA:-.cache/gsm8k_steps_v80_train.jsonl}"
if [ ! -f "$V95_DATA" ]; then
    echo "ERROR: v80 train data not found at $V95_DATA"
    exit 1
fi

# ---- V77 DAG path ON (AR token generation) ----
export V77_DAG_TRAINING=1
export V77_N_LAYERS=7

# ---- V85 queryable structures OFF (AR not slot) ----
export V85_QUERYABLE=0

# ---- AR-CORRECT MASKING (matches v80 paradigm) ----
export V79_CAUSAL_MASKS=1     # kv_mask + notebook_pool_mask (correct for AR)
export V81_MAIN_ATTN_MASK=0   # OFF — preserves AR feedback loop

# ---- Multi-head OFF (single-head AR decode) ----
export MULTI_HEAD_WAIST=0

# ---- v82-v94 paradigms OFF ----
export V82_PARALLEL_DIFFUSION=0
export V83_ANYTIME_SUPERVISION=0
export V83_GRADUATION=1
export V91_SIMPLIFIED_ARGS=0
export V87_REINIT_SLOT_POS=0
export V88_REINIT_KV_PROJ=0
export V92_REINIT_ARG_POS_EMB=0
export V92_RESET_ACTIVE_HEAD_NEUTRAL=0
export V90_RESET_ACTIVE_HEAD=0

# ---- v77b knobs ----
export BREATH_EMBED_ORTHO_INIT=2.0
export PER_BREATH_TEMP=1
export BREATH_NORM_OSC=1

# ---- v78 knobs (V78_HEAD_CODEBOOK_N=12 to match v80_prod_step400) ----
export MAX_STEP_BASE=2.0
export MAX_STEP_MIN=0.1
export NOTEBOOK_ACCUMULATE_ENABLED=1
export NOTEBOOK_NO_DETACH=1
export V78_HEAD_CODEBOOK=1
export V78_HEAD_CODEBOOK_N=12    # CRITICAL: match v80 ckpt (don't use 32)

# ---- v78b WAIST_ATTN_SUPERVISION (the stash + aux loss machinery) ----
export CONTROLLER_N_LAYERS=4
export WAIST_ATTN_SUPERVISION=1
# v95 piggybacks on the WAIST_ATTN_AUX_WEIGHT scalar — set it to V95_OPERAND_AUX_WEIGHT.
export WAIST_ATTN_AUX_WEIGHT="${V95_OPERAND_AUX_WEIGHT:-0.5}"

# ---- v95 NEW knob ----
export V95_OPERAND_AUX=1
export V95_OPERAND_AUX_WEIGHT="${V95_OPERAND_AUX_WEIGHT:-0.5}"

# ---- SCHED_SAMPLE off (regular JIT path; SS has known hang risk with aux losses) ----
export SCHED_SAMPLE_RATE=0.0

# ---- v66 architecture (inherited verbatim — match v80_prod training config) ----
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

# ---- LR cosine decay (no warmup needed — warm-start) ----
export LR_DECAY_TO_ZERO=1
export V92_LR_WARMUP_STEPS=0

# ---- Training config — SMOKE (300 steps; CKPT every 50; gentle LR 1e-5) ----
export DEV=PCI+AMD
export LEVEL=GSM8K_STEPS
export GSM8K_STEPS_PATH="$V95_DATA"
export V77_TEST_PATH="$V95_DATA"
export GSM8K_STEPS_MIN_K=2
export GSM8K_STEPS_MAX_K=6
export SPACE_DIGITS=1
export BATCH="${BATCH:-4}"
export FIXED_LEN="${FIXED_LEN:-256}"
export STEPS="${STEPS:-300}"
export LR="${LR:-1e-5}"
export TRAIN_LOOPS="${TRAIN_LOOPS:-7}"
export EVAL_LOOPS="${EVAL_LOOPS:-7}"
export ACC_EVAL_EVERY=10000
export SKIP_FINAL_ACC=1
export CKPT_EVERY="${CKPT_EVERY:-50}"
export NUM_EVAL=20
export NUM_PROBLEMS=20000
export EVAL_BATCH=4
export EVAL_CACHE_LEN="${EVAL_CACHE_LEN:-264}"
export LOOKUP_AUX_WEIGHT=0.0
export USE_JIT=1
export USE_KV_CACHE=1
export CKPT_LABEL="${CKPT_LABEL:-v95_smoke}"
export RESUME_FROM="$V95_INIT"

/home/bryce/mycelium/.venv/bin/python -u scripts/l3_train.py 2>&1 | tee .cache/v95_smoke_train.log
