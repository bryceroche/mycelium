#!/bin/bash
# v97 SMOKE — Bombe-inspired CALIBRATION HEAD on v80 AR baseline.
#
# Architecturally clean test of self-assessment without the v96 consolidation-
# table feedback loop. v96.0/1/2 all diverged via positive feedback (table read
# → residual stream → next-breath table write). v97 bypasses this entirely:
#
#   - NO consolidation table (V96_CONSOLIDATION=0)
#   - NO new KV stream to WaistController
#   - Small Linear(1024 → 1) calibration head reads each breath's pooled output
#   - Per-breath progression target derived from final-breath CE (the proxy for
#     "will the model's answer be correct").
#   - Pure auxiliary loss — NEVER feeds back into residual stream.
#
# This isolates the question: "does calibration as an aux objective unlock
# accuracy?" with no architectural risk of v96-style feedback explosion.
#
# Warm-start: v80_prod_step400.safetensors (known-good 78% parse / 1.7% acc).
# All v80 settings preserved; only the v97 head is added.
#
# Success criteria:
#   - waist_norm STAYS BELOW 5 throughout training (stability check)
#   - B6 CE doesn't ascend > 20 consecutive steps
#   - Accuracy > 3% at any of step 100/200/300/500 → breakthrough; recommend prod
#   - Accuracy stays ~1.7% → final write-up; calibration aux alone insufficient

set -e
cd "$(dirname "$0")/.."

V97_INIT="${V97_INIT:-.cache/gsm8k_steps_ckpts/v80_prod_step400.safetensors}"
if [ ! -f "$V97_INIT" ]; then
    echo "ERROR: v80 prod warm-start ckpt not found at $V97_INIT"
    exit 1
fi
V97_DATA="${V97_DATA:-.cache/gsm8k_steps_v80_train.jsonl}"
if [ ! -f "$V97_DATA" ]; then
    echo "ERROR: v80 train data not found at $V97_DATA"
    exit 1
fi

# ---- V97 KNOBS — Bombe-inspired calibration head, pure aux loss ----
export V97_CALIBRATION=1
export V97_CALIB_WEIGHT="${V97_CALIB_WEIGHT:-0.1}"

# ---- V96 OFF (no consolidation table, no constraint heads) ----
export V96_CONSOLIDATION=0
export V96_TEMPERATURE_DECAY=0
export V96_ENERGY_WEIGHT=0.0

# ---- V77 DAG path ON (AR token generation) ----
export V77_DAG_TRAINING=1
export V77_N_LAYERS=7

# ---- V85 queryable structures OFF (AR not slot) ----
export V85_QUERYABLE=0

# ---- AR-CORRECT MASKING (paradigm-aware: 2 masks for AR) ----
export V79_CAUSAL_MASKS=1     # kv_mask + notebook_pool_mask (correct for AR)
export V81_MAIN_ATTN_MASK=0   # OFF — preserves AR feedback loop

# ---- Multi-head OFF (single-head AR decode) ----
export MULTI_HEAD_WAIST=0

# ---- v82-v95 paradigms OFF ----
export V82_PARALLEL_DIFFUSION=0
export V83_ANYTIME_SUPERVISION=0
export V83_GRADUATION=1
export V91_SIMPLIFIED_ARGS=0
export V95_OPERAND_AUX=0

# All slot-decoder reinits OFF
export V87_REINIT_SLOT_POS=0
export V88_REINIT_KV_PROJ=0
export V92_REINIT_ARG_POS_EMB=0
export V92_RESET_ACTIVE_HEAD_NEUTRAL=0
export V90_RESET_ACTIVE_HEAD=0

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

# ---- v78b knobs (waist attention supervision = the existing constraint mechanism) ----
export CONTROLLER_N_LAYERS=4
export WAIST_ATTN_SUPERVISION=1
export WAIST_ATTN_AUX_WEIGHT=0.5

# ---- SCHED_SAMPLE off (regular JIT path; v96/v97 don't support SS) ----
export SCHED_SAMPLE_RATE=0.0

# ---- waist STATIC (no schedule — proven by v94) ----
unset BFIELD_WAIST_SCHEDULE 2>/dev/null || true

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
export ABLATE_BREATH_ROTATION=1   # match v80 era (rotation was ablated)
export QUADRATURE_HEADS=0
export PROMPT_REFRESH_ALPHA=0.1
export BOUNDARY_AUX_WEIGHT=0.0
export BOUNDARY_POS_WEIGHT=5.0
export PER_BREATH_FULL_ANSWER=0

# ---- LR cosine decay (no warmup needed — warm-start from v80) ----
export LR_DECAY_TO_ZERO=1
export V92_LR_WARMUP_STEPS=0

# ---- Training config — SMOKE (500 steps; CKPT every 100; LR 1.5e-5 same as v80 continue) ----
export DEV=PCI+AMD
export LEVEL=GSM8K_STEPS
export GSM8K_STEPS_PATH="$V97_DATA"
export V77_TEST_PATH="$V97_DATA"
export GSM8K_STEPS_MIN_K=2
export GSM8K_STEPS_MAX_K=6
export SPACE_DIGITS=1
export BATCH="${BATCH:-4}"
export FIXED_LEN="${FIXED_LEN:-256}"
export STEPS="${STEPS:-500}"
export LR="${LR:-1.5e-5}"
export TRAIN_LOOPS="${TRAIN_LOOPS:-7}"
export EVAL_LOOPS="${EVAL_LOOPS:-7}"
export ACC_EVAL_EVERY=10000
export SKIP_FINAL_ACC=1
export CKPT_EVERY="${CKPT_EVERY:-100}"
export NUM_EVAL=20
export NUM_PROBLEMS=20000
export EVAL_BATCH=4
export EVAL_CACHE_LEN="${EVAL_CACHE_LEN:-264}"
export LOOKUP_AUX_WEIGHT=0.0
export USE_JIT=1
export USE_KV_CACHE=1
export CKPT_LABEL="${CKPT_LABEL:-v97_smoke}"
export RESUME_FROM="$V97_INIT"

/home/bryce/mycelium/.venv/bin/python -u scripts/l3_train.py 2>&1 | tee .cache/v97_smoke_train.log
