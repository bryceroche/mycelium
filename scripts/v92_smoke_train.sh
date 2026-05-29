#!/bin/bash
# v92 SMOKE — targeted init fixes for v91's simplified args.
#
# v91 confirmed the architectural hypothesis (15-25× gradient attenuation →
# 2-5×, args_ce dropped 37×, 1.7% first non-zero accuracy in 11 attempts).
# But the simplification exposed two init bottlenecks:
#
#   1. arg_pos_emb collapsed: zero-init for an additive (2, 1024) param on
#      slot_query (L2≈10). Rows stayed nearly identical (cos 0.915) →
#      args1 ≈ args2 → same pointer for both arg positions.
#   2. Active head over-corrected: v90 reset bias to -1.0 + v91 100 more
#      steps → active_logits mean = -1.67, all slots False → eval falls
#      back to 1-slot DAG.
#
# v92 multi-knob init fix:
#   Fix 1: V92_REINIT_ARG_POS_EMB=1 + V92_ARG_POS_EMB_SCALE=0.5
#          → uniform(-0.5, 0.5) instead of zero-init, args1/args2 start orthogonal.
#   Fix 2: V92_RESET_ACTIVE_HEAD_NEUTRAL=1
#          → active_head_b = 0.0 (sigmoid bias = 0.5), learn from data balance.
#   Fix 3: V92_LR_WARMUP_STEPS=50
#          → linear warmup; the v91 args_ce 60→1 transient at steps 0-50
#          spiked the loss at 50-60. Slower start lets args projection settle.
#   Fix 4: V92_ACTIVE_EVAL_THRESHOLD (default 0.5, tunable for eval safety net).
#
# Smoke success bar (STRICT BAIL AT STEP 100):
#   - Accuracy > 5% at step 100 (the 12-attempt gate)
#   - active=True fraction at step 100: 20-40% (vs v91's 0%)
#   - args1 != args2 in sample DAGs

set -e
cd "$(dirname "$0")/.."

V92_INIT="${V92_INIT:-.cache/gsm8k_steps_ckpts/v91_smoke_step100.safetensors}"
if [ ! -f "$V92_INIT" ]; then
    echo "ERROR: v92 warm-start ckpt not found at $V92_INIT"
    exit 1
fi
V92_DATA="${V92_DATA:-.cache/gsm8k_steps_v85_train.jsonl}"
if [ ! -f "$V92_DATA" ]; then
    echo "ERROR: v92 train data not found at $V92_DATA"
    exit 1
fi

# ---- V77 path off (v85+v91 has its own loader + train step) ----
export V77_DAG_TRAINING=0

# ---- v85 knobs ----
export V85_QUERYABLE=1
export V85_K_MAX=10
export V85_N_MAX=20
export V85_TYPES_N=32

# ---- v86 knobs (KEEP) ----
export V86_ARGS_CROSS_ATTN=1
export V86_ACTIVE_POS_WEIGHT="${V86_ACTIVE_POS_WEIGHT:-1.0}"

# ---- v87 knobs (preserve slot_pos diversity from v90/v91 ckpt — DON'T reinit) ----
export V87_SLOT_POS_INIT_SCALE="${V87_SLOT_POS_INIT_SCALE:-0.5}"
export V87_REINIT_SLOT_POS="${V87_REINIT_SLOT_POS:-0}"

# ---- v88 knobs (K/V projs deleted from forward under v91 — no reinit needed) ----
export V88_REINIT_KV_PROJ="${V88_REINIT_KV_PROJ:-0}"
export V88_KV_PROJ_INIT_SCALE="${V88_KV_PROJ_INIT_SCALE:-0.02}"

# ---- v89 knobs (DISABLED — v91 has no cross-attn) ----
export V89_SUPERVISED_ATTN="${V89_SUPERVISED_ATTN:-0}"
export V89_SUPERVISED_ATTN_WEIGHT="${V89_SUPERVISED_ATTN_WEIGHT:-0.0}"
export V89_PROJ_INIT_SCALE="${V89_PROJ_INIT_SCALE:-0.02}"
export V89_INHERIT_V86="${V89_INHERIT_V86:-0}"

# ---- v90 knobs (DISABLED for v92 — v92 supersedes with neutral reset) ----
export V90_RESET_ACTIVE_HEAD="${V90_RESET_ACTIVE_HEAD:-0}"
export V90_ACTIVE_BIAS="${V90_ACTIVE_BIAS:--1.0}"

# ---- v91 knobs (KEEP — simplified args pathway) ----
export V91_SIMPLIFIED_ARGS=1

# ---- v92 knobs (NEW) ----
export V92_REINIT_ARG_POS_EMB="${V92_REINIT_ARG_POS_EMB:-1}"
export V92_ARG_POS_EMB_SCALE="${V92_ARG_POS_EMB_SCALE:-0.5}"
export V92_RESET_ACTIVE_HEAD_NEUTRAL="${V92_RESET_ACTIVE_HEAD_NEUTRAL:-1}"
export V92_LR_WARMUP_STEPS="${V92_LR_WARMUP_STEPS:-50}"

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

# ---- v78b knobs ----
export CONTROLLER_N_LAYERS=4
export WAIST_ATTN_SUPERVISION=0
export WAIST_ATTN_AUX_WEIGHT=0.0

# ---- v79/v81 knobs ----
export V79_CAUSAL_MASKS=1
export SCHED_SAMPLE_RATE=0.0
export MULTI_HEAD_WAIST=0
export V81_MAIN_ATTN_MASK=1

# ---- v82-v84 knobs ----
export V82_PARALLEL_DIFFUSION=0
export V83_ANYTIME_SUPERVISION=0
export V83_GRADUATION=1

# ---- v83 waist schedule ----
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

# ---- Training config — SMOKE (150 steps; 50 warmup + 100 effective; ckpt every 25) ----
export DEV=PCI+AMD
export LEVEL=GSM8K_STEPS
export GSM8K_STEPS_PATH="$V92_DATA"
export V77_TEST_PATH="$V92_DATA"
export GSM8K_STEPS_MIN_K=2
export GSM8K_STEPS_MAX_K=6
export SPACE_DIGITS=1
export BATCH="${BATCH:-4}"
export FIXED_LEN="${FIXED_LEN:-224}"
export STEPS="${STEPS:-150}"
export LR="${LR:-3e-5}"
export TRAIN_LOOPS="${TRAIN_LOOPS:-5}"
export EVAL_LOOPS="${EVAL_LOOPS:-5}"
export V77_N_LAYERS="${V77_N_LAYERS:-5}"
export ACC_EVAL_EVERY=10000
export SKIP_FINAL_ACC=1
export CKPT_EVERY="${CKPT_EVERY:-25}"
export NUM_EVAL=20
export NUM_PROBLEMS=20000
export EVAL_BATCH=4
export EVAL_CACHE_LEN="${EVAL_CACHE_LEN:-232}"
export LOOKUP_AUX_WEIGHT=0.0
export USE_JIT=1
export USE_KV_CACHE=1
export CKPT_LABEL="${CKPT_LABEL:-v92_smoke}"
export RESUME_FROM="$V92_INIT"

/home/bryce/mycelium/.venv/bin/python -u scripts/l3_train.py 2>&1 | tee .cache/v92_smoke_train.log
