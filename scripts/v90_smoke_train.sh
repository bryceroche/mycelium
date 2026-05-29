#!/bin/bash
# v90 SMOKE — fix active-head over-firing.
#
# v89 worked at the attention layer (supervised attn loss pushed args1/args2
# cross-attn distributions toward gold number positions) but accuracy stayed
# at 0% because the active head over-fires: every problem has
# active=[T,T,T,T,T,T,F,F,F,F] (exactly 6 True) regardless of gold n_steps.
# The 4 phantom slots inherit the degenerate "args2 = 20+k" pattern from
# unsupervised slot-arg pairs and pollute the DAG.
#
# v90 single-knob fix:
#   1. Lower V86_ACTIVE_POS_WEIGHT 5.0 -> 1.0. With pos_weight=5.0, FN cost is
#      5x FP — combined with the dataset's ~80% False rate this baked in a
#      strong over-fire bias. pos_weight=1.0 lets BCE balance correctly.
#   2. V90_RESET_ACTIVE_HEAD=1 — reset active_head_b to -1.0 after load so the
#      starting prediction defaults to False (sigmoid(-1.0) ≈ 0.27). The
#      weight matrix is preserved (keeps learned slot-query features); only
#      the bias is reset to undo 200 steps of inherited over-fire.
#   3. Eval already truncates correctly via _render_dag(if not active[k]: continue).
#      v90 adds a fallback: if NO slots are active, force slot 0 active.
#
# Smoke success bar: accuracy > 5% at step 100. active=True frac should drop
# from ~60% (v89 baseline) to ~20-30% (matching gold n_steps fraction).

set -e
cd "$(dirname "$0")/.."

V90_INIT="${V90_INIT:-.cache/gsm8k_steps_ckpts/v89_smoke_step200.safetensors}"
if [ ! -f "$V90_INIT" ]; then
    echo "ERROR: v90 warm-start ckpt not found at $V90_INIT"
    exit 1
fi
V90_DATA="${V90_DATA:-.cache/gsm8k_steps_v85_train.jsonl}"
if [ ! -f "$V90_DATA" ]; then
    echo "ERROR: v90 train data not found at $V90_DATA"
    exit 1
fi

# ---- V77 path off (v85/v86/v87/v88/v89/v90 has its own loader + train step) ----
export V77_DAG_TRAINING=0

# ---- v85 knobs ----
export V85_QUERYABLE=1
export V85_K_MAX=10
export V85_N_MAX=20
export V85_TYPES_N=32

# ---- v86 knobs (CHANGED: pos_weight 5.0 -> 1.0) ----
export V86_ARGS_CROSS_ATTN=1
export V86_ACTIVE_POS_WEIGHT="${V86_ACTIVE_POS_WEIGHT:-1.0}"

# ---- v87 knobs (preserve slot_pos diversity from v89 ckpt) ----
export V87_SLOT_POS_INIT_SCALE="${V87_SLOT_POS_INIT_SCALE:-0.5}"
export V87_REINIT_SLOT_POS="${V87_REINIT_SLOT_POS:-0}"

# ---- v88 knobs (DO NOT re-reinit; v89 ckpt has trained K/V) ----
export V88_REINIT_KV_PROJ="${V88_REINIT_KV_PROJ:-0}"
export V88_KV_PROJ_INIT_SCALE="${V88_KV_PROJ_INIT_SCALE:-0.02}"

# ---- v89 knobs (preserve trained supervised attn) ----
export V89_SUPERVISED_ATTN="${V89_SUPERVISED_ATTN:-1}"
export V89_SUPERVISED_ATTN_WEIGHT="${V89_SUPERVISED_ATTN_WEIGHT:-0.5}"
export V89_PROJ_INIT_SCALE="${V89_PROJ_INIT_SCALE:-0.02}"
export V89_INHERIT_V86="${V89_INHERIT_V86:-0}"

# ---- v90 knobs (NEW) ----
export V90_RESET_ACTIVE_HEAD="${V90_RESET_ACTIVE_HEAD:-1}"
export V90_ACTIVE_BIAS="${V90_ACTIVE_BIAS:--1.0}"

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

# ---- Training config — SMOKE (100 steps, ckpt every 25, bail at 100) ----
export DEV=PCI+AMD
export LEVEL=GSM8K_STEPS
export GSM8K_STEPS_PATH="$V90_DATA"
export V77_TEST_PATH="$V90_DATA"
export GSM8K_STEPS_MIN_K=2
export GSM8K_STEPS_MAX_K=6
export SPACE_DIGITS=1
export BATCH="${BATCH:-4}"
export FIXED_LEN="${FIXED_LEN:-224}"
export STEPS="${STEPS:-100}"
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
export CKPT_LABEL="${CKPT_LABEL:-v90_smoke}"
export RESUME_FROM="$V90_INIT"

/home/bryce/mycelium/.venv/bin/python -u scripts/l3_train.py 2>&1 | tee .cache/v90_smoke_train.log
