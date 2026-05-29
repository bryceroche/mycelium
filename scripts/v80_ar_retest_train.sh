#!/bin/bash
# v80 AR RETEST — autoregressive DAG generation with all 4 masks active from step 0.
#
# Pivot rationale: 12 iterations of V85SlotDecoder (v85→v94) all converge to a
# byte-identical 4-slot template ("multiply two numbers, square it, add a number,
# double it"). v94 photon (cold-start, static waist, rotation osc, norm osc)
# produced LITERALLY IDENTICAL output to v92 prod (warm-start, varied schedule).
# This is conclusive evidence that the V85SlotDecoder family has a structural
# attractor at this template that no init/schedule/supervision change can break.
#
# v80 paradigm: model emits the full DAG string AUTOREGRESSIVELY token-by-token
# ("x0 = 50 / 60 ; x1 = x0 * 12 ; answer = x1"). Variable n_steps natural via
# AR. Sequential causation: each token conditioned on prior tokens.
#
# The hypothesis: v80 hit 28% parse / 1.7% accuracy ORIGINALLY but was trained
# with only 2 of 4 masks (kv_mask + notebook_pool_mask, MISSING main_attn_mask +
# embed_mask). The model was learning to peek at the answer span via the
# unmasked paths. v80 retest: same paradigm but with all 4 masks from step 0,
# so the model has to learn proper attention without cheating.
#
# Cold-start from v66 backbone (no v77+ slot decoder weights to inherit).

set -e
cd "$(dirname "$0")/.."

V80_INIT="${V80_INIT:-.cache/gsm8k_steps_ckpts/v66_sched_sampling_step3000.safetensors}"
if [ ! -f "$V80_INIT" ]; then
    echo "ERROR: v80 warm-start ckpt (v66) not found at $V80_INIT"
    exit 1
fi
V80_DATA="${V80_DATA:-.cache/gsm8k_steps_v80_train.jsonl}"
if [ ! -f "$V80_DATA" ]; then
    echo "ERROR: v80 train data not found at $V80_DATA"
    exit 1
fi

# ---- V77 DAG path ON (this is what makes it v80 paradigm) ----
export V77_DAG_TRAINING=1
export V77_N_LAYERS=7

# ---- V85 queryable structures OFF (we're back to AR token generation) ----
export V85_QUERYABLE=0

# ---- AR-CORRECT MASKING (take 2) ----
# kv_mask + notebook_pool_mask: yes (notebook should not peek at answer-span)
# main_attn_mask + embed_mask: NO — those break the AR feedback loop. The model
# must see its own prior generated tokens via main attention + embedding to do
# autoregression. Standard causal attention (transformer default) handles the
# "no future tokens" concern. v81 mask discovery was correct for SLOT decoders
# (parallel emission) but wrong for AR (sequential emission).
export V79_CAUSAL_MASKS=1     # kv_mask + notebook_pool_mask
export V81_MAIN_ATTN_MASK=0   # OFF — let main attn + embed work for AR

# ---- Multi-head OFF (single-head AR decode like v77/v80 original) ----
export MULTI_HEAD_WAIST=0

# ---- v82-v92 paradigms OFF ----
export V82_PARALLEL_DIFFUSION=0
export V83_ANYTIME_SUPERVISION=0
export V83_GRADUATION=1     # keep this — works across paradigms
export V91_SIMPLIFIED_ARGS=0

# All v87/v88/v92 init reinits OFF (no slot decoder to reinit)
export V87_REINIT_SLOT_POS=0
export V88_REINIT_KV_PROJ=0
export V92_REINIT_ARG_POS_EMB=0
export V92_RESET_ACTIVE_HEAD_NEUTRAL=0
export V90_RESET_ACTIVE_HEAD=0

# ---- v77b knobs ----
export BREATH_EMBED_ORTHO_INIT=2.0
export PER_BREATH_TEMP=1
export BREATH_NORM_OSC=1

# ---- v78 knobs (these were active in v80 era) ----
export MAX_STEP_BASE=2.0
export MAX_STEP_MIN=0.1
export NOTEBOOK_ACCUMULATE_ENABLED=1
export NOTEBOOK_NO_DETACH=1
export V78_HEAD_CODEBOOK=1
export V78_HEAD_CODEBOOK_N=32

# ---- v78b knobs ----
export CONTROLLER_N_LAYERS=4
export WAIST_ATTN_SUPERVISION=1
export WAIST_ATTN_AUX_WEIGHT=0.5

# ---- v79/v81 knobs already set above ----
export SCHED_SAMPLE_RATE=0.0

# ---- waist STATIC (no schedule — same as v94) ----
unset BFIELD_WAIST_SCHEDULE 2>/dev/null || true

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
export ABLATE_BREATH_ROTATION=0   # ← re-enable rotation (v94 learning carried forward)
export QUADRATURE_HEADS=0
export PROMPT_REFRESH_ALPHA=0.1
export BOUNDARY_AUX_WEIGHT=0.0
export BOUNDARY_POS_WEIGHT=5.0
export PER_BREATH_FULL_ANSWER=0

# ---- LR with warmup ----
export LR_DECAY_TO_ZERO=1
export V92_LR_WARMUP_STEPS=50

# ---- Training config — SMOKE (500 steps to validate AR paradigm with 4 masks) ----
export DEV=PCI+AMD
export LEVEL=GSM8K_STEPS
export GSM8K_STEPS_PATH="$V80_DATA"
export V77_TEST_PATH="$V80_DATA"
export GSM8K_STEPS_MIN_K=2
export GSM8K_STEPS_MAX_K=6
export SPACE_DIGITS=1
export BATCH="${BATCH:-4}"
export FIXED_LEN="${FIXED_LEN:-256}"
export STEPS="${STEPS:-500}"
export LR="${LR:-3e-5}"
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
export CKPT_LABEL="${CKPT_LABEL:-v80_ar_retest_v2}"
export RESUME_FROM="$V80_INIT"

/home/bryce/mycelium/.venv/bin/python -u scripts/l3_train.py 2>&1 | tee .cache/v80_ar_retest_train.log
