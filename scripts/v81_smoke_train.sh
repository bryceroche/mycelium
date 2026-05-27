#!/bin/bash
# v81 SMOKE — multi-head WaistController training on v81 data.
#
# Hypothesis: with v81's 4-list parallel supervision (ops/types/args1/args2 separated
# by " | "), the multi-head WaistController eliminates the per-breath CE ladder. Each
# head specializes in its list; per-breath CE becomes UNIFORM (all breaths similar).
# v8 v3 step 70 reference: pb_ce=[3.10, 3.68, 4.02, 4.20, 4.38, 4.32, 3.66], spread 1.28.
#
# Architecture changes vs v80:
#   - MULTI_HEAD_WAIST=1: WaistController emits 4 parallel heads (ops/types/args1/args2)
#   - V81_MAIN_ATTN_MASK=1: main-self-attn keys masked + input embeddings zeroed at
#     answer-span positions. Combined with V79's notebook + cross-attn masks, this makes
#     the model fully prompt-conditional (no teacher-forcing leak). Verified via
#     scripts/diag_v81_masking_audit.py — MUST pass before training.
#   - Data: .cache/gsm8k_steps_v81_train.jsonl (v81 multi-list targets).

set -e
cd "$(dirname "$0")/.."

V81_INIT="${V81_INIT:-.cache/gsm8k_steps_ckpts/v66_sched_sampling_step3000.safetensors}"
if [ ! -f "$V81_INIT" ]; then
    echo "ERROR: warm-start ckpt not found at $V81_INIT"
    exit 1
fi
V80_SMOKE_DATA="${V80_SMOKE_DATA:-.cache/gsm8k_steps_v81_train.jsonl}"
if [ ! -f "$V80_SMOKE_DATA" ]; then
    echo "ERROR: v81 train data not found at $V80_SMOKE_DATA"
    exit 1
fi

# ---- V77 path (inherits the data + JIT branch in l3_train.py / l3_training.py) ----
export V77_DAG_TRAINING=1
export V77_N_LAYERS=7

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
export V78_HEAD_CODEBOOK_N="${V78_HEAD_CODEBOOK_N:-32}"

# ---- v78b knobs ----
export CONTROLLER_N_LAYERS=4
export WAIST_ATTN_SUPERVISION=1
export WAIST_ATTN_AUX_WEIGHT=0.5

# ---- v79 NEW knobs ----
export V79_CAUSAL_MASKS=1
# v81 (2026-05-26) DISABLE scheduled sampling — AMD JIT hangs with SS path + new aux losses
# (documented in memory/reference_tinygrad_am_quirks.md). The v81 smoke (70 steps) runs at
# step_idx < SCHED_SAMPLE warmup=500 anyway, so the sampling rate would always be 0.0;
# we just need to take the regular JIT path (SCHED_SAMPLE_RATE=0) to avoid the hang.
export SCHED_SAMPLE_RATE=0.0

# ---- v81 NEW knobs ----
export MULTI_HEAD_WAIST=1
export V81_MAIN_ATTN_MASK=1

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

# ---- Training config — SMOKE (70 steps on v81 data) ----
export DEV=PCI+AMD
export LEVEL=GSM8K_STEPS
export GSM8K_STEPS_PATH="$V80_SMOKE_DATA"
export V77_TEST_PATH="$V80_SMOKE_DATA"
export GSM8K_STEPS_MIN_K=2
export GSM8K_STEPS_MAX_K=6
export SPACE_DIGITS=1
export BATCH="${BATCH:-2}"
export FIXED_LEN="${FIXED_LEN:-320}"
export STEPS="${STEPS:-70}"
export LR=3e-5
export TRAIN_LOOPS=7
export EVAL_LOOPS=7
export ACC_EVAL_EVERY=10000
export SKIP_FINAL_ACC=1
export CKPT_EVERY="${CKPT_EVERY:-10}"
export NUM_EVAL=20
export NUM_PROBLEMS=20000
export EVAL_BATCH=4
export EVAL_CACHE_LEN=328
export LOOKUP_AUX_WEIGHT=0.0
export USE_JIT=1
export USE_KV_CACHE=1
export CKPT_LABEL="${CKPT_LABEL:-v81_smoke}"
export RESUME_FROM="$V81_INIT"

/home/bryce/mycelium/.venv/bin/python scripts/l3_train.py 2>&1 | tee .cache/v81_smoke_train.log
