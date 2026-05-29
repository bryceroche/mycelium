#!/bin/bash
# v82 SMOKE — SINGLE-HEAD parallel-diffusion training on v82 data.
#
# v82 is the next architecture after v81. v81 failed because multi-head per-position
# CE diluted content learning into structural-token CE. v82 fixes this with SINGLE-
# HEAD full-sequence CE on a parallel-diffusion target schedule.
#
# Each breath emits the FULL 3-list sequence at a different precision level:
#   B0: all '?' placeholders
#   B1: ops exact + types depth-1 + args magnitude
#   B2: types depth-2 + args exact
#   B3..B6: types leaf + args exact (refinement passes)
#
# Architecture (vs v81 smoke):
#   - MULTI_HEAD_WAIST=0 (revert from v81)
#   - V82_PARALLEL_DIFFUSION=1
#   - V81_MAIN_ATTN_MASK=1 + V79_CAUSAL_MASKS=1 (keep all 4 masks for clean train/eval)
#   - V78_HEAD_CODEBOOK_N=32 (keep)
#   - BATCH=4, FIXED_LEN=256 (sequences fit; we ran a 256-token sanity check)
#
# Data: .cache/gsm8k_steps_v82_train.jsonl (3 lists, parallel-diffusion targets).

set -e
cd "$(dirname "$0")/.."

V82_INIT="${V82_INIT:-.cache/gsm8k_steps_ckpts/v66_sched_sampling_step3000.safetensors}"
if [ ! -f "$V82_INIT" ]; then
    echo "ERROR: warm-start ckpt not found at $V82_INIT"
    exit 1
fi
V80_SMOKE_DATA="${V80_SMOKE_DATA:-.cache/gsm8k_steps_v82_train.jsonl}"
if [ ! -f "$V80_SMOKE_DATA" ]; then
    echo "ERROR: v82 train data not found at $V80_SMOKE_DATA"
    exit 1
fi

# ---- V77 path (per-breath layered supervision; reused by v82) ----
export V77_DAG_TRAINING=1

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
# v82 (2026-05-27) — like v81, disable scheduled sampling to avoid AMD JIT hangs
# with the new aux losses + per-breath SS path. The 70-step smoke is well under
# the SCHED_SAMPLE warmup=500 step anyway.
export SCHED_SAMPLE_RATE=0.0

# ---- v81 knobs (KEEP — all 4 masks are still required for clean train/eval) ----
export MULTI_HEAD_WAIST=0
export V81_MAIN_ATTN_MASK=1

# ---- v82 NEW knobs ----
export V82_PARALLEL_DIFFUSION=1

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

# ---- Training config — SMOKE (70 steps on v82 data) ----
export DEV=PCI+AMD
export LEVEL=GSM8K_STEPS
export GSM8K_STEPS_PATH="$V80_SMOKE_DATA"
export V77_TEST_PATH="$V80_SMOKE_DATA"
export GSM8K_STEPS_MIN_K=2
export GSM8K_STEPS_MAX_K=6
export SPACE_DIGITS=1
export BATCH="${BATCH:-4}"
export FIXED_LEN="${FIXED_LEN:-256}"
export STEPS="${STEPS:-70}"
export LR=3e-5
export TRAIN_LOOPS="${TRAIN_LOOPS:-7}"
export EVAL_LOOPS="${EVAL_LOOPS:-7}"
export V77_N_LAYERS="${V77_N_LAYERS:-7}"
export ACC_EVAL_EVERY=10000
export SKIP_FINAL_ACC=1
export CKPT_EVERY="${CKPT_EVERY:-10}"
export NUM_EVAL=20
export NUM_PROBLEMS=20000
export EVAL_BATCH=4
export EVAL_CACHE_LEN="${EVAL_CACHE_LEN:-264}"
export LOOKUP_AUX_WEIGHT=0.0
export USE_JIT=1
export USE_KV_CACHE=1
export CKPT_LABEL="${CKPT_LABEL:-v82_smoke}"
export RESUME_FROM="$V82_INIT"

/home/bryce/mycelium/.venv/bin/python scripts/l3_train.py 2>&1 | tee .cache/v82_smoke_train.log
