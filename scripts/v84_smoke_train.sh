#!/bin/bash
# v84 SMOKE — OUTLINE-TO-ESSAY training.
#
# v84 is the breakthrough paradigm after v83. v83 hit 20% parse rate at grad
# step 400 (first time post-mask-fix) but 0% accuracy because args content is
# random. v84 replaces v83's parallel-diffusion supervision with OUTLINE-TO-
# ESSAY supervision: each breath ADDS a new dimension rather than refining
# all dimensions in parallel.
#
# Per-breath target schedule (K=5):
#
#   B0:  ops only                    "2,0"
#   B1:  + types                     "2,0 | 0.1.1,0.0.1"
#   B2:  + arg magnitudes            "2,0 | 0.1.1,0.0.1 | 50,60,r,10"
#   B3:  + arg exact                 "2,0 | 0.1.1,0.0.1 | 50,60,-1,12"
#   B4:  refinement (same as B3)     "2,0 | 0.1.1,0.0.1 | 50,60,-1,12"
#
# Key properties:
# - Monotone-length: B0 shortest, B3+ longest. Model learns to ADD content per
#   breath, not refine.
# - No placeholders: B0 just emits ops, no `?` tokens. Each breath is a
#   complete partial answer.
# - V83_ANYTIME_SUPERVISION=1 keeps the "capable students read ahead" semantics:
#   per-position min CE between the schedule target and the full target. At B0,
#   schedule has 2 tokens of supervision; the model is rewarded if it emits the
#   full essay at later positions when it can.
#
# Warm-start: v83_smoke_grad_step400 (the best v83 ckpt — 20% parse, 0% acc).
# Sequences: K=5 breaths, FIXED_LEN=224 (proven safe).
# Data: .cache/gsm8k_steps_v84_train.jsonl (3852 records — same as v82/v83).
#
# v84-specific additions:
#   - LR_DECAY_TO_ZERO=1 — cosine decay to 0 over STEPS. Stabilizes peak.
#   - V83_GRAD_TEMP=0.5 + V83_GRAD_MAINTENANCE=0.15 — sharper graduation softmax
#     with a higher floor (the sharper softmax narrows the distribution; the
#     higher floor ensures no breath starves).
#   - CKPT_EVERY=50 — catch the peak.

set -e
cd "$(dirname "$0")/.."

V84_INIT="${V84_INIT:-.cache/gsm8k_steps_ckpts/v83_smoke_grad_step400.safetensors}"
if [ ! -f "$V84_INIT" ]; then
    echo "ERROR: warm-start ckpt not found at $V84_INIT"
    exit 1
fi
V84_DATA="${V84_DATA:-.cache/gsm8k_steps_v84_train.jsonl}"
if [ ! -f "$V84_DATA" ]; then
    echo "ERROR: v84 train data not found at $V84_DATA"
    exit 1
fi

# ---- V77 path (per-breath layered supervision; reused by v82/v83/v84) ----
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

# ---- v79 knobs ----
export V79_CAUSAL_MASKS=1
export SCHED_SAMPLE_RATE=0.0

# ---- v81 knobs (KEEP — all 4 masks are still required for clean train/eval) ----
export MULTI_HEAD_WAIST=0
export V81_MAIN_ATTN_MASK=1

# ---- v82 knobs (inherited) ----
export V82_PARALLEL_DIFFUSION=1

# ---- v83 knobs (inherited — anytime + graduation + waist schedule) ----
# v84-tuned graduation: sharper temperature + higher maintenance floor.
export BFIELD_WAIST_SCHEDULE="${BFIELD_WAIST_SCHEDULE:-64,256,384,512,512}"
export V83_ANYTIME_SUPERVISION="${V83_ANYTIME_SUPERVISION:-1}"
export V83_GRADUATION="${V83_GRADUATION:-1}"
export V83_GRAD_MAINTENANCE="${V83_GRAD_MAINTENANCE:-0.15}"
export V83_GRAD_EMA_ALPHA="${V83_GRAD_EMA_ALPHA:-0.1}"
export V83_GRAD_TEMP="${V83_GRAD_TEMP:-0.5}"

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

# ---- v84 NEW knob: cosine LR decay to zero over STEPS ----
export LR_DECAY_TO_ZERO=1

# ---- Training config — SMOKE (500 steps on v84 data, K=5 breaths) ----
export DEV=PCI+AMD
export LEVEL=GSM8K_STEPS
export GSM8K_STEPS_PATH="$V84_DATA"
export V77_TEST_PATH="$V84_DATA"
export GSM8K_STEPS_MIN_K=2
export GSM8K_STEPS_MAX_K=6
export SPACE_DIGITS=1
export BATCH="${BATCH:-4}"
export FIXED_LEN="${FIXED_LEN:-224}"
export STEPS="${STEPS:-500}"
export LR=3e-5
# K=5 breaths matches the v84 schedule (B0..B4).
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
export CKPT_LABEL="${CKPT_LABEL:-v84_smoke}"
export RESUME_FROM="$V84_INIT"

/home/bryce/mycelium/.venv/bin/python scripts/l3_train.py 2>&1 | tee .cache/v84_smoke_train.log
