#!/bin/bash
# v88 SMOKE — targeted K/V projection reinit fix for args cross-attention.
#
# v87 fixed the slot Q-side diversity (slot_pos_embed reinit at scale 0.5),
# confirmed by diag_v87_slot_pos_init.py: slots are now diverse on the Q side
# (cosine off-diag 0.007). But the args cross-attention attention pattern
# STILL collapsed:
#   - v86_args_k_proj L2 = 0.08 → 0.10 (essentially zero after 300 v87 steps)
#   - v86_args_v_proj L2 = 0.28 → 0.39 (also essentially zero)
#   - attn_scores = q @ k.T ≈ 0  →  softmax(zero) ≈ uniform  →  all slots get
#     the SAME attention regardless of Q diversity.
# This is a chicken-and-egg vanishing-gradient problem: weak K means tiny score
# variation, which means tiny softmax gradient, which means K can't lift.
#
# v88 fix:
#   - V88_REINIT_KV_PROJ=1: REINITIALIZE v86_args_k_proj and v86_args_v_proj
#     AFTER load_state_dict at scale V88_KV_PROJ_INIT_SCALE (default 0.02).
#   - Warm-start from v87_smoke_step300.safetensors (preserves the breakthrough
#     slot_pos diversity); we do NOT re-reinit slot_pos (V87_REINIT_SLOT_POS=0).
#
# Everything else cloned verbatim from v87 smoke. One knob.

set -e
cd "$(dirname "$0")/.."

V88_INIT="${V88_INIT:-.cache/gsm8k_steps_ckpts/v87_smoke_step300.safetensors}"
if [ ! -f "$V88_INIT" ]; then
    echo "ERROR: v88 warm-start ckpt not found at $V88_INIT"
    exit 1
fi
V88_DATA="${V88_DATA:-.cache/gsm8k_steps_v85_train.jsonl}"
if [ ! -f "$V88_DATA" ]; then
    echo "ERROR: v88 train data not found at $V88_DATA"
    exit 1
fi

# ---- V77 path off (v85/v86/v87/v88 has its own loader + train step) ----
export V77_DAG_TRAINING=0

# ---- v85 knobs ----
export V85_QUERYABLE=1
export V85_K_MAX=10
export V85_N_MAX=20
export V85_TYPES_N=32

# ---- v86 knobs ----
export V86_ARGS_CROSS_ATTN=1
export V86_ACTIVE_POS_WEIGHT="${V86_ACTIVE_POS_WEIGHT:-5.0}"

# ---- v87 knobs (KEEP slot_pos init scale; do NOT re-reinit since the v87
# ckpt already has the differentiated values saved) ----
export V87_SLOT_POS_INIT_SCALE="${V87_SLOT_POS_INIT_SCALE:-0.5}"
export V87_REINIT_SLOT_POS="${V87_REINIT_SLOT_POS:-0}"

# ---- v88 knobs (NEW — the only change) ----
export V88_REINIT_KV_PROJ="${V88_REINIT_KV_PROJ:-1}"
export V88_KV_PROJ_INIT_SCALE="${V88_KV_PROJ_INIT_SCALE:-0.02}"

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

# ---- Training config — SMOKE (300 steps, K=5 breaths) ----
export DEV=PCI+AMD
export LEVEL=GSM8K_STEPS
export GSM8K_STEPS_PATH="$V88_DATA"
export V77_TEST_PATH="$V88_DATA"
export GSM8K_STEPS_MIN_K=2
export GSM8K_STEPS_MAX_K=6
export SPACE_DIGITS=1
export BATCH="${BATCH:-4}"
export FIXED_LEN="${FIXED_LEN:-224}"
export STEPS="${STEPS:-300}"
export LR="${LR:-3e-5}"
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
export CKPT_LABEL="${CKPT_LABEL:-v88_smoke}"
export RESUME_FROM="$V88_INIT"

/home/bryce/mycelium/.venv/bin/python -u scripts/l3_train.py 2>&1 | tee .cache/v88_smoke_train.log
