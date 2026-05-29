#!/bin/bash
# v93 COLD START — clone of v92_smoke_train.sh but warm-starts from v66 baseline
# instead of v92/v91/v90. The architecture is identical to v92; only the weights
# differ.
#
# Hypothesis: 12 iterations of warm-starting through broken slot-decoder
# mechanisms (v85→v86→v87→v88→v89→v90→v91→v92) baked degenerate patterns into
# the V85SlotDecoder weights. The architecture is now audit-verified, but the
# inherited weights can't escape the template basin (v92 → 1.7% ceiling, args_pred
# stuck at [0, 20, 21, 22, ...] template across problems).
#
# v93 tests: architecture is right + clean weights = breakthrough.
#
# Cold-start definition:
#   - KEEP v66_step3000 backbone weights (Pythia-410M + breathing components —
#     these were learned cleanly before the v85 era).
#   - DROP all V85SlotDecoder weights — fresh init by construction, then the
#     post-load v92 reinit blocks override arg_pos_emb and active_head_b.
#   - Note: v66 doesn't have V85SlotDecoder at all → warm-starting from v66
#     IS the cold-start for the slot decoder.
#
# Verify in log:
#   - "ckpt missing N keys" with N=~115 (96 architectural + 19 v85)
#   - "[v92] reinitialized arg_pos_emb at scale 0.5"
#   - "[v92] reset active_head_b NEUTRAL: ... -> 0.0"
#   - "  v85 types_codebook init from .cache/ib_centroids.npz"
#
# Decision point: step 500 args_pred dump. If templated → KILL. If varies → continue.

set -e
cd "$(dirname "$0")/.."

V93_INIT="${V93_INIT:-.cache/gsm8k_steps_ckpts/v66_sched_sampling_step3000.safetensors}"
if [ ! -f "$V93_INIT" ]; then
    echo "ERROR: v93 warm-start ckpt (v66) not found at $V93_INIT"
    exit 1
fi
V93_DATA="${V93_DATA:-.cache/gsm8k_steps_v85_train.jsonl}"
if [ ! -f "$V93_DATA" ]; then
    echo "ERROR: v93 train data not found at $V93_DATA"
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

# ---- v87 knobs (slot_pos is fresh-init by construction — no reinit needed) ----
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

# ---- v90 knobs (DISABLED — v92 supersedes) ----
export V90_RESET_ACTIVE_HEAD="${V90_RESET_ACTIVE_HEAD:-0}"
export V90_ACTIVE_BIAS="${V90_ACTIVE_BIAS:--1.0}"

# ---- v91 knobs (KEEP — simplified args pathway) ----
export V91_SIMPLIFIED_ARGS=1

# ---- v92 knobs — apply the FIXED init (the slot decoder is fresh from v66, so
# these reinits apply on top of construction defaults). arg_pos_emb fresh-init
# is zero (line 2946) so the v92 reinit at scale 0.5 is essential. active_head_b
# fresh-init is zero (line 2950) — v92 neutral reset is a no-op here but harmless.
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

# ---- Training config — COLD-START PROD (2000 steps; ckpt every 200) ----
export DEV=PCI+AMD
export LEVEL=GSM8K_STEPS
export GSM8K_STEPS_PATH="$V93_DATA"
export V77_TEST_PATH="$V93_DATA"
export GSM8K_STEPS_MIN_K=2
export GSM8K_STEPS_MAX_K=6
export SPACE_DIGITS=1
export BATCH="${BATCH:-4}"
export FIXED_LEN="${FIXED_LEN:-224}"
export STEPS="${STEPS:-2000}"
export LR="${LR:-3e-5}"
export TRAIN_LOOPS="${TRAIN_LOOPS:-5}"
export EVAL_LOOPS="${EVAL_LOOPS:-5}"
export V77_N_LAYERS="${V77_N_LAYERS:-5}"
export ACC_EVAL_EVERY=10000
export SKIP_FINAL_ACC=1
export CKPT_EVERY="${CKPT_EVERY:-200}"
export NUM_EVAL=20
export NUM_PROBLEMS=20000
export EVAL_BATCH=4
export EVAL_CACHE_LEN="${EVAL_CACHE_LEN:-232}"
export LOOKUP_AUX_WEIGHT=0.0
export USE_JIT=1
export USE_KV_CACHE=1
export CKPT_LABEL="${CKPT_LABEL:-v93_cold}"
export RESUME_FROM="$V93_INIT"

/home/bryce/mycelium/.venv/bin/python -u scripts/l3_train.py 2>&1 | tee .cache/v93_cold_train.log
