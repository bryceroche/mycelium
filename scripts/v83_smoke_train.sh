#!/bin/bash
# v83 SMOKE — PHOTON-MODE parallel-diffusion training.
#
# v83 is the next architecture after v82. v82 worked architecturally (first model
# in 6 attempts to emit `|` separators) but the per-breath supervision was too
# rigid and the waist was fixed-width. v83 fixes both:
#
#   1. Per-breath VARIED WAIST WIDTH (BFIELD_WAIST_SCHEDULE):
#      The B field becomes a precision ladder phase-locked to the E field
#      rotation. At breath b only the first k_b channels of the 512d waist are
#      kept (the rest are mask-zeroed BEFORE GELU + codebook injection). Narrow
#      waist at B0 physically limits precision (can't emit "50,60,-1,12" through
#      a 64d bottleneck); wide waist at B3+ allows full precision.
#
#      Schedule: "64,256,384,512,512" — K=5 breaths from narrow to wide.
#
#   2. ANYTIME SUPERVISION (V83_ANYTIME_SUPERVISION=1):
#      Each breath has TWO targets — the breath-specific scheduled target (v82's
#      precision-locked label) and the final-breath FULL-precision target. Per-
#      position CE is min(ce_schedule, ce_full): a strong student that emits
#      full precision early gets credit at any position where ce_full < ce_sched;
#      a weak student emits the schedule precision (ce_schedule low). The
#      capable students "read ahead" — bounded by the physical waist limit.
#
# Combined: phase-locked photon mode. E (rotation), B (waist width), and
# supervision (anytime precision) all oscillate together across breaths.
#
# Warm-start: v82_smoke_k5_step500 (continue v82's structural signal).
# Sequences: K=5 breaths, FIXED_LEN=224 (proven safe).
# Data: same v82 train/test data (NO Haiku regen needed).

set -e
cd "$(dirname "$0")/.."

V83_INIT="${V83_INIT:-.cache/gsm8k_steps_ckpts/v82_smoke_k5_step500.safetensors}"
if [ ! -f "$V83_INIT" ]; then
    echo "ERROR: warm-start ckpt not found at $V83_INIT"
    exit 1
fi
V82_DATA="${V82_DATA:-.cache/gsm8k_steps_v82_train.jsonl}"
if [ ! -f "$V82_DATA" ]; then
    echo "ERROR: v82 train data not found at $V82_DATA"
    exit 1
fi

# ---- V77 path (per-breath layered supervision; reused by v82/v83) ----
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
# v83 (2026-05-27) — like v82, disable scheduled sampling. The 500-step smoke
# is under the SCHED_SAMPLE warmup=500 anyway.
export SCHED_SAMPLE_RATE=0.0

# ---- v81 knobs (KEEP — all 4 masks are still required for clean train/eval) ----
export MULTI_HEAD_WAIST=0
export V81_MAIN_ATTN_MASK=1

# ---- v82 knobs (inherited) ----
export V82_PARALLEL_DIFFUSION=1

# ---- v83 NEW knobs ----
# Per-breath waist-width schedule. K=5 breaths: narrow → wide. The wide-end
# values exceed BFIELD_WAIST=512 (clipped at allocation), so 512 is the cap.
export BFIELD_WAIST_SCHEDULE="${BFIELD_WAIST_SCHEDULE:-64,256,384,512,512}"
# Anytime supervision — per-position min CE between scheduled and full targets.
export V83_ANYTIME_SUPERVISION="${V83_ANYTIME_SUPERVISION:-1}"
# Graduation — dynamic per-breath loss weights via softmax(EMA(pb_ce)/T).
# Maintenance floor ensures graduated breaths still get a refresher (no forgetting).
export V83_GRADUATION="${V83_GRADUATION:-1}"
export V83_GRAD_MAINTENANCE="${V83_GRAD_MAINTENANCE:-0.2}"
export V83_GRAD_EMA_ALPHA="${V83_GRAD_EMA_ALPHA:-0.1}"
export V83_GRAD_TEMP="${V83_GRAD_TEMP:-1.0}"

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

# ---- Training config — SMOKE (500 steps on v82 data, K=5 breaths) ----
export DEV=PCI+AMD
export LEVEL=GSM8K_STEPS
export GSM8K_STEPS_PATH="$V82_DATA"
export V77_TEST_PATH="$V82_DATA"
export GSM8K_STEPS_MIN_K=2
export GSM8K_STEPS_MAX_K=6
export SPACE_DIGITS=1
export BATCH="${BATCH:-4}"
export FIXED_LEN="${FIXED_LEN:-224}"
export STEPS="${STEPS:-500}"
export LR=3e-5
# K=5 breaths matches v82_smoke_k5_step500 warm-start. The schedule above has 5 entries.
export TRAIN_LOOPS="${TRAIN_LOOPS:-5}"
export EVAL_LOOPS="${EVAL_LOOPS:-5}"
export V77_N_LAYERS="${V77_N_LAYERS:-5}"
export ACC_EVAL_EVERY=10000
export SKIP_FINAL_ACC=1
export CKPT_EVERY="${CKPT_EVERY:-100}"
export NUM_EVAL=20
export NUM_PROBLEMS=20000
export EVAL_BATCH=4
export EVAL_CACHE_LEN="${EVAL_CACHE_LEN:-232}"
export LOOKUP_AUX_WEIGHT=0.0
export USE_JIT=1
export USE_KV_CACHE=1
export CKPT_LABEL="${CKPT_LABEL:-v83_smoke}"
export RESUME_FROM="$V83_INIT"

/home/bryce/mycelium/.venv/bin/python scripts/l3_train.py 2>&1 | tee .cache/v83_smoke_train.log
