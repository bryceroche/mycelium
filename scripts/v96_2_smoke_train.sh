#!/bin/bash
# v96.2 SMOKE — CONSTRAINT PROPAGATION on top of v96.1 consolidation table.
#
# Bombe-inspired elimination over generation:
#   - v96.1 generates DAGs via per-breath consolidation rows (ops/types/conf).
#   - v96.2 adds three SELF-SUPERVISED constraint losses that turn the breaths
#     into PARALLEL CONSTRAINT CHECKERS that eliminate inconsistent partial
#     assignments:
#         (A) per-breath temperature decay on ops/types logits (broad → sharp)
#         (B) ref_validity_head: model penalizes its own future-referencing rows
#         (C) arg_order_head:    model penalizes low arg-ordering plausibility
#                                weighted by non-commutative op prob
#         (D) calibration_loss:  per-breath confidence tracks expected correctness
#                                (proxy: sigmoid(-pb_ce[K-1] + 4) * (k/(K-1)))
#
# Total energy_loss = w_ref*ref + w_arg*arg + w_calib*calib
# total_loss = task_loss + V96_ENERGY_WEIGHT * energy_loss
#
# All v96.1 fixes preserved (zero-init table_kv_proj + v96_table_alpha).
#
# Success bar (smoke):
#   - Accuracy > 3% at any of step 100/200/300/500 (breakthrough from 0-1.7%)
#   - Per-breath CE ladder visible: B0 high → B6 low monotonically
#   - v96.2 metrics descend: ref_loss, arg_loss, calib_loss
#   - waist_norm stable < 10 (no v96.0-style positive-feedback explosion)

set -e
cd "$(dirname "$0")/.."

V96_INIT="${V96_INIT:-.cache/gsm8k_steps_ckpts/v66_sched_sampling_step3000.safetensors}"
if [ ! -f "$V96_INIT" ]; then
    # Fall back to v80_prod_step400 if v66 ckpt isn't present
    V96_INIT=".cache/gsm8k_steps_ckpts/v80_prod_step400.safetensors"
    echo "[v96.2] v66 ckpt not found; falling back to $V96_INIT"
fi
if [ ! -f "$V96_INIT" ]; then
    echo "ERROR: warm-start ckpt not found at $V96_INIT"
    exit 1
fi
V96_DATA="${V96_DATA:-.cache/gsm8k_steps_v80_train.jsonl}"
if [ ! -f "$V96_DATA" ]; then
    echo "ERROR: v80 train data not found at $V96_DATA"
    exit 1
fi

# ---- V96 CONSOLIDATION TABLE (v96.1 foundation, all params preserved) ----
export V96_CONSOLIDATION=1
export V96_LABEL_SMOOTHING_START="${V96_LABEL_SMOOTHING_START:-0.5}"
export V96_W_OPS="${V96_W_OPS:-1.0}"
export V96_W_TYPES="${V96_W_TYPES:-0.5}"
export V96_W_CONF="${V96_W_CONF:-0.1}"

# ---- V96.2 NEW KNOBS — CONSTRAINT PROPAGATION ----
# Per-breath temperature decay on artifact's ops/types logits.
export V96_TEMPERATURE_DECAY=1
export V96_T_START="${V96_T_START:-2.0}"   # T at B0 (broad)
export V96_T_END="${V96_T_END:-0.3}"       # T at B_{K-1} (sharp)

# Energy loss weight (bundle = ref + arg + calib).
export V96_ENERGY_WEIGHT="${V96_ENERGY_WEIGHT:-0.3}"
# Component weights inside the energy bundle.
export V96_W_REF="${V96_W_REF:-1.0}"
export V96_W_ARG="${V96_W_ARG:-1.0}"
export V96_W_CALIB="${V96_W_CALIB:-1.0}"
# Calibration target mode: "proxy" (cheap, uses pb_ce[K-1] as a fixed signal)
# vs "sympy" (training-time AR decode + SymPy execution — not implemented in
# this smoke; falls through to proxy).
export V96_CALIB_LOSS_MODE="${V96_CALIB_LOSS_MODE:-proxy}"

# ---- V77 DAG path ON (AR token generation, final breath) ----
export V77_DAG_TRAINING=1
export V77_N_LAYERS=7

# ---- V85 queryable structures OFF (AR not slot) ----
export V85_QUERYABLE=0

# ---- AR-CORRECT MASKING (2-mask paradigm per project_v81_four_mask_discovery) ----
export V79_CAUSAL_MASKS=1     # kv_mask + notebook_pool_mask (correct for AR)
export V81_MAIN_ATTN_MASK=0   # OFF — preserves AR feedback loop

# ---- Multi-head OFF (single-head AR decode for v96; v96 multi-head is the table itself) ----
export MULTI_HEAD_WAIST=0

# ---- v82-v95 paradigms OFF ----
export V82_PARALLEL_DIFFUSION=0
export V83_ANYTIME_SUPERVISION=0
export V83_GRADUATION=1
export V91_SIMPLIFIED_ARGS=0

export V87_REINIT_SLOT_POS=0
export V88_REINIT_KV_PROJ=0
export V92_REINIT_ARG_POS_EMB=0
export V92_RESET_ACTIVE_HEAD_NEUTRAL=0
export V90_RESET_ACTIVE_HEAD=0

# ---- v77b knobs ----
export BREATH_EMBED_ORTHO_INIT=2.0
export PER_BREATH_TEMP=1
export BREATH_NORM_OSC=1

# ---- v78 knobs — N=32 codebook (matches v96 types codebook size) ----
export MAX_STEP_BASE=2.0
export MAX_STEP_MIN=0.1
export NOTEBOOK_ACCUMULATE_ENABLED=1
export NOTEBOOK_NO_DETACH=1
export V78_HEAD_CODEBOOK=1
export V78_HEAD_CODEBOOK_N=32

# ---- WAIST_ATTN_SUPERVISION OFF for v96 smoke (v96 provides its own per-row supervision) ----
export CONTROLLER_N_LAYERS=4
export WAIST_ATTN_SUPERVISION=0
export WAIST_ATTN_AUX_WEIGHT=0.0
export V95_OPERAND_AUX=0

# ---- SCHED_SAMPLE off (regular JIT path; v96 SS not supported) ----
export SCHED_SAMPLE_RATE=0.0

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

# ---- LR with WARMUP (per v92 lesson: any new attention pathway benefits from warmup) ----
export LR_DECAY_TO_ZERO=1
export V92_LR_WARMUP_STEPS=50

# ---- Training config — SMOKE (500 steps; CKPT every 100; LR 3e-5 with warmup) ----
export DEV=PCI+AMD
export LEVEL=GSM8K_STEPS
export GSM8K_STEPS_PATH="$V96_DATA"
export V77_TEST_PATH="$V96_DATA"
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
export CKPT_LABEL="${CKPT_LABEL:-v96_2_smoke}"
export RESUME_FROM="$V96_INIT"

/home/bryce/mycelium/.venv/bin/python -u scripts/l3_train.py 2>&1 | tee .cache/v96_2_smoke_train.log
