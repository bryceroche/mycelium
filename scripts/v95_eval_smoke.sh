#!/bin/bash
# v95 SMOKE EVAL — run eval_v77_dag.py at smoke ckpts and summarize.
#
# Uses the same env config as v95_smoke_train.sh (so the model topology matches).
# Dumps every decode to a JSONL, then aggregates via v95_eval_summary.py.

set -e
cd "$(dirname "$0")/.."

CKPT_DIR=".cache/gsm8k_steps_ckpts"
CKPT_LABEL="${CKPT_LABEL:-v95_smoke}"
NUM_EVAL="${NUM_EVAL:-60}"
TEST_PATH="${TEST_PATH:-.cache/gsm8k_steps_v80_test.jsonl}"

# ---- Match the v80/v95 model topology (env vars read by model loader) ----
export V77_DAG_TRAINING=1
export V77_N_LAYERS=7
export V85_QUERYABLE=0
export V79_CAUSAL_MASKS=1
export V81_MAIN_ATTN_MASK=0
export MULTI_HEAD_WAIST=0
export V82_PARALLEL_DIFFUSION=0
export V83_ANYTIME_SUPERVISION=0
export V83_GRADUATION=1
export V91_SIMPLIFIED_ARGS=0
export V87_REINIT_SLOT_POS=0
export V88_REINIT_KV_PROJ=0
export V92_REINIT_ARG_POS_EMB=0
export V92_RESET_ACTIVE_HEAD_NEUTRAL=0
export V90_RESET_ACTIVE_HEAD=0
export BREATH_EMBED_ORTHO_INIT=2.0
export PER_BREATH_TEMP=1
export BREATH_NORM_OSC=1
export MAX_STEP_BASE=2.0
export MAX_STEP_MIN=0.1
export NOTEBOOK_ACCUMULATE_ENABLED=1
export NOTEBOOK_NO_DETACH=1
export V78_HEAD_CODEBOOK=1
export V78_HEAD_CODEBOOK_N=12
export CONTROLLER_N_LAYERS=4
export WAIST_ATTN_SUPERVISION=1
export WAIST_ATTN_AUX_WEIGHT=0.5
export V95_OPERAND_AUX=1
export V95_OPERAND_AUX_WEIGHT=0.5
export SCHED_SAMPLE_RATE=0.0
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
export DEV=PCI+AMD

# Eval-specific
export K=7
export FIXED_LEN=256
export BATCH=4
export MAX_NEW=120
export NUM_EVAL
export V77_TEST_PATH="$TEST_PATH"

# Iterate every ckpt with the given label (steps 50..300)
for STEP in 100 200 300; do
    CKPT="${CKPT_DIR}/${CKPT_LABEL}_step${STEP}.safetensors"
    if [ ! -f "$CKPT" ]; then
        echo "[skip] $CKPT not found"
        continue
    fi
    DUMP=".cache/${CKPT_LABEL}_step${STEP}_eval_dump.jsonl"
    rm -f "$DUMP"
    echo "=== eval $CKPT ==="
    CKPT="$CKPT" DUMP_ALL_DECODES="$DUMP" \
        /home/bryce/mycelium/.venv/bin/python -u scripts/eval_v77_dag.py 2>&1 | tail -40
    echo ""
    echo "=== summary $CKPT ==="
    /home/bryce/mycelium/.venv/bin/python scripts/v95_eval_summary.py "$DUMP"
    echo ""
done
