#!/usr/bin/env bash
# V24c notebook component ablation. Disables individual notebooks at eval time
# to isolate which one is load-bearing.
#
# Configs:
#   BOTH         — both notebooks active (training config)
#   ACC_ONLY     — accumulate only (NOTEBOOK_DUAL=0)
#   REP_ONLY     — replace only (NOTEBOOK_DUAL=1, NOTEBOOK_ACCUMULATE_ENABLED=0)
#   NEITHER      — both off (NOTEBOOK_V24=0) — baseline
#
# Usage: bash scripts/v24c_notebook_ablation.sh <ckpt_path>
# Default ckpt: v24c step 500 champion (96/94/91 on L4_MIXED).

set -euo pipefail

CKPT="${1:-/home/bryce/mycelium/.cache/l4_mixed_ckpts/l4_mixed_v24c_dual_notebook_step500.safetensors}"
if [ ! -f "$CKPT" ]; then
    echo "ckpt not found: $CKPT"; exit 1
fi
LOOPS="${LOOPS:-1,4,8}"
NUM_EVAL="${NUM_EVAL:-100}"
LEVEL="${LEVEL:-L4_MIXED}"

PY=".venv/bin/python"
SCRIPT="scripts/eval_l4.py"

COMMON="DEV=PCI+AMD PER_HEAD_PITCH=1 CONSTANT_RADIUS=1 CROSS_BREATH_HANDOFF=1 \
BREATH_TIME_EMBED=1 BREATH_TIME_INIT_SCALE=0.0 ABLATE_BREATH_ROTATION=1 \
SINE_TEMP=1 SINE_TEMP_MAX=2.0 SINE_TEMP_MIN=0.7 SPACE_DIGITS=1 \
NOTEBOOK_INIT_SCALE=0.02 NOTEBOOK_POOL_MODE=attn \
LEVEL=$LEVEL LOOPS=$LOOPS NUM_EVAL=$NUM_EVAL CKPT=$CKPT"

echo "==================================================================="
echo "V24c notebook ablation: $(basename $CKPT)"
echo "level=$LEVEL loops=$LOOPS num_eval=$NUM_EVAL"
echo "==================================================================="

run_config() {
    local name="$1"
    local env_extra="$2"
    echo ""
    echo "==================================================================="
    echo "[$name] $env_extra"
    echo "==================================================================="
    env $COMMON $env_extra $PY $SCRIPT 2>&1 | grep -E "acc @|notebook|Error|Traceback" | head -20
}

run_config "BOTH"     "NOTEBOOK_V24=1 NOTEBOOK_DUAL=1 NOTEBOOK_ACCUMULATE_ENABLED=1"
run_config "ACC_ONLY" "NOTEBOOK_V24=1 NOTEBOOK_DUAL=0 NOTEBOOK_ACCUMULATE_ENABLED=1"
run_config "REP_ONLY" "NOTEBOOK_V24=1 NOTEBOOK_DUAL=1 NOTEBOOK_ACCUMULATE_ENABLED=0"
run_config "NEITHER"  "NOTEBOOK_V24=0"

echo ""
echo "==================================================================="
echo "DONE"
echo "==================================================================="
