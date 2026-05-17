#!/usr/bin/env bash
# V24 component ablation eval. Runs the standard L4_MIXED accuracy eval on a
# specific ckpt with each of v24's three photon mechanisms toggled OFF, isolating
# which is helping or hurting at the regression point.
#
# Usage: bash scripts/v24_ablate_eval.sh <ckpt_path> [LOOPS=1,4,8] [NUM_EVAL=100]
#
# Output: 5 acc readouts:
#   FULL       — all three v24 components ON (matches training config)
#   NO_TEMP    — temp wave OFF (uses across-breath SINE_TEMP baseline)
#   NO_NORM    — norm oscillation OFF (uses end-of-breath CRP only)
#   NO_NB      — notebook OFF (zero contribution, photon read/write skipped)
#   ALL_OFF    — v23a-equivalent (per-head pitch only, no photon)
#
# Each run does fresh JIT compile (~30s overhead per config).

set -euo pipefail

CKPT="${1:-}"
if [ -z "$CKPT" ] || [ ! -f "$CKPT" ]; then
    echo "usage: bash $0 <ckpt_path>"
    echo "ckpt not found: $CKPT"
    exit 1
fi
LOOPS="${LOOPS:-1,4,8}"
NUM_EVAL="${NUM_EVAL:-100}"
LEVEL="${LEVEL:-L4_MIXED}"

PY=".venv/bin/python"
SCRIPT="scripts/eval_l4.py"

# All v24 configs share the v23a + warm-start config baseline.
COMMON_ENV="DEV=PCI+AMD PER_HEAD_PITCH=1 CONSTANT_RADIUS=1 CROSS_BREATH_HANDOFF=1 \
BREATH_TIME_EMBED=1 BREATH_TIME_INIT_SCALE=0.0 ABLATE_BREATH_ROTATION=1 \
SINE_TEMP=1 SINE_TEMP_MAX=2.0 SINE_TEMP_MIN=0.7 SPACE_DIGITS=1 \
LEVEL=$LEVEL LOOPS=$LOOPS NUM_EVAL=$NUM_EVAL CKPT=$CKPT"

echo "==================================================================="
echo "V24 ablation eval on $(basename $CKPT)"
echo "level=$LEVEL loops=$LOOPS num_eval=$NUM_EVAL"
echo "==================================================================="

run_config() {
    local name="$1"
    local extra_env="$2"
    echo ""
    echo "==================================================================="
    echo "[$name] $extra_env"
    echo "==================================================================="
    env $COMMON_ENV $extra_env $PY $SCRIPT 2>&1 | grep -E "acc @|error|Error" | head -20
}

run_config "FULL"     "PER_BREATH_TEMP=1 BREATH_NORM_OSC=1 NOTEBOOK_V24=1"
run_config "NO_TEMP"  "PER_BREATH_TEMP=0 BREATH_NORM_OSC=1 NOTEBOOK_V24=1"
run_config "NO_NORM"  "PER_BREATH_TEMP=1 BREATH_NORM_OSC=0 NOTEBOOK_V24=1"
run_config "NO_NB"    "PER_BREATH_TEMP=1 BREATH_NORM_OSC=1 NOTEBOOK_V24=0"
run_config "ALL_OFF"  "PER_BREATH_TEMP=0 BREATH_NORM_OSC=0 NOTEBOOK_V24=0"

echo ""
echo "==================================================================="
echo "DONE — compare acc across configs to isolate the load-bearing piece"
echo "==================================================================="
