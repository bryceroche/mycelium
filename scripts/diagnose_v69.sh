#!/bin/bash
# Segment-count diagnostic dump for v69.
# Dumps 100 examples to JSONL then runs analyze_gsm8k_failures.py.
#
# Compares to v66 baseline metrics:
#   Avg segments emitted: K=4 → 2.10, K=5 → 2.39, K=6 → 1.83
#   Name preservation in originals: 33/86 = 38%
#   Repetition collapse: 4/100
#
# Usage:
#   CKPT=.cache/gsm8k_steps_ckpts/v69_collapse_step3000.safetensors \
#     bash scripts/diagnose_v69.sh

set -e
cd "$(dirname "$0")/.."

CKPT="${CKPT:-.cache/gsm8k_steps_ckpts/v69_collapse_step3000.safetensors}"

if [ ! -f "$CKPT" ]; then
    echo "ERROR: ckpt not found: $CKPT"
    exit 1
fi

DIAG_OUT="${DIAG_OUT:-/tmp/v69_diag.jsonl}"

# --- v69 architecture ---
export COLLAPSE_V69=1
export COLLAPSE_WAIST_DIM=128
export COLLAPSE_CODEBOOK_N=256
export COLLAPSE_TAU=1.0
export COLLAPSE_GATE_BIAS=2.0
export COLLAPSE_ENTROPY_REG=0.01

export TWO_PHASE=0
export NOTEBOOK_DAG=0
export PROMPT_REFRESH_ALPHA=0.1
export BOUNDARY_AUX_WEIGHT=0.1
export CONTROLLER_DECODE=1
export CONTROLLER_N_LAYERS=2
export PER_BREATH_DECODE=1
export BFIELD_WAIST=512
export BFIELD_END_OF_BREATH=1
export BFIELD_ENFORCED=0
export BFIELD_ALPHA=1.0
export WAIST_CODEBOOK_N=64
export WAIST_CODEBOOK_INJECT_WEIGHT=1.0
export NOTEBOOK_V24=1
export NOTEBOOK_ACCUMULATE_ENABLED=0
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
export SCHED_SAMPLE_RATE=0.3
export BOUNDARY_POS_WEIGHT=5.0

export DEV=PCI+AMD
export LEVEL=GSM8K_STEPS
export GSM8K_STEPS_PATH=.cache/gsm8k_steps_v1_test.jsonl
export GSM8K_STEPS_MIN_K=2
export GSM8K_STEPS_MAX_K=6
export NUM_EVAL=100
export BATCH=2
export FIXED_LEN=400
export MAX_NEW=120
export USE_KV_CACHE=0
export CKPT="$CKPT"
export DIAG_OUT="$DIAG_OUT"

echo "=== v69 segment-count diagnostic dump ==="
echo "  ckpt: $CKPT"
echo "  output: $DIAG_OUT"
echo ""

/home/bryce/mycelium/.venv/bin/python scripts/diagnose_gsm8k_failures.py 2>&1 | tee /tmp/v69_diag.log

echo ""
echo "=== analyzing failure categories ==="
/home/bryce/mycelium/.venv/bin/python scripts/analyze_gsm8k_failures.py "$DIAG_OUT"

echo ""
echo "=== v66 baseline for comparison ==="
echo "  Avg segments emitted: K=4→2.10, K=5→2.39, K=6→1.83"
echo "  Name preservation: 33/86 = 38%"
echo "  Repetition collapse: 4/100"
