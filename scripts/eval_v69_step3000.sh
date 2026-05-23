#!/bin/bash
# Eval v69 step 3000 on GSM8K test, 300 examples.
# v69 = v66 + JPEG-inspired collapse pipeline (codebook + gate + proj_down/up).
# USE_KV_CACHE=0 because Stage 1 JIT is not yet v69-aware (v69 changes waist shape).
#
# Usage:
#   CKPT=.cache/gsm8k_steps_ckpts/v69_collapse_step3000.safetensors \
#     bash scripts/eval_v69_step3000.sh
#
# Override ckpt with the CKPT env var; default assumes step 3000 label.

set -e
cd "$(dirname "$0")/.."

CKPT="${CKPT:-.cache/gsm8k_steps_ckpts/v69_collapse_step3000.safetensors}"

if [ ! -f "$CKPT" ]; then
    echo "ERROR: ckpt not found: $CKPT"
    echo "Set CKPT= to the correct path, e.g.:"
    echo "  CKPT=.cache/gsm8k_steps_ckpts/v69_collapse_step2500.safetensors bash $0"
    exit 1
fi

# --- v69 architecture (must match training flags in v69_collapse.sh) ---
# New collapse pipeline
export COLLAPSE_V69=1
export COLLAPSE_WAIST_DIM=128
export COLLAPSE_CODEBOOK_N=256
export COLLAPSE_TAU=1.0
export COLLAPSE_GATE_BIAS=2.0
export COLLAPSE_ENTROPY_REG=0.01

# v66 base architecture (preserved in v69)
export TWO_PHASE=0
export NOTEBOOK_DAG=0
export PROMPT_REFRESH_ALPHA=0.1
export BOUNDARY_AUX_WEIGHT=0.1
export CONTROLLER_DECODE=1
export CONTROLLER_N_LAYERS=2
export PER_BREATH_DECODE=1
export BFIELD_WAIST=512          # legacy shape for state-dict compat; unused at runtime
export BFIELD_END_OF_BREATH=1    # ignored when COLLAPSE_V69=1
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

# Eval settings
export DEV=PCI+AMD
export LEVEL=GSM8K_STEPS
export GSM8K_STEPS_PATH=.cache/gsm8k_steps_v1_test.jsonl
export GSM8K_STEPS_MIN_K=2
export GSM8K_STEPS_MAX_K=6
export SPACE_DIGITS=1
export NUM_EVAL=300
export BATCH=2
export FIXED_LEN=400
export MAX_NEW=120
# KV cache OFF: Stage 1 JIT not yet v69-aware (collapse waist dim changed 512→128)
export USE_KV_CACHE=0
export CKPT="$CKPT"

echo "=== v69 eval ==="
echo "  ckpt: $CKPT"
echo "  USE_KV_CACHE=0 (Stage 1 JIT not v69-aware — expected ~3× slower than v66 cached)"
echo ""

/home/bryce/mycelium/.venv/bin/python scripts/eval_ckpt_controller_segmented.py 2>&1 | tee /tmp/v69_eval.log
