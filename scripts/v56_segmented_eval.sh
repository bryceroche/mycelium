#!/bin/bash
# Segmented decode eval: each breath decodes its own step's tokens.
# Same env as v56 training, but with the segmented eval script.
#
# Usage:
#   scripts/v56_segmented_eval.sh <ckpt_path> [K] [LEVEL] [FIXED_LEN] [MAX_NEW]
# Defaults: K=3, LEVEL=L4.5, FIXED_LEN=192, MAX_NEW=80

set -e
cd "$(dirname "$0")/.."

# v55/v56 architecture
export CONTROLLER_DECODE=1
export CONTROLLER_N_LAYERS=1
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
export ACROSS_LAYER_PITCH_TARGET=0
export PER_LAYER_OFFSETS_RADIANS=""

# Eval params
export DEV=PCI+AMD
export NUM_EVAL=100
export BATCH=16
export CKPT="${1:?usage: $0 <ckpt_path> [K] [LEVEL] [FIXED_LEN] [MAX_NEW]}"
export K="${2:-3}"
export LEVEL="${3:-L4.5}"
export FIXED_LEN="${4:-192}"
export MAX_NEW="${5:-80}"

/home/bryce/mycelium/.venv/bin/python scripts/eval_ckpt_controller_segmented.py
