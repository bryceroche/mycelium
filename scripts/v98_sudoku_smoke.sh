#!/bin/bash
# v98 sudoku smoke training launcher.
#
# Trains the breathing transformer on EASY-only sudoku puzzles for 1000 steps.
# Tests:
#   - Per-breath CE ladder forms (B0 high → B_K-1 low)
#   - Cell + puzzle accuracy emerges (target: >50% puzzle acc on easy at step 1000)
#   - Constraint energy descends to <0.5 per puzzle
#   - No divergence (loss never explodes)
#
# Phase 3 of the v98 strategic pivot (see memory/project_v98_sudoku_spec.md).
# After this smoke validates, Phase 4 will train on the full curriculum.

set -e
cd "$(dirname "$0")/.."

# ---- DATA ----
export SUDOKU_TRAIN="${SUDOKU_TRAIN:-.cache/sudoku_train.jsonl}"
export SUDOKU_VAL="${SUDOKU_VAL:-.cache/sudoku_val.jsonl}"
if [ ! -f "$SUDOKU_TRAIN" ]; then
    echo "ERROR: sudoku train data not found at $SUDOKU_TRAIN"
    echo "Run: python scripts/build_sudoku_data.py --out $SUDOKU_TRAIN --n 50000 --seed 42 --workers 8"
    exit 1
fi
if [ ! -f "$SUDOKU_VAL" ]; then
    echo "ERROR: sudoku val data not found at $SUDOKU_VAL"
    exit 1
fi

# ---- SUDOKU-SPECIFIC ----
export SUDOKU_TASK=1
export SUDOKU_K_MAX="${SUDOKU_K_MAX:-30}"
export SUDOKU_CONSTRAINT_WEIGHT="${SUDOKU_CONSTRAINT_WEIGHT:-0.3}"
export SUDOKU_CALIB_WEIGHT="${SUDOKU_CALIB_WEIGHT:-0.1}"
export SUDOKU_DIFFICULTY_FILTER="${SUDOKU_DIFFICULTY_FILTER:-easy}"

# ---- BACKBONE — Pythia warm-init, no special breathing extras (sudoku layer
# has its own attention path, doesn't use RoPE / sine temp / waist / etc) ----
export PYTHIA_INIT=1

# Ensure GSM8K-era flags are OFF so they don't accidentally hijack control flow
# (sudoku forward doesn't call them, but the model still allocates them — keep
# the dormant code paths quiet).
export V77_DAG_TRAINING=0
export V85_QUERYABLE=0
export V96_CONSOLIDATION=0
export V97_CALIBRATION=0
export NOTEBOOK_V24=0
export NOTEBOOK_DAG=0
export BFIELD_WAIST=0
export COLLAPSE_V69=0
export COLLAPSE_V70=0
export COLLAPSE_V71=0

# ---- DEVICE + PERF ----
export DEV="${DEV:-PCI+AMD}"
export BATCH="${BATCH:-8}"
export STEPS="${STEPS:-1000}"
export LR="${LR:-3e-5}"

# ---- LOGGING + CKPT ----
export CKPT_EVERY="${CKPT_EVERY:-200}"
export EVAL_EVERY="${EVAL_EVERY:-100}"
export LOG_EVERY="${LOG_EVERY:-10}"
export PER_BREATH_CE_EVERY="${PER_BREATH_CE_EVERY:-50}"
export GC_EVERY="${GC_EVERY:-50}"
export EVAL_BATCHES="${EVAL_BATCHES:-20}"
export EVAL_BATCH="${EVAL_BATCH:-8}"

export CKPT_LABEL="${CKPT_LABEL:-v98_smoke}"

# ---- OPTIONAL WARM-START ----
# export RESUME_FROM=".cache/sudoku_ckpts/v98_smoke_step1000.safetensors"

mkdir -p .cache

LOG_FILE="${LOG_FILE:-.cache/v98_sudoku_smoke.log}"
echo "logging to: $LOG_FILE"
/home/bryce/mycelium/.venv/bin/python -u scripts/sudoku_train.py 2>&1 | tee "$LOG_FILE"
