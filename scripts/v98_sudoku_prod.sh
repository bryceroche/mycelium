#!/bin/bash
# v98 sudoku PROD training launcher (Phase 4).
#
# Activated ONLY if the smoke (v98_sudoku_smoke.sh) hit the success bar:
#   - cell or puzzle accuracy > 50% on easy at step 1000
#   - per-breath CE ladder visible
#
# This launcher trains with the full curriculum:
#   - Starts 100% easy, anneals to uniform across difficulties over 1000 steps
#   - Resumes from the smoke ckpt (v98_smoke_step1000.safetensors) by default
#   - 5000 steps, K=20 (deeper than smoke for harder puzzles)
#   - Target: >80% puzzle acc on expert at step 5000 (per spec)

set -e
cd "$(dirname "$0")/.."

# ---- DATA ----
export SUDOKU_TRAIN="${SUDOKU_TRAIN:-.cache/sudoku_train.jsonl}"
export SUDOKU_VAL="${SUDOKU_VAL:-.cache/sudoku_val.jsonl}"
if [ ! -f "$SUDOKU_TRAIN" ]; then
    echo "ERROR: $SUDOKU_TRAIN not found"
    exit 1
fi

# ---- ARCH ----
export SUDOKU_TASK=1
export SUDOKU_K_MAX="${SUDOKU_K_MAX:-20}"
export SUDOKU_CONSTRAINT_WEIGHT="${SUDOKU_CONSTRAINT_WEIGHT:-0.005}"
export SUDOKU_CALIB_WEIGHT="${SUDOKU_CALIB_WEIGHT:-0.05}"
export SUDOKU_BREATH_EMBED_SCALE="${SUDOKU_BREATH_EMBED_SCALE:-0.5}"

# Full curriculum (annealed) instead of easy-only filter
unset SUDOKU_DIFFICULTY_FILTER 2>/dev/null || true
export SUDOKU_CURRICULUM=1
export SUDOKU_CURRICULUM_ANNEAL_STEPS="${SUDOKU_CURRICULUM_ANNEAL_STEPS:-1500}"

# ---- BACKBONE — Pythia warm-init still ----
export PYTHIA_INIT=1

# Quiet the dormant GSM8K paths.
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
export STEPS="${STEPS:-5000}"
export LR="${LR:-3e-5}"

# ---- LOGGING + CKPT ----
export CKPT_EVERY="${CKPT_EVERY:-500}"
export EVAL_EVERY="${EVAL_EVERY:-250}"
export LOG_EVERY="${LOG_EVERY:-20}"
export PER_BREATH_CE_EVERY="${PER_BREATH_CE_EVERY:-100}"
export GC_EVERY="${GC_EVERY:-50}"
export EVAL_BATCHES="${EVAL_BATCHES:-40}"
export EVAL_BATCH="${EVAL_BATCH:-8}"

export CKPT_LABEL="${CKPT_LABEL:-v98_prod}"

# ---- WARM-START FROM SMOKE CKPT ----
# Spec: use the step 400 ckpt — smoke was killed at 400 due to perf bug, not 1000.
export RESUME_FROM="${RESUME_FROM:-.cache/sudoku_ckpts/v98_smoke_step400.safetensors}"
if [ ! -f "$RESUME_FROM" ]; then
    echo "Warning: warm-start ckpt $RESUME_FROM not found. Will cold-start."
    unset RESUME_FROM
fi

mkdir -p .cache

LOG_FILE="${LOG_FILE:-.cache/v98_sudoku_prod.log}"
echo "logging to: $LOG_FILE"
/home/bryce/mycelium/.venv/bin/python -u scripts/sudoku_train.py 2>&1 | tee "$LOG_FILE"
