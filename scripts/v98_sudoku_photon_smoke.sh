#!/usr/bin/env bash
# v98 Sudoku + photon mechanism smoke.
#
# Adds coherent E+B photon on top of v98 (BreathingTransformer + Pythia-410M
# L0-L3 + shared L0 wv). E-field: per-breath Q rotation phase=N·k·π/K_max.
# B-field: sin²(N·k·π/K_max) gate × (x @ W_compress @ W_expand). W_expand
# is zero-init so step 0 is byte-identical to plain v98 even with the photon
# attached.
#
# Frequency coupling: V98_PHOTON_FREQ_MULT controls BOTH fields (coherent
# photon). Setting V98_PHOTON_ALPHA=0 skips the photon entirely.
#
# Warm-start from v98_prod_final (97.65% cell / 79% puzzle on easy).
# Goal: do val[easy/medium] survive the photon addition? Does freq sweep
# show a resonant frequency that lifts hard?
set -euo pipefail

cd "$(dirname "$0")/.."

# ---- DATA ----
export SUDOKU_TRAIN="${SUDOKU_TRAIN:-.cache/sudoku_train.jsonl}"
export SUDOKU_VAL="${SUDOKU_VAL:-.cache/sudoku_val.jsonl}"
if [ ! -f "$SUDOKU_TRAIN" ]; then
    echo "ERROR: sudoku train data not found at $SUDOKU_TRAIN"
    exit 1
fi

# ---- v98 core (matches scripts/v98_sudoku_smoke.sh validated values) ----
export SUDOKU_TASK=1
export SUDOKU_K_MAX="${SUDOKU_K_MAX:-20}"
export SUDOKU_CONSTRAINT_WEIGHT="${SUDOKU_CONSTRAINT_WEIGHT:-0.005}"
export SUDOKU_CALIB_WEIGHT="${SUDOKU_CALIB_WEIGHT:-0.05}"
export SUDOKU_DIFFICULTY_FILTER="${SUDOKU_DIFFICULTY_FILTER:-easy}"
export PYTHIA_INIT=1

# ---- Quiesce dormant non-sudoku flags ----
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

# ---- Photon mechanism (the new bit) ----
export SUDOKU_PHOTON_ENABLE="${SUDOKU_PHOTON_ENABLE:-1}"
export SUDOKU_PHOTON_ALPHA="${SUDOKU_PHOTON_ALPHA:-1.0}"
export SUDOKU_PHOTON_FREQ_MULT="${SUDOKU_PHOTON_FREQ_MULT:-1.0}"
export SUDOKU_PHOTON_WAIST_DIM="${SUDOKU_PHOTON_WAIST_DIM:-256}"

# ---- DEVICE + PERF ----
export DEV="${DEV:-PCI+AMD}"
export BATCH="${BATCH:-12}"
export STEPS="${STEPS:-500}"
export LR="${LR:-3e-5}"

# ---- LOGGING + CKPT ----
export CKPT_EVERY="${CKPT_EVERY:-250}"
export EVAL_EVERY="${EVAL_EVERY:-100}"
export LOG_EVERY="${LOG_EVERY:-10}"
export PER_BREATH_CE_EVERY="${PER_BREATH_CE_EVERY:-50}"
export GC_EVERY="${GC_EVERY:-50}"
export EVAL_BATCHES="${EVAL_BATCHES:-20}"
export EVAL_BATCH="${EVAL_BATCH:-12}"

# Warm-start from v98 prod (the trained ckpt). Missing photon keys are
# tolerated by load_ckpt (kept at the orthonormal/zero init from attach).
export RESUME_FROM="${RESUME_FROM:-.cache/sudoku_ckpts/v98_prod_final.safetensors}"

export CKPT_LABEL="${CKPT_LABEL:-v98_photon_smoke}"

mkdir -p .cache .cache/logs

LOG_FILE="${LOG_FILE:-.cache/logs/${CKPT_LABEL}.log}"
echo "v98 + photon: alpha=$SUDOKU_PHOTON_ALPHA freq=$SUDOKU_PHOTON_FREQ_MULT waist_dim=$SUDOKU_PHOTON_WAIST_DIM"
echo "warm-start:  $RESUME_FROM"
echo "logging to:  $LOG_FILE"
/home/bryce/mycelium/.venv/bin/python -u scripts/sudoku_train.py 2>&1 | tee "$LOG_FILE"
