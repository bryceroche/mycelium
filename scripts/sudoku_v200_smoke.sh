#!/bin/bash
# v200-Sudoku smoke launcher: 500 steps, EASY-only.
#
# Two backbones selectable via V200_SUDOKU_BASE (default: smollm2_1_7b):
#
#   SmolLM2-1.7B (default):
#     V200_SUDOKU_BASE=smollm2_1_7b BATCH=8 STEPS=500 bash scripts/sudoku_v200_smoke.sh
#     Hidden=2048, 32 heads, mask 10+10+10+2, backbone ~268M params
#     Expected: ~2-4s/step after JIT compile (~5-10 min on AMD 7900 XTX)
#
#   Pythia-410M (4-5× lighter, matches v98):
#     V200_SUDOKU_BASE=pythia_410m BATCH=12 STEPS=500 bash scripts/sudoku_v200_smoke.sh
#     Hidden=1024, 16 heads, mask 5+5+5+1, backbone ~50M params
#     Expected: ~10-12s/step after JIT compile (at K=20, BATCH=12)
#     CPU micro-smoke (K=1): ~30s for 5 steps (no JIT needed)
#
# Tests:
#   - Backbone L0-L3 weights load cleanly
#   - Forward produces finite cell logits (B, 81, 9)
#   - Per-breath CE ladder forms over K=20 (B0 > B_K-1)
#   - Cell accuracy on easy climbs above 0.30 by step 500
#   - No divergence (loss finite throughout)
#
# Smoke criterion for "v98 paradigm transfers":
#   cell_acc > 0.30 on easy at step 500  (chance=11%; v98 at step 500 was ~0.50)

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

# ---- BACKBONE SELECTION ----
# Default: smollm2_1_7b.  Override: V200_SUDOKU_BASE=pythia_410m
export V200_SUDOKU_BASE="${V200_SUDOKU_BASE:-smollm2_1_7b}"

# ---- V200-SUDOKU FLAGS ----
export V200_SUDOKU_TASK=1
export V200_SUDOKU_K_MAX="${V200_SUDOKU_K_MAX:-20}"
export V200_SUDOKU_CONSTRAINT_WEIGHT="${V200_SUDOKU_CONSTRAINT_WEIGHT:-0.3}"
export V200_SUDOKU_CALIB_WEIGHT="${V200_SUDOKU_CALIB_WEIGHT:-0.1}"
export DIFFICULTY_FILTER="${DIFFICULTY_FILTER:-easy}"

# ---- DEVICE + PERF ----
export DEV="${DEV:-PCI+AMD}"
export BATCH="${BATCH:-8}"
export STEPS="${STEPS:-500}"
export LR="${LR:-3e-5}"

# ---- LOGGING + CKPT ----
export CKPT_EVERY="${CKPT_EVERY:-500}"
export EVAL_EVERY="${EVAL_EVERY:-100}"
export LOG_EVERY="${LOG_EVERY:-10}"
export PER_BREATH_CE_EVERY="${PER_BREATH_CE_EVERY:-50}"
export GC_EVERY="${GC_EVERY:-50}"
export EVAL_BATCHES="${EVAL_BATCHES:-20}"
export EVAL_BATCH="${EVAL_BATCH:-8}"
export CKPT_LABEL="${CKPT_LABEL:-sudoku_v200_smoke_${V200_SUDOKU_BASE}}"

# Optional warm-start:
# export RESUME_FROM=".cache/sudoku_v200_ckpts/sudoku_v200_smoke_pythia_410m_final.safetensors"

# Optional SmolLM2 override (default: auto-resolved from .cache/llama-3.2-1b/):
# export LLAMA_WEIGHTS="/path/to/smollm2/model.safetensors"

mkdir -p .cache .cache/sudoku_v200_ckpts

LOG_FILE="${LOG_FILE:-.cache/sudoku_v200_smoke_${V200_SUDOKU_BASE}.log}"
echo "backbone: $V200_SUDOKU_BASE"
echo "logging to: $LOG_FILE"
/home/bryce/mycelium/.venv/bin/python -u scripts/sudoku_v200_train.py 2>&1 | tee "$LOG_FILE"
