#!/usr/bin/env bash
# v105 factor graph — smoke run (50 steps, fresh random init, no warm-start)
#
# Architecture: digit-by-digit codebook (10-way per digit, N_DIGITS=5 positions)
# Key difference from v104: no 100-way domain codebook; 10-way digit codebook shared
# across all N_DIGITS positions and all N_VARS variable slots.
# T = N_MAX * N_DIGITS + F_MAX = 16*5+8 = 88  (vs 24 in v100)
#
# Acceptance criteria:
#   1. No NaN (isfinite check passes every step)
#   2. Loss decreases over 50 steps
#   3. Step time ≤ 3.0s/step  (T=88 is larger than v100's T=24)
#   4. Per-breath CE ladder forms (B0 > B{K-1}, delta ≥ 0.1)
#   5. Per-digit accuracy on observed variables ≥ 95% (should trivially pass via aligned init)
#   6. Per-digit accuracy on unobserved variables > 10% after 50 steps
#
# Note: NO warm-start (RESUME_FROM unset) — v105 has different parameter shapes.
#       PYTHIA_INIT=1 still loads Pythia transformer backbone.
#
# Usage:
#   bash scripts/v105_factor_graph_smoke.sh
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-.venv/bin/python}"

DEV='PCI+AMD' \
V105_TASK=1 \
V105_K_MAX=8 \
V105_N_DIGITS=5 \
V105_N_MAX=16 \
V105_F_MAX=8 \
V105_ENERGY_WEIGHT=0.01 \
V105_FACTOR_AUX_WEIGHT=0.5 \
V105_CALIB_WEIGHT=0.05 \
V105_DIFFICULTY_FILTER=easy \
V105_TRAIN=.cache/factor_graph_train.jsonl \
V105_VAL=.cache/factor_graph_test.jsonl \
V105_GSM8K_TRAIN=.cache/gsm8k_factor_graphs_train.jsonl \
V105_GSM8K_RATIO=0.3 \
BATCH=4 \
STEPS=50 \
LR=3e-5 \
LOG_EVERY=5 \
PER_BREATH_CE_EVERY=10 \
EVAL_EVERY=25 \
EVAL_BATCHES=5 \
EVAL_BATCH=4 \
CKPT_EVERY=500 \
CKPT_LABEL=v105_smoke \
PYTHIA_INIT=1 \
SUDOKU_TASK=0 \
V99_TASK=0 \
V100_TASK=0 \
"$PYTHON" -u scripts/v105_factor_graph_train.py
