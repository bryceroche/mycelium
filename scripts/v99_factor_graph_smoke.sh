#!/usr/bin/env bash
# v99 factor graph — smoke run (50 steps, easy only, fast verification)
# Usage: bash scripts/v99_factor_graph_smoke.sh
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-.venv/bin/python}"

# NOTE: K_MAX=20 crashes the AMD AM driver (JIT graph too large for K=20×4-layer×T=24 backward).
# K=10 is the working limit. Bryce should investigate graph decomposition before using K=20.
DEV='PCI+AMD' \
V99_TASK=1 \
V99_K_MAX=10 \
V99_N_MAX=16 \
V99_F_MAX=8 \
V99_ENERGY_WEIGHT=0.1 \
V99_CALIB_WEIGHT=0.05 \
V99_DIFFICULTY_FILTER=easy \
V99_TRAIN=.cache/factor_graph_train.jsonl \
V99_VAL=.cache/factor_graph_test.jsonl \
BATCH=4 \
STEPS=50 \
LR=3e-5 \
LOG_EVERY=5 \
PER_BREATH_CE_EVERY=10 \
EVAL_EVERY=25 \
EVAL_BATCHES=5 \
EVAL_BATCH=4 \
CKPT_EVERY=500 \
CKPT_LABEL=v99_smoke \
PYTHIA_INIT=1 \
SUDOKU_TASK=0 \
"$PYTHON" -u scripts/v99_factor_graph_train.py
