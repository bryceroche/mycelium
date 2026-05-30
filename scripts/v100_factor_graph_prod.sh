#!/usr/bin/env bash
# v100 factor graph — production run (2000 steps, full curriculum)
# Usage: bash scripts/v100_factor_graph_prod.sh
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-.venv/bin/python}"

# K=10: AMD JIT working limit (same as v99).
# Curriculum: easy-only for 0-1500 steps, then anneal to full distribution.
DEV='PCI+AMD' \
V100_TASK=1 \
V100_K_MAX=10 \
V100_N_MAX=16 \
V100_F_MAX=8 \
V100_CALIB_WEIGHT=0.05 \
V100_FACTOR_AUX_WEIGHT=0.5 \
V100_CURRICULUM=1 \
V100_CURRICULUM_ANNEAL=1500 \
V100_TRAIN=.cache/factor_graph_train.jsonl \
V100_VAL=.cache/factor_graph_test.jsonl \
BATCH="${BATCH:-8}" \
STEPS="${STEPS:-2000}" \
LR="${LR:-3e-5}" \
LOG_EVERY="${LOG_EVERY:-10}" \
PER_BREATH_CE_EVERY="${PER_BREATH_CE_EVERY:-50}" \
KL_DIAG_EVERY="${KL_DIAG_EVERY:-100}" \
EVAL_EVERY="${EVAL_EVERY:-250}" \
EVAL_BATCHES="${EVAL_BATCHES:-20}" \
EVAL_BATCH="${EVAL_BATCH:-8}" \
CKPT_EVERY="${CKPT_EVERY:-500}" \
CKPT_LABEL="${CKPT_LABEL:-v100_prod}" \
RESUME_FROM="${RESUME_FROM:-}" \
PYTHIA_INIT=1 \
SUDOKU_TASK=0 \
V99_TASK=0 \
"$PYTHON" -u scripts/v100_factor_graph_train.py
