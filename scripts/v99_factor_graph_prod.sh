#!/usr/bin/env bash
# v99 factor graph — production run (2000 steps, full curriculum)
# Usage: bash scripts/v99_factor_graph_prod.sh
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-.venv/bin/python}"

# NOTE: K_MAX=20 crashes AMD AM driver (JIT graph too large). K=10 is working limit.
# To use K=20: need per-breath graph decomposition or reduced backward complexity.
DEV='PCI+AMD' \
V99_TASK=1 \
V99_K_MAX=10 \
V99_N_MAX=16 \
V99_F_MAX=8 \
V99_ENERGY_WEIGHT=0.1 \
V99_CALIB_WEIGHT=0.05 \
V99_CURRICULUM=1 \
V99_CURRICULUM_ANNEAL=1000 \
V99_TRAIN=.cache/factor_graph_train.jsonl \
V99_VAL=.cache/factor_graph_test.jsonl \
BATCH=8 \
STEPS=2000 \
LR=3e-5 \
LOG_EVERY=10 \
PER_BREATH_CE_EVERY=50 \
EVAL_EVERY=250 \
EVAL_BATCHES=20 \
EVAL_BATCH=8 \
CKPT_EVERY=500 \
CKPT_LABEL=v99_prod \
PYTHIA_INIT=1 \
SUDOKU_TASK=0 \
"$PYTHON" -u scripts/v99_factor_graph_train.py
