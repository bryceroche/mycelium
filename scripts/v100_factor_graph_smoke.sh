#!/usr/bin/env bash
# v100 factor graph — smoke run (50 steps, easy only, fast verification)
# Key test: does per_breath_ce show a LADDER (B0 > B1 > ... > B9)?
# If yes → topological staging is working. If flat → investigate.
# Usage: bash scripts/v100_factor_graph_smoke.sh
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-.venv/bin/python}"

# K=10: each breath k reveals factor results at depth <= k+1.
# Ladder expected: B0 sees only depth-0 leaves + depth-1 results;
# B9 sees everything → should have lower CE.
DEV='PCI+AMD' \
V100_TASK=1 \
V100_K_MAX=10 \
V100_N_MAX=16 \
V100_F_MAX=8 \
V100_CALIB_WEIGHT=0.05 \
V100_FACTOR_AUX_WEIGHT=0.5 \
V100_DIFFICULTY_FILTER=easy \
V100_TRAIN=.cache/factor_graph_train.jsonl \
V100_VAL=.cache/factor_graph_test.jsonl \
BATCH=4 \
STEPS=50 \
LR=3e-5 \
LOG_EVERY=5 \
PER_BREATH_CE_EVERY=10 \
KL_DIAG_EVERY=25 \
EVAL_EVERY=25 \
EVAL_BATCHES=5 \
EVAL_BATCH=4 \
CKPT_EVERY=500 \
CKPT_LABEL=v100_smoke \
PYTHIA_INIT=1 \
SUDOKU_TASK=0 \
V99_TASK=0 \
"$PYTHON" -u scripts/v100_factor_graph_train.py
