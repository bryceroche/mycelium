#!/usr/bin/env bash
# v101 factor graph — smoke run (50 steps, easy only, warm-start verification)
#
# Key tests:
#   1. Loss doesn't explode at step 0 — near-identity waist init preserved warm-start
#   2. Loss decreases over 50 steps — waist is learning useful compression
#   3. Step time ≤ 2s/step — two extra matmuls per breath (10 breaths) are cheap
#   4. Per-breath ladder still forms (B0 > B9 by ≥ 0.1)
#   5. Cell accuracy ≥ 40% (v100 step 8000 was ~50%; waist shouldn't destroy that)
#
# Usage: bash scripts/v101_factor_graph_smoke.sh
#   or:  RESUME_FROM=.cache/fg_v100_ckpts/v100_continue_step8000.safetensors \
#        V100_TASK=1 bash scripts/v101_factor_graph_smoke.sh
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-.venv/bin/python}"

DEV='PCI+AMD' \
V101_TASK=1 \
V100_TASK=1 \
V101_K_MAX=10 \
V101_N_MAX=16 \
V101_F_MAX=8 \
V101_WAIST=512 \
V101_CALIB_WEIGHT=0.05 \
V101_FACTOR_AUX_WEIGHT=0.5 \
V101_DIFFICULTY_FILTER=easy \
V101_TRAIN=.cache/factor_graph_train.jsonl \
V101_VAL=.cache/factor_graph_test.jsonl \
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
CKPT_LABEL=v101_smoke \
RESUME_FROM="${RESUME_FROM:-.cache/fg_v100_ckpts/v100_continue_step8000.safetensors}" \
PYTHIA_INIT=1 \
SUDOKU_TASK=0 \
V99_TASK=0 \
V100_TASK=0 \
"$PYTHON" -u scripts/v101_factor_graph_train.py
