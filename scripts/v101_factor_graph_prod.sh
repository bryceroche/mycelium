#!/usr/bin/env bash
# v101 factor graph — production run (5000 steps, full curriculum)
#
# Hypothesis: the per-breath waist compression (JPEG Quantize step) provides
# commitment that v100 lacked, enabling higher accuracy on hard/deep problems.
#
# Warm-starts from v100_continue_step8000 (v100 baseline at ~50% easy cell_acc).
# 5000 steps = 2.5× v100's production run, giving the waist time to learn.
#
# Usage: bash scripts/v101_factor_graph_prod.sh
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-.venv/bin/python}"

DEV='PCI+AMD' \
V101_TASK=1 \
V101_K_MAX=10 \
V101_N_MAX=16 \
V101_F_MAX=8 \
V101_WAIST=512 \
V101_CALIB_WEIGHT=0.05 \
V101_FACTOR_AUX_WEIGHT=0.5 \
V101_CURRICULUM=1 \
V101_CURRICULUM_ANNEAL=2000 \
V101_TRAIN=.cache/factor_graph_train.jsonl \
V101_VAL=.cache/factor_graph_test.jsonl \
BATCH="${BATCH:-8}" \
STEPS="${STEPS:-5000}" \
LR="${LR:-3e-5}" \
LOG_EVERY="${LOG_EVERY:-10}" \
PER_BREATH_CE_EVERY="${PER_BREATH_CE_EVERY:-50}" \
KL_DIAG_EVERY="${KL_DIAG_EVERY:-100}" \
EVAL_EVERY="${EVAL_EVERY:-250}" \
EVAL_BATCHES="${EVAL_BATCHES:-20}" \
EVAL_BATCH="${EVAL_BATCH:-8}" \
CKPT_EVERY="${CKPT_EVERY:-500}" \
CKPT_LABEL="${CKPT_LABEL:-v101_prod}" \
RESUME_FROM="${RESUME_FROM:-.cache/fg_v100_ckpts/v100_continue_step8000.safetensors}" \
PYTHIA_INIT=1 \
SUDOKU_TASK=0 \
V99_TASK=0 \
V100_TASK=0 \
"$PYTHON" -u scripts/v101_factor_graph_train.py
