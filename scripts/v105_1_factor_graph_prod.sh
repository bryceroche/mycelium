#!/usr/bin/env bash
# v105.1 factor graph — production run (3000 steps, full curriculum, digit RoPE)
#
# Architecture: digit RoPE replaces additive digit_pos_embed.
# v105.1 addresses the v105 0%-val failure: inter-digit correlation now
# emerges from RoPE-aware attention rather than requiring the loss to enforce it.
#
# Training:
#   - 50/50 mix of synthetic [0,99] + GSM8K
#   - Curriculum: easy-only for first 1000 steps, then annealed to uniform
#   - K=8 iterative-prefill breaths
#   - T=88 sequence length (same as v105)
#
# DO NOT launch without smoke-run passing first.
# Usage: bash scripts/v105_1_factor_graph_prod.sh
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-.venv/bin/python}"

DEV='PCI+AMD' \
V105_TASK=1 \
V105_K_MAX=8 \
V105_N_DIGITS=5 \
V105_N_MAX=16 \
V105_F_MAX=8 \
V105_ENERGY_WEIGHT="${V105_ENERGY_WEIGHT:-0.01}" \
V105_FACTOR_AUX_WEIGHT="${V105_FACTOR_AUX_WEIGHT:-0.5}" \
V105_CALIB_WEIGHT="${V105_CALIB_WEIGHT:-0.05}" \
V105_1_ROPE_BASE="${V105_1_ROPE_BASE:-10000}" \
V105_CURRICULUM=1 \
V105_CURRICULUM_ANNEAL=1000 \
V105_TRAIN=.cache/factor_graph_train.jsonl \
V105_VAL=.cache/factor_graph_test.jsonl \
V105_GSM8K_TRAIN=.cache/gsm8k_factor_graphs_train.jsonl \
V105_GSM8K_RATIO=0.5 \
BATCH="${BATCH:-8}" \
STEPS="${STEPS:-3000}" \
LR="${LR:-3e-5}" \
LOG_EVERY="${LOG_EVERY:-10}" \
PER_BREATH_CE_EVERY="${PER_BREATH_CE_EVERY:-50}" \
EVAL_EVERY="${EVAL_EVERY:-250}" \
EVAL_BATCHES="${EVAL_BATCHES:-20}" \
EVAL_BATCH="${EVAL_BATCH:-8}" \
CKPT_EVERY="${CKPT_EVERY:-500}" \
CKPT_LABEL="${CKPT_LABEL:-v105_1_prod}" \
PYTHIA_INIT=1 \
SUDOKU_TASK=0 \
V99_TASK=0 \
V100_TASK=0 \
"$PYTHON" -u scripts/v105_1_factor_graph_train.py
