#!/usr/bin/env bash
# v107 factor graph — production run (3000 steps, full curriculum, 50/50 mix)
#
# Architecture: hybrid 200-bin codebook
#   - 100 linear bins for [0,99] (exact match for synthetic data)
#   - 50 log-spaced bins for [100,999]
#   - 50 log-spaced bins for [1000,9999]
#
# Dual dataset: 50/50 mix of synthetic [0,99] + GSM8K full value range.
# No [0,99] filtering of GSM8K — all 4261 records used (values > 9999 clamped to bin 199).
#
# Comparison baselines:
#   v104 (100-way, [0,99] filtered):  ~53% cell acc on synthetic
#   v107 hypothesis: 35-45% query acc on GSM8K val (lifts from 0%)
#
# DO NOT launch without smoke-run passing first.
# Usage: bash scripts/v107_factor_graph_prod.sh
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-.venv/bin/python}"

DEV='PCI+AMD' \
V107_TASK=1 \
V107_BIN_COUNT=200 \
V107_K_MAX=10 \
V107_N_MAX=16 \
V107_F_MAX=8 \
V107_CODEBOOK_N=32 \
V107_IB_CENTROIDS=.cache/ib_centroids_gsm8k_partial.npz \
V107_CALIB_WEIGHT=0.05 \
V107_FACTOR_AUX_WEIGHT=0.5 \
V107_ENERGY_WEIGHT=0.01 \
V107_CURRICULUM=1 \
V107_CURRICULUM_ANNEAL=2000 \
V107_TRAIN=.cache/factor_graph_train.jsonl \
V107_VAL=.cache/factor_graph_test.jsonl \
V107_GSM8K_TRAIN=.cache/gsm8k_factor_graphs_train.jsonl \
V107_GSM8K_RATIO=0.5 \
BATCH="${BATCH:-8}" \
STEPS="${STEPS:-3000}" \
LR="${LR:-3e-5}" \
LOG_EVERY="${LOG_EVERY:-10}" \
PER_BREATH_CE_EVERY="${PER_BREATH_CE_EVERY:-50}" \
EVAL_EVERY="${EVAL_EVERY:-250}" \
EVAL_BATCHES="${EVAL_BATCHES:-20}" \
EVAL_BATCH="${EVAL_BATCH:-8}" \
CKPT_EVERY="${CKPT_EVERY:-500}" \
CKPT_LABEL="${CKPT_LABEL:-v107_prod}" \
RESUME_FROM="${RESUME_FROM:-.cache/fg_v104_ckpts/v104_prod_step3000.safetensors}" \
PYTHIA_INIT=1 \
"$PYTHON" -u scripts/v107_factor_graph_train.py
