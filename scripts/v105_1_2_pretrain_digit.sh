#!/usr/bin/env bash
# Stage 1 of v105.1.2 v2 compositional training:
# pre-train single-digit arithmetic into the weights.
#
# Goal: ~5 minute run that bakes (a, b) -> a +/- b / * / / into the model
# so v105.1.2 v2 doesn't have to discover arithmetic during the main
# factor-graph training. Resumes from v104_prod_step3000 (same warm-start
# as the existing prod launcher) and saves to:
#   .cache/fg_v105_1_2_ckpts/v105_1_2_pretrain_digit_final.safetensors
#
# Acceptance: position 4 (ones) val acc > 70% on the digit arith eval data.
#
# Usage: bash scripts/v105_1_2_pretrain_digit.sh
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-.venv/bin/python}"
CKPT="${CKPT:-.cache/fg_v104_ckpts/v104_prod_step3000.safetensors}"

DEV='PCI+AMD' \
V105_1_2_TASK=1 \
V105_1_2_K_MAX=8 \
V105_1_2_N_DIGITS=5 \
V105_1_2_N_MAX=16 \
V105_1_2_F_MAX=8 \
V105_1_2_WAIST="${V105_1_2_WAIST:-512}" \
V105_1_2_CODEBOOK_N="${V105_1_2_CODEBOOK_N:-32}" \
V105_1_2_IB_CENTROIDS=.cache/ib_centroids_gsm8k_partial.npz \
V105_1_2_ENERGY_WEIGHT="${V105_1_2_ENERGY_WEIGHT:-0.01}" \
V105_1_2_FACTOR_AUX_WEIGHT="${V105_1_2_FACTOR_AUX_WEIGHT:-1.0}" \
V105_1_2_CALIB_WEIGHT="${V105_1_2_CALIB_WEIGHT:-0.05}" \
V105_1_2_ROPE_BASE=10000 \
V105_1_2_IB_INIT=1 \
V105_1_2_WAIST_LORA_INIT=1 \
V105_CURRICULUM=0 \
V105_TRAIN=.cache/digit_arith_train.jsonl \
V105_VAL=.cache/digit_arith_val.jsonl \
V105_GSM8K_TRAIN= \
V105_GSM8K_RATIO=0.0 \
BATCH="${BATCH:-8}" \
STEPS="${STEPS:-300}" \
LR="${LR:-5e-5}" \
LOG_EVERY="${LOG_EVERY:-10}" \
PER_BREATH_CE_EVERY="${PER_BREATH_CE_EVERY:-50}" \
EVAL_EVERY="${EVAL_EVERY:-100}" \
EVAL_BATCHES="${EVAL_BATCHES:-10}" \
EVAL_BATCH="${EVAL_BATCH:-8}" \
CKPT_EVERY="${CKPT_EVERY:-300}" \
CKPT_LABEL="${CKPT_LABEL:-v105_1_2_pretrain_digit}" \
PYTHIA_INIT=1 \
RESUME_FROM="$CKPT" \
SUDOKU_TASK=0 \
V99_TASK=0 \
V100_TASK=0 \
V101_TASK=0 \
V104_TASK=0 \
V105_TASK=0 \
V105_2_TASK=0 \
"$PYTHON" -u scripts/v105_1_2_factor_graph_train.py
