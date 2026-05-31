#!/usr/bin/env bash
# Stage 3 of v105.1.2 v2 compositional training plan:
# main factor-graph training from the digit-arithmetic pre-train.
#
# Same architecture and hyperparameters as v105_1_2_factor_graph_prod.sh, with:
#   - log-uniform synthetic factor graphs (operand range [1, 9999], 5-digit
#     coverage at all positions, vs the original generator's [1, 20] range)
#   - warm-start from the digit-arithmetic pre-train checkpoint (Stage 1
#     output), NOT directly from v104. The pre-train already bakes
#     (a, b) -> a +/- b / * / / into the weights so the model can spend
#     its compositional capacity on the factor-graph application.
#   - 5000 steps instead of 3000 (more headroom now that the ones digit
#     is bootstrapped).
#
# Checkpoints to .cache/fg_v105_1_2_ckpts/v105_1_2_loguniform_prod_*.safetensors.
#
# DO NOT launch without inspecting Stage 1 / Stage 2 outputs first.
# Usage: bash scripts/v105_1_2_v2_loguniform_prod.sh
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-.venv/bin/python}"
CKPT="${CKPT:-.cache/fg_v105_1_2_ckpts/v105_1_2_pretrain_digit_final.safetensors}"

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
V105_CURRICULUM=1 \
V105_CURRICULUM_ANNEAL=1000 \
V105_TRAIN=.cache/factor_graph_train_loguniform.jsonl \
V105_VAL=.cache/factor_graph_test_loguniform.jsonl \
V105_GSM8K_TRAIN=.cache/gsm8k_factor_graphs_train.jsonl \
V105_GSM8K_RATIO="${V105_GSM8K_RATIO:-0.5}" \
BATCH="${BATCH:-8}" \
STEPS="${STEPS:-5000}" \
LR="${LR:-3e-5}" \
LOG_EVERY="${LOG_EVERY:-10}" \
PER_BREATH_CE_EVERY="${PER_BREATH_CE_EVERY:-50}" \
EVAL_EVERY="${EVAL_EVERY:-250}" \
EVAL_BATCHES="${EVAL_BATCHES:-30}" \
EVAL_BATCH="${EVAL_BATCH:-8}" \
CKPT_EVERY="${CKPT_EVERY:-500}" \
CKPT_LABEL="${CKPT_LABEL:-v105_1_2_loguniform_prod}" \
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
