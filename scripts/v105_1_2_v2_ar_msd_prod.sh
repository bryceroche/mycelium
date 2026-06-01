#!/usr/bin/env bash
# v105.1.2 v2 + MSD-FIRST AUTOREGRESSIVE DIGIT DECODING — production run
#
# Single-variable change vs scripts/v105_1_2_v2_ar_prod.sh:
#   V105_1_2_AR_MSD_FIRST=1
#
# Iteration order: position 0 (ten-thousands) FIRST, then 1, 2, 3, 4 (ones).
# Each position conditions on accumulated soft embeddings of all MORE
# SIGNIFICANT (coarser) digits committed earlier.
#
# Conceptual model: "value tree traversal" — magnitude decision first,
# then leading digit, then second digit, ... finally ones. Each level
# refines the previous one's coarser commitment.
#
# Caveat with our loguniform [1, 9999] data: position 0 is ALWAYS
# invalid (no values >= 10000). The first AR iteration trivially
# predicts pos 0 (no supervision). The meaningful chain starts at
# position 1 (thousands, supervised for values >= 1000).
#
# Expected behavior vs LSD-first:
#   - First meaningful prediction is "thousands digit" not "ones digit"
#   - Gradient flows from later (less significant) positions back to
#     earlier (more significant) — the magnitude decisions get the
#     accumulated correction signal from ones/tens/hundreds
#   - If "value tree" intuition is right, this should converge faster
#     than LSD-first because hierarchical decisions stabilize earlier
#
# Usage: bash scripts/v105_1_2_v2_ar_msd_prod.sh
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
V105_1_2_AR_DIGITS=1 \
V105_1_2_AR_MSD_FIRST=1 \
V105_1_2_AR_COND_SCALE="${V105_1_2_AR_COND_SCALE:-0.5}" \
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
CKPT_LABEL="${CKPT_LABEL:-v105_1_2_ar_msd_prod}" \
PYTHIA_INIT=1 \
RESUME_FROM="$CKPT" \
SUDOKU_TASK=0 \
V99_TASK=0 \
V100_TASK=0 \
V101_TASK=0 \
V104_TASK=0 \
V105_TASK=0 \
"$PYTHON" -u scripts/v105_1_2_factor_graph_train.py
