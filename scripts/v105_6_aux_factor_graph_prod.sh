#!/usr/bin/env bash
# v105.6 + DIAGNOSTIC aux distinctness loss — production run (5000 steps)
#
# Same as v105_6_factor_graph_prod.sh but with V105_AUX_DISTINCT_WEIGHT=1.0
# enabled: an explicit penalty on the mean off-diagonal cosine similarity
# between digit-position hidden states within each unobserved cell (the
# terminal tensor right before the digit codebook readout).
#
# Hypothesis under test:
#   v105 per-layer cos_sim diagnostic showed digit positions of one cell
#   collapse to cos_sim 0.99+ in the terminal hidden state. Even v105.6's
#   per-position L0 W_in didn't change this — the per-position weights
#   themselves stayed at cos_sim 0.999 across positions, suggesting the
#   model has no gradient pressure to differentiate per-position.
#
#   This script asks: if we add an EXPLICIT loss term penalizing high
#   cos_sim between positions, can the model find a low-cos solution?
#   - YES → architecture is CAPABLE but UNMOTIVATED. Then we combine the
#     aux loss with v105.6's per-position weights to see if accuracy lifts.
#   - NO  → architecture is at its FUNDAMENTAL CEILING. Pivot warranted
#     (per-number readout in v107, etc.).
#
# Lambda choice (V105_AUX_DISTINCT_WEIGHT=1.0):
#   - aux_distinct_loss starts ~1.0 (collapsed cos_sim across positions).
#   - var_loss + factor_aux_loss ~2.0-3.0 at step 0.
#   - λ=1.0 makes aux meaningful but not dominant. Adjust if needed.
#
# Other settings identical to v105_6_factor_graph_prod.sh.
#
# DO NOT launch without smoke-run passing first.
# Usage: bash scripts/v105_6_aux_factor_graph_prod.sh
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-.venv/bin/python}"
CKPT="${CKPT:-.cache/fg_v104_ckpts/v104_prod_step3000.safetensors}"

DEV='PCI+AMD' \
V105_5_TASK=1 \
V105_5_K_MAX=8 \
V105_5_N_DIGITS=5 \
V105_5_N_MAX=16 \
V105_5_F_MAX=8 \
V105_5_WAIST="${V105_5_WAIST:-512}" \
V105_5_CODEBOOK_N="${V105_5_CODEBOOK_N:-32}" \
V105_5_IB_CENTROIDS=.cache/ib_centroids_gsm8k_partial.npz \
V105_5_IB_TREE=.cache/ib_tree_gsm8k_partial.json \
V105_5_ENERGY_WEIGHT="${V105_5_ENERGY_WEIGHT:-0.01}" \
V105_5_FACTOR_AUX_WEIGHT="${V105_5_FACTOR_AUX_WEIGHT:-1.0}" \
V105_5_CALIB_WEIGHT="${V105_5_CALIB_WEIGHT:-0.05}" \
V105_5_MAGNITUDE_WEIGHT="${V105_5_MAGNITUDE_WEIGHT:-0.3}" \
V105_5_ROPE_BASE=10000 \
V105_5_IB_INIT=1 \
V105_5_WAIST_LORA_INIT=1 \
V105_5_AR_DIGITS=1 \
V105_5_AR_COND_SCALE="${V105_5_AR_COND_SCALE:-0.5}" \
V105_5_AR_MSD_FIRST=0 \
V105_5_PERPOS_FFN="${V105_5_PERPOS_FFN:-0}" \
V105_5_PERPOS_FFN_DIM="${V105_5_PERPOS_FFN_DIM:-2048}" \
V105_5_BLOCK_WITHIN_VAR="${V105_5_BLOCK_WITHIN_VAR:-1}" \
V105_6_PERPOS_L0="${V105_6_PERPOS_L0:-1}" \
V105_AUX_DISTINCT_WEIGHT="${V105_AUX_DISTINCT_WEIGHT:-1.0}" \
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
CKPT_LABEL="${CKPT_LABEL:-v105_6_aux_prod}" \
PYTHIA_INIT=1 \
RESUME_FROM="$CKPT" \
SUDOKU_TASK=0 \
V99_TASK=0 \
V100_TASK=0 \
V101_TASK=0 \
V104_TASK=0 \
V105_TASK=0 \
V105_1_2_TASK=0 \
V105_2_TASK=0 \
V105_4_TASK=0 \
"$PYTHON" -u scripts/v105_5_factor_graph_train.py
