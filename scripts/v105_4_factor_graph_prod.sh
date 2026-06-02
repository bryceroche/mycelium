#!/usr/bin/env bash
# v105.4 factor graph — production run (5000 steps)
#
# Architecture: v105.3 (LSD-first array + digit RoPE + projection waist + IB
# codebook + AR digit decoding) PLUS four v105.4 additions:
#   1. magnitude head (4-way "n_digits" classification per cell)
#   2. per-position digit codebooks (5 codebooks instead of 1 shared)
#   3. hierarchical IB attention (4-way family × 32-leaf gating)
#   4. soft magnitude-derived valid mask on factor_aux
#
# Training:
#   - 50/50 mix of synthetic + GSM8K (NO range filter)
#   - Curriculum: easy-only for first 1000 steps, then annealed to uniform
#   - K=8 iterative-prefill breaths
#   - T=88 sequence length
#   - Energy weight 0.01 (differentiable constraint energy, LSD place values)
#   - Magnitude weight 0.3 (4-way CE on number-of-digits)
#   - Warm-start from v104_prod_step3000.safetensors
#
# DO NOT launch without smoke-run passing first.
# Usage: bash scripts/v105_4_factor_graph_prod.sh
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-.venv/bin/python}"
CKPT="${CKPT:-.cache/fg_v104_ckpts/v104_prod_step3000.safetensors}"

DEV='PCI+AMD' \
V105_4_TASK=1 \
V105_4_K_MAX=8 \
V105_4_N_DIGITS=5 \
V105_4_N_MAX=16 \
V105_4_F_MAX=8 \
V105_4_WAIST="${V105_4_WAIST:-512}" \
V105_4_CODEBOOK_N="${V105_4_CODEBOOK_N:-32}" \
V105_4_IB_CENTROIDS=.cache/ib_centroids_gsm8k_partial.npz \
V105_4_IB_TREE=.cache/ib_tree_gsm8k_partial.json \
V105_4_ENERGY_WEIGHT="${V105_4_ENERGY_WEIGHT:-0.01}" \
V105_4_FACTOR_AUX_WEIGHT="${V105_4_FACTOR_AUX_WEIGHT:-1.0}" \
V105_4_CALIB_WEIGHT="${V105_4_CALIB_WEIGHT:-0.05}" \
V105_4_MAGNITUDE_WEIGHT="${V105_4_MAGNITUDE_WEIGHT:-0.3}" \
V105_4_ROPE_BASE=10000 \
V105_4_IB_INIT=1 \
V105_4_WAIST_LORA_INIT=1 \
V105_4_AR_DIGITS=1 \
V105_4_AR_COND_SCALE="${V105_4_AR_COND_SCALE:-0.5}" \
V105_4_AR_MSD_FIRST=0 \
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
CKPT_LABEL="${CKPT_LABEL:-v105_4_prod}" \
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
"$PYTHON" -u scripts/v105_4_factor_graph_train.py
