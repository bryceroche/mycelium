#!/usr/bin/env bash
# v105.4 factor graph — smoke run (50 steps)
#
# Architecture: v105.3 (LSD-first array + digit RoPE + projection waist + IB
# codebook + AR digit decoding) PLUS four v105.4 additions:
#   1. magnitude head (4-way "n_digits" classification per cell)
#   2. per-position digit codebooks (5 codebooks instead of 1 shared)
#   3. hierarchical IB attention (4-way family × 32-leaf gating)
#   4. soft magnitude-derived valid mask on factor_aux
#
# Warm-start from v104_prod_step3000.safetensors:
#   - Backbone (shared.*, phase*.*, ln_f.*) loaded
#   - fg_v104.codebook → fg_v105_4.ib_codebook  (if shapes match)
#   - Fresh-init digit_codebook (now per-position), family_centroids,
#     leaf_to_family, magnitude head
#
# Acceptance criteria:
#   1. No NaN (isfinite check passes every step)
#   2. Step time <= 2.5s
#   3. magnitude_loss drops from log(4)=1.39 toward < 0.5
#   4. var_loss + factor_aux similar magnitudes to v105.3
#
# Usage:
#   bash scripts/v105_4_factor_graph_smoke.sh
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
V105_4_WAIST=512 \
V105_4_CODEBOOK_N=32 \
V105_4_IB_CENTROIDS=.cache/ib_centroids_gsm8k_partial.npz \
V105_4_IB_TREE=.cache/ib_tree_gsm8k_partial.json \
V105_4_ENERGY_WEIGHT=0.01 \
V105_4_FACTOR_AUX_WEIGHT=1.0 \
V105_4_CALIB_WEIGHT=0.05 \
V105_4_MAGNITUDE_WEIGHT=0.3 \
V105_4_ROPE_BASE=10000 \
V105_4_IB_INIT=1 \
V105_4_WAIST_LORA_INIT=1 \
V105_4_AR_DIGITS=1 \
V105_4_AR_COND_SCALE=0.5 \
V105_4_AR_MSD_FIRST=0 \
V105_DIFFICULTY_FILTER=easy \
V105_TRAIN=.cache/factor_graph_train.jsonl \
V105_VAL=.cache/factor_graph_test.jsonl \
V105_GSM8K_TRAIN=.cache/gsm8k_factor_graphs_train.jsonl \
V105_GSM8K_RATIO=0.3 \
BATCH=4 \
STEPS=50 \
LR=3e-5 \
LOG_EVERY=5 \
PER_BREATH_CE_EVERY=10 \
EVAL_EVERY=25 \
EVAL_BATCHES=5 \
EVAL_BATCH=4 \
CKPT_EVERY=500 \
CKPT_LABEL=v105_4_smoke \
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
