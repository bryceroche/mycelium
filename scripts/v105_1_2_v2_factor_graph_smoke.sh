#!/usr/bin/env bash
# v105.1.2 v2 factor graph — smoke run (200 steps)
#
# Architecture: v105.1.2 + MSD-first digit encoding + right-aligned RoPE +
# digit_valid_mask.  The MSD-first layout matches breathing's coarse-to-fine
# refinement philosophy (breath 0 attends to magnitude, breath K attends to
# ones).  Right-aligned RoPE anchors the ones digit at position 0 regardless
# of n_digits, and the digit_valid_mask zeros out the loss at leading-zero
# padding positions so the model can't collapse to "always predict 0".
#
# This is the architectural fix Bryce identified: v105.2 (LSD-first) worked
# numerically but opposed the architecture's semantics.  v105.1.2 v2 keeps
# MSD-first AND adds the masking infrastructure.
#
# Warm-start from v104_prod_step3000.safetensors.
#
# Acceptance criteria (smoke @ step 200):
#   1. No NaN (isfinite check passes every step)
#   2. Step time <= 2.5s
#   3. Per-breath ladder positive: B0 > B_last, delta > 0.07 (matches v105.2's +0.071)
#   4. Val cell_acc on synthetic easy > 3% (matches v105.2's 3.7%)
#
# Usage:
#   bash scripts/v105_1_2_v2_factor_graph_smoke.sh
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
V105_1_2_WAIST=512 \
V105_1_2_CODEBOOK_N=32 \
V105_1_2_IB_CENTROIDS=.cache/ib_centroids_gsm8k_partial.npz \
V105_1_2_ENERGY_WEIGHT=0.01 \
V105_1_2_FACTOR_AUX_WEIGHT=1.0 \
V105_1_2_CALIB_WEIGHT=0.05 \
V105_1_2_ROPE_BASE=10000 \
V105_1_2_IB_INIT=1 \
V105_1_2_WAIST_LORA_INIT=1 \
V105_DIFFICULTY_FILTER=easy \
V105_TRAIN=.cache/factor_graph_train.jsonl \
V105_VAL=.cache/factor_graph_test.jsonl \
V105_GSM8K_TRAIN=.cache/gsm8k_factor_graphs_train.jsonl \
V105_GSM8K_RATIO=0.3 \
BATCH=4 \
STEPS=200 \
LR=3e-5 \
LOG_EVERY=10 \
PER_BREATH_CE_EVERY=25 \
EVAL_EVERY=100 \
EVAL_BATCHES=5 \
EVAL_BATCH=4 \
CKPT_EVERY=500 \
CKPT_LABEL=v105_1_2_v2_smoke \
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
