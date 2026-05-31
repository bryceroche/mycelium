#!/usr/bin/env bash
# v105.2 factor graph — smoke run (50 steps)
#
# Architecture: v105.1.2 + LSD-first digit encoding + digit_valid_mask
#
#   - Position 0 = ones place (ALWAYS valid for any non-negative integer).
#   - digit_valid_mask zeros out the loss at "leading zero" padding positions.
#   - This eliminates the v105.1.2 collapse mode: with MSD-first + uniform CE,
#     [0,99] data had 3 trivially-zero MSB positions that drove the model to
#     "always predict 0" (which scored 1.0 on positions 0-2 of length-5 LSD-free
#     gold, then bled into the actual tens/ones positions as a noise floor).
#
# Warm-start from v104_prod_step3000.safetensors:
#   - Backbone (shared.*, phase*.*, ln_f.*) loaded
#   - fg_v104.codebook → fg_v105_2.ib_codebook
#   - Fresh-init digit_codebook, RoPE, waist, breath_embed
#   - Zero-init waist W_expand + delta_gate_quant → step-0 = v104 backbone
#
# Acceptance criteria (smoke @ step 50):
#   1. No NaN (isfinite check passes every step)
#   2. Step time <= 2.5s
#   3. Per-breath ladder positive: B0 > B_last, delta > 0.1
#   4. Train cell_acc lifts off zero (target ≥ 20%)
#   5. Val cell_acc on synthetic easy ≥ 5%
#
# Usage:
#   bash scripts/v105_2_factor_graph_smoke.sh
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-.venv/bin/python}"
CKPT="${CKPT:-.cache/fg_v104_ckpts/v104_prod_step3000.safetensors}"

DEV='PCI+AMD' \
V105_2_TASK=1 \
V105_2_K_MAX=8 \
V105_2_N_DIGITS=5 \
V105_2_N_MAX=16 \
V105_2_F_MAX=8 \
V105_2_WAIST=512 \
V105_2_CODEBOOK_N=32 \
V105_2_IB_CENTROIDS=.cache/ib_centroids_gsm8k_partial.npz \
V105_2_ENERGY_WEIGHT=0.01 \
V105_2_FACTOR_AUX_WEIGHT=1.0 \
V105_2_CALIB_WEIGHT=0.05 \
V105_2_ROPE_BASE=10000 \
V105_2_IB_INIT=1 \
V105_2_WAIST_LORA_INIT=1 \
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
CKPT_LABEL=v105_2_smoke \
PYTHIA_INIT=1 \
RESUME_FROM="$CKPT" \
SUDOKU_TASK=0 \
V99_TASK=0 \
V100_TASK=0 \
V101_TASK=0 \
V104_TASK=0 \
V105_TASK=0 \
V105_1_2_TASK=0 \
"$PYTHON" -u scripts/v105_2_factor_graph_train.py
