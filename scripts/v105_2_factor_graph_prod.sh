#!/usr/bin/env bash
# v105.2 factor graph — production run (3000 steps)
#
# Architecture: v105.1.2 + LSD-first digit encoding + digit_valid_mask
#
#   - LSD-first: position 0 = ones place, position p = 10^p place.
#     Position 0 is the only position GUARANTEED valid for any non-negative
#     integer.  Smaller values have natural padding at higher positions, but
#     those positions are masked out in the loss via digit_valid_mask.
#   - digit_valid_mask: (B, N_MAX, N_DIGITS) — 1 for positions used by the
#     gold value, 0 for "leading zero" padding above the most-significant digit.
#   - factor_digit_valid_mask: same for factor result gold values.
#   - All places (ones, tens, hundreds, ...) contribute to per-breath CE only
#     when they correspond to natural digits — no more "all leading zeros"
#     trivial attractor that crushed v105 → v105.1 → v105.1.2.
#
# Training (same as v105.1.2 prod):
#   - 50/50 mix of synthetic [0,99] + GSM8K (NO range filter)
#   - Curriculum: easy-only for first 1000 steps, then annealed to uniform
#   - K=8 iterative-prefill breaths
#   - T=88 sequence length
#   - Energy weight 0.01 (LSD-first expected-value energy)
#   - Warm-start from v104_prod_step3000.safetensors
#
# Comparison baselines:
#   v100 (100-way codebook):               K=10 = 40.7% cell acc
#   v101 (+ 512d waist):                   K=10 = 47.6% cell acc
#   v104 (+ IB 32-entry codebook):         K=10 = 47.7% cell acc
#   v105.1 (digit RoPE):                   TBD
#   v105.1.2 (full stack, MSD):            0% val (constant-predictor collapse)
#   v105.2 (LSD + valid_mask):             target >= 30% on synth easy
#
# DO NOT launch without smoke-run passing first.
# Usage: bash scripts/v105_2_factor_graph_prod.sh
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
V105_2_WAIST="${V105_2_WAIST:-512}" \
V105_2_CODEBOOK_N="${V105_2_CODEBOOK_N:-32}" \
V105_2_IB_CENTROIDS=.cache/ib_centroids_gsm8k_partial.npz \
V105_2_ENERGY_WEIGHT="${V105_2_ENERGY_WEIGHT:-0.01}" \
V105_2_FACTOR_AUX_WEIGHT="${V105_2_FACTOR_AUX_WEIGHT:-1.0}" \
V105_2_CALIB_WEIGHT="${V105_2_CALIB_WEIGHT:-0.05}" \
V105_2_ROPE_BASE=10000 \
V105_2_IB_INIT=1 \
V105_2_WAIST_LORA_INIT=1 \
V105_CURRICULUM=1 \
V105_CURRICULUM_ANNEAL=1000 \
V105_TRAIN=.cache/factor_graph_train.jsonl \
V105_VAL=.cache/factor_graph_test.jsonl \
V105_GSM8K_TRAIN=.cache/gsm8k_factor_graphs_train.jsonl \
V105_GSM8K_RATIO="${V105_GSM8K_RATIO:-0.5}" \
BATCH="${BATCH:-8}" \
STEPS="${STEPS:-3000}" \
LR="${LR:-3e-5}" \
LOG_EVERY="${LOG_EVERY:-10}" \
PER_BREATH_CE_EVERY="${PER_BREATH_CE_EVERY:-50}" \
EVAL_EVERY="${EVAL_EVERY:-250}" \
EVAL_BATCHES="${EVAL_BATCHES:-20}" \
EVAL_BATCH="${EVAL_BATCH:-8}" \
CKPT_EVERY="${CKPT_EVERY:-500}" \
CKPT_LABEL="${CKPT_LABEL:-v105_2_prod}" \
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
