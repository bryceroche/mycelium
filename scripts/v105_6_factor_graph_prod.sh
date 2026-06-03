#!/usr/bin/env bash
# v105.6 factor graph — production run (5000 steps)
#
# Architecture: v105.5 minus PPFFN, plus a v105.6 addition:
#   PER-POSITION W_IN AT L0 FFN (REPLACEMENT path) — the L0 FFN's W_in is
#   per-digit-position for digit tokens (no shared Pythia w_in fallback at
#   digit positions). Factor tokens still use the shared Pythia w_in. The
#   shared Pythia W_out is preserved (warm-start), and the n_digits=5
#   per-position W_in matrices are init'd as (Pythia L0 w_in) + tiny noise.
#
# Why replacement, not add-alongside:
#   PPFFN (add-alongside) trained but stayed functionally inert — the shared
#   Pythia path gradient-suppressed the parallel PPFFN path. The replacement
#   path forces gradient through per-position weights at L0 digit tokens
#   (no shared w_in fallback there at all).
#
# Training:
#   - 50/50 mix of synthetic + GSM8K (NO range filter)
#   - Curriculum: easy-only for first 1000 steps, then annealed to uniform
#   - K=8 iterative-prefill breaths
#   - T=88 sequence length
#   - Energy weight 0.01 (differentiable constraint energy, LSD place values)
#   - Magnitude weight 0.3 (4-way CE on number-of-digits)
#   - V105_5_BLOCK_WITHIN_VAR=1 (hard-block within-variable Y-soft attention)
#   - V105_5_PERPOS_FFN=0 (PPFFN is dropped entirely)
#   - V105_6_PERPOS_L0=1 (NEW)
#   - Warm-start from v104_prod_step3000.safetensors
#
# DO NOT launch without smoke-run passing first.
# Usage: bash scripts/v105_6_factor_graph_prod.sh
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
CKPT_LABEL="${CKPT_LABEL:-v105_6_prod}" \
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
