#!/usr/bin/env bash
# v105.8 factor graph — production run (5000 steps)
#
# Architecture: v105.5 per-position token architecture (5 tokens per cell,
# Pythia L0-L3 shared, IB hierarchical codebook, projection waist) WITH
# variable supervision switched from per-digit CE to per-NUMBER CE on a
# (200, H) number_codebook (v107's hybrid bin scheme: 100 linear + 50 log
# [100,999] + 50 log [1000,9999]).
#
# Why:
#   Per-layer cos_sim diagnostic on v105.6 step 1000 confirmed terminal
#   hidden state cos_sim ~ 1.0 across digit positions of each cell. Per-digit
#   CE is equivalent to per-NUMBER CE under the conditional-independence
#   factorization that position-specific codebook entries implicitly assume,
#   so the loss form provides NO gradient pressure against collapse — it
#   *expects* a shared hidden state. v105.8 drops per-digit CE entirely;
#   positions are now allowed to collapse (the mean is still a valid number
#   representation) and the 5 per-position tokens act as "thinking slots"
#   rather than digit decoders.
#
# Reference for "why this is principled":
#   v107 (per-NUMBER 200-bin codebook, no per-position) gets 24% GSM8K cell
#   accuracy. v105 family (per-position digit codebook) plateaus at ~5%.
#   v105.8 unifies them: keep the per-position residual capacity, but use
#   per-NUMBER CE so collapse is permitted.
#
# Forced env overrides applied Python-side when V105_8_PER_NUMBER_READOUT=1:
#   V105_5_VAR_LOSS_WEIGHT  → 0  (drop per-digit CE)
#   V105_5_MAGNITUDE_WEIGHT → 0  (magnitude head still runs but is supervised by
#                                  per-NUMBER factor_aux only)
#   V105_5_ENERGY_WEIGHT    → 0  (energy depended on per-digit logits)
#   V105_AUX_DISTINCT_WEIGHT → 0 (no per-position pressure anymore)
#   V105_5_FACTOR_AUX_WEIGHT preserved at 1.0 (per-NUMBER MSE — complementary)
#
# Training:
#   - 50/50 mix of synthetic + GSM8K (NO range filter)
#   - Curriculum: easy-only for first 1000 steps, then annealed to uniform
#   - K=8 iterative-prefill breaths
#   - T=88 sequence length
#   - 200-bin hybrid codebook (matches v107 — direct comparability)
#   - V105_5_BLOCK_WITHIN_VAR=0 (positions can mix freely now)
#   - V105_5_PERPOS_FFN=0 (PPFFN dropped)
#   - V105_6_PERPOS_L0=0 (per-position L0 W_in dropped — was for digit-level
#                          differentiation, no longer needed)
#   - Warm-start from v104_prod_step3000.safetensors (backbone + IB codebook)
#
# DO NOT launch without smoke-run passing first.
# Usage: bash scripts/v105_8_factor_graph_prod.sh
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
V105_5_ENERGY_WEIGHT="${V105_5_ENERGY_WEIGHT:-0}" \
V105_5_FACTOR_AUX_WEIGHT="${V105_5_FACTOR_AUX_WEIGHT:-1.0}" \
V105_5_CALIB_WEIGHT="${V105_5_CALIB_WEIGHT:-0.05}" \
V105_5_MAGNITUDE_WEIGHT="${V105_5_MAGNITUDE_WEIGHT:-0}" \
V105_5_VAR_LOSS_WEIGHT="${V105_5_VAR_LOSS_WEIGHT:-0}" \
V105_AUX_DISTINCT_WEIGHT="${V105_AUX_DISTINCT_WEIGHT:-0}" \
V105_5_ROPE_BASE=10000 \
V105_5_IB_INIT=1 \
V105_5_WAIST_LORA_INIT=1 \
V105_5_AR_DIGITS=1 \
V105_5_AR_COND_SCALE="${V105_5_AR_COND_SCALE:-0.5}" \
V105_5_AR_MSD_FIRST=0 \
V105_5_PERPOS_FFN="${V105_5_PERPOS_FFN:-0}" \
V105_5_PERPOS_FFN_DIM="${V105_5_PERPOS_FFN_DIM:-2048}" \
V105_5_BLOCK_WITHIN_VAR="${V105_5_BLOCK_WITHIN_VAR:-0}" \
V105_6_PERPOS_L0="${V105_6_PERPOS_L0:-0}" \
V105_8_PER_NUMBER_READOUT="${V105_8_PER_NUMBER_READOUT:-1}" \
V105_8_N_NUMBER_BINS="${V105_8_N_NUMBER_BINS:-200}" \
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
CKPT_LABEL="${CKPT_LABEL:-v105_8_prod}" \
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
