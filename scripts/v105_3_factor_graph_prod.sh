#!/usr/bin/env bash
# v105.3 factor graph — production run (5000 steps)
#
# Architecture: LSD-first array layout + digit RoPE (no reversal) + projection
# waist + IB semantic codebook + per-NUMBER aux loss + AR LSD-first decoding.
#
# Functionally equivalent to v105.1.2 v2 with AR LSD-first enabled, but uses
# LSD-first array layout natively — array index 0 = ones digit naturally,
# RoPE position 0 is no rotation, place values are [10^0, 10^1, ...].
#
# Active variant: V105_3_AR_DIGITS=1, V105_3_AR_MSD_FIRST=0 (LSD-first AR — the
# direction that produced the diversity-signal breakthrough at v105.1.2 v2 LSD
# AR step 1000 ckpt).
#
# Training:
#   - 50/50 mix of synthetic + GSM8K (NO range filter)
#   - Curriculum: easy-only for first 1000 steps, then annealed to uniform
#   - K=8 iterative-prefill breaths
#   - T=88 sequence length
#   - Energy weight 0.01 (differentiable constraint energy, LSD place values)
#   - Warm-start from v104_prod_step3000.safetensors
#     (backbone + IB codebook; fresh digit/waist/RoPE params)
#
# DO NOT launch without smoke-run passing first.
# Usage: bash scripts/v105_3_factor_graph_prod.sh
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-.venv/bin/python}"
CKPT="${CKPT:-.cache/fg_v104_ckpts/v104_prod_step3000.safetensors}"

DEV='PCI+AMD' \
V105_3_TASK=1 \
V105_3_K_MAX=8 \
V105_3_N_DIGITS=5 \
V105_3_N_MAX=16 \
V105_3_F_MAX=8 \
V105_3_WAIST="${V105_3_WAIST:-512}" \
V105_3_CODEBOOK_N="${V105_3_CODEBOOK_N:-32}" \
V105_3_IB_CENTROIDS=.cache/ib_centroids_gsm8k_partial.npz \
V105_3_ENERGY_WEIGHT="${V105_3_ENERGY_WEIGHT:-0.01}" \
V105_3_FACTOR_AUX_WEIGHT="${V105_3_FACTOR_AUX_WEIGHT:-1.0}" \
V105_3_CALIB_WEIGHT="${V105_3_CALIB_WEIGHT:-0.05}" \
V105_3_ROPE_BASE=10000 \
V105_3_IB_INIT=1 \
V105_3_WAIST_LORA_INIT=1 \
V105_3_AR_DIGITS=1 \
V105_3_AR_COND_SCALE="${V105_3_AR_COND_SCALE:-0.5}" \
V105_3_AR_MSD_FIRST=0 \
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
CKPT_LABEL="${CKPT_LABEL:-v105_3_prod}" \
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
"$PYTHON" -u scripts/v105_3_factor_graph_train.py
