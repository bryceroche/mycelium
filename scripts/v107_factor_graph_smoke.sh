#!/usr/bin/env bash
# v107 factor graph — smoke run (50 steps, warm-start from v104_prod_step3000)
#
# Architecture: hybrid 200-bin codebook (100 linear + 50 log[100,999] + 50 log[1000,9999])
#   PREDICTION: 200-way softmax (one number, fully correlated)
#   SEARCH: MCTS derives digits from predicted bin (not predicted here)
#
# Warm-start from v104_prod_step3000.safetensors:
#   - shared.* + phase*.* (transformer backbone preserved)
#   - fg_v104.codebook → fg_v107.semantic_codebook (IB centroids preserved)
#   - fg_v107.domain_codebook (200, H) reinitialised random orthonormal
#   - fg_v100.domain_codebook (100, H) skipped (wrong shape)
#
# Smoke acceptance criteria:
#   1. No NaN
#   2. Step time <= 1.5s
#   3. Per-breath ladder Δ >= 0.1 (B0 > B_last)
#   4. Cell accuracy >= 30% on [0,99] subset at step 50
#   5. val accuracy on GSM8K >= 5% (KEY TEST — lifts off 0%)
#
# Usage:
#   bash scripts/v107_factor_graph_smoke.sh
#   RESUME_FROM=<other_ckpt> bash scripts/v107_factor_graph_smoke.sh
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-.venv/bin/python}"

DEV='PCI+AMD' \
V107_TASK=1 \
V107_BIN_COUNT=200 \
V107_K_MAX=10 \
V107_N_MAX=16 \
V107_F_MAX=8 \
V107_CODEBOOK_N=32 \
V107_IB_CENTROIDS=.cache/ib_centroids_gsm8k_partial.npz \
V107_CALIB_WEIGHT=0.05 \
V107_FACTOR_AUX_WEIGHT=0.5 \
V107_ENERGY_WEIGHT=0.01 \
V107_DIFFICULTY_FILTER=easy \
V107_TRAIN=.cache/factor_graph_train.jsonl \
V107_VAL=.cache/factor_graph_test.jsonl \
V107_GSM8K_TRAIN=.cache/gsm8k_factor_graphs_train.jsonl \
V107_GSM8K_RATIO=0.5 \
BATCH=4 \
STEPS=50 \
LR=3e-5 \
LOG_EVERY=5 \
PER_BREATH_CE_EVERY=10 \
EVAL_EVERY=25 \
EVAL_BATCHES=5 \
EVAL_BATCH=4 \
CKPT_EVERY=500 \
CKPT_LABEL=v107_smoke \
RESUME_FROM="${RESUME_FROM:-.cache/fg_v104_ckpts/v104_prod_step3000.safetensors}" \
PYTHIA_INIT=1 \
"$PYTHON" -u scripts/v107_factor_graph_train.py
