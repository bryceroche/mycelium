#!/usr/bin/env bash
# v104 factor graph — production run (3000 steps, full curriculum, IB codebook)
#
# Architecture: IB-anchored 1024d residual codebook matching (v102 forward path).
# Initialized from 32 semantic centroids (Pythia embeddings of GSM8K variable
# descriptions, hierarchically clustered per OP family via IB criterion).
#
# Dual dataset training: 50/50 mix of synthetic [0,99] + GSM8K filtered to [0,99].
# Distribution alignment: the model trains on the distribution we want to test on.
#
# Comparison baseline:
#   v100 (no codebook):               K=10 = 40.7% overall cell acc
#   v101 (projection waist 512d):     K=10 = 47.6% overall cell acc  ← best
#   v102 (random 32-entry codebook):  K=10 ~ 46.6% overall cell acc
#   v104 (IB-anchored 32-entry):      hypothesis ≥ 50% overall cell acc
#
# Same warm-start as v102: v100_continue_step8000 (no v104 keys in ckpt → IB init kept).
# Same training budget: 3000 steps, EVAL_EVERY=250.
#
# DO NOT launch without smoke-run passing first.
# Usage: bash scripts/v104_factor_graph_prod.sh
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-.venv/bin/python}"

DEV='PCI+AMD' \
V104_TASK=1 \
V100_TASK=1 \
V104_K_MAX=10 \
V104_N_MAX=16 \
V104_F_MAX=8 \
V104_CODEBOOK_N=32 \
V104_IB_CENTROIDS=.cache/ib_centroids_gsm8k_partial.npz \
V104_CALIB_WEIGHT=0.05 \
V104_FACTOR_AUX_WEIGHT=0.5 \
V104_CURRICULUM=1 \
V104_CURRICULUM_ANNEAL=2000 \
V104_TRAIN=.cache/factor_graph_train.jsonl \
V104_VAL=.cache/factor_graph_test.jsonl \
V104_GSM8K_TRAIN=.cache/gsm8k_factor_graphs_train.jsonl \
V104_GSM8K_RATIO=0.5 \
BATCH="${BATCH:-8}" \
STEPS="${STEPS:-3000}" \
LR="${LR:-3e-5}" \
LOG_EVERY="${LOG_EVERY:-10}" \
PER_BREATH_CE_EVERY="${PER_BREATH_CE_EVERY:-50}" \
KL_DIAG_EVERY="${KL_DIAG_EVERY:-100}" \
EVAL_EVERY="${EVAL_EVERY:-250}" \
EVAL_BATCHES="${EVAL_BATCHES:-20}" \
EVAL_BATCH="${EVAL_BATCH:-8}" \
CKPT_EVERY="${CKPT_EVERY:-500}" \
CKPT_LABEL="${CKPT_LABEL:-v104_prod}" \
RESUME_FROM="${RESUME_FROM:-.cache/fg_v100_ckpts/v100_continue_step8000.safetensors}" \
PYTHIA_INIT=1 \
SUDOKU_TASK=0 \
V99_TASK=0 \
"$PYTHON" -u scripts/v104_factor_graph_train.py
