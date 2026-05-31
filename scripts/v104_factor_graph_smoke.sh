#!/usr/bin/env bash
# v104 factor graph — smoke run (50 steps, easy only, warm-start verification)
#
# Architecture: IB-anchored 1024d residual codebook matching (v102 forward path).
# The KEY change from v102/v103: codebook initialized from IB semantic centroids
# (Pythia embeddings of variable descriptions, clustered per OP family), not random.
#
# Hypothesis under test: semantic codebook init > random init.
# Baseline (v102 random 32-entry): ~46.6% cell acc.  Target: ≥ 50%.
#
# Dual dataset: 50/50 mix of synthetic [0,99] + GSM8K filtered to [0,99].
#
# Smoke acceptance criteria:
#   1. No NaN — delta_gate_quant=0 at init means codebook path silent
#   2. Loss decreases over 50 steps
#   3. Step time ≤ 1.5s/step
#   4. Per-breath ladder Δ ≥ 0.1
#   5. Cell accuracy ≥ 40% (preserves v100 step 8000 warm-start)
#
# Usage:
#   bash scripts/v104_factor_graph_smoke.sh
#   RESUME_FROM=.cache/fg_v100_ckpts/v100_continue_step8000.safetensors \
#     bash scripts/v104_factor_graph_smoke.sh
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
V104_DIFFICULTY_FILTER=easy \
V104_TRAIN=.cache/factor_graph_train.jsonl \
V104_VAL=.cache/factor_graph_test.jsonl \
V104_GSM8K_TRAIN=.cache/gsm8k_factor_graphs_train.jsonl \
V104_GSM8K_RATIO=0.5 \
BATCH=4 \
STEPS=50 \
LR=3e-5 \
LOG_EVERY=5 \
PER_BREATH_CE_EVERY=10 \
KL_DIAG_EVERY=25 \
EVAL_EVERY=25 \
EVAL_BATCHES=5 \
EVAL_BATCH=4 \
CKPT_EVERY=500 \
CKPT_LABEL=v104_smoke \
RESUME_FROM="${RESUME_FROM:-.cache/fg_v100_ckpts/v100_continue_step8000.safetensors}" \
PYTHIA_INIT=1 \
SUDOKU_TASK=0 \
V99_TASK=0 \
V100_TASK=0 \
"$PYTHON" -u scripts/v104_factor_graph_train.py
