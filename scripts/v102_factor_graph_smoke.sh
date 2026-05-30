#!/usr/bin/env bash
# v102 factor graph — smoke run (50 steps, easy only, warm-start verification)
#
# Key tests:
#   1. Loss doesn't explode at step 0 — delta_gate_quant=0 means codebook doesn't
#      affect forward at init (byte-identical to v100)
#   2. Loss decreases over 50 steps — codebook starts contributing after step 1-2
#   3. Step time <= 1.5s/step — codebook match is cheap (B×T×N_CODE×H matmul)
#   4. Per-breath ladder still forms (B0 > B9 by >= 0.1)
#   5. Cell accuracy >= 40% (v100 step 8000 was ~50%; codebook shouldn't destroy that)
#
# Usage: bash scripts/v102_factor_graph_smoke.sh
#   or:  RESUME_FROM=.cache/fg_v100_ckpts/v100_continue_step8000.safetensors \
#        bash scripts/v102_factor_graph_smoke.sh
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-.venv/bin/python}"

DEV='PCI+AMD' \
V102_TASK=1 \
V100_TASK=1 \
V102_K_MAX=10 \
V102_N_MAX=16 \
V102_F_MAX=8 \
V102_CODEBOOK_N=32 \
V102_CALIB_WEIGHT=0.05 \
V102_FACTOR_AUX_WEIGHT=0.5 \
V102_DIFFICULTY_FILTER=easy \
V102_TRAIN=.cache/factor_graph_train.jsonl \
V102_VAL=.cache/factor_graph_test.jsonl \
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
CKPT_LABEL=v102_smoke \
RESUME_FROM="${RESUME_FROM:-.cache/fg_v100_ckpts/v100_continue_step8000.safetensors}" \
PYTHIA_INIT=1 \
SUDOKU_TASK=0 \
V99_TASK=0 \
V100_TASK=0 \
"$PYTHON" -u scripts/v102_factor_graph_train.py
