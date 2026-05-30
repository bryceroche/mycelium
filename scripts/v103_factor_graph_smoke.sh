#!/usr/bin/env bash
# v103 factor graph — smoke run (50 steps, easy only, warm-start verification)
#
# Architecture: VQ-VAE-style encoder (1024→512) → codebook (32×512d) → decoder (512→1024).
# The codebook is the ONLY information path from encoder to decoder.
#
# Key tests:
#   1. No NaN, no loss explosion at step 0 — W_expand=0 AND delta_gate_quant=0 means
#      the VQ-VAE path is completely silent at init (forward = v100, byte-identical)
#   2. Loss decreases over 50 steps — W_expand unlocks after step 1 (receives gradient)
#   3. Step time <= 1.5s/step — one extra matmul vs v101: 512→32 score computation
#   4. Per-breath ladder forms (B0 > B9 by >= 0.1)
#   5. Cell accuracy >= 40% on easy (preserves v100 step 8000 ~50% warm-start)
#
# Usage:
#   bash scripts/v103_factor_graph_smoke.sh
#   RESUME_FROM=.cache/fg_v100_ckpts/v100_continue_step8000.safetensors \
#     bash scripts/v103_factor_graph_smoke.sh
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-.venv/bin/python}"

DEV='PCI+AMD' \
V103_TASK=1 \
V100_TASK=1 \
V103_K_MAX=10 \
V103_N_MAX=16 \
V103_F_MAX=8 \
V103_CODEBOOK_N=32 \
V103_WAIST=512 \
V103_CALIB_WEIGHT=0.05 \
V103_FACTOR_AUX_WEIGHT=0.5 \
V103_DIFFICULTY_FILTER=easy \
V103_TRAIN=.cache/factor_graph_train.jsonl \
V103_VAL=.cache/factor_graph_test.jsonl \
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
CKPT_LABEL=v103_smoke \
RESUME_FROM="${RESUME_FROM:-.cache/fg_v100_ckpts/v100_continue_step8000.safetensors}" \
PYTHIA_INIT=1 \
SUDOKU_TASK=0 \
V99_TASK=0 \
V100_TASK=0 \
"$PYTHON" -u scripts/v103_factor_graph_train.py
