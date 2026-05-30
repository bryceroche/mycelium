#!/usr/bin/env bash
# v103 factor graph — production run (3000 steps, full curriculum)
#
# Architecture: VQ-VAE-style encoder (1024→512) → codebook (32×512d) → decoder (512→1024).
# Hypothesis: the 1024d residual space is too large for 32 codebook entries to meaningfully
# represent.  Project to 512d first (where 32 primitives is the right granularity per IB
# tree analysis), THEN quantize via codebook, THEN reconstruct back to 1024d.
#
# Predicted outcome:
#   v100 baseline:         ~50% easy
#   v101 (waist only):     53% peak → overfit to 46%
#   v102 (codebook only):  TBD
#   v103 (waist+codebook): predicted 55-60%+ (combines bottleneck + topology-invariance)
#
# Key differences from v101/v102:
#   - Codebook in 512d (not 1024d): 32×512 is a more constrained subspace
#   - Decoder input is QUANTIZED waist, not raw waist: topology memorization blocked
#   - Two zero-init gates (W_expand=0 AND delta_gate_quant=0): cleaner warm-start
#
# STEPS=3000 (match v102), EVAL_EVERY=250 to catch overfit early.
# v101 overfitted by step 4500 (53% peak → 46%); stopping at 3000 per v102 lesson.
#
# Usage: bash scripts/v103_factor_graph_prod.sh
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
V103_CURRICULUM=1 \
V103_CURRICULUM_ANNEAL=2000 \
V103_TRAIN=.cache/factor_graph_train.jsonl \
V103_VAL=.cache/factor_graph_test.jsonl \
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
CKPT_LABEL="${CKPT_LABEL:-v103_prod}" \
RESUME_FROM="${RESUME_FROM:-.cache/fg_v100_ckpts/v100_continue_step8000.safetensors}" \
PYTHIA_INIT=1 \
SUDOKU_TASK=0 \
V99_TASK=0 \
"$PYTHON" -u scripts/v103_factor_graph_train.py
