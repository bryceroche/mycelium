#!/usr/bin/env bash
# v102 factor graph — production run (3000 steps, full curriculum)
#
# Hypothesis: shared-codebook compression generalizes where v101's per-instance
# projection memorized.  The codebook forces every problem to represent its
# state as a soft combination of 32 shared primitives — topology-invariant.
#
# v101 overfitted by step 4500 (53% peak → 46% final).  STEPS=3000 + EVAL_EVERY=250
# lets us track trajectory closely and stop early if overfit appears again.
#
# Warm-starts from v100_continue_step8000 (v100 baseline at ~50% easy cell_acc).
#
# Usage: bash scripts/v102_factor_graph_prod.sh
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
V102_CURRICULUM=1 \
V102_CURRICULUM_ANNEAL=2000 \
V102_TRAIN=.cache/factor_graph_train.jsonl \
V102_VAL=.cache/factor_graph_test.jsonl \
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
CKPT_LABEL="${CKPT_LABEL:-v102_prod}" \
RESUME_FROM="${RESUME_FROM:-.cache/fg_v100_ckpts/v100_continue_step8000.safetensors}" \
PYTHIA_INIT=1 \
SUDOKU_TASK=0 \
V99_TASK=0 \
"$PYTHON" -u scripts/v102_factor_graph_train.py
