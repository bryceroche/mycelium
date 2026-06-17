#!/usr/bin/env bash
# Perceiver-Poincaré BRICK-1 smoke — the MAKE-OR-BREAK anchored-perceiver test.
#
# Confirms (on the AM driver):
#   (a) compiles (d_hyp cross-attn READ/WRITE + Pythia THINK + readout),
#   (b) t=0 ANCHOR-CHECK: anchored routing == factor-graph membership? (single vs
#       per_constraint reported + SELECTED),
#   (c) ~40-60 steps: loss off chance (cell_acc > 1/N = 0.143),
#   (d) ENGAGEMENT (the kill metric) stays alive (read/write select_norm non-zero,
#       not flatlining toward 0),
#   (e) no NaN, grads finite.
# Throwaway RUN_NAME. K=12 for a fast smoke compile (design K_max=20).
set -euo pipefail
cd "$(dirname "$0")/.."

PERCEIVER_TASK=1 \
PERCEIVER_K_MAX="${PERCEIVER_K_MAX:-12}" \
PERCEIVER_BALL_PATH="${PERCEIVER_BALL_PATH:-auto}" \
PERCEIVER_TAU="${PERCEIVER_TAU:-0.5}" \
PERCEIVER_RHO="${PERCEIVER_RHO:-0.7}" \
PERCEIVER_DIM="${PERCEIVER_DIM:-48}" \
PERCEIVER_N_GLOBAL="${PERCEIVER_N_GLOBAL:-4}" \
BATCH="${BATCH:-8}" \
STEPS="${STEPS:-50}" \
LR="${LR:-3e-5}" \
GRAD_CLIP="${GRAD_CLIP:-1.0}" \
MAX_ZNORM="${MAX_ZNORM:-0.9}" \
LOG_EVERY="${LOG_EVERY:-5}" \
EVAL_EVERY="${EVAL_EVERY:-0}" \
RUN_NAME="${RUN_NAME:-perceiver_brick1_smoke}" \
KENKEN_TRAIN="${KENKEN_TRAIN:-.cache/kenken_train_curriculum.jsonl}" \
KENKEN_TEST="${KENKEN_TEST:-.cache/kenken_test_curriculum.jsonl}" \
SEED="${SEED:-42}" \
.venv/bin/python scripts/perceiver_train.py
