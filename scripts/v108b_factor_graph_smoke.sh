#!/usr/bin/env bash
# v108b factor graph — smoke (500 steps, cold-start from Pythia-410M)
#
# Architecture: digit-decomposed INPUT + tree codebook OUTPUT.
#   Input:  sum of learned per-(pos, digit) embeddings → ONE token per variable
#   Output: 5-level tree codebook (same as v108)
#
# Cold-start (no v107 warm): v107's domain_codebook is incompatible with the
# digit-input path. v108b starts from pure Pythia-410M backbone.
#
# Hypothesis: with digit-precise input, low-order digit accuracy (pos3, pos4)
# should jump from ~50% / ~15% (v108 with bin input) to 80%+ / 70%+.
#
# Smoke acceptance criteria:
#   1. No NaN, step time <= 1.5s
#   2. Per-breath ladder forms (delta >= 0.05 by step 250)
#   3. pos3 acc >= 0.65 at step 500 (vs v108's 0.46)
#   4. pos4 acc >= 0.40 at step 500 (vs v108's 0.18)
#   5. cell_acc easy >= 0.30 at step 500 (vs v108's 0.11)
#
# Usage:
#   bash scripts/v108b_factor_graph_smoke.sh
#   V108B_HARD_BREATH_LEVEL=1 bash scripts/v108b_factor_graph_smoke.sh
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-.venv/bin/python}"

DEV='PCI+AMD' \
V108B_TASK=1 \
V108B_K_MAX=8 \
V108B_N_DIGITS=5 \
V108B_N_MAX=16 \
V108B_F_MAX=8 \
V108B_HARD_BREATH_LEVEL="${V108B_HARD_BREATH_LEVEL:-0}" \
V108B_VAR_LOSS_WEIGHT=1.0 \
V108B_CALIB_WEIGHT=0.05 \
V108B_FACTOR_AUX_WEIGHT=0.5 \
V108B_DIGIT_EMBED_SCALE=0.1 \
V108B_CODEBOOK_N=32 \
V108B_IB_CENTROIDS=.cache/ib_centroids_gsm8k_partial.npz \
V108B_DIFFICULTY_FILTER=easy \
V108B_TRAIN=.cache/factor_graph_train.jsonl \
V108B_VAL=.cache/factor_graph_test.jsonl \
V108B_GSM8K_TRAIN=.cache/gsm8k_factor_graphs_train.jsonl \
V108B_GSM8K_RATIO=0.5 \
BATCH=4 \
STEPS=500 \
LR=3e-5 \
LOG_EVERY=10 \
PER_BREATH_CE_EVERY=50 \
EVAL_EVERY=100 \
EVAL_BATCHES=10 \
EVAL_BATCH=4 \
CKPT_EVERY=250 \
CKPT_LABEL=v108b_smoke \
PYTHIA_INIT=1 \
"$PYTHON" -u scripts/v108b_factor_graph_train.py
