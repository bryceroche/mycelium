#!/usr/bin/env bash
# v108 factor graph — smoke run (500 steps, warm-start from v107 step 1000)
#
# Architecture: v107 single-token-per-variable substrate + 5-level tree codebook
# readout (5 levels × 10 digit entries). The tree codebook makes digit-level
# decomposition explicit at the OUTPUT without paying the v105 L0 mean-field
# collapse cost (variables stay single-token in Pythia attention).
#
# Default mode: V108_HARD_BREATH_LEVEL=0 (SOFT — all levels supervised every
# breath). This tests whether the tree codebook structure adds value over
# v107's flat 200-bin codebook.
#
# To test HARD breath-to-level assignment (other-Claude's design):
#   V108_HARD_BREATH_LEVEL=1 bash scripts/v108_factor_graph_smoke.sh
#
# Smoke acceptance criteria:
#   1. No NaN
#   2. Step time <= 1.5s
#   3. Per-breath ladder Δ >= 0.05 (looser than v107 since tree CE is per-level
#      ~log(10)=2.3 vs 200-bin ~log(200)=5.3)
#   4. In-dist cell_acc (ALL digits match) >= 0.10 at step 500 on easy
#      (v107 baseline = 0.24 at step 1000 on whole dataset — this is a 500-step
#      smoke on a subset, so we use 0.10 as the bar)
#   5. Per-position digit_acc[0..4] varies meaningfully (not all the same)
#
# Usage:
#   bash scripts/v108_factor_graph_smoke.sh
#   V108_HARD_BREATH_LEVEL=1 bash scripts/v108_factor_graph_smoke.sh
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-.venv/bin/python}"

DEV='PCI+AMD' \
V108_TASK=1 \
V108_K_MAX=8 \
V108_N_DIGITS=5 \
V108_N_MAX=16 \
V108_F_MAX=8 \
V108_HARD_BREATH_LEVEL="${V108_HARD_BREATH_LEVEL:-0}" \
V108_VAR_LOSS_WEIGHT=1.0 \
V108_CALIB_WEIGHT=0.05 \
V108_FACTOR_AUX_WEIGHT=0.5 \
V108_CODEBOOK_N=32 \
V108_IB_CENTROIDS=.cache/ib_centroids_gsm8k_partial.npz \
V108_DIFFICULTY_FILTER=easy \
V108_TRAIN=.cache/factor_graph_train.jsonl \
V108_VAL=.cache/factor_graph_test.jsonl \
V108_GSM8K_TRAIN=.cache/gsm8k_factor_graphs_train.jsonl \
V108_GSM8K_RATIO=0.5 \
BATCH=4 \
STEPS=500 \
LR=3e-5 \
LOG_EVERY=10 \
PER_BREATH_CE_EVERY=50 \
EVAL_EVERY=100 \
EVAL_BATCHES=10 \
EVAL_BATCH=4 \
CKPT_EVERY=250 \
CKPT_LABEL=v108_smoke \
RESUME_FROM="${RESUME_FROM:-.cache/fg_v107_ckpts/v107_prod_step1000.safetensors}" \
PYTHIA_INIT=1 \
"$PYTHON" -u scripts/v108_factor_graph_train.py
