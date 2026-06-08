#!/usr/bin/env bash
# v113 prod — per-digit input tokens + v112b topology gate.
# Cold-start (no v112b warm transfer). 3000 steps.
#
# Tests: does per-digit input lift hard cell_acc above v112b's 0.3945?
# Ckpts every 500 steps; eval every 500 steps.
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-.venv/bin/python}"

DEV='PCI+AMD' \
V113_TASK=1 \
V113_K_MAX=8 \
V113_N_DIGITS=5 \
V113_N_MAX=16 \
V113_F_MAX=8 \
V113_WAIST_DIM=512 \
V113_TOPOLOGY_DIM=64 \
V113_CODEBOOK_N=32 \
V113_IB_CENTROIDS=.cache/ib_centroids_gsm8k_partial.npz \
V110_STEP3_ALTERNATION=1 \
V110_STEP3_PHASE_SCALE=1.0 \
V110_STEP3_GATE_PROFILE=sin2_pi \
V110_STEP3_PHOTON_ALPHA=0.5 \
V110_STEP3_BALANCE_WEIGHT=0.05 \
V110_STEP3_UNCERTAINTY_MIN=0.05 \
V110_STEP3_HARD_BREATH_LEVEL=0 \
V110_STEP3_VAR_LOSS_WEIGHT=1.0 \
V110_STEP3_CALIB_WEIGHT=0.05 \
V110_STEP3_FACTOR_AUX_WEIGHT=0.5 \
V113_TRAIN=.cache/factor_graph_train.jsonl \
V113_VAL=.cache/factor_graph_test.jsonl \
V113_GSM8K_TRAIN=.cache/gsm8k_factor_graphs_train.jsonl \
V113_GSM8K_RATIO=0.5 \
V113_SBP_NOISE_SCALE=0.0 \
BATCH=8 \
STEPS="${STEPS:-3000}" \
LR=3e-5 \
LOG_EVERY=25 \
PER_BREATH_CE_EVERY=100 \
EVAL_EVERY=500 \
EVAL_BATCHES=10 \
EVAL_BATCH=8 \
CKPT_EVERY=500 \
CKPT_LABEL="${CKPT_LABEL:-v113_prod}" \
PYTHIA_INIT=1 \
"$PYTHON" -u scripts/v113_train.py
