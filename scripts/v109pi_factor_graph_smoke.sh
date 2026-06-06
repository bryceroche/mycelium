#!/usr/bin/env bash
# v109pi smoke — v109 + per-breath Q rotation (Option A π-cycled).
#
# Architecture (warm-start from v109_prod_step8500):
#   v109 base (waist + alternation + tree codebook)
#   + per-breath Q rotation by phase k·π/K_max (uniform across positions)
#
# Byte-safe at step 0: at k=0 phase=0 → Q unchanged → identical to v109.
# Other breaths apply increasing rotation, making attention patterns differ.
#
# Smoke uses v109_prod_step8500 ckpt — the all-three-above-0.30 peak.
# 500 steps to see if π-cycling adds lift on top of v109's saturated state.
#
# Set V109PI_PHASE_SCALE=0.5 (half rotation) or 2.0 (overshoot) to ablate.
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-.venv/bin/python}"
CKPT="${CKPT:-.cache/fg_v109_ckpts/v109_prod_step8500.safetensors}"

DEV='PCI+AMD' \
V109PI_TASK=1 \
V109PI_K_MAX=8 \
V109PI_N_DIGITS=5 \
V109PI_N_MAX=16 \
V109PI_F_MAX=8 \
V109PI_WAIST_DIM=512 \
V109PI_ALTERNATION="${V109PI_ALTERNATION:-1}" \
V109PI_PHASE_SCALE="${V109PI_PHASE_SCALE:-1.0}" \
V109PI_HARD_BREATH_LEVEL=0 \
V109PI_VAR_LOSS_WEIGHT=1.0 \
V109PI_CALIB_WEIGHT=0.05 \
V109PI_FACTOR_AUX_WEIGHT=0.5 \
V109PI_CODEBOOK_N=32 \
V109PI_IB_CENTROIDS=.cache/ib_centroids_gsm8k_partial.npz \
V109PI_TRAIN=.cache/factor_graph_train.jsonl \
V109PI_VAL=.cache/factor_graph_test.jsonl \
V109PI_GSM8K_TRAIN=.cache/gsm8k_factor_graphs_train.jsonl \
V109PI_GSM8K_RATIO=0.5 \
BATCH=8 \
STEPS="${STEPS:-500}" \
LR=3e-5 \
LOG_EVERY=10 \
PER_BREATH_CE_EVERY=50 \
EVAL_EVERY=100 \
EVAL_BATCHES=30 \
EVAL_BATCH=8 \
CKPT_EVERY=250 \
CKPT_LABEL="${CKPT_LABEL:-v109pi_smoke}" \
RESUME_FROM="$CKPT" \
PYTHIA_INIT=1 \
"$PYTHON" -u scripts/v109pi_factor_graph_train.py
