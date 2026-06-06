#!/usr/bin/env bash
# v109 PROD — overnight run from v109_smoke_step500 warm-start
#
# Architecture (proven configuration from the smoke):
#   v108 base + 512d LoRA waist + alternation (waist on EVEN breaths only)
#   K=8 (proven; do NOT extend to K=16+ without retraining breath_embed)
#
# Differences from smoke:
#   STEPS:       10000 (vs smoke 500)
#   BATCH:       8 (vs smoke 4) — full prod batch
#   EVAL_BATCH:  8
#   CKPT_EVERY:  500
#   EVAL_EVERY:  500
#   DIFFICULTY_FILTER: empty (use ALL difficulties, not just easy)
#   CKPT_LABEL:  v109_prod
#
# Warm-start: from v109_smoke_step500.safetensors (v109 architecture
#   trained 500 steps from v108 step 500). The waist's W_expand is now
#   non-zero (LoRA has activated). Continuing training preserves the
#   smoke's gains.
#
# Expected runtime: ~10000 steps × ~0.85s/step = ~140 min = ~2.5 hr.
#   Plus eval overhead at every 500 steps: ~3 hr total.
#
# Eval cadence: every 500 steps on the full test set (n=5000 across 3
#   difficulties). Per-position acc + cell_acc per difficulty.
#
# Usage:
#   bash scripts/v109_factor_graph_prod.sh
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-.venv/bin/python}"
CKPT="${CKPT:-.cache/fg_v109_ckpts/v109_smoke_step500.safetensors}"

DEV='PCI+AMD' \
V109_TASK=1 \
V109_K_MAX=8 \
V109_N_DIGITS=5 \
V109_N_MAX=16 \
V109_F_MAX=8 \
V109_WAIST_DIM=512 \
V109_ALTERNATION=1 \
V109_HARD_BREATH_LEVEL=0 \
V109_VAR_LOSS_WEIGHT=1.0 \
V109_CALIB_WEIGHT=0.05 \
V109_FACTOR_AUX_WEIGHT=0.5 \
V109_CODEBOOK_N=32 \
V109_IB_CENTROIDS=.cache/ib_centroids_gsm8k_partial.npz \
V109_TRAIN=.cache/factor_graph_train.jsonl \
V109_VAL=.cache/factor_graph_test.jsonl \
V109_GSM8K_TRAIN=.cache/gsm8k_factor_graphs_train.jsonl \
V109_GSM8K_RATIO=0.5 \
BATCH=8 \
STEPS="${STEPS:-10000}" \
LR=3e-5 \
LOG_EVERY=20 \
PER_BREATH_CE_EVERY=100 \
EVAL_EVERY=500 \
EVAL_BATCHES=30 \
EVAL_BATCH=8 \
CKPT_EVERY=500 \
CKPT_LABEL="${CKPT_LABEL:-v109_prod}" \
RESUME_FROM="$CKPT" \
PYTHIA_INIT=1 \
"$PYTHON" -u scripts/v109_factor_graph_train.py
