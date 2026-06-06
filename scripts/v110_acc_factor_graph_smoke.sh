#!/usr/bin/env bash
# v110-acc smoke — v109pi + ACCUMULATE notebook.
#
# Architecture (warm-start from v109pi_cont9_step500):
#   v109pi base (waist + alternation + per-breath Q rotation)
#   + write-once K-slot notebook (B, K_max, H)
#   + causal cross-attn read at start of each breath (W_o zero-init)
#
# Byte-safe at step 0: W_o=0 → cross-attn contribution=0 → identical to v109pi.
# Smoke uses v109pi_cont9_step500 ckpt (effective step 9000, triple new-high).
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-.venv/bin/python}"
CKPT="${CKPT:-.cache/fg_v109pi_ckpts/v109pi_cont9_step500.safetensors}"

DEV='PCI+AMD' \
V110_ACC_TASK=1 \
V110_ACC_K_MAX=8 \
V110_ACC_N_DIGITS=5 \
V110_ACC_N_MAX=16 \
V110_ACC_F_MAX=8 \
V110_ACC_WAIST_DIM=512 \
V110_ACC_ALTERNATION="${V110_ACC_ALTERNATION:-1}" \
V110_ACC_PHASE_SCALE="${V110_ACC_PHASE_SCALE:-1.0}" \
V110_ACC_HARD_BREATH_LEVEL=0 \
V110_ACC_VAR_LOSS_WEIGHT=1.0 \
V110_ACC_CALIB_WEIGHT=0.05 \
V110_ACC_FACTOR_AUX_WEIGHT=0.5 \
V110_ACC_CODEBOOK_N=32 \
V110_ACC_IB_CENTROIDS=.cache/ib_centroids_gsm8k_partial.npz \
V110_ACC_TRAIN=.cache/factor_graph_train.jsonl \
V110_ACC_VAL=.cache/factor_graph_test.jsonl \
V110_ACC_GSM8K_TRAIN=.cache/gsm8k_factor_graphs_train.jsonl \
V110_ACC_GSM8K_RATIO=0.5 \
BATCH=8 \
STEPS="${STEPS:-500}" \
LR=3e-5 \
LOG_EVERY=10 \
PER_BREATH_CE_EVERY=50 \
EVAL_EVERY=100 \
EVAL_BATCHES=30 \
EVAL_BATCH=8 \
CKPT_EVERY=250 \
CKPT_LABEL="${CKPT_LABEL:-v110_acc_smoke}" \
RESUME_FROM="$CKPT" \
PYTHIA_INIT=1 \
"$PYTHON" -u scripts/v110_acc_factor_graph_train.py
