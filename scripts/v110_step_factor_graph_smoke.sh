#!/usr/bin/env bash
# v110-step smoke — v110-full + step-size variance penalty.
#
# Architecture (warm-start from v109pi_cont9_step500):
#   v110-full base (notebook + sin² photon gate)
#   + step-size CoV penalty: aux_loss += balance_weight · std(step_mag_k) / mean(step_mag_k)
#
# The penalty pushes per-breath step magnitudes toward uniformity ("Goldilocks
# bite-sized steps") — load distributed across all K breath cycles.
#
# At balance_weight=0: byte-identical to v110-full.
# Default balance_weight=0.05: mild push, won't dominate the main task loss.
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-.venv/bin/python}"
CKPT="${CKPT:-.cache/fg_v109pi_ckpts/v109pi_cont9_step500.safetensors}"

DEV='PCI+AMD' \
V110_STEP_TASK=1 \
V110_STEP_K_MAX=8 \
V110_STEP_N_DIGITS=5 \
V110_STEP_N_MAX=16 \
V110_STEP_F_MAX=8 \
V110_STEP_WAIST_DIM=512 \
V110_STEP_ALTERNATION="${V110_STEP_ALTERNATION:-1}" \
V110_STEP_PHASE_SCALE="${V110_STEP_PHASE_SCALE:-1.0}" \
V110_STEP_GATE_PROFILE="${V110_STEP_GATE_PROFILE:-sin2_pi}" \
V110_STEP_PHOTON_ALPHA="${V110_STEP_PHOTON_ALPHA:-0.5}" \
V110_STEP_BALANCE_WEIGHT="${V110_STEP_BALANCE_WEIGHT:-0.05}" \
V110_STEP_HARD_BREATH_LEVEL=0 \
V110_STEP_VAR_LOSS_WEIGHT=1.0 \
V110_STEP_CALIB_WEIGHT=0.05 \
V110_STEP_FACTOR_AUX_WEIGHT=0.5 \
V110_STEP_CODEBOOK_N=32 \
V110_STEP_IB_CENTROIDS=.cache/ib_centroids_gsm8k_partial.npz \
V110_STEP_TRAIN=.cache/factor_graph_train.jsonl \
V110_STEP_VAL=.cache/factor_graph_test.jsonl \
V110_STEP_GSM8K_TRAIN=.cache/gsm8k_factor_graphs_train.jsonl \
V110_STEP_GSM8K_RATIO=0.5 \
BATCH=8 \
STEPS="${STEPS:-500}" \
LR=3e-5 \
LOG_EVERY=10 \
PER_BREATH_CE_EVERY=50 \
EVAL_EVERY=100 \
EVAL_BATCHES=30 \
EVAL_BATCH=8 \
CKPT_EVERY=250 \
CKPT_LABEL="${CKPT_LABEL:-v110_step_smoke}" \
RESUME_FROM="$CKPT" \
PYTHIA_INIT=1 \
"$PYTHON" -u scripts/v110_step_factor_graph_train.py
