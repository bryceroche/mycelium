#!/usr/bin/env bash
# v110-step2 smoke — CE-normalized Goldilocks penalty.
#
# Architecture (warm-start from v109pi_cont9_step500):
#   v110-full base (notebook + sin² photon gate)
#   + step-balance penalty: aux_loss += bw · std(step_k/CE_k) / mean(step_k/CE_k)
#
# CE is DETACHED — only the step side gets gradient. Predicts the model's
# step distribution should INVERT v110-step's late-breath-dominant ramp,
# becoming early-breath-dominant where CE is highest.
#
# At balance_weight=0: byte-identical to v110-full.
# Default balance_weight=0.05: mild push toward step_k ∝ CE_k.
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-.venv/bin/python}"
CKPT="${CKPT:-.cache/fg_v109pi_ckpts/v109pi_cont9_step500.safetensors}"

DEV='PCI+AMD' \
V110_STEP2_TASK=1 \
V110_STEP2_K_MAX=8 \
V110_STEP2_N_DIGITS=5 \
V110_STEP2_N_MAX=16 \
V110_STEP2_F_MAX=8 \
V110_STEP2_WAIST_DIM=512 \
V110_STEP2_ALTERNATION="${V110_STEP2_ALTERNATION:-1}" \
V110_STEP2_PHASE_SCALE="${V110_STEP2_PHASE_SCALE:-1.0}" \
V110_STEP2_GATE_PROFILE="${V110_STEP2_GATE_PROFILE:-sin2_pi}" \
V110_STEP2_PHOTON_ALPHA="${V110_STEP2_PHOTON_ALPHA:-0.5}" \
V110_STEP2_BALANCE_WEIGHT="${V110_STEP2_BALANCE_WEIGHT:-0.05}" \
V110_STEP2_HARD_BREATH_LEVEL=0 \
V110_STEP2_VAR_LOSS_WEIGHT=1.0 \
V110_STEP2_CALIB_WEIGHT=0.05 \
V110_STEP2_FACTOR_AUX_WEIGHT=0.5 \
V110_STEP2_CODEBOOK_N=32 \
V110_STEP2_IB_CENTROIDS=.cache/ib_centroids_gsm8k_partial.npz \
V110_STEP2_TRAIN=.cache/factor_graph_train.jsonl \
V110_STEP2_VAL=.cache/factor_graph_test.jsonl \
V110_STEP2_GSM8K_TRAIN=.cache/gsm8k_factor_graphs_train.jsonl \
V110_STEP2_GSM8K_RATIO=0.5 \
BATCH=8 \
STEPS="${STEPS:-500}" \
LR=3e-5 \
LOG_EVERY=10 \
PER_BREATH_CE_EVERY=50 \
EVAL_EVERY=100 \
EVAL_BATCHES=30 \
EVAL_BATCH=8 \
CKPT_EVERY=250 \
CKPT_LABEL="${CKPT_LABEL:-v110_step2_smoke}" \
RESUME_FROM="$CKPT" \
PYTHIA_INIT=1 \
"$PYTHON" -u scripts/v110_step2_factor_graph_train.py
