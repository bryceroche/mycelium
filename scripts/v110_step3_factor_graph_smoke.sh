#!/usr/bin/env bash
# v110-step3 smoke — calibration-driven Goldilocks penalty.
#
# Architecture (warm-start from v109pi_cont9_step500):
#   v110-full base (notebook + sin² photon gate)
#   + step balance: aux_loss += bw · std(step_k / max(1-calib_k, u_min))
#                                  / mean(step_k / max(1-calib_k, u_min))
#
# Calibration head output drives the step regulator. calib is DETACHED so
# only the step side gets gradient. This makes the calibration head
# functional — currently it's trained with weight 0.05 but unused.
# Side effect: calibration accuracy starts mattering for the main task.
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-.venv/bin/python}"
CKPT="${CKPT:-.cache/fg_v109pi_ckpts/v109pi_cont9_step500.safetensors}"

DEV='PCI+AMD' \
V110_STEP3_TASK=1 \
V110_STEP3_K_MAX=8 \
V110_STEP3_N_DIGITS=5 \
V110_STEP3_N_MAX=16 \
V110_STEP3_F_MAX=8 \
V110_STEP3_WAIST_DIM=512 \
V110_STEP3_ALTERNATION="${V110_STEP3_ALTERNATION:-1}" \
V110_STEP3_PHASE_SCALE="${V110_STEP3_PHASE_SCALE:-1.0}" \
V110_STEP3_GATE_PROFILE="${V110_STEP3_GATE_PROFILE:-sin2_pi}" \
V110_STEP3_PHOTON_ALPHA="${V110_STEP3_PHOTON_ALPHA:-0.5}" \
V110_STEP3_BALANCE_WEIGHT="${V110_STEP3_BALANCE_WEIGHT:-0.05}" \
V110_STEP3_UNCERTAINTY_MIN="${V110_STEP3_UNCERTAINTY_MIN:-0.05}" \
V110_STEP3_HARD_BREATH_LEVEL=0 \
V110_STEP3_VAR_LOSS_WEIGHT=1.0 \
V110_STEP3_CALIB_WEIGHT=0.05 \
V110_STEP3_FACTOR_AUX_WEIGHT=0.5 \
V110_STEP3_CODEBOOK_N=32 \
V110_STEP3_IB_CENTROIDS=.cache/ib_centroids_gsm8k_partial.npz \
V110_STEP3_TRAIN=.cache/factor_graph_train.jsonl \
V110_STEP3_VAL=.cache/factor_graph_test.jsonl \
V110_STEP3_GSM8K_TRAIN=.cache/gsm8k_factor_graphs_train.jsonl \
V110_STEP3_GSM8K_RATIO=0.5 \
BATCH=8 \
STEPS="${STEPS:-500}" \
LR=3e-5 \
LOG_EVERY=10 \
PER_BREATH_CE_EVERY=50 \
EVAL_EVERY=100 \
EVAL_BATCHES=30 \
EVAL_BATCH=8 \
CKPT_EVERY=250 \
CKPT_LABEL="${CKPT_LABEL:-v110_step3_smoke}" \
RESUME_FROM="$CKPT" \
PYTHIA_INIT=1 \
"$PYTHON" -u scripts/v110_step3_factor_graph_train.py
