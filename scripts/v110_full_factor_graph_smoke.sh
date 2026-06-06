#!/usr/bin/env bash
# v110-full smoke — v109pi + ACCUMULATE notebook + photon-shaped waist gate.
#
# Architecture (warm-start from v109pi_cont9_step500):
#   v109pi base (waist + alternation + per-breath Q rotation)
#   + ACCUMULATE notebook (v110-acc, write-once K-slot memory)
#   + sin² photon gate (v110-photon, smooth B-field/E-field phase-lock)
#
# Tests whether the lifts compound: hard from notebook (v110-acc gave +0.022)
# AND easy from smooth gate (v110-photon gave +0.009).
#
# Notebook W_o is zero-init, photon alpha=0.5 blend with binary alternation.
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-.venv/bin/python}"
CKPT="${CKPT:-.cache/fg_v109pi_ckpts/v109pi_cont9_step500.safetensors}"

DEV='PCI+AMD' \
V110_FULL_TASK=1 \
V110_FULL_K_MAX=8 \
V110_FULL_N_DIGITS=5 \
V110_FULL_N_MAX=16 \
V110_FULL_F_MAX=8 \
V110_FULL_WAIST_DIM=512 \
V110_FULL_ALTERNATION="${V110_FULL_ALTERNATION:-1}" \
V110_FULL_PHASE_SCALE="${V110_FULL_PHASE_SCALE:-1.0}" \
V110_FULL_GATE_PROFILE="${V110_FULL_GATE_PROFILE:-sin2_pi}" \
V110_FULL_PHOTON_ALPHA="${V110_FULL_PHOTON_ALPHA:-0.5}" \
V110_FULL_HARD_BREATH_LEVEL=0 \
V110_FULL_VAR_LOSS_WEIGHT=1.0 \
V110_FULL_CALIB_WEIGHT=0.05 \
V110_FULL_FACTOR_AUX_WEIGHT=0.5 \
V110_FULL_CODEBOOK_N=32 \
V110_FULL_IB_CENTROIDS=.cache/ib_centroids_gsm8k_partial.npz \
V110_FULL_TRAIN=.cache/factor_graph_train.jsonl \
V110_FULL_VAL=.cache/factor_graph_test.jsonl \
V110_FULL_GSM8K_TRAIN=.cache/gsm8k_factor_graphs_train.jsonl \
V110_FULL_GSM8K_RATIO=0.5 \
BATCH=8 \
STEPS="${STEPS:-500}" \
LR=3e-5 \
LOG_EVERY=10 \
PER_BREATH_CE_EVERY=50 \
EVAL_EVERY=100 \
EVAL_BATCHES=30 \
EVAL_BATCH=8 \
CKPT_EVERY=250 \
CKPT_LABEL="${CKPT_LABEL:-v110_full_smoke}" \
RESUME_FROM="$CKPT" \
PYTHIA_INIT=1 \
"$PYTHON" -u scripts/v110_full_factor_graph_train.py
