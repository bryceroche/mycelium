#!/usr/bin/env bash
# v121 smoke: energy-selected perceiver writes — 300 steps warm from v112b_cont1_final.
#
# Mechanism: perceiver reads intermediate Pythia layer states (L0..L3 outputs),
# computes per-position energy, soft-selects high-energy positions, augments
# the accumulate notebook write via zero-init W_select.
#
# Expected at step 0: byte-identical output to v112b (W_select=0).
# Expected by step 100: select_norm > 0.05 (gradient analysis confirms non-zero grad).
# Expected at step 300: val[hard] > 0.362 (beats v118-a no-perceiver baseline).
#
# PASS: select_norm > 0.05 by step 100 AND val[hard] > 0.362 by step 300
# FAIL: select_norm flat OR val[hard] < 0.355
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-.venv/bin/python}"

V110_STEP3_TASK=1 \
V110_STEP3_GATE_PROFILE=sin2_pi \
V110_STEP3_K_MAX=8 \
V110_STEP3_N_DIGITS=5 \
V110_STEP3_N_MAX=16 \
V110_STEP3_F_MAX=8 \
V110_STEP3_WAIST_DIM=512 \
V112B_TOPOLOGY_DIM=64 \
V110_STEP3_CODEBOOK_N=32 \
V110_STEP3_IB_CENTROIDS=.cache/ib_centroids_gsm8k_partial.npz \
V110_STEP3_ALTERNATION=1 \
V110_STEP3_PHASE_SCALE=1.0 \
V110_STEP3_PHOTON_ALPHA=0.5 \
V110_STEP3_BALANCE_WEIGHT=0.05 \
V110_STEP3_UNCERTAINTY_MIN=0.05 \
V110_STEP3_HARD_BREATH_LEVEL=0 \
V110_STEP3_VAR_LOSS_WEIGHT=1.0 \
V110_STEP3_CALIB_WEIGHT=0.05 \
V110_STEP3_FACTOR_AUX_WEIGHT=0.5 \
V110_STEP3_TRAIN=.cache/factor_graph_train.jsonl \
V110_STEP3_VAL=.cache/factor_graph_test.jsonl \
V110_STEP3_GSM8K_TRAIN=.cache/gsm8k_factor_graphs_train.jsonl \
V110_STEP3_GSM8K_RATIO=0.5 \
V110_STEP3_SBP_NOISE_SCALE=0.0 \
V121_ENERGY_TEMP_INIT=1.0 \
WARM_FROM=v112b_cont1_final \
BATCH=8 \
STEPS="${STEPS:-300}" \
LR=3e-5 \
LOG_EVERY=10 \
PER_BREATH_CE_EVERY=50 \
EVAL_EVERY=300 \
EVAL_BATCHES=10 \
EVAL_BATCH=8 \
CKPT_EVERY=300 \
CKPT_LABEL="${CKPT_LABEL:-v121_smoke}" \
PYTHIA_INIT=1 \
DEV='PCI+AMD' \
"$PYTHON" -u scripts/v121_train.py
