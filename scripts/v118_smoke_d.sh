#!/usr/bin/env bash
# v118-d smoke: CONTROL — sin2_pi gate profile, Perceiver=0, LoRAs=0.
#
# Sanity control: should behave identically to v112b training at step 0.
# Uses the existing sin2_pi gate profile (v112b/v114 default) with NO new
# mechanisms. This confirms the v118 training driver is correct and that
# the new code path doesn't break the baseline.
#
# Success criteria:
#   1. Code + JIT compile cleanly
#   2. Step 0 loss should match v112b's first warm-start step
#   3. hard cell_acc at step 300 should match v112b baseline ~0.3945
#   4. This is the "do nothing new" baseline for the ablation table
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-.venv/bin/python}"

V110_STEP3_TASK=1 \
V118_GATE_PROFILE=sin2_pi \
V118_PERCEIVER_ENABLED=0 \
V118_PER_PHASE_LORAS=0 \
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
CKPT_LABEL="${CKPT_LABEL:-v118_d_ctrl}" \
PYTHIA_INIT=1 \
DEV='PCI+AMD' \
"$PYTHON" -u scripts/v118_train.py
