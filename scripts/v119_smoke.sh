#!/usr/bin/env bash
# v119 smoke: v118-b config + perceiver bootstrap fix (V119_PERCEIVER_W_OUT=1).
#
# Compared to v118-b which had perc_gate stuck at -0.0005 across 300 steps,
# v119 adds a zero-init (H, H) output projection W_perc_out so:
#   delta = ctx_e @ W_perc_out    (= 0 at init)
#   out = x + g * delta            (gate opens onto zero-magnitude delta)
#
# Two-knob graduation pattern: gate AND W_out both need to learn together.
# Same pattern that worked for v112b residual gate (topology @ W_res_gate).
#
# Success signals:
#   1. perc_gate breaks through ±0.005 by step 100 (vs stuck at -0.0005 in v118-b)
#   2. perc_W_out_norm grows above zero (validates W_out learning)
#   3. val[hard] beats v118-b's 0.322 (was -0.054 vs v112b baseline)
#   4. Ideally val[hard] matches or beats v118-a's 0.362 (no perceiver baseline)
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-.venv/bin/python}"

V110_STEP3_TASK=1 \
V118_GATE_PROFILE=piecewise_4phase \
V118_PERCEIVER_ENABLED=1 \
V118_PERCEIVER_N_LATENTS=16 \
V118_PER_PHASE_LORAS=0 \
V119_PERCEIVER_W_OUT=1 \
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
CKPT_LABEL="${CKPT_LABEL:-v119_smoke}" \
PYTHIA_INIT=1 \
DEV='PCI+AMD' \
"$PYTHON" -u scripts/v118_train.py
