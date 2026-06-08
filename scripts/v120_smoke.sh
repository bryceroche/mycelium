#!/usr/bin/env bash
# v120 smoke: persistent cross-breath latent notebook.
#
# FOURTH attempt at perceiver mechanism — redesigned as NOTEBOOK (not FILTER):
#   Latents persist across all K=8 breaths. Each breath:
#     READ  (residual attends to latents)
#     BREATH (standard v112b 4-layer forward)
#     WRITE  (latents updated from residual)
#   IB-centroid init → meaningful cross-attention from step 0.
#   Two zero-init W_out + (1+g) amplifier → gradient flows from step 1.
#
# Warm-starts from v112b_cont1_final.
# Step-0 forward should be byte-identical to v112b (W_out=0 → delta=0).
#
# Success criteria:
#   1. Code + JIT compile cleanly
#   2. 300 steps without crash/OOM/hang
#   3. Loss finite throughout
#   4. read_W_out_norm and write_W_out_norm BOTH growing from 0 by step 100
#      (this is the bootstrap signal; if both stay 0, mechanism is still locked)
#   5. hard cell_acc at step 300 ≥ v112b baseline 0.3945
#
# v120-specific diagnostics logged every 10 steps:
#   r_gate   — read gate scalar (expect: grows slowly from 0)
#   w_gate   — write gate scalar (expect: grows slowly from 0)
#   rW       — read_W_out Frobenius norm (expect: >0.001 by step 50)
#   wW       — write_W_out Frobenius norm (expect: follows rW with lag)
#   lat_drift — mean norm delta from IB init (expect: small but non-zero by step 100)
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
V120_IB_CENTROIDS=.cache/ib_centroids_gsm8k_partial.npz \
V120_N_LATENTS=16 \
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
CKPT_LABEL="${CKPT_LABEL:-v120_smoke}" \
PYTHIA_INIT=1 \
DEV='PCI+AMD' \
"$PYTHON" -u scripts/v120_train.py
