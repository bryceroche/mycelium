#!/usr/bin/env bash
# v105.11 factor graph — production run (5000 steps), DROP-THE-CODEBOOK.
#
# Architecture: AR digit decoder only (v105.9 base), NO codebook readout.
#   - Per-digit CE on pooled-cell AR digit logits (the breathing must shape
#     cell_hidden for digit extraction; no codebook shortcut).
#   - Log-number-MSE on the AR-reconstructed soft expected value
#     (provides precise-number gradient through the SAME AR path as CE).
#   - Stronger AR cond_scale (2.0 vs 0.5 in v105.10) — Mechanism 1 forcing the
#     conditioning to be load-bearing.
#
# Loss combination (inside JIT):
#   total = factor_aux + calib + var_loss_pooled + V105_11_NUMBER_MSE_BETA * log_mse
#         (var_loss, magnitude, energy, aux_distinct forced to 0 Python-side via
#          the v105.11 mutex + the v105.9 cascade; factor_aux kept at 1.0)
#
# Why:
#   v105.10 step 5000 diagnostics returned three findings:
#     1. In-distribution 5-token win is real (v105.10 +16.8pt over v107 on val[medium]).
#     2. OOD compositionality FAILED (per-digit acc 10.3% = chance on [10010, 99998]).
#     3. AR conditioning DECORATIVE (consistency test UNRESPONSIVE — d2 ignores d1).
#   Result 3 explains result 2. v105.11 (a) drops the codebook so the
#   breathing has no shortcut, (b) replaces the precision signal via log-MSE
#   through AR reconstruction, AND (c) forces the AR conditioning to be
#   load-bearing via V105_9_AR_COND_SCALE=2.0.
#
# Warm-start from v105.8 step-5000: the breathing is already well-trained for
# precision via number_CE; v105.11 starts from there and lets the AR decoder
# bear all the load. The fresh digit_codebook (per-position, attached fresh)
# is randomly initialized.
#
# Forced env overrides applied Python-side when V105_11_NUMBER_MSE=1 (mirror
# of the module-load mutex that flips V105_8 off, V105_9 on, V105_10 off):
#   V105_5_VAR_LOSS_WEIGHT  → 0  (per-position digit CE replaced by pooled-AR)
#   V105_5_MAGNITUDE_WEIGHT → 0  (magnitude head decoupled from cell_hidden)
#   V105_5_ENERGY_WEIGHT    → 0  (depended on per-position digit logits)
#   V105_AUX_DISTINCT_WEIGHT → 0
#   V105_5_FACTOR_AUX_WEIGHT preserved at 1.0
#
# Usage: bash scripts/v105_11_factor_graph_prod.sh
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-.venv/bin/python}"
CKPT="${CKPT:-.cache/fg_v105_5_ckpts/v105_8_prod_step5000.safetensors}"

DEV='PCI+AMD' \
V105_5_TASK=1 \
V105_5_K_MAX=8 \
V105_5_N_DIGITS=5 \
V105_5_N_MAX=16 \
V105_5_F_MAX=8 \
V105_5_WAIST="${V105_5_WAIST:-512}" \
V105_5_CODEBOOK_N="${V105_5_CODEBOOK_N:-32}" \
V105_5_IB_CENTROIDS=.cache/ib_centroids_gsm8k_partial.npz \
V105_5_IB_TREE=.cache/ib_tree_gsm8k_partial.json \
V105_5_ENERGY_WEIGHT="${V105_5_ENERGY_WEIGHT:-0}" \
V105_5_FACTOR_AUX_WEIGHT="${V105_5_FACTOR_AUX_WEIGHT:-1.0}" \
V105_5_CALIB_WEIGHT="${V105_5_CALIB_WEIGHT:-0.05}" \
V105_5_MAGNITUDE_WEIGHT="${V105_5_MAGNITUDE_WEIGHT:-0}" \
V105_5_VAR_LOSS_WEIGHT="${V105_5_VAR_LOSS_WEIGHT:-0}" \
V105_AUX_DISTINCT_WEIGHT="${V105_AUX_DISTINCT_WEIGHT:-0}" \
V105_5_ROPE_BASE=10000 \
V105_5_IB_INIT=1 \
V105_5_WAIST_LORA_INIT=1 \
V105_5_AR_DIGITS=1 \
V105_5_AR_COND_SCALE="${V105_5_AR_COND_SCALE:-0.5}" \
V105_5_AR_MSD_FIRST=0 \
V105_5_PERPOS_FFN="${V105_5_PERPOS_FFN:-0}" \
V105_5_PERPOS_FFN_DIM="${V105_5_PERPOS_FFN_DIM:-2048}" \
V105_5_BLOCK_WITHIN_VAR="${V105_5_BLOCK_WITHIN_VAR:-0}" \
V105_6_PERPOS_L0="${V105_6_PERPOS_L0:-0}" \
V105_8_PER_NUMBER_READOUT="${V105_8_PER_NUMBER_READOUT:-0}" \
V105_8_N_NUMBER_BINS="${V105_8_N_NUMBER_BINS:-200}" \
V105_9_AR_DIGIT_DECODER="${V105_9_AR_DIGIT_DECODER:-1}" \
V105_9_AR_COND_SCALE="${V105_9_AR_COND_SCALE:-2.0}" \
V105_10_DUAL_READOUT="${V105_10_DUAL_READOUT:-0}" \
V105_10_DIGIT_WEIGHT="${V105_10_DIGIT_WEIGHT:-0.3}" \
V105_11_NUMBER_MSE="${V105_11_NUMBER_MSE:-1}" \
V105_11_NUMBER_MSE_BETA="${V105_11_NUMBER_MSE_BETA:-1.0}" \
V105_11_CONCAT_COND="${V105_11_CONCAT_COND:-0}" \
V105_11_COND_DROPOUT="${V105_11_COND_DROPOUT:-0.0}" \
V105_11_LN_COND="${V105_11_LN_COND:-1}" \
V105_CURRICULUM=1 \
V105_CURRICULUM_ANNEAL=1000 \
V105_TRAIN=.cache/factor_graph_train_loguniform.jsonl \
V105_VAL=.cache/factor_graph_test_loguniform.jsonl \
V105_GSM8K_TRAIN=.cache/gsm8k_factor_graphs_train.jsonl \
V105_GSM8K_RATIO="${V105_GSM8K_RATIO:-0.5}" \
BATCH="${BATCH:-8}" \
STEPS="${STEPS:-5000}" \
LR="${LR:-3e-5}" \
LOG_EVERY="${LOG_EVERY:-10}" \
PER_BREATH_CE_EVERY="${PER_BREATH_CE_EVERY:-50}" \
EVAL_EVERY="${EVAL_EVERY:-250}" \
EVAL_BATCHES="${EVAL_BATCHES:-30}" \
EVAL_BATCH="${EVAL_BATCH:-8}" \
CKPT_EVERY="${CKPT_EVERY:-500}" \
CKPT_LABEL="${CKPT_LABEL:-v105_11_ln_prod}" \
PYTHIA_INIT=1 \
RESUME_FROM="$CKPT" \
SUDOKU_TASK=0 \
V99_TASK=0 \
V100_TASK=0 \
V101_TASK=0 \
V104_TASK=0 \
V105_TASK=0 \
V105_1_2_TASK=0 \
V105_2_TASK=0 \
V105_4_TASK=0 \
"$PYTHON" -u scripts/v105_5_factor_graph_train.py
