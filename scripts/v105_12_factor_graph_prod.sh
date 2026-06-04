#!/usr/bin/env bash
# v105.12 factor graph — production run (15000 steps), FINAL COMPOSITIONALITY EXPERIMENT.
#
# Combines every proven v105 mechanism with three new principled additions:
#   - v105.8 200-bin number codebook readout (precision pressure)
#   - v105.9 AR digit decoder (compositional extraction)
#   - v105.10 dual readout (both active concurrently)
#   - v105.11 log-MSE through AR reconstruction (number-precision via AR path)
#   - v105.11 LN_COND (magnitude-equalization fix — AR conditioning RESPONSIVE)
#
# v105.12 additions:
#   Change 1 — V105_12_PREFILL_ISOLATE=1: magnitude head + per-position digit
#     codebook readout + magnitude_embed addition all gated to k==K-1 only.
#     Breathing loop becomes pure constraint propagation; readout once at end.
#
#   Change 2 — V105_12_FOURIER_DECODE_INIT=1: replace QR-random init of
#     fg_v105_5_digit_codebook with Fourier basis (cos/sin pairs at frequencies
#     1..H/2, angle = 2π·d·(k+1)/10, global 1/sqrt(H) scale). With prefill
#     isolation the codebook is touched only at the decode breath so the
#     Fourier structure isn't washed out.
#
#   Change 3 — V105_12_CODEBOOK_ANNEAL=1: number_codebook CE weight follows a
#     schedule (1.0 bootstrap → 0.15 maintenance) via runtime Tensor input to
#     the JIT step. Lets the codebook provide precision pressure early while
#     the AR decoder takes over for OOD compositionality later.
#
# Schedule (training driver):
#   steps 0    – 2000:   codebook_weight = 1.0  (bootstrap precision)
#   steps 2000 – 5000:   linear 1.0 → 0.15
#   steps 5000 – 15000:  codebook_weight = 0.15 (maintenance)
#
# Diagnostic checkpoints (CKPT_EVERY=500 gives step 2000/5000/10000/15000):
#   step 2000: consistency test (RESPONSIVE should be preserved via LN_COND)
#   step 5000: OOD test #1 (codebook partially annealed)
#   step 10000: OOD test #2 (grokking window)
#   step 15000: OOD test #3 (final) + matched comparison vs v107/v105.10
#
# Loss combination (inside JIT):
#   total = factor_aux + calib
#         + codebook_weight * number_ce           (v105.8, runtime-scheduled)
#         + V105_10_DIGIT_WEIGHT * var_loss_pooled (v105.9 AR per-digit CE)
#         + V105_11_NUMBER_MSE_BETA * num_mse     (v105.11 log-MSE via AR)
#
# var_loss / magnitude / energy / aux_distinct forced to 0 Python-side.
# factor_aux_weight = 1.0, calib_weight = 0.05.
#
# Warm-start from v105.8 step-5000 (matches v105.10/v105.11 anchor).
#
# Usage: bash scripts/v105_12_factor_graph_prod.sh
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
V105_8_PER_NUMBER_READOUT="${V105_8_PER_NUMBER_READOUT:-1}" \
V105_8_N_NUMBER_BINS="${V105_8_N_NUMBER_BINS:-200}" \
V105_9_AR_DIGIT_DECODER="${V105_9_AR_DIGIT_DECODER:-1}" \
V105_9_AR_COND_SCALE="${V105_9_AR_COND_SCALE:-2.0}" \
V105_10_DUAL_READOUT="${V105_10_DUAL_READOUT:-1}" \
V105_10_DIGIT_WEIGHT="${V105_10_DIGIT_WEIGHT:-0.3}" \
V105_11_NUMBER_MSE="${V105_11_NUMBER_MSE:-1}" \
V105_11_NUMBER_MSE_BETA="${V105_11_NUMBER_MSE_BETA:-1.0}" \
V105_11_CONCAT_COND="${V105_11_CONCAT_COND:-0}" \
V105_11_COND_DROPOUT="${V105_11_COND_DROPOUT:-0.0}" \
V105_11_LN_COND="${V105_11_LN_COND:-1}" \
V105_12_PREFILL_ISOLATE="${V105_12_PREFILL_ISOLATE:-1}" \
V105_12_FOURIER_DECODE_INIT="${V105_12_FOURIER_DECODE_INIT:-1}" \
V105_12_CODEBOOK_ANNEAL="${V105_12_CODEBOOK_ANNEAL:-1}" \
V105_CURRICULUM=1 \
V105_CURRICULUM_ANNEAL=1000 \
V105_TRAIN=.cache/factor_graph_train_loguniform.jsonl \
V105_VAL=.cache/factor_graph_test_loguniform.jsonl \
V105_GSM8K_TRAIN=.cache/gsm8k_factor_graphs_train.jsonl \
V105_GSM8K_RATIO="${V105_GSM8K_RATIO:-0.5}" \
BATCH="${BATCH:-8}" \
STEPS="${STEPS:-15000}" \
LR="${LR:-3e-5}" \
LOG_EVERY="${LOG_EVERY:-10}" \
PER_BREATH_CE_EVERY="${PER_BREATH_CE_EVERY:-50}" \
EVAL_EVERY="${EVAL_EVERY:-250}" \
EVAL_BATCHES="${EVAL_BATCHES:-30}" \
EVAL_BATCH="${EVAL_BATCH:-8}" \
CKPT_EVERY="${CKPT_EVERY:-500}" \
CKPT_LABEL="${CKPT_LABEL:-v105_12_prod}" \
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
