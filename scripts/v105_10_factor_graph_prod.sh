#!/usr/bin/env bash
# v105.10 factor graph — production run (5000 steps), DUAL READOUT.
#
# Architecture: v105.8 (200-bin per-NUMBER readout) + v105.9 (AR digit decoder)
# combined in a single architecture. BOTH readouts are computed from the same
# pooled cell_hidden at the final breath. Number CE trains the breathing for
# precision; AR digit CE trains the decoder to extract compositional digits.
#
# Loss combination (inside JIT):
#   total = ... + number_ce_loss + V105_10_DIGIT_WEIGHT * digit_ce_loss
#         (var_loss, magnitude_loss, energy_loss, aux_distinct_loss forced to 0
#          Python-side; factor_aux_loss kept at weight 1.0)
#
# Why:
#   v105.8 (200-bin number CE only) hit 29.7% cell_acc on val[medium] at step
#   5000 — strongest factor-graph paradigm result. But the codebook has no bins
#   above 9999, so OOD on [10000, 99999] is guaranteed 0%. v105.9 (AR digit
#   decoder only) confirmed the architecture works but cell_hidden was too
#   vague without number supervision (probe R² = -614, 6.3% bin acc).
#   v105.10 unifies them: number_CE provides the precision gradient, digit_CE
#   trains the decoder to extract digits compositionally. The killer experiment
#   is OOD generalization — eval on values above the training range.
#
# Warm-start from v105.8 step-5000 (the breathing is already well-trained for
# precision via number_CE; v105.10 adds digit_CE on top via the fresh-init
# digit_codebook).
#
# Forced env overrides applied Python-side when V105_10_DUAL_READOUT=1:
#   V105_5_VAR_LOSS_WEIGHT  → 0  (per-position digit CE replaced by pooled-AR)
#   V105_5_MAGNITUDE_WEIGHT → 0  (was already zero in v105.8/v105.9)
#   V105_5_ENERGY_WEIGHT    → 0  (depended on per-position digit logits)
#   V105_AUX_DISTINCT_WEIGHT → 0
#   V105_5_FACTOR_AUX_WEIGHT preserved at 1.0
#
# Usage: bash scripts/v105_10_factor_graph_prod.sh
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
V105_9_AR_COND_SCALE="${V105_9_AR_COND_SCALE:-0.5}" \
V105_10_DUAL_READOUT="${V105_10_DUAL_READOUT:-1}" \
V105_10_DIGIT_WEIGHT="${V105_10_DIGIT_WEIGHT:-0.3}" \
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
CKPT_LABEL="${CKPT_LABEL:-v105_10_prod}" \
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
