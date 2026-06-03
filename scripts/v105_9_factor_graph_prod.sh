#!/usr/bin/env bash
# v105.9 factor graph — production run (5000 steps)
#
# Architecture: v105.5 per-position token architecture (5 tokens per cell,
# Pythia L0-L3 shared, IB hierarchical codebook, projection waist) with the
# variable supervision MOVED from per-position digit CE (collapsed) and from
# v105.8's 200-bin number codebook (lossy memorization) to a SMALL AR DIGIT
# DECODER over POOLED cell_hidden.
#
# v105.9 keeps digit-decomposition where it WORKS (readout, small decoder)
# and lets the breathing collapse where it MUST (representation = consensus).
#
# Forward, last breath only:
#   cell_hidden = mean(var_tokens_with_mag, axis=2)          # (B, n_max, H)
#   For each digit position p (LSD-first):
#     logits_p = (cell_hidden + cond) @ digit_codebook[p].T
#     cond    += AR_COND_SCALE * (softmax(logits_p) @ digit_codebook[p])
#   digit_logits_pooled = stack(logits_p)                    # (B, n_max, n_digits, 10)
#
# Why this is principled:
#   - Breathing collapses to "the number is X" in cell_hidden (consensus,
#     not bug). v105 per-position CE failed because it asked the breathing
#     to disagree per-position to fit a factorized digit-product target —
#     mechanically impossible after attention averaging (cos_sim ~ 1.0).
#   - v105.8 (per-NUMBER 200-bin codebook) works but loses compositionality:
#     40k pairwise entries for 4-digit numbers, no OOD generalization to
#     unseen magnitudes.
#   - v105.9 puts digit decomposition in the READOUT (a 5-step AR chain
#     reading one pooled hidden) where collapse is fine. Per-digit CE
#     flows through ONE pooled hidden — single coherent gradient.
#
# Forced env overrides applied Python-side when V105_9_AR_DIGIT_DECODER=1:
#   V105_5_VAR_LOSS_WEIGHT  → 0  (drop per-POSITION digit CE)
#   V105_5_ENERGY_WEIGHT    → 0  (depended on per-position digit logits)
#   V105_AUX_DISTINCT_WEIGHT → 0 (no per-position pressure anymore)
#   The pooled-AR per-digit CE is added unconditionally inside the JIT step
#   with weight 1.0 — it IS the variable supervision.
#
# Training:
#   - 50/50 mix of synthetic + GSM8K (NO range filter)
#   - Curriculum: easy-only for first 1000 steps, then annealed to uniform
#   - K=8 iterative-prefill breaths
#   - T=88 sequence length
#   - V105_5_BLOCK_WITHIN_VAR=0 (positions can mix freely — collapse is fine)
#   - V105_5_PERPOS_FFN=0 (PPFFN dropped — positions are pooled anyway)
#   - V105_6_PERPOS_L0=0 (per-position L0 dropped — positions are pooled)
#   - V105_5_AR_DIGITS=0 (per-position-hidden AR is REPLACED by pooled-AR)
#   - V105_5_MAGNITUDE_WEIGHT=0.3 (kept — magnitude head reads cell_hidden,
#     complementary aux)
#   - V105_5_FACTOR_AUX_WEIGHT=1.0 (kept — per-NUMBER MSE over factor cells)
#   - Warm-start from v104_prod_step3000.safetensors (backbone + IB codebook)
#
# DO NOT launch without smoke-run passing first.
# Usage: bash scripts/v105_9_factor_graph_prod.sh
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-.venv/bin/python}"
CKPT="${CKPT:-.cache/fg_v104_ckpts/v104_prod_step3000.safetensors}"

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
V105_5_MAGNITUDE_WEIGHT="${V105_5_MAGNITUDE_WEIGHT:-0.3}" \
V105_5_VAR_LOSS_WEIGHT="${V105_5_VAR_LOSS_WEIGHT:-0}" \
V105_AUX_DISTINCT_WEIGHT="${V105_AUX_DISTINCT_WEIGHT:-0}" \
V105_5_ROPE_BASE=10000 \
V105_5_IB_INIT=1 \
V105_5_WAIST_LORA_INIT=1 \
V105_5_AR_DIGITS=0 \
V105_5_AR_COND_SCALE="${V105_5_AR_COND_SCALE:-0.5}" \
V105_5_AR_MSD_FIRST=0 \
V105_5_PERPOS_FFN="${V105_5_PERPOS_FFN:-0}" \
V105_5_PERPOS_FFN_DIM="${V105_5_PERPOS_FFN_DIM:-2048}" \
V105_5_BLOCK_WITHIN_VAR="${V105_5_BLOCK_WITHIN_VAR:-0}" \
V105_6_PERPOS_L0="${V105_6_PERPOS_L0:-0}" \
V105_8_PER_NUMBER_READOUT="${V105_8_PER_NUMBER_READOUT:-0}" \
V105_9_AR_DIGIT_DECODER="${V105_9_AR_DIGIT_DECODER:-1}" \
V105_9_AR_COND_SCALE="${V105_9_AR_COND_SCALE:-0.5}" \
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
CKPT_LABEL="${CKPT_LABEL:-v105_9_prod}" \
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
