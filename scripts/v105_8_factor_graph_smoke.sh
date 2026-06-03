#!/usr/bin/env bash
# v105.8 factor graph — smoke run (50 steps)
#
# Architecture: keep v105.5 per-position token architecture (5 tokens per cell,
# Pythia L0-L3 shared, IB hierarchical codebook, projection waist, magnitude
# head still RUNS in forward — but its CE contribution is zero). Replace the
# variable supervision: instead of per-digit CE (which assumed independent
# decoding from a shared hidden state and let positions collapse to cos 1.0),
# train against a SINGLE per-NUMBER CE on the mean-pooled cell hidden state
# projected through a (n_bins, H) number_codebook (200-bin hybrid v107 scheme).
#
# WHY:
#   v105 per-layer cos_sim diagnostic on v105.6 step 1000 pinned the collapse
#   to L0 (input cos 0.74 → post-L0 0.90; everything after L3 preserves 0.991).
#   Per-digit CE = CE on the joint distribution under conditional independence,
#   so position-specific codebook entries provide the SAME gradient as a
#   per-NUMBER CE on the joint when positions share a hidden state. The loss
#   form can't escape collapse. v105.8 drops per-digit CE entirely; positions
#   are now allowed to collapse without losing gradient signal (the mean of
#   collapsed identical vectors is still a valid number representation).
#
# Acceptance criteria:
#   1. No NaN (isfinite check passes every step)
#   2. Step time <= 5.0s after JIT warmup
#   3. number_ce_loss decreases over 50 steps
#   4. total loss decreases over 50 steps
#
# Warm-start from v104_prod_step3000.safetensors:
#   - Backbone (shared.*, phase*.*, ln_f.*) loaded
#   - fg_v104.codebook → fg_v105_5.ib_codebook
#   - number_codebook init: random orthonormal × 0.1 (no shape match in v104;
#     200 bins are introduced by v105.8/v107). If a v107 ckpt is also present
#     the fg_v107.domain_codebook is preferred (exact bin-scheme match).
#
# Usage:
#   bash scripts/v105_8_factor_graph_smoke.sh
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
V105_5_WAIST=512 \
V105_5_CODEBOOK_N=32 \
V105_5_IB_CENTROIDS=.cache/ib_centroids_gsm8k_partial.npz \
V105_5_IB_TREE=.cache/ib_tree_gsm8k_partial.json \
V105_5_ENERGY_WEIGHT=0.0 \
V105_5_FACTOR_AUX_WEIGHT=1.0 \
V105_5_CALIB_WEIGHT=0.05 \
V105_5_MAGNITUDE_WEIGHT=0.0 \
V105_5_VAR_LOSS_WEIGHT=0.0 \
V105_AUX_DISTINCT_WEIGHT=0.0 \
V105_5_ROPE_BASE=10000 \
V105_5_IB_INIT=1 \
V105_5_WAIST_LORA_INIT=1 \
V105_5_AR_DIGITS=1 \
V105_5_AR_COND_SCALE=0.5 \
V105_5_AR_MSD_FIRST=0 \
V105_5_PERPOS_FFN=0 \
V105_5_BLOCK_WITHIN_VAR=0 \
V105_6_PERPOS_L0=0 \
V105_8_PER_NUMBER_READOUT=1 \
V105_8_N_NUMBER_BINS=200 \
V105_DIFFICULTY_FILTER=easy \
V105_TRAIN=.cache/factor_graph_train.jsonl \
V105_VAL=.cache/factor_graph_test.jsonl \
V105_GSM8K_TRAIN=.cache/gsm8k_factor_graphs_train.jsonl \
V105_GSM8K_RATIO=0.3 \
BATCH=4 \
STEPS=50 \
LR=3e-5 \
LOG_EVERY=5 \
PER_BREATH_CE_EVERY=10 \
EVAL_EVERY=25 \
EVAL_BATCHES=5 \
EVAL_BATCH=4 \
CKPT_EVERY=500 \
CKPT_LABEL=v105_8_smoke \
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
