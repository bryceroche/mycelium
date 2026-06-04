#!/usr/bin/env bash
# v105.13 wave-guide smoke (500 steps from v105.12 step 5000 warm-start).
#
# Architecture: v105.5 base + LoRA-init 1024→512→1024 waist
#   + WAVE-GUIDE MASK on the LoRA correction (V105_13_WAVEGUIDE_PRESERVE_DIMS=512)
#
# Wave-guide behavior:
#   commit channel (dims 0..511):    receives full LoRA correction each breath
#   preserve channel (dims 512..1023): receives NO correction, passes through
#   At init: byte-identical to v105.12 (W_expand=0 → quantize=0 → mask irrelevant)
#
# Hypothesis (Bryce + other-Claude):
#   The carry signal is high-frequency info that gets smoothed by repeated
#   waist compression. A skip-channel through the waist preserves it.
#
# Caveat (from v108 analysis):
#   v105 family also has the L0 multi-token collapse problem (5 tokens per
#   variable averaged at Pythia L0). Wave-guide alone doesn't fix L0 collapse,
#   so per-position digit acc may still be capped. We test the wave-guide
#   alongside this confound; result is informative either way.
#
# Smoke acceptance:
#   1. No NaN, step time <= 2.5s
#   2. Byte-identical eval at step 0 vs v105.12 step 5000 (sanity check that
#      mask doesn't break ckpt loading)
#   3. Per-position pool readout — track pos3/pos4 in particular
#      v105.12 baseline: pool cell_acc = 6.4% medium
#      Wave-guide WIN: pool cell_acc > 10% medium (pos3/pos4 lift)
#      Wave-guide NULL: stays at 5-7% (preserve channel doesn't help)
#
# Usage:
#   bash scripts/v105_13_waveguide_smoke.sh
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-.venv/bin/python}"
CKPT="${CKPT:-.cache/fg_v105_5_ckpts/v105_12_prod_step5000_source.safetensors}"

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
V105_13_WAVEGUIDE_PRESERVE_DIMS=512 \
V105_5_AR_DIGITS=1 \
V105_5_AR_COND_SCALE="${V105_5_AR_COND_SCALE:-0.5}" \
V105_5_AR_MSD_FIRST=0 \
V105_5_PERPOS_FFN="${V105_5_PERPOS_FFN:-0}" \
V105_5_PERPOS_FFN_DIM="${V105_5_PERPOS_FFN_DIM:-2048}" \
V105_5_BLOCK_WITHIN_VAR="${V105_5_BLOCK_WITHIN_VAR:-0}" \
V105_6_PERPOS_L0="${V105_6_PERPOS_L0:-0}" \
V105_8_PER_NUMBER_READOUT=1 \
V105_8_N_NUMBER_BINS=200 \
V105_9_AR_DIGIT_DECODER=1 \
V105_9_AR_COND_SCALE=2.0 \
V105_10_DUAL_READOUT=1 \
V105_10_DIGIT_WEIGHT=0.3 \
V105_11_NUMBER_MSE=1 \
V105_11_NUMBER_MSE_BETA=1.0 \
V105_11_CONCAT_COND=0 \
V105_11_COND_DROPOUT=0.0 \
V105_11_LN_COND=1 \
V105_12_PREFILL_ISOLATE=1 \
V105_12_FOURIER_DECODE_INIT=1 \
V105_12_CODEBOOK_ANNEAL=0 \
V105_CURRICULUM=0 \
V105_TRAIN=.cache/factor_graph_train_loguniform.jsonl \
V105_VAL=.cache/factor_graph_test_loguniform.jsonl \
V105_GSM8K_TRAIN=.cache/gsm8k_factor_graphs_train.jsonl \
V105_GSM8K_RATIO="${V105_GSM8K_RATIO:-0.5}" \
BATCH="${BATCH:-8}" \
STEPS="${STEPS:-500}" \
LR="${LR:-3e-5}" \
LOG_EVERY="${LOG_EVERY:-10}" \
PER_BREATH_CE_EVERY="${PER_BREATH_CE_EVERY:-50}" \
EVAL_EVERY="${EVAL_EVERY:-100}" \
EVAL_BATCHES="${EVAL_BATCHES:-20}" \
EVAL_BATCH="${EVAL_BATCH:-8}" \
CKPT_EVERY="${CKPT_EVERY:-250}" \
CKPT_LABEL="${CKPT_LABEL:-v105_13_waveguide_smoke}" \
PYTHIA_INIT=1 \
RESUME_FROM="$CKPT" \
SUDOKU_TASK=0 \
V99_TASK=0 V100_TASK=0 V101_TASK=0 V104_TASK=0 \
V105_TASK=0 V105_1_2_TASK=0 V105_2_TASK=0 V105_4_TASK=0 \
"$PYTHON" -u scripts/v105_5_factor_graph_train.py
