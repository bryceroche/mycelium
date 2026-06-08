#!/usr/bin/env bash
# v118-c smoke: FULL — piecewise_4phase + Perceiver + 4 phase-gated LoRAs.
#
# Tests all three mechanisms together. This is the heaviest v118 variant.
# LoRA params: rank=16 × 6 projections × 4 layers × 4 phases = ~3M new params.
# Up-projections are zero-init → step 0 forward byte-identical to v112b.
#
# JIT note: LoRA phase selection is Python-level (not inside JIT). If JIT
# compile takes >10min, reduce LORA_RANK (try 8 or 4) or file a bug.
#
# Success criteria:
#   1. Code + JIT compile cleanly (monitor compile time)
#   2. LoRA up-projections start at zero, grow from gradient
#   3. 300 steps without crash/OOM/hang
#   4. hard cell_acc at step 300 vs v118-b (same - LoRAs)
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-.venv/bin/python}"

V110_STEP3_TASK=1 \
V118_GATE_PROFILE=piecewise_4phase \
V118_PERCEIVER_ENABLED=1 \
V118_PERCEIVER_N_LATENTS=16 \
V118_PER_PHASE_LORAS=1 \
V118_LORA_RANK=16 \
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
CKPT_LABEL="${CKPT_LABEL:-v118_c_smoke}" \
PYTHIA_INIT=1 \
DEV='PCI+AMD' \
"$PYTHON" -u scripts/v118_train.py
