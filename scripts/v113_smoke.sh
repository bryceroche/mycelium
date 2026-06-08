#!/usr/bin/env bash
# v113 smoke — per-digit input tokens + v112b topology gate.
#
# Architecture:
#   v112b base (photon waist + notebook + topology gate)
#   + per-digit input tokens (N_MAX * n_digits separate tokens per variable)
#   + right-aligned RoPE (ones digit = rope_pos 0, frozen sinusoidal)
#   + valid_mask: leading-zero padding tokens blocked as keys in attention
#
# T_new = N_MAX * n_digits + F_MAX = 16 * 5 + 8 = 88
# Cold-start only (v112b ckpt incompatible — different T, no domain_codebook).
#
# Smoke success criteria:
#   1. Code compiles + JIT step compiles within 5 minutes
#   2. 50-100 steps without crash/OOM/hang
#   3. Loss finite throughout
#   4. Loss decreases from step 1 to step 50+
#   5. Memory < 22GB
#   6. Per-step time < 10s
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-.venv/bin/python}"

DEV='PCI+AMD' \
V113_TASK=1 \
V113_K_MAX=8 \
V113_N_DIGITS=5 \
V113_N_MAX=16 \
V113_F_MAX=8 \
V113_WAIST_DIM=512 \
V113_TOPOLOGY_DIM=64 \
V113_CODEBOOK_N=32 \
V113_IB_CENTROIDS=.cache/ib_centroids_gsm8k_partial.npz \
V110_STEP3_ALTERNATION=1 \
V110_STEP3_PHASE_SCALE=1.0 \
V110_STEP3_GATE_PROFILE=sin2_pi \
V110_STEP3_PHOTON_ALPHA=0.5 \
V110_STEP3_BALANCE_WEIGHT=0.05 \
V110_STEP3_UNCERTAINTY_MIN=0.05 \
V110_STEP3_HARD_BREATH_LEVEL=0 \
V110_STEP3_VAR_LOSS_WEIGHT=1.0 \
V110_STEP3_CALIB_WEIGHT=0.05 \
V110_STEP3_FACTOR_AUX_WEIGHT=0.5 \
V113_TRAIN=.cache/factor_graph_train.jsonl \
V113_VAL=.cache/factor_graph_test.jsonl \
V113_GSM8K_TRAIN=.cache/gsm8k_factor_graphs_train.jsonl \
V113_GSM8K_RATIO=0.5 \
V113_SBP_NOISE_SCALE=0.0 \
BATCH=8 \
STEPS="${STEPS:-75}" \
LR=3e-5 \
LOG_EVERY=5 \
PER_BREATH_CE_EVERY=25 \
EVAL_EVERY=9999 \
EVAL_BATCHES=5 \
EVAL_BATCH=8 \
CKPT_EVERY=9999 \
CKPT_LABEL="${CKPT_LABEL:-v113_smoke}" \
PYTHIA_INIT=1 \
"$PYTHON" -u scripts/v113_train.py
