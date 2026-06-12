#!/usr/bin/env bash
# Standard-val eval of v112b_cont1_final.
#
# Recovers v112b cont1 final's val numbers under the SAME protocol cont2 uses
# (n~88/83/69 per difficulty, standard val, not 50-puzzle MC-BP subset).
# Resolves the protocol-mismatch caveat: published v112b numbers (easy 0.6371)
# are MC-BP @ noise=0.005 on 50-puzzle subsets; standard val will differ.
#
# Result gates interpretation of: cont2 trajectory means, real mechanism Δ
# vs natural drift, and any future Phase 2 comparison anchored on cont1.
set -euo pipefail

cd "$(dirname "$0")/.."

DEV='PCI+AMD' \
V110_STEP3_TASK=1 \
V110_STEP3_K_MAX=8 \
V110_STEP3_N_DIGITS=5 \
V110_STEP3_N_MAX=16 \
V110_STEP3_F_MAX=8 \
V110_STEP3_WAIST_DIM=512 \
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
V110_STEP3_CODEBOOK_N=32 \
V110_STEP3_IB_CENTROIDS=.cache/ib_centroids_gsm8k_partial.npz \
V110_STEP3_TRAIN=.cache/factor_graph_train.jsonl \
V110_STEP3_VAL=.cache/factor_graph_test.jsonl \
V110_STEP3_GSM8K_TRAIN=.cache/gsm8k_factor_graphs_train.jsonl \
V110_STEP3_GSM8K_RATIO=0.5 \
BATCH=8 \
STEPS=1 \
LR=3e-5 \
LOG_EVERY=1 \
PER_BREATH_CE_EVERY=1 \
EVAL_EVERY=1 \
EVAL_BATCHES=30 \
EVAL_BATCH=8 \
CKPT_EVERY=999 \
CKPT_LABEL=v112b_cont1_standard_val \
PYTHIA_INIT=1 \
RESUME_FROM=.cache/fg_v112b_ckpts/v112b_cont1_final.safetensors \
/home/bryce/mycelium/.venv/bin/python -u scripts/v112b_train.py 2>&1 | tee .cache/logs/v112b_cont1_standard_val.log
