#!/usr/bin/env bash
# v109 smoke — v108 base + 512d LoRA waist + alternation (even breaths commit)
#
# Architecture:
#   v108 backbone (Pythia-410M L0-L3, single-token-per-variable, tree codebook)
#   + 1024 → 512 → 1024 LoRA-init waist (W_expand zero-init for byte-safe warm-start)
#   + ALTERNATION: waist active on EVEN breaths (0,2,4,6); BYPASSED on ODD (1,3,5,7)
#
# Warm-start from v108 step 500. At step 0 the waist is identity (W_expand=0),
# so eval should match v108 step 500 exactly. As training updates W_expand,
# the waist becomes load-bearing on commit breaths only.
#
# Smoke success criteria:
#   1. No NaN, step time <= 2.5s
#   2. At step 0: byte-identical to v108 (sanity check)
#   3. Per-breath ladder forms by step 200 (>0.05 delta)
#   4. In-dist cell_acc matches or exceeds v108 baseline (0.11 medium, 0.05 hard)
#   5. PRIMARY: K-sweep at step 500 shows pos4 hard CHANGE direction vs v108
#      v108 baseline:  pos4 hard 0.191 → 0.138 → 0.132 (decreases with K)
#      v109 target:    pos4 hard rises or stays flat with K (alternation hypothesis)
#
# Set V109_ALTERNATION=0 to test the v101-style "waist every breath" variant
# (control: does alternation help vs always-on waist?).
#
# Usage:
#   bash scripts/v109_factor_graph_smoke.sh
#   V109_ALTERNATION=0 CKPT_LABEL=v109_always_on_smoke bash scripts/v109_factor_graph_smoke.sh
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-.venv/bin/python}"

DEV='PCI+AMD' \
V109_TASK=1 \
V109_K_MAX=8 \
V109_N_DIGITS=5 \
V109_N_MAX=16 \
V109_F_MAX=8 \
V109_WAIST_DIM=512 \
V109_ALTERNATION="${V109_ALTERNATION:-1}" \
V109_HARD_BREATH_LEVEL=0 \
V109_VAR_LOSS_WEIGHT=1.0 \
V109_CALIB_WEIGHT=0.05 \
V109_FACTOR_AUX_WEIGHT=0.5 \
V109_CODEBOOK_N=32 \
V109_IB_CENTROIDS=.cache/ib_centroids_gsm8k_partial.npz \
V109_DIFFICULTY_FILTER=easy \
V109_TRAIN=.cache/factor_graph_train.jsonl \
V109_VAL=.cache/factor_graph_test.jsonl \
V109_GSM8K_TRAIN=.cache/gsm8k_factor_graphs_train.jsonl \
V109_GSM8K_RATIO=0.5 \
BATCH=4 \
STEPS=500 \
LR=3e-5 \
LOG_EVERY=10 \
PER_BREATH_CE_EVERY=50 \
EVAL_EVERY=100 \
EVAL_BATCHES=10 \
EVAL_BATCH=4 \
CKPT_EVERY=250 \
CKPT_LABEL="${CKPT_LABEL:-v109_smoke}" \
RESUME_FROM="${RESUME_FROM:-.cache/fg_v108_ckpts/v108_smoke_step500.safetensors}" \
PYTHIA_INIT=1 \
"$PYTHON" -u scripts/v109_factor_graph_train.py
