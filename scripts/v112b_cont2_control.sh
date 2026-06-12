#!/usr/bin/env bash
# v112b cont2 natural-continuation control.
#
# Dual-purpose run (per Jun 9 methodology upgrade):
#   1. Bounds how much of v112b cont1's +0.0564 (easy) / +0.0225 (med) /
#      +0.0123 (hard) was drift vs mechanism. If 5000 more cont steps move
#      val numbers significantly with no new mechanism, the Phase 1
#      attribution needs caveats.
#   2. Establishes the proper baseline for Phase 2 per-position tests. Every
#      Phase 2 mechanism warm-starts from v112b_cont1_final; the right
#      comparison anchor is v112b_cont2_step{N}, not frozen v112b_cont1_final.
#
# Intermediate evals at step 500/1000/2500/5000 (read drift trajectory early).
# 5000 steps, ~2hr on AMD 7900 XTX. Uses V110_STEP3_* env vars (v112b_train
# reuses v110-step3's config surface).
set -euo pipefail

cd "$(dirname "$0")/.."

CKPT="${CKPT:-.cache/fg_v112b_ckpts/v112b_cont1_final.safetensors}"
STEPS="${STEPS:-5000}"

if [ ! -f "$CKPT" ]; then
    echo "ERROR: warm-start ckpt not found: $CKPT"
    exit 1
fi

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
STEPS="$STEPS" \
LR=3e-5 \
LOG_EVERY=50 \
PER_BREATH_CE_EVERY=250 \
EVAL_EVERY=500 \
EVAL_BATCHES=30 \
EVAL_BATCH=8 \
CKPT_EVERY=500 \
CKPT_LABEL=v112b_cont2_control \
PYTHIA_INIT=1 \
RESUME_FROM="$CKPT" \
/home/bryce/mycelium/.venv/bin/python -u scripts/v112b_train.py 2>&1 | tee .cache/logs/v112b_cont2_control.log
