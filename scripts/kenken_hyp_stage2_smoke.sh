#!/usr/bin/env bash
# STAGE-2 RELAXATION smoke — HYPERBOLIC ARM (spec docs/hyperbolic_mask_generator_spec.md
# §8.1 capacity-matching + §8.6 Stage 2: the DeepSets g_phi STRUCTURE ENCODER).
#
# Stage 1 proved the slot-anchor cage field CANNOT generalize (held-out flat 0.827). Stage
# 2 replaces it with cage_coord = anchor[id] + g_phi(cell-set), a SHARED encoder that can
# zero-shot interpolate to UNSEEN cage configs. g_phi (pos_emb + phi + rho MLPs) JOINS the
# coord-only param group; the v98 backbone STAYS FROZEN. rho's output layer is zero-init,
# so g_phi==0 at t=0 -> the bias is the BIT-EXACT validated foothold mask.
#
# ~15 steps. Confirms (the FIRST DeepSets + d_hyp backward on the AM driver):
#   - ZERO-INIT t=0 bias == foothold hyperbolic bias to ~1e-3 (the anchor check),
#   - backward-through (segment-mean -> phi/rho -> d_hyp) COMPILES on the AM driver,
#   - g_phi grad NORMS are FINITE + bounded (the [coord-grad] line's gphi= field),
#   - backbone params bit-identical pre/post (the [freeze-check] lines),
#   - the held-out per-band eval runs.
# Throwaway RUN_NAME + low CKPT_EVERY. alpha_margin=1.5 + RELAX_BLOCK_ARG=6 (soft mask ->
# two-sided gradient, t=0 leak <1e-3 per Stage-1 FINDING A).
set -euo pipefail
cd "$(dirname "$0")/.."

CKPT=".cache/kenken_ckpts/kenken_curric_k16_cont/kenken_curric_k16_cont_final.safetensors"

KENKEN_TASK=1 \
KENKEN_HYP_MASK=1 \
KENKEN_HYP_RELAX=1 \
KENKEN_HYP_GPHI=1 \
KENKEN_HYP_EUCLID=0 \
KENKEN_HYP_ALPHA_MARGIN=1.5 \
KENKEN_HYP_RELAX_BLOCK_ARG=6.0 \
KENKEN_HYP_GPHI_DPOS=32 \
KENKEN_HYP_GPHI_WIDTH=64 \
KENKEN_HYP_GPHI_LAYERS=2 \
KENKEN_HYP_LR=1e-4 \
KENKEN_HYP_WARMUP=200 \
KENKEN_HYP_GRAD_CLIP=1.0 \
KENKEN_HYP_JITTER=1e-3 \
KENKEN_HYP_MAX_ZNORM=0.9 \
KENKEN_K_MAX=16 \
BATCH=8 \
STEPS=15 \
LR=3e-5 \
CKPT_EVERY=10 \
EVAL_EVERY=15 \
EVAL_BATCHES=4 \
LOG_EVERY=1 \
PER_BREATH_CE_EVERY=50 \
GC_EVERY=50 \
RUN_NAME=kenken_hyp_stage2_smoke \
RESUME_FROM="$CKPT" \
KENKEN_TRAIN=.cache/kenken_train_curriculum.jsonl \
KENKEN_TEST=.cache/kenken_test_curriculum.jsonl \
PYTHIA_INIT=1 \
SEED=42 \
.venv/bin/python scripts/kenken_train.py
