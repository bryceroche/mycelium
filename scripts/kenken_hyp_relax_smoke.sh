#!/usr/bin/env bash
# STAGE-1 RELAXATION smoke (spec docs/hyperbolic_mask_generator_spec.md §8.2/§8.6 Stage 1).
#
# The CRITICAL SAFETY TEST: this is the FIRST time the backward pass through d_hyp runs,
# so the 1/(1-|z|^2) boundary-gradient landmine is LIVE. ~15 steps with the v98 backbone
# FROZEN and ONLY the ~3 hyperbolic coord tensors trained. Confirms:
#   - backward-through-d_hyp COMPILES on the AM driver,
#   - coord grad NORMS are FINITE + bounded (the headline safety read — watch [coord-grad]),
#   - loss finite (no NaN/explosion),
#   - backbone params bit-identical pre/post (the [freeze-check] lines),
#   - the held-out per-band eval runs.
#
# Throwaway RUN_NAME + low CKPT_EVERY (one mid-run coord ckpt) so no real backbone ckpt is
# relied on. alpha_margin lowered to 1.5 (the §8.2 boundary-gradient softening before fire).
set -euo pipefail
cd "$(dirname "$0")/.."

CKPT=".cache/kenken_ckpts/kenken_curric_k16_cont/kenken_curric_k16_cont_final.safetensors"

KENKEN_TASK=1 \
KENKEN_HYP_MASK=1 \
KENKEN_HYP_RELAX=1 \
KENKEN_HYP_ALPHA_MARGIN=1.5 \
KENKEN_HYP_LR=1e-4 \
KENKEN_HYP_WARMUP=200 \
KENKEN_HYP_GRAD_CLIP=1.0 \
KENKEN_HYP_JITTER=1e-3 \
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
RUN_NAME=kenken_hyp_relax_smoke \
RESUME_FROM="$CKPT" \
KENKEN_TRAIN=.cache/kenken_train_curriculum.jsonl \
KENKEN_TEST=.cache/kenken_test_curriculum.jsonl \
PYTHIA_INIT=1 \
SEED=42 \
.venv/bin/python scripts/kenken_train.py
