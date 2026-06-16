#!/usr/bin/env bash
# STAGE-1 RELAXATION empirical-null run (spec §8.2/§8.6 Stage 1).
#
# FREEZE the v98 backbone, unfreeze + train ONLY the existing hyperbolic coords
# (kenken_hyp_v_row, kenken_hyp_v_col, kenken_hyp_cage_anchors). ~600 steps, per-band
# held-out eval every 100. EXPECTED per §8.6: row/col relax (little to learn), cage
# slot-anchors fail to generalize (in-dist maybe up, held-out flat/down) — the honest
# first data point that the slot-anchor parameterization can't structurally interpolate
# (gates whether to build the Stage-2 structure-based cage encoder).
#
# Guards LIVE (first backward through d_hyp): coord-grad clip + bounded tangent norms +
# where()-gated NaN guard. alpha recalibrated for gradient flow (RELAX_BLOCK_ARG=20 ->
# faithful mask, leak exp(-20)~2e-9, responsive softplus). Watch the [coord-grad] line.
set -euo pipefail
cd "$(dirname "$0")/.."

CKPT=".cache/kenken_ckpts/kenken_curric_k16_cont/kenken_curric_k16_cont_final.safetensors"

KENKEN_TASK=1 \
KENKEN_HYP_MASK=1 \
KENKEN_HYP_RELAX=1 \
KENKEN_HYP_ALPHA_MARGIN=1.5 \
KENKEN_HYP_RELAX_BLOCK_ARG=20.0 \
KENKEN_HYP_LR=1e-4 \
KENKEN_HYP_WARMUP=200 \
KENKEN_HYP_GRAD_CLIP=1.0 \
KENKEN_HYP_JITTER=1e-3 \
KENKEN_HYP_MAX_ZNORM=0.9 \
KENKEN_K_MAX=16 \
BATCH=8 \
STEPS=600 \
LR=3e-5 \
CKPT_EVERY=200 \
EVAL_EVERY=100 \
EVAL_BATCHES=20 \
EVAL_BATCH=8 \
LOG_EVERY=10 \
PER_BREATH_CE_EVERY=50 \
GC_EVERY=50 \
RUN_NAME=kenken_hyp_relax_stage1 \
RESUME_FROM="$CKPT" \
KENKEN_TRAIN=.cache/kenken_train_curriculum.jsonl \
KENKEN_TEST=.cache/kenken_test_curriculum.jsonl \
PYTHIA_INIT=1 \
SEED=42 \
.venv/bin/python scripts/kenken_train.py
