#!/usr/bin/env bash
# STAGE-2 RELAXATION smoke — EUCLIDEAN CONTROL ARM (spec §8.1 capacity-matched control).
#
# The v112b attribution control: the SAME g_phi encoder (identical param count/shapes),
# but Euclidean ||u-v|| instead of d_hyp (drop exp_0; the coord IS the point). r/alpha are
# recalibrated to the EUCLIDEAN anchor distances so this arm ALSO reproduces the hard mask
# at t=0 (same zero-init discipline). Identical DOF -> any hyperbolic>Euclidean gap in the
# full run is the GEOMETRY, not capacity.
#
# ~15 steps. Same zero-init t=0 check + compiles + finite g_phi grads as the hyperbolic arm.
# Throwaway RUN_NAME. alpha_margin=1.5 + RELAX_BLOCK_ARG=6 (soft mask -> two-sided gradient).
set -euo pipefail
cd "$(dirname "$0")/.."

CKPT=".cache/kenken_ckpts/kenken_curric_k16_cont/kenken_curric_k16_cont_final.safetensors"

KENKEN_TASK=1 \
KENKEN_HYP_MASK=1 \
KENKEN_HYP_RELAX=1 \
KENKEN_HYP_GPHI=1 \
KENKEN_HYP_EUCLID=1 \
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
RUN_NAME=kenken_hyp_stage2_euclid_smoke \
RESUME_FROM="$CKPT" \
KENKEN_TRAIN=.cache/kenken_train_curriculum.jsonl \
KENKEN_TEST=.cache/kenken_test_curriculum.jsonl \
PYTHIA_INIT=1 \
SEED=42 \
.venv/bin/python scripts/kenken_train.py
