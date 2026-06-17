#!/usr/bin/env bash
# Perceiver-Poincaré BRICK-2 smoke — the DEDUCTION-TEST short re-smoke.
#
# Brick-1 BREATHED (engagement genuine, wall cured) but with FROZEN routing only
# reached off-chance (0.187 vs 0.143). Brick-2 UNFREEZES g_phi so the engine can
# learn to DEDUCE + fixes the two brick-1 caveats. This SHORT smoke (~20-30 steps,
# g_phi UNFROZEN, per_constraint) confirms before a human fires the ~600-1000 step
# deduction run:
#   (a) compiles + trains (no NaN, grads finite INCL the now-trainable g_phi),
#   (b) the RECALIBRATED kill metric reads correctly (select_norm vs the per-field
#       uniform floor 1/sqrt(S) + the ALIVE/DEAD decision; should read ALIVE),
#   (c) the trajectory JSONL is written + parseable (run_dir/trajectory.jsonl),
#   (d) g_phi actually MOVES (drift > 0, unfrozen) + cell_acc not collapsing.
#
# CEILINGS (do NOT regress): K=8 (K>=12 HUNG the AM driver), fp32 THINK (fp16
# overflowed late-breath), the brick-1 graph trims (engagement on first+last
# breath only, deferred logging). Throwaway RUN_NAME. PER-CONSTRAINT path.
set -euo pipefail
cd "$(dirname "$0")/.."

PERCEIVER_TASK=1 \
PERCEIVER_K_MAX="${PERCEIVER_K_MAX:-8}" \
PERCEIVER_BALL_PATH="${PERCEIVER_BALL_PATH:-per_constraint}" \
PERCEIVER_TAU="${PERCEIVER_TAU:-0.5}" \
PERCEIVER_RHO="${PERCEIVER_RHO:-0.7}" \
PERCEIVER_DIM="${PERCEIVER_DIM:-48}" \
PERCEIVER_N_GLOBAL="${PERCEIVER_N_GLOBAL:-4}" \
BATCH="${BATCH:-8}" \
STEPS="${STEPS:-24}" \
LR="${LR:-3e-5}" \
GRAD_CLIP="${GRAD_CLIP:-1.0}" \
MAX_ZNORM="${MAX_ZNORM:-0.9}" \
LOG_EVERY="${LOG_EVERY:-4}" \
EVAL_EVERY="${EVAL_EVERY:-0}" \
RUN_NAME="${RUN_NAME:-perceiver_brick2_smoke}" \
KENKEN_TRAIN="${KENKEN_TRAIN:-.cache/kenken_train_curriculum.jsonl}" \
KENKEN_TEST="${KENKEN_TEST:-.cache/kenken_test_curriculum.jsonl}" \
SEED="${SEED:-42}" \
.venv/bin/python scripts/perceiver_train.py
