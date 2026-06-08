#!/usr/bin/env bash
# v115 smoke — monotonic waist envelope (basin landing test).
#
# Architecture: v112b model (unchanged) + GATE_PROFILE=cos2_pi2 instead of
# sin2_pi. cos2_pi2 is monotonically decreasing 1.0 → 0.0 across K=8 breaths.
# Combined with v109 alternation (waist on even breaths only), the active
# (even) breaths get waist amplitude: k=0→1.0, k=2→0.81, k=4→0.41, k=6→0.07.
# HEAVIEST compression at breath 0 (basin landing), lightest at breath 6.
#
# Inverts the sin² photon profile (light-heavy-light) to test the basin
# landing principle: coarse representation at breath 0 forces commitment
# to the right basin in a low-state landscape; progressive widening allows
# refinement within the chosen basin.
#
# Warm-start from v112b_cont1_final (NEW PROJECT HIGH at 0.3945 hard).
# 300 steps to see if monotonic waist lifts hard cell_acc.
#
# Mirror is DISABLED (V114_MIRROR_AT_K=0) — this isolates the waist envelope
# question from the mirror question. Reuses v114 train infrastructure since
# WARM_FROM and the staging-mask gate are already wired there.
#
# Predicted signature (from eg prior data — easy↑, hard↓):
#   easy:   should lift (basin obvious, early compression wins)
#   medium: should lift modestly
#   hard:   may HURT (exploration before commitment matters on hard)
#
# If hard lifts: basin landing principle validated → next test is v113+monotonic
# If hard hurts: exploration before commit is real → sin² is right for hard
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-.venv/bin/python}"

V114_MIRROR_AT_K=0 \
V110_STEP3_TASK=1 \
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
V110_STEP3_GATE_PROFILE=cos2_pi2 \
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
CKPT_LABEL="${CKPT_LABEL:-v115_monotonic}" \
PYTHIA_INIT=1 \
DEV='PCI+AMD' \
"$PYTHON" -u scripts/v114_train.py
