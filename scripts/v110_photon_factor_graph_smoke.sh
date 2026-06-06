#!/usr/bin/env bash
# v110-photon smoke — v109pi + smooth cos² waist gate (B-field) phase-locked to Q sin rotation (E-field).
#
# Architecture (warm-start from v109pi_cont9_step500):
#   v109pi base (waist + alternation + per-breath Q rotation)
#   + waist gating: photon_alpha * smooth_profile + (1-photon_alpha) * binary
#
# At photon_alpha=0: byte-identical to v109pi.
# At photon_alpha=1: fully smooth profile.
# Default photon_alpha=0.5: half-and-half warm-start; lets gradient nudge it.
#
# Profiles (V110_PHOTON_GATE_PROFILE):
#   alt_smooth — same peaks as binary at integer breaths but cosine-smoothed
#   cos2_pi2   — monotonic 1→0 across K breaths (commit then expand)
#   cos2_pi    — peaks at k=0 and k=K-1, dips at K/2
#   sin2_pi    — peaks at K/2, zero at boundaries (the literal "B-field")
#   binary     — control, same as v109pi
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-.venv/bin/python}"
CKPT="${CKPT:-.cache/fg_v109pi_ckpts/v109pi_cont9_step500.safetensors}"

DEV='PCI+AMD' \
V110_PHOTON_TASK=1 \
V110_PHOTON_K_MAX=8 \
V110_PHOTON_N_DIGITS=5 \
V110_PHOTON_N_MAX=16 \
V110_PHOTON_F_MAX=8 \
V110_PHOTON_WAIST_DIM=512 \
V110_PHOTON_ALTERNATION="${V110_PHOTON_ALTERNATION:-1}" \
V110_PHOTON_PHASE_SCALE="${V110_PHOTON_PHASE_SCALE:-1.0}" \
V110_PHOTON_GATE_PROFILE="${V110_PHOTON_GATE_PROFILE:-sin2_pi}" \
V110_PHOTON_ALPHA="${V110_PHOTON_ALPHA:-0.5}" \
V110_PHOTON_HARD_BREATH_LEVEL=0 \
V110_PHOTON_VAR_LOSS_WEIGHT=1.0 \
V110_PHOTON_CALIB_WEIGHT=0.05 \
V110_PHOTON_FACTOR_AUX_WEIGHT=0.5 \
V110_PHOTON_CODEBOOK_N=32 \
V110_PHOTON_IB_CENTROIDS=.cache/ib_centroids_gsm8k_partial.npz \
V110_PHOTON_TRAIN=.cache/factor_graph_train.jsonl \
V110_PHOTON_VAL=.cache/factor_graph_test.jsonl \
V110_PHOTON_GSM8K_TRAIN=.cache/gsm8k_factor_graphs_train.jsonl \
V110_PHOTON_GSM8K_RATIO=0.5 \
BATCH=8 \
STEPS="${STEPS:-500}" \
LR=3e-5 \
LOG_EVERY=10 \
PER_BREATH_CE_EVERY=50 \
EVAL_EVERY=100 \
EVAL_BATCHES=30 \
EVAL_BATCH=8 \
CKPT_EVERY=250 \
CKPT_LABEL="${CKPT_LABEL:-v110_photon_smoke}" \
RESUME_FROM="$CKPT" \
PYTHIA_INIT=1 \
"$PYTHON" -u scripts/v110_photon_factor_graph_train.py
