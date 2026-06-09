#!/usr/bin/env bash
# v200 Stage 2a smoke: alternating waist on even breaths (COMMIT phase).
#
# Stage 2a adds:
#   fg_v200_W_compress  (2048, 512)  QR-init ×0.01
#   fg_v200_W_expand    (512, 2048)  ZERO-INIT (bootstrap safe)
#   fg_v200_waist_gate  scalar       zero-init, (1+g) amplifier
#   Total new params: ~2.1M
#
# Step 0 byte-identity: W_expand=0 → delta=0 → latents unchanged.
# Gradient at step 1: W_compress non-zero → z non-zero → ∂L/∂W_expand non-zero.
#
# PASS criteria (200 steps):
#   - W_expand_norm > 0 at step 10 (gradient flows)
#   - per_breath_ce ladder delta > 0.05 by step 200
#   - drift pattern: even breaths > odd breaths (asymmetry confirmed)
#   - No NaN, no crash
#
# INVESTIGATE criteria:
#   - W_expand_norm stays 0.0000 (bootstrap failed — gradient not flowing)
#   - Drift uniform across even/odd (waist not creating rhythm)
#   - waist_gate trends strongly negative (model dampening, v119 pattern)
#
# Run:
#   cd /home/bryce/mycelium && bash scripts/v200_smoke_2a.sh
#
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-.venv/bin/python}"

V200_TASK=1 \
V200_STAGE2A_WAIST="${V200_STAGE2A_WAIST:-1}" \
V200_WAIST_DIM="${V200_WAIST_DIM:-512}" \
V200_K_MAX="${V200_K_MAX:-8}" \
V200_N_LATENTS=32 \
V200_N_VAR_LAT=16 \
V200_N_DIGITS=5 \
V200_N_MAX=16 \
V200_F_MAX=8 \
V200_CALIB_WEIGHT=0.05 \
V200_TRAIN=.cache/factor_graph_train.jsonl \
V200_VAL=.cache/factor_graph_test.jsonl \
V200_GSM8K_TRAIN=.cache/gsm8k_factor_graphs_train.jsonl \
V200_GSM8K_RATIO=0.5 \
BATCH="${BATCH:-12}" \
STEPS="${STEPS:-1000}" \
LR=1e-4 \
LOG_EVERY=10 \
PER_BREATH_CE_EVERY=50 \
EVAL_EVERY=250 \
EVAL_BATCHES=10 \
EVAL_BATCH="${BATCH:-12}" \
CKPT_EVERY=250 \
CKPT_LABEL="${CKPT_LABEL:-v200_smoke_2a}" \
SEED=42 \
DEV="${DEV:-PCI+AMD}" \
"$PYTHON" -u scripts/v200_train.py 2>&1 | tee .cache/v200_smoke_2a_train.log

echo ""
echo "=== Stage 2a smoke complete. Check .cache/v200_smoke_2a_train.log ==="
echo ""
echo "Key signals to check:"
echo "  W_expand_norm > 0 at step 10     (gradient flow OK)"
echo "  LADDER delta > 0.05 at step 200  (breathing rhythm)"
echo "  DRIFT even > odd                 (waist asymmetry confirmed)"
