#!/usr/bin/env bash
# v200 Stage 1C smoke wrapper — perceiver-CORE training driver.
#
# Runs scripts/v200_perceiver_train.py for 200 steps with the §1A.B
# training contract verification. Tees output to SMOKE_DIR log.
#
# Artifacts written:
#   .cache/v200_smoke/train_200_step.log      — full log with PASSED/FAILED
#   .cache/v200_smoke/step200_eval.json       — eval metrics + cont-control
#   .cache/v200_smoke/grad_norms.npz          — per-group grad norms
#   .cache/v200_smoke/persistence/step200_z.npz
#   .cache/v200_smoke/step200_provenance.json
#
# Run:
#   cd /home/bryce/mycelium && bash scripts/v200_smoke.sh
#
# Override BATCH, STEPS, or SMOKE_DIR with env vars.
#
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-.venv/bin/python}"
SMOKE_DIR="${SMOKE_DIR:-.cache/v200_smoke}"
mkdir -p "$SMOKE_DIR" "$SMOKE_DIR/persistence"

echo "=== v200 Stage 1C Smoke (perceiver-CORE) ==="
echo "  SMOKE_DIR: $SMOKE_DIR"
echo "  PYTHON: $PYTHON"
echo ""

V200_TASK=1 \
V200_STAGE2A_WAIST=1 \
V200_K_MAX="${V200_K_MAX:-8}" \
V200_N_LATENTS=32 \
V200_N_VAR_LAT=16 \
V200_N_DIGITS=5 \
V200_N_MAX=16 \
V200_F_MAX=8 \
V200_CALIB_WEIGHT=0.05 \
V200_TRAIN="${V200_TRAIN:-.cache/factor_graph_train.jsonl}" \
V200_VAL="${V200_VAL:-.cache/factor_graph_test.jsonl}" \
V200_GSM8K="${V200_GSM8K:-.cache/gsm8k_factor_graphs_train.jsonl}" \
V200_GSM8K_RATIO=0.5 \
BATCH="${BATCH:-8}" \
STEPS="${STEPS:-200}" \
LR="${LR:-3e-4}" \
LOG_EVERY=10 \
PER_BREATH_EVERY=50 \
EVAL_BATCHES="${EVAL_BATCHES:-8}" \
GRAD_NORM_EVERY="${GRAD_NORM_EVERY:-100}" \
CKPT_EVERY=200 \
CKPT_LABEL="${CKPT_LABEL:-v200_perceiver_smoke}" \
SEED=42 \
SMOKE_DIR="$SMOKE_DIR" \
"$PYTHON" -u scripts/v200_perceiver_train.py 2>&1 | tee "$SMOKE_DIR/train_200_step.log"

echo ""
echo "=== Smoke run complete ==="
echo "Final log line:"
tail -1 "$SMOKE_DIR/train_200_step.log"
