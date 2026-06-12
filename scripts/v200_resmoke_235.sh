#!/usr/bin/env bash
# v200 Stage 1C #235 smoke — §1A.E.4 READ-dominance fix.
#
# Change vs #234 (prenorm4 + delta_gate=-2.0):
#   +norm_read_ctx  RMSNorm on read_ctx before residual add (ones-init gain)
#   +alpha_read     learnable scalar α init=1.0 (NOT zero-init per §1A.E.8)
#   arch_version suffix: prenorm4_gate-2_readnorm
#
# §1A.E.9 quantitative predictions (per-element scale, 4 checkpoints):
#   post-norm_breath ~1.0  post-READ-add ~2.0  post-THINK ~3.5  post-blend ~1.6
#
# Runs:
#   1. Reference curve regeneration on corrected architecture (§5/§6/§7)
#   2. 200-step training smoke at K=8, LR=3e-4 (spec defaults)
#   3. §1A.B C1-C6 verification + §1A.E.4 grid reading
#   4. §1A.E.9 within-breath 4-checkpoint scale trajectory
#   5. §10 row 5 alpha_read strain detection
#
# First line of log has ADVISORY: prefix if any spec deviation occurred.
# Final line is STAGE 1C SMOKE PASSED/FAILED per §11.
#
# Run:
#   cd /home/bryce/mycelium && bash scripts/v200_resmoke_235.sh
#
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-.venv/bin/python}"
SMOKE_DIR="${SMOKE_DIR:-.cache/v200_smoke}"
mkdir -p "$SMOKE_DIR" "$SMOKE_DIR/persistence" "$SMOKE_DIR/reference_curves"

echo "=== v200 Stage 1C #235 Smoke (§1A.E.4 READ-dominance fix) ==="
echo "  SMOKE_DIR: $SMOKE_DIR"
echo "  PYTHON: $PYTHON"
echo "  Fix: α·RMSNorm(read_ctx) at READ residual-add site"
echo "  α init=1.0 (NOT zero-init — READ is information inlet per §1A.E.8)"
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
CKPT_LABEL="${CKPT_LABEL:-v200_perceiver_235_readnorm}" \
SEED=42 \
B_REF="${B_REF:-4}" \
SMOKE_DIR="$SMOKE_DIR" \
"$PYTHON" -u scripts/v200_resmoke_235.py 2>&1 | tee "$SMOKE_DIR/train_200_step_235.log"

echo ""
echo "=== #235 smoke complete ==="
echo "Final log line:"
tail -1 "$SMOKE_DIR/train_200_step_235.log"
echo ""
echo "§1A.E.4 cell:"
python3 -c "import json; d=json.load(open('$SMOKE_DIR/step200_eval_235.json')); print('  ', d.get('e4_grid',{}).get('cell','?'))" 2>/dev/null || echo "  (eval JSON not yet written)"
echo ""
echo "§1A.E.9 trajectory match:"
python3 -c "import json; d=json.load(open('$SMOKE_DIR/step200_eval_235.json')); print('  match=', d.get('trajectory_match'), '  avg=', d.get('measured_trajectory_average_breaths'))" 2>/dev/null || echo "  (eval JSON not yet written)"
echo ""
echo "alpha_read strain:"
python3 -c "import json; d=json.load(open('$SMOKE_DIR/step200_eval_235.json')); print('  alpha_read=', d.get('alpha_read_at_step200'), '  strain=', d.get('alpha_read_strain_signal'))" 2>/dev/null || echo "  (eval JSON not yet written)"
