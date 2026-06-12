#!/usr/bin/env bash
# v200 Stage 1C SPEC-RESTORE re-smoke (#234: prenorm4 + delta_gate init=-2.0).
#
# Two spec restorations bundled (both documented fixes, not new variables):
#   1. 4th RMSNorm (norm_breath) at breath boundary — fixes z oscillation 0.77→19160 (#233)
#   2. delta_gate init -2.0 (sigmoid→0.119) — Pythia cold-start finding (240× gradient speed)
#
# Runs:
#   1. Reference curve regeneration on spec-restored arch (§5/§6/§7)
#   2. 200-step training smoke at K=8, LR=3e-4 (spec defaults)
#   3. §1A.B C1-C6 verification + §1A.E.4 grid reading
#
# First line of log has ADVISORY: prefix if any spec deviation occurred.
# Final line is STAGE 1C SMOKE PASSED/FAILED per §11.
#
# Run:
#   cd /home/bryce/mycelium && bash scripts/v200_resmoke.sh
#
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-.venv/bin/python}"
SMOKE_DIR="${SMOKE_DIR:-.cache/v200_smoke}"
mkdir -p "$SMOKE_DIR" "$SMOKE_DIR/persistence" "$SMOKE_DIR/reference_curves"

echo "=== v200 Stage 1C Re-Smoke (substrate-fix) ==="
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
CKPT_LABEL="${CKPT_LABEL:-v200_perceiver_specrestore}" \
SEED=42 \
B_REF="${B_REF:-4}" \
SMOKE_DIR="$SMOKE_DIR" \
"$PYTHON" -u scripts/v200_resmoke.py 2>&1 | tee "$SMOKE_DIR/train_200_step_specrestore.log"

echo ""
echo "=== Spec-restore re-smoke complete ==="
echo "Final log line:"
tail -1 "$SMOKE_DIR/train_200_step_specrestore.log"
echo ""
echo "§1A.E.4 cell (from eval JSON):"
python3 -c "import json; d=json.load(open('$SMOKE_DIR/step200_eval_specrestore.json')); print('  ', d.get('e4_grid',{}).get('cell','?'))" 2>/dev/null || echo "  (eval JSON not yet written)"
