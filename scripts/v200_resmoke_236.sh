#!/usr/bin/env bash
# v200 Stage 1C #236 smoke — §2 Seam 3 norm_blend (pre-blend RMSNorm).
#
# Change vs #235 (prenorm4 + alpha_read):
#   +norm_blend  6th RMSNorm, Seam 3 — pre-blend, applied after THINK (and after
#                COMMIT on even breaths), BEFORE the delta_gate convex blend.
#                Principle: bound the seams, not the organ. Pre-blend keeps gate
#                semantics interpretable (norm-after-blend launders mismatch).
#   arch_version suffix: prenorm5_seamthree_gate-2
#
# §1A.E.9 quantitative predictions (#236 recalibrated, 5 checkpoints):
#   post-norm_breath ~1.0  post-READ-add ~2.0
#   post-THINK ~4-50 (Llama natural attractor — NOT A CRITERION)
#   post-norm_blend ~1.0  post-blend ~1.0
#
# §7 new metric: concentration-drift (top-10/2048 dim energy fraction)
#   Reference: init-Llama=0.98, v235=0.258, random=0.048
#
# Runs:
#   1. Reference curve regeneration on corrected architecture (§5/§6/§7)
#   2. 200-step training smoke at K=8, LR=3e-4 (spec defaults)
#   3. §1A.B C1-C6 verification + §1A.E.4 grid reading
#   4. §1A.E.9 within-breath 5-checkpoint scale trajectory (recalibrated)
#   5. §10 alpha_read + norm_blend.weight strain detection
#   6. §7 concentration-drift metric at step 200
#
# First line of log has ADVISORY: prefix if any spec deviation occurred.
# Final line is STAGE 1C SMOKE PASSED/FAILED per §11.
#
# Run:
#   cd /home/bryce/mycelium && bash scripts/v200_resmoke_236.sh
#
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-.venv/bin/python}"
SMOKE_DIR="${SMOKE_DIR:-.cache/v200_smoke}"
mkdir -p "$SMOKE_DIR" "$SMOKE_DIR/persistence" "$SMOKE_DIR/reference_curves"

echo "=== v200 Stage 1C #236 Smoke (§2 Seam 3: norm_blend pre-blend RMSNorm) ==="
echo "  SMOKE_DIR: $SMOKE_DIR"
echo "  PYTHON: $PYTHON"
echo "  Fix: norm_blend RMSNorm at Seam 3 (between THINK/COMMIT and delta_gate blend)"
echo "  Placement: PRE-blend (not post-blend) — bounds seam, preserves gate semantics"
echo "  Principle: bound the seams, not the organ (Llama L0-L3 breathes large)"
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
CKPT_LABEL="${CKPT_LABEL:-v200_perceiver_236_seamthree}" \
SEED=42 \
B_REF="${B_REF:-4}" \
SMOKE_DIR="$SMOKE_DIR" \
"$PYTHON" -u scripts/v200_resmoke_236.py 2>&1 | tee "$SMOKE_DIR/train_200_step_236.log"

echo ""
echo "=== #236 smoke complete ==="
echo "Final log line:"
tail -1 "$SMOKE_DIR/train_200_step_236.log"
echo ""
echo "§1A.E.4 cell:"
python3 -c "import json; d=json.load(open('$SMOKE_DIR/step200_eval_236.json')); print('  ', d.get('e4_grid',{}).get('cell','?'))" 2>/dev/null || echo "  (eval JSON not yet written)"
echo ""
echo "§1A.E.9 trajectory match (5 checkpoints, post_THINK not a criterion):"
python3 -c "import json; d=json.load(open('$SMOKE_DIR/step200_eval_236.json')); print('  match=', d.get('trajectory_match'), '  avg=', d.get('measured_trajectory_average_breaths'))" 2>/dev/null || echo "  (eval JSON not yet written)"
echo ""
echo "alpha_read + norm_blend strain:"
python3 -c "import json; d=json.load(open('$SMOKE_DIR/step200_eval_236.json')); print('  alpha_read=', d.get('alpha_read_at_step200'), '  strain=', d.get('alpha_read_strain_signal'), '  blend_strain=', d.get('norm_blend_strain_signal'))" 2>/dev/null || echo "  (eval JSON not yet written)"
echo ""
echo "§7 concentration-drift at step 200:"
python3 -c "import json; d=json.load(open('$SMOKE_DIR/step200_eval_236.json')); c=d.get('concentration_drift',{}); print('  post_think=', c.get('post_think_top10_frac'), '  post_blend=', c.get('post_blend_top10_frac'))" 2>/dev/null || echo "  (eval JSON not yet written)"
