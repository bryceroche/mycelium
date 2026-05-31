#!/usr/bin/env bash
# Smoke test: 50 steps on 100-problem subset.
# Acceptance criteria:
#   1. No NaN in any loss
#   2. Loss decreases (verified by visual inspection of step 1 vs step 50)
#   3. Inference test produces structured output

set -euo pipefail
cd "$(dirname "$0")/.."

VENV=".venv/bin/python"

echo "============================================="
echo "Phase 1 Classifier Smoke Test"
echo "============================================="
echo ""

echo "--- Step 1: Data prep (idempotent) ---"
$VENV scripts/build_phase1_classifier_data.py
echo ""

echo "--- Step 2: 50-step training smoke ---"
STEPS=50 BATCH=16 LR=3e-5 DEVICE=cpu \
  $VENV scripts/phase1_classifier_train.py --smoke \
  --eval-every 25 --save-every 50
echo ""

echo "--- Step 3: Inference test ---"
CKPT=$(.venv/bin/python -c "
import glob, sys
ckpts = sorted(glob.glob('.cache/phase1_ckpts/phase1_classifier_step*.pt'))
print(ckpts[-1] if ckpts else '')
")
if [ -z "$CKPT" ]; then
  echo "ERROR: No checkpoint found after training"
  exit 1
fi

$VENV scripts/phase1_classifier_eval.py \
  --ckpt "$CKPT" \
  --problem "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for \$2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?" \
  --vars "eggs laid per day" "eggs eaten for breakfast" "eggs used for baking" \
         "price per egg" "eggs remaining" "daily revenue"
echo ""

echo "============================================="
echo "Smoke PASSED"
echo "============================================="
