#!/bin/bash
# v96.2 eval-all-ckpts — once training finishes, evaluate all v96_2_smoke ckpts.
#
# Used as: bash scripts/v96_2_eval_all_ckpts.sh > .cache/v96_2_eval_all.log 2>&1
#
# Waits for the training process to exit, then sequentially evals each ckpt.

set -e
cd "$(dirname "$0")/.."

# Wait for training process to exit (max 4 hours).
echo "[eval_all] waiting for l3_train.py to exit..."
deadline=$((SECONDS + 14400))
while pgrep -f "l3_train.py" > /dev/null; do
    if [ $SECONDS -gt $deadline ]; then
        echo "[eval_all] deadline reached without training exit; bailing"
        exit 1
    fi
    sleep 30
done
echo "[eval_all] training exited. Evaluating ckpts..."
sleep 5  # let the lock file release

CKPT_DIR="${CKPT_DIR:-.cache/gsm8k_steps_ckpts}"

for step in 100 200 300 400 500; do
    ckpt="$CKPT_DIR/v96_2_smoke_step${step}.safetensors"
    if [ -f "$ckpt" ]; then
        echo ""
        echo "======================================"
        echo "[eval_all] EVAL step ${step}: $ckpt"
        echo "======================================"
        NUM_EVAL=60 BATCH=4 MAX_NEW=120 bash scripts/v96_2_eval.sh "$ckpt" 2>&1 | tail -50 || true
    else
        echo "[eval_all] missing ckpt: $ckpt (skipping)"
    fi
done
echo ""
echo "[eval_all] DONE."
