#!/usr/bin/env bash
# Phase 1 ablation screening: which of the 7 closed-loop components is load-bearing?
#
# Three short ARITH_HARD runs (150 steps each, ~10× smaller than the v1 baseline)
# vary one component at a time via existing env-var flags — no code changes
# required. Resumes from the same L3 ctrl_v5_step375 starting point as the
# ARITH_HARD v1 baseline so the only difference between runs is the ablated
# component.
#
# Runs:
#   ablate_baseline   — all 7 components on (matches arith_hard_v1 config)
#   ablate_no_ctrl    — controller stays at random init (CTRL_TRAIN=0)
#   ablate_no_lookup  — no lookup-table aux loss (LOOKUP_AUX_WEIGHT=0)
#
# Each run prints its final accuracy at A=1/2/4/8 and lookup-eval. A summary
# table prints at the end. Compare against arith_hard_v1's step 150 numbers
# (the inflection point in the full 1500-step run was step 1000, so 150-step
# results are early-trajectory comparisons, not converged-state).
#
# Total wall time: ~50 min (3 runs × ~16 min each).

set -euo pipefail

cd "$(dirname "$0")/.."
CACHE=.cache
CKPT=$CACHE/l3_ckpts/l3_ctrl_v5_step375.safetensors

if [[ ! -f "$CKPT" ]]; then
  echo "missing warm-start checkpoint: $CKPT" >&2
  exit 1
fi

run_ablation() {
  local label=$1
  local ctrl_train=$2
  local lookup_aux_weight=$3
  local log="$CACHE/${label}.log"

  echo
  echo "==================================================================="
  echo "  $label  (CTRL_TRAIN=$ctrl_train  LOOKUP_AUX_WEIGHT=$lookup_aux_weight)"
  echo "==================================================================="
  date

  DEV='PCI+AMD' \
    LEVEL=ARITH_HARD \
    BATCH=32 \
    STEPS=150 \
    TRAIN_LOOPS='1,2,4,8' \
    SPACE_DIGITS=1 \
    CTRL_TRAIN="$ctrl_train" \
    CTRL_LR=1e-4 \
    CTRL_MAX_LOOPS=2 \
    CTRL_TRAIN_EVERY=4 \
    LOOKUP_AUX_WEIGHT="$lookup_aux_weight" \
    USE_JIT=1 \
    RESUME_FROM="$CKPT" \
    NUM_EVAL=50 \
    ACC_EVAL_EVERY=150 \
    LOSS_EVAL_EVERY=150 \
    CKPT_EVERY=150 \
    CKPT_LABEL="$label" \
    .venv/bin/python scripts/l3_train.py > "$log" 2>&1

  echo "  --- $label final eval ---"
  grep -E "acc @ A=|lookup-eval @" "$log" | tail -5
}

run_ablation ablate_baseline   1 0.1
run_ablation ablate_no_ctrl    0 0.1
run_ablation ablate_no_lookup  1 0.0

echo
echo "==================================================================="
echo "  Summary"
echo "==================================================================="
for label in ablate_baseline ablate_no_ctrl ablate_no_lookup; do
  log="$CACHE/${label}.log"
  echo
  echo "$label:"
  grep -E "acc @ A=|lookup-eval @" "$log" | tail -5 | sed 's/^/  /'
done
