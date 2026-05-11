#!/usr/bin/env bash
# Phase 2+3 ablation screening — the five components Phase 1 didn't reach.
#
# Phase 2 (small code patches, env-var gated in breathing.py):
#   ablate_const_temp       — controller's temperature multiplier pinned to 1.0
#   ablate_const_step       — controller's step_mult pinned to 1.0
#
# Phase 3 (more involved patches, also env-var gated):
#   ablate_no_rotation      — RoPE per-head + per-loop phase offsets zeroed (no π cycling)
#   ablate_no_integration   — running integral reset every breath (last-breath-only)
#   ablate_no_notebook      — notebook cleared before every controller call (no x-breath memory)
#
# Same shape as Phase 1: 150 steps on ARITH_HARD, resume from l3_ctrl_v5_step375,
# single final eval at NUM_EVAL=50. Sequential runs so failures don't kill the
# sequence — continues to the next ablation if one crashes.
#
# Estimated wall: ~80 min (5 runs × ~16 min each).

set -uo pipefail   # NOT -e — we want to continue past single-ablation failures

cd "$(dirname "$0")/.."
CACHE=.cache
CKPT=$CACHE/l3_ckpts/l3_ctrl_v5_step375.safetensors

if [[ ! -f "$CKPT" ]]; then
  echo "missing warm-start checkpoint: $CKPT" >&2
  exit 1
fi

run_ablation() {
  local label=$1
  local ablate_var=$2  # name of the ABLATE_* env var to set to 1
  local log="$CACHE/${label}.log"

  echo
  echo "==================================================================="
  echo "  $label  ($ablate_var=1)"
  echo "==================================================================="
  date

  env \
    "$ablate_var=1" \
    DEV='PCI+AMD' \
    LEVEL=ARITH_HARD \
    BATCH=32 \
    STEPS=150 \
    TRAIN_LOOPS='1,2,4,8' \
    SPACE_DIGITS=1 \
    CTRL_TRAIN=1 \
    CTRL_LR=1e-4 \
    CTRL_MAX_LOOPS=2 \
    CTRL_TRAIN_EVERY=4 \
    LOOKUP_AUX_WEIGHT=0.1 \
    USE_JIT=1 \
    RESUME_FROM="$CKPT" \
    NUM_EVAL=50 \
    ACC_EVAL_EVERY=150 \
    LOSS_EVAL_EVERY=150 \
    CKPT_EVERY=150 \
    CKPT_LABEL="$label" \
    .venv/bin/python scripts/l3_train.py > "$log" 2>&1
  local rc=$?

  if (( rc == 0 )); then
    echo "  --- $label final eval ---"
    grep -E "acc @ A=|lookup-eval @" "$log" | tail -5
  else
    echo "  --- $label CRASHED (exit $rc) ---"
    tail -10 "$log" | sed 's/^/  /'
  fi
}

run_ablation ablate_const_temp     ABLATE_TEMP
run_ablation ablate_const_step     ABLATE_STEP_MULT
run_ablation ablate_no_rotation    ABLATE_ROTATION
run_ablation ablate_no_integration ABLATE_INTEGRATION
run_ablation ablate_no_notebook    ABLATE_NOTEBOOK

echo
echo "==================================================================="
echo "  Phase 2+3 Summary"
echo "==================================================================="
for label in ablate_const_temp ablate_const_step ablate_no_rotation ablate_no_integration ablate_no_notebook; do
  log="$CACHE/${label}.log"
  echo
  echo "$label:"
  if [[ -f "$log" ]]; then
    grep -E "acc @ A=|lookup-eval @" "$log" | tail -5 | sed 's/^/  /' || echo "  (no eval line found — likely crashed)"
  else
    echo "  (no log file)"
  fi
done
