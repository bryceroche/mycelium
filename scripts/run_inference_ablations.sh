#!/usr/bin/env bash
# Inference-time ablations on the converged ARITH_HARD step-1500 checkpoint.
#
# The question shifts from "does training with X help?" (Phase 1-3 ablations)
# to "did the trained model learn to USE X?" — a sharper test. Each run reuses
# the same converged ckpt but evaluates with one closed-loop component
# disabled. Compares against the unablated baseline eval at A=8.
#
# Six runs at N=100, A=8:
#   inf_baseline       — all 7 components on (reference)
#   inf_const_temp     — controller's temperature pinned to 1.0
#   inf_const_step     — controller's step_mult pinned to 1.0
#   inf_const_gate     — controller's integration gate pinned to 1.0 (uniform breath weighting)
#   inf_no_integration — last-breath-only (no running integral)
#   inf_no_notebook    — notebook cleared every breath (no cross-breath memory)
#   inf_no_rotation    — uniform RoPE (no per-head / per-loop phase offset; sanity)
#   inf_all_const      — temp + step + gate all pinned (combined controller-decisions ablation)
#
# Each eval is ~1-2 min on the batched path. Total ~15-20 min.

set -uo pipefail
cd "$(dirname "$0")/.."

CKPT=.cache/arith_hard_ckpts/arith_hard_v1_step1500.safetensors
if [[ ! -f "$CKPT" ]]; then
  echo "missing checkpoint: $CKPT" >&2
  exit 1
fi

run_inf() {
  local label=$1
  shift
  local log=".cache/${label}.log"

  echo
  echo "==================================================================="
  echo "  $label  ($*)"
  echo "==================================================================="
  date

  env "$@" \
    DEV='PCI+AMD' \
    CKPT="$CKPT" \
    LEVEL=ARITH_HARD \
    LOOPS=8 \
    NUM_EVAL=100 \
    SPACE_DIGITS=1 \
    .venv/bin/python scripts/eval_l4.py > "$log" 2>&1
  local rc=$?

  if (( rc == 0 )); then
    echo "  --- $label final ---"
    grep -E "acc @ A=|\[ABLATE\] active" "$log" | head -5
  else
    echo "  --- $label CRASHED (exit $rc) ---"
    tail -10 "$log" | sed 's/^/  /'
  fi
}

# Unablated baseline (no flags)
run_inf inf_baseline
run_inf inf_const_temp     ABLATE_TEMP=1
run_inf inf_const_step     ABLATE_STEP_MULT=1
run_inf inf_const_gate     ABLATE_GATE=1
run_inf inf_no_integration ABLATE_INTEGRATION=1
run_inf inf_no_notebook    ABLATE_NOTEBOOK=1
run_inf inf_no_rotation    ABLATE_ROTATION=1
run_inf inf_all_const      ABLATE_TEMP=1 ABLATE_STEP_MULT=1 ABLATE_GATE=1

echo
echo "==================================================================="
echo "  Inference-Time Ablation Summary (A=8, N=100, ARITH_HARD step-1500 ckpt)"
echo "==================================================================="
for label in inf_baseline inf_const_temp inf_const_step inf_const_gate \
             inf_no_integration inf_no_notebook inf_no_rotation inf_all_const; do
  log=".cache/${label}.log"
  if [[ -f "$log" ]]; then
    acc=$(grep -E "acc @ A=8" "$log" | tail -1 | sed -E 's/.*: ([0-9.]+)%.*/\1/')
    printf '  %-22s %s%%\n' "$label" "${acc:-—}"
  else
    printf '  %-22s (no log)\n' "$label"
  fi
done
