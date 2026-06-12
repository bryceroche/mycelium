#!/usr/bin/env bash
# #237 post-run sweep — order is BINDING (E.10→E.14):
#   1. Specialization diag over all dense ckpts (rho trajectory at 100-step
#      resolution through the freeze window; dim identities persisted —
#      REQUIRED before the ablation's carrier conditions can run).
#   2. Six-condition ablation on steps 1000 + 2000 (baseline, gate-0,
#      waist-off, carrier boundary/bus/both) with the four-cell read.
# The declaration is then composed in pinned order: rho trajectory
# (fifth-outcome rule + decay-front-vs-freeze-ordering) -> C4 branch ->
# ablation cells -> carrier four-cell with exposure tags -> wiring
# inspection (c) before (a)/(b) -> transfer split. Every cell read twice.
set -euo pipefail
cd "$(dirname "$0")/.."

# Refuse to fire while the training run is alive
if pgrep -f "v200_resmoke_237.py" > /dev/null; then
  echo "REFUSING: #237 training still running"; exit 1
fi
test -f .cache/v200_smoke/step2000_eval_237.json || {
  echo "REFUSING: step2000_eval_237.json missing — run not complete"; exit 1; }

# E.14 NaN-cascade contingency FIRED: step-2000 state poisoned (all-nan
# instrumentation at the final checkpoint; 173/173 skips from step 1828).
# Terminal analysis point = step 1750 (last clean dense ckpt); step-1500
# checkpoint is the last clean instrumented read. 2000 EXCLUDED from cells.
STEPS_ALL="0,100,200,300,400,500,600,700,800,900,1000,1250,1500,1750"

echo "=== [1/2] specialization sweep (rho + dims), ${STEPS_ALL} ==="
STEPS_LIST="$STEPS_ALL" .venv/bin/python scripts/diag_v237_specialization.py \
  2>&1 | tee .cache/v200_smoke/sweep_specialization_237.log

echo "=== [2/2] six-condition ablation, steps 1000,1750 ==="
STEPS_LIST=1000,1750 .venv/bin/python scripts/diag_v237_waist_ablation.py \
  2>&1 | tee .cache/v200_smoke/sweep_ablation_237.log

echo "=== sweep artifacts ==="
ls -la .cache/v200_smoke/specialization_237.json .cache/v200_smoke/waist_ablation_237.json
