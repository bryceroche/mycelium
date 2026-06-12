#!/usr/bin/env bash
# #238 post-run sweep — order BINDING (specialization before ablation;
# carrier conditions read specialization_238.json). Then slot probe with
# vector persistence, torsion refresh (incl. 2000 bundle), and the E.16
# depth split on BOTH finals (the control's and #238's).
set -euo pipefail
cd "$(dirname "$0")/.."
if pgrep -f "v200_resmoke_238.py" > /dev/null; then
  echo "REFUSING: #238 training still running"; exit 1
fi
test -f .cache/v200_smoke/step2000_eval_238.json || {
  echo "REFUSING: step2000_eval_238.json missing"; exit 1; }

STEPS_ALL="0,100,200,300,400,500,600,700,800,900,1000,1250,1500,1750,2000"

echo "=== [1/5] specialization sweep (rho + dims), RUN_TAG=238 ==="
RUN_TAG=238 STEPS_LIST="$STEPS_ALL" .venv/bin/python scripts/diag_v237_specialization.py \
  2>&1 | tee .cache/v200_smoke/sweep_specialization_238.log

echo "=== [2/5] six-condition ablation (cross-read armed), steps 1000,2000 ==="
RUN_TAG=238 STEPS_LIST=1000,2000 .venv/bin/python scripts/diag_v237_waist_ablation.py \
  2>&1 | tee .cache/v200_smoke/sweep_ablation_238.log

echo "=== [3/5] slot probe with vector persistence, steps 1750,2000 ==="
STEPS_LIST=1750,2000 .venv/bin/python scripts/diag_v238_slots.py \
  2>&1 | tee .cache/v200_smoke/sweep_slots_238.log

echo "=== [4/5] torsion refresh (incl. 2000 bundles) ==="
.venv/bin/python scripts/diag_torsion.py 2>&1 | tee .cache/v200_smoke/sweep_torsion.log

echo "=== [5/5] E.16 depth split — control final, then #238 final ==="
RUN_CKPT=.cache/v200_perceiver_ckpts/v200_perceiver_237_5_substrate_step2000.safetensors \
  .venv/bin/python scripts/diag_e16_depth_split.py 2>&1 | tee .cache/v200_smoke/sweep_e16_2375.log
mv .cache/v200_smoke/e16_depth_split.json .cache/v200_smoke/e16_depth_split_237_5.json
RUN_CKPT=.cache/v200_perceiver_ckpts/v200_perceiver_238_write8_step2000.safetensors \
  .venv/bin/python scripts/diag_e16_depth_split.py 2>&1 | tee .cache/v200_smoke/sweep_e16_238.log
mv .cache/v200_smoke/e16_depth_split.json .cache/v200_smoke/e16_depth_split_238.json

echo "=== sweep artifacts ==="
ls -la .cache/v200_smoke/specialization_238.json .cache/v200_smoke/waist_ablation_238.json \
  .cache/v200_smoke/slot_distinctness_238.json .cache/v200_smoke/torsion_diag.json \
  .cache/v200_smoke/e16_depth_split_237_5.json .cache/v200_smoke/e16_depth_split_238.json
