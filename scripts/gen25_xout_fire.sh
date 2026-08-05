#!/bin/bash
# gen25_xout_fire.sh — THE REVERSE GEAR'S TRAINING FIRE (2026-08-05; word
# given; registration + pinned bars in the ledger BEFORE this ran). Four
# arms, cont-control: same warm start (g24 rings lineage, gentle
# continuation), same seed/data order; the only delta is the release arm.
# BENCH fire: nothing promotes; the gate is untouched. Revoke gold =
# self-labeled wrong bindings (two-pass step); mix unchanged (#126:
# machinery rides no diet). The re-bind three-way read (same-wrong /
# new-right / new-wrong, SPLIT BY FILLER-BEARING VS CLEAN rows — the
# rider) and the sufficiency read run post-fire on the banked ckpts.
set -eo pipefail
cd /home/bryce/mycelium
export DEV=AMD ALG2=1 ALG_FTYPES=8 ALG_DUP=1 ALG_HW=512 ALG_WIDE=1 ALG_ALLOW_PEN_TRAIN=1
export ALG_TEST=.cache/algebra_nl_test.jsonl ALG_TEST_NAME=test23
export ALG_RINGS=1 ALG_BREATH=3
PY=.venv/bin/python3

for ARM in ctl dump graded elastic; do
  if [ "$ARM" = "ctl" ]; then export ALG_XOUT=0; else export ALG_XOUT=1 ALG_XARM=$ARM; fi
  echo "=== XOUT FIRE [$ARM]: 4k gentle continuation from g24_rings ==="
  env WARM_FROM=.cache/g24_rings_rings.safetensors ALG_TRAIN=.cache/gen23_mix.jsonl \
      ALG_TRAIN_NAME=gen23 ALG_CKPT=.cache/g25_xout_${ARM}.safetensors \
      STEPS=4000 LR=1e-4 BATCH=8 SEED=251 SNAP_EVERY=0 \
      $PY scripts/phase1_algebra_head.py --train | tail -3
  echo "=== [$ARM] eval bigtest (no-regression bar: within -5 of ctl) ==="
  env ALG_CKPT=.cache/g25_xout_${ARM}.safetensors ALG_TEST=.cache/algebra_nl_bigtest.jsonl \
      ALG_TEST_NAME=bigtest $PY scripts/phase1_algebra_head.py --eval | tail -2
done
echo "=== THE XOUT FIRE IS BURNED — the three-way read comes next ==="
