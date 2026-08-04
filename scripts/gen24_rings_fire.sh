#!/bin/bash
# gen24_rings_fire.sh — THE RUNG-3 BENCH FIRE (2026-08-04, the word given;
# registration + pinned bars in the ledger BEFORE this ran). Two arms,
# cont-control: same warm start, same seed/data order; the only delta is
# the pawl. BENCH fire: nothing promotes; the gate is untouched.
set -eo pipefail
cd /home/bryce/mycelium
export DEV=AMD ALG2=1 ALG_FTYPES=8 ALG_DUP=1 ALG_HW=512 ALG_WIDE=1 ALG_ALLOW_PEN_TRAIN=1
export ALG_TEST=.cache/algebra_nl_test.jsonl ALG_TEST_NAME=test23
PY=.venv/bin/python3

for ARM in ctl rings; do
  if [ "$ARM" = "rings" ]; then export ALG_RINGS=1; else export ALG_RINGS=0; fi
  export ALG_BREATH=3
  echo "=== RINGS FIRE [$ARM]: 4k from g23 (breath params fresh, init-closed) ==="
  env WARM_FROM=.cache/g23.safetensors ALG_TRAIN=.cache/gen23_mix.jsonl \
      ALG_TRAIN_NAME=gen23 ALG_CKPT=.cache/g24_rings_${ARM}.safetensors \
      STEPS=4000 LR=1e-4 BATCH=8 SEED=241 SNAP_EVERY=0 \
      $PY scripts/phase1_algebra_head.py --train | tail -3
  for FIX in "adupheld .cache/gen17_adup_held.jsonl" "bigtest .cache/algebra_nl_bigtest.jsonl"; do
    set -- $FIX
    echo "=== [$ARM] eval $1 ==="
    env ALG_CKPT=.cache/g24_rings_${ARM}.safetensors ALG_TEST=$2 ALG_TEST_NAME=$1 \
        $PY scripts/phase1_algebra_head.py --eval | tail -2
  done
done
unset ALG_RINGS; unset ALG_BREATH
echo "=== [g23-static] eval adupheld (third line) ==="
env ALG_CKPT=.cache/g23.safetensors ALG_TEST=.cache/gen17_adup_held.jsonl ALG_TEST_NAME=adupheld \
    $PY scripts/phase1_algebra_head.py --eval | tail -2
echo "=== THE RUNG-3 BENCH FIRE IS BURNED — the verdict reads next ==="
