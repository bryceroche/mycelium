#!/bin/bash
set -eo pipefail
cd /home/bryce/mycelium
export DEV=AMD ALG2=1 ALG_FTYPES=8 ALG_DUP=1 ALG_HW=512 ALG_WIDE=1 ALG_ALLOW_PEN_TRAIN=1
PY=.venv/bin/python3
echo "=== ASSEMBLE: form_mix states + gold ==="
$PY scripts/form_assemble2.py
echo "=== DOOR #38 [form]: 4k from g23v5, size 8x + rehearsal(dup+formation) 3x, h_dup FROZEN ==="
env ALG_TEST=.cache/algebra_nl_test.jsonl ALG_TEST_NAME=test23 \
    WARM_FROM=.cache/g23v5.safetensors ALG_TRAIN=.cache/form_mix2.jsonl ALG_TRAIN_NAME=form \
    ALG_CKPT=.cache/g38_form.safetensors ALG_FREEZE_DUP=1 STEPS=4000 LR=1e-4 BATCH=8 SEED=127 SNAP_EVERY=0 \
    RATION_FILE=.cache/size_sliver_idx.json RATION_W=8 \
    RATION_FILE2=.cache/form_rehearsal_idx2.json RATION_W2=3 \
    $PY scripts/phase1_algebra_head.py --train | tail -2
echo "=== BURNED; CONTROL-ASSERT THEN THE READ ==="
env CK=.cache/g23v5.safetensors OUT_JSON=.cache/calib_control38.json $PY scripts/calibrated_rite.py | grep -E "headroom|fold-ready"
echo "== control passed; candidate =="
env CK=.cache/g38_form.safetensors OUT_JSON=.cache/calib_g38.json CK_OUT=.cache/g38_form_refold.safetensors $PY scripts/calibrated_rite.py || echo "[g37] NO-HEADROOM"
echo "== g37 REFOLD scan =="
env CK=.cache/g38_form_refold.safetensors $PY scripts/dup_axis_scan2.py | grep "^\[scan\]"
echo "== row-grade autopsy re-read =="
env AUTOPSY_CKS="gate:.cache/g23v5.safetensors,g37:.cache/g38_form_refold.safetensors" $PY scripts/formation_autopsy.py 2>/dev/null | grep -E "cured|row|fails" || true
echo "== bigtest =="
env ALG_CKPT=.cache/g38_form_refold.safetensors ALG_TEST=.cache/algebra_nl_bigtest.jsonl ALG_TEST_NAME=bigtest $PY scripts/phase1_algebra_head.py --eval | grep TOTAL
echo "== alg4test =="
env ALG_CKPT=.cache/g38_form_refold.safetensors ALG_TEST=.cache/algebra4_nl_test.jsonl ALG_TEST_NAME=alg4test $PY scripts/phase1_algebra_head.py --eval | grep TOTAL
echo "== sentinels =="
env CK=.cache/g38_form_refold.safetensors $PY scripts/aim_battery.py | grep sentinel
echo "== DOOR #38 BATTERY COMPLETE =="
