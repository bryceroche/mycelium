#!/bin/bash
set -eo pipefail
cd /home/bryce/mycelium
export DEV=AMD ALG2=1 ALG_FTYPES=8 ALG_DUP=1 ALG_HW=512 ALG_WIDE=1 ALG_ALLOW_PEN_TRAIN=1
PY=.venv/bin/python3
echo "=== DOOR #44-RESCUE [form]: 4k from g23v5, size 8x + rehearsal(dup+formation) 3x, h_dup FROZEN ==="
env ALG_TEST=.cache/algebra_nl_test.jsonl ALG_TEST_NAME=test23 \
    WARM_FROM=.cache/g23v5.safetensors ALG_TRAIN=.cache/form_mix3.jsonl ALG_TRAIN_NAME=form3 \
    ALG_CKPT=.cache/g44r_breath.safetensors ALG_FREEZE_DUP=1 ALG_BREATH=3 BREATH_GATE_INIT=0.0 STEPS=4000 LR=1e-4 BATCH=8 SEED=127 SNAP_EVERY=0 \
    RATION_FILE=.cache/ration41_idx.json RATION_W=8 \
    RATION_FILE2=.cache/dup_only_idx.json RATION_W2=3 \
    $PY scripts/phase1_algebra_head.py --train | tail -2
echo "=== BURNED; THE PROBE READS FIRST (the spec's order) ==="
env ALG_BREATH=3 PROBE_CKPT=.cache/g44r_breath.safetensors PROBE_OUT=.cache/pointer_probe_g44r.json $PY scripts/pointer_probe.py 2>/dev/null | grep -E "pools|probe"
echo "=== CONTROL-ASSERT THEN THE READ ==="
env CK=.cache/g23v5.safetensors OUT_JSON=.cache/calib_control44r.json $PY scripts/calibrated_rite.py | grep -E "headroom|fold-ready"
echo "== control passed; candidate =="
env ALG_BREATH=3 CK=.cache/g44r_breath.safetensors OUT_JSON=.cache/calib_g44r.json CK_OUT=.cache/g44r_breath_refold.safetensors $PY scripts/calibrated_rite.py || echo "[g38] NO-HEADROOM"
echo "== g44 REFOLD scan =="
env ALG_BREATH=3 CK=.cache/g44r_breath_refold.safetensors $PY scripts/dup_axis_scan2.py | grep "^\[scan\]"
echo "== row-grade autopsy: nd=4 =="
env AUTOPSY_ND=4 AUTOPSY_CKS="g44r:.cache/g44r_breath_refold.safetensors" ALG_BREATH=3 $PY scripts/formation_autopsy.py 2>/dev/null | grep -E "cured|fails" || true
echo "== row-grade autopsy: nd=0 =="
env AUTOPSY_ND=0 AUTOPSY_CKS="g44r:.cache/g44r_breath_refold.safetensors" ALG_BREATH=3 $PY scripts/formation_autopsy.py 2>/dev/null | grep -E "cured|fails|row" || true
echo "== OP AUTOPSY (primary read: add-logit return) =="
env ND=0 OPAUT_NAME=g44 OPAUT_CK=.cache/g44r_breath_refold.safetensors $PY scripts/op_autopsy.py 2>/dev/null | grep -E "gold=|miss|fork" || true
echo "== frontier =="
env ALG_BREATH=3 WFF_CKPT=.cache/g44r_breath_refold.safetensors WFF_OUT=.cache/wff_g44rrefold.json $PY scripts/wild_frontier_fixture.py | grep -E "DEPLOYED|STRESS|LIE" || true
echo "== bigtest =="
env ALG_BREATH=3 ALG_CKPT=.cache/g44r_breath_refold.safetensors ALG_TEST=.cache/algebra_nl_bigtest.jsonl ALG_TEST_NAME=bigtest $PY scripts/phase1_algebra_head.py --eval | grep TOTAL
echo "== alg4test =="
env ALG_BREATH=3 ALG_CKPT=.cache/g44r_breath_refold.safetensors ALG_TEST=.cache/algebra4_nl_test.jsonl ALG_TEST_NAME=alg4test $PY scripts/phase1_algebra_head.py --eval | grep TOTAL
echo "== sentinels =="
env ALG_BREATH=3 CK=.cache/g44r_breath_refold.safetensors $PY scripts/aim_battery.py | grep sentinel
echo "== P-233 READ (ordinal-graded conversions) =="
env ALG_BREATH=3 CENSUS_CKPT=.cache/g44r_breath_refold.safetensors CENSUS_OUT=.cache/miss_census_g44r.json $PY scripts/miss_census.py 2>/dev/null | grep -E "census|len\]" || true
$PY scripts/p233_read.py
echo "== DOOR #44-RESCUE BATTERY COMPLETE =="
