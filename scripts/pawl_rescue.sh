#!/bin/bash
set -eo pipefail
cd /home/bryce/mycelium
export DEV=AMD ALG2=1 ALG_FTYPES=8 ALG_DUP=1 ALG_HW=512 ALG_WIDE=1 ALG_ALLOW_PEN_TRAIN=1
PY=.venv/bin/python3
echo "=== DOOR #54-R: BIAS-OPEN PAWL ==="
env ALG_TEST=.cache/algebra_nl_test.jsonl ALG_TEST_NAME=test23 \
    WARM_FROM=.cache/g23v5.safetensors ALG_TRAIN=.cache/form_mix3.jsonl ALG_TRAIN_NAME=form3 \
    ALG_CKPT=.cache/g54r_open.safetensors ALG_FREEZE_DUP=1 ALG_BREATH=3 ALG_RINGS=1 CMT_B_INIT=-2.0 BREATH_NORM=1 BREATH_WARM_BO=1 BREATH_GATE_INIT=0.0 BREATH_DROPOUT=0.5 ALG_RINGS=1 \
    STEPS=4000 LR=1e-4 BATCH=8 SEED=127 SNAP_EVERY=0 \
    RATION_FILE=.cache/ration41_idx.json RATION_W=8 \
    RATION_FILE2=.cache/dup_only_idx.json RATION_W2=3 \
    $PY scripts/phase1_algebra_head.py --train | tail -1
echo "=== BURNED; vacuousness (organ + PAWL) ==="
$PY -c "
import numpy as np
from tinygrad.nn.state import safe_load
sd=safe_load('.cache/g54r_open.safetensors')
g=sd['breath_gate'].numpy(); print('[organ] gates',g,'sigmoid',1/(1+np.exp(-g)))
print(f'[organ] |W_bo| {np.abs(sd[\"W_bo\"].numpy()).mean():.5f}')
print(f'[pawl] |W_cmt| {np.abs(sd[\"W_cmt\"].numpy()).mean():.5f}  b {float(sd[\"W_cmt_b\"].numpy()[0]):+.3f} (init -4.0)')"
echo "== SILENT GRAIN (deploy target): cells + lean + score =="
env ALG_BREATH=3 ALG_RINGS=1 CK=.cache/g54r_open.safetensors $PY scripts/dup_axis_scan2.py | grep "^\[scan\]"
env ALG_BREATH=3 ALG_RINGS=1 CENSUS_CKPT=.cache/g54r_open.safetensors CENSUS_OUT=.cache/mc_g54r_sp.json $PY scripts/miss_census.py 2>/dev/null | grep census
env P233_NEW=.cache/mc_g54r_sp.json $PY scripts/p233_read.py
env ALG_BREATH=3 ALG_RINGS=1 BREATH_SILENT=1 ALG_CKPT=.cache/g54r_open.safetensors ALG_TEST=.cache/algebra_nl_bigtest.jsonl ALG_TEST_NAME=bigtest $PY scripts/phase1_algebra_head.py --eval | grep TOTAL
echo "== ENGAGED GRAIN (record) =="
env ALG_BREATH=3 ALG_RINGS=1 TWO_PASS=1 CK=.cache/g54r_open.safetensors $PY scripts/dup_axis_scan2.py | grep "^\[scan\]"
env ALG_BREATH=3 ALG_RINGS=1 TWO_PASS=1 CENSUS_CKPT=.cache/g54r_open.safetensors CENSUS_OUT=.cache/mc_g54r_2p.json $PY scripts/miss_census.py 2>/dev/null | grep census
env P233_NEW=.cache/mc_g54r_2p.json $PY scripts/p233_read.py
env ALG_BREATH=3 ALG_RINGS=1 ALG_CKPT=.cache/g54r_open.safetensors ALG_TEST=.cache/algebra_nl_bigtest.jsonl ALG_TEST_NAME=bigtest $PY scripts/phase1_algebra_head.py --eval | grep TOTAL
echo "== ring extras =="
env ALG_BREATH=3 ALG_RINGS=1 BREATH_SILENT=1 ALG_CKPT=.cache/g54r_open.safetensors ALG_TEST=.cache/algebra4_nl_test.jsonl ALG_TEST_NAME=alg4test $PY scripts/phase1_algebra_head.py --eval | grep TOTAL
env ALG_BREATH=3 ALG_RINGS=1 CK=.cache/g54r_open.safetensors $PY scripts/aim_battery.py | grep sentinel
echo "== DOOR #52 COMPLETE =="
