#!/bin/bash
set -eo pipefail
cd /home/bryce/mycelium
export DEV=AMD ALG2=1 ALG_FTYPES=8 ALG_DUP=1 ALG_HW=512 ALG_WIDE=1 ALG_ALLOW_PEN_TRAIN=1
PY=.venv/bin/python3
echo "=== DOOR #55: NAZARE TRAINING (focused regime, NAZ_BG=0.05) ==="
env ALG_TEST=.cache/algebra_nl_test.jsonl ALG_TEST_NAME=test23 \
    WARM_FROM=.cache/g23v5.safetensors ALG_TRAIN=.cache/form_mix3.jsonl ALG_TRAIN_NAME=form3 \
    ALG_CKPT=.cache/g55_nazare.safetensors ALG_FREEZE_DUP=1 ALG_BREATH=3 BREATH_NORM=1 BREATH_WARM_BO=1 BREATH_GATE_INIT=0.0 NAZ_TRAIN=1 NAZ_BG=0.05 \
    STEPS=4000 LR=1e-4 BATCH=8 SEED=127 SNAP_EVERY=0 \
    RATION_FILE=.cache/ration41_idx.json RATION_W=8 \
    RATION_FILE2=.cache/dup_only_idx.json RATION_W2=3 \
    $PY scripts/phase1_algebra_head.py --train | tail -1
echo "=== BURNED; vacuousness ==="
$PY -c "
import numpy as np
from tinygrad.nn.state import safe_load
sd=safe_load('.cache/g55_nazare.safetensors')
g=sd['breath_gate'].numpy(); print('[organ] gates',g,'sigmoid',1/(1+np.exp(-g)))
print(f'[organ] |W_bo| {np.abs(sd[\"W_bo\"].numpy()).mean():.5f}')"
echo "== REMOVAL TRIPLE: silent / engaged-uniform / (focused = the deploy read, via smoke harness after) =="
env ALG_BREATH=3 CK=.cache/g55_nazare.safetensors $PY scripts/dup_axis_scan2.py | grep "^\[scan\]"
env ALG_BREATH=3 TWO_PASS=1 CK=.cache/g55_nazare.safetensors $PY scripts/dup_axis_scan2.py | grep "^\[scan\]"
env ALG_BREATH=3 CENSUS_CKPT=.cache/g55_nazare.safetensors CENSUS_OUT=.cache/mc_g55_sp.json $PY scripts/miss_census.py 2>/dev/null | grep census
env P233_NEW=.cache/mc_g55_sp.json $PY scripts/p233_read.py
env ALG_BREATH=3 TWO_PASS=1 CENSUS_CKPT=.cache/g55_nazare.safetensors CENSUS_OUT=.cache/mc_g55_2p.json $PY scripts/miss_census.py 2>/dev/null | grep census
env P233_NEW=.cache/mc_g55_2p.json $PY scripts/p233_read.py
echo "== ring (silent-grain reported) =="
env ALG_BREATH=3 BREATH_SILENT=1 ALG_CKPT=.cache/g55_nazare.safetensors ALG_TEST=.cache/algebra_nl_bigtest.jsonl ALG_TEST_NAME=bigtest $PY scripts/phase1_algebra_head.py --eval | grep TOTAL
env ALG_BREATH=3 ALG_CKPT=.cache/g55_nazare.safetensors ALG_TEST=.cache/algebra_nl_bigtest.jsonl ALG_TEST_NAME=bigtest $PY scripts/phase1_algebra_head.py --eval | grep TOTAL
env ALG_BREATH=3 CK=.cache/g55_nazare.safetensors $PY scripts/aim_battery.py | grep sentinel
echo "== DOOR #55 BATTERY COMPLETE (focused-grain reads follow via harness) =="
