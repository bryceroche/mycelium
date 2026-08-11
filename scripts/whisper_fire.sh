#!/bin/bash
set -eo pipefail
cd /home/bryce/mycelium
export DEV=AMD ALG2=1 ALG_FTYPES=8 ALG_DUP=1 ALG_HW=512 ALG_WIDE=1 ALG_ALLOW_PEN_TRAIN=1
PY=.venv/bin/python3
echo "=== DOOR #51: WHISPER TITRATION (5% blend) ==="
env ALG_TEST=.cache/algebra_nl_test.jsonl ALG_TEST_NAME=test23 \
    WARM_FROM=.cache/g23v5.safetensors ALG_TRAIN=.cache/form_mix3.jsonl ALG_TRAIN_NAME=form3 \
    ALG_CKPT=.cache/g51_whisper.safetensors ALG_FREEZE_DUP=1 ALG_BREATH=3 BREATH_NORM=1 BREATH_WARM_BO=1 BREATH_GATE_INIT=-3.0 \
    STEPS=4000 LR=1e-4 BATCH=8 SEED=127 SNAP_EVERY=0 \
    RATION_FILE=.cache/ration41_idx.json RATION_W=8 \
    RATION_FILE2=.cache/dup_only_idx.json RATION_W2=3 \
    $PY scripts/phase1_algebra_head.py --train | tail -1
echo "=== BURNED; VACUOUSNESS CHECK FIRST ==="
$PY -c "
import numpy as np
from tinygrad.nn.state import safe_load
sd=safe_load('.cache/g51_whisper.safetensors'); ref=safe_load('.cache/g23v5.safetensors')
g=sd['breath_gate'].numpy(); import math
print('[organ] gates', g, 'sigmoid', 1/(1+np.exp(-g)))
d=(sd['W_bo'].numpy()-ref['attn_wo'].numpy())
print(f'[organ] |W_bo - attn_wo(seed)| {np.abs(d).mean():.5f}  |W_bo| {np.abs(sd[\"W_bo\"].numpy()).mean():.5f}')"
echo "== SURVIVAL PRIMARY: nd2 + wall (breath env on reads) =="
env ALG_BREATH=3 CK=.cache/g51_whisper.safetensors $PY scripts/dup_axis_scan2.py | grep "^\[scan\]"
echo "== pointer probe (transport) =="
env ALG_BREATH=3 PROBE_CKPT=.cache/g51_whisper.safetensors PROBE_OUT=.cache/probe_g51.json $PY scripts/pointer_probe.py 2>/dev/null | grep probe
echo "== P-233 lean =="
env ALG_BREATH=3 CENSUS_CKPT=.cache/g51_whisper.safetensors CENSUS_OUT=.cache/miss_census_g51.json $PY scripts/miss_census.py 2>/dev/null | grep census
env P233_NEW=.cache/miss_census_g51.json $PY scripts/p233_read.py
echo "== ring (reported; score soft) =="
env ALG_BREATH=3 ALG_CKPT=.cache/g51_whisper.safetensors ALG_TEST=.cache/algebra_nl_bigtest.jsonl ALG_TEST_NAME=bigtest $PY scripts/phase1_algebra_head.py --eval | grep TOTAL
env ALG_BREATH=3 ALG_CKPT=.cache/g51_whisper.safetensors ALG_TEST=.cache/algebra4_nl_test.jsonl ALG_TEST_NAME=alg4test $PY scripts/phase1_algebra_head.py --eval | grep TOTAL
env ALG_BREATH=3 CK=.cache/g51_whisper.safetensors $PY scripts/aim_battery.py | grep sentinel
echo "== DOOR #50 COMPLETE =="
