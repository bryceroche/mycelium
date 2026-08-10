#!/bin/bash
set -eo pipefail
cd /home/bryce/mycelium
export ALG2=1 ALG_FTYPES=8 ALG_HW=512 ALG_DUP=1 ALG_WIDE=1
PY=.venv/bin/python3
echo "== CONTROL-ASSERT: the gate (g23v5) under the calibrated rite =="
env CK=.cache/g23v5.safetensors OUT_JSON=.cache/calib_control.json $PY scripts/calibrated_rite.py
echo "== CONTROL PASSED — the instrument reads. Candidates: =="
echo "== g35_size8x =="
env CK=.cache/g35_size8x.safetensors OUT_JSON=.cache/calib_g35.json CK_OUT=.cache/g35_size8x_refold.safetensors $PY scripts/calibrated_rite.py || echo "[g35] NO-HEADROOM"
echo "== g36_freeze8x =="
env CK=.cache/g36_freeze8x.safetensors OUT_JSON=.cache/calib_g36.json CK_OUT=.cache/g36_freeze8x_refold.safetensors $PY scripts/calibrated_rite.py || echo "[g36] NO-HEADROOM"
echo "== CALIBRATED READS COMPLETE =="
