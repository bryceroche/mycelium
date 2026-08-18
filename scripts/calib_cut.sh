#!/bin/bash
set -eo pipefail
cd /home/bryce/mycelium
PY=.venv/bin/python3
echo "== CUT CALIBRATION: K3@74k profile =="
env DEV=AMD ALG2=1 ALG_FTYPES=9 ALG_DUP=1 ALG_HW=512 ALG_WIDE=1 ALG_SIXWAVE=1 ALG_BREATH=3 BREATH_NORM=1 ALG_DEEPSUP=1 \
  PR_CK=.cache/gnat_native.safetensors ALG_TEST=.cache/algebra_nl_bigtest.jsonl ALG_TEST_NAME=bigtest $PY scripts/breath_profile.py
echo "== K5@74k resample (seed 99) =="
env DEV=AMD ALG2=1 ALG_FTYPES=9 ALG_DUP=1 ALG_HW=512 ALG_WIDE=1 ALG_SEPHASE=1 ALG_SIXWAVE=1 ALG_BREATH=5 BREATH_NORM=1 ALG_DEEPSUP=1 \
  PR_CK=.cache/gk5_arm.safetensors ALG_TEST=.cache/algebra_nl_bigtest.jsonl ALG_TEST_NAME=bigtest $PY - << 'PEOF'
import sys
sys.argv=["x"]
exec(open('scripts/breath_profile.py').read().replace("default_rng(41)","default_rng(99)"))
PEOF
echo "== CALIB COMPLETE =="
