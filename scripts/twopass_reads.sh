#!/bin/bash
set -eo pipefail
cd /home/bryce/mycelium
export ALG2=1 ALG_FTYPES=8 ALG_HW=512 ALG_DUP=1 ALG_WIDE=1 ALG_BREATH=3 TWO_PASS=1
PY=.venv/bin/python3
for ARM in g50_boot g50r_low g51_whisper; do
  echo "=== TWO-PASS RE-READ: $ARM ==="
  env CK=.cache/$ARM.safetensors $PY scripts/dup_axis_scan2.py | grep "^\[scan\]"
  env CENSUS_CKPT=.cache/$ARM.safetensors CENSUS_OUT=.cache/miss_census_${ARM}_2p.json $PY scripts/miss_census.py 2>/dev/null | grep census
  env P233_NEW=.cache/miss_census_${ARM}_2p.json $PY scripts/p233_read.py
done
echo "== TWO-PASS RE-READS COMPLETE =="
