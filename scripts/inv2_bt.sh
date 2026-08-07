#!/bin/bash
set -eo pipefail
cd /home/bryce/mycelium
export ALG2=1 ALG_FTYPES=8 ALG_HW=512 ALG_DUP=1
for CK in ctl inv; do
  echo "== $CK =="
  env ALG_CKPT=.cache/g25v2_inv_${CK}.safetensors ALG_TEST=.cache/algebra_nl_bigtest.jsonl ALG_TEST_NAME=bigtest .venv/bin/python3 scripts/phase1_algebra_head.py --eval | grep -E "TOTAL"
done
