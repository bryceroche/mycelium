#!/bin/bash
set -eo pipefail
cd /home/bryce/mycelium
export DEV=AMD ALG2=1 ALG_FTYPES=8 ALG_DUP=1 ALG_HW=512 ALG_WIDE=1
export ALG_CKPT=.cache/g41_onemass_refold.safetensors
export ALG_TEST=.cache/algebra_nl_bigtest.jsonl ALG_TEST_NAME=bigtest
echo "=== ENGINEERED VIEWS (scope pinned: decorrelation-beats-convenience) ==="
.venv/bin/python3 scripts/engineered_views.py
echo "== EVIEW COMPLETE =="
