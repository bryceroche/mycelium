#!/bin/bash
set -o pipefail
cd /home/bryce/mycelium
export ALG2=1 ALG_FTYPES=8 ALG_HW=512 ALG_DUP=1 ALG_WIDE=1
for CK in g35_size8x g36_freeze8x g23v5; do
  echo "== $CK =="
  env WILD_EXCLUDE=0 CK=.cache/$CK.safetensors CK_OUT=.cache/scratch_dist.safetensors \
    .venv/bin/python3 scripts/refold_rite.py 2>&1 | grep -E "dist\]|headroom|ABORT|fold\]"
done
rm -f .cache/scratch_dist.safetensors
echo "== DIST READ COMPLETE =="
