#!/bin/bash
# g17_heldfix.sh — re-eval the two held fixtures (now real-keyed) for both
# arms, splice into the battery logs, re-invoke the pen.
set -eo pipefail
cd /home/bryce/mycelium
export DEV=AMD ALG2=1 ALG_FTYPES=8 ALG_DUP=1
PY=.venv/bin/python3

for cand in F R; do
  ckpt=.cache/g17_arm${cand}.safetensors
  log=.cache/gen17_${cand}.log
  LOGFILE=$log $PY - << 'PYEOF'
import os, re
log_path = os.environ["LOGFILE"]
log = open(log_path).read()
log = re.sub(r"--- h3held ---.*?(?=--- |\Z)", "", log, flags=re.S)
log = re.sub(r"--- adupheld ---.*?(?=--- |\Z)", "", log, flags=re.S)
open(log_path, "w").write(log)
PYEOF
  for fx in h3held:gen17_hundreds_held adupheld:gen17_adup_held; do
    name=${fx%%:*}; file=${fx##*:}
    echo "--- $name ---" >> "$log"
    ALG_CKPT=$ckpt ALG_TEST=.cache/$file.jsonl ALG_TEST_NAME=$name \
      $PY scripts/phase1_algebra_head.py --eval >> "$log" 2>&1
    echo "[$cand/$name] $(grep -E 'TOTAL' "$log" | tail -1)"
  done
done
echo "=== RE-EVAL DONE — the pen speaks ==="
$PY scripts/gen17_verdict.py || true
