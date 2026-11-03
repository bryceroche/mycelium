#!/usr/bin/bash
# THE RoPE RETIREMENT RULING (Bryce 11-03): apply scripts/patches/rope_retire.py once lfmbr22 + lfbt7b (pinned on the old defaults) are done
set -eo pipefail; cd /home/bryce/mycelium; exec > .cache/run_lfrope.log 2>&1
echo "== start $(date +%H:%M) =="
while ! grep -qE "== done" .cache/run_lfmbr22.log 2>/dev/null || ! grep -qE "== done" .cache/run_lfbt7b.log 2>/dev/null; do sleep 300; done
while pgrep -f 'scripts/(distill_reader|gsm_breath_reader|gsm_bt7_read).py' >/dev/null; do sleep 60; done
PY=.venv/bin/python3; $PY scripts/patches/rope_retire.py
( RD_STATES=gsmb_student_states.npy $PY -c "import sys; sys.path.insert(0,'scripts'); import gsm_trunk_reader as TR; TR.load()" 2>&1 | grep -q "RoPE-era" ) && echo "guard: legacy states refused OK"
$PY -c "import sys; sys.path.insert(0,'scripts'); import gsm_trunk_reader as TR; D, s = TR.load(); print('default states', s.shape)"
echo "== done $(date +%H:%M) =="
