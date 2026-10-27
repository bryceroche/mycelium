#!/usr/bin/bash
# THE EQ_SLOT TASK (ledger 2026-10-27, word given): BREATH vs VANILLA x seeds 0,1 on the reader B2 picked; solver-graded holdout.
set -eo pipefail; cd /home/bryce/mycelium; exec > .cache/run_eqslot.log 2>&1
while systemctl --user is-active -q headcont || systemctl --user is-active -q t3e2; do sleep 60; done   # CPU runs alone
# THE READER RULE (pinned 10-27): the student's states iff B2's PARITY passed (HCS - HCT >= -0.020 on the paired wild slots)
R=$(.venv/bin/python3 -c "import numpy as np; s=np.load('.cache/ps_headcont_HCS.npz')['ok'].mean(); t=np.load('.cache/ps_headcont_HCT.npz')['ok'].mean(); print('STU' if s - t >= -0.020 else 'BP')")
echo "== start $(date +%H:%M) | reader $R =="
for S in 0 1; do for A in BREATH VANILLA; do
  EQ_ARM=$A EQ_SEED=$S EQ_READER=$R TOY_THREADS=16 .venv/bin/python3 scripts/eq_slot_breath.py 2>&1 | grep -v -i warn
done; done
.venv/bin/python3 scripts/eq_slot_verdict.py $R
echo "== done $(date +%H:%M) =="
