#!/usr/bin/bash
# BEAM SPEED TEST 8M, BEAM=2 half rerun (ledger 11-03): the first BEAM=2 run never searched -- tinygrad's spawn-mode worker pool
# re-executes distill_reader.py (no __main__ guard) and every worker died claiming the GPU. PARALLEL=0 = the search runs in-process.
set -eo pipefail; cd /home/bryce/mycelium; exec > .cache/run_lfbeam8b.log 2>&1
echo "== start $(date +%H:%M) =="; PY=.venv/bin/python3
A="DEV=PCI+AMD DR_FF=0 DR_D=768 DR_LAYERS=2 DR_DB=1 DR_T6=3 DR_T9NS=1 DR_BREATHS=4 DR_STEPS=601"
T=$(date +%s)
flock -w 36000 .cache/gpu.lock timeout 7200 env $A BEAM=2 PARALLEL=0 DR_TAG=beam8t2b $PY scripts/distill_reader.py 2>&1 | grep -v -i warn | tee .cache/beam8t2b.log | grep -E "step (0|200|400|600) " || echo "BEAM=2 run did not finish (timeout 2 h or error)"
rm -f .cache/distill_reader_beam8t2b.*
r=$(grep -oP 'step (200|400) .*\| \K[0-9.]+(?=s/step)' .cache/beam8t2b.log | tr '\n' ' ')
echo "BEAM=2 PARALLEL=0 wall $(( $(date +%s) - T ))s | steady s/step (200->400): $(echo $r | awk '{if (NF==2) printf "%.3f", ($2*401-$1*201)/200; else print "n/a"}') | BEAM=0 was 0.267"
echo "== done $(date +%H:%M) =="
