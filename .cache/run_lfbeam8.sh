#!/usr/bin/bash
# BEAM SPEED TEST (Bryce 11-03 word given): tinygrad kernel search BEAM=0 vs BEAM=2 on the 8M MBR shape (2 tied x 4 breaths, 8.07M; Bryce 11-03: all in on 8M).
# Timing only, no claims: 601 steps each; steady rate = steps 200->400 from the cumulative s/step lines (step 0 compile/BEAM search excluded).
# Waits for lfmbr22's whole chain (GPU) and lfrope (it edits distill_reader.py) to finish.
set -eo pipefail; cd /home/bryce/mycelium; exec > .cache/run_lfbeam8.log 2>&1
echo "== start $(date +%H:%M) =="
while systemctl --user is-active -q lfbeam; do sleep 30; done
echo "gpu free $(date +%H:%M)"
PY=.venv/bin/python3
A="DEV=PCI+AMD DR_FF=0 DR_D=768 DR_LAYERS=2 DR_DB=1 DR_T6=3 DR_T9NS=1 DR_BREATHS=4 DR_STEPS=601"
for BM in 0 2; do
  T=$(date +%s)
  flock -w 36000 .cache/gpu.lock env $A BEAM=$BM DR_TAG=beam8t$BM $PY scripts/distill_reader.py 2>&1 | grep -v -i warn | tee .cache/beam8t$BM.log | grep -E "step (0|200|400|600) "
  rm -f .cache/distill_reader_beam8t$BM.*
  r=$(grep -oP 'step (200|400) .*\| \K[0-9.]+(?=s/step)' .cache/beam8t$BM.log | tr '\n' ' ')
  echo "BEAM=$BM wall $(( $(date +%s) - T ))s | steady s/step (200->400): $(echo $r | awk '{printf "%.3f", ($2*401-$1*201)/200}')"
done
echo "== done $(date +%H:%M) =="
