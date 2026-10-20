#!/bin/bash
# The dictionary read, queued behind rung 1T's full run (no overlapping CPU jobs: the 10-20 oversubscription lesson).
set -eo pipefail
cd /home/bryce/mycelium
while systemctl --user is-active -q complex-r1t-full; do sleep 60; done
for tap in 4 7; do
  TOY_THREADS=16 .venv/bin/python3 scripts/toys/dict_sae.py --tap $tap
done
