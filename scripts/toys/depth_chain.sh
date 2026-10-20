#!/bin/bash
# Opus's steps 1-2 on rung 1T (Bryce's word 10-20): the depth-generalization run, the code-overlap read on its checkpoints,
# then the second configuration (K=4). Queued behind the dictionaries and the FNet arm (no overlapping CPU jobs).
set -eo pipefail
cd /home/bryce/mycelium
while systemctl --user is-active -q dict-chain || systemctl --user is-active -q fnet-chain; do sleep 60; done
UT_MAXHOP=5 UM_ARMS=PH_oracle,PH_learn,MX_orth,TL_hop UM_TAG=_depth UM_SAVE=1 UM_LR_FROM=.cache/complex_unbind_machine.json \
  TOY_THREADS=16 .venv/bin/python3 scripts/toys/complex_unbind_machine.py
.venv/bin/python3 scripts/toys/complex_unbind_codes.py .cache/complex_unbind_machine_depth
UT_K=4 UM_ARMS=PH_oracle,PH_learn,MX_learn,MX_orth UM_TAG=_k4 UM_SAVE=1 UM_VERDICT=order UM_LR_FROM=.cache/complex_unbind_machine.json \
  TOY_THREADS=16 .venv/bin/python3 scripts/toys/complex_unbind_machine.py
