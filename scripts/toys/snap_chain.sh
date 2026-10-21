#!/bin/bash
# The snap control (Opus) + the static-memory split (coordinator), Bryce's word 10-20. One CPU job.
set -eo pipefail
cd /home/bryce/mycelium
UT_MAXHOP=5 UM_ARMS=TL_snap,TL_snapmem UM_TAG=_snap TOY_THREADS=16 .venv/bin/python3 scripts/toys/complex_unbind_machine.py
