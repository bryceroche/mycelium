#!/bin/bash
# The time-vs-content rung's feasibility read (Bryce's word 10-20), behind the snap control. Bars pin after it.
set -eo pipefail
cd /home/bryce/mycelium
while systemctl --user is-active -q snap-chain; do sleep 60; done
TOY_THREADS=16 .venv/bin/python3 scripts/toys/time_content_rung.py --feas
