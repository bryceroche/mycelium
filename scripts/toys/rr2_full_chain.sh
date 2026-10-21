#!/bin/bash
# Router race 2 (slowest gear vs bounded ramp) (Bryce's word 10-21). Bars pinned in the ledger before any full-length read.
set -eo pipefail
cd /home/bryce/mycelium
RR_RACE=2 TOY_THREADS=16 .venv/bin/python3 scripts/toys/router_race.py  # RR_RACE=2
