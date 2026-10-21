#!/bin/bash
# Router race 3 (gate then compare; odometer; fuzzy gates) (Bryce's word 10-21). Bars pinned in the ledger before any full-length read.
set -eo pipefail
cd /home/bryce/mycelium
RR_RACE=3 TOY_THREADS=16 .venv/bin/python3 scripts/toys/router_race.py  # RR_RACE=3
