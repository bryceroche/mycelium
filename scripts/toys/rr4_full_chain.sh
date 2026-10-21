#!/bin/bash
# Router race 4 (the fuzzy gate with a known leak) (Bryce's word 10-21). Bars pinned in the ledger before any full-length read.
set -eo pipefail
cd /home/bryce/mycelium
RR_RACE=4 TOY_THREADS=16 .venv/bin/python3 scripts/toys/router_race.py  # RR_RACE=4
