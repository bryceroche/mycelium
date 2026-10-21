#!/bin/bash
# The multi-hop beam race's full run (Bryce's word 10-21). Bars pinned in the ledger.
set -eo pipefail
cd /home/bryce/mycelium
TOY_THREADS=16 .venv/bin/python3 scripts/toys/hop_race.py
