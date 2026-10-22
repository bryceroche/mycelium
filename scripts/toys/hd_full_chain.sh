#!/bin/bash
# Train wide, read narrow: the full run (Bryce's word 10-21). Bars pinned in the ledger 18:18.
set -eo pipefail
cd /home/bryce/mycelium
TOY_THREADS=16 .venv/bin/python3 scripts/toys/hop_read.py
