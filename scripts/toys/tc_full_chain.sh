#!/bin/bash
# The time-vs-content full run with the registers extension (Bryce's word 10-20 ~20:50). Bars pinned in the ledger.
set -eo pipefail
cd /home/bryce/mycelium
TOY_THREADS=16 .venv/bin/python3 scripts/toys/time_content_rung.py
