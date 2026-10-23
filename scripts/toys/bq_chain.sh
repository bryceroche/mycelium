#!/bin/bash
# Bryce's word 10-22 (~19:10): T2g the blind query (probe gate first; arms only if the gate passes).
set -eo pipefail
cd /home/bryce/mycelium
TOY_THREADS=16 .venv/bin/python3 scripts/toys/blind_query_toy.py
