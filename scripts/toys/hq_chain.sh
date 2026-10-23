#!/bin/bash
# Bryce's word 10-23: T2h impose the second hop (vs the learned two-hop and the one-hop reader).
set -eo pipefail
cd /home/bryce/mycelium
TOY_THREADS=16 .venv/bin/python3 scripts/toys/hop_query_toy.py
