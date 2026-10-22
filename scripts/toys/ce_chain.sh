#!/bin/bash
# T2b, composite entities (Bryce's word 10-22): after T2 (bound-mem) ends; a 20-step code check, then the full run.
set -eo pipefail
cd /home/bryce/mycelium
while systemctl --user is-active --quiet bound-mem; do sleep 30; done
CE_STEPS=20 TOY_THREADS=16 .venv/bin/python3 scripts/toys/composite_entity_toy.py --smoke
TOY_THREADS=16 .venv/bin/python3 scripts/toys/composite_entity_toy.py
