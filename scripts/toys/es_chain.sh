#!/bin/bash
# Bryce's word 10-23: T2i entity slots on the write side (+ held-out names).
set -eo pipefail
cd /home/bryce/mycelium
TOY_THREADS=16 .venv/bin/python3 scripts/toys/entity_slot_toy.py
