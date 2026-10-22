#!/bin/bash
# Bryce's word 10-22 15:1x: T2e label noise, then the T3 follow-ups (parity + per-hop table). Sequential.
set -eo pipefail
cd /home/bryce/mycelium
TOY_THREADS=16 .venv/bin/python3 scripts/toys/label_noise_toy.py
TOY_THREADS=16 .venv/bin/python3 scripts/toys/cj_followup.py
