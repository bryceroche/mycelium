#!/bin/bash
# Bryce's word 10-22 (~17:40): T2f construction-tied label errors. Single job.
set -eo pipefail
cd /home/bryce/mycelium
TOY_THREADS=16 .venv/bin/python3 scripts/toys/label_construct_toy.py
