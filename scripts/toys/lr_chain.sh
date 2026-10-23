#!/bin/bash
# Bryce's word 10-23: T2j learned resolver on realistic mentions (aliases, families, possessives, role pronouns).
set -eo pipefail
cd /home/bryce/mycelium
TOY_THREADS=16 .venv/bin/python3 scripts/toys/learned_resolver_toy.py
