#!/bin/bash
# T2, the bound working memory (Bryce's word 10-22): feasibility (seed 0) then, only on PASS, the full 3-seed run.
set -eo pipefail
cd /home/bryce/mycelium
TOY_THREADS=16 .venv/bin/python3 scripts/toys/bound_memory_toy.py --feas
grep -q "FEASIBILITY GATE.*PASS" .cache/bound_memory_feas.txt
TOY_THREADS=16 .venv/bin/python3 scripts/toys/bound_memory_toy.py
