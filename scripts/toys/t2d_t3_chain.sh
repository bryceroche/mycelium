#!/bin/bash
# Bryce's word 10-22 ~12:00: T2d (the distractor-owner implicit query) then T3 (the cross-join toy, W 32 then W 128). Sequential: never overlap CPU jobs.
set -eo pipefail
cd /home/bryce/mycelium
SS_DISTRACT=1 TOY_THREADS=16 .venv/bin/python3 scripts/toys/supervision_split_toy.py --feas
grep -q "FEASIBILITY GATE.*PASS" .cache/supervision_split_d_feas.txt
SS_DISTRACT=1 TOY_THREADS=16 .venv/bin/python3 scripts/toys/supervision_split_toy.py
/home/bryce/mycelium/scripts/toys/cj_chain.sh
