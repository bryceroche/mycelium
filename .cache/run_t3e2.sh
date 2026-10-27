#!/usr/bin/bash
# T3e2 (ledger 2026-10-27, word given): NB 24 on FRESH seeds 5-9 (REREAD, BUS, BUS256) + the BUS256 width read at NB 12 (seeds 0-4).
set -eo pipefail; cd /home/bryce/mycelium; exec > .cache/run_t3e2.log 2>&1
while systemctl --user is-active -q headcont; do sleep 60; done          # CPU toys run alone (the 10-26 contention lesson)
echo "== start $(date +%H:%M) =="
TE_TAG=2 TE_ARMS=REREAD,BUS,BUS256 TM_SEEDS=5,6,7,8,9 TM_NB=24 TOY_THREADS=16 .venv/bin/python3 scripts/toys/reread_toy.py
TE_TAG=2 TE_ARMS=BUS256 TM_SEEDS=0,1,2,3,4 TM_NB=12 TOY_THREADS=16 .venv/bin/python3 scripts/toys/reread_toy.py
.venv/bin/python3 scripts/toys/reread_t3e2_verdict.py
echo "== done $(date +%H:%M) =="
