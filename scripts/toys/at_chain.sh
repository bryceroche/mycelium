#!/bin/bash
# Bryce's word 10-22: T3c (the Auto-Tune snap), queued behind unit ln-cj (never overlap CPU jobs).
set -eo pipefail
cd /home/bryce/mycelium
while systemctl --user is-active --quiet ln-cj; do sleep 30; done
TOY_THREADS=16 .venv/bin/python3 scripts/toys/autotune_snap.py
