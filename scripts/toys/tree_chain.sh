#!/bin/bash
# The tree rung's full run (Bryce's word 10-20), queued behind depth-chain (no overlapping CPU jobs).
set -eo pipefail
cd /home/bryce/mycelium
while systemctl --user is-active -q dict-chain || systemctl --user is-active -q fnet-chain || systemctl --user is-active -q depth-chain; do sleep 60; done
TOY_THREADS=16 .venv/bin/python3 scripts/toys/tree_unbind_machine.py
