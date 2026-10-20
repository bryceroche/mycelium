#!/bin/bash
# The FNet context arm, queued behind the dictionary chain (no overlapping CPU jobs).
set -eo pipefail
cd /home/bryce/mycelium
while systemctl --user is-active -q complex-r1t-full || systemctl --user is-active -q dict-chain; do sleep 60; done
UM_ARMS=T6_fnet UM_TAG=_fnet TOY_THREADS=16 .venv/bin/python3 scripts/toys/complex_unbind_machine.py
