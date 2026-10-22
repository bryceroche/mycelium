#!/bin/bash
# T3, the cross-join toy (Bryce's word): W 32 (construction confirmations) then W 128 (the real bars + confirmations).
set -eo pipefail
cd /home/bryce/mycelium
CJ_W=32 TOY_THREADS=16 .venv/bin/python3 scripts/toys/cross_join_toy.py
CJ_W=128 TOY_THREADS=16 .venv/bin/python3 scripts/toys/cross_join_toy.py
