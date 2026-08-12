#!/bin/bash
set -eo pipefail
cd /home/bryce/mycelium
PY=.venv/bin/python3
echo "== FOCUSED GRAIN (deploy): fixture cells =="
env BG_AUTH=0.05 CK_OVERRIDE=.cache/g55_nazare.safetensors $PY scripts/bsite_smoke.py | grep bsite
echo "== FOCUSED GRAIN: passer cost =="
env BG_AUTH=0.05 CK_OVERRIDE=.cache/g55_nazare.safetensors $PY scripts/focus_cost.py | grep "focus cost"
echo "== FOCUSED READS COMPLETE =="
