#!/usr/bin/env bash
# v200 #237 — structural READ masks (§2 per-latent topology mask, partition 1a).
#
# Single experimental variable vs #236: latent_topology_mask (L=32, T=24),
# partition 1a = 24 per-token + 4 per-op + 4 global. Everything else is #236's
# at-spec configuration (K=8, LR=3e-4, BATCH=8, 6-RMSNorm substrate,
# delta_gate -2.0, alpha_read 1.0). Cold-start.
#
# ADVISORY (§11, surfaces in line 1): per-op mask family uses token_index-mod-4
# proxy, not §2 op_type routing. Pre-pinned consequence: per-op group's
# entropy-separation read does not bind; per-token + global + C4 carry it.
#
# Horizon: 2000 steps. Eval checkpoints {200, 500, 1000, 1500, 2000}.
# Dense model ckpts every 100 steps to 1K, every 250 after (§9 — this run is
# the control-generator).
#
# Sequence:
#   1. STEP-0 GATE: per-mask-family entropy assertion at random init (§1A.E.10/§7).
#      Fails => mask not wired => script aborts BEFORE training.
#   2. Reference curves regenerated under masked arch (MEASURED entropies, §6 fresh null).
#   3. 2000-step training with per-checkpoint instrumentation (single tapped forward).
#   4. Final read: C1-C6 (C5 recalibrated), §1A.E.4 grid, E.10 reads, #238 routing.
#
# Run:
#   cd /home/bryce/mycelium && bash scripts/v200_resmoke_237_5.sh
#
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-.venv/bin/python}"
SMOKE_DIR="${SMOKE_DIR:-.cache/v200_smoke}"
mkdir -p "$SMOKE_DIR" "$SMOKE_DIR/persistence" "$SMOKE_DIR/reference_curves"

echo "=== v200 #237.5 — SUBSTRATE RESTORATION (detached seams + fp32 chain + where-guard) ==="
echo "  SMOKE_DIR: $SMOKE_DIR"
echo "  PYTHON: $PYTHON"
echo "  Substrate fixes vs #237 (masks inherited): detached seams + fp32 chain + where-guard; mask (24 per-token + 4 per-op + 4 global)"
echo "  STEP-0 GATE: per-group entropy < log(support) [1/6/24], per-group never mean"
echo "  ADVISORY: per-op family = mod-4 proxy (see line 1 of train_237_5.log)"
echo ""

V200_TASK=1 \
V200_STAGE2A_WAIST=1 \
V200_K_MAX="${V200_K_MAX:-8}" \
V200_N_LATENTS=32 \
V200_N_VAR_LAT=16 \
V200_N_DIGITS=5 \
V200_N_MAX=16 \
V200_F_MAX=8 \
V200_CALIB_WEIGHT=0.05 \
V200_TRAIN="${V200_TRAIN:-.cache/factor_graph_train.jsonl}" \
V200_VAL="${V200_VAL:-.cache/factor_graph_test.jsonl}" \
V200_GSM8K="${V200_GSM8K:-.cache/gsm8k_factor_graphs_train.jsonl}" \
V200_GSM8K_RATIO=0.5 \
BATCH="${BATCH:-8}" \
STEPS="${STEPS:-2000}" \
LR="${LR:-3e-4}" \
LOG_EVERY=10 \
PER_BREATH_EVERY=50 \
EVAL_BATCHES="${EVAL_BATCHES:-8}" \
GRAD_NORM_EVERY="${GRAD_NORM_EVERY:-100}" \
CKPT_LABEL="${CKPT_LABEL:-v200_perceiver_237_5_substrate}" \
SEED=42 \
B_REF="${B_REF:-4}" \
SMOKE_DIR="$SMOKE_DIR" \
"$PYTHON" -u scripts/v200_resmoke_237_5.py 2>&1 | tee "$SMOKE_DIR/train_237_5_console.log"

STEPS_DONE="${STEPS:-2000}"
EVAL_JSON="$SMOKE_DIR/step${STEPS_DONE}_eval_237_5.json"

echo ""
echo "=== #237.5 substrate run complete ==="
echo "Final log line:"
tail -1 "$SMOKE_DIR/train_237_5.log"
echo ""
echo "§1A.E.4 cell + E.10 reads:"
python3 -c "import json; d=json.load(open('$EVAL_JSON')); print('  cell:', d.get('e4_grid',{}).get('cell','?')); print('  e10:', json.dumps(d.get('e10_reads',{}), indent=2)[:600])" 2>/dev/null || echo "  (eval JSON not yet written)"
echo ""
echo "#238 routing (per-latent THINK entropy):"
python3 -c "import json; d=json.load(open('$EVAL_JSON')); print('  ', json.dumps(d.get('routing_238',{})))" 2>/dev/null || echo "  (eval JSON not yet written)"
echo ""
echo "C5 recalibrated:"
python3 -c "import json; d=json.load(open('$EVAL_JSON')); print('  ', json.dumps(d.get('c5_recalibrated',{}))[:400])" 2>/dev/null || echo "  (eval JSON not yet written)"
echo ""
echo "Instrumentation timeseries: $SMOKE_DIR/instrumentation_237_5.json"
