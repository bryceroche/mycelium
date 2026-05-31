#!/usr/bin/env bash
# v106 MCTS-on-digit-codebook — smoke test
#
# Algorithm: AlphaZero-pattern MCTS with v105 breathing transformer as evaluator.
#   - Phase 0: BP-only (fast path)
#   - Phase 1: MCTS rollouts for hard problems (calib < threshold)
#
# Acceptance criteria:
#   1. No crash — MCTS engine compiles and runs
#   2. MCTS triggers correctly (calib < 0.7 → MCTS; calib > 0.7 → BP-only)
#   3. Rollout count reported per problem
#   4. Wallclock < 30s/problem on hard problems
#   5. Summary table prints BP vs BP+MCTS accuracy
#
# Uses the v105 smoke ckpt if available:
#   .cache/fg_v105_ckpts/v105_smoke_final.safetensors
#
# Modify V106_N_ROLLOUTS to reduce smoke-time on CPU.
#
# Usage:
#   bash scripts/v106_mcts_smoke.sh              # full smoke (50 rollouts)
#   V106_N_ROLLOUTS=5 bash scripts/v106_mcts_smoke.sh   # quick verify (5 rollouts)

set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-.venv/bin/python}"

CKPT="${CKPT:-.cache/fg_v105_ckpts/v105_smoke_final.safetensors}"

if [ ! -f "$CKPT" ]; then
  echo "[v106 smoke] WARNING: v105 ckpt not found at $CKPT"
  echo "[v106 smoke] Will run with random-init model (algorithm validation only)."
  echo "[v106 smoke] Set CKPT=/path/to/v105.safetensors to use a trained model."
fi

DEV='PCI+AMD' \
V105_TASK=1 \
V105_K_MAX=8 \
V105_N_DIGITS=5 \
V105_N_MAX=16 \
V105_F_MAX=8 \
V106_N_ROLLOUTS="${V106_N_ROLLOUTS:-50}" \
V106_CALIB_THRESHOLD="${V106_CALIB_THRESHOLD:-0.7}" \
V106_K_BREATHS="${V106_K_BREATHS:-8}" \
"$PYTHON" -u scripts/v106_mcts_eval.py \
  "$CKPT" \
  --gsm8k .cache/gsm8k_factor_graphs_200test.jsonl \
  --n_problems 50 \
  --n_rollouts "${V106_N_ROLLOUTS:-50}" \
  --calib_threshold "${V106_CALIB_THRESHOLD:-0.7}" \
  --K "${V106_K_BREATHS:-8}" \
  --seed 42
