#!/usr/bin/env bash
# v200 smoke: perceiver-CORE architecture, 200-step cold-start.
#
# Base model: SmolLM2-1.7B (LlamaForCausalLM, hidden=2048, non-gated).
# To use meta-llama/Llama-3.2-1B instead, set LLAMA_WEIGHTS to the
# local path of model.safetensors after huggingface-cli login + download.
#
# Architecture summary:
#   32 latents × 2048d (QR-orthogonal init)
#   4 Llama layers (L0-L3) for THINK self-attention
#   READ cross-attention (latents←tokens, 32×24)
#   Tree codebook readout on first 16 latents
#   Per-breath weighted CE ladder + calibration BCE
#
# Memory estimate at BATCH=4, K=8, fp16 activations:
#   Llama 1B fp32 weights (4 layers):  ~0.45 GB
#   Latents (B×32×2048 fp16):          ~0.0005 GB per batch
#   Activations per breath (B×32×2048): ~0.005 GB
#   K=8 unrolled JIT: ~8× activation memory = ~0.04 GB
#   Total estimate: ~0.5 GB (fits comfortably in 24 GB)
#   At BATCH=8: ~1 GB — also safe
#
# PASS criteria (200 steps):
#   - No NaN / crash
#   - loss decreases over first 200 steps
#   - per_breath_ce ladder: B0 > B7 by step 200 (any positive delta)
#   - cell_acc > 0.01 (above random chance at 1/10^5 = 1e-5)
#
# Run:
#   cd /home/bryce/mycelium && bash scripts/v200_smoke.sh
#
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-.venv/bin/python}"

V200_TASK=1 \
V200_K_MAX="${V200_K_MAX:-8}" \
V200_N_LATENTS=32 \
V200_N_VAR_LAT=16 \
V200_N_DIGITS=5 \
V200_N_MAX=16 \
V200_F_MAX=8 \
V200_CALIB_WEIGHT=0.05 \
V200_TRAIN=.cache/factor_graph_train.jsonl \
V200_VAL=.cache/factor_graph_test.jsonl \
V200_GSM8K_TRAIN=.cache/gsm8k_factor_graphs_train.jsonl \
V200_GSM8K_RATIO=0.5 \
BATCH="${BATCH:-4}" \
STEPS="${STEPS:-200}" \
LR=1e-4 \
LOG_EVERY=10 \
PER_BREATH_CE_EVERY=50 \
EVAL_EVERY=200 \
EVAL_BATCHES=5 \
EVAL_BATCH="${BATCH:-4}" \
CKPT_EVERY=200 \
CKPT_LABEL="${CKPT_LABEL:-v200_smoke}" \
SEED=42 \
DEV="${DEV:-PCI+AMD}" \
"$PYTHON" -u scripts/v200_train.py 2>&1 | tee .cache/v200_smoke_train.log

echo ""
echo "=== smoke complete. Check .cache/v200_smoke_train.log ==="
