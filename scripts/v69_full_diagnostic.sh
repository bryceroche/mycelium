#!/bin/bash
# v69 full diagnostic battery.
# Fires all four diagnostics sequentially once training finishes.
#
# Steps:
#   1a. Full GSM8K eval (300 examples, per-K breakdown)
#   1b. Rename diagnostic (entity grounding, 20 examples)
#   1c. Segment-count dump + failure analysis (100 examples)
#   1d. Codebook usage histogram (100 examples)
#
# Usage:
#   CKPT=.cache/gsm8k_steps_ckpts/v69_collapse_step3000.safetensors \
#     bash scripts/v69_full_diagnostic.sh
#
# Each sub-script also reads CKPT from the env. All logs are tee'd to /tmp/v69_*.

set -e
cd "$(dirname "$0")/.."

CKPT="${CKPT:-.cache/gsm8k_steps_ckpts/v69_collapse_step3000.safetensors}"
export CKPT

if [ ! -f "$CKPT" ]; then
    echo "ERROR: ckpt not found: $CKPT"
    echo "Set CKPT= to the correct path."
    exit 1
fi

echo ""
echo "============================================================"
echo " v69 full diagnostic battery"
echo " ckpt: $CKPT"
echo " start: $(date)"
echo "============================================================"
echo ""

# --- 1a. Full GSM8K eval (300 examples) ---
echo "--- [1a] Full GSM8K eval (300 examples) ---"
bash scripts/eval_v69_step3000.sh
echo ""
echo "[1a] done. Log: /tmp/v69_eval.log"
echo ""

# --- 1b. Rename diagnostic ---
echo "--- [1b] Rename diagnostic (entity grounding, 20 examples) ---"
bash scripts/run_rename_diag_v69.sh
echo ""
echo "[1b] done. Log: /tmp/v69_rename_diag.log  JSONL: /tmp/v69_rename_diag.jsonl"
echo ""

# --- 1c. Segment-count dump + failure analysis ---
echo "--- [1c] Segment-count dump + failure analysis (100 examples) ---"
bash scripts/diagnose_v69.sh
echo ""
echo "[1c] done. Log: /tmp/v69_diag.log  JSONL: /tmp/v69_diag.jsonl"
echo ""

# --- 1d. Codebook usage histogram ---
echo "--- [1d] Codebook usage histogram (100 examples) ---"
HIST_OUT=/tmp/v69_codebook_hist.txt \
  GSM8K_PATH=.cache/gsm8k_steps_v1_test.jsonl \
  NUM_EVAL=100 \
  TRAIN_LOOPS=4 \
  /home/bryce/mycelium/.venv/bin/python scripts/codebook_usage_histogram.py 2>&1 | tee /tmp/v69_codebook_hist.log
echo ""
echo "[1d] done. Log: /tmp/v69_codebook_hist.log"
echo "     Histogram: /tmp/v69_codebook_hist.txt"
echo "     Summary:   /tmp/v69_codebook_hist_summary.json"
echo ""

echo "============================================================"
echo " v69 diagnostic battery complete"
echo " end: $(date)"
echo "============================================================"
echo ""
echo "Summary of output files:"
echo "  /tmp/v69_eval.log             — full eval log"
echo "  /tmp/v69_rename_diag.log      — rename diagnostic log"
echo "  /tmp/v69_rename_diag.jsonl    — per-example rename results"
echo "  /tmp/v69_diag.log             — segment-count diagnostic log"
echo "  /tmp/v69_diag.jsonl           — per-example segment dump"
echo "  /tmp/v69_codebook_hist.log    — codebook histogram log"
echo "  /tmp/v69_codebook_hist.txt    — formatted histogram report"
echo "  /tmp/v69_codebook_hist_summary.json  — JSON summary"
