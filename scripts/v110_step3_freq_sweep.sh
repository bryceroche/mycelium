#!/usr/bin/env bash
# v110-step3 photon frequency sweep — 4 sequential 500-step smokes.
#
# All warm-start from v110_step3_cont8_step1000 (the current strongest baseline).
# Profile: sin2_pi, alpha=1.0 (pure photon), varying V110_PHOTON_FREQ_MULT.
# Each smoke ~13 min after JIT compile; ~52 min total.
#
# Compares pure-photon gate at four frequencies. Already-tested:
#   freq=1.0 alpha=0.5 — current prod (baseline)
#   freq=4.0 alpha=1.0 — antiphase binary (refuted: hard 0.375 = baseline)
#
# New runs in this sweep:
#   freq=0.5 alpha=1.0 — quarter cycle (very slow oscillation)
#   freq=1.5 alpha=1.0 — 3/4 cycle (asymmetric, peaks past mid-K)
#   freq=2.0 alpha=1.0 — full cycle (two compress-relax phases per K)
#   freq=3.0 alpha=1.0 — 1.5 cycles (three lobes)
set -euo pipefail

cd "$(dirname "$0")/.."

mkdir -p .cache/logs

CKPT="${CKPT:-.cache/fg_v110_step3_ckpts/v110_step3_cont8_step1000.safetensors}"
STEPS="${STEPS:-500}"

if [ ! -f "$CKPT" ]; then
    echo "ERROR: warm-start ckpt not found: $CKPT"
    exit 1
fi

for FREQ in 0.5 1.5 2.0 3.0; do
    LABEL="v110_step3_freq${FREQ//./p}_alpha1"
    LOG=".cache/logs/${LABEL}.log"
    echo ""
    echo "=== Firing $LABEL (freq=$FREQ alpha=1.0) ==="
    echo "    log: $LOG"

    V110_PHOTON_FREQ_MULT="$FREQ" \
    V110_STEP3_PHOTON_ALPHA=1.0 \
    STEPS="$STEPS" \
    CKPT="$CKPT" \
    CKPT_LABEL="$LABEL" \
        bash scripts/v110_step3_factor_graph_smoke.sh > "$LOG" 2>&1

    echo "=== $LABEL done. Final val: ==="
    grep -E "val\[" "$LOG" | tail -3
done

echo ""
echo "=== Freq sweep complete. Summary of all step-500 vals: ==="
for FREQ in 0.5 1.5 2.0 3.0; do
    LABEL="v110_step3_freq${FREQ//./p}_alpha1"
    LOG=".cache/logs/${LABEL}.log"
    echo ""
    echo "  $LABEL:"
    grep -E "val\[" "$LOG" | tail -3 | sed 's/^/    /'
done
