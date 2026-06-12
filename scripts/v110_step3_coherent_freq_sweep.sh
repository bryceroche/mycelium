#!/usr/bin/env bash
# v110-step3 COHERENT photon frequency sweep — 4 sequential 500-step smokes.
#
# Genuine photon-frequency variation: BOTH knobs scaled together.
#   V110_STEP3_PHASE_SCALE = freq  (E-field, Q rotation per breath)
#   V110_PHOTON_FREQ_MULT  = freq  (B-field, waist gate sin² amplitude)
#
# At freq=1.0 this matches the current prod (v110-step3 sin2_pi alpha=0.5
# rotation phase = k·π/K). At freq=N the photon (E+B coupled) oscillates
# N× per K breaths.
#
# All warm-start from v110_step3_cont8_step1000. Profile sin2_pi, alpha=1.0
# (pure photon, no binary blend) for cleanest comparison.
#
# Pythia-410M L0-L3 backbone (the v110-step3 prod chain).
#
# Reference baselines:
#   prod (freq=1.0 alpha=0.5):     easy 0.572 / med 0.491 / hard 0.376
#   incoherent freq=4 alpha=1.0:   easy 0.547 / med 0.493 / hard 0.375
#   incoherent freq=0.5 alpha=1.0: easy 0.578 / med 0.491 / hard 0.386
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
    LABEL="v110_step3_coh_freq${FREQ//./p}"
    LOG=".cache/logs/${LABEL}.log"
    echo ""
    echo "=== Firing $LABEL (coherent freq=$FREQ, E and B both scaled) ==="
    echo "    log: $LOG"

    V110_PHOTON_FREQ_MULT="$FREQ" \
    V110_STEP3_PHASE_SCALE="$FREQ" \
    V110_STEP3_PHOTON_ALPHA=1.0 \
    STEPS="$STEPS" \
    CKPT="$CKPT" \
    CKPT_LABEL="$LABEL" \
        bash scripts/v110_step3_factor_graph_smoke.sh > "$LOG" 2>&1

    echo "=== $LABEL done. Final val: ==="
    grep -E "val\[" "$LOG" | tail -3
done

echo ""
echo "=== Coherent freq sweep complete. Summary of all step-500 vals: ==="
for FREQ in 0.5 1.5 2.0 3.0; do
    LABEL="v110_step3_coh_freq${FREQ//./p}"
    LOG=".cache/logs/${LABEL}.log"
    echo ""
    echo "  $LABEL:"
    grep -E "val\[" "$LOG" | tail -3 | sed 's/^/    /'
done
