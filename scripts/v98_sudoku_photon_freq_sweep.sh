#!/usr/bin/env bash
# v98 Sudoku coherent-photon frequency sweep — 4 sequential 500-step smokes.
#
# All warm-start from v98_prod_final.safetensors (97.65% cell / 79% puzzle
# on easy). E and B fields both scaled by V98_PHOTON_FREQ_MULT (coherent).
# alpha=1.0 (pure photon).
#
# Predictions (from the other-Claude framing, adapted to K=20):
#   freq=0.5 → half cycle (one slow commit at breath ~10). Easy specialist?
#   freq=1.0 → full cycle (one commit-relax). Default test.
#   freq=2.0 → two cycles (commits at breaths ~5 and ~15). Hard specialist?
#   freq=3.0 → three cycles (more commits, possibly noisy).
set -euo pipefail

cd "$(dirname "$0")/.."

mkdir -p .cache/logs

CKPT="${CKPT:-.cache/sudoku_ckpts/v98_prod_final.safetensors}"
STEPS="${STEPS:-500}"

if [ ! -f "$CKPT" ]; then
    echo "ERROR: warm-start ckpt not found: $CKPT"
    exit 1
fi

for FREQ in 0.5 1.5 2.0 3.0; do
    LABEL="v98_photon_ramp_freq${FREQ//./p}"
    LOG=".cache/logs/${LABEL}.log"
    echo ""
    echo "=== Firing $LABEL (coherent freq=$FREQ + 100-step rotation ramp) ==="
    echo "    log: $LOG"

    SUDOKU_PHOTON_ENABLE=1 \
    SUDOKU_PHOTON_ALPHA=1.0 \
    SUDOKU_PHOTON_FREQ_MULT="$FREQ" \
    SUDOKU_PHOTON_ROT_RAMP_STEPS="${SUDOKU_PHOTON_ROT_RAMP_STEPS:-100}" \
    STEPS="$STEPS" \
    RESUME_FROM="$CKPT" \
    CKPT_LABEL="$LABEL" \
        bash scripts/v98_sudoku_photon_smoke.sh > "$LOG" 2>&1

    echo "=== $LABEL done. ==="
    echo "    pre-step-0 (yank diagnostic — should match v98 prod if ramp works):"
    grep -E "pre\[" "$LOG" | head -4 | sed 's/^/    /'
    echo "    final val:"
    grep -E "val\[" "$LOG" | tail -4 | sed 's/^/    /'
done

echo ""
echo "=== Photon freq sweep complete. Step-500 vals: ==="
for FREQ in 0.5 1.5 2.0 3.0; do
    LABEL="v98_photon_ramp_freq${FREQ//./p}"
    LOG=".cache/logs/${LABEL}.log"
    echo ""
    echo "  $LABEL pre-step-0 (yank size at this freq):"
    grep -E "pre\[" "$LOG" | head -4 | sed 's/^/    /'
    echo "  $LABEL step-500 val:"
    grep -E "val\[" "$LOG" | tail -4 | sed 's/^/    /'
done
