#!/usr/bin/env bash
# v110-step3 chain — cont1 through cont9, each 1000 steps.
#
# Total: 500 (smoke) + 9 × 1000 (conts) = 9500 effective steps.
# Warm-start from v110_step3_smoke_final.safetensors.
#
# Each cont saves step250, step500, step750, step1000, final ckpts.
# Effective step for cont_i = 500 + i*1000.
set -euo pipefail

cd "$(dirname "$0")/.."

LOG=/tmp/v110_step3_chain.log
CKPT_DIR=.cache/fg_v110_step3_ckpts

# Starting point: smoke final (effective step 500)
CKPT_PREV="${CKPT_DIR}/v110_step3_smoke_final.safetensors"

if [ ! -f "$CKPT_PREV" ]; then
    echo "ERROR: starting ckpt not found: $CKPT_PREV" >&2
    exit 1
fi

echo "=== v110-step3 chain to step 9500 starting ===" | tee -a "$LOG"
echo "starting ckpt: $CKPT_PREV" | tee -a "$LOG"
date | tee -a "$LOG"
echo "" | tee -a "$LOG"

for i in 1 2 3 4 5 6 7 8 9; do
    if [ "$i" = "1" ]; then
        LABEL="v110_step3_cont"
    else
        LABEL="v110_step3_cont${i}"
    fi

    EFF_END=$((500 + i*1000))
    echo "=== launching cont$i (effective step $EFF_END) ===" | tee -a "$LOG"
    date | tee -a "$LOG"

    STEPS=1000 CKPT_LABEL="$LABEL" CKPT="$CKPT_PREV" \
        bash scripts/v110_step3_factor_graph_smoke.sh >> "$LOG" 2>&1

    CKPT_PREV="${CKPT_DIR}/${LABEL}_final.safetensors"

    if [ ! -f "$CKPT_PREV" ]; then
        echo "ERROR: cont$i final ckpt missing: $CKPT_PREV" | tee -a "$LOG" >&2
        exit 1
    fi

    echo "=== cont$i (effective step $EFF_END) done ===" | tee -a "$LOG"
    date | tee -a "$LOG"
    echo "" | tee -a "$LOG"
done

echo "=== v110-step3 9500-step chain done ===" | tee -a "$LOG"
date | tee -a "$LOG"
