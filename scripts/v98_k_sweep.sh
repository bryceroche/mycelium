#!/bin/bash
# K-sweep eval on v98 final ckpt: K ∈ {1, 3, 5, 8, 12, 15, 18, 20}
# n=200 per difficulty (easy, medium, hard).
set -e
CKPT="${CKPT:-/home/bryce/mycelium/.cache/sudoku_ckpts/v98_prod_final.safetensors}"
N="${N:-200}"
BATCH="${BATCH:-8}"
OUT_DIR="${OUT_DIR:-/home/bryce/mycelium/.cache/v98_ksweep}"
TEST="${TEST:-.cache/sudoku_test.jsonl}"

mkdir -p "$OUT_DIR"
echo "=== v98 K-sweep ==="
echo "ckpt=$CKPT  test=$TEST  n=$N  batch=$BATCH"
echo "out=$OUT_DIR"
echo

source .venv/bin/activate
export DEV='PCI+AMD'
export SUDOKU_TASK=1

for K in 1 3 5 8 12 15 18 20; do
    echo "[K=$K] starting at $(date +%H:%M:%S)..."
    SUDOKU_K_MAX=20 python scripts/eval_v98_sudoku.py "$CKPT" \
        --test "$TEST" --K "$K" --k_alloc 20 --n "$N" --batch "$BATCH" --show 0 \
        2>&1 | tee "$OUT_DIR/K${K}.log"
    echo
done

echo "=== sweep complete ==="
