"""Per-breath convergence diagnostic for v98 Sudoku.

For each puzzle, runs K=20 breaths and tracks how many cells CHANGE
between consecutive breaths. Plots the convergence curve per difficulty.

If the model is doing approximate BP, the curve should:
  - Decrease monotonically (messages stabilize)
  - Plateau near zero (BP fixed point reached)
  - Plateau LATER for harder puzzles (longer mixing time)

Usage:
  python scripts/diag_v98_per_breath_convergence.py CKPT --n 100 --batch 8
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from tinygrad import Tensor, Device, dtypes
from tinygrad.nn.state import safe_load

from mycelium import Config
from mycelium.loader import _load_state, load_breathing
from mycelium.sudoku import (
    SUDOKU_K_MAX,
    attach_sudoku_params, sudoku_breathing_forward,
)
from mycelium.sudoku_data import SudokuLoader

# Reuse helpers from the eval script.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_v98_sudoku import cast_layers_fp32, load_ckpt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", help="Path to v98 final.safetensors")
    ap.add_argument("--test", default=".cache/sudoku_test.jsonl")
    ap.add_argument("--K", type=int, default=20)
    ap.add_argument("--n", type=int, default=100, help="Puzzles per difficulty")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--out", default=".cache/v98_per_breath_convergence.json")
    args = ap.parse_args()

    print(f"=== per-breath convergence diagnostic ===")
    print(f"ckpt={args.ckpt}  K={args.K}  n_per_diff={args.n}  batch={args.batch}")

    cfg = Config()
    sd = _load_state()
    model = load_breathing(cfg, sd=sd)
    del sd
    cast_layers_fp32(model)
    attach_sudoku_params(model, hidden=cfg.hidden, n_heads=cfg.n_heads, k_max=args.K)
    Device[Device.DEFAULT].synchronize()
    print(f"loading ckpt {args.ckpt}...")
    load_ckpt(model, args.ckpt)

    loader = SudokuLoader(args.test, batch_size=args.batch, seed=0)
    Tensor.training = False

    # Per-difficulty: list of (K-1) delta values per puzzle.
    # delta_k = number of cells whose argmax changed from breath k-1 to breath k.
    per_diff_deltas = {}
    per_diff_acc_by_breath = {}  # cells-correct trajectory
    diff_counts = {}

    t0 = time.time()
    for input_cells, gold, picks in loader.iter_eval(batch_size=args.batch):
        cell_logits_history, _ = sudoku_breathing_forward(model, input_cells, K=args.K)
        # Realize each breath's argmax: (B, 81) int.
        preds_per_breath = []
        for logits in cell_logits_history:
            pred = (logits.argmax(axis=-1) + 1).realize().numpy()
            preds_per_breath.append(pred)
        gold_np = gold.realize().numpy()
        B = len(picks)

        for b in range(B):
            diff = picks[b].get("difficulty", "easy")
            if diff_counts.get(diff, 0) >= args.n:
                continue
            diff_counts[diff] = diff_counts.get(diff, 0) + 1

            if diff not in per_diff_deltas:
                per_diff_deltas[diff] = np.zeros(args.K - 1, dtype=np.float64)
                per_diff_acc_by_breath[diff] = np.zeros(args.K, dtype=np.float64)

            # Delta: cells that changed between consecutive breaths.
            for k in range(1, args.K):
                changed = int((preds_per_breath[k][b] != preds_per_breath[k - 1][b]).sum())
                per_diff_deltas[diff][k - 1] += changed

            # Cells correct trajectory.
            for k in range(args.K):
                n_correct = int((preds_per_breath[k][b] == gold_np[b]).sum())
                per_diff_acc_by_breath[diff][k] += n_correct

        # Stop early if all difficulty bands capped.
        if all(diff_counts.get(d, 0) >= args.n for d in ["easy", "medium", "hard"]):
            break

    dt = time.time() - t0
    print(f"diagnostic complete: {sum(diff_counts.values())} puzzles in {dt:.1f}s")
    print()

    # Compute averages and print convergence curves.
    out = {"K": args.K, "n_per_diff": args.n, "by_difficulty": {}}
    for diff in ["easy", "medium", "hard"]:
        if diff not in per_diff_deltas:
            continue
        n = diff_counts[diff]
        avg_delta = (per_diff_deltas[diff] / n).tolist()
        avg_acc = (per_diff_acc_by_breath[diff] / (n * 81)).tolist()

        out["by_difficulty"][diff] = {
            "n": n,
            "avg_delta_per_breath": avg_delta,   # cells changing per breath
            "avg_cell_acc_per_breath": avg_acc,  # cumulative cell accuracy
        }

        print(f"--- {diff} (n={n}) ---")
        print(f"  Δ (cells changed between breaths):")
        for k in range(args.K - 1):
            bar = "█" * int(avg_delta[k] / 2)
            print(f"    B{k+1:02d}: {avg_delta[k]:5.2f}  {bar}")
        print(f"  Cumulative cell accuracy:")
        for k in range(args.K):
            bar = "█" * int(avg_acc[k] * 30)
            print(f"    B{k:02d}: {avg_acc[k]:.4f}  {bar}")
        print()

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
