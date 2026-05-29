"""Standalone v98 sudoku evaluation: per-difficulty accuracy + constraint
violation stats + variable-K efficiency check.

Usage:
  python scripts/eval_v98_sudoku.py CKPT_PATH [--test PATH] [--K N] [--n N]
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
    attach_sudoku_params, sudoku_breathing_forward, sudoku_constraint_energy,
)
from mycelium.sudoku_data import SudokuLoader


def cast_layers_fp32(model):
    def _cast(obj, attr):
        t = getattr(obj, attr)
        if t.dtype == dtypes.half:
            setattr(obj, attr, t.cast(dtypes.float).contiguous().realize())
    _cast(model.embed, "weight")
    sw = model.block.shared
    for a in ("wv", "bv", "wo", "bo", "w_out", "b_out"):
        _cast(sw, a)
    for layer in model.block.layers:
        for a in ("wq", "bq", "wk", "bk", "w_in", "b_in"):
            _cast(layer, a)


def model_state_dict_sudoku(model) -> dict:
    sd = {"ln_f.g": model.ln_f_g, "ln_f.b": model.ln_f_b}
    sw = model.block.shared
    for a in ("wv", "bv", "wo", "bo", "w_out", "b_out",
              "in_ln_g", "in_ln_b", "post_ln_g", "post_ln_b"):
        sd[f"shared.{a}"] = getattr(sw, a)
    for i, layer in enumerate(model.block.layers):
        for a in ("wq", "bq", "wk", "bk", "w_in", "b_in"):
            sd[f"phase{i}.{a}"] = getattr(layer, a)
    sd["sudoku.state_embed"] = model.sudoku_state_embed
    sd["sudoku.position_embed"] = model.sudoku_position_embed
    sd["sudoku.digit_codebook"] = model.sudoku_digit_codebook
    sd["sudoku.calib_head_w"] = model.sudoku_calib_head_w
    sd["sudoku.calib_head_b"] = model.sudoku_calib_head_b
    sd["sudoku.breath_embed"] = model.sudoku_breath_embed
    sd["sudoku.delta_gate"] = model.sudoku_delta_gate
    return sd


def load_ckpt(model, path: str):
    sd = safe_load(path)
    targets = model_state_dict_sudoku(model)
    for name, dst in targets.items():
        if name not in sd:
            continue
        src = sd[name].to(dst.device).realize()
        if src.shape != dst.shape:
            src = src.reshape(dst.shape)
        if src.dtype != dst.dtype:
            src = src.cast(dst.dtype)
        dst.assign(src).realize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", help="Path to a .safetensors v98 sudoku ckpt.")
    ap.add_argument("--test", default=".cache/sudoku_test.jsonl")
    ap.add_argument("--K", type=int, default=int(os.environ.get("SUDOKU_K_MAX", SUDOKU_K_MAX)))
    ap.add_argument("--k_alloc", type=int, default=None,
                    help="K_max used for model allocation (must match ckpt). Defaults to --K.")
    ap.add_argument("--n", type=int, default=500, help="Number of test puzzles per difficulty (cap).")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--show", type=int, default=2, help="Show N sample puzzles (in,gold,pred).")
    args = ap.parse_args()

    k_alloc = args.k_alloc if args.k_alloc is not None else args.K
    print(f"=== v98 sudoku eval ===")
    print(f"ckpt={args.ckpt}  test={args.test}  K={args.K}  k_alloc={k_alloc}  n_per_diff={args.n}  batch={args.batch}")

    cfg = Config()
    sd = _load_state()
    from mycelium import BreathingTransformer
    model = load_breathing(cfg, sd=sd)
    del sd
    cast_layers_fp32(model)
    attach_sudoku_params(model, hidden=cfg.hidden, n_heads=cfg.n_heads, k_max=k_alloc)
    Device[Device.DEFAULT].synchronize()
    print(f"loading ckpt {args.ckpt}...")
    load_ckpt(model, args.ckpt)

    loader = SudokuLoader(args.test, batch_size=args.batch, seed=0)
    Tensor.training = False

    # Per-difficulty stats
    agg = {}
    sample_shown = 0

    t0 = time.time()
    n_seen = 0
    for input_cells, gold, picks in loader.iter_eval(batch_size=args.batch):
        cell_logits_history, calib_history = sudoku_breathing_forward(model, input_cells, K=args.K)
        final_logits = cell_logits_history[-1]
        pred = final_logits.argmax(axis=-1) + 1
        eq = (pred == gold).cast(dtypes.float)
        eq_np = eq.realize().numpy()
        pred_np = pred.realize().numpy()
        input_np = input_cells.realize().numpy()
        gold_np = gold.realize().numpy()

        # Final-breath probs → constraint violation
        final_probs = final_logits.softmax(axis=-1)
        violations = sudoku_constraint_energy(final_probs).realize().numpy()

        for b, rec in enumerate(picks):
            diff = rec.get("difficulty", "easy")
            if diff not in agg:
                agg[diff] = {"cell_eq": 0, "n_cells": 0, "puzzle_eq": 0,
                             "n_puzzles": 0, "viol_sum": 0.0, "n_givens_sum": 0,
                             "calib_final_sum": 0.0}
            # Cap each difficulty's count at args.n.
            if agg[diff]["n_puzzles"] >= args.n:
                continue
            agg[diff]["cell_eq"] += int(eq_np[b].sum())
            agg[diff]["n_cells"] += 81
            agg[diff]["puzzle_eq"] += int(eq_np[b].prod())
            agg[diff]["n_puzzles"] += 1
            agg[diff]["viol_sum"] += float(violations[b])
            agg[diff]["n_givens_sum"] += int(rec.get("n_givens", 0))
            agg[diff]["calib_final_sum"] += float(calib_history[-1].realize().numpy()[b])

            if sample_shown < args.show:
                print()
                print(f"--- sample ({diff}, n_givens={rec.get('n_givens')}) ---")
                print(f"INPUT (0=unknown):  {input_np[b].tolist()}")
                print(f"PREDICTION:          {pred_np[b].tolist()}")
                print(f"GOLD:                {gold_np[b].tolist()}")
                n_correct = int(eq_np[b].sum())
                print(f"Cells correct: {n_correct}/81  |  Puzzle correct: {bool(eq_np[b].prod())}")
                print(f"Constraint energy (row+col+box L2 to 1.0): {violations[b]:.3f}")
                sample_shown += 1

        n_seen += len(picks)
        # Stop early if all difficulty bands are capped.
        if all(agg.get(d, {}).get("n_puzzles", 0) >= args.n for d in agg) \
                and len(agg) >= 1:
            # Check we've actually filled SOME difficulty
            pass

    dt = time.time() - t0
    print()
    print(f"Eval complete: {n_seen} puzzles in {dt:.1f}s ({n_seen/dt:.1f} puzzles/s)")
    print()

    out_rows = []
    for d in ["easy", "medium", "hard", "expert"]:
        if d not in agg:
            continue
        v = agg[d]
        n = v["n_puzzles"]
        if n == 0:
            continue
        out = {
            "difficulty": d,
            "n_puzzles": n,
            "cell_acc": v["cell_eq"] / v["n_cells"],
            "puzzle_acc": v["puzzle_eq"] / n,
            "avg_violation": v["viol_sum"] / n,
            "avg_n_givens": v["n_givens_sum"] / n,
            "avg_calib_final": v["calib_final_sum"] / n,
        }
        out_rows.append(out)
        print(f"[{d:7s}] cell_acc={out['cell_acc']:.4f} "
              f"puzzle_acc={out['puzzle_acc']:.4f} "
              f"avg_viol={out['avg_violation']:.3f} "
              f"avg_calib_final={out['avg_calib_final']:.3f} "
              f"n={n} avg_G={out['avg_n_givens']:.1f}")

    # Overall
    if out_rows:
        n_total = sum(r["n_puzzles"] for r in out_rows)
        cell_acc_overall = sum(r["cell_acc"] * r["n_puzzles"] for r in out_rows) / n_total
        puzzle_acc_overall = sum(r["puzzle_acc"] * r["n_puzzles"] for r in out_rows) / n_total
        print()
        print(f"OVERALL: cell_acc={cell_acc_overall:.4f}  puzzle_acc={puzzle_acc_overall:.4f}  n={n_total}")


if __name__ == "__main__":
    main()
