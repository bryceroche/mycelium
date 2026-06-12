"""v98 photon yank-only diagnostic.

For each frequency in {1.5, 2.0, 3.0}, attach the photon mechanism to v98
prod ckpt with rot_scale=1.0 (full rotation, no ramp), run ONE eval pass,
report per-difficulty cell_acc / puzzle_acc.

This measures the freq-dependent rotation perturbation magnitude at warm-
start. We already have freq=0.5 from the killed sweep:
  freq=0.5 → easy 0.966/0.609, medium 0.816/0.024
v98 prod baseline:
  easy   0.977/0.79, medium 0.79/0.0, hard ?, expert ?

Backbone-agnostic information about rotation-magnitude vs freq, useful when
wiring the ramp into the factor-graph (v110-step3) photon port.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("DEV", "PCI+AMD")
os.environ.setdefault("SUDOKU_TASK", "1")
os.environ.setdefault("SUDOKU_PHOTON_ENABLE", "1")
os.environ.setdefault("SUDOKU_K_MAX", "20")

import numpy as np
from tinygrad import Tensor, Device, dtypes
from tinygrad.nn.state import safe_load

from mycelium import Config, BreathingTransformer
from mycelium.loader import _load_state, load_breathing
from mycelium.sudoku import (
    attach_sudoku_params,
    _compile_jit_sudoku_eval,
)
from mycelium.sudoku_data import SudokuLoader

K = 20
BATCH = 12
CKPT = ".cache/sudoku_ckpts/v98_prod_final.safetensors"
VAL = ".cache/sudoku_val.jsonl"
EVAL_BATCHES = 20

cfg = Config()
print("loading Pythia + v98 prod ckpt...")
sd_pythia = _load_state()
model = load_breathing(cfg, sd=sd_pythia)
del sd_pythia
attach_sudoku_params(model, hidden=cfg.hidden, n_heads=cfg.n_heads, k_max=K)

# Load v98 prod ckpt — tolerates missing photon keys (kept at init)
sd_ckpt = safe_load(CKPT)
# Manually map ckpt names → model attrs (compact ckpt format from sudoku_train.py)
from mycelium.sudoku import sudoku_state_dict
sudoku_targets = sudoku_state_dict(model)
shared_attrs = ("wv", "bv", "wo", "bo", "w_out", "b_out",
                "in_ln_g", "in_ln_b", "post_ln_g", "post_ln_b")
phase_attrs = ("wq", "bq", "wk", "bk", "w_in", "b_in")
loaded = 0
skipped = []
for name, dst in sudoku_targets.items():
    if name not in sd_ckpt:
        skipped.append(name)
        continue
    src = sd_ckpt[name].to(dst.device).realize()
    if src.dtype != dst.dtype:
        src = src.cast(dst.dtype)
    dst.assign(src).realize()
    loaded += 1
sw = model.block.shared
for a in shared_attrs:
    key = f"shared.{a}"
    if key in sd_ckpt:
        src = sd_ckpt[key].to(getattr(sw, a).device).realize()
        if src.dtype != getattr(sw, a).dtype:
            src = src.cast(getattr(sw, a).dtype)
        getattr(sw, a).assign(src).realize()
for i, layer in enumerate(model.block.layers):
    for a in phase_attrs:
        key = f"phase{i}.{a}"
        if key in sd_ckpt:
            src = sd_ckpt[key].to(getattr(layer, a).device).realize()
            if src.dtype != getattr(layer, a).dtype:
                src = src.cast(getattr(layer, a).dtype)
            getattr(layer, a).assign(src).realize()
for nm in ("ln_f.g", "ln_f.b"):
    if nm in sd_ckpt:
        dst = model.ln_f_g if nm == "ln_f.g" else model.ln_f_b
        src = sd_ckpt[nm].to(dst.device).realize()
        if src.dtype != dst.dtype:
            src = src.cast(dst.dtype)
        dst.assign(src).realize()
print(f"loaded {loaded} sudoku keys; skipped {len(skipped)} (kept at init: e.g. photon)")

val_loader = SudokuLoader(VAL, batch_size=BATCH, difficulty_filter=None,
                            curriculum=False, seed=42)


def run_eval(photon_alpha: float, photon_freq_mult: float, rot_scale: float):
    """Compile eval JIT for given config, set rot_scale, run val sweep."""
    # Set rot_scale
    model.sudoku_photon_rot_scale.assign(
        Tensor(np.array(rot_scale, dtype=np.float32), dtype=dtypes.float)
    ).realize()
    eval_fn = _compile_jit_sudoku_eval(
        model, K=K, B=BATCH,
        photon_alpha=photon_alpha, photon_freq_mult=photon_freq_mult,
    )
    Tensor.training = False
    agg = {}
    n_batches = 0
    for input_cells, gold, picks in val_loader.iter_eval(batch_size=BATCH):
        eq, _ca, _pa = eval_fn(input_cells, gold)
        eq_np = eq.numpy()
        for b, rec in enumerate(picks):
            diff = rec.get("difficulty", "easy")
            if diff not in agg:
                agg[diff] = dict(cell_eq=0.0, n_cells=0, puzzle_eq=0, n_puzzles=0)
            agg[diff]["cell_eq"]   += float(eq_np[b].sum())
            agg[diff]["n_cells"]   += 81
            agg[diff]["puzzle_eq"] += int(eq_np[b].prod())
            agg[diff]["n_puzzles"] += 1
        n_batches += 1
        if n_batches >= EVAL_BATCHES:
            break
    out = {}
    for d, v in agg.items():
        if v["n_puzzles"] == 0:
            continue
        out[d] = dict(
            cell_acc=v["cell_eq"] / v["n_cells"],
            puzzle_acc=v["puzzle_eq"] / v["n_puzzles"],
            n_puzzles=v["n_puzzles"],
        )
    return out


# Baseline: no photon (alpha=0)
print()
print("=" * 60)
print("BASELINE: no photon (alpha=0)")
print("=" * 60)
t0 = time.time()
results_base = run_eval(photon_alpha=0.0, photon_freq_mult=1.0, rot_scale=0.0)
for d in ["easy", "medium", "hard", "expert"]:
    if d not in results_base:
        continue
    v = results_base[d]
    print(f"  baseline[{d:7s}]: cell_acc={v['cell_acc']:.3f} "
          f"puzzle_acc={v['puzzle_acc']:.3f} (n={v['n_puzzles']})")
print(f"  ({time.time() - t0:.1f}s)")


# Yank sweep
yank_results = {0.5: {"easy": dict(cell_acc=0.966, puzzle_acc=0.609),
                      "medium": dict(cell_acc=0.816, puzzle_acc=0.024)}}
for FREQ in [1.5, 2.0, 3.0]:
    print()
    print("=" * 60)
    print(f"YANK at freq={FREQ} (rot_scale=1.0 full rotation, no ramp)")
    print("=" * 60)
    t0 = time.time()
    results = run_eval(photon_alpha=1.0, photon_freq_mult=FREQ, rot_scale=1.0)
    yank_results[FREQ] = results
    for d in ["easy", "medium", "hard", "expert"]:
        if d not in results:
            continue
        v = results[d]
        print(f"  yank[{d:7s}] freq={FREQ}: cell_acc={v['cell_acc']:.3f} "
              f"puzzle_acc={v['puzzle_acc']:.3f} (n={v['n_puzzles']})")
    print(f"  ({time.time() - t0:.1f}s)")


# Summary
print()
print("=" * 60)
print("SUMMARY: rotation perturbation magnitude vs frequency")
print("=" * 60)
print(f"  v98 prod baseline: easy 0.977/0.79, medium 0.79/0.0 (from CLAUDE.md)")
print(f"  freq=0.5 (cached): easy 0.966/0.609, medium 0.816/0.024")
for FREQ in [1.5, 2.0, 3.0]:
    if FREQ in yank_results and "easy" in yank_results[FREQ]:
        e = yank_results[FREQ]["easy"]
        m = yank_results[FREQ].get("medium", dict(cell_acc=-1, puzzle_acc=-1))
        print(f"  freq={FREQ}:        easy {e['cell_acc']:.3f}/{e['puzzle_acc']:.3f}, "
              f"medium {m['cell_acc']:.3f}/{m['puzzle_acc']:.3f}")
print()
print("If yank monotonically increases with freq → rotation-magnitude confound real.")
print("If flat across freq → photon damage isn't rotation-magnitude; need diff cause.")
