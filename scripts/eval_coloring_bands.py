"""eval_coloring_bands.py — Characterize graph-coloring generality across difficulty.

Path 3 (consolidation): loads the trained fg_coloring_k16 checkpoint and runs a full
test-split evaluation bucketed by density BAND, DEDUCTION_DEPTH, and graph SIZE.

MODEL CONSTRUCTION mirrors factor_graph_train.py EXACTLY:
  - Pythia-410M L0-L3 backbone (same cast_layers_fp32 path)
  - attach_factor_graph_params with coloring spec (s_max=49, n_values=3,
    n_factor_types=1, n_heads=16, k_max=16, has_factor_inlet=False)
  - load_ckpt with the same model_state_dict_fg key set

TEST SPLIT reproduced identically to the training run:
  - GraphColoringLoader(n_instances=8000, s_max=49, k_colors=3, batch_size=8,
    seed=42) — same default test_frac/bands/regular_frac — so loader.test_records
    is byte-identical to the trained run's held-out set.

GPU-FREE BUILD: ast.parse at the bottom verifies this file parses cleanly on CPU.
Run with DEV=AMD for the actual GPU eval:
  DEV=AMD FG_TASK=coloring FG_N_VALUES=3 FG_N_INSTANCES=8000 \\
      .venv/bin/python3 scripts/eval_coloring_bands.py
"""
import ast
import os
import sys

# --------------------------------------------------------------------------
# GPU-free build gate (CPU import — no tinygrad Device ops until __main__)
# --------------------------------------------------------------------------
_THIS_FILE = os.path.abspath(__file__)

sys.path.insert(0, os.path.dirname(os.path.dirname(_THIS_FILE)))

import numpy as np

# --- parse gate (always runs, even on CPU) ----------------------------------

def _ast_parse_ok() -> bool:
    with open(_THIS_FILE) as f:
        src = f.read()
    try:
        ast.parse(src)
        return True
    except SyntaxError as e:
        print(f"[ast.parse] FAILED: {e}", flush=True)
        return False


# --------------------------------------------------------------------------
# Mirror of factor_graph_train.cast_layers_fp32
# --------------------------------------------------------------------------

def cast_layers_fp32(model):
    from tinygrad import dtypes
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


# --------------------------------------------------------------------------
# Mirror of factor_graph_train.model_state_dict_fg / load_ckpt
# --------------------------------------------------------------------------

_FG_PARAM_NAMES = [
    "fg_state_embed", "fg_position_embed", "fg_value_codebook",
    "fg_calib_head_w", "fg_calib_head_b", "fg_breath_embed", "fg_delta_gate",
]


def model_state_dict_fg(model) -> dict:
    sd = {"ln_f.g": model.ln_f_g, "ln_f.b": model.ln_f_b}
    sw = model.block.shared
    for a in ("wv", "bv", "wo", "bo", "w_out", "b_out",
              "in_ln_g", "in_ln_b", "post_ln_g", "post_ln_b"):
        sd[f"shared.{a}"] = getattr(sw, a)
    for i, layer in enumerate(model.block.layers):
        for a in ("wq", "bq", "wk", "bk", "w_in", "b_in"):
            sd[f"phase{i}.{a}"] = getattr(layer, a)
    for nm in _FG_PARAM_NAMES:
        sd[nm] = getattr(model, nm)
    # No kenken inlet params for the coloring task.
    return sd


def load_ckpt(model, path: str):
    from tinygrad.nn.state import safe_load
    sd = safe_load(path)
    targets = model_state_dict_fg(model)
    missing = []
    for name, dst in targets.items():
        if name not in sd:
            missing.append(name)
            continue
        src = sd[name].to(dst.device).realize()
        if src.shape != dst.shape:
            try:
                src = src.reshape(dst.shape)
            except Exception:
                missing.append(f"{name} (shape mismatch)")
                continue
        if src.dtype != dst.dtype:
            src = src.cast(dst.dtype)
        dst.assign(src).realize()
    if missing:
        print(f"  ckpt missing {len(missing)} keys: "
              f"{missing[:5]}{'...' if len(missing) > 5 else ''}", flush=True)
    else:
        print(f"  ckpt loaded cleanly ({len(targets)} keys).", flush=True)


# --------------------------------------------------------------------------
# Bucketing helpers
# --------------------------------------------------------------------------

def _n_bucket(n: int) -> str:
    """Coarse graph-size bucket."""
    if n <= 10:
        return "n≤10"
    if n <= 20:
        return "n11-20"
    if n <= 35:
        return "n21-35"
    return "n36+"


def _print_table(title: str, rows: dict, col_order=None) -> None:
    """Print a flat {key: {cell_eq_sum, n_cells, puzzle_eq_sum, n_puzzles}} table."""
    print(f"\n{'='*60}", flush=True)
    print(f"  {title}", flush=True)
    print(f"{'='*60}", flush=True)
    keys = col_order if col_order is not None else sorted(rows.keys())
    hdr = f"  {'bucket':<14}  {'cell_acc':>8}  {'puzzle_acc':>10}  {'count':>6}"
    print(hdr, flush=True)
    print("  " + "-" * (len(hdr) - 2), flush=True)
    total_cells = total_ceq = total_puz = total_peq = 0
    for k in keys:
        r = rows.get(k)
        if r is None:
            continue
        nc = r["n_cells"]
        np_ = r["n_puzzles"]
        if nc == 0:
            continue
        ca = r["cell_eq_sum"] / nc
        pa = r["puzzle_eq_sum"] / max(np_, 1)
        print(f"  {str(k):<14}  {ca:8.3f}  {pa:10.3f}  {np_:6d}", flush=True)
        total_cells += nc
        total_ceq += r["cell_eq_sum"]
        total_puz += np_
        total_peq += r["puzzle_eq_sum"]
    if total_cells > 0:
        oca = total_ceq / total_cells
        opa = total_peq / max(total_puz, 1)
        print("  " + "-" * (len(hdr) - 2), flush=True)
        print(f"  {'OVERALL':<14}  {oca:8.3f}  {opa:10.3f}  {total_puz:6d}", flush=True)
    print("", flush=True)


def _print_ladder(pb_ce: list[float], K: int) -> None:
    print(f"\n{'='*60}", flush=True)
    print("  Per-breath CE ladder (mean over test set; B0..B15)", flush=True)
    print(f"{'='*60}", flush=True)
    n = len(pb_ce)
    # Print in rows of 8
    for row_start in range(0, n, 8):
        chunk = pb_ce[row_start:row_start + 8]
        labels = "  ".join(f"B{row_start+i}={v:.3f}" for i, v in enumerate(chunk))
        print(f"  {labels}", flush=True)
    if n >= 2:
        drop = pb_ce[0] - pb_ce[-1]
        print(f"\n  CE drop B0->B{n-1}: {pb_ce[0]:.3f} -> {pb_ce[-1]:.3f} "
              f"(Δ={drop:.3f})", flush=True)
    print("", flush=True)


# --------------------------------------------------------------------------
# Main evaluation
# --------------------------------------------------------------------------

def main():
    import gc
    from tinygrad import Tensor, Device
    from tinygrad.helpers import getenv

    CKPT = getenv("FG_CKPT",
                  ".cache/fg_ckpts/fg_coloring_k16/fg_coloring_k16_final.safetensors")
    K = int(getenv("FG_K_MAX", getenv("K", "16")))
    EVAL_BATCH = int(getenv("EVAL_BATCH", getenv("BATCH", "8")))
    SEED = int(getenv("SEED", "42"))
    N_INSTANCES = int(getenv("FG_N_INSTANCES", "8000"))
    S_MAX = int(getenv("FG_S_MAX", "49"))
    N_VALUES = int(getenv("FG_N_VALUES", "3"))

    print("=== eval_coloring_bands.py — graph-coloring generality sweep ===",
          flush=True)
    print(f"device={Device.DEFAULT}  ckpt={CKPT}", flush=True)
    print(f"K={K}  EVAL_BATCH={EVAL_BATCH}  seed={SEED}  "
          f"n_instances={N_INSTANCES}  s_max={S_MAX}  k={N_VALUES}", flush=True)
    print(flush=True)

    # ---- build model (mirrors factor_graph_train.main) ---------------------
    from mycelium import Config, BreathingTransformer
    from mycelium.loader import _load_state, load_breathing
    from mycelium.factor_graph_engine import (
        FactorGraphSpec,
        attach_factor_graph_params,
        factor_breathing_forward,
    )
    from mycelium.graph_coloring_data import GraphColoringLoader

    spec = FactorGraphSpec(
        s_max=S_MAX,
        n_values=N_VALUES,
        n_factor_types=1,
        n_heads=16,
        k_max=K,
        has_factor_inlet=False,
    )

    print("loading Pythia-410M -> BreathingTransformer...", flush=True)
    cfg = Config()
    sd = _load_state()
    model = load_breathing(cfg, sd=sd)
    del sd
    gc.collect()
    cast_layers_fp32(model)

    attach_factor_graph_params(model, hidden=cfg.hidden, spec=spec)
    Device[Device.DEFAULT].synchronize()

    print(f"loading checkpoint: {CKPT}", flush=True)
    load_ckpt(model, CKPT)

    # ---- reconstruct IDENTICAL test split ----------------------------------
    # GraphColoringLoader(n_instances=8000, s_max=49, k_colors=3, seed=42)
    # uses DEFAULT test_frac=0.15, bands=BANDS, regular_frac=0.4
    # => exactly mirrors _build_coloring_task in factor_graph_train.py.
    print(f"\nreconstructing test split "
          f"(n_instances={N_INSTANCES}, seed={SEED})...", flush=True)
    loader = GraphColoringLoader(
        n_instances=N_INSTANCES,
        s_max=S_MAX,
        k_colors=N_VALUES,
        batch_size=EVAL_BATCH,
        seed=SEED,
    )
    n_test = len(loader.test_records)
    print(f"  test set: {n_test} instances", flush=True)

    # ---- accumulators (per bucket) -----------------------------------------
    BAND_ORDER = ["d10", "d15", "d20", "d25"]
    DEPTH_ORDER = [0, 1, 2, 3, 4]

    def _mk() -> dict:
        return {"cell_eq_sum": 0.0, "n_cells": 0,
                "puzzle_eq_sum": 0, "n_puzzles": 0}

    band_acc:  dict[str, dict] = {b: _mk() for b in BAND_ORDER}
    depth_acc: dict[int, dict] = {d: _mk() for d in DEPTH_ORDER}
    size_acc:  dict[str, dict] = {}
    overall   = _mk()

    # Per-breath CE (accumulated over all batches)
    pb_ce_sum: list[float] = [0.0] * K
    pb_ce_n_batches = 0

    Tensor.training = False

    print("\nrunning forward passes over test set...", flush=True)
    n_batches_done = 0

    for batch in loader.iter_eval(batch_size=EVAL_BATCH):
        # Forward: K breaths
        logits_history, _ = factor_breathing_forward(model, batch, spec, K=K)
        final_logits = logits_history[-1]

        # Decode
        cell_valid_np = batch.cell_valid.realize().numpy()     # (B, S)
        gold_np       = batch.gold.realize().numpy().astype(np.int32)      # (B, S)
        pred_np       = (final_logits.argmax(axis=-1) + 1).realize().numpy().astype(np.int32)  # (B, S)
        eq_np         = (pred_np == gold_np).astype(np.float32) * cell_valid_np  # (B, S)

        # Per-breath CE (all batches, supervise = cell_valid since no givens)
        B_cur = int(cell_valid_np.shape[0])
        supervise_flat = batch.cell_valid.reshape(B_cur * S_MAX)
        sup_sum_t = supervise_flat.sum() + 1e-6
        gold_idx_flat = (batch.gold - 1).clip(0, N_VALUES - 1).reshape(B_cur * S_MAX)
        for k_idx, logits_k in enumerate(logits_history):
            ce_elems = logits_k.reshape(B_cur * S_MAX, N_VALUES).sparse_categorical_crossentropy(
                gold_idx_flat, reduction="none")
            ce_k = float(((ce_elems * supervise_flat).sum() / sup_sum_t).realize().numpy())
            pb_ce_sum[k_idx] += ce_k
        pb_ce_n_batches += 1

        # Python metadata for bucketing
        meta_band  = batch.band            # list[str], len B
        meta_depth = batch.deduction_depth # list[int], len B
        meta_n     = batch.n               # list[int], len B

        for b in range(B_cur):
            valid_mask = cell_valid_np[b] > 0.5
            nv = int(valid_mask.sum())
            if nv == 0:
                continue  # padding-only row (shouldn't happen but be safe)

            ceq = float(eq_np[b].sum())
            peq = int(np.all(pred_np[b][valid_mask] == gold_np[b][valid_mask]))

            band_key  = meta_band[b]
            depth_key = meta_depth[b]
            sz_key    = _n_bucket(meta_n[b])

            # overall
            overall["cell_eq_sum"]   += ceq
            overall["n_cells"]       += nv
            overall["puzzle_eq_sum"] += peq
            overall["n_puzzles"]     += 1

            # band
            if band_key not in band_acc:
                band_acc[band_key] = _mk()
            band_acc[band_key]["cell_eq_sum"]   += ceq
            band_acc[band_key]["n_cells"]       += nv
            band_acc[band_key]["puzzle_eq_sum"] += peq
            band_acc[band_key]["n_puzzles"]     += 1

            # depth
            if depth_key not in depth_acc:
                depth_acc[depth_key] = _mk()
            depth_acc[depth_key]["cell_eq_sum"]   += ceq
            depth_acc[depth_key]["n_cells"]       += nv
            depth_acc[depth_key]["puzzle_eq_sum"] += peq
            depth_acc[depth_key]["n_puzzles"]     += 1

            # size
            if sz_key not in size_acc:
                size_acc[sz_key] = _mk()
            size_acc[sz_key]["cell_eq_sum"]   += ceq
            size_acc[sz_key]["n_cells"]       += nv
            size_acc[sz_key]["puzzle_eq_sum"] += peq
            size_acc[sz_key]["n_puzzles"]     += 1

        n_batches_done += 1
        if n_batches_done % 10 == 0:
            ov_ca = overall["cell_eq_sum"] / max(overall["n_cells"], 1)
            ov_pa = overall["puzzle_eq_sum"] / max(overall["n_puzzles"], 1)
            print(f"  [{n_batches_done} batches] cell_acc={ov_ca:.3f} "
                  f"puzzle_acc={ov_pa:.3f} "
                  f"n_puzzles={overall['n_puzzles']}", flush=True)

    # ---- per-breath CE (mean over all batches) ------------------------------
    pb_ce_mean = [v / max(pb_ce_n_batches, 1) for v in pb_ce_sum]

    # ---- overall -----------------------------------------------------------
    ov_cell   = overall["cell_eq_sum"] / max(overall["n_cells"], 1)
    ov_puzzle = overall["puzzle_eq_sum"] / max(overall["n_puzzles"], 1)
    print(f"\n{'='*60}", flush=True)
    print(f"  OVERALL (full test set)", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  cell_acc  = {ov_cell:.4f}", flush=True)
    print(f"  puzzle_acc = {ov_puzzle:.4f}", flush=True)
    print(f"  n_puzzles  = {overall['n_puzzles']}", flush=True)
    print(f"  n_valid_cells = {overall['n_cells']}", flush=True)

    # ---- per-breath ladder -------------------------------------------------
    _print_ladder(pb_ce_mean, K)

    # ---- band breakdown ----------------------------------------------------
    _print_table(
        "cell_acc / puzzle_acc by DENSITY BAND (easy d10 → hard d25)",
        band_acc,
        col_order=BAND_ORDER,
    )

    # ---- depth breakdown ---------------------------------------------------
    depth_label = {
        0: "depth=0 (greedy)",
        1: "depth=1 (bt≤2)",
        2: "depth=2 (bt≤8)",
        3: "depth=3 (bt≤32)",
        4: "depth=4 (bt>32)",
    }
    depth_rows = {depth_label.get(d, str(d)): depth_acc[d]
                  for d in DEPTH_ORDER if d in depth_acc}
    _print_table(
        "cell_acc / puzzle_acc by DEDUCTION_DEPTH (DSATUR backtrack bucket)",
        depth_rows,
        col_order=list(depth_rows.keys()),
    )

    # ---- size breakdown ----------------------------------------------------
    size_order = ["n≤10", "n11-20", "n21-35", "n36+"]
    _print_table(
        "cell_acc / puzzle_acc by GRAPH SIZE (n vertices)",
        size_acc,
        col_order=size_order,
    )

    # ---- interpretation notes ----------------------------------------------
    print("=== INTERPRETATION GUIDE ===", flush=True)
    print("  Band d10 (1.0 edges/n) = sparsest / easiest (fewer constraints).", flush=True)
    print("  Band d25 (2.5 edges/n) = densest  / hardest (more constraints).", flush=True)
    print("  Depth 0 = DSATUR greedy-solvable (0 backtracks).", flush=True)
    print("  Depth 4 = hardest (>32 DSATUR backtracks).", flush=True)
    print("  Generality holds if cell_acc is stable across bands and depths.", flush=True)
    print("  Collapse on d25/depth-3/4 means the BP engine can't handle dense", flush=True)
    print("  constraint propagation — would gate Tier-2 foothold progression.", flush=True)
    print(flush=True)


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

if __name__ == "__main__":
    ok = _ast_parse_ok()
    print(f"[ast.parse] astparse_ok={ok}", flush=True)
    if not ok:
        sys.exit(1)
    main()
