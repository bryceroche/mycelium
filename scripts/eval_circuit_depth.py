"""eval_circuit_depth.py — Characterize Boolean-circuit deduction depth (Rung 1).

THE KEY RUNG-1 READ. A Boolean circuit is a DAG: each gate's output is a deduction
over its inputs, and a deep circuit chains many such deductions. So the binding
question for the breathing-as-BP claim is NOT just "does accuracy hold across whole
circuits" but "does accuracy fall off at deeper LEVELS" — the per-NODE lvl curve is
the deduction-depth ceiling. (KenKen is flat lateral cliques and graph coloring is
binary not-equal edges; neither exposes a depth axis the way a circuit DAG does.)

Loads a trained fg_circuit checkpoint and runs a full test-split evaluation bucketed
by:
  (a) CIRCUIT_DEPTH band D2..D5            — whole-circuit difficulty.
  (b) PER-NODE lvl (topological depth)      — THE deduction-depth ceiling read.
  (c) GRAPH SIZE n                          — does it scale with circuit width.
Plus the per-breath CE ladder (does K matter).

MODEL CONSTRUCTION mirrors factor_graph_train.py EXACTLY:
  - Pythia-410M L0-L3 backbone (same cast_layers_fp32 path)
  - attach_factor_graph_params with the circuit spec (s_max=49, n_values=2,
    n_factor_types=T [3 AND/OR/NOT, 4 with XOR], n_heads=16, k_max=K,
    has_factor_inlet=False).  T is read from the CircuitLoader (authoritative) so
    the spec always matches the membership/latent_type the engine sees.
  - load_ckpt with the same model_state_dict_fg key set.

TEST SPLIT reproduced identically to the training run:
  - CircuitLoader(n_instances=<FG_N_INSTANCES>, s_max=49, n_values=2,
    batch_size=<EVAL_BATCH>, seed=<SEED>, gate_types=..., use_xor=...) — same
    defaults => loader.test_records byte-identical to the trained run's held-out set.

GPU-FREE BUILD: ast.parse at the bottom verifies this file parses cleanly on CPU.
Run with DEV=AMD for the actual GPU eval:
  DEV=AMD FG_TASK=circuit FG_N_INSTANCES=8000 \\
      FG_CKPT=.cache/fg_ckpts/fg_circuit_k16/fg_circuit_k16_final.safetensors \\
      .venv/bin/python3 scripts/eval_circuit_depth.py
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
    from mycelium.factor_graph_engine import FG_HYP_MASK
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
    # No kenken inlet params for the circuit task (has_factor_inlet=False).
    # Per-type hyperbolic anchor tables (only present when FG_HYP_MASK=1).
    # Saved so load_ckpt restores the relaxed anchors from a trained checkpoint.
    if FG_HYP_MASK:
        t_idx = 0
        while True:
            anchors = getattr(model, f"fg_hyp_anchors_{t_idx}", None)
            if anchors is None:
                if t_idx > 64:
                    break
                t_idx += 1
                continue
            sd[f"fg_hyp_anchors_{t_idx}"] = anchors
            t_idx += 1
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
    """Coarse circuit-size (node-count) bucket."""
    if n <= 10:
        return "n≤10"
    if n <= 20:
        return "n11-20"
    if n <= 35:
        return "n21-35"
    return "n36+"


def _print_table(title: str, rows: dict, col_order=None) -> None:
    """Print a flat {key: {cell_eq_sum, n_cells, puzzle_eq_sum, n_puzzles}} table."""
    print(f"\n{'='*64}", flush=True)
    print(f"  {title}", flush=True)
    print(f"{'='*64}", flush=True)
    keys = col_order if col_order is not None else sorted(rows.keys())
    hdr = f"  {'bucket':<18}  {'cell_acc':>8}  {'whole_acc':>10}  {'count':>7}"
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
        print(f"  {str(k):<18}  {ca:8.3f}  {pa:10.3f}  {np_:7d}", flush=True)
        total_cells += nc
        total_ceq += r["cell_eq_sum"]
        total_puz += np_
        total_peq += r["puzzle_eq_sum"]
    if total_cells > 0:
        oca = total_ceq / total_cells
        opa = total_peq / max(total_puz, 1)
        print("  " + "-" * (len(hdr) - 2), flush=True)
        print(f"  {'OVERALL':<18}  {oca:8.3f}  {opa:10.3f}  {total_puz:7d}", flush=True)
    print("", flush=True)


def _print_node_level_table(title: str, rows: dict, max_lvl: int) -> None:
    """Per-NODE lvl table: one row per topological depth.  Here a 'cell' IS a node,
    so cell_acc == fraction of nodes at this lvl predicted correctly, and there is
    no whole-circuit column (a level is a set of nodes, not a circuit)."""
    print(f"\n{'='*64}", flush=True)
    print(f"  {title}", flush=True)
    print(f"{'='*64}", flush=True)
    hdr = f"  {'lvl':<8}  {'node_acc':>9}  {'n_nodes':>9}"
    print(hdr, flush=True)
    print("  " + "-" * (len(hdr) - 2), flush=True)
    prev_acc = None
    for lvl in range(0, max_lvl + 1):
        r = rows.get(lvl)
        if r is None or r["n_cells"] == 0:
            continue
        acc = r["cell_eq_sum"] / r["n_cells"]
        delta = "" if prev_acc is None else f"  (Δ vs lvl-1: {acc - prev_acc:+.3f})"
        print(f"  {lvl:<8}  {acc:9.3f}  {r['n_cells']:9d}{delta}", flush=True)
        prev_acc = acc
    print("", flush=True)


def _print_ladder(pb_ce: list[float], K: int) -> None:
    print(f"\n{'='*64}", flush=True)
    print("  Per-breath CE ladder (mean over test set; B0..B{})".format(K - 1),
          flush=True)
    print(f"{'='*64}", flush=True)
    n = len(pb_ce)
    for row_start in range(0, n, 8):
        chunk = pb_ce[row_start:row_start + 8]
        labels = "  ".join(f"B{row_start+i}={v:.3f}" for i, v in enumerate(chunk))
        print(f"  {labels}", flush=True)
    if n >= 2:
        drop = pb_ce[0] - pb_ce[-1]
        print(f"\n  CE drop B0->B{n-1}: {pb_ce[0]:.3f} -> {pb_ce[-1]:.3f} "
              f"(Δ={drop:.3f})", flush=True)
    print("", flush=True)


def _node_levels(batch, b: int, s_max: int) -> np.ndarray:
    """Pull per-node topological levels for batch element b as an int array of
    length s_max (>=0 for real nodes; the value at padding slots is ignored — we
    only read levels where cell_valid is set).

    The CircuitBatch contract exposes per-NODE lvl as python-side metadata. We
    accept either:
      - batch.lvl[b] a length-s_max array/list (per-slot levels), or
      - batch.lvl[b] a length-n list (per-real-node levels, in node order) —
        in which case real nodes are the first n cell_valid slots in index order.
    The first form (length s_max) is preferred and what the encoder emits.
    """
    lvl_b = batch.lvl[b]
    arr = np.asarray(lvl_b, dtype=np.int64)
    if arr.shape[0] == s_max:
        return arr
    # length-n form: scatter onto valid slots in index order.
    out = np.full((s_max,), -1, dtype=np.int64)
    out[:arr.shape[0]] = arr
    return out


# --------------------------------------------------------------------------
# Main evaluation
# --------------------------------------------------------------------------

def main():
    import gc
    from tinygrad import Tensor, Device, dtypes
    from tinygrad.helpers import getenv

    CKPT = getenv("FG_CKPT",
                  ".cache/fg_ckpts/fg_circuit_k16/fg_circuit_k16_final.safetensors")
    K = int(getenv("FG_K_MAX", getenv("K", "16")))
    EVAL_BATCH = int(getenv("EVAL_BATCH", getenv("BATCH", "8")))
    SEED = int(getenv("SEED", "42"))
    N_INSTANCES = int(getenv("FG_N_INSTANCES", "8000"))
    S_MAX = int(getenv("FG_S_MAX", "49"))
    N_VALUES = 2  # Boolean

    print("=== eval_circuit_depth.py — Boolean-circuit deduction-depth sweep ===",
          flush=True)
    print(f"device={Device.DEFAULT}  ckpt={CKPT}", flush=True)
    print(f"K={K}  EVAL_BATCH={EVAL_BATCH}  seed={SEED}  "
          f"n_instances={N_INSTANCES}  s_max={S_MAX}  n_values={N_VALUES}", flush=True)
    print(flush=True)

    # ---- build model (mirrors factor_graph_train.main) ---------------------
    from mycelium import Config, BreathingTransformer
    from mycelium.loader import _load_state, load_breathing
    from mycelium.factor_graph_engine import (
        FactorGraphSpec,
        attach_factor_graph_params,
        factor_breathing_forward,
        FG_HYP_MASK,
    )
    from mycelium.factor_masks import attach_factor_hyperbolic_params
    from mycelium.circuit_data import CircuitLoader

    # Gate-type / XOR knobs — MUST match the training run so T (and therefore the
    # spec + checkpoint shapes) line up.  The loader owns T; the spec follows it.
    # Keys MUST be UPPERCASE — CircuitLoader keys on ('AND','OR','NOT','XOR').
    gate_types_env = getenv("FG_CIRCUIT_GATE_TYPES", "").strip()
    use_xor = int(getenv("FG_CIRCUIT_XOR", "0")) > 0
    if gate_types_env:
        gate_types: tuple[str, ...] = tuple(g.strip().upper() for g in gate_types_env.split(",") if g.strip())
    elif use_xor:
        gate_types = ("AND", "OR", "NOT", "XOR")
    else:
        gate_types = ("AND", "OR", "NOT")          # loader default

    print(f"reconstructing CircuitLoader "
          f"(n_instances={N_INSTANCES}, seed={SEED}, "
          f"gate_types={gate_types})...", flush=True)
    loader = CircuitLoader(
        n_instances=N_INSTANCES,
        s_max=S_MAX,
        n_values=N_VALUES,
        batch_size=EVAL_BATCH,
        seed=SEED,
        gate_types=gate_types,
    )
    n_factor_types = int(loader.n_factor_types)
    n_test = len(loader.test_records)
    print(f"  test set: {n_test} instances  T(n_factor_types)={n_factor_types}  "
          f"n_gates_max={loader.n_gates_max}", flush=True)

    spec = FactorGraphSpec(
        s_max=S_MAX,
        n_values=N_VALUES,
        n_factor_types=n_factor_types,
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

    # FG_HYP_MASK=1 (frozen-confirm): build the circuit anchor tables from a
    # representative batch (one anchor table per gate type, sized by n_gates_max).
    if FG_HYP_MASK:
        print(f"[FG_HYP_MASK=1] building circuit anchor tables ...", flush=True)
        _ref_batch = loader.sample_batch()
        _mem_np = _ref_batch.membership.realize().numpy()   # (B, n_gates_max, S)
        _lt_np  = _ref_batch.latent_type.realize().numpy()  # (B, n_gates_max)
        attach_factor_hyperbolic_params(
            model,
            n_heads=spec.n_heads,
            n_factor_types=spec.n_factor_types,
            s_max=spec.s_max,
            membership_np=_mem_np,
            latent_type_np=_lt_np,
        )
        del _ref_batch, _mem_np, _lt_np
        print(f"  circuit hyperbolic params attached (frozen).", flush=True)

    Device[Device.DEFAULT].synchronize()

    print(f"loading checkpoint: {CKPT}", flush=True)
    load_ckpt(model, CKPT)

    # ---- accumulators (per bucket) -----------------------------------------
    # circuit_depth bands D2..D5 (the whole-circuit difficulty axis).
    DEPTH_BAND_ORDER = ["D2", "D3", "D4", "D5"]

    def _mk() -> dict:
        return {"cell_eq_sum": 0.0, "n_cells": 0,
                "puzzle_eq_sum": 0, "n_puzzles": 0}

    band_acc:  dict[str, dict] = {b: _mk() for b in DEPTH_BAND_ORDER}
    lvl_acc:   dict[int, dict] = {}   # per-NODE topological level -> acc (the key read)
    size_acc:  dict[str, dict] = {}
    overall    = _mk()
    max_lvl_seen = 0

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
        cell_valid_np = batch.cell_valid.realize().numpy()                       # (B, S)
        gold_np       = batch.gold.realize().numpy().astype(np.int32)            # (B, S)
        pred_np       = (final_logits.argmax(axis=-1) + 1).realize().numpy().astype(np.int32)  # (B, S)
        eq_np         = (pred_np == gold_np).astype(np.float32) * cell_valid_np  # (B, S)

        # Per-breath CE (all batches).  Mirror the trainer's supervision mask:
        # leaves are GIVEN (input_cells > 0), gate nodes are supervised.
        B_cur = int(cell_valid_np.shape[0])
        observed = (batch.input_cells > 0).cast(dtypes.float)      # (B, S) — leaves
        supervise_flat = (batch.cell_valid * (1.0 - observed)).reshape(B_cur * S_MAX)
        sup_sum_t = supervise_flat.sum() + 1e-6
        gold_idx_flat = (batch.gold - 1).clip(0, N_VALUES - 1).reshape(B_cur * S_MAX)
        for k_idx, logits_k in enumerate(logits_history):
            ce_elems = logits_k.reshape(B_cur * S_MAX, N_VALUES).sparse_categorical_crossentropy(
                gold_idx_flat, reduction="none")
            ce_k = float(((ce_elems * supervise_flat).sum() / sup_sum_t).realize().numpy())
            pb_ce_sum[k_idx] += ce_k
        pb_ce_n_batches += 1

        # Python metadata for bucketing
        meta_band  = batch.band           # list[str], len B (D2..D5)
        meta_depth = batch.circuit_depth  # list[int], len B (per-instance DAG depth)
        meta_n     = batch.n              # list[int], len B (node count)

        for b in range(B_cur):
            valid_mask = cell_valid_np[b] > 0.5
            nv = int(valid_mask.sum())
            if nv == 0:
                continue  # padding-only row (shouldn't happen but be safe)

            ceq = float(eq_np[b].sum())
            peq = int(np.all(pred_np[b][valid_mask] == gold_np[b][valid_mask]))

            # band key: prefer the explicit band string; fall back to a depth bucket.
            band_key = meta_band[b]
            if band_key not in band_acc:
                # accept a numeric depth -> "D<depth>" label if band string absent.
                band_key = f"D{int(meta_depth[b])}"
            sz_key = _n_bucket(meta_n[b])

            # overall
            overall["cell_eq_sum"]   += ceq
            overall["n_cells"]       += nv
            overall["puzzle_eq_sum"] += peq
            overall["n_puzzles"]     += 1

            # circuit-depth band
            if band_key not in band_acc:
                band_acc[band_key] = _mk()
            band_acc[band_key]["cell_eq_sum"]   += ceq
            band_acc[band_key]["n_cells"]       += nv
            band_acc[band_key]["puzzle_eq_sum"] += peq
            band_acc[band_key]["n_puzzles"]     += 1

            # size
            if sz_key not in size_acc:
                size_acc[sz_key] = _mk()
            size_acc[sz_key]["cell_eq_sum"]   += ceq
            size_acc[sz_key]["n_cells"]       += nv
            size_acc[sz_key]["puzzle_eq_sum"] += peq
            size_acc[sz_key]["n_puzzles"]     += 1

            # ---- PER-NODE lvl (the deduction-depth ceiling read) ----
            levels_b = _node_levels(batch, b, S_MAX)         # (S,) int (-1 = pad)
            for s in range(S_MAX):
                if not valid_mask[s]:
                    continue
                lvl = int(levels_b[s])
                if lvl < 0:
                    continue
                if lvl > max_lvl_seen:
                    max_lvl_seen = lvl
                if lvl not in lvl_acc:
                    lvl_acc[lvl] = _mk()
                lvl_acc[lvl]["cell_eq_sum"] += float(eq_np[b, s])
                lvl_acc[lvl]["n_cells"]     += 1
                # no whole-circuit count at node granularity.

        n_batches_done += 1
        if n_batches_done % 10 == 0:
            ov_ca = overall["cell_eq_sum"] / max(overall["n_cells"], 1)
            ov_pa = overall["puzzle_eq_sum"] / max(overall["n_puzzles"], 1)
            print(f"  [{n_batches_done} batches] cell_acc={ov_ca:.3f} "
                  f"circuit_acc={ov_pa:.3f} "
                  f"n_circuits={overall['n_puzzles']}", flush=True)

    # ---- per-breath CE (mean over all batches) ------------------------------
    pb_ce_mean = [v / max(pb_ce_n_batches, 1) for v in pb_ce_sum]

    # ---- overall -----------------------------------------------------------
    ov_cell   = overall["cell_eq_sum"] / max(overall["n_cells"], 1)
    ov_puzzle = overall["puzzle_eq_sum"] / max(overall["n_puzzles"], 1)
    print(f"\n{'='*64}", flush=True)
    print(f"  OVERALL (full test set)", flush=True)
    print(f"{'='*64}", flush=True)
    print(f"  cell_acc    (per-node)        = {ov_cell:.4f}", flush=True)
    print(f"  circuit_acc (whole-circuit)   = {ov_puzzle:.4f}", flush=True)
    print(f"  n_circuits                    = {overall['n_puzzles']}", flush=True)
    print(f"  n_valid_nodes                 = {overall['n_cells']}", flush=True)

    # ---- per-breath ladder -------------------------------------------------
    _print_ladder(pb_ce_mean, K)

    # ---- (a) circuit-depth band breakdown ----------------------------------
    _print_table(
        "cell_acc / whole-circuit acc by CIRCUIT_DEPTH band (D2 shallow -> D5 deep)",
        band_acc,
        col_order=DEPTH_BAND_ORDER,
    )

    # ---- (b) PER-NODE lvl breakdown — THE KEY RUNG-1 READ -------------------
    _print_node_level_table(
        "node_acc by PER-NODE lvl (topological depth) — the deduction-depth ceiling",
        lvl_acc,
        max_lvl_seen,
    )

    # ---- (c) size breakdown ------------------------------------------------
    size_order = ["n≤10", "n11-20", "n21-35", "n36+"]
    _print_table(
        "cell_acc / whole-circuit acc by CIRCUIT SIZE (n nodes)",
        size_acc,
        col_order=size_order,
    )

    # ---- interpretation notes ----------------------------------------------
    print("=== INTERPRETATION GUIDE ===", flush=True)
    print("  Circuit = DAG; a gate output is a deduction over its inputs.", flush=True)
    print("  D2 band = shallow circuits; D5 = deepest (longest deduction chains).", flush=True)
    print("  PER-NODE lvl = topological depth of a node (input=0, output=deepest).", flush=True)
    print("  THE KEY READ: if node_acc falls off as lvl grows, the BP engine has a", flush=True)
    print("  deduction-depth ceiling — deeper gates depend on resolving shallower", flush=True)
    print("  ones first, and K breaths may be too few to propagate that far.", flush=True)
    print("  Flat node_acc across lvl => the breath cycle propagates to full depth.", flush=True)
    print("  Watch the per-breath CE ladder: a real drop B0->B{K-1} means K matters.",
          flush=True)
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
