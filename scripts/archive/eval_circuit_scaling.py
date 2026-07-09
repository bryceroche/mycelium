"""eval_circuit_scaling.py — Deep-circuit SCALING PROBE (Rung-2, D6..D16).

THE SCALING PROBE.  Rung-1 confirmed the engine solves shallow Boolean circuits
(D≤5, ~0.97 cell accuracy, flat per-level curve).  This script pushes the same
model to DEPTH D=16 — harder than any circuit it saw at D≤5 — and runs two
measurements:

  (A) PER-D ACCURACY: cell_acc and whole-circuit acc for each deep band D6..D16
      using circuits from the SAME model that trained on D≤5.  If accuracy holds
      at D=16, the engine generalizes to arbitrary depth.  If it falls off, there
      is a depth-ceiling and more training (with deep bands mixed in) is needed.

  (B) K-SWEEP: evaluate the SAME model at K' ∈ {4, 8, 12, 16} on deep circuits.
      This is the PRIMARY diagnostic for PARALLEL vs SEQUENTIAL deduction:
        * If acc stays HIGH for K' << D  (e.g. K'=4, D=16) → the engine solves
          depth-16 in only 4 breaths → PARALLEL message-passing (many sub-problems
          processed simultaneously per breath).
        * If acc COLLAPSES when K' < D → needs at least D breaths for depth-D →
          SEQUENTIAL: each breath resolves one level of the chain before the next.
      The breathing-as-BP hypothesis predicts PARALLEL (loopy BP converges in far
      fewer rounds than the longest path).  A clean K-sweep at various D gives the
      empirical answer.

  (C) D×K' MATRIX: prints a table with rows = D band and cols = K' value.

TRAINED CHECKPOINT:
  The script loads a checkpoint trained on D2..D5 (the baseline B3 checkpoint).
  For the best read, also train a DEEP MIX (D4..D16) and compare (the DEEP MIX
  train command is printed at the end of this script's output).

USAGE:
  GPU run (AMD):
    DEV=AMD FG_CKPT=.cache/fg_ckpts/fg_circuit_k16_final.safetensors \\
        FG_TASK=circuit FG_N_INSTANCES=8000 \\
        .venv/bin/python3 scripts/eval_circuit_scaling.py \\
        --bands D6,D8,D10,D12,D14,D16 --k-sweep 4,8,12,16 --n-eval 400

  CPU import/ast check (GPU-free):
    .venv/bin/python3 scripts/eval_circuit_scaling.py --cpu-smoke

Env vars:
  FG_CKPT           path to safetensors checkpoint (required for full eval)
  FG_K_MAX / K      K_max the model was trained with (default 16)
  FG_N_INSTANCES    corpus size to reproduce the loader (default 8000)
  EVAL_BATCH        batch size for eval (default 8)
  SEED              corpus RNG seed (default 42)
  FG_CIRCUIT_GATE_TYPES  comma-separated gate types (default AND,OR,NOT)
  FG_CIRCUIT_XOR    1 to include XOR gate type (default 0)
"""
from __future__ import annotations

import ast
import os
import sys

_THIS_FILE = os.path.abspath(__file__)
_ROOT = os.path.dirname(os.path.dirname(_THIS_FILE))
sys.path.insert(0, _ROOT)


# ---------------------------------------------------------------------------
# ast.parse gate — always runs, even on CPU
# ---------------------------------------------------------------------------

def _ast_parse_ok() -> bool:
    with open(_THIS_FILE) as f:
        src = f.read()
    try:
        ast.parse(src)
        return True
    except SyntaxError as e:
        print(f"[ast.parse] FAILED: {e}", flush=True)
        return False


# ---------------------------------------------------------------------------
# Mirrors of helpers from factor_graph_train / eval_circuit_depth
# ---------------------------------------------------------------------------

def cast_layers_fp32(model) -> None:
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


def load_ckpt(model, path: str) -> None:
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


# ---------------------------------------------------------------------------
# Evaluation over a set of circuits at a specific K'
# ---------------------------------------------------------------------------

def eval_at_k(model, loader, spec_template, K_prime: int,
              n_eval: int, batch_size: int,
              bands_to_eval: list[str],
              ) -> dict[str, dict]:
    """Evaluate the model at K_prime breaths over n_eval deep-band circuits.

    Returns a nested dict: band -> {"cell_eq": float, "n_cells": int,
                                    "puzzle_eq": int, "n_puzzles": int}
    Also includes a "_total" key for the aggregate.

    Generates n_eval instances PER BAND from generate_skinny_instance() (NOT
    the corpus loader's test set, because the original loader has only D2..D5).
    A fresh RNG seeded per band ensures reproducibility.
    """
    import random as _random

    import numpy as np
    from tinygrad import Tensor, dtypes

    from mycelium.circuit_data import (
        generate_skinny_instance, encode_instance,
        _DEEP_BAND_TARGET_D, _ALL_BAND_TARGET_D,
    )
    from mycelium.factor_graph_engine import (
        FactorGraphSpec, factor_breathing_forward,
    )
    from mycelium.circuit_data import CircuitBatch

    # Build spec for K_prime (k_max only affects breath_embed size assertion).
    # k_max must be >= K_prime; reuse the trained k_max from spec_template.
    spec = spec_template

    S = spec.s_max
    N = spec.n_values
    T = spec.n_factor_types
    K_train = int(spec.k_max)

    assert K_prime <= K_train, (
        f"K_prime={K_prime} > K_train={K_train}; the model was trained with "
        f"at most K_train breaths; reduce K_prime or train with higher K.")

    accum: dict[str, dict] = {
        b: {"cell_eq": 0.0, "n_cells": 0, "puzzle_eq": 0, "n_puzzles": 0}
        for b in bands_to_eval
    }
    accum["_total"] = {"cell_eq": 0.0, "n_cells": 0, "puzzle_eq": 0, "n_puzzles": 0}

    gate_types = loader.gate_types   # must match training

    n_gates_max = loader.n_gates_max   # static topology width from the loader

    for band in bands_to_eval:
        rng = _random.Random(hash(band) ^ 0xDEAD_C0DE)
        instances_done = 0
        attempts = 0
        batch_buf: list[dict] = []

        while instances_done < n_eval and attempts < n_eval * 200 + 500:
            attempts += 1
            # Use the skinny generator for deep bands, generic for shallow.
            if band in _DEEP_BAND_TARGET_D:
                inst = generate_skinny_instance(rng, S, band,
                                                gate_types=gate_types)
            else:
                from mycelium.circuit_data import generate_instance
                inst = generate_instance(rng, S, band, gate_types=gate_types)
            if inst is None:
                continue
            batch_buf.append(inst)

            if len(batch_buf) == batch_size or (
                    instances_done + len(batch_buf) >= n_eval
                    and len(batch_buf) > 0):
                # Pad batch to batch_size with repeats if at the end.
                while len(batch_buf) < batch_size:
                    batch_buf.append(batch_buf[0])

                # Encode and stack into a CircuitBatch.
                import numpy as _np
                encs = [encode_instance(r, S, n_gates_max, T) for r in batch_buf]

                def _stack_int(key):
                    from tinygrad import dtypes as _dtypes
                    return Tensor(
                        _np.stack([e[key] for e in encs]).astype(_np.int32),
                        dtype=_dtypes.int).contiguous().realize()

                def _stack_f(key):
                    from tinygrad import dtypes as _dtypes
                    return Tensor(
                        _np.stack([e[key] for e in encs]).astype(_np.float32),
                        dtype=_dtypes.float).contiguous().realize()

                d = {
                    "input_cells":        _stack_int("input_cells"),
                    "cell_valid":         _stack_f("cell_valid"),
                    "value_domain_mask":  _stack_f("value_domain_mask"),
                    "gold":               _stack_int("gold"),
                    "membership":         _stack_f("membership"),
                    "latent_type":        _stack_int("latent_type"),
                    "lvl":                _np.stack([e["lvl"] for e in encs]).astype(_np.int32),
                    "is_leaf":            _np.stack([e["is_leaf"] for e in encs]).astype(_np.float32),
                    "circuit_depth":      [e["circuit_depth"] for e in encs],
                    "n":                  [e["n"] for e in encs],
                    "n_leaves":           [e["n_leaves"] for e in encs],
                    "n_gates":            [e["n_gates"] for e in encs],
                    "band":               [e["band"] for e in encs],
                    "depth_shuffled":     False,
                }
                cb = CircuitBatch(d)

                # Forward at K_prime.
                Tensor.training = False
                logits_hist, _ = factor_breathing_forward(model, cb, spec, K=K_prime)
                final_logits = logits_hist[-1]

                cv_np   = cb.cell_valid.realize().numpy()
                gold_np = cb.gold.realize().numpy().astype(_np.int32)
                pred_np = (final_logits.argmax(axis=-1) + 1).realize().numpy().astype(_np.int32)
                eq_np   = (pred_np == gold_np).astype(_np.float32) * cv_np

                real_in_batch = min(batch_size, n_eval - instances_done)
                for bi in range(real_in_batch):
                    valid = cv_np[bi] > 0.5
                    nv = int(valid.sum())
                    if nv == 0:
                        continue
                    ceq = float(eq_np[bi].sum())
                    peq = int(_np.all(pred_np[bi][valid] == gold_np[bi][valid]))
                    accum[band]["cell_eq"]    += ceq
                    accum[band]["n_cells"]    += nv
                    accum[band]["puzzle_eq"]  += peq
                    accum[band]["n_puzzles"]  += 1
                    accum["_total"]["cell_eq"]   += ceq
                    accum["_total"]["n_cells"]   += nv
                    accum["_total"]["puzzle_eq"] += peq
                    accum["_total"]["n_puzzles"] += 1

                instances_done += real_in_batch
                batch_buf = []

        if instances_done < n_eval:
            print(f"  [WARNING] {band}: only {instances_done}/{n_eval} instances "
                  f"generated in {attempts} attempts", flush=True)

    return accum


# ---------------------------------------------------------------------------
# Print helpers
# ---------------------------------------------------------------------------

def _cell_acc(r: dict) -> float:
    return r["cell_eq"] / max(r["n_cells"], 1)


def _puzzle_acc(r: dict) -> float:
    return r["puzzle_eq"] / max(r["n_puzzles"], 1)


def _print_per_d_table(per_d: dict[str, dict], bands: list[str]) -> None:
    """Print cell_acc / whole-circuit acc by depth band."""
    print(f"\n{'='*72}", flush=True)
    print("  PER-D ACCURACY (deep circuits, K=K_train)", flush=True)
    print(f"{'='*72}", flush=True)
    hdr = f"  {'band':<8}  {'cell_acc':>8}  {'puzzle_acc':>10}  {'n_puzzles':>9}"
    print(hdr, flush=True)
    print("  " + "-" * (len(hdr) - 2), flush=True)
    for band in bands:
        r = per_d.get(band)
        if r is None or r["n_puzzles"] == 0:
            print(f"  {band:<8}  {'(no data)':>8}", flush=True)
            continue
        ca = _cell_acc(r)
        pa = _puzzle_acc(r)
        print(f"  {band:<8}  {ca:8.3f}  {pa:10.3f}  {r['n_puzzles']:9d}", flush=True)
    tot = per_d.get("_total")
    if tot and tot["n_puzzles"] > 0:
        print("  " + "-" * (len(hdr) - 2), flush=True)
        print(f"  {'OVERALL':<8}  {_cell_acc(tot):8.3f}  "
              f"{_puzzle_acc(tot):10.3f}  {tot['n_puzzles']:9d}", flush=True)
    print("", flush=True)


def _print_k_sweep_matrix(matrix: dict[str, dict[int, dict]],
                           bands: list[str], k_values: list[int]) -> None:
    """Print D × K' cell_acc matrix."""
    print(f"\n{'='*72}", flush=True)
    print("  K-SWEEP MATRIX: cell_acc[D][K']", flush=True)
    print(f"{'='*72}", flush=True)
    print("  (Each cell = cell_acc for that depth band at that K' budget)", flush=True)
    print("  READ: K' << D but high acc -> PARALLEL deduction", flush=True)
    print("         acc collapses when K' < D   -> SEQUENTIAL deduction", flush=True)
    print(flush=True)

    # Header
    k_col_width = 8
    band_col_width = 8
    hdr_parts = [f"  {'band':<{band_col_width}}"]
    for k in k_values:
        hdr_parts.append(f"  {'K='+str(k):>{k_col_width}}")
    print("".join(hdr_parts), flush=True)
    print("  " + "-" * (band_col_width + (k_col_width + 2) * len(k_values) + 2),
          flush=True)

    for band in bands:
        row_parts = [f"  {band:<{band_col_width}}"]
        for k in k_values:
            r = matrix.get(band, {}).get(k)
            if r is None or r["n_puzzles"] == 0:
                row_parts.append(f"  {'n/a':>{k_col_width}}")
            else:
                row_parts.append(f"  {_cell_acc(r):{k_col_width}.3f}")
        print("".join(row_parts), flush=True)
    print("", flush=True)

    # Row: ratio K'=4 vs K'=max (diagnostic for parallel vs sequential).
    k_min = min(k_values)
    k_max_v = max(k_values)
    print(f"  Ratio K'={k_min}/K'={k_max_v} per band "
          f"(near 1.0 = PARALLEL; << 1.0 = SEQUENTIAL):", flush=True)
    for band in bands:
        r_min = matrix.get(band, {}).get(k_min)
        r_max = matrix.get(band, {}).get(k_max_v)
        if (r_min and r_min["n_puzzles"] > 0 and
                r_max and r_max["n_puzzles"] > 0):
            ratio = _cell_acc(r_min) / max(_cell_acc(r_max), 1e-6)
            verdict = "PARALLEL" if ratio >= 0.90 else ("MIXED" if ratio >= 0.70 else "SEQUENTIAL")
            print(f"    {band}: {ratio:.2f}  [{verdict}]", flush=True)
    print("", flush=True)


# ---------------------------------------------------------------------------
# CPU smoke (GPU-free) — validates the eval infrastructure without a GPU
# ---------------------------------------------------------------------------

def _cpu_smoke() -> None:
    """CPU-only smoke: generate deep circuits + verify the eval_at_k plumbing.

    Does NOT run factor_breathing_forward (requires the full model + GPU).
    Instead verifies:
      1. Deep-skinny generation works (D=12 fits in 49 nodes, gold correct).
      2. CircuitBatch packing works for deep bands.
      3. encode_instance + stack shapes are correct.
    """
    import random as _random
    import numpy as _np
    from mycelium.circuit_data import (
        generate_skinny_instance, encode_instance,
        _longest_path_lvl, _eval_gate, DEEP_BANDS, _DEEP_BAND_TARGET_D,
        CircuitLoader,
    )

    print("[eval_circuit_scaling] CPU smoke...", flush=True)

    # 1. Deep-skinny D=12 instance.
    rng = _random.Random(12345)
    found = None
    for _ in range(500):
        inst = generate_skinny_instance(rng, 49, "D12")
        if inst is not None:
            found = inst
            break
    assert found is not None, "D12 skinny instance: None after 500 attempts"
    assert found["n"] <= 49, f"D12 n={found['n']} > 49"
    assert found["circuit_depth"] == 12, \
        f"D12 circuit_depth={found['circuit_depth']} != 12"

    # verify gold via independent topo-eval
    n_i = found["n"]
    recomputed = []
    for v in range(n_i):
        if not found["operands"][v]:
            recomputed.append(found["bits"][v])
        else:
            val = _eval_gate(found["gtypes"][v],
                             [recomputed[o] for o in found["operands"][v]])
            recomputed.append(val)
    for v in range(n_i):
        assert recomputed[v] == found["bits"][v], \
            f"D12 gold[{v}]={found['bits'][v]} != reeval {recomputed[v]}"

    # Verify longest-path is exactly 12.
    lvl_re = _longest_path_lvl(n_i, found["operands"])
    assert max(lvl_re) == 12, f"D12 longest-path={max(lvl_re)} != 12"
    print(f"  D12 instance: n={found['n']} nodes, gold verified, "
          f"longest-path={max(lvl_re)}", flush=True)

    # 2. encode_instance packing.
    enc = encode_instance(found, 49, found["n_gates"] + 2, 3)
    assert enc["input_cells"].shape == (49,)
    assert enc["gold"].shape == (49,)
    assert enc["membership"].shape[1] == 49
    print(f"  encode_instance OK: shapes correct.", flush=True)

    # 3. CircuitLoader with deep bands.
    loader = CircuitLoader(
        n_instances=80, s_max=49, n_values=2,
        batch_size=4, seed=7,
        bands=["D6", "D8", "D10"],
    )
    batch = loader.sample_batch()
    assert batch.input_cells.shape == (4, 49)
    for idx in range(4):
        d_b = batch.circuit_depth[idx]
        n_b = batch.n[idx]
        assert 6 <= d_b <= 10, f"batch.circuit_depth={d_b} not in [6,10]"
        assert n_b <= 49, f"n_b={n_b} > 49"
    print(f"  CircuitLoader(D6/D8/D10) sample: OK (depths "
          f"{batch.circuit_depth}, sizes {batch.n})", flush=True)

    print("[eval_circuit_scaling] CPU smoke PASSED.", flush=True)


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    import argparse
    import gc

    parser = argparse.ArgumentParser(description="Deep-circuit scaling probe")
    parser.add_argument("--bands",
                        default="D6,D8,D10,D12,D14,D16",
                        help="Comma-separated depth bands to eval (default D6..D16)")
    parser.add_argument("--k-sweep",
                        default="4,8,12,16",
                        help="Comma-separated K' values for the K-sweep")
    parser.add_argument("--n-eval",
                        type=int, default=200,
                        help="Instances to eval PER band (default 200)")
    parser.add_argument("--ckpt",
                        default=None,
                        help="Checkpoint path (overrides FG_CKPT env var)")
    parser.add_argument("--cpu-smoke",
                        action="store_true",
                        help="Run GPU-free CPU smoke and exit")
    args = parser.parse_args(argv)

    if args.cpu_smoke:
        _cpu_smoke()
        return

    from tinygrad import Tensor, Device, dtypes
    from tinygrad.helpers import getenv

    CKPT = args.ckpt or getenv("FG_CKPT", "")
    K_TRAIN = int(getenv("FG_K_MAX", getenv("K", "16")))
    EVAL_BATCH = int(getenv("EVAL_BATCH", getenv("BATCH", "8")))
    SEED = int(getenv("SEED", "42"))
    N_INSTANCES = int(getenv("FG_N_INSTANCES", "8000"))
    S_MAX = int(getenv("FG_S_MAX", "49"))
    N_VALUES = 2

    bands = [b.strip() for b in args.bands.split(",") if b.strip()]
    k_values = [int(k.strip()) for k in args.k_sweep.split(",") if k.strip()]
    k_values = sorted(set(k_values))
    n_eval = int(args.n_eval)

    print("=== eval_circuit_scaling.py — deep-circuit scaling probe ===",
          flush=True)
    print(f"device={Device.DEFAULT}  ckpt={CKPT}", flush=True)
    print(f"K_train={K_TRAIN}  k_sweep={k_values}  n_eval_per_band={n_eval}",
          flush=True)
    print(f"bands={bands}", flush=True)
    print(f"EVAL_BATCH={EVAL_BATCH}  seed={SEED}  n_instances={N_INSTANCES}  "
          f"s_max={S_MAX}", flush=True)
    print(flush=True)

    if not CKPT:
        print("ERROR: FG_CKPT not set and --ckpt not provided.", flush=True)
        print("Set FG_CKPT=<path> or pass --ckpt <path>.", flush=True)
        sys.exit(1)

    # ---- gate types (MUST match training run) --------------------------------
    gate_types_env = getenv("FG_CIRCUIT_GATE_TYPES", "").strip()
    use_xor = int(getenv("FG_CIRCUIT_XOR", "0")) > 0
    if gate_types_env:
        gate_types: tuple[str, ...] = tuple(
            g.strip().upper() for g in gate_types_env.split(",") if g.strip())
    elif use_xor:
        gate_types = ("AND", "OR", "NOT", "XOR")
    else:
        gate_types = ("AND", "OR", "NOT")

    # ---- build a shallow loader to get n_gates_max + n_factor_types ---------
    # We reproduce the ORIGINAL training loader (D2..D5 shallow) to get the
    # JIT topology width (n_gates_max) that the checkpoint was trained with.
    # The eval generates DEEP circuits on the fly using the same gate_types.
    print(f"Reproducing training loader (D2..D5, n={N_INSTANCES}) to fix "
          f"n_gates_max...", flush=True)
    from mycelium.circuit_data import CircuitLoader
    shallow_loader = CircuitLoader(
        n_instances=N_INSTANCES,
        s_max=S_MAX,
        n_values=N_VALUES,
        batch_size=EVAL_BATCH,
        seed=SEED,
        gate_types=gate_types,
    )
    n_factor_types = int(shallow_loader.n_factor_types)
    n_gates_max = int(shallow_loader.n_gates_max)
    print(f"  n_gates_max={n_gates_max}  n_factor_types(T)={n_factor_types}",
          flush=True)

    # ---- build model ---------------------------------------------------------
    from mycelium import Config, BreathingTransformer
    from mycelium.loader import _load_state, load_breathing
    from mycelium.factor_graph_engine import (
        FactorGraphSpec,
        attach_factor_graph_params,
        factor_breathing_forward,
        FG_HYP_MASK,
    )
    from mycelium.factor_masks import attach_factor_hyperbolic_params

    spec = FactorGraphSpec(
        s_max=S_MAX,
        n_values=N_VALUES,
        n_factor_types=n_factor_types,
        n_heads=16,
        k_max=K_TRAIN,
        has_factor_inlet=False,
    )

    print("Loading Pythia-410M -> BreathingTransformer...", flush=True)
    cfg = Config()
    sd = _load_state()
    model = load_breathing(cfg, sd=sd)
    del sd
    gc.collect()
    cast_layers_fp32(model)
    attach_factor_graph_params(model, hidden=cfg.hidden, spec=spec)

    if FG_HYP_MASK:
        print("[FG_HYP_MASK=1] building circuit anchor tables...", flush=True)
        _ref_batch = shallow_loader.sample_batch()
        _mem_np = _ref_batch.membership.realize().numpy()
        _lt_np = _ref_batch.latent_type.realize().numpy()
        attach_factor_hyperbolic_params(
            model,
            n_heads=spec.n_heads,
            n_factor_types=spec.n_factor_types,
            s_max=spec.s_max,
            membership_np=_mem_np,
            latent_type_np=_lt_np,
        )
        del _ref_batch, _mem_np, _lt_np

    Device[Device.DEFAULT].synchronize()
    print(f"Loading checkpoint: {CKPT}", flush=True)
    load_ckpt(model, CKPT)
    Tensor.training = False

    # ---- (A) PER-D accuracy at K=K_train ------------------------------------
    print(f"\n[A] PER-D ACCURACY at K_train={K_TRAIN}...", flush=True)
    per_d_accum: dict[str, dict] = {}
    for band in bands:
        print(f"  evaluating band {band} ({n_eval} instances)...", flush=True)
        res = eval_at_k(model, shallow_loader, spec, K_TRAIN,
                        n_eval, EVAL_BATCH, [band])
        per_d_accum[band] = res[band]
        per_d_accum.setdefault("_total", {"cell_eq": 0.0, "n_cells": 0,
                                           "puzzle_eq": 0, "n_puzzles": 0})
        per_d_accum["_total"]["cell_eq"]    += res["_total"]["cell_eq"]
        per_d_accum["_total"]["n_cells"]    += res["_total"]["n_cells"]
        per_d_accum["_total"]["puzzle_eq"]  += res["_total"]["puzzle_eq"]
        per_d_accum["_total"]["n_puzzles"]  += res["_total"]["n_puzzles"]
        ca = _cell_acc(res[band])
        pa = _puzzle_acc(res[band])
        print(f"    {band}: cell_acc={ca:.3f}  puzzle_acc={pa:.3f}  "
              f"n={res[band]['n_puzzles']}", flush=True)

    _print_per_d_table(per_d_accum, bands)

    # ---- (B) K-SWEEP --------------------------------------------------------
    print(f"\n[B] K-SWEEP across K' in {k_values}...", flush=True)
    # matrix[band][k_prime] = accum dict
    matrix: dict[str, dict[int, dict]] = {b: {} for b in bands}

    for k_prime in k_values:
        print(f"  K'={k_prime}:", flush=True)
        for band in bands:
            res = eval_at_k(model, shallow_loader, spec, k_prime,
                            n_eval, EVAL_BATCH, [band])
            matrix[band][k_prime] = res[band]
            ca = _cell_acc(res[band])
            print(f"    {band}: cell_acc={ca:.3f}", flush=True)

    _print_k_sweep_matrix(matrix, bands, k_values)

    # ---- (C) Interpretation notes -------------------------------------------
    print("=== INTERPRETATION GUIDE ===", flush=True)
    print("  (A) PER-D curve: flat = engine depth-generalizes; "
          "falling = depth ceiling.", flush=True)
    print("  (B) K-SWEEP matrix: KEY DIAGNOSTIC for parallel vs sequential:", flush=True)
    print("      acc(K'=4, D=16) ≈ acc(K'=16, D=16)  → PARALLEL (many levels / breath)", flush=True)
    print("      acc(K'=4, D=16) << acc(K'=16, D=16) → SEQUENTIAL (need ≥D breaths)", flush=True)
    print("  If SEQUENTIAL: train with deep-mix bands (D4..D16) to see if the engine", flush=True)
    print("  learns to parallelize (or increase K; K=28 HUNG — stay at K≤16).", flush=True)
    print(flush=True)
    print("=== DEEP-MIX TRAIN COMMAND ===", flush=True)
    print("  To train a model on deep-mix D4..D16 (K=16, s_max=49):", flush=True)
    print("  FG_TASK=circuit FG_N_INSTANCES=16000 \\", flush=True)
    print("      FG_CIRCUIT_BANDS=D4,D5,D6,D8,D10,D12,D14,D16 \\", flush=True)
    print("      FG_S_MAX=49 K=16 BATCH=8 STEPS=4000 \\", flush=True)
    print("      .venv/bin/python3 scripts/factor_graph_train.py", flush=True)
    print(flush=True)
    print("=== K-SWEEP EVAL COMMAND ===", flush=True)
    print("  DEV=AMD FG_CKPT=<deep_mix_ckpt.safetensors> \\", flush=True)
    print("      .venv/bin/python3 scripts/eval_circuit_scaling.py \\", flush=True)
    print("      --bands D6,D8,D10,D12,D14,D16 --k-sweep 4,8,12,16 --n-eval 400",
          flush=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ok = _ast_parse_ok()
    print(f"[ast.parse] astparse_ok={ok}", flush=True)
    if not ok:
        sys.exit(1)
    main()
