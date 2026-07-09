"""probe_svd_collapse_multitask.py — Representation collapse on the GENERAL-WEIGHTS ckpt.

Re-probes the validated MULTI-TASK general-weights checkpoint (one shared Pythia
backbone co-trained on coloring + circuit + kenken) and compares, PER DOMAIN, the
same three collapse measurements the single-domain baseline (scripts/probe_svd_collapse.py,
project_svd_collapse_baseline) banked on the frozen single-domain ckpts. The delta
(single-domain rank - multi-task rank) is the MULTI-TASK-INDUCED collapse, if any.
It also reads the CATHEDRAL TRIGGER (a forgetting signature: deductions that resolve
then degrade across breaths -- the single residual failing to HOLD resolved
intermediates), which gates the cross-breath cathedral memory.

NON-INVASIVE. NEVER edits mycelium/factor_graph_engine.py or mycelium/kenken.py.
All intermediate capture is via the SAME monkeypatch the baseline probe used
(scripts/probe_svd_collapse.py): breathing._layernorm (readout LN only, gamma IS
model.ln_f_g) for per-breath reps, and factor_graph_engine.kenken_layer_forward for
per-head context. Both modules' forwards on the multi-task path call the IDENTICAL
readout LN + kenken_layer_forward (the multi-task batch only changes the attn_bias
BUILDER via batch.head_type_oh, not the layer/readout calls), so the hooks transfer
verbatim.

MULTI-TASK MODEL BUILD. Reuses scripts/factor_graph_train.py end-to-end so the
per-domain head allocation + the generic gated inlet wiring match training EXACTLY:
  * load_breathing + cast_layers_fp32                (the Pythia-410M backbone)
  * attach_factor_inlet_params                       (the generic semantics inlet)
  * attach_factor_graph_params with the UNIFIED spec (s_max=49, n_values=7,
    n_factor_types=N_GLOBAL_TYPES=8, has_factor_inlet=True)
  * _build_multitask_adapter(domain, ...)            (per-domain -> _MultiTaskBatch;
    REMAPs latent_type to GLOBAL ids, pads membership/vdm, builds the per-batch
    NATIVE head allocation tensors, emits the inlet op/target/size ids)
  * load_ckpt                                        (the train script's loader; loads
    backbone + fg_* + generic inlet keys)
The generic inlet is built EAGERLY per batch (build_generic_factor_inlet) exactly as
evaluate_multitask does, so the TRAINED, GATE-OPENED inlet params are read at probe
time too (the gated inlet is part of what the multi-task backbone learned).

THE THREE MEASUREMENTS (per domain on the multi-task ckpt; vs the single-domain baseline):
  1. BREATH rank/collapse: per-breath effective rank (singular-value entropy) of the
     valid-cell readout-LN reps + consec-breath cosine across K=16. The KEY delta:
     does the multi-task breath cone collapse FURTHER (lower eff_rank) or go static
     EARLIER than the single-domain baseline?
  2. WITHIN-GROUP HEAD redundancy: per-relation-group eff-rank of per-head context,
     under each domain's NATIVE multi-task allocation (coloring 15 edge-heads;
     kenken 5 row / 5 col / 5 cage). Grouped by native_head_alloc_for_present_types
     (the SAME allocation the engine routes on), NOT the union cell_mp_head_allocation.
  3. CODEBOOK collinearity: the UNIVERSAL N=7 codebook -- off-diag cosine gram +
     eff-rank, read ONCE (domain-independent; it is one shared tensor). SPECIFICALLY
     the SHARED LOW ROWS: rows 0-1 (all 3 domains), row 2 (coloring k=3 + kenken),
     rows 3-6 (kenken-only). Have the shared rows COLLINEARIZED / over-specialized?

THE CATHEDRAL TRIGGER. Per domain, per breath, the VALID-UNOBSERVED-cell accuracy
(argmax==gold). A FORGETTING signature = a breath where accuracy RISES then FALLS
(resolve-then-degrade), or a breath cone that collapses harder than the single-domain
baseline (the backbone forgetting under multi-domain load). Reported EXPLICITLY:
FORGETTING SIGNATURE PRESENT (cathedral trigger FIRES) / ABSENT (cathedral parked).

SELFTEST (CPU, GPU-free): reuses scripts/probe_svd_collapse.selftest (the same
effective-rank + cosine-gram validation) AND adds a multi-task-specific check on the
cathedral resolve-then-degrade detector (synthetic monotone-up vs up-then-down curves).

USAGE:
  CPU selftest (GPU-free):
    SELFTEST_ONLY=1 .venv/bin/python3 scripts/probe_svd_collapse_multitask.py
  GPU run on the multi-task ckpt (AMD):
    DEV=AMD .venv/bin/python3 scripts/probe_svd_collapse_multitask.py
  One domain:
    DEV=AMD PROBE_ONLY=kenken .venv/bin/python3 scripts/probe_svd_collapse_multitask.py

Env vars:
  SELFTEST_ONLY  1 -> CPU selftest only (default 0).
  FG_CKPT        multi-task ckpt path
                 (default .cache/fg_ckpts/fg_multi_fair/fg_multi_fair_final.safetensors).
  PROBE_ONLY     coloring | circuit | kenken | "" (all) — default "".
  FG_MIX         domain set for the model build (default coloring,circuit,kenken — must
                 include every domain probed; L_max + adapters are built from it).
  PROBE_N_INST   instances to sample per domain (default 64).
  PROBE_BATCH / EVAL_BATCH  eval batch size (default 8).
  K              breaths (default 16; the trained K_max).
  PROBE_HEAD_LAYER / PROBE_HEAD_BREATH  capture layer/breath for per-head ctx
                 (default layer 3, last breath).
  PROBE_MAX_ROWS rows cap for SVD cost (default 4000).
  FG_TRAIN / FG_TEST  kenken corpora (default .cache/kenken_{train,test}.jsonl).
"""
from __future__ import annotations

import ast
import os
import sys

_THIS_FILE = os.path.abspath(__file__)
_ROOT = os.path.dirname(os.path.dirname(_THIS_FILE))
sys.path.insert(0, _ROOT)

import numpy as np

# Reuse the validated numerical machinery + reporting from the baseline probe.
import scripts.probe_svd_collapse as base


# ===========================================================================
# ast.parse gate — always runs, even on CPU
# ===========================================================================

def _ast_parse_ok() -> bool:
    with open(_THIS_FILE) as f:
        src = f.read()
    try:
        ast.parse(src)
        return True
    except SyntaxError as e:
        print(f"[ast.parse] FAILED: {e}", flush=True)
        return False


# ===========================================================================
# Cathedral resolve-then-degrade detector (the forgetting signature)
# ===========================================================================

def resolve_then_degrade(curve: list[float], tol: float = 0.0) -> dict:
    """Detect a resolve-then-degrade (rise-then-fall) signature in a per-breath curve.

    The cathedral (cross-breath memory) is gated on a MEASURED forgetting signature:
    a quantity that RESOLVES (rises to a peak) then DEGRADES (falls back) across
    breaths -- the single residual cannot HOLD a resolved intermediate. Clean
    iterative-solver settling is MONOTONE-up-then-flat (a fixed point), NOT a peak.

    Returns a dict with:
      peak_idx   : argmax breath.
      peak       : max value.
      last       : final value.
      drop       : peak - last (how much was forgotten after the peak).
      drop_frac  : drop / max(peak, eps).
      forgetting : True if drop > tol AND the peak is strictly interior
                   (0 < peak_idx < len-1) — i.e. it rose, peaked, then fell.
    """
    c = [float(x) for x in curve]
    n = len(c)
    if n < 3:
        return {"peak_idx": 0, "peak": (c[0] if c else 0.0),
                "last": (c[-1] if c else 0.0), "drop": 0.0,
                "drop_frac": 0.0, "forgetting": False}
    peak_idx = int(np.argmax(c))
    peak = c[peak_idx]
    last = c[-1]
    drop = peak - last
    drop_frac = drop / max(abs(peak), 1e-9)
    # Interior peak that then falls = resolve-then-degrade.
    interior = 0 < peak_idx < (n - 1)
    forgetting = bool(interior and drop > tol)
    return {"peak_idx": peak_idx, "peak": peak, "last": last,
            "drop": drop, "drop_frac": drop_frac, "forgetting": forgetting}


# ===========================================================================
# SELFTEST (CPU) — reuse the baseline selftest + the cathedral detector check
# ===========================================================================

def selftest() -> bool:
    print("=== probe_svd_collapse_multitask SELFTEST (CPU) ===", flush=True)
    # 1. The full effective-rank / cosine-gram machinery (delegated to the baseline).
    ok = base.selftest()

    # 2. The cathedral resolve-then-degrade detector.
    print("\n  --- cathedral resolve-then-degrade detector ---", flush=True)
    # Monotone-up-then-flat (a clean fixed point): NO forgetting.
    mono = [0.10, 0.40, 0.70, 0.85, 0.90, 0.90, 0.90]
    r_mono = resolve_then_degrade(mono)
    print(f"  [monotone settle] peak@{r_mono['peak_idx']} drop={r_mono['drop']:.3f} "
          f"forgetting={r_mono['forgetting']} (expect False)", flush=True)
    cond_a = (not r_mono["forgetting"])
    ok &= cond_a

    # Rise-then-fall (resolve then degrade): forgetting PRESENT.
    updown = [0.10, 0.55, 0.88, 0.90, 0.72, 0.61, 0.55]
    r_ud = resolve_then_degrade(updown)
    print(f"  [resolve->degrade] peak@{r_ud['peak_idx']} drop={r_ud['drop']:.3f} "
          f"forgetting={r_ud['forgetting']} (expect True; peak interior)", flush=True)
    cond_b = (r_ud["forgetting"] and r_ud["peak_idx"] == 3
              and abs(r_ud["drop"] - 0.35) < 1e-6)
    ok &= cond_b

    # Peak at the LAST breath = still resolving, NOT degrading.
    last_peak = [0.10, 0.30, 0.55, 0.70, 0.82, 0.91, 0.95]
    r_lp = resolve_then_degrade(last_peak)
    print(f"  [peak-at-last] peak@{r_lp['peak_idx']} forgetting={r_lp['forgetting']} "
          f"(expect False; peak not interior)", flush=True)
    cond_c = (not r_lp["forgetting"])
    ok &= cond_c

    print(f"\n  MULTITASK SELFTEST {'PASSED' if ok else 'FAILED'}", flush=True)
    return ok


# ===========================================================================
# Multi-task model build (reuse scripts/factor_graph_train.py)
# ===========================================================================

def _build_multitask_model(K: int, mix: list[str], train_path: str, test_path: str,
                           BATCH: int, EVAL_BATCH: int, SEED: int):
    """Build the multi-task model + per-domain adapters EXACTLY as the trainer does.

    Returns (model, spec, adapters, eval_loaders, n_heads, head_dim, cfg).
    """
    import gc
    from tinygrad import Device, Tensor

    import scripts.factor_graph_train as fgt
    from mycelium import Config
    from mycelium.loader import _load_state, load_breathing
    from mycelium.factor_graph_engine import (
        attach_factor_graph_params, FG_HYP_MASK,
    )
    from mycelium.factor_inlet import attach_factor_inlet_params, N_GLOBAL_TYPES

    if FG_HYP_MASK:
        raise SystemExit(
            "FG_HYP_MASK=1 is incompatible with the multi-task probe (hyperbolic masks "
            "are single-domain Tier-2 research; the multi-task ckpt was trained "
            "FG_HYP_MASK=0). Re-run with FG_HYP_MASK=0.")

    cfg = Config()
    n_heads = cfg.n_heads
    hidden = cfg.hidden
    head_dim = hidden // n_heads

    print(f"loading Pythia-410M -> breathing transformer ...", flush=True)
    sd = _load_state()
    model = load_breathing(cfg, sd=sd)
    del sd
    gc.collect()
    fgt.cast_layers_fp32(model)

    # Generic inlet params FIRST (the adapters call build_generic_factor_inlet on model).
    attach_factor_inlet_params(model, hidden=hidden)
    print(f"  [multi] generic inlet attached (N_GLOBAL_TYPES={N_GLOBAL_TYPES})",
          flush=True)

    # Build the multi-task task (the unified spec + per-domain adapters + eval loaders).
    task = fgt._build_multitask_task(
        K, BATCH, EVAL_BATCH, SEED, hidden, n_heads, model,
        train_path, test_path, mix, {m: 1.0 for m in mix})
    spec = task.spec
    print(f"  [multi] unified spec: s_max={spec.s_max} N={spec.n_values} "
          f"T={spec.n_factor_types} inlet={spec.has_factor_inlet} L_max={task.L_max}",
          flush=True)

    # The general factor-graph params (fg_*).
    attach_factor_graph_params(model, hidden=hidden, spec=spec)
    Device[Device.DEFAULT].synchronize()

    fgt.model = model  # the trainer's eval helpers read it as a module global.
    return (model, spec, task.adapters, task.eval_loaders, n_heads, head_dim,
            cfg, task)


# ===========================================================================
# Per-domain GPU probe (non-invasive capture; reuse the baseline hooks)
# ===========================================================================

def _present_global_types_for(domain: str, eval_loaders: dict):
    """The GLOBAL factor-type ids present in a pure single-domain batch of `domain`.

    Mirrors factor_graph_train._present_global_types (re-derived here so the head
    grouping uses the SAME allocation the engine routes on). KenKen's eval loader is the
    _KKWrap; circuit's gate_types come from the underlying CircuitLoader.
    """
    from mycelium.factor_inlet import GLOBAL_TYPE_IDS
    if domain == "coloring":
        return [GLOBAL_TYPE_IDS["coloring_edge"]]
    if domain == "kenken":
        return [GLOBAL_TYPE_IDS["kenken_row"], GLOBAL_TYPE_IDS["kenken_col"],
                GLOBAL_TYPE_IDS["kenken_cage"]]
    if domain == "circuit":
        loader = eval_loaders["circuit"]
        gate_types = tuple(getattr(loader, "gate_types"))
        from mycelium.factor_inlet import GLOBAL_TYPE_IDS as G
        return [G[f"circuit_{g.lower()}"] for g in gate_types]
    raise ValueError(f"unknown domain {domain!r}")


def _relation_names_mt(domain: str, present: list[int]) -> dict[int, str]:
    """GLOBAL-type-id -> human name, for the head-group report."""
    inv = {
        0: "coloring_edge", 1: "circuit_and", 2: "circuit_or", 3: "circuit_not",
        4: "circuit_xor", 5: "kenken_row", 6: "kenken_col", 7: "kenken_cage",
    }
    from mycelium.factor_masks import CELL_MP_HEAD_GLOBAL
    names = {gid: inv.get(gid, f"type{gid}") for gid in present}
    names[CELL_MP_HEAD_GLOBAL] = "global"
    return names


def run_gpu_probe_multitask(domain: str, model, spec, adapter, eval_loader,
                            n_heads: int, head_dim: int, cfg, K: int) -> dict:
    """Run the breath loop on a sample of `domain` instances; capture reps + per-head ctx.

    Non-invasive: install the SAME two monkeypatches the baseline probe uses
    (breathing._layernorm readout-only, factor_graph_engine.kenken_layer_forward) and
    additionally track per-breath valid-unobserved-cell accuracy (cathedral trigger).
    """
    import math
    from tinygrad import Tensor, dtypes, Device
    import mycelium.breathing as breathing_mod
    import mycelium.factor_graph_engine as fge
    from mycelium.factor_graph_engine import factor_breathing_forward
    from mycelium.factor_inlet import build_generic_factor_inlet
    from mycelium.factor_masks import (
        native_head_alloc_for_present_types,
    )

    HEAD_LAYER = int(os.environ.get("PROBE_HEAD_LAYER", "3"))
    HEAD_BREATH = int(os.environ.get("PROBE_HEAD_BREATH", str(K - 1)))
    N_INST = int(os.environ.get("PROBE_N_INST", "64"))
    MAX_ROWS = int(os.environ.get("PROBE_MAX_ROWS", "4000"))

    print(f"\n{'#'*70}\n#  MULTI-TASK GPU PROBE: domain={domain}\n{'#'*70}", flush=True)

    # Per-domain NATIVE head->GLOBAL-type allocation (the SAME the engine routes on).
    present = _present_global_types_for(domain, {"circuit": eval_loader}
                                        if domain == "circuit" else {})
    alloc = native_head_alloc_for_present_types(present, n_heads)   # (H,) global ids / -1
    rel_name = _relation_names_mt(domain, present)
    print(f"  present global types: {present}", flush=True)
    print(f"  native head alloc (head->global type, -1=global): {alloc.tolist()}",
          flush=True)

    Tensor.training = False

    # ---- HOOK A: per-breath readout-LN reps (gamma IS model.ln_f_g). ----
    orig_layernorm = breathing_mod._layernorm
    breath_capture: list[np.ndarray] = []

    def _patched_layernorm(x, gamma, beta, eps=1e-5):
        out = orig_layernorm(x, gamma, beta, eps)
        if gamma is model.ln_f_g:
            breath_capture.append(out.cast(dtypes.float).realize().numpy())
        return out

    # ---- HOOK B: per-head ctx at (HEAD_LAYER, HEAD_BREATH). ----
    orig_kenken_layer = fge.kenken_layer_forward
    layer_index = {id(L): i for i, L in enumerate(model.block.layers)}
    call_state = {"breath": 0, "layer_calls": 0}
    head_capture: dict = {"ctx": None}

    def _patched_kenken_layer(layer, x, attn_bias, q_rot_cos=None, q_rot_sin=None):
        li = layer_index.get(id(layer), -1)
        if li == 0:
            call_state["breath"] = call_state["layer_calls"] // 4
            call_state["layer_calls"] += 1
        else:
            call_state["layer_calls"] += 1
        if (li == HEAD_LAYER and call_state["breath"] == HEAD_BREATH
                and head_capture["ctx"] is None):
            cfgl = layer.cfg
            B, Sx, Hx = x.shape
            attn_in = orig_layernorm(x, layer.shared.in_ln_g,
                                     layer.shared.in_ln_b, cfgl.layer_norm_eps)
            attn_in_dt = attn_in.cast(x.dtype) if attn_in.dtype != x.dtype else attn_in
            q = (attn_in_dt @ layer.wq + layer.bq).reshape(
                B, Sx, cfgl.n_heads, cfgl.head_dim).transpose(1, 2)
            k = (attn_in_dt @ layer.wk + layer.bk).reshape(
                B, Sx, cfgl.n_heads, cfgl.head_dim).transpose(1, 2)
            v = (attn_in_dt @ layer.shared.wv + layer.shared.bv).reshape(
                B, Sx, cfgl.n_heads, cfgl.head_dim).transpose(1, 2)
            scale = 1.0 / math.sqrt(cfgl.head_dim)
            scores = q @ k.transpose(-2, -1) * scale
            scores = scores + attn_bias.cast(scores.dtype)
            attn = scores.clip(-1e4, 1e4).softmax(-1)
            ctx = (attn @ v).transpose(1, 2)
            head_capture["ctx"] = ctx.cast(dtypes.float).realize().numpy()
        return orig_kenken_layer(layer, x, attn_bias, q_rot_cos, q_rot_sin)

    # accumulators
    per_breath_rows: list[list[np.ndarray]] = [[] for _ in range(K)]
    head_group_rows: dict[int, list[np.ndarray]] = {}
    # cathedral: per-breath correct-cell + total-cell counts over valid-unobserved cells.
    pb_correct = np.zeros(K, dtype=np.float64)
    pb_total = np.zeros(K, dtype=np.float64)

    breathing_mod._layernorm = _patched_layernorm
    fge.kenken_layer_forward = _patched_kenken_layer
    try:
        done = 0
        for native in eval_loader.iter_eval():
            fb = adapter(native)
            # Build the generic inlet EAGERLY (reads the trained, gate-opened inlet) —
            # exactly as evaluate_multitask does.
            fb.factor_inlet = build_generic_factor_inlet(
                model, fb.membership, fb.latent_type, fb.cell_valid,
                op=fb.inlet_op, target=fb.inlet_target, size=fb.inlet_size).realize()

            breath_capture.clear()
            head_capture["ctx"] = None
            call_state["breath"] = 0
            call_state["layer_calls"] = 0

            logits_history, _ = factor_breathing_forward(model, fb, spec, K=K)
            assert len(breath_capture) == K, \
                f"expected {K} readout captures, got {len(breath_capture)}"

            cv = fb.cell_valid.realize().numpy()                     # (B, S)
            gold = fb.gold.realize().numpy().astype(np.int32)        # (B, S)
            observed = (fb.input_cells.realize().numpy().astype(np.int32) > 0)  # (B,S)
            B_cur = cv.shape[0]
            real = min(B_cur, N_INST - done)

            # cathedral: per-breath accuracy over valid & UNOBSERVED cells (the deduced
            # cells — the ones the breaths actually have to resolve).
            sup = (cv > 0.5) & (~observed)                          # (B, S)
            for kk in range(K):
                pred_k = (logits_history[kk].argmax(axis=-1) + 1
                          ).realize().numpy().astype(np.int32)       # (B, S)
                eq_k = (pred_k == gold) & sup
                pb_correct[kk] += float(eq_k[:real].sum())
                pb_total[kk] += float(sup[:real].sum())

            for bi in range(real):
                valid = cv[bi] > 0.5
                if not np.any(valid):
                    continue
                for kk in range(K):
                    per_breath_rows[kk].append(breath_capture[kk][bi][valid])

            ctx = head_capture["ctx"]
            if ctx is not None:
                for bi in range(real):
                    valid = cv[bi] > 0.5
                    if not np.any(valid):
                        continue
                    for hh in range(n_heads):
                        rel = int(alloc[hh])
                        head_group_rows.setdefault(rel, []).append(
                            ctx[bi, valid, hh, :])

            done += real
            if done >= N_INST:
                break
    finally:
        breathing_mod._layernorm = orig_layernorm
        fge.kenken_layer_forward = orig_kenken_layer

    print(f"  captured {done} instances.", flush=True)

    # ---- measurement 2 (this domain): breath rank / collapse ----
    breath_eff: list[float] = []
    breath_frac: list[float] = []
    full_breath = [np.concatenate(per_breath_rows[kk], axis=0) for kk in range(K)]
    for kk in range(K):
        stk = full_breath[kk]
        if stk.shape[0] > MAX_ROWS:
            idx = np.linspace(0, stk.shape[0] - 1, MAX_ROWS).astype(int)
            stk = stk[idx]
        eff, _, _ = base.effective_rank(stk)
        max_rank = min(stk.shape[0], stk.shape[1])
        breath_eff.append(eff)
        breath_frac.append(eff / max(max_rank, 1))
    consec = [base.consecutive_cosine(full_breath[kk], full_breath[kk + 1])
              for kk in range(K - 1)]
    breath_max_rank = min(full_breath[0].shape[0], full_breath[0].shape[1])
    breath_res = {
        "eff": breath_eff, "frac": breath_frac, "consec": consec,
        "max_rank": breath_max_rank, "H": cfg.hidden,
        "n_rows": full_breath[0].shape[0],
    }

    # ---- measurement 3 (this domain): within-group head redundancy ----
    head_res = base._compute_head_groups(head_group_rows, alloc, n_heads, head_dim,
                                         rel_name)

    # ---- cathedral: per-breath accuracy curve + forgetting read ----
    pb_acc = (pb_correct / np.maximum(pb_total, 1.0)).tolist()
    cathedral = resolve_then_degrade(pb_acc)
    cathedral["curve"] = pb_acc
    cathedral["n_deduced_cells"] = float(pb_total[0])

    return {
        "domain": domain, "K": K,
        "breath": breath_res, "head": head_res,
        "head_dim": head_dim, "n_heads": n_heads,
        "rel_name": rel_name, "alloc": alloc.tolist(),
        "cathedral": cathedral,
    }


# ===========================================================================
# Reporting
# ===========================================================================

# Single-domain baseline numbers (project_svd_collapse_baseline, commit a9a0de4).
_BASELINE = {
    "coloring": {
        "codebook_eff": 2.99, "codebook_max": 3,
        "breath_eff_first": 217.0, "breath_eff_last": 76.0,
        "breath_consec_settle": "0.988 -> 1.000 by breath ~8-11",
        "head_frac": "94-99% of group rank", "head_meancos": "0.05-0.14",
        "head_group": "15 edge + 1 global",
    },
    "circuit": {
        "codebook_eff": 1.99, "codebook_max": 2,
        "breath_eff_first": None, "breath_eff_last": None,
        "breath_consec_settle": "0.988 -> 1.000 by breath ~8-11",
        "head_frac": "94-99% of group rank", "head_meancos": "0.05-0.14",
        "head_group": "5 AND / 5 OR / 5 NOT + 1 global",
    },
    "kenken": {
        # The single-domain baseline probe did not run KenKen (the original CKPTS map
        # had coloring/circuit/circuit_deep). The KenKen codebook is N=7; no direct
        # single-domain rank to subtract -> report as new (the universal codebook IS the
        # KenKen-sized object).
        "codebook_eff": None, "codebook_max": 7,
        "breath_eff_first": None, "breath_eff_last": None,
        "breath_consec_settle": "(no single-domain KenKen baseline)",
        "head_frac": "(no single-domain KenKen baseline)", "head_meancos": "n/a",
        "head_group": "5 row / 5 col / 5 cage + 1 global",
    },
}


def _report_codebook(cb: dict) -> str:
    """Report the UNIVERSAL N=7 codebook + the shared-row collinearity verdict."""
    print(f"\n{'='*72}", flush=True)
    print(f"  UNIVERSAL CODEBOOK  (value_codebook N={cb['n']} x H={cb['h']})", flush=True)
    print(f"{'='*72}", flush=True)
    print(f"  singular values : {np.array2string(cb['sv'], precision=4)}", flush=True)
    print(f"  effective rank  = {cb['eff_rank']:.4f}  (max = N = {cb['n']})", flush=True)
    print(f"  hard rank       = {cb['hard_rank']}", flush=True)
    print(f"  max |off-diag cos| = {cb['max_offcos']:.4f}", flush=True)
    print(f"  mean|off-diag cos| = {cb['mean_offcos']:.4f}", flush=True)
    g = cb["gram"]
    print(f"  off-diag cosine gram (rows are value indices 0..{cb['n']-1}):", flush=True)
    for i in range(g.shape[0]):
        rowstr = "  ".join(f"{g[i, j]:+.3f}" for j in range(g.shape[1]))
        print(f"    [{i}] {rowstr}", flush=True)

    # Shared-row analysis: rows 0-1 (all 3 domains), row 2 (coloring k=3 + kenken),
    # rows 3-6 (kenken-only). Are the SHARED rows collinearized vs the kenken-only rows?
    n = cb["n"]
    shared_rows = [r for r in (0, 1, 2) if r < n]      # used by >=2 domains
    kk_only = [r for r in (3, 4, 5, 6) if r < n]       # kenken-only
    print(f"\n  SHARED-ROW analysis (rows used by >=2 domains: {shared_rows}; "
          f"kenken-only: {kk_only}):", flush=True)

    def _max_abs_offdiag_for(rows: list[int]) -> tuple[float, float, str]:
        if len(rows) < 2:
            return 0.0, 0.0, "(only one shared row -> no within-shared off-diagonal)"
        sub = np.abs(g[np.ix_(rows, rows)])
        iu = np.triu_indices(len(rows), k=1)
        vals = sub[iu]
        # find the most-collinear shared pair
        pi = int(np.argmax(vals))
        pairs = list(zip(*np.triu_indices(len(rows), k=1)))
        ri, rj = pairs[pi]
        return float(vals.max()), float(vals.mean()), \
            f"most-collinear shared pair = ({rows[ri]},{rows[rj]}) |cos|={vals.max():.3f}"

    sh_max, sh_mean, sh_desc = _max_abs_offdiag_for(shared_rows)
    print(f"    within-SHARED |cos|: max={sh_max:.3f} mean={sh_mean:.3f}  {sh_desc}",
          flush=True)
    if kk_only:
        kk_max, kk_mean, kk_desc = _max_abs_offdiag_for(kk_only)
        print(f"    within-KENKEN-ONLY |cos|: max={kk_max:.3f} mean={kk_mean:.3f}  "
              f"{kk_desc}", flush=True)
    # cross block: shared vs kenken-only
    if shared_rows and kk_only:
        cross = np.abs(g[np.ix_(shared_rows, kk_only)])
        print(f"    SHARED x KENKEN-ONLY |cos|: max={cross.max():.3f} "
              f"mean={cross.mean():.3f}", flush=True)

    frac = cb["eff_rank"] / max(cb["n"], 1)
    # Shared-row collinearity verdict.
    shared_collinear = sh_max > 0.5
    full_rank = frac >= 0.75 and cb["mean_offcos"] <= 0.5
    if shared_collinear:
        sr_verdict = (f"SHARED ROWS COLLINEARIZED (max within-shared |cos|={sh_max:.3f} "
                      f"> 0.5 -> shared low rows over-specialized / collapsed together)")
    else:
        sr_verdict = (f"SHARED ROWS STILL DISTINCT (max within-shared |cos|={sh_max:.3f} "
                      f"<= 0.5 -> the universal low rows did NOT collinearize)")
    full_verdict = base._verdict_codebook(cb["eff_rank"], cb["n"], cb["h"],
                                          cb["mean_offcos"])
    print(f"\n  VERDICT (full codebook): {full_verdict}", flush=True)
    print(f"  VERDICT (shared rows)  : {sr_verdict}", flush=True)
    return f"full={full_verdict} | shared={sr_verdict}"


def _report_domain(res: dict) -> dict:
    domain = res["domain"]
    br = res["breath"]
    hd = res["head"]
    K = res["K"]
    head_dim = res["head_dim"]
    cath = res["cathedral"]
    bl = _BASELINE.get(domain, {})

    print(f"\n{'='*72}", flush=True)
    print(f"  DOMAIN: {domain}  (multi-task ckpt)", flush=True)
    print(f"{'='*72}", flush=True)

    # ---- [2] breath ----
    print(f"\n  [BREATH] rank / collapse  (K={K}; {br['n_rows']} valid-cell rows, "
          f"H={br['H']}, max_rank={br['max_rank']})", flush=True)
    print(f"    per-breath effective rank (B0..B{K-1}):", flush=True)
    for r0 in range(0, K, 8):
        chunk = br["eff"][r0:r0 + 8]
        fr = br["frac"][r0:r0 + 8]
        s = "  ".join(f"B{r0+i}={chunk[i]:.1f}({fr[i]:.0%})" for i in range(len(chunk)))
        print(f"      {s}", flush=True)
    print(f"    consecutive-breath cosine (B0->B1 .. B{K-2}->B{K-1}):", flush=True)
    for r0 in range(0, K - 1, 8):
        chunk = br["consec"][r0:r0 + 8]
        s = "  ".join(f"{r0+i}:{chunk[i]:.3f}" for i in range(len(chunk)))
        print(f"      {s}", flush=True)
    br_verdict = base._verdict_breath(br["eff"], br["max_rank"], br["consec"])
    print(f"    VERDICT: {br_verdict}", flush=True)
    # Delta vs single-domain baseline.
    if bl.get("breath_eff_first") is not None:
        d_first = bl["breath_eff_first"] - br["eff"][0]
        d_last = bl["breath_eff_last"] - br["eff"][-1]
        print(f"    DELTA vs single-domain baseline (positive = multi-task collapsed "
              f"FURTHER):", flush=True)
        print(f"      first-breath eff_rank: single={bl['breath_eff_first']:.0f} "
              f"multi={br['eff'][0]:.0f}  delta={d_first:+.0f}", flush=True)
        print(f"      last-breath  eff_rank: single={bl['breath_eff_last']:.0f} "
              f"multi={br['eff'][-1]:.0f}  delta={d_last:+.0f}", flush=True)
    print(f"    baseline settle: {bl.get('breath_consec_settle','n/a')}", flush=True)

    # ---- [3] heads ----
    print(f"\n  [HEADS] within-group redundancy  (native alloc; head_dim={head_dim})",
          flush=True)
    print(f"    baseline single-domain heads: {bl.get('head_frac','n/a')}, "
          f"mean|cos| {bl.get('head_meancos','n/a')} ({bl.get('head_group','')})",
          flush=True)
    head_verdicts = {}
    for rel in sorted(hd.keys()):
        h = hd[rel]
        print(f"    group '{h['name']}' (heads {h['heads']}, n={h['n_heads']}):",
              flush=True)
        print(f"      eff_rank over per-head mean vectors = {h['eff_rank_headmeans']:.3f} "
              f"(max = {h['max_rank_headmeans']})", flush=True)
        print(f"      mean |pairwise cos| (head means)    = {h['mean_abscos_headmeans']:.4f}"
              f"  (max {h['max_abscos_headmeans']:.4f})", flush=True)
        v = base._verdict_head_group(h["eff_rank_headmeans"], h["n_heads"], head_dim,
                                     h["mean_abscos_headmeans"])
        head_verdicts[rel] = v
        print(f"      VERDICT: {v}", flush=True)

    # ---- cathedral trigger ----
    print(f"\n  [CATHEDRAL TRIGGER] per-breath valid-UNOBSERVED-cell accuracy "
          f"({int(cath['n_deduced_cells'])} deduced cells):", flush=True)
    curve = cath["curve"]
    for r0 in range(0, K, 8):
        chunk = curve[r0:r0 + 8]
        s = "  ".join(f"B{r0+i}={chunk[i]:.3f}" for i in range(len(chunk)))
        print(f"      {s}", flush=True)
    print(f"    peak @ B{cath['peak_idx']} = {cath['peak']:.3f}; "
          f"final B{K-1} = {cath['last']:.3f}; "
          f"post-peak drop = {cath['drop']:.3f} ({cath['drop_frac']:.0%})", flush=True)
    if cath["forgetting"]:
        cath_verdict = (f"FORGETTING SIGNATURE PRESENT (resolve-then-degrade: accuracy "
                        f"peaked at B{cath['peak_idx']}={cath['peak']:.3f} then fell to "
                        f"B{K-1}={cath['last']:.3f}, drop {cath['drop']:.3f}) -> "
                        f"CATHEDRAL TRIGGER FIRES")
    else:
        cath_verdict = (f"FORGETTING SIGNATURE ABSENT (no interior-peak resolve-then-"
                        f"degrade; accuracy {'monotone/settling' if cath['drop']<=0 else 'peak-at-last'}, "
                        f"peak@B{cath['peak_idx']}) -> CATHEDRAL TRIGGER does NOT fire")
    print(f"    VERDICT: {cath_verdict}", flush=True)

    return {"breath": br_verdict, "head": head_verdicts,
            "cathedral": cath_verdict, "cathedral_fires": cath["forgetting"]}


# ===========================================================================
# Main
# ===========================================================================

def main():
    if int(os.environ.get("SELFTEST_ONLY", "0")) > 0:
        ok = selftest()
        sys.exit(0 if ok else 1)

    print("Running CPU selftest before GPU probe...", flush=True)
    if not selftest():
        print("SELFTEST FAILED — aborting GPU probe.", flush=True)
        sys.exit(1)

    from tinygrad import Tensor

    K = int(os.environ.get("K", os.environ.get("FG_K_MAX", "16")))
    BATCH = int(os.environ.get("PROBE_BATCH", os.environ.get("BATCH", "8")))
    EVAL_BATCH = int(os.environ.get("EVAL_BATCH", str(BATCH)))
    SEED = int(os.environ.get("SEED", "42"))
    CKPT = os.environ.get(
        "FG_CKPT",
        ".cache/fg_ckpts/fg_multi_fair/fg_multi_fair_final.safetensors")
    FG_TRAIN = os.environ.get("FG_TRAIN", ".cache/kenken_train.jsonl")
    FG_TEST = os.environ.get("FG_TEST", ".cache/kenken_test.jsonl")
    MIX = [m.strip().lower() for m in
           os.environ.get("FG_MIX", "coloring,circuit,kenken").split(",") if m.strip()]
    only = os.environ.get("PROBE_ONLY", "").strip().lower()
    probe_domains = [only] if only else list(MIX)
    for d in probe_domains:
        assert d in MIX, f"PROBE_ONLY={d} not in FG_MIX={MIX}"

    if not os.path.exists(CKPT):
        print(f"[abort] ckpt not found: {CKPT}", flush=True)
        sys.exit(1)

    import scripts.factor_graph_train as fgt

    np.random.seed(SEED)
    Tensor.manual_seed(SEED)

    (model, spec, adapters, eval_loaders, n_heads, head_dim, cfg, task) = \
        _build_multitask_model(K, MIX, FG_TRAIN, FG_TEST, BATCH, EVAL_BATCH, SEED)

    print(f"\nloading multi-task checkpoint: {CKPT}", flush=True)
    fgt.load_ckpt(model, CKPT)
    Tensor.training = False

    # ---- measurement 1: the UNIVERSAL codebook (read once; domain-independent). ----
    from tinygrad import dtypes
    cb_np = model.fg_value_codebook.realize().numpy().astype(np.float64)
    cb_res = base.probe_codebook(cb_np, h=cfg.hidden)
    cb_verdict = _report_codebook(cb_res)

    # ---- per-domain breath + head + cathedral. ----
    all_verdicts = {}
    for d in probe_domains:
        res = run_gpu_probe_multitask(
            d, model, spec, adapters[d], eval_loaders[d],
            n_heads, head_dim, cfg, K)
        all_verdicts[d] = _report_domain(res)

    # ---- overall read ----
    print(f"\n{'='*72}", flush=True)
    print(f"  OVERALL — MULTI-TASK vs SINGLE-DOMAIN BASELINE", flush=True)
    print(f"{'='*72}", flush=True)
    print(f"\n  UNIVERSAL CODEBOOK: {cb_verdict}", flush=True)
    any_fires = False
    for d, v in all_verdicts.items():
        print(f"\n  {d}:", flush=True)
        print(f"    breath   : {v['breath']}", flush=True)
        for rel, hv in v["head"].items():
            print(f"    head[{rel}]: {hv}", flush=True)
        print(f"    cathedral: {v['cathedral']}", flush=True)
        any_fires = any_fires or v["cathedral_fires"]
    print(f"\n  CATHEDRAL TRIGGER (any domain): "
          f"{'FIRES — forgetting signature present' if any_fires else 'PARKED — no forgetting signature'}",
          flush=True)
    print(flush=True)


if __name__ == "__main__":
    ok = _ast_parse_ok()
    print(f"[ast.parse] astparse_ok={ok}", flush=True)
    if not ok:
        sys.exit(1)
    main()
