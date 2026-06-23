"""diag_kenken_granularity_probe.py — the KenKen PER-BREATH GRANULARITY probe (SKELETON).

THE HYPOTHESIS (see docs/granularity_probe_spec.md for the full design)
======================================================================
The deducer runs K breaths through 4 SHARED Pythia L0-L3 layers, accumulating a 1024d
residual (the persistent factor-graph belief state). DOES that residual STRATIFY BY SCALE
ACROSS BREATHS? Three decision outcomes:

  (1) COARSE-EARLY + FINE-LATE wave  -> expand-collapse waist JUSTIFIED
      (puzzle-level structure decodable early; per-cell structure sharpens late).
  (2) V-CYCLE / OSCILLATING           -> oscillating (multigrid) waist JUSTIFIED
      (coarse peaks at k=0, dips at k~K/2, peaks again at k=K-1).
  (3) FLAT across all scales/breaths  -> REFUTED (no stratification; don't build a waist).
  (+) WEAK / INCONCLUSIVE fall-through, and a FRESH RADIAL-DEPTH control (depth orthogonal
      to the granularity axis?).

THE THREE GRANULARITY SCALES (decoded from the per-breath readout-LN residual)
=============================================================================
  COARSE / GLOBAL  : puzzle size N / difficulty band / #cages / #givens
                     -> from the POOLED (B, H) per-breath rep (mean over valid cells).
  MEDIUM / REGIONAL: cage ID per cell / constraint type
                     -> from the UN-POOLED (B, S, H) per-cell rep, masked by cell_valid.
  FINE / LOCAL     : per-cell gold value / per-cage arithmetic satisfaction
                     -> same per-cell reps, optionally cage-aggregated.
  Decodability metric per breath k: held-out by-instance CV logistic AUC; the TREND across
  k drives the verdict.

LEAK-FREE METHODOLOGY (reuses dart_cluster_probe machinery UNCHANGED)
====================================================================
  * BY-INSTANCE CV split (FIX 1): whole instances assigned to folds via assign_instance_folds
    -> no cell/breath of a train-instance leaks into the held-out fold.
  * per-breath probe loop: standardize on TRAIN stats per (fold, breath), _logreg_fit, score
    held-out, auc_mann_whitney; mean over folds -> auc_per_breath[k].
  * SHUFFLE NULL: same probe on shuffled labels -> expect ~0.5 at every k.

CAPTURE (non-invasive, eager-only; engine + oracle stay git-clean)
==================================================================
Monkeypatch mycelium.breathing._layernorm; record when gamma IS model.ln_f_g. Each breath
calls the readout LN exactly once (factor_graph_engine ~line 478), so APPENDING (not
overwriting, the one difference from _DartCapture) yields K x (B, S, H) per-breath reps.

STATUS: SKELETON. Full structure + stubs + arg parsing + the probe/eval scaffolding. The GPU
forward loop (build model, run K-breath eager forward, collect per-breath reps) is STUBBED
(raises NotImplementedError) — this file is IMPORT-CLEAN and ast.parse-clean; it does NOT run
a GPU job. Bryce/main-thread fires the GPU run once a KenKen ckpt exists (FG_TASK=kenken on
scripts/factor_graph_train.py would produce one; per CLAUDE.md no KenKen ckpt exists yet).

USAGE (when a KenKen ckpt exists — Bryce/main-thread runs this; agents do NOT):
  DEV=AMD \
  FG_CKPT=.cache/fg_ckpts/fg_kenken_k16/fg_kenken_k16_final.safetensors \
  .venv/bin/python3 scripts/diag_kenken_granularity_probe.py \
      --bands g40,g30,g20,g10 --per-band 50 --K 16 --eval-batch 8 --folds 5 --seed 0
"""
from __future__ import annotations

import argparse
import ast
import os
import sys
import time

_THIS_FILE = os.path.abspath(__file__)
_ROOT = os.path.dirname(os.path.dirname(_THIS_FILE))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.dirname(_THIS_FILE))  # so `import dart_cluster_probe` works

import numpy as np  # noqa: E402

# --- REUSE dart_cluster_probe's VALIDATED machinery (leak-free protocol depends on it) ---
# auc_mann_whitney   : the EXACT rank-based AUC (handles ties)
# _logreg_fit        : the tiny L2-regularized logistic regression (pure numpy GD)
# _center_by_instance: per-instance-mean centering (identity removal; optional here)
# _cv_logistic_auc   : the by-instance CV held-out AUC (FIX 1 baked in)
from dart_cluster_probe import (  # noqa: E402
    auc_mann_whitney,
    _logreg_fit,
    _center_by_instance,
    _cv_logistic_auc,
)
# assign_instance_folds: the WHOLE-instance fold split (no leakage), reused verbatim.
from learned_waist_gate import assign_instance_folds  # noqa: E402

# --- KenKen label constants (OP_VOCAB = ["given","add","sub","mul","div"], ids 0..4) ---
_OP_GIVEN, _OP_ADD, _OP_SUB, _OP_MUL, _OP_DIV = 0, 1, 2, 3, 4
# Difficulty band -> ordinal (g10 hard .. g40 easy); used as a COARSE categorical target.
_BAND_ORD = {"g10": 0, "g20": 1, "g30": 2, "g40": 3}


# ===========================================================================
# ast.parse gate — always runs (the import-clean contract for this skeleton)
# ===========================================================================

def _ast_parse_ok() -> bool:
    with open(_THIS_FILE) as f:
        src = f.read()
    try:
        ast.parse(src)
        return True
    except SyntaxError as e:  # pragma: no cover
        print(f"[ast.parse] FAILED: {e}", flush=True)
        return False


# ===========================================================================
# MODEL BUILD (KenKen) — STUB
# ===========================================================================
# amortized_frontier_measure._build_deducer_model is COLORING-SPECIFIC (imports
# scripts.search_coloring + GraphColoringLoader; --domain kenken raises NotImplementedError)
# AND KenKen needs has_factor_inlet=True (the verification inlet). So we provide a KenKen
# builder that mirrors the validated build path for the KenKen spec instead.

def build_kenken_spec(K: int):
    """Return the FactorGraphSpec for KenKen (s_max=49, N=7, T=3 row/col/cage, 16 heads,
    has_factor_inlet=True). K sets k_max.

    Mirrors the KenKen path: FactorGraphSpec(s_max=N_CELLS, n_values=N_MAX,
    n_factor_types=3, n_heads=16, k_max=K, has_factor_inlet=True).
    """
    from mycelium.factor_graph_engine import FactorGraphSpec
    from mycelium.kenken_data import N_MAX, N_CELLS
    return FactorGraphSpec(
        s_max=N_CELLS, n_values=N_MAX, n_factor_types=3,
        n_heads=16, k_max=int(K), has_factor_inlet=True,
    )


def build_kenken_deducer_model(spec, ckpt: str, seed: int):
    """Build the Pythia-410M breathing deducer for KenKen + load the ckpt.

    Mirrors the PARITY-VALIDATED build path in scripts/kenken_volume_eval.build_model:
      load_breathing(Config, sd=_load_state) -> cast_layers_fp32
      -> attach_kenken_params (the verification-inlet op/target/size tables + kenken_inlet_w/b)
      -> attach_factor_graph_params(model, hidden, spec) (the fg_* params)
      -> load_ckpt(model, ckpt).
    The order (kenken inlet params BEFORE fg params) matches factor_graph_train.main()'s
    kenken path, which is what the ckpt was saved against. K is taken from spec.k_max.

    Returns the model with factor_graph_* + kenken inlet params attached and ckpt restored.
    """
    from tinygrad import Device
    from mycelium import Config
    from mycelium.loader import _load_state, load_breathing
    from scripts.factor_graph_train import (
        cast_layers_fp32, load_ckpt, attach_factor_graph_params,
    )
    from mycelium.kenken import attach_kenken_params

    cfg = Config()
    print(f"[build] Pythia-410M -> breathing transformer (hidden={cfg.hidden} "
          f"n_heads={cfg.n_heads})", flush=True)
    sd = _load_state()
    model = load_breathing(cfg, sd=sd)
    del sd
    cast_layers_fp32(model)
    hidden, n_heads = cfg.hidden, cfg.n_heads
    K = int(spec.k_max)

    # kenken inlet tables FIRST (matches factor_graph_train.main()'s kenken path / volume_eval).
    attach_kenken_params(model, hidden=hidden, n_heads=n_heads, k_max=K)
    # the general factor-graph params (fg_*).
    attach_factor_graph_params(model, hidden=hidden, spec=spec)

    Device[Device.DEFAULT].synchronize()
    print(f"[build] resuming from fg ckpt: {ckpt}", flush=True)
    load_ckpt(model, ckpt)
    print("[build]   loaded.", flush=True)
    return model


# ===========================================================================
# CAPTURE — per-breath readout-LN hook (mirrors _DartCapture; APPENDS per breath)
# ===========================================================================

def install_readout_capture_hook(model):
    """Monkeypatch mycelium.breathing._layernorm to capture per-breath readout (B, S, H).

    THE ONE DIFFERENCE FROM _DartCapture: _DartCapture overwrites a single slot (keeps only
    the final breath); here we APPEND every readout-LN call (gamma IS model.ln_f_g) so a
    K-breath forward leaves K x (B, S, H) reps in slot['captures'] (one per breath, in order).

    Non-invasive + eager-only: the engine imports _layernorm locally inside
    factor_breathing_forward, so patching mycelium.breathing._layernorm intercepts the
    readout call without touching the engine. Returns a handle dict:
      {"uninstall": callable, "arm": callable, "slot": {"captures": [...]}}
    arm() resets the capture list before each forward.
    """
    import mycelium.breathing as breathing_mod
    from tinygrad import dtypes

    slot = {"captures": []}  # list of K x (B, S, H) numpy fp32 (one per breath)
    orig_ln = breathing_mod._layernorm

    def patched_ln(x, gamma, beta, eps=1e-5):
        out = orig_ln(x, gamma, beta, eps)
        if gamma is model.ln_f_g:                       # the readout LN (unambiguous)
            slot["captures"].append(out.cast(dtypes.float).realize().numpy())  # (B, S, H)
        return out

    breathing_mod._layernorm = patched_ln

    def _uninstall():
        breathing_mod._layernorm = orig_ln

    def _arm():
        slot["captures"] = []

    return {"uninstall": _uninstall, "arm": _arm, "slot": slot}


# ===========================================================================
# FORWARD — per-breath KenKen forward, collect per-breath reps via the hook
# ===========================================================================

def run_kenken_per_breath_forward(model, kb, spec, K, hook=None):
    """Run ONE eager K-breath KenKen forward; return per-breath RESIDUAL reps via the
    engine's fg_resid_capture sink (STEP 1 hook; mirrors fg_waist_capture).

    Builds the FactorGraphBatch from a KenKenBatch (make_kenken_factor_batch) WITH the
    verification inlet LIVE (build_verification_inlet, since KenKen sets has_factor_inlet=
    True), then calls factor_breathing_forward(model, batch, spec, K). The engine sink
    appends one (B, S, H) residual per breath into model.fg_resid_capture.

    EAGER ONLY (no JIT): the sink fires only on the eager forward path. `hook` is accepted
    for back-compat but the engine sink is the capture mechanism (no _layernorm patch needed).

    Returns:
      readout_reps_per_breath : (K, B, S, H) float32  [stacked from the engine sink]
      value_logits_history    : list of K (B, S, N) Tensors (for cross-check / argmax)
    """
    from tinygrad import Tensor
    from mycelium.factor_graph_engine import make_kenken_factor_batch, factor_breathing_forward
    from mycelium.kenken import build_verification_inlet

    Tensor.training = False

    # Build the verification inlet EAGERLY (must be LIVE — KenKen has_factor_inlet=True).
    inlet = build_verification_inlet(
        model, kb.cage_op, kb.cage_target, kb.cage_size, kb.cell_cage_id).realize()
    batch = make_kenken_factor_batch(kb, spec, prebuilt_inlet=inlet)

    # Arm the engine per-breath residual sink (STEP 1). Eager-only; None by default elsewhere.
    model.fg_resid_capture = []
    try:
        logits_hist, _calib = factor_breathing_forward(model, batch, spec, K)
        # Force realization of the forward (sink appends happen during the eager forward).
        _ = logits_hist[-1].realize()
        caps = list(model.fg_resid_capture)
    finally:
        model.fg_resid_capture = None  # disarm -> back to byte-identical no-op state.

    reps = np.stack(caps, axis=0).astype(np.float32)   # (K, B, S, H)
    assert reps.shape[0] == K, f"expected {K} per-breath captures, got {reps.shape[0]}"
    return reps, logits_hist


# ===========================================================================
# LABELS — extract instance-level + per-cell granularity labels from a KenKenBatch
# ===========================================================================

def extract_granularity_features(kb, spec):
    """Extract COARSE/MEDIUM/FINE labels + masks from a KenKenBatch.

    Reads the KenKenBatch python-side metadata (N, band, deduction_depth, n_givens) and the
    tensors (gold, cell_valid, cell_cage_id) and returns the label dict the probe consumes.

    Returns dict {
      'instance_id'    : (B,) int   — per-row stable id (offset-tagged across bands upstream)
      'band'           : (B,) str   — difficulty band ('g40'..'g10')
      'deduction_depth': (B,) int   — radial-depth control x-axis
      'n'              : (B,) int    — puzzle size N (3..7)
      'n_givens'       : (B,) int
      'coarse_label'   : (B,) int    — COARSE target (N by default; n_givens for the regression arm)
      'medium_label'   : (B, S) int  — cage_id per cell (-1 pad)
      'fine_label'     : (B, S) int  — gold value per cell (1..N, 0=unknown)
      'cell_valid'     : (B, S) bool
    }

    NOTE: kenken-specific extraction. The COARSE/MEDIUM/FINE label CHOICE is documented in
    docs/granularity_probe_spec.md s2; swap coarse_label to n_givens / #cages there.
    """
    B = int(kb.input_cells.shape[0])
    S = spec.s_max

    cell_valid = (kb.cell_valid > 0.5).realize().numpy()           # (B, S) bool
    gold = kb.gold.realize().numpy().astype(np.int64)              # (B, S) 1..N or 0
    input_cells = kb.input_cells.realize().numpy().astype(np.int64)  # (B, S) given val or 0
    is_given = (input_cells > 0)                                    # (B, S) bool — copied input
    cage_id = kb.cell_cage_id.realize().numpy().astype(np.int64)   # (B, S) cage id (-1 pad)
    cage_op = kb.cage_op.realize().numpy().astype(np.int64)        # (B, C) op id per cage
    cage_size = kb.cage_size.realize().numpy().astype(np.int64)    # (B, C) size per cage (0=pad)

    n = np.asarray(kb.N, dtype=np.int64)                          # (B,)
    deduction_depth = np.asarray(kb.deduction_depth, dtype=np.int64)
    band = np.asarray(kb.band, dtype=object)
    n_givens = np.asarray(kb.n_givens, dtype=np.int64)
    # #cages per puzzle = #cages with size > 0 (the COARSE #cages target).
    n_cages = (cage_size > 0).sum(axis=1).astype(np.int64)         # (B,)
    # band ordinal (g10..g40) — a COARSE categorical with variance across the bank.
    band_ord = np.array([_BAND_ORD.get(str(b), -1) for b in band], dtype=np.int64)

    # REGIONAL: per-cell cage OP id. Gather cage_op by the cell's cage id (-1 pad -> op 0).
    C = cage_op.shape[1]
    safe_cid = np.clip(cage_id, 0, C - 1)                          # (B, S)
    regional_op = np.take_along_axis(cage_op, safe_cid, axis=1)    # (B, S) op id per cell
    regional_op = np.where(cage_id >= 0, regional_op, -1)          # pad cells -> -1

    inst_id = np.arange(B, dtype=np.int64)  # caller offsets per band to keep ids global

    return {
        "instance_id": inst_id,
        "band": band,
        "band_ord": band_ord,
        "deduction_depth": deduction_depth,
        "n": n,
        "n_givens": n_givens,
        "n_cages": n_cages,
        "coarse_label": n,            # default COARSE target = puzzle size N
        "regional_label": regional_op,  # (B, S) cage OP id per cell (-1 pad)
        "local_label": gold,          # (B, S) gold value per cell (1..N, 0=unknown)
        "is_given": is_given,         # (B, S) bool — cell value copied from the input
        "cell_valid": cell_valid,     # (B, S) bool
    }


def _binarize_multiclass(labels, valid=None, seed=0):
    """Reduce a multiclass label array to a balanced BINARY contrast so the Mann-Whitney AUC
    machinery applies unchanged (printed in the table header for audit).

    Strategy (stable, documented): label > median(labels[valid]) -> positive. For cage_id /
    gold this is the 'high-id vs low-id' / 'high-value vs low-value' contrast; for N it is
    'large puzzle vs small'. A one-vs-rest sweep is the richer alternative (see spec s2).

    Returns (binary_labels bool, descr str).
    """
    labels = np.asarray(labels)
    sel = labels[valid] if valid is not None else labels
    if sel.size == 0:
        return np.zeros(labels.shape, dtype=bool), "empty"
    thr = float(np.median(sel))
    return (labels > thr), f"(> median {thr:g})"


# ===========================================================================
# PROBE — per-breath decodability of COARSE / MEDIUM / FINE (leak-free, by-instance CV)
# ===========================================================================

def _cv_logistic_auc_by_instance(reps, labels, fold_id, n_folds, inst_id_flat=None):
    """Per-breath by-instance CV held-out AUC for a SINGLE breath's reps.

    reps   : (n, H) standardize-per-fold inside; labels : (n,) bool; fold_id : (n,) int fold
    assignment (already by-instance via assign_instance_folds). For per-cell scales the rows
    are cells and `inst_id_flat` carries the per-cell instance id so the by-instance fold map
    can be re-derived at cell granularity (when fold_id was built at instance granularity).

    Mirrors _cv_logistic_auc's per-fold standardize -> _logreg_fit -> score -> pooled-OOF
    auc_mann_whitney, but takes a PRE-BUILT fold_id (so every breath/scale shares folds).

    Returns float AUC (nan if a class/fold is degenerate).
    """
    reps = np.asarray(reps, dtype=np.float64)
    labels = np.asarray(labels, dtype=bool)
    n = reps.shape[0]
    if n == 0 or labels.sum() == 0 or (~labels).sum() == 0:
        return float("nan")
    # If cell-level fold ids are needed, map them from inst_id_flat (built at instance level).
    if inst_id_flat is not None:
        # fold_id here is indexed by the (B,) instances; re-derive a cell-level fold per row.
        # Caller passes fold_id already aligned to inst_id_flat when sizes match; otherwise
        # this is the seam to translate inst_id_flat -> fold via the same assign map.
        fold_id = np.asarray(fold_id)
    oof = np.full(n, np.nan)
    for f in range(n_folds):
        te = fold_id == f
        tr = ~te
        if labels[tr].sum() == 0 or (~labels[tr]).sum() == 0 or te.sum() == 0:
            continue
        Xtr, Xte = reps[tr], reps[te]
        mu = Xtr.mean(axis=0, keepdims=True)
        sd = Xtr.std(axis=0, keepdims=True)
        sd = np.where(sd < 1e-8, 1.0, sd)
        Xtr = (Xtr - mu) / sd
        Xte = (Xte - mu) / sd
        w, b = _logreg_fit(Xtr, labels[tr], l2=1.0)
        oof[te] = Xte @ w + b
    ok = ~np.isnan(oof)
    if ok.sum() == 0 or labels[ok].sum() == 0 or (~labels[ok]).sum() == 0:
        return float("nan")
    return auc_mann_whitney(oof[ok], labels[ok])


def probe_per_breath_decodability(readout_reps_per_breath, labels_dict, spec, K,
                                  folds=5, seed=0, shuffle_null=False):
    """Probe decodability of COARSE/REGIONAL/LOCAL at each breath k (leak-free by-instance CV).

    readout_reps_per_breath : (K, B, S, H) per-breath per-cell RESIDUAL reps (engine sink).
    labels_dict             : from extract_granularity_features.
    shuffle_null            : if True, permute labels (independent of reps) -> expect ~0.5.

    Scales (each a binary contrast so auc_mann_whitney applies unchanged):
      COARSE  : pooled rep (mean over valid cells) -> {N>median, band-ordinal>median,
                #cages>median}. Three coarse targets; the headline COARSE = mean of the three.
      REGIONAL: per-cell rep (valid ARITHMETIC cells, op!=given) -> cage is MULTIPLICATIVE
                (mul/div) vs ADDITIVE (add/sub). Per-cell by-instance CV.
      LOCAL   : per-cell rep (valid cells) -> gold value > median. Per-cell by-instance CV.

    Returns: list of K dicts, each {'k', 'scales': {...}} with COARSE/COARSE_N/COARSE_BAND/
    COARSE_NCAGES/REGIONAL/LOCAL.
    """
    reps = np.asarray(readout_reps_per_breath)
    K_actual = reps.shape[0]
    assert K_actual == K, f"expected K={K} breaths, got {K_actual}"
    B = reps.shape[1]
    S = spec.s_max

    cell_valid = labels_dict["cell_valid"].astype(bool)            # (B, S)
    inst_id = labels_dict.get("instance_id", np.arange(B))
    rng = np.random.RandomState(seed)

    # by-INSTANCE fold map (shared across all breaths + scales).
    fold_id_inst, n_folds = assign_instance_folds(inst_id, folds=folds, seed=seed)

    # ---- COARSE labels (instance level) — three independent binary targets. ----
    n_bin, _ = _binarize_multiclass(labels_dict["n"])
    band_bin, _ = _binarize_multiclass(labels_dict["band_ord"])
    ncages_bin, _ = _binarize_multiclass(labels_dict["n_cages"])

    # ---- per-cell labels (REGIONAL + LOCAL) — recode over VALID cells only. ----
    valid_flat = cell_valid.reshape(-1)
    inst_flat = np.repeat(inst_id, S)
    fold_flat = np.repeat(fold_id_inst, S)

    # REGIONAL: op-type per cell. Binary = MULTIPLICATIVE (mul/div) vs ADDITIVE (add/sub).
    # Restrict to ARITHMETIC cells (op in {add,sub,mul,div}; exclude given/pad). This is the
    # meaningful regional contrast (given cells carry no cage-relation structure).
    op_flat = labels_dict["regional_label"].reshape(-1)            # (B*S,) op id (-1 pad)
    reg_is_arith = np.isin(op_flat, [_OP_ADD, _OP_SUB, _OP_MUL, _OP_DIV]) & valid_flat
    reg_bin = np.isin(op_flat, [_OP_MUL, _OP_DIV])                 # True = multiplicative

    # LOCAL: gold value per cell > median. Restrict to NON-GIVEN cells (input_cells==0) so
    # decodability reflects DEDUCED values, not values copied from the input (a given cell's
    # value is trivially in the residual at breath 0 via the input embed -> a confound the
    # non-given restriction removes). The median split is computed over the non-given valids.
    is_given_flat = labels_dict["is_given"].reshape(-1)
    local_eval_mask = valid_flat & (~is_given_flat)                # non-given, valid cells
    local_flat = labels_dict["local_label"].reshape(-1)
    local_bin, _ = _binarize_multiclass(local_flat, valid=local_eval_mask)

    if shuffle_null:
        # Permute each label set independently of the reps (the chance control).
        n_bin = n_bin[rng.permutation(n_bin.shape[0])]
        band_bin = band_bin[rng.permutation(band_bin.shape[0])]
        ncages_bin = ncages_bin[rng.permutation(ncages_bin.shape[0])]
        reg_bin = reg_bin[rng.permutation(reg_bin.shape[0])]
        local_bin = local_bin[rng.permutation(local_bin.shape[0])]

    def _nanmean(vals):
        vals = [v for v in vals if np.isfinite(v)]
        return float(np.mean(vals)) if vals else float("nan")

    results_per_breath = []
    for k in range(K):
        reps_k = reps[k]                                          # (B, S, H)
        scales = {}

        # COARSE: pool over valid cells -> (B, H).
        cv = cell_valid.astype(float)
        reps_coarse = ((reps_k * cv[:, :, None]).sum(axis=1)
                       / (cv.sum(axis=1, keepdims=True) + 1e-6))   # (B, H)
        auc_n = _cv_logistic_auc_by_instance(reps_coarse, n_bin, fold_id_inst, n_folds)
        auc_band = _cv_logistic_auc_by_instance(reps_coarse, band_bin, fold_id_inst, n_folds)
        auc_nc = _cv_logistic_auc_by_instance(reps_coarse, ncages_bin, fold_id_inst, n_folds)
        scales["COARSE_N"] = auc_n
        scales["COARSE_BAND"] = auc_band
        scales["COARSE_NCAGES"] = auc_nc
        scales["COARSE"] = _nanmean([auc_n, auc_band, auc_nc])

        # REGIONAL: per-cell, valid ARITHMETIC cells only.
        reps_cells = reps_k.reshape(-1, reps_k.shape[-1])          # (B*S, H)
        rmask = reg_is_arith
        scales["REGIONAL"] = _cv_logistic_auc_by_instance(
            reps_cells[rmask], reg_bin[rmask], fold_flat[rmask], n_folds,
            inst_id_flat=inst_flat[rmask])

        # LOCAL: per-cell, valid NON-GIVEN (deduced) cells.
        lmask = local_eval_mask
        scales["LOCAL"] = _cv_logistic_auc_by_instance(
            reps_cells[lmask], local_bin[lmask], fold_flat[lmask], n_folds,
            inst_id_flat=inst_flat[lmask])

        results_per_breath.append({"k": k, "scales": scales})
    return results_per_breath


# ===========================================================================
# RADIAL-DEPTH CONTROL — coarse-decodability slope vs deduction_depth (orthogonality)
# ===========================================================================

def _pearson(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size < 3 or a.std() < 1e-9 or b.std() < 1e-9:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def radial_depth_correlation(results_per_breath, labels_dict, K, settle_breath=None, seed=0):
    """RADIAL-DEPTH control (a distinct claim, re-tested fresh).

    The radial-depth deep-prize claim is that breath allocation is DEPTH-ORDERED. The cheap
    test: per-instance SETTLE BREATH (the breath at which the model's per-cell argmax stops
    changing, gold-free) vs per-instance deduction_depth. If rho is small (|rho|<0.30, with
    a clean shuffle null) the depth axis is ORTHOGONAL to the granularity dynamics (consistent
    with the refuted deep-prize, re-confirmed here).

    settle_breath : (N_inst,) per-instance settle breath in 0..K-1 (from the captured per-breath
                    argmax stability; None -> the control is skipped, rho=nan).

    Also reports the population COARSE slope (AUC_final - AUC_0) for context.
    """
    coarse = np.array([r["scales"].get("COARSE", np.nan) for r in results_per_breath])
    slope = float(coarse[-1] - coarse[0]) if coarse.size >= 2 else float("nan")
    depth = np.asarray(labels_dict.get("deduction_depth", []), dtype=float)

    depth_rho = float("nan")
    null_mean = float("nan")
    null_p95 = float("nan")
    if settle_breath is not None and depth.size == np.asarray(settle_breath).size:
        sb = np.asarray(settle_breath, dtype=float)
        depth_rho = _pearson(depth, sb)
        # shuffle null on the depth labels: |rho| distribution under permutation.
        rng = np.random.RandomState(seed)
        null = np.array([abs(_pearson(rng.permutation(depth), sb)) for _ in range(500)])
        null = null[np.isfinite(null)]
        if null.size:
            null_mean = float(null.mean())
            null_p95 = float(np.percentile(null, 95))
    bar = 0.30
    orthogonal = bool(np.isfinite(depth_rho) and abs(depth_rho) < bar)
    return {"coarse_slope": slope, "depth_rho": depth_rho,
            "depth_null_bar": bar, "orthogonal": orthogonal,
            "null_abs_rho_mean": null_mean, "null_abs_rho_p95": null_p95,
            "depth_spread": float(depth.std()) if depth.size else float("nan"),
            "settle_spread": (float(np.asarray(settle_breath).std())
                              if settle_breath is not None else float("nan"))}


# ===========================================================================
# VERDICT — the decision rule (thresholds stated, not hidden)
# ===========================================================================

_SCALES = ["COARSE", "REGIONAL", "LOCAL"]
_FLAT_THRESH = 0.05   # |AUC-0.5| below this at EVERY breath/scale => flat/refuted.


def interpret_verdict(results_per_breath, K, null_per_breath=None, radial=None):
    """Print the per-breath AUC table + the thresholded verdict (spec s5).

    Patterns (thresholds explicit):
      COARSE-EARLY : mean(COARSE[:K/2])>0.65 and mean(COARSE[K/2:])<0.60
      FINE-LATE    : mean(LOCAL[K/2:])>0.65  and mean(LOCAL[:K/2])<0.60
      V-CYCLE      : max(COARSE[:K/3])>0.65 and min(COARSE[K/3:2K/3])<0.55 and
                     max(COARSE[2K/3:])>0.65
      FLAT (REFUTED): |AUC-0.5|.max() < 0.05 for ALL scales/breaths AND null-indistinguishable.
    Returns dict of the boolean flags.
    """
    aucs = {s: np.array([r["scales"].get(s, np.nan) for r in results_per_breath])
            for s in _SCALES}
    nulls = ({s: np.array([r["scales"].get(s, np.nan) for r in null_per_breath])
              for s in _SCALES} if null_per_breath is not None else None)

    print("\n=== GRANULARITY PROBE VERDICT ===", flush=True)
    print("Decodability (by-instance CV AUC) by breath and scale "
          "(null in [] = shuffled-label control):", flush=True)
    if nulls is not None:
        print(f"  {'k':>3} | {'COARSE':>16} | {'REGIONAL':>16} | {'LOCAL':>16}", flush=True)
        for k in range(K):
            def _sn(x, n):
                xs = f"{x:6.3f}" if np.isfinite(x) else "  nan "
                ns = f"{n:6.3f}" if np.isfinite(n) else "  nan "
                return f"{xs} [{ns}]"
            print(f"  {k:3d} | {_sn(aucs['COARSE'][k], nulls['COARSE'][k])} | "
                  f"{_sn(aucs['REGIONAL'][k], nulls['REGIONAL'][k])} | "
                  f"{_sn(aucs['LOCAL'][k], nulls['LOCAL'][k])}", flush=True)
    else:
        print(f"  {'k':>3} | {'COARSE':>8} | {'REGIONAL':>8} | {'LOCAL':>8}", flush=True)
        for k in range(K):
            def _s(x):
                return f"{x:8.3f}" if np.isfinite(x) else f"{'nan':>8}"
            print(f"  {k:3d} | {_s(aucs['COARSE'][k])} | {_s(aucs['REGIONAL'][k])} | "
                  f"{_s(aucs['LOCAL'][k])}", flush=True)

    def _nanmean(a):
        a = a[np.isfinite(a)]
        return float(a.mean()) if a.size else float("nan")

    coarse_early = (_nanmean(aucs["COARSE"][:K // 2]) > 0.65
                    and _nanmean(aucs["COARSE"][K // 2:]) < 0.60)
    fine_late = (_nanmean(aucs["LOCAL"][K // 2:]) > 0.65
                 and _nanmean(aucs["LOCAL"][:K // 2]) < 0.60)
    c = aucs["COARSE"]
    v_cycle = (np.nanmax(c[:max(1, K // 3)]) > 0.65
               and np.nanmin(c[K // 3:2 * K // 3]) < 0.55
               and np.nanmax(c[2 * K // 3:]) > 0.65)
    flat = all(np.nanmax(np.abs(aucs[s] - 0.5)) < _FLAT_THRESH
               for s in _SCALES if np.isfinite(aucs[s]).any())

    # Null-indistinguishable check (real-minus-null max deviation per scale).
    null_indist = True
    if nulls is not None:
        for s in _SCALES:
            dev = np.nanmax(np.abs(aucs[s] - nulls[s])) if np.isfinite(aucs[s]).any() else 0.0
            if np.isfinite(dev) and dev >= _FLAT_THRESH:
                null_indist = False

    if coarse_early and fine_late:
        print("  VERDICT: COARSE-EARLY + FINE-LATE WAVE CONFIRMED", flush=True)
        print("    -> Expand-collapse waist CONFIRMED (compress coarse early, refine late)",
              flush=True)
    elif v_cycle:
        print("  VERDICT: V-CYCLE / OSCILLATION DETECTED", flush=True)
        print("    -> Oscillating (V-cycle) waist JUSTIFIED", flush=True)
    elif flat and null_indist:
        print("  VERDICT: FLAT ACROSS ALL SCALES AND BREATHS — REFUTED", flush=True)
        print("    -> No scale stratification; expand-collapse waist NOT justified",
              flush=True)
    else:
        print("  VERDICT: INCONCLUSIVE — mixed/weak signal (above null somewhere, but "
              "no clean coarse-early/fine-late wave or V-cycle)", flush=True)

    if radial is not None:
        print(f"\n  [radial-depth control] coarse pop-slope (AUC_final-AUC_0)="
              f"{radial['coarse_slope']:.3f}  depth<->settle_breath rho={radial['depth_rho']}  "
              f"(null |rho| mean={radial['null_abs_rho_mean']} p95={radial['null_abs_rho_p95']}; "
              f"orthogonal if |rho|<{radial['depth_null_bar']}: {radial['orthogonal']})",
              flush=True)
        print(f"    depth spread={radial['depth_spread']:.3f}  "
              f"settle spread={radial['settle_spread']:.3f}", flush=True)

    return {"coarse_early": coarse_early, "fine_late": fine_late,
            "v_cycle": v_cycle, "flat": flat, "null_indist": null_indist}


# ===========================================================================
# DRIVER — collect per-breath reps over the bank, probe, verdict
# ===========================================================================

def _sample_balanced_records(curriculum_path, bands, total, n_values, seed):
    """Sample `total` puzzle records from the curriculum, balanced across (band x N) so the
    COARSE band/N targets have variance. Returns a list of records (each a dict)."""
    from mycelium.kenken_data import load_jsonl
    recs = load_jsonl(curriculum_path)
    # bucket by (band, N).
    buckets = {}
    for r in recs:
        b = str(r.get("band", "all"))
        nN = int(r["N"])
        if b in bands and nN in n_values:
            buckets.setdefault((b, nN), []).append(r)
    keys = [(b, nN) for b in bands for nN in n_values if (b, nN) in buckets]
    if not keys:
        raise RuntimeError(f"no records matched bands={bands} N={n_values} in {curriculum_path}")
    rng = np.random.RandomState(seed)
    per_key = max(1, total // len(keys))
    picked = []
    for k in keys:
        pool = buckets[k]
        idx = rng.permutation(len(pool))[:min(per_key, len(pool))]
        picked.extend(pool[i] for i in idx)
    rng.shuffle(picked)
    return picked[:total]


def _settle_breath_from_logits(logits_hist):
    """Per-instance settle breath (gold-free): the last breath at which the per-cell argmax
    CHANGED, +1 (so larger = settles later). logits_hist : list of K (B, S, N) Tensors.

    Returns (B,) int. Uses per-cell argmax stability across breaths (no gold)."""
    preds = np.stack([lg.argmax(axis=-1).realize().numpy() for lg in logits_hist], axis=0)
    K = preds.shape[0]
    B = preds.shape[1]
    settle = np.zeros(B, dtype=np.int64)
    for k in range(1, K):
        changed = (preds[k] != preds[k - 1]).any(axis=1)   # (B,) any cell flipped this breath
        settle = np.where(changed, k, settle)
    return settle


def collect_bank_reps(model, spec, curriculum_path, bands, total, K, eval_batch, n_values,
                      seed):
    """Sample a balanced KenKen bank from the curriculum, run the per-breath eager forward,
    and accumulate per-breath residual reps + labels + per-instance settle breaths.

    Returns:
      reps        : (K, N_inst, S, H) float32 (per-breath per-cell residual, engine sink).
      labels      : merged extract_granularity_features over the bank (instance_id GLOBAL).
      settle      : (N_inst,) per-instance settle breath (for the radial-depth control).
    """
    from mycelium.kenken_data import N_MAX, N_CELLS
    N_CAGES_MAX = 41   # pinned: the ckpt's training topology (g40 max cages = 40 <= 41).

    recs = _sample_balanced_records(curriculum_path, bands, total, n_values, seed)
    print(f"[collect] sampled {len(recs)} puzzles balanced across "
          f"bands={bands} x N={n_values}", flush=True)

    reps_batches = []
    settle_batches = []
    label_keys = ["band", "band_ord", "deduction_depth", "n", "n_givens", "n_cages",
                  "coarse_label", "cell_valid"]
    cell_label_keys = ["regional_label", "local_label", "is_given"]
    merged = {k: [] for k in label_keys + cell_label_keys}
    inst_ids = []
    running = 0

    t0 = time.time()
    for start in range(0, len(recs), eval_batch):
        picks = recs[start:start + eval_batch]
        kb = stack_records(picks, N_CAGES_MAX)
        reps, logits_hist = run_kenken_per_breath_forward(model, kb, spec, K)  # (K,b,S,H)
        feats = extract_granularity_features(kb, spec)
        settle = _settle_breath_from_logits(logits_hist)

        b = reps.shape[1]
        reps_batches.append(reps)
        settle_batches.append(settle)
        inst_ids.append(np.arange(running, running + b, dtype=np.int64))
        running += b
        for k in label_keys + cell_label_keys:
            merged[k].append(feats[k])
        print(f"[collect] batch {start // eval_batch + 1} "
              f"({running} insts, {time.time() - t0:.0f}s)", flush=True)

    reps_all = np.concatenate(reps_batches, axis=1)         # (K, N_inst, S, H)
    settle_all = np.concatenate(settle_batches, axis=0)     # (N_inst,)
    labels = {}
    for k in label_keys:
        labels[k] = np.concatenate([np.atleast_1d(v) for v in merged[k]], axis=0)
    for k in cell_label_keys:
        labels[k] = np.concatenate(merged[k], axis=0)       # (N_inst, S)
    labels["instance_id"] = np.concatenate(inst_ids, axis=0)
    # keep aliases the probe reads:
    labels["coarse_label"] = labels["n"]
    return reps_all, labels, settle_all


def stack_records(recs, n_cages_max):
    """Stack a list of puzzle records into a KenKenBatch (mirrors kenken_volume_eval)."""
    from tinygrad import Tensor, dtypes
    from mycelium.kenken_data import encode_puzzle, KenKenBatch
    encs = [encode_puzzle(r, n_cages_max) for r in recs]

    def stack_int(key):
        return Tensor(np.stack([e[key] for e in encs]).astype(np.int32),
                      dtype=dtypes.int).contiguous().realize()

    def stack_f(key):
        return Tensor(np.stack([e[key] for e in encs]).astype(np.float32),
                      dtype=dtypes.float).contiguous().realize()

    d = {
        "input_cells": stack_int("input_cells"),
        "gold": stack_int("gold"),
        "cell_valid": stack_f("cell_valid"),
        "cage_mask": stack_f("cage_mask"),
        "cell_cage_id": stack_int("cell_cage_id"),
        "cage_cell_count_per_cell": stack_int("cage_cell_count_per_cell"),
        "value_domain_mask": stack_f("value_domain_mask"),
        "cage_op": stack_int("cage_op"),
        "cage_target": stack_int("cage_target"),
        "cage_size": stack_int("cage_size"),
        "deduction_depth": [e["deduction_depth"] for e in encs],
        "N": [e["N"] for e in encs],
        "n_givens": [e["n_givens"] for e in encs],
        "band": [e["band"] for e in encs],
    }
    return KenKenBatch(d)


def main(argv=None):
    P = argparse.ArgumentParser(description="KenKen per-breath granularity probe")
    P.add_argument("--bands", default="g10,g20,g30,g40",
                   help="comma-separated bands to sample (default: all KenKen bands)")
    P.add_argument("--n", type=int, default=300,
                   help="total puzzles to sample (balanced across band x N)")
    P.add_argument("--curriculum", default=".cache/kenken_test_curriculum.jsonl",
                   help="curriculum jsonl to sample from (has band + N variance)")
    P.add_argument("--ckpt",
                   default=os.environ.get(
                       "FG_CKPT",
                       ".cache/fg_ckpts/fg_kenken_k16_reg/fg_kenken_k16_reg_final.safetensors"),
                   help="model checkpoint path")
    P.add_argument("--K", type=int, default=16, help="number of breaths")
    P.add_argument("--eval-batch", type=int, default=8, help="forward batch size")
    P.add_argument("--folds", type=int, default=5, help="CV folds (by-instance split)")
    P.add_argument("--seed", type=int, default=0)
    P.add_argument("--n-values", default="5,6,7",
                   help="puzzle sizes N to include (curriculum has 5,6,7)")
    args = P.parse_args(argv)

    bands = [b.strip() for b in args.bands.split(",") if b.strip()]
    n_values = [int(x) for x in args.n_values.split(",") if x.strip()]

    print(f"[granularity-probe] bands={bands} n={args.n} N={n_values} K={args.K} "
          f"ckpt={args.ckpt} folds={args.folds} seed={args.seed}", flush=True)
    if not os.path.exists(args.curriculum):
        print(f"[granularity-probe] ERROR: curriculum not found: {args.curriculum}", flush=True)
        return 2
    if not os.path.exists(args.ckpt):
        print(f"[granularity-probe] ERROR: ckpt not found: {args.ckpt}", flush=True)
        return 2

    spec = build_kenken_spec(args.K)
    model = build_kenken_deducer_model(spec, args.ckpt, args.seed)

    reps, labels, settle = collect_bank_reps(
        model, spec, args.curriculum, bands, args.n, args.K, args.eval_batch,
        n_values, args.seed)
    print(f"[granularity-probe] reps shape = {reps.shape}  "
          f"(K, N_inst, S, H); N_inst={reps.shape[1]}", flush=True)

    results = probe_per_breath_decodability(
        reps, labels, spec, args.K, folds=args.folds, seed=args.seed)
    null = probe_per_breath_decodability(
        reps, labels, spec, args.K, folds=args.folds, seed=args.seed, shuffle_null=True)
    radial = radial_depth_correlation(
        results, labels, args.K, settle_breath=settle, seed=args.seed)

    # Detailed COARSE sub-target table (N / band / #cages).
    print("\n=== COARSE sub-targets (pooled rep) by breath ===", flush=True)
    print(f"  {'k':>3} | {'N>med':>7} | {'band>med':>8} | {'#cages>med':>10} | "
          f"{'COARSE(mean)':>12}", flush=True)
    for k in range(args.K):
        sc = results[k]["scales"]
        def _s(x):
            return f"{x:7.3f}" if np.isfinite(x) else f"{'nan':>7}"
        print(f"  {k:3d} | {_s(sc['COARSE_N'])} | {_s(sc['COARSE_BAND'])} | "
              f"{_s(sc['COARSE_NCAGES']):>10} | {_s(sc['COARSE']):>12}", flush=True)

    flags = interpret_verdict(results, args.K, null_per_breath=null, radial=radial)
    print("\nGranularity probe complete.", flush=True)
    return 0


if __name__ == "__main__":
    parse_ok = _ast_parse_ok()
    print(f"[ast.parse] astparse_ok={parse_ok}", flush=True)
    if not parse_ok:
        sys.exit(1)
    sys.exit(main())
