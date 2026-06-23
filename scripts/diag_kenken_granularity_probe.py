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
    """Build the Pythia-410M breathing deducer for KenKen + load the ckpt. STUB.

    Mirrors the validated build path used elsewhere (amortized_frontier_measure
    ._build_deducer_model / search_coloring): load_breathing(Config) -> cast_layers_fp32
    -> attach_factor_graph_params(model, hidden, spec) -> load_ckpt(model, ckpt). KenKen
    additionally needs the verification-inlet params (mycelium.kenken op/target/size embeds
    + kenken_inlet_w/b) attached so build_verification_inlet can run.

    Returns the model with factor_graph_* + kenken inlet params attached and ckpt restored.

    NOT IMPLEMENTED in the skeleton (no GPU job from agents; no KenKen ckpt exists yet).
    """
    raise NotImplementedError(
        "build_kenken_deducer_model: GPU build stubbed in the skeleton. "
        "Mirror amortized_frontier_measure._build_deducer_model but for the KenKen spec "
        "(attach_factor_graph_params + the kenken verification-inlet params), then "
        "sc.load_ckpt(model, ckpt). Requires a KenKen fg ckpt (FG_TASK=kenken on "
        "scripts/factor_graph_train.py); none exists yet (CLAUDE.md s5)."
    )


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

def run_kenken_per_breath_forward(model, kb, spec, K, hook):
    """Run ONE eager K-breath KenKen forward; return per-breath readout reps via the hook.

    Builds the FactorGraphBatch from a KenKenBatch (make_kenken_factor_batch) WITH the
    verification inlet (build_verification_inlet -> prebuilt_inlet) since KenKen sets
    has_factor_inlet=True, then calls factor_breathing_forward(model, batch, spec, K). The
    installed hook appends one (B, S, H) rep per breath into hook["slot"]["captures"].

    EAGER ONLY (no JIT): the JIT-unrolled K-graph would not surface per-breath _layernorm
    calls to the monkeypatch. Call hook["arm"]() before the forward.

    Returns:
      readout_reps_per_breath : (K, B, S, H) float32  [stacked from the hook captures]
      value_logits_history    : list of K (B, S, N) Tensors (for cross-check / argmax)

    NOT IMPLEMENTED in the skeleton (stubbed; no GPU job from agents).
    """
    raise NotImplementedError(
        "run_kenken_per_breath_forward: GPU forward stubbed in the skeleton. "
        "Reference wiring:\n"
        "  from mycelium.factor_graph_engine import make_kenken_factor_batch, "
        "factor_breathing_forward\n"
        "  from mycelium.kenken import build_verification_inlet\n"
        "  inlet = build_verification_inlet(model, kb.cage_op, kb.cage_target, "
        "kb.cage_size, kb.cell_cage_id)\n"
        "  batch = make_kenken_factor_batch(kb, spec, prebuilt_inlet=inlet)\n"
        "  hook['arm']()\n"
        "  logits_hist, _calib = factor_breathing_forward(model, batch, spec, K)\n"
        "  reps = np.stack(hook['slot']['captures'], axis=0)  # (K, B, S, H)\n"
        "  assert reps.shape[0] == K, 'expected one readout capture per breath'\n"
        "  return reps, logits_hist"
    )


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
    cage_id = kb.cell_cage_id.realize().numpy().astype(np.int64)   # (B, S) cage id (-1 pad)

    n = np.asarray(kb.N, dtype=np.int64)                          # (B,)
    deduction_depth = np.asarray(kb.deduction_depth, dtype=np.int64)
    band = np.asarray(kb.band, dtype=object)
    n_givens = np.asarray(kb.n_givens, dtype=np.int64)

    inst_id = np.arange(B, dtype=np.int64)  # caller offsets per band to keep ids global

    return {
        "instance_id": inst_id,
        "band": band,
        "deduction_depth": deduction_depth,
        "n": n,
        "n_givens": n_givens,
        "coarse_label": n,            # default COARSE target = puzzle size N
        "medium_label": cage_id,      # (B, S)
        "fine_label": gold,           # (B, S)
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
    """Probe decodability of COARSE/MEDIUM/FINE at each breath k (leak-free by-instance CV).

    readout_reps_per_breath : (K, B, S, H) per-breath per-cell reps (from the hook).
    labels_dict             : from extract_granularity_features (coarse/medium/fine + masks).
    shuffle_null            : if True, permute labels (independent of reps) -> expect ~0.5.

    For each breath k:
      COARSE : pool reps over valid cells -> (B, H); binary-recode coarse_label; by-instance
               CV AUC.
      MEDIUM : per-cell reps (B, S, H) -> flatten valid cells; binary-recode cage_id;
               per-cell by-instance CV AUC (cell instance ids derived from inst_id repeat).
      FINE   : per-cell reps; binary-recode gold value; per-cell by-instance CV AUC.

    Returns: list of K dicts, each {'k': k, 'scales': {'COARSE':auc, 'MEDIUM':auc, 'FINE':auc}}.
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

    # COARSE label (instance level) + binary recode.
    coarse_lab, coarse_descr = _binarize_multiclass(labels_dict["coarse_label"])
    # MEDIUM / FINE labels (cell level) — recode over VALID cells only.
    med_flat = labels_dict["medium_label"].reshape(-1)
    fine_flat = labels_dict["fine_label"].reshape(-1)
    valid_flat = cell_valid.reshape(-1)
    med_bin, _ = _binarize_multiclass(med_flat, valid=valid_flat)
    fine_bin, _ = _binarize_multiclass(fine_flat, valid=valid_flat)
    # per-cell instance id + per-cell fold id (instance fold broadcast over S, masked valid).
    inst_flat = np.repeat(inst_id, S)
    fold_flat = np.repeat(fold_id_inst, S)

    if shuffle_null:
        coarse_lab = coarse_lab[rng.permutation(coarse_lab.shape[0])]
        perm = rng.permutation(med_bin.shape[0])
        med_bin = med_bin[perm]
        fine_bin = fine_bin[perm]

    results_per_breath = []
    for k in range(K):
        reps_k = reps[k]                                          # (B, S, H)
        scales = {}

        # COARSE: pool over valid cells -> (B, H).
        cv = cell_valid.astype(float)
        reps_coarse = ((reps_k * cv[:, :, None]).sum(axis=1)
                       / (cv.sum(axis=1, keepdims=True) + 1e-6))   # (B, H)
        scales["COARSE"] = _cv_logistic_auc_by_instance(
            reps_coarse, coarse_lab, fold_id_inst, n_folds)

        # MEDIUM + FINE: per-cell, valid cells only.
        reps_cells = reps_k.reshape(-1, reps_k.shape[-1])          # (B*S, H)
        mask = valid_flat
        reps_valid = reps_cells[mask]
        fold_valid = fold_flat[mask]
        inst_valid = inst_flat[mask]
        scales["MEDIUM"] = _cv_logistic_auc_by_instance(
            reps_valid, med_bin[mask], fold_valid, n_folds, inst_id_flat=inst_valid)
        scales["FINE"] = _cv_logistic_auc_by_instance(
            reps_valid, fine_bin[mask], fold_valid, n_folds, inst_id_flat=inst_valid)

        results_per_breath.append({"k": k, "scales": scales,
                                   "coarse_descr": coarse_descr})
    return results_per_breath


# ===========================================================================
# RADIAL-DEPTH CONTROL — coarse-decodability slope vs deduction_depth (orthogonality)
# ===========================================================================

def radial_depth_correlation(results_per_breath, labels_dict, K, seed=0):
    """Correlate the COARSE decodability SLOPE (AUC_final - AUC_0) with deduction_depth.

    A POPULATION-LEVEL slope (one AUC per breath) gives a single number, so the control here
    is the cross-instance Pearson rho between per-instance deduction_depth and a per-instance
    coarse-refinement proxy. The skeleton computes the GLOBAL slope (auc[K-1]-auc[0]) and the
    depth spread; the per-instance proxy + shuffle-null bar is the seam to fill at run time.

    Returns dict {'coarse_slope': float, 'depth_rho': float|nan, 'depth_null_bar': 0.30,
                  'orthogonal': bool}.
    """
    coarse = np.array([r["scales"].get("COARSE", np.nan) for r in results_per_breath])
    slope = float(coarse[-1] - coarse[0]) if coarse.size >= 2 else float("nan")
    depth = np.asarray(labels_dict.get("deduction_depth", []), dtype=float)
    # per-instance coarse-refinement proxy + rho is the run-time seam; default nan here.
    depth_rho = float("nan")
    bar = 0.30
    orthogonal = bool(np.isfinite(depth_rho) and abs(depth_rho) < bar)
    return {"coarse_slope": slope, "depth_rho": depth_rho,
            "depth_null_bar": bar, "orthogonal": orthogonal,
            "depth_spread": float(depth.std()) if depth.size else float("nan")}


# ===========================================================================
# VERDICT — the decision rule (thresholds stated, not hidden)
# ===========================================================================

def interpret_verdict(results_per_breath, K, radial=None):
    """Print the per-breath AUC table + the thresholded verdict (see spec s5).

    Patterns (thresholds explicit):
      COARSE-EARLY : mean(COARSE[:K/2])>0.65 and mean(COARSE[K/2:])<0.60
      FINE-LATE    : mean(FINE[K/2:])>0.65   and mean(FINE[:K/2])<0.60
      V-CYCLE      : max(COARSE[:K/3])>0.65 and min(COARSE[K/3:2K/3])<0.55 and
                     max(COARSE[2K/3:])>0.65
      FLAT (REFUTED): |AUC-0.5|.max() < 0.05 for ALL scales/breaths.
    Returns dict of the boolean flags.
    """
    aucs = {scale: np.array([r["scales"].get(scale, np.nan) for r in results_per_breath])
            for scale in ["COARSE", "MEDIUM", "FINE"]}

    print("\n=== GRANULARITY PROBE VERDICT ===", flush=True)
    print("Decodability (by-instance CV AUC) by breath and scale:", flush=True)
    print(f"  {'k':>3} | {'COARSE':>8} | {'MEDIUM':>8} | {'FINE':>8}", flush=True)
    for k in range(K):
        def _s(x):
            return f"{x:8.3f}" if np.isfinite(x) else f"{'nan':>8}"
        print(f"  {k:3d} | {_s(aucs['COARSE'][k])} | {_s(aucs['MEDIUM'][k])} | "
              f"{_s(aucs['FINE'][k])}", flush=True)

    def _nanmean(a):
        a = a[np.isfinite(a)]
        return float(a.mean()) if a.size else float("nan")

    coarse_early = (_nanmean(aucs["COARSE"][:K // 2]) > 0.65
                    and _nanmean(aucs["COARSE"][K // 2:]) < 0.60)
    fine_late = (_nanmean(aucs["FINE"][K // 2:]) > 0.65
                 and _nanmean(aucs["FINE"][:K // 2]) < 0.60)
    c = aucs["COARSE"]
    v_cycle = (np.nanmax(c[:max(1, K // 3)]) > 0.65
               and np.nanmin(c[K // 3:2 * K // 3]) < 0.55
               and np.nanmax(c[2 * K // 3:]) > 0.65)
    flat = all(np.nanmax(np.abs(aucs[s] - 0.5)) < 0.05
               for s in ["COARSE", "MEDIUM", "FINE"]
               if np.isfinite(aucs[s]).any())

    if coarse_early and fine_late:
        print("  VERDICT: COARSE-EARLY + FINE-LATE WAVE CONFIRMED", flush=True)
        print("    -> Expand-collapse waist JUSTIFIED (compress coarse early, refine late)",
              flush=True)
    elif v_cycle:
        print("  VERDICT: V-CYCLE / OSCILLATION DETECTED", flush=True)
        print("    -> Expand-collapse with oscillating waist JUSTIFIED", flush=True)
    elif flat:
        print("  VERDICT: FLAT ACROSS ALL SCALES AND BREATHS — REFUTED", flush=True)
        print("    -> No scale stratification; expand-collapse waist NOT justified",
              flush=True)
    else:
        print("  VERDICT: INCONCLUSIVE — mixed/weak signal", flush=True)

    if radial is not None:
        print(f"\n  [radial-depth control] coarse slope (AUC_final-AUC_0)="
              f"{radial['coarse_slope']:.3f}  depth_rho={radial['depth_rho']}  "
              f"(orthogonal if |rho|<{radial['depth_null_bar']}: {radial['orthogonal']})",
              flush=True)

    return {"coarse_early": coarse_early, "fine_late": fine_late,
            "v_cycle": v_cycle, "flat": flat}


# ===========================================================================
# DRIVER — collect per-breath reps over the bank, probe, verdict
# ===========================================================================

def collect_bank_reps(model, spec, bands_test, per_band, K, eval_batch, seed):
    """Iterate the KenKen bands, run the per-breath forward, and accumulate per-breath reps
    + labels across all instances. STUB (orchestration shape only — no GPU from agents).

    Shape contract (what the run-time fill must return):
      readout_reps_per_breath : (K, N_inst, S, H) float32 (stacked across batches on axis 1)
      labels_dict             : merged extract_granularity_features over the bank, with
                                instance_id offset per band so ids are GLOBAL (no collision).

    Reference loop:
      hook = install_readout_capture_hook(model)
      for band, path in bands_test.items():
        loader = KenKenLoader(path, batch_size=eval_batch, seed=seed)
        for kb in loader.iter_eval(eval_batch):   # bounded by per_band upstream
          reps, _ = run_kenken_per_breath_forward(model, kb, spec, K, hook)  # (K,B,S,H)
          feats = extract_granularity_features(kb, spec)
          # offset feats['instance_id'] by a running counter; append reps + feats
      hook['uninstall']()
      # concat reps on axis 1 -> (K, N_inst, S, H); merge labels.
    """
    raise NotImplementedError(
        "collect_bank_reps: GPU orchestration stubbed in the skeleton. See docstring for "
        "the reference loop (install hook -> per-band KenKenLoader -> per-breath forward -> "
        "extract labels -> accumulate, offsetting instance_id per band)."
    )


def main(argv=None):
    P = argparse.ArgumentParser(description="KenKen per-breath granularity probe (skeleton)")
    P.add_argument("--bands", default="g40,g30,g20,g10",
                   help="comma-separated bands to test (default: all KenKen bands)")
    P.add_argument("--per-band", type=int, default=50, help="instances per band")
    P.add_argument("--ckpt",
                   default=os.environ.get(
                       "FG_CKPT",
                       ".cache/fg_ckpts/fg_kenken_k16/fg_kenken_k16_final.safetensors"),
                   help="model checkpoint path")
    P.add_argument("--K", type=int, default=16, help="number of breaths")
    P.add_argument("--eval-batch", type=int, default=8, help="batch size")
    P.add_argument("--folds", type=int, default=5, help="CV folds (by-instance split)")
    P.add_argument("--seed", type=int, default=0)
    P.add_argument("--shuffle-null", action="store_true",
                   help="run the probe on SHUFFLED labels (control; expect ~0.5 everywhere)")
    P.add_argument("--test-band-dir", default="/tmp",
                   help="dir holding kk_test_<band>.jsonl files (default /tmp)")
    args = P.parse_args(argv)

    # Resolve the per-band KenKen test files (the spec's convention).
    bands = [b.strip() for b in args.bands.split(",") if b.strip()]
    bands_test = {b: os.path.join(args.test_band_dir, f"kk_test_{b}.jsonl") for b in bands}
    missing = [p for p in bands_test.values() if not os.path.exists(p)]

    print(f"[granularity-probe] bands={bands} per_band={args.per_band} K={args.K} "
          f"ckpt={args.ckpt} folds={args.folds} seed={args.seed}", flush=True)
    if missing:
        print(f"[granularity-probe] WARNING: missing test files: {missing}", flush=True)

    # --- GPU pipeline (stubbed in the skeleton; Bryce/main-thread fills + fires) ---
    #   spec  = build_kenken_spec(args.K)
    #   model = build_kenken_deducer_model(spec, args.ckpt, args.seed)
    #   reps, labels = collect_bank_reps(model, spec, bands_test, args.per_band,
    #                                    args.K, args.eval_batch, args.seed)
    #   results = probe_per_breath_decodability(reps, labels, spec, args.K,
    #                                           folds=args.folds, seed=args.seed)
    #   null    = probe_per_breath_decodability(reps, labels, spec, args.K,
    #                                           folds=args.folds, seed=args.seed,
    #                                           shuffle_null=True)  # expect ~0.5
    #   radial  = radial_depth_correlation(results, labels, args.K, seed=args.seed)
    #   interpret_verdict(results, args.K, radial=radial)
    print("[granularity-probe] SKELETON: GPU pipeline is stubbed (build_kenken_deducer_model "
          "/ run_kenken_per_breath_forward / collect_bank_reps raise NotImplementedError). "
          "Fill the stubs + run on AMD (eager forward) once a KenKen ckpt exists.", flush=True)
    print("Granularity probe (skeleton) complete.", flush=True)
    return 0


if __name__ == "__main__":
    parse_ok = _ast_parse_ok()
    print(f"[ast.parse] astparse_ok={parse_ok}", flush=True)
    if not parse_ok:
        sys.exit(1)
    sys.exit(main())
