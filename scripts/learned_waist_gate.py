"""learned_waist_gate.py — does a LEARNED, NONLINEAR, SUPERVISED projection BEAT the
PCA linear/unsupervised LOWER BOUND at separating VALID from INVALID darts on the SAME
data + SAME protocol? (the go/no-go gate for an in-deducer learned waist.)

THE QUESTION. `dart_cluster_probe.py` found the Anna-Karenina common-mode signal is WEAK
at raw 1024d (transferable, instance-identity-removed, by-instance CV AUC = 0.582) but
SHARPENS under PCA compression (centered-global AUC 0.658 at d=256). PCA is a LINEAR,
UNSUPERVISED lower bound. This script asks: can a LEARNED, NONLINEAR, SUPERVISED head
(trained to separate valid vs invalid) beat that 0.658 floor on the SAME centered reps,
SAME by-instance CV folds, SAME Mann-Whitney AUC? If yes -> a learned in-deducer waist
has real headroom -> worth building. If it only ties PCA -> the rep has no extra
nonlinear-separable structure -> the expensive in-deducer waist is NOT justified.

THE GATE LOGIC (stated, not hidden):
  GREEN LIGHT (build the in-deducer waist) iff at some d:
    learned_nonlinear_AUC >= PCA_AUC + 0.05  AND  learned_nonlinear_AUC >= 0.70.
  NOT JUSTIFIED iff learned ~= PCA (within ~0.02) at every d (the rep has no extra
    nonlinear-separable structure beyond the linear/unsupervised PCA read -> faint signal).
  Also reports nonlinear-vs-linear: does the nonlinear hidden layer buy anything over a
    plain linear (1024->1) head?

APPLES-TO-APPLES (fairness depends on it — REUSED, not re-implemented):
  * by-INSTANCE CV split  -> reproduced byte-faithfully from dart_cluster_probe's
    `_cv_logistic_auc` (same RandomState(seed) permutation/round-robin over instances)
    so the learned head and the PCA baseline see the IDENTICAL train/test folds.
  * per-instance-mean CENTERING (instance-identity removal) -> imported
    `dart_cluster_probe._center_by_instance` (the exact metric_C transform).
  * Mann-Whitney AUC -> imported `dart_cluster_probe.auc_mann_whitney` (the exact metric).
  * per-fold standardization (train stats) -> mirrors `_cv_logistic_auc`.
  * the PCA baseline AT EACH d -> the EXISTING `_cv_logistic_auc(..., max_dim=d)` path
    (the SAME train-fit SVD->d + tiny logistic the probe uses inside metric_C), run on
    the SAME centered reps + SAME folds. So PCA_AUC here == the probe's centered-global
    AUC at that d, by construction.

LEARNED HEADS (tiny, CPU-capable, tinygrad; DEV-agnostic):
  (a) LINEAR     : 1024 -> 1 logit (logistic; the learned linear sanity, ~= PCA+logistic
                   upper-bounded since it sees ALL 1024 dims supervised).
  (b) NONLINEAR  : 1024 -> d -> GELU -> 1 logit, the d-dim hidden being the "learned
                   common-mode waist" silhouette. d in {512, 256, 128}.
  Trained with class-WEIGHTED binary cross-entropy (valid is the minority ~15%), L2 reg,
  EARLY-STOP on a within-train (by-instance) val split, modest epochs. Held-out (out-of-
  fold, by-instance) scores are pooled -> ONE AUC per (head, d), matching the probe.

HONEST CAVEATS (always printed): (1) this trains on FROZEN baseline silhouettes -> a
LOWER BOUND on an in-deducer waist that ALSO reshapes the reps during training; a null
here is suggestive, not fully conclusive. (2) separating valid/invalid is the OPERATIONAL
proxy for "good clusters" — a strong classifier => a usable common mode for steering/a
library. (3) by-instance CV + centering remove the graph-identity confound (the raw
uncentered AUC ~0.755 is NOT the target; 0.5 is the floor).

SELFTEST (CPU, no npz, no GPU): a NONLINEAR-separable synthetic (XOR-like in a 2d
subspace embedded in H-d; validity not linearly separable) -> learned_nonlinear AUC HIGH
while PCA+logistic ~0.5 (proves the gate DETECTS nonlinear headroom PCA misses); + a NULL
(validity independent of the rep) -> ALL ~0.5 by-instance (no false green light). Both
ASSERTED. astparse gate.

USAGE:
  CPU selftest (GPU-free):
    DEV=CPU .venv/bin/python3 scripts/learned_waist_gate.py --selftest
  Run the gate on the captured darts:
    DEV=CPU .venv/bin/python3 scripts/learned_waist_gate.py \
        --npz .cache/dart_silhouettes_fg_coloring_k16.npz --dims 512,256,128
"""
from __future__ import annotations

import argparse
import ast
import os
import sys

_THIS_FILE = os.path.abspath(__file__)

import numpy as np

# Keep tinygrad on CPU by default for this tiny-MLP build (DEV-agnostic: respects an
# externally-set DEV/GPU, but never FORCES a GPU). Must precede the tinygrad import.
os.environ.setdefault("DEV", "CPU")

# --- REUSE dart_cluster_probe's VALIDATED machinery (fairness depends on identity). ---
sys.path.insert(0, os.path.dirname(_THIS_FILE))
import dart_cluster_probe as DCP  # noqa: E402
from dart_cluster_probe import (  # noqa: E402
    auc_mann_whitney,          # the EXACT Mann-Whitney AUC metric
    _center_by_instance,       # the EXACT per-instance-mean centering (identity removal)
    _cv_logistic_auc,          # the EXACT PCA->d + tiny-logistic by-instance CV baseline
)

from tinygrad import Tensor  # noqa: E402
from tinygrad.nn.optim import Adam  # noqa: E402


# ===========================================================================
# ast.parse gate
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
# BY-INSTANCE FOLD ASSIGNMENT — reproduced BYTE-FAITHFULLY from
# dart_cluster_probe._cv_logistic_auc so the learned head and the PCA baseline see the
# IDENTICAL folds. Same RandomState(seed) permutation + round-robin over the unique
# instance ids -> same uid->fold map -> same per-dart fold id. (We need the per-dart fold
# ids explicitly to drive the learned head's training/eval; the PCA baseline reconstructs
# the SAME map internally because it is given the SAME inst_id + seed.)
# ===========================================================================

def assign_instance_folds(inst_id, folds=5, seed=0):
    """Return (fold_id (n,), n_folds) assigning WHOLE instances to folds, EXACTLY as
    dart_cluster_probe._cv_logistic_auc does (no dart of a train-instance in test)."""
    inst_id = np.asarray(inst_id)
    uids = np.unique(inst_id)
    n_inst = uids.shape[0]
    folds = min(folds, n_inst)                      # can't have more folds than instances.
    rng = np.random.RandomState(seed)
    inst_perm = rng.permutation(n_inst)
    inst_fold = np.zeros(n_inst, dtype=int)
    inst_fold[inst_perm] = np.arange(n_inst) % folds
    uid_to_fold = {int(u): int(f) for u, f in zip(uids, inst_fold)}
    fold_id = np.array([uid_to_fold[int(u)] for u in inst_id], dtype=int)
    return fold_id, folds


def _by_instance_val_split(inst_id_train, val_frac=0.2, seed=0):
    """Split a TRAIN set into (fit, val) BY INSTANCE (no dart of a fit-instance in val),
    so the early-stop val signal is itself instance-clean. Returns boolean fit/val masks
    over the train rows."""
    inst_id_train = np.asarray(inst_id_train)
    uids = np.unique(inst_id_train)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(uids.shape[0])
    n_val = max(1, int(round(val_frac * uids.shape[0])))
    val_uids = set(int(u) for u in uids[perm[:n_val]])
    val_mask = np.array([int(u) in val_uids for u in inst_id_train], dtype=bool)
    # guard: if the val split happens to be single-class, fall back to all-train (no ES).
    return ~val_mask, val_mask


# ===========================================================================
# THE TINY LEARNED HEAD (tinygrad; CPU-capable, DEV-agnostic)
# ===========================================================================
#   LINEAR    : x(1024) -> Wx + b                       (1 logit)
#   NONLINEAR : GELU(x @ W1 + b1)(d) -> @ W2 + b2        (1 logit); the d-dim layer = waist
#
# class-WEIGHTED BCE (pos=valid is the minority), L2 weight decay, EARLY-STOP on a
# by-instance within-train val split. Full-batch GD (12800x1024 is tiny). Modest epochs.

class _Head:
    def __init__(self, in_dim, hidden=None, seed=0):
        rng = np.random.RandomState(seed)
        self.hidden = hidden
        if hidden is None:                          # LINEAR head
            scale = 1.0 / np.sqrt(in_dim)
            self.W = Tensor((rng.randn(in_dim, 1) * scale).astype(np.float32),
                            requires_grad=True)
            self.b = Tensor.zeros(1, requires_grad=True)
            self.params = [self.W, self.b]
        else:                                       # NONLINEAR head: in -> hidden -> 1
            s1 = 1.0 / np.sqrt(in_dim)
            s2 = 1.0 / np.sqrt(hidden)
            self.W1 = Tensor((rng.randn(in_dim, hidden) * s1).astype(np.float32),
                             requires_grad=True)
            self.b1 = Tensor.zeros(hidden, requires_grad=True)
            self.W2 = Tensor((rng.randn(hidden, 1) * s2).astype(np.float32),
                             requires_grad=True)
            self.b2 = Tensor.zeros(1, requires_grad=True)
            self.params = [self.W1, self.b1, self.W2, self.b2]

    def logits(self, x: Tensor) -> Tensor:
        if self.hidden is None:
            return (x @ self.W + self.b).reshape(-1)
        h = (x @ self.W1 + self.b1).gelu()
        return (h @ self.W2 + self.b2).reshape(-1)

    def l2(self) -> Tensor:
        if self.hidden is None:
            return (self.W * self.W).sum()
        return (self.W1 * self.W1).sum() + (self.W2 * self.W2).sum()


def _bce_weighted(logits: Tensor, y: Tensor, w_pos: float, w_neg: float) -> Tensor:
    """Class-weighted BCE-with-logits. y in {0,1}. Numerically stable softplus form:
       loss_i = weight_i * [ softplus(z) - y*z ],  weight_i = w_pos if y else w_neg.
    softplus(z) = max(z,0) + log1p(exp(-|z|))."""
    z = logits
    sp = z.relu() + (1.0 + (-z.abs()).exp()).log()  # softplus(z)
    per = sp - y * z
    weight = y * w_pos + (1.0 - y) * w_neg
    return (weight * per).sum() / weight.sum()


def train_head(Xtr, ytr, Xval, yval, hidden, l2=1e-3, lr=3e-3, max_epochs=300,
               patience=25, seed=0):
    """Train one head with class-weighted BCE + L2, EARLY-STOP on by-instance val AUC.
    Returns the trained head with its BEST (val-AUC) weights restored.

    class weights: w_pos = n_neg/n, w_neg = n_pos/n -> rebalances the minority. Full-batch
    GD (data is tiny). Early-stop on val Mann-Whitney AUC (the deployed metric), restoring
    the best-AUC snapshot. If the val split is single-class, falls back to fixed epochs."""
    in_dim = Xtr.shape[1]
    head = _Head(in_dim, hidden=hidden, seed=seed)
    opt = Adam(head.params, lr=lr)
    Xt = Tensor(Xtr.astype(np.float32))
    yt = Tensor(ytr.astype(np.float32))
    n_pos = float(ytr.sum())
    n_neg = float((ytr == 0).sum())
    n_all = float(ytr.shape[0])
    w_pos = (n_neg / n_all) if n_pos > 0 else 1.0
    w_neg = (n_pos / n_all) if n_neg > 0 else 1.0

    have_val = (Xval is not None and yval is not None and yval.size > 0
                and 0 < yval.sum() < yval.size)
    Xv = Tensor(Xval.astype(np.float32)) if have_val else None

    best_auc, best_snap, since = -1.0, None, 0
    for ep in range(max_epochs):
        with Tensor.train():
            opt.zero_grad()
            loss = _bce_weighted(head.logits(Xt), yt, w_pos, w_neg) + l2 * head.l2()
            loss.backward()
            opt.step()
        if have_val:
            sv = head.logits(Xv).numpy()
            vauc = auc_mann_whitney(sv, yval.astype(bool))
            if vauc > best_auc + 1e-4:
                best_auc = vauc
                best_snap = [p.numpy().copy() for p in head.params]
                since = 0
            else:
                since += 1
                if since >= patience:
                    break
    if best_snap is not None:                       # restore best-val-AUC weights.
        for p, snap in zip(head.params, best_snap):
            p.assign(Tensor(snap))
    return head


# ===========================================================================
# THE GATE — learned (linear + nonlinear@d) vs the EXISTING PCA baseline@d, SAME
# centered reps, SAME by-instance folds, SAME AUC.
# ===========================================================================

def learned_cv_auc(reps_centered, valid, inst_id, fold_id, n_folds, hidden, l2=1e-3,
                   lr=3e-3, max_epochs=300, patience=25, seed=0):
    """Cross-validated held-out AUC of a learned head (hidden=None -> linear; int ->
    nonlinear with that hidden/waist dim). Per fold: standardize on TRAIN stats (mirrors
    _cv_logistic_auc), by-instance within-train val split for early-stop, fit, score the
    held-out fold. POOL all out-of-fold scores -> ONE global Mann-Whitney AUC (the SAME
    pooling the probe uses)."""
    reps_centered = np.asarray(reps_centered, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    inst_id = np.asarray(inst_id)
    n = reps_centered.shape[0]
    oof = np.full(n, np.nan)
    for f in range(n_folds):
        te = fold_id == f
        tr = ~te
        if valid[tr].sum() == 0 or (~valid[tr]).sum() == 0:
            continue
        Xtr_raw, Xte_raw = reps_centered[tr], reps_centered[te]
        # standardize on TRAIN stats (identical to _cv_logistic_auc).
        mu = Xtr_raw.mean(axis=0, keepdims=True)
        sd = Xtr_raw.std(axis=0, keepdims=True)
        sd = np.where(sd < 1e-8, 1.0, sd)
        Xtr = (Xtr_raw - mu) / sd
        Xte = (Xte_raw - mu) / sd
        ytr = valid[tr]
        # BY-INSTANCE within-train val split for early-stop: split the TRAIN-row instance
        # ids so no dart of a fit-instance leaks into the ES val (keeps ES instance-clean,
        # same discipline as the outer CV).
        inst_tr = inst_id[tr]
        fit_m, val_m = _by_instance_val_split(inst_tr, val_frac=0.2, seed=seed)
        if val_m.sum() == 0 or ytr[val_m].sum() == 0 or (~ytr[val_m]).sum() == 0:
            fit_m = np.ones(ytr.shape[0], dtype=bool)   # no clean val -> fixed epochs.
            val_m = np.zeros(ytr.shape[0], dtype=bool)
        Xfit, yfit = Xtr[fit_m], ytr[fit_m]
        Xval = Xtr[val_m] if val_m.any() else None
        yval = ytr[val_m] if val_m.any() else None
        head = train_head(Xfit, yfit, Xval, yval, hidden=hidden, l2=l2, lr=lr,
                          max_epochs=max_epochs, patience=patience, seed=seed + f)
        oof[te] = head.logits(Tensor(Xte.astype(np.float32))).numpy()
    ok = ~np.isnan(oof)
    if ok.sum() == 0:
        return float("nan"), 0
    return auc_mann_whitney(oof[ok], valid[ok]), int(ok.sum())


def run_gate(reps, valid, inst_id, dims=(512, 256, 128), folds=5, seed=0,
             l2=1e-3, lr=3e-3, max_epochs=300, patience=25, verbose=True):
    """The gate. CENTER by instance (identity removal), assign by-instance folds, then for
    each requested waist dim d compute three SAME-protocol held-out AUCs:
        PCA_AUC               = _cv_logistic_auc(centered, max_dim=d)  [existing path]
        learned_linear_AUC    = learned head, NO hidden (1024->1), supervised
        learned_nonlinear_AUC = learned head, 1024->d->GELU->1, supervised
    Linear head is computed ONCE (d-independent). Returns the table rows + verdict."""
    reps = np.asarray(reps, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    inst_id = np.asarray(inst_id)
    centered = _center_by_instance(reps, inst_id)
    fold_id, n_folds = assign_instance_folds(inst_id, folds=folds, seed=seed)

    # learned LINEAR head — d-independent (sees all 1024 dims). Compute once.
    lin_auc, lin_n = learned_cv_auc(centered, valid, inst_id, fold_id, n_folds,
                                    hidden=None, l2=l2, lr=lr, max_epochs=max_epochs,
                                    patience=patience, seed=seed)

    rows = []
    for d in dims:
        pca = _cv_logistic_auc(centered, valid, inst_id, folds=folds, seed=seed, max_dim=d)
        nl_auc, nl_n = learned_cv_auc(centered, valid, inst_id, fold_id, n_folds,
                                      hidden=int(d), l2=l2, lr=lr, max_epochs=max_epochs,
                                      patience=patience, seed=seed)
        rows.append({
            "dim": int(d),
            "pca_auc": float(pca["auc"]),
            "learned_linear_auc": float(lin_auc),
            "learned_nonlinear_auc": float(nl_auc),
            "pca_n": int(pca["n"]),
            "nl_n": int(nl_n),
        })

    V = gate_verdict(rows, lin_auc)
    if verbose:
        _print_gate(reps.shape, valid, inst_id, rows, lin_auc, lin_n, V, seed)
    return {"rows": rows, "learned_linear_auc": float(lin_auc), "verdict": V}


def gate_verdict(rows, lin_auc) -> dict:
    """The go/no-go read (thresholds stated):
      GREEN LIGHT iff at some d: nonlinear >= PCA+0.05 AND nonlinear >= 0.70.
      NOT JUSTIFIED iff at every d: |nonlinear - PCA| <= 0.02 (ties PCA everywhere).
      else WEAK/PARTIAL: beats PCA but not the green bar (or beats at some d only).
    Also: nonlinearity HELPS iff best nonlinear > best (PCA, linear) by > 0.02."""
    def _f(x):
        return isinstance(x, float) and np.isfinite(x)

    usable = [r for r in rows if _f(r["pca_auc"]) and _f(r["learned_nonlinear_auc"])]
    green = [r for r in usable
             if r["learned_nonlinear_auc"] >= r["pca_auc"] + 0.05
             and r["learned_nonlinear_auc"] >= 0.70]
    ties = bool(usable) and all(abs(r["learned_nonlinear_auc"] - r["pca_auc"]) <= 0.02
                                for r in usable)

    best_pca = max((r["pca_auc"] for r in usable), default=float("nan"))
    best_nl = max((r["learned_nonlinear_auc"] for r in usable), default=float("nan"))
    best_nl_row = (max(usable, key=lambda r: r["learned_nonlinear_auc"])
                   if usable else None)
    # does NONLINEARITY buy anything over the best LINEAR read (PCA-linear or learned-linear)?
    best_linear_read = max([v for v in [best_pca, lin_auc] if _f(v)], default=float("nan"))
    nonlinear_helps = _f(best_nl) and _f(best_linear_read) and best_nl > best_linear_read + 0.02

    lines = []
    if green:
        gr = max(green, key=lambda r: r["learned_nonlinear_auc"])
        lines.append(
            f"GREEN LIGHT — build the in-deducer waist. At d={gr['dim']} the LEARNED "
            f"NONLINEAR head reaches AUC={gr['learned_nonlinear_auc']:.3f} vs PCA "
            f"{gr['pca_auc']:.3f} (>= PCA+0.05 AND >= 0.70): there is real nonlinear-"
            f"separable headroom the linear/unsupervised PCA read misses. Best waist "
            f"dim ~{gr['dim']}.")
    elif ties:
        lines.append(
            f"NOT JUSTIFIED — the learned NONLINEAR head ties PCA at EVERY d "
            f"(|nonlinear-PCA| <= 0.02 throughout; best nonlinear={best_nl:.3f} vs best "
            f"PCA={best_pca:.3f}). The frozen rep has NO extra nonlinear-separable "
            f"structure beyond the linear/unsupervised PCA lower bound -> the signal is "
            f"faint; the expensive in-deducer waist is NOT justified by this evidence.")
    else:
        bd = best_nl_row["dim"] if best_nl_row else "?"
        lines.append(
            f"WEAK / PARTIAL — the LEARNED NONLINEAR head BEATS PCA but not the GREEN bar "
            f"(best nonlinear={best_nl:.3f} at d={bd} vs best PCA={best_pca:.3f}; needs "
            f">= PCA+0.05 AND >= 0.70). Some supervised/nonlinear headroom exists but it "
            f"is modest -> an in-deducer waist might help, evidence is suggestive not "
            f"decisive.")

    if nonlinear_helps:
        lines.append(
            f"NONLINEARITY HELPS: best nonlinear AUC={best_nl:.3f} > best LINEAR read "
            f"(max of PCA={best_pca:.3f}, learned-linear={lin_auc:.3f})={best_linear_read:.3f} "
            f"by > 0.02 — the hidden layer captures structure a linear map cannot.")
    else:
        lines.append(
            f"NONLINEARITY DOES NOT HELP: best nonlinear AUC={best_nl:.3f} ~<= best LINEAR "
            f"read={best_linear_read:.3f} — a plain linear separator is as good; the "
            f"separable signal (if any) is essentially LINEAR (so PCA already captures it).")

    return {
        "green_light": bool(green),
        "not_justified": bool(ties),
        "nonlinear_helps": bool(nonlinear_helps),
        "best_nonlinear_auc": float(best_nl) if _f(best_nl) else float("nan"),
        "best_pca_auc": float(best_pca) if _f(best_pca) else float("nan"),
        "best_nonlinear_dim": (best_nl_row["dim"] if best_nl_row else None),
        "lines": lines,
    }


def _print_gate(shape, valid, inst_id, rows, lin_auc, lin_n, V, seed) -> None:
    n, H = shape
    print(f"\n{'='*78}", flush=True)
    print(f"  LEARNED WAIST GATE — learned (supervised, nonlinear) vs PCA "
          f"(linear, unsupervised) LOWER BOUND", flush=True)
    print(f"{'='*78}", flush=True)
    print(f"  n_darts={n}  H={H}  VALID={int(valid.sum())} "
          f"({valid.mean()*100:.1f}%, minority)  INVALID={int((~valid).sum())}  "
          f"instances={int(np.unique(inst_id).shape[0])}  seed={seed}", flush=True)
    print(f"  protocol: per-instance-mean CENTERED reps (identity removed) | by-INSTANCE "
          f"5-fold CV | per-fold standardize | pooled OOF Mann-Whitney AUC", flush=True)
    print(f"  (REUSED from dart_cluster_probe: _center_by_instance, the by-instance fold "
          f"map, _cv_logistic_auc PCA baseline, auc_mann_whitney)", flush=True)

    print(f"\n  LEARNED LINEAR head (1024->1, supervised, all dims) AUC = {lin_auc:.3f}  "
          f"(n={lin_n}) — the learned LINEAR sanity row", flush=True)
    print(f"\n  {'dim':>5} | {'PCA_AUC':>9} | {'learned_linear':>14} | "
          f"{'learned_nonlinear':>17}", flush=True)
    print(f"  {'-'*5}-+-{'-'*9}-+-{'-'*14}-+-{'-'*17}", flush=True)
    for r in rows:
        nl = r["learned_nonlinear_auc"]
        pca = r["pca_auc"]
        flag = "  <- beats PCA+0.05" if (np.isfinite(nl) and np.isfinite(pca)
                                         and nl >= pca + 0.05) else ""
        print(f"  {r['dim']:>5} | {pca:>9.3f} | {lin_auc:>14.3f} | "
              f"{nl:>17.3f}{flag}", flush=True)
    print(f"  (PCA_AUC == dart_cluster_probe centered-global AUC at that d, by construction. "
          f"learned_linear is d-independent.)", flush=True)

    print(f"\n  VERDICT", flush=True)
    for ln in V["lines"]:
        print(f"    - {ln}", flush=True)

    print(f"\n  CAVEATS (honest, do NOT over-read):", flush=True)
    print(f"    (1) FROZEN-REP LOWER BOUND: this trains on FROZEN baseline silhouettes; an "
          f"in-deducer waist ALSO RESHAPES the reps during training. A null here is "
          f"SUGGESTIVE, not fully conclusive, for the in-deducer waist.", flush=True)
    print(f"    (2) SEPARABILITY-AS-PROXY: separating valid/invalid is the OPERATIONAL "
          f"proxy for 'good clusters' — a strong classifier => a usable common mode for "
          f"steering / a reusable library.", flush=True)
    print(f"    (3) CENTERING / BY-INSTANCE: by-instance CV + per-instance centering remove "
          f"the graph-identity confound. The raw uncentered AUC (~0.755) is NOT the target; "
          f"0.5 is the floor, the PCA centered-global ~0.658 is the bar to beat.", flush=True)


# ===========================================================================
# CPU SELFTEST — nonlinear-separable (learned >> PCA) + null (all ~0.5).
# ===========================================================================

def _make_nonlinear_synth(n_inst, m, H, seed=0, p_valid=0.4):
    """XOR-like NONLINEAR-separable bank: in a 2d subspace embedded in H-d, validity =
    XOR(sign(a), sign(b)) (a quadrant pattern). A LINEAR separator (PCA+logistic) is ~0.5
    on this; a nonlinear head can solve it. Each instance gets a graph-IDENTITY mean
    (removed by centering) so the by-instance protocol is exercised. The 2d signal axes are
    SHARED across instances (a transferable nonlinear good-pattern) so the held-out fold has
    the same boundary."""
    rng = np.random.RandomState(seed)
    # two SHARED orthonormal signal axes (the transferable nonlinear boundary).
    Q, _ = np.linalg.qr(rng.randn(H, 2))
    ax_a, ax_b = Q[:, 0], Q[:, 1]
    reps, valid, inst_id = [], [], []
    for g in range(n_inst):
        identity = rng.randn(H) * 3.0               # graph identity (centering removes it).
        for _j in range(m):
            a = rng.randn()
            b = rng.randn()
            is_valid = bool((a > 0) != (b > 0))     # XOR quadrants -> NOT linearly separable.
            # bias the marginal toward p_valid by occasionally flipping label-less darts:
            if rng.rand() > (2 * p_valid if is_valid else 2 * (1 - p_valid)):
                # downsample the over-represented class to hit ~p_valid prior.
                continue
            vec = (identity + 2.5 * a * ax_a + 2.5 * b * ax_b + 0.15 * rng.randn(H))
            reps.append(vec.astype(np.float32))
            valid.append(is_valid)
            inst_id.append(g)
    return (np.asarray(reps, dtype=np.float64), np.asarray(valid, dtype=bool),
            np.asarray(inst_id, dtype=np.int64))


def _make_null_synth(n_inst, m, H, seed=0, p_valid=0.3):
    """NULL: validity is an independent coin flip, rep is identity + isotropic noise. Both
    PCA and the learned head must read ~0.5 by-instance (no false green light)."""
    rng = np.random.RandomState(seed)
    reps, valid, inst_id = [], [], []
    for g in range(n_inst):
        identity = rng.randn(H) * 3.0
        for _j in range(m):
            is_valid = bool(rng.rand() < p_valid)
            vec = identity + 1.0 * rng.randn(H)
            reps.append(vec.astype(np.float32))
            valid.append(is_valid)
            inst_id.append(g)
    return (np.asarray(reps, dtype=np.float64), np.asarray(valid, dtype=bool),
            np.asarray(inst_id, dtype=np.int64))


def selftest() -> bool:
    print("=== learned_waist_gate SELFTEST (CPU) ===", flush=True)
    ok = True
    H = 64

    # --- NONLINEAR-separable: learned_nonlinear HIGH, PCA (linear) ~0.5. ---
    reps, valid, inst = _make_nonlinear_synth(30, 40, H, seed=1, p_valid=0.4)
    print(f"  [nonlinear-synth] n={reps.shape[0]} valid_frac={valid.mean():.2f} "
          f"instances={len(np.unique(inst))}", flush=True)
    res = run_gate(reps, valid, inst, dims=(16, 8), folds=5, seed=1,
                   l2=1e-3, lr=5e-3, max_epochs=400, patience=40, verbose=False)
    best_nl = res["verdict"]["best_nonlinear_auc"]
    best_pca = res["verdict"]["best_pca_auc"]
    lin = res["learned_linear_auc"]
    print(f"  [nonlinear-synth] best_nonlinear={best_nl:.3f} (expect HIGH >0.75) | "
          f"best_PCA={best_pca:.3f} (expect ~0.5) | learned_linear={lin:.3f} (~0.5)",
          flush=True)
    cond_nl = (best_nl > 0.75 and best_pca < 0.62 and best_nl > best_pca + 0.10)
    print(f"  [nonlinear-synth] EXPECT learned_nonlinear DETECTS what PCA misses -> "
          f"{'PASS' if cond_nl else 'FAIL'}", flush=True)
    ok &= cond_nl
    # the gate must NOT report 'not_justified' here (nonlinear headroom is real).
    print(f"  [nonlinear-synth] verdict green_light={res['verdict']['green_light']} "
          f"not_justified={res['verdict']['not_justified']} "
          f"nonlinear_helps={res['verdict']['nonlinear_helps']}", flush=True)
    ok &= (res["verdict"]["not_justified"] is False
           and res["verdict"]["nonlinear_helps"] is True)

    # --- NULL: everything ~0.5 by-instance (no false green light). ---
    reps0, valid0, inst0 = _make_null_synth(80, 40, H, seed=2, p_valid=0.3)
    print(f"\n  [null-synth] n={reps0.shape[0]} valid_frac={valid0.mean():.2f} "
          f"instances={len(np.unique(inst0))}", flush=True)
    res0 = run_gate(reps0, valid0, inst0, dims=(16, 8), folds=5, seed=2,
                    l2=1e-3, lr=5e-3, max_epochs=400, patience=40, verbose=False)
    best_nl0 = res0["verdict"]["best_nonlinear_auc"]
    best_pca0 = res0["verdict"]["best_pca_auc"]
    lin0 = res0["learned_linear_auc"]
    print(f"  [null-synth] best_nonlinear={best_nl0:.3f} best_PCA={best_pca0:.3f} "
          f"learned_linear={lin0:.3f} (ALL expect ~0.5)", flush=True)
    cond_null = (abs(best_nl0 - 0.5) <= 0.07 and abs(best_pca0 - 0.5) <= 0.07
                 and abs(lin0 - 0.5) <= 0.07)
    print(f"  [null-synth] EXPECT all ~0.5 by-instance (no false green light) -> "
          f"{'PASS' if cond_null else 'FAIL'}", flush=True)
    ok &= cond_null
    # the gate must NOT green-light the null.
    print(f"  [null-synth] verdict green_light={res0['verdict']['green_light']} "
          f"(expect False)", flush=True)
    ok &= res0["verdict"]["green_light"] is False

    print(f"\n  SELFTEST {'PASSED' if ok else 'FAILED'}", flush=True)
    return ok


# ===========================================================================
# Main
# ===========================================================================

def run_on_npz(npz_path, dims, folds=5, seed=0, l2=1e-3, lr=3e-3, max_epochs=300,
               patience=25):
    z = np.load(npz_path, allow_pickle=True)
    reps = np.asarray(z["reps"], dtype=np.float64)
    valid = np.asarray(z["valid"], dtype=bool)
    inst_id = np.asarray(z["inst_id"])
    return run_gate(reps, valid, inst_id, dims=tuple(dims), folds=folds, seed=seed,
                    l2=l2, lr=lr, max_epochs=max_epochs, patience=patience, verbose=True)


def main(argv=None) -> int:
    P = argparse.ArgumentParser(
        description="Learned-waist gate: learned nonlinear vs PCA lower bound on dart reps")
    P.add_argument("--npz", default=os.environ.get("DART_NPZ", None),
                   help="path to the per-dart silhouette .npz")
    P.add_argument("--selftest", action="store_true",
                   help="run the CPU selftest (nonlinear-separable + null) and exit")
    P.add_argument("--dims", default="512,256,128",
                   help="comma list of waist dims to sweep (default 512,256,128)")
    P.add_argument("--folds", type=int, default=5, help="CV folds (by instance)")
    P.add_argument("--seed", type=int, default=0, help="CV / init seed")
    P.add_argument("--l2", type=float, default=1e-3, help="L2 weight decay")
    P.add_argument("--lr", type=float, default=3e-3, help="Adam lr")
    P.add_argument("--max-epochs", type=int, default=300, help="max epochs (early-stopped)")
    P.add_argument("--patience", type=int, default=25, help="early-stop patience (epochs)")
    args = P.parse_args(argv)

    if args.selftest or os.environ.get("SELFTEST_ONLY", "0") == "1":
        return 0 if selftest() else 1
    if not args.npz:
        print("error: pass --npz PATH (or --selftest).", flush=True)
        return 2
    if not os.path.exists(args.npz):
        print(f"error: npz not found: {args.npz}", flush=True)
        return 2
    try:
        dims = sorted({int(t) for t in args.dims.split(",") if t.strip()}, reverse=True)
    except ValueError:
        print(f"error: --dims must be a comma list of ints, got {args.dims!r}", flush=True)
        return 2
    if not dims:
        print("error: --dims produced no valid dims", flush=True)
        return 2
    run_on_npz(args.npz, dims, folds=args.folds, seed=args.seed, l2=args.l2, lr=args.lr,
               max_epochs=args.max_epochs, patience=args.patience)
    return 0


if __name__ == "__main__":
    parse_ok = _ast_parse_ok()
    print(f"[ast.parse] astparse_ok={parse_ok}", flush=True)
    if not parse_ok:
        sys.exit(1)
    sys.exit(main())
