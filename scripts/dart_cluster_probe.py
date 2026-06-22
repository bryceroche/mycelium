"""dart_cluster_probe.py — the ANNA-KARENINA per-dart cluster probe (pure numpy, CPU).

THE HYPOTHESIS (Anna-Karenina / George-Hotz "good drivers are good in the SAME way"):
in the generate-and-verify volume run, the deducer throws M solution-preserving symmetry
darts per instance; a FREE exact verifier tags each VALID or INVALID. The claim: VALID
darts CLUSTER (a shared "common mode" / silhouette) while INVALID darts SCATTER. If true,
a centroid of valid darts is a reusable "good pattern" we could later match/bias toward
(raising per-dart p). This script TESTS it on the captured per-dart silhouettes.

THE INPUT: an .npz from `scripts/amortized_frontier_measure.py --capture-darts` with
  reps    (n_darts, H) float32  — per-dart SILHOUETTE = final-breath readout-LN 1024d
                                  hidden, mean-pooled over valid cells (one (H,) per dart)
  valid   (n_darts,) bool       — the FREE exact-verifier flag (VALID coloring or not)
  inst_id (n_darts,) int        — STABLE global id, band-namespaced (distinct per
                                  (band, instance)) -> the WITHIN-INSTANCE grouping key
  band    (n_darts,) float      — edge-density band c
  meta    (object)              — dict {ckpt, mech, H, rep, ...}

THE FOUR METRICS (with an explicit VERDICT):

(A) ANNA-KARENINA VARIANCE TEST. For each instance with >=3 VALID and >=3 INVALID darts,
    compute the SPREAD (mean cosine distance to the class centroid) of its VALID darts vs
    its INVALID darts. Aggregate: the FRACTION of such instances where VALID spread <
    INVALID spread (good is TIGHTER), and the mean valid/invalid spread RATIO. Prediction
    if the hypothesis holds: valid tighter -> fraction >> 0.5, ratio < 1.

(B) WITHIN-INSTANCE SEPARABILITY. Per mixed instance, score each dart by cosine-similarity
    to that instance's VALID centroid (mean of its valid reps); compute AUC(valid vs
    invalid) via Mann-Whitney; report the MEAN AUC over instances (+ distribution). AUC~0.5
    = no signal; >>0.5 = valid darts are separable from invalid by proximity to the good
    centroid. WITHIN-INSTANCE -> measures SOLUTION QUALITY (same graph), not graph-identity.

(C) GLOBAL / TRANSFERABLE GOOD-PATTERN (the library question). The confound: across
    instances the rep is dominated by graph-IDENTITY, not solution-quality. Control for it:
    subtract each instance's OWN mean rep (over its darts) from every dart -> instance-
    identity-removed reps. Then test global VALID-vs-INVALID separability (a cross-validated
    logistic probe, held-out AUC) on the CENTERED reps. Also report the RAW (uncentered)
    global AUC for contrast and FLAG that raw global separability is confounded by graph-
    identity. Library-viable signal = centered-global AUC meaningfully > 0.5.

(D) VERDICT. A clear, quantitative read: does the common-mode hypothesis hold WITHIN-
    instance (the necessary precursor: valid tighter + within-AUC > ~0.65)? Is a
    transferable good-pattern plausible (centered-global AUC > ~0.6)? Or is the 1024d
    readout rep NOT separating valid from invalid (-> the signal may need the compression
    of a 512-waist to emerge, OR the hypothesis fails)? Honest + quantitative; raw-global is
    NOT over-read (it is identity-confounded).

CPU SELFTEST (no GPU, no npz): SELFTEST_ONLY=1 (or --selftest) builds synthetic data where
valid darts ARE a tight cluster + invalid scattered -> the probe must report tighter + high
AUC; and a NULL where valid/invalid are intermixed -> the probe must report ~0.5 AUC /
ratio ~1 (no false positive). Both are ASSERTED.

USAGE:
  CPU selftest (GPU-free):
    .venv/bin/python3 scripts/dart_cluster_probe.py --selftest
  Probe a captured npz:
    .venv/bin/python3 scripts/dart_cluster_probe.py --npz .cache/dart_silhouettes.npz
"""
from __future__ import annotations

import argparse
import ast
import os
import sys

_THIS_FILE = os.path.abspath(__file__)

import numpy as np


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
# Core numerical machinery (pure numpy)
# ===========================================================================

def _l2norm_rows(x: np.ndarray) -> np.ndarray:
    """Row-normalize (unit L2). Zero rows -> left as zero (denominator clamped)."""
    x = np.asarray(x, dtype=np.float64)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.where(n < 1e-12, 1.0, n)
    return x / n


def _mean_cosdist_to_centroid(rows: np.ndarray) -> float:
    """Mean cosine DISTANCE (1 - cos) of `rows` to their (unit-normalized) centroid.

    SPREAD measure: 0 = all identical direction (tight), up to 2 = antipodal (scattered).
    Uses the MEAN of unit rows as the centroid direction (a robust cluster-spread proxy).

    NOTE: this is the IN-SAMPLE centroid -> the spread it reports is DOWNWARD-biased for
    small n (the centroid overfits its own points: with n points it explains ~1/n of the
    variance for free). So this raw estimator is NOT n-fair across classes of different size
    -- use `_spread_at_n` to compare two classes (it subsamples to a common n). FIX 2."""
    rows = np.asarray(rows, dtype=np.float64)
    if rows.shape[0] < 2:
        return 0.0
    unit = _l2norm_rows(rows)
    centroid = unit.mean(axis=0)
    cn = np.linalg.norm(centroid)
    if cn < 1e-12:
        # rows cancel out -> maximally scattered.
        return 1.0
    centroid = centroid / cn
    cos = unit @ centroid                      # (n,) cos to centroid direction
    return float(np.mean(1.0 - cos))


def _spread_at_n(rows: np.ndarray, n_target: int, n_sub: int, rng) -> float:
    """n-FAIR spread of `rows`: mean cos-distance to the in-sample centroid, but estimated
    from a random subsample of exactly `n_target` rows (averaged over `n_sub` subsamples for
    stability). FIX 2 — the in-sample centroid overfits its own points, so a class with FEWER
    darts reads artificially TIGHTER; estimating both classes at the SAME n removes that bias.

    If rows already has <= n_target rows, all rows are used (no subsample needed)."""
    rows = np.asarray(rows, dtype=np.float64)
    n = rows.shape[0]
    if n <= n_target:
        return _mean_cosdist_to_centroid(rows)
    vals = []
    for _ in range(n_sub):
        idx = rng.choice(n, size=n_target, replace=False)
        vals.append(_mean_cosdist_to_centroid(rows[idx]))
    return float(np.mean(vals))


def _cos_to_vector(rows: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Cosine similarity of each row to `vec` -> (n,)."""
    unit = _l2norm_rows(rows)
    v = np.asarray(vec, dtype=np.float64)
    vn = np.linalg.norm(v)
    if vn < 1e-12:
        return np.zeros(unit.shape[0])
    return unit @ (v / vn)


def auc_mann_whitney(scores: np.ndarray, labels: np.ndarray) -> float:
    """AUC(positive vs negative) via the Mann-Whitney U statistic (rank-based; ties
    handled by average ranks). scores higher => more positive. labels bool (True=pos).

    Returns 0.5 when either class is empty (no signal / undefined)."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=bool)
    n_pos = int(labels.sum())
    n_neg = int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5
    # average ranks (1..N), ties shared.
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(scores.shape[0], dtype=np.float64)
    sorted_scores = scores[order]
    i = 0
    N = scores.shape[0]
    while i < N:
        j = i
        while j + 1 < N and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0          # 1-based average rank for the tie block
        ranks[order[i:j + 1]] = avg_rank
        i = j + 1
    sum_ranks_pos = ranks[labels].sum()
    u_pos = sum_ranks_pos - n_pos * (n_pos + 1) / 2.0
    return float(u_pos / (n_pos * n_neg))


# ===========================================================================
# (A) ANNA-KARENINA VARIANCE TEST
# ===========================================================================

def metric_A_variance(reps, valid, inst_id, min_per_class=3, n_sub=12, seed=0) -> dict:
    """For each instance with >=min_per_class VALID and >=min_per_class INVALID darts,
    compare VALID spread vs INVALID spread (mean cos-dist to class centroid).

    n-FAIR (FIX 2): the in-sample centroid is downward-biased for small n, so the SMALLER
    class would read artificially tighter. Before comparing, both classes are estimated at
    the SAME n = min(nv, ni) by subsampling the larger class down (averaged over `n_sub`
    random subsamples for stability). This makes the spread comparison size-fair so the NULL
    ratio reads ~1.0 (not <1) even when valid is the minority.

    Returns:
      n_instances     : # mixed instances meeting the >=k/k threshold.
      frac_valid_tighter : fraction where valid spread < invalid spread (good >> 0.5).
      mean_ratio      : mean over instances of (valid_spread / invalid_spread) (good < 1).
      median_ratio    : median ratio.
      valid_spreads / invalid_spreads : per-instance spread lists (for the distribution).
    """
    rng = np.random.RandomState(seed)
    uids = np.unique(inst_id)
    vspreads, ispreads, ratios = [], [], []
    tighter = 0
    n_used = 0
    for uid in uids:
        sel = inst_id == uid
        vmask = sel & valid
        imask = sel & (~valid)
        nv, ni = int(vmask.sum()), int(imask.sum())
        if nv < min_per_class or ni < min_per_class:
            continue
        n_common = min(nv, ni)                 # estimate BOTH classes at this same n.
        vs = _spread_at_n(reps[vmask], n_common, n_sub, rng)
        is_ = _spread_at_n(reps[imask], n_common, n_sub, rng)
        vspreads.append(vs)
        ispreads.append(is_)
        ratios.append(vs / is_ if is_ > 1e-12 else float("inf"))
        if vs < is_:
            tighter += 1
        n_used += 1
    finite_ratios = [r for r in ratios if np.isfinite(r)]
    return {
        "n_instances": n_used,
        "frac_valid_tighter": (tighter / n_used) if n_used else float("nan"),
        "mean_ratio": (float(np.mean(finite_ratios)) if finite_ratios else float("nan")),
        "median_ratio": (float(np.median(finite_ratios)) if finite_ratios else float("nan")),
        "mean_valid_spread": (float(np.mean(vspreads)) if vspreads else float("nan")),
        "mean_invalid_spread": (float(np.mean(ispreads)) if ispreads else float("nan")),
        "valid_spreads": vspreads, "invalid_spreads": ispreads,
        "min_per_class": min_per_class,
    }


# ===========================================================================
# (B) WITHIN-INSTANCE SEPARABILITY (proximity to the per-instance VALID centroid)
# ===========================================================================

def metric_B_within_auc(reps, valid, inst_id, min_per_class=3, seed=0) -> dict:
    """Per mixed instance, score each dart by cosine-sim to that instance's VALID centroid
    (mean of its valid reps); AUC(valid vs invalid). Report MEAN AUC over instances.

    WITHIN-INSTANCE -> measures SOLUTION QUALITY on the SAME graph (NOT graph-identity).

    SYMMETRIC INCLUSION (FIX 3): the prior code scored VALID darts against a leave-one-out
    valid centroid (without themselves) but INVALID darts against the FULL valid centroid.
    That asymmetry biases the AUC: valid darts paid the self-EXCLUSION penalty, invalid darts
    did not -> valid scored lower than it should, but more importantly the two classes were
    scored against DIFFERENT centroids under DIFFERENT rules.

    Unbiased symmetric rule: build the valid centroid from a FIXED random subsample of the
    instance's valid darts (the "anchor" set), then score EVERY dart NOT in that anchor set
    (valid or invalid) against that SAME centroid under the SAME rule. Valid anchors are
    excluded from scoring (they helped build the centroid), so no dart is ever scored against
    a centroid that contains it. Under the NULL this reads ~0.5."""
    rng = np.random.RandomState(seed)
    uids = np.unique(inst_id)
    aucs = []
    for uid in uids:
        sel = inst_id == uid
        vmask = sel & valid
        imask = sel & (~valid)
        nv, ni = int(vmask.sum()), int(imask.sum())
        if nv < min_per_class or ni < min_per_class:
            continue
        v_idx = np.nonzero(vmask)[0]
        i_idx = np.nonzero(imask)[0]
        # FIXED anchor subset of valid darts -> the centroid. Hold out at least 1 valid dart
        # to score (so valid is represented on the scoring side too). Half the valid darts,
        # capped so >=1 remains for scoring.
        n_anchor = max(1, min(nv - 1, nv // 2))
        anchor_local = rng.choice(nv, size=n_anchor, replace=False)
        anchor_set = set(int(g) for g in v_idx[anchor_local])
        centroid = reps[v_idx[anchor_local]].astype(np.float64).mean(axis=0)
        # score EVERY dart not in the anchor set against the SAME centroid, SAME rule.
        score_valid = np.array([g for g in v_idx if int(g) not in anchor_set])
        score_idx = np.concatenate([score_valid, i_idx]) if score_valid.size else i_idx
        if score_valid.size == 0:
            continue                            # no valid dart left to score -> skip instance.
        scores = _cos_to_vector(reps[score_idx], centroid)
        labels = valid[score_idx]
        aucs.append(auc_mann_whitney(scores, labels))
    aucs = np.asarray(aucs, dtype=np.float64)
    return {
        "n_instances": int(aucs.shape[0]),
        "mean_auc": (float(np.mean(aucs)) if aucs.size else float("nan")),
        "median_auc": (float(np.median(aucs)) if aucs.size else float("nan")),
        "std_auc": (float(np.std(aucs)) if aucs.size else float("nan")),
        "p25_auc": (float(np.percentile(aucs, 25)) if aucs.size else float("nan")),
        "p75_auc": (float(np.percentile(aucs, 75)) if aucs.size else float("nan")),
        "frac_auc_gt_065": (float(np.mean(aucs > 0.65)) if aucs.size else float("nan")),
        "aucs": aucs.tolist(),
        "min_per_class": min_per_class,
    }


# ===========================================================================
# (C) GLOBAL / TRANSFERABLE GOOD-PATTERN (instance-identity-removed logistic probe)
# ===========================================================================

def _center_by_instance(reps, inst_id) -> np.ndarray:
    """Subtract each instance's OWN mean rep (over its darts) from every dart. Removes the
    graph-IDENTITY common mode so what remains is the WITHIN-graph solution-quality axis."""
    out = np.array(reps, dtype=np.float64, copy=True)
    for uid in np.unique(inst_id):
        sel = inst_id == uid
        out[sel] -= out[sel].mean(axis=0, keepdims=True)
    return out


def _logreg_fit(X, y, l2=1.0, iters=300, lr=0.5):
    """Tiny L2-regularized logistic regression via gradient descent (pure numpy). X is
    (n, d) standardized; y bool. Returns weights (d,) + bias. No external deps."""
    n, d = X.shape
    w = np.zeros(d, dtype=np.float64)
    b = 0.0
    yf = y.astype(np.float64)
    for _ in range(iters):
        z = X @ w + b
        z = np.clip(z, -30, 30)
        p = 1.0 / (1.0 + np.exp(-z))
        g = p - yf
        gw = X.T @ g / n + l2 * w / n
        gb = float(g.mean())
        w -= lr * gw
        b -= lr * gb
    return w, b


def _cv_logistic_auc(reps, labels, inst_id, folds=5, l2=1.0, seed=0, max_dim=256) -> dict:
    """Cross-validated held-out AUC of a logistic probe predicting `labels` from `reps`.

    Standardizes per fold (train stats), optionally PCA-reduces to <=max_dim (train-fit),
    fits the tiny logreg, scores held-out, pools out-of-fold scores -> ONE global AUC. This
    is the honest 'is the class linearly separable, generalizing across instances' test.

    BY-INSTANCE SPLIT (FIX 1): folds are assigned by WHOLE INSTANCE (using `inst_id`), so no
    dart of a train-instance ever appears in test. Splitting by individual dart leaks residual
    per-instance structure (which survives even instance-mean centering) into the held-out
    fold -> an inflated CENTERED-global AUC (a false 'transferable good-pattern' signal). With
    the by-instance split, the NULL centered-AUC drops to ~0.5."""
    reps = np.asarray(reps, dtype=np.float64)
    labels = np.asarray(labels, dtype=bool)
    inst_id = np.asarray(inst_id)
    n = reps.shape[0]
    uids = np.unique(inst_id)
    n_inst = uids.shape[0]
    folds = min(folds, n_inst)                  # can't have more folds than instances.
    if (n < 2 * folds or folds < 2 or labels.sum() == 0 or (~labels).sum() == 0):
        return {"auc": float("nan"), "n": n, "note": "too few samples/instances / one class"}
    rng = np.random.RandomState(seed)
    # assign WHOLE instances to folds (round-robin over a permuted instance order).
    inst_perm = rng.permutation(n_inst)
    inst_fold = np.zeros(n_inst, dtype=int)
    inst_fold[inst_perm] = np.arange(n_inst) % folds
    uid_to_fold = {int(u): int(fld) for u, fld in zip(uids, inst_fold)}
    fold_id = np.array([uid_to_fold[int(u)] for u in inst_id], dtype=int)
    oof_scores = np.full(n, np.nan)
    for f in range(folds):
        te = fold_id == f
        tr = ~te
        if labels[tr].sum() == 0 or (~labels[tr]).sum() == 0:
            continue
        Xtr, Xte = reps[tr], reps[te]
        # standardize on train stats.
        mu = Xtr.mean(axis=0, keepdims=True)
        sd = Xtr.std(axis=0, keepdims=True)
        sd = np.where(sd < 1e-8, 1.0, sd)
        Xtr = (Xtr - mu) / sd
        Xte = (Xte - mu) / sd
        # PCA reduce (train-fit) when high-dim, to keep the tiny GD probe well-conditioned.
        if Xtr.shape[1] > max_dim:
            U, S, Vt = np.linalg.svd(Xtr, full_matrices=False)
            comps = Vt[:max_dim].T                     # (d, max_dim)
            Xtr = Xtr @ comps
            Xte = Xte @ comps
        w, b = _logreg_fit(Xtr, labels[tr], l2=l2)
        oof_scores[te] = Xte @ w + b
    valid_oof = ~np.isnan(oof_scores)
    if valid_oof.sum() == 0:
        return {"auc": float("nan"), "n": n, "note": "no scored folds"}
    auc = auc_mann_whitney(oof_scores[valid_oof], labels[valid_oof])
    return {"auc": auc, "n": int(valid_oof.sum()), "note": ""}


def metric_C_global(reps, valid, inst_id, seed=0) -> dict:
    """Global VALID-vs-INVALID separability: CENTERED (instance-identity-removed; the
    library-viable signal) vs RAW (uncentered; identity-CONFOUNDED — flagged, not over-read).
    Both via the SAME cross-validated logistic probe (held-out AUC), split BY INSTANCE
    (FIX 1) so no dart of a train-instance leaks into test."""
    raw = _cv_logistic_auc(reps, valid, inst_id, seed=seed)
    centered = _cv_logistic_auc(_center_by_instance(reps, inst_id), valid, inst_id, seed=seed)
    return {
        "raw_global_auc": raw["auc"], "raw_n": raw["n"], "raw_note": raw["note"],
        "centered_global_auc": centered["auc"], "centered_n": centered["n"],
        "centered_note": centered["note"],
        "n_valid": int(valid.sum()), "n_invalid": int((~valid).sum()),
        "n_instances": int(np.unique(inst_id).shape[0]),
    }


# ===========================================================================
# (D) VERDICT
# ===========================================================================

def verdict(A, B, C) -> dict:
    """Synthesize the read. Thresholds (stated, not hidden):
      WITHIN-instance hypothesis HOLDS iff valid tighter (frac_valid_tighter > 0.5 AND
        mean_ratio < 1) AND within-AUC mean > 0.65 (the necessary precursor).
      TRANSFERABLE good-pattern PLAUSIBLE iff centered-global AUC > 0.6.
      NOT-SEPARATING iff within-AUC ~ 0.5 (|.-0.5| < 0.05) AND mean_ratio ~ 1."""
    ft = A["frac_valid_tighter"]
    mr = A["mean_ratio"]
    wauc = B["mean_auc"]
    cauc = C["centered_global_auc"]
    rauc = C["raw_global_auc"]

    def _f(x):
        return (not isinstance(x, float)) or np.isfinite(x)

    within_holds = (_f(ft) and _f(mr) and _f(wauc)
                    and ft > 0.5 and mr < 1.0 and wauc > 0.65)
    transferable = _f(cauc) and cauc > 0.6
    not_separating = (_f(wauc) and abs(wauc - 0.5) < 0.05
                      and _f(mr) and abs(mr - 1.0) < 0.10)

    lines = []
    if within_holds:
        lines.append(
            f"WITHIN-INSTANCE COMMON-MODE HOLDS: valid darts are TIGHTER "
            f"(frac_tighter={ft:.2f}>0.5, ratio={mr:.2f}<1) AND separable from invalid by "
            f"proximity to the valid centroid (mean within-AUC={wauc:.3f}>0.65). The "
            f"'good darts cluster' precursor is REAL on this graph.")
    elif not_separating:
        lines.append(
            f"NOT SEPARATING: the 1024d readout silhouette does NOT distinguish valid from "
            f"invalid within-instance (mean within-AUC={wauc:.3f}~0.5, spread ratio={mr:.2f}"
            f"~1). The common-mode signal is ABSENT in this rep — it may need the "
            f"compression of a 512-waist to emerge, OR the hypothesis fails here.")
    else:
        lines.append(
            f"WEAK / PARTIAL within-instance signal: frac_tighter={ft:.2f}, spread "
            f"ratio={mr:.2f}, mean within-AUC={wauc:.3f}. Below the 'holds' bar "
            f"(tighter + AUC>0.65) but not flat at chance — a faint common mode, not a "
            f"reliable one.")

    if transferable:
        lines.append(
            f"TRANSFERABLE GOOD-PATTERN PLAUSIBLE: instance-identity-removed (CENTERED) "
            f"global AUC={cauc:.3f}>0.6 — a cross-instance 'good' direction survives "
            f"removing graph-identity. A reusable centroid/library is worth a shot.")
    else:
        lines.append(
            f"TRANSFERABLE GOOD-PATTERN NOT SUPPORTED: centered-global AUC={cauc:.3f}<=0.6 — "
            f"after removing graph-identity, no robust cross-instance good-direction. A "
            f"shared library centroid is NOT justified by this rep.")
    lines.append(
        f"(RAW global AUC={rauc:.3f} is CONFOUNDED by graph-identity — darts of the same "
        f"graph cluster regardless of validity — so it is NOT evidence for a transferable "
        f"good-pattern; read the CENTERED number.)")

    return {
        "within_holds": bool(within_holds),
        "transferable": bool(transferable),
        "not_separating": bool(not_separating),
        "lines": lines,
    }


# ===========================================================================
# PCA DIMENSION SWEEP — "what dim should the common-mode / waist be?"
# ===========================================================================
#
# THE QUESTION (Bryce's hypothesis): a 512d waist's COMPRESSION is what creates the
# common mode. So the Anna-Karenina cluster signal may SHARPEN as we PCA-compress
# 1024 -> 512 -> 256 (dropping idiosyncratic noise dims) even if it is DILUTED at the
# full 1024d rep. We PCA-reduce the captured silhouettes to several dims and re-run all
# three (de-biased) metrics at each dim, looking for the dim where the signal PEAKS.
#
# CAVEAT (printed, do NOT over-read): PCA is LINEAR + UNSUPERVISED, so this is a LOWER
# BOUND on what a learned NONLINEAR waist could recover. Sharpening = green light;
# flat = INCONCLUSIVE (a learned waist might still work) — never "compression fails".


def _pca_project(reps: np.ndarray, d: int) -> np.ndarray:
    """PCA-reduce `reps` (n, H) to `d` dims: center by the GLOBAL mean, SVD, keep the
    top-`d` right-singular vectors, project. Pure numpy (matches the in-fold SVD at
    `_cv_logistic_auc`). When d >= rank/H the projection is rank-preserving (an
    orthonormal rotation of the same information) -> effectively the full/identity rep,
    so d=1024 is the byte-faithful BASELINE row of the sweep.

    Returns (n, d') where d' = min(d, H, n-ish rank). Centering is intentional: PCA
    components are defined on the centered data; the metrics are all
    centering/scale-robust (cosine + per-instance recentering + per-fold standardize),
    so the global-mean removal does not bias them."""
    X = np.asarray(reps, dtype=np.float64)
    mu = X.mean(axis=0, keepdims=True)
    Xc = X - mu
    # economy SVD: Vt rows are the principal directions, ordered by singular value.
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    k = int(min(d, Vt.shape[0]))
    comps = Vt[:k].T                              # (H, k)
    return Xc @ comps                            # (n, k)


def pca_dim_sweep(reps, valid, inst_id, dims, min_per_class=3, seed=0) -> list:
    """Run all three metrics at each PCA dim in `dims`. Reuses the EXISTING de-biased
    metric functions UNCHANGED — just feeds them the reduced reps (metric C still does
    its own per-instance-mean centering inside, on the reduced reps, which is fine).

    Returns a list of per-dim dicts (sorted by requested order) with the headline
    numbers for the table: dim | A_ratio | A_frac_tighter | B_within_AUC | C_centered."""
    rows = []
    for d in dims:
        Xd = _pca_project(reps, d) if d < reps.shape[1] else np.asarray(reps, dtype=np.float64)
        A = metric_A_variance(Xd, valid, inst_id, min_per_class=min_per_class, seed=seed)
        B = metric_B_within_auc(Xd, valid, inst_id, min_per_class=min_per_class, seed=seed)
        C = metric_C_global(Xd, valid, inst_id, seed=seed)
        rows.append({
            "dim_requested": int(d),
            "dim_effective": int(Xd.shape[1]),
            "A_ratio": A["mean_ratio"],
            "A_frac_tighter": A["frac_valid_tighter"],
            "B_within_auc": B["mean_auc"],
            "C_centered_auc": C["centered_global_auc"],
            "A": A, "B": B, "C": C,
        })
    return rows


def _sweep_read(rows) -> str:
    """One-line READ over the sweep: where does the cluster signal PEAK? We rank by the
    combined within-instance signal (B within-AUC is the necessary precursor; the A
    spread-ratio is the corroborator: lower=tighter). The peak dim is the suggested waist
    dimension IF the signal is meaningfully above chance there; otherwise 'no dim
    separates'."""
    def _finite(x):
        return isinstance(x, float) and np.isfinite(x)

    cand = [r for r in rows if _finite(r["B_within_auc"])]
    if not cand:
        return ("READ: no dim produced a usable within-AUC (too few mixed instances) -> "
                "INCONCLUSIVE.")
    best = max(cand, key=lambda r: r["B_within_auc"])
    bauc = best["B_within_auc"]
    # baseline = the largest requested dim (the full/identity row, signal at no compression).
    base = max(rows, key=lambda r: r["dim_requested"])
    base_auc = base["B_within_auc"]
    bd = best["dim_effective"]
    # "separates" bar: within-AUC clears 0.55 (above the de-biased ~0.5 null floor).
    if bauc <= 0.55:
        return (f"READ: NO dim linearly separates valid from invalid (peak within-AUC="
                f"{bauc:.3f} at d={bd}, ~chance) -> PCA compression does NOT reveal a "
                f"linear common mode. INCONCLUSIVE, not a refutation (see caveat).")
    sharpened = (_finite(base_auc) and bd < base["dim_effective"]
                 and bauc > base_auc + 0.01)
    msg = (f"READ: cluster signal PEAKS at d={bd} (within-AUC={bauc:.3f}) "
           f"=> suggested waist dim ~{bd}.")
    if sharpened:
        msg += (f" It SHARPENED under compression (d={base['dim_effective']} AUC="
                f"{base_auc:.3f} -> d={bd} AUC={bauc:.3f}): GREEN LIGHT for a waist near "
                f"this dim.")
    elif _finite(base_auc):
        msg += (f" It did NOT sharpen vs the full rep (d={base['dim_effective']} AUC="
                f"{base_auc:.3f}): compression neither helps nor hurts the LINEAR read.")
    return msg


def print_pca_sweep(rows) -> None:
    """Print the dim-sweep table + the one-line READ + the lower-bound caveat."""
    print(f"\n  [PCA SWEEP] cluster signal vs PCA dim "
          f"(linear/unsupervised LOWER BOUND on a learned waist)", flush=True)
    print(f"      {'dim':>6} | {'A_ratio':>9} | {'A_frac_tight':>12} | "
          f"{'B_within_AUC':>12} | {'C_centered_AUC':>14}", flush=True)
    print(f"      {'-'*6}-+-{'-'*9}-+-{'-'*12}-+-{'-'*12}-+-{'-'*14}", flush=True)
    for r in rows:
        deff = r["dim_effective"]
        dlabel = f"{deff}" if deff == r["dim_requested"] else f"{deff}*"
        print(f"      {dlabel:>6} | {r['A_ratio']:>9.3f} | {r['A_frac_tighter']:>12.3f} | "
              f"{r['B_within_auc']:>12.3f} | {r['C_centered_auc']:>14.3f}", flush=True)
    print(f"      (* = requested dim exceeded the rep rank; shown at the effective dim. "
          f"A_ratio<1 & A_frac_tight>0.5 & AUC>0.5 = tighter/separable.)", flush=True)
    print(f"      {_sweep_read(rows)}", flush=True)
    print(f"      CAVEAT: PCA is LINEAR + UNSUPERVISED -> a LOWER BOUND on a learned "
          f"nonlinear waist. Sharpening = green light; flat = INCONCLUSIVE (a learned "
          f"waist may still work), NOT a refutation.", flush=True)


# ===========================================================================
# Driver
# ===========================================================================

def run_probe(npz_path: str, min_per_class=3, seed=0, pca_dims=None) -> dict:
    z = np.load(npz_path, allow_pickle=True)
    reps = np.asarray(z["reps"], dtype=np.float64)
    valid = np.asarray(z["valid"], dtype=bool)
    inst_id = np.asarray(z["inst_id"])
    band = np.asarray(z["band"], dtype=np.float64) if "band" in z else None
    meta = z["meta"].item() if "meta" in z else {}

    n, H = reps.shape
    print(f"\n{'='*74}", flush=True)
    print(f"  DART CLUSTER PROBE — {npz_path}", flush=True)
    print(f"{'='*74}", flush=True)
    print(f"  n_darts={n}  H={H}  VALID={int(valid.sum())}  INVALID={int((~valid).sum())}  "
          f"instances={int(np.unique(inst_id).shape[0])}", flush=True)
    if meta:
        print(f"  meta: ckpt={meta.get('ckpt','?')} mech={meta.get('mech','?')} "
              f"K={meta.get('K','?')} m_max={meta.get('m_max','?')} "
              f"bands={meta.get('bands','?')}", flush=True)
        print(f"  rep: {meta.get('rep','?')}", flush=True)
    if band is not None:
        for c in np.unique(band):
            bsel = band == c
            print(f"    band c={c:.2f}: {int(bsel.sum())} darts "
                  f"({int((valid & bsel).sum())} valid)", flush=True)

    A = metric_A_variance(reps, valid, inst_id, min_per_class=min_per_class, seed=seed)
    B = metric_B_within_auc(reps, valid, inst_id, min_per_class=min_per_class, seed=seed)
    C = metric_C_global(reps, valid, inst_id, seed=seed)
    V = verdict(A, B, C)

    print(f"\n  [A] ANNA-KARENINA VARIANCE TEST (valid spread vs invalid spread; "
          f">={min_per_class}/{min_per_class} per class)", flush=True)
    print(f"      mixed instances used      = {A['n_instances']}", flush=True)
    print(f"      frac valid TIGHTER        = {A['frac_valid_tighter']:.3f}  "
          f"(hypothesis: >> 0.5)", flush=True)
    print(f"      mean valid/invalid ratio  = {A['mean_ratio']:.3f}  "
          f"(median {A['median_ratio']:.3f}; hypothesis: < 1)", flush=True)
    print(f"      mean spreads: valid={A['mean_valid_spread']:.4f}  "
          f"invalid={A['mean_invalid_spread']:.4f}  (cos-dist to centroid)", flush=True)

    print(f"\n  [B] WITHIN-INSTANCE SEPARABILITY (cos-sim to per-instance VALID centroid; "
          f"solution-quality, NOT identity)", flush=True)
    print(f"      mixed instances used      = {B['n_instances']}", flush=True)
    print(f"      MEAN within-AUC           = {B['mean_auc']:.3f}  "
          f"(median {B['median_auc']:.3f}, std {B['std_auc']:.3f})", flush=True)
    print(f"      AUC IQR [p25,p75]         = [{B['p25_auc']:.3f}, {B['p75_auc']:.3f}]",
          flush=True)
    print(f"      frac instances AUC>0.65   = {B['frac_auc_gt_065']:.3f}  "
          f"(0.5=no signal; >>0.5=valid darts separable)", flush=True)

    print(f"\n  [C] GLOBAL / TRANSFERABLE GOOD-PATTERN (cross-validated logistic probe, "
          f"held-out AUC)", flush=True)
    print(f"      CENTERED  global AUC      = {C['centered_global_auc']:.3f}  "
          f"(instance-identity REMOVED — the LIBRARY-viable signal; >0.6 = plausible)",
          flush=True)
    print(f"      RAW       global AUC      = {C['raw_global_auc']:.3f}  "
          f"[CONFOUNDED by graph-identity — do NOT over-read]", flush=True)
    print(f"      probe n (centered/raw)    = {C['centered_n']}/{C['raw_n']}  "
          f"valid={C['n_valid']} invalid={C['n_invalid']} instances={C['n_instances']}",
          flush=True)

    print(f"\n  [D] VERDICT", flush=True)
    for ln in V["lines"]:
        print(f"      - {ln}", flush=True)

    sweep = None
    if pca_dims:
        sweep = pca_dim_sweep(reps, valid, inst_id, pca_dims,
                              min_per_class=min_per_class, seed=seed)
        print_pca_sweep(sweep)

    return {"A": A, "B": B, "C": C, "verdict": V, "meta": meta,
            "n_darts": n, "H": H, "pca_sweep": sweep}


# ===========================================================================
# CPU SELFTEST — synthetic clustered (signal) + null (no false positive)
# ===========================================================================

def _make_synthetic(n_inst, m, H, clustered, seed=0, p_valid=0.5):
    """Build a synthetic per-dart npz-like (reps, valid, inst_id) bank.

    clustered=True : per instance, VALID darts share a tight per-instance 'good' direction
                     (small noise) AND a SHARED global good axis (the transferable signal);
                     INVALID darts scatter widely. -> probe must detect tighter + high AUC.
    clustered=False: NULL — valid/invalid are intermixed (same distribution); validity is a
                     coin flip independent of the rep. -> probe must report ~0.5 / ratio~1.

    p_valid : P(dart is valid). Set < 0.5 to make VALID the MINORITY class (the real-data
              regime), which EXERCISES FIX 2's size-fair spread estimator under the null."""
    rng = np.random.RandomState(seed)
    reps, valid, inst_id = [], [], []
    # a SHARED global 'good' axis (the transferable common mode) for the clustered case.
    global_good = rng.randn(H)
    global_good /= np.linalg.norm(global_good)
    for g in range(n_inst):
        # per-instance graph-IDENTITY mean (dominates the raw rep — the confound).
        identity = rng.randn(H) * 3.0
        good_dir = rng.randn(H)
        good_dir /= np.linalg.norm(good_dir)
        for j in range(m):
            is_valid = bool(rng.rand() < p_valid)
            if clustered:
                if is_valid:
                    # tight: identity + shared global good + per-instance good + small noise.
                    vec = (identity + 1.5 * global_good + 1.0 * good_dir
                           + 0.10 * rng.randn(H))
                else:
                    # scattered: identity + large isotropic noise (no good direction).
                    vec = identity + 1.5 * rng.randn(H)
            else:
                # NULL: identity + isotropic noise, validity independent of the rep.
                vec = identity + 1.0 * rng.randn(H)
            reps.append(vec.astype(np.float32))
            valid.append(is_valid)
            inst_id.append(g)
    return (np.asarray(reps, dtype=np.float64), np.asarray(valid, dtype=bool),
            np.asarray(inst_id, dtype=np.int64))


def selftest() -> bool:
    print("=== dart_cluster_probe SELFTEST (CPU) ===", flush=True)
    ok = True
    H = 32

    # --- AUC machinery sanity: perfectly separable -> 1.0; identical -> 0.5. ---
    s = np.array([0.1, 0.2, 0.3, 0.9, 1.0, 1.1])
    y = np.array([False, False, False, True, True, True])
    auc_sep = auc_mann_whitney(s, y)
    auc_flat = auc_mann_whitney(np.ones(6), y)
    print(f"  [auc] separable={auc_sep:.3f} (expect 1.0)  flat={auc_flat:.3f} (expect 0.5)",
          flush=True)
    ok &= abs(auc_sep - 1.0) < 1e-9 and abs(auc_flat - 0.5) < 1e-9

    # --- CLUSTERED: tighter + high within-AUC + (likely) centered-global signal. ---
    reps, valid, inst_id = _make_synthetic(40, 24, H, clustered=True, seed=1)
    A = metric_A_variance(reps, valid, inst_id, seed=1)
    B = metric_B_within_auc(reps, valid, inst_id, seed=1)
    C = metric_C_global(reps, valid, inst_id, seed=1)
    print(f"\n  [clustered] A.frac_tighter={A['frac_valid_tighter']:.3f} "
          f"A.mean_ratio={A['mean_ratio']:.3f} | B.mean_auc={B['mean_auc']:.3f} | "
          f"C.centered={C['centered_global_auc']:.3f} C.raw={C['raw_global_auc']:.3f}",
          flush=True)
    cond_clust = (A["frac_valid_tighter"] > 0.8 and A["mean_ratio"] < 0.8
                  and B["mean_auc"] > 0.8 and C["centered_global_auc"] > 0.7)
    print(f"  [clustered] EXPECT tighter+high-AUC+centered-signal -> "
          f"{'PASS' if cond_clust else 'FAIL'}", flush=True)
    ok &= cond_clust

    # --- NULL: the anti-false-positive guard. valid/invalid intermixed, SAME distribution.
    # After ALL THREE fixes the null must read ~0.5 on all three biased-prone metrics:
    #   within-AUC in [0.45,0.55] (FIX 3), spread ratio in [0.9,1.1] (FIX 2), AND
    #   centered-global AUC in [0.45,0.55] (FIX 1 — before the by-instance split it inflates).
    # Use enough instances/darts that the null is statistically stable, and make valid the
    # MINORITY (p_valid=0.3) so FIX 2's size-fairness is actually exercised.
    reps0, valid0, inst0 = _make_synthetic(120, 40, H, clustered=False, seed=2, p_valid=0.3)
    A0 = metric_A_variance(reps0, valid0, inst0, seed=2)
    B0 = metric_B_within_auc(reps0, valid0, inst0, seed=2)
    C0 = metric_C_global(reps0, valid0, inst0, seed=2)
    print(f"\n  [null] A.frac_tighter={A0['frac_valid_tighter']:.3f} "
          f"A.mean_ratio={A0['mean_ratio']:.3f} | B.mean_auc={B0['mean_auc']:.3f} | "
          f"C.centered={C0['centered_global_auc']:.3f} C.raw={C0['raw_global_auc']:.3f}",
          flush=True)
    null_within_ok = abs(B0["mean_auc"] - 0.5) <= 0.05
    null_ratio_ok = 0.9 <= A0["mean_ratio"] <= 1.1
    null_centered_ok = abs(C0["centered_global_auc"] - 0.5) <= 0.05
    cond_null = null_within_ok and null_ratio_ok and null_centered_ok
    print(f"  [null] within-AUC in[.45,.55]={null_within_ok} (FIX3)  "
          f"ratio in[.9,1.1]={null_ratio_ok} (FIX2)  "
          f"centered-AUC in[.45,.55]={null_centered_ok} (FIX1)", flush=True)
    print(f"  [null] EXPECT all three ~0.5 (no false positive) -> "
          f"{'PASS' if cond_null else 'FAIL'}", flush=True)
    ok &= cond_null

    # --- VERDICT routing: clustered -> holds; null -> not_separating. ---
    Vc = verdict(A, B, C)
    Vn = verdict(A0, B0, C0)
    print(f"\n  [verdict] clustered.within_holds={Vc['within_holds']} (expect True); "
          f"null.not_separating={Vn['not_separating']} (expect True)", flush=True)
    ok &= Vc["within_holds"] is True and Vn["not_separating"] is True

    # --- PCA DIMENSION SWEEP: signal RETAINED/sharpened at reduced dims on a LOW-RANK
    # clustered bank; NULL reads ~0.5 at EVERY dim (no false positive from PCA). ---
    # Use a higher H (256) so the sweep dims (256..16) actually compress; the clustered
    # signal is intrinsically low-rank (identity + shared global + per-instance good live
    # in << 256 dims), so PCA to d=16 must KEEP a high within-AUC. The full-rank isotropic
    # noise lives in the dropped tail components -> compression cannot hurt (often helps).
    Hsw = 256
    sweep_dims = [256, 128, 64, 32, 16]
    repsS, validS, instS = _make_synthetic(40, 24, Hsw, clustered=True, seed=3)
    rows_clust = pca_dim_sweep(repsS, validS, instS, sweep_dims, seed=3)
    print(f"\n  [pca-sweep clustered] (low-rank cluster; within-AUC must stay high under "
          f"compression)", flush=True)
    print_pca_sweep(rows_clust)
    # signal must be RETAINED at the most-compressed dim (d=16) AND not collapse anywhere.
    low_dim_row = next(r for r in rows_clust if r["dim_effective"] == 16)
    sweep_clust_ok = (low_dim_row["B_within_auc"] > 0.8
                      and all(r["B_within_auc"] > 0.8 for r in rows_clust))
    print(f"  [pca-sweep clustered] EXPECT within-AUC>0.8 at ALL dims incl d=16 "
          f"(d16 AUC={low_dim_row['B_within_auc']:.3f}) -> "
          f"{'PASS' if sweep_clust_ok else 'FAIL'}", flush=True)
    ok &= sweep_clust_ok

    # NULL sweep: PCA must NOT manufacture a cluster signal at ANY dim.
    repsSn, validSn, instSn = _make_synthetic(120, 40, Hsw, clustered=False, seed=4,
                                              p_valid=0.3)
    rows_null = pca_dim_sweep(repsSn, validSn, instSn, sweep_dims, seed=4)
    print(f"\n  [pca-sweep null] (PCA must NOT manufacture a signal at any dim)", flush=True)
    print_pca_sweep(rows_null)
    sweep_null_ok = all(abs(r["B_within_auc"] - 0.5) <= 0.05
                        and abs(r["C_centered_auc"] - 0.5) <= 0.06
                        for r in rows_null)
    print(f"  [pca-sweep null] EXPECT every dim within-AUC~0.5 & centered-AUC~0.5 -> "
          f"{'PASS' if sweep_null_ok else 'FAIL'}", flush=True)
    ok &= sweep_null_ok

    # --- npz round-trip: dump clustered -> run_probe loads + reports. ---
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "syn.npz")
        np.savez(p, reps=reps.astype(np.float32), valid=valid, inst_id=inst_id,
                 band=np.full(reps.shape[0], 2.0),
                 meta=np.array({"H": H, "mech": "symmetry", "rep": "synthetic"},
                               dtype=object))
        res = run_probe(p)
        ok &= res["verdict"]["within_holds"] is True
        # also exercise the --pca-dims driver path end-to-end (table + READ printed).
        res2 = run_probe(p, pca_dims=[32, 16, 8])
        ok &= res2["pca_sweep"] is not None and len(res2["pca_sweep"]) == 3

    print(f"\n  SELFTEST {'PASSED' if ok else 'FAILED'}", flush=True)
    return ok


# ===========================================================================
# Main
# ===========================================================================

def main(argv=None) -> int:
    P = argparse.ArgumentParser(description="Anna-Karenina per-dart cluster probe")
    P.add_argument("--npz", default=os.environ.get("DART_NPZ", None),
                   help="path to the per-dart silhouette .npz (from --capture-darts)")
    P.add_argument("--selftest", action="store_true",
                   help="run the CPU selftest (clustered + null synthetic) and exit")
    P.add_argument("--min-per-class", type=int, default=3,
                   help="min VALID and INVALID darts per instance for (A)/(B) (default 3)")
    P.add_argument("--seed", type=int, default=0, help="CV / probe seed")
    P.add_argument("--pca-dims", default=None,
                   help="OPTIONAL comma list of PCA dims to sweep, e.g. '1024,512,256,128,"
                        "64'. When set, re-runs all three metrics on PCA-reduced reps at "
                        "each dim to find where the cluster signal peaks (=> suggested "
                        "waist dim). Default off => base behavior is byte-identical.")
    args = P.parse_args(argv)

    pca_dims = None
    if args.pca_dims:
        try:
            pca_dims = [int(tok) for tok in args.pca_dims.split(",") if tok.strip()]
        except ValueError:
            print(f"error: --pca-dims must be a comma list of ints, got {args.pca_dims!r}",
                  flush=True)
            return 2
        pca_dims = sorted({d for d in pca_dims if d >= 1}, reverse=True)
        if not pca_dims:
            print("error: --pca-dims produced no valid (>=1) dims", flush=True)
            return 2

    if args.selftest or os.environ.get("SELFTEST_ONLY", "0") == "1":
        return 0 if selftest() else 1
    if not args.npz:
        print("error: pass --npz PATH (or --selftest). "
              "PATH is the npz from amortized_frontier_measure.py --capture-darts.",
              flush=True)
        return 2
    if not os.path.exists(args.npz):
        print(f"error: npz not found: {args.npz}", flush=True)
        return 2
    run_probe(args.npz, min_per_class=args.min_per_class, seed=args.seed, pca_dims=pca_dims)
    return 0


if __name__ == "__main__":
    parse_ok = _ast_parse_ok()
    print(f"[ast.parse] astparse_ok={parse_ok}", flush=True)
    if not parse_ok:
        sys.exit(1)
    sys.exit(main())
