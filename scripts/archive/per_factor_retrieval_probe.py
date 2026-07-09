"""per_factor_retrieval_probe.py — does PART-LEVEL (per-edge / per-cell) cross-instance
MATCHING beat a RANDOM control, where WHOLE-graph matching LOST (matched-minus-random
= -0.048, GLOBAL_PRIOR_SUFFICES)? (pure numpy, CPU; no GPU/training; no engine edits).

THE COMPOSITION HYPOTHESIS (Bryce, 2026-06-22)
==============================================
The whole-graph centroid retrieval FAILED (scripts/retrieval_utility_probe.py on
.cache/dart_silhouettes_fg_coloring_k16.npz: matched-minus-random = -0.048 ->
GLOBAL_PRIOR_SUFFICES) because each solution was POOLED into ONE whole-graph silhouette,
DESTROYING the compositional structure. On RANDOM coloring graphs the WHOLE solution is
unique per graph (nothing to retrieve), but the PARTS recur: a triangle needs 3 colors, a
path 2-colors, a k-clique k colors, a degree-1 node is free. The factor graph is ALREADY
factored -> the "primes" are the local factors / motifs. So PART-LEVEL matching may carry
the transferable signal the WHOLE-level pool lacked. Rigorous form = sparse-dictionary /
compositional / part-based matching.

THE TEST (the decisive question)
================================
Does PART-LEVEL cross-instance matching beat a RANDOM control, where WHOLE-level matching
lost (-0.048)? If part-level matched-minus-random goes clearly POSITIVE (> a margin) AND
beats the whole-level -0.048 AND cross-instance recurrence is high -> composition is the
missing ingredient -> the prime/dictionary path is real, even on random graphs. If
part-level also ties random -> composition does NOT rescue retrieval either.

THE DATA CONTRACT (.cache/dart_cells_fg_coloring_k16.npz, from --capture-cells)
==============================================================================
  dart_reps    (D, S, H) float16 — PER-CELL final-breath readout-LN rep (pad cells zeroed)
  dart_colors  (D, S)    int16   — per-cell argmax color (1-based; 0 = pad)
  dart_inst_id (D,)      int64   — owning instance id (band-namespaced grouping key)
  dart_idx     (D,)      int32   — dart index (0 = argmax/identity)
  dart_valid   (D,)      bool    — whole-graph proper flag for the dart
  inst_id      (I,)      int64   — instance ids (deduped side-table key)
  inst_membership (I,L,S) float16 — edge membership (2 ones per real edge row)
  inst_cell_valid (I,S)  bool    — real-cell mask
  inst_degree  (I,S)     int16   — per-cell degree (real edge rows incident)
  inst_band    (I,)      float64 — band c
  inst_n       (I,)      int32   — real vertex count
  meta         (object)          — dict {ckpt, domain, mech, K, m_max, edge_ltype, ...}

WHAT THIS PROBE DOES
====================
PART CONSTRUCTION at TWO granularities (both reported):
  (a) PER-EDGE = the factor = a 2-member not-equal constraint. rep = pool (mean) the 2
      endpoint cells' reps; LABEL = SATISFIED (endpoints differ) or VIOLATED (same color),
      per (edge, dart). Attributes: same-X flags built offline (triangle-edge? both-endpoint
      degrees), band.
  (b) PER-CELL = each cell's rep directly (the K-breath attention already summarizes the
      cell's local motif). Attributes: degree, triangle-membership, band. The cell "role"
      label = whether the cell sits on any VIOLATED edge in that dart (a stuck cell).

RECURRENCE TEST (do the primes RECUR cross-instance, unlike whole-silhouettes?):
  k-means on the part-level reps; CROSS-INSTANCE cluster coverage — a "prime" cluster should
  contain parts from MANY instances. Contrast with the whole-graph case (each whole
  silhouette is instance-unique -> a whole-level cluster covers ~1 instance). High
  cross-instance coverage = the primes recur.

RETRIEVAL-UTILITY TEST (the decisive one — MIRRORS retrieval_utility_probe EXACTLY so it
  is apples-to-apples with the whole-level -0.048):
  by-INSTANCE split (library = parts from TRAIN-fold instances ONLY, never the query
  instance's own parts). For each held-out part, retrieve the nearest library "prime"
  (part-centroid); the DECISIVE control = MATCHED vs RANDOM library prime. HELP metric:
  does blending the query part toward the matched prime predict/improve the part LABEL
  (edge satisfied / cell role) better than the alpha=0 baseline — AUC(own-positive vs
  own-negative) over the query's OWN parts, alpha-sweep (mirrors auc_curve_for_centroid).
  KEY OUTPUT: part-level matched-minus-random, head-to-head with whole-level -0.048.

TRIVIALITY GUARD (critical)
===========================
An edge/cell rep TRIVIALLY encodes its own color (the readout determines the color), so
"predict satisfaction from the rep" can be true WITHOUT retrieval. The matched-vs-RANDOM
control ISOLATES whether the LIBRARY/MATCHING adds signal beyond reading the local state:
the RANDOM control reads the SAME local state (same query rep, same alpha-sweep) but blends
a RANDOM library prime; only a MATCHED-minus-RANDOM gap > 0 means the MATCHING (the prime
choice) — not the local rep — carries transferable signal. Additionally the by-instance
split means the query part is NEVER in its own library (no self-match leak), and at alpha=0
matched==random==global==baseline (only the local rep), so any separation at alpha=0 is
attributed to NEITHER control -> matched-minus-random is a clean retrieval-value isolate.

PER-MATCH LOG (csv + json)
==========================
One row per held-out part: query attrs (band, degree, motif=triangle/free/edge-type),
matched library part attrs, same-motif / same-degree / same-band flags, match distance,
matched vs random improvement, helped(bool) -> infer WHICH primes transfer (triangle-edges
match triangle-edges and help; random edges do not).

VERDICT
=======
PRIMES_CONFIRMED iff part-level matched-minus-random is clearly positive (> MARGIN) AND
  beats the whole-level -0.048 AND cross-instance recurrence is high.
PARTIAL / NULL otherwise (composition does not rescue retrieval). Honest caveats printed
  (frozen-rep proxy; baseline-model reps; final-breath; this tests retrieval SIGNAL, the
  in-deducer use is a further step).

SELFTEST (CPU, no npz)
======================
RECURRING-PRIME synthetic: a per-part PRIME drives cross-instance matching (matched>>random
  at PART level) WHILE whole-graph aggregation washes it out (matched~=random at WHOLE
  level) -> proves the probe can DETECT composition-rescues-retrieval. + a NULL (no recurring
  primes) -> matched~=random at BOTH levels. Both ASSERTED.

USAGE
=====
  CPU selftest (GPU-free):
    .venv/bin/python3 scripts/per_factor_retrieval_probe.py --selftest
  Probe the captured npz (the gate):
    .venv/bin/python3 scripts/per_factor_retrieval_probe.py \
        --npz .cache/dart_cells_fg_coloring_k16.npz \
        --out-prefix .cache/per_factor_retrieval_fg_coloring_k16
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import sys

_THIS_FILE = os.path.abspath(__file__)

import numpy as np

# Reuse the VALIDATED machinery from retrieval_utility_probe + dart_cluster_probe so the
# part-level result is APPLES-TO-APPLES with the whole-level -0.048 (same cosine, centering,
# Mann-Whitney AUC, by-instance folds, blend, the auc-curve help metric). Best-effort import;
# identical fallbacks below if the siblings are unavailable.
_SCRIPTS_DIR = os.path.dirname(_THIS_FILE)
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)
try:
    from retrieval_utility_probe import (  # type: ignore
        auc_mann_whitney as _auc_mann_whitney,
        _cos_to_vector as _ru_cos_to_vector,
        _cos_one as _ru_cos_one,
        _blend as _ru_blend,
        assign_instance_folds as _ru_assign_folds,
        ALPHAS_DEFAULT as _RU_ALPHAS_DEFAULT,
    )
    _REUSED_HELPERS = True
except Exception:  # pragma: no cover - fallback only if the sibling import breaks
    _REUSED_HELPERS = False
    _auc_mann_whitney = None
    _ru_cos_to_vector = None
    _ru_cos_one = None
    _ru_blend = None
    _ru_assign_folds = None
    _RU_ALPHAS_DEFAULT = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)

ALPHAS_DEFAULT = tuple(_RU_ALPHAS_DEFAULT)

# The whole-graph (pooled-silhouette) result we are trying to BEAT. From
# scripts/retrieval_utility_probe.py on .cache/dart_silhouettes_fg_coloring_k16.npz:
# matched-minus-random = -0.048, verdict GLOBAL_PRIOR_SUFFICES.
WHOLE_LEVEL_MMR = -0.048


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
# Numerical machinery (reuse retrieval_utility_probe; identical fallbacks)
# ===========================================================================

def _l2norm_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.where(n < 1e-12, 1.0, n)
    return x / n


def _cos_to_vector(rows: np.ndarray, vec: np.ndarray) -> np.ndarray:
    if _REUSED_HELPERS:
        return _ru_cos_to_vector(rows, vec)
    unit = _l2norm_rows(rows)
    v = np.asarray(vec, dtype=np.float64)
    vn = np.linalg.norm(v)
    if vn < 1e-12:
        return np.zeros(unit.shape[0])
    return unit @ (v / vn)


def _cos_one(a: np.ndarray, b: np.ndarray) -> float:
    if _REUSED_HELPERS:
        return _ru_cos_one(a, b)
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    an, bn = np.linalg.norm(a), np.linalg.norm(b)
    if an < 1e-12 or bn < 1e-12:
        return 0.0
    return float((a @ b) / (an * bn))


def auc_mann_whitney(scores: np.ndarray, labels: np.ndarray) -> float:
    if _REUSED_HELPERS:
        return _auc_mann_whitney(scores, labels)
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=bool)
    n_pos = int(labels.sum())
    n_neg = int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(scores.shape[0], dtype=np.float64)
    ss = scores[order]
    i, N = 0, scores.shape[0]
    while i < N:
        j = i
        while j + 1 < N and ss[j + 1] == ss[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    u_pos = ranks[labels].sum() - n_pos * (n_pos + 1) / 2.0
    return float(u_pos / (n_pos * n_neg))


def _blend(query_vec, centroid_vec, alpha):
    if _REUSED_HELPERS:
        return _ru_blend(query_vec, centroid_vec, alpha)
    return (1.0 - alpha) * np.asarray(query_vec, dtype=np.float64) \
        + alpha * np.asarray(centroid_vec, dtype=np.float64)


def assign_instance_folds(uids, folds, seed=0):
    if _REUSED_HELPERS:
        return _ru_assign_folds(uids, folds, seed=seed)
    uids = np.asarray(uids)
    n_inst = uids.shape[0]
    folds = max(2, min(folds, n_inst))
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n_inst)
    fold_of = np.zeros(n_inst, dtype=int)
    fold_of[perm] = np.arange(n_inst) % folds
    return {int(u): int(f) for u, f in zip(uids, fold_of)}, folds


def _safe_mean(x):
    x = [v for v in x if v is not None and np.isfinite(v)]
    return float(np.mean(x)) if x else float("nan")


def _safe_median(x):
    x = [v for v in x if v is not None and np.isfinite(v)]
    return float(np.median(x)) if x else float("nan")


# ===========================================================================
# PART CONSTRUCTION — build per-edge + per-cell parts from the per-cell capture
# ===========================================================================

def _triangle_membership(adj: dict, n: int) -> np.ndarray:
    """Per-vertex flag: is the vertex in any TRIANGLE? adj: {v: set(neighbors)}; returns
    (n,) bool. A triangle at (u,v) means u,v adjacent AND share a common neighbor."""
    tri = np.zeros((n,), dtype=bool)
    for u in range(n):
        nbrs = adj.get(u, set())
        for v in nbrs:
            if v <= u:
                continue
            # common neighbor of u and v -> triangle u-v-w.
            if adj.get(v, set()) & nbrs:
                tri[u] = True
                tri[v] = True
    return tri


def _instance_adjacency(mem_inst: np.ndarray, cv_inst: np.ndarray, edge_ltype: int,
                        lt_inst=None) -> tuple:
    """Build (adjacency dict, edge list) from one instance's membership (L, S).

    A real edge row has EXACTLY two 1s (at the endpoints). We take rows with exactly 2 ones
    among VALID cells (pad rows are all-zero / global sentinel). Returns:
      adj   : {v: set(neighbors)} over real vertices
      edges : list of (u, v) with u < v (deduped)
    """
    L, S = mem_inst.shape
    valid = cv_inst > 0.5
    adj: dict = {}
    edges = set()
    for e in range(L):
        if lt_inst is not None and int(lt_inst[e]) != edge_ltype:
            continue
        ones = np.nonzero(mem_inst[e] > 0.5)[0]
        if ones.shape[0] != 2:
            continue
        u, v = int(ones[0]), int(ones[1])
        if not (valid[u] and valid[v]):
            continue
        a, b = (u, v) if u < v else (v, u)
        edges.add((a, b))
        adj.setdefault(u, set()).add(v)
        adj.setdefault(v, set()).add(u)
    return adj, sorted(edges)


def build_parts(z, *, max_darts_per_inst=None, seed=0):
    """Build PER-EDGE and PER-CELL part tables from a loaded per-cell capture npz `z`.

    Returns a dict with two granularities, each a flat record set:
      edges: dict of arrays (one row per (instance, edge, dart)):
        reps     (Ne, H) float64  — pooled mean of the 2 endpoint cell reps for that dart
        label    (Ne,)   bool     — SATISFIED (endpoints differ); ~label = VIOLATED
        inst_id  (Ne,)   int64    — owning instance (grouping key for by-instance folds)
        band     (Ne,)   float64
        dart_idx (Ne,)   int32
        deg_min/deg_max (Ne,) int — endpoint degrees (motif attrs)
        is_tri   (Ne,)   bool     — edge sits inside a triangle (both endpoints triangle)
      cells: dict of arrays (one row per (instance, cell, dart)):
        reps     (Nc, H) float64  — the cell rep itself
        label    (Nc,)   bool     — cell sits on >=1 VIOLATED edge in that dart (stuck role)
        inst_id  (Nc,)   int64
        band     (Nc,)   float64
        dart_idx (Nc,)   int32
        degree   (Nc,)   int
        is_tri   (Nc,)   bool
    A per-instance dart subsample (max_darts_per_inst) keeps the part count tractable.
    """
    dart_reps = np.asarray(z["dart_reps"], dtype=np.float32)        # (D, S, H)
    dart_colors = np.asarray(z["dart_colors"], dtype=np.int64)      # (D, S)
    dart_inst_id = np.asarray(z["dart_inst_id"], dtype=np.int64)    # (D,)
    dart_idx = np.asarray(z["dart_idx"], dtype=np.int64)            # (D,)
    inst_id = np.asarray(z["inst_id"], dtype=np.int64)              # (I,)
    inst_membership = np.asarray(z["inst_membership"], dtype=np.float32)  # (I, L, S)
    inst_cell_valid = np.asarray(z["inst_cell_valid"], dtype=bool)  # (I, S)
    inst_degree = np.asarray(z["inst_degree"], dtype=np.int64)      # (I, S)
    inst_band = np.asarray(z["inst_band"], dtype=np.float64)        # (I,)
    inst_n = np.asarray(z["inst_n"], dtype=np.int64) if "inst_n" in z \
        else inst_cell_valid.sum(axis=1).astype(np.int64)
    meta = z["meta"].item() if "meta" in z else {}
    edge_ltype = int(meta.get("edge_ltype", 0))

    # per-instance precompute: adjacency + triangle flags + edge list (membership has no
    # latent_type column stored, so we infer real edges as exactly-2-ones rows).
    inst_pos = {int(iid): i for i, iid in enumerate(inst_id)}
    inst_adj = {}
    inst_edges = {}
    inst_tri = {}
    for i, iid in enumerate(inst_id):
        cv = inst_cell_valid[i]
        adj, edges = _instance_adjacency(inst_membership[i], cv, edge_ltype, lt_inst=None)
        inst_adj[int(iid)] = adj
        inst_edges[int(iid)] = edges
        inst_tri[int(iid)] = _triangle_membership(adj, inst_membership[i].shape[1])

    # optional per-instance dart subsample (deterministic).
    rng = np.random.RandomState(seed)
    keep_dart = np.ones(dart_reps.shape[0], dtype=bool)
    if max_darts_per_inst is not None:
        for iid in np.unique(dart_inst_id):
            sel = np.nonzero(dart_inst_id == iid)[0]
            if sel.shape[0] > max_darts_per_inst:
                drop = rng.permutation(sel)[max_darts_per_inst:]
                keep_dart[drop] = False

    edge_reps, edge_label, edge_inst, edge_band = [], [], [], []
    edge_dart, edge_degmin, edge_degmax, edge_tri = [], [], [], []
    cell_reps, cell_label, cell_inst, cell_band = [], [], [], []
    cell_dart, cell_deg, cell_tri = [], [], []

    for d in range(dart_reps.shape[0]):
        if not keep_dart[d]:
            continue
        iid = int(dart_inst_id[d])
        if iid not in inst_pos:
            continue
        ipos = inst_pos[iid]
        cv = inst_cell_valid[ipos]
        deg = inst_degree[ipos]
        band = float(inst_band[ipos])
        di = int(dart_idx[d])
        tri = inst_tri[iid]
        colors = dart_colors[d]
        reps = dart_reps[d]                                          # (S, H)
        edges = inst_edges[iid]

        # per-cell: which cells sit on a violated edge in this dart (the stuck role label).
        on_violated = np.zeros(cv.shape[0], dtype=bool)
        for (u, v) in edges:
            if colors[u] == colors[v]:          # same color -> VIOLATED edge
                on_violated[u] = True
                on_violated[v] = True

        # PER-EDGE parts.
        for (u, v) in edges:
            ev = (reps[u].astype(np.float64) + reps[v].astype(np.float64)) / 2.0
            sat = bool(colors[u] != colors[v])  # SATISFIED iff endpoints differ
            edge_reps.append(ev)
            edge_label.append(sat)
            edge_inst.append(iid)
            edge_band.append(band)
            edge_dart.append(di)
            du, dv = int(deg[u]), int(deg[v])
            edge_degmin.append(min(du, dv))
            edge_degmax.append(max(du, dv))
            edge_tri.append(bool(tri[u] and tri[v]))

        # PER-CELL parts.
        valid_idx = np.nonzero(cv)[0]
        for cidx in valid_idx:
            cell_reps.append(reps[cidx].astype(np.float64))
            cell_label.append(bool(on_violated[cidx]))   # True = stuck (on a violated edge)
            cell_inst.append(iid)
            cell_band.append(band)
            cell_dart.append(di)
            cell_deg.append(int(deg[cidx]))
            cell_tri.append(bool(tri[cidx]))

    def _pack(reps, label, inst, band, dart, extra):
        out = {
            "reps": (np.asarray(reps, dtype=np.float64) if reps
                     else np.zeros((0, dart_reps.shape[2]), dtype=np.float64)),
            "label": np.asarray(label, dtype=bool),
            "inst_id": np.asarray(inst, dtype=np.int64),
            "band": np.asarray(band, dtype=np.float64),
            "dart_idx": np.asarray(dart, dtype=np.int32),
        }
        out.update(extra)
        return out

    edges_tab = _pack(edge_reps, edge_label, edge_inst, edge_band, edge_dart, {
        "deg_min": np.asarray(edge_degmin, dtype=np.int64),
        "deg_max": np.asarray(edge_degmax, dtype=np.int64),
        "is_tri": np.asarray(edge_tri, dtype=bool),
    })
    cells_tab = _pack(cell_reps, cell_label, cell_inst, cell_band, cell_dart, {
        "degree": np.asarray(cell_deg, dtype=np.int64),
        "is_tri": np.asarray(cell_tri, dtype=bool),
    })
    return {"edges": edges_tab, "cells": cells_tab, "meta": meta,
            "n_instances": int(inst_id.shape[0])}


# ===========================================================================
# RECURRENCE TEST — do part-level reps CLUSTER cross-instance? (vs instance-unique whole)
# ===========================================================================

def _kmeans(X, k, seed=0, iters=50):
    """Tiny k-means (cosine via L2-normalized rows -> euclidean on the sphere). Returns
    (labels (n,), centers (k, H)). Pure numpy; deterministic seed; empty-cluster re-seed."""
    X = _l2norm_rows(np.asarray(X, dtype=np.float64))
    n = X.shape[0]
    k = max(1, min(k, n))
    rng = np.random.RandomState(seed)
    centers = X[rng.choice(n, size=k, replace=False)].copy()
    labels = np.zeros(n, dtype=np.int64)
    for _ in range(iters):
        d = -(X @ centers.T)                    # (n, k) negative cosine = distance proxy
        new_labels = np.argmin(d, axis=1)
        if np.array_equal(new_labels, labels) and _ > 0:
            labels = new_labels
            break
        labels = new_labels
        for c in range(k):
            m = labels == c
            if m.sum() == 0:
                centers[c] = X[rng.choice(n)]   # re-seed empty cluster
            else:
                v = X[m].mean(axis=0)
                nv = np.linalg.norm(v)
                centers[c] = v / nv if nv > 1e-12 else v
    return labels, centers


def recurrence_test(reps, inst_id, *, k_clusters=None, seed=0):
    """Cross-instance cluster coverage: does a PART cluster contain parts from MANY
    instances (a recurring "prime"), unlike whole-graph silhouettes (instance-unique)?

    Returns:
      k_clusters             : number of clusters used.
      n_parts / n_instances  : counts.
      mean_instances_per_cluster : mean distinct instances per cluster (the recurrence
                               headline; >> 1 = parts recur across instances).
      median_instances_per_cluster.
      mean_cross_instance_coverage : mean over clusters of (distinct instances / cluster size)
                               clipped — fraction-distinct (1.0 = every part a different
                               instance; low = many parts share an instance). Reported for
                               completeness; the headline is instances-per-cluster.
      frac_clusters_multi_instance : fraction of clusters spanning >= 2 instances.
      whole_graph_baseline   : 1.0 (a whole silhouette cluster covers ~1 instance —
                               instance-unique by construction; the contrast).
      cluster_sizes / cluster_n_instances : per-cluster arrays.
    """
    reps = np.asarray(reps, dtype=np.float64)
    inst_id = np.asarray(inst_id, dtype=np.int64)
    n = reps.shape[0]
    n_inst = int(np.unique(inst_id).shape[0])
    if n < 4 or n_inst < 2:
        return {"k_clusters": 0, "n_parts": n, "n_instances": n_inst,
                "mean_instances_per_cluster": float("nan"),
                "median_instances_per_cluster": float("nan"),
                "mean_cross_instance_coverage": float("nan"),
                "frac_clusters_multi_instance": float("nan"),
                "whole_graph_baseline": 1.0,
                "cluster_sizes": [], "cluster_n_instances": []}
    if k_clusters is None:
        # heuristic: ~ #instances (so a "prime per instance" null would give ~1 inst/cluster,
        # while genuinely recurring primes give >> 1).
        k_clusters = max(2, min(n_inst, n // 4))

    def _cluster_stats(rep_mat):
        labels, _ = _kmeans(rep_mat, k_clusters, seed=seed)
        sizes, n_insts = [], []
        for c in range(k_clusters):
            m = labels == c
            sz = int(m.sum())
            if sz == 0:
                continue
            sizes.append(sz)
            n_insts.append(int(np.unique(inst_id[m]).shape[0]))
        return sizes, n_insts

    sizes, n_insts = _cluster_stats(reps)
    n_insts_arr = np.asarray(n_insts, dtype=np.float64)
    # SHUFFLE-NULL (the honesty guard): instances-per-cluster when the inst_id labels are
    # destroyed (cluster the SAME reps but the high instances/cluster could be a pure size
    # artifact — clustering noise still spreads instances). We instead PERMUTE the inst_id
    # tags and recompute on the SAME clustering geometry: this is the instances-per-cluster a
    # cluster of that SIZE would get if membership were random. If observed ~= shuffle, the
    # 'recurrence' is a size artifact, NOT genuine cross-instance prime recurrence.
    rng_sh = np.random.RandomState(seed + 1)
    sh_vals = []
    for _ in range(5):
        perm = rng_sh.permutation(n)
        inst_perm = inst_id[perm]
        labels, _ = _kmeans(reps, k_clusters, seed=seed)
        per = []
        for c in range(k_clusters):
            m = labels == c
            if m.sum() == 0:
                continue
            per.append(int(np.unique(inst_perm[m]).shape[0]))
        sh_vals.append(float(np.mean(per)) if per else float("nan"))
    shuffle_null = float(np.nanmean(sh_vals)) if sh_vals else float("nan")
    cov = [ni / sz for ni, sz in zip(n_insts, sizes)]
    return {
        "k_clusters": int(k_clusters),
        "n_parts": n,
        "n_instances": n_inst,
        "mean_instances_per_cluster": float(np.mean(n_insts_arr)),
        "median_instances_per_cluster": float(np.median(n_insts_arr)),
        "mean_cross_instance_coverage": float(np.mean(cov)),
        "frac_clusters_multi_instance": float(np.mean(n_insts_arr >= 2)),
        "shuffle_null_instances_per_cluster": shuffle_null,
        "whole_graph_baseline": 1.0,
        "cluster_sizes": sizes,
        "cluster_n_instances": n_insts,
    }


# ===========================================================================
# RETRIEVAL-UTILITY TEST (by-instance, matched-vs-random; mirrors retrieval_utility_probe)
# ===========================================================================

def _instance_part_tables(tab, min_pos=2):
    """Per-instance part tables (mirror retrieval_utility_probe.build_instance_tables but on
    PARTS instead of darts, and with a generic POSITIVE label instead of 'valid').

    For each instance: pos_centroid = mean of its POSITIVE parts (the recurring 'good prime');
    query_mean = mean over ALL its parts (the deduction-state stand-in); own part indices +
    labels. lib_eligible iff n_pos >= min_pos; query_eligible iff >=1 pos AND >=1 neg (so
    own-pos-vs-neg AUC is defined)."""
    reps = tab["reps"]
    label = tab["label"]
    inst = tab["inst_id"]
    band = tab["band"]
    tables = {}
    for uid in np.unique(inst):
        sel = np.nonzero(inst == uid)[0]
        lab = label[sel]
        p_idx = sel[lab]
        n_idx = sel[~lab]
        npos, nneg = int(p_idx.shape[0]), int(n_idx.shape[0])
        pc = reps[p_idx].mean(axis=0) if npos > 0 else None
        qmean = reps[sel].mean(axis=0)
        b = float(band[sel[0]]) if sel.shape[0] else float("nan")
        tables[int(uid)] = {
            "uid": int(uid),
            "part_idx": sel,
            "pos_idx": p_idx,
            "neg_idx": n_idx,
            "pos_centroid": pc,
            "query_mean": qmean,
            "band": b,
            "n_pos": npos,
            "n_neg": nneg,
            "lib_eligible": npos >= min_pos,
            "query_eligible": (npos >= 1 and nneg >= 1),
        }
    return tables


def _auc_curve(tab, reps, centroid_vec, alphas):
    """AUC(own-positive vs own-negative) of the query's OWN parts scored by cosine-sim to the
    blended point, per alpha. blended = (1-alpha)*query_mean + alpha*centroid. alpha=0 = the
    no-blend baseline (ONLY the local rep — the triviality anchor). Mirrors
    retrieval_utility_probe.auc_curve_for_centroid exactly (same blend, same Mann-Whitney)."""
    own_idx = np.concatenate([tab["pos_idx"], tab["neg_idx"]])
    own_labels = np.concatenate([
        np.ones(tab["pos_idx"].shape[0], dtype=bool),
        np.zeros(tab["neg_idx"].shape[0], dtype=bool),
    ])
    own_reps = reps[own_idx]
    aucs = np.empty(len(alphas), dtype=np.float64)
    for ki, a in enumerate(alphas):
        bp = _blend(tab["query_mean"], centroid_vec, a)
        aucs[ki] = auc_mann_whitney(_cos_to_vector(own_reps, bp), own_labels)
    return aucs


def _retrieve(query_vec, lib_uids, lib_centroids, rng):
    """Nearest library prime by RAW cosine + a RANDOM library prime + the GLOBAL good-prime.
    (Mirrors retrieval_utility_probe._retrieve, raw geometry — the headline for the whole-
    level result. We keep it raw for apples-to-apples; centered is summarized via the global
    control already.)"""
    lib_uids = np.asarray(lib_uids)
    lib_centroids = np.asarray(lib_centroids, dtype=np.float64)
    L = lib_centroids.shape[0]
    sims = _cos_to_vector(lib_centroids, query_vec)
    best = int(np.argmax(sims))
    rnd = int(rng.randint(L))
    return {
        "matched": {"uid": int(lib_uids[best]), "centroid": lib_centroids[best],
                    "cosdist": float(1.0 - sims[best])},
        "random": {"uid": int(lib_uids[rnd]), "centroid": lib_centroids[rnd]},
        "global": {"centroid": lib_centroids.mean(axis=0)},
    }


def _motif_of_query(tab_attr):
    """A coarse motif label for the per-match log (used to infer which primes transfer)."""
    return tab_attr


def run_retrieval(tab, attrs, *, granularity, min_pos=2, folds=5, alphas=ALPHAS_DEFAULT,
                  seed=0, verbose=False):
    """By-instance matched-vs-random retrieval-utility on a PART table (edges or cells).

    tab     : the part table (reps/label/inst_id/band/...).
    attrs   : dict of per-part attribute arrays for the per-match log (degree/motif).
    granularity : 'edge' or 'cell' (for logging).
    Returns a result dict with aggregate matched-minus-random + per-match rows + verdict-
    inputs, mirroring retrieval_utility_probe's aggregate exactly for comparability."""
    reps = tab["reps"]
    alphas = tuple(float(a) for a in alphas)
    tables = _instance_part_tables(tab, min_pos=min_pos)
    uids = np.array(sorted(tables.keys()))
    fold_of, folds = assign_instance_folds(uids, folds, seed=seed)

    lib_eligible = [u for u in uids if tables[int(u)]["lib_eligible"]]
    query_eligible = [u for u in uids if tables[int(u)]["query_eligible"]]
    rng = np.random.RandomState(seed)

    curve_sum = {ctrl: np.zeros(len(alphas)) for ctrl in ("matched", "random", "global")}
    curve_cnt = 0
    rows = []
    agg = {ctrl: {"improvements": [], "opt_alphas": [], "auc_opt": [], "auc0": []}
           for ctrl in ("matched", "random", "global")}
    n_skipped = 0

    # per-instance attribute summaries (mean degree / triangle fraction) for the match log.
    # Attrs are optional (the whole-level proxy passes attrs={}); default to 0.0 then.
    def _inst_attr(uid):
        sel = tables[uid]["part_idx"]
        out = {"band": tables[uid]["band"]}
        if "is_tri" in attrs:
            out["tri_frac"] = float(np.mean(attrs["is_tri"][sel])) if sel.shape[0] else 0.0
        else:
            out["tri_frac"] = 0.0
        if granularity == "edge" and "deg_max" in attrs:
            out["deg_max_med"] = (float(np.median(attrs["deg_max"][sel]))
                                  if sel.shape[0] else 0.0)
        elif "degree" in attrs:
            out["deg_med"] = (float(np.median(attrs["degree"][sel]))
                              if sel.shape[0] else 0.0)
        else:
            out["deg_max_med"] = 0.0
            out["deg_med"] = 0.0
        return out

    for qu in query_eligible:
        qu = int(qu)
        tabq = tables[qu]
        qfold = fold_of[qu]
        lib_uids = [u for u in lib_eligible
                    if fold_of[int(u)] != qfold and int(u) != qu]
        if len(lib_uids) < 2:
            n_skipped += 1
            continue
        lib_centroids = np.stack([tables[int(u)]["pos_centroid"] for u in lib_uids])
        ret = _retrieve(tabq["query_mean"], lib_uids, lib_centroids, rng)

        matched_auc = _auc_curve(tabq, reps, ret["matched"]["centroid"], alphas)
        random_auc = _auc_curve(tabq, reps, ret["random"]["centroid"], alphas)
        global_auc = _auc_curve(tabq, reps, ret["global"]["centroid"], alphas)

        curve_sum["matched"] += matched_auc
        curve_sum["random"] += random_auc
        curve_sum["global"] += global_auc
        curve_cnt += 1

        def _opt(auc_vec):
            kb = int(np.argmax(auc_vec))
            return (float(alphas[kb]), float(auc_vec[kb]), float(auc_vec[0]),
                    float(auc_vec[kb] - auc_vec[0]))

        m_alpha, m_aopt, m_a0, m_imp = _opt(matched_auc)
        r_alpha, r_aopt, r_a0, r_imp = _opt(random_auc)
        g_alpha, g_aopt, g_a0, g_imp = _opt(global_auc)
        for ctrl, vals in (("matched", (m_alpha, m_aopt, m_a0, m_imp)),
                           ("random", (r_alpha, r_aopt, r_a0, r_imp)),
                           ("global", (g_alpha, g_aopt, g_a0, g_imp))):
            agg[ctrl]["opt_alphas"].append(vals[0])
            agg[ctrl]["auc_opt"].append(vals[1])
            agg[ctrl]["auc0"].append(vals[2])
            agg[ctrl]["improvements"].append(vals[3])

        mmr = m_imp - r_imp
        helped = bool(m_imp > 0.0 and mmr > 1e-3)
        q_attr = _inst_attr(qu)
        m_attr = _inst_attr(ret["matched"]["uid"])
        same_band = bool(tabq["band"] == tables[ret["matched"]["uid"]]["band"])
        same_tri = (abs(q_attr.get("tri_frac", 0.0) - m_attr.get("tri_frac", 0.0)) < 0.25)
        if granularity == "edge":
            same_deg = (abs(q_attr.get("deg_max_med", 0.0)
                            - m_attr.get("deg_max_med", 0.0)) <= 1.0)
            q_deg, m_deg = q_attr.get("deg_max_med", 0.0), m_attr.get("deg_max_med", 0.0)
        else:
            same_deg = (abs(q_attr.get("deg_med", 0.0)
                            - m_attr.get("deg_med", 0.0)) <= 1.0)
            q_deg, m_deg = q_attr.get("deg_med", 0.0), m_attr.get("deg_med", 0.0)
        rows.append({
            "granularity": granularity,
            "query_inst_id": qu,
            "query_band": tabq["band"],
            "query_n_pos": tabq["n_pos"],
            "query_n_neg": tabq["n_neg"],
            "query_fold": qfold,
            "query_tri_frac": q_attr.get("tri_frac", float("nan")),
            "query_deg": q_deg,
            "lib_size": len(lib_uids),
            "matched_lib_inst_id": ret["matched"]["uid"],
            "matched_band": tables[ret["matched"]["uid"]]["band"],
            "matched_tri_frac": m_attr.get("tri_frac", float("nan")),
            "matched_deg": m_deg,
            "same_band": same_band,
            "same_motif_tri": same_tri,
            "same_degree": same_deg,
            "match_cosine_distance": ret["matched"]["cosdist"],
            "optimal_alpha_matched": m_alpha,
            "auc_at_optimal_matched": m_aopt,
            "auc_at_alpha0": m_a0,
            "improvement_matched": m_imp,
            "optimal_alpha_random": r_alpha,
            "improvement_random": r_imp,
            "improvement_global": g_imp,
            "matched_minus_random": mmr,
            "helped": helped,
        })

    c = max(1, curve_cnt)
    curves = {ctrl: (curve_sum[ctrl] / c).tolist()
              for ctrl in ("matched", "random", "global")}
    mmr_list = [a - b for a, b in zip(agg["matched"]["improvements"],
                                      agg["random"]["improvements"])]
    aggregate = {
        "granularity": granularity,
        "n": len(agg["matched"]["improvements"]),
        "frac_helped_matched": _safe_mean(
            [1.0 if v > 0 else 0.0 for v in agg["matched"]["improvements"]]),
        "frac_helped_random": _safe_mean(
            [1.0 if v > 0 else 0.0 for v in agg["random"]["improvements"]]),
        "mean_improvement_matched": _safe_mean(agg["matched"]["improvements"]),
        "mean_improvement_random": _safe_mean(agg["random"]["improvements"]),
        "mean_improvement_global": _safe_mean(agg["global"]["improvements"]),
        "mean_matched_minus_random": _safe_mean(mmr_list),
        "median_matched_minus_random": _safe_median(mmr_list),
        "frac_matched_beats_random": _safe_mean([1.0 if v > 1e-3 else 0.0 for v in mmr_list]),
        "mean_auc0": _safe_mean(agg["matched"]["auc0"]),
        "mean_auc_opt_matched": _safe_mean(agg["matched"]["auc_opt"]),
        "mean_auc_opt_random": _safe_mean(agg["random"]["auc_opt"]),
        "mean_auc_opt_global": _safe_mean(agg["global"]["auc_opt"]),
        "mean_opt_alpha_matched": _safe_mean(agg["matched"]["opt_alphas"]),
        "frac_opt_alpha_gt0": _safe_mean(
            [1.0 if v > 0 else 0.0 for v in agg["matched"]["opt_alphas"]]),
    }
    # attribute drivers: help-rate by same-motif / same-band (which primes transfer?).
    def _help_rate(pred):
        sub = [r for r in rows if pred(r)]
        return (_safe_mean([1.0 if r["helped"] else 0.0 for r in sub]), len(sub))
    hr_tri, n_tri = _help_rate(lambda r: r["same_motif_tri"])
    hr_ntri, n_ntri = _help_rate(lambda r: not r["same_motif_tri"])
    hr_sb, n_sb = _help_rate(lambda r: r["same_band"])
    hr_xb, n_xb = _help_rate(lambda r: not r["same_band"])
    aggregate["attributes"] = {
        "help_rate_same_motif": hr_tri, "n_same_motif": n_tri,
        "help_rate_diff_motif": hr_ntri, "n_diff_motif": n_ntri,
        "help_rate_same_band": hr_sb, "n_same_band": n_sb,
        "help_rate_cross_band": hr_xb, "n_cross_band": n_xb,
    }
    result = {
        "granularity": granularity,
        "alphas": list(alphas),
        "min_pos": min_pos,
        "folds": folds,
        "seed": seed,
        "n_instances_total": int(uids.shape[0]),
        "n_lib_eligible": len(lib_eligible),
        "n_query_eligible": len(query_eligible),
        "n_queries_scored": len(rows),
        "n_skipped_no_lib": n_skipped,
        "curves": curves,
        "aggregate": aggregate,
        "rows": rows,
    }
    if verbose:
        _print_retrieval(result)
    return result


# ===========================================================================
# VERDICT
# ===========================================================================

def make_verdict(edge_res, cell_res, edge_rec, cell_rec, *, margin=0.02,
                 recurrence_bar=2.0):
    """PRIMES_CONFIRMED iff (the BEST part granularity's) matched-minus-random > margin AND
    beats the whole-level -0.048 AND cross-instance recurrence is high (mean instances per
    cluster > recurrence_bar). PARTIAL if part beats whole + random by a hair but misses a
    gate; NULL if part also ties random.
    """
    def _mmr(res):
        return res["aggregate"]["mean_matched_minus_random"]
    def _fbeats(res):
        return res["aggregate"]["frac_matched_beats_random"]
    e_mmr, c_mmr = _mmr(edge_res), _mmr(cell_res)
    # the BEST (most positive) part granularity drives the headline.
    if (np.isfinite(e_mmr) and (not np.isfinite(c_mmr) or e_mmr >= c_mmr)):
        best_g, best_mmr, best_res = "edge", e_mmr, edge_res
        best_rec = edge_rec
    else:
        best_g, best_mmr, best_res = "cell", c_mmr, cell_res
        best_rec = cell_rec
    rec_val = best_rec.get("mean_instances_per_cluster", float("nan"))
    rec_null = best_rec.get("shuffle_null_instances_per_cluster", float("nan"))

    def _f(x):
        return isinstance(x, float) and np.isfinite(x)

    beats_whole = _f(best_mmr) and best_mmr > WHOLE_LEVEL_MMR + margin
    clearly_pos = _f(best_mmr) and best_mmr > margin
    fbeats = best_res["aggregate"]["frac_matched_beats_random"]
    consistent = _f(fbeats) and fbeats > 0.55
    # Recurrence must beat its OWN shuffle-null (label-permuted cluster-size artifact),
    # not a raw absolute bar: genuine cross-instance recurrence = observed clearly above
    # the shuffle-null. Fall back to the absolute bar only if the null is unavailable.
    if _f(rec_val) and _f(rec_null):
        recurs = rec_val > rec_null * 1.1
    else:
        recurs = _f(rec_val) and rec_val > recurrence_bar

    lines = []
    if clearly_pos and beats_whole and recurs and consistent:
        label = "PRIMES_CONFIRMED"
        lines.append(
            f"PRIMES CONFIRMED (best granularity={best_g}): part-level matched-minus-random="
            f"{best_mmr:+.4f} > margin {margin} AND beats the whole-level {WHOLE_LEVEL_MMR:+.3f}; "
            f"frac matched>random={fbeats:.2f}>0.55; cross-instance recurrence="
            f"{rec_val:.1f} instances/cluster > {recurrence_bar} (the primes RECUR). "
            f"COMPOSITION is the missing ingredient -> the prime/dictionary path is real, "
            f"even on random graphs. The whole-graph pool destroyed it.")
    elif (_f(best_mmr) and best_mmr > margin / 2.0
          and best_mmr > WHOLE_LEVEL_MMR + margin and (clearly_pos or recurs)):
        # PARTIAL requires the part level to be at least WEAKLY positive (> margin/2), not
        # merely 'less negative than the whole'. A null part-level (matched ~= random,
        # mmr ~ 0) with high recurrence-from-noise must NOT read as PARTIAL -> it is NULL.
        label = "PARTIAL"
        lines.append(
            f"PARTIAL (best granularity={best_g}): part-level matched-minus-random="
            f"{best_mmr:+.4f} is weakly positive and BEATS the whole-level {WHOLE_LEVEL_MMR:+.3f} "
            f"(composition helps relative to the pool) but does not clear ALL gates "
            f"(clearly_positive={clearly_pos}, recurrence={rec_val:.1f}>{recurrence_bar}="
            f"{recurs}, frac_beats={fbeats:.2f}>0.55={consistent}). Suggestive, not "
            f"confirmed; composition partially rescues retrieval.")
    else:
        label = "NULL"
        lines.append(
            f"NULL (best granularity={best_g}): part-level matched-minus-random="
            f"{best_mmr:+.4f} does NOT clearly beat random and/or does not beat the whole-"
            f"level {WHOLE_LEVEL_MMR:+.3f} by the margin {margin}. Composition does NOT rescue "
            f"retrieval — even the local factors/motifs do not carry transferable matched "
            f"signal beyond a random good-prime on these random graphs.")
    lines.append(
        f"(edge mmr={e_mmr:+.4f} rec={edge_rec.get('mean_instances_per_cluster', float('nan')):.1f}; "
        f"cell mmr={c_mmr:+.4f} rec={cell_rec.get('mean_instances_per_cluster', float('nan')):.1f}; "
        f"whole-level reference mmr={WHOLE_LEVEL_MMR:+.3f}.)")
    return {"label": label, "best_granularity": best_g, "best_matched_minus_random": best_mmr,
            "beats_whole_level": bool(beats_whole), "clearly_positive": bool(clearly_pos),
            "recurs": bool(recurs), "consistent": bool(consistent),
            "whole_level_mmr": WHOLE_LEVEL_MMR, "margin": margin,
            "recurrence_bar": recurrence_bar, "lines": lines}


# ===========================================================================
# Reporting
# ===========================================================================

def _print_curve(alphas, curves):
    head = "  alpha:    " + "  ".join(f"{a:>5.2f}" for a in alphas)
    print(f"      {head}", flush=True)
    for ctrl in ("matched", "random", "global"):
        vals = "  ".join(f"{v:>5.3f}" for v in curves[ctrl])
        print(f"        {ctrl:>8}: {vals}", flush=True)


def _print_recurrence(name, rec):
    print(f"    [{name}] k_clusters={rec['k_clusters']} parts={rec['n_parts']} "
          f"instances={rec['n_instances']}", flush=True)
    print(f"      instances/cluster: mean={rec['mean_instances_per_cluster']:.2f} "
          f"median={rec['median_instances_per_cluster']:.1f}  "
          f"frac multi-instance={rec['frac_clusters_multi_instance']:.2f}", flush=True)
    print(f"      shuffle-null instances/cluster={rec.get('shuffle_null_instances_per_cluster', float('nan')):.2f} "
          f"(size artifact floor; observed >> this = genuine recurrence)  "
          f"whole-graph baseline ~ {rec['whole_graph_baseline']:.1f} (instance-unique)",
          flush=True)


def _print_retrieval(res):
    a = res["aggregate"]
    print(f"    GRANULARITY={res['granularity']} (queries scored={res['n_queries_scored']}, "
          f"lib-eligible={res['n_lib_eligible']}, skipped(no lib)={res['n_skipped_no_lib']})",
          flush=True)
    print(f"      AUC(alpha) curves (mean over queries; own-positive vs own-negative):",
          flush=True)
    _print_curve(res["alphas"], res["curves"])
    print(f"      frac helped:   matched={a['frac_helped_matched']:.3f}  "
          f"random={a['frac_helped_random']:.3f}", flush=True)
    print(f"      mean improve:  matched={a['mean_improvement_matched']:+.4f}  "
          f"random={a['mean_improvement_random']:+.4f}  "
          f"global={a['mean_improvement_global']:+.4f}", flush=True)
    print(f"      MATCHED-minus-RANDOM (the retrieval value): "
          f"mean={a['mean_matched_minus_random']:+.4f}  "
          f"median={a['median_matched_minus_random']:+.4f}  "
          f"frac_matched>random={a['frac_matched_beats_random']:.3f}", flush=True)
    print(f"      AUC: baseline(a=0)={a['mean_auc0']:.3f}  "
          f"matched-opt={a['mean_auc_opt_matched']:.3f}  "
          f"random-opt={a['mean_auc_opt_random']:.3f}  "
          f"global-opt={a['mean_auc_opt_global']:.3f}", flush=True)
    at = a["attributes"]
    print(f"      [which primes transfer] help-rate same-motif={at['help_rate_same_motif']:.3f}"
          f" (n={at['n_same_motif']})  diff-motif={at['help_rate_diff_motif']:.3f} "
          f"(n={at['n_diff_motif']})  same-band={at['help_rate_same_band']:.3f} "
          f"(n={at['n_same_band']})", flush=True)


def _print_report(result):
    print(f"\n{'='*78}", flush=True)
    print(f"  PER-FACTOR RETRIEVAL PROBE — part-level (edge/cell) cross-instance matching",
          flush=True)
    print(f"{'='*78}", flush=True)
    m = result.get("meta", {})
    if m:
        print(f"  source: ckpt={m.get('ckpt','?')} domain={m.get('domain','?')} "
              f"mech={m.get('mech','?')} K={m.get('K','?')} m_max={m.get('m_max','?')} "
              f"bands={m.get('bands','?')}", flush=True)
    print(f"  parts: edges={result['n_edges_parts']}  cells={result['n_cells_parts']}  "
          f"instances={result['n_instances']}  "
          f"reused retrieval_utility helpers={_REUSED_HELPERS}", flush=True)

    print(f"\n  [RECURRENCE TEST — do part-level reps CLUSTER cross-instance?]", flush=True)
    _print_recurrence("edge", result["recurrence"]["edge"])
    _print_recurrence("cell", result["recurrence"]["cell"])

    print(f"\n  [RETRIEVAL-UTILITY TEST — by-instance matched vs random "
          f"(mirrors retrieval_utility_probe; whole-level mmr={WHOLE_LEVEL_MMR:+.3f})]",
          flush=True)
    _print_retrieval(result["retrieval"]["edge"])
    _print_retrieval(result["retrieval"]["cell"])

    print(f"\n  [HEAD-TO-HEAD vs WHOLE-LEVEL]", flush=True)
    e_mmr = result["retrieval"]["edge"]["aggregate"]["mean_matched_minus_random"]
    c_mmr = result["retrieval"]["cell"]["aggregate"]["mean_matched_minus_random"]
    print(f"      whole-level (pooled silhouette): matched-minus-random={WHOLE_LEVEL_MMR:+.3f}"
          f" (GLOBAL_PRIOR_SUFFICES)", flush=True)
    print(f"      part-level EDGE: matched-minus-random={e_mmr:+.4f}  "
          f"(delta vs whole={e_mmr - WHOLE_LEVEL_MMR:+.4f})", flush=True)
    print(f"      part-level CELL: matched-minus-random={c_mmr:+.4f}  "
          f"(delta vs whole={c_mmr - WHOLE_LEVEL_MMR:+.4f})", flush=True)

    print(f"\n  [VERDICT] label={result['verdict']['label']}", flush=True)
    for ln in result["verdict"]["lines"]:
        print(f"      - {ln}", flush=True)
    _print_caveats()


def _print_caveats():
    print(f"\n  [HONEST CAVEATS — read before acting on the verdict]", flush=True)
    print(f"      (1) FROZEN-REP PROXY: blends in the CAPTURED (baseline, final-breath) "
          f"per-cell rep space — a proxy/LOWER BOUND for real in-deducer injection (which "
          f"re-runs deduction with the blend). A positive is SUGGESTIVE; a null is damning.",
          flush=True)
    print(f"      (2) BASELINE-MODEL REPS: from the baseline deducer ckpt; an energy/"
          f"contrastive ckpt may cluster the primes more sharply (sharper matches).",
          flush=True)
    print(f"      (3) FINAL-BREATH: the part reps are the FINAL-breath readout; the realistic "
          f"in-deducer query is an EARLY-breath state. A follow-up early-breath per-cell "
          f"capture would test the deployed regime.", flush=True)
    print(f"      (4) THIS TESTS RETRIEVAL SIGNAL (does a matched library prime carry "
          f"transferable part signal), NOT the in-deducer use. The triviality guard "
          f"(matched-vs-random, reading the SAME local rep) isolates the MATCHING value; "
          f"actually wiring a part-dictionary into the deducer is a further step.", flush=True)


# ===========================================================================
# Per-match log writers
# ===========================================================================

_CSV_COLUMNS = [
    "granularity", "query_inst_id", "query_band", "query_n_pos", "query_n_neg",
    "query_fold", "query_tri_frac", "query_deg", "lib_size",
    "matched_lib_inst_id", "matched_band", "matched_tri_frac", "matched_deg",
    "same_band", "same_motif_tri", "same_degree", "match_cosine_distance",
    "optimal_alpha_matched", "auc_at_optimal_matched", "auc_at_alpha0",
    "improvement_matched", "optimal_alpha_random", "improvement_random",
    "improvement_global", "matched_minus_random", "helped",
]


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def write_logs(result, out_prefix):
    """Write the per-match CSV (edge + cell rows) + the full JSON."""
    csv_path = out_prefix + "_permatch.csv"
    json_path = out_prefix + ".json"
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)) or ".", exist_ok=True)
    all_rows = (result["retrieval"]["edge"]["rows"] + result["retrieval"]["cell"]["rows"])
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_CSV_COLUMNS)
        w.writeheader()
        for r in all_rows:
            w.writerow({k: r.get(k, "") for k in _CSV_COLUMNS})
    payload = dict(result)
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2, default=_json_default)
    print(f"\n  [WROTE] per-match CSV : {os.path.abspath(csv_path)} ({len(all_rows)} rows)",
          flush=True)
    print(f"  [WROTE] full JSON    : {os.path.abspath(json_path)}", flush=True)
    return csv_path, json_path


# ===========================================================================
# Top-level run
# ===========================================================================

def run_probe(parts, *, min_pos=2, folds=5, alphas=ALPHAS_DEFAULT, seed=0,
              k_clusters=None, verbose=True):
    """Run recurrence + retrieval on BOTH granularities, plus the verdict."""
    edges_tab, cells_tab = parts["edges"], parts["cells"]
    edge_rec = recurrence_test(edges_tab["reps"], edges_tab["inst_id"],
                               k_clusters=k_clusters, seed=seed)
    cell_rec = recurrence_test(cells_tab["reps"], cells_tab["inst_id"],
                               k_clusters=k_clusters, seed=seed)
    edge_attrs = {"deg_min": edges_tab["deg_min"], "deg_max": edges_tab["deg_max"],
                  "is_tri": edges_tab["is_tri"]}
    cell_attrs = {"degree": cells_tab["degree"], "is_tri": cells_tab["is_tri"]}
    edge_res = run_retrieval(edges_tab, edge_attrs, granularity="edge", min_pos=min_pos,
                             folds=folds, alphas=alphas, seed=seed, verbose=False)
    cell_res = run_retrieval(cells_tab, cell_attrs, granularity="cell", min_pos=min_pos,
                             folds=folds, alphas=alphas, seed=seed, verbose=False)
    V = make_verdict(edge_res, cell_res, edge_rec, cell_rec)
    result = {
        "meta": parts.get("meta", {}),
        "n_instances": parts["n_instances"],
        "n_edges_parts": int(edges_tab["reps"].shape[0]),
        "n_cells_parts": int(cells_tab["reps"].shape[0]),
        "whole_level_mmr": WHOLE_LEVEL_MMR,
        "recurrence": {"edge": edge_rec, "cell": cell_rec},
        "retrieval": {"edge": edge_res, "cell": cell_res},
        "verdict": V,
        "reused_helpers": _REUSED_HELPERS,
    }
    if verbose:
        _print_report(result)
    return result


def run_from_npz(npz_path, *, min_pos=2, folds=5, seed=0, max_darts_per_inst=None,
                 k_clusters=None, out_prefix=None):
    z = np.load(npz_path, allow_pickle=True)
    print(f"\n  LOADED {npz_path}", flush=True)
    print(f"  darts={z['dart_reps'].shape[0]}  per-cell reps shape={z['dart_reps'].shape}  "
          f"instances={z['inst_id'].shape[0]}  "
          f"VALID darts={int(np.asarray(z['dart_valid']).sum())}", flush=True)
    parts = build_parts(z, max_darts_per_inst=max_darts_per_inst, seed=seed)
    res = run_probe(parts, min_pos=min_pos, folds=folds, seed=seed,
                    k_clusters=k_clusters, verbose=True)
    res["npz_path"] = npz_path
    if out_prefix:
        write_logs(res, out_prefix)
    return res


# ===========================================================================
# SELFTEST — recurring-prime (part detects, whole washes out) + null (no false positive)
# ===========================================================================

def _make_synth_capture(n_inst, m_darts, H, n_per_inst, recurring, seed=0):
    """Build a SYNTHETIC per-cell capture npz dict that exercises the COMPOSITION mechanism.

    GEOMETRY (mirrors the PROVEN retrieval_utility_probe synthetic, lifted to the PART level).
    Each instance belongs to a FAMILY (a recurring 'prime'). The family GOOD direction is the
    transferable signal: it DISCRIMINATES satisfied from violated edges and RECURS across the
    family's instances. To create HEADROOM (so the alpha=0 baseline does NOT already separate
    own-positive from own-negative -- the triviality anchor), per-instance IDENTITY dominates
    a single part's rep and heavy per-part NOISE buries a single part's good direction; only a
    library POS-CENTROID (averaged over MANY satisfied parts of OTHER family instances ->
    identity + noise cancel, family good survives clean) recovers it.

      * PART level (recurring): a query part's nearest library pos-centroid is a SAME-FAMILY
        instance's centroid -> it carries the clean family good direction -> blending injects
        the RIGHT direction -> AUC(own-pos vs own-neg) rises. A RANDOM library centroid carries
        a foreign family's good direction -> no rise. matched >> random.
      * WHOLE level (recurring): pooling ALL of an instance's parts (sat + violated) per dart
        into ONE whole silhouette is dominated by per-instance IDENTITY (the unique graph) and
        the family good direction is HALVED (only sat parts carry it) + averaged against
        violated parts -> the whole silhouette matches on IDENTITY, not the prime -> matched
        ~= random (washes out). This is the -0.048 regime we are trying to beat.
    recurring=False : NULL -- each instance has its OWN independent good direction (no shared
      family) -> no library centroid agrees with a query part -> matched ~= random at BOTH
      levels.

    Construction is at the EDGE-part level (the probe's per-edge path): each edge's 2 endpoint
    cells each carry HALF the family good direction when satisfied (so the 2-endpoint MEAN ==
    the per-edge rep lands at identity + GOOD_MAG*family_good + noise), none when violated.
    """
    rng = np.random.RandomState(seed)
    # FEW families (recurring 'primes') so a RANDOM library pick is rarely the query's HOME
    # family (matched must genuinely beat random), but each family recurs across many
    # instances. THE KEY to whole-washout: an instance is a MIX of families across its edges
    # (a random graph is a MIX of motifs), with a skew toward a per-instance HOME family. So:
    #   * per-INSTANCE pos-centroid leans toward the HOME family -> a query edge of family F
    #     matches the library centroids of instances whose home is F -> transferable (part).
    #   * the WHOLE silhouette = mean over the instance's MIX of families -> a unique blend
    #     per instance (instance-IDENTITY) with NO single transferable family -> matches on
    #     the blend, not a prime -> washes out (the real -0.048 regime).
    n_families = 5
    fam_good = rng.randn(n_families, H)
    fam_good /= np.linalg.norm(fam_good, axis=1, keepdims=True)
    inst_identity = rng.randn(max(1, n_inst), H)   # per-instance identity (unique graph)
    inst_identity /= np.linalg.norm(inst_identity, axis=1, keepdims=True)

    # THE PART-vs-WHOLE ASYMMETRY (faithful to 'parts recur, wholes are unique').
    #   * IDENTITY is a LARGE per-instance offset on EVERY cell. It is the SAME within an
    #     instance (so it dominates the WHOLE silhouette) but ORTHOGONAL ACROSS instances
    #     (random) -> in a cross-instance PART match the identity contributes ~0 to the cosine
    #     (orthogonal) so the match is driven by the SHARED family good. At the WHOLE level
    #     the family good is DILUTED (only satisfied parts carry it, averaged with violated
    #     ones) so identity dominates -> the whole matches on IDENTITY (unique, untransferable)
    #     -> washes out (the real -0.048 regime).
    #   * FAMILY GOOD is at FULL magnitude in a single satisfied edge (a local motif) and
    #     RECURS across the family's instances -> the per-edge part match recovers it.
    # PART vs WHOLE: the per-EDGE label (satisfied) is tied to the recurring FAMILY good (the
    # part match transfers); the WHOLE label is a coin flip uncorrelated with any transferable
    # direction (the whole match washes out — see whole_valid below). IDENTITY is a moderate
    # per-instance offset (orthogonal across instances) that keeps the part match driven by the
    # SHARED family good while making each whole silhouette instance-unique.
    GOOD_MAG = 4.0        # family good in a SATISFIED edge mean (full at the part level)
    ID_MAG = 2.0          # per-instance identity (orthogonal across instances)
    NOISE = 1.6           # per-part noise (buries single-part + query-mean good_dir)
    HOME_FRAC = 0.8       # fraction of an instance's edges from its HOME family (the skew)
    S = max(4, 2 * n_per_inst)
    n_edges = min(n_per_inst, S // 2)
    L = n_edges

    dart_reps, dart_colors, dart_inst_id, dart_idx, dart_valid = [], [], [], [], []
    inst_id_l, inst_membership, inst_cell_valid, inst_degree, inst_band, inst_n = \
        [], [], [], [], [], []

    for g in range(n_inst):
        gid = 2000 * 1_000_000 + g
        home_fam = g % n_families
        identity = ID_MAG * inst_identity[g]
        mem = np.zeros((L, S), dtype=np.float16)
        cv = np.zeros((S,), dtype=bool)
        deg = np.zeros((S,), dtype=np.int16)
        edge_endpoints = []
        edge_fam = []
        for e in range(n_edges):
            u, v = 2 * e, 2 * e + 1
            mem[e, u] = 1.0
            mem[e, v] = 1.0
            cv[u] = True
            cv[v] = True
            deg[u] = 1
            deg[v] = 1
            edge_endpoints.append((u, v))
            # each edge's family: HOME with prob HOME_FRAC, else a random OTHER family (the
            # mix that makes the WHOLE blend unique).
            if rng.rand() < HOME_FRAC:
                edge_fam.append(home_fam)
            else:
                edge_fam.append(int(rng.randint(n_families)))
        inst_id_l.append(gid)
        inst_membership.append(mem)
        inst_cell_valid.append(cv)
        inst_degree.append(deg)
        inst_band.append(float([1.0, 1.5, 2.0, 2.5][g % 4]))
        inst_n.append(int(cv.sum()))

        # per-instance independent good directions for the NULL (no shared family).
        null_good = None
        if not recurring:
            grng = np.random.RandomState(seed * 100003 + g * 131)
            null_good = grng.randn(n_edges, H)
            null_good /= np.linalg.norm(null_good, axis=1, keepdims=True)

        for di in range(m_darts):
            reps = np.zeros((S, H), dtype=np.float16)
            colors = np.zeros((S,), dtype=np.int16)
            for e, (u, v) in enumerate(edge_endpoints):
                sat = bool(rng.rand() < 0.5)
                if sat:
                    colors[u] = 1
                    colors[v] = 2                  # differ -> satisfied
                else:
                    colors[u] = 1
                    colors[v] = 1                  # same -> violated
                if recurring:
                    gd = fam_good[edge_fam[e]]     # SHARED within the edge's family
                else:
                    gd = null_good[e]              # independent per (instance, edge)
                # endpoint reps: each carries HALF the family good when satisfied (so the
                # 2-endpoint MEAN == per-edge rep carries the FULL GOOD_MAG*gd), the dominating
                # per-instance identity (ORTHOGONAL across instances), + heavy per-endpoint
                # noise.
                base_u = identity + NOISE * rng.randn(H)
                base_v = identity + NOISE * rng.randn(H)
                if sat:
                    base_u = base_u + GOOD_MAG * gd
                    base_v = base_v + GOOD_MAG * gd
                reps[u] = base_u.astype(np.float16)
                reps[v] = base_v.astype(np.float16)
            # whole-graph valid flag: a COIN FLIP, INDEPENDENT of the family-good content.
            # THE POINT (faithful to 'wholes are unique on random graphs'): the whole pos vs
            # neg darts of an instance do NOT differ by a transferable family direction (the
            # whole silhouette is dominated by instance IDENTITY + isotropic noise), so the
            # whole pos-centroid carries NO transferable discriminator -> cross-instance whole
            # matching adds nothing over random (matched ~= random, washes out), exactly the
            # real -0.048 regime. Meanwhile the per-EDGE label (satisfied) stays tied to the
            # recurring family prime -> the PART match transfers. (Each instance still gets a
            # mix of T/F whole labels so the whole pos/neg split is defined.)
            whole_valid = bool(rng.rand() < 0.5)
            dart_reps.append(reps)
            dart_colors.append(colors)
            dart_inst_id.append(gid)
            dart_idx.append(di)
            dart_valid.append(whole_valid)

    z = {
        "dart_reps": np.stack(dart_reps).astype(np.float16),
        "dart_colors": np.stack(dart_colors).astype(np.int16),
        "dart_inst_id": np.asarray(dart_inst_id, dtype=np.int64),
        "dart_idx": np.asarray(dart_idx, dtype=np.int32),
        "dart_valid": np.asarray(dart_valid, dtype=bool),
        "inst_id": np.asarray(inst_id_l, dtype=np.int64),
        "inst_membership": np.stack(inst_membership).astype(np.float16),
        "inst_cell_valid": np.stack(inst_cell_valid).astype(bool),
        "inst_degree": np.stack(inst_degree).astype(np.int16),
        "inst_band": np.asarray(inst_band, dtype=np.float64),
        "inst_n": np.asarray(inst_n, dtype=np.int32),
        "meta": np.array({"edge_ltype": 0, "synthetic": True, "recurring": recurring},
                         dtype=object),
    }
    return z


def _whole_level_mmr_from_capture(z, *, min_pos=2, folds=5, seed=0):
    """Compute the WHOLE-LEVEL matched-minus-random on the SAME synthetic capture, by POOLING
    each dart's parts into one whole-graph silhouette (mean over all valid cells) — exactly
    what the whole-level retrieval_utility_probe did. Used by the selftest to assert that the
    recurring-prime signal WASHES OUT at the whole level (matched ~= random) while it
    DETECTS at the part level.

    We treat each dart as a 'whole silhouette' = mean over valid cells; label = the dart's
    whole_valid flag; group by instance. This mirrors the dart_silhouettes contract the
    whole-level probe consumed.
    """
    dart_reps = np.asarray(z["dart_reps"], dtype=np.float32)
    dart_inst = np.asarray(z["dart_inst_id"], dtype=np.int64)
    dart_valid = np.asarray(z["dart_valid"], dtype=bool)
    inst_id = np.asarray(z["inst_id"], dtype=np.int64)
    inst_cv = np.asarray(z["inst_cell_valid"], dtype=bool)
    ipos = {int(i): k for k, i in enumerate(inst_id)}
    reps, label, inst = [], [], []
    for d in range(dart_reps.shape[0]):
        iid = int(dart_inst[d])
        cv = inst_cv[ipos[iid]]
        sil = dart_reps[d][cv].mean(axis=0)
        reps.append(sil.astype(np.float64))
        label.append(bool(dart_valid[d]))
        inst.append(iid)
    tab = {"reps": np.asarray(reps, dtype=np.float64), "label": np.asarray(label, dtype=bool),
           "inst_id": np.asarray(inst, dtype=np.int64),
           "band": np.zeros(len(inst), dtype=np.float64)}
    # reuse the SAME retrieval machinery (granularity='whole' for logging).
    res = run_retrieval(tab, {}, granularity="cell", min_pos=min_pos, folds=folds, seed=seed,
                        verbose=False)
    return res["aggregate"]["mean_matched_minus_random"]


def selftest() -> bool:
    print("=== per_factor_retrieval_probe SELFTEST (CPU) ===", flush=True)
    ok = True

    # --- AUC machinery sanity (reused helper). ---
    s = np.array([0.1, 0.2, 0.3, 0.9, 1.0, 1.1])
    y = np.array([False, False, False, True, True, True])
    auc_sep = auc_mann_whitney(s, y)
    auc_flat = auc_mann_whitney(np.ones(6), y)
    print(f"  [auc] separable={auc_sep:.3f} (1.0)  flat={auc_flat:.3f} (0.5)  "
          f"reused_helpers={_REUSED_HELPERS}", flush=True)
    ok &= abs(auc_sep - 1.0) < 1e-9 and abs(auc_flat - 0.5) < 1e-9

    # --- build_parts contract on a tiny synthetic capture (edge sat/violated + degree). ---
    z_tiny = _make_synth_capture(6, 4, 16, 4, recurring=True, seed=3)
    parts_tiny = build_parts(z_tiny, seed=0)
    e = parts_tiny["edges"]
    print(f"  [build_parts] edges={e['reps'].shape[0]} sat={int(e['label'].sum())} "
          f"violated={int((~e['label']).sum())} cells={parts_tiny['cells']['reps'].shape[0]}",
          flush=True)
    ok &= (e["reps"].shape[0] > 0 and e["label"].sum() > 0 and (~e["label"]).sum() > 0)
    # the per-edge label MUST equal endpoints-differ derived from dart_colors (cross-check
    # the first few edges directly).
    cross_ok = True
    z = z_tiny
    inst_cv = np.asarray(z["inst_cell_valid"], dtype=bool)
    inst_mem = np.asarray(z["inst_membership"], dtype=np.float32)
    inst_ids = np.asarray(z["inst_id"], dtype=np.int64)
    dart_inst = np.asarray(z["dart_inst_id"], dtype=np.int64)
    dart_col = np.asarray(z["dart_colors"], dtype=np.int64)
    # recompute one edge's label and compare to build_parts.
    ip = {int(i): k for k, i in enumerate(inst_ids)}
    # find the first part: it belongs to dart 0's first instance, first edge.
    first_iid = int(e["inst_id"][0])
    first_dart = int(e["dart_idx"][0])
    didx = np.nonzero((dart_inst == first_iid))[0]
    didx = [d for d in didx if int(z["dart_idx"][d]) == first_dart][0]
    mem0 = inst_mem[ip[first_iid]]
    ones = np.nonzero(mem0[0] > 0.5)[0]
    expect_sat = bool(dart_col[didx][ones[0]] != dart_col[didx][ones[1]])
    cross_ok = (bool(e["label"][0]) == expect_sat)
    print(f"  [build_parts] first-edge label matches endpoints-differ: {cross_ok}", flush=True)
    ok &= cross_ok

    H = 40
    # --- RECURRING: matched>>random at PART level; verdict PRIMES_CONFIRMED (or PARTIAL). ---
    z_rec = _make_synth_capture(80, 12, H, 6, recurring=True, seed=1)
    parts_rec = build_parts(z_rec, seed=1)
    res_rec = run_probe(parts_rec, min_pos=2, folds=5, seed=1, verbose=False)
    e_mmr = res_rec["retrieval"]["edge"]["aggregate"]["mean_matched_minus_random"]
    e_rec = res_rec["recurrence"]["edge"]["mean_instances_per_cluster"]
    whole_mmr = _whole_level_mmr_from_capture(z_rec, seed=1)
    print(f"\n  [recurring] PART edge mmr={e_mmr:+.4f}  recurrence={e_rec:.1f} inst/cluster  "
          f"WHOLE mmr={whole_mmr:+.4f}  verdict={res_rec['verdict']['label']}", flush=True)
    # the DECISIVE composition assertions:
    #  (1) part-level matched >> random (clearly positive),
    #  (2) part-level BEATS the whole level, AND the whole level WASHES OUT (whole-level mmr
    #      well BELOW part-level — exactly the real-data -0.048 regime the part rescues),
    #  (3) recurrence high (primes recur across instances).
    cond_part = e_mmr > 0.02
    cond_beats_whole = e_mmr > whole_mmr + 0.02
    cond_recur = e_rec > 2.0
    cond_washes = whole_mmr < e_mmr - 0.02    # whole washes out, well below the part level
    cond_verdict = res_rec["verdict"]["label"] in ("PRIMES_CONFIRMED", "PARTIAL")
    print(f"  [recurring] EXPECT part>>random ({cond_part}), part beats whole "
          f"({cond_beats_whole}), recurrence high ({cond_recur}), whole washes ({cond_washes}),"
          f" verdict confirmed/partial ({cond_verdict})", flush=True)
    rec_pass = cond_part and cond_beats_whole and cond_recur and cond_washes and cond_verdict
    print(f"  [recurring] -> {'PASS' if rec_pass else 'FAIL'}", flush=True)
    ok &= rec_pass

    # --- NULL: no recurring primes -> matched ~= random at BOTH levels; verdict NULL. ---
    z_null = _make_synth_capture(80, 12, H, 6, recurring=False, seed=2)
    parts_null = build_parts(z_null, seed=2)
    res_null = run_probe(parts_null, min_pos=2, folds=5, seed=2, verbose=False)
    n_mmr = res_null["retrieval"]["edge"]["aggregate"]["mean_matched_minus_random"]
    whole_mmr_n = _whole_level_mmr_from_capture(z_null, seed=2)
    print(f"\n  [null] PART edge mmr={n_mmr:+.4f}  WHOLE mmr={whole_mmr_n:+.4f}  "
          f"verdict={res_null['verdict']['label']}", flush=True)
    cond_null_part = n_mmr <= 0.02
    cond_null_verdict = res_null["verdict"]["label"] != "PRIMES_CONFIRMED"
    print(f"  [null] EXPECT part~=random (mmr<=0.02; {cond_null_part}) + NOT confirmed "
          f"({cond_null_verdict})", flush=True)
    null_pass = cond_null_part and cond_null_verdict
    print(f"  [null] -> {'PASS' if null_pass else 'FAIL'}", flush=True)
    ok &= null_pass

    # --- no-leak structural check: a query part NEVER matches its own instance. ---
    leaked = any(r["matched_lib_inst_id"] == r["query_inst_id"]
                 for r in (res_rec["retrieval"]["edge"]["rows"]
                           + res_rec["retrieval"]["cell"]["rows"]
                           + res_null["retrieval"]["edge"]["rows"]))
    print(f"\n  [no-leak] any query matched its OWN instance? {leaked} (expect False)",
          flush=True)
    ok &= (leaked is False)

    # --- csv/json round-trip on the recurring run. ---
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        prefix = os.path.join(td, "syn_perfactor")
        csv_path, json_path = write_logs(res_rec, prefix)
        ok &= os.path.exists(csv_path) and os.path.exists(json_path)
        with open(json_path) as f:
            loaded = json.load(f)
        ok &= ("verdict" in loaded and "retrieval" in loaded)
        with open(csv_path) as f:
            header = f.readline().strip().split(",")
        required = {"granularity", "query_inst_id", "matched_lib_inst_id",
                    "same_motif_tri", "match_cosine_distance", "improvement_matched",
                    "improvement_random", "matched_minus_random", "helped"}
        missing = required - set(header)
        print(f"  [io] csv+json round-trip; missing required cols={missing} (expect empty)",
              flush=True)
        ok &= (len(missing) == 0)

    # --- exercise the full verbose report once. ---
    print("\n  [report] full verbose report on the RECURRING run:", flush=True)
    _print_report(res_rec)

    print(f"\n  SELFTEST {'PASSED' if ok else 'FAILED'}", flush=True)
    return ok


# ===========================================================================
# Main
# ===========================================================================

def main(argv=None) -> int:
    P = argparse.ArgumentParser(
        description="Part-level (per-edge/per-cell) cross-instance retrieval probe — does "
                    "composition rescue retrieval where the whole-graph pool failed?")
    P.add_argument("--npz", default=os.environ.get("CELL_NPZ", None),
                   help="path to the per-cell capture .npz (from --capture-cells)")
    P.add_argument("--selftest", action="store_true",
                   help="run the CPU selftest (recurring-prime + null) and exit")
    P.add_argument("--min-pos", type=int, default=2,
                   help="min POSITIVE parts for an instance to enter the library (default 2)")
    P.add_argument("--folds", type=int, default=5,
                   help="k-fold over INSTANCES (by-instance no-leak split; default 5)")
    P.add_argument("--seed", type=int, default=0, help="fold/random-control seed")
    P.add_argument("--max-darts-per-inst", type=int,
                   default=int(os.environ["MAX_DARTS_PER_INST"])
                   if os.environ.get("MAX_DARTS_PER_INST") else None,
                   help="optional per-instance dart subsample to bound the part count")
    P.add_argument("--k-clusters", type=int, default=None,
                   help="k-means clusters for the recurrence test (default ~#instances)")
    P.add_argument("--out-prefix", default=None,
                   help="write <prefix>_permatch.csv + <prefix>.json (per-match log)")
    args = P.parse_args(argv)

    if args.selftest or os.environ.get("SELFTEST_ONLY", "0") == "1":
        return 0 if selftest() else 1
    if not args.npz:
        print("error: pass --npz PATH (or --selftest). PATH is the per-cell capture npz "
              "from amortized_frontier_measure.py --capture-cells.", flush=True)
        return 2
    if not os.path.exists(args.npz):
        print(f"error: npz not found: {args.npz}", flush=True)
        return 2
    run_from_npz(args.npz, min_pos=args.min_pos, folds=args.folds, seed=args.seed,
                 max_darts_per_inst=args.max_darts_per_inst, k_clusters=args.k_clusters,
                 out_prefix=args.out_prefix)
    return 0


if __name__ == "__main__":
    parse_ok = _ast_parse_ok()
    print(f"[ast.parse] astparse_ok={parse_ok}", flush=True)
    if not parse_ok:
        sys.exit(1)
    sys.exit(main())
