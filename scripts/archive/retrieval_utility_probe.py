"""retrieval_utility_probe.py — does CROSS-INSTANCE RETRIEVAL of a good-pattern centroid
carry SOLUTION-RELEVANT signal? (pure numpy, CPU; no GPU/training; no engine edits).

THE TWO-PATH IDEA WE ARE GATING. A would-be architecture: a normal deduction path PLUS a
"waist" path that, each breath, compresses the residual to a low-dim COMMON MODE (a
silhouette), MATCHES it against a LIBRARY of known good-pattern centroids (silhouettes of
CORRECT solutions from already-solved problems), and INJECTS the matched centroid back to
steer the deduction (case-based / amortized reuse of solution shapes). Before paying for
that (an extra path + a training run + the attention-bootstrap risk of a retrieval pathway),
we GATE it with a cheap OFFLINE probe on captured silhouettes we already have.

THE CLAIM TO TEST — retrieval UTILITY, not just separability. The prior dart_cluster_probe
asked "do valid darts cluster (is there a good pattern)?" and gated at centered-global
AUC>0.6 (separability). Separability is NECESSARY but NOT SUFFICIENT. This probe asks the
DECISIVE question: if we retrieve a good-pattern centroid from OTHER (solved) instances and
blend a held-out instance TOWARD it, does that move the held-out instance toward ITS OWN
valid region? I.e. does cross-instance RETRIEVAL carry signal that helps the SPECIFIC query?

THE DATA CONTRACT (.cache/dart_silhouettes_fg_coloring_k16.npz, from the BASELINE deducer):
  reps    (n_darts, H) float32  — per-dart FINAL-breath readout silhouette (pooled valid cells)
  valid   (n_darts,)  bool      — FREE exact-verifier flag (VALID coloring or not)
  inst_id (n_darts,)  int64     — band-namespaced per-instance id (the grouping key)
  band    (n_darts,)  float64   — edge-density band c (1.0/1.5/2.0/2.5)
  meta    (object)              — dict {ckpt, mech, K, m_max, bands, ...}

WHAT THIS PROBE DOES:
  LIBRARY. For each instance with >= MIN_VALID valid darts (default 3), its good-pattern
    centroid = mean of its VALID dart silhouettes. The library = these per-instance
    valid-centroids.
  BY-INSTANCE SPLIT (the load-bearing no-leak guard). k-fold (default 5) over INSTANCES. For
    a held-out (query) instance, the library is ONLY the centroids of TRAIN-fold instances —
    NEVER the query instance's own centroid. The query never matches against itself.
  QUERY rep. The held-out instance's mean silhouette over its darts (a stand-in for "where
    the deduction currently is"). RETRIEVE the nearest library centroid by COSINE — in TWO
    geometries (compared): RAW (uncentered: matches on graph-IDENTITY / instance-similarity =
    case-based "find the most similar solved graph") and CENTERED (subtract the GLOBAL mean of
    library centroids: matches on the transferable ABSTRACT good-pattern, identity removed).
  THE DECISIVE CONTROLS (matched vs not). For each query, also a RANDOM library centroid (a
    random train-instance centroid) and the GLOBAL good-centroid (mean of ALL train
    valid-centroids). The verdict HINGES on MATCHED beating RANDOM (and the global mean): if
    matched ~= random, the MATCHING adds nothing (any good centroid blends the same) -> the
    two-path RETRIEVAL is NOT justified; a single GLOBAL good-prior would do (simpler).
  ALPHA-SWEEP (the blend question). blended = (1-alpha)*query + alpha*centroid for alpha in
    {0,0.1,0.25,0.5,0.75,1.0} (alpha=0 = no blend = baseline). HELP METRIC: does the blended
    point point toward the query instance's OWN valid region better than its OWN invalid
    region? Score each of the query's own darts by cosine-sim to the blended point;
    AUC(own-valid vs own-invalid) via Mann-Whitney. The alpha maximizing this AUC = the
    optimal blend for that query. Report the AUC(alpha) curve PER control.
  STUCK-RESCUE variant. Take the query's own INVALID darts; does blending them toward the
    matched centroid move them closer to the query's own VALID-centroid than to its own
    INVALID-centroid? (Does retrieval RESCUE stuck states.)

PER-MATCH LOG (csv + json). One row per query: query/matched inst ids + bands, same_band,
  match cosine-distance, optimal_alpha + AUC at optimal + AUC at alpha0 (baseline) +
  improvement, for MATCHED and RANDOM, matched_minus_random (the per-query retrieval value),
  helped(bool). -> we can INFER which attributes drive help (same-band? close match?).

VERDICT. RETRIEVAL JUSTIFIED iff MATCHED beats RANDOM by a clear margin AND a consistent
  alpha>0 helps. If matched ~= random -> a single GLOBAL good-prior suffices (no retrieval).
  If no alpha>0 helps -> the centroid does not point the right direction (retrieval not
  useful in silhouette space).

HONEST CAVEATS (printed). (1) FROZEN-SILHOUETTE PROXY: blends in the captured (baseline,
  final-breath) silhouette space — a proxy/LOWER BOUND for real in-deducer injection (which
  re-runs deduction with the blend). A positive is suggestive, not proof; a null is fairly
  damning. (2) FINAL-BREATH query: the realistic query is an EARLY-breath state (retrieve
  early to guide the rest); a follow-up capture would test that. (3) BASELINE-model reps: an
  energy ckpt may cluster better. (4) RANDOM coloring graphs are a WEAK testbed (less shared
  structure); structured/recurring domains (the MATH-500 north-star) would favor retrieval more.

SELFTEST (CPU, no npz). PLANTED-MATCHED: each query's OWN valid region is near its planted
  matched centroid (and far from random ones) -> matched >> random, optimal alpha>0. NULL:
  centroids unrelated to query valid regions -> matched ~= random, no alpha>0 helps. Both ASSERTED.

USAGE:
  CPU selftest (GPU-free):
    .venv/bin/python3 scripts/retrieval_utility_probe.py --selftest
  Probe the captured npz (the gate):
    .venv/bin/python3 scripts/retrieval_utility_probe.py \
        --npz .cache/dart_silhouettes_fg_coloring_k16.npz \
        --out-prefix .cache/retrieval_utility_fg_coloring_k16
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

# Reuse the validated helpers from dart_cluster_probe for CONSISTENCY (same cosine,
# centering, Mann-Whitney AUC machinery). Import is best-effort: if the sibling script is
# unavailable we fall back to local identical copies (kept byte-equivalent below).
_SCRIPTS_DIR = os.path.dirname(_THIS_FILE)
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)
try:
    from dart_cluster_probe import (  # type: ignore
        auc_mann_whitney as _auc_mann_whitney,
        _l2norm_rows as _dcp_l2norm_rows,
        _cos_to_vector as _dcp_cos_to_vector,
    )
    _REUSED_HELPERS = True
except Exception:  # pragma: no cover - fallback only if the sibling import breaks
    _REUSED_HELPERS = False
    _auc_mann_whitney = None
    _dcp_l2norm_rows = None
    _dcp_cos_to_vector = None


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
# Core numerical machinery (reuse dart_cluster_probe; identical fallbacks)
# ===========================================================================

def _l2norm_rows(x: np.ndarray) -> np.ndarray:
    """Row-normalize (unit L2). Zero rows left as zero (denominator clamped)."""
    if _REUSED_HELPERS:
        return _dcp_l2norm_rows(x)
    x = np.asarray(x, dtype=np.float64)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.where(n < 1e-12, 1.0, n)
    return x / n


def _cos_to_vector(rows: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Cosine similarity of each row to `vec` -> (n,)."""
    if _REUSED_HELPERS:
        return _dcp_cos_to_vector(rows, vec)
    unit = _l2norm_rows(rows)
    v = np.asarray(vec, dtype=np.float64)
    vn = np.linalg.norm(v)
    if vn < 1e-12:
        return np.zeros(unit.shape[0])
    return unit @ (v / vn)


def auc_mann_whitney(scores: np.ndarray, labels: np.ndarray) -> float:
    """AUC(positive vs negative) via Mann-Whitney U (rank-based, ties via average ranks).
    scores higher => more positive; labels bool. 0.5 if a class is empty."""
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
    sorted_scores = scores[order]
    i = 0
    N = scores.shape[0]
    while i < N:
        j = i
        while j + 1 < N and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        ranks[order[i:j + 1]] = avg_rank
        i = j + 1
    sum_ranks_pos = ranks[labels].sum()
    u_pos = sum_ranks_pos - n_pos * (n_pos + 1) / 2.0
    return float(u_pos / (n_pos * n_neg))


def _cos_one(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors (float)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    an, bn = np.linalg.norm(a), np.linalg.norm(b)
    if an < 1e-12 or bn < 1e-12:
        return 0.0
    return float((a @ b) / (an * bn))


ALPHAS_DEFAULT = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)


# ===========================================================================
# Library construction (per-instance VALID centroids) + by-instance folds
# ===========================================================================

def build_instance_tables(reps, valid, inst_id, band, min_valid=3):
    """Per-instance tables needed by the probe. Returns a dict keyed by uid with:
        valid_centroid : mean of the instance's VALID dart silhouettes (None if < min_valid)
        invalid_centroid : mean of its INVALID darts (None if 0 invalid)
        query_mean : mean over ALL its darts (the QUERY stand-in for the deduction state)
        dart_idx, valid_idx, invalid_idx : global row indices
        band : the instance's band (single, band-namespaced id)
        n_valid, n_invalid : counts
        lib_eligible : n_valid >= min_valid (library inclusion gate)
        query_eligible : has BOTH >=1 valid AND >=1 invalid (so own-valid-vs-invalid AUC defined)
    """
    reps = np.asarray(reps, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    inst_id = np.asarray(inst_id)
    band = np.asarray(band, dtype=np.float64) if band is not None else None
    tables = {}
    for uid in np.unique(inst_id):
        sel = np.nonzero(inst_id == uid)[0]
        vmask = valid[sel]
        v_idx = sel[vmask]
        i_idx = sel[~vmask]
        nv, ni = int(v_idx.shape[0]), int(i_idx.shape[0])
        vc = reps[v_idx].mean(axis=0) if nv > 0 else None
        ic = reps[i_idx].mean(axis=0) if ni > 0 else None
        qmean = reps[sel].mean(axis=0)
        b = float(band[sel[0]]) if band is not None else float("nan")
        tables[int(uid)] = {
            "uid": int(uid),
            "dart_idx": sel,
            "valid_idx": v_idx,
            "invalid_idx": i_idx,
            "valid_centroid": vc,
            "invalid_centroid": ic,
            "query_mean": qmean,
            "band": b,
            "n_valid": nv,
            "n_invalid": ni,
            "lib_eligible": nv >= min_valid,
            "query_eligible": (nv >= 1 and ni >= 1),
        }
    return tables


def assign_instance_folds(uids, folds, seed=0):
    """Round-robin fold assignment over a permuted instance order (whole-instance split)."""
    uids = np.asarray(uids)
    n_inst = uids.shape[0]
    folds = max(2, min(folds, n_inst))
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n_inst)
    fold_of = np.zeros(n_inst, dtype=int)
    fold_of[perm] = np.arange(n_inst) % folds
    return {int(u): int(f) for u, f in zip(uids, fold_of)}, folds


# ===========================================================================
# The blend AUC help metric + stuck-rescue
# ===========================================================================

def _blend(query_vec, centroid_vec, alpha):
    """blended = (1-alpha)*query + alpha*centroid (alpha=0 -> baseline query)."""
    return (1.0 - alpha) * np.asarray(query_vec, dtype=np.float64) \
        + alpha * np.asarray(centroid_vec, dtype=np.float64)


def auc_curve_for_centroid(tab, reps, centroid_vec, alphas):
    """For a query instance `tab` and a target `centroid_vec`, return the AUC(own-valid vs
    own-invalid) of the query's OWN darts scored by cosine-sim to the blended point, for each
    alpha. blended = (1-alpha)*query_mean + alpha*centroid. alpha=0 = no-blend baseline.

    Returns (aucs (len(alphas),), own_dart_global_idx, own_labels) — the labels/idx reused
    across controls for the same query (consistency).
    """
    own_idx = np.concatenate([tab["valid_idx"], tab["invalid_idx"]])
    own_labels = np.concatenate([
        np.ones(tab["valid_idx"].shape[0], dtype=bool),
        np.zeros(tab["invalid_idx"].shape[0], dtype=bool),
    ])
    own_reps = reps[own_idx]
    aucs = np.empty(len(alphas), dtype=np.float64)
    for k, a in enumerate(alphas):
        bp = _blend(tab["query_mean"], centroid_vec, a)
        scores = _cos_to_vector(own_reps, bp)
        aucs[k] = auc_mann_whitney(scores, own_labels)
    return aucs, own_idx, own_labels


def stuck_rescue_for_centroid(tab, reps, centroid_vec, alpha):
    """STUCK-RESCUE: take the query's OWN INVALID darts; blend each toward `centroid_vec` at
    `alpha`; does the blended invalid dart land closer (cosine) to the query's OWN
    VALID-centroid than to its OWN INVALID-centroid? Returns the fraction of invalid darts
    rescued (closer-to-valid) and the mean (cos_to_valid - cos_to_invalid) margin. Needs
    the query to have both a valid centroid (>=1 valid) and invalid darts.
    """
    if tab["valid_centroid"] is None or tab["n_invalid"] == 0:
        return {"frac_rescued": float("nan"), "mean_margin": float("nan"), "n": 0}
    inv = reps[tab["invalid_idx"]]
    blended = _blend(inv, centroid_vec[None, :], alpha)  # (n_inv, H)
    cos_v = _cos_to_vector(blended, tab["valid_centroid"])
    cos_i = _cos_to_vector(blended, tab["invalid_centroid"]) \
        if tab["invalid_centroid"] is not None else np.zeros(blended.shape[0])
    margin = cos_v - cos_i
    return {
        "frac_rescued": float(np.mean(margin > 0.0)),
        "mean_margin": float(np.mean(margin)),
        "n": int(blended.shape[0]),
    }


# ===========================================================================
# Retrieval: nearest library centroid (raw + centered geometries) + controls
# ===========================================================================

def _retrieve(query_vec, lib_uids, lib_centroids, query_uid, centered_basis_mean, rng):
    """Given a query vector and a library (uids + centroids, none belonging to the query),
    return:
        matched_uid, matched_centroid (RAW-cosine nearest), match_cosdist
        random_uid, random_centroid   (a uniformly random library centroid != query)
        global_centroid               (mean of all library centroids)
    in BOTH raw and centered geometry. `centered_basis_mean` is the mean over the (train)
    library centroids; centered matching subtracts it from both query and library centroids
    BEFORE the cosine (matches on the abstract good-pattern, graph-identity removed).
    The matched_centroid returned for BLENDING is always the RAW centroid (we blend in the
    native silhouette space); the geometry only changes WHICH library entry is selected.
    """
    lib_uids = np.asarray(lib_uids)
    lib_centroids = np.asarray(lib_centroids, dtype=np.float64)  # (L, H)
    L = lib_centroids.shape[0]

    # --- RAW geometry: cosine in native space ---
    sims_raw = _cos_to_vector(lib_centroids, query_vec)
    raw_best = int(np.argmax(sims_raw))
    raw_uid = int(lib_uids[raw_best])
    raw_centroid = lib_centroids[raw_best]
    raw_cosdist = float(1.0 - sims_raw[raw_best])

    # --- CENTERED geometry: subtract the library mean from both sides ---
    qc = np.asarray(query_vec, dtype=np.float64) - centered_basis_mean
    libc = lib_centroids - centered_basis_mean[None, :]
    sims_cen = _cos_to_vector(libc, qc)
    cen_best = int(np.argmax(sims_cen))
    cen_uid = int(lib_uids[cen_best])
    cen_centroid = lib_centroids[cen_best]  # blend the RAW centroid of the centered-selected entry
    cen_cosdist = float(1.0 - sims_cen[cen_best])

    # --- RANDOM control: a uniformly random library entry ---
    rnd = int(rng.randint(L))
    rnd_uid = int(lib_uids[rnd])
    rnd_centroid = lib_centroids[rnd]

    # --- GLOBAL good-centroid: mean of all library centroids ---
    global_centroid = lib_centroids.mean(axis=0)

    return {
        "raw": {"uid": raw_uid, "centroid": raw_centroid, "cosdist": raw_cosdist},
        "centered": {"uid": cen_uid, "centroid": cen_centroid, "cosdist": cen_cosdist},
        "random": {"uid": rnd_uid, "centroid": rnd_centroid},
        "global": {"centroid": global_centroid},
    }


# ===========================================================================
# Main probe driver
# ===========================================================================

def run_retrieval_probe(reps, valid, inst_id, band, *, min_valid=3, folds=5,
                        alphas=ALPHAS_DEFAULT, seed=0, geometry_for_log="raw",
                        rescue_alpha=0.5, verbose=True):
    """Run the full by-instance retrieval-utility probe. Returns a result dict with the
    per-query rows, aggregate stats, the per-control AUC(alpha) curves, and the verdict.

    geometry_for_log : which retrieval geometry ('raw' or 'centered') the per-match LOG and
        the headline matched-vs-random comparison use (both are computed; the OTHER geometry
        is summarized in aggregate for the raw-vs-centered comparison)."""
    reps = np.asarray(reps, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    inst_id = np.asarray(inst_id)
    band = np.asarray(band, dtype=np.float64) if band is not None else None
    alphas = tuple(float(a) for a in alphas)

    tables = build_instance_tables(reps, valid, inst_id, band, min_valid=min_valid)
    uids = np.array(sorted(tables.keys()))
    fold_of, folds = assign_instance_folds(uids, folds, seed=seed)

    lib_eligible = [u for u in uids if tables[int(u)]["lib_eligible"]]
    query_eligible = [u for u in uids if tables[int(u)]["query_eligible"]]

    rng = np.random.RandomState(seed)

    # accumulate per-control AUC(alpha) sums (for the curve) and per-query rows.
    geoms = ["raw", "centered"]
    curve_sum = {g: {ctrl: np.zeros(len(alphas)) for ctrl in
                     ("matched", "random", "global")} for g in geoms}
    curve_cnt = {g: 0 for g in geoms}
    rows = []
    # aggregate help stats per geometry/control.
    agg = {g: {ctrl: {"improvements": [], "opt_alphas": [], "auc_opt": [], "auc0": []}
               for ctrl in ("matched", "random", "global")} for g in geoms}
    rescue_agg = {g: {"matched_frac": [], "random_frac": [],
                      "matched_margin": [], "random_margin": []} for g in geoms}

    n_skipped_no_lib = 0
    for qu in query_eligible:
        qu = int(qu)
        tab = tables[qu]
        qfold = fold_of[qu]
        # library = TRAIN-fold lib-eligible instances, EXCLUDING the query (no leak).
        lib_uids = [u for u in lib_eligible
                    if fold_of[int(u)] != qfold and int(u) != qu]
        if len(lib_uids) < 2:
            n_skipped_no_lib += 1
            continue
        lib_centroids = np.stack([tables[int(u)]["valid_centroid"] for u in lib_uids])
        centered_basis_mean = lib_centroids.mean(axis=0)

        ret = _retrieve(tab["query_mean"], lib_uids, lib_centroids, qu,
                        centered_basis_mean, rng)

        # global is geometry-independent; precompute its curve once.
        global_auc, _, _ = auc_curve_for_centroid(tab, reps, ret["global"]["centroid"], alphas)
        # random control: blend the SAME randomly-chosen entry across geometries (it is not
        # geometry-dependent — it is a random entry — so reuse it for both geoms).
        random_auc, _, _ = auc_curve_for_centroid(tab, reps, ret["random"]["centroid"], alphas)

        per_geom_log = {}
        for g in geoms:
            matched_centroid = ret[g]["centroid"]
            matched_auc, _, _ = auc_curve_for_centroid(tab, reps, matched_centroid, alphas)

            curve_sum[g]["matched"] += matched_auc
            curve_sum[g]["random"] += random_auc
            curve_sum[g]["global"] += global_auc
            curve_cnt[g] += 1

            # per-control optimal alpha + improvement over alpha=0 (the baseline).
            def _opt(auc_vec):
                k_best = int(np.argmax(auc_vec))
                return (float(alphas[k_best]), float(auc_vec[k_best]),
                        float(auc_vec[0]), float(auc_vec[k_best] - auc_vec[0]))

            m_alpha, m_aopt, m_a0, m_imp = _opt(matched_auc)
            r_alpha, r_aopt, r_a0, r_imp = _opt(random_auc)
            g_alpha, g_aopt, g_a0, g_imp = _opt(global_auc)

            agg[g]["matched"]["improvements"].append(m_imp)
            agg[g]["matched"]["opt_alphas"].append(m_alpha)
            agg[g]["matched"]["auc_opt"].append(m_aopt)
            agg[g]["matched"]["auc0"].append(m_a0)
            agg[g]["random"]["improvements"].append(r_imp)
            agg[g]["random"]["opt_alphas"].append(r_alpha)
            agg[g]["random"]["auc_opt"].append(r_aopt)
            agg[g]["random"]["auc0"].append(r_a0)
            agg[g]["global"]["improvements"].append(g_imp)
            agg[g]["global"]["opt_alphas"].append(g_alpha)
            agg[g]["global"]["auc_opt"].append(g_aopt)
            agg[g]["global"]["auc0"].append(g_a0)

            # stuck-rescue at the fixed rescue_alpha, for matched + random.
            sr_m = stuck_rescue_for_centroid(tab, reps, matched_centroid, rescue_alpha)
            sr_r = stuck_rescue_for_centroid(tab, reps, ret["random"]["centroid"], rescue_alpha)
            if np.isfinite(sr_m["frac_rescued"]):
                rescue_agg[g]["matched_frac"].append(sr_m["frac_rescued"])
                rescue_agg[g]["matched_margin"].append(sr_m["mean_margin"])
            if np.isfinite(sr_r["frac_rescued"]):
                rescue_agg[g]["random_frac"].append(sr_r["frac_rescued"])
                rescue_agg[g]["random_margin"].append(sr_r["mean_margin"])

            matched_minus_random = m_imp - r_imp
            helped = bool(m_imp > 0.0 and matched_minus_random > 1e-3)
            matched_uid = ret[g]["uid"]
            same_band = bool(tab["band"] == tables[matched_uid]["band"])
            per_geom_log[g] = {
                "matched_uid": matched_uid,
                "matched_band": tables[matched_uid]["band"],
                "same_band": same_band,
                "match_cosdist": ret[g]["cosdist"],
                "opt_alpha_matched": m_alpha,
                "auc_opt_matched": m_aopt,
                "auc0": m_a0,
                "improvement_matched": m_imp,
                "opt_alpha_random": r_alpha,
                "improvement_random": r_imp,
                "matched_minus_random": matched_minus_random,
                "improvement_global": g_imp,
                "helped": helped,
                "stuck_rescue_matched_frac": sr_m["frac_rescued"],
                "stuck_rescue_random_frac": sr_r["frac_rescued"],
            }

        # the LOG row uses geometry_for_log as the headline geometry; both stored in json.
        gl = per_geom_log[geometry_for_log]
        row = {
            "query_inst_id": qu,
            "query_band": tab["band"],
            "query_n_valid": tab["n_valid"],
            "query_n_invalid": tab["n_invalid"],
            "query_fold": qfold,
            "lib_size": len(lib_uids),
            "geometry": geometry_for_log,
            "matched_lib_inst_id": gl["matched_uid"],
            "matched_band": gl["matched_band"],
            "same_band": gl["same_band"],
            "match_cosine_distance": gl["match_cosdist"],
            "optimal_alpha_matched": gl["opt_alpha_matched"],
            "auc_at_optimal_matched": gl["auc_opt_matched"],
            "auc_at_alpha0": gl["auc0"],
            "improvement_matched": gl["improvement_matched"],
            "optimal_alpha_random": gl["opt_alpha_random"],
            "improvement_random": gl["improvement_random"],
            "improvement_global": gl["improvement_global"],
            "matched_minus_random": gl["matched_minus_random"],
            "helped": gl["helped"],
            "stuck_rescue_matched_frac": gl["stuck_rescue_matched_frac"],
            "stuck_rescue_random_frac": gl["stuck_rescue_random_frac"],
            # the OTHER geometry, for the raw-vs-centered comparison in the json.
            "other_geometry": ("centered" if geometry_for_log == "raw" else "raw"),
            "other_matched_lib_inst_id": per_geom_log[
                "centered" if geometry_for_log == "raw" else "raw"]["matched_uid"],
            "other_improvement_matched": per_geom_log[
                "centered" if geometry_for_log == "raw" else "raw"]["improvement_matched"],
            "other_matched_minus_random": per_geom_log[
                "centered" if geometry_for_log == "raw" else "raw"]["matched_minus_random"],
        }
        rows.append(row)

    # build the per-control AUC(alpha) curves (mean over queries) per geometry.
    curves = {}
    for g in geoms:
        c = max(1, curve_cnt[g])
        curves[g] = {ctrl: (curve_sum[g][ctrl] / c).tolist()
                     for ctrl in ("matched", "random", "global")}

    aggregate = _aggregate(agg, rescue_agg, rows, alphas, geometry_for_log, curve_cnt)
    V = _verdict(aggregate, geometry_for_log)

    result = {
        "alphas": list(alphas),
        "min_valid": min_valid,
        "folds": folds,
        "seed": seed,
        "geometry_for_log": geometry_for_log,
        "rescue_alpha": rescue_alpha,
        "n_instances_total": int(uids.shape[0]),
        "n_lib_eligible": len(lib_eligible),
        "n_query_eligible": len(query_eligible),
        "n_queries_scored": len(rows),
        "n_skipped_no_lib": n_skipped_no_lib,
        "curves": curves,
        "aggregate": aggregate,
        "verdict": V,
        "rows": rows,
        "reused_dart_cluster_helpers": _REUSED_HELPERS,
    }
    if verbose:
        _print_report(result)
    return result


def _safe_mean(x):
    x = [v for v in x if v is not None and np.isfinite(v)]
    return float(np.mean(x)) if x else float("nan")


def _safe_median(x):
    x = [v for v in x if v is not None and np.isfinite(v)]
    return float(np.median(x)) if x else float("nan")


def _aggregate(agg, rescue_agg, rows, alphas, geom, curve_cnt):
    """Aggregate help stats. The headline geometry (geom) drives the verdict; the other is
    summarized for the raw-vs-centered comparison."""
    out = {"per_geometry": {}}
    for g in agg:
        m = agg[g]["matched"]
        r = agg[g]["random"]
        gl = agg[g]["global"]
        frac_helped_matched = _safe_mean([1.0 if v > 0 else 0.0 for v in m["improvements"]])
        frac_helped_random = _safe_mean([1.0 if v > 0 else 0.0 for v in r["improvements"]])
        # matched-minus-random per query (paired).
        mmr = [a - b for a, b in zip(m["improvements"], r["improvements"])]
        opt_alphas = m["opt_alphas"]
        # opt-alpha histogram over the discrete grid.
        hist = {f"{a:.2f}": int(sum(1 for v in opt_alphas if abs(v - a) < 1e-9))
                for a in alphas}
        out["per_geometry"][g] = {
            "n": len(m["improvements"]),
            "frac_helped_matched": frac_helped_matched,
            "frac_helped_random": frac_helped_random,
            "mean_improvement_matched": _safe_mean(m["improvements"]),
            "mean_improvement_random": _safe_mean(r["improvements"]),
            "mean_improvement_global": _safe_mean(gl["improvements"]),
            "mean_matched_minus_random": _safe_mean(mmr),
            "median_matched_minus_random": _safe_median(mmr),
            "frac_matched_beats_random": _safe_mean(
                [1.0 if v > 1e-3 else 0.0 for v in mmr]),
            "mean_auc0": _safe_mean(m["auc0"]),
            "mean_auc_opt_matched": _safe_mean(m["auc_opt"]),
            "mean_auc_opt_random": _safe_mean(r["auc_opt"]),
            "mean_auc_opt_global": _safe_mean(gl["auc_opt"]),
            "mean_opt_alpha_matched": _safe_mean(opt_alphas),
            "median_opt_alpha_matched": _safe_median(opt_alphas),
            "opt_alpha_hist": hist,
            "frac_opt_alpha_gt0": _safe_mean([1.0 if v > 0 else 0.0 for v in opt_alphas]),
            "stuck_rescue_matched_frac": _safe_mean(rescue_agg[g]["matched_frac"]),
            "stuck_rescue_random_frac": _safe_mean(rescue_agg[g]["random_frac"]),
            "stuck_rescue_matched_margin": _safe_mean(rescue_agg[g]["matched_margin"]),
            "stuck_rescue_random_margin": _safe_mean(rescue_agg[g]["random_margin"]),
        }

    # attribute correlations on the headline-geometry rows.
    same = [r for r in rows if r["same_band"]]
    cross = [r for r in rows if not r["same_band"]]
    helped_same = _safe_mean([1.0 if r["helped"] else 0.0 for r in same])
    helped_cross = _safe_mean([1.0 if r["helped"] else 0.0 for r in cross])
    # correlation of match-distance with matched improvement (closer match -> more help?).
    md = np.array([r["match_cosine_distance"] for r in rows], dtype=np.float64)
    imp = np.array([r["improvement_matched"] for r in rows], dtype=np.float64)
    fin = np.isfinite(md) & np.isfinite(imp)
    if fin.sum() >= 3 and np.std(md[fin]) > 1e-12 and np.std(imp[fin]) > 1e-12:
        corr_dist_imp = float(np.corrcoef(md[fin], imp[fin])[0, 1])
    else:
        corr_dist_imp = float("nan")
    out["attributes"] = {
        "geometry": geom,
        "n_same_band": len(same),
        "n_cross_band": len(cross),
        "help_rate_same_band": helped_same,
        "help_rate_cross_band": helped_cross,
        "corr_matchdist_vs_improvement": corr_dist_imp,
        "frac_same_band_matches": _safe_mean(
            [1.0 if r["same_band"] else 0.0 for r in rows]),
    }
    out["headline_geometry"] = geom
    return out


# ===========================================================================
# VERDICT
# ===========================================================================

def _verdict(aggregate, geom):
    """Thresholds (stated, not hidden), on the HEADLINE geometry:
      RETRIEVAL JUSTIFIED iff mean_matched_minus_random > MARGIN (0.02) AND
        frac_matched_beats_random > 0.55 AND a consistent alpha>0 helps
        (frac_opt_alpha_gt0 > 0.5 AND mean_improvement_matched > 0).
      GLOBAL-PRIOR-SUFFICES iff matched ~= random (|mmr| <= MARGIN) BUT a good centroid
        still helps (mean_improvement_global > 0 OR mean_improvement_matched > 0 with
        alpha>0) -> any good centroid blends the same; a single global good-prior is enough,
        no per-query matching/retrieval needed (simpler).
      NOT-USEFUL iff no alpha>0 helps (frac_opt_alpha_gt0 <= 0.5 AND
        mean_improvement_matched <= 0) -> the centroid does not point the right direction.
    """
    MARGIN = 0.02
    g = aggregate["per_geometry"][geom]
    mmr = g["mean_matched_minus_random"]
    fbeats = g["frac_matched_beats_random"]
    imp_m = g["mean_improvement_matched"]
    imp_g = g["mean_improvement_global"]
    frac_a = g["frac_opt_alpha_gt0"]

    def _f(x):
        return isinstance(x, float) and np.isfinite(x)

    alpha_helps = _f(frac_a) and frac_a > 0.5 and _f(imp_m) and imp_m > 0.0
    matched_beats = (_f(mmr) and mmr > MARGIN and _f(fbeats) and fbeats > 0.55)
    no_alpha_helps = (_f(frac_a) and frac_a <= 0.5 and _f(imp_m) and imp_m <= 0.0)
    global_helps = (_f(imp_g) and imp_g > 0.0) or alpha_helps

    lines = []
    if no_alpha_helps:
        label = "NOT_USEFUL"
        lines.append(
            f"RETRIEVAL NOT USEFUL ({geom}): no alpha>0 consistently helps "
            f"(frac_opt_alpha>0={frac_a:.2f}<=0.5, mean matched improvement={imp_m:+.3f}<=0). "
            f"Blending toward a good centroid does NOT point the query toward its own valid "
            f"region in this silhouette space — the centroid does not carry the right "
            f"direction. The two-path retrieval is NOT supported here.")
    elif matched_beats and alpha_helps:
        label = "RETRIEVAL_JUSTIFIED"
        lines.append(
            f"RETRIEVAL JUSTIFIED ({geom}): MATCHED beats RANDOM by a clear margin "
            f"(mean matched-minus-random={mmr:+.3f}>{MARGIN}, frac queries where matched "
            f"beats random={fbeats:.2f}>0.55) AND a consistent alpha>0 helps "
            f"(frac_opt_alpha>0={frac_a:.2f}, mean matched improvement={imp_m:+.3f}>0). "
            f"The MATCHING — not just any good centroid — carries query-specific solution "
            f"signal: the two-path retrieve-and-inject idea is worth building.")
    else:
        # matched ~= random (or beats only weakly) but a good centroid still helps.
        label = "GLOBAL_PRIOR_SUFFICES"
        lines.append(
            f"GLOBAL PRIOR SUFFICES ({geom}): blending toward a good centroid helps "
            f"(mean matched improvement={imp_m:+.3f}, mean global improvement={imp_g:+.3f}, "
            f"frac_opt_alpha>0={frac_a:.2f}) BUT matched does NOT clearly beat random "
            f"(mean matched-minus-random={mmr:+.3f}<={MARGIN}, frac beats random="
            f"{fbeats:.2f}). Any good centroid blends the same -> the per-query MATCHING/"
            f"RETRIEVAL adds little; a single GLOBAL good-prior (cheaper, no library/match) "
            f"would capture the benefit. Two-path retrieval NOT justified over a global prior.")
    lines.append(
        f"(matched mean optimal-alpha={g['mean_opt_alpha_matched']:.2f} "
        f"median={g['median_opt_alpha_matched']:.2f}; baseline AUC(alpha=0)="
        f"{g['mean_auc0']:.3f} -> matched-opt AUC={g['mean_auc_opt_matched']:.3f}, "
        f"random-opt AUC={g['mean_auc_opt_random']:.3f}, global-opt AUC="
        f"{g['mean_auc_opt_global']:.3f}.)")
    return {"label": label, "geometry": geom, "lines": lines,
            "margin_threshold": MARGIN,
            "matched_beats_random": bool(matched_beats),
            "alpha_helps": bool(alpha_helps),
            "no_alpha_helps": bool(no_alpha_helps),
            "global_helps": bool(global_helps)}


# ===========================================================================
# Reporting + caveats
# ===========================================================================

def _print_curve(name, alphas, curve):
    head = "  alpha:    " + "  ".join(f"{a:>5.2f}" for a in alphas)
    print(f"      {name}", flush=True)
    print(f"      {head}", flush=True)
    for ctrl in ("matched", "random", "global"):
        vals = "  ".join(f"{v:>5.3f}" for v in curve[ctrl])
        print(f"        {ctrl:>8}: {vals}", flush=True)


def _print_report(res):
    print(f"\n{'='*78}", flush=True)
    print(f"  RETRIEVAL UTILITY PROBE — cross-instance good-pattern matching", flush=True)
    print(f"{'='*78}", flush=True)
    print(f"  instances total={res['n_instances_total']}  "
          f"library-eligible(>= {res['min_valid']} valid)={res['n_lib_eligible']}  "
          f"query-eligible(>=1 valid & >=1 invalid)={res['n_query_eligible']}", flush=True)
    print(f"  queries scored={res['n_queries_scored']}  "
          f"skipped(no library in train fold)={res['n_skipped_no_lib']}  "
          f"folds={res['folds']} (BY-INSTANCE, no query-self leak)  seed={res['seed']}",
          flush=True)
    print(f"  reused dart_cluster_probe helpers: {res['reused_dart_cluster_helpers']}",
          flush=True)
    print(f"  alphas={res['alphas']}  rescue_alpha={res['rescue_alpha']}", flush=True)

    print(f"\n  [AUC(alpha) curves — mean over queries; "
          f"AUC(own-valid vs own-invalid) of blended point]", flush=True)
    for g in ("raw", "centered"):
        _print_curve(f"GEOMETRY={g}", res["alphas"], res["curves"][g])

    print(f"\n  [AGGREGATE help — matched vs random vs global, per geometry]", flush=True)
    for g in ("raw", "centered"):
        a = res["aggregate"]["per_geometry"][g]
        print(f"    GEOMETRY={g} (n={a['n']}):", flush=True)
        print(f"      frac helped:   matched={a['frac_helped_matched']:.3f}  "
              f"random={a['frac_helped_random']:.3f}", flush=True)
        print(f"      mean improve:  matched={a['mean_improvement_matched']:+.4f}  "
              f"random={a['mean_improvement_random']:+.4f}  "
              f"global={a['mean_improvement_global']:+.4f}", flush=True)
        print(f"      MATCHED-minus-RANDOM (retrieval value): "
              f"mean={a['mean_matched_minus_random']:+.4f}  "
              f"median={a['median_matched_minus_random']:+.4f}  "
              f"frac_matched>random={a['frac_matched_beats_random']:.3f}", flush=True)
        print(f"      AUC: baseline(a=0)={a['mean_auc0']:.3f}  "
              f"matched-opt={a['mean_auc_opt_matched']:.3f}  "
              f"random-opt={a['mean_auc_opt_random']:.3f}  "
              f"global-opt={a['mean_auc_opt_global']:.3f}", flush=True)
        print(f"      optimal-alpha: mean={a['mean_opt_alpha_matched']:.3f}  "
              f"median={a['median_opt_alpha_matched']:.3f}  "
              f"frac>0={a['frac_opt_alpha_gt0']:.3f}  hist={a['opt_alpha_hist']}", flush=True)
        print(f"      STUCK-RESCUE (alpha={res['rescue_alpha']}; frac invalid darts moved "
              f"closer to own-valid than own-invalid centroid):", flush=True)
        print(f"        matched frac={a['stuck_rescue_matched_frac']:.3f} "
              f"(margin={a['stuck_rescue_matched_margin']:+.4f})  "
              f"random frac={a['stuck_rescue_random_frac']:.3f} "
              f"(margin={a['stuck_rescue_random_margin']:+.4f})", flush=True)

    att = res["aggregate"]["attributes"]
    print(f"\n  [ATTRIBUTE drivers — headline geometry={att['geometry']}]", flush=True)
    print(f"      help-rate same-band={att['help_rate_same_band']:.3f} "
          f"(n={att['n_same_band']})  cross-band={att['help_rate_cross_band']:.3f} "
          f"(n={att['n_cross_band']})", flush=True)
    print(f"      frac matches that are same-band={att['frac_same_band_matches']:.3f}",
          flush=True)
    print(f"      corr(match-distance, matched improvement)="
          f"{att['corr_matchdist_vs_improvement']:+.3f}  "
          f"(negative => closer match helps more)", flush=True)

    print(f"\n  [VERDICT] (headline geometry={res['verdict']['geometry']}; "
          f"label={res['verdict']['label']})", flush=True)
    for ln in res["verdict"]["lines"]:
        print(f"      - {ln}", flush=True)

    _print_caveats()


def _print_caveats():
    print(f"\n  [HONEST CAVEATS — read before acting on the verdict]", flush=True)
    print(f"      (1) FROZEN-SILHOUETTE PROXY: this blends in the CAPTURED (baseline, "
          f"final-breath) silhouette space — a proxy/LOWER BOUND for the real in-deducer "
          f"injection, which would RE-RUN the deduction with the blend (changing the whole "
          f"trajectory). A positive here is SUGGESTIVE, not proof; a null is fairly damning.",
          flush=True)
    print(f"      (2) FINAL-BREATH QUERY: the realistic query is an EARLY-breath state "
          f"(retrieve early to guide the REST of the breathing). Here the query is the "
          f"final-breath mean. A follow-up early-breath capture would test the real regime.",
          flush=True)
    print(f"      (3) BASELINE-MODEL REPS: these are from the baseline deducer ckpt; an "
          f"energy/contrastive ckpt may cluster good-patterns more sharply (better matches).",
          flush=True)
    print(f"      (4) WEAK TESTBED: RANDOM coloring graphs share LITTLE structure across "
          f"instances, so retrieval has little to reuse — a conservative testbed. Structured/"
          f"recurring domains (the MATH-500 north-star) would favor case-based retrieval more.",
          flush=True)


# ===========================================================================
# Per-match log writers (csv + json)
# ===========================================================================

_CSV_COLUMNS = [
    "query_inst_id", "query_band", "query_n_valid", "query_n_invalid", "query_fold",
    "lib_size", "geometry",
    "matched_lib_inst_id", "matched_band", "same_band", "match_cosine_distance",
    "optimal_alpha_matched", "auc_at_optimal_matched", "auc_at_alpha0",
    "improvement_matched", "optimal_alpha_random", "improvement_random",
    "improvement_global", "matched_minus_random", "helped",
    "stuck_rescue_matched_frac", "stuck_rescue_random_frac",
    "other_geometry", "other_matched_lib_inst_id", "other_improvement_matched",
    "other_matched_minus_random",
]


def write_logs(result, out_prefix):
    """Write the per-match CSV + the full JSON (rows + aggregate + verdict + curves)."""
    csv_path = out_prefix + "_permatch.csv"
    json_path = out_prefix + ".json"
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)) or ".", exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_CSV_COLUMNS)
        w.writeheader()
        for r in result["rows"]:
            w.writerow({k: r.get(k, "") for k in _CSV_COLUMNS})
    payload = {k: v for k, v in result.items() if k != "rows"}
    payload["rows"] = result["rows"]
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2, default=_json_default)
    print(f"\n  [WROTE] per-match CSV : {os.path.abspath(csv_path)} "
          f"({len(result['rows'])} rows)", flush=True)
    print(f"  [WROTE] full JSON    : {os.path.abspath(json_path)}", flush=True)
    return csv_path, json_path


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


# ===========================================================================
# npz loader + run
# ===========================================================================

def run_from_npz(npz_path, *, min_valid=3, folds=5, seed=0, geometry_for_log="raw",
                 rescue_alpha=0.5, out_prefix=None):
    z = np.load(npz_path, allow_pickle=True)
    reps = np.asarray(z["reps"], dtype=np.float64)
    valid = np.asarray(z["valid"], dtype=bool)
    inst_id = np.asarray(z["inst_id"])
    band = np.asarray(z["band"], dtype=np.float64) if "band" in z else None
    meta = z["meta"].item() if "meta" in z else {}

    n, H = reps.shape
    print(f"\n  LOADED {npz_path}", flush=True)
    print(f"  n_darts={n}  H={H}  VALID={int(valid.sum())}  INVALID={int((~valid).sum())}  "
          f"instances={int(np.unique(inst_id).shape[0])}", flush=True)
    if meta:
        print(f"  meta: ckpt={meta.get('ckpt','?')} domain={meta.get('domain','?')} "
              f"K={meta.get('K','?')} m_max={meta.get('m_max','?')} "
              f"bands={meta.get('bands','?')}", flush=True)
        print(f"  rep: {meta.get('rep','?')}", flush=True)

    res = run_retrieval_probe(reps, valid, inst_id, band, min_valid=min_valid,
                              folds=folds, seed=seed, geometry_for_log=geometry_for_log,
                              rescue_alpha=rescue_alpha, verbose=True)
    res["meta"] = meta
    res["npz_path"] = npz_path
    if out_prefix:
        write_logs(res, out_prefix)
    return res


# ===========================================================================
# SELFTEST — planted-matched (detects) + null (no false positive)
# ===========================================================================

def _make_synthetic_retrieval(n_inst, m, H, planted, seed=0, p_valid=0.4):
    """Synthetic per-dart bank for the RETRIEVAL probe.

    THE MECHANISM WE MUST EXERCISE. The probe's help metric is: does blending the QUERY mean
    toward a centroid RAISE AUC(own-valid vs own-invalid)? For there to be HEADROOM, the
    query-mean baseline (alpha=0) must NOT already separate own-valid from own-invalid. We
    arrange that by making the per-instance graph-IDENTITY DOMINATE the query mean so the
    discriminating GOOD direction is washed out at alpha=0 — then blending in a centroid that
    carries a CLEAN, identity-free good direction injects the missing signal.

    planted=True : the good direction is SHARED across instances in the same 'family'. The
        library valid-centroids of same-family instances therefore carry that family's
        good direction (their own identities average toward ~0 across many darts but each
        single centroid still has its own identity; what makes a centroid a good MATCH for the
        query is that the query's identity correlates with same-family library identities AND
        the good direction agrees). Matched (same-family-ish) centroid -> injects the RIGHT
        good direction -> AUC rises; a RANDOM centroid injects a wrong/other good direction
        + foreign identity -> little/no rise. -> matched >> random, optimal alpha > 0,
        verdict RETRIEVAL_JUSTIFIED.
    planted=False : NULL — each instance's good direction is INDEPENDENT (no shared family),
        so no library centroid's good direction agrees with the query's own valid region. The
        nearest centroid by cosine matches on IDENTITY noise, carrying an unrelated good
        direction. -> matched ~= random, no alpha>0 consistently helps.

    Construction: identity has a per-FAMILY shared component (so same-family instances are
    cosine-near -> retrievable) plus per-instance noise. Valid darts = identity + good_dir +
    noise; invalid darts = identity + isotropic noise. Identity magnitude >> good_dir so the
    query mean does NOT trivially separate (headroom), but the good_dir is consistent enough
    that a clean centroid injection recovers it."""
    rng = np.random.RandomState(seed)
    n_families = 8
    # family good directions (the transferable, retrievable signal). These are STRONG +
    # CLEAN in a library centroid (averaged over many valid darts -> noise cancels) but
    # DILUTED in the query mean (few valid darts among many invalid -> washed out). That
    # asymmetry is exactly the headroom the help metric needs. n_families high so a RANDOM
    # library pick is rarely same-family (matched must genuinely beat random).
    fam_good = rng.randn(n_families, H)
    fam_good /= np.linalg.norm(fam_good, axis=1, keepdims=True)
    # family identity anchors (so same-family instances are cosine-near -> a real match).
    # SMALL relative to the good direction: identity is enough to retrieve the right family
    # but it is NOT what discriminates own-valid from own-invalid (both share identity).
    fam_identity = rng.randn(n_families, H)
    fam_identity /= np.linalg.norm(fam_identity, axis=1, keepdims=True)
    reps, valid, inst_id, band = [], [], [], []
    # GEOMETRY OF THE HELP SIGNAL:
    #   * GOOD direction is the DISCRIMINATOR (valid darts have it, invalid do not). A library
    #     centroid AVERAGES it clean -> blending the matched centroid injects a strong, clean
    #     good direction -> raises own-valid-vs-invalid AUC.
    #   * identity is a SMALL shared offset (drives the cosine MATCH to the right family but
    #     does NOT discriminate own valid/invalid; blending it alone does nothing).
    #   * heavy per-dart noise buries a SINGLE dart's good direction AND the query mean's (few
    #     valid darts among many invalid) -> LOW baseline AUC -> real headroom for the blend.
    ID_MAG = 0.7            # shared family identity offset magnitude (match signal)
    ID_NOISE = 0.3          # per-instance identity jitter
    GOOD_MAG = 3.0          # the discriminating good-direction magnitude in a valid dart
    DART_NOISE = 2.2        # per-dart noise (buries single-dart + query-mean good_dir)
    # moderate p_valid: enough valid darts per instance to average a CLEAN centroid good
    # direction, while the query mean (all darts) under-separates at baseline.
    p_valid = min(max(p_valid, 0.3), 0.45)
    for g in range(n_inst):
        fam = g % n_families
        identity = ID_MAG * fam_identity[fam] + ID_NOISE * rng.randn(H)
        if planted:
            good_dir = fam_good[fam]           # SHARED within family -> retrievable match
        else:
            good_dir = rng.randn(H)
            good_dir /= np.linalg.norm(good_dir)
        b = float([1.0, 1.5, 2.0, 2.5][g % 4])
        n_valid = max(3, int(round(m * p_valid)))
        for j in range(m):
            is_valid = j < n_valid
            if is_valid:
                vec = identity + GOOD_MAG * good_dir + DART_NOISE * rng.randn(H)
            else:
                vec = identity + DART_NOISE * rng.randn(H)
            reps.append(vec.astype(np.float32))
            valid.append(is_valid)
            inst_id.append(g)
            band.append(b)
    return (np.asarray(reps, dtype=np.float64), np.asarray(valid, dtype=bool),
            np.asarray(inst_id, dtype=np.int64), np.asarray(band, dtype=np.float64))


def selftest() -> bool:
    print("=== retrieval_utility_probe SELFTEST (CPU) ===", flush=True)
    ok = True

    # --- AUC machinery sanity (reused helper). ---
    s = np.array([0.1, 0.2, 0.3, 0.9, 1.0, 1.1])
    y = np.array([False, False, False, True, True, True])
    auc_sep = auc_mann_whitney(s, y)
    auc_flat = auc_mann_whitney(np.ones(6), y)
    print(f"  [auc] separable={auc_sep:.3f} (expect 1.0)  flat={auc_flat:.3f} (expect 0.5)  "
          f"reused_helpers={_REUSED_HELPERS}", flush=True)
    ok &= abs(auc_sep - 1.0) < 1e-9 and abs(auc_flat - 0.5) < 1e-9

    H = 64
    # --- PLANTED: matched >> random; consistent alpha>0; verdict RETRIEVAL_JUSTIFIED. ---
    reps, valid, inst, band = _make_synthetic_retrieval(80, 40, H, planted=True, seed=1)
    res_p = run_retrieval_probe(reps, valid, inst, band, min_valid=3, folds=5, seed=1,
                                geometry_for_log="raw", verbose=False)
    gp = res_p["aggregate"]["per_geometry"]["raw"]
    print(f"\n  [planted] matched_imp={gp['mean_improvement_matched']:+.3f} "
          f"random_imp={gp['mean_improvement_random']:+.3f} "
          f"mmr={gp['mean_matched_minus_random']:+.3f} "
          f"frac_matched>random={gp['frac_matched_beats_random']:.3f} "
          f"frac_alpha>0={gp['frac_opt_alpha_gt0']:.3f} "
          f"verdict={res_p['verdict']['label']}", flush=True)
    cond_planted = (gp["mean_matched_minus_random"] > 0.02
                    and gp["frac_matched_beats_random"] > 0.55
                    and gp["frac_opt_alpha_gt0"] > 0.5
                    and gp["mean_improvement_matched"] > 0.0
                    and res_p["verdict"]["label"] == "RETRIEVAL_JUSTIFIED")
    print(f"  [planted] EXPECT matched>>random + alpha>0 + JUSTIFIED -> "
          f"{'PASS' if cond_planted else 'FAIL'}", flush=True)
    ok &= cond_planted

    # --- NULL: matched ~= random; verdict NOT RETRIEVAL_JUSTIFIED. ---
    reps0, valid0, inst0, band0 = _make_synthetic_retrieval(80, 40, H, planted=False, seed=2)
    res_n = run_retrieval_probe(reps0, valid0, inst0, band0, min_valid=3, folds=5, seed=2,
                                geometry_for_log="raw", verbose=False)
    gn = res_n["aggregate"]["per_geometry"]["raw"]
    print(f"\n  [null] matched_imp={gn['mean_improvement_matched']:+.3f} "
          f"random_imp={gn['mean_improvement_random']:+.3f} "
          f"mmr={gn['mean_matched_minus_random']:+.3f} "
          f"frac_matched>random={gn['frac_matched_beats_random']:.3f} "
          f"verdict={res_n['verdict']['label']}", flush=True)
    # the decisive null assertion: matching adds nothing -> matched does NOT clearly beat
    # random, and the verdict is NOT 'RETRIEVAL_JUSTIFIED'.
    cond_null = (gn["mean_matched_minus_random"] <= 0.02
                 and res_n["verdict"]["label"] != "RETRIEVAL_JUSTIFIED")
    print(f"  [null] EXPECT matched~=random (mmr<=0.02) + NOT JUSTIFIED -> "
          f"{'PASS' if cond_null else 'FAIL'}", flush=True)
    ok &= cond_null

    # --- no-leak structural check: query never sees its own centroid. We verify that the
    # matched library inst id is NEVER the query id in either run. ---
    leaked = any(r["matched_lib_inst_id"] == r["query_inst_id"]
                 for r in res_p["rows"] + res_n["rows"])
    print(f"\n  [no-leak] any query matched its OWN centroid? {leaked} (expect False)",
          flush=True)
    ok &= (leaked is False)

    # --- csv/json round-trip on the planted run. ---
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        prefix = os.path.join(td, "syn_retrieval")
        csv_path, json_path = write_logs(res_p, prefix)
        ok &= os.path.exists(csv_path) and os.path.exists(json_path)
        with open(json_path) as f:
            loaded = json.load(f)
        ok &= ("verdict" in loaded and "rows" in loaded
               and len(loaded["rows"]) == len(res_p["rows"]))
        # csv header has all required per-match columns.
        with open(csv_path) as f:
            header = f.readline().strip().split(",")
        required = {"query_inst_id", "matched_lib_inst_id", "same_band",
                    "match_cosine_distance", "optimal_alpha_matched",
                    "improvement_matched", "improvement_random", "matched_minus_random",
                    "helped"}
        missing = required - set(header)
        print(f"  [io] csv+json round-trip; missing required cols={missing} (expect empty)",
              flush=True)
        ok &= (len(missing) == 0)

    # --- exercise the full verbose report once (prints curves + verdict + caveats). ---
    print("\n  [report] full verbose report on the PLANTED run:", flush=True)
    _print_report(res_p)

    print(f"\n  SELFTEST {'PASSED' if ok else 'FAILED'}", flush=True)
    return ok


# ===========================================================================
# Main
# ===========================================================================

def main(argv=None) -> int:
    P = argparse.ArgumentParser(
        description="Cross-instance retrieval-utility probe on captured dart silhouettes")
    P.add_argument("--npz", default=os.environ.get("DART_NPZ", None),
                   help="path to the per-dart silhouette .npz (from --capture-darts)")
    P.add_argument("--selftest", action="store_true",
                   help="run the CPU selftest (planted-matched + null) and exit")
    P.add_argument("--min-valid", type=int, default=3,
                   help="min VALID darts for an instance to enter the library (default 3)")
    P.add_argument("--folds", type=int, default=5,
                   help="k-fold over INSTANCES (by-instance no-leak split; default 5)")
    P.add_argument("--seed", type=int, default=0, help="fold/random-control seed")
    P.add_argument("--geometry", default="raw", choices=["raw", "centered"],
                   help="headline retrieval geometry for the log + verdict (both are "
                        "computed; default raw = case-based instance-similarity match)")
    P.add_argument("--rescue-alpha", type=float, default=0.5,
                   help="alpha for the stuck-rescue variant (default 0.5)")
    P.add_argument("--out-prefix", default=None,
                   help="write <prefix>_permatch.csv + <prefix>.json (per-match log)")
    args = P.parse_args(argv)

    if args.selftest or os.environ.get("SELFTEST_ONLY", "0") == "1":
        return 0 if selftest() else 1
    if not args.npz:
        print("error: pass --npz PATH (or --selftest). PATH is the per-dart silhouette npz "
              "from amortized_frontier_measure.py --capture-darts.", flush=True)
        return 2
    if not os.path.exists(args.npz):
        print(f"error: npz not found: {args.npz}", flush=True)
        return 2
    run_from_npz(args.npz, min_valid=args.min_valid, folds=args.folds, seed=args.seed,
                 geometry_for_log=args.geometry, rescue_alpha=args.rescue_alpha,
                 out_prefix=args.out_prefix)
    return 0


if __name__ == "__main__":
    parse_ok = _ast_parse_ok()
    print(f"[ast.parse] astparse_ok={parse_ok}", flush=True)
    if not parse_ok:
        sys.exit(1)
    sys.exit(main())
