#!/usr/bin/env python3
"""POST-HOC binding analyzer for KenKen Property 2 — the adaptive-depth telegraph.

This is the BINDING verdict for Property 2 (the trainer's per-eval test-set
property2 JSONL is only a cheap during-training monitor). It applies the FULL
control suite finalized in `memory/project_csp_target_survey_jun14.md`
("PROPERTY-2 SUCCESS CRITERION — FINALIZED Jun 14") — that pinned spec is
authoritative; this script implements it faithfully and does NOT improvise the bar.

WHAT PROPERTY 2 IS: does breath-count-to-convergence TRACK deduction-depth at
FIXED board size N? Harder instances should recruit more breaths; easy ones stop
early. That is "allocates compute to instance difficulty" — what fixed-iteration
recurrent-BP incumbents (RRN/NeuroSAT) do NOT claim.

TWO MODES
---------
MODE A (model-over-corpus): load a checkpoint into the KenKen breathing model
  (mirrors scripts/kenken_train.py model construction + load_ckpt), run the
  forward over the chosen corpus (DEFAULT = the TRAIN corpus ~340/N — the BINDING
  corpus, NOT the underpowered 60/N test split), call convergence_instrument to
  produce per-puzzle records, AND additionally capture n_givens and a per-puzzle
  breath_count_frac (the secondary convergence def: first breath where >=95% of
  valid cells have per-cell JSD<0.01). Writes the raw per-puzzle records to a JSONL.

  ** MODE A RUNS THE MODEL ON THE GPU. ** It is intentionally NOT exercised by the
  self-test (the GPU is busy with the live training run). A human runs MODE A later
  when the GPU is free. The numpy STATISTICS path (MODE B / the stats core) is
  self-tested CPU-only on synthetic data.

MODE B (analyze existing JSONL): skip the model entirely; read a property2 records
  JSONL (tolerates the trainer's schema {breath_count, deduction_depth, converged,
  correct, status, N, step, core} — `core`/`step`/`n_givens`/`breath_count_frac`
  are all optional). Computes the same statistics + verdict.

THE STATISTICS (load-bearing; pure-numpy implementations of Spearman, Kendall
tau-b, a within-bin permutation null, and a bootstrap CI — scipy is used ONLY as
an optional cross-check if importable):
  Per N-bin in {5,6,7}, on (a) SETTLED-ONLY (status==settled) PRIMARY and
  (b) CONVERGED-COMPANION (status in {settled,stuck}):
    - Spearman rho (PRIMARY) between breath_count and deduction_depth.
    - Kendall tau-b (tie-robust COMPANION; breath_count is a heavily-tied small int).
    - Permutation null p-value (>=10,000 within-bin shuffles of breath_count, one-sided).
    - Bootstrap 95% CI for rho (>=10,000 resamples); the LOWER bound gates the bar.
  Plus per bin:
    - frac_settled_strict = fraction of settled puzzles with breath_count < 20.
    - rho WITH and WITHOUT ceiling-hit (breath_count==20) puzzles; ceiling-EXCLUDED
      is primary if they disagree on the verdict (flagged).
    - settled / stuck / not_converged COUNTS + DEPTH DISTRIBUTIONS; not_converged
      FRACTION; a restriction-of-range flag when the settled depth-range is
      materially narrower than the full corpus range.
    - ROBUSTNESS companions (reported, NOT gating): rho(breath_count,n_givens),
      rho(breath_count, board-size-N pooled), and the secondary breath_count_frac
      correlation alongside the primary all-cells-AND breath_count.

THE FOUR-WAY READ (>=2 of 3 N-bins must AGREE — 2/3 MAJORITY, not any-of-3; lean on
N=6/N=7 because N=5 is tail-starved). A bin QUALIFIES only if settled-n>=50 AND
frac_settled_strict>=0.80:
  - HILL-STANDS  : lower-CI rho > 0.30 AND perm p < 0.01 in >=2/3 qualifying bins
                   (STRONG-WIN sub-tier if point rho >= 0.50).
  - WEAK         : p < 0.01 but lower-CI rho in [0.15, 0.30] in >=2/3 qualifying bins.
  - NULL         : p >= 0.05 (or lower-CI rho <= 0.15) in >=2/3 bins THAT HAVE POWER.
  - UNTESTABLE   : false-win guard trips (no breath_count spread w/ depth) OR
                   settled-n<50 OR frac_strict<0.80 in >=2/3 bins. Action:
                   regenerate harder/deeper corpus; do NOT declare null.
  The 0.30 bar is SPEARMAN-calibrated — applied to Spearman rho; tau-b is reported
  descriptively (NOT gated).

CLI:
  analyze_kenken_property2.py CKPT_PATH [--corpus PATH] [--jsonl PATH]
                              [--out PATH] [--records-out PATH] [--K N]
                              [--n-perm N] [--n-boot N] [--seed N]
  - CKPT_PATH : checkpoint to load (MODE A). Ignored if --jsonl is given.
  - --corpus  : corpus to run MODE A over (DEFAULT = .cache/kenken_train.jsonl,
                the binding TRAIN corpus — NOT the underpowered test split).
  - --jsonl   : analyze an EXISTING property2 records JSONL (MODE B); skips the model.
  - --out     : verdict JSON output path (default: <jsonl-or-ckpt-dir>/property2_verdict.json).
  - --records-out : MODE A raw per-puzzle records JSONL (default beside the verdict).

SELF-TEST:
  analyze_kenken_property2.py --selftest
  Runs the CPU-only synthetic + schema-parse self-tests (no GPU, no model).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

# ---------------------------------------------------------------------------
# Constants (pinned)
# ---------------------------------------------------------------------------
N_BINS = (5, 6, 7)                 # board sizes
K_CEILING_DEFAULT = 20             # KENKEN_K_MAX; breath_count==K means "hit ceiling"
QUALIFY_MIN_SETTLED_N = 50         # settled-n >= 50 to qualify a bin
QUALIFY_MIN_FRAC_STRICT = 0.80     # frac_settled_strict >= 0.80 to qualify a bin
HILL_RHO_BAR = 0.30                # SPEARMAN-calibrated lower-CI bar (HILL-STANDS)
WEAK_RHO_LO = 0.15                 # lower-CI rho band for WEAK
STRONG_RHO = 0.50                  # point-rho STRONG-WIN sub-tier
P_STRICT = 0.01                    # permutation p threshold for HILL / WEAK
P_NULL = 0.05                      # permutation p threshold for NULL
RESTRICTION_RANGE_FRAC = 0.6       # settled depth-range < 0.6x corpus range -> flag
DEFAULT_TRAIN_CORPUS = ".cache/kenken_train.jsonl"
N_PERM_DEFAULT = 10000
N_BOOT_DEFAULT = 10000

# ---- INSTRUMENT SELECTORS (the #238 U-curve min-based decision) ------------
# PRIMARY = min-based (gold-free argmin consecutive-belief-change). It uses
# breath_count_min + status_min, where status_min in {settled, stuck} (NO
# not_converged — an argmin always exists). The JSD-floor instrument
# (breath_count + status with not_converged) is the TAIL-CONTAMINATED SECONDARY.
# See memory/project_csp_target_survey_jun14.md, "THE U-CURVE IS
# ARCHITECTURE-GENERAL + MIN-BASED CONVERGENCE".
PRIMARY_BREATH_FIELD = "breath_count_min"
PRIMARY_STATUS_FIELD = "status_min"
SECONDARY_BREATH_FIELD = "breath_count"      # JSD-floor (tail-contaminated)
SECONDARY_STATUS_FIELD = "status"


def has_min_instrument(records: list[dict]) -> bool:
    """True iff the records carry the min-based fields (MODE A output / current
    convergence_instrument). MODE B over the LIVE trainer's JSON-floor-only
    records will NOT have these -> the analyzer falls back to the contaminated
    JSD-floor read and flags it loudly."""
    return bool(records) and all(
        (PRIMARY_BREATH_FIELD in r and PRIMARY_STATUS_FIELD in r) for r in records)


# ===========================================================================
# PURE-NUMPY STATISTICS (the load-bearing core; scipy-free by construction)
# ===========================================================================

def _rankdata(a: np.ndarray) -> np.ndarray:
    """Average-rank of a 1-D array (ties get the mean of the spanned ranks).

    Equivalent to scipy.stats.rankdata(a, method='average').
    """
    a = np.asarray(a, dtype=np.float64)
    n = a.size
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    sorted_a = a[order]
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        # ranks i..j (0-based) get the average of (i+1 .. j+1) (1-based)
        avg = (i + j) / 2.0 + 1.0
        ranks[order[i:j + 1]] = avg
        i = j + 1
    return ranks


def spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation = Pearson on average-ranks (tie-correct).

    Returns NaN if either variable has zero rank-variance (constant), which is
    the correct "undefined correlation" signal.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 3:
        return float("nan")
    rx = _rankdata(x)
    ry = _rankdata(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    if denom == 0.0:
        return float("nan")
    return float((rx * ry).sum() / denom)


def kendall_tau_b(x: np.ndarray, y: np.ndarray) -> float:
    """Kendall tau-b (tie-corrected). O(n^2) — fine for these bin sizes (<=~340).

    tau_b = (C - D) / sqrt((C+D+Tx) * (C+D+Ty))
      C = concordant pairs, D = discordant, Tx/Ty = pairs tied only in x / only in y.
    Returns NaN when the denominator is 0 (one variable constant).
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = x.size
    if n < 3:
        return float("nan")
    C = D = Tx = Ty = 0
    for i in range(n - 1):
        dx = x[i + 1:] - x[i]
        dy = y[i + 1:] - y[i]
        sx = np.sign(dx)
        sy = np.sign(dy)
        prod = sx * sy
        C += int(np.sum(prod > 0))
        D += int(np.sum(prod < 0))
        Tx += int(np.sum((sx == 0) & (sy != 0)))
        Ty += int(np.sum((sy == 0) & (sx != 0)))
    denom = np.sqrt(float(C + D + Tx) * float(C + D + Ty))
    if denom == 0.0:
        return float("nan")
    return float((C - D) / denom)


def permutation_p_spearman(x: np.ndarray, y: np.ndarray,
                           n_perm: int = N_PERM_DEFAULT,
                           rng: np.random.Generator | None = None) -> float:
    """One-sided permutation p-value for a POSITIVE Spearman rho.

    Shuffles `x` (breath_count) WITHIN the bin, recomputes rho on ranks, and
    counts how often the permuted rho >= the observed rho. One-sided because
    Property 2's directional hypothesis is "more depth -> more breaths" (positive).

    p = (1 + #{perm_rho >= obs_rho}) / (n_perm + 1)   [add-one; never reports 0].

    Returns 1.0 when rho is undefined (constant variable) — correctly non-significant.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    obs = spearman_rho(x, y)
    if not np.isfinite(obs):
        return 1.0
    if rng is None:
        rng = np.random.default_rng(0)
    ry = _rankdata(y)
    ry_c = ry - ry.mean()
    ry_ss = (ry_c * ry_c).sum()
    if ry_ss == 0.0:
        return 1.0
    rx = _rankdata(x)            # ranks of x; permutation of values == permutation of ranks
    rx_c = rx - rx.mean()
    rx_ss = (rx_c * rx_c).sum()
    if rx_ss == 0.0:
        return 1.0
    denom = np.sqrt(rx_ss * ry_ss)
    count = 0
    n = x.size
    for _ in range(n_perm):
        perm = rng.permutation(n)
        rho_p = float((rx_c[perm] * ry_c).sum() / denom)
        if rho_p >= obs - 1e-12:
            count += 1
    return (1 + count) / (n_perm + 1)


def bootstrap_ci_spearman(x: np.ndarray, y: np.ndarray,
                          n_boot: int = N_BOOT_DEFAULT,
                          ci: float = 0.95,
                          rng: np.random.Generator | None = None) -> tuple[float, float, float]:
    """Bootstrap percentile CI for Spearman rho via paired resampling-with-replacement.

    Returns (lower, upper, point_rho). NaN bootstrap replicates (a constant
    resample) are dropped before taking percentiles. If too few finite replicates
    survive, the CI bounds are NaN (correctly "undetermined").
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    point = spearman_rho(x, y)
    n = x.size
    if n < 3 or not np.isfinite(point):
        return (float("nan"), float("nan"), point)
    if rng is None:
        rng = np.random.default_rng(0)
    reps = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        reps[b] = spearman_rho(x[idx], y[idx])
    reps = reps[np.isfinite(reps)]
    if reps.size < max(20, n_boot // 100):
        return (float("nan"), float("nan"), point)
    lo_q = (1.0 - ci) / 2.0 * 100.0
    hi_q = (1.0 + ci) / 2.0 * 100.0
    lo = float(np.percentile(reps, lo_q))
    hi = float(np.percentile(reps, hi_q))
    return (lo, hi, point)


def _scipy_crosscheck(x: np.ndarray, y: np.ndarray) -> dict | None:
    """Optional scipy cross-check of the hand-rolled rho/tau (reported if importable)."""
    try:
        import warnings
        from scipy import stats  # noqa: WPS433 (optional)
    except Exception:
        return None
    with warnings.catch_warnings():
        # constant-input on degenerate (untrained) data is expected; rho is NaN.
        warnings.simplefilter("ignore")
        try:
            rho = float(stats.spearmanr(x, y).statistic)
        except Exception:
            rho = float("nan")
        try:
            tau = float(stats.kendalltau(x, y, variant="b").statistic)
        except Exception:
            tau = float("nan")
    return {"scipy_spearman": rho, "scipy_kendall_tau_b": tau}


# ===========================================================================
# PER-BIN ANALYSIS
# ===========================================================================

def _depth_hist(depths: np.ndarray) -> dict:
    if depths.size == 0:
        return {}
    vals, counts = np.unique(depths.astype(np.int64), return_counts=True)
    return {int(v): int(c) for v, c in zip(vals, counts)}


def analyze_subset(breath: np.ndarray, depth: np.ndarray,
                   k_ceiling: int, label: str,
                   n_perm: int, n_boot: int,
                   rng: np.random.Generator,
                   corpus_depth_range: tuple[int, int] | None = None) -> dict:
    """Compute the full statistic block for ONE subset (settled-only OR converged).

    Returns a dict with rho/tau/p/CI (full subset), the ceiling-excluded rho block,
    frac_settled_strict, restriction-of-range flag, and counts.
    """
    breath = np.asarray(breath, dtype=np.float64)
    depth = np.asarray(depth, dtype=np.float64)
    n = breath.size

    out: dict = {"label": label, "n": int(n)}
    if n == 0:
        out.update({
            "rho": float("nan"), "tau_b": float("nan"), "p_perm": float("nan"),
            "ci_lower": float("nan"), "ci_upper": float("nan"),
            "frac_settled_strict": float("nan"),
            "rho_no_ceiling": float("nan"), "ci_lower_no_ceiling": float("nan"),
            "ci_upper_no_ceiling": float("nan"), "p_perm_no_ceiling": float("nan"),
            "n_no_ceiling": 0, "n_ceiling_hit": 0,
            "breath_spread": False, "restriction_of_range": False,
        })
        return out

    # frac converging STRICTLY before the K ceiling.
    ceiling_hit = breath >= float(k_ceiling)
    n_ceiling = int(ceiling_hit.sum())
    frac_strict = float((~ceiling_hit).sum()) / float(n)

    # FALSE-WIN GUARD: breath_count must show spread (>1 distinct value) to be testable.
    breath_spread = int(np.unique(breath).size) > 1

    # Full-subset statistics.
    rho = spearman_rho(breath, depth)
    tau = kendall_tau_b(breath, depth)
    p = permutation_p_spearman(breath, depth, n_perm=n_perm, rng=rng)
    lo, hi, _ = bootstrap_ci_spearman(breath, depth, n_boot=n_boot, rng=rng)

    # Ceiling-EXCLUDED statistics (drop breath_count == ceiling).
    keep = ~ceiling_hit
    b_nc = breath[keep]
    d_nc = depth[keep]
    if b_nc.size >= 3:
        rho_nc = spearman_rho(b_nc, d_nc)
        p_nc = permutation_p_spearman(b_nc, d_nc, n_perm=n_perm, rng=rng)
        lo_nc, hi_nc, _ = bootstrap_ci_spearman(b_nc, d_nc, n_boot=n_boot, rng=rng)
    else:
        rho_nc = p_nc = lo_nc = hi_nc = float("nan")

    # Restriction-of-range: settled depth-range materially narrower than corpus range.
    restriction = False
    if corpus_depth_range is not None and n >= 2:
        sub_range = float(depth.max() - depth.min())
        corp_range = float(corpus_depth_range[1] - corpus_depth_range[0])
        if corp_range > 0 and sub_range < RESTRICTION_RANGE_FRAC * corp_range:
            restriction = True

    out.update({
        "rho": rho,
        "tau_b": tau,
        "p_perm": p,
        "ci_lower": lo,
        "ci_upper": hi,
        "frac_settled_strict": frac_strict,
        "n_ceiling_hit": n_ceiling,
        "n_no_ceiling": int(b_nc.size),
        "rho_no_ceiling": rho_nc,
        "p_perm_no_ceiling": p_nc,
        "ci_lower_no_ceiling": lo_nc,
        "ci_upper_no_ceiling": hi_nc,
        "breath_spread": breath_spread,
        "restriction_of_range": restriction,
        "depth_min": int(depth.min()),
        "depth_max": int(depth.max()),
        "depth_hist": _depth_hist(depth),
        "scipy_crosscheck": _scipy_crosscheck(breath, depth),
    })
    return out


def analyze_bin(records: list[dict], N: int, k_ceiling: int,
                n_perm: int, n_boot: int, rng: np.random.Generator,
                breath_field: str = SECONDARY_BREATH_FIELD,
                status_field: str = SECONDARY_STATUS_FIELD) -> dict:
    """Full per-N-bin analysis: settled-only PRIMARY + converged-companion + companions.

    INSTRUMENT-AWARE (the #238 U-curve min-based decision). `breath_field` /
    `status_field` select WHICH convergence instrument drives this bin:
      - PRIMARY (min-based): breath_field=PRIMARY_BREATH_FIELD ("breath_count_min"),
        status_field=PRIMARY_STATUS_FIELD ("status_min"). status_min in
        {settled, stuck} only — there is NO not_converged for the min instrument
        (an argmin always exists), so the converged-companion = settled+stuck =
        ALL puzzles in the bin.
      - SECONDARY (JSD-floor, TAIL-CONTAMINATED): breath_field="breath_count",
        status_field="status". status carries not_converged; under the #238
        U-curve a puzzle SOLVED at the refinement minimum then DRIFTING in the
        tail is mislabeled not_converged, censoring the settled set. Reported but
        FLAGGED; never drives the verdict when the min instrument is present.

    `breath_count_min` lives in [1, K-1] (transition index), reported 1-based as
    [2, K]; the K ceiling binds only when the LAST transition is the global argmin
    (rare). frac_settled_strict therefore means, for the min instrument, "fraction
    of settled puzzles whose SETTLE breath is not pinned at the K ceiling" — its
    meaning is recomputed by analyze_subset against the same k_ceiling but the
    interpretation differs (documented here, see also analyze_subset).
    """
    rows = [r for r in records if int(r.get("N", -1)) == N]
    breath_all = np.array([float(r[breath_field]) for r in rows], dtype=np.float64)
    depth_all = np.array([float(r["deduction_depth"]) for r in rows], dtype=np.float64)
    status_all = [str(r.get(status_field, "")) for r in rows]

    settled_mask = np.array([s == "settled" for s in status_all], dtype=bool)
    converged_mask = np.array([s in ("settled", "stuck") for s in status_all], dtype=bool)
    notconv_mask = np.array([s == "not_converged" for s in status_all], dtype=bool)

    n_settled = int(settled_mask.sum())
    n_stuck = int(np.array([s == "stuck" for s in status_all], dtype=bool).sum())
    n_notconv = int(notconv_mask.sum())
    n_total = len(rows)

    corpus_range = (int(depth_all.min()), int(depth_all.max())) if n_total else None

    # PRIMARY: settled-only.
    settled = analyze_subset(
        breath_all[settled_mask], depth_all[settled_mask], k_ceiling,
        "settled_only", n_perm, n_boot, rng, corpus_depth_range=corpus_range)
    # COMPANION: converged set (settled + stuck), regardless of correctness.
    # For the min instrument this is ALL puzzles (no not_converged exists).
    converged = analyze_subset(
        breath_all[converged_mask], depth_all[converged_mask], k_ceiling,
        "converged_companion", n_perm, n_boot, rng, corpus_depth_range=corpus_range)

    # ---- ROBUSTNESS COMPANIONS (reported, NOT gating) ----
    companions: dict = {}
    # breath_count vs n_givens (settled subset, if available).
    givens = [r.get("n_givens", None) for r in rows]
    if all(g is not None for g in givens) and n_settled >= 3:
        g_arr = np.array(givens, dtype=np.float64)[settled_mask]
        companions["rho_breath_vs_ngivens_settled"] = spearman_rho(
            breath_all[settled_mask], g_arr)
    else:
        companions["rho_breath_vs_ngivens_settled"] = None
    # secondary breath_count_frac correlation (settled subset, if present).
    bcf = [r.get("breath_count_frac", None) for r in rows]
    if all(b is not None for b in bcf) and n_settled >= 3:
        bcf_arr = np.array(bcf, dtype=np.float64)[settled_mask]
        companions["rho_breathfrac_vs_depth_settled"] = spearman_rho(
            bcf_arr, depth_all[settled_mask])
    else:
        companions["rho_breathfrac_vs_depth_settled"] = None
    # CE-companion breath_count_min_ce correlation (settled subset, if present).
    # USES GOLD upstream -> reported only, NEVER the correlation axis (leakage).
    bcc = [r.get("breath_count_min_ce", None) for r in rows]
    if all(b is not None for b in bcc) and n_settled >= 3:
        bcc_arr = np.array(bcc, dtype=np.float64)[settled_mask]
        companions["rho_breathmince_vs_depth_settled"] = spearman_rho(
            bcc_arr, depth_all[settled_mask])
    else:
        companions["rho_breathmince_vs_depth_settled"] = None

    # not_converged depth distribution (does it concentrate at high depth?).
    notconv_depths = depth_all[notconv_mask]
    settled_depths = depth_all[settled_mask]
    stuck_depths = depth_all[np.array([s == "stuck" for s in status_all], dtype=bool)]

    # ---- QUALIFICATION (settled-n>=50 AND frac_settled_strict>=0.80) ----
    frac_strict = settled["frac_settled_strict"]
    has_power = (n_settled >= QUALIFY_MIN_SETTLED_N)
    qualifies = bool(
        n_settled >= QUALIFY_MIN_SETTLED_N
        and np.isfinite(frac_strict)
        and frac_strict >= QUALIFY_MIN_FRAC_STRICT
        and settled["breath_spread"]
    )

    # ---- ceiling with/without disagreement on the primary (settled) ----
    ceiling_disagree = False
    rho_full = settled["rho"]
    rho_nc = settled["rho_no_ceiling"]
    if np.isfinite(rho_full) and np.isfinite(rho_nc):
        # disagree on the HILL bar (one over 0.30, the other under)
        if (rho_full > HILL_RHO_BAR) != (rho_nc > HILL_RHO_BAR):
            ceiling_disagree = True

    return {
        "N": N,
        "n_total": n_total,
        "breath_field": breath_field,
        "status_field": status_field,
        "counts": {"settled": n_settled, "stuck": n_stuck,
                   "not_converged": n_notconv},
        "not_converged_fraction": (n_notconv / n_total) if n_total else float("nan"),
        "corpus_depth_range": corpus_range,
        "depth_dist": {
            "settled": _depth_hist(settled_depths),
            "stuck": _depth_hist(stuck_depths),
            "not_converged": _depth_hist(notconv_depths),
        },
        "settled_only": settled,           # PRIMARY
        "converged_companion": converged,  # companion
        "robustness_companions": companions,
        "qualifies": qualifies,
        "has_power": has_power,
        "ceiling_with_without_disagree": ceiling_disagree,
    }


# ===========================================================================
# FOUR-WAY READ (the verdict)
# ===========================================================================

def _bin_primary_rho_p(bin_res: dict) -> tuple[float, float, float]:
    """Return (rho_point, lower_CI, p) used for the verdict on a bin.

    Ceiling-EXCLUDED is primary IF it disagrees with the full-subset verdict
    (per the pinned ceiling-control rule); else use the full subset.
    """
    s = bin_res["settled_only"]
    if bin_res["ceiling_with_without_disagree"]:
        return (s["rho_no_ceiling"], s["ci_lower_no_ceiling"], s["p_perm_no_ceiling"])
    return (s["rho"], s["ci_lower"], s["p_perm"])


def four_way_read(bin_results: list[dict]) -> dict:
    """Apply the pinned four-way read over the {5,6,7} bins.

    >=2 of 3 N-bins must AGREE (2/3 majority). A bin only contributes to HILL/WEAK
    if it QUALIFIES; a bin only contributes to NULL if it HAS POWER.
    """
    per_bin_calls: dict[int, str] = {}
    per_bin_detail: dict[int, dict] = {}

    n_qualifying = 0
    n_untestable_bins = 0
    n_power_bins = 0

    hill_votes = 0
    strong_votes = 0
    weak_votes = 0
    null_votes = 0

    for br in bin_results:
        N = br["N"]
        s = br["settled_only"]
        rho_pt, lo_ci, p = _bin_primary_rho_p(br)
        qualifies = br["qualifies"]
        has_power = br["has_power"]
        breath_spread = s["breath_spread"]
        # RESTRICTION-OF-RANGE GATE (pinned: memory/project_csp_target_survey_jun14.md
        # "PROPERTY-2 SUCCESS CRITERION" + maintainer point 3): a settled set whose
        # depth axis is too compressed to span the corpus cannot be distinguished
        # from a true null. Such a bin must NOT vote NULL — a flat/weak rho on a
        # depth range that doesn't span the corpus is UNTESTABLE, not null. A real
        # POSITIVE still stands under restriction (conservative — a true positive
        # survives a narrowed range), so HILL/WEAK are NOT blocked; only the NULL
        # direction is.
        restriction = bool(s.get("restriction_of_range"))

        if has_power:
            n_power_bins += 1

        call = "untestable"
        untestable_by_restriction = False
        if not breath_spread or not qualifies:
            # UNTESTABLE in this bin (no spread, or fails settled-n>=50 / frac_strict>=0.80).
            call = "untestable"
            n_untestable_bins += 1
        else:
            n_qualifying += 1
            if np.isfinite(lo_ci) and np.isfinite(p) and lo_ci > HILL_RHO_BAR and p < P_STRICT:
                call = "hill"
                hill_votes += 1
                if np.isfinite(rho_pt) and rho_pt >= STRONG_RHO:
                    strong_votes += 1
            elif np.isfinite(lo_ci) and np.isfinite(p) and (WEAK_RHO_LO <= lo_ci <= HILL_RHO_BAR) and p < P_STRICT:
                call = "weak"
                weak_votes += 1
            elif restriction:
                # Qualifies + restriction-of-range but did NOT clear HILL/WEAK.
                # It cannot be read as null (range too compressed to detect a
                # positive) — count it UNTESTABLE so the verdict can reach
                # UNTESTABLE via the n_untestable_bins>=2 branch instead of
                # falling silently to INCONCLUSIVE.
                call = "untestable"
                untestable_by_restriction = True
                n_qualifying -= 1
                n_untestable_bins += 1
            else:
                call = "weak_or_null"  # qualifies but neither hill nor clean weak

        # NULL vote: bin has POWER and is refuted (p>=0.05 OR lower-CI<=0.15).
        # A restriction-of-range bin must NOT contribute a NULL vote (you cannot
        # claim a null on a depth axis too compressed to detect a positive).
        if has_power and breath_spread and not restriction:
            if (np.isfinite(p) and p >= P_NULL) or (np.isfinite(lo_ci) and lo_ci <= WEAK_RHO_LO):
                null_votes += 1

        per_bin_calls[N] = call
        per_bin_detail[N] = {
            "rho_point": rho_pt, "lower_ci": lo_ci, "p_perm": p,
            "qualifies": qualifies, "has_power": has_power,
            "breath_spread": breath_spread,
            "ceiling_excluded_primary": br["ceiling_with_without_disagree"],
            "restriction_of_range": restriction,
            "untestable_by_restriction": untestable_by_restriction,
        }

    # Decision: 2/3 majority. LEAN note is informational (N=5 tail-starved).
    verdict = "UNTESTABLE"
    reason = ""
    if hill_votes >= 2:
        verdict = "HILL-STANDS"
        reason = f"{hill_votes}/3 qualifying bins clear lower-CI rho>{HILL_RHO_BAR} & p<{P_STRICT}"
        if strong_votes >= 2:
            verdict = "HILL-STANDS (STRONG-WIN)"
            reason += f"; {strong_votes}/3 bins point-rho>={STRONG_RHO}"
    elif (hill_votes + weak_votes) >= 2 and weak_votes >= 1:
        # >=2 bins suggestive (hill OR weak) with at least one clean weak -> WEAK overall.
        verdict = "WEAK"
        reason = f"{hill_votes} hill + {weak_votes} weak qualifying bins (>=2 suggestive)"
    elif null_votes >= 2:
        verdict = "NULL"
        reason = f"{null_votes}/3 power-bins refuted (p>={P_NULL} or lower-CI<={WEAK_RHO_LO})"
    elif n_untestable_bins >= 2:
        verdict = "UNTESTABLE"
        reason = (f"{n_untestable_bins}/3 bins fail power/spread "
                  f"(settled-n<{QUALIFY_MIN_SETTLED_N} or frac_strict<{QUALIFY_MIN_FRAC_STRICT} "
                  f"or no breath spread) — regenerate harder/deeper corpus, do NOT declare null")
    else:
        verdict = "INCONCLUSIVE"
        reason = (f"no 2/3 majority: hill={hill_votes} weak={weak_votes} "
                  f"null={null_votes} untestable={n_untestable_bins}")

    return {
        "verdict": verdict,
        "reason": reason,
        "votes": {"hill": hill_votes, "strong": strong_votes, "weak": weak_votes,
                  "null": null_votes, "untestable_bins": n_untestable_bins,
                  "qualifying_bins": n_qualifying, "power_bins": n_power_bins},
        "per_bin_calls": per_bin_calls,
        "per_bin_detail": per_bin_detail,
        "lean_note": ("LEAN on N=6/N=7 in close calls; N=5 is tail-starved "
                      "(depth range compressed, achievable rho structurally lower)"),
    }


def collect_caveats(bin_results: list[dict]) -> list[str]:
    caveats = []
    for br in bin_results:
        N = br["N"]
        s = br["settled_only"]
        if br["has_power"] and not br["qualifies"]:
            if np.isfinite(s["frac_settled_strict"]) and s["frac_settled_strict"] < QUALIFY_MIN_FRAC_STRICT:
                caveats.append(
                    f"N={N}: frac_settled_strict={s['frac_settled_strict']:.2f} < "
                    f"{QUALIFY_MIN_FRAC_STRICT} (too many ceiling-pins; bin does NOT qualify)")
        if not br["has_power"]:
            caveats.append(
                f"N={N}: settled-n={br['counts']['settled']} < {QUALIFY_MIN_SETTLED_N} "
                f"(underpowered; cannot vote NULL here)")
        if not s["breath_spread"] and br["counts"]["settled"] > 0:
            caveats.append(
                f"N={N}: NO breath_count spread among settled puzzles "
                f"(false-win guard tripped — UNTESTABLE, not null)")
        if s.get("restriction_of_range"):
            caveats.append(
                f"N={N}: settled depth-range [{s.get('depth_min')},{s.get('depth_max')}] "
                f"materially narrower than corpus range {br['corpus_depth_range']} "
                f"(restriction-of-range — attenuates rho)")
        if br["ceiling_with_without_disagree"]:
            caveats.append(
                f"N={N}: ceiling-WITH vs ceiling-WITHOUT rho disagree on the bar "
                f"(rho={s['rho']:.3f} vs no-ceiling={s['rho_no_ceiling']:.3f}) — "
                f"ceiling-excluded is primary")
        nc_frac = br["not_converged_fraction"]
        if np.isfinite(nc_frac) and nc_frac > 0.2:
            caveats.append(
                f"N={N}: not_converged fraction={nc_frac:.2f} high "
                f"(if concentrated at high depth, settled rho is on a truncated subset)")
    return caveats


# ===========================================================================
# RECORD I/O + DRIVING
# ===========================================================================

def load_property2_jsonl(path: str) -> list[dict]:
    """Read a property2 records JSONL; tolerate the trainer schema.

    Required fields: breath_count, deduction_depth, N, status. Optional and
    tolerated: converged, correct, step, core, n_givens, breath_count_frac.
    """
    out = []
    with open(path) as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{ln}: invalid JSON: {e}") from e
            for req in ("breath_count", "deduction_depth", "N", "status"):
                if req not in r:
                    raise ValueError(f"{path}:{ln}: missing required field '{req}'")
            out.append(r)
    return out


def _analyze_with_instrument(records: list[dict], k_ceiling: int,
                             n_perm: int, n_boot: int, rng: np.random.Generator,
                             breath_field: str, status_field: str) -> list[dict]:
    """Per-bin analysis using a chosen convergence instrument (breath/status fields)."""
    bin_results = []
    for N in N_BINS:
        if any(int(r.get("N", -1)) == N for r in records):
            bin_results.append(analyze_bin(
                records, N, k_ceiling, n_perm, n_boot, rng,
                breath_field=breath_field, status_field=status_field))
    return bin_results


def run_full_analysis(records: list[dict], k_ceiling: int,
                      n_perm: int, n_boot: int, seed: int) -> dict:
    """Run per-bin analysis + four-way read over a list of property2 records.

    INSTRUMENT SELECTION (the #238 U-curve min-based decision):
      - If the records carry the min-based fields (MODE A output / the current
        convergence_instrument), the PRIMARY read is min-based
        (breath_count_min / status_min) and DRIVES the verdict. A SECONDARY read
        on the JSD-floor instrument (breath_count / status) is ALSO computed and
        reported, but FLAGGED tail-contaminated and never gates the verdict.
      - If the records lack the min-based fields (MODE B over the LIVE trainer's
        JSON-floor-only records), there is no min-based read to do: the analyzer
        falls back to the JSD-floor instrument, drives the verdict from it, and
        flags loudly that this is "JSD-floor only (tail-contaminated); run MODE A
        for the binding min-based read."
    """
    rng = np.random.default_rng(seed)
    min_available = has_min_instrument(records)

    if min_available:
        # PRIMARY = min-based (gold-free). This drives the verdict.
        bin_results = _analyze_with_instrument(
            records, k_ceiling, n_perm, n_boot, rng,
            breath_field=PRIMARY_BREATH_FIELD, status_field=PRIMARY_STATUS_FIELD)
        # SECONDARY = JSD-floor (tail-contaminated). Reported, NOT gating.
        secondary_bins = _analyze_with_instrument(
            records, k_ceiling, n_perm, n_boot, rng,
            breath_field=SECONDARY_BREATH_FIELD, status_field=SECONDARY_STATUS_FIELD)
        instrument = "min_based_primary"
        instrument_note = (
            "PRIMARY = min-based (argmin consecutive-belief-change; gold-free; "
            "U-curve robust). SECONDARY = JSD-floor (tail-contaminated; reported, "
            "NOT gating).")
    else:
        # MODE B fallback: JSD-floor only — drives the verdict, flagged loudly.
        bin_results = _analyze_with_instrument(
            records, k_ceiling, n_perm, n_boot, rng,
            breath_field=SECONDARY_BREATH_FIELD, status_field=SECONDARY_STATUS_FIELD)
        secondary_bins = None
        instrument = "jsd_floor_fallback"
        instrument_note = (
            "JSD-floor only (tail-contaminated); the min-based fields "
            "(breath_count_min / status_min) are ABSENT from these records — this "
            "is the LIVE trainer's JSON-floor-only schema. Run MODE A "
            "(analyze_kenken_property2.py CKPT) for the binding min-based read.")

    verdict = four_way_read(bin_results)
    caveats = collect_caveats(bin_results)
    if not min_available:
        caveats.insert(
            0, "INSTRUMENT: JSD-floor only (tail-contaminated under the #238 "
               "U-curve); min-based breath_count_min ABSENT. Run MODE A for the "
               "binding min-based read.")

    out = {
        "k_ceiling": k_ceiling,
        "n_records": len(records),
        "n_perm": n_perm,
        "n_boot": n_boot,
        "seed": seed,
        "instrument": instrument,
        "instrument_note": instrument_note,
        "bins": bin_results,
        "verdict": verdict,
        "caveats": caveats,
    }
    if secondary_bins is not None:
        # Contaminated SECONDARY verdict reported alongside (NOT the binding read).
        out["secondary_jsd_floor"] = {
            "note": "TAIL-CONTAMINATED JSD-floor read; reported for comparison, "
                    "NOT the binding verdict.",
            "bins": secondary_bins,
            "verdict": four_way_read(secondary_bins),
            "caveats": collect_caveats(secondary_bins),
        }
    return out


def _json_sanitize(obj):
    """Recursively convert numpy scalars -> python, and non-finite floats -> None,
    so the verdict JSON is STRICT-JSON valid (no bare NaN/Infinity tokens)."""
    if isinstance(obj, dict):
        return {(_json_sanitize(k) if not isinstance(k, str) else k): _json_sanitize(v)
                for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_sanitize(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return f if np.isfinite(f) else None
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def _fmt(x, nd=3):
    if x is None:
        return "  -  "
    try:
        if not np.isfinite(x):
            return " nan "
    except (TypeError, ValueError):
        return str(x)
    return f"{x:.{nd}f}"


def print_summary(result: dict) -> None:
    print("=" * 92)
    print("KENKEN PROPERTY-2 — ADAPTIVE-DEPTH TELEGRAPH — POST-HOC BINDING ANALYSIS")
    print("=" * 92)
    print(f"records={result['n_records']}  K_ceiling={result['k_ceiling']}  "
          f"n_perm={result['n_perm']}  n_boot={result['n_boot']}  seed={result['seed']}")
    instrument = result.get("instrument", "jsd_floor_fallback")
    print(f"INSTRUMENT: {instrument}")
    print(f"  {result.get('instrument_note', '')}")
    if instrument == "jsd_floor_fallback":
        print("  *** WARNING: min-based (binding) read UNAVAILABLE on these records. ***")
    print()
    print("PRIMARY = settled-only; COMPANION = converged (settled+stuck). "
          "rho=Spearman (gated), tau-b descriptive.")
    if result.get("bins"):
        bf = result["bins"][0].get("breath_field", "breath_count")
        sf = result["bins"][0].get("status_field", "status")
        print(f"(breath_field={bf}  status_field={sf})")
    print("-" * 92)
    header = (f"{'N':>2} {'set':>4} {'qual':>4} {'rho':>6} {'loCI':>6} {'hiCI':>6} "
              f"{'p':>7} {'tau-b':>6} {'fstr':>5} {'rho_nc':>6} {'set/stk/nc':>12}")
    print("SETTLED-ONLY (PRIMARY):")
    print(header)
    for br in result["bins"]:
        s = br["settled_only"]
        c = br["counts"]
        print(f"{br['N']:>2} {c['settled']:>4} {('Y' if br['qualifies'] else 'n'):>4} "
              f"{_fmt(s['rho']):>6} {_fmt(s['ci_lower']):>6} {_fmt(s['ci_upper']):>6} "
              f"{_fmt(s['p_perm'],4):>7} {_fmt(s['tau_b']):>6} "
              f"{_fmt(s['frac_settled_strict'],2):>5} {_fmt(s['rho_no_ceiling']):>6} "
              f"{c['settled']:>3}/{c['stuck']:>3}/{c['not_converged']:>3}")
    print()
    print("CONVERGED-COMPANION (settled+stuck):")
    print(header)
    for br in result["bins"]:
        cc = br["converged_companion"]
        c = br["counts"]
        print(f"{br['N']:>2} {cc['n']:>4} {'-':>4} "
              f"{_fmt(cc['rho']):>6} {_fmt(cc['ci_lower']):>6} {_fmt(cc['ci_upper']):>6} "
              f"{_fmt(cc['p_perm'],4):>7} {_fmt(cc['tau_b']):>6} "
              f"{_fmt(cc['frac_settled_strict'],2):>5} {_fmt(cc['rho_no_ceiling']):>6} "
              f"{c['settled']:>3}/{c['stuck']:>3}/{c['not_converged']:>3}")
    print()
    print("ROBUSTNESS COMPANIONS (reported, NOT gating):")
    for br in result["bins"]:
        rc = br["robustness_companions"]
        print(f"  N={br['N']}: rho(breath,n_givens|settled)="
              f"{_fmt(rc.get('rho_breath_vs_ngivens_settled'))}  "
              f"rho(breath_frac,depth|settled)="
              f"{_fmt(rc.get('rho_breathfrac_vs_depth_settled'))}  "
              f"rho(breath_min_ce,depth|settled)="
              f"{_fmt(rc.get('rho_breathmince_vs_depth_settled'))}  "
              f"not_conv_frac={_fmt(br['not_converged_fraction'],2)}")
    # pooled board-size companion
    print()
    if result["caveats"]:
        print("CAVEATS:")
        for cv in result["caveats"]:
            print(f"  - {cv}")
    else:
        print("CAVEATS: none flagged")
    print()
    v = result["verdict"]
    print("=" * 92)
    print(f"VERDICT ({result.get('instrument', 'jsd_floor_fallback')}): {v['verdict']}")
    print(f"  reason : {v['reason']}")
    print(f"  votes  : {v['votes']}")
    print(f"  per-bin: {v['per_bin_calls']}")
    print(f"  {v['lean_note']}")
    print("=" * 92)

    # Contaminated SECONDARY verdict (reported only; NEVER the binding read).
    sec = result.get("secondary_jsd_floor")
    if sec is not None:
        sv = sec["verdict"]
        print()
        print("-" * 92)
        print("SECONDARY (JSD-FLOOR, TAIL-CONTAMINATED — reported, NOT the binding verdict):")
        for br in sec["bins"]:
            s = br["settled_only"]
            c = br["counts"]
            print(f"  N={br['N']}: rho={_fmt(s['rho'])} loCI={_fmt(s['ci_lower'])} "
                  f"p={_fmt(s['p_perm'],4)} set/stk/nc="
                  f"{c['settled']}/{c['stuck']}/{c['not_converged']}")
        print(f"  SECONDARY verdict (FLAGGED tail-contaminated): {sv['verdict']} "
              f"({sv['reason']})")
        print("-" * 92)


# ---------------------------------------------------------------------------
# MODE A — model over corpus (NOT exercised by self-test; GPU is busy)
# ---------------------------------------------------------------------------

def _breath_count_frac(beliefs: list[np.ndarray], valid: np.ndarray,
                       threshold: float, frac_required: float = 0.95,
                       K: int | None = None) -> int:
    """Secondary convergence def for ONE puzzle: first 1-based breath k where
    >= frac_required of valid cells have per-cell JSD(belief_k, belief_{k-1}) < threshold.

    beliefs: list of (49, N_MAX) softmax arrays for this puzzle (length K).
    valid:   (49,) bool. Returns K (=ceiling) if never satisfied.
    """
    from mycelium.kenken import _jsd  # reuse the exact instrument JSD (base-2)
    Kc = len(beliefs) if K is None else K
    n_valid = int(valid.sum())
    if n_valid == 0:
        return Kc
    for k in range(1, Kc):
        jsd_k = _jsd(beliefs[k], beliefs[k - 1], axis=-1)   # (49,)
        below = jsd_k[valid] < threshold
        if float(below.mean()) >= frac_required:
            return k + 1
    return Kc


def run_mode_a(ckpt_path: str, corpus_path: str, records_out: str,
               K: int) -> list[dict]:
    """MODE A: load the ckpt into the KenKen breathing model and run forward over
    the corpus, producing per-puzzle Property-2 records (with n_givens and the
    secondary breath_count_frac). Writes records_out JSONL and returns the records.

    *** RUNS THE MODEL ON THE GPU. *** Mirrors scripts/kenken_train.py model
    construction + load_ckpt. A human runs this when the GPU is free.
    """
    # Imports are local so MODE B / self-test never touch tinygrad / the GPU.
    import numpy as _np
    from tinygrad import Tensor, Device, dtypes  # noqa: F401
    from tinygrad.helpers import getenv

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from mycelium import Config, BreathingTransformer
    from mycelium.loader import _load_state, load_breathing
    from mycelium.kenken import (
        attach_kenken_params, kenken_breathing_forward, convergence_instrument,
        KENKEN_CONVERGE_JSD,
    )
    from mycelium.kenken_data import KenKenLoader, load_jsonl, N_MAX as _NMAX
    # Mirror kenken_train.py helpers exactly (model build + ckpt load).
    from scripts.kenken_train import cast_layers_fp32, load_ckpt

    PYTHIA_INIT = int(getenv("PYTHIA_INIT", 1)) > 0
    EVAL_BATCH = int(getenv("EVAL_BATCH", getenv("BATCH", 8)))
    SEED = int(getenv("SEED", 42))

    # n_cages_max consistent with the trainer: max over the corpus we read AND the
    # trainer's paired corpus if present (graph-stability law). We pin to the
    # corpus we actually run (analysis is eager, single corpus) but allow override.
    recs = load_jsonl(corpus_path)
    corpus_n_cages_max = max(len(r["cages"]) for r in recs)
    # Mirror the trainer's train+test max if the sibling corpus exists.
    train_p = os.environ.get("KENKEN_TRAIN", DEFAULT_TRAIN_CORPUS)
    test_p = os.environ.get("KENKEN_TEST", ".cache/kenken_test.jsonl")
    try:
        if os.path.exists(train_p) and os.path.exists(test_p):
            tr = load_jsonl(train_p)
            te = load_jsonl(test_p)
            corpus_n_cages_max = max(corpus_n_cages_max,
                                     max(len(r["cages"]) for r in tr),
                                     max(len(r["cages"]) for r in te))
    except Exception:
        pass
    N_CAGES_MAX = int(getenv("KENKEN_N_CAGES_MAX", str(corpus_n_cages_max)))

    cfg = Config()
    print(f"[MODE A] building model (PYTHIA_INIT={PYTHIA_INIT}); "
          f"corpus={corpus_path} ({len(recs)} puzzles); "
          f"n_cages_max={N_CAGES_MAX}; K={K}", flush=True)
    if PYTHIA_INIT:
        sd = _load_state()
        model = load_breathing(cfg, sd=sd)
        del sd
    else:
        model = BreathingTransformer(cfg)
    cast_layers_fp32(model)
    attach_kenken_params(model, hidden=cfg.hidden, n_heads=cfg.n_heads, k_max=K)
    Device[Device.DEFAULT].synchronize()
    print(f"[MODE A] loading checkpoint {ckpt_path}", flush=True)
    load_ckpt(model, ckpt_path)

    loader = KenKenLoader(corpus_path, batch_size=EVAL_BATCH, seed=SEED + 7,
                          n_cages_max=N_CAGES_MAX)

    Tensor.training = False
    threshold = KENKEN_CONVERGE_JSD
    records: list[dict] = []
    n_seen = 0
    total = len(loader)
    for batch in loader.iter_eval(batch_size=EVAL_BATCH):
        cell_logits_history, _ = kenken_breathing_forward(model, batch, K=K)
        # Primary records (all-cells-AND breath_count, status, correct).
        recs_b = convergence_instrument(cell_logits_history, batch, threshold=threshold)

        # Secondary breath_count_frac needs per-cell beliefs; recompute once.
        beliefs = []
        for logits in cell_logits_history:
            l = logits.realize().numpy().astype(_np.float64)
            l = l - l.max(axis=-1, keepdims=True)
            e = _np.exp(l)
            beliefs.append(e / (e.sum(axis=-1, keepdims=True) + 1e-12))  # (B,49,NMAX)
        cell_valid_np = batch.cell_valid.realize().numpy().astype(bool)   # (B,49)

        Bn = len(recs_b)
        for b in range(Bn):
            # Stop once we've covered the real corpus (iter_eval pads the last batch).
            if n_seen >= total:
                break
            rec = dict(recs_b[b])
            # recs_b already carries the full instrument schema (the min-based
            # PRIMARY fields breath_count_min / correct_min / status_min, the
            # companions breath_count_min_ce / breath_count_frac, AND the
            # JSD-floor SECONDARY breath_count / status). We only annotate N /
            # n_givens / core here.
            rec["N"] = int(batch.N[b])
            rec["n_givens"] = int(batch.n_givens[b])
            rec["core"] = "v98_residual"
            valid_b = cell_valid_np[b]
            beliefs_b = [beliefs[k][b] for k in range(len(beliefs))]
            # Defensive cross-check: recompute breath_count_frac from the same
            # base-2 _jsd and overwrite. This is identical to the instrument's own
            # breath_count_frac (same threshold, frac_required=0.95) — a single
            # source-of-truth guard, not a second definition.
            rec["breath_count_frac"] = _breath_count_frac(
                beliefs_b, valid_b, threshold=threshold, frac_required=0.95, K=K)
            records.append(rec)
            n_seen += 1
        if n_seen >= total:
            break

    os.makedirs(os.path.dirname(os.path.abspath(records_out)), exist_ok=True)
    with open(records_out, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    print(f"[MODE A] wrote {len(records)} per-puzzle records -> {records_out}", flush=True)
    Tensor.training = True
    return records


# ===========================================================================
# SELF-TEST (CPU-only; no GPU, no model)
# ===========================================================================

def _train_depth_distributions(train_path: str) -> dict:
    """Read deduction_depth values per N from the TRAIN corpus (for synthetic draws)."""
    by_N: dict[int, list[int]] = {5: [], 6: [], 7: []}
    with open(train_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            N = int(r["N"])
            if N in by_N:
                by_N[N].append(int(r["deduction_depth"]))
    return {N: np.array(v, dtype=np.float64) for N, v in by_N.items() if v}


def _synth_records(depths_by_N: dict, mode: str, rng: np.random.Generator,
                   k_ceiling: int = K_CEILING_DEFAULT,
                   strong_slope: float = 0.45, noise_sd: float = 1.0,
                   ceiling_frac: float = 0.25, with_min: bool = True) -> list[dict]:
    """Build synthetic property2 records from real per-N depth distributions.

    mode:
      'strong'  : breath_count = clip(round(slope*depth + N(0,noise_sd)), 1, 20),
                  status=settled for all -> a genuine positive link.
      'null'    : breath_count drawn independent of depth (shuffle of the strong
                  breath_counts within bin) -> no link.
      'ceiling' : strong link, but a RANDOM `ceiling_frac` of puzzles are PINNED to
                  breath_count==20. Per the pinned spec, AND-gate ceiling-pins (one
                  oscillating cell pins the whole puzzle to 20) are depth-INDEPENDENT
                  noise that dilutes rho toward 0 — so the pin is drawn at RANDOM
                  (depth-independent); excluding those pins is signal-recovery, i.e.
                  the WITH-ceiling rho is ATTENUATED below the no-ceiling rho.

    `with_min` (default True): also emit the MIN-BASED instrument fields by
    MIRRORING the JSD-floor fields (breath_count_min := breath_count,
    status_min := status, and the companions). This routes the existing
    strong/null/ceiling self-tests through the PRIMARY min instrument (which is
    what the analyzer now uses as the binding read) while preserving their
    behaviour. Set with_min=False to emit a JSON-floor-only record (the LIVE
    trainer schema) for the fallback test.
    """
    records = []
    for N, depths in depths_by_N.items():
        n = depths.size
        base = np.clip(np.round(strong_slope * depths + rng.normal(0, noise_sd, n)),
                       1, k_ceiling).astype(int)
        if mode == "strong":
            bc = base
        elif mode == "null":
            bc = base.copy()
            rng.shuffle(bc)             # break the depth link within the bin
        elif mode == "ceiling":
            bc = base.copy()
            # pin a RANDOM (depth-INDEPENDENT) ceiling_frac to the K ceiling — the
            # spec's AND-gate-pin noise model. Avoid already-at-ceiling rows so the
            # pin genuinely adds dilution (a high-depth strong-link row could already
            # sit near 20). This makes the with-ceiling rho < no-ceiling rho.
            n_pin = int(round(ceiling_frac * n))
            if n_pin > 0:
                candidates = np.where(base < k_ceiling)[0]
                if candidates.size >= n_pin:
                    pin_idx = rng.choice(candidates, size=n_pin, replace=False)
                else:
                    pin_idx = candidates
                bc[pin_idx] = k_ceiling
        else:
            raise ValueError(mode)
        for i in range(n):
            rec = {
                "breath_count": int(bc[i]),
                "deduction_depth": int(depths[i]),
                "N": int(N),
                "status": "settled",
                "converged": True,
                "correct": True,
                "n_givens": int(rng.integers(3, 8)),
            }
            if with_min:
                # Mirror the JSD-floor field into the min instrument so the
                # PRIMARY (min-based) read exercises these records.
                rec["breath_count_min"] = int(bc[i])
                rec["status_min"] = "settled"
                rec["correct_min"] = True
                rec["breath_count_min_ce"] = int(bc[i])
                rec["breath_count_frac"] = int(bc[i])
            records.append(rec)
    return records


# ---------------------------------------------------------------------------
# BELIEF-SEQUENCE self-test machinery (exercises the REAL convergence_instrument
# from mycelium.kenken, CPU-only — no GPU, no model, no MODE A).
# ---------------------------------------------------------------------------

class _FakeT:
    """A numpy-backed stand-in for a tinygrad Tensor that supports the small
    surface convergence_instrument uses: .shape, .realize(), .numpy().

    This lets the self-test feed numpy belief logits / batch tensors through the
    REAL mycelium.kenken.convergence_instrument WITHOUT touching tinygrad's GPU
    device (no real Tensor is ever realized on a device)."""
    def __init__(self, arr: np.ndarray):
        self._arr = np.asarray(arr)

    @property
    def shape(self):
        return self._arr.shape

    def realize(self):
        return self

    def numpy(self):
        return self._arr


class _FakeBatch:
    """Minimal batch carrying exactly the attributes convergence_instrument reads:
    cell_valid, gold, input_cells (all _FakeT), and deduction_depth (a list)."""
    def __init__(self, cell_valid, gold, input_cells, deduction_depth):
        self.cell_valid = _FakeT(cell_valid)
        self.gold = _FakeT(gold)
        self.input_cells = _FakeT(input_cells)
        self.deduction_depth = list(deduction_depth)


def _logits_from_pred(pred_value: int, n_max: int, conf: float = 9.0) -> np.ndarray:
    """One-cell logit vector (n_max,) sharply favouring `pred_value` (1-based).

    `conf` is the peak logit; the rest are 0 -> belief sharpness grows with conf.
    """
    v = np.zeros(n_max, dtype=np.float64)
    v[pred_value - 1] = conf
    return v


def _logits_from_belief(belief: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Logits whose softmax reproduces `belief` (log of clipped belief). Lets the
    self-test specify beliefs directly in PROBABILITY space (so consecutive-JSD is
    controllable) and feed them to the logit-consuming instrument unchanged."""
    return np.log(np.clip(belief, eps, 1.0))


def _build_ucurve_logits(K: int, n_cells: int, valid_idx: list[int],
                         gold_vals: np.ndarray, settle_breath_1based: int,
                         n_max: int = 7,
                         tail_wrong_value_fn=None) -> list[_FakeT]:
    """Build a K-long per-breath logit history (each a _FakeT of shape (1, n_cells, n_max))
    that:
      - REFINES toward the gold answer, reaching it (sharp + CORRECT) at
        `settle_breath_1based` (1-based breath index),
      - then DRIFTS over the tail: beliefs keep MOVING (consecutive-JSD RISES again
        AND STAYS elevated — it does NOT re-settle) and the FINAL breath is WRONG
        on at least one valid cell.

    The settle breath is the UNIQUE global argmin of the mean-over-valid
    consecutive-JSD because (a) belief change shrinks to ~0 as the puzzle locks
    onto gold near the settle breath, then (b) the tail drift ROTATES the wrong
    target each breath so consecutive-JSD stays large all the way to the end — it
    never re-collapses to a second near-zero (which would create a spurious second
    minimum). This is the #238 U-curve at the BELIEF level (not just CE level).

    `tail_wrong_value_fn(cell, gold_value) -> wrong_value` chooses the tail's
    (wrong) drift target per cell; default picks gold+1 (mod n_max). The tail
    additionally ROTATES that wrong target by the breath index so successive tail
    beliefs differ -> non-saturating drift.
    """
    s = settle_breath_1based                       # breath where belief == gold, sharp
    history = []
    if tail_wrong_value_fn is None:
        def tail_wrong_value_fn(cell, gv):
            return (gv % n_max) + 1                 # any value != gold (1..n_max)
    # Sharp beliefs built in PROBABILITY space. The tail ALTERNATES between two
    # DISTINCT wrong peaks (w1, w2, never gold) so every tail consecutive-JSD is
    # large and CONSTANT (two sharp distributions on different peaks ~0.93) and
    # never re-collapses to ~0 (no spurious second minimum). The approach ramps
    # gold confidence so the INTO-SETTLE transition is the UNIQUE near-zero global
    # min. The final breath is a sharp WRONG peak.
    sharp = 0.985                                   # peak prob of a "sharp" belief
    off = (1.0 - sharp) / float(n_max - 1)

    def _sharp_belief(val: int) -> np.ndarray:
        b = np.full(n_max, off, dtype=np.float64)
        b[val - 1] = sharp
        return b

    for k in range(K):  # k is 0-based breath
        kb = k + 1      # 1-based
        grid = np.zeros((1, n_cells, n_max), dtype=np.float64)
        for ci, cell in enumerate(valid_idx):
            gv = int(gold_vals[cell])
            w1 = int(tail_wrong_value_fn(cell, gv))
            if w1 == gv:
                w1 = (gv % n_max) + 1
            # a SECOND distinct wrong value != gv and != w1 for the alternation.
            w2 = (w1 % n_max) + 1
            if w2 == gv:
                w2 = (w2 % n_max) + 1
            if w2 == w1:
                w2 = (w2 % n_max) + 1
            if kb < s:
                # APPROACH: belief interpolates uniform -> gold along a CONCAVE ramp
                # (frac saturates fast) so by breath s-1 it is already ~sharp gold
                # and the INTO-SETTLE transition (s-1 -> s) is the UNIQUE near-zero
                # global minimum. Earlier approach transitions are larger.
                frac = 1.0 - (1.0 - kb / float(s)) ** 2
                belief = (1.0 - frac) * np.full(n_max, 1.0 / n_max) + frac * _sharp_belief(gv)
            elif kb == s:
                # SETTLE: pure sharp gold (the least-moving breath = global argmin).
                belief = _sharp_belief(gv)
            else:
                # TAIL DRIFT: alternate w1/w2 each breath -> constant large JSD, no
                # re-settle. The FINAL breath is sharp wrong (final-breath incorrect).
                belief = _sharp_belief(w1 if ((kb - s) % 2 == 1) else w2)
            grid[0, cell] = _logits_from_belief(belief)
        history.append(_FakeT(grid))
    return history


def run_selftest() -> int:
    print("#" * 92)
    print("# SELF-TEST (CPU-only, no GPU, no model) — analyze_kenken_property2.py")
    print("#" * 92)
    # Smaller perm/boot for self-test speed; the stats logic is identical.
    N_PERM = 10000
    N_BOOT = 10000
    train_path = DEFAULT_TRAIN_CORPUS
    if not os.path.exists(train_path):
        print(f"FAIL: train corpus not found at {train_path}; cannot draw synthetic depths")
        return 1
    depths_by_N = _train_depth_distributions(train_path)
    print(f"[setup] train depth draws per N: "
          f"{ {N: (int(d.min()), int(d.max()), d.size) for N, d in depths_by_N.items()} }")
    print()

    failures = []

    # ---------- 1. SYNTHETIC STRONG ----------
    print("-" * 92)
    print("[1] SYNTHETIC STRONG link: breath = clip(round(0.45*depth + N(0,1)), 1, 20)")
    rng = np.random.default_rng(123)
    recs = _synth_records(depths_by_N, "strong", rng)
    res = run_full_analysis(recs, K_CEILING_DEFAULT, N_PERM, N_BOOT, seed=7)
    print_summary(res)
    # Expectation: p<0.01 AND positive lower-CI in qualifying bins; verdict HILL/WEAK/STRONG.
    ok_bins = 0
    for br in res["bins"]:
        s = br["settled_only"]
        if br["qualifies"] and np.isfinite(s["p_perm"]) and s["p_perm"] < 0.01 \
                and np.isfinite(s["ci_lower"]) and s["ci_lower"] > 0:
            ok_bins += 1
    v = res["verdict"]["verdict"]
    strong_ok = (ok_bins >= 2) and (v.startswith("HILL") or v == "WEAK")
    print(f"\n[1] RESULT: qualifying bins with p<0.01 & lower-CI>0 = {ok_bins}/3; "
          f"verdict={v}  ->  {'PASS' if strong_ok else 'FAIL'}")
    if not strong_ok:
        failures.append("synthetic STRONG did not fire p<0.01 + positive lower-CI in >=2 bins")
    print()

    # ---------- 2. SYNTHETIC NULL ----------
    print("-" * 92)
    print("[2] SYNTHETIC NULL: breath_count independent of depth (within-bin shuffle)")
    rng = np.random.default_rng(456)
    recs = _synth_records(depths_by_N, "null", rng)
    res = run_full_analysis(recs, K_CEILING_DEFAULT, N_PERM, N_BOOT, seed=11)
    print_summary(res)
    # Expectation: NOT a false HILL; in power bins p>=0.05 OR lower-CI<=0.15; verdict NULL/INCONCLUSIVE/UNTESTABLE.
    false_hill = 0
    null_like_bins = 0
    for br in res["bins"]:
        s = br["settled_only"]
        if br["qualifies"] and np.isfinite(s["p_perm"]) and s["p_perm"] < 0.01 \
                and np.isfinite(s["ci_lower"]) and s["ci_lower"] > HILL_RHO_BAR:
            false_hill += 1
        if br["has_power"] and ((np.isfinite(s["p_perm"]) and s["p_perm"] >= 0.05)
                                or (np.isfinite(s["ci_lower"]) and s["ci_lower"] <= 0.15)):
            null_like_bins += 1
    v = res["verdict"]["verdict"]
    null_ok = (false_hill == 0) and (not v.startswith("HILL")) and (null_like_bins >= 2)
    print(f"\n[2] RESULT: false-HILL bins={false_hill} (want 0); "
          f"null-like power bins={null_like_bins} (want >=2); verdict={v}  ->  "
          f"{'PASS' if null_ok else 'FAIL'}")
    if not null_ok:
        failures.append("synthetic NULL false-positived or failed to read null where power exists")
    print()

    # ---------- 3. SYNTHETIC CEILING-CENSORED ----------
    print("-" * 92)
    print("[3] SYNTHETIC CEILING-CENSORED: 25% of high-depth puzzles pinned to breath=20")
    rng = np.random.default_rng(789)
    recs_clean = _synth_records(depths_by_N, "strong", np.random.default_rng(789))
    rng = np.random.default_rng(789)
    recs_ceil = _synth_records(depths_by_N, "ceiling", rng, ceiling_frac=0.25)
    res = run_full_analysis(recs_ceil, K_CEILING_DEFAULT, N_PERM, N_BOOT, seed=13)
    print_summary(res)
    # Expectation: frac_settled_strict drops below 1.0 (ideally near 0.75) and the
    # with-ceiling rho is ATTENUATED vs the no-ceiling rho (with-ceiling < no-ceiling).
    frac_dropped = 0
    attenuated = 0
    for br in res["bins"]:
        s = br["settled_only"]
        if np.isfinite(s["frac_settled_strict"]) and s["frac_settled_strict"] < 0.95:
            frac_dropped += 1
        if np.isfinite(s["rho"]) and np.isfinite(s["rho_no_ceiling"]) \
                and s["rho"] < s["rho_no_ceiling"] + 1e-9:
            attenuated += 1
    # Also show the qualification gate flipping off where frac<0.80.
    n_disq_by_frac = sum(
        1 for br in res["bins"]
        if br["has_power"] and np.isfinite(br["settled_only"]["frac_settled_strict"])
        and br["settled_only"]["frac_settled_strict"] < QUALIFY_MIN_FRAC_STRICT)
    ceil_ok = (frac_dropped >= 2) and (attenuated >= 2)
    print(f"\n[3] RESULT: bins with frac_strict dropped (<0.95)={frac_dropped}/3 (want >=2); "
          f"bins where with-ceiling rho <= no-ceiling rho (attenuated)={attenuated}/3 (want >=2); "
          f"bins disqualified by frac<0.80={n_disq_by_frac}  ->  "
          f"{'PASS' if ceil_ok else 'FAIL'}")
    if not ceil_ok:
        failures.append("synthetic CEILING did not drop frac_strict and/or attenuate with-ceiling rho")
    print()

    # ---------- 4. SCHEMA: parse the dry-run records ----------
    print("-" * 92)
    print("[4] SCHEMA parse: dry-run property2 JSONL (untrained -> degenerate, expect UNTESTABLE)")
    dry = [
        ".cache/kenken_ckpts/dryrun/property2_step15.jsonl",
        ".cache/kenken_ckpts/dryrun/property2_step30.jsonl",
    ]
    schema_ok = True
    for p in dry:
        if not os.path.exists(p):
            print(f"  MISSING: {p}  -> FAIL")
            schema_ok = False
            continue
        try:
            recs = load_property2_jsonl(p)
        except Exception as e:
            print(f"  PARSE ERROR on {p}: {e}  -> FAIL")
            schema_ok = False
            continue
        res = run_full_analysis(recs, K_CEILING_DEFAULT, n_perm=2000, n_boot=2000, seed=3)
        v = res["verdict"]["verdict"]
        bc = sorted({int(r["breath_count"]) for r in recs})
        statuses = sorted({str(r["status"]) for r in recs})
        has_core = any("core" in r for r in recs)
        # The LIVE trainer's JSON-floor-only records LACK breath_count_min -> the
        # analyzer MUST take the JSD-floor FALLBACK and flag it (Task-2 MODE B).
        instrument = res.get("instrument")
        fallback_ok = (instrument == "jsd_floor_fallback")
        # Untrained -> all breath_count==2, all stuck -> no spread + no settled -> UNTESTABLE.
        deg_ok = (v == "UNTESTABLE") and fallback_ok
        print(f"  {os.path.basename(p)}: parsed {len(recs)} recs; "
              f"breath_counts={bc}; statuses={statuses}; core_field_present={has_core}; "
              f"instrument={instrument}; verdict={v}  -> {'PASS' if deg_ok else 'FAIL'}")
        if not deg_ok:
            schema_ok = False
    print(f"\n[4] RESULT: dry-run schema parse + degenerate->UNTESTABLE  ->  "
          f"{'PASS' if schema_ok else 'FAIL'}")
    if not schema_ok:
        failures.append("dry-run schema parse failed or did not read degenerate as UNTESTABLE")
    print()

    # ---------- 5. U-CURVE RECOVERY (the #238 instrument test, REAL instrument) ----------
    print("-" * 92)
    print("[5] U-CURVE RECOVERY: belief CORRECT+settled at breath 4 then DRIFTS (final wrong).")
    print("    (i) JSD-floor instrument -> not_converged / late;  (ii) min-based -> settled @ ~4, correct_min=True.")
    # Put the repo root on sys.path so `mycelium` imports when run as a script
    # (CPU-only: importing kenken does NOT touch the GPU; only realizing real
    # Tensors on a device would — and the self-test never does that).
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from mycelium.kenken import convergence_instrument as _ci, N_MAX as _NMAX
    n_max = int(_NMAX)
    n_cells = n_max * n_max
    K = K_CEILING_DEFAULT
    # A small 5x5 puzzle's worth of valid cells; gold values 1..n_max cycled.
    Nside = 5
    valid_idx = [r * n_max + c for r in range(Nside) for c in range(Nside)]
    gold_vals = np.zeros(n_cells, dtype=np.int32)
    for ci, cell in enumerate(valid_idx):
        gold_vals[cell] = (ci % n_max) + 1
    cell_valid = np.zeros(n_cells, dtype=np.float32)
    cell_valid[valid_idx] = 1.0
    # input_cells: a couple of GIVENs (so the supervised/gold paths are exercised);
    # given cells equal gold so they don't affect correctness.
    input_cells = np.zeros(n_cells, dtype=np.int32)
    for cell in valid_idx[:3]:
        input_cells[cell] = int(gold_vals[cell])
    settle = 4
    history = _build_ucurve_logits(K, n_cells, valid_idx, gold_vals,
                                   settle_breath_1based=settle, n_max=n_max)
    fake_batch = _FakeBatch(cell_valid.reshape(1, n_cells),
                            gold_vals.reshape(1, n_cells),
                            input_cells.reshape(1, n_cells),
                            deduction_depth=[settle])
    recs5 = _ci(history, fake_batch, threshold=0.01)
    r5 = recs5[0]
    print(f"    instrument record: {r5}")
    # (i) JSD-floor: NOT settled at the true minimum. Under the U it is either
    #     not_converged OR (if some late tail breath drops below thr) converged LATE
    #     (breath_count > settle+1) and STUCK (final wrong). Either way NOT 'settled'
    #     at the minimum.
    jsdfloor_censors = (r5["status"] != "settled") or (r5["breath_count"] > settle + 1)
    # (ii) min-based RECOVERS: settled at ~settle (within +/-1), correct_min True.
    min_recovers = (r5["status_min"] == "settled" and bool(r5["correct_min"]) and
                    abs(int(r5["breath_count_min"]) - settle) <= 1)
    # The final breath is genuinely wrong (the drift censored the JSD-floor view).
    final_wrong = (not bool(r5["correct"]))
    ucurve_ok = bool(jsdfloor_censors and min_recovers and final_wrong)
    print(f"    JSD-floor: status={r5['status']} breath_count={r5['breath_count']} "
          f"correct(final)={r5['correct']}  -> censors_settled={jsdfloor_censors}")
    print(f"    MIN-based: status_min={r5['status_min']} breath_count_min={r5['breath_count_min']} "
          f"correct_min={r5['correct_min']}  -> recovers={min_recovers}")
    print(f"    companions: breath_count_min_ce={r5['breath_count_min_ce']} "
          f"breath_count_frac={r5['breath_count_frac']}")
    print(f"\n[5] RESULT: JSD-floor CENSORS (status!=settled or late)={jsdfloor_censors}; "
          f"final-breath wrong={final_wrong}; MIN-based RECOVERS settled@~{settle}="
          f"{min_recovers}  ->  {'PASS' if ucurve_ok else 'FAIL'}")
    if not ucurve_ok:
        failures.append("U-curve recovery: min-based did not recover the puzzle the JSD-floor censors")
    print()

    # ---------- 6. MIN-BASED breath_count TRACKS DEPTH (analyzer rho, tail-drift immune) ----------
    print("-" * 92)
    print("[6] DEPTH-TRACKING: settle breath increases with a difficulty knob; tail-drift")
    print("    is added AFTER settle. min-based breath_count_min must TRACK depth; the")
    print("    analyzer's min-based rho must detect it while the JSD-floor secondary is contaminated.")
    rng6 = np.random.default_rng(2026)
    recs6: list[dict] = []
    K6 = K_CEILING_DEFAULT
    # For each N bin, draw enough puzzles; depth-knob maps to a settle breath via a
    # positive slope; per puzzle we BUILD belief sequences (settle at depth-driven
    # breath, then RANDOM tail drift) and run the REAL instrument -> record.
    for N in N_BINS:
        # depth domain per bin: small spread, deeper -> later settle.
        n_pz = 70
        depths = rng6.integers(2, 9, size=n_pz)          # difficulty knob 2..8
        for d in depths:
            # settle breath grows with depth: settle = clip(2 + d, 2, K6-3) so there
            # is room for a tail. Add +-1 jitter so it is not a perfect step.
            settle_b = int(np.clip(2 + d + rng6.integers(-1, 2), 2, K6 - 3))
            Nside = N
            vidx = [r * n_max + c for r in range(Nside) for c in range(Nside)]
            gvals = np.zeros(n_cells, dtype=np.int32)
            for ci, cell in enumerate(vidx):
                gvals[cell] = (ci % n_max) + 1
            cvalid = np.zeros(n_cells, dtype=np.float32)
            cvalid[vidx] = 1.0
            icells = np.zeros(n_cells, dtype=np.int32)
            for cell in vidx[:2]:
                icells[cell] = int(gvals[cell])
            # random tail-drift target per puzzle (depth-INDEPENDENT noise after settle).
            wbase = int(rng6.integers(1, n_max))
            def _twf(cell, gv, _wbase=wbase):
                w = ((gv - 1 + _wbase) % n_max) + 1
                return w
            hist = _build_ucurve_logits(K6, n_cells, vidx, gvals,
                                        settle_breath_1based=settle_b, n_max=n_max,
                                        tail_wrong_value_fn=_twf)
            fb = _FakeBatch(cvalid.reshape(1, n_cells), gvals.reshape(1, n_cells),
                            icells.reshape(1, n_cells), deduction_depth=[int(d)])
            rec = _ci(hist, fb, threshold=0.01)[0]
            rec["N"] = int(N)
            rec["n_givens"] = 2
            recs6.append(rec)
    # Sanity: the min instrument should track the settle breath (== depth-driven).
    res6 = run_full_analysis(recs6, K6, N_PERM, N_BOOT, seed=17)
    print_summary(res6)
    # PRIMARY (min-based) should fire a positive link in >=2/3 power bins.
    min_fires = 0
    for br in res6["bins"]:
        s = br["settled_only"]
        if br["has_power"] and np.isfinite(s["p_perm"]) and s["p_perm"] < 0.01 \
                and np.isfinite(s["ci_lower"]) and s["ci_lower"] > WEAK_RHO_LO:
            min_fires += 1
    # The analyzer must be using the MIN instrument (binding read), and the field
    # must be breath_count_min.
    instrument_is_min = (res6.get("instrument") == "min_based_primary"
                         and res6["bins"][0].get("breath_field") == PRIMARY_BREATH_FIELD)
    v6 = res6["verdict"]["verdict"]
    track_ok = bool(instrument_is_min and min_fires >= 2 and
                    (v6.startswith("HILL") or v6 == "WEAK"))
    print(f"\n[6] RESULT: instrument=min_based={instrument_is_min}; "
          f"min-based power bins with p<0.01 & lower-CI>{WEAK_RHO_LO}={min_fires}/3 (want >=2); "
          f"verdict={v6}  ->  {'PASS' if track_ok else 'FAIL'}")
    if not track_ok:
        failures.append("depth-tracking: analyzer min-based rho did not detect the settle-breath/depth link")
    print()

    # ---------- summary ----------
    print("#" * 92)
    if failures:
        print(f"# SELF-TEST: {len(failures)} FAILURE(S):")
        for fmsg in failures:
            print(f"#   - {fmsg}")
        print("#" * 92)
        return 1
    print("# SELF-TEST: ALL PASS (strong fires, null does not false-positive, "
          "ceiling attenuates, schema parses, U-curve recovers, min tracks depth)")
    print("#" * 92)
    return 0


# ===========================================================================
# CLI
# ===========================================================================

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="POST-HOC binding analyzer for KenKen Property 2 (adaptive-depth telegraph).")
    ap.add_argument("ckpt", nargs="?", default=None,
                    help="checkpoint path (MODE A); ignored when --jsonl is given")
    ap.add_argument("--corpus", default=DEFAULT_TRAIN_CORPUS,
                    help=f"corpus for MODE A (default TRAIN binding corpus: {DEFAULT_TRAIN_CORPUS})")
    ap.add_argument("--jsonl", default=None,
                    help="analyze an EXISTING property2 records JSONL (MODE B; skips the model)")
    ap.add_argument("--out", default=None, help="verdict JSON output path")
    ap.add_argument("--records-out", default=None,
                    help="MODE A raw per-puzzle records JSONL (default beside the verdict)")
    ap.add_argument("--K", type=int, default=K_CEILING_DEFAULT,
                    help=f"K_max / breath ceiling (default {K_CEILING_DEFAULT})")
    ap.add_argument("--n-perm", type=int, default=N_PERM_DEFAULT)
    ap.add_argument("--n-boot", type=int, default=N_BOOT_DEFAULT)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--selftest", action="store_true",
                    help="run CPU-only synthetic + schema self-tests (no GPU / no model)")
    args = ap.parse_args(argv)

    if args.selftest:
        return run_selftest()

    # ---- MODE B: analyze an existing JSONL ----
    if args.jsonl:
        records = load_property2_jsonl(args.jsonl)
        print(f"[MODE B] loaded {len(records)} property2 records from {args.jsonl}")
        out_path = args.out or os.path.join(os.path.dirname(os.path.abspath(args.jsonl)),
                                            "property2_verdict.json")
    # ---- MODE A: run the model over the corpus ----
    else:
        if not args.ckpt:
            ap.error("MODE A requires a checkpoint path (or pass --jsonl for MODE B, "
                     "or --selftest for the self-test)")
        ckpt_dir = os.path.dirname(os.path.abspath(args.ckpt))
        records_out = args.records_out or os.path.join(
            ckpt_dir, "property2_analysis_records.jsonl")
        records = run_mode_a(args.ckpt, args.corpus, records_out, K=args.K)
        out_path = args.out or os.path.join(ckpt_dir, "property2_verdict.json")

    result = run_full_analysis(records, args.K, args.n_perm, args.n_boot, args.seed)
    print_summary(result)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(_json_sanitize(result), f, indent=2)
    print(f"\nwrote verdict JSON -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
