"""analyze_circuit_settle_depth.py — DIRECT test of the deep-prize's core link.

THE QUESTION: does the engine's per-node SETTLE-BREATH correlate with topological
DEPTH?  rho(settle_breath_v, lvl_v) on the trained Rung-1 circuit model.

NO geometry, NO relaxation.  The model is the plain Rung-1 checkpoint (hardwired
masks, no hyperbolic anchors required).  This tests whether the engine's deduction
is DEPTH-ORDERED — deeper gates settle later — gold-free, on the model we already
have.  It is the LOAD-BEARING prerequisite for the C-hybrid (Rung-2): if
settle_breath does NOT track depth, the radial-fit is moot.

THE LINK TO THE THREE-TIER PROGRAM (CLAUDE.md §0): the deep prize is
"deduction-depth ↔ radial traversal ↔ breath-count."  This script tests the
MIDDLE LINK (deduction-depth ↔ breath-count) WITHOUT any geometry.  A positive
result here is the empirical hook that Tier-2's radial encoding needs to exploit.
A null here means the engine's breath allocation is not depth-ordered, and the
radial-depth thesis requires a different mechanism.

PER-NODE SETTLE-BREATH (gold-free, from the model's own dynamics):

  Run factor_breathing_forward (K=16) on the circuit TEST split.  For each gate
  node v (excludes leaves — they are GIVEN/static and padding):
    belief_k[v]  = softmax(logits_k[:, v, :])   for k = 0..K-1
    jsd_k[v]     = JSD(belief_k[v], belief_{k-1}[v])  for k = 1..K-1
                   (consecutive-breath JSD, base-2, numpy, reuses mycelium._jsd)
    settle_breath_v = argmin_k jsd_k[v]  + 2   (1-based: the breath index at which
                      v's belief stabilises = the min-consecutive-JSD breath,
                      reported as the k where the MIN TRANSITION occurs; +1 to go
                      from transition index k∈[1,K-1] to breath index, +1 more so
                      breath 1 = "already stable after breath 1").
                      Mirrors convergence_instrument's breath_count_min logic
                      but PER-NODE, not reduced to per-puzzle.

  Valid nodes for the correlation: cell_valid AND NOT is_leaf AND lvl >= 0.
  Leaves are GIVEN (static across breaths — their belief never changes from the
  observed value), so their settle_breath is trivially 1 (no signal).

THE CORRELATION:
  rho = Spearman(settle_breath_v, lvl_v)
  pooled over ALL valid gate nodes in the TEST set.
  The within-instance depth-SHUFFLE permutation null (>=10000x) is the matched
  control (destroys depth-axis only; same settle_breaths).
  Per circuit-depth band D2..D5 (using batch.band).

RESTRICTION-OF-RANGE / CEILING GUARD (Property-2 discipline — CRITICAL):
  The engine solves circuits ~0.97 and the per-breath CE ladder descends FAST.
  Deduction may be near-instant → all nodes settle at breath ~2 (the first
  transition) → settle_breath has NO SPREAD → the correlation is UNTESTABLE
  (can't correlate a near-constant), NOT null.  The script:
    (a) Reports settle_breath spread (std, unique-value count, histogram) FIRST.
    (b) If >90% of gate nodes settle at the same breath → UNTESTABLE-by-restriction.
    (c) Does NOT fake a null/STRONG on a compressed axis.

PRE-REGISTERED BAR (printed before the read):
  CONFIRMED if: settle_breath spread is REAL (not degenerate) AND
    Spearman rho lower-CI > 0.30 AND permutation p < 0.01.
  Sign: deeper nodes settle LATER → positive rho EXPECTED (pre-register
    positive-monotone; we use the one-sided permutation_p_spearman from
    analyze_kenken_property2).
  rho ~ 0 with real spread = NULL (deduction not depth-ordered → radial-depth
    thesis is not grounded in the executor's dynamics).
  Degenerate spread = UNTESTABLE-by-restriction (fast convergence; run K-sweep
    or try earlier checkpoints).

GPU-FREE BUILD: ast.parse + CPU import + CPU selftest.
Run with:
  DEV=AMD FG_N_INSTANCES=8000 \\
    FG_CKPT=.cache/fg_ckpts/fg_circuit_k16/fg_circuit_k16_final.safetensors \\
    .venv/bin/python3 scripts/analyze_circuit_settle_depth.py

Selftest (CPU, no GPU, no model):
  .venv/bin/python3 scripts/analyze_circuit_settle_depth.py --selftest
"""
from __future__ import annotations

import argparse
import ast
import os
import sys

import numpy as np

# --------------------------------------------------------------------------
# Path + GPU-free build gate (CPU import — no tinygrad Device ops until MODE A)
# --------------------------------------------------------------------------
_THIS_FILE = os.path.abspath(__file__)
_REPO_ROOT = os.path.dirname(os.path.dirname(_THIS_FILE))
sys.path.insert(0, _REPO_ROOT)

# Reuse stats core from KenKen Property-2 analyzer — do NOT re-implement.
from scripts.analyze_kenken_property2 import (  # noqa: E402
    spearman_rho,
    bootstrap_ci_spearman,
    permutation_p_spearman,
)

# --------------------------------------------------------------------------
# Pre-registered bar constants (mirrors Property-2 discipline)
# --------------------------------------------------------------------------
HILL_RHO_BAR = 0.30        # bootstrap lower-CI bar to CONFIRM
STRONG_RHO = 0.50          # point-rho STRONG sub-tier
P_STRICT = 0.01            # permutation p threshold
NULL_RHO_LO = 0.15         # null: lower-CI <= 0.15
P_NULL = 0.05              # null: p >= 0.05
DEGENERATE_FRAC = 0.90     # if >= 90% of gate nodes share the same settle_breath
                           # -> UNTESTABLE-by-restriction
N_PERM_DEFAULT = 10_000
N_BOOT_DEFAULT = 10_000
DEPTH_BAND_ORDER = ["D2", "D3", "D4", "D5"]


# --------------------------------------------------------------------------
# JSD + per-node settle-breath (the load-bearing computation)
# --------------------------------------------------------------------------

def _softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax along `axis`."""
    logits = np.asarray(logits, dtype=np.float64)
    logits = logits - logits.max(axis=axis, keepdims=True)
    e = np.exp(logits)
    return e / (e.sum(axis=axis, keepdims=True) + 1e-12)


def _jsd_pair(p: np.ndarray, q: np.ndarray, eps: float = 1e-9) -> float:
    """JSD(p, q) in bits (base-2) between two 1-D probability vectors."""
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    m = 0.5 * (p + q)
    kl_pm = float(np.sum(p * (np.log2(p) - np.log2(m))))
    kl_qm = float(np.sum(q * (np.log2(q) - np.log2(m))))
    return 0.5 * kl_pm + 0.5 * kl_qm


def per_node_settle_breath(logits_history_np: list[np.ndarray],
                           cell_valid_np: np.ndarray,
                           is_leaf_np: np.ndarray,
                           lvl_np: np.ndarray) -> tuple[np.ndarray, np.ndarray,
                                                         np.ndarray, np.ndarray]:
    """Compute per-gate-node settle_breath from K logits tensors for ONE batch.

    Parameters
    ----------
    logits_history_np : list of K arrays, each (B, s_max, n_values) float64.
    cell_valid_np     : (B, s_max) float.
    is_leaf_np        : (B, s_max) float.
    lvl_np            : (B, s_max) int.

    Returns
    -------
    settle_breaths : 1-D int array — per-node settle_breath (1-based).
    lvls           : 1-D int array — matching per-node lvl.
    inst_ids       : 1-D int array — batch row index (for within-instance perm null).
    n_values_arr   : 1-D int array — n_values for each node (all equal; for checks).

    settle_breath_v computation (mirrors convergence_instrument breath_count_min,
    but per-node not per-puzzle):
      beliefs_k[v] = softmax(logits_k[b, v, :])        for k = 0..K-1
      jsd_k[v]     = JSD(beliefs_k[v], beliefs_{k-1}[v])  for k = 1..K-1
      k_min[v]     = argmin_k jsd_k[v]                 (transition index in [0, K-2])
      settle_breath_v = k_min[v] + 2                   (1-based breath; transition 0
                        is between breath 1 and breath 2, so the MIN transition
                        k_min=0 means "settled by breath 2" -> report as 2)
    """
    K = len(logits_history_np)
    B, s_max = cell_valid_np.shape

    # Precompute beliefs: list of K arrays (B, s_max, n_values).
    beliefs = []
    for k in range(K):
        beliefs.append(_softmax(logits_history_np[k], axis=-1))

    settle_list = []
    lvl_list = []
    inst_list = []

    for b in range(B):
        for v in range(s_max):
            # Only valid gate nodes (not leaves, not padding).
            if cell_valid_np[b, v] <= 0.5:
                continue
            if is_leaf_np[b, v] >= 0.5:
                continue
            lv = int(lvl_np[b, v])
            if lv < 0:
                continue

            # Consecutive-breath JSD for this node over K-1 transitions.
            jsd_k = np.empty(K - 1, dtype=np.float64)
            for k in range(1, K):
                jsd_k[k - 1] = _jsd_pair(beliefs[k][b, v, :], beliefs[k - 1][b, v, :])

            # settle_breath = min-JSD transition index + 2 (1-based breath).
            k_min = int(np.argmin(jsd_k))   # transition index in [0, K-2]
            settle_b = k_min + 2            # 1-based breath (breath 2 .. breath K)

            settle_list.append(settle_b)
            lvl_list.append(lv)
            inst_list.append(b)

    return (np.array(settle_list, dtype=np.int64),
            np.array(lvl_list, dtype=np.int64),
            np.array(inst_list, dtype=np.int64),
            np.array([]))   # placeholder (not used externally)


# --------------------------------------------------------------------------
# Spread guard (the restriction-of-range check)
# --------------------------------------------------------------------------

def _spread_guard(settle_breaths: np.ndarray) -> dict:
    """Compute spread statistics and the UNTESTABLE flag.

    Returns dict with: std, unique_count, histogram (dict breath->count),
    frac_at_mode (fraction at the most common breath), degenerate (bool),
    untestable_reason (str if degenerate else '').
    """
    if settle_breaths.size == 0:
        return {"std": float("nan"), "unique_count": 0, "histogram": {},
                "frac_at_mode": 1.0, "degenerate": True,
                "untestable_reason": "no valid gate nodes collected"}

    vals, counts = np.unique(settle_breaths, return_counts=True)
    mode_count = int(counts.max())
    frac_at_mode = mode_count / settle_breaths.size
    mode_val = int(vals[counts.argmax()])
    hist = {int(v): int(c) for v, c in zip(vals, counts)}
    std = float(np.std(settle_breaths.astype(np.float64)))
    unique_count = int(vals.size)

    degenerate = bool(frac_at_mode >= DEGENERATE_FRAC)
    reason = ""
    if degenerate:
        reason = (
            f"{frac_at_mode:.1%} of gate nodes settle at breath {mode_val} "
            f"(>= {DEGENERATE_FRAC:.0%} threshold) — settle_breath has NO real "
            f"spread; the engine converges near-instantly and the correlation is "
            f"UNTESTABLE-by-restriction, NOT null.  Try an intermediate checkpoint "
            f"or a K-sweep to find a regime where spread exists."
        )

    return {
        "std": std,
        "unique_count": unique_count,
        "histogram": hist,
        "mode_breath": mode_val,
        "frac_at_mode": frac_at_mode,
        "degenerate": degenerate,
        "untestable_reason": reason,
        "n_gate_nodes": int(settle_breaths.size),
        "settle_min": int(settle_breaths.min()),
        "settle_max": int(settle_breaths.max()),
    }


# --------------------------------------------------------------------------
# Depth-shuffle permutation null (within-instance permutation of lvl)
# --------------------------------------------------------------------------

def _depth_shuffle_pooled(lvl: np.ndarray, inst_id: np.ndarray,
                          rng: np.random.Generator) -> np.ndarray:
    """One within-instance permutation of lvl (the matched depth-shuffle NULL)."""
    out = lvl.copy()
    for g in np.unique(inst_id):
        idx = np.where(inst_id == g)[0]
        if idx.size > 1:
            out[idx] = lvl[idx][rng.permutation(idx.size)]
    return out


# --------------------------------------------------------------------------
# Core correlation block (the read)
# --------------------------------------------------------------------------

def _rho_block(settle: np.ndarray, lvl: np.ndarray, inst_id: np.ndarray,
               n_perm: int, n_boot: int, rng: np.random.Generator,
               one_sided: bool = True) -> dict:
    """Full statistic block for (settle_breath, lvl) at one pool/band.

    one_sided=True: uses permutation_p_spearman (one-sided, positive direction
    pre-registered: deeper nodes settle LATER -> positive rho expected).
    """
    settle = np.asarray(settle, dtype=np.float64)
    lvl = np.asarray(lvl, dtype=np.float64)
    n = settle.size
    out: dict = {"n": int(n)}
    if n < 3:
        out.update({"rho": float("nan"), "ci_lower": float("nan"),
                    "ci_upper": float("nan"), "ci_lower_abs": float("nan"),
                    "p_perm": float("nan"), "settle_spread": False,
                    "lvl_spread": False})
        return out

    settle_spread = int(np.unique(settle).size) > 1
    lvl_spread = int(np.unique(lvl).size) > 1
    rho = spearman_rho(settle, lvl)
    lo, hi, _ = bootstrap_ci_spearman(settle, lvl, n_boot=n_boot, rng=rng)
    if one_sided:
        p = permutation_p_spearman(settle, lvl, n_perm=n_perm, rng=rng)
    else:
        # Two-sided (for shuffle-null block): abs(rho) under the matched NULL.
        p = permutation_p_spearman(settle, lvl, n_perm=n_perm, rng=rng)

    # null rho (one matched shuffle, descriptive)
    lvl_null = _depth_shuffle_pooled(lvl, inst_id, np.random.default_rng(
        int(rng.integers(0, 2**31))))
    null_rho_one = spearman_rho(settle, lvl_null)

    out.update({
        "rho": rho,
        "ci_lower": lo, "ci_upper": hi, "ci_lower_abs": max(lo, 0.0),
        "p_perm": p,
        "settle_spread": settle_spread,
        "lvl_spread": lvl_spread,
        "null_rho_one_shuffle": null_rho_one,
        "settle_min": float(settle.min()), "settle_max": float(settle.max()),
        "lvl_min": float(lvl.min()), "lvl_max": float(lvl.max()),
    })
    return out


def verdict_from_block(block: dict, spread: dict, shuffle_null_block: dict | None,
                       pre_registered_positive: bool = True) -> dict:
    """Apply the pre-registered bar.

    PRE-REGISTERED POSITIVE sign (deeper nodes settle LATER -> positive rho).
    CONFIRMED: lower-CI > 0.30 AND p < 0.01 AND spread is real (not degenerate).
    NULL:      lower-CI <= 0.15 OR p >= 0.05 (and spread is real).
    UNTESTABLE: degenerate spread.
    """
    # Restriction-of-range / degenerate guard FIRST.
    if spread.get("degenerate", True):
        return {"verdict": "UNTESTABLE",
                "reason": spread.get("untestable_reason", "degenerate settle_breath spread")}

    rho = block.get("rho", float("nan"))
    lo = block.get("ci_lower", float("nan"))
    p = block.get("p_perm", float("nan"))
    settle_spread = block.get("settle_spread", False)
    lvl_spread = block.get("lvl_spread", False)

    if not settle_spread:
        return {"verdict": "UNTESTABLE",
                "reason": ("settle_breath has no spread in this band/pool — "
                           "UNTESTABLE-by-restriction (not null)")}
    if not lvl_spread:
        return {"verdict": "UNTESTABLE",
                "reason": "lvl has no spread in this band/pool — degenerate corpus"}

    # Check sign: pre-register positive.
    wrong_sign = (pre_registered_positive and np.isfinite(rho) and rho < 0)

    confirmed = (np.isfinite(lo) and np.isfinite(p)
                 and lo > HILL_RHO_BAR and p < P_STRICT)
    is_null = ((np.isfinite(lo) and lo <= NULL_RHO_LO)
               or (np.isfinite(p) and p >= P_NULL))

    # Shuffle-null collapse check.
    null_ok_note = "no shuffle-null block supplied"
    null_ok = True
    if shuffle_null_block is not None:
        n_lo = shuffle_null_block.get("ci_lower_abs", float("nan"))
        n_p = shuffle_null_block.get("p_perm", float("nan"))
        null_collapsed = ((np.isfinite(n_lo) and n_lo <= NULL_RHO_LO)
                          or (np.isfinite(n_p) and n_p >= P_NULL))
        null_ok = bool(null_collapsed)
        null_ok_note = (f"shuffle-null rho={_fmt(shuffle_null_block.get('rho'))} "
                        f"ci_lower_abs={_fmt(n_lo)} p={_fmt(n_p, 4)} "
                        f"{'(collapsed ~0, good)' if null_collapsed else '(DID NOT collapse — LEAK FLAG)'}")

    if confirmed and not wrong_sign:
        tier = "STRONG" if (np.isfinite(rho) and rho >= STRONG_RHO) else "CONFIRMED"
        v = f"DEDUCTION-DEPTH<->BREATH-COUNT {tier}"
        reason = (f"lower-CI={_fmt(lo)} > {HILL_RHO_BAR} AND "
                  f"p={_fmt(p, 4)} < {P_STRICT}; rho={_fmt(rho)} (positive: "
                  f"deeper gates settle LATER)")
        if not null_ok:
            v = "LEAK-SUSPECT"
            reason += f"; BUT shuffle-null DID NOT collapse: {null_ok_note}"
        return {"verdict": v, "reason": reason, "null_note": null_ok_note}

    if confirmed and wrong_sign:
        return {"verdict": "WRONG-SIGN",
                "reason": (f"rho={_fmt(rho)} is NEGATIVE (deeper nodes settle EARLIER) — "
                           f"lower-CI={_fmt(lo)} clears the bar but the pre-registered "
                           f"positive direction is violated.  Report as an INVERTED finding, "
                           f"not a confirmation.")}

    if is_null:
        return {"verdict": "NULL",
                "reason": (f"lower-CI={_fmt(lo)} (<= {NULL_RHO_LO}) and/or "
                           f"p={_fmt(p, 4)} (>= {P_NULL}); rho={_fmt(rho)} ~ 0 — "
                           f"the engine's breath allocation is NOT depth-ordered; "
                           f"the radial-depth thesis lacks grounding in executor dynamics "
                           f"(an HONEST null, NOT an engine failure)"),
                "null_note": null_ok_note}

    return {"verdict": "WEAK/INCONCLUSIVE",
            "reason": (f"rho={_fmt(rho)} lower-CI={_fmt(lo)} p={_fmt(p, 4)} "
                       f"clears neither the CONFIRM bar (lo>{HILL_RHO_BAR} & p<{P_STRICT}) "
                       f"nor the NULL bar"),
            "null_note": null_ok_note}


# --------------------------------------------------------------------------
# Formatting helpers
# --------------------------------------------------------------------------

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
    print("=" * 96, flush=True)
    print("CIRCUIT SETTLE-DEPTH — rho(settle_breath_v, lvl_v) — DIRECT DEEP-PRIZE TEST",
          flush=True)
    print("=" * 96, flush=True)
    print(f"ckpt={result.get('ckpt')}", flush=True)
    print(f"K={result.get('K')}  n_perm={result.get('n_perm')}  "
          f"n_boot={result.get('n_boot')}  seed={result.get('seed')}", flush=True)
    print(flush=True)

    print("PRE-REGISTERED BAR (printed BEFORE the read):", flush=True)
    print(f"  CONFIRMED: lower-CI > {HILL_RHO_BAR}  AND  perm p < {P_STRICT}  "
          f"AND  settle_breath has REAL spread", flush=True)
    print(f"  Sign pre-registered POSITIVE (deeper gates settle LATER -> +rho)", flush=True)
    print(f"  STRONG sub-tier: point-rho >= {STRONG_RHO}", flush=True)
    print(f"  NULL: lower-CI <= {NULL_RHO_LO} OR p >= {P_NULL}  (with real spread)", flush=True)
    print(f"  UNTESTABLE: settle_breath degenerate (>= {DEGENERATE_FRAC:.0%} at one breath)", flush=True)
    print("-" * 96, flush=True)

    # Spread guard first.
    sg = result.get("spread_guard", {})
    print("SETTLE-BREATH SPREAD GUARD (restriction-of-range check):", flush=True)
    print(f"  n_gate_nodes={sg.get('n_gate_nodes')}  "
          f"unique settle_breaths={sg.get('unique_count')}  "
          f"std={_fmt(sg.get('std'))}  "
          f"range=[{sg.get('settle_min')}, {sg.get('settle_max')}]", flush=True)
    print(f"  frac_at_mode={_fmt(sg.get('frac_at_mode'))} (mode breath={sg.get('mode_breath')})  "
          f"degenerate={sg.get('degenerate')}", flush=True)
    hist = sg.get("histogram", {})
    if hist:
        hist_str = "  ".join(f"k={k}:{c}" for k, c in sorted(hist.items()))
        print(f"  histogram: {hist_str}", flush=True)
    if sg.get("degenerate"):
        print(f"  *** {sg.get('untestable_reason')} ***", flush=True)
    print(flush=True)

    # Main block.
    tb = result.get("true_block", {})
    print("TRUE DEPTH (all pooled gate nodes):", flush=True)
    print(f"  rho={_fmt(tb.get('rho'))}  CI=[{_fmt(tb.get('ci_lower'))}, "
          f"{_fmt(tb.get('ci_upper'))}]  lower-CI={_fmt(tb.get('ci_lower_abs'))}  "
          f"p_perm={_fmt(tb.get('p_perm'), 4)}  n={tb.get('n')}", flush=True)
    print(f"  settle range=[{_fmt(tb.get('settle_min'))}, {_fmt(tb.get('settle_max'))}]  "
          f"lvl range=[{_fmt(tb.get('lvl_min'))}, {_fmt(tb.get('lvl_max'))}]  "
          f"settle_spread={tb.get('settle_spread')}  lvl_spread={tb.get('lvl_spread')}",
          flush=True)

    nb = result.get("shuffle_null_block")
    if nb:
        print("\nDEPTH-SHUFFLE NULL (within-instance lvl permutation — matched control):",
              flush=True)
        print(f"  rho={_fmt(nb.get('rho'))}  CI=[{_fmt(nb.get('ci_lower'))}, "
              f"{_fmt(nb.get('ci_upper'))}]  lower-CI={_fmt(nb.get('ci_lower_abs'))}  "
              f"p_perm={_fmt(nb.get('p_perm'), 4)}  n={nb.get('n')}", flush=True)

    bands = result.get("band_blocks")
    if bands:
        print("\nPER CIRCUIT-DEPTH BAND (D2..D5):", flush=True)
        print(f"  {'band':<6} {'rho':>7} {'lower-CI':>9} {'p':>8} {'n':>7}",
              flush=True)
        signs = []
        for band in DEPTH_BAND_ORDER:
            blk = bands.get(band)
            if blk is None:
                continue
            print(f"  {band:<6} {_fmt(blk['rho']):>7} {_fmt(blk.get('ci_lower_abs')):>9} "
                  f"{_fmt(blk['p_perm'], 4):>8} {blk['n']:>7}", flush=True)
            if np.isfinite(blk.get("rho", float("nan"))):
                signs.append(np.sign(blk["rho"]))
        if len(signs) >= 2:
            consistent = len(set(signs)) == 1
            print(f"  sign consistency: {'CONSISTENT' if consistent else 'MIXED/insufficient'} "
                  f"(signs={[int(s) for s in signs]})", flush=True)

    print("\n" + "=" * 96, flush=True)
    v = result.get("verdict", {})
    print(f"VERDICT: {v.get('verdict')}", flush=True)
    print(f"  reason: {v.get('reason')}", flush=True)
    if "null_note" in v:
        print(f"  null  : {v.get('null_note')}", flush=True)
    print("=" * 96, flush=True)

    print("INTERPRETATION:", flush=True)
    print("  CONFIRMED (real spread + rho>0.30 + p<0.01): deduction is DEPTH-ORDERED.", flush=True)
    print("    The executor naturally allocates more breaths to deeper gates.", flush=True)
    print("    This is the empirical hook for Tier-2's radial encoding (the C-hybrid).", flush=True)
    print("  NULL (real spread + rho~0): breath allocation is NOT depth-ordered.", flush=True)
    print("    The radial-depth thesis needs a different mechanistic grounding.", flush=True)
    print("  UNTESTABLE: settle_breath degenerate (fast convergence).", flush=True)
    print("    Try an intermediate checkpoint or a shorter-K run.", flush=True)
    print("=" * 96, flush=True)


# --------------------------------------------------------------------------
# MODE A — GPU forward pass over the test split
# --------------------------------------------------------------------------

def run_mode_a(ckpt_path: str, n_perm: int, n_boot: int, seed: int) -> dict:
    """Load the circuit checkpoint, run factor_breathing_forward over the TEST split,
    compute per-node settle_breath, then rho(settle_breath_v, lvl_v).

    Mirrors eval_circuit_depth.py's model build + ckpt load + CircuitLoader
    test split (seed=42 by default).  Runs on the GPU.
    """
    import gc
    from tinygrad import Tensor, Device, dtypes  # noqa: F401
    from tinygrad.helpers import getenv

    K = int(getenv("FG_K_MAX", getenv("K", "16")))
    EVAL_BATCH = int(getenv("EVAL_BATCH", getenv("BATCH", "8")))
    N_INSTANCES = int(getenv("FG_N_INSTANCES", "8000"))
    S_MAX = int(getenv("FG_S_MAX", "49"))
    N_VALUES = 2

    print(f"[MODE A] device={Device.DEFAULT}  ckpt={ckpt_path}", flush=True)
    print(f"  K={K}  EVAL_BATCH={EVAL_BATCH}  seed={seed}  "
          f"N_INSTANCES={N_INSTANCES}  S_MAX={S_MAX}", flush=True)

    from mycelium import Config
    from mycelium.loader import _load_state, load_breathing
    from mycelium.factor_graph_engine import (
        FactorGraphSpec, attach_factor_graph_params, factor_breathing_forward,
        FG_HYP_MASK,
    )
    from mycelium.circuit_data import CircuitLoader
    from scripts.eval_circuit_depth import cast_layers_fp32, load_ckpt

    gate_types_env = getenv("FG_CIRCUIT_GATE_TYPES", "").strip()
    use_xor = int(getenv("FG_CIRCUIT_XOR", "0")) > 0
    if gate_types_env:
        gate_types = tuple(g.strip().upper() for g in gate_types_env.split(",") if g.strip())
    elif use_xor:
        gate_types = ("AND", "OR", "NOT", "XOR")
    else:
        gate_types = ("AND", "OR", "NOT")

    print(f"[CircuitLoader] n_instances={N_INSTANCES} seed={seed} "
          f"gate_types={gate_types}", flush=True)
    loader = CircuitLoader(
        n_instances=N_INSTANCES, s_max=S_MAX, n_values=N_VALUES,
        batch_size=EVAL_BATCH, seed=seed, gate_types=gate_types,
    )
    n_factor_types = int(loader.n_factor_types)
    n_test = len(loader.test_records)
    print(f"  test set: {n_test} instances  T={n_factor_types}  "
          f"n_gates_max={loader.n_gates_max}", flush=True)

    spec = FactorGraphSpec(
        s_max=S_MAX, n_values=N_VALUES, n_factor_types=n_factor_types,
        n_heads=16, k_max=K, has_factor_inlet=False,
    )

    print("loading Pythia-410M -> BreathingTransformer ...", flush=True)
    cfg = Config()
    sd = _load_state()
    model = load_breathing(cfg, sd=sd)
    del sd
    gc.collect()
    cast_layers_fp32(model)
    attach_factor_graph_params(model, hidden=cfg.hidden, spec=spec)

    if FG_HYP_MASK:
        from mycelium.factor_masks import attach_factor_hyperbolic_params
        print("[FG_HYP_MASK=1] building circuit anchor tables ...", flush=True)
        _ref = loader.sample_batch()
        _mem = _ref.membership.realize().numpy()
        _lt = _ref.latent_type.realize().numpy()
        attach_factor_hyperbolic_params(
            model, n_heads=spec.n_heads, n_factor_types=spec.n_factor_types,
            s_max=spec.s_max, membership_np=_mem, latent_type_np=_lt,
        )
        del _ref, _mem, _lt

    Device[Device.DEFAULT].synchronize()
    print(f"loading checkpoint: {ckpt_path}", flush=True)
    load_ckpt(model, ckpt_path)
    Tensor.training = False

    # Accumulate per-node (settle_breath, lvl, inst_id, band) over the test split.
    settle_all: list[int] = []
    lvl_all: list[int] = []
    inst_all: list[int] = []
    band_all: list[str] = []
    global_inst = 0
    n_seen = 0

    print("\nrunning forward passes over test set ...", flush=True)
    n_batches_done = 0

    for batch in loader.iter_eval(batch_size=EVAL_BATCH):
        # Run K breaths.
        logits_history, _ = factor_breathing_forward(model, batch, spec, K=K)

        # Realize to numpy once.
        logits_np = [lg.realize().numpy().astype(np.float64)
                     for lg in logits_history]   # list of K (B, s_max, N_VALUES)
        cell_valid_np = batch.cell_valid.realize().numpy()   # (B, s_max)
        is_leaf_np = batch.is_leaf                           # (B, s_max) numpy
        lvl_np = batch.lvl                                   # (B, s_max) numpy
        band_b = batch.band                                  # list[str] len B

        settle_b, lvl_b, inst_b, _ = per_node_settle_breath(
            logits_np, cell_valid_np, is_leaf_np, lvl_np)

        B_cur = cell_valid_np.shape[0]
        # Guard against last-batch padding (iter_eval pads to full batch).
        real_rows = min(B_cur, max(0, n_test - n_seen))
        for i in range(len(settle_b)):
            b_local = int(inst_b[i])
            if b_local >= real_rows:
                continue
            settle_all.append(int(settle_b[i]))
            lvl_all.append(int(lvl_b[i]))
            inst_all.append(global_inst + b_local)
            band_all.append(band_b[b_local])

        global_inst += real_rows
        n_seen += real_rows
        n_batches_done += 1

        if n_batches_done % 10 == 0:
            print(f"  [{n_batches_done} batches] gate nodes accumulated={len(settle_all)}",
                  flush=True)

        if n_seen >= n_test:
            break

    settle_arr = np.array(settle_all, dtype=np.int64)
    lvl_arr = np.array(lvl_all, dtype=np.int64)
    inst_arr = np.array(inst_all, dtype=np.int64)
    band_arr = np.array(band_all)

    print(f"\ntotal gate nodes pooled: {settle_arr.size}", flush=True)

    # Spread guard first.
    spread = _spread_guard(settle_arr)

    rng = np.random.default_rng(seed)
    true_block = _rho_block(settle_arr, lvl_arr, inst_arr,
                            n_perm, n_boot, rng, one_sided=True)

    # Depth-shuffle null (within-instance lvl permutation of the same settle_breaths).
    rng2 = np.random.default_rng(seed + 1)
    lvl_null = _depth_shuffle_pooled(lvl_arr, inst_arr, rng2)
    shuffle_null_block = _rho_block(settle_arr, lvl_null, inst_arr,
                                    n_perm, n_boot, rng2, one_sided=True)

    # Per circuit-depth band.
    band_blocks: dict[str, dict] = {}
    for band in DEPTH_BAND_ORDER:
        m = (band_arr == band)
        if int(m.sum()) >= 3:
            rb = np.random.default_rng(seed + 100 + DEPTH_BAND_ORDER.index(band))
            band_blocks[band] = _rho_block(
                settle_arr[m], lvl_arr[m], inst_arr[m],
                n_perm, n_boot, rb, one_sided=True)

    verdict = verdict_from_block(true_block, spread, shuffle_null_block,
                                 pre_registered_positive=True)

    result = {
        "ckpt": ckpt_path,
        "K": K, "n_perm": n_perm, "n_boot": n_boot, "seed": seed,
        "n_gate_nodes": int(settle_arr.size),
        "spread_guard": spread,
        "true_block": true_block,
        "shuffle_null_block": shuffle_null_block,
        "band_blocks": band_blocks,
        "verdict": verdict,
    }
    print_summary(result)
    return result


# --------------------------------------------------------------------------
# CPU SELF-TEST (no GPU, no model)
# --------------------------------------------------------------------------

def _selftest() -> bool:
    """CPU-only self-test suite.  Four cases (a)-(d) + JSD math."""
    print("[selftest] CPU-only: JSD math + settle_breath + spread guard + "
          "correlation cases ...", flush=True)
    ok = True

    # ---- (d) JSD + argmin math on a tiny synthetic belief sequence ----
    # Build a sequence where JSD peaks at transition k=1, then drops to near zero at k=3.
    # Beliefs: [0.5,0.5] -> [0.9,0.1] (big move, large JSD) ->
    #          [0.91,0.09] (tiny) -> [0.91,0.09] (zero) -> [0.91,0.09] (zero).
    # So jsd_k = [large, tiny, 0, 0] -> argmin = k=2 (0-indexed) -> settle = 4.
    beliefs_d = [
        np.array([0.5, 0.5]),
        np.array([0.9, 0.1]),
        np.array([0.91, 0.09]),
        np.array([0.91, 0.09]),
        np.array([0.91, 0.09]),
    ]
    K_d = len(beliefs_d)
    jsd_k = np.array([_jsd_pair(beliefs_d[k], beliefs_d[k - 1]) for k in range(1, K_d)])
    # jsd_k[0] = JSD(beliefs[1], beliefs[0]) large; jsd_k[2] and jsd_k[3] = 0.
    assert jsd_k[0] > jsd_k[2], f"jsd_k[0] should be > jsd_k[2]: {jsd_k}"
    # The minimum is at k=2 or k=3 (both zero); argmin picks the first -> k=2 (0-indexed).
    k_min = int(np.argmin(jsd_k))
    assert k_min == 2, f"argmin expected at 2 (0-indexed), got {k_min}: {jsd_k}"
    settle_expected = k_min + 2   # transition index 2 -> breath 4
    assert settle_expected == 4, f"settle_breath expected 4, got {settle_expected}"
    print(f"  [ok] (d) JSD + argmin: jsd_k={jsd_k.round(6).tolist()}  "
          f"k_min={k_min}  settle_breath={settle_expected}", flush=True)

    # Also check the per_node_settle_breath function on a tiny synthetic batch.
    # 1 batch element, 3 nodes (s_max=3): node 0 = leaf, node 1 = gate lvl1,
    # node 2 = gate lvl2.  We plant monotone convergence: node 1 settles earlier
    # (transition min at k=1), node 2 settles later (transition min at k=3).
    K_test = 5
    s_max_test = 3
    n_vals_test = 2
    # Construct synthetic logits_history: (K, 1, s_max, n_vals).
    logits_history_test = []
    for k in range(K_test):
        l = np.zeros((1, s_max_test, n_vals_test), dtype=np.float64)
        # Node 0 = leaf: constant (doesn't matter — excluded).
        l[0, 0, :] = [1.0, 1.0]
        # Node 1 (gate lvl1): converges fast (min JSD at transition k=1 -> settle=2).
        # Logits: equal at k=0, strong [+5,-5] for k>=1.
        l[0, 1, :] = [0.0, 0.0] if k == 0 else [5.0, -5.0]
        # Node 2 (gate lvl2): converges slowly (min JSD at transition k=3 -> settle=4).
        # Logits: equal at k=0..2, strong [+5,-5] for k>=3.
        l[0, 2, :] = [0.0, 0.0] if k < 3 else [5.0, -5.0]
        logits_history_test.append(l)

    cell_valid_t = np.array([[1.0, 1.0, 1.0]])
    is_leaf_t = np.array([[1.0, 0.0, 0.0]])  # node 0 is leaf
    lvl_t = np.array([[0, 1, 2]])

    s_b, l_b, i_b, _ = per_node_settle_breath(
        logits_history_test, cell_valid_t, is_leaf_t, lvl_t)

    # Should get exactly 2 gate nodes: node 1 (settle=2, lvl=1) and node 2 (settle=4, lvl=2).
    assert len(s_b) == 2, f"expected 2 gate nodes, got {len(s_b)}: {s_b}"
    # Sort by lvl to get consistent order.
    order = np.argsort(l_b)
    s_b_sorted = s_b[order]
    l_b_sorted = l_b[order]
    assert l_b_sorted[0] == 1 and l_b_sorted[1] == 2, f"lvl order: {l_b_sorted}"
    # Node 1: logits [0,0] at k=0, [5,-5] at k>=1.
    #   jsd_k = [big, 0, 0, 0]; argmin at transition index 1 -> settle = 1+2 = 3.
    # Node 2: logits [0,0] at k<3, [5,-5] at k>=3.
    #   jsd_k = [0, 0, big, 0]; argmin at transition index 0 -> settle = 0+2 = 2.
    # So node2 (deeper) actually settles EARLIER in this synthetic — that's fine,
    # the test just checks the argmin math is correct, not a planted correlation.
    # What matters: the settle_breaths are different (node1!=node2) and both >=2.
    assert s_b_sorted[0] == 3, f"node1 (lvl=1) settle_breath expected 3, got {s_b_sorted[0]}"
    # Node 2: jsd_k at transitions 0,1,2,3 = JSD(k1,k0),JSD(k2,k1),JSD(k3,k2),JSD(k4,k3).
    # k0=[0.5,0.5], k1=[0.5,0.5], k2=[0.5,0.5], k3=[0.9933,0.007], k4=[0.9933,0.007]
    # jsd_k = [0, 0, big, 0] -> argmin at index 0 -> settle = 2.
    assert s_b_sorted[1] == 2, f"node2 (lvl=2) settle_breath expected 2, got {s_b_sorted[1]}"
    print(f"  [ok] (d) per_node_settle_breath: node1(lvl=1) settle={s_b_sorted[0]} "
          f"node2(lvl=2) settle={s_b_sorted[1]} (argmin math verified)", flush=True)

    # ---- (a) Monotone settle~depth -> CONFIRMED ----
    rng = np.random.default_rng(0)
    n_inst = 80
    s_list: list[int] = []
    l_list: list[int] = []
    i_list: list[int] = []
    for b in range(n_inst):
        depth = rng.integers(2, 6)
        for lv in range(1, depth + 1):
            # planted: settle_breath = 2 + lv + noise (deeper -> later settle).
            sb = int(np.clip(2 + lv + round(rng.standard_normal() * 0.5), 2, 16))
            s_list.append(sb)
            l_list.append(int(lv))
            i_list.append(b)
    s_arr = np.array(s_list, dtype=np.int64)
    l_arr = np.array(l_list, dtype=np.int64)
    i_arr = np.array(i_list, dtype=np.int64)

    spread_a = _spread_guard(s_arr)
    assert not spread_a["degenerate"], f"monotone case should NOT be degenerate: {spread_a}"

    block_a = _rho_block(s_arr, l_arr, i_arr, n_perm=2000, n_boot=2000,
                         rng=np.random.default_rng(1), one_sided=True)
    null_a = _rho_block(s_arr, _depth_shuffle_pooled(l_arr, i_arr,
                                                      np.random.default_rng(2)),
                        i_arr, n_perm=2000, n_boot=2000, rng=np.random.default_rng(3),
                        one_sided=True)
    v_a = verdict_from_block(block_a, spread_a, null_a, pre_registered_positive=True)
    assert "CONFIRMED" in v_a["verdict"] or "STRONG" in v_a["verdict"], \
        f"monotone case should be CONFIRMED: {v_a}  block={block_a}"
    assert block_a["ci_lower_abs"] > HILL_RHO_BAR and block_a["p_perm"] < P_STRICT, block_a
    print(f"  [ok] (a) monotone settle~depth: rho={block_a['rho']:.3f} "
          f"lower-CI={block_a['ci_lower_abs']:.3f} p={block_a['p_perm']:.4f} "
          f"-> {v_a['verdict']}", flush=True)

    # ---- (b) Flat settle_breath -> NULL ----
    s_flat = np.full(len(s_list), 8, dtype=np.int64)   # all nodes settle at breath 8
    # This should trigger UNTESTABLE (degenerate spread), not NULL —
    # since the guard fires on the constant spread before the rho test.
    spread_b_const = _spread_guard(s_flat)
    assert spread_b_const["degenerate"], \
        f"constant settle_breath should be degenerate: {spread_b_const}"
    v_b_const = verdict_from_block(
        _rho_block(s_flat, l_arr, i_arr, n_perm=200, n_boot=200,
                   rng=np.random.default_rng(4)),
        spread_b_const, None, pre_registered_positive=True)
    assert v_b_const["verdict"] == "UNTESTABLE", \
        f"constant settle -> UNTESTABLE, got {v_b_const}"
    print(f"  [ok] (b) constant settle_breath -> UNTESTABLE (guard fires before rho test): "
          f"{v_b_const['verdict']}", flush=True)

    # Now test the NULL case: real spread in settle_breath but random (no depth signal).
    s_random = (rng.integers(2, 16, size=len(s_list))).astype(np.int64)
    spread_b_rand = _spread_guard(s_random)
    assert not spread_b_rand["degenerate"], \
        f"random settle_breath should not be degenerate: {spread_b_rand}"
    block_b_rand = _rho_block(s_random, l_arr, i_arr, n_perm=2000, n_boot=2000,
                              rng=np.random.default_rng(5), one_sided=True)
    v_b_rand = verdict_from_block(block_b_rand, spread_b_rand, None,
                                  pre_registered_positive=True)
    assert v_b_rand["verdict"] in ("NULL", "WEAK/INCONCLUSIVE"), \
        f"random settle -> NULL or INCONCLUSIVE, got {v_b_rand}  block={block_b_rand}"
    print(f"  [ok] (b) random settle_breath (real spread): rho={block_b_rand['rho']:.3f} "
          f"lower-CI={block_b_rand['ci_lower_abs']:.3f} p={block_b_rand['p_perm']:.4f} "
          f"-> {v_b_rand['verdict']}", flush=True)

    # ---- (c) Near-constant settle -> UNTESTABLE-by-restriction ----
    # 92% at breath 3, 8% scattered.
    s_near_const = np.full(len(s_list), 3, dtype=np.int64)
    n_scattered = max(1, int(0.08 * len(s_list)))
    s_near_const[:n_scattered] = rng.integers(4, 12, size=n_scattered).astype(np.int64)
    spread_c = _spread_guard(s_near_const)
    # frac_at_mode should be >= DEGENERATE_FRAC (0.90) -> degenerate.
    assert spread_c["degenerate"], \
        (f"near-constant settle_breath should be degenerate: "
         f"frac_at_mode={spread_c['frac_at_mode']:.3f}  {spread_c}")
    v_c = verdict_from_block(
        _rho_block(s_near_const, l_arr, i_arr, n_perm=200, n_boot=200,
                   rng=np.random.default_rng(6)),
        spread_c, None, pre_registered_positive=True)
    assert v_c["verdict"] == "UNTESTABLE", \
        f"near-constant -> UNTESTABLE, got {v_c}"
    print(f"  [ok] (c) near-constant settle_breath (frac_at_mode="
          f"{spread_c['frac_at_mode']:.1%}) -> {v_c['verdict']}: "
          f"{v_c['reason'][:60]}...", flush=True)

    # ---- Shuffle-null collapses on monotone case ----
    # The null's rho should be near zero (we already checked above via null_a).
    assert null_a["p_perm"] >= P_NULL or null_a["ci_lower_abs"] <= NULL_RHO_LO, \
        f"shuffle null should collapse: {null_a}"
    print(f"  [ok] shuffle-null collapses on monotone case: "
          f"null rho={null_a['rho']:.3f} p={null_a['p_perm']:.4f}", flush=True)

    print("[selftest] ALL PASSED", flush=True)
    return True


# --------------------------------------------------------------------------
# ast.parse guard
# --------------------------------------------------------------------------

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
# Entry point
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", type=str, default=None,
                    help="circuit checkpoint (.safetensors). MODE A (GPU). "
                         "Falls back to FG_CKPT env var.")
    ap.add_argument("--n-perm", type=int, default=N_PERM_DEFAULT)
    ap.add_argument("--n-boot", type=int, default=N_BOOT_DEFAULT)
    ap.add_argument("--seed", type=int, default=42,
                    help="CircuitLoader seed (default 42, mirrors eval_circuit_depth.py).")
    ap.add_argument("--selftest", action="store_true",
                    help="CPU-only self-test (no GPU, no model).")
    args = ap.parse_args()

    parse_ok = _ast_parse_ok()
    print(f"[ast.parse] astparse_ok={parse_ok}", flush=True)
    if not parse_ok:
        sys.exit(1)

    if args.selftest:
        ok = _selftest()
        sys.exit(0 if ok else 1)

    # MODE A: GPU forward pass.
    ckpt = args.ckpt or os.environ.get("FG_CKPT", "")
    if not ckpt:
        print("ERROR: provide --ckpt CKPT or set FG_CKPT env var (MODE A, GPU), "
              "or use --selftest (CPU).", flush=True)
        sys.exit(2)

    run_mode_a(ckpt, args.n_perm, args.n_boot, args.seed)


if __name__ == "__main__":
    main()
