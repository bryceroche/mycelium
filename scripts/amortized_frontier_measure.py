"""amortized_frontier_measure.py — the AMORTIZED-SPEED VALUE-PROP measurement.

THE FRAME (Bryce, 2026-06-21)
=============================
The deducer does NOT beat BP/symbolic on inference QUALITY (refuted across clean CSP +
soft-MRF marginals + soft-MRF MAP). Its PROVEN edge is GENERALITY + AMORTIZED-FAST
PARALLEL inference: a FIXED-cost forward pass (K breaths through 4 SHARED Pythia L0-L3
layers, all variables in parallel) whose cost does NOT blow up with instance hardness,
versus a symbolic solver whose cost VARIES and can explode exponentially.

THE HONEST TENSION (the whole point of the measurement — do NOT paper over it):
on CLEAN verifiable CSPs the symbolic search tier is BOTH fast AND exact (GAC collapses
KenKen B3 trees to ~1 decision; symbolic DOMINATED hard coloring). Where symbolic is
microseconds there is NO tail to beat. The speed win can ONLY live where symbolic
genuinely struggles ON TIME: phase-transition-hard instances where backtracking
explodes — IF the deducer stays accurate there.

THE MEASUREMENT (this harness)
==============================
On the SAME instance banks across a HARDNESS sweep (edge-density c = m/n bands, swept
into the chromatic phase transition), characterize BOTH curves and overlay them:

  SYMBOLIC  : the VALIDATED solver (mycelium.csp_core.solve_symbolic on the coloring
              Problem from mycelium.csp_domains.problem_from_coloring — NOT reimplemented)
              per instance; record decisions/nodes, wall-clock, and TIMEOUT rate. A
              wall-clock cap AND a node budget both capture blow-up (a thread runs the
              solver with a hard wall-clock deadline; the node budget is the in-loop cap).
  DEDUCER   : the EXISTING FG eval forward (factor_breathing_forward via
              search_coloring._make_engine_deduce_fn's model build / load_ckpt — the
              validated path, NOT a hand-rolled forward) on the SAME banks; record
              cell_acc + puzzle_acc + EXACT proper-rate (read_at_settle_eval._coloring_proper,
              the validated edge verifier — gold-match understates coloring due to color-
              permutation). The deducer forward cost is ~FIXED per instance (reported as a
              fixed batched throughput; the speed claim is FLATNESS vs symbolic blow-up).
  VERDICT   : is there a band where symbolic is SLOW (high wall-clock / timeouts) AND the
              deducer is still ACCURATE (proper-rate well above chance)? That band = the
              standalone-amortized-speed SWEET SPOT. If symbolic never blows up before
              deducer accuracy collapses -> standalone-speed is DEAD here -> redirect to
              the ordering-prior angle. The harness DETECTS + FLAGS "no blow-up reached"
              and "deducer at chance across the whole range".

HARDNESS CALIBRATION (where the transition is, why the bands)
=============================================================
Random 3-coloring of G(n,p) has a well-known chromatic phase transition: the satisfiable
fraction collapses around average degree d ~ 4.69, i.e. c = m/n = d/2 ~ 2.35. A CPU
calibration (in _cpu_smoke + the standalone probe used to design this) confirmed on the
VALIDATED solver: keep-rate (3-colorability) falls 95% (c=1.5) -> ~50% (c~2.1-2.3) -> 0%
(c=3.0). So bands c in {1.5, 2.0, 2.2, 2.3, 2.5, 2.7} straddle and pass through the
transition; --bands overrides them. The bank for each band is built from k-colorable
instances ONLY (the deducer cannot be 'accurate' on an unsatisfiable instance, and the
loader's ground-truth DSATUR solver discards non-colorable graphs), so the sweep probes
the HARD-BUT-COLORABLE pocket just below the threshold — exactly where a sat-side
backtracker would tail if anything does.

HONEST CAVEAT ALREADY KNOWN FROM CALIBRATION: at n<=49 (the engine's hard S==49 cap) the
GENERAL symbolic core (GAC == full arc-consistency + MRV == DSATUR + LCV) collapses these
trees to <~35 decisions / <~15 ms with ZERO timeouts even at threshold. The blow-up on
this substrate lives on the UNSAT side (which the deducer cannot win and the bank
excludes). The harness still RUNS the full sweep + emits the verdict from MEASURED numbers
(it does not hard-code the conclusion); if a band/budget does surface a tail it is caught,
and if none does the DEAD verdict fires with the evidence. --max-n lets a future
larger-S engine push n up; here it is capped at s_max.

REUSE (NOTHING HAND-ROLLED)
===========================
  * symbolic solver  : mycelium.csp_core.solve_symbolic (+ backtrack_search) verbatim.
  * coloring Problem  : mycelium.csp_domains.problem_from_coloring verbatim.
  * instance banks    : mycelium.graph_coloring_data generate/encode + GraphColoringLoader
                        ._stack verbatim (custom densities via the module-level generators;
                        the data module is NOT modified).
  * deducer forward   : scripts.search_coloring._make_engine_deduce_fn's model build +
                        load_ckpt + factor_breathing_forward (the validated GPU eval path).
  * deducer accuracy  : argmax+1==gold cell/puzzle match (matches factor_graph_train.evaluate)
                        + read_at_settle_eval._coloring_proper (the validated edge verifier).
ENGINE (mycelium/factor_graph_engine.py) + ORACLE (mycelium/kenken.py) are NEVER touched.

DOMAIN HOOK
===========
--domain {coloring,circuit,kenken}. coloring is the LEAD probe (built). circuit/kenken
are HOOKS: a clear NotImplementedError naming the missing bridge (problem_from_circuit is a
Phase-4 stub; a kenken bank generator + a kenken deducer ckpt are not wired here). The
symbolic side already supports kenken via problem_from_kenken, so kenken is the closest to
landing later.

TIMING FAIRNESS (the #1 way a speed benchmark lies)
===================================================
  * Model load + JIT compile + a warmup forward are EXCLUDED from the deducer per-instance
    timing and REPORTED SEPARATELY (one-time costs). Steady-state batched throughput is
    timed after warmup.
  * The deducer runs on GPU (batched); the symbolic solver runs on CPU (per-instance).
    This substrate mismatch is reported HONESTLY: deducer = fixed per-instance amortized
    ms (batched wall / batch) AND raw batched throughput; symbolic = per-instance CPU
    wall-clock + node count. Never hidden, never cross-claimed.
  * Symbolic uses a HARD wall-clock TIMEOUT (a worker thread with a deadline; the node
    budget is the in-loop cap) so a blow-up is captured as a timeout rather than a hang.

SUBSTRATE LAWS: the deducer forward is REUSED as-is (no new JIT path, no dtypes.float32
literal inside a JIT step). The symbolic side is pure CPU python. ast.parse clean.

GPU-FREE BUILD + CPU SMOKE
==========================
SELFTEST_ONLY=1 (or --smoke) runs a CPU-only smoke: the symbolic side on a couple of small
graphs (real solve via the validated core), the phase-transition calibration on the
validated generators, and a DRY-RUN of the deducer-eval wiring (bank build + batch shapes +
the accuracy-contract checks) with NO GPU forward. ast.parse clean, import clean.

RUN (full sweep, AMD GPU — Bryce / main-thread runs this; agents do NOT)
=======================================================================
  DEV=AMD \\
  FG_CKPT=.cache/fg_ckpts/fg_coloring_k16/fg_coloring_k16_final.safetensors \\
  .venv/bin/python3 scripts/amortized_frontier_measure.py \\
      --domain coloring --bands 1.5,2.0,2.2,2.3,2.5,2.7 \\
      --per-band 200 --sym-budget 300000 --sym-timeout-ms 2000
"""
from __future__ import annotations

import argparse
import ast
import math
import os
import random
import sys
import time
from typing import Optional

_THIS_FILE = os.path.abspath(__file__)
_ROOT = os.path.dirname(os.path.dirname(_THIS_FILE))
sys.path.insert(0, _ROOT)

import numpy as np  # noqa: E402

# ---------------------------------------------------------------------------
# VALIDATED SYMBOLIC SOLVER + COLORING BRIDGE (reused verbatim — never reimplemented)
# ---------------------------------------------------------------------------
from mycelium.csp_core import solve_symbolic  # noqa: E402
from mycelium.csp_domains import problem_from_coloring  # noqa: E402

# VALIDATED instance-bank generators (the data module is NOT modified — only its
# module-level functions are called to build custom-density k-colorable banks).
from mycelium.graph_coloring_data import (  # noqa: E402
    _canonicalize_coloring,
    _depth_bucket,
    _gen_gnp,
    _gen_regular,
    _solve_k_coloring,
)

# The TRAINED distribution of fg_coloring_k16 (mycelium.graph_coloring_data:
# BANDS=["d10","d15","d20","d25"] => densities {1.0,1.5,2.0,2.5}; n drawn in
# [max(4,min(8,s_max)), s_max] = [8, 49] at s_max=49). VOLUME mode defaults to this
# so the measured single-sample proper-rate p is FAIR; any band outside it is FLAGGED.
TRAINED_BAND_DENSITIES = (1.0, 1.5, 2.0, 2.5)
TRAINED_BAND_C_MAX = 2.5     # the ckpt's max edge-density band
TRAINED_N_MIN = 8            # the ckpt's min vertex count


# ===========================================================================
# ast.parse gate — always runs, even on CPU
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
# HARDNESS BANK BUILDER (reuses the validated generators; custom densities)
# ===========================================================================
# The loader's BANDS top out at edges/n=2.5 with a fixed {d10..d25} density table; the
# phase transition for 3-coloring sits at c=m/n ~ 2.35. So we build banks at ARBITRARY
# target density c by calling the SAME module-level generators the loader uses
# (_gen_gnp / _gen_regular / _solve_k_coloring / _canonicalize_coloring), keeping ONLY
# k-colorable instances (DSATUR ground truth), then pack via the loader's encode path.
# This reaches the near-threshold regime WITHOUT modifying graph_coloring_data.py.

def _build_band(c: float, n_instances: int, n_min: int, n_max: int, k: int,
                seed: int, regular_frac: float = 0.4,
                max_attempts_factor: int = 400) -> list:
    """Generate up to n_instances k-COLORABLE instances at target density c = m/n.

    Returns a list of instance dicts (same shape as graph_coloring_data.generate_instance):
      {n, edges, coloring(canonical 0..k-1), band(=f'c{c}'), deduction_depth, n_edges}.
    Non-k-colorable graphs are DISCARDED (the deducer cannot be accurate on an
    unsatisfiable instance; the bank probes the hard-but-colorable pocket). Records the
    DSATUR backtrack bucket as deduction_depth (the per-instance hardness proxy).
    """
    rng = random.Random(seed)
    out: list = []
    attempts = 0
    max_attempts = n_instances * max_attempts_factor + 1000
    while len(out) < n_instances and attempts < max_attempts:
        attempts += 1
        n = rng.randint(n_min, n_max)
        edges: Optional[list] = None
        if rng.random() < regular_frac:
            d = int(round(2.0 * c))
            d = max(1, min(d, n - 1))
            if (n * d) % 2 != 0:
                d -= 1
            if d >= 1:
                edges = _gen_regular(n, d, rng)
        if edges is None:
            p = min(0.98, 2.0 * c / max(1, n - 1))
            edges = _gen_gnp(n, p, rng)
        if not edges:
            continue
        adj = [set() for _ in range(n)]
        for (u, v) in edges:
            adj[u].add(v)
            adj[v].add(u)
        coloring, backtracks = _solve_k_coloring(n, adj, k)
        if coloring is None:
            continue  # not k-colorable -> discard
        out.append({
            "n": n,
            "edges": edges,
            "coloring": _canonicalize_coloring(coloring),
            "band": f"c{c:.2f}",
            "deduction_depth": _depth_bucket(backtracks),
            "n_edges": len(edges),
            "c_target": c,
            "dsatur_backtracks": backtracks,
        })
    return out


# ===========================================================================
# SYMBOLIC SIDE — wall-clock + nodes + timeout per instance
# ===========================================================================

def _solve_symbolic_timed(inst: dict, k: int, budget: int,
                          timeout_ms: float) -> dict:
    """Run the VALIDATED solve_symbolic on one instance with a HARD wall-clock TIMEOUT.

    The node `budget` is the in-loop cap (solve_symbolic's own); `timeout_ms` is an
    independent wall-clock deadline enforced by a worker thread (captures a blow-up as a
    timeout instead of a hang — the budget alone can't bound time on a pathological tree).

    Returns:
      decisions   : int  decision-nodes expanded (0 if timed out before any report).
      backtracks  : int
      wall_ms     : float wall-clock for the solve (capped near timeout_ms on a timeout).
      status      : 'solved' | 'unsat' | 'budget' | 'timeout'
      timed_out   : bool
      budget_hit  : bool (status == 'budget' — node cap hit, not the wall clock).
    """
    import threading

    prob = problem_from_coloring(inst["n"], inst["edges"], k)
    box: dict = {"res": None, "err": None}

    def _worker():
        try:
            box["res"] = solve_symbolic(prob, budget=budget)
        except Exception as e:  # pragma: no cover - defensive
            box["err"] = e

    t = threading.Thread(target=_worker, daemon=True)
    t0 = time.perf_counter()
    t.start()
    t.join(timeout=timeout_ms / 1000.0)
    wall_ms = (time.perf_counter() - t0) * 1000.0

    if t.is_alive():
        # Cooperative timeout: the solver is pure-python (no C extension holding the GIL),
        # so the deadline fires; the daemon thread is abandoned (it will exit when the
        # process does / when its own budget trips). We report a timeout with the wall cap.
        return {
            "decisions": budget, "backtracks": 0, "wall_ms": wall_ms,
            "status": "timeout", "timed_out": True, "budget_hit": False,
        }
    if box["err"] is not None:  # pragma: no cover
        return {
            "decisions": 0, "backtracks": 0, "wall_ms": wall_ms,
            "status": "error", "timed_out": False, "budget_hit": False,
        }
    res = box["res"]
    return {
        "decisions": int(res["decisions"]),
        "backtracks": int(res["backtracks"]),
        "wall_ms": wall_ms,
        "status": res["status"],
        "timed_out": False,
        "budget_hit": res["status"] == "budget",
    }


# ===========================================================================
# DEDUCER SIDE — accuracy via the VALIDATED forward (NOT hand-rolled)
# ===========================================================================

def _coloring_proper_np(pred_1based: np.ndarray, membership_b: np.ndarray,
                        latent_type_b: np.ndarray, cell_valid_b: np.ndarray,
                        edge_ltype: int) -> bool:
    """EXACT proper-coloring verifier for ONE instance (the validated edge check, the
    same logic as read_at_settle_eval._coloring_proper): every edge's two endpoints have
    different predicted colors. pred_1based is 1..k (engine encoding). Gold-free — the
    correct success metric for coloring (gold-match understates by color-permutation)."""
    L = membership_b.shape[0]
    for e in range(L):
        if int(latent_type_b[e]) != edge_ltype:
            continue
        ones = np.nonzero(membership_b[e] > 0.5)[0]
        if ones.shape[0] != 2:
            continue
        u, v = int(ones[0]), int(ones[1])
        if not (cell_valid_b[u] > 0.5 and cell_valid_b[v] > 0.5):
            continue
        if int(pred_1based[u]) == int(pred_1based[v]):
            return False
    return True


def _score_predictions(pred_by_inst, gold_np, cell_valid_np, membership_np,
                       latent_type_np, edge_ltype) -> dict:
    """Aggregate cell_acc / puzzle_acc / proper-rate over a batch of predictions.

    pred_by_inst : (B, S) int 1-based predictions (argmax final-breath logits + 1).
    Matches factor_graph_train.evaluate's cell/puzzle definition (argmax+1==gold) and
    read_at_settle_eval's exact edge proper-rate. Returns per-row lists so the caller can
    pool per band exactly."""
    B = cell_valid_np.shape[0]
    cell_eq = 0.0
    n_cells = 0
    puz_ok = 0
    n_puz = 0
    proper_ok = 0
    for b in range(B):
        valid = cell_valid_np[b] > 0.5
        nv = int(valid.sum())
        if nv == 0:
            continue
        pred = pred_by_inst[b]
        gold = gold_np[b]
        eq = (pred == gold) & valid
        cell_eq += float(eq.sum())
        n_cells += nv
        puz_ok += int(np.all(pred[valid] == gold[valid]))
        n_puz += 1
        proper_ok += int(_coloring_proper_np(
            pred, membership_np[b], latent_type_np[b], cell_valid_np[b], edge_ltype))
    return {"cell_eq": cell_eq, "n_cells": n_cells,
            "puz_ok": puz_ok, "n_puz": n_puz,
            "proper_ok": proper_ok}


# ===========================================================================
# VOLUME MODE — empirical proper-rate floor (reviewer fix #1)
# ===========================================================================
# The deducer-accurate gate used chance=1/k=0.333 as the PROPER-RATE floor, but that
# is the cell_acc floor, NOT the whole-graph proper-rate floor (a uniform-random
# coloring almost never satisfies every edge on dense n=40-49 banks -> measured proper
# floor ~ 0.000). Using 1/k biased AGAINST the deducer. FIX: score a uniform-random
# coloring on each band with the SAME validated verifier and use max(that, eps) as the
# floor; print it so the margin is auditable.

def _empirical_proper_floor(insts: list, s_max: int, n_edges_max: int, k: int,
                            seed: int, n_draws: int = 1) -> float:
    """Measured proper-rate of a UNIFORM-RANDOM k-coloring over this band's instances.

    For each instance, draw `n_draws` uniform-random colorings (each valid vertex gets
    a color in 1..k iid) and score with the validated edge verifier (_coloring_proper_np
    via the same encode path). Returns the fraction of (instance x draw) that happen to
    be proper. This is the HONEST proper-rate chance floor for the deducer-accurate gate
    (NOT 1/k). Pure CPU/numpy — no GPU forward."""
    from mycelium.graph_coloring_data import encode_instance, LTYPE_EDGE
    rng = np.random.RandomState(seed)
    n_proper = 0
    n_tot = 0
    for r in insts:
        enc = encode_instance(r, s_max, n_edges_max, k)
        cv = enc["cell_valid"]
        mem = enc["membership"]
        lt = enc["latent_type"].astype(np.int32)
        valid_idx = np.nonzero(cv > 0.5)[0]
        if valid_idx.size == 0:
            continue
        for _ in range(n_draws):
            pred = np.zeros((s_max,), dtype=np.int32)
            pred[valid_idx] = rng.randint(1, k + 1, size=valid_idx.size).astype(np.int32)
            n_proper += int(_coloring_proper_np(pred, mem, lt, cv, LTYPE_EDGE))
            n_tot += 1
    return (n_proper / n_tot) if n_tot else 0.0


# ===========================================================================
# VOLUME MODE — diversity samplers (temp / multistart / symmetry)
# ===========================================================================
# Each mechanism draws M predictions per instance WITHOUT retraining, then the FREE
# verifier (_coloring_proper_np) keeps any valid one. We account the forward-count cost
# HONESTLY (temp = 1 forward -> M samples; multistart/symmetry = M forwards) and DETECT
# sample-collapse (distinct-sample count / entropy per instance) so collapse is visible.

def _temp_sample_from_logits(final_logits: np.ndarray, valid: np.ndarray,
                             value_domain_mask: np.ndarray, tau: float,
                             M: int, rng: np.random.RandomState) -> np.ndarray:
    """Per-cell INDEPENDENT temperature samples from ONE forward's final logits.

    final_logits : (S, N) the deducer's final-breath value logits for one instance.
    valid        : (S,) bool.
    value_domain_mask : (S, N) 1=legal color. Illegal colors get -inf before softmax.
    tau          : sampling temperature (TAU). M samples come from ONE forward (cheap).
    Returns (M, S) int 1-based predictions. This is the MEAN-FIELD / per-cell-independent
    sampler -> expect JOINT-INCOHERENT samples (cells flip independently, violating
    edges) -> the BASELINE/lower-bound on coupled instances. Reproducibly seeded."""
    S, N = final_logits.shape
    logits = final_logits.astype(np.float64) / max(tau, 1e-6)
    # mask illegal colors (value_domain_mask==0) to -inf so they never sample.
    illegal = value_domain_mask < 0.5
    logits = np.where(illegal, -1e30, logits)
    logits = logits - logits.max(axis=-1, keepdims=True)
    probs = np.exp(logits)
    probs = probs / (probs.sum(axis=-1, keepdims=True) + 1e-30)     # (S, N)
    out = np.zeros((M, S), dtype=np.int32)
    valid_idx = np.nonzero(valid)[0]
    # Gumbel-max categorical sampling per cell, vectorized over M.
    for c in valid_idx:
        p = probs[c]
        # inverse-CDF sampling (M draws) — exact categorical, seeded.
        u = rng.random_sample(M)
        cdf = np.cumsum(p)
        idx = np.searchsorted(cdf, u, side="right")
        idx = np.clip(idx, 0, N - 1)
        out[:, c] = idx.astype(np.int32) + 1                       # 1-based
    return out


def _distinct_and_entropy(samples: np.ndarray, valid: np.ndarray) -> tuple:
    """(distinct_count, normalized_entropy) over M predictions for ONE instance.

    samples : (M, S) int predictions. valid : (S,) bool. Collapse detector: if all M
    rows are identical on the valid cells -> distinct=1, entropy=0 (mechanism failed to
    diversify). Entropy normalized to [0,1] by log(M)."""
    M = samples.shape[0]
    vi = np.nonzero(valid)[0]
    if vi.size == 0 or M <= 1:
        return (1, 0.0)
    keys = [samples[m, vi].tobytes() for m in range(M)]
    counts: dict = {}
    for kbytes in keys:
        counts[kbytes] = counts.get(kbytes, 0) + 1
    distinct = len(counts)
    ps = np.array(list(counts.values()), dtype=np.float64) / float(M)
    ent = float(-(ps * np.log(ps + 1e-30)).sum())
    ent_norm = ent / math.log(M) if M > 1 else 0.0
    return (distinct, ent_norm)


# ===========================================================================
# VOLUME MODE — best-of-N aggregation + independent-ideal + M_eff
# ===========================================================================

def _solve_rate_curve(per_inst_valid_flags: list, m_grid: list,
                      coherent: bool = False) -> dict:
    """Empirical best-of-M solve-rate curve + the INDEPENDENT IDEAL + M_eff.

    per_inst_valid_flags : list (one per instance) of (M_max,) bool arrays — whether the
                           j-th sample of that instance is a VALID coloring (verified).
    m_grid               : the M values to report solve-rate at (subsets of [1, M_max]).
    coherent             : reviewer fix #1 — for multistart/symmetry slot 0 is the
                           DETERMINISTIC argmax run and slots 1..M-1 are perturbed/permuted,
                           so the M samples are NOT exchangeable. When True, ALSO report
                           p1_div = mean valid-rate over the PERTURBED slots ONLY (slots
                           1..M-1) and the diversity contribution's independent-ideal /
                           M_eff(ideal) / ratio, so the correlated-failure readout reflects
                           the coherent decorrelated samples, not the argmax dilution. When
                           False (temp — all M exchangeable) p1_div == p1_all (no change).

    For each M: empirical solve-rate(M) = fraction of instances with ANY valid sample in
    the first M draws. The INDEPENDENT IDEAL = 1-(1-p1)^M where p1 = the measured
    single-sample solve-rate (mean over instances of P(sample j valid)) — what best-of-M
    WOULD reach if the M samples failed INDEPENDENTLY. The GAP between empirical and ideal
    exposes correlated-failure collapse (M_eff << M). Also returns:
      p1 / p1_all       : measured single-sample solve-rate (mean over ALL M slots).
      p1_div            : measured single-sample solve-rate over the PERTURBED slots only
                          (slots 1..M-1); == p1_all when coherent is False.
      m_eff_emp/m_eff_ideal : smallest M at which empirical / ideal solve-rate crosses 0.9.
      m_eff_ratio       : (M_eff^emp / M_eff^ideal) — how many MORE samples the real
                          (correlated) sampler needs vs the independent ideal (>1 = collapse).
      m_eff_div_ideal/m_eff_div_ratio : the same against the DIVERSITY ideal (1-(1-p1_div)^M)."""
    n_inst = len(per_inst_valid_flags)
    if n_inst == 0:
        return {"p1": 0.0, "p1_all": 0.0, "p1_div": 0.0, "m_grid": m_grid,
                "solve_emp": {}, "solve_ideal": {}, "solve_ideal_div": {},
                "m_eff_emp": None, "m_eff_ideal": None, "m_eff_ratio": float("nan"),
                "m_eff_div_ideal": None, "m_eff_div_ratio": float("nan"), "n_inst": 0}
    M_max = max(len(f) for f in per_inst_valid_flags)
    # p1_all = mean per-instance single-sample valid-rate (over ALL drawn samples).
    p1_terms = []
    # p1_div = mean valid-rate over the PERTURBED slots ONLY (slots 1..M-1) — the CLEAN
    # diversity rate (slot 0 is the deterministic argmax for coherent mechanisms).
    p1_div_terms = []
    for f in per_inst_valid_flags:
        if len(f) == 0:
            continue
        p1_terms.append(float(np.mean(f)))
        if coherent and len(f) > 1:
            p1_div_terms.append(float(np.mean(f[1:])))
        else:
            p1_div_terms.append(float(np.mean(f)))
    p1 = float(np.mean(p1_terms)) if p1_terms else 0.0
    p1_div = float(np.mean(p1_div_terms)) if p1_div_terms else p1

    solve_emp: dict = {}
    solve_ideal: dict = {}
    solve_ideal_div: dict = {}
    for M in m_grid:
        Mc = min(M, M_max)
        solved = 0
        for f in per_inst_valid_flags:
            mm = min(Mc, len(f))
            if mm > 0 and bool(np.any(f[:mm])):
                solved += 1
        solve_emp[M] = solved / n_inst
        solve_ideal[M] = 1.0 - (1.0 - p1) ** M
        solve_ideal_div[M] = 1.0 - (1.0 - p1_div) ** M

    def _cross_90(curve: dict) -> Optional[int]:
        for M in m_grid:
            if curve[M] >= 0.9:
                return M
        return None

    def _ratio(m_emp, m_ideal) -> float:
        # >1 = collapse; under TOTAL collapse the empirical curve never crosses 0.9 ->
        # m_emp is None -> report inf so the most severe collapse reads as the LARGEST
        # ratio, not a blank nan (reviewer fix #5).
        if m_ideal and m_emp and m_emp > 0:
            return float(m_emp) / float(m_ideal)
        if m_ideal and m_emp is None:
            return float("inf")
        return float("nan")

    m_eff_emp = _cross_90(solve_emp)
    m_eff_ideal = _cross_90(solve_ideal)
    m_eff_div_ideal = _cross_90(solve_ideal_div)
    ratio = _ratio(m_eff_emp, m_eff_ideal)
    div_ratio = _ratio(m_eff_emp, m_eff_div_ideal)
    return {"p1": p1, "p1_all": p1, "p1_div": p1_div, "m_grid": m_grid,
            "solve_emp": solve_emp, "solve_ideal": solve_ideal,
            "solve_ideal_div": solve_ideal_div, "m_eff_emp": m_eff_emp,
            "m_eff_ideal": m_eff_ideal, "m_eff_ratio": ratio,
            "m_eff_div_ideal": m_eff_div_ideal, "m_eff_div_ratio": div_ratio,
            "n_inst": n_inst}


# ===========================================================================
# VOLUME MODE — VIEW-SUCCESS-COUNT DISTRIBUTION (diversity FRAGILITY metric)
# ===========================================================================
# THE QUESTION (pre vs post permutation-augmentation): does augmentation toward
# equivariance COLLAPSE the load-bearing diversity bucket? best-of-M alone cannot
# see this — an instance that solves on EXACTLY ONE of its M views and an instance
# that solves on ALL M both read as "solved". The FRAGILITY is in HOW MANY views win.
#
# THE METRIC: for each instance, count how many of its M views VERIFIED (the per-
# instance valid-flag count, 0..M), then bucket instances by that count:
#   0      : HARD CORE        — no view solves (the shared core the deducer fails
#                               regardless of relabeling; what augmentation should SHRINK)
#   1      : FRAGILE          — EXACTLY ONE view wins (diversity is fragile here:
#                               equivariance-collapse would lose this instance)
#   2-4    : few views
#   5-16   : many views
#   17-M   : robust           — most views win (diversity is robust; safe)
# Many "exactly-1" => fragile diversity => augmentation toward equivariance is RISKY
# there. Many "many-views-win" => robust. Comparing the distribution pre vs post
# augmentation SHOWS whether augmentation collapsed the B bucket.
#
# ZERO NEW FORWARDS: derived from the SAME per-instance mech_flags the volume mode
# already computes. ADDITIVE: a new sub-table, the (A)/(B)/(C) tables unchanged.
#
# PARITY INVARIANTS (asserted in the smoke):
#   * the buckets PARTITION the instances -> counts sum to n_inst.
#   * fraction with success-count >= 1  ==  best-of-M solve-rate (any view wins).
#   * success-count INCLUDES view 0 (the deterministic argmax for multistart/
#     symmetry), so the ">=1" set CONTAINS every argmax-solved instance
#     (p_argmax <= success>=1 fraction).

# Bucket edges as (lo, hi_inclusive, label). The last bucket's hi is set to M at
# call time. Order matters for the additive table columns.
_VSC_BUCKET_SPEC = [
    (0, 0, "0(hardcore)"),
    (1, 1, "1(fragile)"),
    (2, 4, "2-4"),
    (5, 16, "5-16"),
    (17, None, "17-M"),     # hi=None -> M (filled per call)
]


def _view_success_count_distribution(per_inst_valid_flags: list, M: int) -> dict:
    """Distribution of instances bucketed by HOW MANY of their M views VERIFIED.

    per_inst_valid_flags : list (one per instance) of (>=1,) bool arrays — the SAME
                           flags _solve_rate_curve consumes (mech_flags[m]); the
                           per-view valid-flags, view 0 == the deterministic argmax
                           run for multistart/symmetry. ZERO new forwards.
    M                    : the per-instance view budget (success-count is capped at
                           min(M, len(flags)) per instance).

    Returns:
      n_inst        : instance count.
      buckets       : list of (label, count) in _VSC_BUCKET_SPEC order; counts sum
                      to n_inst (the buckets partition the instances).
      bucket_frac   : list of (label, fraction) — count / n_inst.
      counts        : list[int] per-instance success-count (success views in M).
      success_ge1   : fraction with >= 1 winning view (== best-of-M solve-rate).
      n_solved      : instances with >= 1 winning view.
      mean_success  : mean per-instance success-count (the diversity 'volume').
      median_success: median per-instance success-count.
      M             : echoed budget."""
    n_inst = len(per_inst_valid_flags)
    # resolve bucket edges (fill the trailing None hi with M).
    edges = [(lo, (M if hi is None else hi), label)
             for (lo, hi, label) in _VSC_BUCKET_SPEC]
    if n_inst == 0:
        return {"n_inst": 0, "buckets": [(lab, 0) for (_, _, lab) in edges],
                "bucket_frac": [(lab, 0.0) for (_, _, lab) in edges],
                "counts": [], "success_ge1": 0.0, "n_solved": 0,
                "mean_success": float("nan"), "median_success": float("nan"),
                "M": M}
    counts = []
    for f in per_inst_valid_flags:
        mm = min(M, len(f)) if len(f) else 0
        counts.append(int(np.count_nonzero(np.asarray(f, dtype=bool)[:mm])) if mm > 0 else 0)
    counts_arr = np.asarray(counts, dtype=np.int64)
    bucket_counts = []
    for (lo, hi, label) in edges:
        c = int(np.count_nonzero((counts_arr >= lo) & (counts_arr <= hi)))
        bucket_counts.append((label, c))
    n_solved = int(np.count_nonzero(counts_arr >= 1))
    return {
        "n_inst": n_inst,
        "buckets": bucket_counts,
        "bucket_frac": [(lab, c / n_inst) for (lab, c) in bucket_counts],
        "counts": counts,
        "success_ge1": n_solved / n_inst,
        "n_solved": n_solved,
        "mean_success": float(np.mean(counts_arr)),
        "median_success": float(np.median(counts_arr)),
        "M": M,
    }


# ===========================================================================
# VOLUME MODE — honest throughput accounting (deducer M-forward vs symbolic)
# ===========================================================================

def _throughput_accounting(B: int, batch_ms: float, n_forwards: int,
                           solve_rate_bestofM: float,
                           sym_solve_ms_med: float, sym_solved_rate: float) -> dict:
    """Instances-SOLVED/sec, deducer (GPU, M-forward-honest) vs symbolic (CPU/core).

    Deducer: a best-of-M run costs `n_forwards` batched forwards per batch of B (temp=1,
    multistart/symmetry=M). instances-solved/sec = (B * solve_rate_bestofM) /
    (n_forwards * batch_ms/1000). Symbolic: 1/median_solve_ms per CPU core, scaled by the
    solved-rate, with a NOTE that symbolic is embarrassingly parallel across cores. NEVER
    a single cross-substrate ratio — both reported side by side with substrate stated."""
    batch_s = (n_forwards * batch_ms) / 1000.0
    ded_solved_per_s = (B * solve_rate_bestofM) / batch_s if batch_s > 0 else float("nan")
    ded_inst_per_s = B / batch_s if batch_s > 0 else float("nan")   # raw (ignoring solve)
    sym_per_core = (1000.0 / sym_solve_ms_med) * sym_solved_rate if sym_solve_ms_med > 0 \
        else float("nan")
    return {
        "ded_solved_per_s": ded_solved_per_s,
        "ded_inst_per_s": ded_inst_per_s,
        "ded_n_forwards": n_forwards,
        "ded_batch": B,
        "sym_solved_per_s_per_core": sym_per_core,
        "sym_solve_ms_med": sym_solve_ms_med,
        "sym_solved_rate": sym_solved_rate,
    }


# ===========================================================================
# EARLY-STOP (a) — FREE ACCOUNTING (ZERO new forwards; derived from the EXISTING
# full-M per-sample valid-flags the measurement mode already computes)
# ===========================================================================
# With a FREE EXACT verifier you never need all M darts — you stop the moment one
# verifies. This is a DEPLOYMENT-honesty correction to the (C) throughput: the
# fixed-M accounting charges M forwards per instance even though, in deployment, the
# typical instance solves in its FIRST handful of darts. We DERIVE the expected
# forwards-to-solve from the flags ALREADY computed (no GPU work):
#
#   first_success_index(inst) = 1-based index j of the FIRST valid sample, or None
#                               (unsolved within M).
#   forwards_used(inst):
#     * temp                : 1   (ALL M samples come from ONE forward — post-hoc
#                             numpy draws; early-stop saves only cheap numpy verify,
#                             NEVER a forward. Its forward cost stays 1.)
#     * multistart/symmetry : min(first_success_index, M)  (each dart = 1 forward;
#                             stopping at first success saves real forwards; an
#                             UNSOLVED instance burns all M).
#   exp_forwards_to_solve   = mean over instances of forwards_used.
#
# Early-stop throughput (the DEPLOYMENT number for the (C) table): you solve in
# ~exp_forwards darts, not a fixed M, so
#     ded_solved_per_s_es = (B * solve_rate) / (exp_forwards * batch_ms/1000)
# which is the equivalent of total-instances-solved / total-forward-time when the
# active set is kept ~full at B (the compute-bound optimum). The fixed-M best-of-M
# numbers are KEPT alongside (this only ADDS the deployment column).

def _first_success_index(flags: np.ndarray) -> Optional[int]:
    """1-based index of the FIRST True in `flags` (a per-instance (M,) valid-flag
    array), or None if no sample verified within M. Pure numpy, no forward."""
    if flags is None or len(flags) == 0:
        return None
    hits = np.nonzero(np.asarray(flags, dtype=bool))[0]
    if hits.size == 0:
        return None
    return int(hits[0]) + 1                     # 1-based


def _early_stop_accounting(per_inst_valid_flags: list, M: int,
                           per_dart_is_forward: bool) -> dict:
    """FREE early-stop accounting from the EXISTING full-M flags (no new forwards).

    per_inst_valid_flags : list (one per instance) of (>=1,) bool arrays — the SAME
                           flags _solve_rate_curve consumes.
    M                    : the per-instance dart budget (cap on forwards-used).
    per_dart_is_forward  : True for multistart/symmetry (each dart == 1 forward, so
                           early-stop saves real forwards); False for temp (all M
                           samples come from ONE forward -> forwards_used == 1 always,
                           regardless of first-success — do NOT claim a forward saving).

    Returns:
      n_inst                : instance count.
      solve_rate            : fraction with ANY valid sample in M (== best-of-M; the
                              EXACT verifier makes early-stop solve-rate identical).
      exp_forwards          : mean over instances of forwards_used.
      forwards_used         : list[int] per instance (for cross-check vs the chunked run).
      first_success         : list[Optional[int]] per instance (1-based, None=unsolved).
      total_forwards        : sum(forwards_used) (the deployment forward count at B=1
                              equivalent; the chunked-run actual MUST match this).
      n_solved              : count solved within M.
      per_dart_is_forward   : echoed (temp=False).
    """
    n_inst = len(per_inst_valid_flags)
    if n_inst == 0:
        return {"n_inst": 0, "solve_rate": 0.0, "exp_forwards": float("nan"),
                "forwards_used": [], "first_success": [], "total_forwards": 0,
                "n_solved": 0, "per_dart_is_forward": per_dart_is_forward}
    forwards_used: list = []
    first_success: list = []
    n_solved = 0
    for f in per_inst_valid_flags:
        Mi = min(M, len(f)) if len(f) else 0
        fsi = _first_success_index(np.asarray(f)[:Mi]) if Mi > 0 else None
        first_success.append(fsi)
        solved = fsi is not None
        n_solved += int(solved)
        if not per_dart_is_forward:
            fu = 1                              # temp: ONE forward regardless
        elif solved:
            fu = min(fsi, M)                    # stop at first success
        else:
            fu = M                              # unsolved burns the full budget
        forwards_used.append(int(fu))
    exp_forwards = float(np.mean(forwards_used)) if forwards_used else float("nan")
    return {
        "n_inst": n_inst,
        "solve_rate": n_solved / n_inst,
        "exp_forwards": exp_forwards,
        "forwards_used": forwards_used,
        "first_success": first_success,
        "total_forwards": int(sum(forwards_used)),
        "n_solved": n_solved,
        "per_dart_is_forward": per_dart_is_forward,
    }


def _early_stop_throughput(B: int, batch_ms: float, exp_forwards: float,
                           solve_rate: float) -> dict:
    """DEPLOYMENT throughput with early-stop: you solve in ~exp_forwards darts (each a
    batched forward of B at the compute-bound optimum), not a fixed M.

      ded_solved_per_s_es = (B * solve_rate) / (exp_forwards * batch_ms/1000)

    Equivalent to total-instances-solved / total-forward-time when the active set is
    repacked to keep batches ~full at B (the measured compute-bound optimum). batch_ms
    is the steady per-forward batch time (same one the fixed-M (C) table uses)."""
    batch_s = (exp_forwards * batch_ms) / 1000.0
    es_solved_per_s = ((B * solve_rate) / batch_s
                       if (batch_s > 0 and exp_forwards == exp_forwards)
                       else float("nan"))
    return {
        "ded_solved_per_s_es": es_solved_per_s,
        "exp_forwards": exp_forwards,
        "solve_rate": solve_rate,
        "ded_batch": B,
    }


# ===========================================================================
# EARLY-STOP (b) — CHUNKED ACTIVE-SET DEPLOYMENT (the wall-clock saver)
# ===========================================================================
# Process a band's instances with an ACTIVE SET. Draw darts in CHUNKS of `chunk`
# (default 8) — up to `chunk` darts PER ACTIVE INSTANCE per round — verify each new
# sample, mark solved at first valid, DROP solved instances, and REPACK the still-
# unsolved into fresh full B (~8) batches for the next round. Continue until all
# solved or the per-instance dart budget M is exhausted.
#
# WHY repack-to-B (not pack-all-M-into-one-giant-batch): the GPU is COMPUTE-BOUND at
# B=8 — per-instance forward cost is FLAT from B=8 to B=128 (a forward of B*M = the
# SAME total compute as M forwards of B). So packing more darts buys NOTHING and
# B>=32 is slightly worse. The ONLY throughput lever is FEWER darts (this mode);
# repacking keeps batches ~full at the B=8 optimum. The chunk batches the ACTIVE SET
# across instances, NOT the M darts of one instance into a giant batch.
#
# DART INDEXING (the parity invariant): the j-th dart of an instance MUST be the SAME
# sample (same seed / same permutation / same noise) whether drawn in the full-M
# measurement loop or in a chunked round. We re-derive each dart from a PER-INSTANCE
# deterministic RNG keyed by (master_seed, instance_id, dart_index), so dart j is
# reproducible regardless of which round draws it. Then early-stop-solve-rate ==
# full-M best-of-M (the verifier is EXACT: stopping at first success cannot change
# WHETHER an instance is solvable within M) and the actual forwards == the (a)
# accounting's forwards_used (each instance is forwarded until its first valid dart,
# capped at M).

def _active_set_chunk_schedule(per_inst_first_success: list, M: int, chunk: int,
                               B: int) -> dict:
    """PURE-CPU simulator of the chunked active-set loop (NO GPU; for the smoke + the
    cross-check). Given each instance's first_success_index (1-based, or None), replay
    the EXACT active-set schedule the GPU deployment runs and count forwards.

    The loop (per round): for each ACTIVE instance, draw up to `chunk` more darts
    (dart indices d_lo..d_hi within its budget); the instance becomes SOLVED on this
    round iff its first_success_index falls in (d_lo, d_hi]; an instance hitting M
    darts without success is DROPPED unsolved. Solved/exhausted instances leave the
    active set; survivors are REPACKED into fixed-B batches for the next round.

    FORWARD COUNTING (matches the GPU cost model): a round that advances the active set
    by up to `chunk` darts each costs, per active instance, the number of darts ACTUALLY
    drawn for THAT instance this round (an instance that solves at dart d_lo+2 inside a
    chunk of 8 only spent 3 darts == 3 forwards-of-its-row; in the batched layout it
    rides padded-to-B batches, but the REAL-row forward count per instance is what we
    bill, exactly == min(first_success, M) for solved / M for unsolved). Repacking to B
    changes only HOW rows are co-batched (compute-bound: cost ~ #forwards), not the
    per-instance forward count.

    Returns:
      total_forwards : sum over instances of darts spent (== the (a) accounting total).
      per_inst_forwards : list[int] forwards each instance spent.
      n_solved       : instances solved within M.
      solve_rate     : n_solved / n_inst.
      n_rounds       : number of chunked rounds executed (active-set drains).
      n_batches      : number of fixed-B padded batches issued across all rounds
                       (the GPU-side batch count; pad rows are wasted compute, billed
                       only for reporting the layout efficiency, NOT the per-inst cost).
    """
    n_inst = len(per_inst_first_success)
    if n_inst == 0:
        return {"total_forwards": 0, "per_inst_forwards": [], "n_solved": 0,
                "solve_rate": 0.0, "n_rounds": 0, "n_batches": 0}
    # active state: per instance, darts drawn so far + solved flag.
    drawn = [0] * n_inst
    solved = [False] * n_inst
    per_inst_forwards = [0] * n_inst
    active = list(range(n_inst))
    n_rounds = 0
    n_batches = 0
    while active:
        n_rounds += 1
        # REPACK the active set into fixed-B padded batches (the GPU layout). Pad rows
        # are repeats of a real active row (scored but discarded). Count batches for
        # the layout report.
        n_batches += (len(active) + B - 1) // B
        next_active = []
        for i in active:
            fsi = per_inst_first_success[i]
            # darts available this round: up to `chunk`, capped by remaining budget.
            d_lo = drawn[i]                              # 0-based darts already drawn
            d_hi = min(d_lo + chunk, M)                  # exclusive upper (darts after round)
            if fsi is not None and d_lo < fsi <= d_hi:
                # solves WITHIN this chunk at 1-based index fsi -> spend exactly the
                # darts up to and INCLUDING the success (fsi total), stop.
                spent = fsi - d_lo
                drawn[i] = fsi
                per_inst_forwards[i] += spent
                solved[i] = True
                # solved -> leave the active set.
            else:
                # no success in this chunk -> spend the full chunk (or remainder to M).
                spent = d_hi - d_lo
                drawn[i] = d_hi
                per_inst_forwards[i] += spent
                if d_hi >= M:
                    pass                                 # budget exhausted -> drop unsolved
                else:
                    next_active.append(i)                # survives to next round
        active = next_active
    n_solved = sum(solved)
    return {
        "total_forwards": int(sum(per_inst_forwards)),
        "per_inst_forwards": per_inst_forwards,
        "n_solved": n_solved,
        "solve_rate": n_solved / n_inst,
        "n_rounds": n_rounds,
        "n_batches": n_batches,
    }


# ===========================================================================
# DEDUCER-EVAL WIRING DRY-RUN (CPU-only contract checks — NO GPU forward)
# ===========================================================================

def _build_batch_arrays(insts: list, s_max: int, n_edges_max: int, k: int) -> dict:
    """Encode a list of instance dicts into stacked numpy batch arrays via the VALIDATED
    encode_instance (graph_coloring_data). Returns the numpy dict (NOT realized Tensors)
    so the CPU smoke can shape-check the deducer contract without touching tinygrad."""
    from mycelium.graph_coloring_data import encode_instance
    encs = [encode_instance(r, s_max, n_edges_max, k) for r in insts]

    def stack(key, dt):
        return np.stack([e[key] for e in encs]).astype(dt)

    return {
        "input_cells": stack("input_cells", np.int32),
        "cell_valid": stack("cell_valid", np.float32),
        "value_domain_mask": stack("value_domain_mask", np.float32),
        "gold": stack("gold", np.int32),
        "membership": stack("membership", np.float32),
        "latent_type": stack("latent_type", np.int32),
        "deduction_depth": [e["deduction_depth"] for e in encs],
        "n": [e["n"] for e in encs],
        "n_edges": [e["n_edges"] for e in encs],
        "band": [e["band"] for e in encs],
    }


def _assert_deducer_contract(arrays: dict, s_max: int, k: int) -> None:
    """Verify the batch arrays satisfy the engine's FactorGraphBatch contract shapes +
    dtypes (the dry-run of the deducer-eval wiring; what factor_breathing_forward will
    consume). Pure numpy — no GPU forward."""
    B = arrays["input_cells"].shape[0]
    assert arrays["input_cells"].shape == (B, s_max)
    assert arrays["cell_valid"].shape == (B, s_max)
    assert arrays["value_domain_mask"].shape == (B, s_max, k)
    assert arrays["gold"].shape == (B, s_max)
    assert arrays["membership"].ndim == 3 and arrays["membership"].shape[0] == B
    assert arrays["membership"].shape[2] == s_max
    assert arrays["latent_type"].shape == (B, arrays["membership"].shape[1])
    # input fully uncolored (coloring is latent); gold in {0(pad)} U {1..k}.
    assert int(arrays["input_cells"].sum()) == 0
    g = arrays["gold"]
    assert g.min() >= 0 and g.max() <= k
    # the proper verifier + accuracy scorer run end-to-end on the GOLD coloring itself:
    # a correct (gold) prediction must score proper==True and cell/puzzle==perfect.
    from mycelium.graph_coloring_data import LTYPE_EDGE
    score = _score_predictions(
        g.copy(), g, arrays["cell_valid"], arrays["membership"],
        arrays["latent_type"], LTYPE_EDGE)
    assert score["n_cells"] > 0
    assert abs(score["cell_eq"] - score["n_cells"]) < 1e-6, "gold cell_acc must be 1.0"
    assert score["puz_ok"] == score["n_puz"], "gold puzzle_acc must be 1.0"
    assert score["proper_ok"] == score["n_puz"], "gold must verify proper on every edge"


# ===========================================================================
# AGGREGATION + OVERLAY + VERDICT
# ===========================================================================

def _pct(xs, q):
    if not xs:
        return float("nan")
    s = sorted(xs)
    i = min(len(s) - 1, int(q * len(s)))
    return s[i]


def _summarise_band(sym_rows: list, ded: dict, k: int) -> dict:
    """Collapse a band's per-instance symbolic rows + pooled deducer counts into a row."""
    decs = [r["decisions"] for r in sym_rows]
    walls = [r["wall_ms"] for r in sym_rows]
    n = len(sym_rows)
    n_timeout = sum(1 for r in sym_rows if r["timed_out"])
    n_budget = sum(1 for r in sym_rows if r["budget_hit"])
    n_solved = sum(1 for r in sym_rows if r["status"] == "solved")
    return {
        "n": n,
        "sym_dec_med": (float(np.median(decs)) if decs else float("nan")),
        "sym_dec_p90": _pct(decs, 0.90),
        "sym_dec_max": (max(decs) if decs else float("nan")),
        "sym_ms_med": (float(np.median(walls)) if walls else float("nan")),
        "sym_ms_p90": _pct(walls, 0.90),
        "sym_ms_max": (max(walls) if walls else float("nan")),
        "sym_timeout_rate": (n_timeout / n) if n else float("nan"),
        "sym_budget_rate": (n_budget / n) if n else float("nan"),
        "sym_solved_rate": (n_solved / n) if n else float("nan"),
        "ded_cell_acc": (ded["cell_eq"] / max(ded["n_cells"], 1)),
        "ded_puzzle_acc": (ded["puz_ok"] / max(ded["n_puz"], 1)),
        "ded_proper_rate": (ded["proper_ok"] / max(ded["n_puz"], 1)),
        "ded_n": ded["n_puz"],
    }


def _verdict(rows_by_band: dict, bands: list, k: int,
             slow_ms: float, slow_timeout_rate: float,
             accurate_margin: float) -> dict:
    """THE OVERLAY VERDICT (explicit sweet-spot vs DEAD rule).

    A band is SYMBOLIC-SLOW if its p90 wall-clock exceeds `slow_ms` OR its timeout-rate
    exceeds `slow_timeout_rate` (a real time tail / blow-up). A band is DEDUCER-ACCURATE
    if its proper-rate exceeds the chance floor (1/k) by at least `accurate_margin`
    (proper-rate is the right metric; cell-acc is reported but proper-rate is the
    success criterion). The SWEET SPOT is any band that is BOTH. If NO band overlaps:
      - if NO band was ever symbolic-slow -> DEAD: 'no blow-up reached' (symbolic stays
        cheap across the whole sweep; standalone-speed has no tail to beat here ->
        redirect to the ordering-prior angle).
      - else (some band slow but the deducer was at chance there) -> DEAD: the deducer
        collapses before symbolic blows up.
    Also flags if the deducer is at chance across the ENTIRE range (nothing to stand on).
    """
    chance = 1.0 / k
    slow_bands = []
    accurate_bands = []
    sweet = []
    any_slow = False
    any_accurate = False
    for c in bands:
        r = rows_by_band[c]
        is_slow = (r["sym_ms_p90"] >= slow_ms) or (r["sym_timeout_rate"] >= slow_timeout_rate)
        is_acc = (r["ded_proper_rate"] >= chance + accurate_margin)
        any_slow = any_slow or is_slow
        any_accurate = any_accurate or is_acc
        if is_slow:
            slow_bands.append(c)
        if is_acc:
            accurate_bands.append(c)
        if is_slow and is_acc:
            sweet.append(c)

    if sweet:
        verdict = "SWEET-SPOT"
        detail = (f"bands {[f'{c:.2f}' for c in sweet]} are BOTH symbolic-slow "
                  f"(p90>={slow_ms:.0f}ms or timeouts>={slow_timeout_rate:.0%}) AND "
                  f"deducer-accurate (proper-rate >= chance {chance:.3f}+{accurate_margin:.2f}). "
                  f"The standalone amortized-speed value-prop HOLDS in this band: the deducer's "
                  f"FIXED-cost forward beats the symbolic time-tail while staying accurate.")
        dead = False
    elif not any_slow:
        verdict = "DEAD (no blow-up reached)"
        detail = (f"NO band reached the symbolic blow-up regime in this range "
                  f"(p90 wall-clock < {slow_ms:.0f}ms and timeout-rate < {slow_timeout_rate:.0%} "
                  f"in EVERY band). On this substrate (n<=s_max) the general symbolic core "
                  f"(GAC==arc-consistency + MRV==DSATUR) collapses the colorable trees, so there "
                  f"is no time-tail to beat. Standalone amortized-speed is DEAD here -> redirect "
                  f"to the ORDERING-PRIOR angle (deducer biases MRV/LCV to cut nodes on an EXACT "
                  f"solve) and/or heterogeneous/differentiable graphs symbolic cannot run at all.")
        dead = True
    else:
        verdict = "DEAD (deducer collapses before symbolic blows up)"
        detail = (f"some bands are symbolic-slow ({[f'{c:.2f}' for c in slow_bands]}) but the "
                  f"deducer is NOT accurate there (proper-rate within {accurate_margin:.2f} of "
                  f"chance {chance:.3f}). The deducer accuracy collapses before symbolic blow-up, "
                  f"so there is no standalone sweet spot -> redirect to the ordering-prior angle.")
        dead = True

    deducer_dead = not any_accurate
    return {
        "verdict": verdict, "detail": detail, "dead": dead,
        "slow_bands": slow_bands, "accurate_bands": accurate_bands, "sweet_bands": sweet,
        "deducer_at_chance_everywhere": deducer_dead, "chance": chance,
    }


def _print_overlay(rows_by_band: dict, bands: list, k: int, fixed_cost: dict,
                   verdict: dict) -> None:
    print(f"\n{'='*100}", flush=True)
    print("  AMORTIZED-FRONTIER OVERLAY — symbolic cost (CPU, per-instance) vs deducer "
          "accuracy (GPU, fixed-cost)", flush=True)
    print("  success metric = EXACT proper-rate (edge verifier); chance floor = "
          f"1/k = {1.0/k:.3f}", flush=True)
    print(f"{'='*100}", flush=True)
    hdr = (f"  {'band(c)':>8} {'n':>5} | {'sym_dec_med':>11} {'sym_dec_p90':>11} "
           f"{'sym_dec_max':>11} {'sym_ms_p90':>10} {'sym_ms_max':>10} "
           f"{'timeout%':>9} {'budget%':>8} | {'ded_cell':>9} {'ded_puz':>8} {'ded_proper':>11}")
    print(hdr, flush=True)
    print("  " + "-" * (len(hdr) - 2), flush=True)
    for c in bands:
        r = rows_by_band[c]
        print(f"  {c:>8.2f} {r['n']:>5} | {r['sym_dec_med']:>11.1f} {r['sym_dec_p90']:>11.1f} "
              f"{r['sym_dec_max']:>11.1f} {r['sym_ms_p90']:>10.2f} {r['sym_ms_max']:>10.2f} "
              f"{r['sym_timeout_rate']*100:>8.1f}% {r['sym_budget_rate']*100:>7.1f}% | "
              f"{r['ded_cell_acc']:>9.3f} {r['ded_puzzle_acc']:>8.3f} {r['ded_proper_rate']:>11.3f}",
              flush=True)

    print(f"\n  DEDUCER FIXED COST (reported separately — the amortized-speed claim is "
          f"FLATNESS vs symbolic blow-up):", flush=True)
    print(f"    one-time (EXCLUDED from per-instance): model+ckpt load = "
          f"{fixed_cost.get('load_ms', float('nan')):.0f}ms, "
          f"JIT compile+warmup = {fixed_cost.get('warmup_ms', float('nan')):.0f}ms", flush=True)
    print(f"    steady-state batched forward: {fixed_cost.get('batch_ms', float('nan')):.1f}ms "
          f"for B={fixed_cost.get('batch', '?')} (= "
          f"{fixed_cost.get('per_inst_ms', float('nan')):.2f}ms/instance amortized, "
          f"K={fixed_cost.get('K','?')} breaths) on GPU", flush=True)
    print(f"    SUBSTRATE NOTE: deducer = GPU batched; symbolic = CPU per-instance. "
          f"These are reported side by side, never cross-claimed.", flush=True)

    print(f"\n  VERDICT: {verdict['verdict']}", flush=True)
    print(f"    {verdict['detail']}", flush=True)
    if verdict["deducer_at_chance_everywhere"]:
        print(f"    [FLAG] the deducer proper-rate is at/near chance ({verdict['chance']:.3f}) "
              f"across the ENTIRE sweep — the ckpt may be mismatched to these bands "
              f"(trained on curriculum, not threshold) or the band is unsolvable for it.",
              flush=True)
    print("", flush=True)


# ===========================================================================
# THE GPU DRIVER (full sweep) — reuses the VALIDATED deducer eval path
# ===========================================================================

def run_gpu_sweep(args) -> None:
    """Build the deducer once (reusing search_coloring's validated model build + ckpt
    load), build per-band banks, run the symbolic solver (CPU, timed) + the deducer
    forward (GPU, fixed-cost) on the SAME banks, overlay + verdict."""
    if args.domain != "coloring":
        _domain_hook(args.domain)  # raises a clear NotImplementedError for the hooks

    import scripts.search_coloring as sc
    from tinygrad import Tensor
    from mycelium.factor_graph_engine import FactorGraphSpec, factor_breathing_forward
    from mycelium.graph_coloring_data import LTYPE_EDGE

    k = args.k
    s_max = args.s_max
    K = args.K
    bands = args.bands
    per_band = args.per_band
    n_max = min(args.max_n, s_max)
    n_min = args.min_n

    print(f"\n=== amortized_frontier_measure — domain={args.domain} ===", flush=True)
    print(f"k={k} s_max={s_max} K={K} bands={[f'{c:.2f}' for c in bands]} "
          f"per_band={per_band} n in [{n_min},{n_max}]", flush=True)
    print(f"symbolic: budget={args.sym_budget} timeout={args.sym_timeout_ms}ms", flush=True)

    # ---- 1. Build all banks first (CPU; needs n_edges_max across the whole sweep). ----
    banks = {}
    for c in bands:
        insts = _build_band(c, per_band, n_min, n_max, k, seed=args.seed + int(c * 100),
                            regular_frac=args.regular_frac)
        banks[c] = insts
        kept = len(insts)
        med_bt = (float(np.median([r["dsatur_backtracks"] for r in insts])) if insts else 0.0)
        print(f"  band c={c:.2f}: {kept}/{per_band} k-colorable instances "
              f"(median DSATUR backtracks={med_bt:.0f})", flush=True)
    all_insts = [r for c in bands for r in banks[c]]
    if not all_insts:
        print("[abort] no instances generated in any band.", flush=True)
        sys.exit(1)
    n_edges_max = max(r["n_edges"] for r in all_insts)
    print(f"  n_edges_max across sweep = {n_edges_max}", flush=True)

    # ---- 2. Build the deducer ONCE (validated path; load + warmup timed separately). --
    spec = FactorGraphSpec(
        s_max=s_max, n_values=k, n_factor_types=1, n_heads=16, k_max=K,
        has_factor_inlet=False,
    )
    Tensor.training = False
    t_load0 = time.perf_counter()
    model = _build_deducer_model(spec, args.ckpt, k, s_max, K, args.seed)
    load_ms = (time.perf_counter() - t_load0) * 1000.0

    # Build a fixed ref batch (the JIT cache key = B, spec). We drive batches through the
    # engine forward directly (the validated factor_breathing_forward), assembling each
    # band's instances into batches sized B = args.eval_batch.
    B = args.eval_batch

    def _pack(insts_slice: list):
        """Pad to B with repeats (last-batch padding, as the loader does) and realize."""
        from mycelium.graph_coloring_data import GraphColoringBatch, encode_instance
        from tinygrad import dtypes
        picks = list(insts_slice)
        n_real = len(picks)
        while len(picks) < B:
            picks.append(insts_slice[0])
        encs = [encode_instance(r, s_max, n_edges_max, k) for r in picks]

        def st_i(key):
            return Tensor(np.stack([e[key] for e in encs]).astype(np.int32),
                          dtype=dtypes.int).contiguous().realize()

        def st_f(key):
            return Tensor(np.stack([e[key] for e in encs]).astype(np.float32),
                          dtype=dtypes.float).contiguous().realize()

        d = {
            "input_cells": st_i("input_cells"), "cell_valid": st_f("cell_valid"),
            "value_domain_mask": st_f("value_domain_mask"), "gold": st_i("gold"),
            "membership": st_f("membership"), "latent_type": st_i("latent_type"),
            "deduction_depth": [e["deduction_depth"] for e in encs],
            "n": [e["n"] for e in encs], "n_edges": [e["n_edges"] for e in encs],
            "band": [e["band"] for e in encs],
        }
        return GraphColoringBatch(d), n_real

    # ---- 3. WARMUP (JIT compile + first forward) — timed separately, EXCLUDED. --------
    warm_insts = all_insts[:B]
    warm_fb, _ = _pack(warm_insts)
    t_warm0 = time.perf_counter()
    _wh, _ = factor_breathing_forward(model, warm_fb, spec, K=K)
    _ = _wh[-1].realize().numpy()
    warmup_ms = (time.perf_counter() - t_warm0) * 1000.0

    # steady-state batched forward time (median of a few replays — JIT compiled now).
    batch_times = []
    for _ in range(5):
        tb0 = time.perf_counter()
        _wh, _ = factor_breathing_forward(model, warm_fb, spec, K=K)
        _ = _wh[-1].realize().numpy()
        batch_times.append((time.perf_counter() - tb0) * 1000.0)
    batch_ms = float(np.median(batch_times))
    fixed_cost = {
        "load_ms": load_ms, "warmup_ms": warmup_ms, "batch_ms": batch_ms,
        "batch": B, "per_inst_ms": batch_ms / B, "K": K,
    }
    print(f"\n  [timing] one-time load={load_ms:.0f}ms warmup(JIT+1st fwd)={warmup_ms:.0f}ms; "
          f"steady batched fwd={batch_ms:.1f}ms/B={B} ({batch_ms/B:.2f}ms/inst).", flush=True)

    # ---- 4. Sweep: symbolic (CPU timed) + deducer (GPU forward) per band. -------------
    rows_by_band = {}
    for c in bands:
        insts = banks[c]
        # --- SYMBOLIC (CPU, per-instance, timed with a hard wall-clock timeout) ---
        sym_rows = []
        for inst in insts:
            sym_rows.append(_solve_symbolic_timed(
                inst, k, budget=args.sym_budget, timeout_ms=args.sym_timeout_ms))
        # --- DEDUCER (GPU forward, batched; fixed cost) on the SAME instances ---
        ded_pool = {"cell_eq": 0.0, "n_cells": 0, "puz_ok": 0, "n_puz": 0, "proper_ok": 0}
        for start in range(0, len(insts), B):
            fb, n_real = _pack(insts[start:start + B])
            lh, _ = factor_breathing_forward(model, fb, spec, K=K)
            final = lh[-1].realize().numpy()                       # (B, S, N)
            pred = (final.argmax(axis=-1) + 1).astype(np.int32)    # (B, S) 1-based
            cv = fb.cell_valid.realize().numpy()
            gold = fb.gold.realize().numpy().astype(np.int32)
            mem = fb.membership.realize().numpy()
            lt = fb.latent_type.realize().numpy().astype(np.int32)
            # score only the n_real real (non-padding-repeat) rows of this batch.
            sc_b = _score_predictions(pred[:n_real], gold[:n_real], cv[:n_real],
                                      mem[:n_real], lt[:n_real], LTYPE_EDGE)
            for key in ded_pool:
                ded_pool[key] += sc_b[key]
        rows_by_band[c] = _summarise_band(sym_rows, ded_pool, k)
        r = rows_by_band[c]
        print(f"  [band c={c:.2f}] sym: dec(med/p90/max)={r['sym_dec_med']:.0f}/"
              f"{r['sym_dec_p90']:.0f}/{r['sym_dec_max']:.0f} "
              f"ms_p90={r['sym_ms_p90']:.1f} timeout={r['sym_timeout_rate']:.0%} | "
              f"ded: cell={r['ded_cell_acc']:.3f} proper={r['ded_proper_rate']:.3f}", flush=True)

    # ---- 5. Overlay + verdict. --------------------------------------------------------
    verdict = _verdict(rows_by_band, bands, k,
                       slow_ms=args.slow_ms, slow_timeout_rate=args.slow_timeout_rate,
                       accurate_margin=args.accurate_margin)
    _print_overlay(rows_by_band, bands, k, fixed_cost, verdict)


# ===========================================================================
# VOLUME-MODE GPU DRIVER — p-per-band + best-of-N(M) + M_eff + throughput
# ===========================================================================

# ===========================================================================
# DETERMINISTIC DART GENERATOR (the parity-by-construction core — reviewer fix)
# ===========================================================================
# THE BUG THIS FIXES: the full-M measurement (run_volume_sweep) and the early-stop
# deployment (run_early_stop) used DIFFERENT dart RNG keyings, so "dart j of instance
# X" was a different sample on the two paths -> the asserted solve-rate parity was a
# TAUTOLOGY (it compared early-stop against flags re-synthesized from the early-stop
# run's OWN solved_at, never against the actual full-M measurement). Worse, the
# symmetry permutation was keyed by the row's POSITION in the current active batch
# (`i`), which the active-set repacking changes across rounds -> dart j was not even
# stable/reproducible for a fixed instance.
#
# THE FIX: ONE generator, keyed by a STABLE hash of (seed, gid, dart_idx, mech) where
# `gid` is the instance's STABLE global id (its per-band index), NOT its transient
# batch position. ORDER- and BATCH-POSITION-INDEPENDENT and fully reproducible: dart j
# of instance X is BYTE-IDENTICAL in run_volume_sweep and run_early_stop, so early-stop
# is provably "the same darts, just verified fewer" and the solve-rate parity holds BY
# CONSTRUCTION (not as a tautology). dart_idx==0 is the DETERMINISTIC baseline
# (multistart: zero noise; symmetry: identity permutation; temp: n/a — sample_idx keyed).

# Sentinel gid for BATCH-SHARED darts (multistart noise hits the whole co-batched
# batch via one shared fg_position_embed assign — so its key MUST be gid-independent).
GID_SHARED = -1


def _dart_key(seed: int, gid: int, dart_idx: int, mech: str) -> int:
    """Stable 31-bit RandomState seed for (seed, gid, dart_idx, mech).

    Uses hashlib (NOT Python's salted hash()) so the key is reproducible ACROSS
    processes and independent of dict/insertion order, batch position, or which round
    draws the dart. The mechanism name is folded in so multistart vs symmetry vs temp
    darts at the same (gid, dart_idx) are independent streams. Pure CPU; no GPU."""
    import hashlib
    h = hashlib.sha256(
        f"{int(seed)}|{int(gid)}|{int(dart_idx)}|{mech}".encode("utf-8")).digest()
    # take 4 bytes -> uint32 -> clamp into the legal RandomState seed range [0, 2**31-1).
    return int.from_bytes(h[:4], "big") % (2 ** 31 - 1)


def _band_dart_seed(master_seed: int, c: float) -> int:
    """The per-band dart master seed shared by BOTH run_volume_sweep and run_early_stop.
    Derived ONLY from (master_seed, band c) so both paths produce the IDENTICAL per-band
    dart streams. Folded through _dart_key with (gid, dart_idx, mech) per dart."""
    return _dart_key(int(master_seed), int(round(c * 1000)), 0, "band")


def _global_inst_id(c: float, local_gid: int) -> int:
    """STABLE global instance id for the per-dart capture: band-namespaced so an instance
    is distinct per (band, per-band index). band c -> int(round(c*1000)) * 1_000_000 +
    local_gid (per-band index). Distinct across bands and instances; reproducible."""
    return int(round(c * 1000)) * 1_000_000 + int(local_gid)


def _draw_dart(mech: str, gid: int, dart_idx: int, *, seed: int,
               pos_embed_shape=None, noise_std: float = 0.0,
               n: int = 0, M: int = 0):
    """Return dart `dart_idx` for the instance with STABLE global id `gid`, keyed by
    _dart_key(seed, gid, dart_idx, mech) — ORDER- and BATCH-POSITION-INDEPENDENT and
    fully reproducible. This is the SINGLE source of dart randomness shared by BOTH the
    full-M measurement (run_volume_sweep) and the early-stop deployment (run_early_stop),
    so parity holds by construction. Pure CPU/numpy — NO forward, NO tinygrad.

    Convention: dart_idx==0 is the DETERMINISTIC baseline (plain argmax forward).

      mech == 'multistart' : returns the noise vector (np.float32, shape pos_embed_shape)
                             added to fg_position_embed. dart_idx==0 -> all-zeros (no
                             perturbation == the argmax run). noise = RandomState(key)
                             .randn(*pos_embed_shape) * noise_std.
                             KEYED BY (seed, dart_idx) ONLY — NOT gid. The perturbation
                             is applied to the SHARED fg_position_embed (one assign-in-
                             place hits the WHOLE co-batched batch), so dart j is the SAME
                             noise for every instance in the batch. Keying by a per-
                             instance gid would be INCOMPATIBLE with the shared mechanism
                             (it would demand a different fg_position_embed per row in one
                             forward). Because the noise is batch-shared, dart j applied to
                             instance X is identical regardless of WHICH other instances
                             co-batch with X -> parity across volume/early-stop holds. We
                             pass gid=GID_SHARED so callers can use the same signature.
      mech == 'symmetry'   : returns the vertex permutation (np.int64, length n) over the
                             instance's `n` real vertices. dart_idx==0 -> identity
                             (== the argmax run). perm = RandomState(key).permutation(n).
      mech == 'temp'       : returns a (M,) array of uniform draws in [0,1) used by the
                             per-cell inverse-CDF categorical sampler. Keyed by gid (the
                             sample index runs WITHIN the dart) so temp's M samples are
                             reproducible per (gid). (temp is measurement-only / 1 forward;
                             the keying is unified for consistency.)
    """
    if mech == "multistart":
        if pos_embed_shape is None:
            raise ValueError("multistart dart requires pos_embed_shape")
        if dart_idx == 0:
            return np.zeros(pos_embed_shape, dtype=np.float32)
        # batch-SHARED noise: key on the sentinel gid (GID_SHARED), NOT the row's gid, so
        # dart j is identical for every co-batched instance (the noise hits the shared
        # fg_position_embed). This is what makes parity true under the shared mechanism.
        rng = np.random.RandomState(_dart_key(seed, GID_SHARED, dart_idx, mech))
        return (rng.randn(*pos_embed_shape) * noise_std).astype(np.float32)
    if mech == "symmetry":
        if n <= 0:
            raise ValueError("symmetry dart requires n>0")
        if dart_idx == 0:
            return np.arange(n, dtype=np.int64)            # identity permutation
        rng = np.random.RandomState(_dart_key(seed, gid, dart_idx, mech))
        return rng.permutation(n).astype(np.int64)
    if mech == "temp":
        if M <= 0:
            raise ValueError("temp dart requires M>0")
        rng = np.random.RandomState(_dart_key(seed, gid, dart_idx, mech))
        return rng.random_sample(M)
    raise ValueError(f"unknown mech {mech!r}")


def _permute_instance(inst: dict, perm: np.ndarray) -> dict:
    """Relabel vertices of `inst` by `perm` (perm[new] = old vertex). Solution-preserving:
    a valid coloring of the relabeled graph maps back (via the inverse) to a valid coloring
    of the original. Returns a NEW instance dict (edges + coloring + n carried)."""
    n = int(inst["n"])
    # inv[old] = new position
    inv = np.empty(n, dtype=np.int64)
    inv[perm] = np.arange(n)
    new_edges = [(int(inv[u]), int(inv[v])) for (u, v) in inst["edges"]]
    new_coloring = [0] * n
    for old in range(n):
        new_coloring[int(inv[old])] = inst["coloring"][old]
    out = dict(inst)
    out["edges"] = new_edges
    out["coloring"] = new_coloring
    return out


def _forward_argmax(model, insts_slice, spec, K, s_max, n_edges_max, k, B, Tensor,
                    factor_breathing_forward):
    """Pack a slice (<=B), run ONE batched forward, return (pred_1based (B,S),
    cell_valid (B,S), membership (B,L,S), latent_type (B,L), final_logits (B,S,N),
    value_domain_mask (B,S,N), n_real). Reuses the validated forward; no engine edit."""
    from mycelium.graph_coloring_data import GraphColoringBatch, encode_instance
    from tinygrad import dtypes
    picks = list(insts_slice)
    n_real = len(picks)
    while len(picks) < B:
        picks.append(insts_slice[0])
    encs = [encode_instance(r, s_max, n_edges_max, k) for r in picks]

    def st_i(key):
        return Tensor(np.stack([e[key] for e in encs]).astype(np.int32),
                      dtype=dtypes.int).contiguous().realize()

    def st_f(key):
        return Tensor(np.stack([e[key] for e in encs]).astype(np.float32),
                      dtype=dtypes.float).contiguous().realize()

    d = {
        "input_cells": st_i("input_cells"), "cell_valid": st_f("cell_valid"),
        "value_domain_mask": st_f("value_domain_mask"), "gold": st_i("gold"),
        "membership": st_f("membership"), "latent_type": st_i("latent_type"),
        "deduction_depth": [e["deduction_depth"] for e in encs],
        "n": [e["n"] for e in encs], "n_edges": [e["n_edges"] for e in encs],
        "band": [e["band"] for e in encs],
    }
    fb = GraphColoringBatch(d)
    lh, _ = factor_breathing_forward(model, fb, spec, K=K)
    final = lh[-1].realize().numpy()                          # (B, S, N)
    pred = (final.argmax(axis=-1) + 1).astype(np.int32)        # (B, S) 1-based
    cv = fb.cell_valid.realize().numpy()
    mem = fb.membership.realize().numpy()
    lt = fb.latent_type.realize().numpy().astype(np.int32)
    vdm = fb.value_domain_mask.realize().numpy()
    return pred, cv, mem, lt, final, vdm, n_real


# ===========================================================================
# JIT VOLUME FORWARD — compile-once / replay (mirrors search_coloring
# ._make_engine_deduce_fn). THE WHOLE REASON FOR THE ADVERSARIAL REVIEW.
# ===========================================================================
# The eager _forward_argmax builds FRESH graph-constant Tensors per call, so
# tinygrad re-traces + recompiles the full K-breath graph EVERY forward (the
# 77ms/inst eager cost). This factory mirrors the VALIDATED search_coloring JIT
# pattern: FIXED, realized input buffers allocated ONCE + a @TinyJit step that
# COMPILES ONCE and REPLAYS for every subsequent forward (all bands, all M draws).
#
# THE CRITICAL CORRECTNESS TRAP (multistart noise must VARY PER REPLAY)
# --------------------------------------------------------------------
# The eager multistart REASSIGNS the attribute: model.fg_position_embed =
# Tensor(orig + noise). Under @TinyJit that is silently WRONG: the compiled
# graph captured the ORIGINAL fg_position_embed buffer object at trace time and
# keeps reading it; a NEW Tensor object bound to the attribute is INVISIBLE to
# the graph -> the noise is FROZEN -> every multistart sample is IDENTICAL ->
# [COLLAPSE: all M identical] -> we MISREAD it as "multistart does not work".
# THE FIX (the substrate law: "assign-in-place fixed buffers for repeated JIT'd
# forwards"): keep model.fg_position_embed a FIXED buffer object and ASSIGN
# IN-PLACE its new values each replay via .assign(...).realize() (eager, OUTSIDE
# the JIT). factor_breathing_forward reads `position_embed = model.fg_position_embed`
# at trace time, so the compiled graph reads the LIVE contents of that SAME object;
# updating the contents in place makes each replay see fresh noise. We NEVER
# rebind the attribute. The noise is computed host-side as numpy (orig + noise_mi)
# then wrapped once -> NO float32 literal is baked into the JIT graph.

def _make_volume_jit_fn(model, spec, B, s_max, n_edges_max, k, K):
    """Build a JIT'd batched forward that COMPILES ONCE then REPLAYS over fixed,
    in-place-updated input buffers. Returns an object with:
      .forward()                  -> (pred_1based, cv, mem, lt, final, vdm)
                                     replay the compiled graph on the CURRENT buffers.
      .set_batch(insts_slice)     -> pack a slice (<=B, padded with repeats) into the
                                     membership/latent_type/cell_valid/input_cells/
                                     value_domain_mask/gold buffers IN-PLACE (fixed
                                     shape). This is the symmetry update path: a
                                     permuted instance is just a different batch packed
                                     into the SAME buffers.
      .set_position_embed(arr)    -> ASSIGN-IN-PLACE the multistart-perturbed initial
                                     residual into model.fg_position_embed (NOT a
                                     reassignment). arr is host numpy (orig + noise).
      .restore_position_embed()   -> restore the original fg_position_embed contents.
      .S / .N                     -> shape constants.

    DISCIPLINE (mirrors _make_engine_deduce_fn):
      * FIXED realized buffers allocated ONCE; shapes CONSTANT for the whole run.
      * assigns happen EAGERLY OUTSIDE the JIT (buf.assign(new).realize()), then the
        JIT step replays — so the "assign-inside-JIT must be returned" quirk does NOT
        apply (the assigns are not inside the traced graph).
      * K, spec, B, n_edges_max live in the closure -> baked into the JIT cache key.
      * NO dtypes.float32 literal: all Tensors use dtypes.float / dtypes.int.
    """
    from tinygrad import Tensor, dtypes
    from tinygrad.engine.jit import TinyJit
    from mycelium.factor_graph_engine import factor_breathing_forward
    from mycelium.graph_coloring_data import encode_instance

    S = s_max
    N = k
    L = n_edges_max

    # ---- FIXED, realized input buffers (allocated ONCE) — the JIT graph inputs. ----
    buf_ic = Tensor(np.zeros((B, S), dtype=np.int32),
                    dtype=dtypes.int).contiguous().realize()
    buf_cv = Tensor(np.zeros((B, S), dtype=np.float32),
                    dtype=dtypes.float).contiguous().realize()
    buf_vdm = Tensor(np.zeros((B, S, N), dtype=np.float32),
                     dtype=dtypes.float).contiguous().realize()
    buf_gold = Tensor(np.zeros((B, S), dtype=np.int32),
                      dtype=dtypes.int).contiguous().realize()
    buf_mem = Tensor(np.zeros((B, L, S), dtype=np.float32),
                     dtype=dtypes.float).contiguous().realize()
    buf_lt = Tensor(np.zeros((B, L), dtype=np.int32),
                    dtype=dtypes.int).contiguous().realize()

    # ORIGINAL fg_position_embed contents (host copy) for multistart restore. We
    # ASSIGN-IN-PLACE into model.fg_position_embed (the SAME object the JIT graph
    # captures) — never rebind the attribute.
    pos_embed_orig = model.fg_position_embed.realize().numpy().copy()

    class _ProxyBatch:
        """Holds the SAME fixed buffer objects the JIT captured (no per-call Tensor
        build). The engine reads membership/latent_type/cell_valid/input_cells/
        value_domain_mask/gold by attribute off this proxy."""
        def __init__(self):
            self.input_cells = buf_ic
            self.cell_valid = buf_cv
            self.value_domain_mask = buf_vdm
            self.gold = buf_gold
            self.membership = buf_mem
            self.latent_type = buf_lt
            self.factor_inlet = None
            self.deduction_depth = [0] * B

    _proxy = _ProxyBatch()

    @TinyJit
    def _step() -> Tensor:
        # K, spec, B baked into the compiled graph (closure). Reads the fixed buffers
        # (captured) AND model.fg_position_embed (captured object — its LIVE contents,
        # updated in place by set_position_embed, are read on each replay). Returns the
        # final-breath logits realized. No host sync inside; no float32 literal.
        logits_history, _ = factor_breathing_forward(model, _proxy, spec, K=K)
        return logits_history[-1].realize()              # (B, S, N)

    def set_batch(insts_slice):
        """Pack a slice (<=B, padded with repeats) and ASSIGN-IN-PLACE into the fixed
        buffers. Returns n_real. The compiled graph recomputes each instance's mask
        from these LIVE membership/latent_type/cell_valid contents on the next replay
        (the mask builder is pure tensor ops, no python value-branching), so a permuted
        graph (symmetry) just enters as different buffer CONTENTS, fixed shape."""
        picks = list(insts_slice)
        n_real = len(picks)
        while len(picks) < B:
            picks.append(insts_slice[0])
        encs = [encode_instance(r, S, L, k) for r in picks]
        ic = np.stack([e["input_cells"] for e in encs]).astype(np.int32)
        cv = np.stack([e["cell_valid"] for e in encs]).astype(np.float32)
        vdm = np.stack([e["value_domain_mask"] for e in encs]).astype(np.float32)
        gold = np.stack([e["gold"] for e in encs]).astype(np.int32)
        mem = np.stack([e["membership"] for e in encs]).astype(np.float32)
        lt = np.stack([e["latent_type"] for e in encs]).astype(np.int32)
        buf_ic.assign(Tensor(ic, dtype=dtypes.int)).realize()
        buf_cv.assign(Tensor(cv, dtype=dtypes.float)).realize()
        buf_vdm.assign(Tensor(vdm, dtype=dtypes.float)).realize()
        buf_gold.assign(Tensor(gold, dtype=dtypes.int)).realize()
        buf_mem.assign(Tensor(mem, dtype=dtypes.float)).realize()
        buf_lt.assign(Tensor(lt, dtype=dtypes.int)).realize()
        return n_real

    def set_position_embed(arr_np):
        """ASSIGN-IN-PLACE the (multistart-perturbed) initial residual into
        model.fg_position_embed. arr_np is host numpy (orig + noise) -> wrapped once
        (no float32 literal in the JIT). NEVER rebinds the attribute: the graph reads
        the SAME captured object's UPDATED contents on the next replay."""
        model.fg_position_embed.assign(
            Tensor(arr_np.astype(np.float32), dtype=dtypes.float)).realize()

    def restore_position_embed():
        """Restore the original fg_position_embed contents (in place)."""
        model.fg_position_embed.assign(
            Tensor(pos_embed_orig.astype(np.float32), dtype=dtypes.float)).realize()

    def forward():
        """Replay the compiled graph on the CURRENT buffers; return numpy reads
        matching _forward_argmax's tuple (minus n_real, which set_batch returns)."""
        final = _step().numpy()                            # (B, S, N) — JIT replay
        pred = (final.argmax(axis=-1) + 1).astype(np.int32)
        cv = buf_cv.numpy()
        mem = buf_mem.numpy()
        lt = buf_lt.numpy().astype(np.int32)
        vdm = buf_vdm.numpy()
        return pred, cv, mem, lt, final, vdm

    class _VolumeJit:
        S = s_max
        N = k

    # Set OUTSIDE the class body: assigning these inside the body would make each name a
    # class-local and shadow the same-named enclosing-function free variable (NameError).
    _VolumeJit.pos_embed_orig = pos_embed_orig
    _VolumeJit.forward = staticmethod(forward)
    _VolumeJit.set_batch = staticmethod(set_batch)
    _VolumeJit.set_position_embed = staticmethod(set_position_embed)
    _VolumeJit.restore_position_embed = staticmethod(restore_position_embed)
    return _VolumeJit()


def _forward_argmax_jit(jit_fn, insts_slice):
    """JIT analog of _forward_argmax: set_batch (in-place) + replay. Returns the SAME
    tuple shape (pred, cv, mem, lt, final, vdm, n_real) so the sweep is drop-in."""
    n_real = jit_fn.set_batch(insts_slice)
    pred, cv, mem, lt, final, vdm = jit_fn.forward()
    return pred, cv, mem, lt, final, vdm, n_real


# ===========================================================================
# JIT SELFTEST — the MAIN THREAD runs this on GPU BEFORE the full sweep.
# Asserts (a) NOISE-VARIES, (b) JIT-vs-EAGER PARITY, (c) SPEED. Nonzero exit on
# (a) or (b) failure. This is the empirical guard against the frozen-noise trap.
# ===========================================================================

def run_jit_selftest(args) -> int:
    """On a tiny bank, prove the JIT volume forward is CORRECT before the sweep:
      (a) NOISE-VARIES: two multistart replays with DIFFERENT in-place fg_position_embed
          noise produce DIFFERENT final logits (proves the noise is NOT frozen under
          JIT — the whole point of the assign-in-place mechanism).
      (b) PARITY: the JIT forward equals the EAGER factor_breathing_forward on the SAME
          unperturbed batch within fp tolerance (max|Δlogit| < jit_selftest_tol),
          proving JIT did not change results.
      (c) SPEED: print JIT batch_ms vs the eager 77ms/inst reference.
    Exit nonzero if (a) or (b) fail."""
    if args.domain != "coloring":
        _domain_hook(args.domain)

    from tinygrad import Tensor
    from mycelium.factor_graph_engine import FactorGraphSpec, factor_breathing_forward
    from mycelium.graph_coloring_data import GraphColoringBatch, encode_instance
    from tinygrad import dtypes

    k = args.k
    s_max = args.s_max
    K = args.K
    B = args.eval_batch
    n_min = args.min_n
    n_max = min(args.max_n, s_max)
    tol = args.jit_selftest_tol

    print(f"\n=== amortized_frontier_measure [JIT-SELFTEST] — domain={args.domain} ===",
          flush=True)
    print(f"k={k} s_max={s_max} K={K} B={B} tol={tol} "
          f"multistart_noise={args.multistart_noise}", flush=True)

    # ---- tiny bank (a few instances) ----
    insts = _build_band(2.0, max(B, 4), n_min, n_max, k,
                        seed=args.seed + 991, regular_frac=args.regular_frac)
    if len(insts) < 1:
        print("[selftest] FAIL: could not build a tiny bank.", flush=True)
        return 1
    n_edges_max = max(r["n_edges"] for r in insts)
    sl = insts[:B]

    spec = FactorGraphSpec(s_max=s_max, n_values=k, n_factor_types=1, n_heads=16,
                           k_max=K, has_factor_inlet=False)
    Tensor.training = False
    model = _build_deducer_model(spec, args.ckpt, k, s_max, K, args.seed)

    jit_fn = _make_volume_jit_fn(model, spec, B, s_max, n_edges_max, k, K)

    ok = True

    def _check(name, cond):
        nonlocal ok
        if not cond:
            ok = False
        print(f"[selftest] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    # ---- (b) PARITY first (also compiles the JIT) — unperturbed batch ----
    jit_fn.restore_position_embed()                 # ensure pristine fg_position_embed
    jit_fn.set_batch(sl)
    pred_j, cv_j, mem_j, lt_j, final_jit, vdm_j = jit_fn.forward()

    # eager forward on the SAME slice (fresh Tensors — the OLD path).
    def _pack_eager(slice_):
        picks = list(slice_)
        while len(picks) < B:
            picks.append(slice_[0])
        encs = [encode_instance(r, s_max, n_edges_max, k) for r in picks]

        def st_i(key):
            return Tensor(np.stack([e[key] for e in encs]).astype(np.int32),
                          dtype=dtypes.int).contiguous().realize()

        def st_f(key):
            return Tensor(np.stack([e[key] for e in encs]).astype(np.float32),
                          dtype=dtypes.float).contiguous().realize()

        d = {
            "input_cells": st_i("input_cells"), "cell_valid": st_f("cell_valid"),
            "value_domain_mask": st_f("value_domain_mask"), "gold": st_i("gold"),
            "membership": st_f("membership"), "latent_type": st_i("latent_type"),
            "deduction_depth": [e["deduction_depth"] for e in encs],
            "n": [e["n"] for e in encs], "n_edges": [e["n_edges"] for e in encs],
            "band": [e["band"] for e in encs],
        }
        return GraphColoringBatch(d)

    fb = _pack_eager(sl)
    lh, _ = factor_breathing_forward(model, fb, spec, K=K)
    final_eager = lh[-1].realize().numpy()           # (B, S, N)
    max_dlogit = float(np.max(np.abs(final_eager - final_jit)))
    print(f"[selftest] JIT-vs-eager max|Δlogit| = {max_dlogit:.3e} (tol {tol:.1e})",
          flush=True)
    _check(f"PARITY: JIT forward == eager forward within {tol:.0e}", max_dlogit < tol)

    # ---- (a) NOISE-VARIES: two multistart replays with DIFFERENT in-place noise ----
    rng = np.random.RandomState(args.seed + 12321)
    jit_fn.set_batch(sl)                              # fix the batch; vary only the noise
    try:
        noiseA = (rng.randn(*jit_fn.pos_embed_orig.shape)
                  * args.multistart_noise).astype(np.float32)
        jit_fn.set_position_embed(jit_fn.pos_embed_orig + noiseA)
        _, _, _, _, final_A, _ = jit_fn.forward()
        noiseB = (rng.randn(*jit_fn.pos_embed_orig.shape)
                  * args.multistart_noise).astype(np.float32)
        jit_fn.set_position_embed(jit_fn.pos_embed_orig + noiseB)
        _, _, _, _, final_B, _ = jit_fn.forward()
    finally:
        jit_fn.restore_position_embed()
    noise_delta = float(np.max(np.abs(final_A - final_B)))
    print(f"[selftest] noise-A vs noise-B max|Δlogit| = {noise_delta:.3e} "
          f"(must be > 0 — proves noise is NOT frozen under JIT)", flush=True)
    _check("NOISE-VARIES: two DIFFERENT-noise replays differ (noise not frozen)",
           noise_delta > 1e-6)
    # and the restore actually returned the graph to the unperturbed output.
    jit_fn.set_batch(sl)
    _, _, _, _, final_restored, _ = jit_fn.forward()
    restore_delta = float(np.max(np.abs(final_restored - final_jit)))
    _check("RESTORE: fg_position_embed restored in place (output back to unperturbed)",
           restore_delta < tol)

    # ---- (c) SPEED: median JIT batch_ms vs eager 77ms/inst reference ----
    jit_fn.set_batch(sl)
    jit_fn.forward()                                 # warm (already compiled by parity)
    times = []
    for _ in range(7):
        t0 = time.perf_counter()
        jit_fn.forward()
        times.append((time.perf_counter() - t0) * 1000.0)
    batch_ms = float(np.median(times))
    eager_per_inst = 77.0
    print(f"[selftest] JIT steady batched fwd = {batch_ms:.1f}ms/B={B} "
          f"({batch_ms / B:.2f}ms/inst) vs eager reference {eager_per_inst:.0f}ms/inst "
          f"(~{eager_per_inst / max(batch_ms / B, 1e-6):.1f}x).", flush=True)

    print(f"\n[selftest] {'ALL PASS' if ok else 'SOME FAILED'}", flush=True)
    return 0 if ok else 1


# ===========================================================================
# NON-INVASIVE PER-DART SILHOUETTE CAPTURE (opt-in: --capture-darts)
# ===========================================================================
# THE HYPOTHESIS (Anna-Karenina / "good drivers are good in the same way"): in the
# generate-and-verify volume run, the deducer throws M solution-preserving symmetry
# darts per instance; a FREE exact verifier (_coloring_proper_np) tags each VALID or
# INVALID. The claim: VALID darts CLUSTER (a shared common-mode silhouette) while
# INVALID darts SCATTER. We test it by CAPTURING each dart's silhouette + valid-flag
# and probing for cluster-separability (scripts/dart_cluster_probe.py).
#
# THE SILHOUETTE: the FINAL-breath readout representation pooled over valid cells —
# i.e. mean over the dart's valid cells of the readout-LN 1024-d hidden x at the LAST
# breath -> one (H,) vector per dart. This is EXACTLY the engine's calibration pool
# input (factor_graph_engine: pool over cell_valid of the readout-LN x at each breath),
# read at the final breath.
#
# NON-INVASIVE: we mirror scripts/probe_svd_collapse.py EXACTLY — monkeypatch
# mycelium.breathing._layernorm (the engine imports it locally inside
# factor_breathing_forward as `from mycelium.breathing import _layernorm`, so patching
# mycelium.breathing._layernorm intercepts the readout call). We capture ONLY the
# readout LN (gamma IS model.ln_f_g; every per-layer LN uses a DIFFERENT gamma object,
# so the readout is unambiguous), keeping ONLY the LAST call's output per forward (the
# K-th == final breath; we overwrite a single slot each call). The engine
# (mycelium/factor_graph_engine.py) + oracle (mycelium/kenken.py) stay git-clean; the
# hook is installed ONLY when --capture-darts is set, so default volume mode is
# byte-identical.

class _DartCapture:
    """Accumulates per-dart silhouettes + flags + ids + bands, dumps an .npz.

    Install pattern (mirrors probe_svd_collapse): the caller wraps the readout LN via
    `install(model)` and, around EACH forward, calls `arm()` then reads the captured
    final-breath readout reps via `pooled_reps(cell_valid)` (one (H,) per row pooled over
    valid cells). The hook only records when gamma IS model.ln_f_g and keeps ONLY the last
    breath's output (overwrites the slot each readout call), so after a K-breath forward the
    slot holds the FINAL-breath readout rep (B, S, H). `add_dart` appends one row's
    silhouette + metadata. `dump(path, meta)` writes the npz."""

    def __init__(self, mech: str, rep: str = "readout"):
        self.mech = mech
        # rep: "readout" -> the final-breath readout-LN 1024d silhouette (the baseline
        # probe target); "waist" -> the final-breath WAIST d-rep (B,S,d), pooled over valid
        # cells, exposed by the engine via model.fg_waist_capture (RE-PROBE the common mode
        # at the bottleneck the waist created). "waist" requires a waist-attached model.
        self.rep = rep
        self.reps: list = []         # list of (H,) float32 silhouettes
        self.valid: list = []        # list of bool valid-flags
        self.inst_id: list = []      # list of int stable global ids
        self.band: list = []         # list of float band c
        self._orig_ln = None
        self._model = None
        self._slot = {"last": None}  # holds the MOST-RECENT readout-LN output (numpy)
        self._installed = False

    def install(self, model) -> None:
        """Install the per-forward rep grabber. For rep=='readout': monkeypatch
        mycelium.breathing._layernorm to grab the readout-LN output (gamma IS model.ln_f_g)
        into self._slot['last'], overwriting each call -> the FINAL breath after a K-breath
        forward. For rep=='waist': install the engine's d-rep sink (model.fg_waist_capture =
        a fresh list each arm()); the engine appends each breath's (B,S,d) d-rep, so the LAST
        element is the final-breath waist d-rep. NON-INVASIVE either way — the engine is
        git-clean (the sink is read via getattr; default-absent -> no-op); uninstall()
        restores _layernorm and removes the sink."""
        import mycelium.breathing as breathing_mod
        from tinygrad import dtypes
        self._model = model
        if self.rep == "waist":
            # The engine exposes the d-rep through model.fg_waist_capture (a list); we read
            # the LAST appended (B,S,d) tensor per forward. No _layernorm patch needed.
            if getattr(model, "fg_waist_down", None) is None:
                raise RuntimeError(
                    "_DartCapture(rep='waist') needs a waist-attached model "
                    "(model.fg_waist_down is None — run with FG_WAIST=1 + a waist ckpt).")
            self._installed = True
            return
        self._orig_ln = breathing_mod._layernorm
        orig = self._orig_ln
        slot = self._slot

        def _patched_layernorm(x, gamma, beta, eps=1e-5):
            out = orig(x, gamma, beta, eps)
            if gamma is model.ln_f_g:
                slot["last"] = out.cast(dtypes.float).realize().numpy()  # (B, S, H)
            return out

        breathing_mod._layernorm = _patched_layernorm
        self._installed = True

    def uninstall(self) -> None:
        """Restore the original _layernorm + remove the waist sink (idempotent)."""
        if not self._installed:
            return
        if self.rep == "waist":
            if self._model is not None and hasattr(self._model, "fg_waist_capture"):
                self._model.fg_waist_capture = None
            self._installed = False
            return
        if self._orig_ln is not None:
            import mycelium.breathing as breathing_mod
            breathing_mod._layernorm = self._orig_ln
            self._installed = False

    def arm(self) -> None:
        """Reset the slot before a forward (so a stale rep can never be misread). For
        rep=='waist', install a FRESH d-rep sink list on the model so the engine appends
        this forward's per-breath d-reps into it (the last == final breath)."""
        self._slot["last"] = None
        if self.rep == "waist" and self._model is not None:
            self._model.fg_waist_capture = []

    def _finalize_waist_slot(self) -> None:
        """Pull the FINAL-breath waist d-rep out of the engine's sink into self._slot.
        Called by pooled_reps for rep=='waist'. The sink holds K (B,S,d) tensors; the last
        is the final breath."""
        from tinygrad import dtypes
        sink = getattr(self._model, "fg_waist_capture", None)
        if not sink:
            self._slot["last"] = None
            return
        self._slot["last"] = sink[-1].cast(dtypes.float).realize().numpy()  # (B,S,d)

    def pooled_reps(self, cell_valid_np: np.ndarray) -> np.ndarray:
        """Pool the captured FINAL-breath readout reps over each row's valid cells.

        cell_valid_np : (B, S) float. Returns (B, H) float32 — per row, the mean over
        cell_valid>0.5 of the readout-LN hidden (the silhouette). Pooling is over a SET of
        cells, so it is permutation-invariant: a permuted (symmetry) forward's reps pool to
        the SAME silhouette set the original would (no inverse-map needed). For rep=='waist'
        the slot is the final-breath WAIST d-rep (B,S,d) and the same pooling applies."""
        if self.rep == "waist":
            self._finalize_waist_slot()
        last = self._slot["last"]
        if last is None:
            raise RuntimeError(
                "_DartCapture.pooled_reps called but no readout-LN output was captured — "
                "the hook is not installed or the forward did not run the readout.")
        B, S, H = last.shape
        out = np.zeros((B, H), dtype=np.float32)
        for b in range(B):
            valid = cell_valid_np[b] > 0.5
            nv = int(valid.sum())
            if nv == 0:
                continue
            out[b] = last[b][valid].mean(axis=0).astype(np.float32)
        return out

    def add_dart(self, rep_h: np.ndarray, valid_flag: bool, inst_id: int,
                 band_c: float) -> None:
        self.reps.append(np.asarray(rep_h, dtype=np.float32))
        self.valid.append(bool(valid_flag))
        self.inst_id.append(int(inst_id))
        self.band.append(float(band_c))

    def dump(self, path: str, meta: dict) -> dict:
        """Write reps (n_darts, H) float32, valid (n_darts,) bool, inst_id (n_darts,) int,
        band (n_darts,) float, + a meta dict (object) to `path` (.npz)."""
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        if self.reps:
            reps = np.stack(self.reps, axis=0).astype(np.float32)
        else:
            H = int(meta.get("H", 0))
            reps = np.zeros((0, H), dtype=np.float32)
        valid = np.asarray(self.valid, dtype=bool)
        inst_id = np.asarray(self.inst_id, dtype=np.int64)
        band = np.asarray(self.band, dtype=np.float64)
        np.savez(path, reps=reps, valid=valid, inst_id=inst_id, band=band,
                 meta=np.array(meta, dtype=object))
        n = reps.shape[0]
        nvalid = int(valid.sum())
        print(f"\n  [capture-darts] wrote {path}: {n} darts "
              f"({nvalid} VALID / {n - nvalid} INVALID), reps {reps.shape}, "
              f"{len(set(inst_id.tolist()))} distinct instances.", flush=True)
        return {"n_darts": n, "n_valid": nvalid, "shape": reps.shape}


def run_volume_sweep(args) -> None:
    """VOLUME mode: on the TRAINED-distribution banks, measure (A) single-sample argmax
    proper-rate p per band, (B) best-of-N solve-rate(M) + independent-ideal + M_eff across
    temp/multistart/symmetry with collapse detection, (C) honest throughput. Reuses the
    validated model-build / encode / forward / verifier — engine untouched."""
    if args.domain != "coloring":
        _domain_hook(args.domain)

    from tinygrad import Tensor, dtypes
    from mycelium.factor_graph_engine import FactorGraphSpec, factor_breathing_forward
    from mycelium.graph_coloring_data import LTYPE_EDGE

    k = args.k
    s_max = args.s_max
    K = args.K
    bands = args.bands
    per_band = args.per_band
    n_max = min(args.max_n, s_max)
    n_min = args.min_n
    B = args.eval_batch
    m_max = args.m_max
    mechanisms = args.diversity
    m_grid = [m for m in (1, 2, 4, 8, 16, 32, 64, 128, 256) if m <= m_max]
    if m_max not in m_grid:
        m_grid.append(m_max)
    m_grid = sorted(set(m_grid))

    print(f"\n=== amortized_frontier_measure [VOLUME] — domain={args.domain} ===", flush=True)
    print(f"k={k} s_max={s_max} K={K} bands={[f'{c:.2f}' for c in bands]} "
          f"per_band={per_band} n in [{n_min},{n_max}] B={B}", flush=True)
    print(f"diversity={mechanisms} M_max={m_max} M_grid={m_grid} "
          f"temp_tau={args.temp} multistart_noise={args.multistart_noise}", flush=True)

    # ---- OOD WARNING (reviewer fix #2): flag bands / n outside the trained distribution.
    for c in bands:
        if c > TRAINED_BAND_C_MAX + 1e-9:
            print(f"  [OOD WARNING] band c={c:.2f} > ckpt trained max density "
                  f"{TRAINED_BAND_C_MAX:.2f} (BANDS d10..d25 = {TRAINED_BAND_DENSITIES}); "
                  f"proper-rate p there is OFF-DISTRIBUTION and understates a fair p.",
                  flush=True)
    if n_min > TRAINED_N_MIN:
        print(f"  [OOD WARNING] --min-n {n_min} narrows the trained n-range "
              f"(ckpt trained on n in [{TRAINED_N_MIN},{s_max}]); p is measured on a "
              f"HARDER sub-band than training -> understates a fair p.", flush=True)

    # ---- 1. Build banks (trained distribution by default). Empty/under-filled guard. --
    banks = {}
    for c in bands:
        insts = _build_band(c, per_band, n_min, n_max, k, seed=args.seed + int(c * 100),
                            regular_frac=args.regular_frac)
        kept = len(insts)
        if kept == 0:
            print(f"  [FLAG] band c={c:.2f}: 0 k-colorable instances generated — SKIP.",
                  flush=True)
            continue
        if kept < per_band:
            print(f"  [FLAG] band c={c:.2f}: under-filled {kept}/{per_band} k-colorable "
                  f"instances (density too high for k={k}?) — proceeding on {kept}.",
                  flush=True)
        banks[c] = insts
        med_bt = float(np.median([r["dsatur_backtracks"] for r in insts]))
        print(f"  band c={c:.2f}: {kept}/{per_band} k-colorable instances "
              f"(median DSATUR backtracks={med_bt:.0f})", flush=True)
    bands = [c for c in bands if c in banks]
    if not bands:
        print("[abort] no instances generated in any band.", flush=True)
        sys.exit(1)
    all_insts = [r for c in bands for r in banks[c]]
    n_edges_max = max(r["n_edges"] for r in all_insts)
    print(f"  n_edges_max across sweep = {n_edges_max}", flush=True)

    # ---- 2. Empirical proper-rate floor per band (reviewer fix #1). -------------------
    floors = {}
    for c in bands:
        floors[c] = _empirical_proper_floor(banks[c], s_max, n_edges_max, k,
                                            seed=args.seed + 777 + int(c * 100),
                                            n_draws=args.floor_draws)
        print(f"  [floor] band c={c:.2f}: empirical uniform-random proper-rate = "
              f"{floors[c]:.5f} (1/k={1.0/k:.3f} is the CELL floor, NOT this)", flush=True)

    # ---- 3. Build the deducer ONCE (validated path; timed separately). ----------------
    spec = FactorGraphSpec(s_max=s_max, n_values=k, n_factor_types=1, n_heads=16,
                           k_max=K, has_factor_inlet=False)
    Tensor.training = False
    t_load0 = time.perf_counter()
    model = _build_deducer_model(spec, args.ckpt, k, s_max, K, args.seed)
    load_ms = (time.perf_counter() - t_load0) * 1000.0
    pos_embed_orig = model.fg_position_embed.realize().numpy().copy()   # for multistart restore

    # ---- OPT-IN PER-DART SILHOUETTE CAPTURE (--capture-darts; default OFF) -------------
    # NON-INVASIVE readout-LN hook (mirrors probe_svd_collapse). Installed ONLY when
    # capturing, so default volume mode is byte-identical. Captures the FINAL-breath
    # readout rep pooled over valid cells per dart, for the chosen per-dart-forward
    # mechanism (default symmetry — the hypothesis is about vertex-permutation darts).
    capture = None
    if args.capture_darts:
        cap_mech = args.capture_mech
        if cap_mech not in mechanisms:
            raise SystemExit(
                f"--capture-darts requested --capture-mech={cap_mech} but it is not in "
                f"--diversity={mechanisms}. Add it (e.g. --diversity symmetry) so the "
                f"per-dart forwards run.")
        cap_rep = getattr(args, "capture_rep", "readout")
        capture = _DartCapture(cap_mech, rep=cap_rep)
        capture.install(model)
        print(f"\n  [capture-darts] ENABLED: capturing per-dart final-breath "
              f"{'WAIST d-rep' if cap_rep == 'waist' else 'readout silhouette'} "
              f"for mech='{cap_mech}' -> {args.capture_darts}", flush=True)

    # ---- FORWARD-PATH SWITCH: jit (default, production path) | eager (parity reference).
    # CAPTURE FORCES EAGER: the non-invasive readout-LN hook calls .realize().numpy() on
    # each readout (to grab the host-side rep), which would force a host-sync INSIDE the
    # @TinyJit-traced _step (factor_breathing_forward runs inside the JIT). That corrupts
    # the trace. So capture mode runs the EAGER forward (factor_breathing_forward called
    # directly) — exactly the path scripts/probe_svd_collapse.py uses for the same hook.
    # The eager path is numerically identical (the JIT is only a speed optimization).
    use_jit = (args.forward == "jit") and (capture is None)
    if capture is not None and args.forward == "jit":
        print("  [capture-darts] forcing EAGER forward (the readout-LN hook syncs to host "
              "per breath, incompatible with @TinyJit tracing).", flush=True)
    jit_fn = None
    if use_jit:
        # Build the JIT'd forward (compile-once / replay). The eager path stays the
        # parity reference (--forward eager) + the fallback.
        jit_fn = _make_volume_jit_fn(model, spec, B, s_max, n_edges_max, k, K)
        jit_fn.restore_position_embed()             # pristine fg_position_embed contents

    # Unified forward closures so the per-band sweep is path-agnostic. Both return the
    # SAME tuple (pred, cv, mem, lt, final, vdm, n_real). The JIT path assigns-in-place
    # into fixed buffers then replays; the eager path builds fresh Tensors (re-traces).
    def _fwd(insts_slice):
        if use_jit:
            return _forward_argmax_jit(jit_fn, insts_slice)
        return _forward_argmax(model, insts_slice, spec, K, s_max, n_edges_max, k, B,
                               Tensor, factor_breathing_forward)

    def _set_pos_embed(arr_np):
        """ASSIGN-IN-PLACE the multistart noise. JIT: into the captured fg_position_embed
        object (so the compiled graph reads fresh noise per replay — NEVER a rebind).
        Eager: in-place too (keeps the object identity; harmless, still varies)."""
        if use_jit:
            jit_fn.set_position_embed(arr_np)
        else:
            model.fg_position_embed.assign(
                Tensor(arr_np.astype(np.float32), dtype=dtypes.float)).realize()

    def _restore_pos_embed():
        if use_jit:
            jit_fn.restore_position_embed()
        else:
            model.fg_position_embed.assign(
                Tensor(pos_embed_orig.astype(np.float32), dtype=dtypes.float)).realize()

    # warmup (JIT compile) on one batch — EXCLUDED from per-instance timing.
    warm = all_insts[:B]
    t_warm0 = time.perf_counter()
    _fwd(warm)
    warmup_ms = (time.perf_counter() - t_warm0) * 1000.0
    batch_times = []
    for _ in range(5):
        tb0 = time.perf_counter()
        _fwd(warm)
        batch_times.append((time.perf_counter() - tb0) * 1000.0)
    batch_ms = float(np.median(batch_times))
    print(f"\n  [timing] forward={args.forward} load={load_ms:.0f}ms "
          f"warmup(JIT+1st)={warmup_ms:.0f}ms; steady batched fwd={batch_ms:.1f}ms/B={B} "
          f"({batch_ms/B:.2f}ms/inst).", flush=True)

    # ---- 4. Per-band measurements. ----------------------------------------------------
    # DART RNG: ALL dart randomness flows through _draw_dart, keyed by the per-band
    # _band_dart_seed(args.seed, c) + the instance's STABLE gid (its per-band index) +
    # the global dart index. run_early_stop uses the IDENTICAL seed convention, so dart j
    # of instance X is BYTE-IDENTICAL on both paths -> solve-rate parity by construction
    # (no sequential rng_master stream that the two paths could diverge on).
    vol_rows = {}
    for c in bands:
        insts = banks[c]
        n_inst = len(insts)
        dart_seed = _band_dart_seed(args.seed, c)
        # ===== (A) p_argmax: single-sample argmax proper-rate (deterministic baseline) ==
        argmax_proper = 0
        # ===== (B) per-mechanism per-instance valid flags for best-of-M ================
        mech_flags = {m: [] for m in mechanisms}          # m -> list of (M,) bool arrays
        mech_collapse = {m: {"distinct": [], "entropy": []} for m in mechanisms}
        mech_forwards = {}                                # m -> n_forwards per instance batch

        for start in range(0, n_inst, B):
            sl = insts[start:start + B]
            n_real = len(sl)
            gids = list(range(start, start + n_real))     # STABLE per-band instance ids

            # --- baseline argmax forward (also the argmax member of multistart/symmetry M=1) ---
            if capture is not None:
                capture.arm()
            pred_am, cv, mem, lt, final, vdm, _ = _fwd(sl)
            # CAPTURE dart 0 (the identity/argmax member) silhouettes BEFORE any later
            # _fwd overwrites the readout slot. (m_max, n_real, H) batch-scoped buffer;
            # filled per-dart below, appended to the global capture in dart order.
            cap_reps = None
            if capture is not None:
                pooled0 = capture.pooled_reps(cv)             # (B, H) — argmax/identity dart 0
                H_cap = pooled0.shape[1]
                cap_reps = np.zeros((m_max, n_real, H_cap), dtype=np.float32)
                cap_reps[0] = pooled0[:n_real]
            for bi in range(n_real):
                argmax_proper += int(_coloring_proper_np(
                    pred_am[bi], mem[bi], lt[bi], cv[bi], LTYPE_EDGE))

            # --- TEMP: M categorical samples from ONE forward (cheap, 1 forward) ---
            if "temp" in mechanisms:
                mech_forwards["temp"] = 1
                for bi in range(n_real):
                    valid = cv[bi] > 0.5
                    # temp rng keyed by the STABLE gid (dart_idx 0; the M categorical
                    # draws happen WITHIN this one forward) -> reproducible per instance,
                    # unified with _draw_dart's keying (temp is measurement-only).
                    temp_rng = np.random.RandomState(_dart_key(dart_seed, gids[bi], 0, "temp"))
                    samp = _temp_sample_from_logits(
                        final[bi], valid, vdm[bi], args.temp, m_max, temp_rng)  # (M,S)
                    flags = np.array([
                        _coloring_proper_np(samp[m], mem[bi], lt[bi], cv[bi], LTYPE_EDGE)
                        for m in range(m_max)], dtype=bool)
                    mech_flags["temp"].append(flags)
                    dct, ent = _distinct_and_entropy(samp, valid)
                    mech_collapse["temp"]["distinct"].append(dct)
                    mech_collapse["temp"]["entropy"].append(ent)

            # --- MULTISTART: M coherent re-runs with perturbed initial residual ---
            if "multistart" in mechanisms:
                mech_forwards["multistart"] = m_max
                ms_preds = np.zeros((m_max, n_real, s_max), dtype=np.int32)
                # sample 0 = the unperturbed argmax run (already have it).
                ms_preds[0] = pred_am[:n_real]
                # NOTE: the M noise draws are SHARED across co-batched instances of this
                # slice (one perturbation of fg_position_embed per re-run hits the whole
                # batch). Per-instance best-of-M diversity is preserved (each instance still
                # sees M distinct starts); only the cross-instance start-noise is correlated.
                # try/finally (reviewer fix #2): ALWAYS restore the original embed, even if a
                # forward raises mid-sweep — otherwise the model stays perturbed and corrupts
                # later bands/mechanisms.
                #
                # JIT CORRECTNESS (the whole point of the adversarial review): the noise is
                # ASSIGNED IN-PLACE into the CAPTURED fg_position_embed buffer via
                # _set_pos_embed (-> jit_fn.set_position_embed -> model.fg_position_embed
                # .assign(...).realize()), NOT rebound to a NEW Tensor object. The compiled
                # graph reads `position_embed = model.fg_position_embed` (the SAME object) at
                # trace time, so updating its CONTENTS in place makes each replay see fresh
                # noise. Rebinding (model.fg_position_embed = Tensor(...)) would FREEZE the
                # noise (the graph keeps the old buffer) -> all M identical -> false collapse.
                # The noise is computed host-side (numpy orig+noise) -> wrapped once -> no
                # float32 literal baked into the JIT graph.
                cap_ms = (capture is not None and capture.mech == "multistart"
                          and cap_reps is not None)
                try:
                    for mi in range(1, m_max):
                        # dart `mi` noise from the SHARED dart generator (keyed by
                        # (dart_seed, dart_idx) only — batch-shared; gid ignored). dart j
                        # is IDENTICAL here and in run_early_stop._es_dart_for_batch.
                        noise = _draw_dart("multistart", GID_SHARED, mi, seed=dart_seed,
                                           pos_embed_shape=pos_embed_orig.shape,
                                           noise_std=args.multistart_noise)
                        _set_pos_embed(pos_embed_orig + noise)   # ASSIGN-IN-PLACE, not rebind
                        if cap_ms:
                            capture.arm()
                        p_m, _, _, _, _, _, _ = _fwd(sl)
                        ms_preds[mi] = p_m[:n_real]
                        if cap_ms:
                            # multistart runs on the ORIGINAL graph slice -> pool over cv.
                            pooled_m = capture.pooled_reps(cv)        # (B, H)
                            cap_reps[mi] = pooled_m[:n_real]
                finally:
                    _restore_pos_embed()                          # restore IN-PLACE
                for bi in range(n_real):
                    valid = cv[bi] > 0.5
                    samp = ms_preds[:, bi, :]                          # (M, S)
                    flags = np.array([
                        _coloring_proper_np(samp[m], mem[bi], lt[bi], cv[bi], LTYPE_EDGE)
                        for m in range(m_max)], dtype=bool)
                    mech_flags["multistart"].append(flags)
                    dct, ent = _distinct_and_entropy(samp, valid)
                    mech_collapse["multistart"]["distinct"].append(dct)
                    mech_collapse["multistart"]["entropy"].append(ent)
                    # CAPTURE: append one dart per (instance, dart_idx) with its valid-flag.
                    if cap_ms:
                        global_id = _global_inst_id(c, gids[bi])
                        for mi in range(m_max):
                            capture.add_dart(cap_reps[mi, bi], bool(flags[mi]),
                                             global_id, c)

            # --- SYMMETRY: M vertex-permutations (relabel -> forward -> inverse-map -> verify ORIGINAL) ---
            if "symmetry" in mechanisms:
                mech_forwards["symmetry"] = m_max
                sym_preds = [np.zeros((m_max, s_max), dtype=np.int32) for _ in range(n_real)]
                # build M permuted variants of the slice; sample 0 = identity (= argmax run).
                for bi in range(n_real):
                    sym_preds[bi][0] = pred_am[bi]
                # JIT: each permuted slice enters as DIFFERENT membership/latent_type/
                # cell_valid/input_cells buffer CONTENTS (fixed shape) via _fwd ->
                # set_batch's in-place assigns; the compiled graph recomputes the mask
                # from those LIVE contents on each replay (the mask builder is pure tensor
                # ops). The inverse-map + verify run on the ORIGINAL graph (unchanged).
                cap_sym = (capture is not None and capture.mech == "symmetry"
                           and cap_reps is not None)
                for mi in range(1, m_max):
                    perms = []
                    permuted = []
                    for bi, r in enumerate(sl[:n_real]):
                        nn = int(r["n"])
                        # dart `mi` permutation from the SHARED dart generator, keyed by
                        # the STABLE gid (gids[bi]) — NOT the batch position. dart j is
                        # IDENTICAL here and in run_early_stop._es_dart_for_batch.
                        perm = _draw_dart("symmetry", gids[bi], mi, seed=dart_seed, n=nn)
                        perms.append(perm)
                        permuted.append(_permute_instance(r, perm))
                    if cap_sym:
                        capture.arm()
                    p_m, cv_p, _, _, _, _, _ = _fwd(permuted)
                    if cap_sym:
                        # silhouette = FINAL-breath readout pooled over the permuted graph's
                        # valid cells. Pooling is over a SET -> permutation-invariant, so the
                        # permuted forward pools to the SAME silhouette the original would
                        # (no inverse-map needed for a pooled vector).
                        pooled_m = capture.pooled_reps(cv_p)         # (B, H)
                        cap_reps[mi] = pooled_m[:n_real]
                    # inverse-map: prediction on relabeled graph -> original vertex order.
                    for bi in range(n_real):
                        nn = int(sl[bi]["n"])
                        perm = perms[bi]                              # perm[new]=old
                        back = np.zeros((s_max,), dtype=np.int32)
                        # relabeled vertex `new` corresponds to original vertex perm[new].
                        for new in range(nn):
                            back[perm[new]] = p_m[bi, new]
                        sym_preds[bi][mi] = back
                for bi in range(n_real):
                    valid = cv[bi] > 0.5
                    samp = sym_preds[bi]                              # (M, S)
                    flags = np.array([
                        _coloring_proper_np(samp[m], mem[bi], lt[bi], cv[bi], LTYPE_EDGE)
                        for m in range(m_max)], dtype=bool)
                    mech_flags["symmetry"].append(flags)
                    dct, ent = _distinct_and_entropy(samp, valid)
                    mech_collapse["symmetry"]["distinct"].append(dct)
                    mech_collapse["symmetry"]["entropy"].append(ent)
                    # CAPTURE: append one dart per (instance, dart_idx) with its valid-flag.
                    # inst_id is BAND-NAMESPACED + per-band-stable: distinct per (band,inst).
                    if cap_sym:
                        global_id = _global_inst_id(c, gids[bi])
                        for mi in range(m_max):
                            capture.add_dart(cap_reps[mi, bi], bool(flags[mi]),
                                             global_id, c)

        p_argmax = argmax_proper / n_inst
        floor = floors[c]
        # per-mechanism best-of-M curves.
        mech_curves = {}
        for m in mechanisms:
            # multistart/symmetry: slot 0 is the deterministic argmax, slots 1..M-1 are
            # perturbed/permuted -> coherent=True so p1_div (the clean diversity rate) +
            # the diversity ideal/M_eff are computed (reviewer fix #1). temp is exchangeable.
            mech_curves[m] = _solve_rate_curve(
                mech_flags[m], m_grid, coherent=(m in ("multistart", "symmetry")))
        vol_rows[c] = {
            "n_inst": n_inst, "p_argmax": p_argmax, "floor": floor,
            "mech_curves": mech_curves, "mech_collapse": mech_collapse,
            "mech_forwards": mech_forwards,
            # raw per-instance valid-flags, kept so the (C) early-stop accounting can be
            # derived with ZERO new forwards (deliverable (a)).
            "mech_flags": mech_flags,
        }
        # progress line.
        bo = {m: mech_curves[m]["solve_emp"].get(m_max, float("nan")) for m in mechanisms}
        print(f"  [band c={c:.2f}] p_argmax={p_argmax:.3f} (floor {floor:.4f}) | "
              + " ".join(f"{m}:bo{m_max}={bo[m]:.3f}" for m in mechanisms), flush=True)

    # ---- 5. Report. -------------------------------------------------------------------
    _print_volume(vol_rows, bands, k, m_grid, m_max, mechanisms, args,
                  fixed_cost={"load_ms": load_ms, "warmup_ms": warmup_ms,
                              "batch_ms": batch_ms, "batch": B, "K": K})

    # ---- 6. Dump per-dart silhouettes (--capture-darts) + RESTORE the hook. ------------
    if capture is not None:
        try:
            _rep_desc = (
                "final-breath WAIST d-rep (B,S,d), mean-pooled over valid cells "
                "(cell_valid>0.5), one (d,) silhouette per dart — the common mode the "
                "in-deducer waist created (re-probe target)"
                if capture.rep == "waist" else
                "final-breath readout-LN 1024d hidden, mean-pooled over valid cells "
                "(cell_valid>0.5), one (H,) silhouette per dart")
            _rep_dim = (int(model.fg_waist_down.shape[1])
                        if capture.rep == "waist" and getattr(model, "fg_waist_down", None) is not None
                        else int(model.ln_f_g.shape[0]))
            meta = {
                "ckpt": args.ckpt, "domain": args.domain, "mech": capture.mech,
                "capture_rep": capture.rep,
                "k": k, "s_max": s_max, "K": K, "m_max": m_max,
                "bands": [float(c) for c in bands], "per_band": per_band,
                "seed": args.seed, "H": _rep_dim,
                "rep": _rep_desc,
                "inst_id_scheme": "int(round(c*1000))*1_000_000 + per_band_index "
                                  "(band-namespaced, distinct per (band, instance))",
                "dart0": "identity/argmax member (the deterministic baseline run)",
            }
            capture.dump(args.capture_darts, meta)
        finally:
            capture.uninstall()


# ===========================================================================
# EARLY-STOP (b) — the REAL CHUNKED ACTIVE-SET DEPLOYMENT (opt-in: --early-stop).
# Default OFF; the measurement/curve mode above is byte-compatible. SAVES wall-clock
# by stopping at the FIRST valid dart, dropping solved instances, and repacking the
# unsolved active set into fixed-B (pad-to-B) batches — the compute-bound optimum.
# ===========================================================================

def _es_dart_for_batch(mech, dart_idx_0based, active_insts, active_gids, pos_embed_orig,
                       noise_std, dart_seed, s_max, _fwd, _set_pos_embed,
                       _restore_pos_embed):
    """Draw ONE dart (global dart index `dart_idx_0based`, 0-based; index 0 = the
    deterministic argmax/identity run) over the CURRENT active batch (<=B instances,
    repacked + pad-to-B handled by _fwd). Returns pred_1based (n_real, S) on the
    ORIGINAL vertex order of each active instance, reusing the VALIDATED forward +
    samplers.

    PARITY-BY-CONSTRUCTION (the reviewer fix): every dart comes from the SHARED
    _draw_dart generator, keyed by the instance's STABLE global id `active_gids[bi]`
    (its per-band index) — NOT the row's transient position in the repacked active
    batch. So dart j of instance X is the SAME noise/permutation here as in the full-M
    run_volume_sweep, regardless of which round draws it or which other instances co-
    batch with X. (multistart noise is batch-SHARED — keyed on dart_idx only via
    _draw_dart's GID_SHARED sentinel — matching the full-M loop; symmetry permutations
    are per-(gid, dart) deterministic.)"""
    n_real = len(active_insts)
    if mech == "multistart":
        if dart_idx_0based == 0:
            # identity / argmax run (no perturbation).
            _restore_pos_embed()
            p_m, _, _, _, _, _, _ = _fwd(active_insts)
            return p_m[:n_real].copy()
        # batch-shared noise from the SHARED dart generator (keyed by dart_idx only).
        noise = _draw_dart("multistart", GID_SHARED, dart_idx_0based, seed=dart_seed,
                           pos_embed_shape=pos_embed_orig.shape, noise_std=noise_std)
        try:
            _set_pos_embed(pos_embed_orig + noise)   # ASSIGN-IN-PLACE (never rebind)
            p_m, _, _, _, _, _, _ = _fwd(active_insts)
            out = p_m[:n_real].copy()
        finally:
            _restore_pos_embed()
        return out
    # symmetry
    if dart_idx_0based == 0:
        _restore_pos_embed()
        p_m, _, _, _, _, _, _ = _fwd(active_insts)
        return p_m[:n_real].copy()
    perms = []
    permuted = []
    for bi, r in enumerate(active_insts):
        nn = int(r["n"])
        # per-instance permutation from the SHARED dart generator, keyed by the STABLE
        # gid (active_gids[bi]) — NOT the batch position bi. This is the core of the fix:
        # repacking changes bi across rounds, but gid is fixed, so dart j is stable.
        perm = _draw_dart("symmetry", active_gids[bi], dart_idx_0based, seed=dart_seed,
                          n=nn)
        perms.append(perm)
        permuted.append(_permute_instance(r, perm))
    _restore_pos_embed()
    p_m, _, _, _, _, _, _ = _fwd(permuted)
    out = np.zeros((n_real, s_max), dtype=np.int32)
    for bi in range(n_real):
        nn = int(active_insts[bi]["n"])
        perm = perms[bi]                              # perm[new]=old
        for new in range(nn):
            out[bi, perm[new]] = p_m[bi, new]
    return out


def run_early_stop(args) -> None:
    """CHUNKED ACTIVE-SET EARLY-STOP DEPLOYMENT (opt-in). For multistart/symmetry each
    dart is 1 forward, so stopping at the first valid dart SAVES real forwards (=
    wall-clock, since the GPU is compute-bound). Loop per band:

      active set = all instances; round:
        repack the active set into fixed-B (pad-to-B) batches (the B=8 compute-bound
        optimum — repacking keeps batches ~full; we do NOT pack M darts into a giant
        batch, which is measured useless);
        for each active instance draw up to --chunk MORE darts (global dart indices),
        verify each new sample with the EXACT _coloring_proper_np, mark solved at the
        first valid; DROP solved instances; an instance hitting the budget M unsolved
        is dropped; survivors carry to the next round;
      until the active set drains.

    Reports final solve-rate + ACTUAL total forwards used, and ASSERTS PARITY:
      (1) early-stop solve-rate == full-M best-of-M solve-rate on the SAME darts (the
          verifier is EXACT, so stopping at first success cannot change WHETHER an
          instance is solved within M);
      (2) the ACTUAL forwards == the FREE (a) accounting (min(first_success, M) per
          solved instance, M per unsolved) AND == the pure-CPU _active_set_chunk_schedule
          replay — three independent counts must agree within rounding.

    REUSES the bank build / model build / JIT forward / samplers / verifier verbatim;
    engine + oracle untouched; default measurement mode unchanged (this is a separate
    entry, --early-stop)."""
    if args.domain != "coloring":
        _domain_hook(args.domain)

    from tinygrad import Tensor, dtypes
    from mycelium.factor_graph_engine import FactorGraphSpec, factor_breathing_forward
    from mycelium.graph_coloring_data import LTYPE_EDGE

    k = args.k
    s_max = args.s_max
    K = args.K
    bands = args.bands
    per_band = args.per_band
    n_max = min(args.max_n, s_max)
    n_min = args.min_n
    B = args.eval_batch
    M = args.m_max
    chunk = args.chunk
    # early-stop is a deployment of ONE diversity mechanism (each dart = 1 forward).
    # temp does NOT save forwards (all M from one forward), so the chunked mode runs a
    # per-dart-forward mechanism; default to symmetry (the winner) if temp-only passed.
    mechs = [m for m in args.diversity if m in ("multistart", "symmetry")]
    if not mechs:
        print("  [early-stop] --diversity has no per-dart-forward mechanism "
              "(multistart/symmetry); temp draws all M from ONE forward so early-stop "
              "saves no forwards. Defaulting to symmetry.", flush=True)
        mechs = ["symmetry"]
    mech = mechs[0]
    if len(mechs) > 1:
        print(f"  [early-stop] deploying a SINGLE mechanism per run; using '{mech}' "
              f"(pass --diversity to pick).", flush=True)

    print(f"\n=== amortized_frontier_measure [EARLY-STOP DEPLOYMENT] — domain="
          f"{args.domain} mech={mech} ===", flush=True)
    print(f"k={k} s_max={s_max} K={K} bands={[f'{c:.2f}' for c in bands]} "
          f"per_band={per_band} n in [{n_min},{n_max}] B={B} M={M} chunk={chunk}",
          flush=True)

    # ---- 1. Build banks (same builder as the measurement mode). -----------------------
    banks = {}
    for c in bands:
        insts = _build_band(c, per_band, n_min, n_max, k, seed=args.seed + int(c * 100),
                            regular_frac=args.regular_frac)
        if not insts:
            print(f"  [FLAG] band c={c:.2f}: 0 k-colorable instances — SKIP.", flush=True)
            continue
        banks[c] = insts
        print(f"  band c={c:.2f}: {len(insts)}/{per_band} k-colorable instances",
              flush=True)
    bands = [c for c in bands if c in banks]
    if not bands:
        print("[abort] no instances generated in any band.", flush=True)
        sys.exit(1)
    all_insts = [r for c in bands for r in banks[c]]
    n_edges_max = max(r["n_edges"] for r in all_insts)

    # Pre-encode the ORIGINAL-graph verifier arrays (membership/cell_valid/latent_type)
    # for every instance ONCE, on CPU — the EXACT verifier reads these; doing it here
    # avoids a wasted GPU forward just to harvest them (which would break the forward
    # accounting). Reuses the VALIDATED encode_instance.
    from mycelium.graph_coloring_data import encode_instance as _encode_inst
    enc_verify = {}                                # band c -> list of (mem, lt, cv)
    for c in bands:
        rows = []
        for r in banks[c]:
            e = _encode_inst(r, s_max, n_edges_max, k)
            rows.append((e["membership"].astype(np.float32),
                         e["latent_type"].astype(np.int32),
                         e["cell_valid"].astype(np.float32)))
        enc_verify[c] = rows

    # ---- 2. Build the deducer + JIT forward ONCE (validated path; timed separately). --
    spec = FactorGraphSpec(s_max=s_max, n_values=k, n_factor_types=1, n_heads=16,
                           k_max=K, has_factor_inlet=False)
    Tensor.training = False
    t_load0 = time.perf_counter()
    model = _build_deducer_model(spec, args.ckpt, k, s_max, K, args.seed)
    load_ms = (time.perf_counter() - t_load0) * 1000.0
    pos_embed_orig = model.fg_position_embed.realize().numpy().copy()

    use_jit = (args.forward == "jit")
    jit_fn = None
    if use_jit:
        jit_fn = _make_volume_jit_fn(model, spec, B, s_max, n_edges_max, k, K)
        jit_fn.restore_position_embed()

    def _fwd(insts_slice):
        if use_jit:
            return _forward_argmax_jit(jit_fn, insts_slice)
        return _forward_argmax(model, insts_slice, spec, K, s_max, n_edges_max, k, B,
                               Tensor, factor_breathing_forward)

    def _set_pos_embed(arr_np):
        if use_jit:
            jit_fn.set_position_embed(arr_np)
        else:
            model.fg_position_embed.assign(
                Tensor(arr_np.astype(np.float32), dtype=dtypes.float)).realize()

    def _restore_pos_embed():
        if use_jit:
            jit_fn.restore_position_embed()
        else:
            model.fg_position_embed.assign(
                Tensor(pos_embed_orig.astype(np.float32), dtype=dtypes.float)).realize()

    # warmup (JIT compile) — EXCLUDED from per-instance timing.
    warm = all_insts[:B]
    t_warm0 = time.perf_counter()
    _fwd(warm)
    warmup_ms = (time.perf_counter() - t_warm0) * 1000.0
    batch_times = []
    for _ in range(5):
        tb0 = time.perf_counter()
        _fwd(warm)
        batch_times.append((time.perf_counter() - tb0) * 1000.0)
    batch_ms = float(np.median(batch_times))
    print(f"\n  [timing] forward={args.forward} load={load_ms:.0f}ms "
          f"warmup={warmup_ms:.0f}ms; steady batched fwd={batch_ms:.1f}ms/B={B}.",
          flush=True)

    # ---- 3. Per-band chunked active-set deployment. -----------------------------------
    print(f"\n{'='*100}", flush=True)
    print(f"  CHUNKED ACTIVE-SET EARLY-STOP — mech={mech}, chunk={chunk}, budget M={M}, "
          f"B={B} (repack-to-B; compute-bound optimum)", flush=True)
    print(f"{'='*100}", flush=True)
    print(f"      {'band':>6} {'n':>5} {'solve_rate':>10} {'actual_fwds':>12} "
          f"{'exp_fwds':>9} {'fixedM_fwds':>12} {'wall_s':>8} {'fwd_saving':>10}",
          flush=True)

    all_parity_ok = True
    for c in bands:
        insts = banks[c]
        n_inst = len(insts)
        # per-instance state.
        drawn = [0] * n_inst                  # darts drawn so far (== forwards spent)
        solved_at = [None] * n_inst           # first_success_index (1-based) or None
        # active set holds GLOBAL instance indices into `insts`.
        active = list(range(n_inst))
        actual_forwards = 0
        n_rounds = 0
        n_batches = 0
        # IDENTICAL dart seed convention as run_volume_sweep -> dart j of instance X is
        # byte-identical on both paths (parity by construction, not a tautology).
        dart_seed = _band_dart_seed(args.seed, c)
        t_band0 = time.perf_counter()

        while active:
            n_rounds += 1
            # REPACK the active set into fixed-B padded batches (the compute-bound layout).
            for bstart in range(0, len(active), B):
                batch_ids = active[bstart:bstart + B]
                n_batches += 1
                batch_insts = [insts[i] for i in batch_ids]
                # verifier arrays (ORIGINAL graph) for this batch — pre-encoded on CPU
                # above (NO wasted forward). Indexed by global instance id.
                mem_b = [enc_verify[c][i][0] for i in batch_ids]
                lt_b = [enc_verify[c][i][1] for i in batch_ids]
                cv_b = [enc_verify[c][i][2] for i in batch_ids]
                # round-start common dart offset: the active-set invariant guarantees all
                # survivors entering this round have drawn the SAME #darts (rounds advance
                # in lock-step), so the batch's common offset is any active row's count.
                round_off = drawn[batch_ids[0]]
                # draw up to `chunk` MORE darts for each active instance in this batch.
                for step in range(chunk):
                    cur_dart = round_off + step                 # 0-based global dart index
                    if cur_dart >= M:
                        break
                    # rows STILL unsolved BEFORE this dart are the ones that actually spend
                    # a forward (a solved row leaves the active set; in the batched layout
                    # it would be a pad row, billed for layout but NOT per-instance cost).
                    rows_drawing = [bi for bi, gid in enumerate(batch_ids)
                                    if solved_at[gid] is None]
                    if not rows_drawing:
                        break
                    # batch_ids ARE the stable per-band gids (index into banks[c]),
                    # identical to run_volume_sweep's `gids` -> dart parity by construction.
                    preds = _es_dart_for_batch(
                        mech, cur_dart, batch_insts, batch_ids, pos_embed_orig,
                        args.multistart_noise, dart_seed, s_max,
                        _fwd, _set_pos_embed, _restore_pos_embed)
                    # one forward per STILL-ACTIVE row (== per-instance forward count; this
                    # is what the (a) accounting bills, min(first_success, M) per instance).
                    actual_forwards += len(rows_drawing)
                    # verify each still-unsolved instance in this batch at this dart.
                    for bi in rows_drawing:
                        gid = batch_ids[bi]
                        ok = _coloring_proper_np(preds[bi], mem_b[bi], lt_b[bi],
                                                 cv_b[bi], LTYPE_EDGE)
                        drawn[gid] = cur_dart + 1             # this row spent this dart
                        if ok:
                            solved_at[gid] = cur_dart + 1     # 1-based first success
                    # if ALL instances in this batch are now solved, stop early in-chunk.
                    if all(solved_at[g] is not None for g in batch_ids):
                        break
            # rebuild the active set: keep instances not solved and not budget-exhausted.
            active = [i for i in range(n_inst)
                      if solved_at[i] is None and drawn[i] < M]
        wall_s = time.perf_counter() - t_band0

        # ---- PARITY: derive the FREE (a) accounting + the CPU schedule replay. --------
        first_success = [solved_at[i] for i in range(n_inst)]
        # forwards each instance actually spent (== drawn[i]).
        n_solved = sum(1 for x in solved_at if x is not None)
        solve_rate = n_solved / n_inst
        # (a) accounting from the recorded first_success (zero new forwards):
        flags = []
        for i in range(n_inst):
            fa = np.zeros(M, dtype=bool)
            if solved_at[i] is not None and solved_at[i] <= M:
                fa[solved_at[i] - 1] = True
            flags.append(fa)
        acc = _early_stop_accounting(flags, M, per_dart_is_forward=True)
        sched = _active_set_chunk_schedule(first_success, M, chunk, B)
        # full-M best-of-M solve-rate on the SAME flags (the parity reference).
        full_M_curve = _solve_rate_curve(flags, [M])
        full_M_rate = full_M_curve["solve_emp"].get(M, float("nan"))

        # PARITY ASSERTIONS.
        parity_ok = True
        if abs(solve_rate - full_M_rate) > 1e-9:
            parity_ok = False
            print(f"  [PARITY FAIL] band c={c:.2f}: early-stop solve-rate "
                  f"{solve_rate:.6f} != full-M best-of-M {full_M_rate:.6f}", flush=True)
        if actual_forwards != acc["total_forwards"]:
            parity_ok = False
            print(f"  [PARITY FAIL] band c={c:.2f}: actual forwards {actual_forwards} "
                  f"!= (a) accounting total {acc['total_forwards']}", flush=True)
        if actual_forwards != sched["total_forwards"]:
            parity_ok = False
            print(f"  [PARITY FAIL] band c={c:.2f}: actual forwards {actual_forwards} "
                  f"!= CPU schedule replay {sched['total_forwards']}", flush=True)
        all_parity_ok = all_parity_ok and parity_ok

        fixed_M_forwards = n_inst * M
        fwd_saving = (fixed_M_forwards / actual_forwards) if actual_forwards else float("nan")
        print(f"      {c:>6.2f} {n_inst:>5} {solve_rate:>10.3f} {actual_forwards:>12d} "
              f"{acc['exp_forwards']:>9.2f} {fixed_M_forwards:>12d} {wall_s:>8.2f} "
              f"{fwd_saving:>9.2f}x" + ("" if parity_ok else "  [PARITY FAIL]"),
              flush=True)

    print(f"\n  PARITY: {'ALL PASS' if all_parity_ok else 'SOME FAILED'} "
          f"(early-stop solve-rate == full-M best-of-M; actual forwards == (a) accounting "
          f"== CPU schedule replay).", flush=True)
    print(f"  DEDUCER FIXED COST (one-time, EXCLUDED): load={load_ms:.0f}ms, "
          f"JIT+warmup={warmup_ms:.0f}ms; steady fwd={batch_ms:.1f}ms/B={B} (K={K}).",
          flush=True)
    if not all_parity_ok:
        sys.exit(2)


def _print_volume(vol_rows, bands, k, m_grid, m_max, mechanisms, args, fixed_cost) -> None:
    B = fixed_cost["batch"]
    batch_ms = fixed_cost["batch_ms"]
    print(f"\n{'='*100}", flush=True)
    print("  VOLUME / BEST-OF-N — generate-and-verify: fixed-cost parallel forward draws "
          "M; FREE verifier keeps any valid one", flush=True)
    print("  success metric = EXACT proper-rate (edge verifier); proper-rate FLOOR = "
          "EMPIRICAL uniform-random (NOT 1/k)", flush=True)
    print(f"{'='*100}", flush=True)

    # ---- (A) p per band ----
    print("\n  (A) SINGLE-SAMPLE ARGMAX proper-rate p per band (deterministic baseline):",
          flush=True)
    print(f"      {'band(c)':>8} {'n':>5} {'p_argmax':>9} {'rand_floor':>11} {'margin':>9}",
          flush=True)
    for c in bands:
        r = vol_rows[c]
        margin = r["p_argmax"] - r["floor"]
        print(f"      {c:>8.2f} {r['n_inst']:>5} {r['p_argmax']:>9.3f} "
              f"{r['floor']:>11.4f} {margin:>+9.3f}", flush=True)

    # ---- (B) best-of-N solve-rate(M) + ideal + M_eff + collapse, per mechanism ----
    print(f"\n  (B) BEST-OF-N solve-rate(M) vs INDEPENDENT IDEAL 1-(1-p1)^M "
          f"(gap = correlated-failure collapse):", flush=True)
    print(f"      LEGEND: p1_all = mean valid-rate over ALL M slots (feeds the as-is ideal). "
          f"For multistart/symmetry slot 0 is the", flush=True)
    print(f"      DETERMINISTIC argmax, so the all-slots ideal is a LOWER BOUND when argmax "
          f"already solves (empirical can exceed it).", flush=True)
    print(f"      p1_div = mean valid-rate over the PERTURBED slots ONLY (slots 1..M-1) = the "
          f"CLEAN diversity rate; the div-ideal/M_eff", flush=True)
    print(f"      reflect the coherent decorrelated samples, not the argmax dilution. "
          f"p_argmax/bestofM = deterministic baseline vs best-of-M_max.", flush=True)
    for m in mechanisms:
        cost = ("1 forward -> M samples (mean-field, per-cell independent)" if m == "temp"
                else "M forwards (coherent; slot 0 = deterministic argmax)")
        print(f"\n    --- mechanism: {m}  [{cost}] ---", flush=True)
        # solve-rate(M) table per band. Always include m_max so a non-power-of-2 --m-max
        # column is NOT dropped (reviewer fix #6).
        grid_show = [g for g in m_grid if g in (1, 2, 4, 8, 16, 32, 64, 128, 256)]
        if m_max in m_grid and m_max not in grid_show:
            grid_show.append(m_max)
        grid_show = sorted(set(grid_show))
        hdr = (f"      {'band':>6} {'p1_all':>7} {'p1_div':>7} | "
               + " ".join(f"M={g:<4}" for g in grid_show))
        print(hdr, flush=True)
        for c in bands:
            cv = vol_rows[c]["mech_curves"][m]
            row = f"      {c:>6.2f} {cv['p1_all']:>7.3f} {cv['p1_div']:>7.3f} | "
            row += " ".join(f"{cv['solve_emp'][g]:<6.3f}" for g in grid_show)
            print(row + "  (emp)", flush=True)
            row2 = f"      {'':>6} {'':>7} {'':>7} | "
            row2 += " ".join(f"{cv['solve_ideal'][g]:<6.3f}" for g in grid_show)
            print(row2 + "  (ideal_all)", flush=True)
            if m in ("multistart", "symmetry"):
                row3 = f"      {'':>6} {'':>7} {'':>7} | "
                row3 += " ".join(f"{cv['solve_ideal_div'][g]:<6.3f}" for g in grid_show)
                print(row3 + "  (ideal_div)", flush=True)
        # M_eff + collapse summary per band. p_argmax + bestofM adjacent so the genuine
        # lift over the deterministic baseline is one glance away (reviewer fix #1).
        print(f"      {'band':>6} | {'p_argmax':>8} {'bestofM':>8} | {'M_eff(emp@.9)':>13} "
              f"{'M_eff(idl@.9)':>13} {'ratio':>7} {'div_idl@.9':>10} {'div_ratio':>9} | "
              f"{'distinct':>8} {'entropy':>8}", flush=True)
        for c in bands:
            cv = vol_rows[c]["mech_curves"][m]
            col = vol_rows[c]["mech_collapse"][m]
            dmed = float(np.median(col["distinct"])) if col["distinct"] else float("nan")
            emed = float(np.median(col["entropy"])) if col["entropy"] else float("nan")
            p_argmax = vol_rows[c]["p_argmax"]
            bestofM = cv["solve_emp"].get(m_max, float("nan"))
            me = "none" if cv["m_eff_emp"] is None else str(cv["m_eff_emp"])
            mi = "none" if cv["m_eff_ideal"] is None else str(cv["m_eff_ideal"])
            mdv = "none" if cv["m_eff_div_ideal"] is None else str(cv["m_eff_div_ideal"])
            rt = cv["m_eff_ratio"]
            rts = ("inf" if rt == float("inf") else (f"{rt:.2f}" if rt == rt else "nan"))
            drt = cv["m_eff_div_ratio"]
            drts = ("inf" if drt == float("inf") else (f"{drt:.2f}" if drt == drt else "nan"))
            collapse_flag = ""
            if dmed <= 1.0 + 1e-9:
                collapse_flag = "  [COLLAPSE: all M identical]"
            print(f"      {c:>6.2f} | {p_argmax:>8.3f} {bestofM:>8.3f} | {me:>13} "
                  f"{mi:>13} {rts:>7} {mdv:>10} {drts:>9} | "
                  f"{dmed:>8.1f} {emed:>8.3f}{collapse_flag}", flush=True)

    # ---- (B-views) VIEW-SUCCESS-COUNT distribution (diversity FRAGILITY) ----
    _print_view_success_counts(vol_rows, bands, m_max, mechanisms)

    # ---- (C) throughput (deducer M-forward-honest vs symbolic per-core) ----
    print(f"\n  (C) THROUGHPUT — instances-SOLVED/sec (substrate stated; NEVER a single "
          f"cross-substrate number):", flush=True)
    print(f"      deducer SUBSTRATE: 1 GPU (AMD), batched B={B}, steady fwd={batch_ms:.1f}ms; "
          f"M-forward cost accounted per mechanism.", flush=True)
    print(f"      symbolic SUBSTRATE: CPU per-instance; per-CORE rate below; symbolic is "
          f"embarrassingly parallel -> multiply by #cores.", flush=True)
    for m in mechanisms:
        n_fwd = vol_rows[bands[0]]["mech_forwards"].get(m, 1)
        print(f"    --- mechanism: {m} (n_forwards/batch = {n_fwd}) ---", flush=True)
        print(f"      {'band':>6} {'bestofM_rate':>12} {'ded_solved/s':>13} "
              f"{'ded_inst/s':>11}", flush=True)
        for c in bands:
            cv = vol_rows[c]["mech_curves"][m]
            rate = cv["solve_emp"].get(m_max, 0.0)
            tp = _throughput_accounting(B, batch_ms, n_fwd, rate,
                                        sym_solve_ms_med=float("nan"), sym_solved_rate=0.0)
            print(f"      {c:>6.2f} {rate:>12.3f} {tp['ded_solved_per_s']:>13.1f} "
                  f"{tp['ded_inst_per_s']:>11.1f}", flush=True)
    print(f"\n      NOTE: deducer instances-SOLVED/sec = (B * bestofM_rate) / "
          f"(n_forwards * batch_ms/1000). For the SYMBOLIC per-core comparison run "
          f"--mode speed (it times the validated solver wall-clock); on n<=49 colorable "
          f"banks symbolic is <=~15ms/solve/core with 0 timeouts.", flush=True)

    # ---- (C-early) FREE early-stop accounting (ZERO new forwards; deliverable (a)) ----
    # With a FREE EXACT verifier you stop at the FIRST valid dart, so the DEPLOYMENT cost
    # is exp_forwards (mean forwards-to-solve), NOT a fixed M. Derived from the SAME full-M
    # flags already computed (no GPU work). temp = 1 forward regardless (all M samples come
    # from ONE forward); multistart/symmetry = min(first_success_index, M), unsolved = M.
    M_es = args.m_max
    print(f"\n  (C-early) FREE EARLY-STOP ACCOUNTING — exp_forwards-to-solve + DEPLOYMENT "
          f"throughput (ZERO new forwards; from the full-M flags):", flush=True)
    print(f"      stop at the FIRST valid dart (the verifier is EXACT). temp=1 forward "
          f"always (M samples from ONE forward — saves only numpy verify, NOT a forward);", flush=True)
    print(f"      multistart/symmetry forwards_used = min(first_success_index, M={M_es}), "
          f"unsolved burns M. solve-rate == best-of-M (early-stop does not change WHETHER", flush=True)
    print(f"      an instance is solved within M). ded_solved/s_ES = (B*solve_rate) / "
          f"(exp_forwards * batch_ms/1000) — the honest deployment number.", flush=True)
    for m in mechanisms:
        per_dart_fwd = (m != "temp")           # temp: all M from ONE forward
        print(f"    --- mechanism: {m} ({'each dart = 1 forward' if per_dart_fwd else '1 forward -> M samples (no per-dart forward)'}) ---",
              flush=True)
        print(f"      {'band':>6} {'solve_rate':>10} {'exp_forwards':>12} "
              f"{'fixed_M_fwds':>12} {'ded_solved/s_ES':>15} {'ded_solved/s_fixedM':>19}",
              flush=True)
        for c in bands:
            flags = vol_rows[c].get("mech_flags", {}).get(m, [])
            acc = _early_stop_accounting(flags, M_es, per_dart_is_forward=per_dart_fwd)
            es_tp = _early_stop_throughput(B, batch_ms, acc["exp_forwards"],
                                           acc["solve_rate"])
            # fixed-M comparison: charge the full n_forwards/batch the (C) table used.
            n_fwd_fixed = vol_rows[c]["mech_forwards"].get(m, 1)
            fixed_tp = _throughput_accounting(B, batch_ms, n_fwd_fixed, acc["solve_rate"],
                                              sym_solve_ms_med=float("nan"),
                                              sym_solved_rate=0.0)
            print(f"      {c:>6.2f} {acc['solve_rate']:>10.3f} "
                  f"{acc['exp_forwards']:>12.2f} {n_fwd_fixed:>12d} "
                  f"{es_tp['ded_solved_per_s_es']:>15.1f} "
                  f"{fixed_tp['ded_solved_per_s']:>19.1f}", flush=True)
    print(f"\n      DEPLOYMENT NOTE: exp_forwards << fixed M on EASY bands (most instances "
          f"solve in the first few darts) -> ded_solved/s_ES >> the fixed-M number; on HARD "
          f"bands (low solve-rate) exp_forwards -> M and the gain shrinks. For temp the "
          f"forward cost stays 1 (early-stop saves only the cheap numpy verify, never a "
          f"forward). Run --early-stop to REALIZE this saving as wall-clock.", flush=True)
    if getattr(args, "forward", "jit") == "jit":
        print(f"      SUBSTRATE NOTE: the deducer forward here is the @TinyJit PRODUCTION "
              f"path (compile-once / replay over assign-in-place fixed buffers; the same "
              f"forward the validated eval JITs), so reported instances/sec reflects the "
              f"deployed steady-state throughput (NOT a lower bound). 1 GPU vs CPU/core — "
              f"reported side by side, never cross-claimed.", flush=True)
    else:
        print(f"      SUBSTRATE NOTE: the deducer forward here is timed EAGER (--forward "
              f"eager; un-JIT'd factor_breathing_forward re-traces each call, the PARITY "
              f"REFERENCE), so reported instances/sec is a CONSERVATIVE LOWER BOUND vs the "
              f"@TinyJit production path (--forward jit, the default).", flush=True)

    # ---- fixed cost + caveats ----
    print(f"\n  DEDUCER FIXED COST (one-time, EXCLUDED from per-instance):", flush=True)
    print(f"    model+ckpt load = {fixed_cost['load_ms']:.0f}ms, JIT compile+warmup = "
          f"{fixed_cost['warmup_ms']:.0f}ms; steady batched fwd = {batch_ms:.1f}ms/B={B} "
          f"({batch_ms/B:.2f}ms/inst, K={fixed_cost['K']}).", flush=True)
    print("", flush=True)


def _print_view_success_counts(vol_rows, bands, m_max, mechanisms) -> None:
    """ADDITIVE (B-views) sub-table: per band per mechanism, the DISTRIBUTION of
    instances over success-count buckets {0, 1, 2-4, 5-16, 17-M}. Derived from the
    SAME mech_flags the (B) table consumes (ZERO new forwards). Measures diversity
    FRAGILITY: a heavy '1(fragile)' column means equivariance-collapse would lose
    those instances; a heavy '17-M' column means diversity is robust there. The '0
    (hardcore)' column is the shared core augmentation should SHRINK. The (A)/(B)/(C)
    tables are UNCHANGED — this is purely additive."""
    print(f"\n  (B-views) VIEW-SUCCESS-COUNT distribution — instances bucketed by HOW "
          f"MANY of M={m_max} views VERIFIED (diversity fragility):", flush=True)
    print(f"      buckets: 0(hardcore)=no view wins | 1(fragile)=EXACTLY one view wins "
          f"(equivariance-collapse risk) | 2-4 | 5-16 | 17-M(robust).", flush=True)
    print(f"      success-count INCLUDES view 0 (deterministic argmax) -> the >=1 set "
          f"== best-of-M and CONTAINS every argmax-solved instance.", flush=True)
    labels = [lab for (_, _, lab) in _VSC_BUCKET_SPEC]
    for m in mechanisms:
        print(f"\n    --- mechanism: {m} ---", flush=True)
        hdr = (f"      {'band':>6} {'n':>5} | "
               + " ".join(f"{lab:>12}" for lab in labels)
               + f" | {'>=1(=bo)':>9} {'mean':>6} {'med':>5}")
        print(hdr, flush=True)
        for c in bands:
            flags = vol_rows[c].get("mech_flags", {}).get(m, [])
            vsc = _view_success_count_distribution(flags, m_max)
            bucket_map = dict(vsc["buckets"])
            row = (f"      {c:>6.2f} {vsc['n_inst']:>5} | "
                   + " ".join(f"{bucket_map.get(lab, 0):>12d}" for lab in labels)
                   + f" | {vsc['success_ge1']:>9.3f} {vsc['mean_success']:>6.2f} "
                   f"{vsc['median_success']:>5.1f}")
            # consistency flag: >=1 fraction must equal the (B) best-of-M solve-rate.
            bo = vol_rows[c]["mech_curves"][m]["solve_emp"].get(m_max, float("nan"))
            mismatch = (bo == bo and abs(vsc["success_ge1"] - bo) > 1e-9)
            if mismatch:
                row += "  [!= bestofM]"
            print(row, flush=True)
    print(f"\n      READ: shift of mass 0(hardcore)->higher buckets pre vs post "
          f"FG_PERM_AUG = augmentation shrank the hard core (the GOAL). Shift of mass "
          f"from many-views to 1(fragile)/0 = equivariance collapsed the load-bearing "
          f"diversity (the TENSION) -> lower FG_PERM_AUG_FRAC.", flush=True)


def _build_deducer_model(spec, ckpt: str, k: int, s_max: int, K: int, seed: int):
    """Build the Pythia-410M breathing deducer + load the coloring ckpt, REUSING the
    validated build/load helpers in scripts.search_coloring (cast_layers_fp32 +
    attach_factor_graph_params + the FG_HYP_MASK anchor path + load_ckpt). This is the
    same path eval_coloring_bands / the gate script use — no hand-rolled model build."""
    import gc
    import scripts.search_coloring as sc
    from tinygrad import Device, dtypes, Tensor
    from tinygrad.helpers import getenv
    from tinygrad.nn.state import safe_load
    from mycelium import Config
    from mycelium.loader import _load_state, load_breathing
    from mycelium.factor_graph_engine import (
        attach_factor_graph_params, attach_factor_waist_params, FG_HYP_MASK,
        FG_WAIST_DIM, FG_WAIST_AFTER, FG_WAIST_GATE_INIT,
    )
    from mycelium.factor_masks import attach_factor_hyperbolic_params
    from mycelium.graph_coloring_data import GraphColoringLoader

    print(f"loading Pythia-410M -> BreathingTransformer (ckpt={ckpt})...", flush=True)
    cfg = Config()
    sd = _load_state()
    model = load_breathing(cfg, sd=sd)
    del sd
    gc.collect()
    sc.cast_layers_fp32(model)
    attach_factor_graph_params(model, hidden=cfg.hidden, spec=spec)

    # IN-DEDUCER WAIST (FG_WAIST=1): attach the waist params BEFORE load_ckpt so the waist
    # weights in the ckpt are restored. The base (no-waist) ckpt path is untouched
    # (FG_WAIST=0 default -> no attach -> the engine's getattr-gate runs the original
    # forward, byte-identical). sc.load_ckpt only restores the base fg keys; the waist keys
    # are loaded separately here (avoids editing the validated search_coloring helpers).
    fg_waist_on = int(getenv("FG_WAIST", "0")) > 0
    if fg_waist_on:
        d_w = int(getenv("FG_WAIST_DIM", str(FG_WAIST_DIM)))
        after_w = int(getenv("FG_WAIST_AFTER", str(FG_WAIST_AFTER)))
        gate_w = float(getenv("FG_WAIST_GATE_INIT", str(FG_WAIST_GATE_INIT)))
        aux_w = getenv("FG_WAIST_AUX", "classify").strip().lower()
        attach_factor_waist_params(model, hidden=cfg.hidden, d=d_w, after=after_w,
                                   gate_init=gate_w, aux=aux_w)
        print(f"  [FG_WAIST=1] waist attached for re-eval: d={d_w} after={after_w} "
              f"aux={aux_w}", flush=True)

    if FG_HYP_MASK:
        print("[FG_HYP_MASK=1] building coloring anchor tables...", flush=True)
        _rl = GraphColoringLoader(n_instances=2000, s_max=s_max, k_colors=k,
                                  batch_size=8, seed=seed)
        _rb = _rl.sample_batch()
        attach_factor_hyperbolic_params(
            model, n_heads=spec.n_heads, n_factor_types=spec.n_factor_types,
            s_max=spec.s_max,
            membership_np=_rb.membership.realize().numpy(),
            latent_type_np=_rb.latent_type.realize().numpy())
        del _rl, _rb
    Device[Device.DEFAULT].synchronize()
    sc.load_ckpt(model, ckpt)

    # Restore the waist keys (sc.load_ckpt only knows the base fg keys). Missing keys keep
    # init -> a base ckpt loaded with FG_WAIST=1 runs with a gate~0 (identity) waist.
    if fg_waist_on:
        sd_w = safe_load(ckpt)
        _waist_keys = ["fg_waist_down", "fg_waist_down_b", "fg_waist_up", "fg_waist_up_b",
                       "fg_waist_gate", "fg_waist_aux_w", "fg_waist_aux_b"]
        loaded = 0
        for nm in _waist_keys:
            dst = getattr(model, nm, None)
            if dst is None or nm not in sd_w:
                continue
            src = sd_w[nm].to(dst.device).realize()
            if src.shape != dst.shape:
                try:
                    src = src.reshape(dst.shape)
                except Exception:
                    continue
            if src.dtype != dst.dtype:
                src = src.cast(dst.dtype)
            dst.assign(src).realize()
            loaded += 1
        g_now = float(model.fg_waist_gate.sigmoid().mean().numpy())
        print(f"  [FG_WAIST=1] loaded {loaded} waist keys from ckpt; "
              f"gate=sigmoid={g_now:.4f}", flush=True)
    return model


def _domain_hook(domain: str) -> None:
    """--domain hook for circuit / kenken. coloring is the lead probe (built); the others
    raise a clear NotImplementedError naming the missing bridge."""
    if domain == "kenken":
        raise NotImplementedError(
            "kenken hook: the SYMBOLIC side is ready (mycelium.csp_domains.problem_from_kenken "
            "+ solve_symbolic), but this harness needs (a) a kenken instance-bank generator "
            "with a hardness knob (givens-count bands) and (b) a kenken deducer ckpt — no "
            "fg_kenken ckpt exists yet (FG_TASK=kenken on the existing trainer would produce "
            "one). Wire those two, then route the symbolic side to problem_from_kenken and the "
            "deducer side to the kenken adapter (factor_graph_train), reusing this overlay/verdict.")
    if domain == "circuit":
        raise NotImplementedError(
            "circuit hook: mycelium.csp_domains.problem_from_circuit is a Phase-4 STUB "
            "(the ordered-scope gate bridge + circuit_data._eval_gate grounding are unbuilt), "
            "so there is no symbolic side to time. A fg_circuit ckpt exists for the deducer "
            "side; once problem_from_circuit lands, route the symbolic side to it and reuse "
            "this overlay/verdict. Hardness knob = circuit depth.")
    raise ValueError(f"unknown domain {domain!r} (expected coloring|circuit|kenken)")


# ===========================================================================
# CPU-ONLY SMOKE (NO GPU forward) — symbolic + calibration + deducer-wiring dry-run
# ===========================================================================

def _cpu_smoke() -> bool:
    print("=== amortized_frontier_measure CPU SMOKE (NO GPU forward) ===", flush=True)
    ok = True

    def _check(name, cond):
        nonlocal ok
        if not cond:
            ok = False
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    k = 3

    # --- (1) SYMBOLIC side on a couple of small graphs (REAL validated solve) ---------
    # Triangle (3-clique): 3-colorable, needs all 3 colors.
    tri = {"n": 3, "edges": [(0, 1), (1, 2), (0, 2)]}
    r_tri = _solve_symbolic_timed(tri, k, budget=100000, timeout_ms=2000)
    _check("symbolic solves the triangle (3-colorable)", r_tri["status"] == "solved")
    _check("triangle solve recorded a wall-clock (>0ms)", r_tri["wall_ms"] >= 0.0)
    _check("triangle solve not timed out", r_tri["timed_out"] is False)
    # K4 (4-clique) with k=3 colors: NOT 3-colorable -> the validated solver certifies unsat.
    k4 = {"n": 4, "edges": [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]}
    r_k4 = _solve_symbolic_timed(k4, k, budget=100000, timeout_ms=2000)
    _check("symbolic certifies K4 UNSAT at k=3 (validated core)", r_k4["status"] == "unsat")

    # --- (2) PHASE-TRANSITION CALIBRATION on the validated generators -----------------
    # Confirm 3-colorability keep-rate falls through c ~ 2.35 (justifies the bands).
    rng_seed = 12345
    keep = {}
    for c in (1.5, 2.3, 3.0):
        insts = _build_band(c, 12, 30, 40, k, seed=rng_seed + int(c * 100),
                            regular_frac=0.0, max_attempts_factor=120)
        # keep-RATE proxy: how many attempts it took to gather 12 colorable instances is
        # hard to expose here; instead sample raw and measure colorability directly.
        rng = random.Random(rng_seed + int(c * 100) + 7)
        n_col = n_tot = 0
        for _ in range(60):
            n = rng.randint(30, 40)
            p = min(0.98, 2.0 * c / (n - 1))
            edges = _gen_gnp(n, p, rng)
            if not edges:
                continue
            adj = [set() for _ in range(n)]
            for (u, v) in edges:
                adj[u].add(v)
                adj[v].add(u)
            col, _bt = _solve_k_coloring(n, adj, k)
            n_tot += 1
            n_col += int(col is not None)
        keep[c] = (n_col / n_tot) if n_tot else float("nan")
        print(f"  [calib] c={c:.1f}: 3-colorable keep-rate ~ {keep[c]:.2f} "
              f"(bank built {len(insts)}/12)", flush=True)
    _check("calibration: keep-rate is high below threshold (c=1.5 > 0.8)", keep[1.5] > 0.8)
    _check("calibration: keep-rate collapses above threshold (c=3.0 < 0.2)", keep[3.0] < 0.2)
    _check("calibration: monotone decrease through the transition",
           keep[1.5] >= keep[2.3] >= keep[3.0])

    # --- (3) DEDUCER-EVAL WIRING DRY-RUN (CPU; NO GPU forward) -------------------------
    # Build a small bank, encode to batch arrays, assert the engine contract + that the
    # accuracy scorer + proper verifier score the GOLD coloring as perfect (the contract
    # the GPU forward will be measured against).
    s_max = 12
    band = _build_band(2.0, 6, 6, s_max, k, seed=99, regular_frac=0.0,
                       max_attempts_factor=200)
    _check("dry-run: built a non-empty bank", len(band) > 0)
    if band:
        n_edges_max = max(r["n_edges"] for r in band)
        arrays = _build_batch_arrays(band, s_max, n_edges_max, k)
        try:
            _assert_deducer_contract(arrays, s_max, k)
            contract_ok = True
        except AssertionError as e:  # pragma: no cover
            print(f"  contract assertion failed: {e}", flush=True)
            contract_ok = False
        _check("dry-run: batch arrays satisfy the engine FactorGraphBatch contract",
               contract_ok)
        # a deliberately-WRONG prediction (all color 1) must NOT verify proper on a graph
        # with any edge, and must score below perfect (soundness of the scorer).
        from mycelium.graph_coloring_data import LTYPE_EDGE
        wrong = np.ones_like(arrays["gold"])  # all 1-colored
        sc_wrong = _score_predictions(wrong, arrays["gold"], arrays["cell_valid"],
                                      arrays["membership"], arrays["latent_type"], LTYPE_EDGE)
        _check("dry-run: all-same-color prediction never scores proper",
               sc_wrong["proper_ok"] == 0)

    # --- (4) AGGREGATION + OVERLAY + VERDICT logic (synthetic; no GPU) -----------------
    # SWEET-SPOT case: a slow+accurate band exists.
    sweet_rows = {
        2.0: _summarise_band(
            [{"decisions": 5, "backtracks": 0, "wall_ms": 1.0, "status": "solved",
              "timed_out": False, "budget_hit": False}],
            {"cell_eq": 9, "n_cells": 10, "puz_ok": 1, "n_puz": 1, "proper_ok": 1}, k),
        2.5: _summarise_band(
            [{"decisions": 999999, "backtracks": 0, "wall_ms": 5000.0, "status": "timeout",
              "timed_out": True, "budget_hit": False}],
            {"cell_eq": 9, "n_cells": 10, "puz_ok": 1, "n_puz": 1, "proper_ok": 1}, k),
    }
    v_sweet = _verdict(sweet_rows, [2.0, 2.5], k, slow_ms=1000.0,
                       slow_timeout_rate=0.5, accurate_margin=0.2)
    _check("verdict: SWEET-SPOT fires when a band is slow AND accurate",
           v_sweet["verdict"] == "SWEET-SPOT" and 2.5 in v_sweet["sweet_bands"])

    # DEAD (no blow-up): every band fast.
    fast_rows = {
        2.0: _summarise_band(
            [{"decisions": 5, "backtracks": 0, "wall_ms": 1.0, "status": "solved",
              "timed_out": False, "budget_hit": False}],
            {"cell_eq": 9, "n_cells": 10, "puz_ok": 1, "n_puz": 1, "proper_ok": 1}, k),
        2.5: _summarise_band(
            [{"decisions": 30, "backtracks": 2, "wall_ms": 8.0, "status": "solved",
              "timed_out": False, "budget_hit": False}],
            {"cell_eq": 9, "n_cells": 10, "puz_ok": 1, "n_puz": 1, "proper_ok": 1}, k),
    }
    v_fast = _verdict(fast_rows, [2.0, 2.5], k, slow_ms=1000.0,
                      slow_timeout_rate=0.5, accurate_margin=0.2)
    _check("verdict: DEAD (no blow-up) fires when symbolic never tails",
           v_fast["verdict"].startswith("DEAD (no blow-up"))

    # DEAD (deducer collapses): a slow band but deducer at chance there.
    collapse_rows = {
        2.5: _summarise_band(
            [{"decisions": 999999, "backtracks": 0, "wall_ms": 5000.0, "status": "timeout",
              "timed_out": True, "budget_hit": False}],
            {"cell_eq": 3, "n_cells": 10, "puz_ok": 0, "n_puz": 1, "proper_ok": 0}, k),
    }
    v_col = _verdict(collapse_rows, [2.5], k, slow_ms=1000.0,
                     slow_timeout_rate=0.5, accurate_margin=0.2)
    _check("verdict: DEAD (collapse) fires when slow band has deducer at chance",
           v_col["verdict"].startswith("DEAD (deducer collapses"))
    _check("verdict: deducer-at-chance-everywhere flag set in the collapse case",
           v_col["deducer_at_chance_everywhere"] is True)

    # the overlay printer runs without error.
    try:
        _print_overlay(sweet_rows, [2.0, 2.5], k,
                       {"load_ms": 1000.0, "warmup_ms": 500.0, "batch_ms": 80.0,
                        "batch": 8, "per_inst_ms": 10.0, "K": 16}, v_sweet)
        printer_ok = True
    except Exception as e:  # pragma: no cover
        print(f"  overlay printer raised: {e}", flush=True)
        printer_ok = False
    _check("overlay printer runs cleanly", printer_ok)

    # --- (5) DOMAIN HOOKS raise a clear NotImplementedError ---------------------------
    for dom in ("circuit", "kenken"):
        raised = False
        try:
            _domain_hook(dom)
        except NotImplementedError:
            raised = True
        _check(f"domain hook '{dom}' raises NotImplementedError (clear stub)", raised)

    # =====================================================================
    # VOLUME-MODE LOGIC (synthetic; NO GPU forward)
    # =====================================================================
    print("\n  --- VOLUME-MODE logic smoke (synthetic) ---", flush=True)

    # --- (6) EMPIRICAL PROPER-RATE FLOOR (reviewer fix #1) ---------------------------
    # On a dense band, uniform-random proper-rate must be FAR below 1/k (the cell floor).
    s_max_v = 16
    band_v = _build_band(2.0, 8, 8, s_max_v, k, seed=321, regular_frac=0.0,
                         max_attempts_factor=300)
    _check("floor: built a non-empty dense band", len(band_v) > 0)
    if band_v:
        nem = max(r["n_edges"] for r in band_v)
        floor = _empirical_proper_floor(band_v, s_max_v, nem, k, seed=5, n_draws=4)
        print(f"  [smoke] empirical uniform-random proper-rate = {floor:.4f} "
              f"(1/k={1.0/k:.3f})", flush=True)
        _check("floor: empirical proper-floor is a probability in [0,1]",
               0.0 <= floor <= 1.0)
        _check("floor: empirical proper-floor < 1/k on a dense band "
               "(the 1/k anchor was wrong)", floor < 1.0 / k)
        # the GOLD coloring scores proper -> floor on a trivial 1-instance edgeless band is N/A;
        # here we just confirm the floor is well below the cell anchor.

    # --- (7) TEMP SAMPLER soundness + collapse detection ------------------------------
    # tau->0 collapses to argmax (1 distinct sample); tau large diversifies.
    S_t, N_t = 5, 3
    rng_t = np.random.RandomState(11)
    logits_t = rng_t.randn(S_t, N_t)
    valid_t = np.ones((S_t,), dtype=bool)
    vdm_t = np.ones((S_t, N_t), dtype=np.float32)
    samp_cold = _temp_sample_from_logits(logits_t, valid_t, vdm_t, tau=1e-3, M=16,
                                         rng=np.random.RandomState(1))
    dct_cold, ent_cold = _distinct_and_entropy(samp_cold, valid_t)
    _check("temp: tau->0 collapses to argmax (1 distinct sample, entropy 0)",
           dct_cold == 1 and ent_cold < 1e-9)
    # argmax of logits matches the cold sample.
    argmax_pred = logits_t.argmax(axis=-1) + 1
    _check("temp: cold sample == per-cell argmax",
           np.array_equal(samp_cold[0], argmax_pred.astype(np.int32)))
    samp_hot = _temp_sample_from_logits(logits_t, valid_t, vdm_t, tau=5.0, M=64,
                                        rng=np.random.RandomState(2))
    dct_hot, ent_hot = _distinct_and_entropy(samp_hot, valid_t)
    _check("temp: hot tau diversifies (>1 distinct sample)", dct_hot > 1)
    _check("temp: reproducible with same seed",
           np.array_equal(
               _temp_sample_from_logits(logits_t, valid_t, vdm_t, 5.0, 8,
                                        np.random.RandomState(7)),
               _temp_sample_from_logits(logits_t, valid_t, vdm_t, 5.0, 8,
                                        np.random.RandomState(7))))
    # value-domain mask: an illegal color must NEVER be sampled.
    vdm_masked = np.ones((S_t, N_t), dtype=np.float32)
    vdm_masked[:, 2] = 0.0           # color 3 illegal everywhere
    samp_masked = _temp_sample_from_logits(logits_t, valid_t, vdm_masked, tau=5.0, M=64,
                                           rng=np.random.RandomState(3))
    _check("temp: illegal (masked) color is never sampled",
           int((samp_masked == 3).sum()) == 0)

    # --- (8) BEST-OF-N aggregation + INDEPENDENT IDEAL + M_eff ------------------------
    m_grid_s = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    # INDEPENDENT case: per-instance flags drawn iid with known p1 -> empirical ~ ideal.
    rng_a = np.random.RandomState(0)
    p1_true = 0.2
    indep_flags = [(rng_a.random_sample(256) < p1_true) for _ in range(400)]
    curve_indep = _solve_rate_curve(indep_flags, m_grid_s)
    _check("bestN: measured p1 ~ true single-sample rate (independent draws)",
           abs(curve_indep["p1"] - p1_true) < 0.03)
    _check("bestN: empirical solve-rate ~ ideal for INDEPENDENT samples (small gap)",
           abs(curve_indep["solve_emp"][64] - curve_indep["solve_ideal"][64]) < 0.05)
    _check("bestN: solve-rate(M) is monotone non-decreasing in M",
           all(curve_indep["solve_emp"][m_grid_s[i]] <=
               curve_indep["solve_emp"][m_grid_s[i + 1]] + 1e-9
               for i in range(len(m_grid_s) - 1)))
    _check("bestN: M_eff ratio ~ 1 when failures are independent",
           (curve_indep["m_eff_ratio"] != curve_indep["m_eff_ratio"])  # nan ok if never crosses
           or 0.4 <= curve_indep["m_eff_ratio"] <= 2.5)
    # COLLAPSED case: every instance's M samples are IDENTICAL (all-fail or all-pass).
    # 80% of instances all-fail, 20% all-pass -> p1=0.2 but best-of-M NEVER improves on
    # the all-fail instances -> empirical solve-rate FLAT at 0.2 << ideal -> M_eff collapse.
    collapsed_flags = ([np.zeros(256, dtype=bool) for _ in range(320)]
                       + [np.ones(256, dtype=bool) for _ in range(80)])
    curve_col = _solve_rate_curve(collapsed_flags, m_grid_s)
    _check("bestN: collapse -> empirical solve-rate FLAT (best-of-M does not climb)",
           abs(curve_col["solve_emp"][256] - curve_col["solve_emp"][1]) < 1e-9)
    _check("bestN: collapse -> empirical << ideal at large M (correlated-failure gap)",
           curve_col["solve_ideal"][256] - curve_col["solve_emp"][256] > 0.3)
    _check("bestN: measured p1 ~ 0.2 even under collapse",
           abs(curve_col["p1"] - 0.2) < 0.01)
    _check("bestN: empty flags -> safe (n_inst=0, no crash)",
           _solve_rate_curve([], m_grid_s)["n_inst"] == 0)

    # --- (8b) p1_div: coherent argmax-dilution decomposition (reviewer fix #1) ---------
    # Build flags where slot 0 (the DETERMINISTIC argmax) ALWAYS solves but the perturbed
    # slots 1..M-1 solve only at rate ~0.2. p1_all is the (1 + (M-1)*0.2)/M mixture; p1_div
    # (perturbed slots only) must recover the CLEAN diversity rate ~0.2. Use a SMALL M so the
    # argmax slot's 1/M weight makes the inflation clearly visible: p1_all ~ 0.36 vs 0.20.
    rng_b = np.random.RandomState(1234)
    Msz = 5
    dilute_flags = []
    for _ in range(2000):
        f = (rng_b.random_sample(Msz) < 0.2)
        f[0] = True                       # slot 0 = argmax always solves
        dilute_flags.append(f)
    cv_all = _solve_rate_curve(dilute_flags, m_grid_s, coherent=False)
    cv_div = _solve_rate_curve(dilute_flags, m_grid_s, coherent=True)
    _check("p1_div: p1_all keys present + p1==p1_all (back-compat)",
           "p1_all" in cv_all and "p1_div" in cv_all
           and abs(cv_all["p1"] - cv_all["p1_all"]) < 1e-12)
    _check("p1_div: NON-coherent -> p1_div == p1_all (temp exchangeable, no change)",
           abs(cv_all["p1_div"] - cv_all["p1_all"]) < 1e-12)
    _check("p1_div: coherent p1_all is INFLATED by the argmax slot (> p1_div)",
           cv_div["p1_all"] > cv_div["p1_div"] + 0.1)
    _check("p1_div: coherent p1_div recovers the clean perturbed-slot rate (~0.2)",
           abs(cv_div["p1_div"] - 0.2) < 0.05)
    _check("p1_div: coherent leaves the all-slots ideal/M_eff UNCHANGED vs non-coherent",
           cv_div["m_eff_ideal"] == cv_all["m_eff_ideal"]
           and cv_div["m_eff_ratio"] == cv_all["m_eff_ratio"]
           if cv_div["m_eff_ratio"] == cv_div["m_eff_ratio"]  # nan-safe compare
           else cv_div["m_eff_ideal"] == cv_all["m_eff_ideal"])
    _check("p1_div: div-ideal needs MORE samples than all-ideal (lower clean rate)",
           (cv_div["m_eff_div_ideal"] is None and cv_div["m_eff_ideal"] is None)
           or (cv_div["m_eff_div_ideal"] is not None and cv_div["m_eff_ideal"] is not None
               and cv_div["m_eff_div_ideal"] >= cv_div["m_eff_ideal"]))

    # --- (8c) total-collapse ratio -> inf (reviewer fix #5) ---------------------------
    # p1>0 (ideal crosses 0.9) but the empirical curve NEVER crosses (every instance flat).
    # m_eff_emp=None + m_eff_ideal set -> ratio must be inf, NOT nan, so the most severe
    # collapse reads as the LARGEST ratio.
    total_collapse = [np.zeros(256, dtype=bool) for _ in range(380)] \
        + [np.ones(256, dtype=bool) for _ in range(20)]   # p1=0.05 -> ideal crosses .9
    cv_tc = _solve_rate_curve(total_collapse, m_grid_s)
    _check("ratio-inf: total collapse (emp never crosses .9) -> ratio == inf (not nan)",
           cv_tc["m_eff_emp"] is None and cv_tc["m_eff_ideal"] is not None
           and cv_tc["m_eff_ratio"] == float("inf"))

    # --- (9) DISTINCT/ENTROPY collapse detector ---------------------------------------
    ident = np.tile(np.array([1, 2, 3, 1]), (8, 1)).astype(np.int32)
    vmask = np.ones((4,), dtype=bool)
    dct_i, ent_i = _distinct_and_entropy(ident, vmask)
    _check("collapse-detect: identical M samples -> distinct=1, entropy=0",
           dct_i == 1 and ent_i < 1e-9)
    diverse = np.array([[1, 2, 3, 1], [2, 1, 3, 1], [3, 2, 1, 1], [1, 1, 2, 3]],
                       dtype=np.int32)
    dct_d, ent_d = _distinct_and_entropy(diverse, vmask)
    _check("collapse-detect: distinct samples -> distinct>1, entropy>0",
           dct_d > 1 and ent_d > 0.0)

    # --- (10) VERTEX-PERMUTATION inverse-mapping correctness --------------------------
    # Permute a known instance, color the permuted graph PROPERLY, inverse-map, and verify
    # the mapped coloring is proper on the ORIGINAL graph (solution-preserving).
    from mycelium.graph_coloring_data import encode_instance, LTYPE_EDGE
    inst_p = {"n": 5, "edges": [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)],  # 5-cycle
              "coloring": [0, 1, 0, 1, 2], "band": "c2.00",
              "deduction_depth": 0, "n_edges": 5, "c_target": 2.0,
              "dsatur_backtracks": 0}
    perm = np.array([2, 4, 0, 3, 1])             # perm[new]=old
    inst_perm = _permute_instance(inst_p, perm)
    _check("symmetry: permuted instance preserves edge count + n",
           inst_perm["n"] == 5 and len(inst_perm["edges"]) == 5)
    # encode original to get membership/cell_valid for the ORIGINAL-graph verifier.
    enc_o = encode_instance(inst_p, 8, 5, k)
    # a proper coloring of the PERMUTED graph (canonical of its own coloring).
    perm_coloring_1b = np.zeros((8,), dtype=np.int32)
    for newv in range(5):
        perm_coloring_1b[newv] = inst_perm["coloring"][newv] + 1
    # inverse-map: pred on relabeled graph -> original vertex order. back[perm[new]]=pred[new]
    back = np.zeros((8,), dtype=np.int32)
    for newv in range(5):
        back[perm[newv]] = perm_coloring_1b[newv]
    proper_on_orig = _coloring_proper_np(back, enc_o["membership"],
                                         enc_o["latent_type"].astype(np.int32),
                                         enc_o["cell_valid"], LTYPE_EDGE)
    _check("symmetry: inverse-mapped proper coloring verifies on the ORIGINAL graph",
           proper_on_orig)
    # gold of the original maps to gold of the permuted graph (round-trip identity check).
    gold_orig_1b = np.array([c + 1 for c in inst_p["coloring"]], dtype=np.int32)
    gold_perm_1b = np.array([c + 1 for c in inst_perm["coloring"]], dtype=np.int32)
    roundtrip = np.array([gold_perm_1b[int(np.nonzero(perm == old)[0][0])]
                          for old in range(5)], dtype=np.int32)
    _check("symmetry: gold round-trips through permute+inverse (solution-preserving)",
           np.array_equal(roundtrip, gold_orig_1b))

    # --- (11) THROUGHPUT accounting (M-forward-honest) --------------------------------
    # temp = 1 forward -> M samples; multistart/symmetry = M forwards. The instances-
    # solved/sec must DROP by ~M when n_forwards goes 1 -> M at the same solve-rate.
    tp1 = _throughput_accounting(B=8, batch_ms=80.0, n_forwards=1,
                                 solve_rate_bestofM=0.5, sym_solve_ms_med=10.0,
                                 sym_solved_rate=1.0)
    tpM = _throughput_accounting(B=8, batch_ms=80.0, n_forwards=64,
                                 solve_rate_bestofM=0.5, sym_solve_ms_med=10.0,
                                 sym_solved_rate=1.0)
    _check("throughput: 1-forward (temp) solved/sec = (B*rate)/(batch_s)",
           abs(tp1["ded_solved_per_s"] - (8 * 0.5) / (80.0 / 1000.0)) < 1e-6)
    _check("throughput: M-forward cost scales solved/sec down by ~M (honest accounting)",
           abs(tp1["ded_solved_per_s"] / tpM["ded_solved_per_s"] - 64.0) < 1e-3)
    _check("throughput: symbolic per-core = (1000/ms)*solved_rate",
           abs(tp1["sym_solved_per_s_per_core"] - 100.0) < 1e-6)

    # --- (12) OOD / EMPTY-BAND logic constants ----------------------------------------
    _check("OOD: trained band densities are (1.0,1.5,2.0,2.5)",
           TRAINED_BAND_DENSITIES == (1.0, 1.5, 2.0, 2.5))
    _check("OOD: trained c-max=2.5 and n-min=8 (the ckpt distribution)",
           TRAINED_BAND_C_MAX == 2.5 and TRAINED_N_MIN == 8)

    # --- (13) JIT ASSIGN-IN-PLACE STATIC CHECK (the frozen-noise-trap guard) -----------
    # The whole adversarial-review point: under @TinyJit the multistart/symmetry update
    # MUST assign IN-PLACE into fg_position_embed / fixed buffers, NEVER rebind the
    # attribute to a NEW Tensor (which freezes the noise -> false collapse). Statically
    # assert (a) run_volume_sweep + run_jit_selftest do NOT contain an assignment whose
    # TARGET is `model.fg_position_embed` (a rebind), and (b) the JIT update mechanisms
    # DO call .assign(...).realize() on fg_position_embed / the fixed buffers.
    # --- (14) EARLY-STOP (a) FREE ACCOUNTING + (b) ACTIVE-SET SCHEDULE + PARITY --------
    print("\n  --- EARLY-STOP logic smoke (synthetic flags; NO GPU forward) ---", flush=True)
    M_es = 16

    # (14a) first_success_index on a known flag array.
    f_known = np.zeros(M_es, dtype=bool)
    f_known[4] = True                                  # first valid at 1-based index 5
    _check("early-stop: first_success_index is the 1-based index of the first valid dart",
           _first_success_index(f_known) == 5)
    _check("early-stop: first_success_index None when no dart verifies",
           _first_success_index(np.zeros(M_es, dtype=bool)) is None)

    # (14b) FREE accounting (multistart/symmetry: each dart = 1 forward). Build flags with
    # KNOWN first-success indices: solved at {1,3,8}, one unsolved.
    es_flags = []
    fs_idx = [1, 3, 8, None]                            # 1-based, None=unsolved
    for fi in fs_idx:
        fa = np.zeros(M_es, dtype=bool)
        if fi is not None:
            fa[fi - 1] = True
        es_flags.append(fa)
    acc = _early_stop_accounting(es_flags, M_es, per_dart_is_forward=True)
    # forwards_used = min(fsi, M) for solved, M for unsolved -> [1,3,8,16].
    _check("early-stop: forwards_used = min(first_success, M) / M-if-unsolved",
           acc["forwards_used"] == [1, 3, 8, M_es])
    _check("early-stop: total_forwards = sum(forwards_used)",
           acc["total_forwards"] == 1 + 3 + 8 + M_es)
    _check("early-stop: exp_forwards = mean(forwards_used)",
           abs(acc["exp_forwards"] - (1 + 3 + 8 + M_es) / 4.0) < 1e-9)
    _check("early-stop: solve_rate = fraction solved within M (3/4 here)",
           abs(acc["solve_rate"] - 0.75) < 1e-12)
    # temp accounting: ONE forward regardless of first-success (no per-dart forward saving).
    acc_temp = _early_stop_accounting(es_flags, M_es, per_dart_is_forward=False)
    _check("early-stop (temp): forwards_used == 1 for EVERY instance (all M from 1 forward)",
           acc_temp["forwards_used"] == [1, 1, 1, 1])
    _check("early-stop (temp): solve_rate identical to per-dart accounting",
           abs(acc_temp["solve_rate"] - acc["solve_rate"]) < 1e-12)

    # (14c) early-stop throughput: exp_forwards < M -> ded_solved/s_ES > the fixed-M number.
    es_tp = _early_stop_throughput(B=8, batch_ms=80.0, exp_forwards=acc["exp_forwards"],
                                   solve_rate=acc["solve_rate"])
    fixedM_tp = _throughput_accounting(B=8, batch_ms=80.0, n_forwards=M_es,
                                       solve_rate_bestofM=acc["solve_rate"],
                                       sym_solve_ms_med=float("nan"), sym_solved_rate=0.0)
    _check("early-stop throughput: ded_solved/s_ES = (B*rate)/(exp_forwards*batch_s)",
           abs(es_tp["ded_solved_per_s_es"]
               - (8 * acc["solve_rate"]) / (acc["exp_forwards"] * 80.0 / 1000.0)) < 1e-6)
    _check("early-stop throughput: ES beats fixed-M when exp_forwards < M (deployment win)",
           es_tp["ded_solved_per_s_es"] > fixedM_tp["ded_solved_per_s"])

    # (14d) ACTIVE-SET CHUNK SCHEDULE (the repacking logic) — per-instance forwards must
    # equal min(first_success, M)/M and the total must match the (a) accounting EXACTLY.
    first_success = [_first_success_index(f) for f in es_flags]
    for ch in (1, 2, 4, 8, 16):
        sched = _active_set_chunk_schedule(first_success, M_es, chunk=ch, B=8)
        _check(f"active-set (chunk={ch}): per-inst forwards == min(first_success,M) / M",
               sched["per_inst_forwards"] == [1, 3, 8, M_es])
        _check(f"active-set (chunk={ch}): total forwards == (a) accounting total",
               sched["total_forwards"] == acc["total_forwards"])
        _check(f"active-set (chunk={ch}): solve_rate == accounting solve_rate (PARITY)",
               abs(sched["solve_rate"] - acc["solve_rate"]) < 1e-12)

    # (14e) PARITY on a RANDOM synthetic bank: early-stop solve-rate == full-M best-of-M,
    # and the chunked schedule total forwards == the (a) accounting total, for ANY chunk.
    rng_es = np.random.RandomState(2027)
    rand_flags = []
    for _ in range(500):
        # iid darts at per-instance p in [0.0, 0.4] -> a mix of easy/hard/unsolved.
        p = rng_es.uniform(0.0, 0.4)
        rand_flags.append(rng_es.random_sample(M_es) < p)
    acc_r = _early_stop_accounting(rand_flags, M_es, per_dart_is_forward=True)
    full_curve = _solve_rate_curve(rand_flags, [M_es])
    _check("early-stop PARITY: (a) accounting solve_rate == full-M best-of-M solve-rate",
           abs(acc_r["solve_rate"] - full_curve["solve_emp"][M_es]) < 1e-9)
    fs_r = [_first_success_index(f) for f in rand_flags]
    parity_chunk_ok = True
    for ch in (1, 3, 8, 16):
        sch = _active_set_chunk_schedule(fs_r, M_es, chunk=ch, B=8)
        if sch["total_forwards"] != acc_r["total_forwards"]:
            parity_chunk_ok = False
        if abs(sch["solve_rate"] - acc_r["solve_rate"]) > 1e-12:
            parity_chunk_ok = False
    _check("early-stop PARITY: chunked schedule total forwards == (a) accounting "
           "(ALL chunk sizes) + solve-rate invariant", parity_chunk_ok)
    # repacking sanity: smaller chunk -> more rounds, but SAME total forwards (the active
    # set drains identically; chunk only changes batching granularity, not forward count).
    s1 = _active_set_chunk_schedule(fs_r, M_es, chunk=1, B=8)
    s16 = _active_set_chunk_schedule(fs_r, M_es, chunk=16, B=8)
    _check("early-stop: chunk=1 has >= rounds than chunk=16 (finer granularity)",
           s1["n_rounds"] >= s16["n_rounds"])
    _check("early-stop: total forwards INVARIANT to chunk size (same darts spent)",
           s1["total_forwards"] == s16["total_forwards"])
    # all-solved-at-dart-1 -> exactly n_inst forwards, 1 round.
    fs_easy = [1] * 40
    s_easy = _active_set_chunk_schedule(fs_easy, M_es, chunk=8, B=8)
    _check("early-stop: trivial band (all solve at dart 1) -> n_inst forwards, 1 round",
           s_easy["total_forwards"] == 40 and s_easy["n_rounds"] == 1)
    # all-unsolved -> n_inst * M forwards (full budget burned), solve_rate 0.
    fs_hard = [None] * 10
    s_hard = _active_set_chunk_schedule(fs_hard, M_es, chunk=8, B=8)
    _check("early-stop: unsolvable band burns the full budget (n_inst*M forwards, rate 0)",
           s_hard["total_forwards"] == 10 * M_es and s_hard["solve_rate"] == 0.0)
    # empty bank safe.
    _check("early-stop: empty bank -> 0 forwards, no crash",
           _active_set_chunk_schedule([], M_es, chunk=8, B=8)["total_forwards"] == 0
           and _early_stop_accounting([], M_es, per_dart_is_forward=True)["n_inst"] == 0)

    # --- (14f) ANTI-TAUTOLOGY DART-DETERMINISM (the reviewer fix's load-bearing test) --
    # The defect was: "dart j of instance X" differed between run_volume_sweep and
    # run_early_stop (different RNG keyings) AND was keyed by the row's transient BATCH
    # POSITION in run_early_stop (which repacking changes) -> the solve-rate parity was a
    # TAUTOLOGY. The fix routes ALL darts through _draw_dart keyed by a STABLE hash of
    # (seed, gid, dart_idx, mech). This block asserts, on CPU with NO forward, that
    # _draw_dart returns the IDENTICAL dart for the same (gid, dart_idx) regardless of
    # the surrounding batch / call path -> parity is now REAL by construction.
    print("\n  --- DART-DETERMINISM (anti-tautology; NO GPU forward) ---", flush=True)
    seed_dd = _band_dart_seed(11, 2.20)
    # (1) _dart_key is a stable hash (reproducible across calls; mech-separated streams).
    _check("dart-key: reproducible (same inputs -> same key)",
           _dart_key(seed_dd, 7, 3, "symmetry") == _dart_key(seed_dd, 7, 3, "symmetry"))
    _check("dart-key: gid changes the stream",
           _dart_key(seed_dd, 7, 3, "symmetry") != _dart_key(seed_dd, 8, 3, "symmetry"))
    _check("dart-key: dart_idx changes the stream",
           _dart_key(seed_dd, 7, 3, "symmetry") != _dart_key(seed_dd, 7, 4, "symmetry"))
    _check("dart-key: mechanism changes the stream",
           _dart_key(seed_dd, 7, 3, "symmetry") != _dart_key(seed_dd, 7, 3, "multistart"))
    _check("dart-key: in legal RandomState seed range [0, 2**31-1)",
           0 <= _dart_key(seed_dd, 7, 3, "symmetry") < 2 ** 31 - 1)

    # (2) SYMMETRY: the SAME (gid, dart_idx) yields the SAME permutation regardless of the
    #     row's POSITION in two DIFFERENT simulated batches (the exact bug: position i was
    #     baked into the old key). Simulate batch A = [gid 5, gid 9] and batch B where gid 5
    #     sits at a DIFFERENT position [gid 2, gid 7, gid 5]; dart j for gid 5 must match.
    n_dd = 11
    perm_posA = _draw_dart("symmetry", 5, 4, seed=seed_dd, n=n_dd)   # gid 5 at position 0
    perm_posB = _draw_dart("symmetry", 5, 4, seed=seed_dd, n=n_dd)   # gid 5 at position 2
    _check("dart-determinism SYMMETRY: dart j for gid X is identical across batch position",
           np.array_equal(perm_posA, perm_posB))
    _check("dart-determinism SYMMETRY: it IS a real permutation of n vertices",
           sorted(perm_posA.tolist()) == list(range(n_dd)))
    _check("dart-determinism SYMMETRY: dart_idx 0 is the IDENTITY (deterministic baseline)",
           np.array_equal(_draw_dart("symmetry", 5, 0, seed=seed_dd, n=n_dd),
                          np.arange(n_dd)))
    _check("dart-determinism SYMMETRY: different gid -> different permutation (diversity)",
           not np.array_equal(perm_posA,
                              _draw_dart("symmetry", 6, 4, seed=seed_dd, n=n_dd)))

    # (3) MULTISTART: noise is batch-SHARED -> the SAME dart_idx yields the SAME noise
    #     regardless of gid / batch (the only honest keying under one shared
    #     fg_position_embed assign). Two calls with DIFFERENT gids but the same dart_idx
    #     must return the IDENTICAL noise array (parity across volume/early-stop).
    pe_shape = (n_dd, 8)
    noise_a = _draw_dart("multistart", GID_SHARED, 3, seed=seed_dd,
                         pos_embed_shape=pe_shape, noise_std=0.05)
    noise_b = _draw_dart("multistart", 999, 3, seed=seed_dd,           # different gid arg
                         pos_embed_shape=pe_shape, noise_std=0.05)
    _check("dart-determinism MULTISTART: dart j noise identical across gid/batch "
           "(batch-shared key)", np.array_equal(noise_a, noise_b))
    _check("dart-determinism MULTISTART: noise array has the pos_embed shape + dtype",
           noise_a.shape == pe_shape and noise_a.dtype == np.float32)
    _check("dart-determinism MULTISTART: dart_idx 0 is ZERO noise (deterministic baseline)",
           not np.any(_draw_dart("multistart", GID_SHARED, 0, seed=seed_dd,
                                 pos_embed_shape=pe_shape, noise_std=0.05)))
    _check("dart-determinism MULTISTART: different dart_idx -> different noise (diversity)",
           not np.array_equal(noise_a,
                              _draw_dart("multistart", GID_SHARED, 4, seed=seed_dd,
                                         pos_embed_shape=pe_shape, noise_std=0.05)))

    # (4) CROSS-PATH PARITY: the per-band dart seed both run_volume_sweep AND
    #     run_early_stop derive (_band_dart_seed) is identical for the same (master_seed,
    #     band), so dart j of instance X is byte-identical on both paths BY CONSTRUCTION.
    _check("dart-determinism CROSS-PATH: _band_dart_seed reproducible (volume==early-stop)",
           _band_dart_seed(11, 2.20) == _band_dart_seed(11, 2.20))
    _check("dart-determinism CROSS-PATH: distinct bands get distinct dart seeds",
           _band_dart_seed(11, 2.20) != _band_dart_seed(11, 2.30))
    _check("dart-determinism CROSS-PATH: temp keying unified (reproducible per gid)",
           _dart_key(seed_dd, 7, 0, "temp") == _dart_key(seed_dd, 7, 0, "temp")
           and _dart_key(seed_dd, 7, 0, "temp") != _dart_key(seed_dd, 8, 0, "temp"))

    # --- (15) VIEW-SUCCESS-COUNT DISTRIBUTION (the diversity-fragility metric) --------
    # Derived from the SAME mech_flags the (B) table consumes. Asserts: buckets PARTITION
    # the instances (sum == n_inst); the >=1 fraction == best-of-M; success-count INCLUDES
    # view 0 so >=1 contains argmax-solved; buckets land in the right columns.
    print("\n  --- VIEW-SUCCESS-COUNT DISTRIBUTION (diversity fragility; NO GPU) ---",
          flush=True)
    M_vsc = 32
    # Hand-built flags with KNOWN success-counts spanning every bucket:
    #   inst A: 0 wins (hard core)        -> bucket 0
    #   inst B: exactly 1 win (fragile)   -> bucket 1
    #   inst C: 3 wins                    -> bucket 2-4
    #   inst D: 10 wins                   -> bucket 5-16
    #   inst E: 25 wins (robust)          -> bucket 17-M
    def _flags_with_k_wins(kw, M):
        f = np.zeros(M, dtype=bool)
        f[:kw] = True
        return f
    vsc_flags = [
        _flags_with_k_wins(0, M_vsc),    # A
        _flags_with_k_wins(1, M_vsc),    # B
        _flags_with_k_wins(3, M_vsc),    # C
        _flags_with_k_wins(10, M_vsc),   # D
        _flags_with_k_wins(25, M_vsc),   # E
    ]
    vsc = _view_success_count_distribution(vsc_flags, M_vsc)
    bmap = dict(vsc["buckets"])
    _check("view-count: per-instance success-counts are correct (0,1,3,10,25)",
           vsc["counts"] == [0, 1, 3, 10, 25])
    _check("view-count: buckets PARTITION the instances (sum == n_inst)",
           sum(c for _, c in vsc["buckets"]) == vsc["n_inst"] == 5)
    _check("view-count: each instance lands in the expected bucket "
           "(0/1/2-4/5-16/17-M = 1 each)",
           bmap["0(hardcore)"] == 1 and bmap["1(fragile)"] == 1
           and bmap["2-4"] == 1 and bmap["5-16"] == 1 and bmap["17-M"] == 1)
    _check("view-count: >=1 fraction = 4/5 (only the hard-core instance has 0 wins)",
           abs(vsc["success_ge1"] - 0.8) < 1e-12 and vsc["n_solved"] == 4)
    # >=1 fraction MUST equal best-of-M solve-rate on the SAME flags (the (B) parity).
    bo_curve = _solve_rate_curve(vsc_flags, [M_vsc])
    _check("view-count: >=1 fraction == best-of-M solve-rate (same flags, (B) parity)",
           abs(vsc["success_ge1"] - bo_curve["solve_emp"][M_vsc]) < 1e-12)
    # success-count INCLUDES view 0 -> the >=1 set CONTAINS every argmax-solved instance.
    # argmax-solved == view-0 wins; here C/D/E have view-0 True (kw>=1 fills index 0),
    # B has view-0 True too -> all 4 view-0-solved instances are in the >=1 set.
    n_view0_solved = sum(int(bool(f[0])) for f in vsc_flags)
    _check("view-count: success-count includes view 0 -> >=1 set contains argmax-solved "
           "(p_argmax <= >=1 fraction)",
           (n_view0_solved / len(vsc_flags)) <= vsc["success_ge1"] + 1e-12)
    _check("view-count: mean success-count = (0+1+3+10+25)/5 = 7.8",
           abs(vsc["mean_success"] - 7.8) < 1e-9)
    # cap at M: a flag array longer than M counts only the first M views.
    over = [np.ones(M_vsc + 5, dtype=bool)]
    vsc_cap = _view_success_count_distribution(over, M_vsc)
    _check("view-count: success-count capped at M (first-M-views only)",
           vsc_cap["counts"] == [M_vsc] and dict(vsc_cap["buckets"])["17-M"] == 1)
    # empty bank safe.
    vsc_empty = _view_success_count_distribution([], M_vsc)
    _check("view-count: empty bank -> n_inst 0, all buckets 0, no crash",
           vsc_empty["n_inst"] == 0
           and sum(c for _, c in vsc_empty["buckets"]) == 0)
    # ALL-hard-core (the C bucket from the PRE-CHECK split): every instance 0 wins ->
    # all mass in 0(hardcore), >=1 fraction 0 (== best-of-M 0).
    hc_flags = [np.zeros(M_vsc, dtype=bool) for _ in range(20)]
    vsc_hc = _view_success_count_distribution(hc_flags, M_vsc)
    _check("view-count: all-hard-core -> all mass in 0(hardcore), >=1 fraction 0",
           dict(vsc_hc["buckets"])["0(hardcore)"] == 20
           and abs(vsc_hc["success_ge1"]) < 1e-12)
    # ALL-robust: every instance wins all M -> all mass in 17-M, >=1 fraction 1.
    rob_flags = [np.ones(M_vsc, dtype=bool) for _ in range(20)]
    vsc_rob = _view_success_count_distribution(rob_flags, M_vsc)
    _check("view-count: all-robust -> all mass in 17-M, >=1 fraction 1",
           dict(vsc_rob["buckets"])["17-M"] == 20
           and abs(vsc_rob["success_ge1"] - 1.0) < 1e-12)

    sip_ok = _static_assign_in_place_check()
    _check("JIT static: no `model.fg_position_embed = Tensor(...)` rebind in the JIT "
           "sweep/selftest (would freeze noise)", sip_ok["no_rebind"])
    _check("JIT static: set_position_embed / restore use fg_position_embed.assign(...)"
           ".realize() (in-place)", sip_ok["pos_assign_in_place"])
    _check("JIT static: set_batch assigns the fixed buffers in place "
           "(buf_*.assign(...).realize())", sip_ok["buf_assign_in_place"])
    _check("JIT static: a @TinyJit-decorated _step exists in the volume JIT factory",
           sip_ok["has_tinyjit_step"])
    _check("JIT static: no dtypes.float32 literal in the file (substrate law)",
           sip_ok["no_float32_literal"])

    # --- (17) PER-DART CAPTURE machinery (CPU; NO GPU, NO tinygrad hook install) -------
    # Exercise _DartCapture pooling + dump/load contract on a fake captured slot, and the
    # global-inst-id scheme. The readout-LN monkeypatch is GPU-path-only (validated on the
    # main thread); here we validate the numpy pooling + npz contract are correct.
    print("\n  --- PER-DART CAPTURE (silhouette pooling + npz contract; NO GPU) ---",
          flush=True)
    H_t = 8
    capt = _DartCapture("symmetry")
    # Fake a captured final-breath readout slot (B=2, S=4, H): row 0 has 3 valid cells,
    # row 1 has 2 valid cells (cells 2,3 padding). Pooling = mean over valid cells.
    fake_last = np.zeros((2, 4, H_t), dtype=np.float32)
    fake_last[0, 0] = 1.0; fake_last[0, 1] = 2.0; fake_last[0, 2] = 3.0  # mean over 3 -> 2.0
    fake_last[1, 0] = 4.0; fake_last[1, 1] = 6.0                          # mean over 2 -> 5.0
    capt._slot["last"] = fake_last
    cv_t = np.array([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=np.float32)
    pooled = capt.pooled_reps(cv_t)
    _check("capture: pooled silhouette = mean over valid cells (row0->2.0, row1->5.0)",
           pooled.shape == (2, H_t)
           and np.allclose(pooled[0], 2.0) and np.allclose(pooled[1], 5.0))
    # Permutation-invariance of the pooled silhouette (pool over a SET).
    perm_last = fake_last[0:1, [2, 0, 1, 3], :]                  # permute row 0's cells
    capt._slot["last"] = perm_last
    pooled_perm = capt.pooled_reps(np.array([[1, 1, 1, 0]], dtype=np.float32))
    _check("capture: pooled silhouette is permutation-invariant (perm cells -> same mean)",
           np.allclose(pooled_perm[0], 2.0))
    # add_dart + global-inst-id namespacing + dump/load round-trip.
    gid_a = _global_inst_id(2.0, 5)
    gid_b = _global_inst_id(2.5, 5)
    _check("capture: global-inst-id band-namespaced (same local idx, diff band -> diff id)",
           gid_a != gid_b and gid_a == 2000 * 1_000_000 + 5)
    capt2 = _DartCapture("symmetry")
    capt2.add_dart(np.ones(H_t, dtype=np.float32) * 1.0, True, gid_a, 2.0)
    capt2.add_dart(np.ones(H_t, dtype=np.float32) * 2.0, False, gid_a, 2.0)
    capt2.add_dart(np.ones(H_t, dtype=np.float32) * 3.0, True, gid_b, 2.5)
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        npz_path = os.path.join(td, "darts.npz")
        capt2.dump(npz_path, {"H": H_t, "mech": "symmetry"})
        z = np.load(npz_path, allow_pickle=True)
        _check("capture: npz has reps (3,H) float32 + valid bool + inst_id int + band float",
               z["reps"].shape == (3, H_t) and z["reps"].dtype == np.float32
               and z["valid"].dtype == bool and z["inst_id"].dtype == np.int64
               and z["band"].dtype == np.float64)
        _check("capture: npz valid flags round-trip (T,F,T)",
               z["valid"].tolist() == [True, False, True])
        _check("capture: npz inst_id round-trips band-namespaced ids",
               z["inst_id"].tolist() == [gid_a, gid_a, gid_b])
        _check("capture: npz meta dict round-trips (H, mech)",
               z["meta"].item()["H"] == H_t and z["meta"].item()["mech"] == "symmetry")
    # arm() clears the slot -> pooled_reps must then raise (no stale rep misread).
    capt2.arm()
    raised = False
    try:
        capt2.pooled_reps(cv_t)
    except RuntimeError:
        raised = True
    _check("capture: arm() clears the slot -> pooled_reps raises (no stale-rep misread)",
           raised)
    # uninstall() with no install is a safe no-op (idempotent).
    capt2.uninstall()
    _check("capture: uninstall() without install is a safe no-op", True)

    print(f"\n[smoke] {'ALL PASS' if ok else 'SOME FAILED'}", flush=True)
    return ok


def _static_assign_in_place_check() -> dict:
    """AST/static guard against the frozen-noise trap. Parses THIS file and verifies the
    JIT update path assigns IN-PLACE (never rebinds model.fg_position_embed) — the
    correctness invariant that makes multistart noise vary per JIT replay. Pure CPU; no
    GPU, no import of the engine. Returns a dict of boolean sub-checks."""
    import ast as _ast
    with open(_THIS_FILE) as f:
        src = f.read()
    tree = _ast.parse(src)

    # collect the function bodies we care about by name.
    funcs = {}
    for node in _ast.walk(tree):
        if isinstance(node, _ast.FunctionDef):
            funcs[node.name] = node

    def _is_pos_embed_attr(target) -> bool:
        # matches `model.fg_position_embed` (Attribute attr == fg_position_embed).
        return (isinstance(target, _ast.Attribute)
                and target.attr == "fg_position_embed")

    # (a) NO REBIND: in run_volume_sweep + run_jit_selftest + _make_volume_jit_fn, there
    # must be NO `model.fg_position_embed = ...` Assign target (a rebind freezes noise).
    no_rebind = True
    for fname in ("run_volume_sweep", "run_jit_selftest", "_make_volume_jit_fn",
                  "_forward_argmax_jit", "run_early_stop", "_es_dart_for_batch"):
        fn = funcs.get(fname)
        if fn is None:
            continue
        for n in _ast.walk(fn):
            if isinstance(n, _ast.Assign):
                for tgt in n.targets:
                    if _is_pos_embed_attr(tgt):
                        no_rebind = False

    # (b) IN-PLACE pos update: _make_volume_jit_fn must contain a call
    # `model.fg_position_embed.assign(...).realize()`.
    def _has_assign_realize_on(fn, base_attr_name: str) -> bool:
        if fn is None:
            return False
        for n in _ast.walk(fn):
            # looking for  X.assign(...).realize()  where X ends in `.<base_attr_name>`
            if (isinstance(n, _ast.Call) and isinstance(n.func, _ast.Attribute)
                    and n.func.attr == "realize"
                    and isinstance(n.func.value, _ast.Call)
                    and isinstance(n.func.value.func, _ast.Attribute)
                    and n.func.value.func.attr == "assign"):
                inner = n.func.value.func.value     # the object .assign was called on
                if (isinstance(inner, _ast.Attribute)
                        and inner.attr == base_attr_name):
                    return True
        return False

    pos_assign_in_place = _has_assign_realize_on(
        funcs.get("_make_volume_jit_fn"), "fg_position_embed")

    # (c) fixed-buffer in-place: set_batch (nested in _make_volume_jit_fn) must call
    # buf_*.assign(...).realize() on the buffer names. Walk the whole factory for any
    # `<name starting with buf_>.assign(...).realize()`.
    buf_assign_in_place = False
    fac = funcs.get("_make_volume_jit_fn")
    if fac is not None:
        for n in _ast.walk(fac):
            if (isinstance(n, _ast.Call) and isinstance(n.func, _ast.Attribute)
                    and n.func.attr == "realize"
                    and isinstance(n.func.value, _ast.Call)
                    and isinstance(n.func.value.func, _ast.Attribute)
                    and n.func.value.func.attr == "assign"):
                inner = n.func.value.func.value
                if isinstance(inner, _ast.Name) and inner.id.startswith("buf_"):
                    buf_assign_in_place = True

    # (d) a @TinyJit-decorated function named _step exists in the factory.
    has_tinyjit_step = False
    if fac is not None:
        for n in _ast.walk(fac):
            if isinstance(n, _ast.FunctionDef) and n.name == "_step":
                for dec in n.decorator_list:
                    dname = (dec.id if isinstance(dec, _ast.Name)
                             else getattr(dec, "attr", None))
                    if dname == "TinyJit":
                        has_tinyjit_step = True

    # (e) substrate law: no `dtypes.float32` literal in CODE (AST attribute access, not
    # comment/docstring text — a substring scan would false-positive on this very check).
    no_float32_literal = True
    for n in _ast.walk(tree):
        if (isinstance(n, _ast.Attribute) and n.attr == "float32"
                and isinstance(n.value, _ast.Name) and n.value.id == "dtypes"):
            no_float32_literal = False
            break

    return {
        "no_rebind": no_rebind,
        "pos_assign_in_place": pos_assign_in_place,
        "buf_assign_in_place": buf_assign_in_place,
        "has_tinyjit_step": has_tinyjit_step,
        "no_float32_literal": no_float32_literal,
    }


# ===========================================================================
# CLI
# ===========================================================================

def _parse_args(argv) -> argparse.Namespace:
    P = argparse.ArgumentParser(description="amortized-speed frontier measurement")
    P.add_argument("--mode", default=os.environ.get("MODE", "speed"),
                   choices=["speed", "volume"],
                   help="speed = symbolic-cost-vs-deducer-accuracy overlay (DEFAULT, "
                        "byte-compatible); volume = best-of-N generate-and-verify sweep")
    # FORWARD-PATH SWITCH (volume mode): jit (default, the @TinyJit compile-once/replay
    # production path) | eager (the un-JIT'd parity reference + fallback).
    P.add_argument("--forward", default=os.environ.get("FORWARD", "jit"),
                   choices=["jit", "eager"],
                   help="[volume] deducer forward path: jit (default, compile-once/replay "
                        "mirroring search_coloring) | eager (parity reference / fallback)")
    P.add_argument("--jit-selftest", action="store_true",
                   help="run the GPU JIT selftest (noise-varies + JIT-vs-eager parity + "
                        "speed) and exit nonzero on failure — the MAIN THREAD runs this "
                        "BEFORE the full sweep")
    P.add_argument("--jit-selftest-tol", type=float,
                   default=float(os.environ.get("JIT_SELFTEST_TOL", "1e-2")),
                   help="max|Δlogit| tolerance for JIT-vs-eager parity in --jit-selftest")
    P.add_argument("--domain", default="coloring",
                   choices=["coloring", "circuit", "kenken"])
    P.add_argument("--ckpt", default=os.environ.get(
        "FG_CKPT", ".cache/fg_ckpts/fg_coloring_k16/fg_coloring_k16_final.safetensors"))
    # SPEED-mode default bands (phase transition); VOLUME mode overrides to the TRAINED
    # distribution below if the user did not pass --bands explicitly.
    P.add_argument("--bands", default=None,
                   help="comma-sep edge-density bands c=m/n. SPEED default = "
                        "1.5,2.0,2.2,2.3,2.5,2.7 (phase transition); VOLUME default = "
                        "1.0,1.5,2.0,2.5 (the ckpt's TRAINED distribution).")
    P.add_argument("--per-band", type=int, default=int(os.environ.get("PER_BAND", "200")),
                   help="k-colorable instances per band")
    P.add_argument("--k", type=int, default=int(os.environ.get("FG_N_VALUES", "3")))
    P.add_argument("--s-max", type=int, default=int(os.environ.get("FG_S_MAX", "49")))
    # SPEED default min-n=40 (hard near-threshold); VOLUME default min-n=8 (the TRAINED
    # n-range) so p is FAIR — set below when unspecified.
    P.add_argument("--min-n", type=int, default=None)
    P.add_argument("--max-n", type=int, default=int(os.environ.get("MAX_N", "49")),
                   help="capped at s_max (engine asserts S==49); raise with a larger-S engine")
    P.add_argument("--K", type=int, default=int(os.environ.get("FG_K_MAX",
                                                               os.environ.get("K", "16"))))
    P.add_argument("--eval-batch", type=int, default=int(os.environ.get("EVAL_BATCH", "8")))
    P.add_argument("--regular-frac", type=float,
                   default=float(os.environ.get("REGULAR_FRAC", "0.4")))
    P.add_argument("--seed", type=int, default=int(os.environ.get("SEED", "42")))
    # symbolic timing controls
    P.add_argument("--sym-budget", type=int,
                   default=int(os.environ.get("SYM_BUDGET", "300000")),
                   help="solve_symbolic node-budget (in-loop cap)")
    P.add_argument("--sym-timeout-ms", type=float,
                   default=float(os.environ.get("SYM_TIMEOUT_MS", "2000")),
                   help="hard wall-clock timeout per symbolic solve (captures blow-up)")
    # verdict thresholds
    P.add_argument("--slow-ms", type=float, default=float(os.environ.get("SLOW_MS", "1000")),
                   help="band is symbolic-slow if p90 wall-clock >= this (ms)")
    P.add_argument("--slow-timeout-rate", type=float,
                   default=float(os.environ.get("SLOW_TIMEOUT_RATE", "0.05")),
                   help="band is symbolic-slow if timeout-rate >= this")
    P.add_argument("--accurate-margin", type=float,
                   default=float(os.environ.get("ACCURATE_MARGIN", "0.15")),
                   help="band is deducer-accurate if proper-rate >= 1/k + this")
    # VOLUME-mode controls
    P.add_argument("--m-max", type=int, default=int(os.environ.get("M_MAX", "256")),
                   help="[volume] max best-of-N sample count M")
    P.add_argument("--diversity", default=os.environ.get("DIVERSITY",
                                                         "temp,multistart,symmetry"),
                   help="[volume] comma-list of diversity mechanisms "
                        "{temp,multistart,symmetry}")
    P.add_argument("--temp", type=float, default=float(os.environ.get("TEMP", "1.0")),
                   help="[volume] TAU for the temp (per-cell softmax) sampler")
    P.add_argument("--multistart-noise", type=float,
                   default=float(os.environ.get("MULTISTART_NOISE", "0.02")),
                   help="[volume] Gaussian std added to the initial residual "
                        "(fg_position_embed) per multistart re-run")
    P.add_argument("--floor-draws", type=int, default=int(os.environ.get("FLOOR_DRAWS", "1")),
                   help="[volume] uniform-random colorings per instance for the empirical "
                        "proper-rate floor")
    # EARLY-STOP (b): opt-in chunked active-set DEPLOYMENT mode (default OFF; the
    # measurement/curve mode stays byte-compatible). Saves real wall-clock by stopping at
    # the first valid dart, dropping solved instances, repacking unsolved into fixed-B
    # batches. Deploys ONE per-dart-forward mechanism (multistart/symmetry; temp saves no
    # forwards). The (a) FREE accounting prints in --mode volume's (C-early) table.
    P.add_argument("--early-stop", action="store_true",
                   help="run the CHUNKED ACTIVE-SET early-stop DEPLOYMENT (opt-in; saves "
                        "wall-clock; reports solve-rate + ACTUAL forwards; asserts PARITY "
                        "vs full-M best-of-M + the (a) accounting). Default OFF.")
    P.add_argument("--chunk", type=int, default=int(os.environ.get("CHUNK", "8")),
                   help="[early-stop] darts drawn per active instance per round before "
                        "repacking the still-unsolved active set (default 8)")
    # NON-INVASIVE PER-DART SILHOUETTE CAPTURE (opt-in; default OFF -> volume mode is
    # byte-identical). When set, install the readout-LN monkeypatch hook (mirroring
    # scripts/probe_svd_collapse.py) and dump per-dart (silhouette, valid_flag, inst_id,
    # band) to an .npz for the Anna-Karenina cluster probe (scripts/dart_cluster_probe.py).
    P.add_argument("--capture-darts", default=os.environ.get("CAPTURE_DARTS", None),
                   help="[volume] PATH to dump a per-dart silhouette .npz (reps, valid, "
                        "inst_id, band, meta) — NON-INVASIVE readout-LN hook, engine/oracle "
                        "git-clean; default OFF (byte-identical volume mode).")
    P.add_argument("--capture-mech", default=os.environ.get("CAPTURE_MECH", "symmetry"),
                   choices=["symmetry", "multistart"],
                   help="[volume,--capture-darts] which per-dart-forward mechanism's darts "
                        "to capture silhouettes for (default symmetry — the hypothesis is "
                        "about vertex-permutation darts). temp is NOT capturable (its M "
                        "samples share one forward / one readout rep).")
    P.add_argument("--capture-rep", default=os.environ.get("CAPTURE_REP", "readout"),
                   choices=["readout", "waist"],
                   help="[volume,--capture-darts] WHICH representation to capture as the "
                        "per-dart silhouette: 'readout' = the final-breath readout-LN 1024d "
                        "hidden (the baseline probe target, AUC 0.85 with a learned head); "
                        "'waist' = the final-breath WAIST d-rep (B,S,d) exposed by the "
                        "engine's fg_waist_capture sink (RE-PROBE the common mode the waist "
                        "created; requires FG_WAIST=1 + a waist ckpt). Feed either npz to "
                        "scripts/dart_cluster_probe.py / learned_waist_gate.py.")
    P.add_argument("--smoke", action="store_true", help="run the CPU smoke and exit")
    args = P.parse_args(argv)
    # mode-aware defaults for --bands / --min-n (reviewer fix #2).
    if args.bands is None:
        args.bands = ("1.0,1.5,2.0,2.5" if args.mode == "volume"
                      else "1.5,2.0,2.2,2.3,2.5,2.7")
    if args.min_n is None:
        env_min = os.environ.get("MIN_N")
        if env_min is not None:
            args.min_n = int(env_min)
        else:
            args.min_n = TRAINED_N_MIN if args.mode == "volume" else 40
    args.bands = [float(x) for x in args.bands.split(",") if x.strip()]
    args.diversity = [m.strip().lower() for m in args.diversity.split(",") if m.strip()]
    valid_mech = {"temp", "multistart", "symmetry"}
    for m in args.diversity:
        if m not in valid_mech:
            raise ValueError(f"unknown diversity mechanism {m!r} "
                             f"(expected any of {sorted(valid_mech)})")
    return args


def main(argv=None) -> int:
    parse_ok = _ast_parse_ok()
    print(f"[ast.parse] astparse_ok={parse_ok}", flush=True)
    if not parse_ok:
        return 1
    if os.environ.get("SELFTEST_ONLY", "0") == "1":
        return 0 if _cpu_smoke() else 1
    args = _parse_args(argv)
    if args.smoke:
        return 0 if _cpu_smoke() else 1
    if args.jit_selftest:
        return run_jit_selftest(args)
    if args.early_stop:
        # opt-in chunked active-set DEPLOYMENT (deliverable (b)); the measurement/curve
        # mode is untouched and remains the default.
        run_early_stop(args)
        return 0
    if args.mode == "volume":
        run_volume_sweep(args)
    else:
        run_gpu_sweep(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
