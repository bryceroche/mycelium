"""search_coloring.py — NEURAL plug-ins (the trained deducer as propagator +
ambiguity-detector) injected into the csp_search branch-and-propagate skeleton,
plus the PATH-B ablation eval driver.

PATH B — policy-free verifier-driven branch-and-propagate.
==========================================================
A viability gate (scripts/analyze_search_guidance_gate.py on fg_coloring_k16_final)
found, on the search-hard band (deduction_depth>=3, cell_acc~0.51):
  - G1 FAIL: the learned policy is CONFIDENTLY WRONG at the vertices it gets wrong
    (puts BELOW-chance mass on gold).  => do NOT use the policy as a search prior/value.
  - G2 PASS (AUC 0.69): per-vertex ENTROPY localises WHERE the deducer is unsure.
  - G4 PASS + MONOTONIC: CLAMPing a few vertices to a commitment + re-deducing
    IMPROVES the rest.  => the deducer PROPAGATES partial commitments productively.
So the search SUBSTRATE works (localise + propagate), only the learned guidance is
bad.  Path B uses the deducer ONLY as a propagator + ambiguity-detector and uses the
CSP's EXACT verifier (mycelium/csp_search.py) as the value/pruning function — graph
coloring is verifiable, so NO learned value is needed.

WHAT THIS FILE BUILDS
---------------------
1. NEURAL PLUG-INS for the csp_search skeleton, parameterised by a single
   `deduce_fn(input_cells_np, value_domain_mask_np) -> probs_np (S, N)` callable
   (the deducer wrapped to a per-vertex color softmax for ONE instance):
     - neural_propagate : CLAMP the assigned vertices into the deducer input
       (input_cells + one-hot value_domain_mask, EXACTLY as the gate script clamps),
       run ONE forward, read per-vertex softmax; AUTO-COMMIT any UNASSIGNED vertex
       whose top-color prob > CONF_THRESH that is constraint-consistent (in D(v) AND
       keeps the partial proper).  SOUND: never commit a color violating an assigned
       neighbour; the exact verifier is the backstop.  Stashes beliefs+entropy in
       state.meta for the var-orderer.  Also records auto-commit accuracy stats keyed
       by clamp-depth (for propagation_quality_vs_clampdepth).
     - neural_entropy_varorder : highest-ENTROPY unresolved vertex (G2), reading the
       entropy the propagator stashed in state.meta (recomputes if absent).
     - neural_policy_valorder  : candidate colors ordered by the deducer softmax
       descending, filtered to D(v).
2. THE ABLATION GRID over the HARD band (deduction_depth>=3), all on the SAME hard
   instances, SAME budgets, sharing the ONE skeleton (fair-by-construction):
     B0  pure deduction, no search (one K forward, argmax)            — the floor (~0.51 cell)
     B1  search, NO propagation (noop), DSATUR varorder, LCV valorder — does search alone help
     B2  search, NEURAL propagation, ENTROPY varorder, policy valorder— THE PROPOSAL
     B2b search, NEURAL propagation, DSATUR varorder, LCV valorder    — ablate the orderer
     B3  symbolic ceiling (AC-3 + DSATUR + LCV)                       — the headroom / ceiling
   Sweep budget in SEARCH_BUDGETS (default {25,50,100,200}).  SEARCH_CONFIG selects a
   single config, or "all" runs the whole grid.

   FAIRNESS — neural propagation is ADVISORY (Fix 1).  B2/B2b run the ONE skeleton with
   can_certify_unsat=False + a COMPLETE fallback (B1-style: noop propagate, DSATUR+LCV,
   may certify unsat).  A confidently-WRONG auto-commit (the G1 failure) can manufacture
   a conflict at/near the root and dead-end a SOLVABLE instance — without the fix that
   returns a FALSE 'unsat' (which AC-3 never does), unfairly understating B2/B2b.  With
   the fix the neural search gracefully degrades to complete enumeration: neural prop can
   only ACCELERATE (collapse the tree when right) or fall back safely (when wrong), never
   certify non-colorability.  Soundness is preserved (the exact verifier stays the final
   arbiter; a fallen-back search still returns only verifier-proper solutions).  The
   wasted neural decisions/backtracks/forwards on a failed propagation are counted.
3. METRICS (success = PROPER-coloring-found via the exact verifier, NEVER gold-match):
     - solve_rate@budget : fraction of hard instances with a COMPLETE PROPER coloring
       found within budget (is_complete_proper; frame-invariant).
     - decisions_to_solve : mean+median decision-nodes on SOLVED instances (THE read —
       does the learned propagator collapse the tree like AC-3 does?).
     - forwards_per_solve : mean+median NEURAL forward-passes on SOLVED instances (Fix
       2 — the HONEST compute counterweight to "fewer decisions"). One deduce_fn call =
       one forward. Symbolic B1/B3 -> 0; B0 -> 1; B2/B2b -> the real GPU cost. Read
       TOGETHER with decisions_to_solve: B2/B2b may win on decisions but pay N forwards
       EACH, so "neural collapses the tree" is only honest when both columns are shown.
     - backtracks : mean backtracks per instance.
     - propagation_quality_vs_clampdepth : neural auto-commit accuracy (vs a reference
       proper coloring) bucketed by clamp-count (1-4 / 5-8 / 9+) — the
       off-distribution degradation curve.
     - B0 cell_acc continuity sanity (~0.51).

SUBSTRATE LAWS (tinygrad + AM driver)
-------------------------------------
The neural forward REUSES the engine's existing factor_breathing_forward — no new JIT
graph paths.  No dtypes.float32 literal inside any JIT step; softmax is done in numpy
on the materialised logits (as the gate script does).  Toggles (K, spec fields, batch
size) flow into the engine's existing JIT cache key; the per-instance forward keeps
batch size constant so the cached graph is reused across search nodes.

GPU-FREE BUILD + CPU SMOKE
--------------------------
SELFTEST_ONLY=1 injects a MOCK deduce_fn (synthetic near-deterministic per-vertex
softmax) so the FULL neural-plug-in + skeleton wiring runs on CPU with NO GPU/ckpt —
it proves B2 end-to-end solves a small synthetic 3-colorable instance via mock
propagation, AND proves soundness (a deliberately-wrong mock never yields an improper
'solved').  ast.parse clean, CPU import clean, selftest passes.

RUN COMMAND (AMD GPU)
---------------------
  DEV=AMD \\
  FG_CKPT=.cache/fg_ckpts/fg_coloring_k16/fg_coloring_k16_final.safetensors \\
  FG_N_VALUES=3 FG_N_INSTANCES=8000 SEARCH_CONFIG=all SEARCH_BUDGETS=25,50,100,200 \\
      .venv/bin/python3 scripts/search_coloring.py
"""
from __future__ import annotations

import ast
import math
import os
import sys

_THIS_FILE = os.path.abspath(__file__)
sys.path.insert(0, os.path.dirname(os.path.dirname(_THIS_FILE)))

import numpy as np  # noqa: E402

# PHASE-0 REFACTOR: the search now drives through the GENERAL predicate-driven core
# (mycelium/csp_core.py) + the coloring DOMAIN (mycelium/csp_domains.py), NOT the
# old coloring-specific csp_search module. The neural deducer enters ONLY as ORDERING
# PRIORS wrapping the general MRV/LCV; it is NEVER a propagate path (commit only when
# forced) except behind the explicitly-kept refuted B2 auto-commit ablation arm.
#   DSATUR  -> mrv_varorder on the not-equal registry (proven, parity-pinned)
#   AC-3    -> gac_propagate on the not-equal predicate (proven, parity-pinned)
#   LCV     -> the general lcv_valorder
#   verifier-> verify_complete / is_consistent_partial (via the coloring wrappers)
from mycelium.csp_core import (  # noqa: E402
    UNASSIGNED,
    CSPState,
    Problem,
    assign_var,
    backtrack_search,
    gac_propagate,
    lcv_valorder,
    mrv_varorder,
    noop_propagate,
    solve_symbolic as _core_solve_symbolic,
)
from mycelium.csp_domains import (  # noqa: E402
    coloring_registry,
    edges_from_membership,
    is_complete_proper_coloring,
    is_proper_partial_coloring,
    problem_from_coloring,
)
# Coloring-only graph utilities (edge normalization / adjacency) used by the neural
# plug-ins for entropy-degree tie-breaks and edge handling — pure coloring helpers,
# kept in the legacy module's normalize_edges/build_adjacency (no search logic).
from mycelium.csp_coloring_legacy import (  # noqa: E402
    build_adjacency,
    normalize_edges,
)


# ===========================================================================
# COLORING<->GENERAL-CORE ADAPTERS (keep the driver's coloring-shaped call sites)
# ===========================================================================
# These translate the driver's coloring vocabulary (vertices/colors/edges) into the
# general core's vocabulary (vars/values/factors), so the ablation grid, metrics, and
# run commands are byte-for-byte preserved while the SEARCH runs through csp_core.

def _coloring_problem(n_vertices, edges, k):
    """Build a general not-equal Problem for one coloring instance (cached registry)."""
    return problem_from_coloring(n_vertices, edges, k, registry=coloring_registry())


def is_complete_proper(assignment, edges, n_vertices, k=None):
    """Coloring success arbiter via the GENERAL verify_complete (csp_domains wrapper).
    k defaults to (max color + 1) if omitted, matching the legacy signature's leniency
    — but the driver always passes k, so this is just signature parity."""
    if k is None:
        present = [int(c) for c in (assignment.values() if isinstance(assignment, dict)
                                    else assignment) if int(c) != UNASSIGNED]
        k = (max(present) + 1) if present else 1
    return is_complete_proper_coloring(assignment, n_vertices, edges, k)


def is_proper_partial(assignment, edges, n_vertices=None):
    """Coloring soundness gate via the GENERAL is_consistent_partial (csp_domains
    wrapper). n_vertices defaults to (max index + 1); the driver always passes it."""
    if n_vertices is None:
        if isinstance(assignment, dict):
            n_vertices = (max(assignment) + 1) if assignment else 0
        else:
            n_vertices = len(assignment)
    present = [int(c) for c in (assignment.values() if isinstance(assignment, dict)
                                else assignment) if int(c) != UNASSIGNED]
    k = (max(present) + 1) if present else 1
    return is_proper_partial_coloring(assignment, n_vertices, edges, k)


def solve_symbolic(n_vertices, edges, k, budget=100000, seed=0):
    """The B3 symbolic ceiling, coloring signature -> general core. Plugs
    {gac_propagate(not_equal), mrv_varorder(=DSATUR), lcv_valorder} via the core's
    solve_symbolic on the not-equal Problem."""
    return _core_solve_symbolic(_coloring_problem(n_vertices, edges, k),
                                budget=budget, seed=seed)


def assign_vertex(state, v, c):
    """Coloring alias for the general assign_var (var=vertex, value=color)."""
    return assign_var(state, v, c)


def dsatur_varorder(state):
    """DSATUR == general MRV on the not-equal registry (proven, parity-pinned)."""
    return mrv_varorder(state)


# NOTE: the general lcv_valorder(state, v, prior=None) is imported directly and used by
# the symbolic configs (B1/B3) with the legacy (state, v) call shape — its optional
# `prior` arg defaults to None, so B1/B3 read identically to the legacy LCV.


# ===========================================================================
# FORWARD-PASS COUNTER (Fix 2 — honest neural compute cost)
# ===========================================================================
# decisions_to_solve alone HIDES the neural compute cost: a config that "collapses
# the tree" to few decisions may pay N GPU forwards PER node. To make the
# "neural collapses the tree" claim honest, we count EVERY deduce_fn call (one neural
# forward = one increment) and report forwards_per_solve alongside decisions_to_solve.
#   - symbolic configs (B1 no-prop, B3 ceiling) never call deduce_fn  -> forwards == 0
#   - B0 pure-deduce calls deduce_fn exactly once                     -> forwards == 1
#   - B2/B2b call it once per propagation round per node              -> forwards >> 1
# The counter is reset PER (config, instance) run and read out right after, so each
# config's forward cost is attributed correctly even though deduce_fn is shared.

class ForwardCounter:
    """Wraps a deduce_fn so every call increments a counter. `count` is the running
    total since the last reset(); wrap the SAME underlying deducer once per instance
    and reset() before each config run to attribute forwards per (config, instance)."""

    def __init__(self, deduce_fn):
        self._deduce_fn = deduce_fn
        self.count = 0
        # forward the .set_instance hook (the GPU deduce_fn carries it) so callers can
        # still re-point the engine at the current instance through the wrapper.
        if hasattr(deduce_fn, "set_instance"):
            self.set_instance = deduce_fn.set_instance

    def reset(self):
        self.count = 0

    def __call__(self, ic_np, vdm_np):
        self.count += 1
        return self._deduce_fn(ic_np, vdm_np)


# ===========================================================================
# AST parse gate (always runs, even on CPU)
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
# NEURAL PLUG-IN FACTORY
# ===========================================================================
# A `deduce_fn(input_cells_np, value_domain_mask_np) -> probs_np` is the ONLY thing
# the neural plug-ins need from the deducer.  It takes ONE instance's
#   input_cells_np      : (S,) int      0=unknown, 1..k = clamped color
#   value_domain_mask_np: (S, k) float  1=legal, 0=illegal
# and returns
#   probs_np            : (S, k) float  per-vertex color softmax (final breath).
# The GPU driver builds this from factor_breathing_forward (one instance per call,
# batch slot 0); the selftest injects a MOCK.  The plug-ins below NEVER touch the
# engine directly — that is what makes the CPU smoke possible.

def make_neural_plugins(
    deduce_fn,
    n_vertices: int,
    edges,
    k: int,
    s_max: int,
    cell_valid_np: np.ndarray,
    conf_thresh: float = 0.9,
    reference_coloring=None,
    stats=None,
):
    """Construct (neural_propagate, neural_entropy_varorder, neural_policy_valorder)
    bound to ONE instance's graph + deducer.

    Args:
      deduce_fn(ic_np (S,) int, vdm_np (S,k) float) -> probs_np (S,k) float.
      n_vertices : real vertex count for this instance.
      edges      : normalized 0-indexed edge list.
      k          : number of colors.
      s_max      : padded grid size (engine S; >= n_vertices).
      cell_valid_np : (s_max,) float, 1 for real vertices.
      conf_thresh : auto-commit threshold on the top color prob (CONF_THRESH env).
      reference_coloring : optional list[int] length n_vertices, a proper coloring
        (e.g. symbolic-ceiling solution) used ONLY to score auto-commit accuracy for
        propagation_quality_vs_clampdepth.  NEVER used for search decisions.
      stats : optional dict accumulator; if given, auto-commit accuracy is recorded
        per clamp-depth bucket under stats["clampdepth"][bucket] = [n_correct, n_total].

    Returns (propagate_fn, varorder_fn, valorder_fn) for backtrack_search.
    """
    edges_n = normalize_edges(edges, n_vertices)
    adj = build_adjacency(edges_n, n_vertices)

    def _run_deduce(state: CSPState):
        """Clamp state.assigned() into the deducer input EXACTLY as the gate script
        clamps, run ONE forward, return probs over real vertices (S, k)."""
        ic_np = np.zeros((s_max,), dtype=np.int32)
        vdm_np = np.zeros((s_max, k), dtype=np.float32)
        # real vertices start fully-legal (coloring has no per-vertex domain), pad=0.
        for v in range(n_vertices):
            if cell_valid_np[v] > 0.5:
                vdm_np[v, :] = 1.0
        # clamp assigned vertices: input_cells=color+1, one-hot value_domain_mask.
        for v, c in state.assigned().items():
            if v >= n_vertices or cell_valid_np[v] <= 0.5:
                continue
            ic_np[v] = int(c) + 1            # engine gold/input encoding is color+1
            vdm_np[v, :] = 0.0
            vdm_np[v, c] = 1.0
        probs = deduce_fn(ic_np, vdm_np)     # (s_max, k)
        return np.asarray(probs, dtype=np.float32)

    def _entropy_per_vertex(probs_np: np.ndarray) -> dict:
        ent = {}
        log_k = math.log(float(k)) + 1e-12
        for v in range(n_vertices):
            if cell_valid_np[v] <= 0.5:
                continue
            p = probs_np[v].clip(1e-12, 1.0)
            e = float(-np.sum(p * np.log(p)))     # nats
            ent[v] = e / log_k                    # 0..1 normalised
        return ent

    def _beliefs_dict(probs_np: np.ndarray) -> dict:
        return {v: probs_np[v].tolist() for v in range(n_vertices)
                if cell_valid_np[v] > 0.5}

    def _record_autocommit(v: int, c: int, n_clamped_before: int):
        if stats is None or reference_coloring is None:
            return
        if v >= len(reference_coloring):
            return
        bucket = ("1-4" if n_clamped_before <= 4
                  else "5-8" if n_clamped_before <= 8 else "9+")
        cd = stats.setdefault("clampdepth", {})
        rec = cd.setdefault(bucket, [0, 0])       # [n_correct, n_total]
        rec[1] += 1
        if int(reference_coloring[v]) == int(c):
            rec[0] += 1

    def neural_propagate(state: CSPState) -> CSPState:
        """CLAMP assigned -> ONE forward -> AUTO-COMMIT confident, consistent,
        proper-keeping unassigned vertices.  Iterate until no new commit (each round
        re-deduces with the newly-committed vertices clamped — this is the productive
        propagation that collapses the tree, G4).

        THIS IS THE REFUTED B2 AUTO-COMMIT ARM (spec §3.2): kept ONLY so the negative
        result stays reproducible, run behind can_certify_unsat=False + a complete
        fallback. It is NOT the default and NOT sanctioned — a sound commit must be
        FORCED (gac's sole-survivor), never a 0.9-confident guess. The general core has
        NO prior_biased_propagate; this arm lives entirely in the driver."""
        s = state.copy()
        n_clamped_before = s.n_assigned        # clamp-depth at entry (for stats)
        n_real = state.problem.n_vars
        last_probs = None
        # bounded fixpoint: at most n_vars rounds (each commits >=1 or stops).
        for _round in range(n_real + 1):
            probs_np = _run_deduce(s)
            last_probs = probs_np
            committed_any = False
            # gather candidates sorted by confidence so the most-certain commit first.
            cands = []
            for v in s.unassigned_vars():
                if cell_valid_np[v] <= 0.5:
                    continue
                p = probs_np[v]
                top = int(np.argmax(p))
                cands.append((float(p[top]), v, top))
            cands.sort(reverse=True)            # highest confidence first
            for conf, v, top in cands:
                if s.values[v] != UNASSIGNED:
                    continue                    # already committed this round
                if conf <= conf_thresh:
                    break                       # rest are below threshold
                if top not in s.domains[v]:
                    continue                    # SOUND: respect domain pruning
                cand = assign_var(s, v, top)
                # SOUND: never commit a color that violates an assigned neighbour.
                if not is_proper_partial(cand.values, edges_n, n_real):
                    continue
                _record_autocommit(v, top, n_clamped_before)
                s = cand
                committed_any = True
            if not committed_any:
                break
        # stash beliefs + entropy from the LAST forward for the var-orderer.
        if last_probs is not None:
            s.meta["beliefs"] = _beliefs_dict(last_probs)
            s.meta["entropy"] = _entropy_per_vertex(last_probs)
        return s

    def neural_entropy_varorder(state: CSPState) -> int:
        """MRV primary; deducer-ENTROPY as the tie-break among the MRV-optimal set (G2,
        finding c). This is the SANCTIONED ordering-prior path (spec §3.2): the neural
        signal reorders, the verifier+GAC prune. Reads stashed entropy; if absent (no
        propagation ran), runs one forward to compute it.

        Implemented as the general mrv_varorder with a per-var entropy `prior`, so it
        REDUCES to pure MRV when the deducer is absent/uninformative (strictly additive).
        """
        ent = state.meta.get("entropy")
        if not ent:
            probs_np = _run_deduce(state)
            ent = _entropy_per_vertex(probs_np)
        # Build a length-n_vars prior: entropy where known, else 0 (no bias).
        n_real = state.problem.n_vars
        prior = [float(ent.get(v, 0.0)) for v in range(n_real)]
        return mrv_varorder(state, prior=prior)

    def neural_policy_valorder(state: CSPState, v: int) -> list:
        """LCV order biased by the deducer marginal (advisory, domain-filtered; spec
        §3.2: prob-then-LCV, value-order 2nd-order). Implemented as the general
        lcv_valorder with a per-value `prior` = the deducer softmax for v, so it reduces
        to pure LCV when the deducer is absent (strictly additive)."""
        beliefs = state.meta.get("beliefs") or {}
        p = beliefs.get(v)
        if p is None:
            probs_np = _run_deduce(state)
            p = probs_np[v].tolist()
        return lcv_valorder(state, v, prior=list(p))

    return neural_propagate, neural_entropy_varorder, neural_policy_valorder


# ===========================================================================
# B0 — pure deduction (no search): one forward, argmax, cell_acc continuity
# ===========================================================================

def b0_pure_deduce(deduce_fn, n_vertices, edges, k, s_max, cell_valid_np,
                   gold_np=None):
    """One K-breath forward on the all-zero (no-givens) input; argmax decode.

    Returns dict:
      pred        : list[int] length n_vertices, 0-indexed colors (canonicalised? no
                    — raw argmax; gold-match is NOT the success metric).
      proper      : bool — is the raw argmax a complete proper coloring?
      cell_acc    : float or None — fraction of real vertices matching gold_np
                    (gold_np 1..k, 0=pad), for the ~0.51 continuity sanity ONLY.
    """
    ic_np = np.zeros((s_max,), dtype=np.int32)
    vdm_np = np.zeros((s_max, k), dtype=np.float32)
    for v in range(n_vertices):
        if cell_valid_np[v] > 0.5:
            vdm_np[v, :] = 1.0
    probs = np.asarray(deduce_fn(ic_np, vdm_np), dtype=np.float32)
    pred = [int(np.argmax(probs[v])) for v in range(n_vertices)]
    edges_n = normalize_edges(edges, n_vertices)
    proper = is_complete_proper(pred, edges_n, n_vertices, k=k)
    cell_acc = None
    if gold_np is not None:
        n_ok = n_tot = 0
        for v in range(n_vertices):
            if cell_valid_np[v] <= 0.5:
                continue
            g = int(gold_np[v])
            if g < 1 or g > k:
                continue
            n_tot += 1
            if pred[v] == g - 1:
                n_ok += 1
        cell_acc = (n_ok / n_tot) if n_tot else None
    return {"pred": pred, "proper": proper, "cell_acc": cell_acc}


# ===========================================================================
# CONFIG GRID — the five ablation configs (all share the ONE skeleton)
# ===========================================================================

CONFIGS = ["B0", "B1", "B2", "B2c", "B2b", "B3"]

CONFIG_DESC = {
    "B0":  "pure deduction, no search (one forward, argmax)              [floor]",
    "B1":  "search, NO propagation (noop) + DSATUR + LCV                 [search-only]",
    "B2":  "search, NEURAL prop + ENTROPY varorder + policy valorder     [THE PROPOSAL]",
    "B2c": "search, NEURAL prop + ENTROPY varorder + LCV valorder        [value-order fix]",
    "B2b": "search, NEURAL prop + DSATUR varorder + LCV valorder         [ablate orderer]",
    "B3":  "symbolic ceiling: AC-3 + DSATUR + LCV                        [ceiling]",
}


def run_config_on_instance(
    config: str,
    deduce_fn,
    n_vertices,
    edges,
    k,
    s_max,
    cell_valid_np,
    budget,
    seed=0,
    conf_thresh=0.9,
    reference_coloring=None,
    stats=None,
):
    """Run ONE config on ONE instance at ONE budget.  Returns the skeleton dict
    augmented with 'solved' (bool) per the EXACT verifier and 'forwards' (int) the
    neural forward-pass count for THIS run (Fix 2).  All configs share
    backtrack_search; only the plug-ins differ (fairness).

    FAIRNESS (Fix 1): B2/B2b run with can_certify_unsat=False + a COMPLETE fallback
    (B1-style: noop propagate, DSATUR+LCV, allowed to certify unsat). So a confidently-
    WRONG neural auto-commit that dead-ends a SOLVABLE instance NEVER produces a false
    'unsat' — it gracefully degrades to complete search. The wasted neural decisions/
    backtracks are merged into the result by the skeleton; the wasted forwards are
    captured by the shared ForwardCounter (so the fallback cost is counted honestly).
    """
    edges_n = normalize_edges(edges, n_vertices)
    problem = _coloring_problem(n_vertices, edges_n, k)   # the general not-equal Problem

    # Fix 2: reset the per-run forward counter (no-op for symbolic configs / a plain
    # deduce_fn). Read .count back after the run to attribute forwards to THIS config.
    if isinstance(deduce_fn, ForwardCounter):
        deduce_fn.reset()

    if config == "B3":
        # B3 ceiling = gac_propagate(not_equal) + mrv(=DSATUR) + lcv via the core.
        res = solve_symbolic(n_vertices, edges_n, k, budget=budget, seed=seed)
    elif config == "B1":
        res = backtrack_search(
            problem,
            propagate_fn=noop_propagate,
            varorder_fn=dsatur_varorder,       # == general mrv_varorder
            valorder_fn=lcv_valorder,
            budget=budget, seed=seed,
            can_certify_unsat=True,            # B1 is complete -> may certify unsat
        )
    elif config in ("B2", "B2b", "B2c"):
        prop, ent_var, pol_val = make_neural_plugins(
            deduce_fn, n_vertices, edges_n, k, s_max, cell_valid_np,
            conf_thresh=conf_thresh, reference_coloring=reference_coloring,
            stats=stats,
        )

        def _complete_fallback():
            """COMPLETE search with NO trusted neural commits (noop propagate), allowed
            to certify unsat. This is the graceful B1-style degradation a neural config
            falls back to when an auto-commit dead-ends a solvable instance (Fix 1).
            It shares THIS instance's deduce_fn counter only incidentally — noop never
            calls it, so the only forwards charged to the fallback are the wasted
            neural ones already spent before the dead-end."""
            return backtrack_search(
                problem,
                propagate_fn=noop_propagate,
                varorder_fn=dsatur_varorder,
                valorder_fn=lcv_valorder,
                budget=budget, seed=seed,
                can_certify_unsat=True,
            )

        if config == "B2":
            res = backtrack_search(
                problem,
                propagate_fn=prop,
                varorder_fn=ent_var,
                valorder_fn=pol_val,
                budget=budget, seed=seed,
                can_certify_unsat=False,       # neural prop CANNOT certify unsat (Fix 1)
                fallback_fn=_complete_fallback,
            )
        elif config == "B2c":              # neural prop + ENTROPY varorder + LCV valorder
            res = backtrack_search(         # isolates value-ordering vs B2 (entropy fixed)
                problem,
                propagate_fn=prop,
                varorder_fn=ent_var,
                valorder_fn=lcv_valorder,
                budget=budget, seed=seed,
                can_certify_unsat=False,       # neural prop CANNOT certify unsat (Fix 1)
                fallback_fn=_complete_fallback,
            )
        else:  # B2b: neural prop but symbolic orderer + valorder (ablate the orderer)
            res = backtrack_search(
                problem,
                propagate_fn=prop,
                varorder_fn=dsatur_varorder,
                valorder_fn=lcv_valorder,
                budget=budget, seed=seed,
                can_certify_unsat=False,       # neural prop CANNOT certify unsat (Fix 1)
                fallback_fn=_complete_fallback,
            )
    else:
        raise ValueError(f"unknown config {config!r}")

    # Fix 2: attribute the neural forward cost for THIS run (0 for symbolic configs).
    res["forwards"] = deduce_fn.count if isinstance(deduce_fn, ForwardCounter) else 0

    # success = PROPER coloring found (frame-invariant), via the exact verifier.
    res["solved"] = (
        res["status"] == "solved"
        and is_complete_proper(res["assignment"], edges_n, n_vertices, k=k)
    )
    return res


# ===========================================================================
# METRICS AGGREGATION
# ===========================================================================

def _new_agg():
    return {
        "n": 0,                     # instances seen
        "n_solved": 0,
        "decisions_solved": [],     # decisions on solved instances
        "forwards_solved": [],      # neural forward-passes on solved instances (Fix 2)
        "backtracks_all": [],       # backtracks per instance
    }


def _summarise_agg(agg) -> dict:
    n = agg["n"]
    ns = agg["n_solved"]
    dec = agg["decisions_solved"]
    fwd = agg["forwards_solved"]
    bt = agg["backtracks_all"]
    return {
        "n": n,
        "solve_rate": (ns / n) if n else 0.0,
        "n_solved": ns,
        "decisions_mean": (float(np.mean(dec)) if dec else float("nan")),
        "decisions_median": (float(np.median(dec)) if dec else float("nan")),
        # Fix 2: neural forward-pass cost on solved instances (the honest compute
        # counterweight to "fewer decisions"). 0 for symbolic B1/B3; 1 for B0.
        "forwards_mean": (float(np.mean(fwd)) if fwd else float("nan")),
        "forwards_median": (float(np.median(fwd)) if fwd else float("nan")),
        "backtracks_mean": (float(np.mean(bt)) if bt else float("nan")),
    }


def _print_comparison_table(results_by_config, budgets):
    """results_by_config[config][budget] = summarised agg dict.  Prints solve_rate +
    decisions_to_solve per config per budget, then a final comparison block."""
    print(f"\n{'='*78}", flush=True)
    print("  PATH-B ABLATION GRID — HARD band (deduction_depth>=3)", flush=True)
    print("  success = PROPER coloring found (exact verifier), NOT gold-match",
          flush=True)
    print(f"{'='*78}", flush=True)

    # solve_rate@budget table
    print("\n  solve_rate @ budget", flush=True)
    hdr = f"  {'config':<6} " + " ".join(f"{('b='+str(b)):>9}" for b in budgets)
    print(hdr, flush=True)
    print("  " + "-" * (len(hdr) - 2), flush=True)
    for cfg in CONFIGS:
        if cfg not in results_by_config:
            continue
        cells = []
        for b in budgets:
            s = results_by_config[cfg].get(b)
            cells.append(f"{s['solve_rate']:9.3f}" if s else f"{'-':>9}")
        print(f"  {cfg:<6} " + " ".join(cells), flush=True)

    # decisions_to_solve (mean) table — the decisive tree-size read
    print("\n  decisions_to_solve (mean on solved) @ budget", flush=True)
    print(hdr, flush=True)
    print("  " + "-" * (len(hdr) - 2), flush=True)
    for cfg in CONFIGS:
        if cfg not in results_by_config:
            continue
        cells = []
        for b in budgets:
            s = results_by_config[cfg].get(b)
            if s and not math.isnan(s["decisions_mean"]):
                cells.append(f"{s['decisions_mean']:9.1f}")
            else:
                cells.append(f"{'-':>9}")
        print(f"  {cfg:<6} " + " ".join(cells), flush=True)

    # forwards_per_solve (mean) table — the HONEST neural compute cost (Fix 2).
    # Read TOGETHER with decisions_to_solve: B2/B2b may use FEWER decisions but pay
    # N forwards EACH (B1/B3 forwards==0, B0 forwards==1).
    print("\n  forwards_per_solve (mean neural forwards on solved) @ budget",
          flush=True)
    print(hdr, flush=True)
    print("  " + "-" * (len(hdr) - 2), flush=True)
    for cfg in CONFIGS:
        if cfg not in results_by_config:
            continue
        cells = []
        for b in budgets:
            s = results_by_config[cfg].get(b)
            if s and not math.isnan(s["forwards_mean"]):
                cells.append(f"{s['forwards_mean']:9.1f}")
            else:
                cells.append(f"{'-':>9}")
        print(f"  {cfg:<6} " + " ".join(cells), flush=True)

    # forwards_per_solve (median) table
    print("\n  forwards_per_solve (median neural forwards on solved) @ budget",
          flush=True)
    print(hdr, flush=True)
    print("  " + "-" * (len(hdr) - 2), flush=True)
    for cfg in CONFIGS:
        if cfg not in results_by_config:
            continue
        cells = []
        for b in budgets:
            s = results_by_config[cfg].get(b)
            if s and not math.isnan(s["forwards_median"]):
                cells.append(f"{s['forwards_median']:9.1f}")
            else:
                cells.append(f"{'-':>9}")
        print(f"  {cfg:<6} " + " ".join(cells), flush=True)

    # backtracks (mean) table
    print("\n  backtracks (mean per instance) @ budget", flush=True)
    print(hdr, flush=True)
    print("  " + "-" * (len(hdr) - 2), flush=True)
    for cfg in CONFIGS:
        if cfg not in results_by_config:
            continue
        cells = []
        for b in budgets:
            s = results_by_config[cfg].get(b)
            if s and not math.isnan(s["backtracks_mean"]):
                cells.append(f"{s['backtracks_mean']:9.1f}")
            else:
                cells.append(f"{'-':>9}")
        print(f"  {cfg:<6} " + " ".join(cells), flush=True)
    print("", flush=True)


def _print_clampdepth(stats):
    """propagation_quality_vs_clampdepth: neural auto-commit accuracy by clamp-count
    bucket (1-4 / 5-8 / 9+) — the off-distribution degradation curve."""
    cd = (stats or {}).get("clampdepth", {})
    print(f"\n{'='*60}", flush=True)
    print("  propagation_quality_vs_clampdepth (neural auto-commit acc)", flush=True)
    print("  vs a proper reference coloring; informs Path-A partial-givens retraining",
          flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  {'clamp-depth':<12}  {'auto-commit acc':>16}  {'n_commits':>10}",
          flush=True)
    print("  " + "-" * 44, flush=True)
    for bucket in ("1-4", "5-8", "9+"):
        rec = cd.get(bucket)
        if not rec or rec[1] == 0:
            print(f"  {bucket:<12}  {'-':>16}  {0:>10}", flush=True)
        else:
            acc = rec[0] / rec[1]
            print(f"  {bucket:<12}  {acc:16.3f}  {rec[1]:>10}", flush=True)
    print("", flush=True)


# ===========================================================================
# THE GPU DRIVER (real eval) — builds deduce_fn from the engine forward
# ===========================================================================

def _build_model_and_loader():
    """Build the Pythia-410M deducer + coloring loader EXACTLY as eval_coloring_bands /
    the gate script do.  Returns (model, spec, loader, env)."""
    import gc
    from tinygrad import Device
    from tinygrad.helpers import getenv

    from mycelium import Config
    from mycelium.loader import _load_state, load_breathing
    from mycelium.factor_graph_engine import (
        FactorGraphSpec,
        attach_factor_graph_params,
        FG_HYP_MASK,
    )
    from mycelium.factor_masks import attach_factor_hyperbolic_params
    from mycelium.graph_coloring_data import GraphColoringLoader

    env = {
        "CKPT": getenv(
            "FG_CKPT",
            ".cache/fg_ckpts/fg_coloring_k16/fg_coloring_k16_final.safetensors"),
        "K": int(getenv("FG_K_MAX", getenv("K", "16"))),
        "EVAL_BATCH": int(getenv("EVAL_BATCH", getenv("BATCH", "8"))),
        "SEED": int(getenv("SEED", "42")),
        "N_INSTANCES": int(getenv("FG_N_INSTANCES", "8000")),
        "S_MAX": int(getenv("FG_S_MAX", "49")),
        "N_VALUES": int(getenv("FG_N_VALUES", "3")),
        "CONF_THRESH": float(getenv("CONF_THRESH", "0.9")),
        "HARD_THRESH": int(getenv("HARD_THRESH", "3")),
        "MAX_HARD": int(getenv("MAX_HARD", "0")),   # 0 = all hard instances
        # Fix 2: cheap reference-coloring budget (was a hardcoded 200000). <=49-vertex
        # 3-colorable instances solve in well under ~3000 DSATUR+AC-3 nodes.
        "REF_COLOR_BUDGET": int(getenv("REF_COLOR_BUDGET", "5000")),
    }

    spec = FactorGraphSpec(
        s_max=env["S_MAX"],
        n_values=env["N_VALUES"],
        n_factor_types=1,
        n_heads=16,
        k_max=env["K"],
        has_factor_inlet=False,
    )

    print("loading Pythia-410M -> BreathingTransformer...", flush=True)
    cfg = Config()
    sd = _load_state()
    model = load_breathing(cfg, sd=sd)
    del sd
    gc.collect()
    cast_layers_fp32(model)
    attach_factor_graph_params(model, hidden=cfg.hidden, spec=spec)

    if FG_HYP_MASK:
        print("[FG_HYP_MASK=1] building coloring anchor tables...", flush=True)
        _ref_loader = GraphColoringLoader(
            n_instances=env["N_INSTANCES"], s_max=env["S_MAX"],
            k_colors=env["N_VALUES"], batch_size=env["EVAL_BATCH"], seed=env["SEED"],
        )
        _ref_batch = _ref_loader.sample_batch()
        _mem_np = _ref_batch.membership.realize().numpy()
        _lt_np = _ref_batch.latent_type.realize().numpy()
        attach_factor_hyperbolic_params(
            model, n_heads=spec.n_heads, n_factor_types=spec.n_factor_types,
            s_max=spec.s_max, membership_np=_mem_np, latent_type_np=_lt_np,
        )
        del _ref_loader, _ref_batch, _mem_np, _lt_np
        print("  coloring hyperbolic params attached (frozen).", flush=True)

    Device[Device.DEFAULT].synchronize()
    print(f"loading checkpoint: {env['CKPT']}", flush=True)
    load_ckpt(model, env["CKPT"])

    loader = GraphColoringLoader(
        n_instances=env["N_INSTANCES"], s_max=env["S_MAX"],
        k_colors=env["N_VALUES"], batch_size=env["EVAL_BATCH"], seed=env["SEED"],
    )
    print(f"  test set: {len(loader.test_records)} instances", flush=True)
    return model, spec, loader, env


def cast_layers_fp32(model):
    """Mirror of eval_coloring_bands.cast_layers_fp32 / the gate script."""
    from tinygrad import dtypes

    def _cast(obj, attr):
        t = getattr(obj, attr)
        if t.dtype == dtypes.half:
            setattr(obj, attr, t.cast(dtypes.float).contiguous().realize())

    _cast(model.embed, "weight")
    sw = model.block.shared
    for a in ("wv", "bv", "wo", "bo", "w_out", "b_out"):
        _cast(sw, a)
    for layer in model.block.layers:
        for a in ("wq", "bq", "wk", "bk", "w_in", "b_in"):
            _cast(layer, a)


_FG_PARAM_NAMES = [
    "fg_state_embed", "fg_position_embed", "fg_value_codebook",
    "fg_calib_head_w", "fg_calib_head_b", "fg_breath_embed", "fg_delta_gate",
]


def model_state_dict_fg(model) -> dict:
    sd = {"ln_f.g": model.ln_f_g, "ln_f.b": model.ln_f_b}
    sw = model.block.shared
    for a in ("wv", "bv", "wo", "bo", "w_out", "b_out",
              "in_ln_g", "in_ln_b", "post_ln_g", "post_ln_b"):
        sd[f"shared.{a}"] = getattr(sw, a)
    for i, layer in enumerate(model.block.layers):
        for a in ("wq", "bq", "wk", "bk", "w_in", "b_in"):
            sd[f"phase{i}.{a}"] = getattr(layer, a)
    for nm in _FG_PARAM_NAMES:
        sd[nm] = getattr(model, nm)
    return sd


def load_ckpt(model, path: str):
    from tinygrad.nn.state import safe_load
    sd = safe_load(path)
    targets = model_state_dict_fg(model)
    missing = []
    for name, dst in targets.items():
        if name not in sd:
            missing.append(name)
            continue
        src = sd[name].to(dst.device).realize()
        if src.shape != dst.shape:
            try:
                src = src.reshape(dst.shape)
            except Exception:
                missing.append(f"{name} (shape mismatch)")
                continue
        if src.dtype != dst.dtype:
            src = src.cast(dst.dtype)
        dst.assign(src).realize()
    if missing:
        print(f"  ckpt missing {len(missing)} keys: "
              f"{missing[:5]}{'...' if len(missing) > 5 else ''}", flush=True)
    else:
        print(f"  ckpt loaded cleanly ({len(targets)} keys).", flush=True)


def _make_engine_deduce_fn(model, spec, ref_batch, slot=0):
    """Build a deduce_fn(ic_np (S,), vdm_np (S,k)) -> probs_np (S,k) that runs ONE
    forward through factor_breathing_forward, wrapped in a @TinyJit step that
    COMPILES ONCE and REPLAYS for every subsequent call (all instances, all tree
    nodes).  This is FIX 1: the old path built fresh Tensors as graph constants on
    EVERY call, so tinygrad re-traced + recompiled the full K=16-breath graph for
    each of the THOUSANDS of search-tree-node forwards (the 100%-one-core CPU stall).

    DISCIPLINE
    ----------
    * FIXED, realized input buffers allocated ONCE; shapes are CONSTANT for the whole
      run (B, S, N, L are all fixed by the ref_batch + spec).  These are the buffers
      the @TinyJit step captures as its graph inputs.
    * deduce_fn / set_instance ASSIGN-IN-PLACE into those SAME buffer objects
      (`buf.assign(new).realize()`) OUTSIDE the JIT, then call the JIT step.  The
      .realize() materialises the assign before the JIT replays, mirroring the proven
      factor_graph_train pattern (inputs are realized buffers passed to the @TinyJit
      step; the assign happens eagerly outside the traced graph, so the
      "assign-inside-JIT must be returned" quirk does NOT apply here).
    * The K=16-breath graph, the spec, batch size B, and all toggles live in the
      closure -> they are baked into the compiled graph (the JIT cache key).

    PER-INSTANCE MASK REPLAY — THE CORRECTNESS GUARANTEE
    ----------------------------------------------------
    The attention mask is built INSIDE factor_breathing_forward by
    build_factor_attn_bias / build_factor_hyperbolic_attn_bias from
    (membership, latent_type, cell_valid).  Those builders are PURE TENSOR OPS:
    matmul co-occurrence (m_t^T @ m_t), one_hot, cumsum, broadcast, max, clip,
    where()-free arithmetic — NO python-level branching on the *values* of
    membership/latent_type (only on STATIC shape/head-allocation constants, which are
    identical across instances since L, s_max, n_heads, n_factor_types are fixed).
    Therefore the compiled graph reads membership/latent_type/cell_valid as LIVE
    BUFFER CONTENTS: after set_instance(B) reassigns those buffers, the SAME compiled
    graph recomputes instance-B's mask correctly — there is NO stale instance-A mask
    baked into the graph.  (Verifiable via VERIFY_PARITY=1 below.)

    EFFICIENCY CHOICE: B-BROADCAST (not B=1).
    We keep the ref_batch's batch size B and broadcast the single instance across all
    B slots, reading slot `slot`.  This costs B x the compute per forward, but it
    keeps the engine's JIT cache key (B, spec) identical to the training/eval path and
    avoids a second compiled graph for a B=1 shape.  The load-bearing fix is the JIT
    (compile-once / replay), which removes the per-node recompile entirely; the B x
    factor is a constant multiplier on an already-fast replay.  A dedicated B=1 batch
    was considered but skipped to keep the JIT + mask build trivially correct
    (one graph, one cache key).
    """
    from tinygrad import Tensor, dtypes
    from tinygrad.engine.jit import TinyJit
    from tinygrad.helpers import getenv
    from mycelium.factor_graph_engine import factor_breathing_forward

    B = int(ref_batch.input_cells.shape[0])
    S = int(ref_batch.cell_valid.shape[1])
    L = int(ref_batch.membership.shape[1])
    N = spec.n_values
    K = spec.k_max

    # ---- FIXED, realized input buffers (allocated ONCE) ---------------------------
    # These are the @TinyJit step's graph inputs.  deduce_fn / set_instance assign new
    # numpy values into these SAME objects in place; the compiled graph then replays.
    #   input_cells       (B, S)    int
    #   value_domain_mask (B, S, N) float
    #   membership        (B, L, S) float
    #   latent_type       (B, L)    int
    #   cell_valid        (B, S)    float
    buf_ic = Tensor(np.zeros((B, S), dtype=np.int32),
                    dtype=dtypes.int).contiguous().realize()
    buf_vdm = Tensor(np.zeros((B, S, N), dtype=np.float32),
                     dtype=dtypes.float).contiguous().realize()
    buf_mem = Tensor(np.zeros((B, L, S), dtype=np.float32),
                     dtype=dtypes.float).contiguous().realize()
    buf_lt = Tensor(np.zeros((B, L), dtype=np.int32),
                    dtype=dtypes.int).contiguous().realize()
    buf_cv = Tensor(np.zeros((B, S), dtype=np.float32),
                    dtype=dtypes.float).contiguous().realize()

    class _ProxyBatch:
        """Thin attribute shim so the engine reads the fixed buffers by attr.
        Holds the SAME buffer objects the JIT captured (no per-call Tensor build)."""
        def __init__(self):
            self.input_cells = buf_ic
            self.cell_valid = buf_cv
            self.value_domain_mask = buf_vdm
            self.gold = buf_ic                 # unused for inference (shape-compatible)
            self.membership = buf_mem
            self.latent_type = buf_lt
            self.factor_inlet = None
            self.deduction_depth = [0] * B

    _proxy = _ProxyBatch()

    @TinyJit
    def _step() -> Tensor:
        # All toggles/shape params (K, spec, B) are in the closure -> baked into the
        # compiled graph.  Reads the fixed buffers (captured), returns the final-breath
        # logits realized.  No host sync inside; no dtypes.float32 literal baked.
        logits_history, _ = factor_breathing_forward(model, _proxy, spec, K=K)
        return logits_history[-1].realize()    # (B, S, N)

    def set_instance(mem_row_np, lt_row_np, cv_row_np):
        """mem_row_np (n_edges_max, S), lt_row_np (n_edges_max,), cv_row_np (S,).
        Broadcast the single instance across all B slots and ASSIGN-IN-PLACE into the
        membership/latent_type/cell_valid buffers the JIT captured.  Changing the
        instance does NOT recompile and does NOT leave a stale mask (the mask is
        recomputed from these LIVE buffer contents on the next _step replay)."""
        mem_b = np.broadcast_to(mem_row_np[None], (B,) + mem_row_np.shape).copy()
        lt_b = np.broadcast_to(lt_row_np[None], (B,) + lt_row_np.shape).copy()
        cv_b = np.broadcast_to(cv_row_np[None], (B, S)).copy()
        buf_mem.assign(Tensor(mem_b.astype(np.float32), dtype=dtypes.float)).realize()
        buf_lt.assign(Tensor(lt_b.astype(np.int32), dtype=dtypes.int)).realize()
        buf_cv.assign(Tensor(cv_b.astype(np.float32), dtype=dtypes.float)).realize()

    def deduce_fn(ic_np, vdm_np):
        # ASSIGN-IN-PLACE the new ic/vdm (broadcast over B; read slot only), then
        # REPLAY the compiled graph.  No new graph-constant Tensors -> no recompile.
        ic_b = np.broadcast_to(ic_np[None], (B, S)).copy()
        vdm_b = np.broadcast_to(vdm_np[None], (B, S, N)).copy()
        buf_ic.assign(Tensor(ic_b.astype(np.int32), dtype=dtypes.int)).realize()
        buf_vdm.assign(Tensor(vdm_b.astype(np.float32), dtype=dtypes.float)).realize()
        final_logits = _step()                           # (B, S, N) — JIT replay
        logits_np = final_logits.numpy()[slot]           # (S, N)
        e = np.exp(logits_np - logits_np.max(axis=-1, keepdims=True))
        probs = e / (e.sum(axis=-1, keepdims=True) + 1e-12)
        return probs.astype(np.float32)

    # ---- OPTIONAL PARITY CHECK (VERIFY_PARITY=1) — eager vs JIT, two instances ----
    # Proves (a) the JIT'd deduce_fn matches a one-off eager factor_breathing_forward
    # on instance A (<1e-3), and (b) after set_instance(B) the SAME compiled graph
    # matches instance B's eager forward (NO stale-mask bake-in).  Bryce runs this on
    # the GPU; it is a no-op when VERIFY_PARITY is unset.
    def _verify_parity(insts):
        """insts: list of (mem_row_np, lt_row_np, cv_row_np, ic_np, vdm_np) tuples.
        Returns max|Δ| per instance vs an eager forward built the OLD way (fresh
        Tensors), asserting < 1e-3 and that each instance gets ITS OWN logits."""
        from mycelium.factor_graph_engine import factor_breathing_forward as _fbf

        class _EagerBatch:
            def __init__(self, ic, vdm, mem, lt, cv):
                self.input_cells = ic; self.value_domain_mask = vdm
                self.membership = mem; self.latent_type = lt; self.cell_valid = cv
                self.gold = ic; self.factor_inlet = None
                self.deduction_depth = [0] * B

        deltas = []
        for (mem_row, lt_row, cv_row, ic_np, vdm_np) in insts:
            # --- eager forward (fresh Tensors, the OLD path) ---
            mem_b = np.broadcast_to(mem_row[None], (B,) + mem_row.shape).copy()
            lt_b = np.broadcast_to(lt_row[None], (B,) + lt_row.shape).copy()
            cv_b = np.broadcast_to(cv_row[None], (B, S)).copy()
            ic_b = np.broadcast_to(ic_np[None], (B, S)).copy()
            vdm_b = np.broadcast_to(vdm_np[None], (B, S, N)).copy()
            eb = _EagerBatch(
                Tensor(ic_b.astype(np.int32), dtype=dtypes.int).contiguous().realize(),
                Tensor(vdm_b.astype(np.float32), dtype=dtypes.float).contiguous().realize(),
                Tensor(mem_b.astype(np.float32), dtype=dtypes.float).contiguous().realize(),
                Tensor(lt_b.astype(np.int32), dtype=dtypes.int).contiguous().realize(),
                Tensor(cv_b.astype(np.float32), dtype=dtypes.float).contiguous().realize(),
            )
            eager_logits = _fbf(model, eb, spec, K=K)[0][-1].realize().numpy()[slot]
            # --- JIT'd path (assign-in-place + replay) ---
            set_instance(mem_row, lt_row, cv_row)
            buf_ic.assign(Tensor(ic_b.astype(np.int32), dtype=dtypes.int)).realize()
            buf_vdm.assign(Tensor(vdm_b.astype(np.float32), dtype=dtypes.float)).realize()
            jit_logits = _step().numpy()[slot]
            d = float(np.max(np.abs(eager_logits - jit_logits)))
            deltas.append(d)
            print(f"  [VERIFY_PARITY] instance max|Δlogit| = {d:.2e} "
                  f"({'PASS' if d < 1e-3 else 'FAIL'})", flush=True)
        assert all(d < 1e-3 for d in deltas), \
            f"parity FAILED (stale-mask bake-in?) deltas={deltas}"
        print("  [VERIFY_PARITY] ALL PASS — per-instance mask replay is correct.",
              flush=True)
        return deltas

    deduce_fn.set_instance = set_instance
    deduce_fn.verify_parity = _verify_parity
    return deduce_fn


def run_gpu_eval():
    """The PATH-B ablation eval driver over the HARD band (deduction_depth>=3)."""
    from tinygrad import Tensor
    from tinygrad.helpers import getenv

    model, spec, loader, env = _build_model_and_loader()
    Tensor.training = False

    S_MAX = env["S_MAX"]
    N_VALUES = env["N_VALUES"]
    K = env["K"]
    CONF_THRESH = env["CONF_THRESH"]
    HARD_THRESH = env["HARD_THRESH"]
    MAX_HARD = env["MAX_HARD"]
    REF_COLOR_BUDGET = env["REF_COLOR_BUDGET"]      # Fix 2

    budgets = [int(x) for x in getenv("SEARCH_BUDGETS", "25,50,100,200").split(",")
               if x.strip()]
    cfg_sel = getenv("SEARCH_CONFIG", "all")
    configs = CONFIGS if cfg_sel == "all" else [c for c in CONFIGS
                                                if c.strip() in cfg_sel.split(",")]
    if not configs:
        configs = CONFIGS

    print("\n=== search_coloring.py — PATH-B ablation eval ===", flush=True)
    print(f"K={K}  conf_thresh={CONF_THRESH}  hard_thresh(depth>=)={HARD_THRESH}",
          flush=True)
    print(f"configs={configs}  budgets={budgets}  max_hard={MAX_HARD or 'all'}",
          flush=True)

    # Build a fixed reference batch so the engine JIT cache key (B, spec) is stable.
    ref_batch = loader.sample_batch()
    # Fix 2: wrap the engine deduce_fn in a ForwardCounter so EVERY forward is counted
    # (one neural forward = one increment). reset() per (config, instance) run inside
    # run_config_on_instance / before B0. The .set_instance hook is forwarded.
    _raw_deduce_fn = _make_engine_deduce_fn(model, spec, ref_batch, slot=0)
    deduce_fn = ForwardCounter(_raw_deduce_fn)

    # VERIFY_PARITY=1: GPU parity gate (Fix 1 correctness proof). Run TWO DIFFERENT
    # instances through (a) a fresh-Tensor eager forward and (b) the JIT'd deduce_fn;
    # assert max|Δlogit| < 1e-3 EACH. (b) on instance 2 after set_instance proves the
    # compiled graph recomputes instance-2's mask (NO stale instance-1 bake-in).
    if getenv("VERIFY_PARITY", 0):
        print("\n[VERIFY_PARITY=1] eager-vs-JIT parity on two DIFFERENT instances...",
              flush=True)
        _insts = []
        for _vb in loader.iter_eval(batch_size=env["EVAL_BATCH"]):
            _mem = _vb.membership.realize().numpy()
            _lt = _vb.latent_type.realize().numpy()
            _cv = _vb.cell_valid.realize().numpy()
            for _bi in range(int(_cv.shape[0])):
                _ic = np.zeros((S_MAX,), dtype=np.int32)              # no-givens input
                _vdm = np.zeros((S_MAX, N_VALUES), dtype=np.float32)
                for _v in range(S_MAX):
                    if _cv[_bi, _v] > 0.5:
                        _vdm[_v, :] = 1.0
                _insts.append((_mem[_bi], _lt[_bi], _cv[_bi], _ic, _vdm))
                if len(_insts) >= 2:
                    break
            if len(_insts) >= 2:
                break
        _raw_deduce_fn.verify_parity(_insts)
        print("[VERIFY_PARITY] done.\n", flush=True)

    # accumulators: results_by_config[cfg][budget] = agg
    results = {cfg: {b: _new_agg() for b in budgets} for cfg in configs}
    clamp_stats = {}        # propagation_quality_vs_clampdepth (B2/B2b share)
    b0_cell_accs = []       # B0 continuity sanity

    # iterate the test set, filter to the hard band, run all configs on each instance.
    n_hard = 0
    for batch in loader.iter_eval(batch_size=env["EVAL_BATCH"]):
        mem_np = batch.membership.realize().numpy()        # (B, E, S)
        lt_np = batch.latent_type.realize().numpy()        # (B, E)
        cv_np = batch.cell_valid.realize().numpy()         # (B, S)
        gold_np = batch.gold.realize().numpy().astype(np.int32)  # (B, S)
        Bc = int(cv_np.shape[0])

        for b in range(Bc):
            if int(batch.deduction_depth[b]) < HARD_THRESH:
                continue
            if MAX_HARD and n_hard >= MAX_HARD:
                break
            n_hard += 1

            nv = int(batch.n[b])
            edges = edges_from_membership(mem_np, lt_np, b)
            edges = normalize_edges(edges, nv)

            # point the engine deduce_fn at THIS instance's structure.
            deduce_fn.set_instance(mem_np[b], lt_np[b], cv_np[b])

            # B0 continuity sanity (one forward).
            b0_proper = False
            if "B0" in configs:
                deduce_fn.reset()                  # Fix 2: B0 calls deduce_fn once
                b0 = b0_pure_deduce(deduce_fn, nv, edges, N_VALUES, S_MAX,
                                    cv_np[b], gold_np=gold_np[b])
                b0_forwards = deduce_fn.count      # == 1
                b0_proper = bool(b0["proper"])
                if b0["cell_acc"] is not None:
                    b0_cell_accs.append(b0["cell_acc"])
                # B0 'solve' = the raw argmax already proper (rare on hard band).
                for bud in budgets:
                    agg = results["B0"][bud]
                    agg["n"] += 1
                    if b0["proper"]:
                        agg["n_solved"] += 1
                        agg["decisions_solved"].append(0)
                        agg["forwards_solved"].append(b0_forwards)
                    agg["backtracks_all"].append(0)

            # FIX 2: a reference proper coloring for propagation_quality_vs_clampdepth
            # scoring ONLY (never used for search decisions). Cost reduction:
            #   (a) computed LAZILY — only when a neural config (B2/B2b/B2c) that
            #       actually consumes it is selected (B1/B3-only sweeps skip it);
            #   (b) budget cut 200000 -> REF_COLOR_BUDGET (default 5000). These are
            #       <=49-vertex 3-colorable instances; DSATUR+AC-3 finds a proper
            #       coloring in well under ~3000 nodes, so 200000 was unconsidered
            #       overkill (a pure-Python backtracking burn per hard instance). The
            #       metric is unchanged: a found coloring is still a verifier-proper
            #       reference; if the cheap budget is exhausted the bucket simply
            #       records no commits for that instance (None), exactly as the old
            #       code did on an unsolved ref. 5000 is plenty for this regime; raise
            #       REF_COLOR_BUDGET if any clampdepth bucket looks suspiciously empty.
            needs_ref = any(c in ("B2", "B2b", "B2c") for c in configs)
            reference_coloring = None
            if needs_ref:
                ref_sol = solve_symbolic(nv, edges, N_VALUES, budget=REF_COLOR_BUDGET)
                reference_coloring = (ref_sol["assignment"]
                                      if ref_sol["status"] == "solved" else None)

            # FIX 3: per-instance progress (flush=True) so a long run is never blind.
            inst_solved = {}            # cfg -> solved-at-smallest-budget flag (B1=.. B2=..)
            inst_forwards = 0           # neural forwards spent on THIS instance

            for cfg in configs:
                if cfg == "B0":
                    continue
                for bud in budgets:
                    res = run_config_on_instance(
                        cfg, deduce_fn, nv, edges, N_VALUES, S_MAX, cv_np[b],
                        budget=bud, seed=env["SEED"], conf_thresh=CONF_THRESH,
                        reference_coloring=reference_coloring,
                        stats=clamp_stats if cfg in ("B2", "B2b", "B2c") else None,
                    )
                    agg = results[cfg][bud]
                    agg["n"] += 1
                    if res["solved"]:
                        agg["n_solved"] += 1
                        agg["decisions_solved"].append(res["decisions"])
                        agg["forwards_solved"].append(res["forwards"])
                    agg["backtracks_all"].append(res["backtracks"])
                    inst_solved.setdefault(cfg, res["solved"])  # smallest budget first
                    inst_forwards += int(res.get("forwards", 0))

            # FIX 3: emit one progress line per hard instance.
            solved_str = " ".join(
                f"{c}={'Y' if inst_solved.get(c) else ('Y' if c == 'B0' and b0_proper else 'n')}"
                for c in configs)
            tot = (MAX_HARD if MAX_HARD else "?")
            print(f"[hard {n_hard}/{tot}] depth={int(batch.deduction_depth[b])} "
                  f"nv={nv} solved: {solved_str} forwards={inst_forwards}", flush=True)
        if MAX_HARD and n_hard >= MAX_HARD:
            break

    print(f"\n  evaluated {n_hard} hard instances (depth>={HARD_THRESH}).",
          flush=True)

    # B0 cell_acc continuity sanity.
    if b0_cell_accs:
        print(f"\n  [B0 sanity] mean cell_acc over hard band = "
              f"{float(np.mean(b0_cell_accs)):.3f}  (expect ~0.51)", flush=True)

    # summarise + print.
    summarised = {cfg: {b: _summarise_agg(results[cfg][b]) for b in budgets}
                  for cfg in configs}
    _print_comparison_table(summarised, budgets)
    if any(c in configs for c in ("B2", "B2b", "B2c")):
        _print_clampdepth(clamp_stats)

    print("\n  config legend:", flush=True)
    for c in configs:
        print(f"    {c:<5} {CONFIG_DESC[c]}", flush=True)
    print("\n  read: decisions_to_solve is the decisive tree-size comparison —",
          flush=True)
    print("  does NEURAL prop (B2/B2b) collapse the tree like symbolic AC-3 (B3)?",
          flush=True)


# ===========================================================================
# CPU SELFTEST (SELFTEST_ONLY=1) — MOCK deduce_fn, NO GPU/ckpt
# ===========================================================================

def _mock_deduce_fn(n_vertices, edges, k, s_max, reference_coloring, sharpness=12.0):
    """A stand-in deducer: returns a per-vertex color softmax that PROPAGATES from the
    clamped vertices.  For each unassigned vertex, the score for a color is pushed DOWN
    by clamped/known neighbours already wearing it (so confidence concentrates on a
    locally-consistent color once enough neighbours are clamped), and biased toward the
    reference coloring so the search converges.  This mimics G4: clamping a few vertices
    makes the rest confident + correct.  NO GPU, pure numpy.
    """
    edges_n = normalize_edges(edges, n_vertices)
    adj = build_adjacency(edges_n, n_vertices)

    def deduce_fn(ic_np, vdm_np):
        probs = np.full((s_max, k), 1.0 / k, dtype=np.float32)
        # recover clamped colors from ic_np (color = ic-1).
        clamped = {v: int(ic_np[v]) - 1 for v in range(n_vertices) if ic_np[v] > 0}
        for v in range(n_vertices):
            logits = np.zeros((k,), dtype=np.float32)
            # bias toward the reference coloring (the "right answer" the deducer
            # would tend toward) — modest, so it is NOT a perfect oracle.
            if reference_coloring is not None and v < len(reference_coloring):
                logits[int(reference_coloring[v])] += 1.0
            # push DOWN colors used by clamped neighbours (constraint propagation).
            for w in adj[v]:
                if w in clamped:
                    logits[clamped[w]] -= sharpness
            # vertex itself clamped -> near one-hot.
            if v in clamped:
                logits = np.full((k,), -sharpness, dtype=np.float32)
                logits[clamped[v]] = sharpness
            # respect value_domain_mask (illegal -> very low).
            for c in range(k):
                if vdm_np[v, c] < 0.5:
                    logits[c] -= 50.0
            e = np.exp(logits - logits.max())
            probs[v] = e / (e.sum() + 1e-12)
        return probs

    return deduce_fn


def _selftest() -> bool:
    all_ok = True

    def _check(name, cond):
        nonlocal all_ok
        if not cond:
            all_ok = False
        print(f"[selftest] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    k = 3
    s_max = 12

    # --- a small, search-nontrivial, GENUINELY 3-colorable instance ---
    # Triangular prism (3-prism): two triangles 0-1-2 and 3-4-5 joined by a matching
    # 0-3, 1-4, 2-5.  It is 3-chromatic (each triangle forces all 3 colors) and needs
    # real constraint propagation, but unlike a wheel it IS 3-colorable.
    n = 6
    tri_a = [(0, 1), (1, 2), (0, 2)]
    tri_b = [(3, 4), (4, 5), (3, 5)]
    rungs = [(0, 3), (1, 4), (2, 5)]
    edges = normalize_edges(tri_a + tri_b + rungs, n)
    cell_valid_np = np.zeros((s_max,), dtype=np.float32)
    cell_valid_np[:n] = 1.0

    # reference proper coloring via the symbolic ceiling (ground truth for the mock).
    ref = solve_symbolic(n, edges, k, budget=100000)
    _check("symbolic ceiling solves 3-prism (3-colorable)", ref["status"] == "solved")
    reference_coloring = ref["assignment"]
    _check("ceiling solution is proper",
           is_complete_proper(reference_coloring, edges, n, k=k))

    # Fix 2: wrap the mock in a ForwardCounter so the selftest can assert the
    # forwards_per_solve bookkeeping (run_config_on_instance resets it per run).
    mock = ForwardCounter(_mock_deduce_fn(n, edges, k, s_max, reference_coloring))

    # --- TEST: B0 pure-deduce runs + reports cell_acc ---
    # gold_np = reference coloring +1 (engine 1..k encoding) for the sanity field.
    gold_np = np.zeros((s_max,), dtype=np.int32)
    for v in range(n):
        gold_np[v] = reference_coloring[v] + 1
    mock.reset()
    b0 = b0_pure_deduce(mock, n, edges, k, s_max, cell_valid_np, gold_np=gold_np)
    _check("B0 pure-deduce returns a pred of length n", len(b0["pred"]) == n)
    _check("B0 cell_acc computed (mock biased to ref -> high)",
           b0["cell_acc"] is not None and b0["cell_acc"] > 0.5)
    # Fix 2: B0 calls deduce_fn EXACTLY once.
    _check("fix2: B0 forwards == 1 (one forward, no search)", mock.count == 1)

    # --- TEST: B2 (neural prop + entropy varorder + policy valorder) solves it ---
    stats = {}
    res_b2 = run_config_on_instance(
        "B2", mock, n, edges, k, s_max, cell_valid_np, budget=200, seed=1,
        conf_thresh=0.9, reference_coloring=reference_coloring, stats=stats)
    _check("B2 (neural prop+entropy+policy) solves 3-prism",
           res_b2["solved"] is True)
    _check("B2 solution verifies PROPER (exact verifier)",
           is_complete_proper(res_b2["assignment"], edges, n, k=k))
    _check("B2 recorded auto-commit stats (propagation collapsed the tree)",
           bool(stats.get("clampdepth")))
    # Fix 2: B2 pays N neural forwards (one per propagation round per node) -> > 0.
    _check("fix2: B2 forwards > 0 (neural compute cost recorded)",
           res_b2["forwards"] > 0)

    # --- TEST: B2b (neural prop + DSATUR + LCV) also solves it ---
    res_b2b = run_config_on_instance(
        "B2b", mock, n, edges, k, s_max, cell_valid_np, budget=200, seed=1,
        conf_thresh=0.9, reference_coloring=reference_coloring, stats={})
    _check("B2b (neural prop+DSATUR+LCV) solves 3-prism", res_b2b["solved"] is True)
    _check("fix2: B2b forwards > 0 (neural compute cost recorded)",
           res_b2b["forwards"] > 0)

    # --- TEST: B1 (search, no propagation) solves it (smaller graph, easy) ---
    res_b1 = run_config_on_instance(
        "B1", mock, n, edges, k, s_max, cell_valid_np, budget=500, seed=1)
    _check("B1 (search, no propagation) solves 3-prism", res_b1["solved"] is True)
    # Fix 2: B1 is symbolic (noop propagate) -> never calls deduce_fn -> 0 forwards.
    _check("fix2: B1 forwards == 0 (symbolic, no neural)", res_b1["forwards"] == 0)

    # --- TEST: B3 symbolic ceiling solves it ---
    res_b3 = run_config_on_instance(
        "B3", mock, n, edges, k, s_max, cell_valid_np, budget=200, seed=1)
    _check("B3 (symbolic ceiling) solves 3-prism", res_b3["solved"] is True)
    # Fix 2: B3 is the symbolic ceiling -> 0 forwards.
    _check("fix2: B3 forwards == 0 (symbolic ceiling, no neural)",
           res_b3["forwards"] == 0)

    # --- DECISIVE READ smoke: neural prop should not need MORE decisions than B1 ---
    _check("B2 decisions <= B1 decisions (neural prop collapses tree)",
           res_b2["decisions"] <= res_b1["decisions"] + 1)

    # --- SOUNDNESS (adversarial): a deducer that confidently auto-commits a WRONG
    # (improper) color must NEVER make the skeleton report an improper 'solved'. ---
    # Build a deducer that always says "everyone is color 0 with prob 1.0".
    def bad_deduce(ic_np, vdm_np):
        probs = np.full((s_max, k), 1e-6, dtype=np.float32)
        for v in range(n):
            # if domain forbids 0, fall back legal; else slam color 0 confidently.
            if vdm_np[v, 0] >= 0.5:
                probs[v] = 1e-6
                probs[v, 0] = 1.0
            else:
                legal = [c for c in range(k) if vdm_np[v, c] >= 0.5]
                if legal:
                    probs[v] = 1e-6
                    probs[v, legal[0]] = 1.0
            probs[v] /= probs[v].sum()
        return probs

    res_bad = run_config_on_instance(
        "B2", bad_deduce, n, edges, k, s_max, cell_valid_np, budget=300, seed=1,
        conf_thresh=0.9, reference_coloring=reference_coloring, stats={})
    # if it reports solved, the assignment MUST be verifiably proper (soundness).
    if res_bad["solved"]:
        bad_ok = is_complete_proper(res_bad["assignment"], edges, n, k=k)
    else:
        bad_ok = True   # not solved => sound (the verifier pruned the bad commits)
    _check("SOUNDNESS: all-color-0 deducer never yields improper 'solved'", bad_ok)

    # --- SOUNDNESS (direct): feed an improper coloring to the arbiter ---
    improper = list(reference_coloring)
    # force an edge monochromatic.
    u, w = edges[0]
    improper[w] = improper[u]
    _check("SOUNDNESS: arbiter rejects a hand-built improper coloring",
           is_complete_proper(improper, edges, n, k=k) is False)

    # --- FIX 1 (FAIRNESS): a confidently-WRONG propagator on a SOLVABLE instance must
    # SOLVE via the complete fallback, NEVER return a false 'unsat'. ---------------
    # The 3-prism IS 3-colorable. This deducer confidently auto-commits a sequence of
    # PAIRWISE-consistent colors (each passes the propagator's per-commit proper guard)
    # that GLOBALLY paints the graph into a corner: 0->0,1->1,2->2 (triangle A ok),
    # 3->1,4->0 (rungs/triangle B ok so far) -> vertex 5 is adjacent to 3(=1),4(=0),
    # 2(=2): its domain empties. That manufactures a dead-end at the root on a SOLVABLE
    # instance — exactly the G1 failure the review flagged. The neural propagator's
    # per-commit guard cannot see the global trap, so without Fix 1 the skeleton dead-
    # ends with nothing to backtrack and (with can_certify_unsat=True) would FALSELY
    # report 'unsat'. With Fix 1 (can_certify_unsat=False + complete fallback) B2 falls
    # back to full enumeration and SOLVES.
    _bad_order = {0: 0, 1: 1, 2: 2, 3: 1, 4: 0}     # 5 deliberately left to trap
    _conf_by_v = {0: 0.99, 1: 0.98, 2: 0.97, 3: 0.96, 4: 0.95}  # commit order

    def corner_deduce(ic_np, vdm_np):
        probs = np.full((s_max, k), 1e-6, dtype=np.float32)
        clamped = {v: int(ic_np[v]) - 1 for v in range(n) if ic_np[v] > 0}
        for v in range(n):
            row = np.full((k,), 1e-6, dtype=np.float32)
            if v in clamped:                      # already committed -> echo it
                row[:] = 1e-6
                row[clamped[v]] = 1.0
            elif v in _bad_order and vdm_np[v, _bad_order[v]] >= 0.5:
                tgt = _bad_order[v]
                conf = _conf_by_v[v]
                row[:] = (1.0 - conf) / (k - 1)
                row[tgt] = conf                   # confident on the trap color
            else:
                # vertex 5 (the trap victim) or any domain-blocked vertex: spread over
                # legal colors, low confidence (so it is never auto-committed first).
                legal = [c for c in range(k) if vdm_np[v, c] >= 0.5]
                if legal:
                    for c in legal:
                        row[c] = 1.0 / len(legal)
                else:
                    row[:] = 1.0 / k
            probs[v] = row / row.sum()
        return probs

    corner_mock = ForwardCounter(corner_deduce)

    # (a) WITHOUT the fix: the same neural propagation but allowed to certify unsat
    #     (can_certify_unsat=True, no fallback) returns the BUG — a false 'unsat'.
    prop_bug, ent_bug, pol_bug = make_neural_plugins(
        corner_mock, n, edges, k, s_max, cell_valid_np, conf_thresh=0.9)
    res_bug = backtrack_search(
        _coloring_problem(n, edges, k), propagate_fn=prop_bug, varorder_fn=ent_bug,
        valorder_fn=pol_bug, budget=300, seed=1,
        can_certify_unsat=True, fallback_fn=None)
    _check("fix1: WITHOUT guard, corner-trap propagator returns the BUG (false 'unsat')",
           res_bug["status"] == "unsat")

    # (b) WITH the fix: run via run_config_on_instance (B2 sets can_certify_unsat=False
    #     + the complete fallback) -> SOLVES the SOLVABLE instance via fallback.
    res_fix = run_config_on_instance(
        "B2", corner_mock, n, edges, k, s_max, cell_valid_np, budget=300, seed=1,
        conf_thresh=0.9, reference_coloring=reference_coloring, stats={})
    _check("fix1: WITH guard, B2 falls back and SOLVES (no false unsat on solvable)",
           res_fix["solved"] is True)
    _check("fix1: fallback solution verifies PROPER (soundness preserved)",
           is_complete_proper(res_fix["assignment"], edges, n, k=k) is True)
    _check("fix1: result is flagged fellback (provenance of the graceful degradation)",
           res_fix.get("fellback") is True)
    # the wasted neural forwards on the failed propagation are still counted honestly.
    _check("fix1+fix2: wasted neural forwards on the failed prop are counted (>0)",
           res_fix["forwards"] > 0)

    # --- TEST: the metrics aggregation + table print run without error ---
    agg = _new_agg()
    agg["n"] = 3
    agg["n_solved"] = 2
    agg["decisions_solved"] = [4, 7]
    agg["forwards_solved"] = [10, 20]
    agg["backtracks_all"] = [1, 0, 3]
    summ = _summarise_agg(agg)
    _check("metrics: solve_rate computed", abs(summ["solve_rate"] - 2 / 3) < 1e-9)
    _check("metrics: decisions_median computed",
           abs(summ["decisions_median"] - 5.5) < 1e-9)
    # Fix 2: forwards_per_solve aggregation.
    _check("fix2: forwards_mean computed", abs(summ["forwards_mean"] - 15.0) < 1e-9)
    _check("fix2: forwards_median computed",
           abs(summ["forwards_median"] - 15.0) < 1e-9)
    try:
        _print_comparison_table(
            {"B2": {25: summ}, "B3": {25: summ}}, [25])
        _print_clampdepth({"clampdepth": {"1-4": [3, 4], "5-8": [1, 2]}})
        table_ok = True
    except Exception as e:  # noqa: BLE001
        print(f"  table print raised: {e}", flush=True)
        table_ok = False
    _check("metrics: comparison table + clampdepth print run cleanly", table_ok)

    print(f"[selftest] {'ALL PASS' if all_ok else 'SOME FAILED'}", flush=True)
    return all_ok


# ===========================================================================
# ENTRY POINT
# ===========================================================================

if __name__ == "__main__":
    parse_ok = _ast_parse_ok()
    print(f"[ast.parse] ok={parse_ok}", flush=True)
    if not parse_ok:
        sys.exit(1)
    if os.environ.get("SELFTEST_ONLY", "0") == "1":
        ok = _selftest()
        sys.exit(0 if ok else 1)
    run_gpu_eval()
