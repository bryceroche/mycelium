#!/usr/bin/env python3
"""
frontier_map_dag_probe.py — SALVAGE-or-BURY probe for the soft-MRF frontier.

Pure numpy, CPU-only. NO GPU, NO tinygrad, NO engine code, NO training.

WHY THIS PROBE EXISTS
---------------------
Gate #1 (scripts/frontier_bp_gap_gate.py) tested MARGINALS on the FLAT 2D Ising
torus and returned a SOFT-KILL:
  * the BP-vs-exact MARGINAL gap and its LOCALIZABILITY are ANTI-CORRELATED
    (where the gap is big the per-spin error is UNIFORM noise AND BP stops
    converging; where it is localized the gap is ~0);
  * learned-BP would inherit BP's non-convergence;
  * the engine readout is per-cell independent (a mean-field ceiling -> marginals
    are capped at product-form).
BUT gate-1 surfaced two redirects it could not itself test:
  (A) the MAP (argmax/MPE) gap stays BROAD across the glass band even where
      marginals are close, and MAP sidesteps the mean-field readout ceiling;
  (B) the flat 2D torus may itself be the limiter (the deducer is proven strong
      on DAGs — solves depth-16 circuits in ~4 breaths), so a DAG may carry a
      more STRUCTURED, more correctable gap.

This probe fills the 2x2 {MARGINALS, MAP} x {FLAT 2D Ising, DAG noisy-circuit}
to either SALVAGE soft-MRF (as a learned-MAP-on-DAG frontier) or BURY it.

THE HONEST BAR (the guard against a wishful SALVAGE)
---------------------------------------------------
Optimization (MAP) has strong CHEAP baselines: greedy local search and simulated
annealing. The clean-CSP lesson is that cheap symbolic/heuristic methods beat the
neural approach. So a real frontier requires a regime where SIMULTANEOUSLY:
  (a) vanilla BP (max-product) is INSUFFICIENT (real BP-to-exact MAP gap),
  (b) the error is STRUCTURED / LOCALIZED (BP's own confidence — the max-marginal
      margin — predicts WHICH nodes BP-MAP gets wrong; AUC > ~0.65), AND
  (c) cheap local-search / annealing CANNOT easily close it (they leave a real
      residual gap; BP-MAP sits in basins local moves can't escape).
If cheap methods trivially close the MAP gap -> soft-MRF is buried (same lesson)
-> KILL. If they leave a real residual AND the margin localizes the errors ->
room for a smarter learned/global corrector -> SALVAGE.

DAG NOISY-CIRCUIT SOFT FACTOR GRAPH (the new testbed)
-----------------------------------------------------
We REUSE the gate-DAG TOPOLOGY of mycelium/circuit_data.py (the same layered
Boolean-circuit DAGs the deducer solves hard at ~0.97 — the leveled generator and
the caterpillar deep-skinny construction), re-expressed here in pure numpy/python
(circuit_data.py imports tinygrad at module level for batch packing; we deliberately
do NOT import it, to keep this probe tinygrad-free, and instead mirror its EXACT
topology + gate eval + longest-path depth). Each gate is SOFTENED into a CPD factor:

    phi_g(parents, child) = (1 - eps)   if child == gate(parents)
                            eps         otherwise            (a noisy gate)

Leaf inputs get a (near-deterministic) prior pinned to their sampled bit. The
OUTPUT node(s) are OBSERVED (clamped to the sampled forward-eval value). MAP/MPE =
argmax over the unclamped (leaf+intermediate) binary assignment of the joint
product of CPD factors. We keep <= ~18 binary nodes so brute-force exact MPE
(enumerate 2^n) is feasible, and sweep eps (noise) across a band.

WHAT WE MEASURE (per topology x target x band)
----------------------------------------------
  * MARGINAL gap: mean |P_BP - P_exact|, mean KL.
  * MAP gap: Hamming(BP-MAP, exact-MAP), and the normalized logprob gap
    (logP(exact-MAP) - logP(BP-MAP)) / |nodes|.
  * LOCALIZABILITY (the correctable-signal test, the G2 analog): AUC of
    (low max-marginal margin -> node is wrong in BP-MAP). High AUC = the error is
    identifiable = a corrector has a signal.
  * CHEAP-BASELINE CLOSURE (the honest bar): from BP-MAP run (i) greedy local
    search (single- then pairwise-flip hill-climb to a local optimum) and
    (ii) simulated annealing. Report the FRACTION of the BP-to-exact logprob gap
    each closes + a rough cost (flips evaluated).
  * FLAT-vs-DAG contrast: is the MAP gap markedly more STRUCTURED (higher AUC,
    larger residual-after-local-search) on the DAG than the flat torus?

SELFTEST (SELFTEST_ONLY=1) — guards the building blocks where they MUST be exact:
  * max-product BP gives the EXACT MAP on a TREE,
  * sum-product BP gives EXACT marginals on a TREE,
  * brute-force MPE matches an independent enumeration on a tiny case,
  * greedy local search NEVER decreases logprob,
  * simulated annealing returns a VALID assignment,
  * the DAG soft factor graph's exact MPE on a NOISELESS (eps->0) circuit equals
    the deterministic forward-eval assignment (a hard ground truth).

Usage
-----
  SELFTEST_ONLY=1 .venv/bin/python3 scripts/frontier_map_dag_probe.py
  .venv/bin/python3 scripts/frontier_map_dag_probe.py
Env knobs (all optional, all seeded — NO unseeded randomness anywhere):
  SEED (default 20260620), INSTANCES (per cell, default 8),
  BP_ITERS (default 3000), BP_DAMP (default 0.5), BP_TOL (default 1e-9),
  ISING_SIZES (default "3,4"; add via comma), MAX_NODES (DAG cap, default 18),
  SA_SWEEPS (default 200), SA_SEED_OFFSET (default 0).
"""

import os
import sys
import math
import itertools
import numpy as np


# =========================================================================== #
# PART A — GENERAL FACTOR GRAPH (works for BOTH Ising and DAG)                 #
#                                                                             #
# A factor graph here is a list of FACTORS over BINARY variables x in {0,1}.  #
# Each factor is (vars, logtab) where:                                        #
#   vars   : tuple of variable indices this factor touches (arity 1, 2, or 3) #
#   logtab : np.ndarray of shape (2,)*arity giving LOG-potential as a function #
#            of the (0/1) assignment of `vars` in order.                      #
# The joint LOG-score of a full assignment x is the SUM of factor logtabs at  #
# their local assignments.  P(x) propto exp(score(x)).                        #
#                                                                             #
# This unified representation lets ONE exact solver, ONE max/sum-product BP,  #
# ONE local search, and ONE annealer serve both topologies.                   #
# =========================================================================== #


class FactorGraph:
    """A factor graph over n binary variables; factors carry LOG-potentials.

    factors: list of (vars_tuple, logtab_ndarray).  logtab has shape (2,)*len(vars),
             indexed by the 0/1 values of vars in order.
    clamped: dict var->value for OBSERVED variables (their value is fixed; they are
             NOT part of the MAP/marginal search but DO contribute factor terms).
    The free (unclamped) variables are the inference target.
    """

    def __init__(self, n, factors, clamped=None):
        self.n = int(n)
        self.factors = [(tuple(int(v) for v in vs), np.asarray(tab, dtype=np.float64))
                        for (vs, tab) in factors]
        self.clamped = dict(clamped or {})
        for (vs, tab) in self.factors:
            assert tab.shape == (2,) * len(vs), \
                f"factor {vs} logtab shape {tab.shape} != {(2,)*len(vs)}"
        self.free = [v for v in range(self.n) if v not in self.clamped]

    # ---- exact score of a FULL assignment (length-n int array in {0,1}) ----
    def score(self, x):
        s = 0.0
        for (vs, tab) in self.factors:
            idx = tuple(int(x[v]) for v in vs)
            s += tab[idx]
        return float(s)

    # ---- adjacency: for each variable, the factors that touch it ----
    def var_factors(self):
        vf = [[] for _ in range(self.n)]
        for fi, (vs, tab) in enumerate(self.factors):
            for v in vs:
                vf[v].append(fi)
        return vf


# --------------------------------------------------------------------------- #
# EXACT brute-force inference over the FREE variables (clamped held fixed).    #
# --------------------------------------------------------------------------- #
def fg_exact(fg):
    """Brute-force over 2^(#free) assignments of the FREE variables.

    Returns:
      marg_plus : np.array len n, exact P(x_v = 1) for FREE v; for CLAMPED v it is
                  exactly its clamped value (0 or 1).
      map_x     : np.array len n in {0,1}, the argmax-score FULL assignment.
      map_score : float, score of map_x.
    """
    free = fg.free
    F = len(free)
    assert F <= 22, f"too many free vars for brute force: {F}"
    base = np.zeros(fg.n, dtype=np.int64)
    for v, val in fg.clamped.items():
        base[v] = int(val)

    M = 1 << F
    # build all 2^F assignments of the free vars
    idx = np.arange(M, dtype=np.int64)
    free_bits = ((idx[:, None] >> np.arange(F, dtype=np.int64)[None, :]) & 1)  # (M,F)
    X = np.tile(base[None, :], (M, 1))
    for k, v in enumerate(free):
        X[:, v] = free_bits[:, k]

    # score every row
    scores = np.zeros(M, dtype=np.float64)
    for (vs, tab) in fg.factors:
        # gather the multi-index for these vars across all rows
        flat = np.zeros(M, dtype=np.int64)
        stride = 1
        for v in reversed(vs):
            flat += X[:, v] * stride
            stride *= 2
        scores += tab.reshape(-1)[flat]

    smax = scores.max()
    w = np.exp(scores - smax)
    Z = w.sum()
    probs = w / Z
    marg_plus = np.zeros(fg.n, dtype=np.float64)
    for v in range(fg.n):
        if v in fg.clamped:
            marg_plus[v] = float(fg.clamped[v])
        else:
            marg_plus[v] = float((probs * X[:, v]).sum())
    map_m = int(np.argmax(scores))
    map_x = X[map_m].astype(np.int64).copy()
    return marg_plus, map_x, float(scores[map_m])


# --------------------------------------------------------------------------- #
# Loopy BP on a general factor graph (sum-product & max-product), log-domain,  #
# damped, convergence-capped.  Handles clamped variables via a hard unary.     #
# --------------------------------------------------------------------------- #
def fg_bp(fg, mode, n_iters, damping, tol):
    """Damped loopy BP on a general binary factor graph (log-domain).

    Two message families:
      mv2f[(v,fi)] : log message variable v -> factor fi, over v in {0,1}, shape (2,)
      mf2v[(fi,v)] : log message factor fi -> variable v, over v in {0,1}, shape (2,)

    Update (sum-product; max-product replaces logsumexp by max over the OTHER vars):
      mv2f(v->f)(xv) = sum_{f' in F(v)\f} mf2v(f'->v)(xv)        [+ clamp unary]
      mf2v(f->v)(xv) = reduce_{x_other} [ logtab(x) + sum_{u in vars(f)\v} mv2f(u->f)(xu) ]
    Beliefs:
      b_v(xv) propto sum_{f in F(v)} mf2v(f->v)(xv)               [+ clamp unary]

    Clamping is enforced by a hard unary on v: log u_v(xv) = 0 if xv==clamp else -1e9,
    added wherever v's incoming-message sum is formed (both mv2f source and belief).

    Returns:
      marg_plus : np.array len n.  sum-product -> P(x_v=1); max-product -> argmax
                  belief hardened to {0,1} (the per-node MAP decision).
      margin    : np.array len n.  |b(1)-b(0)| from the NORMALIZED belief (sum-product
                  posterior for 'sum'; normalized max-marginal for 'max').  This is
                  BP's own confidence — the localizability signal.
      converged : bool
      n_used    : iterations run
    """
    n = fg.n
    factors = fg.factors
    vf = fg.var_factors()
    NEG = -1e9

    # hard clamp unary per variable (shape (2,))
    clamp_unary = [np.zeros(2) for _ in range(n)]
    for v, val in fg.clamped.items():
        u = np.full(2, NEG)
        u[int(val)] = 0.0
        clamp_unary[v] = u

    # init messages to uniform (log 0)
    mv2f = {}
    mf2v = {}
    for fi, (vs, tab) in enumerate(factors):
        for v in vs:
            mv2f[(v, fi)] = np.zeros(2)
            mf2v[(fi, v)] = np.zeros(2)

    def lse(a, axis):
        amax = np.max(a, axis=axis, keepdims=True)
        # guard -inf-only slices
        amax = np.where(np.isfinite(amax), amax, 0.0)
        out = amax + np.log(np.sum(np.exp(a - amax), axis=axis, keepdims=True))
        return np.squeeze(out, axis=axis)

    reduce_fn = (lambda a, axis: lse(a, axis)) if mode == "sum" \
        else (lambda a, axis: np.max(a, axis=axis))

    def norm_log(m):
        return m - lse(m, axis=0)

    converged = False
    n_used = n_iters
    for it in range(n_iters):
        max_delta = 0.0

        # --- variable -> factor messages ---
        new_mv2f = {}
        for v in range(n):
            incoming = [mf2v[(fi, v)] for fi in vf[v]]
            total = clamp_unary[v].copy()
            for m in incoming:
                total = total + m
            for fi in vf[v]:
                msg = total - mf2v[(fi, v)]      # exclude target factor
                msg = norm_log(msg)
                old = mv2f[(v, fi)]
                damped = damping * old + (1.0 - damping) * msg
                damped = norm_log(damped)
                new_mv2f[(v, fi)] = damped
                d = float(np.max(np.abs(damped - old)))
                if d > max_delta:
                    max_delta = d
        mv2f = new_mv2f

        # --- factor -> variable messages ---
        new_mf2v = {}
        for fi, (vs, tab) in enumerate(factors):
            arity = len(vs)
            for tvi, tv in enumerate(vs):
                # accumulate logtab + incoming var messages from OTHER vars
                acc = tab.copy()  # shape (2,)*arity
                for ovi, ov in enumerate(vs):
                    if ovi == tvi:
                        continue
                    shape = [1] * arity
                    shape[ovi] = 2
                    acc = acc + mv2f[(ov, fi)].reshape(shape)
                # reduce over all axes except tvi
                axes = tuple(a for a in range(arity) if a != tvi)
                if axes:
                    # reduce one axis at a time (logsumexp/max), keeping order
                    red = acc
                    for ax in sorted(axes, reverse=True):
                        red = reduce_fn(red, axis=ax)
                else:
                    red = acc
                red = norm_log(red)
                old = mf2v[(fi, tv)]
                damped = damping * old + (1.0 - damping) * red
                damped = norm_log(damped)
                new_mf2v[(fi, tv)] = damped
                d = float(np.max(np.abs(damped - old)))
                if d > max_delta:
                    max_delta = d
        mf2v = new_mf2v

        if max_delta < tol:
            converged = True
            n_used = it + 1
            break

    # beliefs
    marg_plus = np.zeros(n)
    margin = np.zeros(n)
    for v in range(n):
        logb = clamp_unary[v].copy()
        for fi in vf[v]:
            logb = logb + mf2v[(fi, v)]
        logb = norm_log(logb)
        b = np.exp(logb)
        if mode == "sum":
            marg_plus[v] = b[1]
        else:
            marg_plus[v] = 1.0 if (logb[1] >= logb[0]) else 0.0
        margin[v] = abs(b[1] - b[0])
    return marg_plus, margin, converged, n_used


# --------------------------------------------------------------------------- #
# Cheap baselines: greedy local search + simulated annealing (the honest bar) #
# --------------------------------------------------------------------------- #
def greedy_local_search(fg, x0, max_passes=200):
    """Hill-climb from x0: single-node then pairwise flips that INCREASE score,
    to a local optimum.  Only FREE vars are flipped; clamped stay fixed.

    Returns (x_best, score_best, n_evals).  Never decreases score (selftest-checked).
    """
    free = fg.free
    x = np.array(x0, dtype=np.int64).copy()
    for v, val in fg.clamped.items():
        x[v] = int(val)
    cur = fg.score(x)
    n_evals = 0
    for _ in range(max_passes):
        improved = False
        # single-node flips
        for v in free:
            x[v] ^= 1
            s = fg.score(x)
            n_evals += 1
            if s > cur + 1e-12:
                cur = s
                improved = True
            else:
                x[v] ^= 1  # revert
        if improved:
            continue
        # pairwise flips (escape simple single-flip optima)
        for ii in range(len(free)):
            for jj in range(ii + 1, len(free)):
                a, b = free[ii], free[jj]
                x[a] ^= 1
                x[b] ^= 1
                s = fg.score(x)
                n_evals += 1
                if s > cur + 1e-12:
                    cur = s
                    improved = True
                else:
                    x[a] ^= 1
                    x[b] ^= 1
        if not improved:
            break
    return x, cur, n_evals


def simulated_annealing(fg, x0, sweeps, rng, T0=2.0, T1=0.02):
    """Single-flip Metropolis SA from x0 over the FREE vars.

    Geometric temperature schedule T0 -> T1 over `sweeps` sweeps (one sweep = one
    proposed flip per free var).  Deterministic given rng.  Returns the BEST
    assignment seen (x_best, score_best, n_evals).
    """
    free = fg.free
    x = np.array(x0, dtype=np.int64).copy()
    for v, val in fg.clamped.items():
        x[v] = int(val)
    cur = fg.score(x)
    best_x = x.copy()
    best_s = cur
    n_evals = 0
    F = len(free)
    if F == 0:
        return best_x, best_s, 0
    ratio = (T1 / T0) ** (1.0 / max(1, sweeps - 1))
    T = T0
    for sw in range(sweeps):
        order = free[:]  # deterministic order; randomness only in accept + proposal pick
        # propose a flip per free var (random var each step to decorrelate)
        for _ in range(F):
            v = free[rng.integers(0, F)]
            x[v] ^= 1
            s = fg.score(x)
            n_evals += 1
            d = s - cur
            if d >= 0 or rng.random() < math.exp(d / max(T, 1e-9)):
                cur = s
                if s > best_s:
                    best_s = s
                    best_x = x.copy()
            else:
                x[v] ^= 1  # reject
        T *= ratio
    return best_x, best_s, n_evals


# --------------------------------------------------------------------------- #
# Localizability AUC: does low BP margin predict a wrong BP-MAP node?          #
# --------------------------------------------------------------------------- #
def localizability_auc(margin, bp_map, exact_map, free):
    """AUC of the ranking (low margin) -> (node wrong in BP-MAP), over FREE nodes.

    score for ranking = -margin (low margin = high 'will be wrong' score).
    label = 1 if bp_map[v] != exact_map[v] else 0.
    Returns (auc, n_wrong, n_free).  auc=NaN if all-correct or all-wrong (undefined).
    """
    s = np.array([-margin[v] for v in free])
    y = np.array([1 if bp_map[v] != exact_map[v] else 0 for v in free])
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan"), n_pos, len(y)
    # rank-based AUC (Mann-Whitney), ties handled by average rank
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=np.float64)
    sorted_s = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0  # 1-based average rank
        ranks[order[i:j + 1]] = avg_rank
        i = j + 1
    sum_ranks_pos = ranks[y == 1].sum()
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc), n_pos, len(y)


# =========================================================================== #
# PART B — FLAT 2D ISING as a FactorGraph                                      #
# =========================================================================== #
def _torus_edges(L):
    N = L * L
    edges = set()
    for r in range(L):
        for c in range(L):
            i = r * L + c
            j = r * L + ((c + 1) % L)
            if i != j:
                edges.add((min(i, j), max(i, j)))
            k = ((r + 1) % L) * L + c
            if i != k:
                edges.add((min(i, k), max(i, k)))
    return sorted(edges), N


def ising_factor_graph(L, beta, rng, h_sigma=0.1):
    """Build a FLAT 2D toroidal Ising as a binary FactorGraph (spin-glass coupling).

    Spins s in {-1,+1} <-> binary x in {0,1} via s = 2x-1.
    Edge factor logtab[xi,xj] = J_ij * (2xi-1)*(2xj-1).
    Field unary logtab[xi] = h_i * (2xi-1).
    NO clamping (the marginal/MAP problem is over all spins).
    Glass regime: J ~ N(0, beta).
    """
    edges, N = _torus_edges(L)
    s = np.array([-1.0, 1.0])  # x=0->-1, x=1->+1
    h = rng.normal(0.0, h_sigma, size=N)
    factors = []
    for i in range(N):
        factors.append(((i,), h[i] * s))  # unary field
    for (i, j) in edges:
        Jij = rng.normal(0.0, beta)
        tab = Jij * np.outer(s, s)  # (2,2): [xi,xj]
        factors.append(((i, j), tab))
    return FactorGraph(N, factors, clamped=None)


# =========================================================================== #
# PART C — DAG NOISY-CIRCUIT as a FactorGraph                                  #
#                                                                             #
# We MIRROR the topology of mycelium/circuit_data.py (leveled + caterpillar    #
# deep-skinny construction, _eval_gate, _longest_path_lvl) in pure python so   #
# the probe needs no tinygrad.  Then we SOFTEN each gate into a CPD factor.    #
# =========================================================================== #
_FANIN = {"AND": 2, "OR": 2, "NOT": 1, "XOR": 2}


def _eval_gate(gtype, operands):
    if gtype == "AND":
        return int(operands[0] & operands[1])
    if gtype == "OR":
        return int(operands[0] | operands[1])
    if gtype == "NOT":
        return int(1 - operands[0])
    if gtype == "XOR":
        return int(operands[0] ^ operands[1])
    raise ValueError(f"unknown gate type {gtype!r}")


def _longest_path_lvl(n, operands):
    lvl = [0] * n
    for v in range(n):
        if operands[v]:
            lvl[v] = 1 + max(lvl[o] for o in operands[v])
        else:
            lvl[v] = 0
    return lvl


def generate_circuit(rng, target_D, max_nodes, gate_types=("AND", "OR", "NOT"),
                     max_side_leaves=2, max_resample=400):
    """Generate ONE leveled Boolean-circuit DAG via the caterpillar (deep-skinny)
    construction of circuit_data.generate_skinny_instance, in pure python.

    rng is a np.random.Generator (seeded).  Returns dict with operands, gtypes,
    bits, leaf_bits, lvl, circuit_depth, n, n_leaves, n_gates — or None on failure.

    Caterpillar: 1 root leaf (id 0) + up to max_side_leaves side leaves (ids 1..),
    then a chain of target_D gates each drawing its FIRST operand from the previous
    chain gate (guaranteeing depth) and a SECOND operand from a side leaf or any
    strictly-lower node.  Topological-order invariant: all leaf ids < all gate ids.
    """
    for _ in range(max_resample):
        n_side = max(0, int(max_side_leaves))
        total_n = 1 + n_side + target_D
        if total_n > max_nodes:
            n_side = max(0, max_nodes - 1 - target_D)
            total_n = 1 + n_side + target_D
        if total_n > max_nodes:
            return None
        n_leaves_total = 1 + n_side
        operands = [[] for _ in range(n_leaves_total)]
        gtypes = [None] * n_leaves_total
        bits = [int(rng.integers(0, 2)) for _ in range(n_leaves_total)]

        gate_slots = list(range(1, target_D))
        rng.shuffle(gate_slots)
        side_leaf_for = {}
        for sl_idx in range(n_side):
            if sl_idx < len(gate_slots):
                side_leaf_for[gate_slots[sl_idx]] = 1 + sl_idx

        prev_id = 0
        for chain_pos in range(1, target_D + 1):
            gt = gate_types[int(rng.integers(0, len(gate_types)))]
            fanin = _FANIN[gt]
            gate_id = n_leaves_total + (chain_pos - 1)
            if fanin == 1:
                ops = [prev_id]
            else:
                sl = side_leaf_for.get(chain_pos)
                if sl is not None:
                    ops = [prev_id, sl]
                else:
                    lower_pool = list(range(gate_id))
                    other_pool = [x for x in lower_pool if x != prev_id]
                    if other_pool:
                        ops = [prev_id, int(rng.choice(other_pool))]
                    else:
                        ops = [prev_id, prev_id]
                rng.shuffle(ops)
            operands.append(list(ops))
            gtypes.append(gt)
            bits.append(_eval_gate(gt, [bits[o] for o in ops]))
            prev_id = gate_id

        n = len(operands)
        if n > max_nodes:
            continue
        lvl = _longest_path_lvl(n, operands)
        D = max(lvl)
        if D != target_D:
            continue
        n_leaves = sum(1 for g in gtypes if g is None)
        n_gates = n - n_leaves
        if n_gates == 0:
            continue
        leaf_bits = [bits[v] for v in range(n) if gtypes[v] is None]
        return {
            "n": n, "n_leaves": n_leaves, "n_gates": n_gates,
            "operands": operands, "gtypes": gtypes, "bits": bits,
            "leaf_bits": leaf_bits, "lvl": lvl, "circuit_depth": D,
        }
    return None


def circuit_factor_graph(inst, eps, leaf_prior_eps=None):
    """Soften a circuit DAG instance into a binary noisy-circuit FactorGraph.

    For each gate g with operands o (1 or 2 of them) and gate type gt:
      phi_g(o..., g) = (1-eps)  if  g == gate(o...)  else  eps
      logtab over (operand0[, operand1], child) indexed in that order.
    Leaf v gets a near-deterministic prior pinned to its sampled bit:
      phi_leaf(v) = (1 - leaf_prior_eps) if v==leaf_bit else leaf_prior_eps.
    The OUTPUT node (the unique deepest gate = the last node id, the chain tip) is
    OBSERVED: clamped to its forward-eval bit.

    Returns (FactorGraph, output_node).  MAP/MPE is over the unclamped leaf+gate
    nodes; the clamp forces the inference to RECONSTRUCT a consistent assignment,
    which is where BP-MAP can land in a wrong basin.
    """
    n = inst["n"]
    operands = inst["operands"]
    gtypes = inst["gtypes"]
    bits = inst["bits"]
    if leaf_prior_eps is None:
        leaf_prior_eps = eps  # leaves are also noisy priors (not hard-pinned)
    le = max(min(float(leaf_prior_eps), 0.5 - 1e-9), 1e-9)
    ge = max(min(float(eps), 0.5 - 1e-9), 1e-9)
    log_hi = math.log(1.0 - ge)
    log_lo = math.log(ge)
    factors = []

    # leaf priors (soft unary toward the sampled leaf bit)
    for v in range(n):
        if gtypes[v] is None:
            lb = int(bits[v])
            tab = np.empty(2)
            tab[lb] = math.log(1.0 - le)
            tab[1 - lb] = math.log(le)
            factors.append(((v,), tab))

    # gate CPD factors
    for v in range(n):
        gt = gtypes[v]
        if gt is None:
            continue
        ops = operands[v]
        if len(ops) == 1:
            vs = (ops[0], v)
            tab = np.empty((2, 2))
            for a in (0, 1):
                want = _eval_gate(gt, [a])
                for c in (0, 1):
                    tab[a, c] = log_hi if c == want else log_lo
            factors.append((vs, tab))
        elif len(ops) == 2:
            vs = (ops[0], ops[1], v)
            tab = np.empty((2, 2, 2))
            for a in (0, 1):
                for b in (0, 1):
                    want = _eval_gate(gt, [a, b])
                    for c in (0, 1):
                        tab[a, b, c] = log_hi if c == want else log_lo
            factors.append((vs, tab))
        else:
            raise ValueError(f"unexpected fan-in {len(ops)}")

    # output node = deepest gate = last node id (chain tip)
    out_node = n - 1
    clamped = {out_node: int(bits[out_node])}
    return FactorGraph(n, factors, clamped=clamped), out_node


# =========================================================================== #
# METRICS                                                                      #
# =========================================================================== #
def marginal_gap(p_bp, p_exact, free):
    eps = 1e-12
    pb = np.clip(np.array([p_bp[v] for v in free]), eps, 1 - eps)
    pe = np.clip(np.array([p_exact[v] for v in free]), eps, 1 - eps)
    mae = float(np.mean(np.abs(pb - pe)))
    kl = float(np.mean(pe * np.log(pe / pb) + (1 - pe) * np.log((1 - pe) / (1 - pb))))
    return mae, kl


def map_gap(bp_map, exact_map, free, fg, exact_score):
    ham = float(np.mean([bp_map[v] != exact_map[v] for v in free]))
    # build full bp assignment honoring clamps
    xbp = np.array(bp_map, dtype=np.int64).copy()
    for v, val in fg.clamped.items():
        xbp[v] = int(val)
    bp_score = fg.score(xbp)
    # normalized logprob gap per node (>=0; exact is the max)
    lp_gap = (exact_score - bp_score) / max(1, len(free))
    return ham, float(lp_gap), float(bp_score)


# =========================================================================== #
# SELFTEST — guard the building blocks where they must be EXACT.               #
# =========================================================================== #
def _tree_chain_fg(n, J, h, rng):
    """A simple chain (tree) Ising as a FactorGraph: edges (0,1),(1,2),...  BP is
    EXACT on a tree."""
    s = np.array([-1.0, 1.0])
    factors = []
    for i in range(n):
        factors.append(((i,), h[i] * s))
    for i in range(n - 1):
        factors.append(((i, i + 1), J[i] * np.outer(s, s)))
    return FactorGraph(n, factors)


def selftest():
    print("=" * 74)
    print("SELFTEST: building-block exactness guards")
    print("=" * 74)
    ok = True
    rng = np.random.default_rng(7)

    # (1) BP on a TREE (chain) is EXACT — both marginals (sum) and MAP (max).
    for trial in range(4):
        n = 6
        J = rng.normal(0.0, 0.8, size=n - 1)
        h = rng.normal(0.0, 0.5, size=n)
        fg = _tree_chain_fg(n, J, h, rng)
        p_ex, map_ex, sc_ex = fg_exact(fg)
        p_bp, marg, conv, nit = fg_bp(fg, "sum", 2000, 0.0, 1e-12)
        mae = float(np.max(np.abs(p_bp - p_ex)))
        bp_map, _, _, _ = fg_bp(fg, "max", 2000, 0.0, 1e-12)
        ham = float(np.mean(bp_map != map_ex))
        passed = (mae < 1e-6) and (ham == 0.0)
        ok = ok and passed
        print(f"  tree chain n={n} trial{trial}: sum-MAE={mae:.2e} (exact?) "
              f"max-Hamming={ham:.3f} (exact MAP?) conv={conv} it={nit} "
              f"-> {'OK' if passed else 'FAIL'}")

    # (2) brute-force MPE matches an INDEPENDENT enumeration on a tiny case.
    n = 5
    J = rng.normal(0, 1.0, size=n - 1)
    h = rng.normal(0, 1.0, size=n)
    fg = _tree_chain_fg(n, J, h, rng)
    _, map_ex, sc_ex = fg_exact(fg)
    # independent enumeration
    best_x, best_s = None, -1e18
    for bits in itertools.product([0, 1], repeat=n):
        x = np.array(bits, dtype=np.int64)
        s = fg.score(x)
        if s > best_s:
            best_s, best_x = s, x.copy()
    passed = (np.array_equal(best_x, map_ex)) and (abs(best_s - sc_ex) < 1e-9)
    ok = ok and passed
    print(f"  brute MPE vs independent enum (n={n}): match={np.array_equal(best_x, map_ex)} "
          f"score_diff={abs(best_s - sc_ex):.2e} -> {'OK' if passed else 'FAIL'}")

    # (3) greedy local search NEVER decreases score; returns >= start.
    rng_g = np.random.default_rng(11)
    inst = generate_circuit(rng_g, target_D=6, max_nodes=16)
    assert inst is not None
    fg_c, out = circuit_factor_graph(inst, eps=0.1)
    x0 = np.array([int(rng_g.integers(0, 2)) for _ in range(fg_c.n)], dtype=np.int64)
    for v, val in fg_c.clamped.items():
        x0[v] = int(val)
    s0 = fg_c.score(x0)
    xls, sls, _ = greedy_local_search(fg_c, x0)
    passed = (sls >= s0 - 1e-9)
    ok = ok and passed
    print(f"  local search monotone: start={s0:.4f} -> end={sls:.4f} "
          f"(>= start?) -> {'OK' if passed else 'FAIL'}")

    # (4) SA returns a VALID assignment (clamps honored, binary).
    rng_sa = np.random.default_rng(3)
    xsa, ssa, _ = simulated_annealing(fg_c, x0, sweeps=60, rng=rng_sa)
    valid = set(np.unique(xsa).tolist()).issubset({0, 1})
    clamp_ok = all(int(xsa[v]) == int(val) for v, val in fg_c.clamped.items())
    passed = valid and clamp_ok and len(xsa) == fg_c.n
    ok = ok and passed
    print(f"  SA valid assignment: binary={valid} clamps_honored={clamp_ok} "
          f"len_ok={len(xsa) == fg_c.n} -> {'OK' if passed else 'FAIL'}")

    # (5) DAG soft FG with eps->0 has exact MPE == deterministic forward-eval.
    rng_d = np.random.default_rng(5)
    n_pass = 0
    n_try = 0
    for _ in range(6):
        inst = generate_circuit(rng_d, target_D=int(rng_d.integers(3, 7)),
                                max_nodes=16)
        if inst is None:
            continue
        n_try += 1
        fg_c, out = circuit_factor_graph(inst, eps=1e-4, leaf_prior_eps=1e-4)
        _, map_ex, _ = fg_exact(fg_c)
        # deterministic forward-eval truth = inst["bits"]
        truth = np.array(inst["bits"], dtype=np.int64)
        match = np.array_equal(map_ex, truth)
        n_pass += int(match)
    passed = (n_try >= 3) and (n_pass == n_try)
    ok = ok and passed
    print(f"  DAG eps->0 MPE == forward-eval: {n_pass}/{n_try} "
          f"-> {'OK' if passed else 'FAIL'}")

    # (6) loopy BP on a tiny LOOP (triangle Ising) — sanity that BP runs and the
    #     marginals are finite & in [0,1] (no exactness claim on a loop).
    s = np.array([-1.0, 1.0])
    factors = [((0, 1), 0.6 * np.outer(s, s)),
               ((1, 2), 0.6 * np.outer(s, s)),
               ((0, 2), 0.6 * np.outer(s, s)),
               ((0,), 0.1 * s), ((1,), -0.1 * s), ((2,), 0.05 * s)]
    fg_tri = FactorGraph(3, factors)
    p_bp, marg, conv, nit = fg_bp(fg_tri, "sum", 3000, 0.5, 1e-10)
    passed = bool(np.all(p_bp >= -1e-9) and np.all(p_bp <= 1 + 1e-9)
                  and np.all(np.isfinite(p_bp)))
    ok = ok and passed
    print(f"  loopy BP on triangle finite & in[0,1]: conv={conv} it={nit} "
          f"p={p_bp.round(3)} -> {'OK' if passed else 'FAIL'}")

    print()
    print(f"SELFTEST {'PASSED' if ok else 'FAILED'}")
    print()
    return ok


# =========================================================================== #
# THE PROBE — fill the 2x2 and the cheap-baseline closure                     #
# =========================================================================== #
def run_probe():
    SEED = int(os.environ.get("SEED", "20260620"))
    INSTANCES = int(os.environ.get("INSTANCES", "8"))
    BP_ITERS = int(os.environ.get("BP_ITERS", "3000"))
    BP_DAMP = float(os.environ.get("BP_DAMP", "0.5"))
    BP_TOL = float(os.environ.get("BP_TOL", "1e-9"))
    ISING_SIZES = [int(x) for x in os.environ.get("ISING_SIZES", "3,4").split(",")]
    MAX_NODES = int(os.environ.get("MAX_NODES", "18"))
    SA_SWEEPS = int(os.environ.get("SA_SWEEPS", "200"))

    print("=" * 100)
    print("FRONTIER MAP+DAG SALVAGE PROBE — {marginals,MAP} x {flat Ising, DAG noisy-circuit}")
    print(f"SEED={SEED} INSTANCES/cell={INSTANCES} BP_ITERS={BP_ITERS} DAMP={BP_DAMP} "
          f"TOL={BP_TOL} MAX_NODES={MAX_NODES} SA_SWEEPS={SA_SWEEPS}")
    print("=" * 100)

    rows = []  # each: dict with topology, band, metrics

    # ----------------------------------------------------------------------- #
    # FLAT 2D ISING (spin-glass)                                              #
    # ----------------------------------------------------------------------- #
    ising_betas = [0.3, 0.5, 0.7, 0.9, 1.1, 1.3]
    print("\n--- FLAT 2D ISING (spin-glass; no clamping) ---")
    hdr = (f"{'L':>3} {'beta':>5} | {'mMAE':>7} {'mKL':>7} | "
           f"{'MAPham':>7} {'lpgap':>7} | {'AUC':>6} {'frac_wrong':>10} | "
           f"{'LSclose':>8} {'SAclose':>8} {'resid_ham':>9} | {'BPconv':>6}")
    print(hdr)
    print("-" * len(hdr))
    for L in ISING_SIZES:
        for beta in ising_betas:
            cell = _accumulate_ising_cell(L, beta, INSTANCES, SEED, BP_ITERS,
                                          BP_DAMP, BP_TOL, SA_SWEEPS)
            rows.append(cell)
            print(_fmt_row(cell))
        print("-" * len(hdr))

    # ----------------------------------------------------------------------- #
    # DAG NOISY-CIRCUIT                                                       #
    # ----------------------------------------------------------------------- #
    print("\n--- DAG NOISY-CIRCUIT (output node clamped; MPE reconstructs) ---")
    print(hdr)
    print("-" * len(hdr))
    # band = (target_depth, eps).  Depth spans shallow->deep; eps spans the noise band.
    dag_depths = [4, 6, 8]
    dag_eps = [0.05, 0.10, 0.15, 0.20]
    for D in dag_depths:
        for eps in dag_eps:
            cell = _accumulate_dag_cell(D, eps, INSTANCES, SEED, MAX_NODES,
                                        BP_ITERS, BP_DAMP, BP_TOL, SA_SWEEPS)
            rows.append(cell)
            print(_fmt_row(cell))
        print("-" * len(hdr))

    verdict(rows)
    return rows


def _fmt_row(c):
    auc = c["auc"]
    auc_s = f"{auc:.3f}" if not math.isnan(auc) else "  nan"
    label = c["label"]
    return (f"{label:>9} | {c['mmae']:>7.4f} {c['mkl']:>7.4f} | "
            f"{c['mapham']:>7.3f} {c['lpgap']:>7.4f} | {auc_s:>6} "
            f"{c['frac_wrong']:>10.3f} | {c['ls_close']:>8.3f} {c['sa_close']:>8.3f} "
            f"{c['resid_ham']:>9.3f} | {100*c['bpconv']:>5.0f}%")


def _accumulate_ising_cell(L, beta, INSTANCES, SEED, BP_ITERS, BP_DAMP, BP_TOL,
                           SA_SWEEPS):
    rng = np.random.default_rng(SEED + L * 1000 + int(beta * 100))
    acc = _new_acc()
    acc["label"] = f"I L{L} b{beta:.1f}"
    acc["topology"] = "flat"
    acc["band"] = f"L{L}_b{beta:.2f}"
    for inst_i in range(INSTANCES):
        fg = ising_factor_graph(L, beta, rng, h_sigma=0.1)
        _eval_one(fg, acc, rng, SEED, inst_i, BP_ITERS, BP_DAMP, BP_TOL, SA_SWEEPS)
    return _finalize_acc(acc)


def _accumulate_dag_cell(D, eps, INSTANCES, SEED, MAX_NODES, BP_ITERS, BP_DAMP,
                         BP_TOL, SA_SWEEPS):
    rng = np.random.default_rng(SEED + D * 7919 + int(eps * 1000))
    acc = _new_acc()
    acc["label"] = f"D{D} e{eps:.2f}"
    acc["topology"] = "dag"
    acc["band"] = f"D{D}_eps{eps:.2f}"
    made = 0
    tries = 0
    while made < INSTANCES and tries < INSTANCES * 50 + 50:
        tries += 1
        inst = generate_circuit(rng, target_D=D, max_nodes=MAX_NODES)
        if inst is None:
            continue
        fg, out = circuit_factor_graph(inst, eps=eps)
        # free count must be brute-forceable
        if len(fg.free) > 20:
            continue
        _eval_one(fg, acc, rng, SEED, made, BP_ITERS, BP_DAMP, BP_TOL, SA_SWEEPS)
        made += 1
    acc["n_made"] = made
    return _finalize_acc(acc)


def _new_acc():
    return {"mmae": [], "mkl": [], "mapham": [], "lpgap": [], "auc": [],
            "frac_wrong": [], "ls_close": [], "sa_close": [], "resid_ham": [],
            "bpconv": [], "ls_evals": [], "sa_evals": [], "n_free": []}


def _eval_one(fg, acc, rng, SEED, inst_i, BP_ITERS, BP_DAMP, BP_TOL, SA_SWEEPS):
    free = fg.free
    p_ex, map_ex, sc_ex = fg_exact(fg)

    # sum-product marginals
    p_bp, _, conv_sum, _ = fg_bp(fg, "sum", BP_ITERS, BP_DAMP, BP_TOL)
    mmae, mkl = marginal_gap(p_bp, p_ex, free)

    # max-product MAP + margins
    bp_map, margin, conv_max, _ = fg_bp(fg, "max", BP_ITERS, BP_DAMP, BP_TOL)
    ham, lpgap, bp_score = map_gap(bp_map, map_ex, free, fg, sc_ex)

    # localizability: low margin -> wrong BP-MAP node
    auc, n_wrong, n_free = localizability_auc(margin, bp_map, map_ex, free)

    # cheap baselines from BP-MAP
    xbp = np.array(bp_map, dtype=np.int64).copy()
    for v, val in fg.clamped.items():
        xbp[v] = int(val)
    xls, sls, ls_evals = greedy_local_search(fg, xbp)
    sa_rng = np.random.default_rng(SEED * 31 + inst_i * 101 + 17)
    xsa, ssa, sa_evals = simulated_annealing(fg, xbp, SA_SWEEPS, sa_rng)
    best_cheap = max(sls, ssa)
    x_best_cheap = xls if sls >= ssa else xsa

    # closure fractions of the BP->exact logprob gap
    gap0 = sc_ex - bp_score
    if gap0 > 1e-9:
        ls_close = (sls - bp_score) / gap0
        sa_close = (ssa - bp_score) / gap0
    else:
        ls_close = 1.0  # no gap to close
        sa_close = 1.0
    ls_close = float(np.clip(ls_close, 0.0, 1.0))
    sa_close = float(np.clip(sa_close, 0.0, 1.0))

    # residual Hamming AFTER the best cheap baseline (vs exact MAP), over free
    resid_ham = float(np.mean([x_best_cheap[v] != map_ex[v] for v in free]))

    acc["mmae"].append(mmae)
    acc["mkl"].append(mkl)
    acc["mapham"].append(ham)
    acc["lpgap"].append(lpgap)
    if not math.isnan(auc):
        acc["auc"].append(auc)
    acc["frac_wrong"].append(n_wrong / max(1, n_free))
    acc["ls_close"].append(ls_close)
    acc["sa_close"].append(sa_close)
    acc["resid_ham"].append(resid_ham)
    acc["bpconv"].append(1.0 if conv_max else 0.0)
    acc["ls_evals"].append(ls_evals)
    acc["sa_evals"].append(sa_evals)
    acc["n_free"].append(len(free))


def _finalize_acc(acc):
    def m(k):
        return float(np.mean(acc[k])) if acc[k] else float("nan")
    out = {
        "label": acc["label"], "topology": acc["topology"], "band": acc["band"],
        "mmae": m("mmae"), "mkl": m("mkl"), "mapham": m("mapham"),
        "lpgap": m("lpgap"), "auc": m("auc"), "frac_wrong": m("frac_wrong"),
        "ls_close": m("ls_close"), "sa_close": m("sa_close"),
        "resid_ham": m("resid_ham"), "bpconv": m("bpconv"),
        "ls_evals": m("ls_evals"), "sa_evals": m("sa_evals"),
        "n_free": m("n_free"), "n_inst": len(acc["mmae"]),
    }
    return out


# =========================================================================== #
# VERDICT                                                                      #
# =========================================================================== #
def verdict(rows):
    print()
    print("=" * 100)
    print("VERDICT")
    print("=" * 100)

    flat = [r for r in rows if r["topology"] == "flat"]
    dag = [r for r in rows if r["topology"] == "dag"]

    AUC_BAR = 0.65       # localizability bar: BP confidence predicts wrong nodes
    GAP_BAR = 0.10       # a "real" MAP Hamming gap
    CLOSE_BAR = 0.80     # cheap baselines closing >= this fraction = "trivially closed"
    RESID_BAR = 0.05     # residual Hamming AFTER cheap = a "real residual"

    def summarize(name, group):
        if not group:
            print(f"[{name}] (no cells)")
            return None
        big = [r for r in group if r["mapham"] >= GAP_BAR]
        # cells with a real gap, localizable, AND cheap leaves a residual = SALVAGE candidates
        salv = [r for r in big
                if (not math.isnan(r["auc"]) and r["auc"] >= AUC_BAR)
                and (max(r["ls_close"], r["sa_close"]) < CLOSE_BAR)
                and (r["resid_ham"] >= RESID_BAR)]
        best_gap = max(group, key=lambda r: r["mapham"])
        # how much does cheap close, on the big-gap cells?
        if big:
            mean_ls = float(np.mean([r["ls_close"] for r in big]))
            mean_sa = float(np.mean([r["sa_close"] for r in big]))
            mean_best_close = float(np.mean([max(r["ls_close"], r["sa_close"]) for r in big]))
            mean_resid = float(np.mean([r["resid_ham"] for r in big]))
            mean_auc = float(np.nanmean([r["auc"] for r in big]))
        else:
            mean_ls = mean_sa = mean_best_close = mean_resid = mean_auc = float("nan")
        print(f"[{name}] cells={len(group)}  big-gap(MAPham>={GAP_BAR})={len(big)}  "
              f"SALVAGE-candidate cells={len(salv)}")
        print(f"        peak MAP gap: {best_gap['label']}  MAPham={best_gap['mapham']:.3f} "
              f"lpgap={best_gap['lpgap']:.4f} AUC={best_gap['auc']:.3f} "
              f"mMAE={best_gap['mmae']:.4f} BPconv={100*best_gap['bpconv']:.0f}%")
        print(f"        on big-gap cells: mean AUC={mean_auc:.3f}  "
              f"mean LS-close={mean_ls:.3f}  mean SA-close={mean_sa:.3f}  "
              f"mean best-close={mean_best_close:.3f}  mean resid-Ham={mean_resid:.3f}")
        return {"big": big, "salv": salv, "best_gap": best_gap,
                "mean_best_close": mean_best_close, "mean_resid": mean_resid,
                "mean_auc": mean_auc}

    print()
    fs = summarize("FLAT 2D Ising", flat)
    print()
    ds = summarize("DAG noisy-circuit", dag)

    # ---- marginal-side sanity (gate-1's KILL): is the marginal gap thin? ----
    print()
    flat_mmae = float(np.mean([r["mmae"] for r in flat])) if flat else float("nan")
    dag_mmae = float(np.mean([r["mmae"] for r in dag])) if dag else float("nan")
    flat_map = float(np.mean([r["mapham"] for r in flat])) if flat else float("nan")
    dag_map = float(np.mean([r["mapham"] for r in dag])) if dag else float("nan")
    print(f"[MARGINALS vs MAP] mean marginal-MAE  flat={flat_mmae:.4f}  dag={dag_mmae:.4f}")
    print(f"                   mean MAP-Hamming    flat={flat_map:.4f}  dag={dag_map:.4f}")

    # ---- flat vs dag contrast ----
    print()
    if fs and ds:
        more_structured = (ds["mean_auc"] > fs["mean_auc"] + 0.05) if (
            not math.isnan(ds["mean_auc"]) and not math.isnan(fs["mean_auc"])) else False
        more_residual = (ds["mean_resid"] > fs["mean_resid"] + 0.02) if (
            not math.isnan(ds["mean_resid"]) and not math.isnan(fs["mean_resid"])) else False
        print(f"[FLAT vs DAG] DAG more localizable (AUC)? {more_structured}  "
              f"(dag {ds['mean_auc']:.3f} vs flat {fs['mean_auc']:.3f})")
        print(f"              DAG bigger residual-after-cheap? {more_residual}  "
              f"(dag {ds['mean_resid']:.3f} vs flat {fs['mean_resid']:.3f})")

    # ---- the verdict ----
    print()
    flat_salv = len(fs["salv"]) if fs else 0
    dag_salv = len(ds["salv"]) if ds else 0
    # cheap-baseline closure is the load-bearing bar
    flat_cheap_wins = (fs is not None and not math.isnan(fs["mean_best_close"])
                       and fs["mean_best_close"] >= CLOSE_BAR)
    dag_cheap_wins = (ds is not None and not math.isnan(ds["mean_best_close"])
                      and ds["mean_best_close"] >= CLOSE_BAR)

    if dag_salv >= 2 and not dag_cheap_wins:
        print(">>> PROVISIONAL: SALVAGE-DAG.  On the DAG noisy-circuit there is a "
              "regime with a LARGE MAP gap, BP confidence LOCALIZES the wrong nodes "
              f"(AUC>={AUC_BAR}), AND cheap local-search/annealing leave a real "
              "residual (do NOT close most of the gap).  This is the learned-MAP-on-"
              "DAG frontier.")
        v = "SALVAGE-DAG"
    elif flat_salv >= 2 and dag_salv >= 2:
        print(">>> PROVISIONAL: SALVAGE-MAP (anywhere).  Both topologies show a "
              "large, localizable, cheap-irreducible MAP gap.")
        v = "SALVAGE-MAP"
    elif (flat_cheap_wins and dag_cheap_wins):
        print(">>> PROVISIONAL: KILL.  Cheap local-search/annealing TRIVIALLY close "
              f"the MAP gap on BOTH topologies (mean best-close >= {CLOSE_BAR}).  "
              "The clean-CSP lesson recurs: cheap methods win, no room for a learned "
              "solver.  Soft-MRF buried.")
        v = "KILL"
    elif (flat_salv == 0 and dag_salv == 0):
        print(">>> PROVISIONAL: KILL.  No cell anywhere clears ALL THREE bars "
              "(large gap + localizable + cheap-irreducible).  Either the MAP gap is "
              "small, or unlocalizable, or cheap baselines close it.  Soft-MRF buried.")
        v = "KILL"
    else:
        print(">>> PROVISIONAL: AMBIGUOUS.  Mixed signals across the bars; see the "
              "per-cell table and the flat-vs-dag contrast above.")
        v = "AMBIGUOUS"
    print(f"    (flat SALVAGE-cells={flat_salv}, dag SALVAGE-cells={dag_salv}, "
          f"flat cheap-wins={flat_cheap_wins}, dag cheap-wins={dag_cheap_wins})")
    return v


# =========================================================================== #
def main():
    if os.environ.get("SELFTEST_ONLY", "0") == "1":
        ok = selftest()
        sys.exit(0 if ok else 1)
    ok = selftest()
    if not ok:
        print("ABORT: selftest failed; not running the probe (baseline suspect).")
        sys.exit(1)
    run_probe()


if __name__ == "__main__":
    main()
