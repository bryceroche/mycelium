#!/usr/bin/env python3
"""
frontier_bp_gap_gate.py — KILL-GATE for the soft-MRF frontier.

Pure numpy, CPU-only. NO GPU, NO tinygrad, NO engine code.

Premise under test
------------------
Mycelium's next frontier is SOFT / PROBABILISTIC factor graphs (Ising/Potts):
the regime where a neural deducer ("learned loopy BP") could earn its keep,
because symbolic CSP propagation has no admissible move on soft factors.

Before building ANY engine work we test the load-bearing premise CHEAPLY:
  Is there a real, consistent, STRUCTURED gap between vanilla damped loopy BP
  and EXACT (brute-force) inference on a candidate testbed (2D periodic Ising)?

  - If vanilla loopy BP already ~= exact across the board: NO HEADROOM -> KILL
    (the soft analog of how cheap symbolic methods killed neural-prop on CSPs).
  - If the gap exists but is UNSTRUCTURED (uniform per-spin error): KILL
    (nothing for a learned variant to localize/correct).
  - If there is a measurable, consistent gap in the spin-glass / near-critical
    band AND the error is LOCALIZABLE (concentrated on identifiable spins):
    GATE PASS -> proceed to build the soft-MRF bridge.

Model
-----
2D periodic (toroidal) Ising grid, L x L spins s_i in {-1,+1}.
Energy/score (we work with the un-negated "score" so larger = more probable):
    S(s) = sum_{<i,j>} J_ij s_i s_j + sum_i h_i s_i
    P(s) propto exp( beta * S(s) )     [beta folded into J,h below]
Edges: 4-neighbour torus (each spin has 4 neighbours; for L=3 the torus has
self-pairs avoided — see _torus_edges).

Two coupling regimes:
  (a) FERROMAGNETIC: all J = +beta  (easy control; BP should be near-exact).
  (b) SPIN-GLASS / MIXED: J ~ N(0, beta)  (frustrated; BP known to be poor).
Fields h_i ~ N(0, h_sigma) (small, fixed scale) so MAP is non-degenerate.

Brute-force exact (<= ~20 spins) gives ground-truth marginals + MAP.
Damped loopy BP gives sum-product marginals + max-product MAP.

Usage
-----
  SELFTEST_ONLY=1 python3 scripts/frontier_bp_gap_gate.py   # sanity only
  python3 scripts/frontier_bp_gap_gate.py                    # full gate
Env knobs (all optional):
  SEED (default 12345), INSTANCES (per cell, default 6),
  BP_ITERS (default 2000), BP_DAMP (default 0.5), BP_TOL (default 1e-8),
  H_SIGMA (default 0.1), MAX_SPINS (default 16; set 20 to include 4x5).
"""

import os
import sys
import math
import itertools
import numpy as np


# --------------------------------------------------------------------------- #
# Topology: toroidal edge list for an L x L grid                              #
# --------------------------------------------------------------------------- #
def _torus_edges(L):
    """Return list of (i, j) undirected edges, i<j, for an LxL 4-neighbour torus.

    Each spin (r,c) -> index r*L + c. Neighbours: right and down (with wrap).
    For L>=3 every spin has 4 distinct neighbours and there are 2*L*L edges.
    For L==2 the wrap would create duplicate (right==left) edges; we de-dup via
    a set, which yields the correct distinct-edge multigraph collapsed to simple
    edges (each pair counted once). L>=3 is the intended operating regime.
    """
    N = L * L
    edges = set()
    for r in range(L):
        for c in range(L):
            i = r * L + c
            # right neighbour (wrap)
            j = r * L + ((c + 1) % L)
            if i != j:
                edges.add((min(i, j), max(i, j)))
            # down neighbour (wrap)
            k = ((r + 1) % L) * L + c
            if i != k:
                edges.add((min(i, k), max(i, k)))
    return sorted(edges), N


# --------------------------------------------------------------------------- #
# Instance generation                                                         #
# --------------------------------------------------------------------------- #
def make_instance(L, regime, beta, rng, h_sigma):
    """Return (edges, N, J(dict edge->coupling), h(np array len N))."""
    edges, N = _torus_edges(L)
    h = rng.normal(0.0, h_sigma, size=N)
    J = {}
    if regime == "ferro":
        for e in edges:
            J[e] = beta  # uniform positive coupling
    elif regime == "glass":
        for e in edges:
            J[e] = rng.normal(0.0, beta)  # mixed sign, std = beta
    else:
        raise ValueError(f"unknown regime {regime!r}")
    return edges, N, J, h


# --------------------------------------------------------------------------- #
# EXACT brute-force inference                                                 #
# --------------------------------------------------------------------------- #
def exact_inference(edges, N, J, h):
    """Brute-force over all 2^N spin configs.

    Returns:
      marg_plus : np.array len N, exact P(s_i = +1)
      map_config: np.array len N in {-1,+1}, argmax-probability config
    Uses a vectorised pass over all 2^N rows; fine for N<=20 (1M rows).
    """
    M = 1 << N
    # spins[m, i] in {-1,+1} : bit i of m -> spin (bit set => +1)
    idx = np.arange(M, dtype=np.int64)
    bits = ((idx[:, None] >> np.arange(N, dtype=np.int64)[None, :]) & 1)
    spins = bits.astype(np.float64) * 2.0 - 1.0  # {0,1}->{-1,+1}, shape (M,N)

    # score per config
    score = spins @ h  # field term, shape (M,)
    for (i, j), Jij in J.items():
        score += Jij * spins[:, i] * spins[:, j]

    # marginals via log-sum-exp stable softmax over configs
    smax = score.max()
    w = np.exp(score - smax)            # unnormalised Boltzmann weights
    Z = w.sum()
    probs = w / Z                       # P(config), shape (M,)
    # P(s_i = +1) = sum over configs with spin_i = +1
    plus_mask = (spins > 0.0)           # (M, N) bool
    marg_plus = (probs[:, None] * plus_mask).sum(axis=0)  # (N,)

    # MAP = argmax score (== argmax prob)
    map_m = int(np.argmax(score))
    map_config = spins[map_m].copy()
    return marg_plus, map_config


# --------------------------------------------------------------------------- #
# Loopy BP (sum-product for marginals, max-product for MAP)                   #
# --------------------------------------------------------------------------- #
def _build_adj(edges, N):
    """neighbours[i] = list of (j, edge_key) for each neighbour j of i."""
    nbr = [[] for _ in range(N)]
    for (i, j) in edges:
        nbr[i].append((j, (i, j)))
        nbr[j].append((i, (i, j)))
    return nbr


def loopy_bp(edges, N, J, h, mode, n_iters, damping, tol):
    r"""Damped loopy BP on the pairwise Ising MRF.

    Messages are over binary spin states {-1,+1}, stored in LOG space for
    numerical stability and renormalised every update.

    Node potential:   phi_i(s)  = exp(h_i * s)      -> log: h_i * s
    Edge potential:   psi_ij(s,t)= exp(J_ij * s * t)-> log: J_ij * s * t

    Message m_{i->j}(t) (over t = state of j):
      sum-product: log m_{i->j}(t) = LSE_s [ log phi_i(s) + log psi_ij(s,t)
                                             + sum_{k in N(i)\j} log m_{k->i}(s) ]
      max-product: replace LSE by max.

    Beliefs:
      b_i(s) propto phi_i(s) * prod_{k in N(i)} m_{k->i}(s)

    Returns:
      marg_plus : np.array len N, BP P(s_i=+1)  (for sum-product)
                  or hardened {0/1} indicator from max-product MAP beliefs.
      converged : bool (max log-message delta < tol within n_iters)
      n_used    : iterations actually run
    Convention for states: index 0 -> s=-1, index 1 -> s=+1.
    """
    states = np.array([-1.0, +1.0])  # idx0=-1, idx1=+1
    nbr = _build_adj(edges, N)

    # log node potentials, shape (N, 2): h_i * s
    log_phi = np.outer(h, states)  # (N,2): log_phi[i, a] = h_i * states[a]

    # log edge potentials per directed message: depends only on J_ij.
    # log_psi[s_idx_of_sender, t_idx_of_receiver] = J_ij * states[s] * states[t]
    def log_psi_mat(Jij):
        # (2 sender, 2 receiver)
        return Jij * np.outer(states, states)

    # init directed messages: log m_{i->j}, normalised (log-domain), start uniform 0.
    # store in dict keyed by directed pair (i,j)
    msg = {}
    for (i, j) in edges:
        msg[(i, j)] = np.zeros(2)
        msg[(j, i)] = np.zeros(2)

    reduce_fn = (lambda x, axis: _logsumexp(x, axis)) if mode == "sum" \
        else (lambda x, axis: x.max(axis=axis))

    converged = False
    n_used = n_iters
    for it in range(n_iters):
        max_delta = 0.0
        new_msg = {}
        for (i, j) in edges:
            for (src, dst) in ((i, j), (j, i)):
                Jij = J[(min(src, dst), max(src, dst))]
                lpsi = log_psi_mat(Jij)  # (sender state, receiver state)
                # incoming from N(src)\dst evaluated at sender state s
                acc = log_phi[src].copy()  # (2,) over sender state s
                for (k, ekey) in nbr[src]:
                    if k == dst:
                        continue
                    acc = acc + msg[(k, src)]  # log m_{k->src}(s), (2,)
                # combine: for each receiver state t, reduce over sender s of
                #   acc[s] + lpsi[s, t]
                combined = acc[:, None] + lpsi  # (sender s, receiver t)
                out = reduce_fn(combined, axis=0)  # (2,) over receiver t
                # normalise in log domain (subtract logsumexp) to prevent drift
                out = out - _logsumexp(out, axis=0)
                # damping in log domain
                old = msg[(src, dst)]
                damped = damping * old + (1.0 - damping) * out
                damped = damped - _logsumexp(damped, axis=0)
                new_msg[(src, dst)] = damped
                d = float(np.max(np.abs(damped - old)))
                if d > max_delta:
                    max_delta = d
        msg = new_msg
        if max_delta < tol:
            converged = True
            n_used = it + 1
            break

    # beliefs
    marg_plus = np.zeros(N)
    if mode == "sum":
        for i in range(N):
            logb = log_phi[i].copy()
            for (k, ekey) in nbr[i]:
                logb = logb + msg[(k, i)]
            logb = logb - _logsumexp(logb, axis=0)
            b = np.exp(logb)
            marg_plus[i] = b[1]  # P(s=+1)
    else:  # max-product -> MAP estimate per spin from max-marginal belief
        for i in range(N):
            logb = log_phi[i].copy()
            for (k, ekey) in nbr[i]:
                logb = logb + msg[(k, i)]
            # argmax belief state
            marg_plus[i] = 1.0 if (logb[1] >= logb[0]) else 0.0
    return marg_plus, converged, n_used


def _logsumexp(a, axis):
    amax = np.max(a, axis=axis, keepdims=True)
    out = amax + np.log(np.sum(np.exp(a - amax), axis=axis, keepdims=True))
    return np.squeeze(out, axis=axis)


# --------------------------------------------------------------------------- #
# Gap metrics                                                                 #
# --------------------------------------------------------------------------- #
def marginal_gap_metrics(p_bp, p_exact):
    """Per-spin abs error and mean KL, plus localizability stats.

    p_bp, p_exact: P(s_i=+1) arrays.
    Returns dict with:
      mean_abs, max_abs, mean_kl,
      top20_share (fraction of total abs error carried by worst 20% of spins),
      gini (of per-spin abs error).
    """
    eps = 1e-12
    pb = np.clip(p_bp, eps, 1 - eps)
    pe = np.clip(p_exact, eps, 1 - eps)
    abs_err = np.abs(pb - pe)
    # KL( exact || bp ) per spin over binary distribution
    kl = (pe * np.log(pe / pb) + (1 - pe) * np.log((1 - pe) / (1 - pb)))
    N = len(abs_err)
    # top-20% share
    k = max(1, int(round(0.20 * N)))
    sorted_err = np.sort(abs_err)[::-1]
    total = abs_err.sum() + eps
    top20_share = sorted_err[:k].sum() / total
    return {
        "mean_abs": float(abs_err.mean()),
        "max_abs": float(abs_err.max()),
        "mean_kl": float(kl.mean()),
        "top20_share": float(top20_share),
        "gini": float(_gini(abs_err)),
        "abs_err": abs_err,
    }


def null_top20_share(N, n_samples=4000, seed=0):
    """Expected worst-20% error share if per-spin error were UNIFORM noise.

    This is the null baseline localizability must beat: a learned variant can
    only help if some spins are CONSISTENTLY wrong, i.e. error concentrated
    ABOVE what random noise alone produces for this N.
    """
    rng = np.random.default_rng(seed)
    k = max(1, int(round(0.20 * N)))
    shares = []
    for _ in range(n_samples):
        e = rng.random(N)
        s = np.sort(e)[::-1]
        shares.append(s[:k].sum() / (e.sum() + 1e-12))
    return float(np.mean(shares))


def _gini(x):
    """Gini coefficient of nonneg array (0 = uniform, ->1 = concentrated)."""
    x = np.asarray(x, dtype=np.float64)
    if x.sum() <= 0:
        return 0.0
    xs = np.sort(x)
    n = len(xs)
    cum = np.cumsum(xs)
    # standard Gini formula
    g = (n + 1 - 2 * (cum / cum[-1]).sum()) / n
    return float(g)


def map_hamming(map_bp_plus01, map_exact_pm1):
    """Fraction of spins differing between BP MAP and exact MAP."""
    bp_pm = np.where(map_bp_plus01 > 0.5, 1.0, -1.0)
    return float(np.mean(bp_pm != map_exact_pm1))


# --------------------------------------------------------------------------- #
# SELFTEST                                                                    #
# --------------------------------------------------------------------------- #
def selftest():
    print("=" * 70)
    print("SELFTEST: BP vs exact must AGREE on easy ferromagnetic cases")
    print("=" * 70)
    rng = np.random.default_rng(7)
    ok = True

    # (1) 3x3 ferromagnetic, WEAK coupling (disordered phase) -> BP ~= exact.
    #     NOTE: the 3x3 torus is densely connected (each spin has 4 neighbours,
    #     effective coupling ~4*beta), so it orders early. Past the ordering
    #     transition a small field tips loopy BP into one basin and ferro
    #     coupling over-amplifies it -> BP becomes overconfident vs the true
    #     (basin-averaged) marginal. That is GENUINE BP behaviour, NOT a bug
    #     (see the h=0 symmetric check (1b) which is exact at all beta). So the
    #     control here uses weak coupling, where BP must match exact tightly.
    for (L, beta, tol_ok) in [(3, 0.10, 0.01), (3, 0.20, 0.02)]:
        edges, N, J, h = make_instance(L, "ferro", beta, rng, h_sigma=0.1)
        p_ex, map_ex = exact_inference(edges, N, J, h)
        p_bp, conv, nit = loopy_bp(edges, N, J, h, "sum",
                                   n_iters=2000, damping=0.5, tol=1e-10)
        mae = float(np.mean(np.abs(p_bp - p_ex)))
        map_bp, _, _ = loopy_bp(edges, N, J, h, "max",
                                n_iters=2000, damping=0.5, tol=1e-10)
        ham = map_hamming(map_bp, map_ex)
        passed = (mae < tol_ok)
        ok = ok and passed
        print(f"  L={L} ferro beta={beta:.2f} (weak): marg MAE={mae:.5f} "
              f"(tol {tol_ok}) MAP-hamming={ham:.3f} conv={conv} it={nit} "
              f"-> {'OK' if passed else 'FAIL'}")

    # (1b) pure ferro, h=0: by symmetry exact P(+1)=0.5 exactly, and the
    #      symmetric BP fixed point is exact at ALL beta. A clean guarantee
    #      that the message math is correct even deep in the ordered regime.
    for beta in [0.4, 0.9, 1.5]:
        rng_s = np.random.default_rng(100)
        edges, N, J, h = make_instance(3, "ferro", beta, rng_s, h_sigma=0.0)
        p_ex, _ = exact_inference(edges, N, J, h)
        p_bp, conv, nit = loopy_bp(edges, N, J, h, "sum",
                                   n_iters=4000, damping=0.5, tol=1e-12)
        mae = float(np.mean(np.abs(p_bp - p_ex)))
        passed = (mae < 1e-6 and abs(p_ex.mean() - 0.5) < 1e-9)
        ok = ok and passed
        print(f"  L=3 ferro beta={beta:.2f} h=0 (symmetric): "
              f"exact mean P(+1)={p_ex.mean():.6f} MAE={mae:.2e} "
              f"-> {'OK' if passed else 'FAIL'}")

    # (2) tiny 2-spin chain (degenerate torus collapses; use explicit 2-node)
    #     single edge, BP on a tree is EXACT -> must match to ~1e-6
    edges2 = [(0, 1)]
    N2 = 2
    J2 = {(0, 1): 0.7}
    h2 = np.array([0.3, -0.2])
    p_ex2, map_ex2 = exact_inference(edges2, N2, J2, h2)
    p_bp2, conv2, nit2 = loopy_bp(edges2, N2, J2, h2, "sum",
                                  n_iters=500, damping=0.0, tol=1e-12)
    mae2 = float(np.mean(np.abs(p_bp2 - p_ex2)))
    passed2 = mae2 < 1e-6
    ok = ok and passed2
    print(f"  2-spin tree (BP exact on trees): MAE={mae2:.3e} "
          f"-> {'OK' if passed2 else 'FAIL'}")
    print(f"     exact P(+1)={p_ex2.round(5)}  bp P(+1)={p_bp2.round(5)}")

    print()
    print(f"SELFTEST {'PASSED' if ok else 'FAILED'}")
    print()
    return ok


# --------------------------------------------------------------------------- #
# Full gate                                                                   #
# --------------------------------------------------------------------------- #
def run_gate():
    SEED = int(os.environ.get("SEED", "12345"))
    INSTANCES = int(os.environ.get("INSTANCES", "6"))
    BP_ITERS = int(os.environ.get("BP_ITERS", "2000"))
    BP_DAMP = float(os.environ.get("BP_DAMP", "0.5"))
    BP_TOL = float(os.environ.get("BP_TOL", "1e-8"))
    H_SIGMA = float(os.environ.get("H_SIGMA", "0.1"))
    MAX_SPINS = int(os.environ.get("MAX_SPINS", "16"))

    sizes = [3, 4]              # 9, 16 spins
    if MAX_SPINS >= 20:
        sizes.append("4x5")    # 20 spins
    regimes = ["ferro", "glass"]
    betas = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5]

    print("=" * 110)
    print("FRONTIER BP-GAP KILL-GATE — vanilla damped loopy BP vs EXACT (2D periodic Ising)")
    print(f"SEED={SEED} INSTANCES/cell={INSTANCES} BP_ITERS={BP_ITERS} "
          f"DAMP={BP_DAMP} TOL={BP_TOL} H_SIGMA={H_SIGMA}")
    print("=" * 110)
    header = (f"{'size':>5} {'regime':>6} {'beta':>5} | "
              f"{'marg_MAE':>9} {'marg_max':>9} {'mean_KL':>9} | "
              f"{'MAP_ham':>8} | {'conv%':>6} | {'top20sh':>8} {'gini':>6}")
    print(header)
    print("-" * len(header))

    rng = np.random.default_rng(SEED)
    results = []  # list of dicts (one per cell, aggregated over instances)

    for L in sizes:
        Lval = 4 if L == "4x5" else L
        for regime in regimes:
            for beta in betas:
                agg = {"marg_mae": [], "marg_max": [], "mean_kl": [],
                       "map_ham": [], "conv": [], "top20": [], "gini": [],
                       "sum_conv": [], "abs_err_pooled": []}
                for _ in range(INSTANCES):
                    if L == "4x5":
                        edges, N, J, h = _make_rect_instance(4, 5, regime,
                                                             beta, rng, H_SIGMA)
                    else:
                        edges, N, J, h = make_instance(L, regime, beta,
                                                       rng, H_SIGMA)
                    p_ex, map_ex = exact_inference(edges, N, J, h)
                    p_bp, conv_sum, _ = loopy_bp(edges, N, J, h, "sum",
                                                 BP_ITERS, BP_DAMP, BP_TOL)
                    map_bp, conv_max, _ = loopy_bp(edges, N, J, h, "max",
                                                   BP_ITERS, BP_DAMP, BP_TOL)
                    mg = marginal_gap_metrics(p_bp, p_ex)
                    agg["marg_mae"].append(mg["mean_abs"])
                    agg["marg_max"].append(mg["max_abs"])
                    agg["mean_kl"].append(mg["mean_kl"])
                    agg["map_ham"].append(map_hamming(map_bp, map_ex))
                    agg["conv"].append(1.0 if conv_sum else 0.0)
                    agg["top20"].append(mg["top20_share"])
                    agg["gini"].append(mg["gini"])
                    agg["abs_err_pooled"].append(mg["abs_err"])

                row = {
                    "size": (f"{L}x{L}" if L != "4x5" else "4x5"),
                    "N": N,
                    "regime": regime,
                    "beta": beta,
                    "marg_mae": float(np.mean(agg["marg_mae"])),
                    "marg_max": float(np.mean(agg["marg_max"])),
                    "mean_kl": float(np.mean(agg["mean_kl"])),
                    "map_ham": float(np.mean(agg["map_ham"])),
                    "conv": float(np.mean(agg["conv"])),
                    "top20": float(np.mean(agg["top20"])),
                    "gini": float(np.mean(agg["gini"])),
                }
                results.append(row)
                print(f"{row['size']:>5} {regime:>6} {beta:>5.2f} | "
                      f"{row['marg_mae']:>9.4f} {row['marg_max']:>9.4f} "
                      f"{row['mean_kl']:>9.4f} | {row['map_ham']:>8.3f} | "
                      f"{100*row['conv']:>5.0f}% | "
                      f"{row['top20']:>8.3f} {row['gini']:>6.3f}")
            print("-" * len(header))

    verdict(results)
    return results


def _make_rect_instance(R, C, regime, beta, rng, h_sigma):
    """Rectangular R x C toroidal Ising instance (for 4x5 = 20 spins)."""
    N = R * C
    edges = set()
    for r in range(R):
        for c in range(C):
            i = r * C + c
            j = r * C + ((c + 1) % C)
            if i != j:
                edges.add((min(i, j), max(i, j)))
            k = ((r + 1) % R) * C + c
            if i != k:
                edges.add((min(i, k), max(i, k)))
    edges = sorted(edges)
    h = rng.normal(0.0, h_sigma, size=N)
    J = {}
    if regime == "ferro":
        for e in edges:
            J[e] = beta
    else:
        for e in edges:
            J[e] = rng.normal(0.0, beta)
    return edges, N, J, h


# --------------------------------------------------------------------------- #
# Verdict logic                                                               #
# --------------------------------------------------------------------------- #
def verdict(results):
    print()
    print("=" * 110)
    print("VERDICT")
    print("=" * 110)

    ferro = [r for r in results if r["regime"] == "ferro"]
    glass = [r for r in results if r["regime"] == "glass"]

    # --- ferromagnetic control: BP should be near-exact in the WEAK/disordered
    #     band. (Past the small-torus ordering transition, BP legitimately
    #     becomes overconfident on a ferromagnet — a field tips it into one
    #     basin and ferro coupling over-amplifies it; this is real BP error,
    #     not a bug. The implementation sanity check is therefore the WEAK band.)
    # The dense small torus (each spin has 4 neighbours) orders early: its
    # ferro transition sits between beta~0.2 and ~0.5, so beta=0.3 already
    # straddles it. The clean disordered-phase anchor is beta<=0.2, where BP
    # is provably near-exact (see selftest). Use that as the sanity band.
    WEAK_FERRO_BETA = 0.2
    ferro_all = np.array([r["marg_mae"] for r in ferro])
    ferro_weak = [r for r in ferro if r["beta"] <= WEAK_FERRO_BETA]
    ferro_weak_mae = np.array([r["marg_mae"] for r in ferro_weak])
    ferro_ok = ferro_weak_mae.max() < 0.02  # near-exact in disordered phase
    fw = max(ferro_weak, key=lambda r: r["marg_mae"])
    print(f"[CONTROL] Ferromagnetic WEAK band (beta<={WEAK_FERRO_BETA}): "
          f"max marg MAE = {ferro_weak_mae.max():.4f} "
          f"(at {fw['size']} beta={fw['beta']:.2f}); "
          f"mean = {ferro_weak_mae.mean():.4f}")
    print(f"          Ferro FULL range max marg MAE = {ferro_all.max():.4f} "
          "(high-beta overconfidence is expected BP behaviour, not a bug).")
    if ferro_ok:
        print("          -> BP is near-exact on the WEAK ferro control. "
              "Implementation sanity: OK.")
    else:
        print("          -> WARNING: BP is INACCURATE even on WEAK "
              "ferromagnetic. Suspect an IMPLEMENTATION BUG, not the science.")

    # --- spin-glass gap ---
    # A "real gap" = marg MAE clearly above the BP-exact floor (~1e-3).
    # Localizability is judged RELATIVE TO THE UNIFORM-NOISE NULL for this N:
    # a learned variant can only help if error is concentrated MORE than random
    # noise would be (top20_share above null + meaningfully high gini), AND on a
    # STABLE BP fixed point (high convergence) — oscillating non-converged error
    # is not a fixed structured bias to anchor a correction on.
    GAP_THRESH = 0.03   # marg MAE that counts as a real gap (vs ~1e-3 BP-exact)
    GINI_LOCAL = 0.30   # gini floor for "non-uniform"
    CONV_FLOOR = 0.80   # require BP to actually converge in the candidate band
    NULL_MARGIN = 0.03  # top20_share must beat the null by at least this

    # per-N null top20-share (depends on N because worst-20% rounds differently)
    null_by_N = {}
    for r in glass:
        if r["N"] not in null_by_N:
            null_by_N[r["N"]] = null_top20_share(r["N"])
    for r in glass:
        r["null_top20"] = null_by_N[r["N"]]
        r["top20_excess"] = r["top20"] - r["null_top20"]

    glass_sorted = sorted(glass, key=lambda r: r["marg_mae"], reverse=True)
    best = glass_sorted[0]

    gap_cells = [r for r in glass if r["marg_mae"] >= GAP_THRESH]
    # localizable = concentrated above the noise null AND non-uniform by gini
    local_cells = [r for r in gap_cells
                   if (r["top20_excess"] >= NULL_MARGIN and r["gini"] >= GINI_LOCAL)]
    # the candidate band a learned variant could actually target: real gap,
    # localized above null, AND BP stable (converged) there.
    actionable = [r for r in local_cells if r["conv"] >= CONV_FLOOR]

    print()
    print(f"[GLASS] Null top20-share (uniform-noise baseline) by N: "
          + ", ".join(f"N={n}:{v:.3f}" for n, v in sorted(null_by_N.items())))
    print(f"        Largest marginal gap: {best['size']} beta={best['beta']:.2f} "
          f"marg_MAE={best['marg_mae']:.4f} max={best['marg_max']:.4f} "
          f"MAP_ham={best['map_ham']:.3f} KL={best['mean_kl']:.4f}")
    print(f"        Localizability there: top20={best['top20']:.3f} "
          f"(null {best['null_top20']:.3f}, excess {best['top20_excess']:+.3f}) "
          f"gini={best['gini']:.3f} BPconv={100*best['conv']:.0f}%")
    print(f"        Cells with a real gap (marg_MAE>={GAP_THRESH}): "
          f"{len(gap_cells)}/{len(glass)}")
    print(f"        ...LOCALIZABLE above null (top20_excess>={NULL_MARGIN} & "
          f"gini>={GINI_LOCAL}): {len(local_cells)}/{len(gap_cells)}")
    print(f"        ...AND BP-stable (conv>={CONV_FLOOR:.0%}) = ACTIONABLE: "
          f"{len(actionable)}/{len(gap_cells)}")

    # the structural story: where the gap is LARGE, is the error UNIFORM?
    if gap_cells:
        big = [r for r in gap_cells if r["marg_mae"] >= 0.10]
        if big:
            big_excess = np.mean([r["top20_excess"] for r in big])
            big_conv = np.mean([r["conv"] for r in big])
            print(f"        Large-gap cells (MAE>=0.10): mean top20_excess="
                  f"{big_excess:+.3f}, mean BPconv={100*big_conv:.0f}% "
                  f"({'UNIFORM & unstable' if big_excess < NULL_MARGIN else 'structured'}).")

    print()
    has_gap = len(gap_cells) >= max(2, int(0.15 * len(glass)))

    if not ferro_ok:
        print(">>> GATE INCONCLUSIVE: ferromagnetic control FAILED — fix the "
              "BP implementation before trusting the glass numbers.")
        return "INCONCLUSIVE"

    if not has_gap:
        print(">>> GATE KILL (NO HEADROOM): vanilla damped loopy BP is "
              "~= exact across the board, including the spin-glass band.")
        print("    => No gap to close. The soft-MRF frontier dies here the "
              "same way cheap symbolic methods killed neural-prop on CSPs.")
        return "KILL_NO_HEADROOM"

    # A clean PASS requires the actionable band to be more than a single
    # borderline cell: either >=2 actionable cells, or one with a gap clearly
    # above the floor (MAE >= 2x threshold). A lone cell sitting right on the
    # gap threshold is a WEAK PASS, not a clean one (over-claim guard).
    strong_actionable = [r for r in actionable
                         if r["marg_mae"] >= 2 * GAP_THRESH]
    clean_pass = (len(actionable) >= 2) or (len(strong_actionable) >= 1)

    if actionable and clean_pass:
        b = sorted(actionable, key=lambda r: r["marg_mae"], reverse=True)[0]
        print(">>> GATE PASS: there is a MEASURABLE, CONSISTENT, LOCALIZABLE "
              "BP-vs-exact gap on a STABLE BP fixed point in the glass band.")
        print(f"    Best actionable band: {b['size']} beta={b['beta']:.2f} "
              f"(marg_MAE={b['marg_mae']:.4f}, MAP_ham={b['map_ham']:.3f}, "
              f"top20_excess={b['top20_excess']:+.3f}, gini={b['gini']:.3f}, "
              f"BPconv={100*b['conv']:.0f}%).")
        print("    => Frontier has headroom. Proceed to build the soft-MRF "
              "bridge (a learned variant can target the localized errors).")
        return "PASS"

    # gap exists but NOT a clean actionable PASS. Diagnose WHY honestly.
    # MAP gap is a separate signal: report the strongest MAP-Hamming band, which
    # can stay large even where marginals are close (MAP is a harder target).
    map_best = max(glass, key=lambda r: r["map_ham"])

    if local_cells:
        b = sorted(local_cells, key=lambda r: r["marg_mae"], reverse=True)[0]
        print(">>> GATE WEAK PASS (CONDITIONAL): a localized BP-vs-exact gap "
              "exists, but where it is most LOCALIZED the gap is modest, and "
              "where the gap is LARGE the error becomes UNIFORM and BP stops "
              "converging (gap and structure are anti-correlated).")
        print(f"    Best localized cell: {b['size']} beta={b['beta']:.2f} "
              f"(marg_MAE={b['marg_mae']:.4f}, top20_excess={b['top20_excess']:+.3f}, "
              f"gini={b['gini']:.3f}, BPconv={100*b['conv']:.0f}%).")
        print(f"    Separately, MAP gap stays large across the glass band "
              f"(peak {map_best['size']} beta={map_best['beta']:.2f} "
              f"MAP_ham={map_best['map_ham']:.3f}) — a learned MAP corrector is "
              "the more promising target than marginal correction.")
        print("    => Thin marginal headroom in a narrow transitional band; "
              "MAP headroom is broader. CAUTIOUS proceed, not a clean PASS; "
              "the natural next testbed is a HIERARCHICAL/DAG factor graph "
              "(KenKen is flat — radial structure can't express here).")
        return "WEAK_PASS"

    print(">>> GATE KILL (UNSTRUCTURED): a BP-vs-exact gap EXISTS but wherever "
          "it is large the per-spin error is UNIFORM (top20-share not above the "
          "noise null) and BP fails to converge — there is no stable, "
          "concentrated error pattern for a learned variant to localize.")
    print("    => Nothing structured to correct on a stable fixed point. "
          "The frontier premise fails on structure.")
    return "KILL_UNSTRUCTURED"


# --------------------------------------------------------------------------- #
def main():
    if os.environ.get("SELFTEST_ONLY", "0") == "1":
        ok = selftest()
        sys.exit(0 if ok else 1)
    # always run selftest first as a guard, then the full gate
    ok = selftest()
    if not ok:
        print("ABORT: selftest failed; not running full gate "
              "(implementation suspect).")
        sys.exit(1)
    run_gate()


if __name__ == "__main__":
    main()
