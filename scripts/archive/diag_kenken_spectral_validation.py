"""diag_kenken_spectral_validation.py — Does spectral decomposition of the KenKen
factor graph surface REGIONAL (cage) structure, against ground-truth cages?

CONTEXT (CLAUDE.md framing)
---------------------------
A KenKen puzzle is a factor graph over the NxN grid of cells. There are THREE
factor TYPES:
  - ROW factors  : all-different over each grid row     (same i//N)
  - COL factors  : all-different over each grid column   (same i%N)
  - CAGE factors : the arithmetic clusters (cell_cage_id)
The deducer's per-head masks are built from membership^T @ membership > 0 per
type (mycelium/kenken.py: build_kenken_attn_bias + _build_fixed_cell_masks).
The FULL cell~cell adjacency = row OR col OR cage.

This script asks the *spectral* question: if you Laplacian-eigendecompose the
cell~cell graph and k-means the eigenvectors, do the clusters line up with the
CAGES (regional structure), the ROW/COL partition (the dense rook backbone), or
neither (vs a random-partition null)?

PREDICTION UNDER TEST (be honest whether it holds):
  KenKen's full graph is a dense rook's graph (each cell ~ its whole row + whole
  col, ~2(N-1) neighbors) with cages as a weak 1-3-edge perturbation. So:
   - full-graph spectral clustering likely recovers ROW/COL, not cages;
   - cages may need the cage-weighted operator, or only show up in a high band.

OPERATORS
  (1) FULL graph          : row OR col OR cage
  (2) cage-weighted       : full graph with cage edges upweighted by w (sweep w)
  (3) cage-only subgraph  : cage edges only (sanity: components == cages)

GROUND TRUTH
  CAGES   : cell_cage_id (over real cells)
  ROW/COL : grid position (row=idx//N, col=idx%N) of each real cell. The encode
            layout (mycelium/kenken_data.py _rc_to_flat) is r*N_MAX + c with
            N_MAX=7, so on the N_MAX grid row=idx//7, col=idx%7. We restrict to
            real cells (cell_valid==1) and rebuild row/col labels from the actual
            flat index (verified, not assumed).

METRICS  NMI + ARI vs CAGES and vs ROW/COL, plus a RANDOM-PARTITION NULL
         (matched cluster count) to calibrate "no alignment".

BANDS    Eigenvectors split into low / mid / high bands; k-means each band and
         report which ground truth it aligns with best (global=low / regional=mid
         / local=high story).

CPU-ONLY: matrices are <=49x49, scipy.linalg.eigh is instant. No GPU, no tinygrad
forward — pure numpy/scipy/sklearn over the encoded membership.

Run: DEV=CPU .venv/bin/python3 scripts/diag_kenken_spectral_validation.py
"""
from __future__ import annotations

import os
import sys
from collections import defaultdict

import numpy as np
from scipy.linalg import eigh
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mycelium import kenken_data as kd  # noqa: E402

N_MAX = kd.N_MAX  # 7

# Reproducibility for k-means + the random null.
GLOBAL_SEED = 0
RNG = np.random.default_rng(GLOBAL_SEED)

# How many puzzles per N to sample.
PUZZLES_PER_N = int(os.environ.get("KK_SPECTRAL_PER_N", "18"))
NS = [5, 6, 7]
CAGE_WEIGHTS = [1, 3, 10, 30]
DATA_PATH = os.environ.get(
    "KK_SPECTRAL_DATA", ".cache/kenken_test_curriculum.jsonl"
)


# --------------------------------------------------------------------------- #
# Adjacency construction (the three operators) + ground-truth partitions.
# --------------------------------------------------------------------------- #
def build_type_membership(enc: dict):
    """From an encode_puzzle dict, return per-type binary membership matrices over
    REAL cells only, plus the real-cell index list and ground-truth labels.

    Returns:
        real_idx   : (n,) flat indices of real cells on the N_MAX grid.
        M_row      : (n, n_rowfac) cell x row-factor membership (one-hot per cell).
        M_col      : (n, n_colfac) cell x col-factor membership.
        M_cage     : (n, n_cagefac) cell x cage-factor membership.
        cage_lab   : (n,) ground-truth cage label per real cell (cell_cage_id).
        row_lab    : (n,) ground-truth row label (idx // N_MAX), N groups.
        col_lab    : (n,) ground-truth col label (idx % N_MAX), N groups.

    NOTE on row/col ground truth: the JOINT (row,col) label is degenerate — each
    real cell has a UNIQUE (row,col) pair, so it is a label-per-cell partition (n
    groups of size 1). A k<<n clustering can never match it (ARI pinned at 0 even
    vs the data), so it is useless as ground truth. The MEANINGFUL rook-backbone
    ground truths are ROW alone (N groups) and COL alone (N groups); we compare
    against each separately, and cluster the graph at the matched k=N when asking
    "does spectral recover row/col?".
    """
    cell_valid = enc["cell_valid"].astype(bool)          # (49,)
    cell_cage_id = enc["cell_cage_id"]                    # (49,) -1 = pad
    cage_mask = enc["cage_mask"]                          # (49,49) cage clique

    real_idx = np.nonzero(cell_valid)[0]                 # (n,)
    n = real_idx.size

    # ---- ROW / COL membership from the ACTUAL flat layout (verified, not assumed).
    # _rc_to_flat is r*N_MAX + c, so on the N_MAX grid: row = idx//N_MAX, col=idx%N_MAX.
    rows = real_idx // N_MAX                              # grid-row id per real cell
    cols = real_idx % N_MAX                               # grid-col id per real cell

    uniq_rows = np.unique(rows)
    uniq_cols = np.unique(cols)
    row_remap = {r: i for i, r in enumerate(uniq_rows)}
    col_remap = {c: i for i, c in enumerate(uniq_cols)}
    row_lab = np.array([row_remap[r] for r in rows], dtype=np.int64)
    col_lab = np.array([col_remap[c] for c in cols], dtype=np.int64)

    M_row = np.zeros((n, uniq_rows.size), dtype=np.float64)
    M_row[np.arange(n), row_lab] = 1.0
    M_col = np.zeros((n, uniq_cols.size), dtype=np.float64)
    M_col[np.arange(n), col_lab] = 1.0

    # ---- CAGE membership + ground-truth cage label, restricted to real cells.
    cage_ids = cell_cage_id[real_idx]                    # (n,) raw cage ids (>=0)
    uniq_cages = np.unique(cage_ids)
    cage_remap = {c: i for i, c in enumerate(uniq_cages)}
    cage_lab = np.array([cage_remap[c] for c in cage_ids], dtype=np.int64)
    M_cage = np.zeros((n, uniq_cages.size), dtype=np.float64)
    M_cage[np.arange(n), cage_lab] = 1.0

    # Sanity-cross-check the cage membership against cage_mask co-occurrence.
    sub_cage_mask = cage_mask[np.ix_(real_idx, real_idx)]   # (n,n)
    cage_adj_from_mask = (sub_cage_mask > 0).astype(np.float64)
    cage_adj_from_M = ((M_cage @ M_cage.T) > 0).astype(np.float64)
    np.fill_diagonal(cage_adj_from_mask, 0.0)
    np.fill_diagonal(cage_adj_from_M, 0.0)
    assert np.array_equal(cage_adj_from_mask, cage_adj_from_M), (
        "cage membership reconstruction disagrees with encode_puzzle cage_mask"
    )

    return real_idx, M_row, M_col, M_cage, cage_lab, row_lab, col_lab


def adj_from_membership(M):
    """adj[i,j] = 1 iff cells i,j share a factor (membership^T @ membership > 0),
    zero diagonal (no self-loops in the graph Laplacian sense)."""
    A = ((M @ M.T) > 0).astype(np.float64)
    np.fill_diagonal(A, 0.0)
    return A


def build_operators(M_row, M_col, M_cage, cage_weight=1.0):
    """Return weighted adjacency matrices for the three operators.

    FULL: row OR col OR cage as a 0/1 graph (each shared factor => an edge).
    CAGE-WEIGHTED: full graph but cage edges weighted by `cage_weight` (>=1).
                   Implemented as a weighted sum: row-edge + col-edge contribute
                   weight 1, cage-edge contributes `cage_weight`; edge weight is
                   the max contribution (so a row+cage co-edge gets cage_weight).
    CAGE-ONLY: cage edges only.
    """
    A_row = adj_from_membership(M_row)
    A_col = adj_from_membership(M_col)
    A_cage = adj_from_membership(M_cage)

    A_full = ((A_row + A_col + A_cage) > 0).astype(np.float64)

    # cage-weighted: base rook weight 1 where any row/col edge, cage edges get w.
    A_rook = ((A_row + A_col) > 0).astype(np.float64)
    A_cw = np.maximum(A_rook, A_cage * cage_weight)

    return A_full, A_cw, A_cage


# --------------------------------------------------------------------------- #
# Laplacians + spectral clustering.
# --------------------------------------------------------------------------- #
def laplacian(A, normalized=False):
    """Unnormalized L = D - A, or symmetric normalized L_sym = I - D^-1/2 A D^-1/2."""
    d = A.sum(axis=1)
    if not normalized:
        return np.diag(d) - A
    dinv_sqrt = np.zeros_like(d)
    nz = d > 0
    dinv_sqrt[nz] = 1.0 / np.sqrt(d[nz])
    Dis = np.diag(dinv_sqrt)
    n = A.shape[0]
    return np.eye(n) - Dis @ A @ Dis


def spectral_embed(L):
    """Return (eigvals, eigvecs) sorted ascending. Symmetric => scipy eigh."""
    w, V = eigh(L)
    order = np.argsort(w)
    return w[order], V[:, order]


def kmeans_labels(X, k, seed=GLOBAL_SEED):
    """k-means cluster rows of X into k clusters; returns integer labels.
    Guards k against the number of rows."""
    n = X.shape[0]
    k = max(1, min(k, n))
    if k == 1:
        return np.zeros(n, dtype=np.int64)
    km = KMeans(n_clusters=k, n_init=10, random_state=seed)
    return km.fit_predict(X)


def cluster_top_k_nontrivial(eigvals, eigvecs, k, eps=1e-8):
    """Spectral clustering: drop near-zero (trivial / disconnected-component) modes,
    take the next k nontrivial eigenvectors, k-means into k clusters.

    For a connected graph there is one zero eigenvalue; we drop all eigenvalues
    <= eps (this also drops the per-component constants if the graph is split).
    We then take up to k nontrivial vectors (the standard spectral-clustering
    embedding dimension == #clusters)."""
    nontrivial = np.nonzero(eigvals > eps)[0]
    if nontrivial.size == 0:
        return np.zeros(eigvecs.shape[0], dtype=np.int64)
    take = nontrivial[: max(1, k)]
    X = eigvecs[:, take]
    return kmeans_labels(X, k)


# --------------------------------------------------------------------------- #
# Alignment metrics.
# --------------------------------------------------------------------------- #
def align(pred, gt):
    return (
        normalized_mutual_info_score(gt, pred),
        adjusted_rand_score(gt, pred),
    )


def random_partition(n, k, seed):
    """Random labeling of n items into (up to) k groups, matched cluster count."""
    rng = np.random.default_rng(seed)
    if k <= 1:
        return np.zeros(n, dtype=np.int64)
    return rng.integers(0, k, size=n)


# --------------------------------------------------------------------------- #
# Per-band analysis.
# --------------------------------------------------------------------------- #
def band_indices(eigvals, eps=1e-8):
    """Split the NONTRIVIAL eigenvector indices into low / mid / high thirds."""
    nontrivial = np.nonzero(eigvals > eps)[0]
    m = nontrivial.size
    if m == 0:
        return {"low": np.array([], int), "mid": np.array([], int),
                "high": np.array([], int)}
    t = m // 3
    low = nontrivial[:max(1, t)]
    mid = nontrivial[max(1, t):max(2, 2 * t)] if m >= 3 else nontrivial[:0]
    high = nontrivial[max(2, 2 * t):] if m >= 3 else nontrivial[:0]
    return {"low": low, "mid": mid, "high": high}


def band_cluster_and_align(eigvecs, idxs, k_cage, n_grid, cage_lab,
                           row_lab, col_lab):
    """For a band: k-means at k=#cages and align with CAGES; SEPARATELY k-means at
    k=N and align with ROW and with COL (the rook backbone, matched cluster count).
    Returns alignment vs cages (k=#cages) and the BETTER of row/col (k=N)."""
    if idxs.size == 0:
        return None
    X = eigvecs[:, idxs]
    # cage alignment at k = #cages
    pred_c = kmeans_labels(X, k_cage)
    nmi_cage, ari_cage = align(pred_c, cage_lab)
    # row/col alignment at k = N (matched to the N-way row/col partitions)
    pred_n = kmeans_labels(X, n_grid)
    nmi_row, ari_row = align(pred_n, row_lab)
    nmi_col, ari_col = align(pred_n, col_lab)
    # take the better-aligned of row/col (the band may pick either axis)
    if (nmi_row + ari_row) >= (nmi_col + ari_col):
        nmi_rc, ari_rc = nmi_row, ari_row
    else:
        nmi_rc, ari_rc = nmi_col, ari_col
    return {
        "nmi_cage": nmi_cage, "ari_cage": ari_cage,
        "nmi_rc": nmi_rc, "ari_rc": ari_rc,
    }


# --------------------------------------------------------------------------- #
# Main per-puzzle pipeline.
# --------------------------------------------------------------------------- #
def cluster_dual_k(eigvals, eigvecs, k_cages, n_grid, cage_lab, row_lab, col_lab):
    """Cluster the spectral embedding twice: at k=#cages (the cage question) and at
    k=N (the row/col question), then align each at its matched cluster count.

    Cage clustering uses the top-k_cages nontrivial eigenvectors; row/col uses the
    top-(N-1) nontrivial eigenvectors (standard spectral-clustering embedding dim
    == #clusters). Returns (cage_align, rc_align) where rc is the BETTER of the
    row-aligned and col-aligned scores."""
    pred_c = cluster_top_k_nontrivial(eigvals, eigvecs, k_cages)
    cage_align = align(pred_c, cage_lab)
    pred_n = cluster_top_k_nontrivial(eigvals, eigvecs, n_grid)
    nmi_row, ari_row = align(pred_n, row_lab)
    nmi_col, ari_col = align(pred_n, col_lab)
    if (nmi_row + ari_row) >= (nmi_col + ari_col):
        rc_align = (nmi_row, ari_row)
    else:
        rc_align = (nmi_col, ari_col)
    return cage_align, rc_align


def analyze_puzzle(enc, puzzle_seed):
    real_idx, M_row, M_col, M_cage, cage_lab, row_lab, col_lab = (
        build_type_membership(enc)
    )
    n = real_idx.size
    k_cages = int(np.unique(cage_lab).size)
    n_grid = int(np.unique(row_lab).size)   # == N (number of grid rows/cols)

    out = {"n": n, "k_cages": k_cages, "N": enc["N"]}

    # Degree stats on the full graph (to characterize the rook backbone).
    A_full, _, A_cage = build_operators(M_row, M_col, M_cage, cage_weight=1.0)
    out["mean_deg_full"] = float(A_full.sum(axis=1).mean())
    out["mean_deg_cage"] = float(A_cage.sum(axis=1).mean())
    # cage edges as a fraction of total edges (the perturbation strength).
    n_full_edges = int(A_full.sum() / 2)
    n_cage_edges = int(A_cage.sum() / 2)
    out["frac_cage_edges"] = (n_cage_edges / n_full_edges) if n_full_edges else 0.0

    # ---- Operator (1) FULL: unnormalized + normalized Laplacian.
    # Cage question clustered at k=#cages; row/col question clustered at k=N.
    for norm_key, normalized in [("unnorm", False), ("norm", True)]:
        L = laplacian(A_full, normalized=normalized)
        w, V = spectral_embed(L)
        cage_a, rc_a = cluster_dual_k(w, V, k_cages, n_grid,
                                      cage_lab, row_lab, col_lab)
        out[f"full_{norm_key}_cage"] = cage_a
        out[f"full_{norm_key}_rc"] = rc_a
        # per-band only on the unnormalized full graph (the headline band test).
        if norm_key == "unnorm":
            bands = band_indices(w)
            for bname, idxs in bands.items():
                r = band_cluster_and_align(V, idxs, k_cages, n_grid,
                                           cage_lab, row_lab, col_lab)
                out[f"band_{bname}"] = r

    # ---- Operator (2) cage-weighted: sweep w (unnormalized Laplacian).
    for cw in CAGE_WEIGHTS:
        _, A_cw, _ = build_operators(M_row, M_col, M_cage, cage_weight=float(cw))
        L = laplacian(A_cw, normalized=False)
        w_, V_ = spectral_embed(L)
        cage_a, rc_a = cluster_dual_k(w_, V_, k_cages, n_grid,
                                      cage_lab, row_lab, col_lab)
        out[f"cw{cw}_cage"] = cage_a
        out[f"cw{cw}_rc"] = rc_a

    # ---- Operator (3) cage-only subgraph: components == cages sanity.
    # Spectral cluster on the cage-only Laplacian (degenerate eigenstructure since
    # it is a disjoint union of cliques) — but the cleaner test is connected
    # components.
    L_cage = laplacian(A_cage, normalized=False)
    w_c, V_c = spectral_embed(L_cage)
    # connected components == #(zero eigenvalues).
    n_components = int(np.sum(w_c < 1e-8))
    # label each cell by its connected component (use cage_lab == component when
    # the cage subgraph is exactly the cage partition — verify via clustering on
    # the trivial eigenspace). Simplest robust check: cluster on the full eigvec
    # set with k = k_cages and compare.
    comp_labels = connected_components(A_cage)
    out["cage_subgraph_ncomp"] = n_components
    out["cage_subgraph_nmi"] = normalized_mutual_info_score(cage_lab, comp_labels)
    out["cage_subgraph_ari"] = adjusted_rand_score(cage_lab, comp_labels)

    # ---- Random-partition null (matched cluster count to each ground truth).
    # cage null: random into k_cages groups vs cage labels.
    rp_cage = random_partition(n, k_cages, seed=puzzle_seed)
    out["null_cage"] = align(rp_cage, cage_lab)
    # row/col null: random into N groups vs the better-aligned of row/col labels.
    rp_n = random_partition(n, n_grid, seed=puzzle_seed + 7)
    nr = align(rp_n, row_lab)
    nc = align(rp_n, col_lab)
    out["null_rc"] = nr if (nr[0] + nr[1]) >= (nc[0] + nc[1]) else nc

    return out


def connected_components(A):
    """Label connected components of a 0/1 adjacency via BFS. Returns int labels."""
    n = A.shape[0]
    labels = np.full(n, -1, dtype=np.int64)
    cur = 0
    for s in range(n):
        if labels[s] != -1:
            continue
        stack = [s]
        labels[s] = cur
        while stack:
            u = stack.pop()
            nbrs = np.nonzero(A[u] > 0)[0]
            for v in nbrs:
                if labels[v] == -1:
                    labels[v] = cur
                    stack.append(int(v))
        cur += 1
    return labels


# --------------------------------------------------------------------------- #
# Aggregation + reporting.
# --------------------------------------------------------------------------- #
def mean_pair(rows, key):
    nmis = [r[key][0] for r in rows if key in r and r[key] is not None]
    aris = [r[key][1] for r in rows if key in r and r[key] is not None]
    if not nmis:
        return (float("nan"), float("nan"))
    return (float(np.mean(nmis)), float(np.mean(aris)))


def mean_band(rows, bname, subkey):
    vals = [r[f"band_{bname}"][subkey] for r in rows
            if r.get(f"band_{bname}") is not None]
    if not vals:
        return float("nan")
    return float(np.mean(vals))


def fmt(x):
    return f"{x:.3f}" if x == x else "  nan"


def main():
    print(f"[diag_kenken_spectral] data={DATA_PATH} per_N={PUZZLES_PER_N} "
          f"Ns={NS} seed={GLOBAL_SEED}", flush=True)
    recs = kd.load_jsonl(DATA_PATH)

    by_N = defaultdict(list)
    for r in recs:
        by_N[int(r["N"])].append(r)

    # Deterministic sample of PUZZLES_PER_N per N.
    sampler = np.random.default_rng(GLOBAL_SEED)
    sampled = {}
    for N in NS:
        pool = by_N.get(N, [])
        if not pool:
            continue
        idx = sampler.permutation(len(pool))[:PUZZLES_PER_N]
        sampled[N] = [pool[i] for i in idx]

    all_rows = {N: [] for N in NS}
    for N in NS:
        for pi, rec in enumerate(sampled.get(N, [])):
            enc = kd.encode_puzzle(rec, n_cages_max=41)
            seed = GLOBAL_SEED + 1000 * N + pi
            row = analyze_puzzle(enc, seed)
            all_rows[N].append(row)

    # ---------------- PER-N summaries ----------------
    print("\n" + "=" * 78)
    print("PER-N SUMMARY (mean NMI / ARI over sampled puzzles)")
    print("=" * 78)
    for N in NS:
        rows = all_rows[N]
        if not rows:
            continue
        kc = np.mean([r["k_cages"] for r in rows])
        deg = np.mean([r["mean_deg_full"] for r in rows])
        dcage = np.mean([r["mean_deg_cage"] for r in rows])
        fce = np.mean([r["frac_cage_edges"] for r in rows])
        print(f"\nN={N}  (n_puzzles={len(rows)}, mean #cages={kc:.1f}, "
              f"mean deg full={deg:.1f}, mean deg cage={dcage:.2f}, "
              f"cage-edge frac={fce:.2f})")
        print(f"  {'operator':<22}{'NMI_cage':>10}{'ARI_cage':>10}"
              f"{'NMI_rc':>10}{'ARI_rc':>10}")

        def line(label, ck, rk):
            nc, ac = mean_pair(rows, ck)
            nr, ar = mean_pair(rows, rk)
            print(f"  {label:<22}{fmt(nc):>10}{fmt(ac):>10}"
                  f"{fmt(nr):>10}{fmt(ar):>10}")

        line("FULL (unnorm)", "full_unnorm_cage", "full_unnorm_rc")
        line("FULL (norm)", "full_norm_cage", "full_norm_rc")
        for cw in CAGE_WEIGHTS:
            line(f"cage-weighted w={cw}", f"cw{cw}_cage", f"cw{cw}_rc")
        line("NULL (random k)", "null_cage", "null_rc")

        # cage subgraph sanity
        nmi = np.mean([r["cage_subgraph_nmi"] for r in rows])
        ari = np.mean([r["cage_subgraph_ari"] for r in rows])
        ncomp = np.mean([r["cage_subgraph_ncomp"] for r in rows])
        kcages = np.mean([r["k_cages"] for r in rows])
        print(f"  cage-only subgraph: NMI={nmi:.3f} ARI={ari:.3f} "
              f"(#components={ncomp:.1f} vs #cages={kcages:.1f})")

        # per-band
        print(f"  PER-BAND (k-means each eigenvector band -> alignment):")
        print(f"    {'band':<8}{'NMI_cage':>10}{'ARI_cage':>10}"
              f"{'NMI_rc':>10}{'ARI_rc':>10}")
        for bname in ["low", "mid", "high"]:
            nc = mean_band(rows, bname, "nmi_cage")
            ac = mean_band(rows, bname, "ari_cage")
            nr = mean_band(rows, bname, "nmi_rc")
            ar = mean_band(rows, bname, "ari_rc")
            print(f"    {bname:<8}{fmt(nc):>10}{fmt(ac):>10}"
                  f"{fmt(nr):>10}{fmt(ar):>10}")

    # ---------------- GRAND (averaged across N) compact table ----------------
    flat = [r for N in NS for r in all_rows[N]]
    print("\n" + "=" * 78)
    print("COMPACT NUMBERS TABLE (averaged across all N, all puzzles)")
    print("=" * 78)
    print(f"{'operator x ground-truth':<26}{'NMI_cage':>10}{'ARI_cage':>10}"
          f"{'NMI_rc':>10}{'ARI_rc':>10}")
    print("-" * 66)

    def grand(label, ck, rk):
        nc, ac = mean_pair(flat, ck)
        nr, ar = mean_pair(flat, rk)
        print(f"{label:<26}{fmt(nc):>10}{fmt(ac):>10}{fmt(nr):>10}{fmt(ar):>10}")

    grand("FULL (unnorm)", "full_unnorm_cage", "full_unnorm_rc")
    grand("FULL (norm)", "full_norm_cage", "full_norm_rc")
    for cw in CAGE_WEIGHTS:
        grand(f"cage-weighted w={cw}", f"cw{cw}_cage", f"cw{cw}_rc")
    grand("NULL (random)", "null_cage", "null_rc")
    print("-" * 66)
    print("PER-BAND (averaged across N):")
    print(f"{'band':<26}{'NMI_cage':>10}{'ARI_cage':>10}"
          f"{'NMI_rc':>10}{'ARI_rc':>10}")
    for bname in ["low", "mid", "high"]:
        nc = mean_band(flat, bname, "nmi_cage")
        ac = mean_band(flat, bname, "ari_cage")
        nr = mean_band(flat, bname, "nmi_rc")
        ar = mean_band(flat, bname, "ari_rc")
        print(f"{bname:<26}{fmt(nc):>10}{fmt(ac):>10}{fmt(nr):>10}{fmt(ar):>10}")
    # cage subgraph grand
    nmi = np.mean([r["cage_subgraph_nmi"] for r in flat])
    ari = np.mean([r["cage_subgraph_ari"] for r in flat])
    print("-" * 66)
    print(f"cage-only subgraph sanity: NMI={nmi:.3f} ARI={ari:.3f} "
          f"(should be ~1.0)")
    print("=" * 78)


if __name__ == "__main__":
    main()
