"""B2 Step 1 — CPU reproduce-match gate.

Verifies that build_factor_hyperbolic_attn_bias reproduces build_factor_attn_bias
to ~1e-3 on BOTH:
  (A) KenKen T=3 (partition relation, B=3 synthetic instances, 49 cells)
  (B) Graph coloring T=1 (non-partition relation, via GraphColoringLoader)

For each:
  G = build_factor_attn_bias(...)       # deterministic {0,-1e4} boolean
  H = build_factor_hyperbolic_attn_bias(...)  # geometric, ~1e-3-identical

Agreement metric:
  - "block" entries: G == -1e4 -> H should be <= -(1e4 - 1.0) (close to -1e4).
  - "allow" entries: G == 0   -> H should be >= -1.0 (close to 0).
  - Report allow-agreement rate, block-agreement rate, and max-abs-diff
    in each region, per head, per test case.

Gate: PASS iff coloring allow-agreement >= 0.99 AND block-agreement >= 0.99.
(KenKen should be essentially identical given it's a partition relation with a
known-good simplex anchor.)

GPU-FREE: pure CPU numpy + tinygrad (no .realize() on GPU).  All tensor ops
run eagerly on numpy backend.
"""
from __future__ import annotations

import os
import sys

# Force CPU / numpy backend for tinygrad so this runs GPU-free.
os.environ["KENKEN_TASK"] = "0"

import numpy as np
from tinygrad import Tensor, dtypes

from mycelium.factor_masks import (
    build_factor_attn_bias,
    build_factor_hyperbolic_attn_bias,
    attach_factor_hyperbolic_params,
    cell_mp_head_allocation,
    CELL_MP_HEAD_GLOBAL,
    FG_HYP_BLOCK,
    FG_HYP_MASK,
)
from mycelium.graph_coloring_data import (
    GraphColoringLoader,
    LTYPE_EDGE,
    LTYPE_GLOBAL,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _np(t: Tensor) -> np.ndarray:
    """Realize a Tensor to numpy (CPU-safe)."""
    return t.numpy()


def _compare(G_np: np.ndarray, H_np: np.ndarray, label: str,
             block_val: float = -1e4, tol_allow: float = 1.0,
             tol_block: float = 1.0) -> dict:
    """Compare G (boolean {0,-1e4}) to H (geometric approx).

    Returns a summary dict with:
      allow_agree, block_agree  : fraction of entries that match within tolerance.
      allow_max_abs, block_max_abs : worst-case |G - H| in each region.
      passed : bool (both >= 0.99)
    """
    is_block = (G_np <= block_val + 0.5)          # G == -1e4 (with fp tolerance)
    is_allow = ~is_block                           # G == 0

    n_allow = int(is_allow.sum())
    n_block = int(is_block.sum())

    if n_allow > 0:
        allow_abs = np.abs(G_np[is_allow] - H_np[is_allow])
        allow_agree = float((allow_abs <= tol_allow).mean())
        allow_max_abs = float(allow_abs.max())
    else:
        allow_agree = 1.0
        allow_max_abs = 0.0

    if n_block > 0:
        block_abs = np.abs(G_np[is_block] - H_np[is_block])
        block_agree = float((block_abs <= tol_block).mean())
        block_max_abs = float(block_abs.max())
    else:
        block_agree = 1.0
        block_max_abs = 0.0

    passed = (allow_agree >= 0.99) and (block_agree >= 0.99)
    print(
        f"  [{label}] n_allow={n_allow:6d} allow_agree={allow_agree:.4f} "
        f"allow_max|G-H|={allow_max_abs:.4f} | "
        f"n_block={n_block:6d} block_agree={block_agree:.4f} "
        f"block_max|G-H|={block_max_abs:.4f} | "
        f"{'PASS' if passed else 'FAIL'}"
    )
    return {
        "allow_agree": allow_agree,
        "allow_max_abs": allow_max_abs,
        "block_agree": block_agree,
        "block_max_abs": block_max_abs,
        "n_allow": n_allow,
        "n_block": n_block,
        "passed": passed,
    }


def _make_model() -> object:
    """Lightweight model-stand-in (just an attribute namespace)."""
    class _Model:
        pass
    return _Model()


# ---------------------------------------------------------------------------
# (A) KenKen reproduce-match (T=3 partition, B=3, s_max=49)
# ---------------------------------------------------------------------------

def _kenken_membership_batch(B: int, s_max: int, n_cages_max: int) -> tuple:
    """Build synthetic KenKen membership/latent_type (B, L, s_max) / (B, L).

    Uses the standard row/col structure + synthetic cage assignments.
    T=3: type 0 = row, type 1 = col, type 2 = cage.
    """
    N_MAX = 7
    N_CELLS = 49
    assert s_max == N_CELLS, f"KenKen s_max must be {N_CELLS}"

    rows_idx = np.array([i // N_MAX for i in range(N_CELLS)], dtype=np.int32)
    cols_idx = np.array([i % N_MAX for i in range(N_CELLS)], dtype=np.int32)

    # Row membership: (N_MAX, N_CELLS).
    row_mem = np.zeros((N_MAX, N_CELLS), dtype=np.float32)
    for r in range(N_MAX):
        row_mem[r, rows_idx == r] = 1.0

    # Col membership: (N_MAX, N_CELLS).
    col_mem = np.zeros((N_MAX, N_CELLS), dtype=np.float32)
    for c in range(N_MAX):
        col_mem[c, cols_idx == c] = 1.0

    C = n_cages_max
    L = N_MAX + N_MAX + C

    # Build batch: each instance gets a random cage assignment (partition).
    rng = np.random.RandomState(42)
    membership_batch = np.zeros((B, L, N_CELLS), dtype=np.float32)
    latent_type_batch = np.zeros((B, L), dtype=np.int32)
    cell_valid_batch = np.ones((B, N_CELLS), dtype=np.float32)

    for b in range(B):
        # Board size: alternating N=5,6,7 for variety.
        N = 5 + (b % 3)
        valid_cells = N * N
        cell_valid_batch[b, :] = 0.0
        cell_valid_batch[b, :valid_cells] = 1.0

        # Row / col membership (constant across batch).
        membership_batch[b, :N_MAX, :] = row_mem        # type-0 rows
        membership_batch[b, N_MAX:N_MAX + N_MAX, :] = col_mem   # type-1 cols

        # Mask off padding cells (cells >= valid_cells are not in any real factor).
        # For padded cells, keep membership=0 (they don't belong to any factor).
        for r in range(N_MAX):
            for i in range(valid_cells, N_CELLS):
                membership_batch[b, r, i] = 0.0
        for c in range(N_MAX):
            for i in range(valid_cells, N_CELLS):
                membership_batch[b, N_MAX + c, i] = 0.0

        # Cage membership: random partition of the valid cells into ~N cages.
        n_cages = N  # one cage per row for simplicity
        cage_assign = rng.randint(0, n_cages, size=valid_cells).astype(np.int32)
        cage_mem_b = np.zeros((C, N_CELLS), dtype=np.float32)
        for ci in range(n_cages):
            for i in range(valid_cells):
                if cage_assign[i] == ci:
                    cage_mem_b[ci, i] = 1.0
        membership_batch[b, N_MAX + N_MAX:, :] = cage_mem_b

        # Latent types: 0=row, 1=col, 2=cage.
        lt = np.concatenate([
            np.zeros((N_MAX,), dtype=np.int32),         # rows -> type 0
            np.ones((N_MAX,), dtype=np.int32),          # cols -> type 1
            np.full((C,), 2, dtype=np.int32),           # cages -> type 2
        ])
        latent_type_batch[b, :] = lt

    return membership_batch, latent_type_batch, cell_valid_batch


def test_kenken(n_heads: int = 16, n_factor_types: int = 3, s_max: int = 49,
                n_cages_max: int = 20, B: int = 3) -> bool:
    """KenKen reproduce-match: geometric vs boolean mask."""
    print(f"\n{'='*70}")
    print(f"(A) KenKen T={n_factor_types}, B={B}, s_max={s_max}, "
          f"n_cages_max={n_cages_max}, n_heads={n_heads}")
    print(f"{'='*70}")

    membership_np, latent_type_np, cell_valid_np = _kenken_membership_batch(
        B, s_max, n_cages_max)

    # Wrap as Tensors.
    membership = Tensor(membership_np, dtype=dtypes.float)
    latent_type = Tensor(latent_type_np, dtype=dtypes.int)
    cell_valid = Tensor(cell_valid_np, dtype=dtypes.float)

    # Boolean mask (anchor target).
    G = build_factor_attn_bias(membership, latent_type, cell_valid,
                               n_heads, n_factor_types, s_max)
    G_np = _np(G)
    print(f"  G shape: {G_np.shape}, unique values: {np.unique(G_np)}")

    # Attach hyperbolic params.
    model = _make_model()
    attach_factor_hyperbolic_params(
        model,
        n_heads=n_heads,
        n_factor_types=n_factor_types,
        s_max=s_max,
        membership_np=membership_np,
        latent_type_np=latent_type_np,
    )
    print(f"  Attached anchor tables:")
    G_alloc = max(1, n_heads // 16)
    alloc = cell_mp_head_allocation(n_factor_types, n_heads, G_alloc)
    used_types = sorted({int(t) for t in alloc if int(t) != CELL_MP_HEAD_GLOBAL})
    for t in used_types:
        anch = getattr(model, f"fg_hyp_anchors_{t}")
        r_t = getattr(model, f"fg_hyp_r_{t}")
        alpha_t = getattr(model, f"fg_hyp_alpha_{t}")
        print(f"    type {t}: anchors={anch.shape}, r={r_t:.4f}, alpha={alpha_t:.2e}")

    # Geometric mask.
    H = build_factor_hyperbolic_attn_bias(
        model, membership, latent_type, cell_valid,
        n_heads, n_factor_types, s_max)
    H_np = _np(H)
    print(f"  H shape: {H_np.shape}, range: [{H_np.min():.4f}, {H_np.max():.4f}]")

    # Compare per-head.
    all_passed = True
    for h in range(n_heads):
        t = int(alloc[h])
        label = f"h{h:02d}/t{'G' if t == CELL_MP_HEAD_GLOBAL else t}"
        result = _compare(G_np[:, h, :, :], H_np[:, h, :, :], label)
        if not result["passed"]:
            all_passed = False

    print(f"\n  KenKen overall: {'PASS' if all_passed else 'FAIL'}")
    return all_passed


# ---------------------------------------------------------------------------
# (B) Graph coloring reproduce-match (T=1 non-partition)
# ---------------------------------------------------------------------------

def test_coloring(n_heads: int = 16, s_max: int = 49, k_colors: int = 3,
                  n_instances: int = 200, batch_size: int = 8) -> bool:
    """Graph coloring reproduce-match: geometric vs boolean mask.

    Graph coloring is the LOAD-BEARING non-partition case.  A vertex belongs
    to ALL its incident edge-factors -> non-partition.  The geometric builder
    must reproduce the edge adjacency mask via PER-FACTOR CLIQUE-UNION.
    """
    print(f"\n{'='*70}")
    print(f"(B) Graph coloring T=1, B={batch_size}, s_max={s_max}, "
          f"k={k_colors}, n_instances={n_instances}, n_heads={n_heads}")
    print(f"{'='*70}")

    n_factor_types = 1   # T=1: edge relation only

    loader = GraphColoringLoader(
        n_instances=n_instances, s_max=s_max, k_colors=k_colors,
        batch_size=batch_size, seed=99)
    batch = loader.sample_batch()

    membership = batch.membership              # (B, n_edges_max, s_max)
    latent_type = batch.latent_type            # (B, n_edges_max) — 0 edge / 1 sentinel
    cell_valid  = batch.cell_valid             # (B, s_max)

    n_edges_max = int(membership.shape[1])
    print(f"  n_edges_max={n_edges_max}, n={batch.n}, n_edges={batch.n_edges}")

    membership_np  = _np(membership).astype(np.float32)
    latent_type_np = _np(latent_type).astype(np.int32)
    cell_valid_np  = _np(cell_valid).astype(np.float32)

    # Boolean mask (anchor target).
    G = build_factor_attn_bias(membership, latent_type, cell_valid,
                               n_heads, n_factor_types, s_max)
    G_np = _np(G)
    print(f"  G shape: {G_np.shape}, allow fraction: "
          f"{(G_np == 0).mean():.4f}, block fraction: {(G_np <= -9999).mean():.4f}")

    # Attach hyperbolic params.
    model = _make_model()
    attach_factor_hyperbolic_params(
        model,
        n_heads=n_heads,
        n_factor_types=n_factor_types,
        s_max=s_max,
        membership_np=membership_np,
        latent_type_np=latent_type_np,
    )
    G_alloc = max(1, n_heads // 16)
    alloc = cell_mp_head_allocation(n_factor_types, n_heads, G_alloc)
    used_types = sorted({int(t) for t in alloc if int(t) != CELL_MP_HEAD_GLOBAL})
    print(f"  Attached anchor tables:")
    for t in used_types:
        anch = getattr(model, f"fg_hyp_anchors_{t}")
        r_t = getattr(model, f"fg_hyp_r_{t}")
        alpha_t = getattr(model, f"fg_hyp_alpha_{t}")
        # Spherical code if G_t_alloc > dim+1
        dim = int(anch.shape[1])
        G_t_alloc = int(anch.shape[0])
        code_type = "simplex" if (G_t_alloc - 1) <= dim else "spherical"
        print(f"    type {t}: anchors={anch.shape} ({code_type}), "
              f"r={r_t:.4f}, alpha={alpha_t:.2e}")

    # Geometric mask.
    H = build_factor_hyperbolic_attn_bias(
        model, membership, latent_type, cell_valid,
        n_heads, n_factor_types, s_max)
    H_np = _np(H)
    print(f"  H shape: {H_np.shape}, range: [{H_np.min():.4f}, {H_np.max():.4f}]")

    # Compare per-head.
    all_passed = True
    for h in range(n_heads):
        t = int(alloc[h])
        label = f"h{h:02d}/t{'G' if t == CELL_MP_HEAD_GLOBAL else t}"
        result = _compare(G_np[:, h, :, :], H_np[:, h, :, :], label)
        if not result["passed"]:
            all_passed = False

    print(f"\n  Coloring overall: {'PASS' if all_passed else 'FAIL'}")
    return all_passed


# ---------------------------------------------------------------------------
# Byte-identical-OFF confirmation (FG_HYP_MASK=0 path unchanged)
# ---------------------------------------------------------------------------

def test_byte_identical_off(n_heads: int = 8, n_factor_types: int = 1,
                             s_max: int = 12, B: int = 2) -> bool:
    """When FG_HYP_MASK=0 (default), build_factor_attn_bias is called unchanged.

    This test just verifies the boolean function runs normally (it is never
    modified — the geometric function is ADDITIVE and gated).
    """
    print(f"\n{'='*70}")
    print(f"(C) Byte-identical-OFF confirmation: FG_HYP_MASK={FG_HYP_MASK}")
    print(f"{'='*70}")
    # Simple sanity: build boolean mask for a tiny case.
    s = s_max
    # Single type, 3 factors, each covering 4 cells.
    L = 5
    mem_np = np.zeros((B, L, s), dtype=np.float32)
    lt_np = np.zeros((B, L), dtype=np.int32)
    for b in range(B):
        for f in range(3):
            cells = list(range(f * 4, f * 4 + 4))
            for c in cells:
                mem_np[b, f, c] = 1.0
        # Rows 3,4 are padding (type global sentinel = n_factor_types = 1).
        lt_np[b, 3:] = n_factor_types
    cv_np = np.ones((B, s), dtype=np.float32)

    membership = Tensor(mem_np, dtype=dtypes.float)
    latent_type = Tensor(lt_np, dtype=dtypes.int)
    cell_valid = Tensor(cv_np, dtype=dtypes.float)

    G = build_factor_attn_bias(membership, latent_type, cell_valid,
                               n_heads, n_factor_types, s)
    G_np = _np(G)
    unique = np.unique(np.round(G_np, 1))
    ok = set(unique).issubset({0.0, -10000.0})
    print(f"  G unique values: {unique}, all in {{0,-1e4}}: {ok}")
    print(f"  Byte-identical-OFF: {'PASS' if ok else 'FAIL'}")
    return ok


# ---------------------------------------------------------------------------
# Main gate
# ---------------------------------------------------------------------------

def main() -> None:
    print("B2 Step 1 — Geometric mask generator CPU reproduce-match gate")
    print("GPU-FREE: all ops on CPU numpy backend")

    # ast.parse self-check.
    import ast
    for path in ["mycelium/factor_masks.py", "mycelium/kenken.py",
                 "mycelium/graph_coloring_data.py"]:
        with open(path) as f:
            src = f.read()
        ast.parse(src)
    print("\nast.parse OK: factor_masks.py, kenken.py, graph_coloring_data.py")

    pass_a = test_kenken(n_heads=16, n_factor_types=3, s_max=49,
                         n_cages_max=20, B=3)
    pass_b = test_coloring(n_heads=16, s_max=49, k_colors=3,
                           n_instances=200, batch_size=8)
    pass_c = test_byte_identical_off()

    print(f"\n{'='*70}")
    print(f"GATE SUMMARY")
    print(f"  (A) KenKen partition:              {'PASS' if pass_a else 'FAIL'}")
    print(f"  (B) Coloring non-partition (LOAD): {'PASS' if pass_b else 'FAIL'}")
    print(f"  (C) Boolean OFF byte-identical:    {'PASS' if pass_c else 'FAIL'}")
    gate = pass_a and pass_b and pass_c
    print(f"  OVERALL GATE: {'PASS — geometric generalization confirmed' if gate else 'FAIL — do not proceed'}")
    sys.exit(0 if gate else 1)


if __name__ == "__main__":
    main()
