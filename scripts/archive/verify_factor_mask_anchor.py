"""verify_factor_mask_anchor.py — CPU-only bit-equality check for factor_masks.

Asserts that build_factor_attn_bias (the general shared builder in
mycelium/factor_masks.py) produces BIT-EQUAL output to build_kenken_attn_bias
(the validated v98 KenKen builder in mycelium/kenken.py) on the same batch.

Run:
    DEV=CPU python scripts/verify_factor_mask_anchor.py

No GPU, no training, no ROCm/CUDA/PyTorch required.

HEAD-ORDER RECONCILIATION
-------------------------
v98 KenKen uses _build_kenken_fixed_masks (n_heads=16) -> 5/5/5/1 split:
  heads  0- 4: ROW   (same-row clique, validity-gated)
  heads  5- 9: COL   (same-col clique, validity-gated)
  heads 10-14: CAGE  (per-instance symmetric cage clique, validity-gated)
  head  15   : GLOBAL (all-valid, validity-gated)

build_factor_attn_bias uses cell_mp_head_allocation(T=3, H=16, G=1):
  R=15, base=5, rem=0 -> alloc = [0,0,0,0,0, 1,1,1,1,1, 2,2,2,2,2, -1]
  where T=0=ROW, T=1=COL, T=2=CAGE, -1=GLOBAL

These are IDENTICAL in both count and contiguous ordering.

GLOBAL-HEAD RECONCILIATION
---------------------------
v98:  fixed mask has mask[h] = np.ones((49,49)) for the global head, then
      validity is applied at the combined-mask level: block pad keys, pad-query
      rows -> self-only.
Factor: global_allow = _validity_mask(ones_49) — same operations on the same
        all-ones starting matrix.  The result must be identical.

SELF-EDGE FIX (step 3 in _validity_mask)
-----------------------------------------
For KenKen, every valid cell appears in exactly one row latent, one col latent,
and one cage latent, so A_t[i,i]==1 for every valid cell i on every type t.
The .maximum(eye * valid_q) step is therefore a no-op and byte-identical.
"""
from __future__ import annotations

import os
import sys

# CPU-only: must be set BEFORE importing tinygrad.
os.environ.setdefault("DEV", "CPU")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from tinygrad import Tensor, dtypes

from mycelium.kenken_data import KenKenLoader, N_MAX, N_CELLS
from mycelium.kenken import build_kenken_attn_bias, _build_kenken_fixed_masks
from mycelium.perceiver_poincare_data import (
    PerceiverLoader, build_constraint_membership,
    LTYPE_ROW, LTYPE_COL, LTYPE_CAGE, LTYPE_GLOBAL,
)
from mycelium.perceiver_poincare import PERCEIVER_N_GLOBAL
from mycelium.factor_masks import build_factor_attn_bias, cell_mp_head_allocation, CELL_MP_HEAD_GLOBAL

TRAIN_PATH = ".cache/kenken_train.jsonl"
BATCH_SIZE = 8
N_HEADS = 16
# T = 3 non-global factor types: ROW (0), COL (1), CAGE (2).
N_FACTOR_TYPES = 3   # matches PERCEIVER_N_LATENT_TYPES - 1 = 4 - 1 = 3
S_MAX = N_CELLS      # 49

# ---------------------------------------------------------------------------
# Step 1: Build a stub model object so build_kenken_attn_bias can read the
#         fixed mask and head split (it reads model.kenken_fixed_mask etc.).
# ---------------------------------------------------------------------------

class _StubModel:
    pass

_model = _StubModel()
_model.kenken_fixed_mask, _model.kenken_head_split = _build_kenken_fixed_masks(N_HEADS)

# ---------------------------------------------------------------------------
# Step 2: Load a real batch via the PerceiverLoader (which gives us both the
#         raw KenKen tensors AND the membership/latent_type tensors).
# ---------------------------------------------------------------------------

loader = PerceiverLoader(TRAIN_PATH, batch_size=BATCH_SIZE, seed=0)
batch = loader.sample_batch()

cage_mask  = batch.cell_cage_id   # we actually need cage_mask from the KenKen batch
# PerceiverBatch doesn't expose cage_mask directly — fetch from the KenKen side.
# We need a plain KenKenLoader batch to get cage_mask.
kk_loader = KenKenLoader(TRAIN_PATH, batch_size=BATCH_SIZE, seed=0,
                         n_cages_max=loader.n_cages_max)
kk_batch = kk_loader.sample_batch()

# Sanity check: same items (same seed -> same batch).
cv_kk  = kk_batch.cell_valid.realize().numpy()
cv_pb  = batch.cell_valid.realize().numpy()
assert np.allclose(cv_kk, cv_pb), "PerceiverLoader and KenKenLoader returned different batches at seed=0"

cage_mask_t = kk_batch.cage_mask          # (B, 49, 49)
cell_valid_t = kk_batch.cell_valid        # (B, 49)

membership_t  = batch.latent_membership   # (B, L, 49)
latent_type_t = batch.latent_type         # (B, L) int

# ---------------------------------------------------------------------------
# Step 3: Compute A = build_factor_attn_bias (the general builder).
# ---------------------------------------------------------------------------

A = build_factor_attn_bias(
    membership  = membership_t,
    latent_type = latent_type_t,
    cell_valid  = cell_valid_t,
    n_heads     = N_HEADS,
    n_factor_types = N_FACTOR_TYPES,
    s_max       = S_MAX,
)
A_np = A.realize().numpy()    # (B, 16, 49, 49)

# ---------------------------------------------------------------------------
# Step 4: Compute B = build_kenken_attn_bias (the validated v98 builder).
# ---------------------------------------------------------------------------

B = build_kenken_attn_bias(_model, cage_mask_t, cell_valid_t)
B_np = B.realize().numpy()    # (B, 16, 49, 49)

# ---------------------------------------------------------------------------
# Step 5: Bit-equality assertion + detailed mismatch report.
# ---------------------------------------------------------------------------

print(f"\nShapes:  A={A_np.shape},  B={B_np.shape}")
print(f"A dtype={A_np.dtype},  B dtype={B_np.dtype}")
print(f"A unique values: {np.unique(A_np)}")
print(f"B unique values: {np.unique(B_np)}")

match = np.array_equal(A_np, B_np)

if match:
    print("\nPASS — A and B are BIT-EQUAL element-wise.\n")
else:
    diff = A_np != B_np                          # (B, 16, 49, 49) bool
    n_mismatch = int(diff.sum())
    print(f"\nFAIL — {n_mismatch} element(s) differ out of {diff.size}.")

    # Per-head mismatch count.
    head_diff = diff.sum(axis=(0, 2, 3))         # (16,) mismatch count per head
    alloc = cell_mp_head_allocation(N_FACTOR_TYPES, N_HEADS,
                                    max(1, N_HEADS // 16))
    type_names = {0: "ROW", 1: "COL", 2: "CAGE", CELL_MP_HEAD_GLOBAL: "GLOBAL"}
    print("\nPer-head mismatch counts:")
    for h in range(N_HEADS):
        if head_diff[h] > 0:
            tname = type_names.get(int(alloc[h]), f"TYPE{alloc[h]}")
            print(f"  head {h:2d} ({tname:6s}): {int(head_diff[h])} differences")

    # Per-batch mismatch count.
    batch_diff = diff.sum(axis=(1, 2, 3))        # (B,) mismatch count per batch item
    print("\nPer-batch mismatch counts:")
    for b in range(BATCH_SIZE):
        if batch_diff[b] > 0:
            n_valid_b = int(cv_kk[b].sum())
            N_b = int(round(n_valid_b ** 0.5))
            print(f"  batch {b}: {int(batch_diff[b])} differences  (N={N_b}, n_valid={n_valid_b})")

    # Show up to 5 specific mismatching positions for diagnosis.
    positions = np.argwhere(diff)[:5]
    print("\nFirst up to 5 mismatching positions (b, head, q, k):")
    for pos in positions:
        b, h, q, k = pos
        tname = type_names.get(int(alloc[h]), f"TYPE{alloc[h]}")
        print(f"  [{b},{h:2d},{q:2d},{k:2d}]  head={tname:6s}  A={A_np[b,h,q,k]:.1f}  B={B_np[b,h,q,k]:.1f}"
              f"  cv_q={cv_kk[b,q]:.0f}  cv_k={cv_kk[b,k]:.0f}")

    print()

# ---------------------------------------------------------------------------
# Step 6: Head-order reconciliation report.
# ---------------------------------------------------------------------------

alloc = cell_mp_head_allocation(N_FACTOR_TYPES, N_HEADS, max(1, N_HEADS // 16))
n_row_f = int((alloc == 0).sum())
n_col_f = int((alloc == 1).sum())
n_cage_f = int((alloc == 2).sum())
n_glob_f = int((alloc == CELL_MP_HEAD_GLOBAL).sum())

n_row_v, n_col_v, n_cage_v, n_glob_v = _model.kenken_head_split

print("Head-order reconciliation:")
print(f"  factor_masks allocation : ROW={n_row_f}  COL={n_col_f}  CAGE={n_cage_f}  GLOBAL={n_glob_f}")
print(f"  v98 kenken_head_split   : ROW={n_row_v}  COL={n_col_v}  CAGE={n_cage_v}  GLOBAL={n_glob_v}")
orders_match = (n_row_f == n_row_v and n_col_f == n_col_v
                and n_cage_f == n_cage_v and n_glob_f == n_glob_v)
print(f"  Counts match:  {'YES' if orders_match else 'NO'}")

print()
print("Global-head reconciliation:")
# The global head (head 15) should have identical masks in A and B.
glob_head = N_HEADS - 1   # head 15 (last head, the global one)
assert int(alloc[glob_head]) == CELL_MP_HEAD_GLOBAL, \
    f"Expected head {glob_head} to be global, got alloc={alloc[glob_head]}"
glob_match = np.array_equal(A_np[:, glob_head, :, :], B_np[:, glob_head, :, :])
print(f"  Global head (head {glob_head}) A==B: {'YES' if glob_match else 'NO'}")
if not glob_match:
    gd = A_np[:, glob_head, :, :] != B_np[:, glob_head, :, :]
    print(f"  Global head differences: {int(gd.sum())}")

print()
if match:
    print("OVERALL: PASS")
else:
    print("OVERALL: FAIL")
    sys.exit(1)
