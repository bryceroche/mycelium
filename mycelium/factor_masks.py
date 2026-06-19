"""factor_masks.py — General per-head factor-graph attention mask builder.

Lifts _cell_mp_head_allocation and _build_cell_mp_bias_perhead from
perceiver_poincare.py into a standalone shared module (no perceiver dependencies).

PUBLIC API
----------
CELL_MP_HEAD_GLOBAL : int (-1)
    Sentinel value returned by cell_mp_head_allocation for global-head slots.

cell_mp_head_allocation(T, H, G) -> np.ndarray
    Deterministic per-head type assignment.  Pure numpy, no Tensor.

build_factor_attn_bias(membership, latent_type, cell_valid,
                       n_heads, n_factor_types, s_max) -> Tensor (B,H,s_max,s_max)
    Build {0,-1e4} per-head attention bias from generic factor membership.

DESIGN NOTES
------------
* No perceiver-specific imports: only tinygrad + numpy.
* s_max is passed explicitly (replaces hard-coded 49/N_CELLS).
* n_factor_types = T = number of non-global relation types (e.g. 3 for row/col/cage).
* Head allocation: G = max(1, n_heads // 16) global heads placed at the END;
  R = n_heads - G remaining heads split evenly in contiguous blocks across T types.
  KenKen (T=3, n_heads=16, G=1): R=15, base=5, rem=0 -> 5/5/5 then 1 global.
* Validity masking:
    - pad key cells blocked  (valid_key=0 -> col zeroed)
    - pad query rows         -> self-only eye (no all-block softmax row)
    - SELF-EDGE FIX          -> force diagonal for every VALID query cell on each
                               per-type allow mask (no-op for KenKen since every
                               valid cell already appears in its own row/col/cage)
* Substrate-legal: no dtypes.float32 Tensor literal baked as a graph constant;
  float consts built as np.float32 then wrapped with dtype=dtypes.float; single
  .contiguous() barrier on the output.
"""
from __future__ import annotations

import numpy as np
from tinygrad import Tensor, dtypes

# Sentinel: a head slot assigned to the global (union/all-valid) mask.
CELL_MP_HEAD_GLOBAL: int = -1


def cell_mp_head_allocation(T: int, H: int, G: int) -> np.ndarray:
    """Deterministic per-head TYPE assignment for a per-head cell-MP bias.

    Pure function: maps (T = number of factor TYPES, H = total head count,
    G = number of GLOBAL heads) -> a length-H int64 array.

    Each entry is a TYPE INDEX in 0..T-1, or CELL_MP_HEAD_GLOBAL (= -1) for
    a global head.  Global heads are placed at the END so the per-type blocks
    are contiguous from head 0 — matching v98 KenKen's 5/5/5/1 layout.

    Allocation rule:
      R = H - G  (non-global head budget)
      base = R // T,  rem = R % T
      First `rem` types get base+1 heads; the rest get base.
      When T > R: base=0, so the first R types get 1 head each; trailing
      T-R types get 0 heads (covered only by the global head).

    KenKen example (T=3, H=16, G=1):
      R=15, base=5, rem=0 -> [0,0,0,0,0, 1,1,1,1,1, 2,2,2,2,2, -1]
      = 5 row-heads, 5 col-heads, 5 cage-heads, 1 global  (v98's 5/5/5/1).

    Parameters
    ----------
    T : int   Number of non-global factor types (>=1).
    H : int   Total number of attention heads (>=1).
    G : int   Number of global heads to reserve at the end (in [0, H]).

    Returns
    -------
    np.ndarray shape (H,) dtype int64.
    """
    T = int(T); H = int(H); G = int(G)
    assert H >= 1, f"H must be >=1, got {H}"
    assert 0 <= G <= H, f"G must be in [0,H], got G={G} H={H}"
    assert T >= 1, f"T (factor types) must be >=1, got {T}"

    alloc = np.full((H,), CELL_MP_HEAD_GLOBAL, dtype=np.int64)
    R = H - G                                       # non-global head budget
    if R <= 0:
        return alloc                                # all heads global (degenerate)

    base = R // T                                   # even share per type
    rem = R % T                                     # leftover, spread one-per-type
    h = 0
    for t in range(T):
        cnt = base + (1 if t < rem else 0)
        for _ in range(cnt):
            alloc[h] = t
            h += 1
    # Remaining slots [h .. H-1] stay CELL_MP_HEAD_GLOBAL (the global block).
    assert h == R, f"allocation filled {h} non-global heads, expected {R}"
    return alloc


def build_factor_attn_bias(
    membership: Tensor,
    latent_type: Tensor,
    cell_valid: Tensor,
    n_heads: int,
    n_factor_types: int,
    s_max: int,
) -> Tensor:
    """Build the (B, n_heads, s_max, s_max) relation-specific per-head bias.

    Mirrors v98 KenKen's 5-row / 5-col / 5-cage / 1-global per-head mask
    split, GENERICALLY:

      1. PER-TYPE ADJACENCY.  For each factor type t in 0..T-1 build the
         type-t membership m_t = membership * (latent_type == t), then
         A_t[b,i,j] = (m_t^T @ m_t)[b,i,j] > 0 — cells that share a type-t
         factor (e.g. same row, or same col, or same cage).
      2. HEAD ALLOCATION.  cell_mp_head_allocation(T, n_heads, G) where
         G = max(1, n_heads // 16).
      3. PER-HEAD MASK.  Head h gets A_{alloc[h]}; a global head gets the
         all-valid mask (every valid cell-pair allowed).

    Validity masking (applied to EACH per-type allow mask before conversion):
      - block pad keys       : allow *= valid_key
      - pad-query self-only  : allow = allow*valid_q + eye*(1-valid_q)
      - self-edge fix        : allow = allow.maximum(eye * valid_q)
        (ensures a valid cell with no type-t peer still attends to itself;
        no-op for KenKen since every valid cell is in its own row/col/cage)

    {0,1} allow -> {0,-1e4} bias.

    Parameters
    ----------
    membership  : Tensor (B, L, s_max) float   — 1 if cell j is in latent l.
    latent_type : Tensor (B, L) int             — type index per latent.
    cell_valid  : Tensor (B, s_max) float       — 1 valid / 0 padding.
    n_heads     : int                           — number of attention heads.
    n_factor_types : int                        — T, the number of non-global types.
    s_max       : int                           — grid size (49 for KenKen).

    Returns
    -------
    Tensor (B, n_heads, s_max, s_max) with values in {0, -1e4}.  Contiguous.
    """
    Bn = int(membership.shape[0])
    T = int(n_factor_types)
    G = max(1, n_heads // 16)
    alloc = cell_mp_head_allocation(T, n_heads, G)   # (H,) int64

    m = membership.cast(dtypes.float)               # (B, L, s_max)
    lt = latent_type                                 # (B, L) int

    # Build the identity matrix as np.float32 then wrap — substrate-legal
    # (no dtypes.float32 Tensor literal baked as a JIT graph constant).
    eye_np = np.eye(s_max, dtype=np.float32)
    eye_s = Tensor(eye_np, dtype=dtypes.float).reshape(1, s_max, s_max)

    valid_key = cell_valid.reshape(Bn, 1, s_max)    # (B, 1, s_max)
    valid_q   = cell_valid.reshape(Bn, s_max, 1)    # (B, s_max, 1)

    def _validity_mask(allow: Tensor) -> Tensor:
        """Apply the three-step validity masking to a (B, s_max, s_max) allow tensor."""
        # Step 1: block pad keys.
        allow = allow * valid_key.cast(allow.dtype)
        # Step 2: pad-query rows -> self-only (no all-block softmax row for pad cells).
        allow = (allow * valid_q.cast(allow.dtype)
                 + eye_s * (1.0 - valid_q.cast(allow.dtype)))
        # Step 3: self-edge fix — force diagonal for every VALID query cell.
        # For KenKen this is a NO-OP (every valid cell is already in its own
        # row/col/cage so A_t[i,i]==1 -> .maximum() changes nothing -> byte-identical).
        allow = allow.maximum(eye_s * valid_q.cast(allow.dtype))
        return allow

    # Build per-type adjacency for each type that has at least one head assigned.
    used_types = sorted({int(t) for t in alloc if int(t) != CELL_MP_HEAD_GLOBAL})
    type_allow: dict[int, Tensor] = {}
    for t in used_types:
        sel = (lt == t).cast(dtypes.float).reshape(Bn, int(lt.shape[1]), 1)  # (B,L,1)
        m_t = m * sel                               # (B, L, s_max) type-t members
        adj_t = m_t.transpose(1, 2) @ m_t          # (B, s_max, s_max) co-occurrence
        type_allow[t] = _validity_mask(adj_t.clip(0.0, 1.0))

    # Global allow: all valid cell-pairs (ones), then validity masking — exactly
    # the v98 global head (full mask gated by cell_valid).
    ones_s = Tensor(np.ones((Bn, s_max, s_max), dtype=np.float32), dtype=dtypes.float)
    global_allow = _validity_mask(ones_s)           # (B, s_max, s_max) {0,1}

    # Assemble per-head allow -> {0, -1e4} bias, head by head.
    head_biases = []
    for h in range(n_heads):
        t = int(alloc[h])
        allow_h = global_allow if t == CELL_MP_HEAD_GLOBAL else type_allow[t]
        bias_h = (1.0 - allow_h) * (-1e4)          # (B, s_max, s_max)
        head_biases.append(bias_h.reshape(Bn, 1, s_max, s_max))

    bias = Tensor.cat(*head_biases, dim=1)          # (B, n_heads, s_max, s_max)
    return bias.contiguous()
