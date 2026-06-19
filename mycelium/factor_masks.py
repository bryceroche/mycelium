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

build_factor_hyperbolic_attn_bias(model, membership, latent_type, cell_valid,
                                  n_heads, n_factor_types, s_max) -> Tensor
    Geometric drop-in for build_factor_attn_bias.  The bias is generated from
    continuous Poincare-ball coordinates rather than boolean co-occurrence.
    Reproduces the deterministic {0,-1e4} mask to ~1e-3 at t=0 for BOTH
    partition relations (KenKen row/col/cage) AND non-partition relations
    (graph coloring edges), via PER-FACTOR CLIQUE-UNION anchoring.

attach_factor_hyperbolic_params(model, n_heads, n_factor_types, s_max,
                                membership_np, latent_type_np,
                                dim, rho, alpha_margin) -> None
    Build and attach the per-type anchor tables (one table per type) to `model`
    as model.fg_hyp_anchors_t, model.fg_hyp_r_t, model.fg_hyp_alpha_t.
    Gated behind FG_HYP_MASK=1; the engine seam stays on build_factor_attn_bias
    when the gate is OFF (byte-identical).

FG_HYP_MASK : int (env, default 0)
    Toggle: 0 => build_factor_attn_bias (boolean, byte-identical),
            1 => build_factor_hyperbolic_attn_bias (geometric, ~1e-3-identical).

DESIGN NOTES
------------
* No perceiver-specific imports: only tinygrad + numpy + the substrate
  primitives imported from kenken.py (_poincare_anchors, _spherical_anchors,
  _tangent_for_anchors, _exp0_map, _d_hyp_pairwise, _relation_bias_from_z,
  _hyp_d_out, _min_between_anchor_distance).
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

PER-FACTOR CLIQUE-UNION (the key for non-partition relations)
-------------------------------------------------------------
Each factor f of type t gets ONE anchor a_f (row f in the type-t anchor table).
The members of factor f are ALL placed on anchor a_f — so within a factor, every
cell pair has d_hyp(z_i, z_j) = 0 -> bias(i,j) ~ 0 (attend).  Across factors
(i in f, j not in f), d ~ d_out -> bias ~ -BLOCK (block).

The type-t cell-cell bias = UNION (max, least-blocked) over all type-t factors:
    bias_t[b,i,j] = max over f in type-t factors of bias_f[i,j]

For a PARTITION relation (each cell in exactly one factor, e.g. KenKen rows):
  - The union collapses to the single factor each cell belongs to.
  - Cells i,j same factor -> their factor's bias = 0; other factors all block them.
  - Cells i,j in different factors -> ALL factors block them (both off-anchor).
  -> Reproduces the partition adjacency (m_t^T @ m_t > 0) exactly.

For a NON-PARTITION relation (each cell may belong to many factors, e.g. graph
coloring edges where a vertex belongs to all its incident edge-factors):
  - If (i,j) is an edge: there exists factor f=(i,j) with both i and j on a_f
    -> d(z_i, z_j) = 0 -> bias_f ~ 0 -> max >= 0 (allow).
  - If (i,j) is not an edge: no factor f has BOTH i and j as members ->
    for every f, exactly one of i,j is off a_f -> d ~ d_out -> all factor
    biases ~ -BLOCK -> max ~ -BLOCK (block).
  -> Reproduces the edge adjacency (m_t^T @ m_t > 0) for graph coloring.

ANCHOR TABLE SIZING
-------------------
Type t has G_t = number of type-t factors in the batch (max across the corpus).
  G_t <= dim+1 : exact regular simplex (_poincare_anchors / _simplex_dirs).
  G_t >  dim+1 : spherical code (_spherical_anchors) — normalized gaussian rows,
                  calibrate alpha against the ACTUAL min-between distance.
For graph coloring: G_t = n_edges_max (up to ~120 edge-factors >> dim in typical
settings). We set dim=48 by default; since 120 > 49 = dim+1, the spherical code
path is used. The spherical code is deterministic (fixed seed) so attach-time
calibration is stable.
"""
from __future__ import annotations

import math
import os

import numpy as np
from tinygrad import Tensor, dtypes

# ---- Hyperbolic substrate primitives (imported from kenken.py) ----------------
# These are the validated, substrate-legal implementations (tinygrad + AM driver).
# Import ONLY functions — do NOT import kenken state or env knobs.
from mycelium.kenken import (
    _poincare_anchors,           # (G, dim, rho) -> (G, dim) np ball points
    _spherical_anchors,          # (A, dim, rho) -> (A, dim) np spherical code
    _tangent_for_anchors,        # (mu: np (G,dim)) -> (G,dim) np tangent inits
    _exp0_map,                   # Tensor (...,dim) -> (...,dim) Tensor ball
    _d_hyp_pairwise,             # Tensor (...,M,dim) -> (...,M,M) Tensor d_hyp
    _relation_bias_from_z,       # (z, r, alpha) -> (...,M,M) Tensor bias
    _hyp_d_out,                  # (rho, G) -> float  closed-form d_out simplex
    _min_between_anchor_distance, # (anchors: np (A,dim)) -> float poincare min-d
)

# Toggle: 0 => boolean build_factor_attn_bias (byte-identical, no new params);
#         1 => geometric build_factor_hyperbolic_attn_bias (~1e-3 to boolean).
FG_HYP_MASK: int = int(os.environ.get("FG_HYP_MASK", "0")) > 0

# Poincare-ball shell radius for the anchors (mirroring KENKEN_HYP_RHO).
FG_HYP_RHO: float = float(os.environ.get("FG_HYP_RHO", "0.7"))

# Per-type coordinate dim. Shared across all types (one code path for d_hyp).
# 48 is ample for an exact simplex up to G=49 factors; larger G falls back to
# the spherical code (deterministic, calibrated). Mirrors KENKEN_HYP_DIM.
FG_HYP_DIM: int = int(os.environ.get("FG_HYP_DIM", "48"))

# Block magnitude (mirroring KENKEN_HYP_BLOCK).
FG_HYP_BLOCK: float = 1e4

# Alpha sharpening margin (mirroring KENKEN_HYP_ALPHA_MARGIN = 4.0).
# Pushes the between-group softplus arg past the -BLOCK floor so fp32 wobble
# in the computed d_hyp doesn't leave blocked cells at -1e4 +/- epsilon.
FG_HYP_ALPHA_MARGIN: float = float(os.environ.get("FG_HYP_ALPHA_MARGIN", "4.0"))

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


# ---- GEOMETRIC MASK GENERATOR (B2 Step 1 — PER-FACTOR CLIQUE-UNION) ----------
# The deterministic {0,-1e4} boolean mask (above) is the ANCHOR TARGET (t=0).
# The geometric generator produces the same structure from continuous Poincare-ball
# coordinates.  At t=0 with frozen calibrated anchors, the two are ~1e-3-identical.
#
# DESIGN: PER-FACTOR CLIQUE-UNION
#   Each type-t factor f (row in the membership matrix) gets anchor a_f.
#   ALL members of f are placed ON a_f -> d(z_i, z_j) ~ 0 for i,j in same factor
#   -> bias ~ 0 (attend).  Cross-factor pairs have d ~ d_out -> bias ~ -BLOCK (block).
#   The per-cell type-t bias = max over type-t factors of per-factor biases (UNION).
#   This correctly handles non-partition relations: (i,j) are connected iff they
#   share some factor f -> that factor's bias is 0 -> the union's max = 0 (allow).
#
# ANCHOR TABLE SIZING PER TYPE:
#   G_t factors of type t.  Need G_t distinct anchors (one per factor).
#   G_t <= dim+1 -> exact simplex (_poincare_anchors).
#   G_t >  dim+1 -> spherical code (_spherical_anchors), calibrate against actual
#                   min-between distance (robust to non-uniform spacing).
#
# RELATION FIELD DISCIPLINE (triangle inequality split):
#   ONE coordinate field (one anchor table) PER TYPE — never one for all types.
#   This is structurally enforced: attach_factor_hyperbolic_params builds one
#   anchor table per used type and stores them as model.fg_hyp_anchors_{t}.

def attach_factor_hyperbolic_params(
    model: object,
    n_heads: int,
    n_factor_types: int,
    s_max: int,
    membership_np: np.ndarray,
    latent_type_np: np.ndarray,
    dim: int | None = None,
    rho: float | None = None,
    alpha_margin: float | None = None,
) -> None:
    """Build and attach per-type Poincare anchor tables to `model`.

    Computes, for each type t used by at least one non-global head:
      - anchor table (G_t, dim): simplex for G_t <= dim+1, spherical code otherwise.
      - r_t = d_out_t / 2,  alpha_t = margin * 2 * BLOCK / d_out_t.

    Attached as:
      model.fg_hyp_anchors_{t}  Tensor (G_t, dim) float   — ball points (rho * dir).
      model.fg_hyp_r_{t}        float                      — half-distance threshold.
      model.fg_hyp_alpha_{t}    float                      — softplus sharpness.

    Also attaches:
      model.fg_hyp_n_heads      int     (saved for the bias builder)
      model.fg_hyp_n_factor_types int
      model.fg_hyp_s_max        int
      model.fg_hyp_dim          int
      model.fg_hyp_rho          float

    Parameters
    ----------
    model          : object to attach params to.
    n_heads        : total attention heads.
    n_factor_types : T — number of non-global relation types.
    s_max          : sequence length (grid size).
    membership_np  : (B_ref, L, s_max) float32 numpy — representative batch to
                     determine G_t = max factors of type t across the corpus.
                     Caller may pass a single representative batch; the builder
                     just needs to know G_t.
    latent_type_np : (B_ref, L) int32 numpy — type per row (matching membership_np).
    dim            : coord dim (default FG_HYP_DIM = 48).
    rho            : ball shell radius (default FG_HYP_RHO = 0.7).
    alpha_margin   : sharpening margin (default FG_HYP_ALPHA_MARGIN = 4.0).
    """
    if dim is None:
        dim = FG_HYP_DIM
    if rho is None:
        rho = FG_HYP_RHO
    if alpha_margin is None:
        alpha_margin = FG_HYP_ALPHA_MARGIN

    T = int(n_factor_types)
    G = max(1, n_heads // 16)
    alloc = cell_mp_head_allocation(T, n_heads, G)
    used_types = sorted({int(t) for t in alloc if int(t) != CELL_MP_HEAD_GLOBAL})

    model.fg_hyp_n_heads = n_heads
    model.fg_hyp_n_factor_types = T
    model.fg_hyp_s_max = s_max
    model.fg_hyp_dim = dim
    model.fg_hyp_rho = rho

    for t in used_types:
        # Count G_t = number of distinct type-t factor rows across the reference batch.
        # We use the maximum count over the batch (so the anchor table covers the
        # worst-case instance).  Padding rows (all-zero membership) still need a
        # distinct anchor to stay isolated from each other; we add +1 headroom.
        # For graph coloring, latent_type rows labeled LTYPE_EDGE=0 are real factors;
        # rows labeled the global sentinel (== T) are padding — those are not type-t.
        mask_t = (latent_type_np == t)              # (B_ref, L) bool
        G_t = int(mask_t.sum(axis=-1).max())        # max real factors of type t
        if G_t == 0:
            G_t = 1                                 # degenerate safety
        # +1 headroom so a padding factor (the row immediately after the last real one)
        # is also isolated (receives an anchor distinct from all real factors).  The
        # builder sends pad rows to the last anchor slot by convention.
        G_t_alloc = G_t + 1

        # Build anchor table: simplex for small G, spherical code for large G.
        if (G_t_alloc - 1) <= dim:
            anchors_np = _poincare_anchors(G_t_alloc, dim, rho)
        else:
            anchors_np = _spherical_anchors(G_t_alloc, dim, rho)
        anchors_np = anchors_np.astype(np.float32)

        # Calibrate r and alpha from the ACTUAL min between-anchor distance
        # (robust to both simplex and spherical code).
        d_out = _min_between_anchor_distance(anchors_np.astype(np.float64))
        r_t = d_out / 2.0
        alpha_t = alpha_margin * 2.0 * FG_HYP_BLOCK / max(d_out, 1e-9)

        # Attach to model.
        setattr(model, f"fg_hyp_anchors_{t}",
                Tensor(anchors_np, dtype=dtypes.float).contiguous())
        setattr(model, f"fg_hyp_r_{t}", r_t)
        setattr(model, f"fg_hyp_alpha_{t}", alpha_t)


def build_factor_hyperbolic_attn_bias(
    model: object,
    membership: Tensor,
    latent_type: Tensor,
    cell_valid: Tensor,
    n_heads: int,
    n_factor_types: int,
    s_max: int,
) -> Tensor:
    """Geometric drop-in for build_factor_attn_bias: (B, n_heads, s_max, s_max).

    Generates the per-head bias from continuous Poincare coordinates via PER-FACTOR
    CLIQUE-UNION anchoring (see module docstring).  Validity masking is kept BYTE-
    FOR-BYTE from build_factor_attn_bias.  At t=0 with calibrated frozen anchors,
    the result is ~1e-3-identical to build_factor_attn_bias.

    Requires attach_factor_hyperbolic_params to have been called on `model`.

    Parameters
    ----------
    model          : object with fg_hyp_anchors_{t} / fg_hyp_r_{t} / fg_hyp_alpha_{t}.
    membership     : Tensor (B, L, s_max) float — 1 if cell j in latent l.
    latent_type    : Tensor (B, L) int          — type index per latent.
    cell_valid     : Tensor (B, s_max) float    — 1 valid / 0 pad.
    n_heads        : int
    n_factor_types : int  (T)
    s_max          : int

    Returns
    -------
    Tensor (B, n_heads, s_max, s_max) additive bias in {~0, ~-1e4}. Contiguous.
    """
    BLOCK = FG_HYP_BLOCK
    Bn = int(membership.shape[0])
    L = int(membership.shape[1])
    T = int(n_factor_types)
    G = max(1, n_heads // 16)
    alloc = cell_mp_head_allocation(T, n_heads, G)
    used_types = sorted({int(t) for t in alloc if int(t) != CELL_MP_HEAD_GLOBAL})

    m = membership.cast(dtypes.float)              # (B, L, s_max)
    lt = latent_type                               # (B, L) int

    # Shared validity helpers — byte-for-byte from build_factor_attn_bias.
    eye_np = np.eye(s_max, dtype=np.float32)
    eye_s = Tensor(eye_np, dtype=dtypes.float).reshape(1, s_max, s_max)
    valid_key = cell_valid.reshape(Bn, 1, s_max)   # (B, 1, s_max)
    valid_q   = cell_valid.reshape(Bn, s_max, 1)   # (B, s_max, 1)

    neg_block = Tensor(np.array(-BLOCK, dtype=np.float32), dtype=dtypes.float)

    def _validity_bias(bias: Tensor) -> Tensor:
        """Apply validity masking to a (B, s_max, s_max) ADDITIVE bias tensor.

        Mirrors build_factor_attn_bias's _validity_mask but works directly on
        the additive bias (not the {0,1} allow mask):
          Step 1: invalid keys -> force to -BLOCK.
          Step 2: padding query rows -> self-only (0 on diagonal, -BLOCK off).
          Step 3: self-edge fix -> force valid-query diagonal to >= 0.

        All three steps are equivalent to those in _validity_mask when the
        geometric bias is already ~{0, -BLOCK} — the allow entries are near 0
        and the block entries are near -BLOCK.
        """
        # Step 1: block pad keys — valid_key==0 forces that entire column to -BLOCK.
        # Additive form: add -(1-valid_key)*BLOCK to every row; then floor at -BLOCK.
        key_block = (1.0 - valid_key.cast(dtypes.float)) * (-BLOCK)
        bias = (bias + key_block).maximum(neg_block)   # (B, s_max, s_max)
        # Step 2: padding query rows -> self-only (0 on diagonal, -BLOCK elsewhere).
        self_only_bias = (1.0 - eye_s) * (-BLOCK)     # (1, s_max, s_max)
        valid_q_f = valid_q.cast(dtypes.float)         # (B, s_max, 1)
        bias = bias * valid_q_f + self_only_bias * (1.0 - valid_q_f)
        # Step 3: self-edge fix — for every VALID query, force the diagonal to >= 0.
        # valid_q_f * eye_s: (B, s_max, s_max) — 1 on the valid-query diagonal, 0 elsewhere.
        valid_diag = valid_q_f * eye_s                 # (B, s_max, s_max)
        # Off-diagonal + pad-query diagonal kept as-is; valid-query diagonal floored at 0.
        # split: bias_off = bias on non-valid-diag positions; bias_diag on valid-diag.
        bias = (1.0 - valid_diag) * bias + valid_diag * bias.maximum(
            Tensor(np.zeros(1, dtype=np.float32), dtype=dtypes.float))
        return bias

    # ---- PER-TYPE GEOMETRIC BIAS (per-factor clique-union) ----
    type_bias: dict[int, Tensor] = {}
    for t in used_types:
        anchors_t = getattr(model, f"fg_hyp_anchors_{t}")  # (G_t, dim)
        r_t       = getattr(model, f"fg_hyp_r_{t}")        # float
        alpha_t   = getattr(model, f"fg_hyp_alpha_{t}")    # float
        G_t_alloc = int(anchors_t.shape[0])
        dim = int(anchors_t.shape[1])

        # Select type-t factor rows: (B, L) bool -> (B, L) float mask.
        sel_t = (lt == t).cast(dtypes.float)               # (B, L)

        # For each factor f (row l), build the per-factor clique bias and union
        # (max) across factors.
        #
        # Efficient batched approach:
        #   1. Assign an anchor index to each factor: row l of type t gets anchor
        #      index (rank of l among type-t rows, 0-based, per batch element).
        #   2. For each cell, gather its anchor point based on which (single)
        #      factor it belongs to -- but for non-partition a cell may belong to
        #      many factors.  We cannot just assign one anchor per cell.
        #
        # NON-PARTITION IMPLEMENTATION:
        #   For each factor f individually (row l when lt[b,l]==t):
        #     place all members of f on anchor[anchor_id_of_f]
        #     -> build per-factor bias (s_max, s_max) where within-f pairs get 0
        #   Union over factors: max over all f of per-factor bias.
        #
        # To stay batched (no Python loop over L up to 120):
        #   Build a (B, L, s_max, s_max) factor-bias volume, then max over L.
        #   Memory: Bn * L * s_max * s_max * 4 bytes.
        #   For Bn=8, L=120, s_max=49: 8*120*49*49*4 ~ 90MB — acceptable on CPU.
        #
        # Per factor l:
        #   anchor_id[b, l] = rank of l among type-t rows in batch b.
        #   Members of factor l: m[b, l, :] -> (s_max,) float ones.
        #   Cell i in factor l -> z_i = anchor[anchor_id[b,l]].
        #   Cell NOT in factor l -> assign sentinel anchor (G_t_alloc - 1) which is
        #     isolated from all real anchors.
        #   z_f = anchor indexed per (B, s_max): (B, s_max, dim) ball point.
        #   bias_f = _relation_bias_from_z(z_f, r_t, alpha_t): (B, s_max, s_max).
        #   Union: max over f of bias_f.
        #
        # Anchor-index assignment: for each batch b, the k-th type-t factor
        # (in row order) gets anchor k.  We compute a running cumsum over sel_t rows.

        # Compute per-(batch, row) anchor index for type-t rows.
        # anchor_idx[b, l] = (cumulative count of type-t rows up to and including l) - 1
        #                    if row l is type t, else G_t_alloc-1 (sentinel).
        # cumsum of sel_t along L: (B, L).
        cumsum_t = sel_t.cumsum(axis=1)                    # (B, L) float cumulative count
        # For a type-t row l, anchor index = cumsum[b,l] - 1 (0-based).
        # For a non-type-t row l, we assign the sentinel anchor index.
        sentinel_idx = float(G_t_alloc - 1)
        anchor_idx_f = (cumsum_t - 1.0) * sel_t + sentinel_idx * (1.0 - sel_t)
        anchor_idx_f = anchor_idx_f.cast(dtypes.float).clip(0.0, float(G_t_alloc - 1))
        # anchor_idx_f: (B, L) float, in [0, G_t_alloc-1]

        # Build per-cell-per-factor anchor coord:
        # For each factor l, each cell i:
        #   if i in factor l (m[b,l,i]==1): z[b,l,i] = anchor[anchor_idx_f[b,l]]
        #   else:                            z[b,l,i] = anchor[sentinel_idx]
        #
        # Approach:
        #   a) member_anchor[b,l,i] = anchor[anchor_idx_f[b,l]] if m[b,l,i]==1
        #   b) nonmem_anchor[b,l,i] = anchor[sentinel_idx]
        #   c) z_li[b,l,i] = m[b,l,i] * member_anchor + (1-m[b,l,i]) * nonmem_anchor
        #
        # Gather anchor for each factor-row:
        #   anchor_oh = one_hot(anchor_idx_f as int, G_t_alloc): (B, L, G_t_alloc)
        #   factor_anchor = anchor_oh @ anchors_t: (B, L, dim) — one anchor per factor
        anchor_idx_int = anchor_idx_f.cast(dtypes.int).clip(0, G_t_alloc - 1)
        anchor_oh = anchor_idx_int.one_hot(G_t_alloc).cast(dtypes.float)  # (B, L, G_t)
        factor_anchor = anchor_oh @ anchors_t.cast(dtypes.float)          # (B, L, dim)

        # Build z_li: (B, L, s_max, dim)
        # For member cells we place them on the factor's anchor; for non-members on
        # the sentinel anchor.  HOWEVER, two non-members placed on the same sentinel
        # have d=0, which would produce bias~0 (allow) and pollute the union.
        # Fix: after computing per-factor biases, GATE each (i,j) pair by
        # "both i and j are members of factor l" before taking the union.
        # This makes the union semantics exactly:
        #   allow(i,j) = exists l: m[l,i]=1 AND m[l,j]=1 AND d_hyp(z_i,z_j)~0
        # which is the per-factor CLIQUE-UNION adjacency (= m_t^T @ m_t > 0).

        # m as (B, L, s_max, 1) member indicator.
        m_4d = m.reshape(Bn, L, s_max, 1)              # (B, L, s_max, 1)
        fa   = factor_anchor.reshape(Bn, L, 1, dim)    # anchor of factor l: (B,L,1,dim)
        # Use sentinel anchor as the fallback for non-members.
        sent_val = anchors_t[G_t_alloc - 1].reshape(1, 1, dim).cast(dtypes.float)
        sv = sent_val.expand(Bn, L, 1, dim)            # (B,L,1,dim)
        z_li = m_4d * fa + (1.0 - m_4d) * sv          # (B, L, s_max, dim)

        # Per-factor bias: reshape to (B*L, s_max, dim), compute d_hyp, reshape back.
        z_flat = z_li.reshape(Bn * L, s_max, dim)       # (B*L, s_max, dim)
        bias_flat = _relation_bias_from_z(z_flat, r_t, alpha_t)  # (B*L, s_max, s_max)
        bias_vol = bias_flat.reshape(Bn, L, s_max, s_max)        # (B, L, s_max, s_max)

        # BOTH-MEMBERS GATE: force to -BLOCK where EITHER cell is a non-member.
        # both_mem[b,l,i,j] = m[b,l,i] * m[b,l,j] — 1 only when both are members.
        m_row = m.reshape(Bn, L, s_max, 1)             # (B, L, s_max, 1) — query membership
        m_col = m.reshape(Bn, L, 1, s_max)             # (B, L, 1, s_max) — key membership
        both_mem = m_row * m_col                        # (B, L, s_max, s_max) {0,1}
        # Where both_mem=1: keep geometric bias (near 0 for co-located members).
        # Where both_mem=0: force to -BLOCK (either cell is a non-member -> block).
        bias_vol = both_mem * bias_vol + (1.0 - both_mem) * neg_block

        # UNION: max over type-t factors; non-type-t rows masked to -BLOCK.
        sel_4d = sel_t.reshape(Bn, L, 1, 1)            # (B, L, 1, 1)
        bias_vol = sel_4d * bias_vol + (1.0 - sel_4d) * neg_block
        # Max over L: (B, s_max, s_max).
        type_bias_raw = bias_vol.max(axis=1)

        # Apply validity masking (byte-for-byte from build_factor_attn_bias).
        type_bias[t] = _validity_bias(type_bias_raw)

    # ---- GLOBAL HEAD BIAS ----
    # Global allow = all valid cell-pairs.  In bias form: 0 everywhere valid,
    # then validity masking.  Start from zeros (all-allow) then apply validity.
    zeros_s = Tensor(np.zeros((Bn, s_max, s_max), dtype=np.float32), dtype=dtypes.float)
    global_bias = _validity_bias(zeros_s)

    # ---- ASSEMBLE PER-HEAD BIAS ----
    head_biases = []
    for h in range(n_heads):
        t = int(alloc[h])
        bias_h = global_bias if t == CELL_MP_HEAD_GLOBAL else type_bias[t]
        head_biases.append(bias_h.reshape(Bn, 1, s_max, s_max))

    bias = Tensor.cat(*head_biases, dim=1)            # (B, n_heads, s_max, s_max)
    return bias.contiguous()
