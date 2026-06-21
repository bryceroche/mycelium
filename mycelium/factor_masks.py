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

RUNG-2 RELAXATION KNOBS (ported from kenken.py; default-off = byte-identical frozen)
-----------------------------------------------------------------------------------
FG_HYP_RELAX : int (env, default 0)
    0 => the proven FROZEN geometric mask: anchors stored as BALL POINTS, used RAW in
         d_hyp (NO exp_0), SATURATED alpha = margin*2*BLOCK/d_out -> exact {0,-1e4} mask.
    1 => RELAXATION: anchors stored as learnable TANGENT params (exp_0 maps them into the
         ball), SOFT-BLOCK alpha (see FG_HYP_RELAX_BLOCK_ARG) so the d_hyp BACKWARD has a
         non-vanishing coord gradient (Finding A), the §7 boundary guards become live.
FG_HYP_RELAX_BLOCK_ARG : float (env, default 20)
    The soft-block target (only when FG_HYP_RELAX=1). alpha = margin*target/r lands the
    between-group softplus arg at margin*target so BOTH tails stay open (gradient ~1e4
    not 0) while the leak exp(-arg) stays well under the 1e-3 attention-weight tolerance.
FG_HYP_JITTER : float (env, default 1e-3)
    Tiny tangent jitter on the relaxed anchor init (break init-degeneracy; t=0 mask still
    matches <1e-3). Applied ONLY under FG_HYP_RELAX.
FG_HYP_MAX_ZNORM : float (env, default 0.9)
    Rim guard: clamp_factor_hyp_tangent_norms shrinks |v| so |z|=tanh(|v|)<=this after
    each optimizer step (keeps 1/(1-|z|^2) bounded in the d_hyp backward).
FG_HYP_EUCLID : int (env, default 0)
    Capacity-matched control arm (spec §8.1): 1 => replace d_hyp with Euclidean ||u-v||
    over the SAME coord params (no exp_0), r/alpha recalibrated to the EUCLIDEAN anchor
    distances so the control ALSO reproduces the hard mask at t=0. To CLAIM geometry,
    hyperbolic must BEAT this control. 0 (default) => hyperbolic d_hyp.

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
    _relation_bias_from_coord_euclid,  # (c, r, alpha) -> (...,M,M) Tensor euclid bias
    _hyp_d_out,                  # (rho, G) -> float  closed-form d_out simplex
    _min_between_anchor_distance, # (anchors: np (A,dim)) -> float poincare min-d
    _min_between_anchor_distance_euclid,  # (anchors: np (A,dim)) -> float L2 min-d
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

# ---- RUNG-2 RELAXATION (the SOFT-BLOCK regime; ported from kenken.py) ----------
# docs spec §8 + Finding A. The FROZEN foothold calibrates alpha = margin*2*BLOCK/d_out
# with BLOCK=1e4 -> alpha ~ 1e4 -> the softplus is FULLY SATURATED on BOTH tails ->
# softplus'(±huge)=0 -> the d_hyp BACKWARD yields EXACTLY ZERO coord gradient (Finding A:
# a faithful sharp mask gives a VANISHING gradient -> 'relaxation does nothing' FALSE
# null). Lowering FG_HYP_ALPHA_MARGIN alone CANNOT fix this — alpha is dominated by the
# 2*BLOCK/d_out term, not the margin. The RELAX knob below recalibrates alpha so the
# between-group softplus ARG lands at a MODEST target (FG_HYP_RELAX_BLOCK_ARG): alpha =
# margin * target / r. At target~20 a blocked entry's bias is ~-margin*20, i.e. the
# attention leak is exp(-arg) (mask faithful WELL under the 1e-3 attention-weight
# tolerance) WHILE the softplus is in its responsive region so BOTH tails stay open and
# coord gradients flow. ONLY used when FG_HYP_RELAX is ON; the frozen foothold keeps the
# BLOCK-saturated calibration (byte-identical). r (= d_out/2) is unchanged in both modes.
#
# FG_HYP_RELAX: master gate. 0 (default) => frozen foothold (ball-point anchors used raw,
#   saturated alpha, NO exp_0) — byte-identical to the proven frozen geometric mask. 1 =>
#   the anchors become learnable TANGENT params (exp_0 maps them into the ball), the
#   soft-block alpha opens both softplus tails, and the §7 boundary guards become live.
FG_HYP_RELAX: int = int(os.environ.get("FG_HYP_RELAX", "0")) > 0
# The soft-block target (only consulted when FG_HYP_RELAX=1). Default 20 mirrors
# KENKEN_HYP_RELAX_BLOCK_ARG: leak ~ exp(-margin*20) << 1e-3, gradient ~1e4 not 0.
FG_HYP_RELAX_BLOCK_ARG: float = float(os.environ.get("FG_HYP_RELAX_BLOCK_ARG", "20.0"))
# Epsilon tangent jitter on the anchor init when relaxing (mirroring KENKEN_HYP_JITTER).
# Breaks any init-degeneracy between same-shape anchor tables so the relations can
# specialize. Tiny (1e-3) so the t=0 mask still matches well under 1e-3. ONLY applied
# under FG_HYP_RELAX; the frozen path stays the exact closed-form anchor (jitter=0).
FG_HYP_JITTER: float = float(os.environ.get("FG_HYP_JITTER", "1e-3"))
# Bounded-tangent-norm rim guard (mirroring KENKEN_HYP_MAX_ZNORM). After each optimizer
# step the trainer clamps |v| so |z|=tanh(|v|) stays <= this (keeps 1/(1-|z|^2) bounded
# in the d_hyp backward — the boundary-gradient landmine). atanh(0.9) ~ 1.4722.
FG_HYP_MAX_ZNORM: float = float(os.environ.get("FG_HYP_MAX_ZNORM", "0.9"))

# ---- EUCLIDEAN CONTROL ARM (spec §8.1 capacity-matched control; ported from kenken) --
# When ON: replace d_hyp(z_i,z_j) with the Euclidean ||u_i - u_j|| over the SAME coord
# params (identical shapes / param-count), so a capacity-matched Euclidean-vs-hyperbolic
# comparison is possible. To CLAIM geometry, hyperbolic must BEAT Euclidean (the v112b
# attribution control). The coord is used DIRECTLY as the Euclidean point (NO exp_0); r
# and alpha are recalibrated to the EUCLIDEAN anchor distances so the Euclidean arm ALSO
# reproduces the hard mask at t=0 (same zero-init discipline). The ONLY differences vs
# the hyperbolic arm are the distance (L2 vs arccosh) and the missing exp_0 — capacity is
# otherwise identical. Default 0 (hyperbolic, unchanged).
FG_HYP_EUCLID: int = int(os.environ.get("FG_HYP_EUCLID", "0")) > 0

# Sentinel: a head slot assigned to the global (union/all-valid) mask.
CELL_MP_HEAD_GLOBAL: int = -1


# ---- PER-RELATION BIAS ARM DISPATCH (ported from kenken._relation_bias_dispatch) -----
# A single seam so the clique-union path stays arm-agnostic. The factor-mask anchors are
# stored as BALL POINTS in the FROZEN path (used directly, NO exp_0) and as TANGENT params
# under RELAX (exp_0 maps them into the ball). The `is_tangent` flag selects which.
#
#   FG_HYP_EUCLID=0 (hyperbolic):
#     is_tangent=True  (relax) -> z = exp_0(coord), then d_hyp.
#     is_tangent=False (frozen)-> coord is ALREADY a ball point, used raw in d_hyp.
#   FG_HYP_EUCLID=1 (control): coord used DIRECTLY as the Euclidean point (NO exp_0),
#     d = ||u-v||. Identical shapes/param-count -> capacity-matched.
#
# NOTE the FROZEN default (FG_HYP_RELAX=0, FG_HYP_EUCLID=0) routes is_tangent=False ->
# coord raw -> d_hyp: the EXACT original computation, so the frozen geometric mask stays
# byte-identical. Only relax flips is_tangent=True (adding the exp_0 reparameterization).
def _factor_relation_bias_dispatch(coord: Tensor, r: float, alpha: float,
                                   is_tangent: bool) -> Tensor:
    """Dispatch one relation's per-factor bias by arm (hyperbolic / Euclidean).

    coord       : (..., M, dim) — a ball point (is_tangent=False) or a tangent param
                  (is_tangent=True). For the Euclidean arm the coord is used directly as
                  the Euclidean point regardless of is_tangent (NO exp_0).
    r, alpha    : the (arm-correct) calibrated softplus threshold / sharpness.
    is_tangent  : True under FG_HYP_RELAX (apply exp_0 for the hyperbolic arm); False for
                  the frozen ball-point path (use raw — byte-identical original).
    """
    if FG_HYP_EUCLID:
        return _relation_bias_from_coord_euclid(coord, r, alpha)
    z = _exp0_map(coord) if is_tangent else coord
    return _relation_bias_from_z(z, r, alpha)


def clamp_factor_hyp_tangent_norms(model: object,
                                   max_znorm: float = FG_HYP_MAX_ZNORM) -> None:
    """Rim guard (spec §7; ported from kenken.clamp_hyp_tangent_norms). Clamp each anchor
    row's tangent norm |v| so |z|=tanh(|v|) stays <= max_znorm (keeps 1/(1-|z|^2) bounded
    in the d_hyp backward — the boundary-gradient landmine). Call AFTER each optimizer
    step, ONLY in relax (FG_HYP_RELAX=1, where the anchors are tangent params).

    Per-ROW scaling: rows with |v| <= atanh(max_znorm) are untouched; longer rows are
    radially shrunk to the cap. In-place via .assign (mirrors the optimizer's own
    updates). JIT/AM-safe pure tensor ops (no float32 literal baked, no isnan loop).

    Iterates over every attached fg_hyp_anchors_{t} tangent table (one per used type).
    No-op when no such tables exist (frozen path attaches ball points, but the trainer
    only calls this under relax where they are tangents).
    """
    max_vnorm = float(math.atanh(min(max(max_znorm, 0.0), 1.0 - 1e-7)))
    t_idx = 0
    while True:
        anchors = getattr(model, f"fg_hyp_anchors_{t_idx}", None)
        if anchors is None:
            # Types are contiguous from 0 in the allocation, but a leading type may be
            # unused (T>R). Probe a few extra slots before giving up.
            if t_idx > 64:
                break
            t_idx += 1
            continue
        v = anchors.cast(dtypes.float)
        norm = (v.pow(2).sum(axis=-1, keepdim=True) + 1e-12).sqrt()           # (M,1)
        # scale = min(1, max_vnorm / |v|): only shrink rows past the cap.
        scale = (Tensor(np.array(max_vnorm, dtype=np.float32), dtype=dtypes.float)
                 / norm).minimum(Tensor(np.array(1.0, dtype=np.float32),
                                        dtype=dtypes.float))
        anchors.assign((v * scale).cast(anchors.dtype)).realize()
        t_idx += 1


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


def native_head_alloc_for_present_types(present_global_types: "list[int]",
                                        n_heads: int) -> np.ndarray:
    """PER-BATCH native head->GLOBAL-type allocation for the multi-task harness.

    THE HEAD-ALLOCATION FIX. The multi-task spec carries the union of all domains'
    global factor-types (T = N_GLOBAL_TYPES = 8). Deriving the head allocation from
    that union (cell_mp_head_allocation(8, 16, 1)) gives only ~2 heads per global
    type, so a PURE single-domain batch (e.g. coloring: only the edge type is
    present) gets just 2 of 16 heads on its one live relation — the other 13 heads
    are allocated to types ABSENT in the batch and sit DEAD. Native single-domain
    coloring (T=1) instead uses 15 edge-heads + 1 global. This crippled coloring
    (15 active heads -> 2) and it stalled at chance.

    The fix: because every multi-task batch is PURE single-domain, allocate the 16
    heads to ONLY the domain's PRESENT global types using the domain's NATIVE
    allocation. This is computed ON THE HOST per batch (the trainer knows the batch
    domain) and threaded into the mask builder as a tensor — NO data-dependent
    control flow inside the JIT graph, and NO per-domain weights (the Q/K/V heads
    are the SAME 16 across all domains; ONLY the per-head MASK assignment changes).

    Mapping: run cell_mp_head_allocation(P, n_heads, G) with P = number of PRESENT
    types -> a LOCAL allocation over 0..P-1 (e.g. coloring P=1 -> 15 local-0 + 1
    global; kenken/circuit P=3 -> 5/5/5 + 1 global). Then translate each local type
    index back to its GLOBAL type id via present_global_types[local_idx]. Global
    heads keep CELL_MP_HEAD_GLOBAL.

    Examples (n_heads=16, G=1):
      coloring present=[0]        -> P=1 -> 15 heads on GLOBAL type 0  + 1 global.
      kenken   present=[5,6,7]    -> P=3 -> 5 on 5, 5 on 6, 5 on 7     + 1 global.
      circuit  present=[1,2,3]    -> P=3 -> 5 on 1, 5 on 2, 5 on 3     + 1 global.

    Parameters
    ----------
    present_global_types : list of distinct GLOBAL type ids present in the batch
                           (sorted or not; order sets the contiguous-block order).
    n_heads              : total attention heads (16).

    Returns
    -------
    np.ndarray shape (n_heads,) int64 — each entry a GLOBAL type id or
    CELL_MP_HEAD_GLOBAL (= -1) for a global head.
    """
    present = list(present_global_types)
    P = len(present)
    assert P >= 1, "need at least one present global type"
    H = int(n_heads)
    G = max(1, H // 16)
    local_alloc = cell_mp_head_allocation(P, H, G)        # (H,) in {0..P-1, -1}
    out = np.full((H,), CELL_MP_HEAD_GLOBAL, dtype=np.int64)
    for h in range(H):
        lt = int(local_alloc[h])
        if lt == CELL_MP_HEAD_GLOBAL:
            continue
        out[h] = int(present[lt])
    return out


def head_alloc_to_tensors(head_global_type: np.ndarray, n_factor_types: int):
    """Convert a per-head GLOBAL-type allocation (host array) into the two JIT-stable
    tensors the multi-task mask builder consumes:

      head_type_oh : Tensor (H, T) float — one-hot over the T global types for each
                     NON-global head; an all-zero row marks a GLOBAL head.
      head_is_global : Tensor (H, 1, 1) float — 1.0 for a global head, else 0.0.

    Tensor (not python) form so the SAME JIT graph topology serves every domain —
    only these input tensors' VALUES change per batch (no recompile, no data-dependent
    python branch on domain inside the graph). T = N_GLOBAL_TYPES (the union spec),
    so the one-hot width is constant across domains.
    """
    H = int(head_global_type.shape[0])
    T = int(n_factor_types)
    oh = np.zeros((H, T), dtype=np.float32)
    isg = np.zeros((H, 1, 1), dtype=np.float32)
    for h in range(H):
        t = int(head_global_type[h])
        if t == CELL_MP_HEAD_GLOBAL:
            isg[h, 0, 0] = 1.0
        else:
            assert 0 <= t < T, f"head {h} global type {t} out of range [0,{T})"
            oh[h, t] = 1.0
    return (Tensor(oh, dtype=dtypes.float).contiguous(),
            Tensor(isg, dtype=dtypes.float).contiguous())


def build_factor_attn_bias_multitask(
    membership: Tensor,
    latent_type: Tensor,
    cell_valid: Tensor,
    head_type_oh: Tensor,
    head_is_global: Tensor,
    n_heads: int,
    n_factor_types: int,
    s_max: int,
) -> Tensor:
    """Multi-task per-head bias with a PER-BATCH (tensor-driven) head->type allocation.

    Same boolean {0,-1e4} mask + validity masking as build_factor_attn_bias, but the
    head->relation assignment is supplied as TENSORS (head_type_oh, head_is_global)
    instead of being derived from the union spec's n_factor_types. This realizes the
    head-allocation fix: a coloring batch's 15 edge-heads (vs 2) and a kenken batch's
    5/5/5 row/col/cage heads, with the SAME shared Q/K/V weights — only the mask
    assignment differs per batch.

    JIT-SAFETY: ONE graph topology for all domains. All T global-type adjacencies are
    built (a python loop over the fixed T, no host-data branch); the per-head selection
    is a tensor weighted-sum over head_type_oh (absent types contribute 0 because no
    head one-hots onto them). No data-dependent python control flow; the head count and
    T are compile-time constants (fixed shapes).

    Parameters
    ----------
    membership     : Tensor (B, L, s_max) float — 1 if cell j in latent l.
    latent_type    : Tensor (B, L) int          — GLOBAL type id per latent.
    cell_valid     : Tensor (B, s_max) float    — 1 valid / 0 pad.
    head_type_oh   : Tensor (n_heads, T) float  — per-head one-hot over global types
                     (all-zero row => global head).
    head_is_global : Tensor (n_heads, 1, 1) f   — 1.0 for a global head, else 0.0.
    n_heads        : int
    n_factor_types : int (T = N_GLOBAL_TYPES, the union width).
    s_max          : int

    Returns
    -------
    Tensor (B, n_heads, s_max, s_max) with values in {0, -1e4}. Contiguous.
    """
    Bn = int(membership.shape[0])
    T = int(n_factor_types)

    m = membership.cast(dtypes.float)               # (B, L, s_max)
    lt = latent_type                                 # (B, L) int

    eye_np = np.eye(s_max, dtype=np.float32)
    eye_s = Tensor(eye_np, dtype=dtypes.float).reshape(1, s_max, s_max)
    valid_key = cell_valid.reshape(Bn, 1, s_max)    # (B, 1, s_max)
    valid_q   = cell_valid.reshape(Bn, s_max, 1)    # (B, s_max, 1)

    def _validity_mask(allow: Tensor) -> Tensor:
        # Byte-for-byte from build_factor_attn_bias._validity_mask.
        allow = allow * valid_key.cast(allow.dtype)
        allow = (allow * valid_q.cast(allow.dtype)
                 + eye_s * (1.0 - valid_q.cast(allow.dtype)))
        allow = allow.maximum(eye_s * valid_q.cast(allow.dtype))
        return allow

    # Per-type allow for EVERY global type t in 0..T-1 (python loop over the fixed,
    # compile-time-constant T — no host-data branch). Absent-in-this-batch types
    # produce an all-block adjacency, but no head one-hots onto them so they are never
    # selected; they are still built so the graph topology is identical across domains.
    type_allows = []
    for t in range(T):
        sel = (lt == t).cast(dtypes.float).reshape(Bn, int(lt.shape[1]), 1)  # (B,L,1)
        m_t = m * sel                               # (B, L, s_max) type-t members
        adj_t = m_t.transpose(1, 2) @ m_t          # (B, s_max, s_max) co-occurrence
        allow_t = _validity_mask(adj_t.clip(0.0, 1.0))                       # (B,S,S)
        type_allows.append(allow_t.reshape(Bn, 1, s_max, s_max))
    type_allow_stack = Tensor.cat(*type_allows, dim=1)   # (B, T, S, S)

    # Global allow: all valid cell-pairs, then validity masking.
    ones_s = Tensor(np.ones((Bn, s_max, s_max), dtype=np.float32), dtype=dtypes.float)
    global_allow = _validity_mask(ones_s).reshape(Bn, 1, s_max, s_max)        # (B,1,S,S)

    # Per-head selection (TENSOR-DRIVEN): allow_h = sum_t head_type_oh[h,t]*type_allow[t],
    # then OR-in the global allow for global heads. head_type_oh: (H,T). Einsum over T.
    #   sel_allow[b,h,i,j] = sum_t head_type_oh[h,t] * type_allow_stack[b,t,i,j]
    # Reshape for broadcast: (1,H,T,1,1) * (B,1,T,S,S) -> sum over T.
    H = int(n_heads)
    oh = head_type_oh.reshape(1, H, T, 1, 1)                                  # (1,H,T,1,1)
    ta = type_allow_stack.reshape(Bn, 1, T, s_max, s_max)                     # (B,1,T,S,S)
    sel_allow = (oh * ta).sum(axis=2)                                         # (B,H,S,S)

    # Global heads: head_is_global (H,1,1) -> (1,H,1,S,S) broadcast picks global_allow.
    isg = head_is_global.reshape(1, H, 1, 1)                                  # (1,H,1,1)
    glob = global_allow.reshape(Bn, 1, s_max, s_max)                          # (B,1,S,S)
    allow = isg * glob + (1.0 - isg) * sel_allow                              # (B,H,S,S)

    bias = (1.0 - allow) * (-1e4)
    return bias.contiguous()


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
      - r_t = d_out_t / 2.
      - alpha_t = margin * 2 * BLOCK / d_out_t (FROZEN, saturated) OR
                  margin * RELAX_BLOCK_ARG / r_t (RELAX, soft-block — Finding A).

    Attached as:
      model.fg_hyp_anchors_{t}  Tensor (G_t, dim) float   — BALL POINTS (rho*dir) when
                                FG_HYP_RELAX=0 (used raw, byte-identical frozen mask); or
                                TANGENT params (exp_0(v)==ball points, +jitter) when
                                FG_HYP_RELAX=1 (learnable; the trainer adds them to the
                                optimizer and clamps their norm via the §7 rim guard).
      model.fg_hyp_r_{t}        float                      — half-distance threshold.
      model.fg_hyp_alpha_{t}    float                      — softplus sharpness (arm- and
                                mode-aware: euclid recalibrates to L2 distances; relax
                                uses the soft-block target).

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

        # Build anchor table (BALL POINTS): simplex for small G, spherical code for large.
        if (G_t_alloc - 1) <= dim:
            anchors_np = _poincare_anchors(G_t_alloc, dim, rho)
        else:
            anchors_np = _spherical_anchors(G_t_alloc, dim, rho)
        anchors_ball = anchors_np.astype(np.float32)                    # (G_t, dim) ball

        # ---- RUNG-2 STORAGE + CALIBRATION (arm- and mode-aware) ----
        # FROZEN (FG_HYP_RELAX=0): store the BALL POINTS and use them RAW in the bias
        #   builder (is_tangent=False, NO exp_0) — the EXACT original computation -> the
        #   frozen geometric mask stays byte-identical.
        # RELAX (FG_HYP_RELAX=1): store TANGENT params v s.t. exp_0(v) == the ball points
        #   (is_tangent=True -> the bias builder applies exp_0). The tangent
        #   parameterization keeps |z|=tanh(|v|)<1 automatically and is what the §7
        #   rim-guard clamp shrinks — the boundary-gradient landmine is live now.
        if FG_HYP_RELAX:
            coords_np = _tangent_for_anchors(
                anchors_ball.astype(np.float64)).astype(np.float32)    # (G_t, dim) tangent
            # tiny tangent jitter (break init-degeneracy across same-shape tables so the
            # relations can specialize). Deterministic seed so a run reproduces; tiny
            # (1e-3) so the t=0 mask still matches well under 1e-3.
            if FG_HYP_JITTER > 0.0:
                rng_j = np.random.RandomState(20260616 + int(t))
                coords_np = (coords_np + FG_HYP_JITTER
                             * rng_j.randn(*coords_np.shape).astype(np.float32))
        else:
            coords_np = anchors_ball                                   # ball points (raw)

        # Calibrate r/alpha from the ACTUAL min between-anchor distance (robust to both
        # simplex and spherical code), in the ARM-CORRECT metric so BOTH arms reproduce
        # the hard mask at t=0:
        #   HYPERBOLIC arm: d_out = Poincare min-distance over the BALL POINTS (exp_0 of
        #     the stored tangents under relax round-trips to these exact ball points, so
        #     the ball-point d_out is correct for the relaxed coord too).
        #   EUCLIDEAN  arm: d_out = L2 min-distance over the COORD AS STORED (the coord IS
        #     the Euclidean point — no exp_0): tangent separation under relax, ball-point
        #     separation when frozen. SAME shapes -> capacity-matched control.
        if FG_HYP_EUCLID:
            d_out = _min_between_anchor_distance_euclid(coords_np.astype(np.float64))
        else:
            d_out = _min_between_anchor_distance(anchors_ball.astype(np.float64))
        r_t = d_out / 2.0
        # alpha: SATURATED (frozen, exact {0,-1e4} mask) vs SOFT-BLOCK (relax, gradient
        # flows). The frozen branch is the original 2*BLOCK/d_out; the relax branch lands
        # the between-group softplus arg at margin*RELAX_BLOCK_ARG (Finding A — unsaturate
        # the BLOCK-dominated alpha so BOTH tails stay open). r is unchanged in both modes.
        if FG_HYP_RELAX:
            alpha_t = alpha_margin * FG_HYP_RELAX_BLOCK_ARG / max(r_t, 1e-9)
        else:
            alpha_t = alpha_margin * 2.0 * FG_HYP_BLOCK / max(d_out, 1e-9)

        # Attach to model.
        setattr(model, f"fg_hyp_anchors_{t}",
                Tensor(coords_np, dtype=dtypes.float).contiguous())
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

        # Per-factor bias: reshape to (B*L, s_max, dim), compute the per-relation bias,
        # reshape back. ARM DISPATCH (Rung-2): FROZEN -> ball points used raw in d_hyp
        # (is_tangent=False -> NO exp_0 -> the EXACT original computation, byte-identical).
        # RELAX -> the coord is a TANGENT param: hyperbolic arm applies exp_0(z_flat) then
        # d_hyp; Euclidean arm (FG_HYP_EUCLID=1) uses z_flat DIRECTLY as the L2 point (no
        # exp_0). The non-member sentinel placement is preserved (the both-members gate
        # below blocks any non-member pair regardless of the geometric value).
        z_flat = z_li.reshape(Bn * L, s_max, dim)       # (B*L, s_max, dim)
        bias_flat = _factor_relation_bias_dispatch(
            z_flat, r_t, alpha_t, is_tangent=FG_HYP_RELAX)  # (B*L, s_max, s_max)
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
