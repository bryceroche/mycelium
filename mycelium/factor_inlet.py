"""factor_inlet.py — Generic semantics-as-input inlet (the neural predicate registry).

THE GENERAL-WEIGHTS THESIS, INLET HALF
--------------------------------------
The validated v98/KenKen executor distinguishes problem domains by the INPUT it is
fed (membership topology + the per-factor semantics), NOT by separate weights. Today
ONLY KenKen feeds per-factor semantics, through a KenKen-specific verification inlet
that lives in kenken.py (THE REGRESSION ORACLE — never modified). This module is the
GENERIC counterpart: ONE inlet channel through which EVERY domain (coloring, circuit,
KenKen) feeds its per-factor semantics, so a single DENSE shared backbone can tell
domains / factor-types apart FROM INPUT.

WHAT THE INLET CARRIES
----------------------
Per LATENT (factor) f, a feature vector:
  - a GLOBAL factor-type id  (coloring-edge=0, circuit-AND/OR/NOT=1/2/3,
    kenken-row/col/cage=4/5/6 — distinct ids so the shared weights separate them),
  - OPTIONAL arithmetic params (op / target-bucket / size) — KenKen only; the other
    domains pass param ids = 0 (a dedicated "no-param" slot in each table).
The per-latent feature is projected to H, SCATTERED to its member cells via the
membership matrix, RMSNorm-bounded, and returned (B, s_max, H). The caller adds it to
the residual EVERY breath (factor_breathing_forward, live at step 0 — non-zero init).

WHY A MEMBERSHIP-SCATTER (not the cage-id scatter in kenken.py)
--------------------------------------------------------------
kenken.py scatters per-CAGE via cell_cage_id (a partition: each cell is in exactly one
cage). The generic inlet scatters per-LATENT via the membership matrix `membership`
(B, L, s_max), which is domain-agnostic and handles NON-partition relations (a cell is
in many factors: its row, its col, its cage; or many circuit gates). For a cell in
several factors we SUM their projected features then RMSNorm — the norm bounds the
magnitude regardless of how many factors a cell belongs to.

SUBSTRATE (tinygrad + AMD AM driver)
------------------------------------
* No dtypes.float32 Tensor literal baked as a JIT graph constant — numpy intermediates
  then wrap once; the {0,-1e4}-style consts are Python floats multiplied in-graph.
* All embedding lookups are one_hot @ table (tinygrad-safe), never integer-index gather.
* RMSNorm is a single-kernel reduction; pad cells re-zeroed after norm.
* This module imports NOTHING from kenken.py (the oracle stays untouched).

PUBLIC API
----------
GLOBAL_TYPE_IDS : dict[str, int]      — the canonical factor-type id registry.
N_GLOBAL_TYPES  : int                 — len(GLOBAL_TYPE_IDS) (= 7 for the Rung-1 mix).
attach_factor_inlet_params(model, hidden, ...) -> None
    Allocate the shared inlet tables (type_embed + op/target/size embeds + W/b proj).
factor_inlet_param_names() -> list[str]
    The attribute names this module attaches (for checkpoint save/load).
factor_inlet_parameters(model) -> list[Tensor]
    The trainable inlet params (for the optimizer).
build_generic_factor_inlet(model, membership, latent_type, cell_valid,
                           op=None, target=None, size=None) -> Tensor (B, s_max, H)
    Build the per-cell inlet from per-latent semantics. op/target/size are KenKen-only
    per-latent param ids (B, L) int; None => the inlet carries the type id alone.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
from tinygrad import Tensor, dtypes


# ---------------------------------------------------------------------------
# Global factor-type id registry (the neural predicate registry's vocabulary).
# Distinct ids so the shared backbone separates every domain's factor-types FROM
# INPUT. Keep this in lock-step with the per-domain latent_type REMAP the adapters
# apply (scripts/factor_graph_train.py multi-task adapters).
# ---------------------------------------------------------------------------

GLOBAL_TYPE_IDS: dict[str, int] = {
    "coloring_edge": 0,
    "circuit_and":   1,
    "circuit_or":    2,
    "circuit_not":   3,
    "circuit_xor":   4,
    "kenken_row":    5,
    "kenken_col":    6,
    "kenken_cage":   7,
}
# +1 slot for the padding / global sentinel (a latent that is a pad row); its inlet
# contribution is zeroed by the membership being all-zero, but we still need a valid
# embedding row to index, so size the type table to N_GLOBAL_TYPES + 1.
N_GLOBAL_TYPES: int = len(GLOBAL_TYPE_IDS)
TYPE_TABLE_ROWS: int = N_GLOBAL_TYPES + 1     # + sentinel row at index N_GLOBAL_TYPES

# KenKen arithmetic-param vocab sizes (mirror kenken_data without importing it, so
# this module never depends on the oracle). These MUST match kenken_data's constants.
INLET_N_OPS: int = 5            # OP_VOCAB = [given, add, sub, mul, div]
INLET_TARGET_BUCKETS: int = 32  # log-bucketed cage target
INLET_MAX_CAGE_SIZE: int = 8    # cage-size one-hot width

# Per-table embedding dims (mirror the KenKen verification inlet defaults).
_D_TYPE: int = 16
_D_OP: int = 32
_D_TGT: int = 64
_D_SIZE: int = 16


# ---------------------------------------------------------------------------
# RMSNorm (no learnable weight) — mirrors kenken.py:_rmsnorm_last verbatim so the
# inlet contribution's magnitude is bounded identically. Re-implemented here (not
# imported) to keep this module independent of the oracle.
# ---------------------------------------------------------------------------

def _rmsnorm_last(x: Tensor, eps: float = 1e-5) -> Tensor:
    """RMSNorm over the last axis (no learnable weight)."""
    x32 = x.cast(dtypes.float)
    rms = (x32.pow(2).mean(axis=-1, keepdim=True) + eps).sqrt()
    return (x32 / rms)


# ---------------------------------------------------------------------------
# Param attach
# ---------------------------------------------------------------------------

_INLET_PARAM_NAMES = [
    "fg_inlet_type_embed",
    "fg_inlet_op_embed",
    "fg_inlet_target_embed",
    "fg_inlet_size_embed",
    "fg_inlet_w",
    "fg_inlet_b",
    "fg_inlet_gate",
]


def factor_inlet_param_names() -> list[str]:
    """Attribute names attach_factor_inlet_params sets (for ckpt save/load)."""
    return list(_INLET_PARAM_NAMES)


def attach_factor_inlet_params(model: Any, hidden: int,
                               d_type: int = _D_TYPE, d_op: int = _D_OP,
                               d_tgt: int = _D_TGT, d_size: int = _D_SIZE,
                               seed: int = 1408) -> None:
    """Allocate the SHARED generic-inlet tables on `model`.

    These are the ONLY inlet params used in the multi-task path — coloring, circuit
    and KenKen all read the SAME W/b projection and the SAME type/op/target/size
    embedding tables. The domain distinction is purely the type_id (and, for KenKen,
    the op/target/size ids) fed as INPUT.

    The inlet is LIVE at init (added to the residual every breath, not behind a
    zero-init gate), so the projection W is Xavier-scaled (non-zero) and the bias is
    zero; RMSNorm bounds the post-projection magnitude so it cannot dominate the
    residual scale at step 0.

    Attributes added
    ----------------
    fg_inlet_type_embed   (TYPE_TABLE_ROWS, d_type)   — global factor-type embedding.
    fg_inlet_op_embed     (INLET_N_OPS, d_op)         — KenKen op embedding.
    fg_inlet_target_embed (INLET_TARGET_BUCKETS, d_tgt) — KenKen target-bucket embedding.
    fg_inlet_size_embed   (INLET_MAX_CAGE_SIZE, d_size) — KenKen cage-size embedding.
    fg_inlet_w            (d_type+d_op+d_tgt+d_size, hidden) — projection.
    fg_inlet_b            (hidden,)                   — projection bias (zero init).
    fg_inlet_gate         (N_GLOBAL_TYPES,)           — per-factor-type inlet gate,
                          ZERO-INIT. The inlet contribution of each latent is scaled by
                          gate[its global type] BEFORE the membership scatter. At init
                          (gate=0) the gated inlet contributes EXACTLY 0 -> the forward
                          is byte-identical to inlet-OFF (the bootstrap-safe zero-init
                          pattern, cf the notebook/cathedral W_o zero-init). Each type's
                          gate opens independently ONLY if that type's inlet earns
                          gradient (KenKen cages: discriminative -> opens; coloring
                          edges: constant DC -> stays ~0, never re-harming coloring).
    """
    rng = np.random.RandomState(seed)
    model.fg_inlet_type_embed = Tensor(
        (rng.randn(TYPE_TABLE_ROWS, d_type) * 0.1).astype(np.float32),
        dtype=dtypes.float).contiguous()
    model.fg_inlet_op_embed = Tensor(
        (rng.randn(INLET_N_OPS, d_op) * 0.1).astype(np.float32),
        dtype=dtypes.float).contiguous()
    model.fg_inlet_target_embed = Tensor(
        (rng.randn(INLET_TARGET_BUCKETS, d_tgt) * 0.1).astype(np.float32),
        dtype=dtypes.float).contiguous()
    model.fg_inlet_size_embed = Tensor(
        (rng.randn(INLET_MAX_CAGE_SIZE, d_size) * 0.1).astype(np.float32),
        dtype=dtypes.float).contiguous()
    d_total = d_type + d_op + d_tgt + d_size
    w_scale = 1.0 / math.sqrt(d_total)
    model.fg_inlet_w = Tensor(
        (rng.randn(d_total, hidden) * w_scale).astype(np.float32),
        dtype=dtypes.float).contiguous()
    model.fg_inlet_b = Tensor.zeros((hidden,), dtype=dtypes.float).contiguous()
    # Per-factor-type inlet gate — ZERO-INIT. One scalar per global factor type. At
    # init the gated inlet contributes EXACTLY 0 (byte-identical to inlet-OFF); each
    # type's gate opens independently only if that type's inlet earns gradient.
    model.fg_inlet_gate = Tensor.zeros((N_GLOBAL_TYPES,), dtype=dtypes.float).contiguous()


def factor_inlet_parameters(model: Any) -> list[Tensor]:
    """Trainable generic-inlet params (for the optimizer)."""
    return [getattr(model, nm) for nm in _INLET_PARAM_NAMES]


# ---------------------------------------------------------------------------
# Generic inlet build
# ---------------------------------------------------------------------------

def build_generic_factor_inlet(model: Any,
                               membership: Tensor,
                               latent_type: Tensor,
                               cell_valid: Tensor,
                               op: "Tensor | None" = None,
                               target: "Tensor | None" = None,
                               size: "Tensor | None" = None) -> Tensor:
    """Build the per-CELL generic semantics inlet, (B, s_max, H).

    Per LATENT (factor) f, encode the GLOBAL type-id (+ optional KenKen op/target/size
    ids), project to H, SCATTER to f's member cells via the membership matrix, SUM
    over factors a cell belongs to, RMSNorm, and return (zeros on pad cells).

    Parameters
    ----------
    model       : object with the fg_inlet_* tables (attach_factor_inlet_params).
    membership  : Tensor (B, L, s_max) float — 1 if cell j is in latent l.
    latent_type : Tensor (B, L) int          — GLOBAL factor-type id per latent.
                  Pad rows carry the sentinel id (>= N_GLOBAL_TYPES); their membership
                  is all-zero so they scatter nothing regardless of the embedding row.
    cell_valid  : Tensor (B, s_max) float    — 1 valid / 0 padding.
    op          : Tensor (B, L) int | None   — KenKen per-latent op id (0..4). None =>
                  the op feature is the table's row 0 ("given"/no-op) for every latent
                  (coloring/circuit carry no arithmetic).
    target      : Tensor (B, L) int | None   — KenKen per-latent target bucket id.
    size        : Tensor (B, L) int | None   — KenKen per-latent cage size.

    Returns
    -------
    Tensor (B, s_max, H) float — the RMSNorm'd inlet (zeros on pad cells).

    NOTE on the membership-scatter vs the kenken cage-id scatter
    ------------------------------------------------------------
    kenken.py:build_verification_inlet scatters per CAGE via cell_cage_id (a clean
    partition). This generic inlet scatters per LATENT via the membership matrix, so
    it handles non-partition relations (a cell in its row AND col AND cage, or in many
    circuit gates): cell_inlet[b,j] = sum_l membership[b,l,j] * latent_feat[b,l]. The
    RMSNorm then bounds the magnitude regardless of how many latents a cell joins.
    """
    type_embed = model.fg_inlet_type_embed       # (TYPE_TABLE_ROWS, d_type)
    op_embed   = model.fg_inlet_op_embed         # (INLET_N_OPS, d_op)
    tgt_embed  = model.fg_inlet_target_embed     # (INLET_TARGET_BUCKETS, d_tgt)
    sz_embed   = model.fg_inlet_size_embed       # (INLET_MAX_CAGE_SIZE, d_size)
    W_inlet    = model.fg_inlet_w                 # (d_total, H)
    b_inlet    = model.fg_inlet_b                 # (H,)

    B = int(membership.shape[0])
    L = int(membership.shape[1])
    S = int(membership.shape[2])
    H = int(W_inlet.shape[-1])

    m = membership.cast(dtypes.float)            # (B, L, S)

    # ---- per-latent type embedding (clip the sentinel into the table). ----
    lt = latent_type.clip(0, TYPE_TABLE_ROWS - 1)                 # (B, L)
    type_oh = lt.one_hot(TYPE_TABLE_ROWS).cast(type_embed.dtype)  # (B, L, TYPE_TABLE_ROWS)
    type_e  = type_oh @ type_embed                                # (B, L, d_type)

    # ---- per-latent KenKen arithmetic params (row-0 default when None). ----
    # When a param is None (coloring/circuit), use a zeros (B, L) int tensor -> row 0
    # of the table for every latent (the table's row 0 is the "no-param" slot). This
    # is a dynamic Tensor.zeros (not a baked float32 const) so the JIT graph stays
    # AM-driver-safe.
    if op is not None:
        op_ids = op.clip(0, INLET_N_OPS - 1)
    else:
        op_ids = Tensor.zeros((B, L), dtype=dtypes.int)
    op_oh = op_ids.one_hot(INLET_N_OPS).cast(op_embed.dtype)       # (B, L, INLET_N_OPS)
    op_e  = op_oh @ op_embed                                       # (B, L, d_op)

    if target is not None:
        tgt_ids = target.clip(0, INLET_TARGET_BUCKETS - 1)
    else:
        tgt_ids = Tensor.zeros((B, L), dtype=dtypes.int)
    tgt_oh = tgt_ids.one_hot(INLET_TARGET_BUCKETS).cast(tgt_embed.dtype)
    tgt_e  = tgt_oh @ tgt_embed                                    # (B, L, d_tgt)

    if size is not None:
        sz_ids = size.clip(0, INLET_MAX_CAGE_SIZE - 1)
    else:
        sz_ids = Tensor.zeros((B, L), dtype=dtypes.int)
    sz_oh = sz_ids.one_hot(INLET_MAX_CAGE_SIZE).cast(sz_embed.dtype)
    sz_e  = sz_oh @ sz_embed                                       # (B, L, d_size)

    # ---- per-latent feature -> project to H. ----
    latent_feat = Tensor.cat(type_e, op_e, tgt_e, sz_e, dim=-1)   # (B, L, d_total)
    latent_inlet = latent_feat @ W_inlet + b_inlet                # (B, L, H)

    # ---- ZERO-INIT per-factor-type gate (the bootstrap-safe fix). ----
    # Each latent's projected inlet is scaled by gate[its global type] BEFORE the
    # membership scatter, so a coloring cell's edge-inlet is scaled by
    # gate[coloring_edge], a kenken cell's cage-inlet by gate[cage], etc. Gates are
    # INDEPENDENT per type, so KenKen's gate opening does NOT re-harm coloring. At
    # init the gate is all-zero -> the gated inlet is EXACTLY 0 -> the whole inlet
    # contribution vanishes -> byte-identical to inlet-OFF (and to native).
    #
    # The gate table is (N_GLOBAL_TYPES,); we append a single non-trainable 0 sentinel
    # slot so the one-hot over TYPE_TABLE_ROWS (= N_GLOBAL_TYPES + 1) lines up. The
    # sentinel slot is a dynamic Tensor.zeros (NOT a baked float32 const) so the JIT
    # graph stays AM-driver-safe; pad latents carry the sentinel id and all-zero
    # membership, so they scatter nothing regardless.
    inlet_gate = getattr(model, "fg_inlet_gate", None)
    if inlet_gate is not None:
        gate_full = Tensor.cat(
            inlet_gate.cast(latent_inlet.dtype),
            Tensor.zeros((1,), dtype=latent_inlet.dtype),       # sentinel slot = 0
            dim=0,
        )                                                        # (TYPE_TABLE_ROWS,)
        # Per-latent gate via the same type one-hot used for the type embedding:
        # (B, L, TYPE_TABLE_ROWS) @ (TYPE_TABLE_ROWS,) -> (B, L).
        gate_per_latent = (type_oh.cast(latent_inlet.dtype) @ gate_full)  # (B, L)
        latent_inlet = latent_inlet * gate_per_latent.reshape(B, L, 1)

    # ---- scatter latent -> cell via membership: (B, S, L) @ (B, L, H) -> (B, S, H).
    # A cell in several latents SUMS their projected features. Pad latents have
    # all-zero membership rows so they scatter nothing (their embedding row is moot).
    cell_inlet = m.transpose(1, 2) @ latent_inlet                 # (B, S, H)

    # ---- RMSNorm + re-zero pad cells. ----
    cell_valid_col = cell_valid.cast(dtypes.float).reshape(B, S, 1)
    cell_inlet = cell_inlet * cell_valid_col                      # zero pad before norm
    inlet_norm = _rmsnorm_last(cell_inlet)                        # (B, S, H)
    inlet_norm = inlet_norm * cell_valid_col                      # re-zero pad after norm
    return inlet_norm.cast(dtypes.float)
