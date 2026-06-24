"""factor_graph_engine.py — General deep executor for typed factor graphs.

Parameterizes the v98 KenKen breathing executor (mycelium/kenken.py:
`kenken_breathing_forward`) for an arbitrary typed factor graph.

DESIGN PRINCIPLE
----------------
kenken.py is the regression oracle — it is NEVER touched. This module:
  - IMPORTS the general helpers from kenken.py (kenken_layer_forward,
    embed_kenken's pattern, _layernorm, codebook_ortho_penalty, ...).
  - RE-IMPLEMENTS only the thin coupled pieces (mask call, inlet plug,
    shape constants) with the domain-specific bits parameterized.
  - Is BYTE-IDENTICAL to the kenken path when driven with KenKen inputs at
    matching hyperparameters (verified by the Step-3 GPU anchor).

PUBLIC API
----------
FactorGraphSpec
    Hyperparameter bundle (s_max, n_values, n_factor_types, n_heads, k_max,
    has_factor_inlet).

FactorGraphBatch
    Attribute contract that factor_breathing_forward reads from the batch
    object.  A KenKenBatch satisfies it (for the KenKen accuracy anchor).

factor_breathing_forward(model, batch, spec, K)
    -> (value_logits_history, calib_history)
    K-breath loop — byte-identical to kenken_breathing_forward save for the
    two coupled call-sites (mask builder + inlet plug).

attach_factor_graph_params(model, hidden, spec)
    Allocate all factor-graph params on `model`.  No fixed mask (masks are
    per-batch from membership).  Position embed is a plain learned (s_max, H).

factor_loss(value_logits_history, calib_history, batch, spec, **weights)
    Per-breath weighted-CE ladder + optional constraint-energy plug +
    calibration.  Parameterized on N (= spec.n_values) and s_max.

SUBSTRATE RULES (tinygrad + AMD AM driver)
------------------------------------------
* No dtypes.float32 literal baked as a JIT graph constant — use numpy
  intermediates then wrap.
* scores.clip(-1e4, 1e4) for attention numerical stability (already inside
  kenken_layer_forward; the general loop just threads attn_bias through it).
* Single-kernel isfinite for NaN guards (no per-element checks inside JIT).
* No host sync (.realize() / .numpy()) inside the breath loop.
* Mirror kenken.py patterns exactly — every deviation is a deliberate
  parameterization, documented inline.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from tinygrad import Tensor, dtypes

from mycelium.factor_masks import (
    build_factor_attn_bias,
    build_factor_attn_bias_multitask,
    build_factor_hyperbolic_attn_bias,
    FG_HYP_MASK,
)
# Import the general helpers that kenken.py already exposes:
from mycelium.kenken import (
    kenken_layer_forward,    # (layer, x, attn_bias, cos, sin) -> x  — general S
    codebook_ortho_penalty,  # (codebook) -> scalar  — domain-free
)

# FG_HYP_FREEZE: when FG_HYP_MASK=1, freeze the hyperbolic anchor params
# (default 1 = frozen).  Set FG_HYP_FREEZE=0 for Step 3 relaxation.
FG_HYP_FREEZE: bool = int(os.environ.get("FG_HYP_FREEZE", "1")) > 0


# ---------------------------------------------------------------------------
# IN-DEDUCER WAIST (FG_WAIST) — additive, zero-init-gated convex-blend bottleneck.
# ---------------------------------------------------------------------------
# An OPTIONAL per-breath bottleneck inserted at a layer boundary (after transformer
# layer FG_WAIST_AFTER, default 1 = between L1/L2, the validated v38 B-field location):
#
#   d     = down(x).gelu()                       # (B, S, d) — the WAIST d-rep
#   up_x  = up(d)                                # (B, S, H)
#   g     = sigmoid(gate_param)                  # scalar in (0,1)
#   x     = (1 - g)*x + g*up_x                   # zero-init-gated convex blend
#
# gate_param is init to a LARGE NEGATIVE (FG_WAIST_GATE_INIT, default -8.0) so g ~ 0
# at start: the blend is x = (1-g)*x + g*up_x ~ x  (warm-start byte-identical bypass).
# Training opens g ONLY if the waist earns its keep. The convex form (NOT v38's residual
# add) means g=0 is an EXACT pass-through regardless of the waist contents, so warm-start
# safety does NOT depend on up() being zero-init (it is small-randn here; the gate carries
# the bypass guarantee). Convex blend also bounds the waist's influence in [0,1].
#
# ADDITIVE + BYTE-IDENTICAL WHEN OFF: the waist runs ONLY when model.fg_waist_down is not
# None (gated by a getattr). A model with NO waist params (every existing single-domain
# ckpt, kenken, coloring, circuit, multi) runs the ORIGINAL forward verbatim — no new op,
# no new tensor, byte-identical. mycelium/kenken.py (the ORACLE) is NEVER imported-from /
# touched by this change.
#
# d-REP EXPOSURE (re-probe the common mode): when a list is present at
# model.fg_waist_capture (set ONLY by the eager re-eval capture path), the per-breath
# waist d-rep (the (B,S,d) tensor BEFORE up()) is appended to it. This is a NO-OP unless a
# caller installs the list (default attribute is absent -> getattr returns None -> skip),
# and it runs ONLY in the eager (non-JIT) capture forward — never inside the JIT graph
# (writing a Python-list side-effect inside a TinyJit trace is the accumulator-stale-ref
# trap; the capture forward is eager, mirroring _DartCapture's readout-LN hook).
FG_WAIST_DIM: int = int(os.environ.get("FG_WAIST_DIM", "256"))
FG_WAIST_AFTER: int = int(os.environ.get("FG_WAIST_AFTER", "1"))
FG_WAIST_GATE_INIT: float = float(os.environ.get("FG_WAIST_GATE_INIT", "-8.0"))


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class FactorGraphSpec:
    """Hyperparameter bundle for one problem domain.

    Parameters
    ----------
    s_max : int
        Grid size — the fixed sequence length the model sees.  49 for KenKen.
    n_values : int
        Codebook size == number of legal cell values.  7 for KenKen (1..7).
    n_factor_types : int
        Number of non-global relation types (T).  3 for KenKen (row/col/cage).
    n_heads : int
        Number of attention heads (must be >= n_factor_types + 1 for at least
        one head per type plus one global head).  16 for KenKen.
    k_max : int
        Maximum number of breaths (breath_embed/delta_gate allocation size).
    has_factor_inlet : bool
        Whether this domain supplies a per-cell inlet tensor via
        ``batch.factor_inlet``.  False => inlet contribution is 0 everywhere.
        For KenKen this is the verification inlet; for a generic domain with no
        arithmetic to verify, pass False.
    continuous_input : bool
        Whether the per-cell INPUT is a CONTINUOUS scalar (e.g. the ECC channel
        LLR) instead of a discrete value index.  False (default) => the validated
        DISCRETE one-hot input embed (kenken/coloring/circuit one-hot the cell
        value through the codebook-aligned state_embed) — byte-identical to the
        current engine.  True => the engine reads ``batch.cont_input`` (B, s_max)
        and maps it through a small learned linear (1 -> H) instead of the one-hot
        path.  This is the ONLY domain-gated divergence in the forward; it is a
        COMPILE-TIME python branch on this spec field, never reached when False.
    reinject_input : bool
        PER-BREATH CHANNEL RE-INJECTION (ECC fix #1).  False (default) => the
        channel/cell embed is added ONLY at B0 (the current B0-only behavior;
        byte-identical).  True => the SAME input embed (the immutable channel
        evidence — embed_factor_cells_continuous of batch.cont_input for the
        continuous path, embed_factor_cells for the discrete path) is RE-ADDED to
        the residual at the START of every breath, alongside the per-breath marker
        be_k.  This makes each breath a real BP variable-node update (channel +
        current messages every round).  The embed is computed ONCE before the loop
        (it does not change) and re-added each breath.  No new params (reuses the
        B0 embed weights).  COMPILE-TIME python branch on this spec field; never
        reached when False, so the forward is byte-identical.
    lora_rank : int
        PER-BREATH LoRA (ECC fix #2; un-shares the tied iterations).  0 (default)
        => no adapters, no new params (byte-identical).  r>0 => K separate rank-r
        low-rank residual adapters (one per breath) are applied at the START of
        each breath: ``x = x + B_k @ (A_k @ x)`` with B_k ZERO-INIT (neutral at
        step 0 -> byte-identical-off + warm-start safe + clean A/B).  The params
        are allocated by attach_factor_lora_params (fg_lora_A/fg_lora_B of shape
        (k_max, r, H) / (k_max, H, r)).  COMPILE-TIME python gate on this field +
        the presence of the params (getattr); never reached when 0 / unattached.
    """
    s_max: int
    n_values: int
    n_factor_types: int
    n_heads: int
    k_max: int
    has_factor_inlet: bool = False
    continuous_input: bool = False
    reinject_input: bool = False
    lora_rank: int = 0


# ---------------------------------------------------------------------------
# Batch attribute contract
# ---------------------------------------------------------------------------

class FactorGraphBatch:
    """Attribute contract that factor_breathing_forward reads.

    A KenKenBatch satisfies this contract (membership/latent_type built from
    cage_mask + a type-assignment helper; cell_valid and value_domain_mask are
    the same).  Use make_kenken_factor_batch() below to adapt a KenKenBatch.

    Required attributes
    -------------------
    input_cells      : Tensor (B, s_max) int  — 0=unknown, 1..N given value.
    cell_valid       : Tensor (B, s_max) float — 1.0 valid / 0.0 padding.
    value_domain_mask: Tensor (B, s_max, N)   — 1.0 for legal values, else 0.
    gold             : Tensor (B, s_max) int  — gold values 1..N (pad = 0).
    membership       : Tensor (B, L, s_max) float — factor membership matrix.
    latent_type      : Tensor (B, L) int      — factor type per latent.

    Optional
    --------
    factor_inlet     : Tensor (B, s_max, H) float — pre-built per-cell inlet.
                       Required when spec.has_factor_inlet=True.
    deduction_depth  : list[int] — Property-2 x-axis (may be all 0s if unknown).
    """
    def __init__(self, d: dict):
        self.input_cells: Tensor       = d["input_cells"]
        self.cell_valid: Tensor        = d["cell_valid"]
        self.value_domain_mask: Tensor = d["value_domain_mask"]
        self.gold: Tensor              = d["gold"]
        self.membership: Tensor        = d["membership"]
        self.latent_type: Tensor       = d["latent_type"]
        self.factor_inlet: Tensor | None = d.get("factor_inlet", None)
        self.deduction_depth: list[int] = d.get("deduction_depth",
                                                  [0] * int(d["input_cells"].shape[0]))


# ---------------------------------------------------------------------------
# KenKen adapter (builds membership / latent_type from KenKenBatch)
# ---------------------------------------------------------------------------

def make_kenken_factor_batch(kb: Any, spec: FactorGraphSpec,
                              prebuilt_inlet: "Tensor | None" = None
                              ) -> FactorGraphBatch:
    """Adapt a KenKenBatch into a FactorGraphBatch for the general engine.

    For KenKen: T=3 (row=0, col=1, cage=2).  The membership matrix L is built
    by stacking one latent per row, one per col, and one per cage-id.

    For the KenKen accuracy anchor (Step 3) this function drives the general
    loop with EXACTLY the same information the kenken forward uses.

    kb            : KenKenBatch (mycelium.kenken_data.KenKenBatch).
    spec          : FactorGraphSpec with n_factor_types=3, s_max=49.
    prebuilt_inlet: (B, s_max, H) Tensor from build_verification_inlet — set
                   this when spec.has_factor_inlet=True.
    """
    from mycelium.kenken_data import N_MAX, N_CELLS

    B = int(kb.input_cells.shape[0])
    S = spec.s_max
    T = spec.n_factor_types   # 3 for KenKen

    # ---- build membership (B, L, s_max) and latent_type (B, L) ----
    # L = N_MAX rows + N_MAX cols + n_cages_max cages = 7+7+C latents.
    C = int(kb.cage_op.shape[1])            # n_cages_max
    L = N_MAX + N_MAX + C                   # row latents | col latents | cage latents

    rows_idx = np.array([i // N_MAX for i in range(N_CELLS)], dtype=np.int32)  # (49,)
    cols_idx = np.array([i % N_MAX  for i in range(N_CELLS)], dtype=np.int32)  # (49,)

    # Row latents: latent r covers cells with row index == r.
    row_mem_np = np.zeros((N_MAX, N_CELLS), dtype=np.float32)
    for r in range(N_MAX):
        row_mem_np[r, rows_idx == r] = 1.0      # (N_MAX, 49)

    # Col latents: latent c covers cells with col index == c.
    col_mem_np = np.zeros((N_MAX, N_CELLS), dtype=np.float32)
    for c in range(N_MAX):
        col_mem_np[c, cols_idx == c] = 1.0      # (N_MAX, 49)

    # Row/col membership: broadcast across batch.
    row_mem = Tensor(row_mem_np, dtype=dtypes.float).reshape(1, N_MAX, N_CELLS).expand(B, N_MAX, N_CELLS)
    col_mem = Tensor(col_mem_np, dtype=dtypes.float).reshape(1, N_MAX, N_CELLS).expand(B, N_MAX, N_CELLS)

    # Cage latents: cage c covers all cells with cell_cage_id == c (per-batch).
    # cell_cage_id: (B, 49) int, -1 = padding.  one_hot(C) per cell -> (B, 49, C).
    cid = kb.cell_cage_id.clip(0, C - 1)                # (B, 49)
    is_real = (kb.cell_cage_id >= 0).cast(dtypes.float).reshape(B, N_CELLS, 1)
    cage_cell_oh = cid.one_hot(C).cast(dtypes.float) * is_real  # (B, 49, C)
    cage_mem = cage_cell_oh.transpose(1, 2)              # (B, C, 49)

    membership = Tensor.cat(row_mem, col_mem, cage_mem, dim=1)  # (B, L, S)

    # latent_type: 0=row, 1=col, 2=cage.
    lt_np = np.concatenate([
        np.zeros((N_MAX,), dtype=np.int32),    # rows -> type 0
        np.ones((N_MAX,), dtype=np.int32),     # cols -> type 1
        np.full((C,), 2, dtype=np.int32),      # cages -> type 2
    ])                                         # (L,)
    latent_type = Tensor(lt_np, dtype=dtypes.int).reshape(1, L).expand(B, L)

    d = {
        "input_cells":       kb.input_cells,
        "cell_valid":        kb.cell_valid,
        "value_domain_mask": kb.value_domain_mask,
        "gold":              kb.gold,
        "membership":        membership.contiguous(),
        "latent_type":       latent_type.contiguous(),
        "deduction_depth":   kb.deduction_depth,
    }
    if prebuilt_inlet is not None:
        d["factor_inlet"] = prebuilt_inlet
    return FactorGraphBatch(d)


# ---------------------------------------------------------------------------
# Per-cell embedding (general — parameterized on N and s_max)
# ---------------------------------------------------------------------------

def embed_factor_cells(input_cells: Tensor, state_embed: Tensor,
                        position_embed: Tensor, n_values: int) -> Tensor:
    """Convert (B, s_max) int cell states -> (B, s_max, H) embeddings.

    Mirrors kenken.py:embed_kenken but parameterized on n_values (N) instead
    of the hard-coded N_MAX=7.

    input_cells  : int Tensor (B, s_max), values in [0, n_values].  0 = unknown.
    state_embed  : (n_values+1, H) — rows for {0=unknown, 1..N=given value}.
    position_embed: (s_max, H) — learned position embedding.
    """
    B = int(input_cells.shape[0])
    S = int(position_embed.shape[0])
    # one-hot over n_values+1 (0 = unknown, 1..N = given value).
    one_hot = input_cells.one_hot(n_values + 1).cast(state_embed.dtype)  # (B,S,N+1)
    state = one_hot @ state_embed                                         # (B,S,H)
    pos = position_embed.reshape(1, S, -1).cast(state.dtype).expand(B, S, -1)
    return state + pos


# ---------------------------------------------------------------------------
# CONTINUOUS per-cell input embed (the ONE additive core edit for ECC / §8.1).
# ---------------------------------------------------------------------------

def embed_factor_cells_continuous(cont_input: Tensor, cont_embed_w: Tensor,
                                   cont_embed_b: Tensor, position_embed: Tensor
                                   ) -> Tensor:
    """Map a per-cell CONTINUOUS scalar (B, s_max) -> (B, s_max, H) embeddings.

    The continuous twin of embed_factor_cells.  Where the DISCRETE path one-hots a
    cell-value index through the codebook-aligned state_embed, the CONTINUOUS path
    is the ECC frontier's input: a per-cell real scalar (the channel LLR), NOT a
    discrete value.  A small LEARNED LINEAR (1 -> H) lifts the scalar into the
    residual stream, then the SAME learned position embedding is added (identical
    to the discrete path's `state + pos` structure).

    This is NOT a new pointer/attention pathway (the attention-bootstrap law): it is
    a per-position scalar -> H lift (one weight vector + bias), which bootstraps from
    the task gradient exactly like a codebook selection.  cont_embed_w is init at
    CODEBOOK SCALE (~0.1) in attach_factor_graph_params so the lifted LLR lands at the
    same magnitude as a discrete value embed at step 0, and the monotone sign->value
    relationship (high-SNR LLR sign == the gold bit) is in reach of the readout from
    the first breath.

    cont_input    : (B, s_max) float — per-cell continuous scalar (e.g. LLR).
    cont_embed_w  : (1, H)     float — learned lift weight (codebook-scale init).
    cont_embed_b  : (H,)       float — learned lift bias (zeros init).
    position_embed: (s_max, H) float — the SAME learned position embedding.

    dtype-preserving: the matmul/add inherit position_embed.dtype after the cast
    chain in the caller; no dtypes.float32 literal is baked here (substrate law).
    """
    B = int(cont_input.shape[0])
    S = int(position_embed.shape[0])
    # (B, S, 1) @ (1, H) -> (B, S, H), then + bias.  Cast the scalar to the embed
    # dtype so the lifted feature shares the residual-stream dtype (no float literal).
    x_scalar = cont_input.reshape(B, S, 1).cast(cont_embed_w.dtype)       # (B,S,1)
    state = x_scalar @ cont_embed_w + cont_embed_b.reshape(1, 1, -1)      # (B,S,H)
    pos = position_embed.reshape(1, S, -1).cast(state.dtype).expand(B, S, -1)
    return state + pos


# ---------------------------------------------------------------------------
# Main breathing forward
# ---------------------------------------------------------------------------

def factor_breathing_forward(model: Any, batch: FactorGraphBatch,
                              spec: FactorGraphSpec, K: int,
                              stoch_keep: "Tensor | None" = None,
                              return_waist: bool = False,
                              ):
    """Run K breaths of factor-graph constraint propagation.

    Byte-identical to kenken_breathing_forward when driven with KenKen inputs
    via make_kenken_factor_batch() (minus the optional verification inlet,
    which must be pre-built and passed as batch.factor_inlet when
    spec.has_factor_inlet=True).

    Coupled pieces replaced vs the kenken original:
      (a) build_kenken_attn_bias -> build_factor_attn_bias (from factor_masks)
      (b) embed_kenken            -> embed_factor_cells    (N + s_max params)
      (c) assert S==N_CELLS       -> assert S==spec.s_max
      (d) value-codebook size     -> spec.n_values
      (e) inlet                   -> batch.factor_inlet or zeros

    Everything else (breath loop, delta_gate, readout, calib) is verbatim from
    the v98 recipe.

    Parameters
    ----------
    model   : object with factor_graph_* attributes (from attach_factor_graph_params).
    batch   : FactorGraphBatch with membership/latent_type/cell_valid/input_cells/
              value_domain_mask and optionally factor_inlet.
    spec    : FactorGraphSpec.
    K       : number of breaths to run (<= spec.k_max).
    stoch_keep : optional (K,) Tensor of per-breath keep-scales (training only).
    return_waist : if True AND the waist is attached, ALSO return the per-breath waist
                   d-rep history (list of K (B, s_max, d) Tensors) as a 3rd element so the
                   trainer can attach the validity-shaping aux objective IN-GRAPH. Default
                   False -> the original 2-tuple signature (byte-identical callers). When
                   the waist is NOT attached, return_waist=True yields an EMPTY list (the
                   aux is a no-op) — so a return_waist caller works on a baseline model too.

    Returns
    -------
    value_logits_history : list of K Tensors, each (B, s_max, n_values) float.
    calib_history        : list of K Tensors, each (B,) float, sigmoid'd.
    [waist_drep_history]  : (only when return_waist) list of K (B, s_max, d) Tensors.
    """
    assert hasattr(model, "fg_state_embed"), \
        "model has no factor_graph params; call attach_factor_graph_params first."

    N   = spec.n_values
    S   = spec.s_max
    H   = int(model.fg_state_embed.shape[-1])

    state_embed    = model.fg_state_embed       # (N+1, H)
    position_embed = model.fg_position_embed    # (s_max, H)
    breath_embed   = model.fg_breath_embed      # (k_max, H)
    delta_gate     = model.fg_delta_gate        # (k_max,)
    value_codebook = model.fg_value_codebook    # (N, H)
    calib_head_w   = model.fg_calib_head_w      # (H, 1)
    calib_head_b   = model.fg_calib_head_b      # (1,)

    input_cells       = batch.input_cells        # (B, s_max) int
    cell_valid        = batch.cell_valid         # (B, s_max) float
    value_domain_mask = batch.value_domain_mask  # (B, s_max, N) float
    membership        = batch.membership         # (B, L, s_max)
    latent_type       = batch.latent_type        # (B, L) int

    B = int(input_cells.shape[0])

    # (a) COUPLED: build per-batch attention bias from factor membership.
    # This is the ONLY call replaced vs the kenken original.
    # FG_HYP_MASK=0 (default) -> boolean {0,-1e4} mask (byte-identical to v98).
    # FG_HYP_MASK=1           -> geometric Poincaré mask (~1e-3-identical at t=0
    #                            when anchors are frozen; requires
    #                            attach_factor_hyperbolic_params to have been called).
    #
    # MULTI-TASK HEAD-ALLOCATION FIX: when the batch carries a PER-BATCH head->type
    # allocation (batch.head_type_oh / batch.head_is_global, set ONLY by the multi-task
    # adapter), route to the tensor-driven multitask builder so each pure single-domain
    # batch uses its NATIVE head allocation (coloring 15+1, kenken/circuit 5/5/5+1) on
    # the SHARED 16 heads. SINGLE-DOMAIN batches never carry these attrs -> the original
    # build_factor_attn_bias call is taken verbatim (byte-identical). The presence of
    # these attrs is a COMPILE-TIME property of the multi-task batch (always set there,
    # never set in single-domain), NOT a runtime domain-branch inside the JIT graph.
    head_type_oh = getattr(batch, "head_type_oh", None)
    head_is_global = getattr(batch, "head_is_global", None)
    if FG_HYP_MASK:
        attn_bias = build_factor_hyperbolic_attn_bias(
            model, membership, latent_type, cell_valid,
            spec.n_heads, spec.n_factor_types, S,
        )  # (B, n_heads, s_max, s_max)
    elif head_type_oh is not None:
        attn_bias = build_factor_attn_bias_multitask(
            membership, latent_type, cell_valid,
            head_type_oh, head_is_global,
            spec.n_heads, spec.n_factor_types, S,
        )  # (B, n_heads, s_max, s_max)
    else:
        attn_bias = build_factor_attn_bias(
            membership, latent_type, cell_valid,
            spec.n_heads, spec.n_factor_types, S,
        )  # (B, n_heads, s_max, s_max)

    # (e) OPTIONAL inlet: domain plug for per-cell arithmetic/structural hints.
    # When has_factor_inlet=False or batch.factor_inlet is None: zeros (no contribution).
    if spec.has_factor_inlet and batch.factor_inlet is not None:
        inlet = batch.factor_inlet.cast(dtypes.float)  # (B, s_max, H)
    else:
        # Zero contribution — don't build a tensor that might bake into the JIT
        # graph as a float32 literal; use Tensor.zeros dynamically.
        inlet = Tensor.zeros((B, S, H), dtype=dtypes.float)

    # Value-domain mask -> additive bias for readout.
    value_bias = (1.0 - value_domain_mask) * (-1e4)  # (B, s_max, N)

    # (b) COUPLED: embed cells, parameterized on N and s_max.
    # CONTINUOUS-INPUT BRANCH (the ONE additive core edit; ECC / §8.1 frontier).
    # spec.continuous_input is a COMPILE-TIME python attribute (NOT a runtime per-
    # batch value branch): it is True ONLY for the ECC spec and False (default) for
    # kenken/coloring/circuit/multi.  When False the ORIGINAL discrete one-hot embed
    # is taken VERBATIM (byte-identical) — the continuous branch is never traced.  When
    # True the engine reads batch.cont_input (a per-cell continuous scalar, e.g. the
    # channel LLR) and lifts it through the small learned cont-embed linear instead of
    # one-hotting a discrete value.  Guarded additionally by the presence of the
    # cont-embed params (getattr) so a spec.continuous_input=True without attached
    # params is a loud failure rather than silent fall-through.
    if getattr(spec, "continuous_input", False):
        cont_input = batch.cont_input                            # (B, s_max) float
        cont_embed_w = model.fg_cont_embed_w                     # (1, H)
        cont_embed_b = model.fg_cont_embed_b                     # (H,)
        x = embed_factor_cells_continuous(cont_input, cont_embed_w,
                                          cont_embed_b, position_embed)
    else:
        x = embed_factor_cells(input_cells, state_embed, position_embed, N)
    x = x.cast(dtypes.half) if x.dtype != dtypes.half else x

    # (g) PER-BREATH CHANNEL RE-INJECTION (ECC fix #1 — spec.reinject_input).
    # When OFF (default) -> the channel/cell embed `x` is used only at B0 (the current
    # B0-only behavior; the whole block below is a COMPILE-TIME python branch on the
    # spec field, never traced -> byte-identical). When ON -> the SAME input embed is the
    # immutable channel evidence and is RE-ADDED to the residual at the START of every
    # breath (alongside be_k), making each breath a real BP variable-node update (channel
    # + current messages every round). The embed does NOT change across breaths, so it is
    # computed ONCE here (== the B0 lift `x` we just built) and re-added each breath. NO
    # new params (reuses the B0 embed weights). dtype-preserving (channel_embed shares
    # x.dtype after the .cast(half) above; the per-breath `+ channel_embed` is the same op
    # as the existing `+ be_k`, no dtypes.float32 literal baked).
    reinject = bool(getattr(spec, "reinject_input", False))
    channel_embed = x if reinject else None                      # (B, S, H) or None

    # (h) PER-BREATH LoRA (ECC fix #2 — spec.lora_rank>0 + params attached). K separate
    # rank-r low-rank residual adapters (one per breath) UN-SHARE the tied iterations:
    # breath k applies x = x + B_k @ (A_k @ x). B_k is ZERO-INIT so the adapter outputs 0
    # at step 0 -> x unchanged -> byte-identical-off + warm-start safe + clean A/B. Gated
    # by BOTH the spec field AND the presence of the params (getattr) so a spec.lora_rank>0
    # without attached params is a loud failure rather than silent fall-through, and a model
    # with no LoRA params (every existing ckpt) takes the original forward verbatim.
    lora_rank = int(getattr(spec, "lora_rank", 0))
    lora_A = getattr(model, "fg_lora_A", None)                   # (k_max, r, H) or None
    use_lora = lora_rank > 0 and lora_A is not None
    if use_lora:
        lora_B = model.fg_lora_B                                 # (k_max, H, r)

    layers = list(model.block.layers)
    assert len(layers) >= 4, f"expected >=4 transformer layers; got {len(layers)}"
    K_max = int(breath_embed.shape[0])
    assert K <= K_max, f"K={K} exceeds k_max={K_max}"

    from mycelium.breathing import _layernorm

    inlet_h = inlet.cast(x.dtype)
    cell_valid_col = cell_valid.reshape(B, S, 1)

    # (f) OPTIONAL WAIST (FG_WAIST) — getattr-gated so OFF == byte-identical. Read the
    # params ONCE outside the loop (they are shared across breaths, like delta_gate).
    # waist_down is None on every model without waist params -> the in-loop branch is a
    # compile-time constant (not a per-breath python value branch), and the original
    # forward is taken verbatim.
    waist_down_w = getattr(model, "fg_waist_down", None)        # (H, d)
    use_waist = waist_down_w is not None
    if use_waist:
        waist_down_b = model.fg_waist_down_b                    # (d,)
        waist_up_w   = model.fg_waist_up                        # (d, H)
        waist_up_b   = model.fg_waist_up_b                      # (H,)
        waist_gate_p = model.fg_waist_gate                      # () scalar param
        waist_after  = int(getattr(model, "fg_waist_after", FG_WAIST_AFTER))
        # d-rep capture sink (eager re-eval only; absent/None by default -> no-op).
        waist_capture = getattr(model, "fg_waist_capture", None)

    value_logits_history: list[Tensor] = []
    calib_history: list[Tensor] = []
    waist_drep_history: list[Tensor] = []   # per-breath (B, S, d); filled only if use_waist

    # Per-breath RESIDUAL capture sink (eager re-eval only; absent/None by default -> no-op).
    # Mirrors the fg_waist_capture pattern (getattr-None gated, append on the eager path):
    # when present (a list), the per-breath readout-point residual x (B, S, H) is appended
    # each breath. None by default -> the append branch is skipped -> byte-identical to the
    # validated training graph (the JIT path never sets fg_resid_capture).
    resid_capture = getattr(model, "fg_resid_capture", None)

    for k in range(K):
        # (h) PER-BREATH LoRA (applied at the START of the breath, before the marker /
        # channel re-add). x = x + B_k @ (A_k @ x). A_k: (r, H), B_k: (H, r); per-breath
        # slice k. B_k zero-init -> the residual `lora_delta` is exactly 0 at step 0 ->
        # x unchanged -> byte-identical-off. Computed in x.dtype (half) — matmuls preserve
        # dtype, no dtypes.float32 literal. This is a per-position (per-cell) low-rank
        # residual (NOT a new attention/pointer pathway), so it bootstraps like a codebook
        # selection. UN-SHARES the iterations: each breath gets its own small transform.
        if use_lora:
            A_k = lora_A[k].cast(x.dtype)                         # (r, H)
            B_k = lora_B[k].cast(x.dtype)                         # (H, r)
            lora_h = x @ A_k.transpose()                          # (B, S, r) = A_k @ x per cell
            lora_delta = lora_h @ B_k.transpose()                # (B, S, H) = B_k @ (A_k @ x)
            x = x + lora_delta

        be_k = breath_embed[k].reshape(1, 1, -1).cast(x.dtype)   # (1, 1, H)
        x_in = x + be_k + inlet_h                                 # add inlet EVERY breath (LIVE)

        # (g) PER-BREATH CHANNEL RE-INJECTION: re-add the immutable channel evidence to
        # the residual at the START of every breath (channel + current messages == the BP
        # variable-node update). Same op shape as `+ be_k`; channel_embed is None when OFF
        # (compile-time branch, never traced -> byte-identical B0-only behavior).
        if reinject:
            x_in = x_in + channel_embed                           # re-add channel evidence

        x_pre = x
        h = x_in
        # (c) COUPLED: pass s_max-length tensor through kenken_layer_forward.
        # The only assertion inside kenken_layer_forward is `assert S==N_CELLS`
        # which fires when S != 49.  For the general case we call the function
        # directly — it works for any S as long as attn_bias has matching shape.
        # For the KenKen anchor (S=49) it is byte-identical to the original call.
        for li, layer in enumerate(layers[:4]):
            h = kenken_layer_forward(layer, h, attn_bias)          # no Q-rotation
            # (f) WAIST: zero-init-gated convex blend at the layer boundary li==waist_after.
            # g ~ 0 at init (gate_param large-negative) -> blend ~ pass-through (warm-start
            # byte-identical). Computed in h.dtype (half) to stay on the validated activation
            # path; no dtypes.float32 literal (sigmoid/gelu/matmul preserve dtype).
            if use_waist and li == waist_after:
                d_rep = (h @ waist_down_w.cast(h.dtype)
                         + waist_down_b.cast(h.dtype)).gelu()       # (B, S, d) — the WAIST d-rep
                up_x = (d_rep @ waist_up_w.cast(h.dtype)
                        + waist_up_b.cast(h.dtype))                 # (B, S, H)
                g = waist_gate_p.cast(h.dtype).sigmoid().reshape(1, 1, 1)
                h = (1.0 - g) * h + g * up_x                        # convex blend
                waist_drep_history.append(d_rep)                    # (B, S, d)
                # d-rep exposure (eager capture only; no-op when sink absent/None).
                if waist_capture is not None:
                    waist_capture.append(d_rep)

        gate_k = delta_gate[k].cast(h.dtype).reshape(1, 1, 1)
        delta = h - x_pre

        if stoch_keep is not None:
            keep_k = stoch_keep[k].cast(h.dtype).reshape(1, 1, 1)
            x = x_pre + (gate_k * keep_k) * delta
        else:
            x = x_pre + gate_k * delta

        # Per-breath residual capture (eager-only; no-op when the sink is absent/None).
        # The persistent 1024d residual x (B, S, H) at the readout point — the factor-graph
        # belief state going into this breath's readout. Realized to fp32 eagerly inside the
        # branch so it never enters the JIT graph (the JIT path never sets fg_resid_capture).
        if resid_capture is not None:
            resid_capture.append(x.cast(dtypes.float).realize().numpy())

        # Readout: project each cell to N-way logit; apply value-domain mask.
        x_ln = _layernorm(x, model.ln_f_g, model.ln_f_b,
                          model.cfg.layer_norm_eps).cast(dtypes.float)
        cell_logits_k = x_ln @ value_codebook.T.cast(dtypes.float)  # (B, s_max, N)
        cell_logits_k = cell_logits_k + value_bias.cast(dtypes.float)
        value_logits_history.append(cell_logits_k)

        # Calibration: mean-pool over VALID cells only.
        pool_num = (x_ln * cell_valid_col.cast(dtypes.float)).sum(axis=1)  # (B, H)
        pool_den = cell_valid_col.cast(dtypes.float).sum(axis=1) + 1e-6    # (B, 1)
        pool = pool_num / pool_den
        calib_logit = pool @ calib_head_w.cast(dtypes.float) + calib_head_b.cast(dtypes.float)
        calib_k = calib_logit.reshape(-1).sigmoid()
        calib_history.append(calib_k)

    if return_waist:
        return value_logits_history, calib_history, waist_drep_history
    return value_logits_history, calib_history


# ---------------------------------------------------------------------------
# Param attach
# ---------------------------------------------------------------------------

def attach_factor_graph_params(model: Any, hidden: int,
                                spec: FactorGraphSpec) -> None:
    """Allocate factor-graph params on `model` (a BreathingTransformer instance).

    Mirrors attach_kenken_params but:
      - N (=spec.n_values) replaces the hard-coded N_MAX=7.
      - s_max (=spec.s_max) replaces the hard-coded N_CELLS=49.
      - NO kenken_fixed_mask / kenken_head_split (masks are per-batch from
        membership; the fixed mask was KenKen-specific).
      - position_embed is a plain learned (s_max, H) — no row/col one-hot
        structural prior (those are domain-specific; supply them at the domain
        layer if needed by initializing fg_position_embed after this call).

    Attributes added
    ----------------
    fg_state_embed      (N+1, hidden)   — {0=unknown, 1..N=given value}
    fg_position_embed   (s_max, hidden) — plain learned embedding (small randn)
    fg_value_codebook   (N, hidden)     — orthonormal rows at scale 0.1
    fg_calib_head_w     (hidden, 1)
    fg_calib_head_b     (1,)
    fg_breath_embed     (k_max, hidden) — orthonormal rows at scale 0.5
    fg_delta_gate       (k_max,)        — ones (full update at init)
    """
    N    = spec.n_values
    S    = spec.s_max
    k_max = spec.k_max

    # Value codebook — orthonormal rows, scale 0.1 (mirror kenken).
    rng_cb = np.random.RandomState(1403)
    raw_cb = rng_cb.randn(max(hidden, N), hidden).astype(np.float32)
    q_cb, _ = np.linalg.qr(raw_cb)
    cb_unit = q_cb[:N].astype(np.float32)                           # (N, hidden)
    model.fg_value_codebook = Tensor(cb_unit * 0.1, dtype=dtypes.float).contiguous()

    # State embedding — N+1 rows: row 0 = unknown, rows 1..N aligned with codebook.
    state = np.zeros((N + 1, hidden), dtype=np.float32)
    state[0] = np.random.RandomState(1402).randn(hidden).astype(np.float32) * 0.02
    state[1:N + 1] = cb_unit                                         # align with codebook
    model.fg_state_embed = Tensor(state, dtype=dtypes.float).contiguous()

    # Position embedding — plain learned (s_max, hidden), small randn.
    rng_pos = np.random.RandomState(1407)
    pos_np = (rng_pos.randn(S, hidden) * 0.02).astype(np.float32)
    model.fg_position_embed = Tensor(pos_np, dtype=dtypes.float).contiguous()

    # Calibration head.
    cw = (np.random.RandomState(1404).randn(hidden, 1) * 0.02).astype(np.float32)
    model.fg_calib_head_w = Tensor(cw, dtype=dtypes.float).contiguous()
    model.fg_calib_head_b = Tensor.zeros((1,), dtype=dtypes.float).contiguous()

    # Breath embedding — orthonormal, scale 0.5 (mirror kenken).
    breath_scale = float(os.environ.get("FG_BREATH_EMBED_SCALE", "0.5"))
    rng_be = np.random.RandomState(1405)
    raw = rng_be.randn(max(k_max, hidden), hidden).astype(np.float32)
    q, _ = np.linalg.qr(raw)
    be = q[:k_max].astype(np.float32) * breath_scale
    model.fg_breath_embed = Tensor(be, dtype=dtypes.float).contiguous()

    # Delta gate — ones (full update; mirror kenken).
    model.fg_delta_gate = Tensor.ones((k_max,), dtype=dtypes.float).contiguous()

    # CONTINUOUS-INPUT embed params (ECC / §8.1).  Allocated ONLY when the spec
    # requests the continuous path; a discrete-input model (kenken/coloring/circuit)
    # never carries these attrs, so factor_graph_parameters / the engine getattr-gate
    # see nothing and the model is byte-identical.  cont_embed_w is init at CODEBOOK
    # SCALE so the lifted scalar lands at the same magnitude as a discrete value embed
    # at step 0 (bootstrap from the task gradient; the attention-bootstrap law for a
    # per-position scalar->H lift).  cont_embed_b is zeros.
    if getattr(spec, "continuous_input", False):
        rng_ce = np.random.RandomState(1409)
        ce_w = (rng_ce.randn(1, hidden) * 0.1).astype(np.float32)   # codebook scale
        model.fg_cont_embed_w = Tensor(ce_w, dtype=dtypes.float).contiguous()
        model.fg_cont_embed_b = Tensor.zeros((hidden,), dtype=dtypes.float).contiguous()


def factor_graph_parameters(model: Any) -> list[Tensor]:
    """Trainable factor-graph params (excludes backbone params).

    The continuous-input embed params (fg_cont_embed_*) are included ONLY when
    attached (ECC spec); absent for discrete-input models -> the list is identical
    to the validated set for kenken/coloring/circuit.

    The per-breath LoRA params (fg_lora_A/B) are included ONLY when attached
    (FG_LORA_RANK>0); absent otherwise -> the list is identical to the validated set.
    """
    params = [
        model.fg_state_embed,
        model.fg_position_embed,
        model.fg_value_codebook,
        model.fg_calib_head_w,
        model.fg_calib_head_b,
        model.fg_breath_embed,
        model.fg_delta_gate,
    ]
    if getattr(model, "fg_cont_embed_w", None) is not None:
        params += [model.fg_cont_embed_w, model.fg_cont_embed_b]
    if getattr(model, "fg_lora_A", None) is not None:
        params += [model.fg_lora_A, model.fg_lora_B]
    return params


# ---------------------------------------------------------------------------
# PER-BREATH LoRA param attach (ECC fix #2) — additive, ZERO-INIT-B gated.
# ---------------------------------------------------------------------------

def attach_factor_lora_params(model: Any, hidden: int, spec: FactorGraphSpec,
                              rank: int) -> None:
    """Allocate the K per-breath rank-r LoRA adapters on `model` (additive; OFF
    unless called with rank>0).

    K = spec.k_max separate adapters, breath k uses adapter k, so each breath gets a
    small UNIQUE low-rank transform (un-shares the otherwise-tied iterations -> toward
    un-tied neural-BP). Applied at the engine level (factor_breathing_forward), NOT
    inside the oracle kenken_layer_forward (the oracle is never touched).

    Params added
    ------------
    fg_lora_A   (k_max, rank, hidden)  — DOWN projection (A_k), small randn 0.02 so the
                                         down-projected feature is well-scaled and the
                                         gradient flows from step 0.
    fg_lora_B   (k_max, hidden, rank)  — UP projection (B_k), ZERO-INIT so the adapter's
                                         residual delta B_k @ (A_k @ x) is EXACTLY 0 at
                                         init -> x is unchanged -> byte-identical-off +
                                         warm-start safe + clean A/B. Gradient still flows
                                         (A is non-zero), so the adapter OPENS if it earns
                                         its keep.

    Param count: 2 * k_max * rank * hidden ~ 2*16*8*1024 = 2.1M floats at K=16 r=8
    (the brief's ~1M is per-direction; both directions ~2M).
    """
    assert int(rank) > 0, "attach_factor_lora_params called with rank<=0"
    r = int(rank)
    k_max = int(spec.k_max)
    H = int(hidden)
    # A (down): small randn 0.02 — well-scaled features, gradient flows from step 0.
    rng_a = np.random.RandomState(2601)
    a_np = (rng_a.randn(k_max, r, H) * 0.02).astype(np.float32)
    model.fg_lora_A = Tensor(a_np, dtype=dtypes.float).contiguous()
    # B (up): ZERO-INIT -> adapter delta == 0 at init (byte-identical-off + warm-start).
    model.fg_lora_B = Tensor.zeros((k_max, H, r), dtype=dtypes.float).contiguous()


def factor_lora_parameters(model: Any) -> list[Tensor]:
    """Trainable per-breath LoRA params (empty when not attached)."""
    if getattr(model, "fg_lora_A", None) is None:
        return []
    return [model.fg_lora_A, model.fg_lora_B]


# ---------------------------------------------------------------------------
# Waist param attach (FG_WAIST) — additive, zero-init-gated.
# ---------------------------------------------------------------------------

def attach_factor_waist_params(model: Any, hidden: int, d: int = FG_WAIST_DIM,
                               after: int = FG_WAIST_AFTER,
                               gate_init: float = FG_WAIST_GATE_INIT,
                               aux: str = "none") -> None:
    """Allocate the in-deducer WAIST params on `model` (additive; OFF unless called).

    The waist is a per-breath zero-init-gated convex-blend bottleneck inserted after
    transformer layer `after` (see the module docstring on FG_WAIST). Calling this is what
    turns the waist ON: factor_breathing_forward reads model.fg_waist_down via getattr and
    skips the whole waist when it is None. So a model WITHOUT this call runs byte-identical.

    Params added
    ------------
    fg_waist_down     (hidden, d)  — down projection (small randn 0.02; mirrors v38).
    fg_waist_down_b   (d,)         — down bias (zeros).
    fg_waist_up       (d, hidden)  — up projection, ZERO-INIT (mirrors v38's zero-init
                                     proj_up). With BOTH the zero-init up AND the gate~0,
                                     warm-start is DOUBLY safe: the waist's contribution
                                     g*up(gelu(down(x))) is EXACTLY 0 at init regardless of
                                     the gate (up=0), so resuming a baseline ckpt is byte-
                                     identical to the gate-only argument; gradient still
                                     flows (down is non-zero, gate is finite), so the waist
                                     can OPEN if it earns its keep. (The convex blend at g~0
                                     is the second, independent bypass guarantee.)
    fg_waist_up_b     (hidden,)    — up bias (zeros).
    fg_waist_gate     ()           — scalar gate logit; init gate_init (large -ve) -> g ~ 0.
    fg_waist_after    int          — the layer-boundary index (python int; not a Tensor).

    AUX HEAD (FG_WAIST_AUX): when aux in {classify, both}, also allocate a validity
    classifier head on the POOLED waist d-rep (mean over valid cells):
    fg_waist_aux_w    (d, 1)       — small randn 0.02.
    fg_waist_aux_b    (1,)         — zeros.
    The 'attract' term needs no params (it pulls toward a running valid centroid, a buffer
    the trainer owns), so aux=='attract' allocates no head. aux=='none' allocates nothing.
    """
    rng_dn = np.random.RandomState(2401)
    dn = (rng_dn.randn(hidden, d) * 0.02).astype(np.float32)
    model.fg_waist_down = Tensor(dn, dtype=dtypes.float).contiguous()
    model.fg_waist_down_b = Tensor.zeros((d,), dtype=dtypes.float).contiguous()

    # up: ZERO-INIT (v38 proj_up pattern) -> waist contribution == 0 at init regardless of
    # the gate (the doubly-safe warm-start bypass; gradient still flows via down + gate).
    model.fg_waist_up = Tensor.zeros((d, hidden), dtype=dtypes.float).contiguous()
    model.fg_waist_up_b = Tensor.zeros((hidden,), dtype=dtypes.float).contiguous()

    # Gate logit -> sigmoid(gate_init) ~ 0 (byte-identical bypass at init). Shape (1,) NOT
    # () — a 0-d optimizer param trips AdamW's moment update (shape (1,) assigned to () ->
    # broadcast error) on this tinygrad. (1,) is reshaped to (1,1,1) in the forward, so the
    # blend is unchanged. Build from a numpy array (NO dtypes.float32 literal as a graph const).
    model.fg_waist_gate = Tensor(
        np.array([gate_init], dtype=np.float32), dtype=dtypes.float).contiguous()
    model.fg_waist_after = int(after)

    if aux in ("classify", "both"):
        rng_aux = np.random.RandomState(2403)
        aw = (rng_aux.randn(d, 1) * 0.02).astype(np.float32)
        model.fg_waist_aux_w = Tensor(aw, dtype=dtypes.float).contiguous()
        model.fg_waist_aux_b = Tensor.zeros((1,), dtype=dtypes.float).contiguous()


def factor_waist_parameters(model: Any) -> list[Tensor]:
    """Trainable waist params (empty when the waist is not attached)."""
    if getattr(model, "fg_waist_down", None) is None:
        return []
    params = [
        model.fg_waist_down, model.fg_waist_down_b,
        model.fg_waist_up, model.fg_waist_up_b,
        model.fg_waist_gate,
    ]
    if getattr(model, "fg_waist_aux_w", None) is not None:
        params += [model.fg_waist_aux_w, model.fg_waist_aux_b]
    return params


def pooled_waist_drep(d_rep: Tensor, cell_valid: Tensor) -> Tensor:
    """Mean-pool a per-cell waist d-rep (B, S, d) over valid cells -> (B, d).

    The SAME pooling the engine's calibration head uses (mean over cell_valid>0) — so the
    pooled d-rep here is the exact 'silhouette' the dart cluster probe re-probes, just read
    at the waist boundary instead of the final readout LN. Used by the aux objective and the
    capture path. Pure tensor ops, dtype-preserving."""
    B = int(d_rep.shape[0])
    S = int(d_rep.shape[1])
    cv = cell_valid.reshape(B, S, 1).cast(d_rep.dtype)
    num = (d_rep * cv).sum(axis=1)                              # (B, d)
    den = cv.sum(axis=1) + 1e-6                                 # (B, 1)
    return num / den


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def factor_loss(
    value_logits_history: list[Tensor],
    calib_history: list[Tensor],
    batch: FactorGraphBatch,
    spec: FactorGraphSpec,
    constraint_weight: float = 0.0,
    calib_weight: float = 0.1,
    ortho_lambda: float = 0.0,
    ortho_codebooks: "list[Tensor] | None" = None,
    constraint_energy_fn: "Callable[[list[Tensor], Any], Tensor] | None" = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Per-breath weighted-CE ladder + optional constraint energy + calibration.

    Mirrors kenken_loss with N_MAX->spec.n_values, N_CELLS->spec.s_max and the
    constraint-energy term replaced by an optional per-domain plug.

    Parameters
    ----------
    value_logits_history : list of K Tensors (B, s_max, N).
    calib_history        : list of K Tensors (B,).
    batch                : FactorGraphBatch with .gold, .cell_valid, .input_cells.
    spec                 : FactorGraphSpec.
    constraint_weight    : weight on the constraint-energy term (0 = off).
    calib_weight         : weight on the calibration MSE term.
    ortho_lambda         : codebook-orthogonality penalty weight (0 = off).
    ortho_codebooks      : list of (R, H) Tensors to penalize off-diagonal cosine.
    constraint_energy_fn : OPTIONAL callable(value_logits_history, batch) -> scalar
                           Tensor.  The domain-specific constraint energy (e.g.
                           kenken_constraint_energy).  None => energy term is 0.

    Returns
    -------
    (total, parts_dict) where parts_dict has keys 'cell_ce', 'energy', 'calib',
    and optionally 'ortho'.
    """
    K   = len(value_logits_history)
    N   = spec.n_values
    S   = spec.s_max
    B   = int(value_logits_history[0].shape[0])

    gold     = batch.gold                                    # (B, s_max) 1..N
    gold_idx = (gold - 1).clip(0, N - 1)                    # (B, s_max) 0..N-1
    cell_valid = batch.cell_valid                            # (B, s_max)

    # Supervised = valid AND not a given cell.
    observed  = (batch.input_cells > 0).cast(dtypes.float)  # (B, s_max)
    supervise = cell_valid * (1.0 - observed)                # (B, s_max)

    supervise_flat = supervise.reshape(B * S)
    sup_sum        = supervise_flat.sum() + 1e-6
    gold_flat      = gold_idx.reshape(B * S)

    # Per-breath weighted-CE ladder (weight_k = 1 + k/(K-1)).
    cell_loss_sum = Tensor.zeros((), dtype=dtypes.float)
    weight_sum = 0.0
    for k, logits in enumerate(value_logits_history):
        weight_k = 1.0 + float(k) / float(K - 1) if K > 1 else 1.0
        ce_elems = logits.reshape(B * S, N).sparse_categorical_crossentropy(
            gold_flat, reduction="none"
        )                                                    # (B*s_max,)
        ce_k = (ce_elems * supervise_flat).sum() / sup_sum
        cell_loss_sum = cell_loss_sum + ce_k * weight_k
        weight_sum += weight_k
    cell_loss = cell_loss_sum / float(weight_sum)

    # Constraint energy (domain plug or zero).
    if constraint_energy_fn is not None and constraint_weight > 0.0:
        energy = constraint_energy_fn(value_logits_history, batch).mean()
    else:
        energy = Tensor.zeros((), dtype=dtypes.float)

    # Calibration MSE against a per-breath correctness target.
    final_argmax = (value_logits_history[-1].argmax(axis=-1) + 1).detach()  # (B, s_max)
    eq = (final_argmax == gold).cast(dtypes.float)
    eq_valid = eq * cell_valid + (1.0 - cell_valid)                          # pad counts as match
    correct = eq_valid.prod(axis=-1)                                         # (B,) 0/1
    calib_loss_sum = Tensor.zeros((), dtype=dtypes.float)
    for k, calib_k in enumerate(calib_history):
        progression = float(k) / float(K - 1) if K > 1 else 1.0
        target_k = 0.5 + (correct - 0.5) * progression
        calib_loss_sum = calib_loss_sum + ((calib_k - target_k) ** 2).mean()
    calib_loss = calib_loss_sum / float(K)

    total = cell_loss + constraint_weight * energy + calib_weight * calib_loss

    parts: dict[str, Tensor] = {
        "cell_ce": cell_loss,
        "energy":  energy,
        "calib":   calib_loss,
    }

    # Codebook-orthogonality penalty (mirror kenken_loss).
    if ortho_lambda > 0.0 and ortho_codebooks:
        ortho = Tensor.zeros((), dtype=dtypes.float)
        for cb in ortho_codebooks:
            ortho = ortho + codebook_ortho_penalty(cb)
        total = total + ortho_lambda * ortho
        parts["ortho"] = ortho

    return total, parts


# ---------------------------------------------------------------------------
# Accuracy helper
# ---------------------------------------------------------------------------

def factor_accuracy(value_logits_final: Tensor, batch: FactorGraphBatch,
                    spec: FactorGraphSpec) -> tuple[float, float]:
    """(cell_accuracy, puzzle_accuracy) over valid cells.

    Mirrors kenken_accuracy but parameterized on N and s_max.
    """
    gold       = batch.gold
    cell_valid = batch.cell_valid
    pred = value_logits_final.argmax(axis=-1) + 1          # (B, s_max)
    eq   = (pred == gold).cast(dtypes.float) * cell_valid
    n_valid = cell_valid.sum() + 1e-6
    cell_acc = eq.sum() / n_valid
    eq_p = (pred == gold).cast(dtypes.float) * cell_valid + (1.0 - cell_valid)
    puzzle_acc = eq_p.prod(axis=-1).mean()
    return float(cell_acc.realize().numpy()), float(puzzle_acc.realize().numpy())
