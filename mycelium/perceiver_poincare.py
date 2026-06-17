"""Perceiver-Poincaré (BRICK-1) — the ANCHORED perceiver revival.

The make-or-break test (docs/perceiver_poincare_design.md): does an ANCHORED
co-embedded perceiver hold the KenKen constraint-propagation engine, or is the
perceiver broken beyond the routing-bootstrap wall that killed it 5x (v118-v121,
v300)?

THE NEW INGREDIENT (the fix the perceiver never had): the Poincaré ball + the
Tier-2 g_phi anchor machinery. The latents anchor to the problem's CONSTRAINTS
(NOT hardcoded row/col/cage roles), each via the closed-form Poincaré anchor +
a ZERO-INIT g_phi over the constraint's cell-set, so at t=0 the latent->cell
routing reproduces the factor-graph membership -> NO random bootstrap.

ARCHITECTURE (minimal co-embedded perceiver, K breaths). Brick-1 ONLY:
  - LATENTS = factor-graph CONSTRAINTS. For KenKen the constraints are the rows,
    cols and cages. The code reads the factor graph GENERICALLY (a constraint =
    a SET of cells); it NEVER branches on a "row"/"col"/"cage" role for routing
    geometry — every constraint latent is anchored by the SAME closed-form anchor
    + zero-init g_phi(cell-set) path (build_constraint_membership in
    perceiver_poincare_data.py turns the raw cages/row/col structure into a
    uniform (L, cell-set) membership table; THIS module never looks at which
    relation a latent came from). Plus a small fixed +k GLOBAL latents at the
    origin (widest horizon).
  - CELLS = the 49 variables, co-embedded in the SAME Poincaré ball, anchored to
    their factor-graph positions (a per-cell tangent coordinate).
  - BREATH k: READ (each latent attends to cells via softmax over the d_hyp
    cross-distance) -> THINK (latent self-attention through the SHARED Pythia
    L0-L3 layers; same weights every breath) -> WRITE (each CELL attends to
    latents via the d_hyp cross-distance) -> per-cell value-codebook readout ->
    cell_logits_k. Persistent state across breaths = the LATENT hidden states.
  - LOSS: the per-breath weighted CE ladder (reuse kenken_loss).

THE TRIANGLE-INEQUALITY DECISION (Tier-2 §0; the t=0 anchor-check decides):
  A cell is in a row AND a col AND a cage. A SINGLE cell coordinate + a single
  d_hyp may not reproduce all three memberships (proven for cell->cell). BUT the
  perceiver graph is BIPARTITE + sparse (a cell links to only ~3 constraint
  latents), which MIGHT fit one ball. So we ATTEMPT the single unified ball
  first, then RUN the t=0 anchor-check (t0_anchor_check below). If a latent's
  top-attended cells do NOT match its constraint's cell-set to tolerance, we FALL
  BACK to per-constraint-type routing (separate cell fields per relation, the
  Tier-2 proven path). build_perceiver attaches BOTH; the breath forward reads
  PERCEIVER_BALL_PATH ("single" | "per_constraint") to pick.

SUBSTRATE LAWS (non-negotiable, Tier-2; ALL reused from kenken.py):
  - d_hyp boundary clamps (|z|^2 <= 1-1e-5, arccosh-arg >= 1+1e-7), where()-gated
    NaN guard (NOT a multiply-gate), no dtypes.float32 literal baked in the JIT
    step, finite -1e4 block, tangent-space coords (standard Adam) + tangent-norm
    rim guard. We REUSE _exp0_map / _d_hyp_* / the simplex anchors / the g_phi
    DeepSets encoder / the value_codebook readout / convergence_instrument.
"""
from __future__ import annotations

import math
import os
from typing import Any

import numpy as np
from tinygrad import Tensor, dtypes

from mycelium.kenken import (
    N_MAX, N_CELLS,
    _poincare_anchors, _tangent_for_anchors, _exp0_map,
    KENKEN_HYP_NORM_CLAMP, KENKEN_HYP_ARG_MIN,
)


# ---- env gates ---------------------------------------------------------------
PERCEIVER_TASK = int(os.environ.get("PERCEIVER_TASK", "0")) > 0
PERCEIVER_K_MAX = int(os.environ.get("PERCEIVER_K_MAX", "20"))
# Poincaré shell radius for the anchors (moderate rho -> off the boundary, the
# Tier-2 backward-stability discipline).
PERCEIVER_RHO = float(os.environ.get("PERCEIVER_RHO", "0.7"))
# Coordinate dim of the ball (shared by cells + latents). 48 holds an exact
# simplex for the corpus's max group count (mirrors KENKEN_HYP_DIM).
PERCEIVER_DIM = int(os.environ.get("PERCEIVER_DIM", "48"))
# READ / WRITE attention temperature on the -d_hyp logits. Lower -> sharper
# routing (closer to a hard top-k); higher -> softer (more gradient). The anchor
# sets the GEOMETRY; tau sets how hard the t=0 routing reads.
PERCEIVER_TAU = float(os.environ.get("PERCEIVER_TAU", "0.5"))
# Number of GLOBAL latents (placed near the origin = widest horizon, see all cells).
PERCEIVER_N_GLOBAL = int(os.environ.get("PERCEIVER_N_GLOBAL", "4"))
# Ball-path: "single" (one unified ball, the attempt) or "per_constraint" (Tier-2
# fallback: per-relation-TYPE cell fields). The t=0 anchor-check SELECTS this; the
# trainer sets the env from the check result. Default "single" (attempt first).
PERCEIVER_BALL_PATH = os.environ.get("PERCEIVER_BALL_PATH", "single").strip().lower()
# DeepSets g_phi geometry (mirror the Tier-2 Stage-2 encoder defaults).
PERCEIVER_GPHI_DPOS = int(os.environ.get("PERCEIVER_GPHI_DPOS", "32"))
PERCEIVER_GPHI_WIDTH = int(os.environ.get("PERCEIVER_GPHI_WIDTH", "64"))
PERCEIVER_GPHI_LAYERS = int(os.environ.get("PERCEIVER_GPHI_LAYERS", "2"))
# Latent hidden init scale (the per-constraint-type learned embedding tail).
PERCEIVER_LATENT_INIT = float(os.environ.get("PERCEIVER_LATENT_INIT", "0.02"))

# Substrate clamp constants (REUSE the exact Tier-2 values).
PERCEIVER_NORM_CLAMP = KENKEN_HYP_NORM_CLAMP
PERCEIVER_ARG_MIN = KENKEN_HYP_ARG_MIN
PERCEIVER_BLOCK = 1e4

# --- PERF FIX A: hoist the loop-invariant latent self-attn bias out of the
# K-breath loop. latent_valid is loop-invariant (extracted once, never mutated
# inside the loop), so _latent_self_attn_bias is byte-identical every breath.
# Building it once (mirroring v98 kenken.py:1121) eliminates K-1 per-breath
# np.eye host->GPU uploads + K-1 .contiguous() fusion barriers. =0 (default):
# rebuild in-loop, byte-identical to committed HEAD. =1: hoist before the loop.
PERCEIVER_HOIST_BIAS = int(os.environ.get("PERCEIVER_HOIST_BIAS", "0")) > 0

# --- PERF FIX B: run the 4-layer Pythia THINK in fp16 activations (AMD packed
# fp16 GEMM path, ~2x throughput, matching v98 kenken.py:1141) WITH a per-breath
# RMSNorm renorm on the accumulated latent residual (the fix for the breath-11
# fp16 overflow — the residual grows ~K x across breaths). =0 (default): fp32
# THINK + NO renorm, byte-identical to committed HEAD. =1: fp16 THINK + renorm.
PERCEIVER_FP16_THINK = int(os.environ.get("PERCEIVER_FP16_THINK", "0")) > 0
# RMSNorm epsilon for the fp16 renorm (Python float scalar — substrate-legal in
# JIT; bakes as a compile-time constant, NOT a float32 Tensor literal).
PERCEIVER_THINK_RMS_EPS = float(os.environ.get("PERCEIVER_THINK_RMS_EPS", "1e-6"))

# --- PERF FIX C: DEFUSE THE WHOLE-BREATH FUSED BACKWARD. Root cause (pinned, not
# re-derived): the breath loop has ZERO materialisation boundaries between
# READ -> THINK -> WRITE, so autograd fuses the entire breath's backward into ONE
# monolithic fp32 reduction mega-kernel (AMD knum 5423, arg 439) that runs at
# occupancy 0 / ~1 GFLOP/s = 44% of the 22.1 s/step (all occ-0 kernels = 95%).
# v98 (kenken.py) does NOT collapse because it materialises per-stage, so its
# backward stays many small occ-5-39 kernels (step 0.84 s). FIX: insert
# .contiguous() at the three breath seams (READ ctx, THINK output, WRITE ctx).
# .contiguous() is a REALIZE BARRIER — it changes ONLY fusion grouping, NOT
# values, so fwd+bwd are byte-identical; it just fragments the mega-kernel into
# the small per-stage kernels that tile well. Substrate-legal (no float32 literal,
# no per-param isnan). =0 (default): NONE fire, byte-identical to committed HEAD.
PERCEIVER_DEFUSE_BREATH = int(os.environ.get("PERCEIVER_DEFUSE_BREATH", "0")) > 0

# --- PERF FIX D: do NOT MATERIALISE the all-param grad_norm as a JIT output.
# Root cause (pinned, controlled A/B + single-output ablation; NOT re-derived):
# the all-param grad-clip sq_sum over the 31.6M param-grads is CHEAP when it is
# CONSUMED IN-GRAPH (it feeds the clip's clip_coef, so it tiles into the backward),
# but it becomes a standalone occ-0 reduction MEGA-KERNEL the moment grad_norm is
# forced to MATERIALISE as a RETURNED scalar (ablation: omit grad_norm -> 0.35 s;
# +grad_norm realized -> 24.07 s; all other outputs cost <=0.5 s combined). FIX:
# keep the in-graph grad-clip EXACTLY as HEAD (compute the norm, scale the grads ->
# training byte-identical), but when =1 DROP grad_norm from the return tuple / the
# packed-log stack so the sq_sum is consumed for the clip and never realized. The
# small g_phi-only grad norm (~0.1M params) is KEPT — it is a tiny reduction, not
# the mega-kernel, and it is the brick-2 re-freeze watch-item. =0 (default):
# return grad_norm, byte-identical to committed HEAD. =1: drop the logged scalar.
PERCEIVER_FAST_GRADNORM = int(os.environ.get("PERCEIVER_FAST_GRADNORM", "0")) > 0


# ---- cross-field hyperbolic distance (latents <-> cells) ---------------------
# kenken._d_hyp_pairwise is WITHIN one field (M,M). The perceiver needs a CROSS
# distance between two fields (latents (L,dim) vs cells (M,dim) -> (L,M)). Same
# substrate laws (boundary clamps, where()-gated guard, no float32 literal).

def _d_hyp_cross(za: Tensor, zb: Tensor) -> Tensor:
    """Pairwise Poincaré distance between two ball-coord fields.

    za: (..., A, dim) ball coords; zb: (..., B, dim) ball coords. Returns
    (..., A, B). arccosh(1 + 2|u-v|^2 / ((1-|u|^2)(1-|v|^2))) with the Tier-2
    substrate guards (spec §4): clamp |z|^2 <= 1-1e-5 for the denom; relu the
    squared-difference numerator; arg >= 1+1e-7; where()-gated isfinite guard.
    No dtypes.float32 literal baked as a graph const.
    """
    a32 = za.cast(dtypes.float)
    b32 = zb.cast(dtypes.float)
    # squared norms (raw for the numerator so coincident points give exactly 0;
    # clamped only protects the denominator factors).
    sa_raw = a32.pow(2).sum(axis=-1)                                   # (...,A)
    sb_raw = b32.pow(2).sum(axis=-1)                                   # (...,B)
    sa = sa_raw.clip(0.0, PERCEIVER_NORM_CLAMP)
    sb = sb_raw.clip(0.0, PERCEIVER_NORM_CLAMP)
    one_minus_a = (1.0 - sa)                                           # (...,A)
    one_minus_b = (1.0 - sb)                                           # (...,B)
    gram = a32 @ b32.transpose(-2, -1)                                 # (...,A,B) u.v
    diff_sq = (sa_raw.unsqueeze(-1) + sb_raw.unsqueeze(-2) - 2.0 * gram)
    diff_sq = diff_sq.relu()                                           # >=0 (fp guard)
    denom = one_minus_a.unsqueeze(-1) * one_minus_b.unsqueeze(-2)      # (...,A,B)
    arg = 1.0 + 2.0 * diff_sq / denom
    arg = arg.maximum(Tensor(PERCEIVER_ARG_MIN, dtype=dtypes.float))   # arg >= 1+1e-7
    # arccosh(arg)=log(arg+sqrt(arg^2-1)). The BACKWARD is 1/sqrt(arg^2-1), which
    # diverges as arg->1 (member cells anchored AT the latent -> d~=0). The Tier-2
    # boundary-gradient landmine. Floor (arg^2-1) by a small eps BEFORE the sqrt so
    # the sqrt backward (0.5/sqrt(inner)) stays finite — keeps the d~=0 member-cell
    # gradient bounded. forward effect negligible (sqrt(1e-10)~1e-5 -> log~1e-5 ~ 0).
    inner = (arg * arg - 1.0).relu() + 1e-10
    d = (arg + inner.sqrt()).log()                                     # arccosh
    big = Tensor(PERCEIVER_BLOCK, dtype=dtypes.float)
    d = d.isfinite().where(d, big)                                     # where()-gated guard
    return d


# ---- READ / WRITE cross-attention (the engagement mechanism) ----------------

def _cross_attn_weights(d_cross: Tensor, valid_src: Tensor, tau: float) -> Tensor:
    """The geometry-driven cross-attention weights (no src_hidden needed).

    attn = softmax_S( -d_cross / tau  + (valid_src-1)*BLOCK ). (B,Q,S) rows sum to
    1. The softmax is the standard Perceiver-IO cross-attention BUT with the
    GEOMETRY as the logit (no learned Q/K projection at the read/write boundary —
    the ball IS the router). Padding sources get a -BLOCK additive logit so they
    never win the softmax.
    """
    Bn = int(d_cross.shape[0])
    S = int(d_cross.shape[2])
    logits = (-d_cross / tau)                                          # (B,Q,S)
    src_block = (valid_src.reshape(Bn, 1, S) - 1.0) * PERCEIVER_BLOCK  # 0 valid / -BLOCK pad
    logits = logits + src_block
    logits = logits.clip(-PERCEIVER_BLOCK, PERCEIVER_BLOCK)
    return logits.softmax(axis=-1)                                     # (B,Q,S)


def _cross_attend(d_cross: Tensor, src_hidden: Tensor, valid_src: Tensor,
                  tau: float) -> tuple[Tensor, Tensor]:
    """One direction of cross-attention over -d_hyp logits.

    d_cross:    (B, Q, S) hyperbolic distance from each QUERY node to each SOURCE.
    src_hidden: (B, S, H) the source nodes' hidden states (what we pull).
    valid_src:  (B, S)    1.0 valid / 0.0 padding source.
    tau:        attention temperature on the -d_hyp logits.

    Returns (ctx (B,Q,H), attn (B,Q,S)).
    """
    attn = _cross_attn_weights(d_cross, valid_src, tau)               # (B,Q,S)
    ctx = attn @ src_hidden.cast(attn.dtype)                          # (B,Q,H)
    return ctx, attn


# ---- DeepSets g_phi over a constraint's cell-set (REUSE the Tier-2 pattern) ---
# Identical mechanism to kenken.gphi_cage_corrections, but GENERIC over the
# constraint-membership table (NOT cage-specific): one zero-init coord correction
# per LATENT slot, aggregated permutation-invariantly (segment-MEAN) over the
# cells in that latent's constraint set. rho's output layer is zero-init -> g_phi
# == 0 at t=0 -> latent coord == the closed-form anchor (the anchor discipline).

def _mlp_forward(x: Tensor, layers: list[tuple[Tensor, Tensor]]) -> Tensor:
    h = x
    n = len(layers)
    for i, (W, b) in enumerate(layers):
        h = h @ W.cast(h.dtype) + b.cast(h.dtype)
        if i < n - 1:
            h = h.gelu()
    return h


def _gphi_phi_layers(model: Any) -> list[tuple[Tensor, Tensor]]:
    return [(model.perc_gphi_phi_w[i], model.perc_gphi_phi_b[i])
            for i in range(len(model.perc_gphi_phi_w))]


def _gphi_rho_layers(model: Any) -> list[tuple[Tensor, Tensor]]:
    return [(model.perc_gphi_rho_w[i], model.perc_gphi_rho_b[i])
            for i in range(len(model.perc_gphi_rho_w))]


def gphi_latent_corrections(model: Any, membership: Tensor) -> Tensor:
    """Zero-init DeepSets correction per LATENT slot -> (B, L, dim).

    membership: (B, L, 49) float — 1.0 if cell j is in latent l's constraint set,
                else 0.0. PAD latents / pad cells are all-zero rows (-> zero aggregate
                -> rho of zeros -> exactly 0 with the zero-init last layer).
    Permutation-invariant by construction (segment-MEAN over a latent's cells; no
    ordering). MASK-ONLY input (cell positions + set size; never op-type/target).
    rho OUTPUT layer zero-init -> g_phi == 0 at t=0 -> latent coord == anchor.
    """
    Bn = int(membership.shape[0])
    L = int(membership.shape[1])
    pos_emb = model.perc_gphi_pos_emb                                  # (49, d_pos)
    phi_cells = _mlp_forward(pos_emb.cast(dtypes.float),
                             _gphi_phi_layers(model))                  # (49, W)
    Wd = int(phi_cells.shape[-1])
    m = membership.cast(dtypes.float)                                  # (B,L,49)
    counts = m.sum(axis=-1, keepdim=True)                              # (B,L,1)
    seg_sum = m @ phi_cells.reshape(1, N_CELLS, Wd)                    # (B,L,W)
    seg_mean = seg_sum / (counts + 1e-6)                               # (B,L,W)
    size_feat = (counts / float(N_CELLS))                             # (B,L,1)
    rho_in = Tensor.cat(seg_mean, size_feat, dim=-1)                  # (B,L,W+1)
    corr = _mlp_forward(rho_in, _gphi_rho_layers(model))             # (B,L,dim)
    return corr.cast(dtypes.float)


# ---- latent + cell coordinate construction (the ANCHOR) ----------------------

def _latent_cell_tangents(model: Any, membership: Tensor, ball_path: str,
                          latent_type: Tensor | None) -> Tensor:
    """Per-batch (B, 49, dim) cell TANGENT field each latent reads its cells from.

    The closed-form latent anchor (below) is the SEGMENT-MEAN of its constraint's
    cell tangents — so the latent lands co-located with its cells -> at t=0 the
    -d_hyp routing reads exactly that constraint's cells (the anchor that
    reproduces factor-graph membership). For "single" the field is the one unified
    cell tangent; for "per_constraint" each latent's cells are read in the latent's
    relation geometry, so we build a per-LATENT cell tangent field gated by
    latent_type (row->v_row, col->v_col, cage->v_cage).
    """
    Bn = int(membership.shape[0])
    if ball_path == "single":
        v = model.perc_cell_v.cast(dtypes.float)                       # (49,dim)
        return v.reshape(1, N_CELLS, -1).expand(Bn, N_CELLS, -1)
    # per_constraint: NOT used for the segment-mean (the anchor uses the relation's
    # OWN cell field). This branch returns the SINGLE field as a fallback; the
    # latent_coords per_constraint path computes a per-type anchor directly.
    return model.perc_cell_v.cast(dtypes.float).reshape(1, N_CELLS, -1).expand(Bn, N_CELLS, -1)


def _segment_mean_tangent(membership: Tensor, cell_tan: Tensor) -> Tensor:
    """Segment-MEAN of the cell tangents over each latent's cell-set -> (B,L,dim).

    membership (B,L,49); cell_tan (B,49,dim). Permutation-invariant (no ordering).
    Latents with no cells (globals / pad) get a zero mean -> origin anchor.
    """
    m = membership.cast(dtypes.float)                                 # (B,L,49)
    counts = m.sum(axis=-1, keepdim=True)                             # (B,L,1)
    seg_sum = m @ cell_tan.cast(dtypes.float)                         # (B,L,dim)
    return seg_sum / (counts + 1e-6)                                  # (B,L,dim)


def latent_coords(model: Any, membership: Tensor, ball_path: str = "single",
                  latent_type: Tensor | None = None) -> Tensor:
    """Constraint-anchored latent ball coords -> (B, L, dim).

    z_latent = exp_0( closed_form_base(constraint) + g_phi(cells_in_constraint) )
    where closed_form_base = the SEGMENT-MEAN of the constraint's cell TANGENTS (so
    the latent is co-located with its cells -> t=0 -d_hyp routing reproduces
    factor-graph membership), and g_phi is the zero-init DeepSets correction over
    the cell-set (-> at t=0 z_latent == the segment-mean anchor, no random
    bootstrap). GLOBAL latents (no membership) get a zero segment-mean -> origin
    -> widest horizon. Anchor added IN TANGENT space (§7 guards + exp_0 apply).

    For "per_constraint" the segment-mean is taken in the latent's RELATION cell
    field (row latent over v_row, etc.), gated by latent_type — so a row latent
    lands on its row anchor in the row geometry (the Tier-2 proven path).
    """
    Bn = int(membership.shape[0])
    L = int(membership.shape[1])
    if ball_path == "single":
        cell_tan = _latent_cell_tangents(model, membership, "single", None)  # (B,49,dim)
        base = _segment_mean_tangent(membership, cell_tan)            # (B,L,dim)
    else:
        # per-relation segment-mean, selected by latent_type (0=row,1=col,2=cage).
        v_row = model.perc_cell_v_row.cast(dtypes.float).reshape(1, N_CELLS, -1).expand(Bn, N_CELLS, -1)
        v_col = model.perc_cell_v_col.cast(dtypes.float).reshape(1, N_CELLS, -1).expand(Bn, N_CELLS, -1)
        v_cage = model.perc_cell_v_cage.cast(dtypes.float).reshape(1, N_CELLS, -1).expand(Bn, N_CELLS, -1)
        base_row = _segment_mean_tangent(membership, v_row)           # (B,L,dim)
        base_col = _segment_mean_tangent(membership, v_col)
        base_cage = _segment_mean_tangent(membership, v_cage)
        t = latent_type.clip(0, 2) if latent_type is not None else None
        if t is None:
            base = base_row
        else:
            is_row = (t == 0).cast(dtypes.float).reshape(Bn, L, 1)
            is_col = (t == 1).cast(dtypes.float).reshape(Bn, L, 1)
            is_cage = (t == 2).cast(dtypes.float).reshape(Bn, L, 1)
            base = base_row * is_row + base_col * is_col + base_cage * is_cage
    corr = gphi_latent_corrections(model, membership)                 # (B,L,dim) zero at t=0
    tan = base + corr                                                 # (B,L,dim) tangent
    return _exp0_map(tan)                                             # (B,L,dim) ball


def cell_coords(model: Any, relation: str = "single") -> Tensor:
    """Cell ball coords -> (49, dim). relation selects the cell field:
      "single"        -> the single unified cell field (the attempt).
      "row"/"col"/"cage" -> the per-constraint-type cell field (Tier-2 fallback).
    Tangent-space param -> exp_0 to the ball.
    """
    if relation == "single":
        v = model.perc_cell_v
    elif relation == "row":
        v = model.perc_cell_v_row
    elif relation == "col":
        v = model.perc_cell_v_col
    elif relation == "cage":
        v = model.perc_cell_v_cage
    else:
        raise ValueError(f"unknown cell relation {relation!r}")
    return _exp0_map(v.cast(dtypes.float))                             # (49, dim)


# ---- the breath cycle: READ -> THINK -> WRITE -> readout --------------------

def perceiver_breathing_forward(model: Any, batch: Any, K: int,
                                ball_path: str | None = None,
                                collect_engagement: bool = False):
    """Run K breaths of the anchored perceiver on a KenKen batch.

    batch carries (REUSE the KenKenBatch + the brick-1 additions in
    perceiver_poincare_data.py): input_cells (B,49) int, gold (B,49) int,
    cell_valid (B,49) f, value_domain_mask (B,49,N_MAX) f, latent_membership
    (B,L,49) f (1 = cell in constraint), latent_valid (B,L) f, latent_type
    (B,L) int (0=row/1=col/2=cage/3=global; ONLY for the per_constraint cell-field
    selection + the learned latent-type embedding — NEVER for routing geometry),
    cell_relation_id (B,49,3) int (the per-constraint-type cell field index per
    cell, only read in the per_constraint path).

    Persistent state across breaths = the LATENT hidden states (the perceiver's
    residual). Per breath:
      READ:  ctx_l = softmax_cells(-d_hyp(z_latent, z_cell)/tau) @ cell_hidden
             latent_in = latent_hidden + ctx_l + breath_marker
      THINK: latent_in -> 4 shared Pythia L0-L3 layers (FULL self-attn over latents)
             -> latent_hidden' (the deduction)
      WRITE: ctx_c = softmax_latents(-d_hyp(z_cell, z_latent)/tau) @ latent_hidden'
             cell_hidden = cell_embed + ctx_c   (cells re-read the latents each breath)
      READOUT: per-cell value-codebook logits (value-domain masked).

    Returns (cell_logits_history[K], engagement_history) where engagement_history
    is a list of K dicts (READ/WRITE select_norm + entropy + max-attn) when
    collect_engagement, else []. The latent hidden states are the residual.
    """
    assert hasattr(model, "perc_cell_v"), \
        "model has no perceiver params; was PERCEIVER_TASK set before attach?"
    if ball_path is None:
        ball_path = PERCEIVER_BALL_PATH

    from mycelium.breathing import _layernorm
    cfg = model.cfg

    state_embed = model.perc_state_embed           # (8, H)   cell input embedding
    position_embed = model.perc_position_embed     # (49, H)  cell position embedding
    breath_embed = model.perc_breath_embed         # (K_max, H) per-breath latent marker
    latent_type_embed = model.perc_latent_type_embed  # (4, H) learned per-type latent init
    value_codebook = model.perc_value_codebook     # (N_MAX, H)
    tau = PERCEIVER_TAU

    input_cells = batch.input_cells                # (B,49) int
    cell_valid = batch.cell_valid                  # (B,49) f
    value_domain_mask = batch.value_domain_mask    # (B,49,N_MAX) f
    membership = batch.latent_membership           # (B,L,49) f
    latent_valid = batch.latent_valid              # (B,L) f
    latent_type = batch.latent_type                # (B,L) int

    Bn = int(input_cells.shape[0])
    L = int(membership.shape[1])

    # --- co-embed: build the cell + latent ball coords (the ANCHOR) ---
    z_latent = latent_coords(model, membership, ball_path, latent_type)  # (B,L,dim)
    if ball_path == "single":
        z_cell = cell_coords(model, "single")                        # (49,dim)
        z_cell_b = z_cell.reshape(1, N_CELLS, -1).expand(Bn, N_CELLS, -1)
        # READ distance: latent -> cell. d_hyp(z_latent, z_cell) -> (B,L,49).
        d_read = _d_hyp_cross(z_latent, z_cell_b)                     # (B,L,49)
        # WRITE distance: cell -> latent = transpose of READ (symmetric metric).
        d_write = d_read.transpose(1, 2)                             # (B,49,L)
    else:
        # PER-CONSTRAINT fallback: each latent reads its cells in ITS relation's
        # geometry. cell_relation_id[b,cell,t] gives the cell-field index for
        # relation t in {row,col,cage}; a latent of type t uses the type-t field.
        z_cell_row = cell_coords(model, "row").reshape(1, N_CELLS, -1).expand(Bn, N_CELLS, -1)
        z_cell_col = cell_coords(model, "col").reshape(1, N_CELLS, -1).expand(Bn, N_CELLS, -1)
        z_cell_cage = cell_coords(model, "cage").reshape(1, N_CELLS, -1).expand(Bn, N_CELLS, -1)
        # per-latent READ distance in the latent's relation geometry.
        d_row = _d_hyp_cross(z_latent, z_cell_row)                   # (B,L,49)
        d_col = _d_hyp_cross(z_latent, z_cell_col)
        d_cage = _d_hyp_cross(z_latent, z_cell_cage)
        # select by latent_type (0=row,1=col,2=cage,3=global). Global latents read
        # in the ROW field (origin anchor -> uniform-ish anyway). one-hot select.
        t = latent_type.clip(0, 2)                                   # (B,L) global->row
        is_row = (t == 0).cast(dtypes.float).reshape(Bn, L, 1)
        is_col = (t == 1).cast(dtypes.float).reshape(Bn, L, 1)
        is_cage = (t == 2).cast(dtypes.float).reshape(Bn, L, 1)
        d_read = d_row * is_row + d_col * is_col + d_cage * is_cage   # (B,L,49)
        # WRITE: each CELL attends to latents. A cell sees a latent through the
        # latent's relation field, so d_write[b,cell,l] = d_read[b,l,cell].
        d_write = d_read.transpose(1, 2)                            # (B,49,L)

    # --- cell embedding (static reference; cells re-read latents each breath) ---
    from mycelium.kenken import embed_kenken
    cell_embed = embed_kenken(input_cells, state_embed, position_embed)  # (B,49,H)
    cell_embed = cell_embed.cast(dtypes.float)

    # --- latent hidden init: learned per-type embedding (NOT routing geometry) ---
    type_oh = latent_type.clip(0, 3).one_hot(4).cast(dtypes.float)    # (B,L,4)
    latent_hidden = (type_oh @ latent_type_embed.cast(dtypes.float))  # (B,L,H)
    latent_hidden = latent_hidden * latent_valid.reshape(Bn, L, 1).cast(dtypes.float)

    value_bias = (1.0 - value_domain_mask) * (-1e4)                   # (B,49,N_MAX)

    layers = list(model.block.layers)
    assert len(layers) >= 4, f"expected >=4 transformer layers; got {len(layers)}"
    K_max = int(breath_embed.shape[0])
    assert K <= K_max, f"K={K} exceeds K_max={K_max}"

    cell_logits_history = []
    engagement_history = []

    # PERF FIX A: hoist the loop-invariant latent self-attn bias BEFORE the loop
    # (latent_valid is loop-invariant — extracted at the top, never mutated in the
    # loop body, which writes only latent_hidden/cell_hidden + appends to history).
    # Same pattern v98 uses (build_kenken_attn_bias hoisted before the loop at
    # kenken.py:1121). Pays the np.eye upload + .contiguous() barrier ONCE instead
    # of K times, eliminating K-1 forced materialisation barriers that break
    # cross-breath kernel fusion. Numerically identical (pure loop-invariant CSE).
    if PERCEIVER_HOIST_BIAS:
        attn_bias = _latent_self_attn_bias(latent_valid, cfg.n_heads)  # (B,n_heads,L,L)

    # cells = the WRITE source's QUERY field; cell_hidden starts as the embedding.
    cell_hidden = cell_embed                                          # (B,49,H)

    for k in range(K):
        # === READ: each latent attends to cells via -d_hyp(z_latent, z_cell). ===
        be_k = breath_embed[k].reshape(1, 1, -1).cast(dtypes.float)   # (1,1,H)
        ctx_l, read_attn = _cross_attend(d_read, cell_hidden, cell_valid, tau)  # (B,L,H),(B,L,49)
        # DEFUSE seam (a): realize-barrier on the READ context so the READ backward
        # cannot fuse into the THINK backward. Byte-identical (.contiguous() = pure
        # fusion-grouping barrier, no value change). See PERCEIVER_DEFUSE_BREATH.
        if PERCEIVER_DEFUSE_BREATH:
            ctx_l = ctx_l.contiguous()
        latent_in = latent_hidden + ctx_l + be_k                     # (B,L,H)
        latent_in = latent_in * latent_valid.reshape(Bn, L, 1)        # zero pad latents

        # === THINK: latent self-attention through the shared Pythia L0-L3. ===
        # Full self-attn over latents (a latent attends to all VALID latents). The
        # mask is per-batch (latent validity). delta_gate-style residual: the THINK
        # output BLENDS with the prior latent state (perceiver residual). The latent
        # field is SMALL (L~59 vs 49 cells), so the THINK runs in FP32 by default —
        # fp16 carry overflowed (max ~65504) on the LATE breaths (the residual stream
        # grows ~K x across the 4-layer stack + K breaths -> breath-11 NaN). PERF
        # FIX B (PERCEIVER_FP16_THINK=1) runs the THINK in fp16 (~2x AMD throughput)
        # and tames that overflow with a per-breath RMSNorm renorm on the latent
        # residual BEFORE the fp16 cast (the renorm rescales the grown residual back
        # to unit RMS, so the fp16 softmax inputs stay bounded at every breath).
        # PERF FIX A (PERCEIVER_HOIST_BIAS): the bias is loop-invariant. When =1 it
        # was hoisted before the loop; when =0 rebuild it here (byte-identical HEAD).
        if not PERCEIVER_HOIST_BIAS:
            attn_bias = _latent_self_attn_bias(latent_valid, cfg.n_heads)  # (B,n_heads,L,L)
        if PERCEIVER_FP16_THINK:
            # Single-kernel RMSNorm renorm on the accumulated latent residual: the
            # residual grows ~K x across breaths, so renormalising to unit RMS
            # before the fp16 cast prevents the breath-11 fp16 overflow at the
            # source. PERCEIVER_THINK_RMS_EPS is a Python float scalar (legal in
            # JIT — bakes as a compile-time constant, NOT a float32 Tensor literal).
            rms = (latent_in.square().mean(axis=-1, keepdim=True) + PERCEIVER_THINK_RMS_EPS).rsqrt()
            h = (latent_in * rms).cast(dtypes.half)
        else:
            h = latent_in.cast(dtypes.float)
        for layer in layers[:4]:
            h = _latent_layer_forward(layer, h, attn_bias)
        # DEFUSE seam (b) — HIGHEST-LEVERAGE: realize-barrier on the THINK output
        # at the THINK/WRITE boundary (the seam the eater name 5423 arg 439 shows
        # fused). This is the cut that fragments the monolithic occ-0 mega-kernel.
        # Placed BEFORE the fp32 cast so it barriers the raw THINK output in either
        # dtype path. Byte-identical (.contiguous() = pure fusion-grouping barrier).
        if PERCEIVER_DEFUSE_BREATH:
            h = h.contiguous()
        # Cast the THINK output back to fp32 for the delta-gate blend so the
        # persistent latent_hidden residual is ALWAYS accumulated in fp32 (the
        # state that grows across K breaths). In the fp32 path h is already fp32
        # and this cast is a no-op (byte-identical to HEAD).
        h = h.cast(dtypes.float)
        gate_k = model.perc_delta_gate[k].cast(dtypes.float).reshape(1, 1, 1)
        latent_hidden = latent_hidden + gate_k * (h - latent_hidden)  # (B,L,H)
        latent_hidden = latent_hidden * latent_valid.reshape(Bn, L, 1)

        # === WRITE: each cell attends to latents via -d_hyp(z_cell, z_latent). ===
        ctx_c, write_attn = _cross_attend(d_write, latent_hidden, latent_valid, tau)  # (B,49,H),(B,49,L)
        # DEFUSE seam (c): realize-barrier on the WRITE context so the WRITE
        # backward cannot fuse back into the THINK backward (and so the next
        # breath's READ starts from a materialised cell state). Byte-identical
        # (.contiguous() = pure fusion-grouping barrier). See PERCEIVER_DEFUSE_BREATH.
        if PERCEIVER_DEFUSE_BREATH:
            ctx_c = ctx_c.contiguous()
        cell_hidden = cell_embed + ctx_c                            # (B,49,H) re-read latents
        cell_hidden = cell_hidden * cell_valid.reshape(Bn, N_CELLS, 1)

        # === READOUT: per-cell value-codebook logits (value-domain masked). ===
        x_ln = _layernorm(cell_hidden, model.ln_f_g, model.ln_f_b,
                          cfg.layer_norm_eps).cast(dtypes.float)
        cell_logits_k = x_ln @ value_codebook.T.cast(dtypes.float)  # (B,49,N_MAX)
        cell_logits_k = cell_logits_k + value_bias.cast(dtypes.float)
        cell_logits_history.append(cell_logits_k)

        # ENGAGEMENT (the kill metric): collect on the FIRST and LAST breath only
        # (the verdict reads b0 -> bN). Per-breath collection blows up the JIT graph
        # (entropy .log() + max + L2 over (B,L,49)+(B,49,L) x K breaths exceeds the
        # AM-driver graph limit -> replay hang). First+last is sufficient + safe.
        if collect_engagement and (k == 0 or k == K - 1):
            engagement_history.append(_engagement_stats(
                read_attn, cell_valid, write_attn, latent_valid))

    return cell_logits_history, engagement_history


def _latent_self_attn_bias(latent_valid: Tensor, n_heads: int) -> Tensor:
    """Per-batch (B, n_heads, L, L) additive bias: a latent attends to all VALID
    latents (full self-attn over the constraint pool); pad latents forced self-only.
    """
    Bn = int(latent_valid.shape[0])
    L = int(latent_valid.shape[1])
    eye = Tensor(np.eye(L, dtype=np.float32), dtype=dtypes.float).reshape(1, 1, L, L)
    valid_key = latent_valid.reshape(Bn, 1, 1, L)                     # key validity
    allow = valid_key.expand(Bn, 1, L, L)                            # (B,1,L,L) {0,1}
    # pad query rows -> self-only (well-defined softmax).
    valid_q = latent_valid.reshape(Bn, 1, L, 1)
    allow = allow * valid_q + eye * (1.0 - valid_q)
    bias = (1.0 - allow) * (-1e4)                                    # 0 allow / -1e4 block
    bias = bias.expand(Bn, n_heads, L, L)
    return bias.contiguous()


def _latent_layer_forward(layer: Any, x: Tensor, attn_bias: Tensor) -> Tensor:
    """One Pythia L0-L3 layer over the LATENT field (full self-attn, no RoPE).

    Mirror of kenken_layer_forward but over the (B, L, H) latent field instead of
    (B, 49, H) cells — the THINK step. Pythia-init shared weights, same every breath.
    """
    cfg = layer.cfg
    Bn, L, H = x.shape
    from mycelium.breathing import _layernorm
    attn_in = _layernorm(x, layer.shared.in_ln_g, layer.shared.in_ln_b, cfg.layer_norm_eps)
    mlp_in = _layernorm(x, layer.shared.post_ln_g, layer.shared.post_ln_b, cfg.layer_norm_eps)
    attn_in_dt = attn_in.cast(x.dtype) if attn_in.dtype != x.dtype else attn_in
    mlp_in_dt = mlp_in.cast(x.dtype) if mlp_in.dtype != x.dtype else mlp_in

    q = (attn_in_dt @ layer.wq + layer.bq).reshape(Bn, L, cfg.n_heads, cfg.head_dim).transpose(1, 2)
    k = (attn_in_dt @ layer.wk + layer.bk).reshape(Bn, L, cfg.n_heads, cfg.head_dim).transpose(1, 2)
    v = (attn_in_dt @ layer.shared.wv + layer.shared.bv).reshape(Bn, L, cfg.n_heads, cfg.head_dim).transpose(1, 2)

    scale = 1.0 / math.sqrt(cfg.head_dim)
    scores = q @ k.transpose(-2, -1) * scale                         # (B,n_heads,L,L)
    scores = scores + attn_bias.cast(scores.dtype)
    attn = scores.clip(-1e4, 1e4).softmax(-1)
    ctx = (attn @ v).transpose(1, 2).reshape(Bn, L, H)
    attn_out = ctx @ layer.shared.wo + layer.shared.bo

    ff = (mlp_in_dt @ layer.w_in + layer.b_in).gelu()
    ffn_out = ff @ layer.shared.w_out + layer.shared.b_out
    return x + attn_out + ffn_out


# ---- ENGAGEMENT INSTRUMENTATION (THE KILL METRIC) ---------------------------
# The prior perceivers DIED with select_norm ~0 (latents never pulled from the
# cells). This is the kill switch: if READ/WRITE engagement flatlines toward 0
# over the smoke, the architecture is dead; if it stays active + pulls gradient,
# the bootstrap is cured. All scalars are computed with pure tensor ops so they
# can live inside the JIT step (returned, .numpy()'d outside).

def _engagement_stats(read_attn: Tensor, cell_valid: Tensor,
                      write_attn: Tensor, latent_valid: Tensor) -> dict[str, Tensor]:
    """Compute the READ + WRITE engagement scalars (the kill metric).

    read_attn:  (B,L,49) latent->cell attention weights (rows sum to 1).
    write_attn: (B,49,L) cell->latent attention weights (rows sum to 1).

    Returns scalars (each a 0-dim Tensor):
      read_select_norm  = mean over valid latents of (1 - max_cell read_attn): how
                          much a latent SPREADS its read (1 = uniform, 0 = one cell).
                          A latent that pulls hard from one cell has a HIGH max -> a
                          LOW spread; the diagnostic engagement we want NON-zero is
                          the actual PULL = mean max-attn (high = engaged). We report
                          BOTH: read_max (mean max-attn, the PULL) + read_entropy.
      read_max          = mean over valid latents of max_cell read_attn (the PULL;
                          high = the latent reads SOME cell hard, not flat ~0).
      read_entropy      = mean over valid latents of the read-attn entropy (nats).
      write_max         = mean over valid cells of max_latent write_attn.
      write_entropy     = mean over valid cells of the write-attn entropy.
      read_select_norm  = mean over valid latents of ||ctx pull|| proxy = the L2 norm
                          of the read-attn vector (a perceiver-style "how much mass
                          is concentrated" scalar; flatlines to ~1/sqrt(S) if dead,
                          rises toward 1 as routing sharpens).
    """
    Bn = int(read_attn.shape[0])
    L = int(read_attn.shape[1])
    S = int(read_attn.shape[2])
    Lw = int(write_attn.shape[2])

    cv = cell_valid.reshape(Bn, 1, S).cast(dtypes.float)             # valid cells
    lv = latent_valid.reshape(Bn, L, 1).cast(dtypes.float)          # valid latents
    lvw = latent_valid.reshape(Bn, 1, Lw).cast(dtypes.float)
    cvw = cell_valid.reshape(Bn, N_CELLS, 1).cast(dtypes.float)

    eps = 1e-9
    # READ over valid latents (mask the per-latent stat by latent validity).
    r = read_attn.cast(dtypes.float)
    r_max = r.max(axis=-1)                                           # (B,L) max-attn
    r_ent = -(r * (r + eps).log()).sum(axis=-1)                     # (B,L) entropy
    r_norm = (r.pow(2).sum(axis=-1) + eps).sqrt()                   # (B,L) L2 (select_norm)
    lvf = latent_valid.cast(dtypes.float)                           # (B,L)
    lden = lvf.sum() + eps
    read_max = (r_max * lvf).sum() / lden
    read_entropy = (r_ent * lvf).sum() / lden
    read_select_norm = (r_norm * lvf).sum() / lden

    # WRITE over valid cells.
    w = write_attn.cast(dtypes.float)
    w_max = w.max(axis=-1)                                          # (B,49)
    w_ent = -(w * (w + eps).log()).sum(axis=-1)                    # (B,49)
    w_norm = (w.pow(2).sum(axis=-1) + eps).sqrt()                  # (B,49)
    cvf = cell_valid.cast(dtypes.float)                            # (B,49)
    cden = cvf.sum() + eps
    write_max = (w_max * cvf).sum() / cden
    write_entropy = (w_ent * cvf).sum() / cden
    write_select_norm = (w_norm * cvf).sum() / cden

    return {
        "read_max": read_max,
        "read_entropy": read_entropy,
        "read_select_norm": read_select_norm,
        "write_max": write_max,
        "write_entropy": write_entropy,
        "write_select_norm": write_select_norm,
    }


# ---- t=0 ANCHOR-CHECK (the ball-path selector) ------------------------------

def t0_anchor_check(model: Any, batch: Any, ball_path: str,
                    top_frac: float = 1.0) -> dict:
    """Does the anchored routing reproduce the factor-graph membership at t=0?

    For each VALID latent l with constraint cell-set C_l (|C_l| = c), check that
    the c cells with the LARGEST read-attention (smallest d_hyp) are EXACTLY C_l.
    Reported metrics (over valid latents, averaged):
      topk_recall  = |topk_cells(l) ∩ C_l| / |C_l|   (== precision since k=|C_l|)
      membership_match = fraction of latents whose topk_cells == C_l exactly
      mean_in_attn  = mean read-attn mass ON the constraint cells (should be ~1
                      if the latent reads its cells and ~nothing else).
    A high topk_recall / membership_match means the anchor reproduces the factor
    graph -> the single ball holds; a low one means a single cell coord can't hold
    row∪col∪cage -> fall back to per_constraint. NO TRAINING — pure t=0 read.
    """
    membership = batch.latent_membership                            # (B,L,49)
    latent_valid = batch.latent_valid                              # (B,L)
    cell_valid = batch.cell_valid                                  # (B,49)

    z_latent = latent_coords(model, membership, ball_path,
                             getattr(batch, "latent_type", None))   # (B,L,dim)
    if ball_path == "single":
        z_cell = cell_coords(model, "single").reshape(
            1, N_CELLS, -1).expand(int(membership.shape[0]), N_CELLS, -1)
        d_read = _d_hyp_cross(z_latent, z_cell)                    # (B,L,49)
    else:
        Bn = int(membership.shape[0])
        z_cell_row = cell_coords(model, "row").reshape(1, N_CELLS, -1).expand(Bn, N_CELLS, -1)
        z_cell_col = cell_coords(model, "col").reshape(1, N_CELLS, -1).expand(Bn, N_CELLS, -1)
        z_cell_cage = cell_coords(model, "cage").reshape(1, N_CELLS, -1).expand(Bn, N_CELLS, -1)
        d_row = _d_hyp_cross(z_latent, z_cell_row)
        d_col = _d_hyp_cross(z_latent, z_cell_col)
        d_cage = _d_hyp_cross(z_latent, z_cell_cage)
        t = batch.latent_type.clip(0, 2)
        is_row = (t == 0).cast(dtypes.float).reshape(Bn, int(z_latent.shape[1]), 1)
        is_col = (t == 1).cast(dtypes.float).reshape(Bn, int(z_latent.shape[1]), 1)
        is_cage = (t == 2).cast(dtypes.float).reshape(Bn, int(z_latent.shape[1]), 1)
        d_read = d_row * is_row + d_col * is_col + d_cage * is_cage

    read_attn = _cross_attn_weights(d_read, cell_valid, PERCEIVER_TAU)  # (B,L,49)
    read_np = read_attn.realize().numpy()                          # (B,L,49)
    mem_np = membership.realize().numpy()                          # (B,L,49)
    lv_np = latent_valid.realize().numpy()                         # (B,L)
    cv_np = cell_valid.realize().numpy()                           # (B,49)

    Bn, L, S = read_np.shape
    recalls = []
    matches = []
    in_attn = []
    for b in range(Bn):
        for l in range(L):
            if lv_np[b, l] < 0.5:
                continue
            cset = np.where(mem_np[b, l] > 0.5)[0]
            c = len(cset)
            if c == 0:
                continue  # global latents (no membership) — skip the recall metric
            # top-c cells by read-attn (largest attn == smallest d_hyp).
            order = np.argsort(-read_np[b, l])
            topk = set(order[:c].tolist())
            gold = set(cset.tolist())
            inter = len(topk & gold)
            recalls.append(inter / c)
            matches.append(1.0 if topk == gold else 0.0)
            in_attn.append(float(read_np[b, l, cset].sum()))
    out = {
        "ball_path": ball_path,
        "n_constraint_latents": len(recalls),
        "topk_recall": float(np.mean(recalls)) if recalls else 0.0,
        "membership_match": float(np.mean(matches)) if matches else 0.0,
        "mean_in_attn": float(np.mean(in_attn)) if in_attn else 0.0,
    }
    return out


# ---- model param attach ------------------------------------------------------

def attach_perceiver_params(model: Any, hidden: int, n_heads: int,
                            L_max: int, k_max: int | None = None,
                            dim: int | None = None, rho: float | None = None) -> None:
    """Allocate perceiver-poincaré params on `model` (a BreathingTransformer).

    Cells reuse the kenken cell readout machinery (value codebook + state embed +
    position embed, aligned-at-init). NEW perceiver params:
      perc_cell_v            (49, dim)    single-ball cell tangent coords (the attempt)
      perc_cell_v_{row,col,cage} (49,dim) per-constraint-type cell tangent (fallback)
      (latent anchor = per-batch SEGMENT-MEAN of the constraint's cell tangents;
       no separate latent-anchor table — see latent_coords)
      perc_latent_type_embed (4, H)       learned per-type latent hidden init
      perc_breath_embed      (K_max, H)   per-breath additive latent marker
      perc_delta_gate        (K_max,)     per-breath THINK residual blend
      g_phi DeepSets encoder (perc_gphi_*) — zero-init rho last layer (anchor disc.)
    """
    if k_max is None:
        k_max = PERCEIVER_K_MAX
    if dim is None:
        dim = PERCEIVER_DIM
    if rho is None:
        rho = PERCEIVER_RHO

    # ---- cell readout (REUSE the kenken aligned-init recipe) ----
    model.perc_position_embed = _build_perceiver_position_features(hidden)
    rng_cb = np.random.RandomState(2403)
    raw_cb = rng_cb.randn(max(hidden, N_MAX), hidden).astype(np.float32)
    q_cb, _ = np.linalg.qr(raw_cb)
    cb_unit = q_cb[:N_MAX].astype(np.float32)                          # (7, hidden)
    model.perc_value_codebook = Tensor(cb_unit * 0.1, dtype=dtypes.float).contiguous()
    state = np.zeros((N_MAX + 1, hidden), dtype=np.float32)
    state[0] = np.random.RandomState(2402).randn(hidden).astype(np.float32) * 0.02
    state[1:N_MAX + 1] = cb_unit                                       # given cells favor own value
    model.perc_state_embed = Tensor(state, dtype=dtypes.float).contiguous()

    # ---- per-breath additive latent marker (orthonormal init, mirror sudoku) ----
    breath_scale = float(os.environ.get("PERCEIVER_BREATH_EMBED_SCALE", "0.5"))
    rng_be = np.random.RandomState(2405)
    raw = rng_be.randn(max(k_max, hidden), hidden).astype(np.float32)
    q, _ = np.linalg.qr(raw)
    be = q[:k_max].astype(np.float32) * breath_scale
    model.perc_breath_embed = Tensor(be, dtype=dtypes.float).contiguous()
    # per-breath THINK residual blend (init 1.0 = full update; mirror sudoku).
    model.perc_delta_gate = Tensor.ones((k_max,), dtype=dtypes.float).contiguous()

    # ---- learned per-type latent hidden init (4 types: row/col/cage/global) ----
    lt = (np.random.RandomState(2407).randn(4, hidden) * PERCEIVER_LATENT_INIT).astype(np.float32)
    model.perc_latent_type_embed = Tensor(lt, dtype=dtypes.float).contiguous()

    # ---- cell ball coords: tangent params -> exp_0 (the ANCHOR for cells) ----
    # Single field: each cell at a max-separated simplex anchor by FLAT cell index
    # (so distinct cells are distinct points). Per-constraint fields: each cell at
    # its relation's GROUP anchor (cells in the same row/col/cage land together ->
    # a latent at that group's anchor reads exactly its cells in that geometry).
    # ALL tangent-space (exp_0 to the ball) so the §7 guards apply.
    model.perc_cell_v = _cell_single_field_tangent(dim, rho)           # (49,dim)
    v_row, v_col, v_cage = _cell_per_constraint_tangents(dim, rho)
    model.perc_cell_v_row = v_row
    model.perc_cell_v_col = v_col
    model.perc_cell_v_cage = v_cage

    # ---- latent anchor = SEGMENT-MEAN of the constraint's CELL tangents (computed
    # per-batch in latent_coords; NO separate latent-anchor table). The closed-form
    # base co-locates each latent with its constraint's cells -> t=0 -d_hyp routing
    # reproduces factor-graph membership (verified by the t=0 anchor-check). g_phi
    # (below, zero-init) adds the learnable per-instance correction. GLOBAL latents
    # have no membership -> a zero segment-mean -> origin -> widest horizon. The
    # `perc_n_global` scalar is kept as the attach sentinel + provenance. ----
    model.perc_n_global = int(PERCEIVER_N_GLOBAL)
    model.perc_dim = int(dim)

    # ---- g_phi DeepSets encoder (zero-init rho last layer -> g_phi==0 at t=0) ----
    attach_perceiver_gphi_params(model, dim=dim)


def _build_perceiver_position_features(hidden: int) -> Tensor:
    """(49, hidden) cell position embedding with row/col one-hots (reuse the kenken
    layout: [row one-hot | col one-hot | randn tail])."""
    pos = np.zeros((N_CELLS, hidden), dtype=np.float32)
    for i in range(N_CELLS):
        r, c = i // N_MAX, i % N_MAX
        if hidden >= N_MAX:
            pos[i, r] = 1.0
        if hidden >= 2 * N_MAX:
            pos[i, N_MAX + c] = 1.0
    rng = np.random.RandomState(240)
    tail = 2 * N_MAX
    if hidden > tail:
        pos[:, tail:] = rng.randn(N_CELLS, hidden - tail).astype(np.float32) * 0.02
    return Tensor(pos, dtype=dtypes.float).contiguous()


def _cell_single_field_tangent(dim: int, rho: float) -> Tensor:
    """(49, dim) single-ball cell tangent coords. Each of the 49 cells at a distinct
    max-separated simplex anchor (a 49-vertex simplex needs dim>=48; PERCEIVER_DIM=48
    holds it exactly). exp_0(this) -> 49 distinct ball points."""
    mu = _poincare_anchors(N_CELLS, dim, rho)                         # (49,dim) ball
    v = _tangent_for_anchors(mu).astype(np.float32)
    return Tensor(v, dtype=dtypes.float).contiguous()


def _cell_per_constraint_tangents(dim: int, rho: float):
    """Per-constraint-type cell tangent fields (Tier-2 fallback). Each cell at its
    GROUP anchor in each relation: row field -> cell's row anchor (7 groups), col
    field -> cell's col anchor, cage field -> a per-cell DISTINCT anchor (cage
    membership is per-instance, so the static cage field places each cell at its
    own anchor and the latent's g_phi handles the per-puzzle grouping). Returns
    (v_row, v_col, v_cage) each (49,dim) tangent Tensors."""
    rows_idx = np.array([i // N_MAX for i in range(N_CELLS)], dtype=np.int64)
    cols_idx = np.array([i % N_MAX for i in range(N_CELLS)], dtype=np.int64)
    row_mu = _poincare_anchors(N_MAX, dim, rho)                       # (7,dim)
    col_mu = _poincare_anchors(N_MAX, dim, rho)                       # (7,dim)
    z_row = row_mu[rows_idx]                                          # (49,dim)
    z_col = col_mu[cols_idx]                                          # (49,dim)
    cage_mu = _poincare_anchors(N_CELLS, dim, rho)                    # (49,dim) per-cell distinct
    v_row = Tensor(_tangent_for_anchors(z_row).astype(np.float32), dtype=dtypes.float).contiguous()
    v_col = Tensor(_tangent_for_anchors(z_col).astype(np.float32), dtype=dtypes.float).contiguous()
    v_cage = Tensor(_tangent_for_anchors(cage_mu).astype(np.float32), dtype=dtypes.float).contiguous()
    return v_row, v_col, v_cage


def attach_perceiver_gphi_params(model: Any, dim: int,
                                 d_pos: int | None = None,
                                 width: int | None = None,
                                 n_layers: int | None = None) -> None:
    """DeepSets g_phi encoder params (mirror the Tier-2 Stage-2 encoder). rho's
    OUTPUT layer is ZERO-INIT so g_phi==0 at t=0 (-> latent coord == anchor)."""
    if d_pos is None:
        d_pos = PERCEIVER_GPHI_DPOS
    if width is None:
        width = PERCEIVER_GPHI_WIDTH
    if n_layers is None:
        n_layers = max(1, PERCEIVER_GPHI_LAYERS)
    rng = np.random.RandomState(20260616)
    model.perc_gphi_pos_emb = Tensor(
        (rng.randn(N_CELLS, d_pos) * 0.1).astype(np.float32), dtype=dtypes.float).contiguous()

    def _mlp(dims: list[int], zero_last: bool):
        Ws, bs = [], []
        Llayers = len(dims) - 1
        for i in range(Llayers):
            din, dout = dims[i], dims[i + 1]
            if zero_last and i == Llayers - 1:
                W = np.zeros((din, dout), dtype=np.float32)
            else:
                W = (rng.randn(din, dout) * (1.0 / math.sqrt(din))).astype(np.float32)
            Ws.append(Tensor(W, dtype=dtypes.float).contiguous())
            bs.append(Tensor.zeros((dout,), dtype=dtypes.float).contiguous())
        return Ws, bs

    phi_dims = [d_pos] + [width] * n_layers
    model.perc_gphi_phi_w, model.perc_gphi_phi_b = _mlp(phi_dims, zero_last=False)
    rho_dims = [width + 1] + [width] * (n_layers - 1) + [dim]
    model.perc_gphi_rho_w, model.perc_gphi_rho_b = _mlp(rho_dims, zero_last=True)


def perceiver_parameters(model: Any, ball_path: str = "single") -> list[Tensor]:
    """Trainable perceiver params for the ACTIVE ball-path. Cell coords ARE trained
    (the cells learn their factor-graph positions; the anchor is the t=0 init) +
    type embed + breath markers + delta gate + value codebook/state/position +
    g_phi encoder.

    ONLY the active path's cell field(s) are returned: single -> perc_cell_v;
    per_constraint -> perc_cell_v_{row,col,cage}. The inactive field gets no
    gradient path, so including it would trip AdamW's grad-is-None assert (the
    same path-awareness kenken_train uses for its gated params)."""
    params: list[Tensor] = []
    if ball_path == "single":
        params.append(model.perc_cell_v)
    else:
        params += [model.perc_cell_v_row, model.perc_cell_v_col, model.perc_cell_v_cage]
    params += [
        model.perc_latent_type_embed,
        model.perc_breath_embed,
        model.perc_delta_gate,
        model.perc_value_codebook,
        model.perc_state_embed,
        model.perc_position_embed,
    ]
    params += perceiver_gphi_parameters(model)
    return params


def perceiver_gphi_parameters(model: Any) -> list[Tensor]:
    """g_phi encoder tensors as a flat list (pos_emb, phi W/b, rho W/b)."""
    if not hasattr(model, "perc_gphi_pos_emb"):
        return []
    params: list[Tensor] = [model.perc_gphi_pos_emb]
    for W, b in zip(model.perc_gphi_phi_w, model.perc_gphi_phi_b):
        params.append(W); params.append(b)
    for W, b in zip(model.perc_gphi_rho_w, model.perc_gphi_rho_b):
        params.append(W); params.append(b)
    return params


def perceiver_active_cell_coords(model: Any, ball_path: str = "single") -> list[Tensor]:
    """The cell tangent-coord tensors that the ACTIVE ball-path trains (gets gradient).
    single -> [perc_cell_v]; per_constraint -> [perc_cell_v_row, _col, _cage]. The
    inactive field has NO grad path -> must NOT be clamped (the brick-1 nice-to-have:
    the rim clamp touched inactive fields, perturbing a tensor that the optimizer
    never updates + that has no Adam moment, mismatching the freeze semantics)."""
    if ball_path == "single":
        return [model.perc_cell_v]
    return [model.perc_cell_v_row, model.perc_cell_v_col, model.perc_cell_v_cage]


def gphi_param_snapshot(model: Any) -> dict[str, "np.ndarray"]:
    """Numpy snapshot of the g_phi encoder tensors (pos_emb + phi/rho W,b) for the
    brick-2 'g_phi MOVES now that it is unfrozen' drift probe. The deduction test's
    make-or-break confirm: under FROZEN routing (brick-1) the rho OUTPUT layer is
    zero-init and the short smoke barely moves it; brick-2 must show non-zero drift."""
    snap: dict[str, np.ndarray] = {}
    snap["pos_emb"] = model.perc_gphi_pos_emb.detach().numpy().copy()
    for i, (W, b) in enumerate(zip(model.perc_gphi_phi_w, model.perc_gphi_phi_b)):
        snap[f"phi_w{i}"] = W.detach().numpy().copy()
        snap[f"phi_b{i}"] = b.detach().numpy().copy()
    for i, (W, b) in enumerate(zip(model.perc_gphi_rho_w, model.perc_gphi_rho_b)):
        snap[f"rho_w{i}"] = W.detach().numpy().copy()
        snap[f"rho_b{i}"] = b.detach().numpy().copy()
    return snap


def gphi_drift(model: Any, snapshot: dict[str, "np.ndarray"]) -> dict[str, float]:
    """Total + per-tensor L2 drift of the g_phi encoder vs a snapshot. The KEY
    brick-2 confirm is rho_drift (the OUTPUT layer): zero-init -> any non-zero drift
    means g_phi is co-adapting (unfrozen + pulling gradient). drift_total > 0 ==
    g_phi MOVES."""
    now = gphi_param_snapshot(model)
    per = {k: float(np.linalg.norm((now[k] - snapshot[k]).ravel()))
           for k in snapshot}
    total = float(np.sqrt(sum(d * d for d in per.values())))
    rho_total = float(np.sqrt(sum(per[k] ** 2 for k in per if k.startswith("rho_"))))
    return {"drift_total": total, "drift_rho": rho_total, **per}


def clamp_perceiver_tangent_norms(model: Any, max_znorm: float = 0.9,
                                  ball_path: str = "single") -> None:
    """Rim guard (Tier-2 §7): clamp the TRAINED tangent coords so |z|=tanh(|v|)
    stays <= max_znorm (keeps 1/(1-|z|^2) bounded in the d_hyp backward). Per-row
    radial shrink; in-place via .assign. Call AFTER each optimizer step.

    BRICK-2 FIX: scope the clamp to the ACTIVE ball_path's cell fields ONLY (the
    brick-1 caveat: the clamp touched all four fields, including the inactive
    single/per_constraint field that the optimizer never trains)."""
    max_vnorm = float(math.atanh(min(max(max_znorm, 0.0), 1.0 - 1e-7)))
    coord_tensors = perceiver_active_cell_coords(model, ball_path)
    for t in coord_tensors:
        v = t.cast(dtypes.float)
        norm = (v.pow(2).sum(axis=-1, keepdim=True) + 1e-12).sqrt()
        scale = (Tensor(max_vnorm, dtype=dtypes.float) / norm).minimum(
            Tensor(1.0, dtype=dtypes.float))
        t.assign((v * scale).cast(t.dtype)).realize()


def perceiver_state_dict(model: Any) -> dict[str, Tensor]:
    """State dict for perceiver params (excludes the FIXED latent anchor table —
    re-derived from the closed-form anchors at attach)."""
    sd = {
        "perc.cell_v": model.perc_cell_v,
        "perc.cell_v_row": model.perc_cell_v_row,
        "perc.cell_v_col": model.perc_cell_v_col,
        "perc.cell_v_cage": model.perc_cell_v_cage,
        "perc.latent_type_embed": model.perc_latent_type_embed,
        "perc.breath_embed": model.perc_breath_embed,
        "perc.delta_gate": model.perc_delta_gate,
        "perc.value_codebook": model.perc_value_codebook,
        "perc.state_embed": model.perc_state_embed,
        "perc.position_embed": model.perc_position_embed,
        "perc.gphi_pos_emb": model.perc_gphi_pos_emb,
    }
    for i, (W, b) in enumerate(zip(model.perc_gphi_phi_w, model.perc_gphi_phi_b)):
        sd[f"perc.gphi_phi_w{i}"] = W
        sd[f"perc.gphi_phi_b{i}"] = b
    for i, (W, b) in enumerate(zip(model.perc_gphi_rho_w, model.perc_gphi_rho_b)):
        sd[f"perc.gphi_rho_w{i}"] = W
        sd[f"perc.gphi_rho_b{i}"] = b
    return sd
