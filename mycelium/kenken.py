"""KenKen module — Sudoku-paradigm constraint propagation on arithmetic cages.

A direct mirror of `mycelium/sudoku.py` (the validated v98 paradigm), adapted to
KenKen. The breathing transformer runs K passes of 4 shared Pythia-410M L0-L3
layers over a FIXED N_max=7 grid (49 cells); each cell projects its final hidden
state through a 7-way number-level codebook (values 1..7).

WHAT IS COPIED UNCHANGED FROM sudoku.py (the validated machinery):
  - K-breath iterative-prefill loop (shared weights, residual stream as state)
  - per-breath additive markers, learnable per-breath delta_gate
  - per-breath calibration head, per-breath weighted CE ladder
  - row / col attention head groups (over the 49-cell N_max grid)
  - the number-level cell codebook readout (the WORKING 40-47% family, NOT the
    failed digit-level v105 family)

THE FOUR LOAD-BEARING DELTAS (pinned in project_csp_target_survey_jun14.md):
  1. VARIABLE-N PADDING. N in {5,6,7}; pad to N_max=7 (49 cells). A CELL-VALIDITY
     mask excludes padding cells everywhere; a VALUE-DOMAIN mask restricts an
     N-board's readout to values 1..N (values N+1..7 masked).
  2. CAGE MASK (replaces sudoku's box head group). The cage head group's mask is
     a SYMMETRIC MEMBERSHIP clique computed PER PUZZLE from `cages` (cell i ~ j
     iff same cage, plus self). It varies per instance → a per-BATCH input
     tensor, NOT baked. op-type (add/sub/mul/div) is NEVER a mask channel
     (the C2-eliminated v100 failure). row/col/global masks stay fixed.
  3. VERIFICATION INLET (the one genuinely new wiring). Per cage, a learned
     encode of [op-type 4-way+given embedding, LOG-BUCKETED target embedding,
     cage-size embedding] is added to EACH cage-cell's residual stream,
     RMSNorm'd (NOT behind a zero-init gate — inlets must be live at init or
     gradient never visits them: funded-vs-starved, §1A.E.8). The target is
     log-magnitude bucketed so a large cage target (e.g. mul up to 343) lives in
     the same bounded space as the codebook (the magnitude-mismatch LN-fix).
  4. CONVERGENCE INSTRUMENT (Property 2, pinned BEFORE the smoke). Per breath,
     per cell, belief = softmax(cell_logits_k). CONVERGED at breath k = all
     valid cells' JSD(belief_k, belief_{k-1}) < KENKEN_CONVERGE_JSD (=0.01).
     breath_count = first converged k (else K). settled = converged AND correct;
     stuck = converged AND wrong. The instrument that will later correlate
     breath_count vs deduction_depth AT FIXED N.

Env var gates (see scripts/kenken_smoke.py):
  KENKEN_TASK=1
  KENKEN_K_MAX=20             — match v98 Sudoku K
  KENKEN_CONSTRAINT_WEIGHT=0.3
  KENKEN_CALIB_WEIGHT=0.1
"""
from __future__ import annotations

import math
import os
import time as _time
from typing import Any

import numpy as np
from tinygrad import Tensor, dtypes
from tinygrad.engine.jit import TinyJit

from mycelium.config import Config
from mycelium.kenken_data import (
    N_MAX, N_CELLS, N_OPS, TARGET_BUCKETS,
)


KENKEN_TASK = int(os.environ.get("KENKEN_TASK", "0")) > 0
KENKEN_K_MAX = int(os.environ.get("KENKEN_K_MAX", "20"))
KENKEN_CONSTRAINT_WEIGHT = float(os.environ.get("KENKEN_CONSTRAINT_WEIGHT", "0.3"))
KENKEN_CALIB_WEIGHT = float(os.environ.get("KENKEN_CALIB_WEIGHT", "0.1"))

# ---- VALUE READOUT-FIX (rare-value undertraining; diag_kenken_cell_difficulty) --
# The diagnostic found round-0 (givens) acc < 1.0, ENTIRELY on N=7 boards, ENTIRELY
# the 6<->7 pair: value-7 appears ONLY on N=7 boards and value-6 only on N=6/N=7, so
# the largest values are gradient-STARVED (rare) -> their codebook rows are under-
# trained -> 6/7 collapse on N=7. Two cheap, env-gated, BYTE-IDENTICAL-WHEN-OFF fixes:
#
#  (1) KENKEN_VALUE_REWEIGHT (default 0 = OFF). When > 0, each supervised cell's CE is
#      weighted by w(gold_value) ∝ inverse frequency of that value in the TRAIN corpus
#      solutions, raised to KENKEN_VALUE_REWEIGHT_POW (default 0.5 = inverse-sqrt, mild),
#      NORMALIZED to mean 1.0 over supervised cells (overall loss scale / LR coupling
#      unchanged). OFF => all weights 1.0 => byte-identical CE.
#  (2) KENKEN_VALUE_BIAS (default = follows KENKEN_VALUE_REWEIGHT). When ON, a small
#      LEARNABLE per-value logit bias kenken_value_bias (N_MAX,), ZERO-INIT, is added
#      to the readout logits BEFORE the value-domain mask. Zero-init => byte-identical
#      at step 0; lets the model calibrate per-value offsets cheaply. The param is only
#      ATTACHED (and added to parameters / state_dict) when this knob is ON, so the OFF
#      path has no new params and is byte-identical to current v98.
KENKEN_VALUE_REWEIGHT = float(os.environ.get("KENKEN_VALUE_REWEIGHT", "0"))
KENKEN_VALUE_REWEIGHT_POW = float(os.environ.get("KENKEN_VALUE_REWEIGHT_POW", "0.5"))
# value-bias defaults to following the reweight knob (1 if reweight>0 else 0).
KENKEN_VALUE_BIAS = int(os.environ.get(
    "KENKEN_VALUE_BIAS", "1" if KENKEN_VALUE_REWEIGHT > 0 else "0")) > 0

# ---- v109pi per-breath Q rotation (π-cycled), OFF by default --------------------
# When ON, each breath k rotates the attention QUERY Q pairwise across head_dim by a
# SINGLE phase theta_k = phase_scale * k * π / K_max, UNIFORM across sequence positions
# (NOT position-dependent RoPE). K stays unrotated (rotating both Q and K by the same
# angle cancels in the dot product). At k=0 theta=0 -> cos=1,sin=0 -> Q unchanged, so
# breath 0 is identity. When OFF (default), the rotation helper is NEVER invoked and the
# forward is BYTE-IDENTICAL to the current v98 KenKen path. Ported from the validated
# v109pi factor-graph mechanism (mycelium/factor_graph_v109pi.py::_rotate_q_pi).
KENKEN_PI_ROPE = int(os.environ.get("KENKEN_PI_ROPE", "0")) > 0
KENKEN_PI_ROPE_PHASE_SCALE = float(os.environ.get("KENKEN_PI_ROPE_PHASE_SCALE", "1.0"))

# ---- MECHANISM 1: CODEBOOK-ORTHOGONALITY PENALTY (root-cause readout fix) ------
# The N=7 6<->7 readout collapse is NOT rare-value undertraining (that fix failed).
# Root cause = CODEBOOK DIRECTIONAL COLLINEARITY: cos(value-6, value-7)=0.173 vs
# mean|cos|=0.025 among values 1-5 (~7x more parallel). A scalar bias / CE reweight
# (the value_reweight + per-value-bias machinery above, left OFF) CANNOT ROTATE
# collinear directions apart — it only shifts logits, never the codebook geometry.
#
# KENKEN_CODEBOOK_ORTHO (float lambda, default 0.0 = OFF). When > 0, add a loss term
#     lambda * mean_{i!=j} ( cos(value_codebook_i, value_codebook_j) )^2
# i.e. the mean-square OFF-DIAGONAL of the ROW-NORMALIZED codebook Gram matrix. This
# is differentiable and pushes the value-codebook ROWS apart (decorrelates 6<->7),
# which a bias/reweight provably can't. Computed with pure tensor ops (normalize rows
# -> gram -> mask diagonal -> mean-square off-diag) entirely INSIDE the JIT-traced
# loss (no dtypes.float32 literal as a graph const, no .numpy() in the traced step).
# OFF (lambda == 0) => the term is exactly 0.0 => byte-identical to current v98.
KENKEN_CODEBOOK_ORTHO = float(os.environ.get("KENKEN_CODEBOOK_ORTHO", "0.0"))
# Optionally also decorrelate the state_embed value-rows (rows 1..N_MAX, the given-
# value lookups, which are aligned with the codebook at init so they share the SAME
# collinearity). Default follows the main knob (apply to both when ortho is ON). Set
# KENKEN_CODEBOOK_ORTHO_STATE=0 to penalize the readout codebook only.
KENKEN_CODEBOOK_ORTHO_STATE = int(os.environ.get(
    "KENKEN_CODEBOOK_ORTHO_STATE", "1" if KENKEN_CODEBOOK_ORTHO > 0 else "0")) > 0

# ---- MECHANISM 2: MASKED-CELL SELF-SUPERVISION (deep-chaining trainer) ---------
# The deep-chaining ceiling: cell-acc decays with deduction depth (r1=0.74 ->
# r5+=0.37). The model does shallow propagation; it can't chain deep. Fewer givens
# => the puzzle requires DEEPER deduction to recover the missing cells, so randomly
# HIDING givens during training directly trains the deep-chaining the ceiling needs.
#
# KENKEN_MASK_GIVENS_P (float prob, default 0.0 = OFF). When > 0, during TRAINING
# ONLY each GIVEN cell is, with probability p, treated as UNOBSERVED:
#   (1) removed from the INPUT (its state token set to 0 = unknown), AND
#   (2) ADDED to the supervised-solve set (the model must PREDICT it).
# Its value_domain (legal values 1..N) stays intact for the readout — only the
# OBSERVED-value input is removed; the model predicts it like any unobserved cell.
# Masked givens include the rare values 6/7 => direct gradient on them (helps the
# readout collinearity too). At EVAL masking is OFF (full givens, deterministic).
# OFF (p == 0) => no givens masked => byte-identical to current v98 trainer.
KENKEN_MASK_GIVENS_P = float(os.environ.get("KENKEN_MASK_GIVENS_P", "0.0"))

# ---- HYPERBOLIC MASK GENERATOR (foothold; FROZEN static-r anchored-to-hard-mask) -
# docs/hyperbolic_mask_generator_spec.md §0-§4 + §6.1. Replaces the HARDWIRED boolean
# attention masks (build_kenken_attn_bias) with masks GENERATED from continuous
# Poincare-ball coordinates, anchored at t=0 to reproduce the v98 hard mask to ~1e-3.
#
# KENKEN_HYP_MASK (default 0 = OFF). When OFF: the forward calls build_kenken_attn_bias
# UNCHANGED and NO new params are attached -> BYTE-IDENTICAL to the current v98 path.
# When ON (=1): the forward calls build_kenken_hyperbolic_attn_bias; attach_kenken_params
# allocates the per-relation TANGENT coordinate params (kenken_hyp_v_row/v_col), the
# fixed cage anchor table (kenken_hyp_cage_anchors), the per-cell group-index maps, and
# the closed-form per-relation r/alpha scalars (frozen here). load_ckpt tolerates a v98
# ckpt that lacks these keys (they are NOT in the saved state_dict; always re-derived
# from the closed-form anchors at attach time), so a hard-mask-trained ckpt evals with
# KENKEN_HYP_MASK=1 unchanged.
#
# FOOTHOLD ONLY: FROZEN coordinates, STATIC global r per relation, calibrated so the
# generated bias reproduces the {0, -1e4} boolean mask. NO relaxation / NO training /
# NO per-breath r_k (those are spec §5 phases 2/3 — explicitly NOT built here).
KENKEN_HYP_MASK = int(os.environ.get("KENKEN_HYP_MASK", "0")) > 0
# Poincare-ball shell radius for the anchors (0<rho<1). Moderate rho keeps anchors away
# from the boundary (spec §4/§7 backward-gradient stability) while giving a large d_out.
KENKEN_HYP_RHO = float(os.environ.get("KENKEN_HYP_RHO", "0.7"))
# Per-relation coordinate dim. Cage needs G-1 dims for an EXACT equiangular simplex up
# to n_cages_max+padding groups; 48 is ample for the corpus max of 41 cages (a 48d
# simplex realizes up to 49 groups exactly) and trivially holds the 7-group row/col
# fields. Shared across relations so d_hyp is one code path. (spec §1 G<=dim+1 simplex.)
KENKEN_HYP_DIM = int(os.environ.get("KENKEN_HYP_DIM", "48"))
# The v98 block magnitude the bias must reproduce (-1e4, finite — NOT -inf; spec §4).
KENKEN_HYP_BLOCK = 1e4
# Substrate clamps (spec §4): keep |z|^2 strictly < 1 and arccosh-arg >= 1.
KENKEN_HYP_NORM_CLAMP = 1.0 - 1e-5
KENKEN_HYP_ARG_MIN = 1.0 + 1e-7
# Calibration SHARPENING margin on alpha (spec §2 "dial it, don't fit it"). The closed-
# form alpha = 2*BLOCK/d_out makes the between-group softplus arg land EXACTLY on BLOCK,
# right at the -BLOCK floor; tiny fp32 wobble in the computed d_hyp can then leave the
# block value at -1e4 +/- ~3e-3. Multiplying alpha by this margin pushes the between-
# group arg WELL past the floor (softplus saturates harder) so the floored block value
# is EXACTLY -1e4 (diff -> 0), while the within-group arg stays deeply negative (-> ~0).
# Pure dialing — does NOT change any allow/block decision; only sharpens saturation.
KENKEN_HYP_ALPHA_MARGIN = float(os.environ.get("KENKEN_HYP_ALPHA_MARGIN", "4.0"))

# ---- STAGE-1 RELAXATION (spec §7/§8.2/§8.6 Stage 1). Default 0 = OFF (frozen foothold;
# NOTHING below changes). Requires KENKEN_HYP_MASK=1 and (in the trainer) a RESUME_FROM
# v98 ckpt. When ON the trainer FREEZES the v98 backbone and trains ONLY the ~3 hyperbolic
# coordinate tensors (kenken_hyp_v_row, kenken_hyp_v_col, kenken_hyp_cage_anchors), and
# the row/col bias is computed INSIDE the differentiable graph (not cached-constant) so
# gradients flow to v_row/v_col. This is the FIRST time the d_hyp BACKWARD runs — the
# 1/(1-|z|^2) boundary-gradient landmine is now live, so the §7 guards (grad clip +
# bounded tangent norms + where()-gated NaN guard) become load-bearing.
KENKEN_HYP_RELAX = int(os.environ.get("KENKEN_HYP_RELAX", "0")) > 0
# Epsilon tangent jitter on the row/col init when relaxing (spec §1 jitter, §8.6 Stage 1):
# row & col currently init to the SAME closed-form anchor table (init-degeneracy flagged
# in the foothold review). Adding eps*randn breaks that symmetry so the two relations can
# specialize. Cage anchors keep their (already-distinct simplex) scheme. ONLY applied when
# KENKEN_HYP_RELAX is ON; the frozen path stays the exact closed-form anchor (jitter=0).
KENKEN_HYP_JITTER = float(os.environ.get("KENKEN_HYP_JITTER", "1e-3"))
# Bounded-tangent-norm rim guard (spec §7 backward stability): after each optimizer step
# the trainer clamps |v| so |z|=tanh(|v|) stays off the rim. Default atanh(0.9) ~ 1.4722
# keeps the z-norm <= ~0.9 (the moderate-rho discipline that keeps 1/(1-|z|^2) bounded).
KENKEN_HYP_MAX_ZNORM = float(os.environ.get("KENKEN_HYP_MAX_ZNORM", "0.9"))
# RELAX-MODE ALPHA RECALIBRATION (the gradient-flow root cause). The frozen foothold
# calibrates alpha = margin*2*BLOCK/d_out with BLOCK=1e4 -> alpha ~ 1e4 -> the softplus is
# FULLY SATURATED on BOTH the within- AND between-group tails -> softplus'(±huge)=0 -> the
# d_hyp backward yields EXACTLY ZERO coord gradient (verified: grad_norm 0.0). Lowering
# KENKEN_HYP_ALPHA_MARGIN alone cannot fix this — alpha is dominated by 2*BLOCK/d_out, not
# the margin. In RELAX mode we instead calibrate alpha so the between-group softplus ARG
# lands at a MODEST target (KENKEN_HYP_RELAX_BLOCK_ARG, default 20): alpha = margin *
# target / r. At target=20 a blocked entry's bias is ~-20, i.e. attention leak exp(-20) ~
# 2e-9 (mask faithful WELL under the 1e-3 attention-weight tolerance) WHILE the softplus
# is in its responsive region so gradients flow (grad_norm ~1e4 instead of 0). ONLY used
# when KENKEN_HYP_RELAX is ON; the frozen foothold keeps the BLOCK-saturated calibration
# (byte-identical). r (= d_out/2) is unchanged in both modes.
KENKEN_HYP_RELAX_BLOCK_ARG = float(os.environ.get("KENKEN_HYP_RELAX_BLOCK_ARG", "20.0"))

# ---- STAGE-2 RELAXATION (spec §8.1/§8.6 Stage 2): DeepSets STRUCTURE ENCODER -----
# docs/hyperbolic_mask_generator_spec.md §8.0 CRUX + §8.6 Stage 2. Stage 1 proved the
# slot-anchor cage field CANNOT generalize under relaxation (held-out flat 0.827 — an
# arbitrary creation-order cage_id carries no structure to interpolate). Stage 2
# replaces the slot-anchor cage field with a SHARED structure encoder g_phi(cell-set)
# so the cage geometry can do ZERO-SHOT interpolation to UNSEEN cage configs:
#
#     cage_coord = anchor[id] + g_phi(cage)                       (zero-init correction)
#     g_phi(cage) = rho( MEAN_{cell in cage} phi(pos_emb[cell])  concat  [|cage|_scaled] )
#
# Requires KENKEN_HYP_GPHI=1 (this gate) AND KENKEN_HYP_MASK=1. KENKEN_HYP_GPHI is
# INDEPENDENT of KENKEN_HYP_RELAX for the bias path (the GPHI cage correction is added
# whenever GPHI is ON), but the TRAINER only trains g_phi under KENKEN_HYP_RELAX=1 (the
# Stage-1 freeze harness). OFF (default 0) => NO g_phi params attached, the cage path is
# the Stage-1/foothold slot-anchor field UNCHANGED -> byte-identical to Stage 1.
#
#  - pos_emb: a NEW learned (N_CELLS, d_pos) position embedding, ENCODER-OWNED (NOT the
#    frozen backbone's position features) — never entangled with the backbone.
#  - phi, rho: small MLPs (KENKEN_HYP_GPHI_LAYERS layers, width KENKEN_HYP_GPHI_WIDTH).
#  - PERMUTATION-INVARIANT by construction: aggregate a cage's cells via SEGMENT-MEAN
#    (one_hot(cage_id).T @ phi(pos_emb) / counts) — NO positional ordering in the agg.
#  - MEAN-pool (NOT sum): sum scales with |cage| -> pushes coords toward the rim (the
#    boundary landmine). |cage| enters as an explicit SCALAR feature (/N_CELLS) so size
#    info is not lost.
#  - MASK-ONLY inputs: cell positions + cage size ONLY (NEVER op-type / target — those
#    are the verification inlet's job; the mask is a pure partition).
#  - rho OUTPUT LAYER ZERO-INIT -> g_phi == 0 at t=0 -> cage_coord == anchor[id] -> the
#    bias is the BIT-EXACT validated foothold mask at t=0 (anchor discipline preserved).
#  - One coord PER CAGE, shared by its cells (clique). Tangent-space (exp_0 to the ball),
#    so the existing guards (tangent clamp via the cage anchor path, grad clip, boundary)
#    apply to g_phi's outputs too (the correction is added IN tangent space, then exp_0).
KENKEN_HYP_GPHI = int(os.environ.get("KENKEN_HYP_GPHI", "0")) > 0
# DeepSets MLP geometry (small per the spec: phi/rho ~1-2 layers, width ~64, d_pos ~32).
KENKEN_HYP_GPHI_DPOS = int(os.environ.get("KENKEN_HYP_GPHI_DPOS", "32"))
KENKEN_HYP_GPHI_WIDTH = int(os.environ.get("KENKEN_HYP_GPHI_WIDTH", "64"))
KENKEN_HYP_GPHI_LAYERS = int(os.environ.get("KENKEN_HYP_GPHI_LAYERS", "2"))
# EUCLIDEAN CONTROL ARM (spec §8.1 capacity-matched control). Requires KENKEN_HYP_GPHI=1.
# When ON: use Euclidean ||u-v|| instead of d_hyp (drop exp_0; the coord IS the point).
# SAME g_phi encoder (identical param count/shapes) — only the distance + the missing
# exp_0 differ. r/alpha recalibrated to the EUCLIDEAN anchor distances so the Euclidean
# arm ALSO reproduces the hard mask at t=0 (same zero-init discipline). EXACT capacity
# match -> any hyperbolic>Euclidean gap is the GEOMETRY, not capacity (the v112b
# attribution control). OFF (default) => the hyperbolic d_hyp path, unchanged.
KENKEN_HYP_EUCLID = int(os.environ.get("KENKEN_HYP_EUCLID", "0")) > 0

# Property-2 convergence instrument threshold (pinned BEFORE the smoke).
KENKEN_CONVERGE_JSD = float(os.environ.get("KENKEN_CONVERGE_JSD", "0.01"))

# Cage-size embedding domain. Cages in the corpus are 1..4 cells; allow a couple
# more for safety. Index 0 is reserved for padding (size-0) cages.
MAX_CAGE_SIZE = 8


# ---- value-frequency reweight table (rare-value undertraining fix) -----------

def compute_value_freq_weights(train_path: str,
                               pow_: float = KENKEN_VALUE_REWEIGHT_POW) -> np.ndarray:
    """Compute per-value CE reweights from the TRAIN corpus solutions.

    Counts how often each value 1..N_MAX appears across ALL solution cells in the
    train corpus, then returns a (N_MAX,) float array of weights:
        w(v) ∝ (1 / freq(v)) ** pow_
    i.e. inverse-frequency raised to `pow_` (0.5 = inverse-sqrt, mild). Values that
    never appear (none, since 1..min-N always do) get weight 0; the caller never
    indexes them (gold is always in [1, N]). The weights are NOT normalized here —
    normalization to mean-1.0 happens PER-BATCH over the supervised cells (so the
    overall loss scale / LR coupling is unchanged regardless of the value mix).

    Index v-1 holds the weight for value v (so a (B,49) gold-1 index maps directly).
    """
    from mycelium.kenken_data import load_jsonl  # local import (avoid import cycle)
    counts = np.zeros((N_MAX,), dtype=np.float64)
    recs = load_jsonl(train_path)
    for rec in recs:
        for row in rec["solution"]:
            for v in row:
                vi = int(v)
                if 1 <= vi <= N_MAX:
                    counts[vi - 1] += 1.0
    # Inverse-frequency ** pow. Guard zero-count values (weight 0; never indexed).
    w = np.zeros((N_MAX,), dtype=np.float32)
    nz = counts > 0
    w[nz] = (1.0 / counts[nz]) ** float(pow_)
    return w.astype(np.float32)


def attach_value_freq_weights(model: Any, train_path: str,
                              pow_: float = KENKEN_VALUE_REWEIGHT_POW) -> None:
    """Compute the value-frequency reweight table once and attach it to `model`.

    Stores both the numpy array (model.kenken_value_freq_weight_np) and a (N_MAX,)
    Tensor (model.kenken_value_freq_weight) for use inside the JIT-traced CE. No-op-
    safe to call multiple times (recomputes identically). Only call when
    KENKEN_VALUE_REWEIGHT > 0; the OFF path never references these attributes.
    """
    w_np = compute_value_freq_weights(train_path, pow_=pow_)
    model.kenken_value_freq_weight_np = w_np
    model.kenken_value_freq_weight = Tensor(w_np, dtype=dtypes.float).contiguous()


# ---- structured attention masks (FIXED part: row / col / global over 49 cells) ----

def _build_kenken_fixed_masks(n_heads: int = 16) -> tuple[Tensor, list[int]]:
    """Return ((n_heads, 49, 49) float mask for the FIXED head groups, head_split).

    Only row / col / global are fixed (they depend only on the N_max grid). The
    CAGE head group is per-instance (built per batch from `cages`) and is layered
    in at forward time — so the cage-head slots in this fixed mask are left as
    ZERO (block-all-but-self placeholder); the forward overwrites them with the
    per-batch cage mask.

    Head assignment (n_heads=16), mirroring sudoku's 5/5/5/1:
      heads 0-4   : ROW    (same row on the N_max grid)
      heads 5-9   : COL    (same col)
      heads 10-14 : CAGE   (per-instance — placeholder here)
      head  15    : GLOBAL (all valid cells; validity applied in forward)

    head_split = [n_row, n_col, n_cage, n_global] so the forward knows which
    head slots to fill with the per-batch cage mask.
    """
    mask = np.zeros((n_heads, N_CELLS, N_CELLS), dtype=np.float32)

    rows_idx = np.array([i // N_MAX for i in range(N_CELLS)])
    cols_idx = np.array([i % N_MAX for i in range(N_CELLS)])
    same_row = (rows_idx[:, None] == rows_idx[None, :]).astype(np.float32)
    same_col = (cols_idx[:, None] == cols_idx[None, :]).astype(np.float32)
    eye = np.eye(N_CELLS, dtype=np.float32)
    full = np.ones((N_CELLS, N_CELLS), dtype=np.float32)

    # 5/5/5/1 split for n_heads=16; degrade gracefully (mirror sudoku exactly).
    n_row = max(1, n_heads * 5 // 16)
    n_col = max(1, n_heads * 5 // 16)
    n_cage = max(1, n_heads * 5 // 16)
    n_global = max(1, n_heads - n_row - n_col - n_cage)
    assigned = n_row + n_col + n_cage + n_global
    if assigned != n_heads:
        n_global += (n_heads - assigned)

    h = 0
    for _ in range(n_row):
        mask[h] = np.maximum(same_row, eye)
        h += 1
    for _ in range(n_col):
        mask[h] = np.maximum(same_col, eye)
        h += 1
    for _ in range(n_cage):
        # placeholder — self-only; forward overwrites these head slots per batch.
        mask[h] = eye
        h += 1
    for _ in range(n_global):
        mask[h] = full
        h += 1

    return (Tensor(mask, dtype=dtypes.float).contiguous(),
            [n_row, n_col, n_cage, n_global])


def _build_kenken_position_features(hidden: int) -> Tensor:
    """Initialize (49, hidden) position embedding with row/col structural priors.

    Mirrors sudoku's _build_sudoku_position_features but for the N_max=7 grid and
    with NO box one-hot (KenKen has cages, not boxes; cage identity is supplied
    per-instance via the cage mask + verification inlet, not via a fixed position
    feature).

    Layout (hidden >= 14):
      [0..6]   = row one-hot (N_max=7)
      [7..13]  = col one-hot
      [14..]   = randn(0.02) learned tail
    """
    pos = np.zeros((N_CELLS, hidden), dtype=np.float32)
    for i in range(N_CELLS):
        r, c = i // N_MAX, i % N_MAX
        if hidden >= N_MAX:
            pos[i, r] = 1.0
        if hidden >= 2 * N_MAX:
            pos[i, N_MAX + c] = 1.0
    rng = np.random.RandomState(140)
    tail = 2 * N_MAX
    if hidden > tail:
        pos[:, tail:] = rng.randn(N_CELLS, hidden - tail).astype(np.float32) * 0.02
    return Tensor(pos, dtype=dtypes.float).contiguous()


# ---- per-cell embedding ------------------------------------------------------

def embed_kenken(input_cells: Tensor, state_embed: Tensor, position_embed: Tensor) -> Tensor:
    """Convert (B, 49) int cell states into (B, 49, hidden) embeddings.

    input_cells: int Tensor with values in [0, 7], where 0 = unknown.
    state_embed: (8, hidden) — lookup for {0=unknown, 1..7 = given value}.
    position_embed: (49, hidden) — learned position embedding (row/col info).

    Returns state + position (sum, matches BPE-token-emb scale; mirror sudoku).
    """
    B = int(input_cells.shape[0])
    one_hot = input_cells.one_hot(N_MAX + 1).cast(state_embed.dtype)   # (B, 49, 8)
    state = one_hot @ state_embed                                       # (B, 49, hidden)
    pos = position_embed.reshape(1, N_CELLS, -1).cast(state.dtype).expand(B, N_CELLS, -1)
    return state + pos


# ---- verification inlet (THE genuinely new wiring; §1A.E.8 normalized-not-gated) ----

def _rmsnorm_last(x: Tensor, eps: float = 1e-5) -> Tensor:
    """RMSNorm over the last axis (no learnable weight — the inlet's own scale is
    learned via the projection; this just bounds the contribution's magnitude so
    it can't collide with / dominate the cell residual scale)."""
    x32 = x.cast(dtypes.float)
    rms = (x32.pow(2).mean(axis=-1, keepdim=True) + eps).sqrt()
    return (x32 / rms)


def build_verification_inlet(model: Any,
                              cage_op: Tensor, cage_target: Tensor,
                              cage_size: Tensor, cell_cage_id: Tensor) -> Tensor:
    """Build the per-CELL verification inlet contribution, (B, 49, H).

    Per CAGE, encode = [op_embed, target_bucket_embed, size_embed] -> concat ->
    project to H. Then SCATTER to each cell via its cage id, RMSNorm the per-cell
    contribution, and return it (the caller adds it to the residual; it is NOT
    behind a zero-init gate — it must be LIVE at init so gradient visits the inlet
    params from step 0).

    Inputs:
      cage_op       (B, C) int   — op id per cage (0=given..4=div)
      cage_target   (B, C) int   — log-bucket id of cage target
      cage_size     (B, C) int   — cell count per cage (0 = padding cage)
      cell_cage_id  (B, 49) int  — which cage each cell is in (-1 = pad cell)

    Returns: (B, 49, H) RMSNorm'd verification inlet (zeros on pad cells).
    """
    op_embed_table     = model.kenken_op_embed       # (N_OPS, d_op)
    target_embed_table = model.kenken_target_embed   # (TARGET_BUCKETS, d_tgt)
    size_embed_table   = model.kenken_size_embed     # (MAX_CAGE_SIZE, d_size)
    W_inlet            = model.kenken_inlet_w         # (d_op+d_tgt+d_size, H)
    b_inlet            = model.kenken_inlet_b         # (H,)

    B = int(cage_op.shape[0])
    C = int(cage_op.shape[1])

    # Per-cage feature lookups via one-hot @ table (tinygrad-safe, mirror embed).
    op_oh  = cage_op.one_hot(N_OPS).cast(op_embed_table.dtype)             # (B, C, N_OPS)
    op_e   = op_oh @ op_embed_table                                        # (B, C, d_op)
    tgt_oh = cage_target.one_hot(TARGET_BUCKETS).cast(target_embed_table.dtype)
    tgt_e  = tgt_oh @ target_embed_table                                  # (B, C, d_tgt)
    sz_oh  = cage_size.clip(0, MAX_CAGE_SIZE - 1).one_hot(MAX_CAGE_SIZE).cast(size_embed_table.dtype)
    sz_e   = sz_oh @ size_embed_table                                     # (B, C, d_size)

    cage_feat = Tensor.cat(op_e, tgt_e, sz_e, dim=-1)                     # (B, C, d_total)
    cage_inlet = cage_feat @ W_inlet + b_inlet                            # (B, C, H)

    # Padding-cage mask: cages with size 0 contribute nothing.
    cage_alive = (cage_size > 0).cast(cage_inlet.dtype).reshape(B, C, 1)  # (B, C, 1)
    cage_inlet = cage_inlet * cage_alive

    # Scatter cage -> cell. cell_cage_id is -1 for padding cells; clip to 0 then
    # zero those rows via a validity mask (a cell is real iff cell_cage_id >= 0).
    cell_valid_cage = (cell_cage_id >= 0).cast(cage_inlet.dtype).reshape(B, N_CELLS, 1)
    cid = cell_cage_id.clip(0, C - 1)                                     # (B, 49)
    cell_oh = cid.one_hot(C).cast(cage_inlet.dtype)                       # (B, 49, C)
    cell_inlet = cell_oh @ cage_inlet                                     # (B, 49, H)
    cell_inlet = cell_inlet * cell_valid_cage                             # zero pad cells

    # RMSNorm the per-cell inlet contribution (bounds its scale; §1A.E.8). Apply
    # only on real cells; pad cells stay zero (their rms would be 0 → guard eps).
    inlet_norm = _rmsnorm_last(cell_inlet)                                # (B, 49, H)
    inlet_norm = inlet_norm * cell_valid_cage                             # re-zero pad cells
    return inlet_norm.cast(dtypes.float)


# ---- one transformer-layer forward (no RoPE, no causal) — mirror sudoku ------

def _rotate_q_pi(q: Tensor, cos_k: float, sin_k: float) -> Tensor:
    """Rotate Q pairwise across head_dim by a single (cos, sin). (v109pi Option A.)

    Input q: (B, n_heads, S, head_dim)
    Output:  same shape, with pairs (2i, 2i+1) rotated by angle theta_k = k·π/K_max.

    UNIFORM rotation across sequence positions (no position-dependence) — this is
    NOT position-dependent RoPE; it is a single per-breath angle applied to every
    position and every (2i,2i+1) pair:
      q_rot[..., 2i]   = cos · q[..., 2i]   - sin · q[..., 2i+1]
      q_rot[..., 2i+1] = sin · q[..., 2i]   + cos · q[..., 2i+1]
    cos_k/sin_k are python scalars (compile-time constants per breath), so the
    rotation bakes into the JIT graph with no new inputs. Ported verbatim from
    mycelium/factor_graph_v109pi.py::_rotate_q_pi.
    """
    H_dim = q.shape[-1]
    assert H_dim % 2 == 0, f"head_dim must be even, got {H_dim}"
    n_pairs = H_dim // 2

    q_pairs = q.reshape(*q.shape[:-1], n_pairs, 2)
    q_even = q_pairs[..., 0]                       # (..., n_pairs)
    q_odd  = q_pairs[..., 1]                       # (..., n_pairs)

    cos_t = Tensor([cos_k], dtype=q.dtype).reshape(*([1] * (q.ndim - 1)))
    sin_t = Tensor([sin_k], dtype=q.dtype).reshape(*([1] * (q.ndim - 1)))

    new_even = cos_t * q_even - sin_t * q_odd
    new_odd  = sin_t * q_even + cos_t * q_odd

    out_pairs = new_even.unsqueeze(-1).cat(new_odd.unsqueeze(-1), dim=-1)
    return out_pairs.reshape(*q.shape)


def kenken_layer_forward(layer: Any, x: Tensor, attn_bias: Tensor,
                         q_rot_cos: float | None = None,
                         q_rot_sin: float | None = None) -> Tensor:
    """Run one BreathingLayer forward with KenKen-style structured attention.

    layer:     a mycelium.breathing.BreathingLayer (Pythia-init wq/wk/.../shared).
    x:         (B, 49, H) residual stream.
    attn_bias: (B, n_heads, 49, 49) PER-BATCH additive bias (0 allow / -1e4 block).
               Per-batch because the cage head group varies per instance.
    q_rot_cos / q_rot_sin: OPTIONAL per-breath Q-rotation scalars (v109pi π-cycling).
               When BOTH are None (the default, and the only path when KENKEN_PI_ROPE
               is OFF) NO rotation tensor is ever constructed and the forward is
               BYTE-IDENTICAL to the original v98 KenKen attention. When supplied, Q
               (and ONLY Q) is rotated pairwise by (cos, sin) BEFORE the Q@K product.

    Mirror of sudoku_layer_forward minus the photon Q-rotation (off by default here).
    """
    cfg = layer.cfg
    B, S, H = x.shape
    assert int(S) == N_CELLS, f"kenken layer expects {N_CELLS} cells, got {S}"

    from mycelium.breathing import _layernorm
    attn_in = _layernorm(x, layer.shared.in_ln_g, layer.shared.in_ln_b, cfg.layer_norm_eps)
    mlp_in = _layernorm(x, layer.shared.post_ln_g, layer.shared.post_ln_b, cfg.layer_norm_eps)
    attn_in_dt = attn_in.cast(x.dtype) if attn_in.dtype != x.dtype else attn_in
    mlp_in_dt = mlp_in.cast(x.dtype) if mlp_in.dtype != x.dtype else mlp_in

    q = (attn_in_dt @ layer.wq + layer.bq).reshape(B, S, cfg.n_heads, cfg.head_dim).transpose(1, 2)
    k = (attn_in_dt @ layer.wk + layer.bk).reshape(B, S, cfg.n_heads, cfg.head_dim).transpose(1, 2)
    v = (attn_in_dt @ layer.shared.wv + layer.shared.bv).reshape(B, S, cfg.n_heads, cfg.head_dim).transpose(1, 2)

    # === v109pi: per-breath Q rotation (Q-ONLY; K stays unrotated). ===
    # Only when explicitly supplied — when OFF, this whole block is skipped so the
    # graph is byte-identical to the original v98 path.
    if q_rot_cos is not None and q_rot_sin is not None:
        q = _rotate_q_pi(q, q_rot_cos, q_rot_sin)

    scale = 1.0 / math.sqrt(cfg.head_dim)
    scores = q @ k.transpose(-2, -1) * scale                       # (B, n_heads, 49, 49)
    # attn_bias is per-batch (B, n_heads, 49, 49) — add directly.
    scores = scores + attn_bias.cast(scores.dtype)
    attn = scores.clip(-1e4, 1e4).softmax(-1)
    ctx = (attn @ v).transpose(1, 2).reshape(B, S, H)
    attn_out = ctx @ layer.shared.wo + layer.shared.bo

    ff = (mlp_in_dt @ layer.w_in + layer.b_in).gelu()
    ffn_out = ff @ layer.shared.w_out + layer.shared.b_out

    return x + attn_out + ffn_out


# ---- per-batch attention bias assembly ---------------------------------------

def build_kenken_attn_bias(model: Any, cage_mask: Tensor, cell_valid: Tensor) -> Tensor:
    """Assemble the per-batch (B, n_heads, 49, 49) additive attention bias.

    row/col/global heads come from the FIXED mask (model.kenken_fixed_mask); the
    CAGE head slots are filled with the per-instance symmetric cage clique. Then
    invalid (padding) cells are excluded everywhere: a padding cell attends only
    to itself, and no valid cell attends to a padding cell.

    cage_mask:  (B, 49, 49) symmetric membership clique (1 = same cage incl self).
    cell_valid: (B, 49) 1.0 valid / 0.0 padding.

    Returns (B, n_heads, 49, 49) bias: 0.0 allow, -1e4 block.
    """
    fixed = model.kenken_fixed_mask                  # (n_heads, 49, 49) {0,1}
    n_row, n_col, n_cage, n_global = model.kenken_head_split
    n_heads = int(fixed.shape[0])
    B = int(cage_mask.shape[0])

    # Broadcast fixed mask to batch.
    mask_b = fixed.reshape(1, n_heads, N_CELLS, N_CELLS).expand(B, n_heads, N_CELLS, N_CELLS)

    # Build the per-batch cage-head block by selecting cage heads with a one-hot
    # head selector (avoids in-place assignment, which tinygrad doesn't support).
    # head_is_cage: (n_heads,) 1 for cage-head slots else 0.
    cage_start = n_row + n_col
    head_is_cage_np = np.zeros((n_heads,), dtype=np.float32)
    head_is_cage_np[cage_start:cage_start + n_cage] = 1.0
    head_is_cage = Tensor(head_is_cage_np, dtype=dtypes.float).reshape(1, n_heads, 1, 1)

    # Add self (eye) to the cage clique so a cage cell always attends to itself.
    eye = Tensor(np.eye(N_CELLS, dtype=np.float32), dtype=dtypes.float).reshape(1, 1, N_CELLS, N_CELLS)
    cage_full = (cage_mask.reshape(B, 1, N_CELLS, N_CELLS).maximum(eye))   # (B,1,49,49)

    # Where head is a cage head, use cage_full; else keep the fixed mask.
    hic = head_is_cage.cast(mask_b.dtype)
    mask_b = mask_b * (1.0 - hic) + cage_full.cast(mask_b.dtype) * hic     # (B,n_heads,49,49)

    # Validity: a cell may be attended only if it is valid; a padding query cell
    # may attend only to itself.
    valid_key = cell_valid.reshape(B, 1, 1, N_CELLS)                       # key validity
    mask_b = mask_b * valid_key.cast(mask_b.dtype)                         # block invalid keys
    # Padding query rows: force self-only so softmax is well-defined (no all-block row).
    valid_q = cell_valid.reshape(B, 1, N_CELLS, 1)                         # query validity
    self_only = eye.expand(B, 1, N_CELLS, N_CELLS).cast(mask_b.dtype)
    mask_b = mask_b * valid_q.cast(mask_b.dtype) + self_only * (1.0 - valid_q.cast(mask_b.dtype))

    # Convert {0,1} allow-mask to additive bias {0, -1e4}.
    bias = (1.0 - mask_b) * (-1e4)
    return bias.contiguous()


# ---- HYPERBOLIC MASK GENERATOR (foothold; spec §0-§4) ------------------------
# Per-relation Poincare coordinate fields generate the SAME {0, -1e4} bias as the
# boolean masks above, but from continuous coordinates anchored at t=0. The
# triangle-inequality split (spec §0) is enforced structurally: ONE coordinate
# field PER RELATION (row / col / cage), never one field for all three.

def _simplex_dirs(G: int, dim: int) -> np.ndarray:
    """G max-separated UNIT directions in R^dim — a regular simplex (equiangular).

    Construction: rows of (I_G - 1/G) are the G vertices of a regular simplex
    centered at the origin (each = e_i - centroid), pairwise cos = -1/(G-1) exactly.
    They live in a (G-1)-dim subspace; normalize to unit and embed in R^dim (dim >=
    G-1 required for an exact simplex — spec §1 `G <= dim+1`). Returns (G, dim).

    EXACT for both KenKen relations: row/col G=7 (cos=-1/6), cage G up to ~41
    (cos~=-1/40), so the between-group distance d_out is UNIFORM -> the closed-form
    calibration (§2) reproduces the boolean {0,-1e4} mask by construction.
    """
    assert dim >= G - 1, f"need dim>=G-1 for an exact simplex (G={G}, dim={dim})"
    M = np.eye(G, dtype=np.float64) - np.ones((G, G), dtype=np.float64) / G   # (G,G)
    norms = np.linalg.norm(M, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    U = (M / norms)                                                           # (G,G) unit
    out = np.zeros((G, dim), dtype=np.float64)
    cols = min(dim, G)
    out[:, :cols] = U[:, :cols]
    return out


def _poincare_anchors(G: int, dim: int, rho: float) -> np.ndarray:
    """G anchors on the ball shell at Euclidean radius rho: mu_g = rho * dir_g.

    dir_g are the simplex unit directions (max-separated). |mu_g| = rho (spec §1).
    Returns (G, dim) float64.
    """
    return rho * _simplex_dirs(G, dim)


def _tangent_for_anchors(mu: np.ndarray) -> np.ndarray:
    """Invert the origin exp-map: v s.t. exp_0(v) = mu (spec §7 tangent param).

    exp_0(v) = tanh(|v|) * v/|v|. For a target ball point mu with |mu| = rho,
    the tangent has direction mu/|mu| and norm atanh(rho): v = atanh(|mu|) * mu/|mu|.
    Init the FROZEN tangent params so exp_0(v) == the closed-form anchors. Returns
    (G, dim) float64 (zeros where |mu|==0).
    """
    norm = np.linalg.norm(mu, axis=-1, keepdims=True)                         # (G,1)
    safe = np.where(norm > 0, norm, 1.0)
    rad = np.arctanh(np.clip(norm, 0.0, 1.0 - 1e-7))                          # atanh(|mu|)
    return (mu / safe) * rad


def _hyp_d_out(rho: float, G: int) -> float:
    """Closed-form between-group Poincare distance for a regular simplex (spec §2).

    Two anchors mu_a, mu_b with |mu|=rho and cos = -1/(G-1):
      |mu_a-mu_b|^2 = 2 rho^2 (1 - cos) = 2 rho^2 (1 + 1/(G-1)) = 2 rho^2 * G/(G-1)
      d_out = arccosh(1 + 2 |mu_a-mu_b|^2 / (1-rho^2)^2)
    Used to dial r = d_out/2, alpha = 2*BLOCK/d_out so within->~0, between->~-BLOCK.
    """
    if G <= 1:
        return 1.0
    sep_sq = 2.0 * rho * rho * (float(G) / float(G - 1))
    arg = 1.0 + 2.0 * sep_sq / ((1.0 - rho * rho) ** 2)
    return float(math.acosh(max(arg, 1.0 + 1e-12)))


def _euclid_d_out_line(rho: float, G: int) -> float:
    """Closed-form EUCLIDEAN between-group distance for the row/col TANGENT anchors (the
    Euclidean control's coord IS the tangent v, used directly — spec §8.1).

    The row/col tangents satisfy exp_0(v) = mu with |mu| = rho, so |v| = atanh(rho) and
    the directions are simplex unit vectors (cos = -1/(G-1)). Two such tangents v_a, v_b:
      ||v_a - v_b||^2 = 2 |v|^2 (1 - cos) = 2 atanh(rho)^2 * G/(G-1)
    so d_out_euclid = sqrt(that). For G=1 return 1.0 (degenerate).
    """
    if G <= 1:
        return 1.0
    vnorm = math.atanh(min(max(rho, 0.0), 1.0 - 1e-7))
    sep_sq = 2.0 * vnorm * vnorm * (float(G) / float(G - 1))
    return float(math.sqrt(max(sep_sq, 1e-24)))


def _min_between_anchor_distance_euclid(anchors: np.ndarray) -> float:
    """Minimum pairwise EUCLIDEAN distance between distinct ball-point anchors (spec §8.1).

    The cage anchors are stored as ball points (rho*dir) and used DIRECTLY as Euclidean
    points in the control arm (no exp_0), so the min between-cage separation is just the
    plain L2 min over those same points. anchors: (A, dim).
    """
    A = anchors.shape[0]
    sq = (anchors ** 2).sum(axis=-1)                                          # (A,)
    gram = anchors @ anchors.T                                                # (A,A)
    diff_sq = np.clip(sq[:, None] + sq[None, :] - 2.0 * gram, 0.0, None)
    d = np.sqrt(diff_sq)
    np.fill_diagonal(d, np.inf)
    return float(d.min())


def _exp0_map(v: Tensor) -> Tensor:
    """Origin exponential map exp_0(v) = tanh(|v|) * v/|v| -> ball point (spec §7).

    v: (..., dim) tangent. Returns (..., dim) ball coords with |z| < 1. JIT-safe:
    pure tensor ops, no float32 literal, eps-guarded division (no where needed — the
    norm guard keeps the divisor strictly positive).
    """
    v32 = v.cast(dtypes.float)
    norm = (v32.pow(2).sum(axis=-1, keepdim=True) + 1e-12).sqrt()             # (...,1)
    scale = norm.tanh() / norm                                               # tanh(|v|)/|v|
    return v32 * scale


def _d_hyp_pairwise(z: Tensor) -> Tensor:
    """Pairwise Poincare distance over a coordinate field z (...,M,dim) -> (...,M,M).

    d_hyp(u,v) = arccosh(1 + 2|u-v|^2 / ((1-|u|^2)(1-|v|^2))), implemented as
    arccosh(x) = log(x + sqrt(x^2-1)) for explicit clamp control (spec §4). Substrate
    laws: clamp |z|^2 <= 1-1e-5 BEFORE the metric; clamp arg >= 1+1e-7; where()-gated
    NaN guard around the metric (NOT a multiply-gate — NaN*0=NaN). No float32 literal.

    z: (..., M, dim). Returns (..., M, M) float.
    """
    z32 = z.cast(dtypes.float)
    M = int(z32.shape[-2])
    # |z|^2 per row, clamped strictly inside the ball (boundary NaN guard, §4).
    sq = z32.pow(2).sum(axis=-1)                                              # (...,M)
    sq = sq.clip(0.0, KENKEN_HYP_NORM_CLAMP)                                  # |z|^2 <= 1-1e-5
    one_minus = (1.0 - sq)                                                    # (...,M) in [1e-5,1]
    # |u-v|^2 = |u|^2 + |v|^2 - 2 u.v  (use the UNCLAMPED sq for the numerator norm so
    # coincident anchors give exactly 0; clamp only protects the denominator factors).
    sq_raw = z32.pow(2).sum(axis=-1)                                          # (...,M)
    gram = z32 @ z32.transpose(-2, -1)                                       # (...,M,M) u.v
    diff_sq = (sq_raw.unsqueeze(-1) + sq_raw.unsqueeze(-2) - 2.0 * gram)      # (...,M,M)
    diff_sq = diff_sq.relu()                                                 # >=0 (fp guard)
    denom = one_minus.unsqueeze(-1) * one_minus.unsqueeze(-2)                 # (...,M,M)
    arg = 1.0 + 2.0 * diff_sq / denom                                        # arccosh arg
    arg = arg.maximum(Tensor(KENKEN_HYP_ARG_MIN, dtype=dtypes.float))        # arg >= 1+1e-7
    # arccosh(arg) = log(arg + sqrt(arg^2 - 1)); arg>=1+1e-7 keeps the sqrt real.
    inner = (arg * arg - 1.0).relu()                                        # >=0
    d = (arg + inner.sqrt()).log()                                          # (...,M,M)
    # where()-gated NaN guard (spec §4): replace any non-finite metric entry with a
    # LARGE finite distance (-> blocked), NOT a multiply-gate. d is finite for the
    # clamped args; this is belt-and-suspenders for the AM driver.
    big = Tensor(1.0e4, dtype=dtypes.float)
    d = d.isfinite().where(d, big)
    return d


def _relation_bias_from_z(z: Tensor, r: float, alpha: float) -> Tensor:
    """Generate one relation's additive bias from coordinates (spec §2).

    bias(i,j) = -softplus(alpha * (d_hyp(z_i,z_j) - r)); 0 = attend, -BLOCK = block.
    With d_in~=0, d_out the simplex separation, r=d_out/2 and alpha=2*BLOCK/d_out:
      within  -> -softplus(-BLOCK) ~= 0
      between -> -softplus(+BLOCK)  = -BLOCK
    z: (..., M, dim). Returns (..., M, M) additive bias (clamped to -BLOCK floor so
    it mirrors the boolean {0,-BLOCK} magnitude rather than diverging).
    """
    d = _d_hyp_pairwise(z)                                                   # (...,M,M)
    arg = alpha * (d - r)                                                    # (...,M,M)
    sp = arg.softplus()                                                      # softplus(arg)
    bias = (0.0 - sp)                                                        # -softplus(.)
    # Floor at -BLOCK to match the boolean block magnitude exactly (softplus saturates
    # to ~BLOCK so this is a no-op for the calibrated values; it caps any overshoot).
    return bias.maximum(Tensor(-KENKEN_HYP_BLOCK, dtype=dtypes.float))


# ---- EUCLIDEAN CONTROL ARM (spec §8.1 capacity-matched control) --------------
# Identical machinery to the hyperbolic path EXCEPT (a) NO exp_0 (the coord IS the
# point — used directly as the Euclidean point) and (b) d = ||u-v|| (plain L2) instead
# of arccosh(...). SAME tensors (v_row/v_col/cage_anchors) -> EXACT capacity match;
# only the distance + the missing exp_0 differ. r/alpha recalibrated to the EUCLIDEAN
# anchor distances so the Euclidean arm ALSO reproduces the hard mask at t=0.

def _d_euclid_pairwise(c: Tensor) -> Tensor:
    """Pairwise EUCLIDEAN distance over a coordinate field c (...,M,dim) -> (...,M,M).

    d(u,v) = ||u - v|| = sqrt(|u|^2 + |v|^2 - 2 u.v). The coord c is used DIRECTLY (no
    exp_0 map — the Euclidean control's coord IS the point, spec §8.1). JIT/AM-safe: a
    relu() guard on the squared norm (fp wobble), a tiny eps under the sqrt (so the zero-
    distance gradient stays finite — the within-group d_in==0 diagonal otherwise yields a
    1/0 sqrt-backward), and the same where()-gated isfinite NaN guard as d_hyp. No
    float32 literal baked as a graph const.
    """
    c32 = c.cast(dtypes.float)
    sq = c32.pow(2).sum(axis=-1)                                             # (...,M) |c|^2
    gram = c32 @ c32.transpose(-2, -1)                                       # (...,M,M) u.v
    diff_sq = (sq.unsqueeze(-1) + sq.unsqueeze(-2) - 2.0 * gram)             # (...,M,M)
    diff_sq = diff_sq.relu()                                                 # >=0 (fp guard)
    d = (diff_sq + 1e-12).sqrt()                                            # ||u-v||, eps-safe
    big = Tensor(1.0e4, dtype=dtypes.float)
    d = d.isfinite().where(d, big)                                          # belt-and-suspenders
    return d


def _relation_bias_from_coord_euclid(c: Tensor, r: float, alpha: float) -> Tensor:
    """Euclidean analogue of _relation_bias_from_z (spec §8.1). Same softplus form:
    bias(i,j) = -softplus(alpha*(||c_i-c_j|| - r)); 0 = attend, -BLOCK = block. The coord
    c is the Euclidean point (NO exp_0). r/alpha are the Euclidean-calibrated constants.
    """
    d = _d_euclid_pairwise(c)                                                # (...,M,M)
    arg = alpha * (d - r)
    sp = arg.softplus()
    bias = (0.0 - sp)
    return bias.maximum(Tensor(-KENKEN_HYP_BLOCK, dtype=dtypes.float))


def _relation_bias_dispatch(coord: Tensor, r: float, alpha: float,
                            is_tangent: bool = True) -> Tensor:
    """Dispatch the per-relation bias by arm.

    is_tangent=True  (row/col): the coord is a TANGENT param. Hyperbolic arm applies
        exp_0(coord) then d_hyp; Euclidean arm uses the coord DIRECTLY as the point
        (drops exp_0, spec §8.1).
    is_tangent=False (cage): the coord is ALREADY a ball point (the foothold stores the
        cage anchors as ball points `rho*dir`, NOT tangents — preserved so the t=0 cage
        bias is BIT-EXACT to the foothold). Hyperbolic arm uses it directly in d_hyp (NO
        exp_0); Euclidean arm uses it directly as the Euclidean point. The g_phi
        correction is added to the cage coord in the SAME (ball / point) space.

    Single seam so the row/col and cage call sites stay arm-agnostic. NOTE: only the
    hyperbolic+tangent branch applies exp_0; this exactly mirrors the original foothold
    (row/col exp_0'd, cage used raw), so KENKEN_HYP_EUCLID=0 + KENKEN_HYP_GPHI=0 is
    byte-identical to the Stage-1 path.
    """
    if KENKEN_HYP_EUCLID:
        return _relation_bias_from_coord_euclid(coord, r, alpha)
    z = _exp0_map(coord) if is_tangent else coord
    return _relation_bias_from_z(z, r, alpha)


# ---- STAGE-2 DeepSets STRUCTURE ENCODER g_phi (spec §8.1/§8.6 Stage 2) --------
# cage_coord = anchor[id] + g_phi(cage), with
#   g_phi(cage) = rho( MEAN_{cell in cage} phi(pos_emb[cell])  concat  [|cage|/N_CELLS] ).
# PERMUTATION-INVARIANT by construction (segment-MEAN over a cage's cells; no ordering).
# rho's OUTPUT LAYER is ZERO-INIT -> g_phi == 0 at t=0 -> cage_coord == anchor[id] -> the
# t=0 bias is the BIT-EXACT validated foothold mask. MASK-ONLY inputs (cell positions +
# cage size; NEVER op-type/target). The correction is added IN TANGENT space (before the
# arm's exp_0 / Euclidean point), so the §7 boundary guards still apply.

def _mlp_forward(x: Tensor, layers: list[tuple[Tensor, Tensor]]) -> Tensor:
    """Tiny MLP: GELU between hidden layers, NO activation after the last (so a zero-init
    last layer yields exactly 0). layers = [(W0,b0), (W1,b1), ...]; x @ W + b each step.
    """
    h = x
    n = len(layers)
    for i, (W, b) in enumerate(layers):
        h = h @ W.cast(h.dtype) + b.cast(h.dtype)
        if i < n - 1:
            h = h.gelu()
    return h


def _gphi_phi_layers(model: Any) -> list[tuple[Tensor, Tensor]]:
    return [(model.kenken_gphi_phi_w[i], model.kenken_gphi_phi_b[i])
            for i in range(len(model.kenken_gphi_phi_w))]


def _gphi_rho_layers(model: Any) -> list[tuple[Tensor, Tensor]]:
    return [(model.kenken_gphi_rho_w[i], model.kenken_gphi_rho_b[i])
            for i in range(len(model.kenken_gphi_rho_w))]


def gphi_cage_corrections(model: Any, cage_ids: Tensor) -> Tensor:
    """DeepSets g_phi cage corrections, ONE coord per CAGE-ANCHOR slot -> (B, A, dim).

    Computes, per batch element b and per cage-anchor slot a:
      g_phi(cage_a) = rho( MEAN_{cell: cell_cage_id[b,cell]==a} phi(pos_emb[cell])
                           concat [count_a / N_CELLS] )
    via a SEGMENT-MEAN (one_hot(cage_id).T @ phi(pos_emb) / counts) — permutation-
    invariant (no cell ordering enters). Empty slots (count 0) get a zero aggregate and a
    zero size-feature -> rho of an all-zero input; with rho's zero-init last layer that is
    exactly 0 at t=0 (and the slot is unused / blocked by validity anyway).

    Returned shape (B, A, dim) aligns with model.kenken_hyp_cage_anchors (A, dim): the
    caller adds it as anchor[a] + correction[b,a], then indexes per cell by cage id. dim
    matches the anchor dim; rho's output width == dim.

    cage_ids: (B, 49) int — cell's cage id (-1 = padding cell, mapped to a per-cell unique
              pad anchor slot by the bias builder; here pad cells are EXCLUDED from every
              real cage's aggregate via the >=0 validity).
    """
    A = int(model.kenken_hyp_cage_anchors.shape[0])
    B = int(cage_ids.shape[0])

    # phi over the ENCODER-OWNED position embedding (NOT the backbone's). (49, d_pos)->(49,W).
    pos_emb = model.kenken_gphi_pos_emb                                      # (49, d_pos)
    phi_cells = _mlp_forward(pos_emb.cast(dtypes.float),
                             _gphi_phi_layers(model))                        # (49, W)
    Wd = int(phi_cells.shape[-1])

    # SEGMENT membership: one-hot(cage_id) over A slots, zeroing PAD cells (cid<0).
    # cid clipped to [0, A-1] for the one-hot; the pad-cell rows are then masked to 0 so
    # they contribute to NO real cage (a pad cell joins no cage's aggregate).
    is_real = (cage_ids >= 0).cast(dtypes.float).reshape(B, N_CELLS, 1)      # (B,49,1)
    cid = cage_ids.cast(dtypes.float).clip(0.0, float(A - 1)).cast(dtypes.int)
    cell_oh = cid.one_hot(A).cast(dtypes.float)                             # (B,49,A)
    cell_oh = cell_oh * is_real                                            # zero pad cells
    # counts per slot (B,A): how many real cells in each cage.
    counts = cell_oh.sum(axis=1)                                           # (B,A)
    # segment SUM of phi over each slot's cells: (B,A,49) @ (49,W) -> (B,A,W).
    seg_sum = cell_oh.transpose(1, 2) @ phi_cells.reshape(1, N_CELLS, Wd)   # (B,A,W)
    # segment MEAN (NOT sum): divide by counts (eps-guarded -> empty slots stay ~0).
    seg_mean = seg_sum / (counts.reshape(B, A, 1) + 1e-6)                   # (B,A,W)
    # explicit |cage| SCALAR feature (mean-pool loses size; add it back), scaled /N_CELLS.
    size_feat = (counts / float(N_CELLS)).reshape(B, A, 1)                  # (B,A,1)
    rho_in = Tensor.cat(seg_mean, size_feat, dim=-1)                        # (B,A,W+1)
    # rho: (W+1) -> dim, ZERO-INIT last layer => g_phi == 0 at t=0.
    corr = _mlp_forward(rho_in, _gphi_rho_layers(model))                    # (B,A,dim)
    return corr.cast(dtypes.float)


def build_kenken_hyperbolic_attn_bias(model: Any, cage_ids: Tensor,
                                      cell_valid: Tensor,
                                      row_col_bias_cache: Tensor | None = None
                                      ) -> Tensor:
    """HYPERBOLIC drop-in for build_kenken_attn_bias (same (B,n_heads,49,49) shape).

    row/col heads come from the FIXED z_row/z_col fields (CONSTANT across batches ->
    cacheable, computed outside the JIT step). cage heads are per-instance: each cell
    is placed on its cage's anchor (a fixed simplex anchor table indexed by cell's
    cage id), so the per-batch cage bias regenerates the membership clique. global
    head = 0 (all attend). Validity masking mirrors build_kenken_attn_bias exactly:
    invalid KEYS blocked everywhere; padding QUERY rows forced self-only.

    cage_ids:  (B, 49) int — cell's cage id (-1 = padding cell). From batch.cell_cage_id.
    cell_valid:(B, 49) float — 1.0 valid / 0.0 padding.
    row_col_bias_cache: OPTIONAL precomputed (2, 49, 49) [row_bias, col_bias] constant
        (spec §3/§4: precompute row/col OUTSIDE the traced step). When None, computed
        here from the frozen fields.

    Returns (B, n_heads, 49, 49) additive bias: 0.0 allow, -1e4 block. ~1e-3-identical
    to build_kenken_attn_bias at t=0 (the foothold replication proof).
    """
    n_row, n_col, n_cage, n_global = model.kenken_head_split
    n_heads = n_row + n_col + n_cage + n_global
    B = int(cage_ids.shape[0])
    BLOCK = KENKEN_HYP_BLOCK

    # --- row / col relation biases (CONSTANT) ---
    # ARM DISPATCH (spec §8.1): KENKEN_HYP_EUCLID ON => Euclidean ||u-v|| on the coord
    # used DIRECTLY (no exp_0); OFF => hyperbolic exp_0(coord) then d_hyp. row/col carry
    # NO g_phi correction (g_phi is cage-only — the partition that VARIES is the cage).
    if row_col_bias_cache is not None:
        row_bias = row_col_bias_cache[0]                                     # (49,49)
        col_bias = row_col_bias_cache[1]                                     # (49,49)
    else:
        row_bias = _relation_bias_dispatch(model.kenken_hyp_v_row,
                                           model.kenken_hyp_r_row,
                                           model.kenken_hyp_alpha_row)        # (49,49)
        col_bias = _relation_bias_dispatch(model.kenken_hyp_v_col,
                                           model.kenken_hyp_r_col,
                                           model.kenken_hyp_alpha_col)        # (49,49)

    # --- cage relation bias (PER-INSTANCE): place each cell on its cage coord ---
    # cage anchors: a fixed (A, dim) simplex table (A = n_cage_anchors >= max cages +
    # padding). Padding cells (cage_id=-1) are sent to a DISTINCT sentinel anchor each
    # (cid = A-1-self? -> instead clip to a per-cell unique pad anchor) so they are far
    # from all real cells (-> -BLOCK), then the self-only override fixes the diagonal.
    cage_anchors = model.kenken_hyp_cage_anchors                             # (A, dim) Tensor
    A = int(cage_anchors.shape[0])
    # STAGE-2 (spec §8.6): the cage COORD per slot = anchor[a] + g_phi(cage_a) (a zero-init
    # structure correction). g_phi==0 at t=0 (rho last layer zero-init) -> cage_coord ==
    # anchor[id] -> the BIT-EXACT validated foothold mask. The correction is added IN the
    # SAME (tangent for hyp / point for euclid) space the anchor lives in, so the arm's
    # exp_0 / Euclidean point + the §7 guards apply unchanged. OFF (GPHI off) => the
    # Stage-1 slot-anchor path, byte-identical.
    if KENKEN_HYP_GPHI:
        cage_corr = gphi_cage_corrections(model, cage_ids)                  # (B,A,dim)
        cage_coord_table = cage_anchors.reshape(1, A, -1).cast(dtypes.float) + cage_corr
    else:
        cage_coord_table = cage_anchors.reshape(1, A, -1).cast(dtypes.float).expand(
            B, A, int(cage_anchors.shape[-1]))                              # (B,A,dim)
    # Real cells: use cage id. Padding cells (cid<0): assign a unique anchor index per
    # cell position so no two padding cells share an anchor (each padding cell isolated;
    # validity masking forces self-only anyway). pad index = (A - N_CELLS + i), guaranteed
    # distinct and disjoint from the real cage-id range [0, max_cages).
    pos = Tensor(np.arange(N_CELLS, dtype=np.float32)).reshape(1, N_CELLS)    # (1,49)
    is_pad = (cage_ids < 0).cast(dtypes.float)                               # (B,49)
    pad_idx = (float(A - N_CELLS) + pos).expand(B, N_CELLS)                  # (B,49)
    real_idx = cage_ids.cast(dtypes.float).clip(0.0, float(A - 1))           # (B,49)
    cid = (is_pad * pad_idx + (1.0 - is_pad) * real_idx)                     # (B,49) float
    cid_int = cid.cast(dtypes.int).clip(0, A - 1)                            # (B,49) int
    # gather per-cell coord: per-batch one-hot @ the per-batch coord table.
    cid_oh = cid_int.one_hot(A).cast(dtypes.float)                          # (B,49,A)
    z_cage = cid_oh @ cage_coord_table                                      # (B,49,dim) ball/point
    cage_bias = _relation_bias_dispatch(z_cage, model.kenken_hyp_r_cage,
                                        model.kenken_hyp_alpha_cage,
                                        is_tangent=False)                    # (B,49,49)

    # --- assemble per-head bias: broadcast each relation across its head slots ---
    # row/col are constants -> broadcast to batch; cage is per-instance; global = 0.
    row_b = row_bias.reshape(1, 1, N_CELLS, N_CELLS).expand(B, n_row, N_CELLS, N_CELLS)
    col_b = col_bias.reshape(1, 1, N_CELLS, N_CELLS).expand(B, n_col, N_CELLS, N_CELLS)
    cage_b = cage_bias.reshape(B, 1, N_CELLS, N_CELLS).expand(B, n_cage, N_CELLS, N_CELLS)
    glob_b = Tensor.zeros((B, n_global, N_CELLS, N_CELLS), dtype=dtypes.float)
    bias = Tensor.cat(row_b, col_b, cage_b, glob_b, dim=1)                   # (B,n_heads,49,49)

    # --- validity masking (mirror build_kenken_attn_bias EXACTLY) ---
    # additive form: blocked KEYS get -BLOCK added; padding QUERY rows forced self-only.
    eye = Tensor(np.eye(N_CELLS, dtype=np.float32), dtype=dtypes.float
                 ).reshape(1, 1, N_CELLS, N_CELLS)
    valid_key = cell_valid.reshape(B, 1, 1, N_CELLS)                         # (B,1,1,49)
    # invalid keys: force their bias to -BLOCK (block) regardless of geometry.
    key_block = (1.0 - valid_key) * (-BLOCK)                                 # (B,1,1,49)
    bias = bias + key_block                                                  # broadcast over heads/queries
    bias = bias.maximum(Tensor(-BLOCK, dtype=dtypes.float))                  # keep finite floor
    # padding query rows: replace the whole row with self-only (0 on diag, -BLOCK off).
    self_only_bias = (1.0 - eye) * (-BLOCK)                                  # (1,1,49,49) self=0
    valid_q = cell_valid.reshape(B, 1, N_CELLS, 1)                           # (B,1,49,1)
    bias = bias * valid_q + self_only_bias * (1.0 - valid_q)                 # (B,n_heads,49,49)
    return bias.contiguous()


def precompute_kenken_hyp_row_col_bias(model: Any) -> Tensor:
    """Precompute the CONSTANT row/col hyperbolic biases -> (2, 49, 49) (spec §3/§4).

    Call once OUTSIDE the JIT-traced step; pass the result to
    build_kenken_hyperbolic_attn_bias so the row/col fields never re-enter the graph.
    Arm-aware: KENKEN_HYP_EUCLID swaps in the Euclidean bias (no exp_0, ||u-v||).
    """
    row_bias = _relation_bias_dispatch(model.kenken_hyp_v_row,
                                       model.kenken_hyp_r_row,
                                       model.kenken_hyp_alpha_row)
    col_bias = _relation_bias_dispatch(model.kenken_hyp_v_col,
                                       model.kenken_hyp_r_col,
                                       model.kenken_hyp_alpha_col)
    return Tensor.stack(row_bias, col_bias, dim=0).contiguous()              # (2,49,49)


# ---- iterative prefill loop --------------------------------------------------

def kenken_breathing_forward(model: Any, batch: Any, K: int, stoch_keep=None):
    """Run K breaths of constraint propagation on a KenKen batch.

    batch: a KenKenBatch (or any object with the same attributes) carrying the
           per-instance tensors (input_cells, cage_mask, cell_valid,
           cell_cage_id, cage_op, cage_target, cage_size, value_domain_mask).

    stoch_keep: OPTIONAL (K,) float Tensor of per-breath keep-SCALES for
           stochastic depth (run-2 reg stack). When supplied (TRAIN only), breath
           k's residual update is scaled by stoch_keep[k] — caller passes
           keep/(1-p) when kept and 0.0 when dropped (ResNet unbiased estimator),
           so E[update] is unchanged. None (the default; ALL eval callers) =>
           every breath fully applied (deterministic forward, byte-identical to
           the pre-run-2 behavior).

    Per-breath structure (mirror sudoku):
      1. Add per-breath additive embedding.
      2. Add the verification inlet (RMSNorm'd, LIVE at init — not gated). The
         inlet is added EVERY breath so its gradient stays live across the loop.
      3. Run 4 shared transformer layers with the per-batch structured mask.
      4. Learnable per-breath delta gate (convex residual blend).
      5. Readout: cell logits (value-domain masked) + calibration confidence.

    Returns:
      cell_logits_history: list of K (B, 49, N_MAX) float Tensors (value-domain
                           masked — illegal values forced to -1e4).
      calib_history:       list of K (B,) float Tensors (sigmoid'd).
    """
    assert hasattr(model, "kenken_state_embed"), \
        "model has no kenken params; was KENKEN_TASK set before attach?"

    state_embed = model.kenken_state_embed         # (8, H)
    position_embed = model.kenken_position_embed   # (49, H)
    breath_embed = model.kenken_breath_embed       # (K_max, H)
    delta_gate = model.kenken_delta_gate           # (K_max,)
    value_codebook = model.kenken_value_codebook   # (N_MAX, H)
    calib_head_w = model.kenken_calib_head_w       # (H, 1)
    calib_head_b = model.kenken_calib_head_b       # (1,)

    input_cells = batch.input_cells                # (B, 49) int
    cage_mask = batch.cage_mask                    # (B, 49, 49)
    cell_valid = batch.cell_valid                  # (B, 49)
    value_domain_mask = batch.value_domain_mask    # (B, 49, N_MAX)

    # Per-batch attention bias (row/col fixed + cage per-instance + validity).
    # ENV-GATE (spec §3): KENKEN_HYP_MASK swaps in the hyperbolic generator. OFF
    # (default) => build_kenken_attn_bias UNCHANGED => byte-identical to v98. ON =>
    # the frozen calibrated Poincare-coordinate generator (reproduces the hard mask).
    if KENKEN_HYP_MASK:
        # STAGE-1 RELAXATION (spec §8): when KENKEN_HYP_RELAX is ON the row/col coords
        # are TRAINABLE, so the row/col bias MUST be computed INSIDE the differentiable
        # graph (pass row_col_bias_cache=None) — caching it as a constant would CUT the
        # gradient to v_row/v_col (the cached tensor is a leaf, no grad path). When OFF
        # (frozen foothold) row/col is a constant -> precompute once per call (cheaper,
        # byte-identical to the foothold). The cage bias is per-instance/in-graph in
        # BOTH paths (it already enters the graph through cell_cage_id).
        if KENKEN_HYP_RELAX:
            attn_bias = build_kenken_hyperbolic_attn_bias(
                model, batch.cell_cage_id, cell_valid, row_col_bias_cache=None)
        else:
            rc_cache = precompute_kenken_hyp_row_col_bias(model)           # (2,49,49)
            attn_bias = build_kenken_hyperbolic_attn_bias(
                model, batch.cell_cage_id, cell_valid, row_col_bias_cache=rc_cache)
    else:
        attn_bias = build_kenken_attn_bias(model, cage_mask, cell_valid)   # (B,n_heads,49,49)

    # Verification inlet (per-cell, RMSNorm'd, LIVE). Built once; added each breath.
    inlet = build_verification_inlet(
        model, batch.cage_op, batch.cage_target, batch.cage_size, batch.cell_cage_id,
    )                                                                      # (B, 49, H)

    # Value-domain bias for readout: illegal values get -1e4 added to their logit.
    value_bias = (1.0 - value_domain_mask) * (-1e4)                        # (B, 49, N_MAX)

    # Learnable per-value logit bias (rare-value calibration). Zero-init => identity
    # at step 0. Added BEFORE the value-domain mask so illegal values still get -1e4
    # afterward (the +bias on a -1e4 illegal logit is negligible). Only present when
    # KENKEN_VALUE_BIAS is ON (attach_kenken_params attaches it under the same gate);
    # OFF => attribute absent => byte-identical to current v98 readout.
    value_logit_bias = getattr(model, "kenken_value_bias", None)          # (N_MAX,) or None
    if value_logit_bias is not None:
        value_logit_bias = value_logit_bias.cast(dtypes.float).reshape(1, 1, N_MAX)

    x = embed_kenken(input_cells, state_embed, position_embed)
    x = x.cast(dtypes.half) if x.dtype != dtypes.half else x

    layers = list(model.block.layers)
    assert len(layers) >= 4, f"expected >=4 transformer layers; got {len(layers)}"
    K_max = int(breath_embed.shape[0])
    assert K <= K_max, f"K={K} exceeds K_max={K_max} for kenken_breath_embed"

    from mycelium.breathing import _layernorm

    inlet_h = inlet.cast(x.dtype)
    cell_valid_col = cell_valid.reshape(int(cell_valid.shape[0]), N_CELLS, 1)

    # v109pi per-breath Q-rotation angles. theta_k = phase_scale * k * π / K_max so
    # k=0 -> 0 (identity, breath 0 unrotated) and the breaths span 0..~π. cos/sin are
    # python scalars (compile-time constants per breath), so they bake into the JIT
    # graph with NO new inputs. When KENKEN_PI_ROPE is OFF we pass (None, None) to
    # kenken_layer_forward, so the rotation block is never entered and the forward is
    # BYTE-IDENTICAL to the current v98 KenKen path.
    if KENKEN_PI_ROPE:
        _phases   = [KENKEN_PI_ROPE_PHASE_SCALE * kk * math.pi / float(K_max)
                     for kk in range(K_max)]
        breath_cos = [math.cos(p) for p in _phases]
        breath_sin = [math.sin(p) for p in _phases]
    else:
        breath_cos = None
        breath_sin = None

    cell_logits_history = []
    calib_history = []

    for k in range(K):
        be_k = breath_embed[k].reshape(1, 1, -1).cast(x.dtype)             # (1,1,H)
        x_in = x + be_k + inlet_h                                          # add LIVE inlet

        cos_k = breath_cos[k] if breath_cos is not None else None
        sin_k = breath_sin[k] if breath_sin is not None else None

        x_pre = x
        h = x_in
        for layer in layers[:4]:
            h = kenken_layer_forward(layer, h, attn_bias, cos_k, sin_k)

        gate_k = delta_gate[k].cast(h.dtype).reshape(1, 1, 1)
        delta = h - x_pre
        # Stochastic depth (run-2): scale this breath's residual update by the
        # per-breath keep-scale (keep/(1-p) or 0). None => full update (eval +
        # the pre-run-2 deterministic path). The scale multiplies the UPDATE only,
        # so a dropped breath passes its input through unchanged (x = x_pre).
        if stoch_keep is not None:
            keep_k = stoch_keep[k].cast(h.dtype).reshape(1, 1, 1)
            x = x_pre + (gate_k * keep_k) * delta
        else:
            x = x_pre + gate_k * delta

        # Readout: project each cell to an N_MAX-way logit; mask illegal values.
        x_ln = _layernorm(x, model.ln_f_g, model.ln_f_b, model.cfg.layer_norm_eps).cast(dtypes.float)
        cell_logits_k = x_ln @ value_codebook.T.cast(dtypes.float)         # (B, 49, N_MAX)
        if value_logit_bias is not None:
            cell_logits_k = cell_logits_k + value_logit_bias              # per-value calib (pre-mask)
        cell_logits_k = cell_logits_k + value_bias.cast(dtypes.float)      # value-domain mask
        cell_logits_history.append(cell_logits_k)

        # Calibration: mean-pool over VALID cells only.
        pool_num = (x_ln * cell_valid_col.cast(dtypes.float)).sum(axis=1)  # (B, H)
        pool_den = cell_valid_col.cast(dtypes.float).sum(axis=1) + 1e-6    # (B, 1)
        pool = pool_num / pool_den
        calib_logit = pool @ calib_head_w.cast(dtypes.float) + calib_head_b.cast(dtypes.float)
        calib_k = calib_logit.reshape(-1).sigmoid()
        calib_history.append(calib_k)

    return cell_logits_history, calib_history


# ---- convergence instrument (Property 2) -------------------------------------

def _jsd(p: np.ndarray, q: np.ndarray, axis: int = -1, eps: float = 1e-9) -> np.ndarray:
    """Jensen-Shannon divergence between two distributions (numpy; base-2)."""
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    m = 0.5 * (p + q)
    def _kl(a, b):
        return np.sum(a * (np.log2(a) - np.log2(b)), axis=axis)
    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def convergence_instrument(cell_logits_history, batch: Any,
                            threshold: float = KENKEN_CONVERGE_JSD,
                            frac_required: float = 0.95) -> list[dict]:
    """Compute the Property-2 convergence artifact per puzzle in the batch.

    TWO families of breath-count are computed (see the U-curve decision pinned in
    `memory/project_csp_target_survey_jun14.md`, section "THE U-CURVE IS
    ARCHITECTURE-GENERAL + MIN-BASED CONVERGENCE"):

    (A) JSD-FLOOR (the ORIGINAL, TAIL-CONTAMINATED secondary; UNCHANGED below):
        For each puzzle: belief_k = softmax(cell_logits_k). Per-cell JSD between
        consecutive breaths. CONVERGED at breath k = ALL valid cells'
        JSD(belief_k, belief_{k-1}) < threshold. breath_count = first such k
        (1-based; else K). settled = converged AND final-breath prediction ==
        gold; stuck = converged AND wrong. Under the #238 U-curve a puzzle SOLVED
        at the refinement minimum (k~3-4) then DRIFTING in the tail keeps a large
        consecutive-JSD, so this instrument labels it not_converged -> censors the
        settled set -> fakes a no-spread NULL. That is why it is now SECONDARY.

    (B) MIN-BASED (the PRIMARY, GOLD-FREE instrument; ADDED here):
        breath_count_min = argmin over k in [1, K-1] of the mean-over-valid-cells
        consecutive-breath JSD(belief_k, belief_{k-1}), reported 1-based as (k+1).
        This is the breath where the loop's OWN DYNAMICS settle. It is robust to
        the U because tail-drift makes consecutive-JSD RISE again, so the global
        argmin is the settle point (~k=3-4). GOLD-FREE by design: gold gates the
        settled SUBSET (status_min) but NEVER the breath-count itself (using gold
        to FIND the breath then correlating with depth would be circular leakage).
        correct_min = all valid cells argmax-correct AT the settle breath (belief
        at breath_count_min). status_min in {settled, stuck}: settled = correct_min,
        stuck otherwise. There is NO not_converged for the min instrument: an
        argmin always exists (the loop always has a least-moving breath), so every
        puzzle is either settled or stuck.

    COMPANIONS (reported, NOT the correlation axis):
      breath_count_min_ce = argmin_k CE-to-gold over the valid+supervised cells
        (where the answer was BEST). USES GOLD -> companion only, never correlated.
      breath_count_frac = first 1-based breath where >= frac_required of valid
        cells have per-cell JSD(belief_k, belief_{k-1}) < threshold (the pinned
        secondary convergence def; else K).

    Returns a list (len B) of dicts. EXISTING fields (do NOT remove/rename — the
    live trainer, MODE B, and self-tests depend on these):
      {breath_count, deduction_depth, converged, correct, status}
      status in {"settled", "stuck", "not_converged"}.
    ADDED fields (purely additive):
      {breath_count_min, correct_min, status_min,
       breath_count_min_ce, breath_count_frac}
      status_min in {"settled", "stuck"}  (no "not_converged" — argmin always exists).
    """
    K = len(cell_logits_history)
    B = int(cell_logits_history[0].shape[0])
    # beliefs[k]: (B, 49, N_MAX) numpy softmax.
    beliefs = []
    for logits in cell_logits_history:
        l = logits.realize().numpy().astype(np.float64)
        l = l - l.max(axis=-1, keepdims=True)
        e = np.exp(l)
        beliefs.append(e / (e.sum(axis=-1, keepdims=True) + 1e-12))

    cell_valid = batch.cell_valid.realize().numpy().astype(bool)           # (B, 49)
    gold = batch.gold.realize().numpy().astype(np.int32)                   # (B, 49) 1..7
    # Supervised cells = valid AND not a given cell (mirror the loss supervision):
    # given cells are already known, so CE-to-gold should not count them.
    given = batch.input_cells.realize().numpy().astype(np.int32) > 0       # (B, 49)
    final_pred = beliefs[-1].argmax(axis=-1).astype(np.int32) + 1          # (B, 49)
    # Per-breath argmax predictions (B, K, 49) +1 -> values 1..N_MAX.
    pred_by_k = np.stack([beliefs[k].argmax(axis=-1).astype(np.int32) + 1
                          for k in range(K)], axis=1)                      # (B, K, 49)

    out = []
    for b in range(B):
        valid = cell_valid[b]                                              # (49,)
        n_valid = int(valid.sum())
        gold_b = gold[b]                                                   # (49,)
        # CE-to-gold supervised cells = valid AND not given.
        sup = valid & (~given[b])                                         # (49,)
        n_sup = int(sup.sum())

        # ---- (A) JSD-FLOOR breath_count (ORIGINAL — unchanged) ----
        breath_count = K
        converged = False
        for k in range(1, K):
            jsd_k = _jsd(beliefs[k][b], beliefs[k - 1][b], axis=-1)        # (49,)
            max_jsd_valid = float(jsd_k[valid].max()) if n_valid > 0 else 0.0
            if max_jsd_valid < threshold:
                breath_count = k + 1   # 1-based: converged after breath index k
                converged = True
                break
        # correctness over valid cells (FINAL-breath prediction)
        correct = bool(np.all(final_pred[b][valid] == gold_b[valid])) if n_valid > 0 else False
        if not converged:
            status = "not_converged"
        elif correct:
            status = "settled"
        else:
            status = "stuck"

        # ---- (B) MIN-BASED breath_count_min (PRIMARY, GOLD-FREE) ----
        # mean-over-valid consecutive-breath JSD per transition k in [1, K-1].
        # argmin -> the settle breath. Always defined (K>=2).
        if K >= 2:
            mean_jsd = np.empty(K - 1, dtype=np.float64)
            for k in range(1, K):
                jsd_k = _jsd(beliefs[k][b], beliefs[k - 1][b], axis=-1)    # (49,)
                mean_jsd[k - 1] = float(jsd_k[valid].mean()) if n_valid > 0 else 0.0
            k_min = int(np.argmin(mean_jsd)) + 1   # transition index k in [1, K-1]
            breath_count_min = k_min + 1           # 1-based, consistent with breath_count
        else:
            k_min = 0
            breath_count_min = 1
        # correctness AT the settle breath (belief at breath_count_min = pred_by_k[k_min]).
        correct_min = bool(np.all(pred_by_k[b, k_min][valid] == gold_b[valid])) \
            if n_valid > 0 else False
        # status_min: NO not_converged (argmin always exists — see docstring).
        status_min = "settled" if correct_min else "stuck"

        # ---- COMPANION: breath_count_min_ce (argmin CE-to-gold; USES GOLD) ----
        if n_sup > 0:
            gold_idx = (gold_b - 1).clip(0, N_MAX - 1)                     # (49,) 0-based
            ce_per_k = np.empty(K, dtype=np.float64)
            for k in range(K):
                p = beliefs[k][b]                                          # (49, N_MAX)
                p_gold = p[np.arange(N_CELLS), gold_idx]                   # (49,)
                ce = -np.log(np.clip(p_gold, 1e-12, 1.0))                  # (49,)
                ce_per_k[k] = float(ce[sup].mean())
            breath_count_min_ce = int(np.argmin(ce_per_k)) + 1            # 1-based
        else:
            breath_count_min_ce = K

        # ---- COMPANION: breath_count_frac (>= frac_required valid cells JSD<thr) ----
        breath_count_frac = K
        for k in range(1, K):
            jsd_k = _jsd(beliefs[k][b], beliefs[k - 1][b], axis=-1)        # (49,)
            if n_valid > 0:
                below = jsd_k[valid] < threshold
                if float(below.mean()) >= frac_required:
                    breath_count_frac = k + 1
                    break
            else:
                breath_count_frac = k + 1
                break

        out.append({
            # EXISTING (do not change): JSD-floor instrument.
            "breath_count": breath_count,
            "deduction_depth": int(batch.deduction_depth[b]),
            "converged": converged,
            "correct": correct,
            "status": status,
            # ADDED: min-based PRIMARY instrument (gold-free breath-count).
            "breath_count_min": breath_count_min,
            "correct_min": correct_min,
            "status_min": status_min,
            # ADDED companions (NOT the correlation axis).
            "breath_count_min_ce": breath_count_min_ce,
            "breath_count_frac": breath_count_frac,
        })
    return out


# ---- losses ------------------------------------------------------------------

def kenken_constraint_energy(probs: Tensor, batch: Any) -> Tensor:
    """Soft constraint-violation diagnostic energy (B,).

    Two terms (mirror sudoku's row/col, plus a cage-arithmetic surrogate):
      - row/col AllDiff: for each row (col) and each value, the summed soft mass
        across the row's valid cells should be ≤ 1 (a value appears at most once
        in a line). Squared excess-over-1 is the penalty (only valid cells, only
        legal values).
      - cage soft-AllDiff: cells in a cage that share a row or col shouldn't both
        take the same value — a cheap surrogate for cage feasibility (we don't
        re-derive arithmetic here; this is a diagnostic only).

    probs: (B, 49, N_MAX) softmax over values (already value-domain-masked).
    """
    B = int(probs.shape[0])
    cell_valid = batch.cell_valid.reshape(B, N_CELLS, 1)                   # (B,49,1)
    pv = probs * cell_valid.cast(probs.dtype)                              # zero pad cells
    grid = pv.reshape(B, N_MAX, N_MAX, N_MAX)                              # (B, row, col, val)

    # Row AllDiff: sum over cols per (row, val); penalize mass > 1.
    row_sums = grid.sum(axis=2)                                            # (B, 7 rows, 7 vals)
    row_excess = (row_sums - 1.0).relu()                                   # only > 1 penalized
    row_violation = (row_excess ** 2).sum(axis=(1, 2))                     # (B,)

    col_sums = grid.sum(axis=1)                                            # (B, 7 cols, 7 vals)
    col_excess = (col_sums - 1.0).relu()
    col_violation = (col_excess ** 2).sum(axis=(1, 2))                     # (B,)

    return row_violation + col_violation                                  # (B,)


def codebook_ortho_penalty(codebook: Tensor, eps: float = 1e-8) -> Tensor:
    """Mean-square off-diagonal of the ROW-NORMALIZED Gram matrix of `codebook`.

    codebook: (R, H) — the value-codebook rows (or state_embed value-rows).
    Returns a scalar Tensor = mean over i!=j of cos(row_i, row_j)^2.

    Pure tensor ops (JIT-safe: no dtypes.float32 literal baked as a graph const, no
    .numpy()/.realize() in the traced body). Row-normalize -> gram = G@G.T -> the
    off-diagonal entries are exactly the pairwise cosines (diagonal = 1). We zero the
    diagonal with a (R,R) eye built from float32 ones (Tensor.eye is a graph-const
    structural mask, NOT a numeric-literal cast) and average the squared off-diagonal
    over the R*(R-1) off-diagonal positions. MINIMIZING this ROTATES collinear rows
    apart — the geometry fix a scalar bias/reweight cannot perform.
    """
    cb = codebook.cast(dtypes.float)                                   # (R, H)
    R = int(cb.shape[0])
    norm = (cb.pow(2).sum(axis=-1, keepdim=True) + eps).sqrt()         # (R, 1)
    unit = cb / norm                                                   # (R, H) row-normalized
    gram = unit @ unit.T                                               # (R, R) cosines
    eye = Tensor.eye(R, dtype=dtypes.float)                            # (R, R) diagonal mask
    off = gram * (1.0 - eye)                                           # zero the diagonal
    n_off = float(R * (R - 1)) if R > 1 else 1.0                       # count of i!=j entries
    return (off * off).sum() / n_off                                  # mean-square off-diag


def kenken_loss(cell_logits_history, calib_history, batch: Any,
                constraint_weight: float = 0.3,
                calib_weight: float = 0.1,
                value_freq_weight: Tensor | None = None,
                value_reweight: float = KENKEN_VALUE_REWEIGHT,
                ortho_lambda: float = 0.0,
                ortho_codebooks: list[Tensor] | None = None) -> tuple[Tensor, dict[str, Tensor]]:
    """Per-breath weighted CE ladder (valid + unobserved cells, value-masked) +
    constraint energy + calibration. Mirror of sudoku_loss.

    The CE is masked to VALID cells AND to UNOBSERVED cells (given cells are not
    supervised — their value is already known). Mirrors sudoku supervising only
    the cells that need solving.

    VALUE REWEIGHT (rare-value undertraining fix): when value_reweight>0 AND a
    value_freq_weight (N_MAX,) table is supplied, each supervised cell's CE is
    scaled by w(gold_value) (inverse-frequency**pow), NORMALIZED to mean 1.0 over
    the supervised cells so the overall loss scale is unchanged. OFF (the default,
    value_reweight==0 OR table None) => all per-cell weights are 1.0 => the CE is
    byte-identical to the original ladder.

    CODEBOOK-ORTHO (MECHANISM 1): when ortho_lambda>0 AND ortho_codebooks is a non-
    empty list of (R,H) tensors, add ortho_lambda * sum_cb mean-square off-diagonal
    cosine of each codebook (codebook_ortho_penalty). OFF (the default, lambda==0 OR
    list None/empty) => no term added => byte-identical total.
    """
    K = len(cell_logits_history)
    B = int(cell_logits_history[0].shape[0])

    gold = batch.gold                                                      # (B, 49) 1..7
    gold_idx = (gold - 1).clip(0, N_MAX - 1)                               # (B, 49) 0..6
    cell_valid = batch.cell_valid                                         # (B, 49)
    # Unobserved = valid AND not a given cell. given cell ⇔ input_cells > 0.
    observed = (batch.input_cells > 0).cast(dtypes.float)                 # (B, 49)
    supervise = cell_valid * (1.0 - observed)                            # (B, 49)

    # Per-cell value reweight (mean-normalized over supervised cells). OFF => ones.
    if value_reweight > 0.0 and value_freq_weight is not None:
        gold_oh = gold_idx.one_hot(N_MAX).cast(dtypes.float)              # (B,49,N_MAX)
        vw = (gold_oh @ value_freq_weight.cast(dtypes.float))            # (B,49) raw w(gold)
        vw = vw * supervise                                              # zero non-supervised
        vw_mean = vw.sum() / (supervise.sum() + 1e-6)                    # mean over supervised
        cell_weight = vw / (vw_mean + 1e-12)                            # mean-1.0 reweight
    else:
        cell_weight = Tensor.ones((B, N_CELLS), dtype=dtypes.float)
    # Effective supervise mask carries the reweight (byte-identical at weight==1).
    supervise = supervise * cell_weight                                  # (B, 49)
    supervise_flat = supervise.reshape(B * N_CELLS)                       # (B*49,)
    sup_sum = supervise_flat.sum() + 1e-6
    gold_flat = gold_idx.reshape(B * N_CELLS)                             # (B*49,)

    cell_loss_sum = Tensor.zeros((), dtype=dtypes.float)
    weight_sum = 0.0
    for k, logits in enumerate(cell_logits_history):
        weight_k = 1.0 + float(k) / float(K - 1) if K > 1 else 1.0
        # Per-element CE (no reduction), then mask to supervised cells.
        ce_elems = logits.reshape(B * N_CELLS, N_MAX).sparse_categorical_crossentropy(
            gold_flat, reduction="none"
        )                                                                 # (B*49,)
        ce_k = (ce_elems * supervise_flat).sum() / sup_sum
        cell_loss_sum = cell_loss_sum + ce_k * weight_k
        weight_sum += weight_k
    cell_loss = cell_loss_sum / float(weight_sum)

    final_probs = cell_logits_history[-1].softmax(axis=-1)
    energy = kenken_constraint_energy(final_probs, batch).mean()

    # Calibration target from detached final-breath argmax-correctness (valid cells).
    final_argmax = (cell_logits_history[-1].argmax(axis=-1) + 1).detach()  # (B, 49)
    eq = (final_argmax == gold).cast(dtypes.float)                         # (B, 49)
    # correct over valid cells: a puzzle is correct iff all valid cells match.
    eq_valid = eq * cell_valid + (1.0 - cell_valid)                        # pad cells count as match
    correct = eq_valid.prod(axis=-1)                                       # (B,) 0/1
    calib_loss_sum = Tensor.zeros((), dtype=dtypes.float)
    for k, calib_k in enumerate(calib_history):
        progression = float(k) / float(K - 1) if K > 1 else 1.0
        target_k = 0.5 + (correct - 0.5) * progression
        calib_loss_sum = calib_loss_sum + ((calib_k - target_k) ** 2).mean()
    calib_loss = calib_loss_sum / float(K)

    total = cell_loss + constraint_weight * energy + calib_weight * calib_loss

    # ---- MECHANISM 1: codebook-orthogonality penalty (OFF => no term, byte-identical) ----
    parts = {"cell_ce": cell_loss, "energy": energy, "calib": calib_loss}
    if ortho_lambda > 0.0 and ortho_codebooks:
        ortho = Tensor.zeros((), dtype=dtypes.float)
        for cb in ortho_codebooks:
            ortho = ortho + codebook_ortho_penalty(cb)
        total = total + ortho_lambda * ortho
        parts["ortho"] = ortho
    return total, parts


def per_breath_ce(cell_logits_history, batch: Any) -> list[float]:
    """Raw per-breath CE (unweighted, supervised cells only) for logging."""
    B = int(cell_logits_history[0].shape[0])
    gold_flat = ((batch.gold - 1).clip(0, N_MAX - 1)).reshape(B * N_CELLS)
    observed = (batch.input_cells > 0).cast(dtypes.float)
    supervise_flat = (batch.cell_valid * (1.0 - observed)).reshape(B * N_CELLS)
    sup_sum = float((supervise_flat.sum() + 1e-6).realize().numpy())
    out = []
    for logits in cell_logits_history:
        ce_elems = logits.reshape(B * N_CELLS, N_MAX).sparse_categorical_crossentropy(
            gold_flat, reduction="none")
        ce = (ce_elems * supervise_flat).sum() / sup_sum
        out.append(float(ce.realize().numpy()))
    return out


# ---- accuracy ----------------------------------------------------------------

def kenken_accuracy(cell_logits_final: Tensor, batch: Any) -> tuple[float, float]:
    """Return (cell_accuracy, puzzle_accuracy) over VALID cells.

    cell_logits_final: (B, 49, N_MAX)
    """
    B = int(cell_logits_final.shape[0])
    gold = batch.gold
    cell_valid = batch.cell_valid
    pred = cell_logits_final.argmax(axis=-1) + 1                           # (B, 49)
    eq = (pred == gold).cast(dtypes.float) * cell_valid                    # (B, 49)
    n_valid = cell_valid.sum() + 1e-6
    cell_acc = eq.sum() / n_valid
    # puzzle correct: all valid cells match (pad cells forced to match)
    eq_p = (pred == gold).cast(dtypes.float) * cell_valid + (1.0 - cell_valid)
    puzzle_acc = eq_p.prod(axis=-1).mean()
    return float(cell_acc.realize().numpy()), float(puzzle_acc.realize().numpy())


# ---- model param attach ------------------------------------------------------

def attach_kenken_params(model: Any, hidden: int, n_heads: int,
                          k_max: int | None = None) -> None:
    """Allocate KenKen-specific params on `model` (a BreathingTransformer instance).

    Attributes added (all trainable params FP32; cast to fp16 on use):
      kenken_state_embed     (8, hidden)          — {0=unknown, 1..7 = given value}
      kenken_position_embed  (49, hidden)         — row/col structural + learned
      kenken_value_codebook  (N_MAX=7, hidden)    — number-level readout codebook
      kenken_calib_head_w    (hidden, 1)
      kenken_calib_head_b    (1,)
      kenken_breath_embed    (K_max, hidden)      — per-breath additive marker
      kenken_delta_gate      (K_max,)             — learnable per-breath delta
      kenken_fixed_mask      (n_heads, 49, 49)    — row/col/global FIXED mask (frozen)
      kenken_head_split      [n_row,n_col,n_cage,n_global]
      VERIFICATION INLET (the genuinely new params):
      kenken_op_embed        (N_OPS=5, d_op)
      kenken_target_embed    (TARGET_BUCKETS=32, d_tgt)
      kenken_size_embed      (MAX_CAGE_SIZE=8, d_size)
      kenken_inlet_w         (d_op+d_tgt+d_size, hidden)
      kenken_inlet_b         (hidden,)
    """
    if k_max is None:
        k_max = KENKEN_K_MAX

    # Position embedding (row/col structural + learned tail).
    model.kenken_position_embed = _build_kenken_position_features(hidden)

    # Value codebook — orthonormal rows scale 0.1 (mirror sudoku digit_codebook).
    rng_cb = np.random.RandomState(1403)
    raw_cb = rng_cb.randn(max(hidden, N_MAX), hidden).astype(np.float32)
    q_cb, _ = np.linalg.qr(raw_cb)
    cb_unit = q_cb[:N_MAX].astype(np.float32)                              # (7, hidden)
    model.kenken_value_codebook = Tensor(cb_unit * 0.1, dtype=dtypes.float).contiguous()

    # State embedding — 8 rows {0=unknown, 1..7 = given value}. Align rows 1..7
    # with the codebook (given cells start favoring their own value) — mirror sudoku.
    state = np.zeros((N_MAX + 1, hidden), dtype=np.float32)
    state[0] = np.random.RandomState(1402).randn(hidden).astype(np.float32) * 0.02
    state[1:N_MAX + 1] = cb_unit                                           # scale 1.0
    model.kenken_state_embed = Tensor(state, dtype=dtypes.float).contiguous()

    # Calibration head.
    cw = (np.random.RandomState(1404).randn(hidden, 1) * 0.02).astype(np.float32)
    model.kenken_calib_head_w = Tensor(cw, dtype=dtypes.float).contiguous()
    model.kenken_calib_head_b = Tensor.zeros((1,), dtype=dtypes.float).contiguous()

    # Per-breath additive embedding — orthonormal init (mirror sudoku).
    breath_scale = float(os.environ.get("KENKEN_BREATH_EMBED_SCALE", "0.5"))
    rng_be = np.random.RandomState(1405)
    raw = rng_be.randn(max(k_max, hidden), hidden).astype(np.float32)
    q, _ = np.linalg.qr(raw)
    be = q[:k_max].astype(np.float32) * breath_scale
    model.kenken_breath_embed = Tensor(be, dtype=dtypes.float).contiguous()

    # Per-breath delta gate (init 1.0 = full update; mirror sudoku).
    model.kenken_delta_gate = Tensor.ones((k_max,), dtype=dtypes.float).contiguous()

    # ---- LEARNABLE per-value logit bias (rare-value calibration). ONLY attached
    # when KENKEN_VALUE_BIAS is ON; zero-init => identity readout at step 0. When OFF
    # the attribute is absent, so the forward / parameters / state_dict skip it and
    # the path is byte-identical to current v98. ----
    if KENKEN_VALUE_BIAS:
        model.kenken_value_bias = Tensor.zeros((N_MAX,), dtype=dtypes.float).contiguous()

    # FIXED structured mask (row/col/global) + head split. CAGE slots placeholder.
    fixed_mask, head_split = _build_kenken_fixed_masks(n_heads)
    model.kenken_fixed_mask = fixed_mask.contiguous()                     # (n_heads,49,49) {0,1}
    model.kenken_head_split = head_split

    # ---- VERIFICATION INLET params (LIVE at init — NOT zero-init / NOT gated) ----
    # Dimensions: small per-feature embeddings, concatenated, projected to H.
    d_op = int(os.environ.get("KENKEN_INLET_D_OP", "32"))
    d_tgt = int(os.environ.get("KENKEN_INLET_D_TGT", "64"))
    d_size = int(os.environ.get("KENKEN_INLET_D_SIZE", "16"))
    rng_in = np.random.RandomState(1406)
    model.kenken_op_embed = Tensor(
        (rng_in.randn(N_OPS, d_op) * 0.1).astype(np.float32), dtype=dtypes.float).contiguous()
    model.kenken_target_embed = Tensor(
        (rng_in.randn(TARGET_BUCKETS, d_tgt) * 0.1).astype(np.float32), dtype=dtypes.float).contiguous()
    model.kenken_size_embed = Tensor(
        (rng_in.randn(MAX_CAGE_SIZE, d_size) * 0.1).astype(np.float32), dtype=dtypes.float).contiguous()
    d_total = d_op + d_tgt + d_size
    # Inlet projection — Xavier-ish scale so the pre-RMSNorm contribution is live
    # (NOT zero-init: the inlet must draw gradient at step 0). Post-RMSNorm bounds
    # the magnitude, so a live W here is safe.
    w_scale = (1.0 / math.sqrt(d_total))
    model.kenken_inlet_w = Tensor(
        (rng_in.randn(d_total, hidden) * w_scale).astype(np.float32), dtype=dtypes.float).contiguous()
    model.kenken_inlet_b = Tensor.zeros((hidden,), dtype=dtypes.float).contiguous()

    # ---- HYPERBOLIC MASK GENERATOR params (foothold; spec §0-§2 + §7) ----
    # ONLY attached when KENKEN_HYP_MASK is ON. The OFF path attaches NOTHING here, so
    # parameters / state_dict / forward are byte-identical to current v98. These params
    # are NEVER saved (kenken_state_dict omits them) and are ALWAYS re-derived from the
    # closed-form anchors at attach time -> a v98 ckpt (which lacks them) loads cleanly
    # under KENKEN_HYP_MASK=1 (load_ckpt tolerates missing keys; these are init here).
    if KENKEN_HYP_MASK:
        attach_kenken_hyperbolic_params(model, n_heads=n_heads)


def attach_kenken_hyperbolic_params(model: Any, n_heads: int,
                                    dim: int | None = None,
                                    rho: float | None = None) -> None:
    """Allocate the FROZEN per-relation Poincare coordinate fields + calibrated r/alpha.

    Per-relation fields (the triangle-inequality split, spec §0 — one PER RELATION):
      kenken_hyp_v_row  (49, dim)  tangent params s.t. exp_0(v_row) = row anchors
      kenken_hyp_v_col  (49, dim)  tangent params s.t. exp_0(v_col) = col anchors
      kenken_hyp_cage_anchors (A, dim)  fixed cage simplex anchor table (per-instance
                                        field built per batch by indexing this table)
    Closed-form per-relation calibration (spec §2; FROZEN here):
      kenken_hyp_r_{row,col,cage}     = d_out / 2
      kenken_hyp_alpha_{row,col,cage} = 2 * BLOCK / d_out
    so within-group d_in~=0 -> bias ~0 and between-group d_out -> bias ~ -BLOCK.

    TANGENT-SPACE parameterization (spec §7): the learnable part is stored as Euclidean
    tangent vectors v_*; z = exp_0(v) when computing d_hyp. Init v_* so exp_0(v_*) == the
    closed-form anchors. FROZEN for the foothold (NOT in kenken_parameters / not trained).
    """
    if dim is None:
        dim = KENKEN_HYP_DIM
    if rho is None:
        rho = KENKEN_HYP_RHO
    n_row, n_col, n_cage, n_global = _head_split_for(n_heads)

    # Per-cell row / col group index on the N_max grid (7 rows, 7 cols).
    rows_idx = np.array([i // N_MAX for i in range(N_CELLS)], dtype=np.int64)  # 0..6
    cols_idx = np.array([i % N_MAX for i in range(N_CELLS)], dtype=np.int64)   # 0..6
    G_line = N_MAX                                                            # 7 groups

    # Row/col anchors: G=7 simplex on the shell; assign each cell to its group's anchor.
    row_mu = _poincare_anchors(G_line, dim, rho)                              # (7, dim)
    col_mu = _poincare_anchors(G_line, dim, rho)                              # (7, dim)
    z_row_np = row_mu[rows_idx]                                              # (49, dim)
    z_col_np = col_mu[cols_idx]                                              # (49, dim)
    # Tangent init s.t. exp_0(v) == anchors (frozen foothold — exact replication, jitter=0).
    v_row_np = _tangent_for_anchors(z_row_np).astype(np.float32)             # (49, dim)
    v_col_np = _tangent_for_anchors(z_col_np).astype(np.float32)             # (49, dim)
    # STAGE-1 RELAXATION (spec §1 jitter / §8.6 Stage 1): row & col init to the SAME
    # closed-form anchor table -> an init-degeneracy (flagged in the foothold review).
    # When relaxing, add eps*randn to the TANGENTS to break the symmetry so the two
    # relations can specialize. Cage anchors keep their distinct-simplex scheme below.
    # ONLY when KENKEN_HYP_RELAX is ON; the frozen path stays the exact anchor (eps=0),
    # so KENKEN_HYP_RELAX=0 is byte-identical to the foothold. Deterministic seed so a
    # given run reproduces; the jitter is tiny (1e-3) so the t=0 mask still matches <1e-3.
    if KENKEN_HYP_RELAX and KENKEN_HYP_JITTER > 0.0:
        rng_j = np.random.RandomState(20260616)
        v_row_np = (v_row_np + KENKEN_HYP_JITTER
                    * rng_j.randn(*v_row_np.shape).astype(np.float32))
        v_col_np = (v_col_np + KENKEN_HYP_JITTER
                    * rng_j.randn(*v_col_np.shape).astype(np.float32))
    model.kenken_hyp_v_row = Tensor(v_row_np, dtype=dtypes.float).contiguous()
    model.kenken_hyp_v_col = Tensor(v_col_np, dtype=dtypes.float).contiguous()

    # Cage anchor table. A must hold the max #cages PLUS one distinct sentinel anchor
    # per cell position for padding cells (cell_cage_id=-1 -> a unique far anchor). The
    # generator sends padding cell i to anchor index (A - N_CELLS + i). So we need
    # A = max_real_cages + N_CELLS spare slots; size it generously from KENKEN_N_CAGES
    # (default 41 = corpus max 40 +1 headroom). All A anchors are simplex vertices ->
    # every pair (incl real-vs-pad and pad-vs-pad) is at the uniform d_out -> blocked.
    max_cages = int(os.environ.get("KENKEN_N_CAGES", "41"))
    A = max_cages + N_CELLS                                                  # real + pad sentinels
    # An exact simplex needs dim >= A-1. If A-1 > dim we fall back to a spherical code
    # (normalized gaussian rows) whose MIN between-distance we calibrate against (§3).
    cage_anchors_np = _poincare_anchors(A, dim, rho) if (A - 1) <= dim \
        else _spherical_anchors(A, dim, rho)
    model.kenken_hyp_cage_anchors = Tensor(
        cage_anchors_np.astype(np.float32), dtype=dtypes.float).contiguous()

    # ---- closed-form calibration r, alpha PER RELATION (spec §2/§3) ----
    # row/col: exact simplex, G=7 -> uniform d_out. r = d_out/2 (UNCHANGED in both modes).
    # alpha:
    #   FROZEN foothold  -> margin * 2*BLOCK/d_out  (BLOCK-saturated, exact {0,-1e4} mask)
    #   RELAX (grad flow)-> margin * RELAX_BLOCK_ARG / r  so the between-group softplus arg
    #                       = margin*RELAX_BLOCK_ARG (responsive, NOT saturated -> coord
    #                       gradient flows; mask leak ~ exp(-arg) << 1e-3). See the
    #                       KENKEN_HYP_RELAX_BLOCK_ARG note above for why margin alone
    #                       can't unsaturate the BLOCK-based alpha.
    #
    # EUCLIDEAN CONTROL ARM (spec §8.1): r/alpha are recalibrated to the EUCLIDEAN anchor
    # distances (the SAME zero-init discipline) so the Euclidean arm ALSO reproduces the
    # hard mask at t=0. The only differences vs the hyperbolic arm are the d_out source
    # (Euclidean vs Poincare) and the missing exp_0 — capacity is otherwise identical.
    margin = KENKEN_HYP_ALPHA_MARGIN
    def _alpha_for(d_out: float, r: float) -> float:
        if KENKEN_HYP_RELAX:
            return margin * KENKEN_HYP_RELAX_BLOCK_ARG / max(r, 1e-9)
        return margin * 2.0 * KENKEN_HYP_BLOCK / d_out
    # row/col d_out: Euclidean uses the TANGENT-anchor separation (the row/col coord is the
    # tangent v used directly, NO exp_0), hyperbolic uses the Poincare simplex separation.
    if KENKEN_HYP_EUCLID:
        d_out_line = _euclid_d_out_line(rho, G_line)
    else:
        d_out_line = _hyp_d_out(rho, G_line)
    r_line = d_out_line / 2.0
    model.kenken_hyp_r_row = r_line
    model.kenken_hyp_alpha_row = _alpha_for(d_out_line, r_line)
    model.kenken_hyp_r_col = r_line
    model.kenken_hyp_alpha_col = _alpha_for(d_out_line, r_line)
    # cage: use the ACTUAL MIN between-anchor distance over the table (spec §3 — anchors
    # may not be a perfect simplex when falling back to a spherical code). For the exact
    # simplex this equals the closed-form d_out; computing it from the table is robust to
    # either construction. The cage anchors are BALL POINTS in both arms, so the Euclidean
    # min-distance is just the L2 min between those same ball points.
    if KENKEN_HYP_EUCLID:
        d_out_cage = _min_between_anchor_distance_euclid(cage_anchors_np)
    else:
        d_out_cage = _min_between_anchor_distance(cage_anchors_np)
    r_cage = d_out_cage / 2.0
    model.kenken_hyp_r_cage = r_cage
    model.kenken_hyp_alpha_cage = _alpha_for(d_out_cage, r_cage)

    # ---- STAGE-2 DeepSets g_phi ENCODER params (spec §8.1/§8.6). ONLY attached when
    # KENKEN_HYP_GPHI is ON; OFF => no g_phi params, the cage path is the Stage-1 slot-
    # anchor field byte-identical. rho's OUTPUT layer is ZERO-INIT => g_phi==0 at t=0. ----
    if KENKEN_HYP_GPHI:
        attach_kenken_gphi_params(model, dim=dim)


def attach_kenken_gphi_params(model: Any, dim: int,
                              d_pos: int | None = None,
                              width: int | None = None,
                              n_layers: int | None = None) -> None:
    """Allocate the STAGE-2 DeepSets STRUCTURE ENCODER g_phi params (spec §8.1/§8.6).

    cage_coord = anchor[id] + g_phi(cage), with
      g_phi(cage) = rho( MEAN_{cell in cage} phi(pos_emb[cell])  concat  [|cage|/N_CELLS] ).

    Params (ALL trainable in Stage-2 relax; NEVER in a v98 ckpt -> init fresh, tolerated):
      kenken_gphi_pos_emb   (N_CELLS, d_pos)  — ENCODER-OWNED learned position embedding
                                                (NOT the frozen backbone's pos features).
      kenken_gphi_phi_w/b   list of (in,out)/(out,) — phi MLP (d_pos -> width, n_layers).
      kenken_gphi_rho_w/b   list of (in,out)/(out,) — rho MLP (width+1 -> dim, n_layers);
                                                the OUTPUT layer is ZERO-INIT so g_phi==0
                                                at t=0 -> cage_coord==anchor[id] (the
                                                BIT-EXACT validated foothold mask).

    `dim` MUST equal the cage-anchor dim (rho's output width) so the correction adds to
    anchor[id] elementwise. MASK-ONLY inputs (cell positions + cage size) — never op/target.
    """
    if d_pos is None:
        d_pos = KENKEN_HYP_GPHI_DPOS
    if width is None:
        width = KENKEN_HYP_GPHI_WIDTH
    if n_layers is None:
        n_layers = max(1, KENKEN_HYP_GPHI_LAYERS)
    rng = np.random.RandomState(20260616)

    # ENCODER-OWNED position embedding — small randn (its own learned features). NOT the
    # backbone's _build_kenken_position_features (never entangle the frozen backbone).
    model.kenken_gphi_pos_emb = Tensor(
        (rng.randn(N_CELLS, d_pos) * 0.1).astype(np.float32), dtype=dtypes.float).contiguous()

    def _mlp(dims: list[int], zero_last: bool) -> tuple[list, list]:
        """Build (W_list, b_list) for an MLP with the given layer dims. He-ish scale on the
        hidden layers; the LAST layer is zero-init when zero_last (rho's g_phi==0 at t=0)."""
        Ws, bs = [], []
        L = len(dims) - 1
        for i in range(L):
            din, dout = dims[i], dims[i + 1]
            if zero_last and i == L - 1:
                W = np.zeros((din, dout), dtype=np.float32)
            else:
                W = (rng.randn(din, dout) * (1.0 / math.sqrt(din))).astype(np.float32)
            Ws.append(Tensor(W, dtype=dtypes.float).contiguous())
            bs.append(Tensor.zeros((dout,), dtype=dtypes.float).contiguous())
        return Ws, bs

    # phi: d_pos -> width (n_layers). NOT zero-init (it must produce live features).
    phi_dims = [d_pos] + [width] * n_layers
    model.kenken_gphi_phi_w, model.kenken_gphi_phi_b = _mlp(phi_dims, zero_last=False)
    # rho: (width + 1 size-feature) -> dim (n_layers). OUTPUT layer ZERO-INIT -> g_phi==0
    # at t=0 (anchor discipline: t=0 bias == the BIT-EXACT foothold mask).
    rho_dims = [width + 1] + [width] * (n_layers - 1) + [dim]
    model.kenken_gphi_rho_w, model.kenken_gphi_rho_b = _mlp(rho_dims, zero_last=True)


def _head_split_for(n_heads: int) -> list[int]:
    """The 5/5/5/1 head split (mirror _build_kenken_fixed_masks)."""
    n_row = max(1, n_heads * 5 // 16)
    n_col = max(1, n_heads * 5 // 16)
    n_cage = max(1, n_heads * 5 // 16)
    n_global = max(1, n_heads - n_row - n_col - n_cage)
    assigned = n_row + n_col + n_cage + n_global
    if assigned != n_heads:
        n_global += (n_heads - assigned)
    return [n_row, n_col, n_cage, n_global]


def _spherical_anchors(A: int, dim: int, rho: float) -> np.ndarray:
    """Fallback max-separated anchors when A-1 > dim (spec §1 spherical code).

    Deterministic normalized-gaussian rows (a spherical code); the caller calibrates
    alpha against the ACTUAL min-between distance (spec §3), so a slightly non-uniform
    code still reproduces the {0,-BLOCK} mask. Returns (A, dim) on the shell radius rho.
    """
    rng = np.random.RandomState(20260616)
    raw = rng.randn(A, dim).astype(np.float64)
    raw = raw / (np.linalg.norm(raw, axis=-1, keepdims=True) + 1e-12)
    return rho * raw


def _min_between_anchor_distance(anchors: np.ndarray) -> float:
    """Minimum pairwise Poincare distance between distinct anchors (spec §3).

    anchors: (A, dim) ball points. Used to calibrate the cage relation's alpha against
    the ACTUAL min between-cage separation (robust to simplex vs spherical code).
    """
    A = anchors.shape[0]
    sq = (anchors ** 2).sum(axis=-1)                                         # (A,)
    one_minus = np.clip(1.0 - sq, 1e-5, 1.0)
    gram = anchors @ anchors.T                                               # (A,A)
    diff_sq = np.clip(sq[:, None] + sq[None, :] - 2.0 * gram, 0.0, None)
    denom = one_minus[:, None] * one_minus[None, :]
    arg = np.clip(1.0 + 2.0 * diff_sq / denom, 1.0 + 1e-12, None)
    d = np.arccosh(arg)
    np.fill_diagonal(d, np.inf)
    return float(d.min())


def kenken_parameters(model: Any) -> list[Tensor]:
    """Trainable KenKen params (everything except the frozen fixed mask)."""
    params = [
        model.kenken_state_embed,
        model.kenken_position_embed,
        model.kenken_value_codebook,
        model.kenken_calib_head_w,
        model.kenken_calib_head_b,
        model.kenken_breath_embed,
        model.kenken_delta_gate,
        # verification inlet
        model.kenken_op_embed,
        model.kenken_target_embed,
        model.kenken_size_embed,
        model.kenken_inlet_w,
        model.kenken_inlet_b,
    ]
    # Learnable per-value logit bias (rare-value calibration), only when attached
    # (KENKEN_VALUE_BIAS ON). Guarded so the OFF path's param list is byte-identical.
    if getattr(model, "kenken_value_bias", None) is not None:
        params.append(model.kenken_value_bias)
    return params


def kenken_hyp_coord_parameters(model: Any) -> dict[str, Tensor]:
    """STAGE-1 RELAXATION coordinate params — EXACTLY the ~3 hyperbolic coord tensors
    (spec §8.2/§8.6 Stage 1). Returned as a NAMED dict so the trainer can build the
    coord-only optimizer AND log a per-relation (row / col / cage) grad norm.

    These are the ONLY tensors trained in Stage 1 (the v98 backbone is frozen):
      kenken_hyp_v_row        (49, dim)  ROW tangent coords  -> z_row = exp_0(v_row)
      kenken_hyp_v_col        (49, dim)  COL tangent coords  -> z_col = exp_0(v_col)
      kenken_hyp_cage_anchors (A, dim)   CAGE simplex anchor table (per-cell field is
                                         built per batch by indexing this table)
    Only valid after attach_kenken_hyperbolic_params (i.e. KENKEN_HYP_MASK=1).
    """
    assert hasattr(model, "kenken_hyp_v_row"), \
        "hyperbolic coords not attached; was KENKEN_HYP_MASK=1 before attach?"
    return {
        "row": model.kenken_hyp_v_row,
        "col": model.kenken_hyp_v_col,
        "cage": model.kenken_hyp_cage_anchors,
    }


def kenken_hyp_gphi_parameters(model: Any) -> list[Tensor]:
    """STAGE-2 DeepSets g_phi ENCODER params, as a FLAT list (spec §8.1/§8.6 Stage 2).

    Returns EXACTLY the g_phi tensors that JOIN the coord-only param group in Stage-2:
      kenken_gphi_pos_emb  (N_CELLS, d_pos)
      kenken_gphi_phi_w[i] / kenken_gphi_phi_b[i]   (phi MLP)
      kenken_gphi_rho_w[i] / kenken_gphi_rho_b[i]   (rho MLP; output layer zero-init)
    Empty list when KENKEN_HYP_GPHI is OFF (no g_phi params attached) -> the trainer's
    Stage-1 path is unaffected. Order is deterministic (pos_emb, phi W/b, rho W/b).
    """
    if not hasattr(model, "kenken_gphi_pos_emb"):
        return []
    params: list[Tensor] = [model.kenken_gphi_pos_emb]
    for W, b in zip(model.kenken_gphi_phi_w, model.kenken_gphi_phi_b):
        params.append(W); params.append(b)
    for W, b in zip(model.kenken_gphi_rho_w, model.kenken_gphi_rho_b):
        params.append(W); params.append(b)
    return params


def clamp_hyp_tangent_norms(model: Any, max_znorm: float = KENKEN_HYP_MAX_ZNORM) -> None:
    """Rim guard (spec §7): clamp each coord row's tangent norm |v| so |z|=tanh(|v|)
    stays <= max_znorm (keeps 1/(1-|z|^2) bounded in the d_hyp backward). Call AFTER
    each optimizer step. Per-ROW scaling (each cell/anchor independently): rows with
    |v| <= atanh(max_znorm) are untouched; longer rows are radially shrunk to the cap.

    In-place via .assign (mirrors the optimizer's own param updates); JIT/AM-safe pure
    tensor ops (no float32 literal baked, no isnan loop). Only call in Stage-1 relax.
    """
    max_vnorm = float(math.atanh(min(max(max_znorm, 0.0), 1.0 - 1e-7)))
    for t in (model.kenken_hyp_v_row, model.kenken_hyp_v_col,
              model.kenken_hyp_cage_anchors):
        v = t.cast(dtypes.float)
        norm = (v.pow(2).sum(axis=-1, keepdim=True) + 1e-12).sqrt()          # (M,1)
        # scale = min(1, max_vnorm / |v|): only shrink rows past the cap.
        scale = (Tensor(max_vnorm, dtype=dtypes.float) / norm).minimum(
            Tensor(1.0, dtype=dtypes.float))
        t.assign((v * scale).cast(t.dtype)).realize()


def kenken_inlet_parameters(model: Any) -> dict[str, Tensor]:
    """Just the verification-inlet params, grouped for the gradient-liveness gate."""
    return {
        "op_embed": model.kenken_op_embed,
        "target_embed": model.kenken_target_embed,
        "size_embed": model.kenken_size_embed,
        "inlet_w": model.kenken_inlet_w,
        "inlet_b": model.kenken_inlet_b,
    }


def kenken_state_dict(model: Any) -> dict[str, Tensor]:
    """State dict entries for KenKen params (excluding the static fixed mask)."""
    sd = {
        "kenken.state_embed": model.kenken_state_embed,
        "kenken.position_embed": model.kenken_position_embed,
        "kenken.value_codebook": model.kenken_value_codebook,
        "kenken.calib_head_w": model.kenken_calib_head_w,
        "kenken.calib_head_b": model.kenken_calib_head_b,
        "kenken.breath_embed": model.kenken_breath_embed,
        "kenken.delta_gate": model.kenken_delta_gate,
        "kenken.op_embed": model.kenken_op_embed,
        "kenken.target_embed": model.kenken_target_embed,
        "kenken.size_embed": model.kenken_size_embed,
        "kenken.inlet_w": model.kenken_inlet_w,
        "kenken.inlet_b": model.kenken_inlet_b,
    }
    # Learnable per-value logit bias — only saved when attached (guarded so the OFF
    # path's checkpoint key set is byte-identical to current v98).
    if getattr(model, "kenken_value_bias", None) is not None:
        sd["kenken.value_bias"] = model.kenken_value_bias
    return sd
