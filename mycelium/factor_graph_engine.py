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

from mycelium.factor_masks import build_factor_attn_bias
# Import the general helpers that kenken.py already exposes:
from mycelium.kenken import (
    kenken_layer_forward,    # (layer, x, attn_bias, cos, sin) -> x  — general S
    codebook_ortho_penalty,  # (codebook) -> scalar  — domain-free
)


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
    """
    s_max: int
    n_values: int
    n_factor_types: int
    n_heads: int
    k_max: int
    has_factor_inlet: bool = False


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
# Main breathing forward
# ---------------------------------------------------------------------------

def factor_breathing_forward(model: Any, batch: FactorGraphBatch,
                              spec: FactorGraphSpec, K: int,
                              stoch_keep: "Tensor | None" = None,
                              ) -> tuple[list[Tensor], list[Tensor]]:
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

    Returns
    -------
    value_logits_history : list of K Tensors, each (B, s_max, n_values) float.
    calib_history        : list of K Tensors, each (B,) float, sigmoid'd.
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
    x = embed_factor_cells(input_cells, state_embed, position_embed, N)
    x = x.cast(dtypes.half) if x.dtype != dtypes.half else x

    layers = list(model.block.layers)
    assert len(layers) >= 4, f"expected >=4 transformer layers; got {len(layers)}"
    K_max = int(breath_embed.shape[0])
    assert K <= K_max, f"K={K} exceeds k_max={K_max}"

    from mycelium.breathing import _layernorm

    inlet_h = inlet.cast(x.dtype)
    cell_valid_col = cell_valid.reshape(B, S, 1)

    value_logits_history: list[Tensor] = []
    calib_history: list[Tensor] = []

    for k in range(K):
        be_k = breath_embed[k].reshape(1, 1, -1).cast(x.dtype)   # (1, 1, H)
        x_in = x + be_k + inlet_h                                 # add inlet EVERY breath (LIVE)

        x_pre = x
        h = x_in
        # (c) COUPLED: pass s_max-length tensor through kenken_layer_forward.
        # The only assertion inside kenken_layer_forward is `assert S==N_CELLS`
        # which fires when S != 49.  For the general case we call the function
        # directly — it works for any S as long as attn_bias has matching shape.
        # For the KenKen anchor (S=49) it is byte-identical to the original call.
        for layer in layers[:4]:
            h = kenken_layer_forward(layer, h, attn_bias)          # no Q-rotation

        gate_k = delta_gate[k].cast(h.dtype).reshape(1, 1, 1)
        delta = h - x_pre

        if stoch_keep is not None:
            keep_k = stoch_keep[k].cast(h.dtype).reshape(1, 1, 1)
            x = x_pre + (gate_k * keep_k) * delta
        else:
            x = x_pre + gate_k * delta

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


def factor_graph_parameters(model: Any) -> list[Tensor]:
    """Trainable factor-graph params (excludes backbone params)."""
    return [
        model.fg_state_embed,
        model.fg_position_embed,
        model.fg_value_codebook,
        model.fg_calib_head_w,
        model.fg_calib_head_b,
        model.fg_breath_embed,
        model.fg_delta_gate,
    ]


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
