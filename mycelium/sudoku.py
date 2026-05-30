"""v98 sudoku module — clean test of constraint propagation paradigm.

Built around the breathing transformer's iterative-prefill insight: K passes of
the same 4 transformer layers refine an 81-cell representation. No NL parsing,
no AR decode, no waist bottleneck. Each cell projects its final hidden state
through a small codebook to a 9-way digit softmax.

Architecture decisions:
  - Reuses BreathingTransformer's L0-L3 weights (Pythia-init): minimal new infra.
  - Bypasses RoPE: cells aren't sequential. Position info comes from a learned
    (81, hidden) embedding that encodes row/col/box structure.
  - Bypasses causal mask: constraint propagation is bidirectional.
  - Structured per-head attention mask: 5 heads each for row/col/box, 1 global
    head. (16 heads total; the existing transformer has exactly 16.)
  - Per-breath supervision (the ladder): every breath's cell predictions get
    weighted CE against gold, plus a constraint-energy term on the final breath
    and a per-breath calibration target.

The forward path lives entirely in this module so existing v77+ code is untouched
(no risk of breaking the AR paths). It calls into BreathingTransformer's
SharedWeights and BreathingLayer attribute access (wq/wk/...) directly so the
Pythia init transfers automatically.

Env var gates (see scripts/v98_sudoku_smoke.sh):
  SUDOKU_TASK=1               — turn on the sudoku forward path (collect_params)
  SUDOKU_K_MAX=30             — number of iterative-prefill breaths
  SUDOKU_CONSTRAINT_WEIGHT=0.3 — weight on row/col/box energy term
  SUDOKU_CALIB_WEIGHT=0.1     — weight on per-breath calibration loss
  SUDOKU_DIFFICULTY_FILTER=easy — limit training data to one difficulty band
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


SUDOKU_TASK = int(os.environ.get("SUDOKU_TASK", "0")) > 0
SUDOKU_K_MAX = int(os.environ.get("SUDOKU_K_MAX", "30"))
SUDOKU_CONSTRAINT_WEIGHT = float(os.environ.get("SUDOKU_CONSTRAINT_WEIGHT", "0.3"))
SUDOKU_CALIB_WEIGHT = float(os.environ.get("SUDOKU_CALIB_WEIGHT", "0.1"))


# ---- structured attention masks (precomputed; deterministic for all 81 cells) ----

def _build_sudoku_attention_masks(n_heads: int = 16) -> Tensor:
    """Return a (n_heads, 81, 81) float Tensor of attention masks.

    Mask value 1.0 = allow attention, 0.0 = block (will translate to large
    negative additive bias before softmax). Each cell ALWAYS attends to itself.

    Default head assignment (n_heads=16):
      heads 0-4   : ROW   (cell attends to other cells in same row)
      heads 5-9   : COL   (same column)
      heads 10-14 : BOX   (same 3x3 box)
      head  15    : GLOBAL (full 81x81 — for cross-pattern constraints)
    """
    mask = np.zeros((n_heads, 81, 81), dtype=np.float32)

    # Precompute per-cell row, col, box.
    rows_idx = np.array([i // 9 for i in range(81)])
    cols_idx = np.array([i % 9 for i in range(81)])
    boxes_idx = np.array([(i // 9 // 3) * 3 + (i % 9 // 3) for i in range(81)])

    # Build same-row/col/box adjacency.
    same_row = (rows_idx[:, None] == rows_idx[None, :]).astype(np.float32)
    same_col = (cols_idx[:, None] == cols_idx[None, :]).astype(np.float32)
    same_box = (boxes_idx[:, None] == boxes_idx[None, :]).astype(np.float32)
    eye = np.eye(81, dtype=np.float32)
    full = np.ones((81, 81), dtype=np.float32)

    # Head assignment — keep 5/5/5/1 split for n_heads=16; degrade gracefully
    # for other counts (caller can override if needed).
    n_row = max(1, n_heads * 5 // 16)
    n_col = max(1, n_heads * 5 // 16)
    n_box = max(1, n_heads * 5 // 16)
    n_global = max(1, n_heads - n_row - n_col - n_box)
    # Pad / clip — guarantee row+col+box+global == n_heads.
    assigned = n_row + n_col + n_box + n_global
    if assigned != n_heads:
        n_global += (n_heads - assigned)

    h = 0
    for _ in range(n_row):
        mask[h] = np.maximum(same_row, eye)
        h += 1
    for _ in range(n_col):
        mask[h] = np.maximum(same_col, eye)
        h += 1
    for _ in range(n_box):
        mask[h] = np.maximum(same_box, eye)
        h += 1
    for _ in range(n_global):
        mask[h] = full
        h += 1

    return Tensor(mask, dtype=dtypes.float).contiguous()


def _build_sudoku_position_features(hidden: int) -> Tensor:
    """Initialize (81, hidden) position embedding with structural priors.

    Encodes row/col/box one-hots into a small fraction of the hidden dim so the
    model gets the structure for free. Remaining channels are randn 0.02 (Pythia
    init) so it can learn additional position-relative features.

    Layout (assuming hidden >= 32):
      [0..8]   = row one-hot
      [9..17]  = col one-hot
      [18..26] = box one-hot
      [27..]   = randn(0.02) (learned)
    """
    pos = np.zeros((81, hidden), dtype=np.float32)
    for i in range(81):
        r, c = i // 9, i % 9
        b = (r // 3) * 3 + (c // 3)
        if hidden >= 9:
            pos[i, r] = 1.0
        if hidden >= 18:
            pos[i, 9 + c] = 1.0
        if hidden >= 27:
            pos[i, 18 + b] = 1.0
    # Randn for the learned tail.
    rng = np.random.RandomState(98)
    if hidden > 27:
        pos[:, 27:] = rng.randn(81, hidden - 27).astype(np.float32) * 0.02
    return Tensor(pos, dtype=dtypes.float).contiguous()


# ---- per-cell embedding ------------------------------------------------------

def embed_sudoku(input_cells: Tensor, state_embed: Tensor, position_embed: Tensor) -> Tensor:
    """Convert (B, 81) int cell states into (B, 81, hidden) embeddings.

    input_cells: int Tensor with values in [0, 9], where 0 = unknown.
    state_embed: (10, hidden) — lookup table for the 10 cell states.
    position_embed: (81, hidden) — learned position embedding (row/col/box info).

    Returns state + position. (Sum, not concat — matches BPE-token-emb scale.)
    """
    B = int(input_cells.shape[0])
    # Tinygrad's embedding lookup: index along axis 0.
    # We use a one-hot multiply since some versions of tinygrad may not have an
    # `Embedding(indices)` callable equivalent. Build a one-hot mask of shape
    # (B, 81, 10) and matmul against state_embed (10, hidden).
    one_hot = input_cells.one_hot(10).cast(state_embed.dtype)   # (B, 81, 10)
    state = one_hot @ state_embed                                # (B, 81, hidden)
    pos = position_embed.reshape(1, 81, -1).cast(state.dtype).expand(B, 81, -1)
    return state + pos


# ---- one transformer-layer forward with structured attention (no RoPE, no causal) ----

def sudoku_layer_forward(layer: Any, x: Tensor, attn_bias: Tensor) -> Tensor:
    """Run one BreathingLayer's forward, but with sudoku-style attention.

    layer:     a mycelium.breathing.BreathingLayer (provides wq/wk/bq/bk/w_in/b_in
               + access to layer.shared for V/O/FFN_out/LN).
    x:         (B, 81, H) residual stream.
    attn_bias: (n_heads, 81, 81) additive bias added to QK^T scores. Use the
               structured mask converted to {0, -1e4} via (1-mask)*(-1e4).

    Differences from BreathingLayer._forward:
      - No RoPE (cells are positional via embed, not via sinusoidal RoPE)
      - No causal mask (constraint propagation is bidirectional)
      - Per-head mask via additive bias
      - Single fixed temperature (1.0 / sqrt(head_dim)) — no per-breath schedule

    Reuses Pythia-init weights (layer.wq, layer.wk, layer.shared.wv/wo/...).
    """
    cfg = layer.cfg
    B, S, H = x.shape
    assert int(S) == 81, f"sudoku layer expects 81 cells, got {S}"

    # Parallel-residual structure (Pythia/GPT-NeoX style)
    from mycelium.breathing import _layernorm
    attn_in = _layernorm(x, layer.shared.in_ln_g, layer.shared.in_ln_b, cfg.layer_norm_eps)
    mlp_in = _layernorm(x, layer.shared.post_ln_g, layer.shared.post_ln_b, cfg.layer_norm_eps)
    attn_in_dt = attn_in.cast(x.dtype) if attn_in.dtype != x.dtype else attn_in
    mlp_in_dt = mlp_in.cast(x.dtype) if mlp_in.dtype != x.dtype else mlp_in

    # Q, K, V projections — same as standard BreathingLayer
    q = (attn_in_dt @ layer.wq + layer.bq).reshape(B, S, cfg.n_heads, cfg.head_dim).transpose(1, 2)
    k = (attn_in_dt @ layer.wk + layer.bk).reshape(B, S, cfg.n_heads, cfg.head_dim).transpose(1, 2)
    v = (attn_in_dt @ layer.shared.wv + layer.shared.bv).reshape(B, S, cfg.n_heads, cfg.head_dim).transpose(1, 2)

    # NO RoPE — cells are addressed by learned position embedding, not by RoPE.
    scale = 1.0 / math.sqrt(cfg.head_dim)
    scores = q @ k.transpose(-2, -1) * scale                       # (B, n_heads, 81, 81)

    # Structured additive mask: bias of 0 where allowed, -1e4 where blocked.
    # attn_bias is (n_heads, 81, 81) → broadcast to (1, n_heads, 81, 81).
    scores = scores + attn_bias.cast(scores.dtype).reshape(1, cfg.n_heads, S, S)
    # Clip pre-softmax (AMD JIT safety).
    attn = scores.clip(-1e4, 1e4).softmax(-1)
    ctx = (attn @ v).transpose(1, 2).reshape(B, S, H)
    attn_out = ctx @ layer.shared.wo + layer.shared.bo

    ff = (mlp_in_dt @ layer.w_in + layer.b_in).gelu()
    ffn_out = ff @ layer.shared.w_out + layer.shared.b_out

    return x + attn_out + ffn_out


# ---- iterative prefill loop --------------------------------------------------

def sudoku_breathing_forward(model: Any, input_cells: Tensor, K: int):
    """Run K breaths of constraint propagation on (B, 81) input cells.

    Per-breath structure (each iteration):
      1. Add per-breath embedding (model knows "I'm on breath k")
      2. Run 4 transformer layers (shared weights — same task each breath)
      3. Apply learned per-breath delta gate (model can shrink late-breath updates)
      4. Readout: cell logits + calibration confidence

    Returns:
      cell_logits_history: list of K Tensors of shape (B, 81, 9). cell_logits[k]
        is the per-cell digit logit BEFORE softmax after breath k.
      calib_history: list of K Tensors of shape (B,). calib[k] is sigmoid'd
        confidence that the model's argmax at breath k matches gold.
    """
    assert hasattr(model, "sudoku_state_embed"), \
        "model has no sudoku params; was SUDOKU_TASK set before model init?"

    # State embedding for 10 cell states, position embedding for 81 cells.
    # Both built FP32 to give the optimizer a stable update path; cast on use.
    state_embed = model.sudoku_state_embed         # (10, H)
    position_embed = model.sudoku_position_embed   # (81, H)
    attn_bias = model.sudoku_attn_bias             # (n_heads, 81, 81) precomputed (1-mask)*(-1e4)
    breath_embed = model.sudoku_breath_embed       # (K_max, H) — per-breath additive bias
    delta_gate = model.sudoku_delta_gate           # (K_max,) — learnable per-breath delta scale

    # Initial embedding — cast to fp16 to match transformer layers' compute dtype.
    x = embed_sudoku(input_cells, state_embed, position_embed)
    x = x.cast(dtypes.half) if x.dtype != dtypes.half else x

    layers = list(model.block.layers)
    assert len(layers) >= 4, f"expected at least 4 transformer layers; got {len(layers)}"
    K_max = int(breath_embed.shape[0])
    assert K <= K_max, f"K={K} exceeds allocated K_max={K_max} for sudoku_breath_embed"

    # Final-LN params (reuse the transformer's ln_f for the cell-projection ln)
    from mycelium.breathing import _layernorm

    digit_codebook = model.sudoku_digit_codebook     # (9, H) — used as @ codebook.T
    calib_head_w = model.sudoku_calib_head_w         # (H, 1)
    calib_head_b = model.sudoku_calib_head_b         # (1,)

    cell_logits_history = []
    calib_history = []

    for k in range(K):
        # 1. Per-breath embedding: add at start of breath. Tells the model
        #    which iteration this is (a la diffusion timestep conditioning).
        be_k = breath_embed[k].reshape(1, 1, -1).cast(x.dtype)            # (1, 1, H)
        x_in = x + be_k

        # 2. Capture pre-layer state for gated delta.
        x_pre = x

        # 3. 4 transformer layers, shared across breaths (the K-iteration "same task" loop).
        h = x_in
        for layer in layers[:4]:
            h = sudoku_layer_forward(layer, h, attn_bias)

        # 4. Learnable per-breath delta gate. delta_gate[k] starts at 1.0 (identity);
        #    gradient lets the model learn to shrink late-breath updates so the
        #    iterative refinement actually CONVERGES rather than drifting.
        gate_k = delta_gate[k].cast(h.dtype).reshape(1, 1, 1)
        delta = h - x_pre
        x = x_pre + gate_k * delta

        # 5. Per-breath readout: project each cell to a 9-way logit via codebook.
        x_ln = _layernorm(x, model.ln_f_g, model.ln_f_b, model.cfg.layer_norm_eps).cast(dtypes.float)
        cell_logits_k = x_ln @ digit_codebook.T.cast(dtypes.float)
        cell_logits_history.append(cell_logits_k)

        # 6. Calibration: mean-pool the 81 cells, project to scalar, sigmoid.
        pool = x_ln.mean(axis=1)                                          # (B, H)
        calib_logit = pool @ calib_head_w.cast(dtypes.float) + calib_head_b.cast(dtypes.float)
        calib_k = calib_logit.reshape(-1).sigmoid()                       # (B,)
        calib_history.append(calib_k)

    return cell_logits_history, calib_history


# ---- losses ------------------------------------------------------------------

def sudoku_constraint_energy(probs: Tensor) -> Tensor:
    """Soft probability over digits at each cell → row/col/box constraint violation.

    probs: (B, 81, 9) softmax distribution per cell.
    Returns: (B,) scalar per-batch energy term. Mean reduces to scalar in caller.
    """
    B = int(probs.shape[0])
    probs_grid = probs.reshape(B, 9, 9, 9)   # (B, row, col, digit)

    # For each row, summing over cols should give exactly 1.0 per digit (each
    # digit appears once in a valid solution). Squared deviation from 1.0 is the
    # penalty.
    row_sums = probs_grid.sum(axis=2)                                     # (B, 9 rows, 9 digits)
    row_violation = ((row_sums - 1.0) ** 2).sum(axis=(1, 2))              # (B,)

    col_sums = probs_grid.sum(axis=1)                                     # (B, 9 cols, 9 digits)
    col_violation = ((col_sums - 1.0) ** 2).sum(axis=(1, 2))              # (B,)

    # Reshape (B, 9, 9, 9) into (B, 3, 3, 3, 3, 9) so axes (1,3) index the box
    # rows, axes (2,4) index the in-box positions.
    # probs_grid[b, r, c, d] = P(cell(r,c) = d)
    # box index: (r//3, c//3), in-box: (r%3, c%3)
    # After reshape: (B, br, ir, bc, ic, d) where br=r//3, ir=r%3, bc=c//3, ic=c%3
    # We want to sum over (ir, ic) — the in-box positions of a single box.
    probs_box = probs_grid.reshape(B, 3, 3, 3, 3, 9)
    # Permute to (B, br, bc, ir, ic, 9) → then sum axes ir, ic → (B, br, bc, 9)
    probs_box_perm = probs_box.permute(0, 1, 3, 2, 4, 5)
    box_sums = probs_box_perm.reshape(B, 9, 9, 9).sum(axis=2)             # (B, 9 boxes, 9 digits)
    box_violation = ((box_sums - 1.0) ** 2).sum(axis=(1, 2))              # (B,)

    return row_violation + col_violation + box_violation                  # (B,)


def sudoku_loss(cell_logits_history, calib_history, gold_solution: Tensor,
                constraint_weight: float = 0.3,
                calib_weight: float = 0.1) -> tuple[Tensor, dict[str, Tensor]]:
    """Per-breath supervision ladder + final-breath constraint energy + calibration.

    cell_logits_history: list of K (B, 81, 9) float Tensors.
    calib_history:        list of K (B,) float Tensors (sigmoid'd already).
    gold_solution:        (B, 81) int Tensor with values 1..9.

    Returns: (total_loss_scalar, {'cell_ce', 'energy', 'calib'} component dict).

    Late-breath weighting (1.0 at B0 → 2.0 at B_{K-1}) so the model is pushed
    toward refinement; combined with per-breath CE, this produces the visible
    ladder that diagnoses whether the iterative prefill is doing work.
    """
    K = len(cell_logits_history)
    B = int(cell_logits_history[0].shape[0])
    # Convert gold from 1..9 to 0..8 indices (sparse_categorical_crossentropy targets).
    # Use sub here to avoid creating a new int tensor on the wrong device.
    gold_idx = gold_solution - 1                                           # (B, 81)
    gold_flat = gold_idx.reshape(B * 81)                                   # (B*81,)

    # Per-breath CE ladder, weighted by progress through K.
    cell_loss_sum = Tensor.zeros((), dtype=dtypes.float)
    weight_sum = 0.0
    for k, logits in enumerate(cell_logits_history):
        # weight_k: linear ramp 1.0 → 2.0 over K breaths (so later breaths matter more)
        if K > 1:
            weight_k = 1.0 + float(k) / float(K - 1)
        else:
            weight_k = 1.0
        ce_k = logits.reshape(B * 81, 9).sparse_categorical_crossentropy(
            gold_flat, reduction="mean"
        )
        cell_loss_sum = cell_loss_sum + ce_k * weight_k
        weight_sum += weight_k
    cell_loss = cell_loss_sum / float(weight_sum)

    # Constraint energy on the final breath's soft predictions.
    final_probs = cell_logits_history[-1].softmax(axis=-1)
    energy_per_batch = sudoku_constraint_energy(final_probs)               # (B,)
    energy = energy_per_batch.mean()

    # Calibration: target_k = 0.5 + (correct - 0.5) * (k / (K-1))
    # correct ∈ {0,1}; B0 target = 0.5 (uncertain), B_{K-1} target = correct.
    # NOTE: correct is computed from FINAL breath argmax — train target is detached
    # so calibration learns "is my final answer right" without affecting backbone.
    # Detach the int argmax tensor BEFORE comparison: argmax propagates
    # requires_grad to its int output, and if it survives in the live scope at
    # backward time it lands in tensors_need_grad (the gc-walk picks it up by
    # requires_grad=True), tripping "only float Tensors have gradient".
    final_argmax = (cell_logits_history[-1].argmax(axis=-1) + 1).detach()  # (B, 81) digits 1..9
    correct = (final_argmax == gold_solution).cast(dtypes.float).prod(axis=-1)  # (B,) 0/1
    calib_loss_sum = Tensor.zeros((), dtype=dtypes.float)
    for k, calib_k in enumerate(calib_history):
        if K > 1:
            progression = float(k) / float(K - 1)
        else:
            progression = 1.0
        target_k = 0.5 + (correct - 0.5) * progression                     # (B,)
        # MSE loss between calibration prediction and target.
        calib_loss_sum = calib_loss_sum + ((calib_k - target_k) ** 2).mean()
    calib_loss = calib_loss_sum / float(K)

    total = cell_loss + constraint_weight * energy + calib_weight * calib_loss
    return total, {"cell_ce": cell_loss, "energy": energy, "calib": calib_loss}


# ---- per-breath CE for logging (no weighting, no aux terms) ------------------

def per_breath_ce(cell_logits_history, gold_solution: Tensor) -> list[float]:
    """Compute the raw per-breath CE (unweighted) for logging / diagnostics."""
    B = int(cell_logits_history[0].shape[0])
    gold_flat = (gold_solution - 1).reshape(B * 81)
    out = []
    for logits in cell_logits_history:
        ce = logits.reshape(B * 81, 9).sparse_categorical_crossentropy(
            gold_flat, reduction="mean"
        )
        out.append(float(ce.realize().numpy()))
    return out


# ---- accuracy ----------------------------------------------------------------

def sudoku_accuracy(cell_logits_final: Tensor, gold_solution: Tensor) -> tuple[float, float]:
    """Return (cell_accuracy, puzzle_accuracy) on a batch.

    cell_logits_final: (B, 81, 9)
    gold_solution:     (B, 81) digits 1..9
    """
    pred = cell_logits_final.argmax(axis=-1) + 1     # (B, 81)
    eq = (pred == gold_solution).cast(dtypes.float)  # (B, 81)
    cell_acc = eq.mean()
    puzzle_acc = eq.prod(axis=-1).mean()
    return float(cell_acc.realize().numpy()), float(puzzle_acc.realize().numpy())


# ---- model param attach ------------------------------------------------------

def attach_sudoku_params(model: Any, hidden: int, n_heads: int,
                          k_max: int | None = None) -> None:
    """Allocate sudoku-specific params on `model` (a BreathingTransformer instance).

    Attributes added:
      sudoku_state_embed     (10, hidden)         — 10 cell states (0=unknown, 1..9)
      sudoku_position_embed  (81, hidden)         — structured + learned
      sudoku_digit_codebook  (9, hidden)          — used as @ T.cast(...)
      sudoku_calib_head_w    (hidden, 1)
      sudoku_calib_head_b    (1,)
      sudoku_breath_embed    (K_max, hidden)      — per-breath additive bias
      sudoku_delta_gate      (K_max,)             — learnable per-breath delta scale
      sudoku_attn_bias       (n_heads, 81, 81)    — precomputed additive bias (frozen)

    All trainable params are FP32 (the optimizer expects fp32 grads on this codebase).
    Cast to fp16 happens on use inside the forward.

    k_max defaults to SUDOKU_K_MAX env var if not supplied.
    """
    if k_max is None:
        k_max = SUDOKU_K_MAX

    # Structured position embedding (row/col/box one-hots + randn tail).
    model.sudoku_position_embed = _build_sudoku_position_features(hidden)

    # Digit codebook — orthonormal init at scale 0.1. Orthogonal rows ensure
    # no single digit dominates at random init (avoids collapse to e.g. 60% mass
    # on digit 2 that we observed with naive randn init).
    rng_cb = np.random.RandomState(9803)
    raw_cb = rng_cb.randn(max(hidden, 9), hidden).astype(np.float32)
    q_cb, _ = np.linalg.qr(raw_cb)
    cb_unit = q_cb[:9].astype(np.float32)  # orthonormal rows, scale 1.0
    model.sudoku_digit_codebook = Tensor(cb_unit * 0.1, dtype=dtypes.float).contiguous()

    # State embedding — 10 rows for {0=unknown, 1..9 = given digit}.
    # CRITICAL INIT: state_embed[i] aligns with digit_codebook[i-1] for digits 1..9
    # so given cells start with logits strongly favoring their given digit. This
    # skips the "learn the identity mapping from scratch" trap. The model still
    # has to learn to propagate from given cells to unknown cells (the real work);
    # this just gives it the trivial 'a given digit is itself' for free.
    # Scale matches the digit_codebook (0.1) so post-LN projection gives logits
    # ~ord(1) immediately for given cells.
    state = np.zeros((10, hidden), dtype=np.float32)
    # row 0 = "unknown" → small random vector (model decides its meaning)
    state[0] = np.random.RandomState(9802).randn(hidden).astype(np.float32) * 0.02
    # rows 1..9 = given digits → align with codebook rows
    state[1:10] = cb_unit  # scale 1.0 (vs codebook 0.1) → big logits on correct digit
    model.sudoku_state_embed = Tensor(state, dtype=dtypes.float).contiguous()

    # Calibration head — Pythia-scale randn, zero bias.
    cw = (np.random.RandomState(9804).randn(hidden, 1) * 0.02).astype(np.float32)
    model.sudoku_calib_head_w = Tensor(cw, dtype=dtypes.float).contiguous()
    model.sudoku_calib_head_b = Tensor.zeros((1,), dtype=dtypes.float).contiguous()

    # Per-breath additive embedding. ORTHOGONAL init at scale 1.0 (or override
    # via SUDOKU_BREATH_EMBED_SCALE env). Each breath gets a unique linearly-
    # independent additive signal so the model can distinguish "which iteration"
    # this is. Zero-init would leave all breaths indistinguishable from each
    # other's residual content (cf v77 → v77b lesson: orthogonal init separates
    # the per-breath gradients).
    breath_scale = float(os.environ.get("SUDOKU_BREATH_EMBED_SCALE", "0.5"))
    rng_be = np.random.RandomState(9805)
    raw = rng_be.randn(max(k_max, hidden), hidden).astype(np.float32)
    # QR-based orthonormal basis, take first k_max rows
    q, _ = np.linalg.qr(raw)
    be = q[:k_max].astype(np.float32) * breath_scale
    model.sudoku_breath_embed = Tensor(be, dtype=dtypes.float).contiguous()

    # Learnable per-breath delta gate. Init at 1.0 so initial behavior is the
    # full residual update; gradient can reduce late-breath gates so the
    # iterative refinement converges (small late deltas → stable late breaths).
    model.sudoku_delta_gate = Tensor.ones((k_max,), dtype=dtypes.float).contiguous()

    # Precomputed structured attention bias. Mask 1.0 → 0 bias; 0.0 → -1e4 bias.
    # FROZEN (no gradient — it's a structural inductive bias).
    mask = _build_sudoku_attention_masks(n_heads)                          # (n_heads, 81, 81)
    bias = (1.0 - mask) * (-1e4)                                           # 0 or -1e4
    model.sudoku_attn_bias = bias.contiguous()


def sudoku_parameters(model: Any) -> list[Tensor]:
    """Trainable sudoku-specific params (everything but the frozen attn mask).

    Caller MUST add these to the AdamW param list when SUDOKU_TASK=1. The bias
    tensor is NOT in this list — it's a structural prior, frozen by construction.
    """
    return [
        model.sudoku_state_embed,
        model.sudoku_position_embed,
        model.sudoku_digit_codebook,
        model.sudoku_calib_head_w,
        model.sudoku_calib_head_b,
        model.sudoku_breath_embed,
        model.sudoku_delta_gate,
    ]


def sudoku_state_dict(model: Any) -> dict[str, Tensor]:
    """State dict entries for sudoku params (excluding the static attn bias)."""
    return {
        "sudoku.state_embed": model.sudoku_state_embed,
        "sudoku.position_embed": model.sudoku_position_embed,
        "sudoku.digit_codebook": model.sudoku_digit_codebook,
        "sudoku.calib_head_w": model.sudoku_calib_head_w,
        "sudoku.calib_head_b": model.sudoku_calib_head_b,
        "sudoku.breath_embed": model.sudoku_breath_embed,
        "sudoku.delta_gate": model.sudoku_delta_gate,
    }


# ---- JIT'd training step ------------------------------------------------------
#
# Without TinyJit, the lazy graph accumulates across steps (each step time grows
# linearly: 4.5s → 8.7s over 400 steps). With TinyJit, the forward+backward+step
# is compiled ONCE and replayed for every subsequent step — flat ~1-2s/step.
#
# The pattern mirrors mycelium/l3_training.py: _compile_jit_per_breath_step.
# Stable input shapes (B, 81) and (B, 81), constant K captured in Python,
# scalar tensor outputs all realized for cheap .numpy() readback.

_JIT_SUDOKU_CACHE: dict = {}


def _compile_jit_sudoku_step(model: Any, opt: Any, K: int, B: int,
                              constraint_weight: float, calib_weight: float,
                              grad_clip: float = 0.0):
    """Compile and return a TinyJit'd train step for the sudoku breathing forward.

    Inputs (stable shapes; pass realized Tensors):
      input_cells   : (B, 81) int   — cell states (0=unknown, 1..9 given digit)
      gold_solution : (B, 81) int   — gold cell digits 1..9

    Returns (each is a realized scalar Tensor):
      total_loss, healthy, cell_ce, energy, calib, ce_per_breath...

    The JIT cache is keyed on (model id, opt id, K, B, weights, grad_clip) so
    repeated calls with the same shape hit the cached step. Different K values
    (e.g. K=15 smoke vs K=20 prod) compile distinct graphs.

    Healthy flag: if total isn't finite (NaN forward), grads are multiplied by 0
    before opt.step — graceful skip without per-param isnan loops (AMD JIT safe).
    """
    key = (id(model), id(opt), int(K), int(B), float(constraint_weight),
           float(calib_weight), float(grad_clip))
    if key in _JIT_SUDOKU_CACHE:
        return _JIT_SUDOKU_CACHE[key]

    cw = float(constraint_weight)
    aw = float(calib_weight)
    gc_val = float(grad_clip)
    params = opt.params

    _jit_compile_start = _time.perf_counter()
    print(f"[JIT] compile sudoku step: K={K} B={B} cw={cw} aw={aw} clip={gc_val}...",
          flush=True)

    @TinyJit
    def _step(input_cells: Tensor, gold_solution: Tensor):
        opt.zero_grad()

        # Forward — sudoku_breathing_forward returns (cell_logits_history, calib_history)
        # Lists of K (B, 81, 9) fp32 and K (B,) fp32 tensors. K is constant so the
        # loop unrolls and the graph topology is fully static.
        cell_logits_history, calib_history = sudoku_breathing_forward(model, input_cells, K=K)

        # Loss — sudoku_loss reduces to a single scalar + a parts dict.
        # We inline it here instead of calling sudoku_loss to (a) keep the parts
        # we need as scalars in the JIT return, and (b) match l3_training.py's
        # pattern (all loss math inside JIT for fusion).
        gold_idx = gold_solution - 1                                       # (B, 81)
        gold_flat = gold_idx.reshape(B * 81)                               # (B*81,)

        # Per-breath weighted CE ladder (linear ramp 1.0 → 2.0)
        cell_loss_sum = Tensor.zeros((), dtype=dtypes.float).contiguous()
        weight_sum = 0.0
        per_breath_ce_losses = []
        for k, logits in enumerate(cell_logits_history):
            if K > 1:
                weight_k = 1.0 + float(k) / float(K - 1)
            else:
                weight_k = 1.0
            ce_k = logits.reshape(B * 81, 9).sparse_categorical_crossentropy(
                gold_flat, reduction="mean"
            )
            per_breath_ce_losses.append(ce_k)
            cell_loss_sum = cell_loss_sum + ce_k * weight_k
            weight_sum += weight_k
        cell_loss = cell_loss_sum / float(weight_sum)

        # Constraint energy on final breath
        final_probs = cell_logits_history[-1].softmax(axis=-1)
        energy_per_batch = sudoku_constraint_energy(final_probs)           # (B,)
        energy = energy_per_batch.mean()

        # Calibration with detached gold check (also reused for train accuracy).
        # argmax propagates requires_grad to its int output via tinygrad's tracking;
        # detach the int tensor BEFORE any further ops so it doesn't end up in
        # tensors_need_grad at backward time (would trip "only float Tensors have
        # gradient" since the int argmax tensor would be picked up by gc-walk).
        final_argmax = (cell_logits_history[-1].argmax(axis=-1) + 1).detach()
        eq = (final_argmax == gold_solution).cast(dtypes.float)            # (B, 81)
        correct = eq.prod(axis=-1)                                          # (B,) 0/1
        # eq itself stays detached because final_argmax was detached (no grad-path).
        calib_loss_sum = Tensor.zeros((), dtype=dtypes.float).contiguous()
        for k, calib_k in enumerate(calib_history):
            if K > 1:
                progression = float(k) / float(K - 1)
            else:
                progression = 1.0
            target_k = 0.5 + (correct - 0.5) * progression
            calib_loss_sum = calib_loss_sum + ((calib_k - target_k) ** 2).mean()
        calib_loss = calib_loss_sum / float(K)

        # Train accuracy (per-cell + per-puzzle) for cheap logging — detached.
        train_cell_acc = eq.mean().detach()
        train_puzzle_acc = eq.prod(axis=-1).mean().detach()

        total = cell_loss + cw * energy + aw * calib_loss
        total.backward()

        # NaN-skip: zero all grads if loss isn't finite. Single-kernel isfinite()
        # check matches the l3_training.py AMD-JIT-safe pattern (avoids per-param
        # isnan() loop that crashes the AM driver in JIT).
        healthy = total.isfinite().cast(dtypes.float)
        for p in params:
            if p.grad is not None:
                p.grad = p.grad * healthy.cast(p.grad.dtype)

        # Optional global-norm gradient clipping
        if gc_val > 0:
            sq_sum = Tensor.zeros((), dtype=dtypes.float).contiguous()
            for p in params:
                if p.grad is not None:
                    sq_sum = sq_sum + p.grad.cast(dtypes.float).square().sum()
            grad_norm = (sq_sum + 1e-12).sqrt()
            clip_coef = (Tensor(gc_val, dtype=dtypes.float) / (grad_norm + 1e-6))
            clip_coef = clip_coef.minimum(Tensor(1.0, dtype=dtypes.float))
            for p in params:
                if p.grad is not None:
                    p.grad = p.grad * clip_coef.cast(p.grad.dtype)

        opt.step()

        # Return all realized scalars so the caller can cheap-readback per-step.
        # Order: total, healthy, cell_ce, energy, calib, train_cell_acc, train_puzzle_acc,
        #        per_breath_ce_0, per_breath_ce_1, ..., per_breath_ce_{K-1}
        return (
            total.realize(),
            healthy.realize(),
            cell_loss.realize(),
            energy.realize(),
            calib_loss.realize(),
            train_cell_acc.realize(),
            train_puzzle_acc.realize(),
            *(ce.realize() for ce in per_breath_ce_losses),
        )

    _JIT_SUDOKU_CACHE[key] = _step
    print(f"[JIT] sudoku step ready (cache size={len(_JIT_SUDOKU_CACHE)}); "
          f"first call will compile (~60-90s)…", flush=True)
    return _step


def _compile_jit_sudoku_eval(model: Any, K: int, B: int):
    """Compile a TinyJit'd eval step (forward-only, no backward).

    Returns (per-cell eq mask (B, 81) float, final cell logits (B, 81, 9)).
    Inference graph is much cheaper than train: no backward, no opt.step.
    Train accuracy is computed inline so eval doesn't need to redo argmax in
    Python land.

    Inputs:
      input_cells   : (B, 81) int
      gold_solution : (B, 81) int   — used only to compute eq mask + accuracies

    Returns:
      eq_mask      : (B, 81) fp32 0/1
      cell_acc     : scalar fp32
      puzzle_acc   : scalar fp32
    """
    key = ("eval", id(model), int(K), int(B))
    if key in _JIT_SUDOKU_CACHE:
        return _JIT_SUDOKU_CACHE[key]

    _jit_compile_start = _time.perf_counter()
    print(f"[JIT] compile sudoku eval: K={K} B={B}…", flush=True)

    @TinyJit
    def _eval(input_cells: Tensor, gold_solution: Tensor):
        cell_logits_history, _ = sudoku_breathing_forward(model, input_cells, K=K)
        final_logits = cell_logits_history[-1]
        pred = final_logits.argmax(axis=-1) + 1
        eq = (pred == gold_solution).cast(dtypes.float)
        cell_acc = eq.mean()
        puzzle_acc = eq.prod(axis=-1).mean()
        return eq.realize(), cell_acc.realize(), puzzle_acc.realize()

    _JIT_SUDOKU_CACHE[key] = _eval
    print(f"[JIT] sudoku eval ready (cache size={len(_JIT_SUDOKU_CACHE)})", flush=True)
    return _eval
