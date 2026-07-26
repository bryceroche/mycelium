"""v99 factor graph breathing transformer.

Generalises the v98 Sudoku iterative-prefill paradigm to arbitrary arithmetic
factor graphs. Instead of an (81,) grid with fixed row/col/box constraints, each
problem is a bipartite graph of variable nodes and factor nodes connected by
arithmetic constraints (add/sub/mul/div).

Architecture decisions:
  - Same 4 shared Pythia-init transformer layers as v98 (reuses BreathingTransformer
    L0-L3 weights, SharedWeights, and attribute access pattern).
  - No RoPE, no causal mask (same as v98) — constraint propagation is bidirectional.
  - Dynamic per-problem (B, T, T) attention mask: variable_i ↔ factor_j iff variable_i
    appears in factor_j's args. Factor type (op) is encoded as additive K/V bias.
  - T_MAX = 24 positions: 0..N_MAX-1 = variable nodes, N_MAX..N_MAX+F_MAX-1 = factor nodes.
  - Domain distribution over [0, 99] decoded via (100, hidden) codebook (vs 9-way for Sudoku).
  - Constraint energy uses moment-matching (differentiable approximation) instead of
    the exact set-equality energy in Sudoku.

Env var gates (set in launchers):
  V99_TASK=1                   — turn on the factor-graph forward path
  V99_K_MAX=20                 — number of iterative-prefill breaths
  V99_ENERGY_WEIGHT=0.1        — weight on constraint energy term
  V99_CALIB_WEIGHT=0.05        — weight on per-breath calibration loss
  V99_N_MAX=16                 — max variable nodes (leaves + results)
  V99_F_MAX=8                  — max factor nodes
"""
from __future__ import annotations

import math
import os
import time as _time
from typing import Any

import numpy as np
from tinygrad import Tensor, dtypes
from tinygrad.engine.jit import TinyJit


V99_TASK = int(os.environ.get("V99_TASK", "0")) > 0
V99_K_MAX = int(os.environ.get("V99_K_MAX", "20"))
V99_ENERGY_WEIGHT = float(os.environ.get("V99_ENERGY_WEIGHT", "0.1"))
V99_CALIB_WEIGHT = float(os.environ.get("V99_CALIB_WEIGHT", "0.05"))
V99_N_MAX = int(os.environ.get("V99_N_MAX", "16"))   # max variable nodes
V99_F_MAX = int(os.environ.get("V99_F_MAX", "8"))    # max factor nodes
V99_T_MAX = V99_N_MAX + V99_F_MAX                    # = 24 total positions

# Op index mapping (fixed)
OP_ADD = 0
OP_SUB = 1
OP_MUL = 2
OP_DIV = 3
OP_PAD = -1   # padding sentinel (not a real factor)

# Node-kind codes
KIND_OBSERVED   = 0
KIND_UNOBSERVED = 1
KIND_FACTOR     = 2


# ---------------------------------------------------------------------------
# Attention mask construction (per-problem, dynamic)
# ---------------------------------------------------------------------------

def build_factor_graph_masks_np(
    factor_types_np: np.ndarray,   # (B, F_MAX) int; -1 = padding
    factor_args_np: np.ndarray,    # (B, F_MAX, 3) int; padding rows irrelevant
    n_vars_np: np.ndarray,         # (B,) int — TOTAL vars (leaves + results)
    n_factors_np: np.ndarray,      # (B,) int
    n_max: int = V99_N_MAX,
    f_max: int = V99_F_MAX,
) -> np.ndarray:
    """Build the per-problem additive attention bias: (B, T_MAX, T_MAX) float32.

    Allowed attention pairs (value = 0.0):
      - Every token attends to itself (diagonal).
      - variable_i ↔ factor_j if variable_i appears in factor_j's args.
      - factor_j self-loop (already covered by diagonal).

    Blocked pairs (value = -1e4):
      - variable_i ↔ variable_j (no direct variable-to-variable communication).
      - factor_j ↔ factor_k (no factor-to-factor communication).
      - Padding positions (beyond n_vars or n_factors).

    Factor positions in the sequence are n_max + fi (fi = 0..F_MAX-1).
    Variable positions are 0..(n_max-1).
    """
    B = factor_types_np.shape[0]
    t_max = n_max + f_max
    bias = np.full((B, t_max, t_max), -1e4, dtype=np.float32)

    for b in range(B):
        nv = int(n_vars_np[b])
        nf = int(n_factors_np[b])

        # Self-attention for all real nodes
        for i in range(nv):
            bias[b, i, i] = 0.0
        for fi in range(nf):
            fpos = n_max + fi
            bias[b, fpos, fpos] = 0.0

        # Variable ↔ factor edges
        for fi in range(nf):
            ft = int(factor_types_np[b, fi])
            if ft < 0:
                continue  # padding factor
            fa = factor_args_np[b, fi]  # [arg1_idx, arg2_idx, result_idx]
            fpos = n_max + fi
            for vi in fa:
                if 0 <= vi < nv:
                    # variable_vi ↔ factor_fi (bidirectional)
                    bias[b, vi, fpos] = 0.0
                    bias[b, fpos, vi] = 0.0

    return bias  # (B, T_MAX, T_MAX) float32


# ---------------------------------------------------------------------------
# Per-factor K/V additive op bias (encodes factor type into key/value)
# ---------------------------------------------------------------------------

def factor_kv_bias_np(
    factor_types_np: np.ndarray,   # (B, F_MAX) int; -1 = padding
    op_embed_np: np.ndarray,       # (4, H) float32 — the learnable op embeddings
    n_max: int = V99_N_MAX,
    f_max: int = V99_F_MAX,
) -> np.ndarray:
    """Build a (B, T_MAX, H) additive bias for K and V projections.

    Only factor positions (n_max .. n_max+F_MAX-1) get a non-zero bias.
    Variable positions and padding positions get zeros.
    This lets the transformer read op-type info from the factor node K/V.
    """
    B = factor_types_np.shape[0]
    t_max = n_max + f_max
    H = op_embed_np.shape[1]
    kv_bias = np.zeros((B, t_max, H), dtype=np.float32)

    for b in range(B):
        for fi in range(f_max):
            ft = int(factor_types_np[b, fi])
            if 0 <= ft < 4:
                fpos = n_max + fi
                kv_bias[b, fpos] = op_embed_np[ft]

    return kv_bias  # (B, T_MAX, H) float32


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def build_kv_bias_tensor(
    factor_types_t: Tensor,  # (B, F_MAX) int — -1=padding, 0..3=op type
    op_embed: Tensor,         # (5, H) learnable — row 0 = zero sentinel, rows 1-4 = ops
    n_max: int = V99_N_MAX,
    f_max: int = V99_F_MAX,
) -> Tensor:
    """Build the (B, T_MAX, H) K/V additive bias from factor_types and op_embed.

    Factor positions (n_max..n_max+F_MAX-1) get the corresponding op embedding.
    Variable positions get zeros via the zero sentinel row.

    This computes kv_bias INSIDE the graph so op_embed receives gradients.
    Uses one_hot encoding of factor_types. The op_embed has 5 rows: row 0 is a
    fixed-zero sentinel for both padding and variable positions; rows 1..4 are
    the learnable add/sub/mul/div embeddings.

    Avoids Tensor.zeros() inside JIT (AMD AM driver stability).
    """
    # factor_types_t: (B, F_MAX) int, values in {-1, 0, 1, 2, 3}
    # Remap: -1 → 0 (zero sentinel), 0..3 → 1..4 (ops).
    ft_remapped = (factor_types_t + 1).cast(dtypes.int).clip(0, 4)  # (B, F_MAX)

    # Build (B, F_MAX+N_MAX = T_MAX) index array:
    # variable positions (0..N_MAX-1) → 0 (zero sentinel)
    # factor positions (N_MAX..T_MAX-1) → ft_remapped[b, fi]
    # Create a sentinel row for variable positions — all zeros (index 0 → zero row)
    # Build (B, T_MAX) index tensor:
    # - Variable positions (0..n_max-1): index = 0 (zero sentinel row → zero kv)
    # - Factor positions (n_max..T_MAX-1): index = ft_remapped[b, fi]
    #
    # AMD-JIT-safe: use Tensor.pad() to prepend N_MAX zeros on the last dim.
    # pad((n_max, 0)) inserts n_max zeros at the start of the last axis.
    t_indices = ft_remapped.pad((n_max, 0))  # (B, T_MAX) — var positions get 0

    # Lookup: (B, T_MAX) int indices → (B, T_MAX, H) via one_hot @ op_embed(5, H)
    t_oh = t_indices.one_hot(5).cast(op_embed.dtype)   # (B, T_MAX, 5)
    kv_bias = t_oh @ op_embed                            # (B, T_MAX, H)
    return kv_bias


def embed_factor_graph(
    domain_init: Tensor,        # (B, N_MAX, 100) float — soft domain dist
    node_kinds: Tensor,          # (B, T_MAX) int — 0=obs, 1=unobs, 2=factor, -1=padding
    var_pos_embed: Tensor,        # (N_MAX, H) learned
    factor_pos_embed: Tensor,     # (F_MAX, H) learned
    node_kind_embed: Tensor,      # (3, H) learned
    domain_codebook: Tensor,      # (100, H) — maps domain dist → hidden
) -> Tensor:
    """Produce initial (B, T_MAX, H) hidden states.

    Variable positions (0..N_MAX-1):
      state = (domain_init @ codebook) + var_pos_embed[i] + node_kind_embed[kind_i]

    Factor positions (N_MAX..N_MAX+F_MAX-1):
      state = factor_pos_embed[fi] + node_kind_embed[2]
      The factor type is encoded in kv_bias (computed separately via build_kv_bias_tensor),
      added to K and V during attention — not to the residual stream here.
    """
    B = int(domain_init.shape[0])
    N_MAX = int(domain_init.shape[1])
    H = int(var_pos_embed.shape[1])
    F_MAX = int(factor_pos_embed.shape[0])

    # Variable states: domain_init (B, N_MAX, 100) @ codebook (100, H) → (B, N_MAX, H)
    var_state = domain_init.cast(domain_codebook.dtype) @ domain_codebook  # (B, N_MAX, H)

    # Add variable position embeddings (broadcast over batch)
    var_pos = var_pos_embed.reshape(1, N_MAX, H).cast(var_state.dtype).expand(B, N_MAX, H)
    var_h = var_state + var_pos

    # Factor states: just the position embedding (type info goes into K/V bias)
    factor_pos = factor_pos_embed.reshape(1, F_MAX, H).cast(var_state.dtype).expand(B, F_MAX, H)

    # Concatenate along the T axis: (B, N_MAX, H) + (B, F_MAX, H) → (B, T_MAX, H)
    x = var_h.cat(factor_pos, dim=1)  # (B, T_MAX, H)

    # Add node-kind embedding: clamp padding positions (-1) to 0 for one_hot safety
    nk_clamped = node_kinds.clip(0, 2)
    nk_one_hot = nk_clamped.one_hot(3).cast(x.dtype)   # (B, T_MAX, 3)
    nk_emb = nk_one_hot @ node_kind_embed.cast(x.dtype)  # (B, T_MAX, H)
    x = x + nk_emb

    return x  # (B, T_MAX, H)


# ---------------------------------------------------------------------------
# One transformer layer forward with factor-graph attention
# ---------------------------------------------------------------------------

def factor_graph_layer_forward(
    layer: Any,
    x: Tensor,           # (B, T_MAX, H)
    attn_bias: Tensor,   # (B, T_MAX, T_MAX) additive mask
    kv_bias: Tensor,     # (B, T_MAX, H) op-type additive K/V
) -> Tensor:
    """Run one BreathingLayer with factor-graph bipartite attention.

    Differences from sudoku_layer_forward:
      - No fixed per-head mask — one shared (B, T_MAX, T_MAX) mask for all heads.
      - kv_bias (B, T_MAX, H) is added to K and V BEFORE projection split so
        factor nodes carry op-type signal in their keys/values.
      - T is dynamic (T_MAX) rather than the fixed 81 cells.
    """
    cfg = layer.cfg
    B, S, H = x.shape
    n_heads = cfg.n_heads
    head_dim = cfg.head_dim

    from mycelium.breathing import _layernorm
    attn_in = _layernorm(x, layer.shared.in_ln_g, layer.shared.in_ln_b, cfg.layer_norm_eps)
    mlp_in  = _layernorm(x, layer.shared.post_ln_g, layer.shared.post_ln_b, cfg.layer_norm_eps)
    attn_in_dt = attn_in.cast(x.dtype) if attn_in.dtype != x.dtype else attn_in
    mlp_in_dt  = mlp_in.cast(x.dtype) if mlp_in.dtype != x.dtype else mlp_in

    # Add op-type bias to K and V input BEFORE projection, so factor nodes'
    # keys and values carry op-type information.
    kv_in = attn_in_dt + kv_bias.cast(attn_in_dt.dtype)

    q = (attn_in_dt @ layer.wq + layer.bq).reshape(B, S, n_heads, head_dim).transpose(1, 2)
    k = (kv_in      @ layer.wk + layer.bk).reshape(B, S, n_heads, head_dim).transpose(1, 2)
    v = (kv_in      @ layer.shared.wv + layer.shared.bv).reshape(B, S, n_heads, head_dim).transpose(1, 2)

    # No RoPE
    scale = 1.0 / math.sqrt(head_dim)
    scores = q @ k.transpose(-2, -1) * scale  # (B, n_heads, S, S)

    # attn_bias is (B, T_MAX, T_MAX): expand to (B, 1, S, S) for broadcast over heads
    bias_expanded = attn_bias.cast(scores.dtype).reshape(B, 1, S, S)
    scores = scores + bias_expanded
    attn = scores.clip(-1e4, 1e4).softmax(-1)
    ctx = (attn @ v).transpose(1, 2).reshape(B, S, H)
    attn_out = ctx @ layer.shared.wo + layer.shared.bo

    ff = (mlp_in_dt @ layer.w_in + layer.b_in).gelu()
    ffn_out = ff @ layer.shared.w_out + layer.shared.b_out

    return x + attn_out + ffn_out


# ---------------------------------------------------------------------------
# Iterative prefill loop (the breathing)
# ---------------------------------------------------------------------------

def factor_graph_breathing_forward(
    model: Any,
    domain_init: Tensor,     # (B, N_MAX, 100) float — initial domain distributions
    node_kinds: Tensor,       # (B, T_MAX) int — 0=observed, 1=unobserved, 2=factor
    attn_bias: Tensor,        # (B, T_MAX, T_MAX) float — precomputed per-problem mask
    factor_types_t: Tensor,   # (B, F_MAX) int — needed to build kv_bias inside graph
    K: int,
    n_max: int = V99_N_MAX,
    f_max: int = V99_F_MAX,
) -> tuple[list[Tensor], list[Tensor]]:
    """Run K iterative-prefill breaths on a batch of factor graphs.

    kv_bias is built INSIDE this function from model.fg_op_embed and factor_types_t
    so that fg_op_embed receives gradients.

    Returns:
      var_logits_history : list of K Tensors, each (B, N_MAX, 100) — domain logits
      calib_history      : list of K Tensors, each (B,) — sigmoid confidence
    """
    assert hasattr(model, "fg_domain_codebook"), \
        "model has no factor-graph params; was V99_TASK set before model init?"

    domain_codebook  = model.fg_domain_codebook   # (100, H)
    var_pos_embed    = model.fg_var_pos_embed      # (N_MAX, H)
    factor_pos_embed = model.fg_factor_pos_embed   # (F_MAX, H)
    node_kind_embed  = model.fg_node_kind_embed    # (3, H)
    op_embed         = model.fg_op_embed           # (4, H)
    breath_embed     = model.fg_breath_embed       # (K_max, H)
    delta_gate       = model.fg_delta_gate         # (K_max,)
    calib_head_w     = model.fg_calib_head_w       # (H, 1)
    calib_head_b     = model.fg_calib_head_b       # (1,)

    # Build kv_bias inside the graph so op_embed gets gradients
    kv_bias = build_kv_bias_tensor(factor_types_t, op_embed, n_max=n_max, f_max=f_max)  # (B, T_MAX, H)

    # Initial embedding → fp16 (matches transformer compute dtype)
    x = embed_factor_graph(
        domain_init, node_kinds,
        var_pos_embed, factor_pos_embed, node_kind_embed, domain_codebook,
    )
    x = x.cast(dtypes.half) if x.dtype != dtypes.half else x

    layers = list(model.block.layers)
    assert len(layers) >= 4, f"expected at least 4 transformer layers; got {len(layers)}"
    K_max = int(breath_embed.shape[0])
    assert K <= K_max, f"K={K} exceeds K_max={K_max} for fg_breath_embed"

    from mycelium.breathing import _layernorm

    var_logits_history = []
    calib_history = []

    for k in range(K):
        be_k = breath_embed[k].reshape(1, 1, -1).cast(x.dtype)
        x_in = x + be_k
        x_pre = x

        h = x_in
        for layer in layers[:4]:
            h = factor_graph_layer_forward(layer, h, attn_bias, kv_bias.cast(h.dtype))

        gate_k = delta_gate[k].cast(h.dtype).reshape(1, 1, 1)
        delta = h - x_pre
        x = x_pre + gate_k * delta

        # Readout: only variable positions (first N_MAX tokens)
        x_ln = _layernorm(x, model.ln_f_g, model.ln_f_b, model.cfg.layer_norm_eps).cast(dtypes.float)
        var_x = x_ln[:, :n_max, :]                                     # (B, N_MAX, H)
        var_logits_k = var_x @ domain_codebook.T.cast(dtypes.float)    # (B, N_MAX, 100)
        var_logits_history.append(var_logits_k)

        # Calibration: mean-pool over all T tokens
        pool = x_ln.mean(axis=1)                                       # (B, H)
        calib_logit = pool @ calib_head_w.cast(dtypes.float) + calib_head_b.cast(dtypes.float)
        calib_k = calib_logit.reshape(-1).sigmoid()                    # (B,)
        calib_history.append(calib_k)

    return var_logits_history, calib_history


# ---------------------------------------------------------------------------
# Constraint energy (differentiable moment-matching)
# ---------------------------------------------------------------------------

def factor_graph_constraint_energy(
    var_logits_k: Tensor,   # (B, N_MAX, 100)
    factor_types_t: Tensor, # (B, F_MAX) int — -1 = padding
    factor_args_t: Tensor,  # (B, F_MAX, 3) int
    n_max: int = V99_N_MAX,
    f_max: int = V99_F_MAX,
) -> Tensor:
    """Compute differentiable constraint energy across all factors.

    For each factor (op, arg1_idx, arg2_idx, result_idx) we measure the
    discrepancy between the predicted moment of result and the expected moment
    given arg1 and arg2's soft distributions.

    Returns: (B,) scalar energy per sample.
    """
    B = int(var_logits_k.shape[0])
    values = Tensor(np.arange(100, dtype=np.float32), dtype=dtypes.float)  # (100,)
    values_sq = values * values  # (100,)

    # Soft probabilities over [0, 99] for all N_MAX variables simultaneously.
    all_probs = var_logits_k.softmax(axis=-1)   # (B, N_MAX, 100)

    # Expected value for each variable: E[X] = sum(p * v)
    mean_all = (all_probs * values.reshape(1, 1, 100)).sum(axis=-1)   # (B, N_MAX)
    # E[X^2] for variance computation
    sq_all   = (all_probs * values_sq.reshape(1, 1, 100)).sum(axis=-1) # (B, N_MAX)
    var_all  = sq_all - mean_all * mean_all                             # (B, N_MAX)

    # We compute energy per factor and sum. Because the factor indices come from
    # Python (factor_args_t is variable across batch items), we loop over the
    # F_MAX factor slots. Each slot contributes zeros for padding factors.
    #
    # We index into mean_all and var_all using gather-like indexing. Tinygrad
    # supports advanced indexing: mean_all[:, idx] for a fixed integer idx.
    # Since factor_args is ragged (each batch item can have different args), we
    # do the loop in Python (F_MAX=8 iterations) but keep all batch operations
    # fused within each iteration.
    #
    # Important: factor_args can refer to indices beyond n_MAX for some problems
    # (shouldn't happen per data generator, but we clamp for safety).

    energy = Tensor.zeros((B,), dtype=dtypes.float).contiguous()

    # Convert factor_types and factor_args to numpy for indexing
    # (they're already numpy-derived realized tensors in the JIT path; realize here)
    ft_np = factor_types_t.numpy()   # (B, F_MAX)
    fa_np = factor_args_t.numpy()    # (B, F_MAX, 3)

    for fi in range(f_max):
        # Per-batch-item: gather the moments for this factor slot.
        # We build a batch of (arg1_mean, arg2_mean, result_mean) tensors.
        # Batching: for each op type, we compute the energy contribution only
        # for batch items where this factor slot has that op type.

        for b_idx in range(B):
            op = int(ft_np[b_idx, fi])
            if op < 0:
                continue  # padding
            a_idx = int(fa_np[b_idx, fi, 0])
            bx_idx = int(fa_np[b_idx, fi, 1])
            r_idx  = int(fa_np[b_idx, fi, 2])

            # Clamp to valid range
            n_total = int(var_logits_k.shape[1])
            if a_idx >= n_total or bx_idx >= n_total or r_idx >= n_total:
                continue

            ma = mean_all[b_idx, a_idx]   # scalar
            mb = mean_all[b_idx, bx_idx]
            mr = mean_all[b_idx, r_idx]
            va = var_all[b_idx, a_idx]
            vb = var_all[b_idx, bx_idx]
            vr = var_all[b_idx, r_idx]

            if op == OP_ADD:
                # E[R] = E[A] + E[B]; Var[R] = Var[A] + Var[B]
                mean_err = (mr - (ma + mb)) ** 2
                var_err  = (vr - (va + vb)) ** 2
                e = mean_err + var_err
            elif op == OP_SUB:
                # E[R] = E[A] - E[B]; Var[R] = Var[A] + Var[B]
                mean_err = (mr - (ma - mb)) ** 2
                var_err  = (vr - (va + vb)) ** 2
                e = mean_err + var_err
            elif op == OP_MUL:
                # E[R] ≈ E[A] * E[B] (ignoring covariance)
                e = (mr - ma * mb) ** 2
            elif op == OP_DIV:
                # E[A] ≈ E[R] * E[B] (rearranged: avoid div by zero)
                e = (mr * mb - ma) ** 2
            else:
                continue

            # Add to energy[b_idx] — use index slice assignment pattern
            # (realized per iteration, acceptable since F_MAX * B is small)
            energy = energy + Tensor(
                np.eye(B, dtype=np.float32)[b_idx],
                dtype=dtypes.float,
            ) * e

    return energy  # (B,)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def factor_graph_loss(
    var_logits_history: list[Tensor],  # K * (B, N_MAX, 100)
    calib_history: list[Tensor],       # K * (B,)
    gold_values: Tensor,               # (B, N_MAX) int — values 0..99
    observed_mask: Tensor,             # (B, N_MAX) bool/int — 1=observed, 0=unobserved
    factor_types_t: Tensor,            # (B, F_MAX) int
    factor_args_t: Tensor,             # (B, F_MAX, 3) int
    energy_weight: float = V99_ENERGY_WEIGHT,
    calib_weight: float = V99_CALIB_WEIGHT,
    n_max: int = V99_N_MAX,
    f_max: int = V99_F_MAX,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Per-breath CE on UNOBSERVED variables + constraint energy + calibration.

    CE is computed ONLY on unobserved variables (mask out observed positions).
    Late-breath weighting ramps 1.0 → 2.0 across K breaths.
    """
    K = len(var_logits_history)
    B = int(var_logits_history[0].shape[0])

    # Build unobserved mask: (B, N_MAX) int — 1 where unobserved, 0 where observed
    unobs_mask = (1 - observed_mask.cast(dtypes.float))  # (B, N_MAX)

    # gold_values: (B, N_MAX) int — values 0..99 (the CE targets)
    # We compute a masked CE: sum(unobs_mask * CE_per_cell) / sum(unobs_mask)
    # Using sparse_categorical_crossentropy over flattened positions then masking.
    gold_flat = gold_values.cast(dtypes.int).reshape(B * n_max)  # (B*N_MAX,)

    var_loss_sum = Tensor.zeros((), dtype=dtypes.float).contiguous()
    weight_sum = 0.0
    per_breath_ce_list = []

    for k, logits in enumerate(var_logits_history):
        weight_k = 1.0 + float(k) / float(max(K - 1, 1))
        # CE per position: (B*N_MAX,) — reduction='none' equivalent via sum/mean trick
        logits_flat = logits.reshape(B * n_max, 100)   # (B*N_MAX, 100)
        # Use sparse_categorical_crossentropy with reduction=mean as baseline,
        # but apply mask manually: compute CE for all, then weight by unobs_mask.
        # To mask: multiply log-probs by unobs, sum, divide by n_unobserved.
        # Tinygrad's sparse_categorical_crossentropy(reduction='none') may not exist,
        # so we implement manually via log_softmax + gather.
        log_probs = logits_flat.log_softmax(axis=-1)         # (B*N_MAX, 100)
        # gather the log-prob at the gold label for each position
        gold_oh = gold_flat.one_hot(100).cast(log_probs.dtype)  # (B*N_MAX, 100)
        nll_per_pos = -(log_probs * gold_oh).sum(axis=-1)    # (B*N_MAX,) — NLL per position

        # Apply unobserved mask
        unobs_flat = unobs_mask.reshape(B * n_max)            # (B*N_MAX,)
        masked_nll = nll_per_pos * unobs_flat.cast(nll_per_pos.dtype)
        n_unobs = unobs_flat.sum() + 1e-8                     # avoid divide-by-zero
        ce_k = masked_nll.sum() / n_unobs

        per_breath_ce_list.append(ce_k)
        var_loss_sum = var_loss_sum + ce_k * weight_k
        weight_sum += weight_k

    var_loss = var_loss_sum / float(weight_sum)

    # Constraint energy on all breaths (weighted by progress)
    energy_sum = Tensor.zeros((), dtype=dtypes.float).contiguous()
    for k, logits in enumerate(var_logits_history):
        weight_e = float(k + 1) / float(K)
        e_k = factor_graph_constraint_energy(
            logits, factor_types_t, factor_args_t, n_max=n_max, f_max=f_max,
        ).mean()
        energy_sum = energy_sum + weight_e * e_k
    energy = energy_sum / float(K)

    # Calibration loss (same formulation as v98)
    final_argmax = var_logits_history[-1].argmax(axis=-1).detach()   # (B, N_MAX)
    # Correct: all unobserved positions match gold (ignore observed)
    eq = (final_argmax == gold_values.cast(dtypes.int)).cast(dtypes.float) * unobs_mask
    # Fraction of unobserved positions correct per sample
    n_unobs_per = unobs_mask.sum(axis=-1) + 1e-8                     # (B,)
    correct = (eq.sum(axis=-1) / n_unobs_per)                        # (B,) ∈ [0,1]

    calib_loss_sum = Tensor.zeros((), dtype=dtypes.float).contiguous()
    for k, calib_k in enumerate(calib_history):
        progression = float(k) / float(max(K - 1, 1))
        target_k = 0.5 + (correct - 0.5) * progression
        calib_loss_sum = calib_loss_sum + ((calib_k - target_k) ** 2).mean()
    calib_loss = calib_loss_sum / float(K)

    total = var_loss + energy_weight * energy + calib_weight * calib_loss
    return total, {
        "var_ce": var_loss,
        "energy": energy,
        "calib": calib_loss,
        "per_breath_ce": per_breath_ce_list,
    }


# ---------------------------------------------------------------------------
# Per-breath CE (for logging, unweighted)
# ---------------------------------------------------------------------------

def per_breath_ce_fg(
    var_logits_history: list[Tensor],
    gold_values: Tensor,       # (B, N_MAX) int
    observed_mask: Tensor,     # (B, N_MAX) int
    n_max: int = V99_N_MAX,
) -> list[float]:
    """Unweighted per-breath CE on unobserved positions only."""
    B = int(var_logits_history[0].shape[0])
    gold_flat = gold_values.cast(dtypes.int).reshape(B * n_max)
    unobs_flat = (1 - observed_mask.cast(dtypes.float)).reshape(B * n_max)

    out = []
    for logits in var_logits_history:
        logits_flat = logits.reshape(B * n_max, 100)
        log_probs = logits_flat.log_softmax(axis=-1)
        gold_oh = gold_flat.one_hot(100).cast(log_probs.dtype)
        nll = -(log_probs * gold_oh).sum(axis=-1)
        masked = nll * unobs_flat.cast(nll.dtype)
        n_unobs = unobs_flat.sum() + 1e-8
        ce = masked.sum() / n_unobs
        out.append(float(ce.realize().numpy()))
    return out


# ---------------------------------------------------------------------------
# Accuracy
# ---------------------------------------------------------------------------

def factor_graph_accuracy(
    var_logits_final: Tensor,  # (B, N_MAX, 100)
    gold_values: Tensor,        # (B, N_MAX) int
    observed_mask: Tensor,      # (B, N_MAX) int
    query_idx: np.ndarray,      # (B,) int — which variable is the query
) -> dict:
    """Return accuracy stats: query_acc, unobserved_cell_acc, per-diff if available.

    query_acc         : fraction of samples where the queried variable is correct.
    unobserved_cell_acc: fraction of unobserved variable positions correct.
    """
    B = int(var_logits_final.shape[0])
    pred = var_logits_final.argmax(axis=-1)  # (B, N_MAX)
    correct = (pred == gold_values.cast(dtypes.int)).cast(dtypes.float)  # (B, N_MAX)
    unobs_mask = (1 - observed_mask.cast(dtypes.float))   # (B, N_MAX)

    # Unobserved cell accuracy
    masked = correct * unobs_mask
    n_unobs = unobs_mask.sum() + 1e-8
    cell_acc = float((masked.sum() / n_unobs).realize().numpy())

    # Query accuracy: for each sample b, check correct[b, query_idx[b]]
    pred_np = pred.cast(dtypes.int).realize().numpy()   # (B, N_MAX)
    gold_np = gold_values.cast(dtypes.int).realize().numpy()
    q_correct = np.array([
        int(pred_np[b, query_idx[b]] == gold_np[b, query_idx[b]])
        for b in range(B)
    ], dtype=np.float32)
    query_acc = float(q_correct.mean())

    return {"cell_acc": cell_acc, "query_acc": query_acc}


# ---------------------------------------------------------------------------
# Model param attachment
# ---------------------------------------------------------------------------

def attach_fg_params(model: Any, hidden: int, n_heads: int,
                     k_max: int | None = None,
                     n_max: int = V99_N_MAX,
                     f_max: int = V99_F_MAX) -> None:
    """Allocate factor-graph-specific params on `model`.

    Attributes added:
      fg_domain_codebook  (100, hidden)   — maps domain distribution → hidden
      fg_var_pos_embed    (N_MAX, hidden) — variable position embeddings
      fg_factor_pos_embed (F_MAX, hidden) — factor position embeddings
      fg_node_kind_embed  (3, hidden)     — observed / unobserved / factor
      fg_op_embed         (4, hidden)     — op-type additive K/V bias
      fg_breath_embed     (K_max, hidden) — per-breath additive, QR-ortho
      fg_delta_gate       (K_max,)        — learnable per-breath delta scale
      fg_calib_head_w     (hidden, 1)
      fg_calib_head_b     (1,)

    All fp32. Cast to fp16 inside the forward (same pattern as v98 sudoku).
    k_max defaults to V99_K_MAX if not supplied.
    """
    if k_max is None:
        k_max = V99_K_MAX

    rng_cb = np.random.RandomState(9901)

    # Domain codebook: (100, hidden) — orthonormal rows at scale 0.1.
    # 100 rows for values 0..99; orthogonal init ensures no single value
    # dominates at random init.
    raw_cb = rng_cb.randn(max(hidden, 100), hidden).astype(np.float32)
    q_cb, _ = np.linalg.qr(raw_cb)
    cb_unit = q_cb[:100].astype(np.float32)
    model.fg_domain_codebook = Tensor(cb_unit * 0.1, dtype=dtypes.float).contiguous()

    # Variable position embeddings: (N_MAX, hidden) — randn 0.02
    vp = rng_cb.randn(n_max, hidden).astype(np.float32) * 0.02
    model.fg_var_pos_embed = Tensor(vp, dtype=dtypes.float).contiguous()

    # Factor position embeddings: (F_MAX, hidden) — randn 0.02
    fp_emb = rng_cb.randn(f_max, hidden).astype(np.float32) * 0.02
    model.fg_factor_pos_embed = Tensor(fp_emb, dtype=dtypes.float).contiguous()

    # Node-kind embedding: (3, hidden) — randn 0.02
    nk = rng_cb.randn(3, hidden).astype(np.float32) * 0.02
    model.fg_node_kind_embed = Tensor(nk, dtype=dtypes.float).contiguous()

    # Op-type embedding: (5, hidden) — row 0 is zero sentinel (variable positions),
    # rows 1..4 are add/sub/mul/div embeddings. Zero sentinel is fixed zero so
    # variable positions always get zero kv_bias. Op rows are randn 0.02 init.
    op_emb = np.zeros((5, hidden), dtype=np.float32)
    op_emb[1:] = rng_cb.randn(4, hidden).astype(np.float32) * 0.02
    model.fg_op_embed = Tensor(op_emb, dtype=dtypes.float).contiguous()

    # Calibration head
    cw = (rng_cb.randn(hidden, 1) * 0.02).astype(np.float32)
    model.fg_calib_head_w = Tensor(cw, dtype=dtypes.float).contiguous()
    model.fg_calib_head_b = Tensor.zeros((1,), dtype=dtypes.float).contiguous()

    # Per-breath embedding: QR-orthonormal, scale 0.5
    breath_scale = float(os.environ.get("V99_BREATH_EMBED_SCALE", "0.5"))
    rng_be = np.random.RandomState(9905)
    raw_be = rng_be.randn(max(k_max, hidden), hidden).astype(np.float32)
    q_be, _ = np.linalg.qr(raw_be)
    be = q_be[:k_max].astype(np.float32) * breath_scale
    model.fg_breath_embed = Tensor(be, dtype=dtypes.float).contiguous()

    # Per-breath delta gate: init 1.0
    model.fg_delta_gate = Tensor.ones((k_max,), dtype=dtypes.float).contiguous()


def fg_parameters(model: Any) -> list[Tensor]:
    """Trainable factor-graph-specific params."""
    return [
        model.fg_domain_codebook,
        model.fg_var_pos_embed,
        model.fg_factor_pos_embed,
        model.fg_node_kind_embed,
        model.fg_op_embed,
        model.fg_calib_head_w,
        model.fg_calib_head_b,
        model.fg_breath_embed,
        model.fg_delta_gate,
    ]


def fg_state_dict(model: Any) -> dict[str, Tensor]:
    """State dict entries for factor-graph params."""
    return {
        "fg.domain_codebook":  model.fg_domain_codebook,
        "fg.var_pos_embed":    model.fg_var_pos_embed,
        "fg.factor_pos_embed": model.fg_factor_pos_embed,
        "fg.node_kind_embed":  model.fg_node_kind_embed,
        "fg.op_embed":         model.fg_op_embed,
        "fg.calib_head_w":     model.fg_calib_head_w,
        "fg.calib_head_b":     model.fg_calib_head_b,
        "fg.breath_embed":     model.fg_breath_embed,
        "fg.delta_gate":       model.fg_delta_gate,
    }


# ---------------------------------------------------------------------------
# JIT training step (analog to _compile_jit_sudoku_step)
# ---------------------------------------------------------------------------

_JIT_FG_CACHE: dict = {}


def _compile_jit_fg_step(
    model: Any,
    opt: Any,
    K: int,
    B: int,
    energy_weight: float,
    calib_weight: float,
    n_max: int = V99_N_MAX,
    f_max: int = V99_F_MAX,
    grad_clip: float = 0.0,
):
    """Compile and return a TinyJit'd train step for the factor-graph forward.

    Inputs (stable shapes; pass realized Tensors):
      domain_init   : (B, N_MAX, 100) float — initial domain distributions
      node_kinds    : (B, T_MAX) int
      attn_bias     : (B, T_MAX, T_MAX) float
      kv_bias       : (B, T_MAX, H) float
      gold_values   : (B, N_MAX) int
      observed_mask : (B, N_MAX) int
      factor_types  : (B, F_MAX) int
      factor_args   : (B, F_MAX, 3) int

    Returns (realized scalars):
      total, healthy, var_ce, energy, calib, query_acc, cell_acc, *pb_ce

    The constraint energy function loops in Python over F_MAX * B (56 iters
    at most) so it IS JIT-compilable as long as B and F_MAX are constant.

    Note: because factor_graph_constraint_energy uses .numpy() internally,
    it CANNOT be inside a TinyJit graph. We split the step into two parts:
      1. JIT'd forward + CE-only loss
      2. Python-level energy computation (not JIT'd)
    This matches the AMD JIT safety requirement (no .numpy() inside JIT).
    """
    key = (id(model), id(opt), int(K), int(B), float(energy_weight),
           float(calib_weight), int(n_max), int(f_max), float(grad_clip))
    if key in _JIT_FG_CACHE:
        return _JIT_FG_CACHE[key]

    ew = float(energy_weight)
    aw = float(calib_weight)
    gc_val = float(grad_clip)
    params = opt.params
    _t0 = _time.perf_counter()
    print(f"[JIT] compile fg step: K={K} B={B} ew={ew} aw={aw}...", flush=True)

    @TinyJit
    def _step(
        domain_init: Tensor,    # (B, N_MAX, 100) fp32
        node_kinds: Tensor,      # (B, T_MAX) int
        attn_bias: Tensor,       # (B, T_MAX, T_MAX) fp32
        gold_values: Tensor,     # (B, N_MAX) int
        observed_mask: Tensor,   # (B, N_MAX) int
        factor_types_t: Tensor,  # (B, F_MAX) int
        factor_args_t: Tensor,   # (B, F_MAX, 3) int — kept for shape stability (not used in JIT)
    ):
        opt.zero_grad()

        var_logits_history, calib_history = factor_graph_breathing_forward(
            model, domain_init, node_kinds, attn_bias, factor_types_t, K=K,
            n_max=n_max, f_max=f_max,
        )

        # CE loss on unobserved positions
        gold_flat = gold_values.cast(dtypes.int).reshape(B * n_max)
        unobs_mask_float = (1 - observed_mask.cast(dtypes.float)).reshape(B * n_max)
        n_unobs = unobs_mask_float.sum() + 1e-8

        var_loss_sum = Tensor.zeros((), dtype=dtypes.float).contiguous()
        weight_sum = 0.0
        per_breath_ce_tensors = []
        for k, logits in enumerate(var_logits_history):
            weight_k = 1.0 + float(k) / float(max(K - 1, 1))
            logits_flat = logits.reshape(B * n_max, 100)
            log_probs = logits_flat.log_softmax(axis=-1)
            gold_oh = gold_flat.one_hot(100).cast(log_probs.dtype)
            nll = -(log_probs * gold_oh).sum(axis=-1)
            masked_nll = nll * unobs_mask_float.cast(nll.dtype)
            ce_k = masked_nll.sum() / n_unobs
            per_breath_ce_tensors.append(ce_k)
            var_loss_sum = var_loss_sum + ce_k * weight_k
            weight_sum += weight_k
        var_loss = var_loss_sum / float(weight_sum)

        # Calibration
        final_argmax = var_logits_history[-1].argmax(axis=-1).detach()
        eq = (final_argmax == gold_values.cast(dtypes.int)).cast(dtypes.float)
        unobs_2d = (1 - observed_mask.cast(dtypes.float))
        eq_unobs = eq * unobs_2d
        n_unobs_per = unobs_2d.sum(axis=-1) + 1e-8
        correct = eq_unobs.sum(axis=-1) / n_unobs_per

        calib_loss_sum = Tensor.zeros((), dtype=dtypes.float).contiguous()
        for k, calib_k in enumerate(calib_history):
            progression = float(k) / float(max(K - 1, 1))
            target_k = 0.5 + (correct - 0.5) * progression
            calib_loss_sum = calib_loss_sum + ((calib_k - target_k) ** 2).mean()
        calib_loss = calib_loss_sum / float(K)

        # Train accuracy metrics (detached)
        cell_acc = (eq_unobs.sum() / (unobs_2d.sum() + 1e-8)).detach()
        query_acc = correct.mean().detach()

        # Total: CE + calibration (energy is handled outside JIT)
        total_no_energy = var_loss + aw * calib_loss
        total_no_energy.backward()

        healthy = total_no_energy.isfinite().cast(dtypes.float)
        for p in params:
            if p.grad is not None:
                p.grad = p.grad * healthy.cast(p.grad.dtype)

        if gc_val > 0:
            sq_sum = Tensor.zeros((), dtype=dtypes.float).contiguous()
            for p in params:
                if p.grad is not None:
                    sq_sum = sq_sum + p.grad.cast(dtypes.float).square().sum()
            grad_norm = (sq_sum + 1e-12).sqrt()
            clip_coef = (Tensor(gc_val, dtype=dtypes.float) / (grad_norm + 1e-6)).minimum(
                Tensor(1.0, dtype=dtypes.float)
            )
            for p in params:
                if p.grad is not None:
                    p.grad = p.grad * clip_coef.cast(p.grad.dtype)

        opt.step()

        return (
            total_no_energy.realize(),
            healthy.realize(),
            var_loss.realize(),
            calib_loss.realize(),
            cell_acc.realize(),
            query_acc.realize(),
            *(ce.realize() for ce in per_breath_ce_tensors),
        )

    _JIT_FG_CACHE[key] = _step
    print(f"[JIT] fg step ready (cache size={len(_JIT_FG_CACHE)}); first call compiles (~60-90s)...",
          flush=True)
    return _step


def _compile_jit_fg_eval(model: Any, K: int, B: int,
                          n_max: int = V99_N_MAX, f_max: int = V99_F_MAX):
    """Compile a TinyJit'd eval step (forward-only, no backward)."""
    key = ("eval", id(model), int(K), int(B), int(n_max), int(f_max))
    if key in _JIT_FG_CACHE:
        return _JIT_FG_CACHE[key]

    print(f"[JIT] compile fg eval: K={K} B={B}...", flush=True)

    @TinyJit
    def _eval(
        domain_init: Tensor,
        node_kinds: Tensor,
        attn_bias: Tensor,
        factor_types_t: Tensor,
        gold_values: Tensor,
        observed_mask: Tensor,
    ):
        var_logits_history, _ = factor_graph_breathing_forward(
            model, domain_init, node_kinds, attn_bias, factor_types_t, K=K,
            n_max=n_max, f_max=f_max,
        )
        final_logits = var_logits_history[-1]   # (B, N_MAX, 100)
        pred = final_logits.argmax(axis=-1)     # (B, N_MAX)
        eq = (pred == gold_values.cast(dtypes.int)).cast(dtypes.float)
        unobs = (1 - observed_mask.cast(dtypes.float))
        cell_acc = (eq * unobs).sum() / (unobs.sum() + 1e-8)
        return pred.realize(), cell_acc.realize()

    _JIT_FG_CACHE[key] = _eval
    print(f"[JIT] fg eval ready (cache size={len(_JIT_FG_CACHE)})", flush=True)
    return _eval
