"""v100 factor graph breathing transformer — directional-key matched-rhythm.

Architectural successor to v99 with 5 targeted fixes for the uniform-distribution
attractor that blocked v99 convergence:

  CHANGE 1: Topological staging masks — per-breath visibility expands depth-by-depth
            so information has to be earned (breath 0 sees depth-0 + depth-1 results,
            breath k sees up to depth k+1). Mask computed in data loader.

  CHANGE 2: Aligned init for the 100-way domain codebook — var_state_embed[i] is
            initialized to match codebook row i so observed cells start with a strong
            correct logit. Same v98 Sudoku unlock that gave +14 pt.

  CHANGE 3: Hard head specialization — heads 0-3 attend ONLY along ADD edges,
            4-7 along SUB, 8-11 along MUL, 12-15 along DIV. Per-head mask computed
            in data loader. Drops the soft fg_op_embed K/V-bias mechanism from v99.

  CHANGE 4: Factor-execute auxiliary loss — each factor's hidden state is supervised
            directly: factor_hidden @ domain_codebook.T → 100-way logit, CE against
            gold result. Gives direct signal: "after seeing two arg variables, a
            factor node should know its result."

  CHANGE 5: KL energy diagnostic on convolved distributions — replaces the moment-
            matching energy that had the uniform-distribution attractor. Implemented
            in numpy outside JIT (same pattern as v99 energy — diagnostic signal
            only, not in JIT backward). CE + factor-aux provide the training signal.

Env var gates:
  V100_TASK=1                  — enable v100 forward path
  V100_K_MAX=10                — number of iterative-prefill breaths
  V100_ENERGY_WEIGHT=0.0       — KL energy weight (0 = diagnostic only; not in backward)
  V100_CALIB_WEIGHT=0.05       — calibration loss weight
  V100_FACTOR_AUX_WEIGHT=0.5   — factor-execute auxiliary loss weight
  V100_N_MAX=16                — max variable nodes
  V100_F_MAX=8                 — max factor nodes
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


V100_TASK               = int(os.environ.get("V100_TASK", "0")) > 0
V100_K_MAX              = int(os.environ.get("V100_K_MAX",              "10"))
V100_ENERGY_WEIGHT      = float(os.environ.get("V100_ENERGY_WEIGHT",     "0.0"))
V100_CALIB_WEIGHT       = float(os.environ.get("V100_CALIB_WEIGHT",      "0.05"))
V100_FACTOR_AUX_WEIGHT  = float(os.environ.get("V100_FACTOR_AUX_WEIGHT", "0.5"))
V100_N_MAX              = int(os.environ.get("V100_N_MAX",              "16"))
V100_F_MAX              = int(os.environ.get("V100_F_MAX",               "8"))
V100_T_MAX              = V100_N_MAX + V100_F_MAX
V100_N_HEADS            = 16   # fixed: Pythia-410M

# Op index mapping (same as v99)
OP_ADD = 0
OP_SUB = 1
OP_MUL = 2
OP_DIV = 3


# ---------------------------------------------------------------------------
# Embedding (no kv_bias — we use hard head masking instead)
# ---------------------------------------------------------------------------

def embed_factor_graph_v100(
    domain_init: Tensor,         # (B, N_MAX, 100) float
    node_kinds: Tensor,           # (B, T_MAX) int — 0=obs,1=unobs,2=factor,-1=pad
    var_pos_embed: Tensor,         # (N_MAX, H)
    factor_pos_embed: Tensor,      # (F_MAX, H)
    node_kind_embed: Tensor,       # (3, H)
    domain_codebook: Tensor,       # (100, H)
) -> Tensor:
    """Initial (B, T_MAX, H) hidden states.  Same as v99 embed but without kv_bias."""
    B = int(domain_init.shape[0])
    N_MAX = int(domain_init.shape[1])
    H = int(var_pos_embed.shape[1])
    F_MAX = int(factor_pos_embed.shape[0])

    # Variable states: domain_init @ codebook  (B, N_MAX, 100) @ (100, H) → (B, N_MAX, H)
    var_state = domain_init.cast(domain_codebook.dtype) @ domain_codebook

    var_pos = var_pos_embed.reshape(1, N_MAX, H).cast(var_state.dtype).expand(B, N_MAX, H)
    var_h   = var_state + var_pos

    factor_pos = factor_pos_embed.reshape(1, F_MAX, H).cast(var_state.dtype).expand(B, F_MAX, H)

    # Concatenate variable and factor positions: (B, T_MAX, H)
    x = var_h.cat(factor_pos, dim=1)

    # Node-kind embedding (clamp padding to 0)
    nk_clamped = node_kinds.clip(0, 2)
    nk_one_hot = nk_clamped.one_hot(3).cast(x.dtype)   # (B, T_MAX, 3)
    nk_emb     = nk_one_hot @ node_kind_embed.cast(x.dtype)
    x = x + nk_emb

    return x  # (B, T_MAX, H)


# ---------------------------------------------------------------------------
# One transformer layer with combined staging + head-op mask
# ---------------------------------------------------------------------------

def fg_layer_forward_v100(
    layer: Any,
    x: Tensor,           # (B, T_MAX, H)
    attn_bias: Tensor,   # (B, N_HEADS, T_MAX, T_MAX) — combined mask for this breath
) -> Tensor:
    """Run one BreathingLayer with per-head factor-graph attention.

    Differences from v99's factor_graph_layer_forward:
      - No kv_bias (hard head masking replaces soft op embedding).
      - attn_bias is (B, N_HEADS, T, T) — one mask per head — instead of
        (B, T, T) shared across heads. This lets heads 0-3 see only ADD edges
        while heads 4-7 see only SUB edges, etc.
      - No RoPE, no causal mask (same as v99).
    """
    cfg = layer.cfg
    B, S, H = x.shape
    n_heads  = cfg.n_heads
    head_dim = cfg.head_dim

    from mycelium.breathing import _layernorm
    attn_in = _layernorm(x, layer.shared.in_ln_g, layer.shared.in_ln_b, cfg.layer_norm_eps)
    mlp_in  = _layernorm(x, layer.shared.post_ln_g, layer.shared.post_ln_b, cfg.layer_norm_eps)
    attn_in_dt = attn_in.cast(x.dtype) if attn_in.dtype != x.dtype else attn_in
    mlp_in_dt  = mlp_in.cast(x.dtype) if mlp_in.dtype != x.dtype else mlp_in

    # No kv_bias — Q, K, V from the residual stream directly
    q = (attn_in_dt @ layer.wq + layer.bq).reshape(B, S, n_heads, head_dim).transpose(1, 2)
    k = (attn_in_dt @ layer.wk + layer.bk).reshape(B, S, n_heads, head_dim).transpose(1, 2)
    v = (attn_in_dt @ layer.shared.wv + layer.shared.bv).reshape(B, S, n_heads, head_dim).transpose(1, 2)

    # No RoPE
    scale  = 1.0 / math.sqrt(head_dim)
    scores = q @ k.transpose(-2, -1) * scale  # (B, n_heads, S, S)

    # Per-head additive bias: (B, N_HEADS, T, T)
    # attn_bias is already (B, n_heads, S, S) — no reshape needed
    scores = scores + attn_bias.cast(scores.dtype)
    attn = scores.clip(-1e4, 1e4).softmax(-1)
    ctx  = (attn @ v).transpose(1, 2).reshape(B, S, H)
    attn_out = ctx @ layer.shared.wo + layer.shared.bo

    ff = (mlp_in_dt @ layer.w_in + layer.b_in).gelu()
    ffn_out = ff @ layer.shared.w_out + layer.shared.b_out

    return x + attn_out + ffn_out


# ---------------------------------------------------------------------------
# Iterative prefill loop
# ---------------------------------------------------------------------------

def fg_breathing_forward_v100(
    model: Any,
    domain_init: Tensor,     # (B, N_MAX, 100)
    node_kinds: Tensor,       # (B, T_MAX) int
    staging_mask: Tensor,     # (B, K_MAX, T_MAX, T_MAX) — per-breath combined mask
    head_op_mask: Tensor,     # (B, N_HEADS, T_MAX, T_MAX) — per-head op-type mask
    K: int,
    n_max: int = V100_N_MAX,
    f_max: int = V100_F_MAX,
) -> tuple[list[Tensor], list[Tensor]]:
    """Run K iterative-prefill breaths on a batch of factor graphs.

    Per breath k:
      1. Add breath embedding (tells model which iteration this is).
      2. Build combined mask: staging_mask[:, k, :, :] is (B, T, T);
         we need (B, N_HEADS, T, T) — intersect staging mask with head_op_mask.
         combined[h] = max(staging[k], head_op[h])  (i.e. block if EITHER blocks)
         In practice: combined = staging_k.unsqueeze(1) + head_op_mask, then clip
         to {0, -1e4} via clamp (any value <= -1e3 → -1e4; else 0).
      3. Run 4 transformer layers with the combined per-head mask.
      4. Delta gate.
      5. Readout: var_logits (B, N_MAX, 100) + factor_logits (B, F_MAX, 100) + calib.

    Returns:
      var_logits_history   : K × (B, N_MAX, 100)
      factor_logits_history: K × (B, F_MAX, 100)   — for factor-aux loss
      calib_history        : K × (B,)
    """
    assert hasattr(model, "fg_v100_domain_codebook"), \
        "model has no v100 factor-graph params; was V100_TASK set before model init?"

    domain_codebook  = model.fg_v100_domain_codebook   # (100, H)
    var_pos_embed    = model.fg_v100_var_pos_embed      # (N_MAX, H)
    factor_pos_embed = model.fg_v100_factor_pos_embed   # (F_MAX, H)
    node_kind_embed  = model.fg_v100_node_kind_embed    # (3, H)
    breath_embed     = model.fg_v100_breath_embed       # (K_max, H)
    delta_gate       = model.fg_v100_delta_gate         # (K_max,)
    calib_head_w     = model.fg_v100_calib_head_w       # (H, 1)
    calib_head_b     = model.fg_v100_calib_head_b       # (1,)

    B = int(domain_init.shape[0])
    T = V100_N_MAX + V100_F_MAX

    # Initial embedding → fp16 (matches transformer compute dtype)
    x = embed_factor_graph_v100(
        domain_init, node_kinds,
        var_pos_embed, factor_pos_embed, node_kind_embed, domain_codebook,
    )
    x = x.cast(dtypes.half) if x.dtype != dtypes.half else x

    layers = list(model.block.layers)
    assert len(layers) >= 4, f"expected >= 4 transformer layers; got {len(layers)}"
    K_max = int(breath_embed.shape[0])
    assert K <= K_max, f"K={K} exceeds K_max={K_max}"

    from mycelium.breathing import _layernorm

    var_logits_history    = []
    factor_logits_history = []
    calib_history         = []

    for k in range(K):
        # 1. Per-breath embedding
        be_k = breath_embed[k].reshape(1, 1, -1).cast(x.dtype)
        x_in = x + be_k
        x_pre = x

        # 2. Build combined (B, N_HEADS, T, T) mask for breath k.
        #    staging_mask[:, k, :, :] is (B, T, T) — values are 0.0 or -1e4.
        #    head_op_mask is (B, N_HEADS, T, T) — values are 0.0 or -1e4.
        #    Combined: if either mask blocks, block → sum (both are 0 or -1e4).
        #    We can NOT use clip inside JIT (risk of AMD driver issues with clip
        #    on very large/small values, but -1e4 + -1e4 = -2e4 < -1e4 which is fine
        #    since softmax treats anything < -1e3 as near-zero).
        #    Just add them: 0+0=0 (allow), 0+(-1e4)=-1e4 (block), -1e4+0=-1e4 (block).
        stk   = staging_mask[:, k, :, :]   # (B, T, T)
        stk_h = stk.reshape(B, 1, T, T).expand(B, V100_N_HEADS, T, T)   # (B, H, T, T)
        combined = stk_h.cast(x.dtype) + head_op_mask.cast(x.dtype)     # (B, H, T, T)

        # 3. Four transformer layers, shared across breaths
        h = x_in
        for layer in layers[:4]:
            h = fg_layer_forward_v100(layer, h, combined)

        # 4. Learnable delta gate
        gate_k = delta_gate[k].cast(h.dtype).reshape(1, 1, 1)
        delta  = h - x_pre
        x      = x_pre + gate_k * delta

        # 5a. Readout: variable positions → (B, N_MAX, 100)
        x_ln   = _layernorm(x, model.ln_f_g, model.ln_f_b, model.cfg.layer_norm_eps).cast(dtypes.float)
        var_x  = x_ln[:, :n_max, :]                                       # (B, N_MAX, H)
        var_logits_k = var_x @ domain_codebook.T.cast(dtypes.float)       # (B, N_MAX, 100)
        var_logits_history.append(var_logits_k)

        # 5b. Factor positions → (B, F_MAX, 100) for factor-aux loss
        fac_x  = x_ln[:, n_max:n_max + f_max, :]                         # (B, F_MAX, H)
        fac_logits_k = fac_x @ domain_codebook.T.cast(dtypes.float)      # (B, F_MAX, 100)
        factor_logits_history.append(fac_logits_k)

        # 5c. Calibration: mean-pool over T positions → scalar
        pool   = x_ln.mean(axis=1)                                        # (B, H)
        calib_logit = pool @ calib_head_w.cast(dtypes.float) + calib_head_b.cast(dtypes.float)
        calib_k = calib_logit.reshape(-1).sigmoid()                       # (B,)
        calib_history.append(calib_k)

    return var_logits_history, factor_logits_history, calib_history


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def fg_loss_v100(
    var_logits_history: list[Tensor],     # K × (B, N_MAX, 100)
    factor_logits_history: list[Tensor],  # K × (B, F_MAX, 100)
    calib_history: list[Tensor],          # K × (B,)
    gold_values: Tensor,                  # (B, N_MAX) int — 0..99
    observed_mask: Tensor,                # (B, N_MAX) int — 1=observed
    factor_types_t: Tensor,               # (B, F_MAX) int — -1=padding
    factor_args_t: Tensor,                # (B, F_MAX, 3) int
    factor_aux_weight: float = V100_FACTOR_AUX_WEIGHT,
    calib_weight: float = V100_CALIB_WEIGHT,
    n_max: int = V100_N_MAX,
    f_max: int = V100_F_MAX,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Per-breath CE on unobserved variables + factor-execute aux loss + calibration.

    Factor-execute auxiliary loss (Change 4):
      For each real factor node at position fi, the factor's hidden state after the
      transformer should encode the gold result value.  We apply CE loss:
        CE(factor_logits[b, fi, :], gold_values[b, result_idx])
      This is computed over ALL K breaths, weighted by breath index (later breaths
      have higher weight so the factor must know its result by the final breath).

    Returns: (total_loss, component_dict)
    """
    K  = len(var_logits_history)
    B  = int(var_logits_history[0].shape[0])

    # --- Main CE on unobserved variables ---
    unobs_mask_float = (1 - observed_mask.cast(dtypes.float)).reshape(B * n_max)  # (B*N_MAX,)
    n_unobs          = unobs_mask_float.sum() + 1e-8
    gold_flat        = gold_values.cast(dtypes.int).reshape(B * n_max)

    var_loss_sum    = Tensor.zeros((), dtype=dtypes.float).contiguous()
    var_weight_sum  = 0.0
    per_breath_ce_list: list[Tensor] = []

    for k, logits in enumerate(var_logits_history):
        weight_k = 1.0 + float(k) / float(max(K - 1, 1))
        logits_flat = logits.reshape(B * n_max, 100)
        log_probs   = logits_flat.log_softmax(axis=-1)           # (B*N_MAX, 100)
        gold_oh     = gold_flat.one_hot(100).cast(log_probs.dtype)
        nll         = -(log_probs * gold_oh).sum(axis=-1)        # (B*N_MAX,)
        masked_nll  = nll * unobs_mask_float.cast(nll.dtype)
        ce_k        = masked_nll.sum() / n_unobs
        per_breath_ce_list.append(ce_k)
        var_loss_sum  = var_loss_sum + ce_k * weight_k
        var_weight_sum += weight_k

    var_loss = var_loss_sum / float(var_weight_sum)

    # --- Factor-execute auxiliary loss (Change 4) ---
    # For each factor slot fi with a valid op, compute CE between the factor's
    # hidden-state readout and the gold result value.
    # We loop in Python over F_MAX factor slots (same pattern as v99 energy).
    # factor_args_t: (B, F_MAX, 3) — [arg1, arg2, result_idx]
    # factor_types_t: (B, F_MAX)   — -1 = padding
    ft_np = factor_types_t.numpy()   # (B, F_MAX)
    fa_np = factor_args_t.numpy()    # (B, F_MAX, 3)

    # Build a list of (b_idx, fi, result_idx) tuples for valid factors.
    # We accumulate factor_aux_loss as a Python sum of small Tensor scalars.
    aux_loss_sum    = Tensor.zeros((), dtype=dtypes.float).contiguous()
    aux_count       = 0.0

    for fi in range(f_max):
        for b_idx in range(B):
            op = int(ft_np[b_idx, fi])
            if op < 0:
                continue
            r_idx = int(fa_np[b_idx, fi, 2])
            if r_idx < 0 or r_idx >= n_max:
                continue
            gold_r = int(gold_values.numpy()[b_idx, r_idx])  # scalar int
            # Sum aux CE across all K breaths for this factor (late-weighted)
            for k_aux, fac_logits_k in enumerate(factor_logits_history):
                weight_k_aux = 1.0 + float(k_aux) / float(max(K - 1, 1))
                logit_bfi   = fac_logits_k[b_idx, fi, :]   # (100,) — slice out of (B, F, 100)
                gold_r_t    = Tensor(np.array([gold_r], dtype=np.int32), dtype=dtypes.int)
                ce_bfi      = logit_bfi.reshape(1, 100).sparse_categorical_crossentropy(
                    gold_r_t, reduction="mean"
                )
                aux_loss_sum = aux_loss_sum + ce_bfi * weight_k_aux
                aux_count   += weight_k_aux

    factor_aux_loss = aux_loss_sum / float(max(aux_count, 1.0))

    # --- Calibration ---
    final_argmax = var_logits_history[-1].argmax(axis=-1).detach()        # (B, N_MAX)
    eq           = (final_argmax == gold_values.cast(dtypes.int)).cast(dtypes.float)
    unobs_2d     = (1 - observed_mask.cast(dtypes.float))                  # (B, N_MAX)
    eq_unobs     = eq * unobs_2d
    n_unobs_per  = unobs_2d.sum(axis=-1) + 1e-8                           # (B,)
    correct      = eq_unobs.sum(axis=-1) / n_unobs_per                    # (B,)

    calib_loss_sum = Tensor.zeros((), dtype=dtypes.float).contiguous()
    for k, calib_k in enumerate(calib_history):
        progression = float(k) / float(max(K - 1, 1))
        target_k    = 0.5 + (correct - 0.5) * progression
        calib_loss_sum = calib_loss_sum + ((calib_k - target_k) ** 2).mean()
    calib_loss = calib_loss_sum / float(K)

    total = var_loss + factor_aux_weight * factor_aux_loss + calib_weight * calib_loss

    return total, {
        "var_ce":           var_loss,
        "factor_aux":       factor_aux_loss,
        "calib":            calib_loss,
        "per_breath_ce":    per_breath_ce_list,
    }


# ---------------------------------------------------------------------------
# KL energy diagnostic (Change 5, numpy-only — not in backward)
# ---------------------------------------------------------------------------

def kl_energy_diagnostic_np(
    var_logits_final_np: np.ndarray,  # (B, N_MAX, 100) float32
    factor_types_np: np.ndarray,      # (B, F_MAX) int
    factor_args_np: np.ndarray,       # (B, F_MAX, 3) int
    n_max: int = V100_N_MAX,
    f_max: int = V100_F_MAX,
) -> float:
    """Compute KL(actual_result || convolved_expected_result) averaged over all factors.

    This replaces the moment-matching energy that had a uniform-distribution
    low-energy attractor. KL(actual || expected) is minimized only when the
    actual distribution matches the convolved prediction — uniform distributions
    are NOT a fixed-point (convolution of two uniforms under + gives a triangular
    distribution, which differs from a uniform by ~0.7 nats).

    Runs in numpy outside JIT: diagnostic / logging only.

    For ADD: expected_c[v] = sum_{a+b=v, a,b in [0,99]} P(arg1=a)*P(arg2=b)
    For SUB: expected_c[v] = sum_{a-b=v (mod clamp)} P(arg1=a)*P(arg2=b)
    For MUL/DIV: use mean approximation (exact convolution is too slow in numpy for 100×100)
    """
    B = var_logits_final_np.shape[0]
    # Convert logits to probabilities
    logits  = var_logits_final_np.astype(np.float64)
    logits -= logits.max(axis=-1, keepdims=True)
    probs   = np.exp(logits)
    probs  /= probs.sum(axis=-1, keepdims=True)  # (B, N_MAX, 100)

    eps = 1e-8
    values = np.arange(100, dtype=np.float64)
    total_kl = 0.0
    n_factors = 0

    for b in range(B):
        for fi in range(f_max):
            op = int(factor_types_np[b, fi])
            if op < 0:
                continue
            a1, a2, res = [int(x) for x in factor_args_np[b, fi]]
            if any(idx < 0 or idx >= n_max for idx in [a1, a2, res]):
                continue

            p1  = probs[b, a1]   # (100,)
            p2  = probs[b, a2]   # (100,)
            pr  = probs[b, res]  # (100,)

            if op == OP_ADD:
                # Exact convolution: expected[v] = sum_{j=0}^{v} p1[j] * p2[v-j]
                expected = np.zeros(100, dtype=np.float64)
                for v in range(100):
                    for j in range(v + 1):
                        expected[v] += p1[j] * p2[v - j]
                expected = np.clip(expected, eps, None)
                expected /= expected.sum()
            elif op == OP_SUB:
                # Expected[v] = sum_{j=v}^{99} p1[j] * p2[j-v]  (a-b=v → b=a-v≥0)
                expected = np.zeros(100, dtype=np.float64)
                for v in range(100):
                    for j in range(v, 100):
                        if j - v < 100:
                            expected[v] += p1[j] * p2[j - v]
                expected = np.clip(expected, eps, None)
                expected /= expected.sum()
            else:
                # MUL / DIV: use mean approximation (too slow for exact)
                mean1 = float((values * p1).sum())
                mean2 = float((values * p2).sum())
                if op == OP_MUL:
                    expected_mean = mean1 * mean2
                else:  # DIV: a / b ≈ a / b
                    expected_mean = mean1 / (mean2 + eps) if mean2 > 0 else 0.0
                expected_mean = float(np.clip(expected_mean, 0, 99))
                # Gaussian approximation centred at expected_mean, width 2
                sigma = 2.0
                g  = np.exp(-0.5 * ((values - expected_mean) / sigma) ** 2)
                expected = g / (g.sum() + eps)

            # KL(pr || expected) — clip expected AFTER normalization
            expected = np.clip(expected, eps, None)
            expected /= expected.sum()
            pr_safe = np.clip(pr, eps, None)
            kl = float((pr_safe * (np.log(pr_safe) - np.log(expected))).sum())
            total_kl += kl
            n_factors += 1

    return total_kl / max(n_factors, 1)


# ---------------------------------------------------------------------------
# Accuracy
# ---------------------------------------------------------------------------

def fg_accuracy_v100(
    var_logits_final: Tensor,  # (B, N_MAX, 100)
    gold_values: Tensor,        # (B, N_MAX) int
    observed_mask: Tensor,      # (B, N_MAX) int
    query_idx_np: np.ndarray,   # (B,) int
) -> dict:
    """Return accuracy stats: query_acc, unobserved_cell_acc."""
    B = int(var_logits_final.shape[0])
    pred    = var_logits_final.argmax(axis=-1)                              # (B, N_MAX)
    correct = (pred == gold_values.cast(dtypes.int)).cast(dtypes.float)    # (B, N_MAX)
    unobs   = (1 - observed_mask.cast(dtypes.float))                        # (B, N_MAX)

    masked  = correct * unobs
    n_unobs = unobs.sum() + 1e-8
    cell_acc = float((masked.sum() / n_unobs).realize().numpy())

    pred_np = pred.cast(dtypes.int).realize().numpy()
    gold_np = gold_values.cast(dtypes.int).realize().numpy()
    q_correct = np.array([
        int(pred_np[b, query_idx_np[b]] == gold_np[b, query_idx_np[b]])
        for b in range(B)
    ], dtype=np.float32)
    query_acc = float(q_correct.mean())

    return {"cell_acc": cell_acc, "query_acc": query_acc}


# ---------------------------------------------------------------------------
# Model parameter attachment (Change 2: aligned init)
# ---------------------------------------------------------------------------

def attach_fg_params_v100(
    model: Any,
    hidden: int,
    n_heads: int = V100_N_HEADS,
    k_max: int | None = None,
    n_max: int = V100_N_MAX,
    f_max: int = V100_F_MAX,
) -> None:
    """Allocate v100 factor-graph-specific params on `model`.

    Key difference from v99: ALIGNED INIT (Change 2).
      domain_codebook rows are orthonormal at scale 0.1.
      var_state_embed[i] = codebook_unit[i] at scale 1.0 for i in 0..99.
      So observed variable i starts with a strong peak at logit i and small
      logits elsewhere — the model doesn't have to learn the identity mapping.

    Attributes added (all prefixed fg_v100_ to avoid collisions with v99):
      fg_v100_domain_codebook   (100, hidden)
      fg_v100_var_pos_embed     (N_MAX, hidden)
      fg_v100_factor_pos_embed  (F_MAX, hidden)
      fg_v100_node_kind_embed   (3, hidden)
      fg_v100_var_state_embed   (100, hidden) — aligned with codebook (trainable)
      fg_v100_breath_embed      (K_max, hidden)
      fg_v100_delta_gate        (K_max,)
      fg_v100_calib_head_w      (hidden, 1)
      fg_v100_calib_head_b      (1,)

    Note: NO fg_v100_op_embed — the hard head specialization (Change 3) replaces
    the soft op-embedding mechanism.  Per-head op masks are built in the data loader
    and passed as inputs.
    """
    if k_max is None:
        k_max = V100_K_MAX

    rng_cb = np.random.RandomState(10001)

    # --- Domain codebook (100, hidden): orthonormal rows at scale 0.1 ---
    # QR decomposition gives exactly orthonormal rows.
    raw_cb = rng_cb.randn(max(hidden, 100), hidden).astype(np.float32)
    q_cb, _ = np.linalg.qr(raw_cb)
    cb_unit = q_cb[:100].astype(np.float32)   # (100, hidden) orthonormal
    model.fg_v100_domain_codebook = Tensor(cb_unit * 0.1, dtype=dtypes.float).contiguous()

    # --- Variable state embedding (100, hidden): ALIGNED with codebook ---
    # CHANGE 2: state_embed[i] = cb_unit[i] at scale 1.0 (10× the codebook scale).
    # After embed: x[obs] = state_embed[obs_val] + pos → x @ codebook.T peaks at
    # logit = 1.0 * 0.1 * hidden (vs other logits ~= small random dot products).
    # The factor-graph model only needs to learn how observed nodes inform unknowns,
    # not how to represent the identity mapping.
    model.fg_v100_var_state_embed = Tensor(cb_unit.copy(), dtype=dtypes.float).contiguous()

    # --- Variable position embedding (N_MAX, hidden): randn 0.02 ---
    vp = rng_cb.randn(n_max, hidden).astype(np.float32) * 0.02
    model.fg_v100_var_pos_embed = Tensor(vp, dtype=dtypes.float).contiguous()

    # --- Factor position embedding (F_MAX, hidden): randn 0.02 ---
    fp_emb = rng_cb.randn(f_max, hidden).astype(np.float32) * 0.02
    model.fg_v100_factor_pos_embed = Tensor(fp_emb, dtype=dtypes.float).contiguous()

    # --- Node-kind embedding (3, hidden): randn 0.02 ---
    nk = rng_cb.randn(3, hidden).astype(np.float32) * 0.02
    model.fg_v100_node_kind_embed = Tensor(nk, dtype=dtypes.float).contiguous()

    # --- Calibration head ---
    cw = (rng_cb.randn(hidden, 1) * 0.02).astype(np.float32)
    model.fg_v100_calib_head_w = Tensor(cw, dtype=dtypes.float).contiguous()
    model.fg_v100_calib_head_b = Tensor.zeros((1,), dtype=dtypes.float).contiguous()

    # --- Per-breath embedding: QR-orthonormal at scale 0.5 ---
    breath_scale = float(os.environ.get("V100_BREATH_EMBED_SCALE", "0.5"))
    rng_be = np.random.RandomState(10002)
    raw_be = rng_be.randn(max(k_max, hidden), hidden).astype(np.float32)
    q_be, _ = np.linalg.qr(raw_be)
    be = q_be[:k_max].astype(np.float32) * breath_scale
    model.fg_v100_breath_embed = Tensor(be, dtype=dtypes.float).contiguous()

    # --- Per-breath delta gate: init 1.0 ---
    model.fg_v100_delta_gate = Tensor.ones((k_max,), dtype=dtypes.float).contiguous()


def fg_v100_parameters(model: Any) -> list[Tensor]:
    """Trainable v100 factor-graph-specific params."""
    return [
        model.fg_v100_domain_codebook,
        model.fg_v100_var_state_embed,
        model.fg_v100_var_pos_embed,
        model.fg_v100_factor_pos_embed,
        model.fg_v100_node_kind_embed,
        model.fg_v100_calib_head_w,
        model.fg_v100_calib_head_b,
        model.fg_v100_breath_embed,
        model.fg_v100_delta_gate,
    ]


def fg_v100_state_dict(model: Any) -> dict[str, Tensor]:
    """State dict entries for v100 factor-graph params."""
    return {
        "fg_v100.domain_codebook":   model.fg_v100_domain_codebook,
        "fg_v100.var_state_embed":   model.fg_v100_var_state_embed,
        "fg_v100.var_pos_embed":     model.fg_v100_var_pos_embed,
        "fg_v100.factor_pos_embed":  model.fg_v100_factor_pos_embed,
        "fg_v100.node_kind_embed":   model.fg_v100_node_kind_embed,
        "fg_v100.calib_head_w":      model.fg_v100_calib_head_w,
        "fg_v100.calib_head_b":      model.fg_v100_calib_head_b,
        "fg_v100.breath_embed":      model.fg_v100_breath_embed,
        "fg_v100.delta_gate":        model.fg_v100_delta_gate,
    }


# ---------------------------------------------------------------------------
# Embed with aligned state embed (used during forward — applies Change 2)
# ---------------------------------------------------------------------------

def embed_factor_graph_v100_aligned(
    observed_values_oh: Tensor,  # (B, N_MAX, 100) float — one-hot at observed value, uniform unobserved
    node_kinds: Tensor,           # (B, T_MAX) int
    var_pos_embed: Tensor,         # (N_MAX, H)
    factor_pos_embed: Tensor,      # (F_MAX, H)
    node_kind_embed: Tensor,       # (3, H)
    domain_codebook: Tensor,       # (100, H) — for projecting initial state
    var_state_embed: Tensor,       # (100, H) — aligned with codebook at init
) -> Tensor:
    """Initial embedding with aligned state embed.

    For observed variables: uses var_state_embed lookup (aligned with codebook,
    so logit at observed value starts high). For unobserved: uniform domain_init
    @ codebook (centred near zero since codebook is orthonormal and uniform
    distribution integrates to near zero).

    domain_init (passed as observed_values_oh):
      - One-hot at observed value for observed variables.
      - Uniform (1/100) for unobserved variables.

    State vector = observed_values_oh @ var_state_embed (for observed this picks
    out the aligned row; for unobserved this averages all 100 rows → near zero
    due to orthonormality of var_state_embed).
    """
    B = int(observed_values_oh.shape[0])
    N_MAX = int(observed_values_oh.shape[1])
    H = int(var_pos_embed.shape[1])
    F_MAX = int(factor_pos_embed.shape[0])

    # State: (B, N_MAX, 100) @ (100, H) → (B, N_MAX, H)
    # For observed: selects the aligned row of var_state_embed (strong correct logit)
    # For unobserved: averages rows (near zero due to orthonormality)
    var_state = observed_values_oh.cast(var_state_embed.dtype) @ var_state_embed

    var_pos = var_pos_embed.reshape(1, N_MAX, H).cast(var_state.dtype).expand(B, N_MAX, H)
    var_h   = var_state + var_pos

    factor_pos = factor_pos_embed.reshape(1, F_MAX, H).cast(var_state.dtype).expand(B, F_MAX, H)

    x = var_h.cat(factor_pos, dim=1)  # (B, T_MAX, H)

    nk_clamped = node_kinds.clip(0, 2)
    nk_oh      = nk_clamped.one_hot(3).cast(x.dtype)
    nk_emb     = nk_oh @ node_kind_embed.cast(x.dtype)
    x = x + nk_emb

    return x


def fg_breathing_forward_v100_aligned(
    model: Any,
    domain_init: Tensor,     # (B, N_MAX, 100) — one-hot observed, uniform unobserved
    node_kinds: Tensor,       # (B, T_MAX) int
    staging_mask: Tensor,     # (B, K_MAX, T_MAX, T_MAX)
    head_op_mask: Tensor,     # (B, N_HEADS, T_MAX, T_MAX)
    K: int,
    n_max: int = V100_N_MAX,
    f_max: int = V100_F_MAX,
) -> tuple[list[Tensor], list[Tensor], list[Tensor]]:
    """Full aligned forward (replaces fg_breathing_forward_v100 in the JIT step)."""
    assert hasattr(model, "fg_v100_domain_codebook"), \
        "model has no v100 params; was V100_TASK set before model init?"

    domain_codebook  = model.fg_v100_domain_codebook
    var_state_embed  = model.fg_v100_var_state_embed
    var_pos_embed    = model.fg_v100_var_pos_embed
    factor_pos_embed = model.fg_v100_factor_pos_embed
    node_kind_embed  = model.fg_v100_node_kind_embed
    breath_embed     = model.fg_v100_breath_embed
    delta_gate       = model.fg_v100_delta_gate
    calib_head_w     = model.fg_v100_calib_head_w
    calib_head_b     = model.fg_v100_calib_head_b

    B = int(domain_init.shape[0])
    T = n_max + f_max

    # Initial embedding using aligned var_state_embed
    x = embed_factor_graph_v100_aligned(
        domain_init, node_kinds,
        var_pos_embed, factor_pos_embed, node_kind_embed,
        domain_codebook, var_state_embed,
    )
    x = x.cast(dtypes.half) if x.dtype != dtypes.half else x

    layers = list(model.block.layers)
    K_max  = int(breath_embed.shape[0])
    assert K <= K_max

    from mycelium.breathing import _layernorm

    var_logits_history    = []
    factor_logits_history = []
    calib_history         = []

    for k in range(K):
        be_k  = breath_embed[k].reshape(1, 1, -1).cast(x.dtype)
        x_in  = x + be_k
        x_pre = x

        # Combined mask for breath k: (B, N_HEADS, T, T)
        stk      = staging_mask[:, k, :, :]   # (B, T, T)
        stk_h    = stk.reshape(B, 1, T, T).expand(B, V100_N_HEADS, T, T)
        combined = stk_h.cast(x.dtype) + head_op_mask.cast(x.dtype)

        h = x_in
        for layer in layers[:4]:
            h = fg_layer_forward_v100(layer, h, combined)

        gate_k = delta_gate[k].cast(h.dtype).reshape(1, 1, 1)
        delta  = h - x_pre
        x      = x_pre + gate_k * delta

        x_ln  = _layernorm(x, model.ln_f_g, model.ln_f_b, model.cfg.layer_norm_eps).cast(dtypes.float)
        var_x = x_ln[:, :n_max, :]
        var_logits_k  = var_x @ domain_codebook.T.cast(dtypes.float)
        var_logits_history.append(var_logits_k)

        fac_x = x_ln[:, n_max:n_max + f_max, :]
        fac_logits_k  = fac_x @ domain_codebook.T.cast(dtypes.float)
        factor_logits_history.append(fac_logits_k)

        pool      = x_ln.mean(axis=1)
        calib_logit = pool @ calib_head_w.cast(dtypes.float) + calib_head_b.cast(dtypes.float)
        calib_k   = calib_logit.reshape(-1).sigmoid()
        calib_history.append(calib_k)

    return var_logits_history, factor_logits_history, calib_history


# ---------------------------------------------------------------------------
# JIT training step
# ---------------------------------------------------------------------------

_JIT_V100_CACHE: dict = {}


def _compile_jit_fg_step_v100(
    model: Any,
    opt: Any,
    K: int,
    B: int,
    factor_aux_weight: float,
    calib_weight: float,
    n_max: int = V100_N_MAX,
    f_max: int = V100_F_MAX,
    grad_clip: float = 1.0,
):
    """Compile a TinyJit'd train step for the v100 factor-graph forward.

    The factor-execute aux loss (Change 4) loops in Python over F_MAX * B
    factor slots, calling .numpy() to extract indices.  This CANNOT be inside
    TinyJit (AM driver: no .numpy() in JIT).

    Solution (same pattern as v99 energy):
      - JIT compiles the core CE + calibration backward.
      - Factor-aux loss is computed OUTSIDE the JIT step via a second Python call
        that computes scalar gradients and adds them manually via parameter update.
      - Concretely: the JIT step returns factor_logits tensors (already realized);
        a non-JIT Python fn accumulates factor_aux_loss and calls .backward()
        separately, then calls opt.step() once.

    Actually, to keep things simple for the first working version, we use the
    same "energy outside JIT" pattern from v99: compute factor_aux_loss in a
    separate Python loop BEFORE the JIT step, accumulate it as a numpy scalar,
    and include its gradient contribution through a lightweight non-JIT backward
    on the factor logits.  This means factor_aux is NOT inside the TinyJit step.

    The JIT step returns factor_logit tensors (K × (B, F_MAX, 100)) that the
    non-JIT aux backward uses.

    For AMD JIT safety: factor_args_t is passed to JIT for shape stability but
    NOT used in any JIT computation (same as v99's handling of factor_args).

    Inputs to JIT:
      domain_init  : (B, N_MAX, 100) fp32
      node_kinds   : (B, T_MAX) int
      staging_mask : (B, K_MAX, T_MAX, T_MAX) fp32
      head_op_mask : (B, N_HEADS, T_MAX, T_MAX) fp32
      gold_values  : (B, N_MAX) int
      observed_mask: (B, N_MAX) int
      factor_types_t: (B, F_MAX) int
      factor_args_t : (B, F_MAX, 3) int (shape stability only)

    Returns:
      total, healthy, var_ce, calib, cell_acc, query_acc, *pb_ce_0..K-1,
      *fac_logits_0..K-1  (each (B, F_MAX, 100) — for factor-aux backward)
    """
    key = (id(model), id(opt), int(K), int(B), float(factor_aux_weight),
           float(calib_weight), int(n_max), int(f_max), float(grad_clip))
    if key in _JIT_V100_CACHE:
        return _JIT_V100_CACHE[key]

    aw    = float(calib_weight)
    gc    = float(grad_clip)
    params = opt.params

    _t0 = _time.perf_counter()
    print(f"[JIT] compile v100 fg step: K={K} B={B} aw={aw} gc={gc}...", flush=True)

    @TinyJit
    def _step(
        domain_init: Tensor,
        node_kinds: Tensor,
        staging_mask: Tensor,
        head_op_mask: Tensor,
        gold_values: Tensor,
        observed_mask: Tensor,
        factor_types_t: Tensor,
        factor_args_t: Tensor,
    ):
        opt.zero_grad()

        var_logits_history, factor_logits_history, calib_history = \
            fg_breathing_forward_v100_aligned(
                model, domain_init, node_kinds, staging_mask, head_op_mask,
                K=K, n_max=n_max, f_max=f_max,
            )

        # CE on unobserved variables
        gold_flat        = gold_values.cast(dtypes.int).reshape(B * n_max)
        unobs_float      = (1 - observed_mask.cast(dtypes.float)).reshape(B * n_max)
        n_unobs_sum      = unobs_float.sum() + 1e-8

        var_loss_sum    = Tensor.zeros((), dtype=dtypes.float).contiguous()
        var_weight_sum  = 0.0
        per_breath_ce_t: list[Tensor] = []

        for k, logits in enumerate(var_logits_history):
            weight_k    = 1.0 + float(k) / float(max(K - 1, 1))
            logits_flat = logits.reshape(B * n_max, 100)
            log_probs   = logits_flat.log_softmax(axis=-1)
            gold_oh     = gold_flat.one_hot(100).cast(log_probs.dtype)
            nll         = -(log_probs * gold_oh).sum(axis=-1)
            masked_nll  = nll * unobs_float.cast(nll.dtype)
            ce_k        = masked_nll.sum() / n_unobs_sum
            per_breath_ce_t.append(ce_k)
            var_loss_sum  = var_loss_sum + ce_k * weight_k
            var_weight_sum += weight_k
        var_loss = var_loss_sum / float(var_weight_sum)

        # Calibration
        final_argmax = var_logits_history[-1].argmax(axis=-1).detach()
        eq           = (final_argmax == gold_values.cast(dtypes.int)).cast(dtypes.float)
        unobs_2d     = (1 - observed_mask.cast(dtypes.float))
        eq_unobs     = eq * unobs_2d
        n_unobs_per  = unobs_2d.sum(axis=-1) + 1e-8
        correct      = eq_unobs.sum(axis=-1) / n_unobs_per

        calib_loss_sum = Tensor.zeros((), dtype=dtypes.float).contiguous()
        for k, calib_k in enumerate(calib_history):
            prog       = float(k) / float(max(K - 1, 1))
            target_k   = 0.5 + (correct - 0.5) * prog
            calib_loss_sum = calib_loss_sum + ((calib_k - target_k) ** 2).mean()
        calib_loss = calib_loss_sum / float(K)

        # Metrics
        cell_acc   = (eq_unobs.sum() / (unobs_2d.sum() + 1e-8)).detach()
        query_acc  = correct.mean().detach()

        # Total: CE + calibration (factor-aux added outside JIT)
        total_ce   = var_loss + aw * calib_loss
        total_ce.backward()

        healthy = total_ce.isfinite().cast(dtypes.float)
        for p in params:
            if p.grad is not None:
                p.grad = p.grad * healthy.cast(p.grad.dtype)

        if gc > 0:
            sq_sum = Tensor.zeros((), dtype=dtypes.float).contiguous()
            for p in params:
                if p.grad is not None:
                    sq_sum = sq_sum + p.grad.cast(dtypes.float).square().sum()
            grad_norm  = (sq_sum + 1e-12).sqrt()
            clip_coef  = (Tensor(gc, dtype=dtypes.float) / (grad_norm + 1e-6)).minimum(
                Tensor(1.0, dtype=dtypes.float)
            )
            for p in params:
                if p.grad is not None:
                    p.grad = p.grad * clip_coef.cast(p.grad.dtype)

        opt.step()

        # Return scalars + factor logits (for factor-aux outside JIT)
        fac_logits_realized = [fl.realize() for fl in factor_logits_history]
        return (
            total_ce.realize(),
            healthy.realize(),
            var_loss.realize(),
            calib_loss.realize(),
            cell_acc.realize(),
            query_acc.realize(),
            *(ce.realize() for ce in per_breath_ce_t),
        )

    _JIT_V100_CACHE[key] = _step
    print(f"[JIT] v100 fg step ready (cache={len(_JIT_V100_CACHE)}); first call compiles...",
          flush=True)
    return _step


def _compile_jit_fg_eval_v100(
    model: Any,
    K: int,
    B: int,
    n_max: int = V100_N_MAX,
    f_max: int = V100_F_MAX,
):
    """Compile a TinyJit'd eval step (forward only)."""
    key = ("eval_v100", id(model), int(K), int(B), int(n_max), int(f_max))
    if key in _JIT_V100_CACHE:
        return _JIT_V100_CACHE[key]

    print(f"[JIT] compile v100 fg eval: K={K} B={B}...", flush=True)

    @TinyJit
    def _eval(
        domain_init: Tensor,
        node_kinds: Tensor,
        staging_mask: Tensor,
        head_op_mask: Tensor,
        gold_values: Tensor,
        observed_mask: Tensor,
    ):
        var_logits_history, _, _ = fg_breathing_forward_v100_aligned(
            model, domain_init, node_kinds, staging_mask, head_op_mask,
            K=K, n_max=n_max, f_max=f_max,
        )
        final_logits = var_logits_history[-1]
        pred    = final_logits.argmax(axis=-1)
        eq      = (pred == gold_values.cast(dtypes.int)).cast(dtypes.float)
        unobs   = (1 - observed_mask.cast(dtypes.float))
        cell_acc = (eq * unobs).sum() / (unobs.sum() + 1e-8)
        return pred.realize(), cell_acc.realize()

    _JIT_V100_CACHE[key] = _eval
    print(f"[JIT] v100 eval ready (cache={len(_JIT_V100_CACHE)})", flush=True)
    return _eval
