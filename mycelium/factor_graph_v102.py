"""v102 factor graph breathing transformer — v100 + shared codebook compression.

Architectural change over v101:
  Per-breath waist compression is replaced with CODEBOOK MATCHING at the same
  insertion point (after 4 transformer layers, before delta_gate residual update).
  The key difference:

    v101 (projection-based, REPLACE THIS):
      waist_h  = h @ W_compress + b_compress     # (B, T, H) → (B, T, 512)
      waist_h  = waist_h.gelu()
      quantize = waist_h @ W_expand + b_expand   # (B, T, 512) → (B, T, H)
      h_quant  = h + quantize

    v102 (codebook matching):
      scores        = x @ codebook.T / temperature   # (B, T, N_CODE)
      weights       = scores.softmax(-1)              # (B, T, N_CODE)
      reconstruction = weights @ codebook             # (B, T, H) — in codebook span
      quantize      = reconstruction - x             # what codebook "thinks" to add
      h_quant       = x + delta_gate_quant[k] * quantize  # LoRA-style residual

  Why codebook matching generalizes better:
    - The reconstruction is bounded to the convex hull of N_CODE=32 shared primitives.
    - The same codebook is used for every problem, every position, every topology.
    - v101's W_compress could learn arbitrary topology-specific projections;
      v102's codebook cannot memorize per-topology schemes.
    - N_CODE=32 matches IB tree leaves (4 ops × ~8 sub-clusters).

  Initialization for EXACT warm-start preservation:
    codebook:         random orthonormal × 0.5  (matches sudoku_digit_codebook pattern)
    delta_gate_quant: all zeros                 (reconstruction multiplied by zero at init)
    temperature:      1.0 (learnable scalar)

    At init: h_quant = x + 0 * (reconstruction - x) = x — byte-identical to v100.
    Gradient flows immediately: dloss/d(delta_gate_quant) is non-zero because
    reconstruction is non-zero (codebook is random but non-zero; weights are non-zero).
    After 1-2 steps delta_gate_quant becomes non-zero and codebook starts contributing.

Env var gates:
  V102_TASK=1                  — enable v102 forward path
  V102_K_MAX=10                — number of iterative-prefill breaths
  V102_ENERGY_WEIGHT=0.0       — KL energy weight (0 = diagnostic only)
  V102_CALIB_WEIGHT=0.05       — calibration loss weight
  V102_FACTOR_AUX_WEIGHT=0.5   — factor-execute auxiliary loss weight
  V102_N_MAX=16                — max variable nodes
  V102_F_MAX=8                 — max factor nodes
  V102_CODEBOOK_N=32           — codebook size (default 32 = IB tree leaves)

All v100 code is imported and reused unchanged.  Only the forward loop and
codebook parameter attachment differ from v101.
"""
from __future__ import annotations

import os
import time as _time
from typing import Any

import numpy as np
from tinygrad import Tensor, dtypes
from tinygrad.engine.jit import TinyJit

from mycelium.config import Config

# Re-export everything from v100 that callers might need unchanged
from mycelium.factor_graph_v100 import (
    embed_factor_graph_v100,
    embed_factor_graph_v100_aligned,
    fg_layer_forward_v100,
    fg_loss_v100,
    kl_energy_diagnostic_np,
    fg_accuracy_v100,
    fg_v100_state_dict,
    V100_N_HEADS,
    OP_ADD, OP_SUB, OP_MUL, OP_DIV,
)


V102_TASK               = int(os.environ.get("V102_TASK", "0")) > 0
V102_K_MAX              = int(os.environ.get("V102_K_MAX",              "10"))
V102_ENERGY_WEIGHT      = float(os.environ.get("V102_ENERGY_WEIGHT",     "0.0"))
V102_CALIB_WEIGHT       = float(os.environ.get("V102_CALIB_WEIGHT",      "0.05"))
V102_FACTOR_AUX_WEIGHT  = float(os.environ.get("V102_FACTOR_AUX_WEIGHT", "0.5"))
V102_N_MAX              = int(os.environ.get("V102_N_MAX",              "16"))
V102_F_MAX              = int(os.environ.get("V102_F_MAX",               "8"))
V102_T_MAX              = V102_N_MAX + V102_F_MAX
V102_N_HEADS            = 16   # fixed: Pythia-410M
V102_KL_DIAG            = int(os.environ.get("V102_KL_DIAG", "0")) > 0
V102_CODEBOOK_N         = int(os.environ.get("V102_CODEBOOK_N",         "32"))


# ---------------------------------------------------------------------------
# Iterative prefill loop with shared codebook compression
# ---------------------------------------------------------------------------

def fg_breathing_forward_v102(
    model: Any,
    domain_init: Tensor,     # (B, N_MAX, 100) — one-hot observed, uniform unobserved
    node_kinds: Tensor,       # (B, T_MAX) int
    staging_mask: Tensor,     # (B, K_MAX, T_MAX, T_MAX)
    head_op_mask: Tensor,     # (B, N_HEADS, T_MAX, T_MAX)
    K: int,
    n_max: int = V102_N_MAX,
    f_max: int = V102_F_MAX,
) -> tuple[list[Tensor], list[Tensor], list[Tensor]]:
    """Run K iterative-prefill breaths on a batch of factor graphs.

    Per breath k:
      1. Add breath embedding.
      2. Build combined mask: staging_mask[:, k] intersected with head_op_mask.
      3. Run 4 transformer layers with the combined per-head mask.
      4. Per-breath codebook matching:
           scores   = x @ codebook.T / temperature    (B, T, N_CODE)
           weights  = scores.softmax(-1)               (B, T, N_CODE)
           recon    = weights @ codebook               (B, T, H)
           quantize = recon - x                        what codebook adds
           h_quant  = x + delta_gate_quant[k] * quantize   LoRA-style
      5. Delta gate residual update over h_quant.
      6. Readout: var_logits, factor_logits, calib.

    The codebook compression is topology-invariant: every problem matches its
    residuals against the SAME set of 32 shared primitives, so no per-topology
    memorization is possible.
    """
    assert hasattr(model, "fg_v100_domain_codebook"), \
        "model has no v100 params; was V102_TASK (and V100_TASK) set before model init?"
    assert hasattr(model, "fg_v102_codebook"), \
        "model has no v102 codebook params; was attach_fg_params_v102 called?"

    domain_codebook  = model.fg_v100_domain_codebook
    var_state_embed  = model.fg_v100_var_state_embed
    var_pos_embed    = model.fg_v100_var_pos_embed
    factor_pos_embed = model.fg_v100_factor_pos_embed
    node_kind_embed  = model.fg_v100_node_kind_embed
    breath_embed     = model.fg_v100_breath_embed
    delta_gate       = model.fg_v100_delta_gate
    calib_head_w     = model.fg_v100_calib_head_w
    calib_head_b     = model.fg_v100_calib_head_b

    # v102 codebook params
    codebook         = model.fg_v102_codebook          # (N_CODE, H)
    delta_gate_quant = model.fg_v102_delta_gate_quant  # (K_max,)
    temperature      = model.fg_v102_temperature       # () scalar

    B = int(domain_init.shape[0])
    T = n_max + f_max

    # Initial embedding using aligned var_state_embed (same as v100)
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
        stk_h    = stk.reshape(B, 1, T, T).expand(B, V102_N_HEADS, T, T)
        combined = stk_h.cast(x.dtype) + head_op_mask.cast(x.dtype)

        # 3. Four transformer layers (same as v100/v101)
        h = x_in
        for layer in layers[:4]:
            h = fg_layer_forward_v100(layer, h, combined)

        # 4. Per-breath codebook compression — the topology-invariant Quantize step
        #
        #    h:        (B, T, H=1024) in fp16
        #    codebook: (N_CODE, H) fp32 — cast to match h
        #    temperature: () scalar fp32 — cast to match h
        #
        #    Reconstruction lies in the convex hull of codebook entries.
        #    delta_gate_quant[k] is zero at init → no codebook influence at step 0.
        cb  = codebook.cast(h.dtype)                                # (N_CODE, H)
        tmp = temperature.cast(h.dtype)                             # ()

        scores  = h @ cb.T / tmp.reshape(1, 1, 1)                  # (B, T, N_CODE)
        weights = scores.clip(-1e4, 1e4).softmax(-1)               # (B, T, N_CODE)
        recon   = weights @ cb                                      # (B, T, H)
        quantize = recon - h                                        # delta: recon - h
        gate_quant_k = delta_gate_quant[k].cast(h.dtype).reshape(1, 1, 1)
        h_quant = h + gate_quant_k * quantize                      # h at init (gate=0)

        # 5. Learnable delta gate over h_quant (same gating structure as v100)
        #    At init: h_quant = h (gate_quant=0), same as v100.
        gate_k = delta_gate[k].cast(h_quant.dtype).reshape(1, 1, 1)
        delta  = h_quant - x_pre
        x      = x_pre + gate_k * delta

        # 6. Readout (identical to v100/v101)
        x_ln  = _layernorm(x, model.ln_f_g, model.ln_f_b, model.cfg.layer_norm_eps).cast(dtypes.float)
        var_x = x_ln[:, :n_max, :]
        var_logits_k  = var_x @ domain_codebook.T.cast(dtypes.float)
        var_logits_history.append(var_logits_k)

        fac_x = x_ln[:, n_max:n_max + f_max, :]
        fac_logits_k  = fac_x @ domain_codebook.T.cast(dtypes.float)
        factor_logits_history.append(fac_logits_k)

        pool        = x_ln.mean(axis=1)
        calib_logit = pool @ calib_head_w.cast(dtypes.float) + calib_head_b.cast(dtypes.float)
        calib_k     = calib_logit.reshape(-1).sigmoid()
        calib_history.append(calib_k)

    return var_logits_history, factor_logits_history, calib_history


# ---------------------------------------------------------------------------
# Model parameter attachment
# ---------------------------------------------------------------------------

def attach_fg_params_v102(
    model: Any,
    hidden: int,
    n_code: int = V102_CODEBOOK_N,
) -> None:
    """Allocate v102 codebook params on `model`.

    ONLY the three codebook tensors are added here.  All v100 params must already
    be attached via attach_fg_params_v100 before calling this function.

    Initialization for EXACT warm-start preservation:
      codebook         (n_code, hidden): random orthonormal × 0.5
                                         (matches sudoku_digit_codebook scale)
      delta_gate_quant (K_max,):         all ZEROS
                                         (reconstruction multiplied by zero at init)
      temperature      ():               1.0 (learnable scalar)

    At init: h_quant = h + 0 * (recon - h) = h — byte-identical to v100.

    Gradient flow: delta_gate_quant[k] receives gradient immediately because recon
    and h are both non-zero, so quantize = (recon - h) is non-zero. After 1-2 steps
    delta_gate_quant becomes non-zero and the codebook starts contributing. The
    codebook itself receives gradient through the softmax weights path.
    """
    assert hasattr(model, "fg_v100_domain_codebook"), \
        "call attach_fg_params_v100 first"

    K_max = int(model.fg_v100_breath_embed.shape[0])

    rng = np.random.RandomState(30001)

    # Codebook: (n_code, hidden) — orthonormal rows at scale 0.5
    # Same init pattern as sudoku_digit_codebook (scale 0.1) but at 0.5 for
    # stronger per-position discrimination before learning starts.
    raw_cb = rng.randn(max(hidden, n_code), hidden).astype(np.float32)
    q_cb, _ = np.linalg.qr(raw_cb)
    cb_unit = q_cb[:n_code].astype(np.float32)   # (n_code, hidden) orthonormal
    model.fg_v102_codebook = Tensor(cb_unit * 0.5, dtype=dtypes.float).contiguous()

    # delta_gate_quant: (K_max,) — all zeros at init
    # At init: h_quant = h + 0 * quantize = h → forward is byte-identical to v100
    model.fg_v102_delta_gate_quant = Tensor.zeros((K_max,), dtype=dtypes.float).contiguous()

    # temperature: () scalar — init 1.0
    # Learnable; controls sharpness of codebook matching (lower = more peaked)
    model.fg_v102_temperature = Tensor(np.array([1.0], dtype=np.float32),
                                        dtype=dtypes.float).contiguous()


def fg_v102_codebook_parameters(model: Any) -> list[Tensor]:
    """Trainable v102 codebook-only params."""
    return [
        model.fg_v102_codebook,
        model.fg_v102_delta_gate_quant,
        model.fg_v102_temperature,
    ]


def fg_v102_state_dict(model: Any) -> dict[str, Tensor]:
    """State dict entries for v102 codebook params."""
    return {
        "fg_v102.codebook":          model.fg_v102_codebook,
        "fg_v102.delta_gate_quant":  model.fg_v102_delta_gate_quant,
        "fg_v102.temperature":       model.fg_v102_temperature,
    }


# ---------------------------------------------------------------------------
# JIT training step
# ---------------------------------------------------------------------------

_JIT_V102_CACHE: dict = {}


def _compile_jit_fg_step_v102(
    model: Any,
    opt: Any,
    K: int,
    B: int,
    factor_aux_weight: float,
    calib_weight: float,
    n_max: int = V102_N_MAX,
    f_max: int = V102_F_MAX,
    grad_clip: float = 1.0,
):
    """Compile a TinyJit'd train step for the v102 factor-graph forward.

    Identical structure to _compile_jit_fg_step_v101 but calls
    fg_breathing_forward_v102 (which uses codebook matching instead of
    projection-based waist compression).

    JIT cache key includes V102_TASK and V102_CODEBOOK_N to avoid stale graphs
    if env vars differ between runs.

    Inputs to JIT:
      domain_init   : (B, N_MAX, 100) fp32
      node_kinds    : (B, T_MAX) int
      staging_mask  : (B, K_MAX, T_MAX, T_MAX) fp32
      head_op_mask  : (B, N_HEADS, T_MAX, T_MAX) fp32
      gold_values   : (B, N_MAX) int
      observed_mask : (B, N_MAX) int
      factor_gold   : (B, F_MAX) int   — pre-indexed gold result per factor
      factor_valid  : (B, F_MAX) float — 1=real, 0=pad

    Returns:
      total, healthy, var_ce, factor_aux, calib, cell_acc, query_acc, *pb_ce_0..K-1
    """
    n_code = int(model.fg_v102_codebook.shape[0])
    key = ("v102", id(model), id(opt), int(K), int(B), float(factor_aux_weight),
           float(calib_weight), int(n_max), int(f_max), float(grad_clip),
           int(n_code))
    if key in _JIT_V102_CACHE:
        return _JIT_V102_CACHE[key]

    aw     = float(calib_weight)
    fw     = float(factor_aux_weight)
    gc     = float(grad_clip)
    params = opt.params

    _t0 = _time.perf_counter()
    print(f"[JIT] compile v102 fg step: K={K} B={B} aw={aw} fw={fw} gc={gc} n_code={n_code}...",
          flush=True)

    @TinyJit
    def _step(
        domain_init: Tensor,
        node_kinds: Tensor,
        staging_mask: Tensor,
        head_op_mask: Tensor,
        gold_values: Tensor,
        observed_mask: Tensor,
        factor_gold: Tensor,    # (B, F_MAX) int
        factor_valid: Tensor,   # (B, F_MAX) float
    ):
        opt.zero_grad()

        var_logits_history, factor_logits_history, calib_history = \
            fg_breathing_forward_v102(
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

        # Factor-execute auxiliary loss (vectorized over B × F_MAX, inside JIT)
        factor_aux_sum   = Tensor.zeros((), dtype=dtypes.float).contiguous()
        factor_aux_w_sum = 0.0
        n_valid_factors  = factor_valid.cast(dtypes.float).sum() + 1e-8
        gold_fac_flat    = factor_gold.cast(dtypes.int).reshape(B * f_max)
        gold_fac_oh      = gold_fac_flat.one_hot(100).cast(dtypes.float)
        valid_flat       = factor_valid.cast(dtypes.float).reshape(B * f_max)

        for k_aux, fac_logits_k in enumerate(factor_logits_history):
            w_k_aux     = 1.0 + float(k_aux) / float(max(K - 1, 1))
            fac_flat    = fac_logits_k.reshape(B * f_max, 100)
            fac_lp      = fac_flat.log_softmax(axis=-1)
            fac_nll     = -(fac_lp * gold_fac_oh).sum(axis=-1)
            fac_masked  = fac_nll * valid_flat
            fac_ce_k    = fac_masked.sum() / n_valid_factors
            factor_aux_sum   = factor_aux_sum + fac_ce_k * w_k_aux
            factor_aux_w_sum += w_k_aux
        factor_aux_loss = factor_aux_sum / float(factor_aux_w_sum)

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

        # Total: CE + factor-aux + calibration
        total_ce   = var_loss + fw * factor_aux_loss + aw * calib_loss
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

        return (
            total_ce.realize(),
            healthy.realize(),
            var_loss.realize(),
            factor_aux_loss.realize(),
            calib_loss.realize(),
            cell_acc.realize(),
            query_acc.realize(),
            *(ce.realize() for ce in per_breath_ce_t),
        )

    _JIT_V102_CACHE[key] = _step
    print(f"[JIT] v102 fg step ready (cache={len(_JIT_V102_CACHE)}); first call compiles...",
          flush=True)
    return _step


def _compile_jit_fg_eval_v102(
    model: Any,
    K: int,
    B: int,
    n_max: int = V102_N_MAX,
    f_max: int = V102_F_MAX,
):
    """Compile a TinyJit'd eval step (forward only)."""
    n_code = int(model.fg_v102_codebook.shape[0])
    key = ("eval_v102", id(model), int(K), int(B), int(n_max), int(f_max), int(n_code))
    if key in _JIT_V102_CACHE:
        return _JIT_V102_CACHE[key]

    print(f"[JIT] compile v102 fg eval: K={K} B={B} n_code={n_code}...", flush=True)

    @TinyJit
    def _eval(
        domain_init: Tensor,
        node_kinds: Tensor,
        staging_mask: Tensor,
        head_op_mask: Tensor,
        gold_values: Tensor,
        observed_mask: Tensor,
    ):
        var_logits_history, _, _ = fg_breathing_forward_v102(
            model, domain_init, node_kinds, staging_mask, head_op_mask,
            K=K, n_max=n_max, f_max=f_max,
        )
        final_logits = var_logits_history[-1]
        pred    = final_logits.argmax(axis=-1)
        eq      = (pred == gold_values.cast(dtypes.int)).cast(dtypes.float)
        unobs   = (1 - observed_mask.cast(dtypes.float))
        cell_acc = (eq * unobs).sum() / (unobs.sum() + 1e-8)
        return pred.realize(), cell_acc.realize()

    _JIT_V102_CACHE[key] = _eval
    print(f"[JIT] v102 eval ready (cache={len(_JIT_V102_CACHE)})", flush=True)
    return _eval
