"""v101 factor graph breathing transformer — v100 + per-breath waist compression.

Architectural change over v100:
  Per-breath waist projection inserted AFTER the 4 transformer layers and BEFORE
  the delta_gate residual update.  This is the JPEG codec Quantize step — the
  lossy compression that forces commitment:

    # Waist compression (the Quantize step):
    waist_h  = h @ W_compress + b_compress     # (B, T, 1024) → (B, T, 512)
    waist_h  = waist_h.gelu()                  # nonlinearity in compressed space
    quantize = waist_h @ W_expand + b_expand   # (B, T, 512) → (B, T, 1024), zero at init
    h_quant  = h + quantize                    # residual correction (LoRA-style)
    # Delta gate (same Encode step as v100, but over h_quant instead of h):
    h = x_pre + delta_gate[k] * (h_quant - x_pre)

  Why it forces commitment (once W_expand learns):
    - 1024 → 512 throws away half the dimensions (lossy)
    - GELU nonlinearity prevents the model learning a pure permutation / lossless route
    - 512 → 1024 must RECONSTRUCT the correction from the compressed representation
    - Information that survives 1024→512→1024 is what the model deems committable

  LoRA-style init (exact warm-start preservation):
    W_compress: shape (1024, 512) — random Gaussian × 0.02  (non-zero at init so
                gradient flows to W_expand immediately once W_expand moves)
    b_compress: zeros
    W_expand:   shape (512, 1024) — ALL ZEROS at init
    b_expand:   zeros

    At init: quantize = GELU(h @ W_compress) @ 0 + 0 = 0.
    Therefore h_quant = h + 0 = h — exactly identical to v100 at step 0.
    First gradient step on W_expand moves it away from zero; then W_compress
    receives gradient through the chain, and the waist begins to function.

    This guarantees: warm-start accuracy at step 0 = v100 baseline (≥ 40% easy).

Env var gates:
  V101_TASK=1                  — enable v101 forward path
  V101_K_MAX=10                — number of iterative-prefill breaths
  V101_ENERGY_WEIGHT=0.0       — KL energy weight (0 = diagnostic only)
  V101_CALIB_WEIGHT=0.05       — calibration loss weight
  V101_FACTOR_AUX_WEIGHT=0.5   — factor-execute auxiliary loss weight
  V101_N_MAX=16                — max variable nodes
  V101_F_MAX=8                 — max factor nodes
  V101_WAIST=512               — waist dimension (default 512)

All v100 code is imported and reused unchanged.  Only the forward loop and
parameter attachment differ.
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
    OP_ADD, OP_SUB, OP_MUL, OP_DIV,
)


V101_TASK               = int(os.environ.get("V101_TASK", "0")) > 0
V101_K_MAX              = int(os.environ.get("V101_K_MAX",              "10"))
V101_ENERGY_WEIGHT      = float(os.environ.get("V101_ENERGY_WEIGHT",     "0.0"))
V101_CALIB_WEIGHT       = float(os.environ.get("V101_CALIB_WEIGHT",      "0.05"))
V101_FACTOR_AUX_WEIGHT  = float(os.environ.get("V101_FACTOR_AUX_WEIGHT", "0.5"))
V101_N_MAX              = int(os.environ.get("V101_N_MAX",              "16"))
V101_F_MAX              = int(os.environ.get("V101_F_MAX",               "8"))
V101_T_MAX              = V101_N_MAX + V101_F_MAX
V101_N_HEADS            = 16   # fixed: Pythia-410M
V101_KL_DIAG            = int(os.environ.get("V101_KL_DIAG", "0")) > 0
V101_WAIST              = int(os.environ.get("V101_WAIST",              "512"))


# ---------------------------------------------------------------------------
# Iterative prefill loop with per-breath waist compression
# ---------------------------------------------------------------------------

def fg_breathing_forward_v101(
    model: Any,
    domain_init: Tensor,     # (B, N_MAX, 100) — one-hot observed, uniform unobserved
    node_kinds: Tensor,       # (B, T_MAX) int
    staging_mask: Tensor,     # (B, K_MAX, T_MAX, T_MAX)
    head_op_mask: Tensor,     # (B, N_HEADS, T_MAX, T_MAX)
    K: int,
    n_max: int = V101_N_MAX,
    f_max: int = V101_F_MAX,
) -> tuple[list[Tensor], list[Tensor], list[Tensor]]:
    """Run K iterative-prefill breaths on a batch of factor graphs.

    Per breath k:
      1. Add breath embedding.
      2. Build combined mask: staging_mask[:, k] intersected with head_op_mask.
      3. Run 4 transformer layers with the combined per-head mask.
      4. Per-breath waist compression: x → 512d → x_compressed (the Quantize step).
      5. Delta gate residual update over x_compressed.
      6. Readout: var_logits, factor_logits, calib.

    The waist is the only structural change vs fg_breathing_forward_v100_aligned.
    """
    assert hasattr(model, "fg_v100_domain_codebook"), \
        "model has no v100 params; was V101_TASK (and V100_TASK) set before model init?"
    assert hasattr(model, "fg_v101_W_compress"), \
        "model has no v101 waist params; was attach_fg_params_v101 called?"

    domain_codebook  = model.fg_v100_domain_codebook
    var_state_embed  = model.fg_v100_var_state_embed
    var_pos_embed    = model.fg_v100_var_pos_embed
    factor_pos_embed = model.fg_v100_factor_pos_embed
    node_kind_embed  = model.fg_v100_node_kind_embed
    breath_embed     = model.fg_v100_breath_embed
    delta_gate       = model.fg_v100_delta_gate
    calib_head_w     = model.fg_v100_calib_head_w
    calib_head_b     = model.fg_v100_calib_head_b

    # v101 waist params
    W_compress = model.fg_v101_W_compress   # (H, waist)
    b_compress = model.fg_v101_b_compress   # (waist,)
    W_expand   = model.fg_v101_W_expand     # (waist, H)
    b_expand   = model.fg_v101_b_expand     # (H,)

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
        stk_h    = stk.reshape(B, 1, T, T).expand(B, V101_N_HEADS, T, T)
        combined = stk_h.cast(x.dtype) + head_op_mask.cast(x.dtype)

        # 3. Four transformer layers (same as v100)
        h = x_in
        for layer in layers[:4]:
            h = fg_layer_forward_v100(layer, h, combined)

        # 4. Per-breath waist compression — the Quantize step (v101 addition)
        #    h: (B, T, H=1024) in fp16
        #    W_compress: (H, waist) fp32 — cast to match h
        #    W_expand:   (waist, H) fp32 — ZERO at init, so quantize=0 at step 0
        wc = W_compress.cast(h.dtype)
        bc = b_compress.reshape(1, 1, -1).cast(h.dtype)
        we = W_expand.cast(h.dtype)
        be = b_expand.reshape(1, 1, -1).cast(h.dtype)

        waist_h  = h @ wc + bc                # (B, T, waist)
        waist_h  = waist_h.gelu()             # nonlinearity in compressed space
        quantize = waist_h @ we + be          # (B, T, H) — zero at init (W_expand=0)
        h_quant  = h + quantize               # residual correction; = h at init

        # 5. Learnable delta gate over h_quant (same gating structure as v100)
        #    At init: h_quant = h, so this is byte-identical to v100
        gate_k = delta_gate[k].cast(h_quant.dtype).reshape(1, 1, 1)
        delta  = h_quant - x_pre
        x      = x_pre + gate_k * delta

        # 6. Readout (identical to v100)
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

def attach_fg_params_v101(
    model: Any,
    hidden: int,
    waist: int = V101_WAIST,
) -> None:
    """Allocate v101 waist params on `model`.

    ONLY the four waist tensors are added here.  All v100 params must already be
    attached via attach_fg_params_v100 before calling this function.

    LoRA-style init for EXACT warm-start preservation:
      W_compress (hidden, waist): randn × 0.02   (non-zero so gradient flows to W_expand)
      b_compress (waist,):        zeros
      W_expand   (waist, hidden): ALL ZEROS      (quantize = GELU(h @ W_compress) @ 0 = 0)
      b_expand   (hidden,):       zeros

    At init: quantize = 0, so h_quant = h + 0 = h — byte-identical to v100.
    The delta_gate then operates on h - x_pre, exactly as v100 did.
    Result: warm-start accuracy at step 0 = v100 baseline (no warm-start destruction).

    Gradient flow: W_expand receives gradient first (output is x_compressed, directly
    in the loss path). Once W_expand moves, W_compress receives gradient through
    GELU(h @ W_compress) → W_expand chain.  LoRA-style asymmetric unlock.
    """
    assert hasattr(model, "fg_v100_domain_codebook"), \
        "call attach_fg_params_v100 first"
    rng = np.random.RandomState(20001)

    # W_compress: (hidden, waist) — small random, non-zero so gradient flows immediately
    W_c = (rng.randn(hidden, waist) * 0.02).astype(np.float32)
    model.fg_v101_W_compress = Tensor(W_c, dtype=dtypes.float).contiguous()
    model.fg_v101_b_compress = Tensor.zeros((waist,), dtype=dtypes.float).contiguous()

    # W_expand: (waist, hidden) — ALL ZEROS at init → quantize = 0 at step 0
    model.fg_v101_W_expand = Tensor.zeros((waist, hidden), dtype=dtypes.float).contiguous()
    model.fg_v101_b_expand = Tensor.zeros((hidden,), dtype=dtypes.float).contiguous()


def fg_v101_waist_parameters(model: Any) -> list[Tensor]:
    """Trainable v101 waist-only params."""
    return [
        model.fg_v101_W_compress,
        model.fg_v101_b_compress,
        model.fg_v101_W_expand,
        model.fg_v101_b_expand,
    ]


def fg_v101_state_dict(model: Any) -> dict[str, Tensor]:
    """State dict entries for v101 waist params."""
    return {
        "fg_v101.W_compress": model.fg_v101_W_compress,
        "fg_v101.b_compress": model.fg_v101_b_compress,
        "fg_v101.W_expand":   model.fg_v101_W_expand,
        "fg_v101.b_expand":   model.fg_v101_b_expand,
    }


# ---------------------------------------------------------------------------
# JIT training step
# ---------------------------------------------------------------------------

_JIT_V101_CACHE: dict = {}


def _compile_jit_fg_step_v101(
    model: Any,
    opt: Any,
    K: int,
    B: int,
    factor_aux_weight: float,
    calib_weight: float,
    n_max: int = V101_N_MAX,
    f_max: int = V101_F_MAX,
    grad_clip: float = 1.0,
):
    """Compile a TinyJit'd train step for the v101 factor-graph forward.

    Identical structure to _compile_jit_fg_step_v100 but calls
    fg_breathing_forward_v101 (which includes the waist compression step).

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
    key = ("v101", id(model), id(opt), int(K), int(B), float(factor_aux_weight),
           float(calib_weight), int(n_max), int(f_max), float(grad_clip))
    if key in _JIT_V101_CACHE:
        return _JIT_V101_CACHE[key]

    aw     = float(calib_weight)
    fw     = float(factor_aux_weight)
    gc     = float(grad_clip)
    params = opt.params

    _t0 = _time.perf_counter()
    print(f"[JIT] compile v101 fg step: K={K} B={B} aw={aw} fw={fw} gc={gc}...", flush=True)

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
            fg_breathing_forward_v101(
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

    _JIT_V101_CACHE[key] = _step
    print(f"[JIT] v101 fg step ready (cache={len(_JIT_V101_CACHE)}); first call compiles...",
          flush=True)
    return _step


def _compile_jit_fg_eval_v101(
    model: Any,
    K: int,
    B: int,
    n_max: int = V101_N_MAX,
    f_max: int = V101_F_MAX,
):
    """Compile a TinyJit'd eval step (forward only)."""
    key = ("eval_v101", id(model), int(K), int(B), int(n_max), int(f_max))
    if key in _JIT_V101_CACHE:
        return _JIT_V101_CACHE[key]

    print(f"[JIT] compile v101 fg eval: K={K} B={B}...", flush=True)

    @TinyJit
    def _eval(
        domain_init: Tensor,
        node_kinds: Tensor,
        staging_mask: Tensor,
        head_op_mask: Tensor,
        gold_values: Tensor,
        observed_mask: Tensor,
    ):
        var_logits_history, _, _ = fg_breathing_forward_v101(
            model, domain_init, node_kinds, staging_mask, head_op_mask,
            K=K, n_max=n_max, f_max=f_max,
        )
        final_logits = var_logits_history[-1]
        pred    = final_logits.argmax(axis=-1)
        eq      = (pred == gold_values.cast(dtypes.int)).cast(dtypes.float)
        unobs   = (1 - observed_mask.cast(dtypes.float))
        cell_acc = (eq * unobs).sum() / (unobs.sum() + 1e-8)
        return pred.realize(), cell_acc.realize()

    _JIT_V101_CACHE[key] = _eval
    print(f"[JIT] v101 eval ready (cache={len(_JIT_V101_CACHE)})", flush=True)
    return _eval
