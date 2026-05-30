"""v103 factor graph breathing transformer — VQ-VAE-style waist + codebook compression.

Architectural change over v101/v102:
  Combines v101's bottleneck projection with v102's topology-invariant codebook,
  applied in the WAIST dimension.  This is a true VQ-VAE structure:

    # Step 1: ENCODE (v101's W_compress, LoRA-init)
    waist    = h @ W_compress + b_compress     # (B, T, 1024) → (B, T, 512)
    waist    = waist.gelu()                    # nonlinearity in compressed space

    # Step 2: QUANTIZE (v102's codebook match, but in 512d space)
    # codebook: (N_CODE=32, 512) — smaller, more constrained than v102's 1024d
    scores   = waist @ codebook.T / temperature    # (B, T, 32)
    weights  = scores.clip(-1e4, 1e4).softmax(-1)  # (B, T, 32)
    quantized = weights @ codebook                  # (B, T, 512) — in codebook span only

    # Step 3: DECODE (v101's W_expand, LoRA-init, input = quantized NOT raw waist)
    recon    = quantized @ W_expand + b_expand  # (B, T, 512) → (B, T, 1024)

    # Step 4: residual gate (same as v101/v102)
    h_quant  = h + delta_gate_quant[k] * recon

  Key differences:
    vs v101: decoder input is QUANTIZED waist (codebook-span only), not raw waist.
             Topology memorization is blocked: codebook is the only information path
             from encoder to decoder.
    vs v102: codebook is in 512d (after compression) instead of 1024d.
             32 entries × 512d is smaller and more constrained.  The 512d space is
             where 32 primitives is the right granularity (IB tree = 32 leaves).

  VQ-VAE analogy:
    Encoder = W_compress (1024 → 512)
    Quantize = soft codebook match (32 entries × 512d) → convex-hull reconstruction
    Decoder = W_expand (512 → 1024, receives ONLY quantized signal)
    Commitment = only information that survives encode → quantize → decode contributes

  Initialization for EXACT warm-start preservation (belt-and-suspenders):
    W_compress: random × 0.02 (non-zero so gradient flows to quantize path immediately)
    b_compress: zeros
    codebook:   random orthonormal × 0.5 (in 512d; same pattern as v102 but smaller)
    temperature: 1.0 (learnable scalar)
    W_expand:   ZERO-INITIALIZED → recon = quantized @ 0 = 0 → no effect at step 0
    b_expand:   zeros
    delta_gate_quant[k]: 0.0 for all k (belt-and-suspenders)

    At step 0: delta_gate_quant=0 AND W_expand=0 → h_quant = h + 0 * 0 = h
    Byte-identical to v100 at step 0.

    After step 1: W_expand receives gradient (loss → recon → W_expand).
    After step 2: W_compress and codebook receive gradient through the chain.

Env var gates:
  V103_TASK=1                  — enable v103 forward path
  V103_K_MAX=10                — number of iterative-prefill breaths
  V103_ENERGY_WEIGHT=0.0       — KL energy weight (0 = diagnostic only)
  V103_CALIB_WEIGHT=0.05       — calibration loss weight
  V103_FACTOR_AUX_WEIGHT=0.5   — factor-execute auxiliary loss weight
  V103_N_MAX=16                — max variable nodes
  V103_F_MAX=8                 — max factor nodes
  V103_CODEBOOK_N=32           — codebook size (default 32 = IB tree leaves)
  V103_WAIST=512               — waist dimension (default 512)

All v100 code is imported and reused unchanged.  Only the forward loop and
parameter attachment differ from v101/v102.
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


V103_TASK               = int(os.environ.get("V103_TASK", "0")) > 0
V103_K_MAX              = int(os.environ.get("V103_K_MAX",              "10"))
V103_ENERGY_WEIGHT      = float(os.environ.get("V103_ENERGY_WEIGHT",     "0.0"))
V103_CALIB_WEIGHT       = float(os.environ.get("V103_CALIB_WEIGHT",      "0.05"))
V103_FACTOR_AUX_WEIGHT  = float(os.environ.get("V103_FACTOR_AUX_WEIGHT", "0.5"))
V103_N_MAX              = int(os.environ.get("V103_N_MAX",              "16"))
V103_F_MAX              = int(os.environ.get("V103_F_MAX",               "8"))
V103_T_MAX              = V103_N_MAX + V103_F_MAX
V103_N_HEADS            = 16   # fixed: Pythia-410M
V103_KL_DIAG            = int(os.environ.get("V103_KL_DIAG", "0")) > 0
V103_CODEBOOK_N         = int(os.environ.get("V103_CODEBOOK_N",         "32"))
V103_WAIST              = int(os.environ.get("V103_WAIST",              "512"))


# ---------------------------------------------------------------------------
# Iterative prefill loop with VQ-VAE-style waist + codebook compression
# ---------------------------------------------------------------------------

def fg_breathing_forward_v103(
    model: Any,
    domain_init: Tensor,     # (B, N_MAX, 100) — one-hot observed, uniform unobserved
    node_kinds: Tensor,       # (B, T_MAX) int
    staging_mask: Tensor,     # (B, K_MAX, T_MAX, T_MAX)
    head_op_mask: Tensor,     # (B, N_HEADS, T_MAX, T_MAX)
    K: int,
    n_max: int = V103_N_MAX,
    f_max: int = V103_F_MAX,
) -> tuple[list[Tensor], list[Tensor], list[Tensor]]:
    """Run K iterative-prefill breaths on a batch of factor graphs.

    Per breath k:
      1. Add breath embedding.
      2. Build combined mask: staging_mask[:, k] intersected with head_op_mask.
      3. Run 4 transformer layers with the combined per-head mask.
      4. Per-breath VQ-VAE compression:
           a. ENCODE:    waist     = GELU(h @ W_compress + b_compress)  (B, T, 512)
           b. QUANTIZE:  scores    = waist @ codebook.T / temp          (B, T, N_CODE)
                         weights   = scores.softmax(-1)                 (B, T, N_CODE)
                         quantized = weights @ codebook                 (B, T, 512)
           c. DECODE:    recon     = quantized @ W_expand + b_expand    (B, T, 1024)
                         W_expand ZERO at init → recon = 0 at step 0
           d. GATE:      h_quant   = h + delta_gate_quant[k] * recon   gate=0 at init
      5. Delta gate residual update over h_quant.
      6. Readout: var_logits, factor_logits, calib.

    Information bottleneck: the only path from h to recon goes through the
    compressed 512d space AND the codebook quantization step.  The codebook
    spans at most N_CODE=32 directions in 512d — a tightly constrained subspace.
    """
    assert hasattr(model, "fg_v100_domain_codebook"), \
        "model has no v100 params; was V103_TASK (and V100_TASK) set before model init?"
    assert hasattr(model, "fg_v103_W_compress"), \
        "model has no v103 params; was attach_fg_params_v103 called?"

    domain_codebook  = model.fg_v100_domain_codebook
    var_state_embed  = model.fg_v100_var_state_embed
    var_pos_embed    = model.fg_v100_var_pos_embed
    factor_pos_embed = model.fg_v100_factor_pos_embed
    node_kind_embed  = model.fg_v100_node_kind_embed
    breath_embed     = model.fg_v100_breath_embed
    delta_gate       = model.fg_v100_delta_gate
    calib_head_w     = model.fg_v100_calib_head_w
    calib_head_b     = model.fg_v100_calib_head_b

    # v103 VQ-VAE params
    W_compress       = model.fg_v103_W_compress       # (H, waist)
    b_compress       = model.fg_v103_b_compress       # (waist,)
    codebook         = model.fg_v103_codebook         # (N_CODE, waist)
    temperature      = model.fg_v103_temperature      # () scalar
    W_expand         = model.fg_v103_W_expand         # (waist, H) — ZERO at init
    b_expand         = model.fg_v103_b_expand         # (H,)
    delta_gate_quant = model.fg_v103_delta_gate_quant # (K_max,) — ZERO at init

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
        stk_h    = stk.reshape(B, 1, T, T).expand(B, V103_N_HEADS, T, T)
        combined = stk_h.cast(x.dtype) + head_op_mask.cast(x.dtype)

        # 3. Four transformer layers (same as v100/v101/v102)
        h = x_in
        for layer in layers[:4]:
            h = fg_layer_forward_v100(layer, h, combined)

        # 4. VQ-VAE compression — encode → quantize → decode
        #
        #    Cast all compression params to match h (fp16 in training)
        #    No .cast(dtypes.float32) inside JIT (AMD constraint)
        wc  = W_compress.cast(h.dtype)            # (H, waist)
        bc  = b_compress.reshape(1, 1, -1).cast(h.dtype)  # (1, 1, waist)
        cb  = codebook.cast(h.dtype)              # (N_CODE, waist)
        tmp = temperature.cast(h.dtype)           # ()
        we  = W_expand.cast(h.dtype)              # (waist, H) — ZERO at init
        be  = b_expand.reshape(1, 1, -1).cast(h.dtype)    # (1, 1, H)

        # Step a: ENCODE — 1024 → 512 with GELU nonlinearity
        waist     = (h @ wc + bc).gelu()          # (B, T, waist)

        # Step b: QUANTIZE — soft codebook match in 512d space
        #   scores: (B, T, N_CODE)  — dot products with each codebook entry
        #   weights: soft assignment, clipped for numerical stability
        #   quantized: convex hull reconstruction in 512d codebook span
        scores    = waist @ cb.T / tmp.reshape(1, 1, 1)   # (B, T, N_CODE)
        weights   = scores.clip(-1e4, 1e4).softmax(-1)    # (B, T, N_CODE)
        quantized = weights @ cb                           # (B, T, waist)

        # Step c: DECODE — 512 → 1024 from QUANTIZED (not raw waist)
        #   W_expand is ZERO at init → recon = b_expand = 0 at step 0
        #   Information path: h → W_compress → GELU → codebook → W_expand → recon
        #   Blocked memorization: only codebook-span info passes to decoder
        recon     = quantized @ we + be            # (B, T, H)

        # Step d: GATE — belt-and-suspenders zero at init
        gate_quant_k = delta_gate_quant[k].cast(h.dtype).reshape(1, 1, 1)
        h_quant      = h + gate_quant_k * recon   # = h at init (gate=0, recon=0)

        # 5. Learnable delta gate over h_quant (same gating structure as v100)
        #    At init: h_quant = h → identical to v100
        gate_k = delta_gate[k].cast(h_quant.dtype).reshape(1, 1, 1)
        delta  = h_quant - x_pre
        x      = x_pre + gate_k * delta

        # 6. Readout (identical to v100/v101/v102)
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

def attach_fg_params_v103(
    model: Any,
    hidden: int,
    n_code: int = V103_CODEBOOK_N,
    waist: int  = V103_WAIST,
) -> None:
    """Allocate v103 VQ-VAE params on `model`.

    ONLY the seven VQ-VAE tensors are added here.  All v100 params must already
    be attached via attach_fg_params_v100 before calling this function.

    Initialization for EXACT warm-start preservation (belt-and-suspenders):
      W_compress (hidden, waist): randn × 0.02  — gradient flows to codebook/W_expand
      b_compress (waist,):        zeros
      codebook   (n_code, waist): random orthonormal × 0.5 in waist-dim space
      temperature ():             1.0 (learnable scalar)
      W_expand   (waist, hidden): ALL ZEROS — recon = quantized @ 0 = 0 at step 0
      b_expand   (hidden,):       zeros
      delta_gate_quant (K_max,):  ALL ZEROS — belt-and-suspenders: h_quant = h at init

    At step 0: W_expand=0 AND delta_gate_quant=0 → h_quant = h + 0 * 0 = h
    Forward is byte-identical to v100.

    Gradient flow:
      Step 1: W_expand receives gradient (loss path through recon).
      Step 2: once W_expand != 0, quantized (and thus codebook + W_compress) get gradient.
      Asymmetric unlock: W_expand is the key that unlocks the whole VQ-VAE path.
    """
    assert hasattr(model, "fg_v100_domain_codebook"), \
        "call attach_fg_params_v100 first"

    K_max = int(model.fg_v100_breath_embed.shape[0])
    rng   = np.random.RandomState(40001)

    # W_compress: (hidden, waist) — small random, non-zero for immediate gradient flow
    W_c = (rng.randn(hidden, waist) * 0.02).astype(np.float32)
    model.fg_v103_W_compress = Tensor(W_c, dtype=dtypes.float).contiguous()
    model.fg_v103_b_compress = Tensor.zeros((waist,), dtype=dtypes.float).contiguous()

    # codebook: (n_code, waist) — orthonormal rows at scale 0.5 in 512d
    # Smaller than v102's 1024d codebook: 32 entries × 512d is more constrained.
    raw_cb = rng.randn(max(waist, n_code), waist).astype(np.float32)
    q_cb, _ = np.linalg.qr(raw_cb)
    cb_unit = q_cb[:n_code].astype(np.float32)   # (n_code, waist) orthonormal
    model.fg_v103_codebook = Tensor(cb_unit * 0.5, dtype=dtypes.float).contiguous()

    # temperature: () scalar — init 1.0
    model.fg_v103_temperature = Tensor(np.array([1.0], dtype=np.float32),
                                        dtype=dtypes.float).contiguous()

    # W_expand: (waist, hidden) — ALL ZEROS at init
    # The ZERO init is load-bearing: recon = quantized @ 0 = 0 → no effect at step 0.
    # W_expand is the "key" that unlocks the VQ-VAE path; it receives gradient first,
    # then propagates back through codebook to W_compress.
    model.fg_v103_W_expand = Tensor.zeros((waist, hidden), dtype=dtypes.float).contiguous()
    model.fg_v103_b_expand = Tensor.zeros((hidden,), dtype=dtypes.float).contiguous()

    # delta_gate_quant: (K_max,) — ALL ZEROS at init (belt-and-suspenders)
    model.fg_v103_delta_gate_quant = Tensor.zeros((K_max,), dtype=dtypes.float).contiguous()


def fg_v103_vqvae_parameters(model: Any) -> list[Tensor]:
    """Trainable v103 VQ-VAE-only params."""
    return [
        model.fg_v103_W_compress,
        model.fg_v103_b_compress,
        model.fg_v103_codebook,
        model.fg_v103_temperature,
        model.fg_v103_W_expand,
        model.fg_v103_b_expand,
        model.fg_v103_delta_gate_quant,
    ]


def fg_v103_state_dict(model: Any) -> dict[str, Tensor]:
    """State dict entries for v103 VQ-VAE params."""
    return {
        "fg_v103.W_compress":       model.fg_v103_W_compress,
        "fg_v103.b_compress":       model.fg_v103_b_compress,
        "fg_v103.codebook":         model.fg_v103_codebook,
        "fg_v103.temperature":      model.fg_v103_temperature,
        "fg_v103.W_expand":         model.fg_v103_W_expand,
        "fg_v103.b_expand":         model.fg_v103_b_expand,
        "fg_v103.delta_gate_quant": model.fg_v103_delta_gate_quant,
    }


# ---------------------------------------------------------------------------
# JIT training step
# ---------------------------------------------------------------------------

_JIT_V103_CACHE: dict = {}


def _compile_jit_fg_step_v103(
    model: Any,
    opt: Any,
    K: int,
    B: int,
    factor_aux_weight: float,
    calib_weight: float,
    n_max: int = V103_N_MAX,
    f_max: int = V103_F_MAX,
    grad_clip: float = 1.0,
):
    """Compile a TinyJit'd train step for the v103 factor-graph forward.

    Identical structure to _compile_jit_fg_step_v102 but calls
    fg_breathing_forward_v103 (which uses VQ-VAE-style waist+codebook compression).

    JIT cache key includes V103_TASK, V103_CODEBOOK_N, V103_WAIST to avoid stale
    graphs if env vars differ between runs.

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
    n_code = int(model.fg_v103_codebook.shape[0])
    waist  = int(model.fg_v103_W_compress.shape[1])
    key = ("v103", id(model), id(opt), int(K), int(B), float(factor_aux_weight),
           float(calib_weight), int(n_max), int(f_max), float(grad_clip),
           int(n_code), int(waist))
    if key in _JIT_V103_CACHE:
        return _JIT_V103_CACHE[key]

    aw     = float(calib_weight)
    fw     = float(factor_aux_weight)
    gc     = float(grad_clip)
    params = opt.params

    _t0 = _time.perf_counter()
    print(f"[JIT] compile v103 fg step: K={K} B={B} aw={aw} fw={fw} gc={gc} "
          f"n_code={n_code} waist={waist}...", flush=True)

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
            fg_breathing_forward_v103(
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

    _JIT_V103_CACHE[key] = _step
    print(f"[JIT] v103 fg step ready (cache={len(_JIT_V103_CACHE)}); first call compiles...",
          flush=True)
    return _step


def _compile_jit_fg_eval_v103(
    model: Any,
    K: int,
    B: int,
    n_max: int = V103_N_MAX,
    f_max: int = V103_F_MAX,
):
    """Compile a TinyJit'd eval step (forward only)."""
    n_code = int(model.fg_v103_codebook.shape[0])
    waist  = int(model.fg_v103_W_compress.shape[1])
    key = ("eval_v103", id(model), int(K), int(B), int(n_max), int(f_max),
           int(n_code), int(waist))
    if key in _JIT_V103_CACHE:
        return _JIT_V103_CACHE[key]

    print(f"[JIT] compile v103 fg eval: K={K} B={B} n_code={n_code} waist={waist}...",
          flush=True)

    @TinyJit
    def _eval(
        domain_init: Tensor,
        node_kinds: Tensor,
        staging_mask: Tensor,
        head_op_mask: Tensor,
        gold_values: Tensor,
        observed_mask: Tensor,
    ):
        var_logits_history, _, _ = fg_breathing_forward_v103(
            model, domain_init, node_kinds, staging_mask, head_op_mask,
            K=K, n_max=n_max, f_max=f_max,
        )
        final_logits = var_logits_history[-1]
        pred    = final_logits.argmax(axis=-1)
        eq      = (pred == gold_values.cast(dtypes.int)).cast(dtypes.float)
        unobs   = (1 - observed_mask.cast(dtypes.float))
        cell_acc = (eq * unobs).sum() / (unobs.sum() + 1e-8)
        return pred.realize(), cell_acc.realize()

    _JIT_V103_CACHE[key] = _eval
    print(f"[JIT] v103 eval ready (cache={len(_JIT_V103_CACHE)})", flush=True)
    return _eval
