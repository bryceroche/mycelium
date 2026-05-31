"""v104 factor graph breathing transformer — IB-anchored codebook initialization.

Identical to v102 (1024d residual codebook matching) EXCEPT the codebook is
initialized from IB-derived semantic centroids instead of random orthonormal vectors.

Architecture (same as v102, NOT v103 VQ-VAE):
  Per breath, in residual 1024d space:
    scores   = h @ codebook.T / temperature     # (B, T, N_CODE=32)
    weights  = scores.clip(-1e4, 1e4).softmax(-1)
    recon    = weights @ codebook               # (B, T, 1024) — in codebook span
    quantize = recon - h                        # delta toward codebook
    h_quant  = h + delta_gate_quant[k] * quantize   # LoRA-style gate

IB codebook initialization (the hypothesis under test):
  Load 32 × 1024 semantic centroids from .cache/ib_centroids_gsm8k_partial.npz.
  These are Pythia embeddings of variable descriptions, clustered via hierarchical
  K-means per OP family (ADD/SUB/MUL/DIV → ~8 sub-clusters each).
  Add small noise (scale 0.01) for tie-breaking.
  Codebook is trainable — gradients refine the IB init.

Initialization for EXACT warm-start preservation:
  codebook         (32, 1024): IB centroids + N(0, 0.01) noise  (non-zero, but small)
  delta_gate_quant (K_max,):   ALL ZEROS — recon multiplied by zero at init
  temperature      ():         1.0 (learnable scalar)

At init: h_quant = h + 0 * (recon - h) = h → byte-identical to v100.
After step 1: delta_gate_quant[k] receives gradient (because recon ≠ 0).

Comparison baseline:
  v100 (no codebook):               K=10 = 40.7% cell acc
  v101 (projection waist 512d):     K=10 = 47.6% cell acc
  v102 (random 32-entry codebook):  K=10 = 46.6% cell acc (v103 had wrong arch label)
  v104 (IB-anchored 32-entry):      hypothesis ≥ 50% cell acc

Env var gates:
  V104_TASK=1                   — enable v104 forward path
  V104_K_MAX=10                 — number of iterative-prefill breaths
  V104_ENERGY_WEIGHT=0.0        — KL energy weight (0 = diagnostic only)
  V104_CALIB_WEIGHT=0.05        — calibration loss weight
  V104_FACTOR_AUX_WEIGHT=0.5    — factor-execute auxiliary loss weight
  V104_N_MAX=16                 — max variable nodes
  V104_F_MAX=8                  — max factor nodes
  V104_CODEBOOK_N=32            — codebook size (must match IB centroids = 32)
  V104_IB_CENTROIDS=.cache/ib_centroids_gsm8k_partial.npz  — path to IB npz

All v100 code is imported and reused unchanged.
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


V104_TASK               = int(os.environ.get("V104_TASK", "0")) > 0
V104_K_MAX              = int(os.environ.get("V104_K_MAX",              "10"))
V104_ENERGY_WEIGHT      = float(os.environ.get("V104_ENERGY_WEIGHT",     "0.0"))
V104_CALIB_WEIGHT       = float(os.environ.get("V104_CALIB_WEIGHT",      "0.05"))
V104_FACTOR_AUX_WEIGHT  = float(os.environ.get("V104_FACTOR_AUX_WEIGHT", "0.5"))
V104_N_MAX              = int(os.environ.get("V104_N_MAX",              "16"))
V104_F_MAX              = int(os.environ.get("V104_F_MAX",               "8"))
V104_T_MAX              = V104_N_MAX + V104_F_MAX
V104_N_HEADS            = 16   # fixed: Pythia-410M
V104_KL_DIAG            = int(os.environ.get("V104_KL_DIAG", "0")) > 0
V104_CODEBOOK_N         = int(os.environ.get("V104_CODEBOOK_N",         "32"))
V104_IB_CENTROIDS       = os.environ.get(
    "V104_IB_CENTROIDS", ".cache/ib_centroids_gsm8k_partial.npz"
)


# ---------------------------------------------------------------------------
# IB centroid loading
# ---------------------------------------------------------------------------

def load_ib_centroids(path: str, n_code: int, hidden: int,
                      noise_scale: float = 0.01,
                      seed: int = 42) -> np.ndarray:
    """Load IB centroids from .npz and return (n_code, hidden) float32 array.

    If the npz contains more/fewer entries than n_code, truncate or pad with
    random orthonormal rows.  Adds small Gaussian noise for tie-breaking.

    Args:
      path:        Path to .npz with keys 'centroids' (N, hidden) and 'leaf_ids' (N,).
      n_code:      Number of codebook entries needed.
      hidden:      Expected centroid dimension (must match).
      noise_scale: Std of added Gaussian noise (0.01 recommended).
      seed:        RNG seed for reproducible tie-breaking noise.

    Returns:
      (n_code, hidden) float32 array.
    """
    rng = np.random.RandomState(seed)

    if not os.path.exists(path):
        print(f"[v104] WARNING: IB centroids not found at {path}; falling back to random init")
        raw = rng.randn(max(hidden, n_code), hidden).astype(np.float32)
        q, _ = np.linalg.qr(raw)
        cb = q[:n_code].astype(np.float32) * 0.5
    else:
        d = np.load(path)
        centroids = d["centroids"].astype(np.float32)  # (N_ib, hidden)
        n_ib, dim_ib = centroids.shape
        assert dim_ib == hidden, (
            f"IB centroid dim {dim_ib} != model hidden {hidden}; "
            f"check that centroids were extracted from the correct Pythia model."
        )
        if n_ib >= n_code:
            cb = centroids[:n_code].copy()
        else:
            # Pad with random orthonormal rows for the remaining entries
            print(f"[v104] IB has {n_ib} centroids < n_code={n_code}; padding with random rows")
            extra = n_code - n_ib
            raw = rng.randn(max(hidden, extra), hidden).astype(np.float32)
            q, _ = np.linalg.qr(raw)
            cb = np.concatenate([centroids, q[:extra] * 0.5], axis=0)

    # Add small noise for tie-breaking (prevents identical softmax weights at init)
    cb = cb + rng.randn(*cb.shape).astype(np.float32) * noise_scale

    # Report stats
    norms = np.linalg.norm(cb, axis=1)
    print(
        f"[v104] IB codebook loaded: shape={cb.shape}  "
        f"norm range=[{norms.min():.4f}, {norms.max():.4f}]  "
        f"noise_scale={noise_scale}"
    )
    return cb


# ---------------------------------------------------------------------------
# Iterative prefill loop with IB-anchored codebook compression
# ---------------------------------------------------------------------------

def fg_breathing_forward_v104(
    model: Any,
    domain_init: Tensor,     # (B, N_MAX, 100) — one-hot observed, uniform unobserved
    node_kinds: Tensor,       # (B, T_MAX) int
    staging_mask: Tensor,     # (B, K_MAX, T_MAX, T_MAX)
    head_op_mask: Tensor,     # (B, N_HEADS, T_MAX, T_MAX)
    K: int,
    n_max: int = V104_N_MAX,
    f_max: int = V104_F_MAX,
) -> tuple[list[Tensor], list[Tensor], list[Tensor]]:
    """Run K iterative-prefill breaths on a batch of factor graphs.

    Identical to fg_breathing_forward_v102 except the codebook was initialized
    from IB semantic centroids (a training-time difference, not a forward-pass
    difference).  At runtime the architecture is identical to v102.

    Per breath k:
      1. Add breath embedding.
      2. Build combined mask: staging_mask[:, k] × head_op_mask.
      3. Run 4 transformer layers.
      4. IB-anchored codebook compression in 1024d residual space:
           scores   = h @ codebook.T / temperature    (B, T, N_CODE=32)
           weights  = scores.clip(-1e4, 1e4).softmax(-1)
           recon    = weights @ codebook               (B, T, 1024)
           quantize = recon - h
           h_quant  = h + delta_gate_quant[k] * quantize
      5. Learnable delta gate over h_quant.
      6. Readout: var_logits, factor_logits, calib.
    """
    assert hasattr(model, "fg_v100_domain_codebook"), \
        "model has no v100 params; was V104_TASK (and V100_TASK) set before model init?"
    assert hasattr(model, "fg_v104_codebook"), \
        "model has no v104 params; was attach_fg_params_v104 called?"

    domain_codebook  = model.fg_v100_domain_codebook
    var_state_embed  = model.fg_v100_var_state_embed
    var_pos_embed    = model.fg_v100_var_pos_embed
    factor_pos_embed = model.fg_v100_factor_pos_embed
    node_kind_embed  = model.fg_v100_node_kind_embed
    breath_embed     = model.fg_v100_breath_embed
    delta_gate       = model.fg_v100_delta_gate
    calib_head_w     = model.fg_v100_calib_head_w
    calib_head_b     = model.fg_v100_calib_head_b

    # v104 IB-anchored codebook params (same names as v102 but initialized differently)
    codebook         = model.fg_v104_codebook          # (N_CODE, H=1024)
    delta_gate_quant = model.fg_v104_delta_gate_quant  # (K_max,)
    temperature      = model.fg_v104_temperature       # () scalar

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
        stk_h    = stk.reshape(B, 1, T, T).expand(B, V104_N_HEADS, T, T)
        combined = stk_h.cast(x.dtype) + head_op_mask.cast(x.dtype)

        # 3. Four transformer layers (same as v100/v101/v102)
        h = x_in
        for layer in layers[:4]:
            h = fg_layer_forward_v100(layer, h, combined)

        # 4. IB-anchored codebook compression in residual 1024d space
        #
        #    Cast codebook and temperature to match h (fp16 in training)
        #    No .cast(dtypes.float32) inside JIT (AMD constraint)
        cb  = codebook.cast(h.dtype)              # (N_CODE, H)
        tmp = temperature.cast(h.dtype)           # ()

        scores   = h @ cb.T / tmp.reshape(1, 1, 1)            # (B, T, N_CODE)
        weights  = scores.clip(-1e4, 1e4).softmax(-1)         # (B, T, N_CODE)
        recon    = weights @ cb                               # (B, T, H)
        quantize = recon - h                                  # delta: toward codebook
        gate_quant_k = delta_gate_quant[k].cast(h.dtype).reshape(1, 1, 1)
        h_quant = h + gate_quant_k * quantize                # = h at init (gate=0)

        # 5. Learnable delta gate over h_quant (same gating structure as v100)
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

def attach_fg_params_v104(
    model: Any,
    hidden: int,
    n_code: int = V104_CODEBOOK_N,
    ib_centroids_path: str = V104_IB_CENTROIDS,
) -> None:
    """Allocate v104 IB-anchored codebook params on `model`.

    ONLY the three codebook tensors are added here.  All v100 params must
    already be attached via attach_fg_params_v100 before calling this function.

    The KEY difference from v102: codebook is initialized from IB semantic
    centroids (Pythia embeddings of variable descriptions, clustered per OP).
    Everything else (delta_gate_quant=0, temperature=1.0, same forward path)
    is identical to v102.

    Initialization:
      codebook         (n_code, hidden): IB centroids + N(0, 0.01) noise
                                          (falls back to random QR if npz missing)
      delta_gate_quant (K_max,):         all ZEROS (belt-and-suspenders warm-start)
      temperature      ():               1.0 (learnable scalar)

    At step 0: delta_gate_quant=0 → h_quant = h → forward byte-identical to v100.
    """
    assert hasattr(model, "fg_v100_domain_codebook"), \
        "call attach_fg_params_v100 first"

    K_max = int(model.fg_v100_breath_embed.shape[0])

    # Load IB-anchored codebook
    cb_np = load_ib_centroids(ib_centroids_path, n_code=n_code, hidden=hidden)
    model.fg_v104_codebook = Tensor(cb_np, dtype=dtypes.float).contiguous()

    # delta_gate_quant: (K_max,) — all zeros at init (warm-start preservation)
    model.fg_v104_delta_gate_quant = Tensor.zeros((K_max,), dtype=dtypes.float).contiguous()

    # temperature: () scalar — init 1.0 (same as v102)
    model.fg_v104_temperature = Tensor(np.array([1.0], dtype=np.float32),
                                        dtype=dtypes.float).contiguous()


def fg_v104_codebook_parameters(model: Any) -> list[Tensor]:
    """Trainable v104 codebook-only params."""
    return [
        model.fg_v104_codebook,
        model.fg_v104_delta_gate_quant,
        model.fg_v104_temperature,
    ]


def fg_v104_state_dict(model: Any) -> dict[str, Tensor]:
    """State dict entries for v104 codebook params."""
    return {
        "fg_v104.codebook":          model.fg_v104_codebook,
        "fg_v104.delta_gate_quant":  model.fg_v104_delta_gate_quant,
        "fg_v104.temperature":       model.fg_v104_temperature,
    }


# ---------------------------------------------------------------------------
# JIT training step
# ---------------------------------------------------------------------------

_JIT_V104_CACHE: dict = {}


def _compile_jit_fg_step_v104(
    model: Any,
    opt: Any,
    K: int,
    B: int,
    factor_aux_weight: float,
    calib_weight: float,
    n_max: int = V104_N_MAX,
    f_max: int = V104_F_MAX,
    grad_clip: float = 1.0,
):
    """Compile a TinyJit'd train step for the v104 factor-graph forward.

    Identical structure to _compile_jit_fg_step_v102 but calls
    fg_breathing_forward_v104 (IB-anchored codebook, same forward path).

    JIT cache key includes V104_TASK and V104_CODEBOOK_N to avoid stale
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
    n_code = int(model.fg_v104_codebook.shape[0])
    key = ("v104", id(model), id(opt), int(K), int(B), float(factor_aux_weight),
           float(calib_weight), int(n_max), int(f_max), float(grad_clip),
           int(n_code))
    if key in _JIT_V104_CACHE:
        return _JIT_V104_CACHE[key]

    aw     = float(calib_weight)
    fw     = float(factor_aux_weight)
    gc     = float(grad_clip)
    params = opt.params

    _t0 = _time.perf_counter()
    print(f"[JIT] compile v104 fg step: K={K} B={B} aw={aw} fw={fw} gc={gc} "
          f"n_code={n_code}...", flush=True)

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
            fg_breathing_forward_v104(
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

    _JIT_V104_CACHE[key] = _step
    print(f"[JIT] v104 fg step ready (cache={len(_JIT_V104_CACHE)}); first call compiles...",
          flush=True)
    return _step


def _compile_jit_fg_eval_v104(
    model: Any,
    K: int,
    B: int,
    n_max: int = V104_N_MAX,
    f_max: int = V104_F_MAX,
):
    """Compile a TinyJit'd eval step (forward only)."""
    n_code = int(model.fg_v104_codebook.shape[0])
    key = ("eval_v104", id(model), int(K), int(B), int(n_max), int(f_max), int(n_code))
    if key in _JIT_V104_CACHE:
        return _JIT_V104_CACHE[key]

    print(f"[JIT] compile v104 fg eval: K={K} B={B} n_code={n_code}...", flush=True)

    @TinyJit
    def _eval(
        domain_init: Tensor,
        node_kinds: Tensor,
        staging_mask: Tensor,
        head_op_mask: Tensor,
        gold_values: Tensor,
        observed_mask: Tensor,
    ):
        var_logits_history, _, _ = fg_breathing_forward_v104(
            model, domain_init, node_kinds, staging_mask, head_op_mask,
            K=K, n_max=n_max, f_max=f_max,
        )
        final_logits = var_logits_history[-1]
        pred    = final_logits.argmax(axis=-1)
        eq      = (pred == gold_values.cast(dtypes.int)).cast(dtypes.float)
        unobs   = (1 - observed_mask.cast(dtypes.float))
        cell_acc = (eq * unobs).sum() / (unobs.sum() + 1e-8)
        return pred.realize(), cell_acc.realize()

    _JIT_V104_CACHE[key] = _eval
    print(f"[JIT] v104 eval ready (cache={len(_JIT_V104_CACHE)})", flush=True)
    return _eval
