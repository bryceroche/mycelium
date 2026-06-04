"""v108 factor graph breathing transformer — tree-structured digit codebook.

Architectural pivot from v107 (flat 200-bin codebook) to v108 (5-level tree
codebook). Same single-token-per-variable substrate (avoids the v105 L0
mean-field collapse on multi-token variables), but the READOUT is hierarchical:

  Level 0 (10000s digit): 10 codebook entries  (coarsest)
  Level 1 (1000s  digit): 10 codebook entries
  Level 2 (100s   digit): 10 codebook entries
  Level 3 (10s    digit): 10 codebook entries
  Level 4 (1s     digit): 10 codebook entries  (finest)
  Total: 50 entries × 1024d = ~50K params (tiny)

The hidden state at each breath is read at all 5 levels in parallel. The full
number is reconstructed as Σ_l 10^(4-l) × argmax(level_l).

Two supervision regimes (env-gated):
  V108_HARD_BREATH_LEVEL=0 (default, SOFT):
    All 5 levels supervised at every breath, per-breath ladder weights breaths.
    loss = Σ_k (1 + k/(K-1)) × mean over levels of CE(level_l, digit_l)

  V108_HARD_BREATH_LEVEL=1 (HARD, the other-Claude design):
    Breath k targets level k for k in [0, N_DIGITS-1].
    Breaths k >= N_DIGITS target all 5 levels (refinement).
    loss = Σ_k (1 + k/(K-1)) × (CE on assigned level(s))

The tree codebook is Fourier-orthogonal initialized per level (each level's 10
entries are distinct multi-harmonic vectors). All v107 backbone params are
shared (warm-start from v107 ckpt is supported).

Env vars:
  V108_TASK=1                   — enable v108 forward path
  V108_K_MAX=8                  — number of breaths
  V108_N_DIGITS=5               — number of tree levels
  V108_HARD_BREATH_LEVEL=0      — soft (default) or hard breath-to-level
  V108_VAR_LOSS_WEIGHT=1.0      — weight on tree CE loss
  V108_KEEP_BIN_LOSS=0          — also compute v107 200-bin CE as diagnostic
"""
from __future__ import annotations

import math
import os
from typing import Any

import numpy as np
from tinygrad import Tensor, dtypes
from tinygrad.engine.jit import TinyJit

from mycelium.config import Config

# Reuse v107's forward kernel pieces
from mycelium.factor_graph_v100 import (
    embed_factor_graph_v100_aligned,
    fg_layer_forward_v100,
    V100_N_HEADS,
)
from mycelium.factor_graph_v107 import (
    V107_N_MAX, V107_F_MAX, V107_N_HEADS,
    V107_ENERGY_WEIGHT, V107_CALIB_WEIGHT, V107_FACTOR_AUX_WEIGHT,
    V107_CODEBOOK_N, V107_IB_CENTROIDS,
    attach_fg_params_v107, fg_v107_parameters, fg_v107_state_dict,
    get_bin_values, bin_to_digits,
)
from mycelium.factor_graph_v104 import load_ib_centroids

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

V108_TASK              = int(os.environ.get("V108_TASK", "0")) > 0
V108_K_MAX             = int(os.environ.get("V108_K_MAX",             "8"))
V108_N_DIGITS          = int(os.environ.get("V108_N_DIGITS",          "5"))
V108_HARD_BREATH_LEVEL = int(os.environ.get("V108_HARD_BREATH_LEVEL", "0")) > 0
V108_VAR_LOSS_WEIGHT   = float(os.environ.get("V108_VAR_LOSS_WEIGHT", "1.0"))
V108_KEEP_BIN_LOSS     = int(os.environ.get("V108_KEEP_BIN_LOSS",     "0")) > 0
V108_N_MAX             = int(os.environ.get("V108_N_MAX",             str(V107_N_MAX)))
V108_F_MAX             = int(os.environ.get("V108_F_MAX",             str(V107_F_MAX)))
V108_T_MAX             = V108_N_MAX + V108_F_MAX
V108_CODEBOOK_N        = int(os.environ.get("V108_CODEBOOK_N",        str(V107_CODEBOOK_N)))
V108_IB_CENTROIDS      = os.environ.get("V108_IB_CENTROIDS",          V107_IB_CENTROIDS)
V108_ENERGY_WEIGHT     = float(os.environ.get("V108_ENERGY_WEIGHT",   "0.0"))   # tree decoder uses digit reconstruction
V108_CALIB_WEIGHT      = float(os.environ.get("V108_CALIB_WEIGHT",    str(V107_CALIB_WEIGHT)))
V108_FACTOR_AUX_WEIGHT = float(os.environ.get("V108_FACTOR_AUX_WEIGHT", str(V107_FACTOR_AUX_WEIGHT)))
V108_N_HEADS           = V107_N_HEADS

# Place values, MSD-first: [10000, 1000, 100, 10, 1]
PLACE_VALUES_MSD = np.array(
    [10 ** (V108_N_DIGITS - 1 - p) for p in range(V108_N_DIGITS)],
    dtype=np.int64,
)


# ---------------------------------------------------------------------------
# Helper: convert integer values → MSD-first digit decomposition
# ---------------------------------------------------------------------------

def values_to_digits_msd(values: np.ndarray, n_digits: int = V108_N_DIGITS) -> np.ndarray:
    """values: (...,) int → digits: (..., n_digits) int, MSD-first.

    For value 1234 with n_digits=5: returns [0, 1, 2, 3, 4].
    Values outside [0, 10**n_digits - 1] are clipped.
    """
    v = values.astype(np.int64).clip(0, 10 ** n_digits - 1)
    out_shape = (*v.shape, n_digits)
    digits = np.zeros(out_shape, dtype=np.int64)
    rem = v.copy()
    for p in range(n_digits):
        place = 10 ** (n_digits - 1 - p)
        digits[..., p] = rem // place
        rem = rem % place
    return digits


def digits_to_value_msd(digits: np.ndarray) -> np.ndarray:
    """Inverse of values_to_digits_msd. digits[..., l] gets weight 10^(N-1-l)."""
    n_digits = digits.shape[-1]
    places = np.array(
        [10 ** (n_digits - 1 - p) for p in range(n_digits)],
        dtype=digits.dtype,
    )
    return (digits * places).sum(axis=-1)


def bins_to_digits_msd(gold_bins: np.ndarray, n_digits: int = V108_N_DIGITS) -> np.ndarray:
    """Map (B, N_MAX) bin indices → (B, N_MAX, n_digits) MSD-first digits.

    Uses get_bin_values()[bin] as the integer value, then decomposes.
    """
    bv = get_bin_values()
    values = bv[gold_bins.astype(np.int64).clip(0, 199)]
    return values_to_digits_msd(values, n_digits=n_digits)


# ---------------------------------------------------------------------------
# Tree codebook Fourier orthogonal init (per level)
# ---------------------------------------------------------------------------

def _fourier_orthogonal_init(n_entries: int, hidden: int, seed: int = 0) -> np.ndarray:
    """Build (n_entries, hidden) orthogonal multi-harmonic Fourier codebook.

    For entry d in [0, n_entries):
      cb[d, 2k]   = cos(2π · d · (k+1) / n_entries)
      cb[d, 2k+1] = sin(2π · d · (k+1) / n_entries)
    Scaled by 0.1/sqrt(hidden) so inner products with LN'd hidden states are
    in a reasonable softmax-friendly range (lesson from v105.12 saturation).
    """
    cb = np.zeros((n_entries, hidden), dtype=np.float32)
    n_pairs = hidden // 2
    for d in range(n_entries):
        for k in range(n_pairs):
            angle = 2.0 * np.pi * d * (k + 1) / float(n_entries)
            cb[d, 2 * k]     = float(np.cos(angle))
            cb[d, 2 * k + 1] = float(np.sin(angle))
    cb = cb * (0.1 / float(np.sqrt(hidden)))
    return cb.astype(np.float32)


# ---------------------------------------------------------------------------
# Iterative prefill loop (v108)
# ---------------------------------------------------------------------------

def fg_breathing_forward_v108(
    model: Any,
    domain_init: Tensor,     # (B, N_MAX, 200) — one-hot observed, uniform unobserved
    node_kinds: Tensor,       # (B, T_MAX) int
    staging_mask: Tensor,     # (B, K_MAX, T_MAX, T_MAX)
    head_op_mask: Tensor,     # (B, N_HEADS, T_MAX, T_MAX)
    K: int,
    n_max: int = V108_N_MAX,
    f_max: int = V108_F_MAX,
    n_digits: int = V108_N_DIGITS,
) -> tuple[list[Tensor], list[Tensor], list[Tensor], list[Tensor]]:
    """Run K iterative-prefill breaths; return per-breath outputs.

    Returns:
      tree_logits_history    : K × (B, N_MAX, n_digits, 10) — per-level digit logits
      var_logits_history     : K × (B, N_MAX, 200)          — 200-bin diagnostic
      factor_logits_history  : K × (B, F_MAX, 200)          — factor bin (v107 path)
      calib_history          : K × (B,)
    """
    assert hasattr(model, "fg_v107_domain_codebook"), \
        "model has no v107 backbone params"
    assert hasattr(model, "fg_v108_tree_codebook"), \
        "model has no v108 tree codebook — was attach_fg_params_v108 called?"

    # Backbone params (shared with v107)
    domain_codebook  = model.fg_v107_domain_codebook    # (200, H)
    var_state_embed  = model.fg_v107_var_state_embed
    var_pos_embed    = model.fg_v107_var_pos_embed
    factor_pos_embed = model.fg_v107_factor_pos_embed
    node_kind_embed  = model.fg_v107_node_kind_embed
    breath_embed     = model.fg_v107_breath_embed
    delta_gate       = model.fg_v107_delta_gate
    calib_head_w     = model.fg_v107_calib_head_w
    calib_head_b     = model.fg_v107_calib_head_b
    semantic_codebook    = model.fg_v107_semantic_codebook
    delta_gate_quant     = model.fg_v107_delta_gate_quant
    temperature          = model.fg_v107_temperature

    # v108 new params
    tree_codebook = model.fg_v108_tree_codebook  # (n_digits, 10, H)
    H = int(tree_codebook.shape[-1])

    B = int(domain_init.shape[0])
    T = n_max + f_max

    x = embed_factor_graph_v100_aligned(
        domain_init, node_kinds,
        var_pos_embed, factor_pos_embed, node_kind_embed,
        domain_codebook, var_state_embed,
    )
    x = x.cast(dtypes.half) if x.dtype != dtypes.half else x

    layers = list(model.block.layers)
    K_max  = int(breath_embed.shape[0])
    assert K <= K_max, f"K={K} > K_max={K_max}"

    from mycelium.breathing import _layernorm

    tree_logits_history   = []
    var_logits_history    = []
    factor_logits_history = []
    calib_history         = []

    # Pre-reshape tree_codebook for matmul: (n_digits * 10, H)
    tree_cb_flat = tree_codebook.reshape(n_digits * 10, H)  # (50, H)

    for k in range(K):
        be_k  = breath_embed[k].reshape(1, 1, -1).cast(x.dtype)
        x_in  = x + be_k
        x_pre = x

        stk      = staging_mask[:, k, :, :]
        stk_h    = stk.reshape(B, 1, T, T).expand(B, V108_N_HEADS, T, T)
        combined = stk_h.cast(x.dtype) + head_op_mask.cast(x.dtype)

        h = x_in
        for layer in layers[:4]:
            h = fg_layer_forward_v100(layer, h, combined)

        # IB semantic codebook compression
        cb  = semantic_codebook.cast(h.dtype)
        tmp = temperature.cast(h.dtype)
        scores   = h @ cb.T / tmp.reshape(1, 1, 1)
        weights  = scores.clip(-1e4, 1e4).softmax(-1)
        recon    = weights @ cb
        quantize = recon - h
        gate_quant_k = delta_gate_quant[k].cast(h.dtype).reshape(1, 1, 1)
        h_quant = h + gate_quant_k * quantize

        gate_k = delta_gate[k].cast(h_quant.dtype).reshape(1, 1, 1)
        delta  = h_quant - x_pre
        x      = x_pre + gate_k * delta

        # Readout
        x_ln  = _layernorm(x, model.ln_f_g, model.ln_f_b,
                           model.cfg.layer_norm_eps).cast(dtypes.float)

        var_x = x_ln[:, :n_max, :]                      # (B, N_MAX, H)

        # v108 tree readout: (B, N_MAX, H) @ (H, 50) → (B, N_MAX, 50) → (B, N_MAX, 5, 10)
        tree_logits_flat = var_x @ tree_cb_flat.T.cast(dtypes.float)
        tree_logits_k = tree_logits_flat.reshape(B, n_max, n_digits, 10)
        tree_logits_history.append(tree_logits_k)

        # v107 200-bin diagnostic readout (kept for comparison eval)
        var_logits_k = var_x @ domain_codebook.T.cast(dtypes.float)
        var_logits_history.append(var_logits_k)

        # Factor readout (still bin-based — factor aux is 200-way)
        fac_x = x_ln[:, n_max:n_max + f_max, :]
        fac_logits_k = fac_x @ domain_codebook.T.cast(dtypes.float)
        factor_logits_history.append(fac_logits_k)

        pool        = x_ln.mean(axis=1)
        calib_logit = pool @ calib_head_w.cast(dtypes.float) + calib_head_b.cast(dtypes.float)
        calib_k     = calib_logit.reshape(-1).sigmoid()
        calib_history.append(calib_k)

    return tree_logits_history, var_logits_history, factor_logits_history, calib_history


# ---------------------------------------------------------------------------
# Accuracy from tree decoder
# ---------------------------------------------------------------------------

def fg_accuracy_v108(
    tree_logits_final: Tensor,   # (B, N_MAX, n_digits, 10)
    gold_digits: Tensor,          # (B, N_MAX, n_digits) int
    observed_mask: Tensor,        # (B, N_MAX) int
    query_idx_np: np.ndarray,     # (B,) int
    n_digits: int = V108_N_DIGITS,
) -> dict:
    """Return cell_acc / query_acc on the reconstructed whole number.

    A cell is correct iff ALL n_digits levels match gold.
    """
    pred_digits = tree_logits_final.argmax(axis=-1)                # (B, N_MAX, n_digits)
    correct_per_pos = (pred_digits == gold_digits.cast(dtypes.int)).cast(dtypes.float)
    correct_all = correct_per_pos.prod(axis=-1)                    # (B, N_MAX)
    unobs = (1 - observed_mask.cast(dtypes.float))
    masked = correct_all * unobs
    n_unobs = unobs.sum() + 1e-8
    cell_acc = float((masked.sum() / n_unobs).realize().numpy())

    pred_np = pred_digits.cast(dtypes.int).realize().numpy()
    gold_np = gold_digits.cast(dtypes.int).realize().numpy()
    B = int(pred_np.shape[0])
    q_correct = np.array([
        int(np.all(pred_np[b, query_idx_np[b]] == gold_np[b, query_idx_np[b]]))
        for b in range(B)
    ], dtype=np.float32)
    query_acc = float(q_correct.mean())

    return {"cell_acc": cell_acc, "query_acc": query_acc}


# ---------------------------------------------------------------------------
# Parameter attachment
# ---------------------------------------------------------------------------

def attach_fg_params_v108(
    model: Any,
    hidden: int,
    n_max: int = V108_N_MAX,
    f_max: int = V108_F_MAX,
    k_max: int | None = None,
    n_digits: int = V108_N_DIGITS,
    n_code: int = V108_CODEBOOK_N,
    ib_centroids_path: str = V108_IB_CENTROIDS,
) -> None:
    """Attach v107 backbone + v108 tree codebook.

    v107 path:
      fg_v107_domain_codebook (200, H), var_state_embed, var_pos_embed,
      factor_pos_embed, node_kind_embed, breath_embed, delta_gate,
      calib_head_w/b, semantic_codebook, delta_gate_quant, temperature

    v108 path:
      fg_v108_tree_codebook (n_digits, 10, H) — Fourier orthogonal init per level,
                                                scaled 0.1/sqrt(H)
    """
    if k_max is None:
        k_max = V108_K_MAX

    # Attach all v107 backbone params (NB: v107 uses V107_K_MAX default, so override)
    attach_fg_params_v107(
        model, hidden=hidden,
        n_max=n_max, f_max=f_max, k_max=k_max,
        n_code=n_code, ib_centroids_path=ib_centroids_path,
    )

    # Tree codebook: (n_digits, 10, H), Fourier orthogonal per level
    # Each level gets its own random-phase shift via seed
    cb = np.zeros((n_digits, 10, hidden), dtype=np.float32)
    for level in range(n_digits):
        cb[level] = _fourier_orthogonal_init(10, hidden, seed=level * 17 + 31)
    model.fg_v108_tree_codebook = Tensor(cb, dtype=dtypes.float).contiguous()

    print(
        f"[v108] tree codebook attached: ({n_digits}, 10, {hidden}) "
        f"= {n_digits*10*hidden/1e3:.1f}K params  (Fourier orthogonal per level, "
        f"scale 0.1/sqrt(H))",
        flush=True,
    )


def fg_v108_parameters(model: Any) -> list[Tensor]:
    """All trainable v108 factor-graph params (backbone + tree)."""
    return fg_v107_parameters(model) + [model.fg_v108_tree_codebook]


def fg_v108_state_dict(model: Any) -> dict[str, Tensor]:
    sd = dict(fg_v107_state_dict(model))
    sd["fg_v108.tree_codebook"] = model.fg_v108_tree_codebook
    return sd


# ---------------------------------------------------------------------------
# JIT training step
# ---------------------------------------------------------------------------

_JIT_V108_CACHE: dict = {}


def _compile_jit_fg_step_v108(
    model: Any,
    opt: Any,
    K: int,
    B: int,
    factor_aux_weight: float = V108_FACTOR_AUX_WEIGHT,
    calib_weight: float = V108_CALIB_WEIGHT,
    var_loss_weight: float = V108_VAR_LOSS_WEIGHT,
    hard_breath_level: bool = V108_HARD_BREATH_LEVEL,
    n_max: int = V108_N_MAX,
    f_max: int = V108_F_MAX,
    n_digits: int = V108_N_DIGITS,
    grad_clip: float = 1.0,
):
    """Compile training step for v108 (per-breath per-level CE)."""
    key = ("v108", id(model), id(opt), int(K), int(B), float(factor_aux_weight),
           float(calib_weight), float(var_loss_weight),
           bool(hard_breath_level),
           int(n_max), int(f_max), int(n_digits), float(grad_clip))
    if key in _JIT_V108_CACHE:
        return _JIT_V108_CACHE[key]

    fw     = float(factor_aux_weight)
    aw     = float(calib_weight)
    vw     = float(var_loss_weight)
    gc     = float(grad_clip)
    params = opt.params

    mode = "HARD" if hard_breath_level else "SOFT"
    print(
        f"[JIT] compile v108 fg step: K={K} B={B} n_digits={n_digits} "
        f"mode={mode} vw={vw} fw={fw} aw={aw} gc={gc}...",
        flush=True,
    )

    @TinyJit
    def _step(
        domain_init: Tensor,       # (B, N_MAX, 200)
        node_kinds: Tensor,
        staging_mask: Tensor,
        head_op_mask: Tensor,
        gold_digits: Tensor,       # (B, N_MAX, n_digits) int, MSD-first
        gold_bins: Tensor,         # (B, N_MAX) int — for calibration target
        observed_mask: Tensor,
        factor_gold_bin: Tensor,
        factor_valid: Tensor,
    ):
        opt.zero_grad()

        tree_lh, var_lh, fac_lh, calib_h = fg_breathing_forward_v108(
            model, domain_init, node_kinds, staging_mask, head_op_mask,
            K=K, n_max=n_max, f_max=f_max, n_digits=n_digits,
        )

        # --- Tree CE (per-breath per-level) ---
        unobs_float = (1 - observed_mask.cast(dtypes.float)).reshape(B * n_max)
        n_unobs_sum = unobs_float.sum() + 1e-8

        # gold_digits: (B, N_MAX, n_digits) → flat (B*N_MAX, n_digits)
        gd_flat = gold_digits.cast(dtypes.int).reshape(B * n_max, n_digits)

        var_loss_sum   = Tensor.zeros((), dtype=dtypes.float).contiguous()
        var_weight_sum = 0.0
        per_breath_ce_t: list[Tensor] = []

        for k, tree_logits_k in enumerate(tree_lh):
            weight_k = 1.0 + float(k) / float(max(K - 1, 1))
            # tree_logits_k: (B, N_MAX, n_digits, 10) → (B*N_MAX, n_digits, 10)
            tl_flat = tree_logits_k.reshape(B * n_max, n_digits, 10)

            # Per-level CE
            if hard_breath_level:
                if k < n_digits:
                    # Only level k is supervised at breath k
                    levels_to_use = [k]
                else:
                    # Refinement breaths: all levels
                    levels_to_use = list(range(n_digits))
            else:
                # Soft: all levels every breath
                levels_to_use = list(range(n_digits))

            ce_breath_sum = Tensor.zeros((), dtype=dtypes.float).contiguous()
            for level in levels_to_use:
                level_logits = tl_flat[:, level, :]                  # (B*N_MAX, 10)
                level_gold   = gd_flat[:, level]                     # (B*N_MAX,)
                log_probs    = level_logits.log_softmax(axis=-1)
                gold_oh      = level_gold.one_hot(10).cast(log_probs.dtype)
                nll          = -(log_probs * gold_oh).sum(axis=-1)
                masked_nll   = nll * unobs_float.cast(nll.dtype)
                ce_level     = masked_nll.sum() / n_unobs_sum
                ce_breath_sum = ce_breath_sum + ce_level
            ce_k = ce_breath_sum / float(len(levels_to_use))
            per_breath_ce_t.append(ce_k)
            var_loss_sum   = var_loss_sum + ce_k * weight_k
            var_weight_sum += weight_k

        var_loss = var_loss_sum / float(var_weight_sum)

        # --- Factor aux (200-way, same as v107) ---
        n_valid_factors  = factor_valid.cast(dtypes.float).sum() + 1e-8
        gold_fac_flat    = factor_gold_bin.cast(dtypes.int).reshape(B * f_max)
        gold_fac_oh      = gold_fac_flat.one_hot(200).cast(dtypes.float)
        valid_flat       = factor_valid.cast(dtypes.float).reshape(B * f_max)

        factor_aux_sum   = Tensor.zeros((), dtype=dtypes.float).contiguous()
        factor_aux_w_sum = 0.0
        for k_aux, fac_logits_k in enumerate(fac_lh):
            w_k_aux   = 1.0 + float(k_aux) / float(max(K - 1, 1))
            fac_flat  = fac_logits_k.reshape(B * f_max, 200)
            fac_lp    = fac_flat.log_softmax(axis=-1)
            fac_nll   = -(fac_lp * gold_fac_oh).sum(axis=-1)
            fac_masked= fac_nll * valid_flat
            fac_ce_k  = fac_masked.sum() / n_valid_factors
            factor_aux_sum   = factor_aux_sum + fac_ce_k * w_k_aux
            factor_aux_w_sum += w_k_aux
        factor_aux_loss = factor_aux_sum / float(factor_aux_w_sum)

        # --- Calibration ---
        # Use reconstructed whole-number correctness as the calibration target
        final_tree = tree_lh[-1]
        pred_digits_final = final_tree.argmax(axis=-1).detach()            # (B, N_MAX, n_digits)
        eq_per_pos        = (pred_digits_final == gold_digits.cast(dtypes.int)).cast(dtypes.float)
        eq                = eq_per_pos.prod(axis=-1)                       # (B, N_MAX)
        unobs_2d          = (1 - observed_mask.cast(dtypes.float))
        eq_unobs          = eq * unobs_2d
        n_unobs_per       = unobs_2d.sum(axis=-1) + 1e-8
        correct           = eq_unobs.sum(axis=-1) / n_unobs_per

        calib_loss_sum = Tensor.zeros((), dtype=dtypes.float).contiguous()
        for kc, calib_k in enumerate(calib_h):
            prog       = float(kc) / float(max(K - 1, 1))
            target_k   = 0.5 + (correct - 0.5) * prog
            calib_loss_sum = calib_loss_sum + ((calib_k - target_k) ** 2).mean()
        calib_loss = calib_loss_sum / float(K)

        # Metrics
        cell_acc  = (eq_unobs.sum() / (unobs_2d.sum() + 1e-8)).detach()
        query_acc = correct.mean().detach()

        # Total
        total_ce = vw * var_loss + fw * factor_aux_loss + aw * calib_loss
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
            grad_norm = (sq_sum + 1e-12).sqrt()
            clip_coef = (Tensor(gc, dtype=dtypes.float) / (grad_norm + 1e-6)).minimum(
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

    _JIT_V108_CACHE[key] = _step
    print(f"[JIT] v108 fg step ready (cache={len(_JIT_V108_CACHE)})", flush=True)
    return _step


def _compile_jit_fg_eval_v108(
    model: Any,
    K: int,
    B: int,
    n_max: int = V108_N_MAX,
    f_max: int = V108_F_MAX,
    n_digits: int = V108_N_DIGITS,
):
    """Compile eval step (forward only, return per-digit argmax + per-level logits)."""
    key = ("eval_v108", id(model), int(K), int(B), int(n_max), int(f_max), int(n_digits))
    if key in _JIT_V108_CACHE:
        return _JIT_V108_CACHE[key]

    print(f"[JIT] compile v108 fg eval: K={K} B={B} n_digits={n_digits}...", flush=True)

    @TinyJit
    def _eval(
        domain_init: Tensor,
        node_kinds: Tensor,
        staging_mask: Tensor,
        head_op_mask: Tensor,
        gold_digits: Tensor,
        observed_mask: Tensor,
    ):
        tree_lh, _, _, _ = fg_breathing_forward_v108(
            model, domain_init, node_kinds, staging_mask, head_op_mask,
            K=K, n_max=n_max, f_max=f_max, n_digits=n_digits,
        )
        final_tree = tree_lh[-1]                                # (B, N_MAX, n_digits, 10)
        pred_digits = final_tree.argmax(axis=-1)                # (B, N_MAX, n_digits)
        eq_per_pos = (pred_digits == gold_digits.cast(dtypes.int)).cast(dtypes.float)
        eq = eq_per_pos.prod(axis=-1)
        unobs = (1 - observed_mask.cast(dtypes.float))
        cell_acc = (eq * unobs).sum() / (unobs.sum() + 1e-8)
        return pred_digits.realize(), cell_acc.realize()

    _JIT_V108_CACHE[key] = _eval
    print(f"[JIT] v108 eval ready (cache={len(_JIT_V108_CACHE)})", flush=True)
    return _eval
