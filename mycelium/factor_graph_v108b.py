"""v108b factor graph — digit-decomposed INPUT + tree codebook OUTPUT.

The v108 smoke showed the tree decoder works (digit_acc 70-75%) but the
per-position accuracy curve EXACTLY matches the v107 200-bin INPUT
resolution curve:

  pos0-2 (leading digits, well-resolved by bins):   96-100%
  pos3   (tens, ambiguous at ~5% log-spacing):      42-52%
  pos4   (ones, below bin resolution):              10-26%

The decoder is reading what the input contains. To unblock pos3-pos4, the
INPUT has to be digit-precise.

v108b architecture:
  Variable encoding (observed): x = Σ_l digit_embed[l, gold_digit_l]
                                ONE token per variable (no L0 collapse).
                                digit_embed: (n_digits, 10, hidden) learned.
                                Sum across positions: BERT-style superposition.
  Variable encoding (unobserved): x = Σ_l mean over digits of digit_embed[l, :]
                                  Same shape as observed (single token).
  All other forward path: identical to v108.
  Tree codebook readout: SAME as v108 (5 × 10 entries, Fourier orthogonal).

Key claim under test: with digit-precise input AND tree-structured output,
the breathing transformer can propagate per-digit information through K=8
breaths and recover ALL 5 digits, not just the leading 2-3.

Env vars:
  V108B_TASK=1                    — enable v108b forward path
  V108B_K_MAX=8
  V108B_N_DIGITS=5
  V108B_HARD_BREATH_LEVEL=0       — soft (default) or hard
  V108B_VAR_LOSS_WEIGHT=1.0
  V108B_DIGIT_EMBED_SCALE=0.1     — init scale for digit_embed (small Gaussian)
"""
from __future__ import annotations

import os
from typing import Any

import numpy as np
from tinygrad import Tensor, dtypes
from tinygrad.engine.jit import TinyJit

from mycelium.config import Config
from mycelium.factor_graph_v100 import (
    fg_layer_forward_v100,
    V100_N_HEADS,
)
from mycelium.factor_graph_v107 import (
    V107_N_MAX, V107_F_MAX, V107_N_HEADS,
    V107_CALIB_WEIGHT, V107_FACTOR_AUX_WEIGHT,
    V107_CODEBOOK_N, V107_IB_CENTROIDS,
    get_bin_values,
)
from mycelium.factor_graph_v108 import (
    V108_K_MAX, V108_N_DIGITS,
    _fourier_orthogonal_init,
    values_to_digits_msd, digits_to_value_msd, bins_to_digits_msd,
)
from mycelium.factor_graph_v104 import load_ib_centroids

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

V108B_TASK              = int(os.environ.get("V108B_TASK", "0")) > 0
V108B_K_MAX             = int(os.environ.get("V108B_K_MAX",             "8"))
V108B_N_DIGITS          = int(os.environ.get("V108B_N_DIGITS",          "5"))
V108B_HARD_BREATH_LEVEL = int(os.environ.get("V108B_HARD_BREATH_LEVEL", "0")) > 0
V108B_VAR_LOSS_WEIGHT   = float(os.environ.get("V108B_VAR_LOSS_WEIGHT", "1.0"))
V108B_DIGIT_EMBED_SCALE = float(os.environ.get("V108B_DIGIT_EMBED_SCALE", "0.1"))
V108B_N_MAX             = int(os.environ.get("V108B_N_MAX",             str(V107_N_MAX)))
V108B_F_MAX             = int(os.environ.get("V108B_F_MAX",             str(V107_F_MAX)))
V108B_T_MAX             = V108B_N_MAX + V108B_F_MAX
V108B_CODEBOOK_N        = int(os.environ.get("V108B_CODEBOOK_N",        str(V107_CODEBOOK_N)))
V108B_IB_CENTROIDS      = os.environ.get("V108B_IB_CENTROIDS",          V107_IB_CENTROIDS)
V108B_CALIB_WEIGHT      = float(os.environ.get("V108B_CALIB_WEIGHT",    str(V107_CALIB_WEIGHT)))
V108B_FACTOR_AUX_WEIGHT = float(os.environ.get("V108B_FACTOR_AUX_WEIGHT", str(V107_FACTOR_AUX_WEIGHT)))
V108B_N_HEADS           = V107_N_HEADS


# ---------------------------------------------------------------------------
# Embedding: digit-decomposed → ONE token per variable (sum across positions)
# ---------------------------------------------------------------------------

def _digit_to_var_token(
    digit_init: Tensor,        # (B, N_MAX, n_digits, 10) — one-hot or uniform
    digit_embed: Tensor,       # (n_digits, 10, H)
) -> Tensor:
    """Sum learned per-(position, digit) embeddings into one token per variable.

    For observed variable v: digit_init[b, v, l, :] is one-hot at gold_digit_l.
                              Token = Σ_l digit_embed[l, gold_digit_l]
    For unobserved variable v: digit_init[b, v, l, :] is uniform 1/10 (mean
                              over digit_embed[l, :] at that position).

    Both paths return shape (B, N_MAX, H) — ONE token per variable.
    """
    # digit_init: (B, N_MAX, n_digits, 10)
    # digit_embed: (n_digits, 10, H)
    # einsum: sum over (n_digits, 10) → (B, N_MAX, H)
    B, N, P, D = digit_init.shape
    H = int(digit_embed.shape[-1])
    # Reshape for matmul: (B*N*P, 10) @ (P, 10, H) … requires broadcasting
    # Simpler: collapse digit_init to (B*N, P, 10), then per-position matmul
    di_flat = digit_init.cast(dtypes.float).reshape(B * N, P, D)             # (BN, P, 10)
    de       = digit_embed.cast(dtypes.float)                                 # (P, 10, H)
    # Per-position matmul: di_flat[:, p, :] @ de[p, :, :] → (BN, H) per p
    # Use broadcasted batched matmul: di_flat.reshape(BN, P, 1, D) @ de.reshape(1, P, D, H)
    #   → (BN, P, 1, H), then sum over P
    di_b = di_flat.reshape(B * N, P, 1, D)                                     # (BN, P, 1, 10)
    de_b = de.reshape(1, P, D, H)                                              # (1, P, 10, H)
    per_pos = (di_b @ de_b).reshape(B * N, P, H)                              # (BN, P, H)
    summed  = per_pos.sum(axis=1)                                              # (BN, H)
    return summed.reshape(B, N, H)


def embed_factor_graph_v108b(
    digit_init: Tensor,        # (B, N_MAX, n_digits, 10)
    node_kinds: Tensor,         # (B, T_MAX) int
    digit_embed: Tensor,        # (n_digits, 10, H)
    var_pos_embed: Tensor,      # (N_MAX, H)
    factor_pos_embed: Tensor,   # (F_MAX, H)
    node_kind_embed: Tensor,    # (3, H)
) -> Tensor:
    """Build (B, T, H) input residual stream.

    Variables: ONE token per variable (sum of digit embeddings).
    Factors:   zero embedding + factor_pos_embed + node_kind_embed[2].
    """
    B = int(digit_init.shape[0])
    N = int(digit_init.shape[1])
    T = int(node_kinds.shape[1])
    F = T - N
    H = int(digit_embed.shape[-1])

    # Variable tokens via digit embedding sum
    var_tokens = _digit_to_var_token(digit_init, digit_embed)                  # (B, N, H)
    var_tokens = var_tokens + var_pos_embed.cast(var_tokens.dtype).reshape(1, N, H)

    # Factor tokens: zero (no observed value) + factor_pos_embed
    fac_zeros  = Tensor.zeros((B, F, H), dtype=var_tokens.dtype).contiguous()
    fac_tokens = fac_zeros + factor_pos_embed.cast(var_tokens.dtype).reshape(1, F, H)

    # node_kind_embed: (3, H) — 0=variable, 1=factor (or 2, depending on data); index by node_kinds
    nk_oh = node_kinds.cast(dtypes.int).clip(0, 2).one_hot(3).cast(var_tokens.dtype)
    nk_v  = nk_oh @ node_kind_embed.cast(var_tokens.dtype)                     # (B, T, H)

    # Concatenate var + factor along token axis, then add node_kind
    full = var_tokens.cat(fac_tokens, dim=1) + nk_v                            # (B, T, H)
    return full


# ---------------------------------------------------------------------------
# Iterative prefill loop (v108b)
# ---------------------------------------------------------------------------

def fg_breathing_forward_v108b(
    model: Any,
    digit_init: Tensor,         # (B, N_MAX, n_digits, 10)
    node_kinds: Tensor,
    staging_mask: Tensor,
    head_op_mask: Tensor,
    K: int,
    n_max: int = V108B_N_MAX,
    f_max: int = V108B_F_MAX,
    n_digits: int = V108B_N_DIGITS,
) -> tuple[list[Tensor], list[Tensor], list[Tensor]]:
    """Run K iterative-prefill breaths with digit-decomposed input.

    Returns:
      tree_logits_history    : K × (B, N_MAX, n_digits, 10)
      factor_logits_history  : K × (B, F_MAX, n_digits, 10) — factor decoded via tree too
      calib_history          : K × (B,)
    """
    assert hasattr(model, "fg_v108b_digit_embed"), \
        "model has no v108b params; was attach_fg_params_v108b called?"

    digit_embed      = model.fg_v108b_digit_embed       # (n_digits, 10, H)
    var_pos_embed    = model.fg_v108b_var_pos_embed
    factor_pos_embed = model.fg_v108b_factor_pos_embed
    node_kind_embed  = model.fg_v108b_node_kind_embed
    breath_embed     = model.fg_v108b_breath_embed
    delta_gate       = model.fg_v108b_delta_gate
    calib_head_w     = model.fg_v108b_calib_head_w
    calib_head_b     = model.fg_v108b_calib_head_b

    semantic_codebook    = model.fg_v108b_semantic_codebook
    delta_gate_quant     = model.fg_v108b_delta_gate_quant
    temperature          = model.fg_v108b_temperature

    tree_codebook = model.fg_v108b_tree_codebook        # (n_digits, 10, H)
    H = int(tree_codebook.shape[-1])

    B = int(digit_init.shape[0])
    T = n_max + f_max

    x = embed_factor_graph_v108b(
        digit_init, node_kinds, digit_embed,
        var_pos_embed, factor_pos_embed, node_kind_embed,
    )
    x = x.cast(dtypes.half) if x.dtype != dtypes.half else x

    layers = list(model.block.layers)
    K_max  = int(breath_embed.shape[0])
    assert K <= K_max, f"K={K} > K_max={K_max}"

    from mycelium.breathing import _layernorm

    tree_logits_history   = []
    factor_logits_history = []
    calib_history         = []

    tree_cb_flat = tree_codebook.reshape(n_digits * 10, H)

    for k in range(K):
        be_k  = breath_embed[k].reshape(1, 1, -1).cast(x.dtype)
        x_in  = x + be_k
        x_pre = x

        stk      = staging_mask[:, k, :, :]
        stk_h    = stk.reshape(B, 1, T, T).expand(B, V108B_N_HEADS, T, T)
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

        x_ln = _layernorm(x, model.ln_f_g, model.ln_f_b,
                          model.cfg.layer_norm_eps).cast(dtypes.float)

        var_x = x_ln[:, :n_max, :]
        fac_x = x_ln[:, n_max:n_max + f_max, :]

        # Tree readout for variables and factors (both decoded as digits)
        var_tree  = (var_x @ tree_cb_flat.T.cast(dtypes.float)).reshape(
            B, n_max, n_digits, 10
        )
        fac_tree  = (fac_x @ tree_cb_flat.T.cast(dtypes.float)).reshape(
            B, f_max, n_digits, 10
        )
        tree_logits_history.append(var_tree)
        factor_logits_history.append(fac_tree)

        pool        = x_ln.mean(axis=1)
        calib_logit = pool @ calib_head_w.cast(dtypes.float) + calib_head_b.cast(dtypes.float)
        calib_history.append(calib_logit.reshape(-1).sigmoid())

    return tree_logits_history, factor_logits_history, calib_history


# ---------------------------------------------------------------------------
# Parameter attachment
# ---------------------------------------------------------------------------

def attach_fg_params_v108b(
    model: Any,
    hidden: int,
    n_max: int = V108B_N_MAX,
    f_max: int = V108B_F_MAX,
    k_max: int | None = None,
    n_digits: int = V108B_N_DIGITS,
    n_code: int = V108B_CODEBOOK_N,
    ib_centroids_path: str = V108B_IB_CENTROIDS,
) -> None:
    """Attach v108b factor-graph parameters."""
    if k_max is None:
        k_max = V108B_K_MAX

    rng = np.random.RandomState(40001)

    # --- Digit embedding (n_digits, 10, H): learned per-(position, digit) ---
    # Small Gaussian init so the BERT-style sum starts near zero (residual safe)
    di = (rng.randn(n_digits, 10, hidden) * V108B_DIGIT_EMBED_SCALE).astype(np.float32)
    model.fg_v108b_digit_embed = Tensor(di, dtype=dtypes.float).contiguous()

    # --- Tree codebook (n_digits, 10, H): Fourier orthogonal per level ---
    cb = np.zeros((n_digits, 10, hidden), dtype=np.float32)
    for level in range(n_digits):
        cb[level] = _fourier_orthogonal_init(10, hidden, seed=level * 17 + 31)
    model.fg_v108b_tree_codebook = Tensor(cb, dtype=dtypes.float).contiguous()

    # --- Pos embeddings (same scale as v107) ---
    vp = (rng.randn(n_max, hidden) * 0.02).astype(np.float32)
    model.fg_v108b_var_pos_embed = Tensor(vp, dtype=dtypes.float).contiguous()
    fp = (rng.randn(f_max, hidden) * 0.02).astype(np.float32)
    model.fg_v108b_factor_pos_embed = Tensor(fp, dtype=dtypes.float).contiguous()
    nk = (rng.randn(3, hidden) * 0.02).astype(np.float32)
    model.fg_v108b_node_kind_embed = Tensor(nk, dtype=dtypes.float).contiguous()

    # --- Calibration head ---
    cw = (rng.randn(hidden, 1) * 0.02).astype(np.float32)
    model.fg_v108b_calib_head_w = Tensor(cw, dtype=dtypes.float).contiguous()
    model.fg_v108b_calib_head_b = Tensor.zeros((1,), dtype=dtypes.float).contiguous()

    # --- Per-breath embedding (orthonormal × 0.5) ---
    rng_be = np.random.RandomState(40002)
    raw_be = rng_be.randn(max(k_max, hidden), hidden).astype(np.float32)
    q_be, _ = np.linalg.qr(raw_be)
    be = (q_be[:k_max] * 0.5).astype(np.float32)
    model.fg_v108b_breath_embed = Tensor(be, dtype=dtypes.float).contiguous()

    # --- Per-breath delta gate: init 1.0 ---
    model.fg_v108b_delta_gate = Tensor.ones((k_max,), dtype=dtypes.float).contiguous()

    # --- IB semantic codebook ---
    cb_ib = load_ib_centroids(ib_centroids_path, n_code=n_code, hidden=hidden)
    model.fg_v108b_semantic_codebook = Tensor(cb_ib, dtype=dtypes.float).contiguous()
    model.fg_v108b_delta_gate_quant = Tensor.zeros((k_max,), dtype=dtypes.float).contiguous()
    model.fg_v108b_temperature = Tensor(
        np.array([1.0], dtype=np.float32), dtype=dtypes.float
    ).contiguous()

    print(
        f"[v108b] params attached: digit_embed=({n_digits},10,{hidden})={n_digits*10*hidden/1e3:.1f}K  "
        f"tree_codebook=({n_digits},10,{hidden})={n_digits*10*hidden/1e3:.1f}K  "
        f"semantic_codebook=({n_code},{hidden})  "
        f"breath_embed=({k_max},{hidden})  T={n_max+f_max}",
        flush=True,
    )


def fg_v108b_parameters(model: Any) -> list[Tensor]:
    return [
        model.fg_v108b_digit_embed,
        model.fg_v108b_tree_codebook,
        model.fg_v108b_var_pos_embed,
        model.fg_v108b_factor_pos_embed,
        model.fg_v108b_node_kind_embed,
        model.fg_v108b_calib_head_w,
        model.fg_v108b_calib_head_b,
        model.fg_v108b_breath_embed,
        model.fg_v108b_delta_gate,
        model.fg_v108b_semantic_codebook,
        model.fg_v108b_delta_gate_quant,
        model.fg_v108b_temperature,
    ]


def fg_v108b_state_dict(model: Any) -> dict[str, Tensor]:
    return {
        "fg_v108b.digit_embed":       model.fg_v108b_digit_embed,
        "fg_v108b.tree_codebook":     model.fg_v108b_tree_codebook,
        "fg_v108b.var_pos_embed":     model.fg_v108b_var_pos_embed,
        "fg_v108b.factor_pos_embed":  model.fg_v108b_factor_pos_embed,
        "fg_v108b.node_kind_embed":   model.fg_v108b_node_kind_embed,
        "fg_v108b.calib_head_w":      model.fg_v108b_calib_head_w,
        "fg_v108b.calib_head_b":      model.fg_v108b_calib_head_b,
        "fg_v108b.breath_embed":      model.fg_v108b_breath_embed,
        "fg_v108b.delta_gate":        model.fg_v108b_delta_gate,
        "fg_v108b.semantic_codebook": model.fg_v108b_semantic_codebook,
        "fg_v108b.delta_gate_quant":  model.fg_v108b_delta_gate_quant,
        "fg_v108b.temperature":       model.fg_v108b_temperature,
    }


# ---------------------------------------------------------------------------
# Build digit_init from gold_digits + observed_mask
# ---------------------------------------------------------------------------

def build_digit_init(
    gold_digits_np: np.ndarray,   # (B, N_MAX, n_digits) int
    observed_mask_np: np.ndarray, # (B, N_MAX) int
    n_digits: int = V108B_N_DIGITS,
) -> np.ndarray:
    """Construct (B, N_MAX, n_digits, 10) one-hot/uniform input.

    Observed variables: one-hot at gold_digit per position.
    Unobserved variables: uniform 1/10 per position.
    """
    B, N = gold_digits_np.shape[:2]
    di = np.full((B, N, n_digits, 10), 0.1, dtype=np.float32)  # uniform
    for b in range(B):
        for v in range(N):
            if observed_mask_np[b, v] == 0:
                continue
            for l in range(n_digits):
                d = int(gold_digits_np[b, v, l])
                if 0 <= d < 10:
                    di[b, v, l] = 0.0
                    di[b, v, l, d] = 1.0
    return di


# ---------------------------------------------------------------------------
# JIT training step
# ---------------------------------------------------------------------------

_JIT_V108B_CACHE: dict = {}


def _compile_jit_fg_step_v108b(
    model: Any,
    opt: Any,
    K: int,
    B: int,
    factor_aux_weight: float = V108B_FACTOR_AUX_WEIGHT,
    calib_weight: float = V108B_CALIB_WEIGHT,
    var_loss_weight: float = V108B_VAR_LOSS_WEIGHT,
    hard_breath_level: bool = V108B_HARD_BREATH_LEVEL,
    n_max: int = V108B_N_MAX,
    f_max: int = V108B_F_MAX,
    n_digits: int = V108B_N_DIGITS,
    grad_clip: float = 1.0,
):
    key = ("v108b", id(model), id(opt), int(K), int(B), float(factor_aux_weight),
           float(calib_weight), float(var_loss_weight),
           bool(hard_breath_level),
           int(n_max), int(f_max), int(n_digits), float(grad_clip))
    if key in _JIT_V108B_CACHE:
        return _JIT_V108B_CACHE[key]

    fw, aw, vw, gc = float(factor_aux_weight), float(calib_weight), \
                     float(var_loss_weight), float(grad_clip)
    params = opt.params

    mode = "HARD" if hard_breath_level else "SOFT"
    print(f"[JIT] compile v108b fg step: K={K} B={B} n_digits={n_digits} "
          f"mode={mode} vw={vw} fw={fw} aw={aw} gc={gc}...", flush=True)

    @TinyJit
    def _step(
        digit_init: Tensor,         # (B, N_MAX, n_digits, 10) — digit-decomposed input
        node_kinds: Tensor,
        staging_mask: Tensor,
        head_op_mask: Tensor,
        gold_digits: Tensor,         # (B, N_MAX, n_digits)
        observed_mask: Tensor,
        factor_gold_digits: Tensor,  # (B, F_MAX, n_digits) — factor gold (decomposed)
        factor_valid: Tensor,
    ):
        opt.zero_grad()

        tree_lh, fac_lh, calib_h = fg_breathing_forward_v108b(
            model, digit_init, node_kinds, staging_mask, head_op_mask,
            K=K, n_max=n_max, f_max=f_max, n_digits=n_digits,
        )

        unobs_float = (1 - observed_mask.cast(dtypes.float)).reshape(B * n_max)
        n_unobs_sum = unobs_float.sum() + 1e-8
        gd_flat     = gold_digits.cast(dtypes.int).reshape(B * n_max, n_digits)

        var_loss_sum   = Tensor.zeros((), dtype=dtypes.float).contiguous()
        var_weight_sum = 0.0
        per_breath_ce_t: list[Tensor] = []

        for k, tree_logits_k in enumerate(tree_lh):
            weight_k = 1.0 + float(k) / float(max(K - 1, 1))
            tl_flat  = tree_logits_k.reshape(B * n_max, n_digits, 10)

            if hard_breath_level:
                levels_to_use = [k] if k < n_digits else list(range(n_digits))
            else:
                levels_to_use = list(range(n_digits))

            ce_breath_sum = Tensor.zeros((), dtype=dtypes.float).contiguous()
            for level in levels_to_use:
                level_logits = tl_flat[:, level, :]
                level_gold   = gd_flat[:, level]
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

        # --- Factor aux (tree CE on factors too) ---
        n_valid_factors = factor_valid.cast(dtypes.float).sum() + 1e-8
        fg_flat   = factor_gold_digits.cast(dtypes.int).reshape(B * f_max, n_digits)
        valid_flat = factor_valid.cast(dtypes.float).reshape(B * f_max)

        factor_aux_sum   = Tensor.zeros((), dtype=dtypes.float).contiguous()
        factor_aux_w_sum = 0.0
        for k_aux, fac_logits_k in enumerate(fac_lh):
            w_k_aux = 1.0 + float(k_aux) / float(max(K - 1, 1))
            ftl     = fac_logits_k.reshape(B * f_max, n_digits, 10)
            ce_fbreath_sum = Tensor.zeros((), dtype=dtypes.float).contiguous()
            for level in range(n_digits):
                lev_log = ftl[:, level, :].log_softmax(axis=-1)
                lev_gold = fg_flat[:, level]
                gold_oh = lev_gold.one_hot(10).cast(lev_log.dtype)
                nll = -(lev_log * gold_oh).sum(axis=-1)
                masked = nll * valid_flat
                ce_fbreath_sum = ce_fbreath_sum + (masked.sum() / n_valid_factors)
            ce_fbreath = ce_fbreath_sum / float(n_digits)
            factor_aux_sum   = factor_aux_sum + ce_fbreath * w_k_aux
            factor_aux_w_sum += w_k_aux
        factor_aux_loss = factor_aux_sum / float(factor_aux_w_sum)

        # --- Calibration ---
        final_tree = tree_lh[-1]
        pred_digits_final = final_tree.argmax(axis=-1).detach()
        eq_per_pos = (pred_digits_final == gold_digits.cast(dtypes.int)).cast(dtypes.float)
        eq         = eq_per_pos.prod(axis=-1)
        unobs_2d   = (1 - observed_mask.cast(dtypes.float))
        eq_unobs   = eq * unobs_2d
        n_unobs_per = unobs_2d.sum(axis=-1) + 1e-8
        correct     = eq_unobs.sum(axis=-1) / n_unobs_per

        calib_loss_sum = Tensor.zeros((), dtype=dtypes.float).contiguous()
        for kc, calib_k in enumerate(calib_h):
            prog     = float(kc) / float(max(K - 1, 1))
            target_k = 0.5 + (correct - 0.5) * prog
            calib_loss_sum = calib_loss_sum + ((calib_k - target_k) ** 2).mean()
        calib_loss = calib_loss_sum / float(K)

        cell_acc  = (eq_unobs.sum() / (unobs_2d.sum() + 1e-8)).detach()
        query_acc = correct.mean().detach()

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

    _JIT_V108B_CACHE[key] = _step
    print(f"[JIT] v108b fg step ready (cache={len(_JIT_V108B_CACHE)})", flush=True)
    return _step


def _compile_jit_fg_eval_v108b(
    model: Any,
    K: int,
    B: int,
    n_max: int = V108B_N_MAX,
    f_max: int = V108B_F_MAX,
    n_digits: int = V108B_N_DIGITS,
):
    key = ("eval_v108b", id(model), int(K), int(B), int(n_max), int(f_max), int(n_digits))
    if key in _JIT_V108B_CACHE:
        return _JIT_V108B_CACHE[key]

    print(f"[JIT] compile v108b fg eval: K={K} B={B} n_digits={n_digits}...", flush=True)

    @TinyJit
    def _eval(
        digit_init: Tensor,
        node_kinds: Tensor,
        staging_mask: Tensor,
        head_op_mask: Tensor,
        gold_digits: Tensor,
        observed_mask: Tensor,
    ):
        tree_lh, _, _ = fg_breathing_forward_v108b(
            model, digit_init, node_kinds, staging_mask, head_op_mask,
            K=K, n_max=n_max, f_max=f_max, n_digits=n_digits,
        )
        final_tree = tree_lh[-1]
        pred_digits = final_tree.argmax(axis=-1)
        eq_per_pos = (pred_digits == gold_digits.cast(dtypes.int)).cast(dtypes.float)
        eq = eq_per_pos.prod(axis=-1)
        unobs = (1 - observed_mask.cast(dtypes.float))
        cell_acc = (eq * unobs).sum() / (unobs.sum() + 1e-8)
        return pred_digits.realize(), cell_acc.realize()

    _JIT_V108B_CACHE[key] = _eval
    print(f"[JIT] v108b eval ready (cache={len(_JIT_V108B_CACHE)})", flush=True)
    return _eval
