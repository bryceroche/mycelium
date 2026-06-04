"""v109 factor graph — v108 base + 512d LoRA waist + alternation.

Architecture:
  v108 substrate (single-token-per-variable, K=8 breaths, tree codebook output)
  + 1024 → 512 → 1024 LoRA-init waist (W_expand zero-init for byte-safe warm-start)
  + ALTERNATION GATE: waist active on EVEN breaths (commit), bypassed on ODD breaths

The alternation cycle (K=8):
  B0  COMMIT    (collapse: waist 1024→512→1024 active)
  B1  EXPAND    (waist bypassed, residual passes through unchanged)
  B2  COMMIT
  B3  EXPAND
  B4  COMMIT
  B5  EXPAND
  B6  COMMIT
  B7  EXPAND    (final readout reads UN-compressed final state)

Rationale: even breaths force commitment through the 512d bottleneck (JPEG
quantize step); odd breaths let beliefs propagate freely at full 1024d without
further compression. Tests the hypothesis that K=8 of REPEATED identical
compressions (v101 style) is worse than 4 commits + 4 propagations.

Hypothesis (testable via K-sweep on v109 ckpt):
  v108 K-sweep showed pos4 hard DECREASES with K (0.191 → 0.138 → 0.132)
  If alternation lets carry signal propagate cleanly through expand phases,
    v109 K-sweep should show pos4 hard rising or at least staying flat with K
  If pos4 hard still falls with K, alternation is decorative

Env vars:
  V109_TASK=1                  — enable v109 forward path
  V109_K_MAX=8                 — number of breaths
  V109_N_DIGITS=5
  V109_WAIST_DIM=512           — bottleneck width
  V109_ALTERNATION=1           — if 1: waist on even breaths only.
                                 If 0: waist on every breath (v101 style ablation)
"""
from __future__ import annotations

import os
from typing import Any

import numpy as np
from tinygrad import Tensor, dtypes
from tinygrad.engine.jit import TinyJit

from mycelium.config import Config
from mycelium.factor_graph_v100 import (
    embed_factor_graph_v100_aligned,
    fg_layer_forward_v100,
)
from mycelium.factor_graph_v107 import V107_N_HEADS, get_bin_values
from mycelium.factor_graph_v108 import (
    V108_N_MAX, V108_F_MAX, V108_N_HEADS, V108_N_DIGITS, V108_K_MAX,
    V108_CODEBOOK_N, V108_IB_CENTROIDS,
    V108_CALIB_WEIGHT, V108_FACTOR_AUX_WEIGHT,
    attach_fg_params_v108, fg_v108_parameters, fg_v108_state_dict,
    bins_to_digits_msd, values_to_digits_msd,
)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

V109_TASK              = int(os.environ.get("V109_TASK", "0")) > 0
V109_K_MAX             = int(os.environ.get("V109_K_MAX",             "8"))
V109_N_DIGITS          = int(os.environ.get("V109_N_DIGITS",          "5"))
V109_WAIST_DIM         = int(os.environ.get("V109_WAIST_DIM",         "512"))
V109_ALTERNATION       = int(os.environ.get("V109_ALTERNATION",       "1")) > 0
V109_HARD_BREATH_LEVEL = int(os.environ.get("V109_HARD_BREATH_LEVEL", "0")) > 0
V109_VAR_LOSS_WEIGHT   = float(os.environ.get("V109_VAR_LOSS_WEIGHT", "1.0"))
V109_N_MAX             = int(os.environ.get("V109_N_MAX",             str(V108_N_MAX)))
V109_F_MAX             = int(os.environ.get("V109_F_MAX",             str(V108_F_MAX)))
V109_T_MAX             = V109_N_MAX + V109_F_MAX
V109_CODEBOOK_N        = int(os.environ.get("V109_CODEBOOK_N",        str(V108_CODEBOOK_N)))
V109_IB_CENTROIDS      = os.environ.get("V109_IB_CENTROIDS",          V108_IB_CENTROIDS)
V109_CALIB_WEIGHT      = float(os.environ.get("V109_CALIB_WEIGHT",    str(V108_CALIB_WEIGHT)))
V109_FACTOR_AUX_WEIGHT = float(os.environ.get("V109_FACTOR_AUX_WEIGHT", str(V108_FACTOR_AUX_WEIGHT)))
V109_N_HEADS           = V108_N_HEADS


# ---------------------------------------------------------------------------
# Waist apply (LoRA-style residual addition, identity at zero-init)
# ---------------------------------------------------------------------------

def _apply_waist_v109(
    h: Tensor,           # (B, T, H)
    W_compress: Tensor,  # (H, waist_dim)
    b_compress: Tensor,  # (waist_dim,)
    W_expand: Tensor,    # (waist_dim, H)  zero-init
    b_expand: Tensor,    # (H,)
) -> Tensor:
    """LoRA-init waist: 1024 → waist_dim → 1024, added as residual.

    At init (W_expand = 0): output = h (byte-identical to no-waist).
    """
    wc = W_compress.cast(h.dtype)
    bc = b_compress.reshape(1, 1, -1).cast(h.dtype)
    we = W_expand.cast(h.dtype)
    be = b_expand.reshape(1, 1, -1).cast(h.dtype)
    z = (h @ wc + bc).gelu()             # (B, T, waist_dim)
    delta = z @ we + be                  # (B, T, H)
    return h + delta


# ---------------------------------------------------------------------------
# Forward pass (v108 substrate + waist + alternation)
# ---------------------------------------------------------------------------

def fg_breathing_forward_v109(
    model: Any,
    domain_init: Tensor,
    node_kinds: Tensor,
    staging_mask: Tensor,
    head_op_mask: Tensor,
    K: int,
    n_max: int = V109_N_MAX,
    f_max: int = V109_F_MAX,
    n_digits: int = V109_N_DIGITS,
    alternation: bool = V109_ALTERNATION,
) -> tuple[list[Tensor], list[Tensor], list[Tensor], list[Tensor]]:
    """Run K iterative-prefill breaths with alternating waist.

    Returns: (tree_logits_history, var_logits_history, factor_logits_history, calib_history)
    """
    assert hasattr(model, "fg_v107_domain_codebook"), \
        "model has no v107 backbone params"
    assert hasattr(model, "fg_v108_tree_codebook"), \
        "model has no v108 tree codebook"
    assert hasattr(model, "fg_v109_W_compress"), \
        "model has no v109 waist params"

    domain_codebook  = model.fg_v107_domain_codebook
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

    tree_codebook = model.fg_v108_tree_codebook  # (n_digits, 10, H)

    # v109 waist params
    W_compress = model.fg_v109_W_compress
    b_compress = model.fg_v109_b_compress
    W_expand   = model.fg_v109_W_expand
    b_expand   = model.fg_v109_b_expand

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

    tree_cb_flat = tree_codebook.reshape(n_digits * 10, H)

    for k in range(K):
        be_k  = breath_embed[k].reshape(1, 1, -1).cast(x.dtype)
        x_in  = x + be_k
        x_pre = x

        stk      = staging_mask[:, k, :, :]
        stk_h    = stk.reshape(B, 1, T, T).expand(B, V109_N_HEADS, T, T)
        combined = stk_h.cast(x.dtype) + head_op_mask.cast(x.dtype)

        h = x_in
        for layer in layers[:4]:
            h = fg_layer_forward_v100(layer, h, combined)

        # IB semantic codebook (kept; gate trained to ~0 in v108, retained here)
        cb  = semantic_codebook.cast(h.dtype)
        tmp = temperature.cast(h.dtype)
        scores   = h @ cb.T / tmp.reshape(1, 1, 1)
        weights  = scores.clip(-1e4, 1e4).softmax(-1)
        recon    = weights @ cb
        quantize = recon - h
        gate_quant_k = delta_gate_quant[k].cast(h.dtype).reshape(1, 1, 1)
        h_quant = h + gate_quant_k * quantize

        # === V109 ALTERNATION: waist active on EVEN breaths only ===
        is_commit_breath = (k % 2 == 0)
        if not alternation or is_commit_breath:
            # COMMIT: apply waist (compress-then-expand LoRA correction)
            h_quant = _apply_waist_v109(
                h_quant, W_compress, b_compress, W_expand, b_expand,
            )
        # else: EXPAND breath — waist bypassed, h_quant passes through

        gate_k = delta_gate[k].cast(h_quant.dtype).reshape(1, 1, 1)
        delta  = h_quant - x_pre
        x      = x_pre + gate_k * delta

        # Readout (same as v108)
        x_ln = _layernorm(x, model.ln_f_g, model.ln_f_b,
                          model.cfg.layer_norm_eps).cast(dtypes.float)
        var_x = x_ln[:, :n_max, :]

        tree_logits_flat = var_x @ tree_cb_flat.T.cast(dtypes.float)
        tree_logits_k    = tree_logits_flat.reshape(B, n_max, n_digits, 10)
        tree_logits_history.append(tree_logits_k)

        var_logits_k = var_x @ domain_codebook.T.cast(dtypes.float)
        var_logits_history.append(var_logits_k)

        fac_x = x_ln[:, n_max:n_max + f_max, :]
        fac_logits_k = fac_x @ domain_codebook.T.cast(dtypes.float)
        factor_logits_history.append(fac_logits_k)

        pool        = x_ln.mean(axis=1)
        calib_logit = pool @ calib_head_w.cast(dtypes.float) + calib_head_b.cast(dtypes.float)
        calib_history.append(calib_logit.reshape(-1).sigmoid())

    return tree_logits_history, var_logits_history, factor_logits_history, calib_history


# ---------------------------------------------------------------------------
# Parameter attachment
# ---------------------------------------------------------------------------

def attach_fg_params_v109(
    model: Any,
    hidden: int,
    n_max: int = V109_N_MAX,
    f_max: int = V109_F_MAX,
    k_max: int | None = None,
    n_digits: int = V109_N_DIGITS,
    n_code: int = V109_CODEBOOK_N,
    ib_centroids_path: str = V109_IB_CENTROIDS,
    waist_dim: int = V109_WAIST_DIM,
) -> None:
    """Attach v108 backbone + v109 waist params.

    Waist params:
      fg_v109_W_compress  (H, waist_dim)         small Kaiming
      fg_v109_b_compress  (waist_dim,)           zero
      fg_v109_W_expand    (waist_dim, H)         ZERO (identity at init)
      fg_v109_b_expand    (H,)                   zero
    """
    if k_max is None:
        k_max = V109_K_MAX

    # Attach v108 backbone (v107 backbone + tree codebook)
    attach_fg_params_v108(
        model, hidden=hidden,
        n_max=n_max, f_max=f_max, k_max=k_max,
        n_digits=n_digits, n_code=n_code, ib_centroids_path=ib_centroids_path,
    )

    rng = np.random.RandomState(50001)

    # Waist params, LoRA init
    wc = (rng.randn(hidden, waist_dim) * (1.0 / np.sqrt(hidden))).astype(np.float32)
    model.fg_v109_W_compress = Tensor(wc, dtype=dtypes.float).contiguous()
    model.fg_v109_b_compress = Tensor.zeros((waist_dim,), dtype=dtypes.float).contiguous()
    # W_expand zero-init → identity at step 0, byte-safe warm-start
    we = np.zeros((waist_dim, hidden), dtype=np.float32)
    model.fg_v109_W_expand = Tensor(we, dtype=dtypes.float).contiguous()
    model.fg_v109_b_expand = Tensor.zeros((hidden,), dtype=dtypes.float).contiguous()

    mode = "ALTERNATING (waist on even breaths only)" if V109_ALTERNATION else "WAIST-EVERY-BREATH (v101 style)"
    print(
        f"[v109] params attached: waist=({hidden}→{waist_dim}→{hidden}) "
        f"~{(hidden*waist_dim*2 + waist_dim + hidden)/1e6:.2f}M params  "
        f"W_expand=ZEROS (byte-safe warm-start)  mode={mode}",
        flush=True,
    )


def fg_v109_parameters(model: Any) -> list[Tensor]:
    return fg_v108_parameters(model) + [
        model.fg_v109_W_compress,
        model.fg_v109_b_compress,
        model.fg_v109_W_expand,
        model.fg_v109_b_expand,
    ]


def fg_v109_state_dict(model: Any) -> dict[str, Tensor]:
    sd = dict(fg_v108_state_dict(model))
    sd["fg_v109.W_compress"] = model.fg_v109_W_compress
    sd["fg_v109.b_compress"] = model.fg_v109_b_compress
    sd["fg_v109.W_expand"]   = model.fg_v109_W_expand
    sd["fg_v109.b_expand"]   = model.fg_v109_b_expand
    return sd


# ---------------------------------------------------------------------------
# JIT training step
# ---------------------------------------------------------------------------

_JIT_V109_CACHE: dict = {}


def _compile_jit_fg_step_v109(
    model: Any,
    opt: Any,
    K: int,
    B: int,
    factor_aux_weight: float = V109_FACTOR_AUX_WEIGHT,
    calib_weight: float = V109_CALIB_WEIGHT,
    var_loss_weight: float = V109_VAR_LOSS_WEIGHT,
    hard_breath_level: bool = V109_HARD_BREATH_LEVEL,
    alternation: bool = V109_ALTERNATION,
    n_max: int = V109_N_MAX,
    f_max: int = V109_F_MAX,
    n_digits: int = V109_N_DIGITS,
    grad_clip: float = 1.0,
):
    key = ("v109", id(model), id(opt), int(K), int(B),
           float(factor_aux_weight), float(calib_weight), float(var_loss_weight),
           bool(hard_breath_level), bool(alternation),
           int(n_max), int(f_max), int(n_digits), float(grad_clip))
    if key in _JIT_V109_CACHE:
        return _JIT_V109_CACHE[key]

    fw, aw, vw, gc = float(factor_aux_weight), float(calib_weight), \
                     float(var_loss_weight), float(grad_clip)
    params = opt.params
    mode = "ALT" if alternation else "ALWAYS"
    bl_mode = "HARD" if hard_breath_level else "SOFT"
    print(f"[JIT] compile v109 fg step: K={K} B={B} n_digits={n_digits} "
          f"waist_mode={mode} breath_level={bl_mode} vw={vw} fw={fw} aw={aw}...",
          flush=True)

    @TinyJit
    def _step(
        domain_init: Tensor,
        node_kinds: Tensor,
        staging_mask: Tensor,
        head_op_mask: Tensor,
        gold_digits: Tensor,
        gold_bins: Tensor,
        observed_mask: Tensor,
        factor_gold_bin: Tensor,
        factor_valid: Tensor,
    ):
        opt.zero_grad()

        tree_lh, var_lh, fac_lh, calib_h = fg_breathing_forward_v109(
            model, domain_init, node_kinds, staging_mask, head_op_mask,
            K=K, n_max=n_max, f_max=f_max, n_digits=n_digits,
            alternation=alternation,
        )

        # Tree CE (same as v108)
        unobs_float = (1 - observed_mask.cast(dtypes.float)).reshape(B * n_max)
        n_unobs_sum = unobs_float.sum() + 1e-8
        gd_flat = gold_digits.cast(dtypes.int).reshape(B * n_max, n_digits)

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

        # Factor aux (200-way bin CE on factors, kept from v108)
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

        # Calibration
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
            prog       = float(kc) / float(max(K - 1, 1))
            target_k   = 0.5 + (correct - 0.5) * prog
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
            total_ce.realize(), healthy.realize(),
            var_loss.realize(), factor_aux_loss.realize(),
            calib_loss.realize(),
            cell_acc.realize(), query_acc.realize(),
            *(ce.realize() for ce in per_breath_ce_t),
        )

    _JIT_V109_CACHE[key] = _step
    print(f"[JIT] v109 fg step ready (cache={len(_JIT_V109_CACHE)})", flush=True)
    return _step


def _compile_jit_fg_eval_v109(
    model: Any,
    K: int,
    B: int,
    n_max: int = V109_N_MAX,
    f_max: int = V109_F_MAX,
    n_digits: int = V109_N_DIGITS,
    alternation: bool = V109_ALTERNATION,
):
    key = ("eval_v109", id(model), int(K), int(B), int(n_max), int(f_max),
           int(n_digits), bool(alternation))
    if key in _JIT_V109_CACHE:
        return _JIT_V109_CACHE[key]

    print(f"[JIT] compile v109 fg eval: K={K} B={B} n_digits={n_digits} "
          f"alternation={alternation}...", flush=True)

    @TinyJit
    def _eval(
        domain_init: Tensor,
        node_kinds: Tensor,
        staging_mask: Tensor,
        head_op_mask: Tensor,
        gold_digits: Tensor,
        observed_mask: Tensor,
    ):
        tree_lh, _, _, _ = fg_breathing_forward_v109(
            model, domain_init, node_kinds, staging_mask, head_op_mask,
            K=K, n_max=n_max, f_max=f_max, n_digits=n_digits,
            alternation=alternation,
        )
        final_tree = tree_lh[-1]
        pred_digits = final_tree.argmax(axis=-1)
        eq_per_pos = (pred_digits == gold_digits.cast(dtypes.int)).cast(dtypes.float)
        eq = eq_per_pos.prod(axis=-1)
        unobs = (1 - observed_mask.cast(dtypes.float))
        cell_acc = (eq * unobs).sum() / (unobs.sum() + 1e-8)
        return pred_digits.realize(), cell_acc.realize()

    _JIT_V109_CACHE[key] = _eval
    print(f"[JIT] v109 eval ready (cache={len(_JIT_V109_CACHE)})", flush=True)
    return _eval
