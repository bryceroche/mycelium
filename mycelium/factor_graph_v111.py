"""v111: Coherent-photon architecture.

THREE coupled oscillating quantities, all peaking at k=K/2 simultaneously,
all returning to identity at k=0 and k=K-1 (the edges). This is the
proper EM-photon analogy:

  1. WAIST AMPLITUDE   (E field)      sin²(πk/(K_max-1))   peaks at K/2,
                                       returns to 0 at edges
  2. Q ROTATION PHASE  (B field)      π · sin²(πk/(K_max-1))  peaks at π
                                       at K/2, returns to 0 at edges
  3. WAIST DIM SCALING (k vector?)    1.0 + gate[k] * dim_delta[d]
                                       per-dim learnable scaling proportional
                                       to the photon amplitude

Differences from v110-step3:
  - Rotation phase: was LINEAR `phase_scale * k * π / K_max` → now
    SIN² `phase_scale * π * sin²(πk/(K_max-1))` so rotation oscillates
    rather than ramps monotonically.
  - Waist envelope: was `sin²(πk/K_max)` (didn't return to 0 at k=K-1) →
    now `sin²(πk/(K_max-1))` (exact zero at both edges).
  - Waist dim scaling: new — at peak gate, each dim of the 512d waist
    intermediate is scaled by `1 + dim_delta[d]`. Zero-init dim_delta
    → byte-identical warm-start. After training, the model can
    enhance/suppress specific dims proportional to the photon amplitude.

Warm-start: with V111_DIM_DELTA_INIT=0 the architecture's only structural
change from baseline is the rotation phase formula. To make it FULLY
byte-identical, we'd need to disable the new rotation too — but the
whole point is to enable it. Expect a small initial loss bump as the
model adapts to the new rotation envelope.
"""
from __future__ import annotations

import math
import os
from typing import Any

import numpy as np
from tinygrad import Tensor, dtypes
from tinygrad.engine.jit import TinyJit

from mycelium.factor_graph_v107 import embed_factor_graph_v100_aligned
from mycelium.factor_graph_v109pi import fg_layer_forward_v109pi
from mycelium.factor_graph_v109 import _apply_waist_v109
from mycelium.factor_graph_v110_acc import (
    _acc_notebook_read, _acc_notebook_write,
)
from mycelium.factor_graph_v110_step import (
    V110_STEP_N_MAX, V110_STEP_F_MAX, V110_STEP_N_DIGITS,
    V110_STEP_K_MAX, V110_STEP_N_HEADS, V110_STEP_ALTERNATION,
    V110_STEP_PHASE_SCALE, V110_STEP_GATE_PROFILE, V110_STEP_PHOTON_ALPHA,
)
from mycelium.factor_graph_v110_step3 import (
    V110_STEP3_N_MAX, V110_STEP3_F_MAX, V110_STEP3_N_DIGITS,
    V110_STEP3_K_MAX, V110_STEP3_WAIST_DIM,
    V110_STEP3_ALTERNATION, V110_STEP3_PHASE_SCALE,
    V110_STEP3_GATE_PROFILE, V110_STEP3_PHOTON_ALPHA,
    attach_fg_params_v110_step3,
)


# Initial scale for fg_v113_dim_delta. Zero = byte-identical (except for
# the rotation envelope change which is structural).
V111_DIM_DELTA_INIT = float(os.environ.get("V111_DIM_DELTA_INIT", "0.0"))


def _v111_waist_envelope(k: int, K_max: int) -> float:
    """sin²(πk/(K_max-1)) — peaks at k=(K_max-1)/2, returns to 0 at edges."""
    if K_max <= 1:
        return 0.0
    return math.sin(math.pi * k / (K_max - 1)) ** 2


def _v111_rotation_phase(k: int, K_max: int, phase_scale: float) -> float:
    """phase_scale · π · sin²(πk/(K_max-1)) — symmetric oscillating phase."""
    return phase_scale * math.pi * _v111_waist_envelope(k, K_max)


def attach_fg_params_v111(
    model: Any,
    hidden: int,
    n_max: int = V110_STEP3_N_MAX,
    f_max: int = V110_STEP3_F_MAX,
    k_max: int = V110_STEP3_K_MAX,
    n_digits: int = V110_STEP3_N_DIGITS,
    n_code: int = 32,
    ib_centroids_path: str = ".cache/ib_codebook_gsm8k_K32.npy",
    waist_dim: int = V110_STEP3_WAIST_DIM,
) -> None:
    """Attach v110-step3 base + v111 dim_delta param."""
    attach_fg_params_v110_step3(
        model, hidden=hidden,
        n_max=n_max, f_max=f_max, k_max=k_max,
        n_digits=n_digits, n_code=n_code, ib_centroids_path=ib_centroids_path,
        waist_dim=waist_dim,
    )
    if hasattr(model, "fg_v113_dim_delta"):
        return
    model.fg_v113_dim_delta = Tensor(
        np.full((waist_dim,), V111_DIM_DELTA_INIT, dtype=np.float32)
    ).contiguous().realize()

    print(f"[v111] coherent-photon params attached: waist_dim={waist_dim}, "
          f"dim_delta init={V111_DIM_DELTA_INIT}",
          flush=True)
    print(f"[v111] rotation: sin²(πk/(K_max-1)) · π  "
          f"(was linear k·π/K_max in v109pi)",
          flush=True)
    print(f"[v111] waist envelope: sin²(πk/(K_max-1))  "
          f"(was sin²(πk/K_max) in v110-photon)",
          flush=True)


def fg_v111_parameters(model: Any) -> list[Tensor]:
    from scripts.v110_acc_factor_graph_train import collect_fg_params_v110_acc
    params = list(collect_fg_params_v110_acc(model))
    params.append(model.fg_v113_dim_delta)
    return params


def fg_v111_state_dict(model: Any) -> dict[str, Tensor]:
    from scripts.v110_acc_factor_graph_train import model_state_dict_v110_acc
    sd = dict(model_state_dict_v110_acc(model))
    sd["fg_v113_dim_delta"] = model.fg_v113_dim_delta
    return sd


def _apply_waist_v111(
    h: Tensor,           # (B, T, H)
    W_compress: Tensor,  # (H, waist_dim)
    b_compress: Tensor,  # (waist_dim,)
    W_expand: Tensor,    # (waist_dim, H)
    b_expand: Tensor,    # (H,)
    dim_delta: Tensor,   # (waist_dim,) — learnable per-dim modulator
    gate_amp: float,     # scalar photon gate amplitude in [0, 1]
) -> Tensor:
    """LoRA-init waist with per-dim scaling proportional to gate amplitude.

    z = (h W_c + b_c).gelu()                                # (B, T, waist_dim)
    dim_scale = 1.0 + gate_amp · dim_delta                  # (waist_dim,)
    z_scaled = z * dim_scale                                # apply per-dim
    delta = z_scaled @ W_e + b_e
    return h + delta
    """
    wc = W_compress.cast(h.dtype)
    bc = b_compress.reshape(1, 1, -1).cast(h.dtype)
    we = W_expand.cast(h.dtype)
    be = b_expand.reshape(1, 1, -1).cast(h.dtype)
    z = (h @ wc + bc).gelu()                                   # (B, T, waist_dim)
    # Per-dim modulator. At init dim_delta=0 → dim_scale=1 → no change.
    dim_scale = (1.0 + gate_amp * dim_delta).cast(h.dtype)     # (waist_dim,)
    z_scaled = z * dim_scale.reshape(1, 1, -1)
    delta = z_scaled @ we + be
    return h + delta


def fg_breathing_forward_v111(
    model: Any,
    domain_init: Tensor,
    node_kinds: Tensor,
    staging_mask: Tensor,
    head_op_mask: Tensor,
    noise: Tensor,
    noise_scale: Tensor,
    K: int,
    n_max: int = V110_STEP3_N_MAX,
    f_max: int = V110_STEP3_F_MAX,
    n_digits: int = V110_STEP3_N_DIGITS,
    alternation: bool = V110_STEP3_ALTERNATION,
    phase_scale: float = V110_STEP3_PHASE_SCALE,
    photon_alpha: float = V110_STEP3_PHOTON_ALPHA,
):
    """v111 coherent-photon forward."""
    domain_codebook  = model.fg_v107_domain_codebook
    var_state_embed  = model.fg_v107_var_state_embed
    var_pos_embed    = model.fg_v107_var_pos_embed
    factor_pos_embed = model.fg_v107_factor_pos_embed
    node_kind_embed  = model.fg_v107_node_kind_embed
    breath_embed     = model.fg_v107_breath_embed
    delta_gate       = model.fg_v107_delta_gate
    calib_head_w     = model.fg_v107_calib_head_w
    calib_head_b     = model.fg_v107_calib_head_b
    semantic_codebook = model.fg_v107_semantic_codebook
    delta_gate_quant  = model.fg_v107_delta_gate_quant
    temperature       = model.fg_v107_temperature
    tree_codebook = model.fg_v108_tree_codebook

    W_compress = model.fg_v109_W_compress
    b_compress = model.fg_v109_b_compress
    W_expand   = model.fg_v109_W_expand
    b_expand   = model.fg_v109_b_expand

    acc_W_q     = model.fg_v110_acc_W_q
    acc_W_k     = model.fg_v110_acc_W_k
    acc_W_v     = model.fg_v110_acc_W_v
    acc_W_o     = model.fg_v110_acc_W_o
    acc_b_o     = model.fg_v110_acc_b_o
    acc_W_write = model.fg_v110_acc_W_write
    acc_b_write = model.fg_v110_acc_b_write

    dim_delta = model.fg_v113_dim_delta

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

    # *** v111 coherent envelopes: rotation phase and waist gate share shape ***
    breath_phases = [_v111_rotation_phase(k, K_max, phase_scale) for k in range(K_max)]
    breath_cos = [math.cos(p) for p in breath_phases]
    breath_sin = [math.sin(p) for p in breath_phases]
    photon_gates = [_v111_waist_envelope(k, K_max) for k in range(K_max)]
    binary_gates = [1.0 if k % 2 == 0 else 0.0 for k in range(K_max)]  # alternation reference

    from mycelium.breathing import _layernorm

    tree_logits_history   = []
    var_logits_history    = []
    factor_logits_history = []
    calib_history         = []
    step_mags_history     = []
    tree_cb_flat = tree_codebook.reshape(n_digits * 10, H)
    notebook_slots: list[Tensor] = []

    for k in range(K):
        x_acc_delta = _acc_notebook_read(
            x, notebook_slots,
            acc_W_q, acc_W_k, acc_W_v, acc_W_o, acc_b_o,
        )
        x = x + x_acc_delta

        x = x + (noise[k] * noise_scale).cast(x.dtype)

        be_k  = breath_embed[k].reshape(1, 1, -1).cast(x.dtype)
        x_in  = x + be_k
        x_pre = x

        stk      = staging_mask[:, k, :, :]
        stk_h    = stk.reshape(B, 1, T, T).expand(B, V110_STEP_N_HEADS, T, T)
        combined = stk_h.cast(x.dtype) + head_op_mask.cast(x.dtype)

        cos_k = breath_cos[k]
        sin_k = breath_sin[k]
        h = x_in
        for layer in layers[:4]:
            h = fg_layer_forward_v109pi(layer, h, combined, cos_k, sin_k)

        cb  = semantic_codebook.cast(h.dtype)
        tmp = temperature.cast(h.dtype)
        scores   = h @ cb.T / tmp.reshape(1, 1, 1)
        weights  = scores.clip(-1e4, 1e4).softmax(-1)
        recon    = weights @ cb
        quantize = recon - h
        gate_quant_k = delta_gate_quant[k].cast(h.dtype).reshape(1, 1, 1)
        h_quant = h + gate_quant_k * quantize

        # *** v111 waist with photon envelope + per-dim modulation ***
        # photon_gates[k] is the sin²(πk/(K-1)) envelope — coherent with rotation
        # alternation blend: keep mechanism for warm-start compatibility
        if alternation:
            gate_k_amp = (1.0 - photon_alpha) * binary_gates[k] + photon_alpha * photon_gates[k]
        else:
            gate_k_amp = photon_gates[k]

        if gate_k_amp > 0.0:
            h_quant_waist = _apply_waist_v111(
                h_quant, W_compress, b_compress, W_expand, b_expand,
                dim_delta, gate_amp=float(gate_k_amp),
            )
            if gate_k_amp >= 1.0:
                h_quant = h_quant_waist
            else:
                h_quant = (1.0 - gate_k_amp) * h_quant + gate_k_amp * h_quant_waist

        gate_k = delta_gate[k].cast(h_quant.dtype).reshape(1, 1, 1)
        delta  = h_quant - x_pre
        step   = gate_k * delta
        x      = x_pre + step

        step_mag_k = step.cast(dtypes.float).square().mean()
        step_mags_history.append(step_mag_k)

        slot_k = _acc_notebook_write(x, acc_W_write, acc_b_write)
        notebook_slots.append(slot_k)

        x_ln  = _layernorm(x, model.ln_f_g, model.ln_f_b,
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

    return (tree_logits_history, var_logits_history, factor_logits_history,
            calib_history, step_mags_history)


# ---------------------------------------------------------------------------
# JIT helpers
# ---------------------------------------------------------------------------

_JIT_V111_CACHE: dict = {}


def _compile_jit_fg_step_v111(
    model: Any,
    opt: Any,
    K: int,
    B: int,
    factor_aux_weight: float,
    calib_weight: float,
    var_loss_weight: float,
    balance_weight: float,
    uncertainty_min: float,
    hard_breath_level: bool,
    alternation: bool,
    phase_scale: float,
    n_max: int,
    f_max: int,
    n_digits: int,
    photon_alpha: float,
    grad_clip: float = 1.0,
):
    key = ("v111", id(model), id(opt), int(K), int(B),
           float(factor_aux_weight), float(calib_weight), float(var_loss_weight),
           float(balance_weight), float(uncertainty_min),
           bool(hard_breath_level), bool(alternation), float(phase_scale),
           int(n_max), int(f_max), int(n_digits),
           float(photon_alpha), float(grad_clip))
    if key in _JIT_V111_CACHE:
        return _JIT_V111_CACHE[key]

    fw, aw, vw, bw, um, gc = (float(factor_aux_weight), float(calib_weight),
                              float(var_loss_weight), float(balance_weight),
                              float(uncertainty_min), float(grad_clip))
    params = opt.params
    print(
        f"[JIT] compile v111 step: K={K} B={B} "
        f"alpha={photon_alpha} balance={bw} u_min={um}...",
        flush=True,
    )

    @TinyJit
    def _step(
        domain_init, node_kinds, staging_mask, head_op_mask,
        gold_digits, gold_bins, observed_mask, factor_gold_bin, factor_valid,
        noise, noise_scale,
    ):
        opt.zero_grad()

        tree_lh, _, fac_lh, calib_h, step_mh = (
            fg_breathing_forward_v111(
                model, domain_init, node_kinds, staging_mask, head_op_mask,
                noise, noise_scale,
                K=K, n_max=n_max, f_max=f_max, n_digits=n_digits,
                alternation=alternation, phase_scale=phase_scale,
                photon_alpha=photon_alpha,
            )
        )

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

        n_valid_factors = factor_valid.cast(dtypes.float).sum() + 1e-8
        gold_fac_flat   = factor_gold_bin.cast(dtypes.int).reshape(B * f_max)
        gold_fac_oh     = gold_fac_flat.one_hot(200).cast(dtypes.float)
        valid_flat      = factor_valid.cast(dtypes.float).reshape(B * f_max)
        factor_aux_sum   = Tensor.zeros((), dtype=dtypes.float).contiguous()
        factor_aux_w_sum = 0.0
        for k_aux, fac_logits_k in enumerate(fac_lh):
            w_k_aux = 1.0 + float(k_aux) / float(max(K - 1, 1))
            fac_flat  = fac_logits_k.reshape(B * f_max, 200)
            fac_lp    = fac_flat.log_softmax(axis=-1)
            fac_nll   = -(fac_lp * gold_fac_oh).sum(axis=-1)
            fac_masked= fac_nll * valid_flat
            fac_ce_k  = fac_masked.sum() / n_valid_factors
            factor_aux_sum   = factor_aux_sum + fac_ce_k * w_k_aux
            factor_aux_w_sum += w_k_aux
        factor_aux_loss = factor_aux_sum / float(factor_aux_w_sum)

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

        calib_per_breath = [c.mean().detach() for c in calib_h]
        calib_stack = Tensor.stack(*calib_per_breath, dim=0)
        uncertainty_k = (1.0 - calib_stack).clip(um, 1.0)
        step_stack = Tensor.stack(*step_mh, dim=0)
        normalized = step_stack / uncertainty_k
        norm_mean  = normalized.mean()
        norm_var   = ((normalized - norm_mean) ** 2).mean()
        norm_std   = (norm_var + 1e-12).sqrt()
        step_balance_loss = norm_std / (norm_mean + 1e-8)

        cell_acc  = (eq_unobs.sum() / (unobs_2d.sum() + 1e-8)).detach()
        query_acc = correct.mean().detach()

        total_ce = (vw * var_loss + fw * factor_aux_loss + aw * calib_loss
                    + bw * step_balance_loss)
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
            step_balance_loss.realize(),
            *(ce.realize() for ce in per_breath_ce_t),
            *(s.realize() for s in step_mh),
            *(c for c in calib_per_breath),
        )

    _JIT_V111_CACHE[key] = _step
    print(f"[JIT] v111 step ready (cache={len(_JIT_V111_CACHE)})", flush=True)
    return _step


def compile_jit_mcbp_v111(
    model: Any,
    K: int = V110_STEP3_K_MAX,
    n_max: int = V110_STEP3_N_MAX,
    f_max: int = V110_STEP3_F_MAX,
    n_digits: int = V110_STEP3_N_DIGITS,
    alternation: bool = V110_STEP3_ALTERNATION,
    phase_scale: float = V110_STEP3_PHASE_SCALE,
    photon_alpha: float = V110_STEP3_PHOTON_ALPHA,
):
    @TinyJit
    def _eval(
        domain_init: Tensor,
        node_kinds: Tensor,
        staging_mask: Tensor,
        head_op_mask: Tensor,
        noise: Tensor,
        noise_scale: Tensor,
    ) -> Tensor:
        Tensor.training = False
        tree_lh, _, _, _, _ = fg_breathing_forward_v111(
            model, domain_init, node_kinds, staging_mask, head_op_mask,
            noise, noise_scale,
            K=K, n_max=n_max, f_max=f_max, n_digits=n_digits,
            alternation=alternation, phase_scale=phase_scale,
            photon_alpha=photon_alpha,
        )
        return tree_lh[-1].realize()
    return _eval


def compile_jit_eval_v111(
    model: Any,
    K: int = V110_STEP3_K_MAX,
    B: int = 8,
    n_max: int = V110_STEP3_N_MAX,
    f_max: int = V110_STEP3_F_MAX,
    n_digits: int = V110_STEP3_N_DIGITS,
    alternation: bool = V110_STEP3_ALTERNATION,
    phase_scale: float = V110_STEP3_PHASE_SCALE,
    photon_alpha: float = V110_STEP3_PHOTON_ALPHA,
):
    T = n_max + f_max
    H = 1024
    eval_noise_zeros = Tensor.zeros(K, B, T, H, dtype=dtypes.half).contiguous().realize()
    eval_noise_scale = Tensor(
        np.array([0.0], dtype=np.float16)
    ).cast(dtypes.half).contiguous().realize()

    @TinyJit
    def _eval(
        domain_init: Tensor,
        node_kinds: Tensor,
        staging_mask: Tensor,
        head_op_mask: Tensor,
        gold_digits: Tensor,
        observed_mask: Tensor,
    ):
        Tensor.training = False
        tree_lh, _, _, _, _ = fg_breathing_forward_v111(
            model, domain_init, node_kinds, staging_mask, head_op_mask,
            eval_noise_zeros, eval_noise_scale,
            K=K, n_max=n_max, f_max=f_max, n_digits=n_digits,
            alternation=alternation, phase_scale=phase_scale,
            photon_alpha=photon_alpha,
        )
        final_tree = tree_lh[-1]
        pred_digits = final_tree.argmax(axis=-1)
        eq_per_pos  = (pred_digits == gold_digits.cast(dtypes.int)).cast(dtypes.float)
        eq          = eq_per_pos.prod(axis=-1)
        unobs       = (1 - observed_mask.cast(dtypes.float))
        cell_acc    = (eq * unobs).sum() / (unobs.sum() + 1e-8)
        return pred_digits.realize(), cell_acc.realize()
    return _eval
