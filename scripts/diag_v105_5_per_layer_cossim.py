"""v105.5 per-layer cos_sim diagnostic.

Find WHERE in the forward pass collapse first appears.

Tap points (within breath 0):
  T0: post-embed (after digit_embed + var/factor_pos + node_kind, then + breath_embed)
  T1: post-L0
  T2: post-L1
  T3: post-L2
  T4: post-L3
  T5: post-IB
  T6: post-waist
  T7: post-PPFFN
  T8: post-delta_gate (end of breath 0)

Per-breath taps (end of breath after delta_gate):
  E0, E2, E4, E7

For each (tap, digit_position), compute mean pairwise cos similarity of the
1024-d hidden state across DIFFERENT problems' SAME (variable, digit_position)
cells. If cos_mean is high (>0.9), positions are collapsed at that tap.

Run:
  CKPT=.cache/fg_v105_5_ckpts/v105_5_prod_step1000.safetensors \
    DEV='PCI+AMD' .venv/bin/python -u scripts/diag_v105_5_per_layer_cossim.py
"""
from __future__ import annotations

import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from tinygrad import Tensor, Device, dtypes
from tinygrad.helpers import getenv

from mycelium import Config
from mycelium.loader import _load_state, load_breathing
from mycelium.factor_graph_v105_5 import (
    attach_fg_params_v105_5,
    load_ckpt_v105_5,
    embed_factor_graph_v105_5,
    fg_layer_forward_v105_5,
    fg_layer_forward_v105_6_l0,
    apply_hierarchical_ib_codebook,
    apply_projection_waist,
    apply_perpos_ffn,
    V105_5_N_HEADS,
    V105_5_PERPOS_FFN,
    V105_6_PERPOS_L0,
)
from mycelium.factor_graph_data_v105_5 import FactorGraphLoaderV105_5
from mycelium.breathing import _layernorm


# ---- config ----

K_MAX     = int(getenv("V105_5_K_MAX", "8"))
N_DIGITS  = int(getenv("V105_5_N_DIGITS", "5"))
N_MAX     = int(getenv("V105_5_N_MAX", "16"))
F_MAX     = int(getenv("V105_5_F_MAX", "8"))
WAIST     = int(getenv("V105_5_WAIST", "512"))
N_CODE    = int(getenv("V105_5_CODEBOOK_N", "32"))
N_HEADS   = 16

N_BATCHES = int(getenv("N_BATCHES", "16"))
BATCH     = int(getenv("BATCH", "8"))
CKPT      = getenv("CKPT", ".cache/fg_v105_5_ckpts/v105_5_prod_step1000.safetensors")
VAL_PATH  = getenv("VAL_PATH", ".cache/factor_graph_test_loguniform.jsonl")
DIFF_FILTER = getenv("DIFF_FILTER", "easy") or None

COS_SIM_PAIRS = int(getenv("COS_SIM_PAIRS", "1500"))

POS_LABELS = ["ones", "tens", "hundreds", "thousands", "ten-thousands"]

# Per-breath end taps (subset to keep output manageable)
PER_BREATH_END_TAPS = [0, 2, 4, 7]


def cast_layers_fp32(model):
    def _cast(obj, attr):
        t = getattr(obj, attr)
        if t.dtype == dtypes.half:
            setattr(obj, attr, t.cast(dtypes.float).contiguous().realize())
    _cast(model.embed, "weight")
    sw = model.block.shared
    for a in ("wv", "bv", "wo", "bo", "w_out", "b_out"):
        _cast(sw, a)
    for layer in model.block.layers:
        for a in ("wq", "bq", "wk", "bk", "w_in", "b_in"):
            _cast(layer, a)


def fg_forward_taps_v105_5(
    model,
    digit_init,
    node_kinds,
    staging_mask,
    head_op_mask,
    K=K_MAX,
    n_max=N_MAX,
    f_max=F_MAX,
    n_digits=N_DIGITS,
):
    """Forward pass that returns a dict of intermediate hidden states.

    Returns dict mapping tap_name → (B, n_var_tokens, H) numpy array
    where n_var_tokens = n_max * n_digits.
    """
    digit_codebook   = model.fg_v105_5_digit_codebook
    digit_rope_cos   = model.fg_v105_5_digit_rope_cos
    digit_rope_sin   = model.fg_v105_5_digit_rope_sin
    var_pos_embed    = model.fg_v105_5_var_pos_embed
    factor_pos_embed = model.fg_v105_5_factor_pos_embed
    node_kind_embed  = model.fg_v105_5_node_kind_embed
    breath_embed     = model.fg_v105_5_breath_embed
    delta_gate       = model.fg_v105_5_delta_gate

    W_compress = model.fg_v105_5_W_compress
    b_compress = model.fg_v105_5_b_compress
    W_expand   = model.fg_v105_5_W_expand
    b_expand   = model.fg_v105_5_b_expand

    ib_codebook       = model.fg_v105_5_ib_codebook
    family_centroids  = model.fg_v105_5_family_centroids
    leaf_to_family_oh = model.fg_v105_5_leaf_to_family_oh
    delta_gate_quant  = model.fg_v105_5_delta_gate_quant
    ib_temperature    = model.fg_v105_5_ib_temperature

    ppffn_ln_g  = getattr(model, "fg_v105_5_ppffn_ln_g",  None)
    ppffn_ln_b  = getattr(model, "fg_v105_5_ppffn_ln_b",  None)
    ppffn_W_in  = getattr(model, "fg_v105_5_ppffn_W_in",  None)
    ppffn_b_in  = getattr(model, "fg_v105_5_ppffn_b_in",  None)
    ppffn_W_out = getattr(model, "fg_v105_5_ppffn_W_out", None)
    ppffn_b_out = getattr(model, "fg_v105_5_ppffn_b_out", None)
    # v105.6 — per-position L0 W_in/b_in (replacement at L0).
    v6_l0_w_in  = getattr(model, "fg_v105_6_l0_perpos_w_in", None)
    v6_l0_b_in  = getattr(model, "fg_v105_6_l0_perpos_b_in", None)

    B = int(digit_init.shape[0])
    T = n_max * n_digits + f_max
    n_var = n_max * n_digits

    taps = {}

    def tap(name, x):
        # Extract variable digit tokens only, cast to float
        v = x[:, :n_var, :].cast(dtypes.float).realize().numpy()
        taps[name] = v

    x = embed_factor_graph_v105_5(
        digit_init, node_kinds,
        var_pos_embed, factor_pos_embed, node_kind_embed,
        digit_codebook, digit_rope_cos, digit_rope_sin,
        n_max=n_max, n_digits=n_digits, f_max=f_max,
    )
    x = x.cast(dtypes.half) if x.dtype != dtypes.half else x

    layers = list(model.block.layers)

    for k in range(K):
        be_k  = breath_embed[k].reshape(1, 1, -1).cast(x.dtype)
        x_in  = x + be_k
        x_pre = x

        stk   = staging_mask[:, k, :, :]
        stk_h = stk.reshape(B, 1, T, T).expand(B, V105_5_N_HEADS, T, T)
        combined = stk_h.cast(x.dtype) + head_op_mask.cast(x.dtype)

        # tap T0: input to PPFFN (post-embed + breath_embed)
        if k == 0:
            tap("T0_post_embed", x_in)

        # PPFFN moved to BEFORE L0 as of 2026-06-02
        if V105_5_PERPOS_FFN and ppffn_W_out is not None:
            x_in = apply_perpos_ffn(
                x_in, ppffn_ln_g, ppffn_ln_b,
                ppffn_W_in, ppffn_b_in, ppffn_W_out, ppffn_b_out,
                n_max=n_max, n_digits=n_digits, f_max=f_max,
                eps=model.cfg.layer_norm_eps,
            )
            if k == 0: tap("T0b_post_PPFFN", x_in)

        h = x_in
        if V105_6_PERPOS_L0 and v6_l0_w_in is not None:
            h = fg_layer_forward_v105_6_l0(
                layers[0], h, combined,
                v6_l0_w_in, v6_l0_b_in,
                n_max=n_max, n_digits=n_digits, f_max=f_max,
            )
        else:
            h = fg_layer_forward_v105_5(layers[0], h, combined)
        if k == 0: tap("T1_post_L0", h)
        h = fg_layer_forward_v105_5(layers[1], h, combined)
        if k == 0: tap("T2_post_L1", h)
        h = fg_layer_forward_v105_5(layers[2], h, combined)
        if k == 0: tap("T3_post_L2", h)
        h = fg_layer_forward_v105_5(layers[3], h, combined)
        if k == 0: tap("T4_post_L3", h)

        h = apply_hierarchical_ib_codebook(
            h, ib_codebook, family_centroids, leaf_to_family_oh,
            ib_temperature, delta_gate_quant[k],
        )
        if k == 0: tap("T5_post_IB", h)

        h = apply_projection_waist(h, W_compress, W_expand, b_compress, b_expand)
        if k == 0: tap("T6_post_waist", h)

        gate_k = delta_gate[k].cast(h.dtype).reshape(1, 1, 1)
        delta  = h - x_pre
        x      = x_pre + gate_k * delta

        if k == 0: tap("T8_post_delta_gate", x)
        if k in PER_BREATH_END_TAPS:
            tap(f"E{k}_breath_end", x)

    return taps, n_var


def estimate_pairwise_cos(X, n_pairs=COS_SIM_PAIRS, seed=0):
    rng = np.random.RandomState(seed)
    N = X.shape[0]
    if N < 2:
        return float("nan"), float("nan")
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    Xn = X / norms
    n_pairs = min(n_pairs, N * (N - 1) // 2)
    i = rng.randint(0, N, size=n_pairs)
    j = rng.randint(0, N, size=n_pairs)
    mask = i != j
    i = i[mask]; j = j[mask]
    if len(i) == 0:
        return float("nan"), float("nan")
    cos = (Xn[i] * Xn[j]).sum(axis=1)
    return float(cos.mean()), float(cos.std())


def main():
    print("=== v105.5 per-layer cos_sim diagnostic ===", flush=True)
    print(f"K={K_MAX}  n_digits={N_DIGITS}  n_batches={N_BATCHES}  batch={BATCH}", flush=True)
    print(f"ckpt: {CKPT}", flush=True)
    print(f"val:  {VAL_PATH}  difficulty_filter={DIFF_FILTER}", flush=True)
    print("", flush=True)

    cfg = Config()
    sd  = _load_state()
    model = load_breathing(cfg, sd=sd); del sd
    cast_layers_fp32(model)
    attach_fg_params_v105_5(
        model, hidden=cfg.hidden,
        n_digits=N_DIGITS, n_max=N_MAX, f_max=F_MAX, k_max=K_MAX,
        waist=WAIST, n_code=N_CODE,
    )
    Device[Device.DEFAULT].synchronize()
    load_ckpt_v105_5(model, CKPT)
    print(f"loaded ckpt OK  hidden={cfg.hidden}", flush=True)

    val_loader = FactorGraphLoaderV105_5(
        VAL_PATH, batch_size=BATCH,
        difficulty_filter=DIFF_FILTER, curriculum=False,
        n_max=N_MAX, f_max=F_MAX, k_max=K_MAX, n_heads=N_HEADS,
        n_digits=N_DIGITS, seed=43,
    )

    H = cfg.hidden

    # Tap names in order (will be defined after first batch runs)
    tap_order = None
    feats_by_tap_pos = None

    Tensor.training = False
    nb = 0
    for batch in val_loader.iter_eval(batch_size=BATCH):
        taps, n_var = fg_forward_taps_v105_5(
            model, batch["digit_init"], batch["node_kinds"],
            batch["staging_mask"], batch["head_op_mask"],
            K=K_MAX, n_max=N_MAX, f_max=F_MAX, n_digits=N_DIGITS,
        )

        if tap_order is None:
            tap_order = list(taps.keys())
            feats_by_tap_pos = {
                t: [[] for _ in range(N_DIGITS)] for t in tap_order
            }

        obs   = batch["observed_mask"].numpy()
        valid = batch["digit_valid_mask"].numpy()

        B = obs.shape[0]
        for t_name in tap_order:
            arr = taps[t_name]  # (B, n_var, H)
            arr_r = arr.reshape(B, N_MAX, N_DIGITS, H)
            for b in range(B):
                nv = int(batch["n_vars_total"][b])
                for v in range(min(nv, N_MAX)):
                    if obs[b, v] == 1:
                        continue
                    if valid[b, v].sum() == 0:
                        continue
                    for p in range(N_DIGITS):
                        if valid[b, v, p] < 0.5:
                            continue
                        feats_by_tap_pos[t_name][p].append(arr_r[b, v, p])

        nb += 1
        if nb >= N_BATCHES:
            break
    Tensor.training = True
    print(f"extracted from {nb} batches", flush=True)
    print("", flush=True)

    # Compute cos_sim per (tap, position)
    print("=== cos_sim per (tap, digit_position) ===", flush=True)
    header = f"{'tap':<22s}  " + "  ".join(f"{POS_LABELS[p][:6]:>10s}" for p in range(N_DIGITS))
    print(header, flush=True)
    print("-" * len(header), flush=True)

    for t_name in tap_order:
        line = f"{t_name:<22s}  "
        for p in range(N_DIGITS):
            cells = feats_by_tap_pos[t_name][p]
            if len(cells) < 2:
                line += f"{'n/a':>10s}  "
                continue
            X = np.stack(cells, axis=0).astype(np.float32)
            cos_mean, cos_std = estimate_pairwise_cos(X, n_pairs=COS_SIM_PAIRS, seed=p)
            line += f"{cos_mean:>10.3f}  "
        print(line, flush=True)

    # Also report per-position n_cells (one fixed tap is enough)
    print("", flush=True)
    print(f"n_cells per position (across {N_BATCHES} batches):", flush=True)
    for p in range(N_DIGITS):
        n = len(feats_by_tap_pos[tap_order[0]][p])
        print(f"  pos{p} ({POS_LABELS[p]:>14s}): {n}", flush=True)

    # Verdict heuristic
    print("", flush=True)
    print("=== VERDICT ===", flush=True)
    first_collapse = {}
    for p in range(N_DIGITS):
        for t_name in tap_order:
            cells = feats_by_tap_pos[t_name][p]
            if len(cells) < 2:
                continue
            X = np.stack(cells, axis=0).astype(np.float32)
            cos_mean, _ = estimate_pairwise_cos(X, n_pairs=500, seed=p+100)
            if cos_mean > 0.9:
                first_collapse[p] = t_name
                break
    for p in range(N_DIGITS):
        if p in first_collapse:
            print(f"  pos{p} ({POS_LABELS[p]:>14s}): first cos>0.9 at  {first_collapse[p]}", flush=True)
        else:
            print(f"  pos{p} ({POS_LABELS[p]:>14s}): never collapses (cos stays <0.9)", flush=True)


if __name__ == "__main__":
    main()
