"""Interior dynamics probe — per-layer attention JSD + residual delta metrics.

Runs EAGER (no TinyJit). For each puzzle:
  - Runs the forward pass with per-layer, per-breath hooks to extract:
      A. Attention weights (B, n_heads, T, T) per layer per breath
      B. Post-layer residual stream x (B, T, H) per layer per breath

  - Computes:
      Metric A: JSD of attn[q, :] distribution between consecutive breaths,
                averaged within head-group (mask family), mean and max per row.
      Metric B: rel_norm(Δ_k) = ||x_L_k+1 - x_L_k|| / ||x_L_k|| per position
                cos(Δ_k, Δ_{k+1}) — directional persistence

  - Splits by solved vs failed (final-breath cell_acc > 0.5 threshold per puzzle)
  - Splits by node type: given vs non-given (v98 clue/fill; v110 obs/unobs)

Head-group structure:
  v110-step3: 16 heads, 4 groups of 4: ADD(0-3), SUB(4-7), MUL(8-11), DIV(12-15)
  v98 sudoku: 16 heads, 4 groups: ROW(0-4), COL(5-9), BOX(10-14), GLOBAL(15)

Output: .cache/interior_probe_v110_v98.json with raw per-breath traces.

Usage:
  cd /home/bryce/mycelium
  V110_STEP3_TASK=1 V110_STEP3_K_MAX=8 V110_STEP3_N_DIGITS=5 \
    V110_STEP3_N_MAX=16 V110_STEP3_F_MAX=8 V110_STEP3_WAIST_DIM=512 \
    V110_STEP3_ALTERNATION=1 V110_STEP3_PHASE_SCALE=1.0 SUDOKU_TASK=1 \
    SUDOKU_K_MAX=20 .venv/bin/python scripts/diag_interior_probe_v110_v98.py
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---- env vars must be set BEFORE imports ----
os.environ.setdefault("V110_STEP3_TASK", "1")
os.environ.setdefault("V110_STEP3_K_MAX", "8")
os.environ.setdefault("V110_STEP3_N_DIGITS", "5")
os.environ.setdefault("V110_STEP3_N_MAX", "16")
os.environ.setdefault("V110_STEP3_F_MAX", "8")
os.environ.setdefault("V110_STEP3_WAIST_DIM", "512")
os.environ.setdefault("V110_STEP3_ALTERNATION", "1")
os.environ.setdefault("V110_STEP3_PHASE_SCALE", "1.0")
os.environ.setdefault("SUDOKU_TASK", "1")
os.environ.setdefault("SUDOKU_K_MAX", "20")

import numpy as np
from tinygrad import Device, Tensor, dtypes
from tinygrad.nn.state import safe_load

from mycelium import Config
from mycelium.loader import _load_state, load_breathing
from mycelium.breathing import _layernorm

# ---- v110-step3 imports ----
from mycelium.factor_graph_v110_step3 import (
    V110_STEP3_K_MAX, V110_STEP3_N_MAX, V110_STEP3_F_MAX, V110_STEP3_N_HEADS,
    V110_STEP3_N_DIGITS, V110_STEP3_WAIST_DIM, V110_STEP3_ALTERNATION,
    V110_STEP3_PHASE_SCALE, V110_STEP3_GATE_PROFILE, V110_STEP3_PHOTON_ALPHA,
    V110_STEP3_CODEBOOK_N, V110_STEP3_IB_CENTROIDS,
    attach_fg_params_v110_step3,
)
from mycelium.factor_graph_v110_step import fg_breathing_forward_v110_step
from mycelium.factor_graph_v109pi import fg_layer_forward_v109pi, _rotate_q_pi
from mycelium.factor_graph_v110_acc import _acc_notebook_read, _acc_notebook_write
from mycelium.factor_graph_v110_photon import _photon_gate
from mycelium.factor_graph_v109 import _apply_waist_v109
from mycelium.factor_graph_v107 import embed_factor_graph_v100_aligned
from mycelium.factor_graph_v108 import bins_to_digits_msd
from mycelium.factor_graph_data_v107 import (
    FactorGraphLoaderV107, _records_to_batch_v107,
)

# ---- v98 sudoku imports ----
from mycelium.sudoku import (
    attach_sudoku_params, sudoku_layer_forward, embed_sudoku,
    SUDOKU_K_MAX as SUDOKU_K_MAX_DEFAULT,
)
from mycelium.sudoku_data import SudokuLoader

# ---- v110 ckpt loader (same as v110_acc) ----
from scripts.v108_factor_graph_train import cast_layers_fp32
from scripts.v110_acc_factor_graph_train import load_ckpt_v110_acc as load_ckpt_v110_step3

# ---- sudoku ckpt loader ----
from scripts.sudoku_train import load_ckpt as load_sudoku_ckpt


# ============================================================
# Config
# ============================================================
EPS = 1e-10
N_PUZZLES = int(os.environ.get("N_PUZZLES", "50"))
SEED = 42
SOLVED_THRESHOLD = 0.5   # cell_acc threshold to classify puzzle as solved

V110_CKPT = ".cache/fg_v110_step3_ckpts/v110_step3_cont8_step1000.safetensors"
V98_CKPT  = ".cache/sudoku_ckpts/v98_prod_final.safetensors"
V110_VAL  = ".cache/factor_graph_test.jsonl"
V98_VAL   = ".cache/sudoku_val.jsonl"

# Head groups
V110_HEAD_GROUPS = {
    "ADD": list(range(0, 4)),
    "SUB": list(range(4, 8)),
    "MUL": list(range(8, 12)),
    "DIV": list(range(12, 16)),
}
V98_HEAD_GROUPS = {
    "ROW":    list(range(0, 5)),
    "COL":    list(range(5, 10)),
    "BOX":    list(range(10, 15)),
    "GLOBAL": [15],
}


# ============================================================
# JSD utilities
# ============================================================

def jsd_rows(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """JSD between corresponding rows of two (N, C) probability arrays.

    Returns (N,) array of JSD values. EPS-stabilized.
    p, q: (N, C) — assumed non-negative, will be renormalized.
    """
    p = p + EPS
    q = q + EPS
    p = p / p.sum(axis=-1, keepdims=True)
    q = q / q.sum(axis=-1, keepdims=True)
    m = 0.5 * (p + q)
    kl_pm = (p * np.log(p / m)).sum(axis=-1)
    kl_qm = (q * np.log(q / m)).sum(axis=-1)
    return 0.5 * kl_pm + 0.5 * kl_qm


def freeze_breath(trace: list[float], eps_frac: float = 0.05) -> int:
    """Return first k where value <= eps_frac * trace[0]. Returns K-2 if never."""
    if not trace or trace[0] < EPS:
        return 0
    threshold = eps_frac * trace[0]
    for k, v in enumerate(trace):
        if v <= threshold:
            return k
    return len(trace) - 1


# ============================================================
# v110 eager forward with per-layer hooks
# ============================================================

def v110_eager_forward_with_hooks(
    model,
    domain_init, node_kinds, staging_mask, head_op_mask,
    K: int,
    n_max: int, f_max: int, n_digits: int,
    alternation: bool, phase_scale: float,
    gate_profile: str, photon_alpha: float,
):
    """
    Runs v110-step3 forward eagerly, capturing per-layer per-breath:
      - attn_weights: list[K][4 layers] of (n_heads, T, T) numpy arrays
      - post_layer_x: list[K][4 layers] of (T, H) numpy arrays (B=1 squeezed)
      - pre_breath_x: list[K] of (T, H) numpy arrays
    Also returns final tree_logits for accuracy evaluation.
    """
    assert int(domain_init.shape[0]) == 1, "B must be 1 for eager probe"

    domain_codebook   = model.fg_v107_domain_codebook
    var_state_embed   = model.fg_v107_var_state_embed
    var_pos_embed     = model.fg_v107_var_pos_embed
    factor_pos_embed  = model.fg_v107_factor_pos_embed
    node_kind_embed   = model.fg_v107_node_kind_embed
    breath_embed      = model.fg_v107_breath_embed
    delta_gate        = model.fg_v107_delta_gate
    calib_head_w      = model.fg_v107_calib_head_w
    calib_head_b      = model.fg_v107_calib_head_b
    semantic_codebook = model.fg_v107_semantic_codebook
    delta_gate_quant  = model.fg_v107_delta_gate_quant
    temperature       = model.fg_v107_temperature
    tree_codebook     = model.fg_v108_tree_codebook

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

    H  = int(tree_codebook.shape[-1])
    B  = 1
    T  = n_max + f_max

    x = embed_factor_graph_v100_aligned(
        domain_init, node_kinds,
        var_pos_embed, factor_pos_embed, node_kind_embed,
        domain_codebook, var_state_embed,
    )
    x = x.cast(dtypes.half) if x.dtype != dtypes.half else x

    layers = list(model.block.layers)
    K_max  = int(breath_embed.shape[0])

    breath_phases = [phase_scale * k * math.pi / float(K_max) for k in range(K_max)]
    breath_cos = [math.cos(p) for p in breath_phases]
    breath_sin = [math.sin(p) for p in breath_phases]

    photon_gates = [_photon_gate(k, K_max, gate_profile) for k in range(K_max)]
    binary_gates = [_photon_gate(k, K_max, "binary")    for k in range(K_max)]

    attn_per_breath  = []   # list[K] of list[4] of (n_heads, T, T) np
    post_x_per_breath = []  # list[K] of list[4] of (T, H) np
    pre_x_per_breath  = []  # list[K] of (T, H) np
    tree_logits_history = []

    tree_cb_flat = tree_codebook.reshape(n_digits * 10, H)
    notebook_slots: list[Tensor] = []

    for k in range(K):
        x_acc_delta = _acc_notebook_read(
            x, notebook_slots,
            acc_W_q, acc_W_k, acc_W_v, acc_W_o, acc_b_o,
        )
        x = x + x_acc_delta

        be_k  = breath_embed[k].reshape(1, 1, -1).cast(x.dtype)
        x_in  = x + be_k
        x_pre = x

        # Save pre-layer residual for this breath
        pre_x_per_breath.append(x_pre.cast(dtypes.float).realize().numpy()[0].copy())  # (T, H)

        stk      = staging_mask[:, k, :, :]
        stk_h    = stk.reshape(B, 1, T, T).expand(B, V110_STEP3_N_HEADS, T, T)
        combined = stk_h.cast(x.dtype) + head_op_mask.cast(x.dtype)

        cos_k = breath_cos[k]
        sin_k = breath_sin[k]

        # Run each layer with explicit attention extraction
        attn_breath_k = []
        post_x_breath_k = []
        h = x_in
        for li, layer in enumerate(layers[:4]):
            h, attn_np = _v110_layer_forward_hooked(layer, h, combined, cos_k, sin_k)
            attn_breath_k.append(attn_np)  # (n_heads, T, T)
            post_x_breath_k.append(h.cast(dtypes.float).realize().numpy()[0].copy())  # (T, H)

        attn_per_breath.append(attn_breath_k)
        post_x_per_breath.append(post_x_breath_k)

        # Codebook quantize
        cb  = semantic_codebook.cast(h.dtype)
        tmp = temperature.cast(h.dtype)
        scores   = h @ cb.T / tmp.reshape(1, 1, 1)
        weights  = scores.clip(-1e4, 1e4).softmax(-1)
        recon    = weights @ cb
        quantize = recon - h
        gate_quant_k = delta_gate_quant[k].cast(h.dtype).reshape(1, 1, 1)
        h_quant = h + gate_quant_k * quantize

        if alternation:
            gate_k_amp = (1.0 - photon_alpha) * binary_gates[k] + photon_alpha * photon_gates[k]
        else:
            gate_k_amp = 1.0

        if gate_k_amp > 0.0:
            h_quant_waist = _apply_waist_v109(
                h_quant, W_compress, b_compress, W_expand, b_expand,
            )
            if gate_k_amp >= 1.0:
                h_quant = h_quant_waist
            else:
                h_quant = (1.0 - gate_k_amp) * h_quant + gate_k_amp * h_quant_waist

        gate_k = delta_gate[k].cast(h_quant.dtype).reshape(1, 1, 1)
        delta  = h_quant - x_pre
        x      = x_pre + gate_k * delta

        slot_k = _acc_notebook_write(x, acc_W_write, acc_b_write)
        notebook_slots.append(slot_k)

        x_ln = _layernorm(x, model.ln_f_g, model.ln_f_b,
                          model.cfg.layer_norm_eps).cast(dtypes.float)
        var_x = x_ln[:, :n_max, :]
        tree_logits_flat = var_x @ tree_cb_flat.T.cast(dtypes.float)
        tree_logits_k    = tree_logits_flat.reshape(B, n_max, n_digits, 10)
        tree_logits_history.append(tree_logits_k.realize().numpy()[0])  # (n_max, n_digits, 10)

    return attn_per_breath, post_x_per_breath, pre_x_per_breath, tree_logits_history


def _v110_layer_forward_hooked(layer, x, attn_bias, q_rot_cos, q_rot_sin):
    """v109pi layer forward but also returns attention weights as numpy."""
    cfg = layer.cfg
    B, S, H = x.shape
    n_heads  = cfg.n_heads
    head_dim = cfg.head_dim

    attn_in = _layernorm(x, layer.shared.in_ln_g, layer.shared.in_ln_b, cfg.layer_norm_eps)
    mlp_in  = _layernorm(x, layer.shared.post_ln_g, layer.shared.post_ln_b, cfg.layer_norm_eps)
    attn_in_dt = attn_in.cast(x.dtype) if attn_in.dtype != x.dtype else attn_in
    mlp_in_dt  = mlp_in.cast(x.dtype) if mlp_in.dtype != x.dtype else mlp_in

    q = (attn_in_dt @ layer.wq + layer.bq).reshape(B, S, n_heads, head_dim).transpose(1, 2)
    k = (attn_in_dt @ layer.wk + layer.bk).reshape(B, S, n_heads, head_dim).transpose(1, 2)
    v = (attn_in_dt @ layer.shared.wv + layer.shared.bv).reshape(B, S, n_heads, head_dim).transpose(1, 2)

    q = _rotate_q_pi(q, q_rot_cos, q_rot_sin)

    scale  = 1.0 / math.sqrt(head_dim)
    scores = q @ k.transpose(-2, -1) * scale
    scores = scores + attn_bias.cast(scores.dtype)
    attn = scores.clip(-1e4, 1e4).softmax(-1)

    # Extract attention weights as numpy: (B, n_heads, S, S) → (n_heads, S, S)
    attn_np = attn.cast(dtypes.float).realize().numpy()[0].copy()  # (n_heads, T, T)

    ctx = (attn @ v).transpose(1, 2).reshape(B, S, H)
    attn_out = ctx @ layer.shared.wo + layer.shared.bo

    ff = (mlp_in_dt @ layer.w_in + layer.b_in).gelu()
    ffn_out = ff @ layer.shared.w_out + layer.shared.b_out

    return x + attn_out + ffn_out, attn_np


# ============================================================
# v98 sudoku eager forward with per-layer hooks
# ============================================================

def v98_eager_forward_with_hooks(
    model,
    input_cells: Tensor,
    K: int,
):
    """
    Runs v98 sudoku forward eagerly, capturing per-layer per-breath:
      - attn_weights: list[K][4 layers] of (n_heads, 81, 81) numpy arrays
      - post_layer_x: list[K][4 layers] of (81, H) numpy arrays
      - pre_breath_x: list[K] of (81, H) numpy arrays
    Also returns final cell_logits for accuracy evaluation.
    """
    assert int(input_cells.shape[0]) == 1, "B must be 1 for eager probe"

    state_embed    = model.sudoku_state_embed
    position_embed = model.sudoku_position_embed
    attn_bias_t    = model.sudoku_attn_bias   # (n_heads, 81, 81) additive bias in {0, -1e4}
    breath_embed   = model.sudoku_breath_embed
    delta_gate     = model.sudoku_delta_gate

    x = embed_sudoku(input_cells, state_embed, position_embed)
    x = x.cast(dtypes.half) if x.dtype != dtypes.half else x

    layers = list(model.block.layers)
    K_max  = int(breath_embed.shape[0])

    # Photon disabled (alpha=0 default in v98 checkpoint)
    breath_cos   = [1.0] * K_max
    breath_sin   = [0.0] * K_max

    attn_per_breath   = []   # list[K] of list[4] of (n_heads, 81, 81) np
    post_x_per_breath = []   # list[K] of list[4] of (81, H) np
    pre_x_per_breath  = []   # list[K] of (81, H) np
    cell_logits_history = []

    digit_codebook = model.sudoku_digit_codebook

    for k in range(K):
        be_k  = breath_embed[k].reshape(1, 1, -1).cast(x.dtype)
        x_in  = x + be_k
        x_pre = x

        pre_x_per_breath.append(x_pre.cast(dtypes.float).realize().numpy()[0].copy())  # (81, H)

        cos_k = breath_cos[k]
        sin_k = breath_sin[k]

        attn_breath_k  = []
        post_x_breath_k = []
        h = x_in
        for li, layer in enumerate(layers[:4]):
            h, attn_np = _v98_layer_forward_hooked(layer, h, attn_bias_t, cos_k, sin_k)
            attn_breath_k.append(attn_np)   # (n_heads, 81, 81)
            post_x_breath_k.append(h.cast(dtypes.float).realize().numpy()[0].copy())  # (81, H)

        attn_per_breath.append(attn_breath_k)
        post_x_per_breath.append(post_x_breath_k)

        gate_k = delta_gate[k].cast(h.dtype).reshape(1, 1, 1)
        delta  = h - x_pre
        x      = x_pre + gate_k * delta

        x_ln = _layernorm(x, model.ln_f_g, model.ln_f_b, model.cfg.layer_norm_eps).cast(dtypes.float)
        cell_logits_k = x_ln @ digit_codebook.T.cast(dtypes.float)
        cell_logits_history.append(cell_logits_k.realize().numpy()[0])  # (81, 9)

    return attn_per_breath, post_x_per_breath, pre_x_per_breath, cell_logits_history


def _v98_layer_forward_hooked(layer, x, attn_bias_t, q_rot_cos=1.0, q_rot_sin=0.0):
    """Sudoku layer forward with attention weight extraction."""
    cfg = layer.cfg
    B, S, H = x.shape
    assert int(S) == 81, f"sudoku layer expects 81 cells, got {S}"
    n_heads = cfg.n_heads

    attn_in = _layernorm(x, layer.shared.in_ln_g, layer.shared.in_ln_b, cfg.layer_norm_eps)
    mlp_in  = _layernorm(x, layer.shared.post_ln_g, layer.shared.post_ln_b, cfg.layer_norm_eps)
    attn_in_dt = attn_in.cast(x.dtype) if attn_in.dtype != x.dtype else attn_in
    mlp_in_dt  = mlp_in.cast(x.dtype) if mlp_in.dtype != x.dtype else mlp_in

    q = (attn_in_dt @ layer.wq + layer.bq).reshape(B, S, n_heads, cfg.head_dim).transpose(1, 2)
    k = (attn_in_dt @ layer.wk + layer.bk).reshape(B, S, n_heads, cfg.head_dim).transpose(1, 2)
    v = (attn_in_dt @ layer.shared.wv + layer.shared.bv).reshape(B, S, n_heads, cfg.head_dim).transpose(1, 2)

    scale  = 1.0 / math.sqrt(cfg.head_dim)
    scores = q @ k.transpose(-2, -1) * scale
    scores = scores + attn_bias_t.cast(scores.dtype).reshape(1, n_heads, S, S)
    attn   = scores.clip(-1e4, 1e4).softmax(-1)

    attn_np = attn.cast(dtypes.float).realize().numpy()[0].copy()  # (n_heads, 81, 81)

    ctx = (attn @ v).transpose(1, 2).reshape(B, S, H)
    attn_out = ctx @ layer.shared.wo + layer.shared.bo

    ff      = (mlp_in_dt @ layer.w_in + layer.b_in).gelu()
    ffn_out = ff @ layer.shared.w_out + layer.shared.b_out

    return x + attn_out + ffn_out, attn_np


# ============================================================
# Per-puzzle metric computation
# ============================================================

def compute_attn_jsd_metrics(
    attn_per_breath,  # list[K][n_layers] of (n_heads, T, T) np
    head_groups: dict,  # {group_name: [head_indices]}
    n_layers: int = 4,
    K: int = 8,
):
    """
    Returns:
      attn_jsd_mean: (n_layers, n_groups, K-1) float array — mean JSD per row
      attn_jsd_max:  (n_layers, n_groups, K-1) float array — max JSD per row
    Averages over all query rows (all positions, no masking).
    """
    group_names = list(head_groups.keys())
    n_groups = len(group_names)
    attn_jsd_mean = np.zeros((n_layers, n_groups, K - 1))
    attn_jsd_max  = np.zeros((n_layers, n_groups, K - 1))

    for li in range(n_layers):
        for ki in range(K - 1):
            attn_k   = attn_per_breath[ki][li]    # (n_heads, T, T)
            attn_k1  = attn_per_breath[ki+1][li]  # (n_heads, T, T)
            n_heads, T, _ = attn_k.shape

            for gi, gname in enumerate(group_names):
                heads = head_groups[gname]
                # Collect JSD per query row, averaged within head group
                jsd_rows_list = []
                for q_row in range(T):
                    p_heads = attn_k[heads, q_row, :]    # (n_group_heads, T)
                    q_heads = attn_k1[heads, q_row, :]   # (n_group_heads, T)
                    # Average across heads in group first, then JSD
                    p_mean = p_heads.mean(axis=0)         # (T,)
                    q_mean = q_heads.mean(axis=0)         # (T,)
                    jsd_val = jsd_rows(
                        p_mean[np.newaxis, :],
                        q_mean[np.newaxis, :]
                    )[0]
                    jsd_rows_list.append(jsd_val)

                jsd_arr = np.array(jsd_rows_list)  # (T,)
                attn_jsd_mean[li, gi, ki] = jsd_arr.mean()
                attn_jsd_max[li, gi, ki]  = jsd_arr.max()

    return attn_jsd_mean, attn_jsd_max


def compute_residual_metrics(
    post_x_per_breath,  # list[K][n_layers] of (T, H) np
    node_type_mask,     # (T,) bool array — True = "given" / observed type
    K: int,
    n_layers: int = 4,
):
    """
    Returns:
      rel_norm: (n_layers, K-1, 3) — pooled, given, non-given
      cos_delta:(n_layers, K-1, 3) — directional persistence
    For K-1 consecutive pairs.
    3 columns: all, given, non-given.
    """
    rel_norm  = np.full((n_layers, K - 1, 3), np.nan)
    cos_delta = np.full((n_layers, K - 1, 3), np.nan)

    for li in range(n_layers):
        for ki in range(K - 1):
            x_k   = post_x_per_breath[ki][li]    # (T, H)
            x_k1  = post_x_per_breath[ki+1][li]  # (T, H)
            delta_k = x_k1 - x_k                  # (T, H)

            norm_k  = np.linalg.norm(x_k, axis=-1)   # (T,)
            norm_dk = np.linalg.norm(delta_k, axis=-1) # (T,)
            rn_k    = norm_dk / (norm_k + EPS)          # (T,)

            # Cosine persistence: cos(Δ_k, Δ_{k+1}) — need Δ_{k+1}
            if ki + 1 < K - 1:
                x_k2    = post_x_per_breath[ki+2][li]
                delta_k1 = x_k2 - x_k1             # (T, H)
                # per-position cosine between delta_k and delta_k1
                dot = (delta_k * delta_k1).sum(axis=-1)  # (T,)
                nd  = (np.linalg.norm(delta_k, axis=-1) *
                       np.linalg.norm(delta_k1, axis=-1))  # (T,)
                cos_k = dot / (nd + EPS)
            else:
                cos_k = np.full(len(node_type_mask), np.nan)

            # Pool: all, given, non-given
            for ci, mask in enumerate([
                np.ones(len(node_type_mask), dtype=bool),
                node_type_mask,
                ~node_type_mask,
            ]):
                if mask.sum() > 0:
                    rel_norm[li, ki, ci]  = rn_k[mask].mean()
                    if not np.all(np.isnan(cos_k)):
                        cos_delta[li, ki, ci] = cos_k[mask][~np.isnan(cos_k[mask])].mean() if (~np.isnan(cos_k[mask])).sum() > 0 else np.nan

    return rel_norm, cos_delta


# ============================================================
# Aggregate metrics over puzzles, split by outcome
# ============================================================

def accumulate_metrics(results_list, K, n_layers, n_groups):
    """
    results_list: list of dicts with keys:
      'solved': bool
      'attn_jsd_mean': (n_layers, n_groups, K-1)
      'attn_jsd_max':  (n_layers, n_groups, K-1)
      'rel_norm':      (n_layers, K-1, 3)
      'cos_delta':     (n_layers, K-1, 3)

    Returns:
      For each outcome (all, solved, failed):
        mean of each metric across puzzles.
    """
    def _mean(arrs):
        if not arrs:
            return None
        stacked = np.stack(arrs, axis=0)
        return np.nanmean(stacked, axis=0)

    splits = {
        "all":    [r for r in results_list],
        "solved": [r for r in results_list if r["solved"]],
        "failed": [r for r in results_list if not r["solved"]],
    }

    out = {}
    for split_name, items in splits.items():
        if not items:
            out[split_name] = None
            continue
        out[split_name] = {
            "attn_jsd_mean": _mean([r["attn_jsd_mean"] for r in items]),
            "attn_jsd_max":  _mean([r["attn_jsd_max"]  for r in items]),
            "rel_norm":      _mean([r["rel_norm"]       for r in items]),
            "cos_delta":     _mean([r["cos_delta"]      for r in items]),
            "n": len(items),
        }
    return out


# ============================================================
# Freeze-breath tables
# ============================================================

def make_freeze_table(agg, K, group_names, eps_frac=0.05):
    """
    agg: result of accumulate_metrics for a single split.
    Returns string table.
    """
    if agg is None:
        return "  (no data)"

    n_layers = 4
    layer_names = ["L0", "L1", "L2", "L3"]
    n_groups = len(group_names)

    lines = []

    # attn_jsd_mean
    lines.append("  attn-JSD-mean (freeze breath per layer):")
    header = "  Metric          " + "  ".join(f"{ln:6s}" for ln in layer_names)
    lines.append(header)
    for gi, gname in enumerate(group_names):
        row = f"  {gname:14s}"
        for li in range(n_layers):
            trace = agg["attn_jsd_mean"][li, gi, :].tolist()
            fk = freeze_breath(trace, eps_frac)
            row += f"  k={fk:2d}  "
        lines.append(row)

    # attn_jsd_max
    lines.append("")
    lines.append("  attn-JSD-max (freeze breath per layer):")
    lines.append(header)
    for gi, gname in enumerate(group_names):
        row = f"  {gname:14s}"
        for li in range(n_layers):
            trace = agg["attn_jsd_max"][li, gi, :].tolist()
            fk = freeze_breath(trace, eps_frac)
            row += f"  k={fk:2d}  "
        lines.append(row)

    # residual rel_norm (pooled = col 0)
    lines.append("")
    lines.append("  residual-rel-norm (freeze breath per layer, pooled):")
    row = "  "
    for li in range(n_layers):
        trace = agg["rel_norm"][li, :, 0].tolist()
        fk = freeze_breath(trace, eps_frac)
        row += f"{layer_names[li]}=k{fk:2d}  "
    lines.append(row)

    # residual cos_delta (pooled = col 0)
    lines.append("")
    lines.append("  residual-cos-delta (freeze breath per layer, pooled):")
    row2 = "  "
    for li in range(n_layers):
        trace_raw = agg["cos_delta"][li, :, 0].tolist()
        # cos-delta: high = frozen direction. Look for where it SATURATES (>0.95)
        # Use inverse: 1 - |cos| as "direction change signal", freeze where it drops
        trace = [1.0 - abs(v) if not math.isnan(v) else np.nan for v in trace_raw]
        fk = freeze_breath([v for v in trace if not math.isnan(v)], eps_frac) if any(not math.isnan(v) for v in trace) else K-2
        row2 += f"{layer_names[li]}=k{fk:2d}  "
    lines.append(row2)

    return "\n".join(lines)


def make_raw_trace_block(agg, K, group_names, n_layers=4):
    """Print raw per-breath trace vectors."""
    if agg is None:
        return "  (no data)"
    lines = []
    layer_names = ["L0", "L1", "L2", "L3"]
    breath_labels = [f"k{i}→{i+1}" for i in range(K - 1)]

    # attn_jsd_mean traces
    lines.append("  Raw attn-JSD-mean traces (per group, per layer):")
    lines.append("  " + " ".join(f"{lb:8s}" for lb in breath_labels))
    for li in range(n_layers):
        for gi, gname in enumerate(group_names):
            trace = agg["attn_jsd_mean"][li, gi, :]
            vals = " ".join(f"{v:.5f}" for v in trace)
            lines.append(f"  {layer_names[li]}-{gname:8s}: {vals}")

    lines.append("")
    lines.append("  Raw attn-JSD-max traces (per group, per layer):")
    lines.append("  " + " ".join(f"{lb:8s}" for lb in breath_labels))
    for li in range(n_layers):
        for gi, gname in enumerate(group_names):
            trace = agg["attn_jsd_max"][li, gi, :]
            vals = " ".join(f"{v:.5f}" for v in trace)
            lines.append(f"  {layer_names[li]}-{gname:8s}: {vals}")

    lines.append("")
    lines.append("  Raw residual rel_norm traces (all / given / non-given):")
    for li in range(n_layers):
        for ci, cname in enumerate(["all", "given", "non-given"]):
            trace = agg["rel_norm"][li, :, ci]
            vals = " ".join(f"{v:.5f}" for v in trace)
            lines.append(f"  {layer_names[li]}-{cname:10s}: {vals}")

    lines.append("")
    lines.append("  Raw residual cos-delta traces (all / given / non-given):")
    for li in range(n_layers):
        for ci, cname in enumerate(["all", "given", "non-given"]):
            trace = agg["cos_delta"][li, :, ci]
            vals = " ".join(f"{v:+.4f}" if not math.isnan(v) else "   nan  " for v in trace)
            lines.append(f"  {layer_names[li]}-{cname:10s}: {vals}")

    return "\n".join(lines)


def make_verdict_2x2(agg, K, group_names, eps_frac=0.05):
    """2x2: attn × residual, moving vs frozen per layer."""
    if agg is None:
        return "  (no data)"
    lines = []
    layer_names = ["L0", "L1", "L2", "L3"]
    for li in range(n_layers := 4):
        # Attn: aggregate freeze breath across groups, take MINIMUM (most moving group)
        attn_fk_by_group = []
        for gi in range(len(group_names)):
            trace = agg["attn_jsd_mean"][li, gi, :].tolist()
            attn_fk_by_group.append(freeze_breath(trace, eps_frac))
        attn_min_freeze = min(attn_fk_by_group)
        attn_max_freeze = max(attn_fk_by_group)

        # Residual: use rel_norm pooled
        res_trace = agg["rel_norm"][li, :, 0].tolist()
        res_fk = freeze_breath(res_trace, eps_frac)

        # Verdict: frozen if freeze_breath <= 4 (absolute; i.e. freeze happens in first half)
        half_K = (K - 1) // 2
        attn_verdict = "FROZEN" if attn_min_freeze <= half_K else "MOVING"
        res_verdict  = "FROZEN" if res_fk        <= half_K else "MOVING"

        lines.append(f"  {layer_names[li]}: attn={attn_verdict} (min_freeze=k{attn_min_freeze}, max_freeze=k{attn_max_freeze})  "
                     f"residual={res_verdict} (freeze=k{res_fk})")

    return "\n".join(lines)


# ============================================================
# Main
# ============================================================

def run_v110_probe(model, records, K, n_max, f_max, n_digits,
                   alternation, phase_scale, gate_profile, photon_alpha,
                   head_groups):
    """Run probe on v110-step3 checkpoint. Returns list of per-puzzle dicts."""
    Tensor.training = False
    results = []
    n_groups = len(head_groups)
    group_names = list(head_groups.keys())

    print(f"\n--- v110-step3 probe: {len(records)} puzzles, K={K} ---")
    t0 = time.time()

    for pi, rec in enumerate(records):
        batch_np = _records_to_batch_v107(
            [rec],
            n_max=n_max, f_max=f_max, k_max=K, n_heads=V110_STEP3_N_HEADS,
        )
        domain_init    = Tensor(batch_np["domain_init"]).cast(dtypes.float).contiguous().realize()
        node_kinds     = Tensor(batch_np["node_kinds"]).cast(dtypes.int).contiguous().realize()
        staging_mask_t = Tensor(batch_np["staging_mask"]).cast(dtypes.float).contiguous().realize()
        head_op_mask_t = Tensor(batch_np["head_op_mask"]).cast(dtypes.float).contiguous().realize()
        obs_mask       = batch_np["observed_mask"].numpy() if hasattr(batch_np["observed_mask"], "numpy") else np.array(batch_np["observed_mask"])  # (1, n_max)
        gold_bins_np   = batch_np["gold_bins"].numpy() if hasattr(batch_np["gold_bins"], "numpy") else np.array(batch_np["gold_bins"])  # (1, n_max)

        attn_per_breath, post_x_per_breath, pre_x_per_breath, tree_logits_history = \
            v110_eager_forward_with_hooks(
                model, domain_init, node_kinds, staging_mask_t, head_op_mask_t,
                K=K, n_max=n_max, f_max=f_max, n_digits=n_digits,
                alternation=alternation, phase_scale=phase_scale,
                gate_profile=gate_profile, photon_alpha=photon_alpha,
            )

        # Compute cell_acc
        final_logits = tree_logits_history[-1]  # (n_max, n_digits, 10)
        pred_digits = final_logits.argmax(axis=-1)  # (n_max, n_digits)
        gold_digits = bins_to_digits_msd(gold_bins_np[0:1], n_digits=n_digits)[0]  # (n_max, n_digits)
        cell_eq     = (pred_digits == gold_digits).all(axis=-1)  # (n_max,)
        unobs_mask  = (1 - obs_mask[0]).astype(bool)  # (n_max,)
        n_unobs     = unobs_mask.sum()
        cell_acc    = float(cell_eq[unobs_mask].sum()) / max(n_unobs, 1)
        solved      = cell_acc >= SOLVED_THRESHOLD

        # observed_mask for node types (T = n_max + f_max)
        # first n_max positions are variables; obs_mask=1 → given
        T = n_max + f_max
        node_type_mask = np.zeros(T, dtype=bool)
        node_type_mask[:n_max] = obs_mask[0].astype(bool)

        attn_jsd_mean, attn_jsd_max = compute_attn_jsd_metrics(
            attn_per_breath, head_groups, n_layers=4, K=K
        )
        rel_norm, cos_delta = compute_residual_metrics(
            post_x_per_breath, node_type_mask, K=K, n_layers=4
        )

        results.append({
            "solved":        solved,
            "cell_acc":      cell_acc,
            "attn_jsd_mean": attn_jsd_mean.tolist(),
            "attn_jsd_max":  attn_jsd_max.tolist(),
            "rel_norm":      rel_norm.tolist(),
            "cos_delta":     cos_delta.tolist(),
        })

        if (pi + 1) % 10 == 0 or pi == 0:
            dt = time.time() - t0
            n_solved = sum(1 for r in results if r["solved"])
            print(f"  [{pi+1:3d}/{len(records)}] dt={dt:.0f}s  cell_acc={cell_acc:.3f}  "
                  f"solved={n_solved}/{len(results)}", flush=True)

    return results


def run_v98_probe(model, loader_records, K, head_groups, given_mask_fn=None):
    """Run probe on v98 checkpoint. Returns list of per-puzzle dicts."""
    Tensor.training = False
    results = []
    group_names = list(head_groups.keys())
    n_groups = len(group_names)

    print(f"\n--- v98 sudoku probe: {len(loader_records)} puzzles, K={K} ---")
    t0 = time.time()

    for pi, (input_np, gold_np) in enumerate(loader_records):
        # input_np: (81,) int, gold_np: (81,) int 1..9
        input_t = Tensor(input_np[np.newaxis], dtype=dtypes.int).contiguous().realize()  # (1, 81)
        gold_t  = Tensor(gold_np[np.newaxis], dtype=dtypes.int).contiguous().realize()   # (1, 81)

        attn_per_breath, post_x_per_breath, pre_x_per_breath, cell_logits_history = \
            v98_eager_forward_with_hooks(model, input_t, K)

        # Compute cell_acc (all 81 cells, no masking — v98 predicts all)
        final_logits = cell_logits_history[-1]  # (81, 9)
        pred_digits  = final_logits.argmax(axis=-1) + 1  # (81,) 1..9
        cell_acc     = float((pred_digits == gold_np).mean())
        solved       = cell_acc >= SOLVED_THRESHOLD

        # node type mask: given cells are where input_np > 0
        # S=81 for sudoku (no factor nodes)
        node_type_mask = (input_np > 0)  # (81,) bool — given=True

        attn_jsd_mean, attn_jsd_max = compute_attn_jsd_metrics(
            attn_per_breath, head_groups, n_layers=4, K=K
        )
        rel_norm, cos_delta = compute_residual_metrics(
            post_x_per_breath, node_type_mask, K=K, n_layers=4
        )

        results.append({
            "solved":        solved,
            "cell_acc":      cell_acc,
            "attn_jsd_mean": attn_jsd_mean.tolist(),
            "attn_jsd_max":  attn_jsd_max.tolist(),
            "rel_norm":      rel_norm.tolist(),
            "cos_delta":     cos_delta.tolist(),
        })

        if (pi + 1) % 10 == 0 or pi == 0:
            dt = time.time() - t0
            n_solved = sum(1 for r in results if r["solved"])
            print(f"  [{pi+1:3d}/{len(loader_records)}] dt={dt:.0f}s  cell_acc={cell_acc:.3f}  "
                  f"solved={n_solved}/{len(results)}", flush=True)

    return results


# ============================================================
# Report printing
# ============================================================

def print_report(name, results, K, head_groups, eps_frac=0.05):
    group_names = list(head_groups.keys())
    n_groups = len(group_names)
    n_layers = 4
    layer_names = ["L0", "L1", "L2", "L3"]

    agg = accumulate_metrics(results, K, n_layers, n_groups)

    print(f"\n{'='*70}")
    print(f"REPORT: {name}  K={K}  n_puzzles={len(results)}")
    print(f"{'='*70}")
    n_solved = sum(1 for r in results if r["solved"])
    n_failed = len(results) - n_solved
    print(f"  solved={n_solved}  failed={n_failed}  threshold={SOLVED_THRESHOLD}")
    print(f"  eps_frac={eps_frac} (threshold = 5% of breath-0 value)")

    for split_name in ["all", "solved", "failed"]:
        a = agg[split_name]
        if a is None:
            print(f"\n  [{split_name.upper()}] (no data)")
            continue
        print(f"\n  [{split_name.upper()} n={a['n']}]")
        print()
        print("  --- FREEZE-BREATH TABLE ---")
        print(make_freeze_table(a, K, group_names, eps_frac))
        print()
        print("  --- RAW TRACE VECTORS ---")
        print(make_raw_trace_block(a, K, group_names, n_layers))
        print()
        print("  --- 2x2 VERDICT (attention x residual) ---")
        print(make_verdict_2x2(a, K, group_names, eps_frac))

    return agg


# ============================================================
# Delta-gate clamp ablation (v110 only)
# ============================================================

def run_delta_gate_clamp_ablation(model, records, K, n_max, f_max, n_digits,
                                   alternation, phase_scale, gate_profile,
                                   photon_alpha, head_groups, clamp_val=1.27):
    """
    Re-run v110 probe with delta_gate[4:] clamped to clamp_val.
    Returns results list same format as run_v110_probe.
    """
    orig_dg = model.fg_v107_delta_gate.numpy().copy()
    clamped = orig_dg.copy()
    clamped[4:] = clamp_val
    model.fg_v107_delta_gate.assign(
        Tensor(clamped, dtype=dtypes.float)
    ).realize()
    print(f"\n  [delta_gate clamp ablation] B4-B7 set to {clamp_val}")
    print(f"  original: {orig_dg}")
    print(f"  clamped:  {clamped}")

    results = run_v110_probe(
        model, records, K, n_max, f_max, n_digits,
        alternation, phase_scale, gate_profile, photon_alpha, head_groups,
    )

    # Restore
    model.fg_v107_delta_gate.assign(
        Tensor(orig_dg, dtype=dtypes.float)
    ).realize()
    print(f"  [delta_gate clamp ablation] restored original gates")

    return results


# ============================================================
# Entry point
# ============================================================

def main():
    print("=" * 70)
    print("Interior dynamics probe: v110-step3 + v98 sudoku")
    print("=" * 70)
    print(f"  N_PUZZLES={N_PUZZLES}  SEED={SEED}  SOLVED_THRESHOLD={SOLVED_THRESHOLD}")
    print(f"  EPS={EPS}")
    print()

    np.random.seed(SEED)
    Tensor.manual_seed(SEED)

    # ---- Load shared Pythia backbone ----
    cfg = Config()
    sd = _load_state()
    model = load_breathing(cfg, sd=sd)
    del sd

    all_results = {}

    # ============================================================
    # v110-step3 probe
    # ============================================================
    print("\n[1/2] Setting up v110-step3 ...")
    cast_layers_fp32(model)
    attach_fg_params_v110_step3(
        model, hidden=cfg.hidden,
        n_max=V110_STEP3_N_MAX, f_max=V110_STEP3_F_MAX, k_max=V110_STEP3_K_MAX,
        n_digits=V110_STEP3_N_DIGITS, n_code=V110_STEP3_CODEBOOK_N,
        ib_centroids_path=V110_STEP3_IB_CENTROIDS,
        waist_dim=V110_STEP3_WAIST_DIM,
    )
    Device[Device.DEFAULT].synchronize()
    load_ckpt_v110_step3(model, V110_CKPT)
    print(f"  Loaded: {V110_CKPT}")
    print(f"  K={V110_STEP3_K_MAX}, N_DIGITS={V110_STEP3_N_DIGITS}, "
          f"N_MAX={V110_STEP3_N_MAX}, F_MAX={V110_STEP3_F_MAX}")

    v110_loader = FactorGraphLoaderV107(
        V110_VAL, batch_size=1,
        difficulty_filter="hard", curriculum=False,
        n_max=V110_STEP3_N_MAX, f_max=V110_STEP3_F_MAX, k_max=V110_STEP3_K_MAX,
        n_heads=V110_STEP3_N_HEADS,
        seed=SEED + 2,
    )
    v110_records = v110_loader.records[:N_PUZZLES]
    print(f"  Records loaded: {len(v110_records)} (difficulty=hard)")

    v110_results = run_v110_probe(
        model, v110_records,
        K=V110_STEP3_K_MAX,
        n_max=V110_STEP3_N_MAX, f_max=V110_STEP3_F_MAX, n_digits=V110_STEP3_N_DIGITS,
        alternation=V110_STEP3_ALTERNATION, phase_scale=V110_STEP3_PHASE_SCALE,
        gate_profile=V110_STEP3_GATE_PROFILE, photon_alpha=V110_STEP3_PHOTON_ALPHA,
        head_groups=V110_HEAD_GROUPS,
    )
    all_results["v110_step3_hard"] = v110_results
    v110_agg = print_report(
        "v110-step3, hard, K=8", v110_results,
        K=V110_STEP3_K_MAX, head_groups=V110_HEAD_GROUPS,
    )

    # ---- Delta-gate clamp ablation ----
    print(f"\n{'='*70}")
    print("ABLATION: delta_gate[B4-B7] clamped to 1.27 (peak)")
    print("="*70)
    v110_clamp_results = run_delta_gate_clamp_ablation(
        model, v110_records,
        K=V110_STEP3_K_MAX,
        n_max=V110_STEP3_N_MAX, f_max=V110_STEP3_F_MAX, n_digits=V110_STEP3_N_DIGITS,
        alternation=V110_STEP3_ALTERNATION, phase_scale=V110_STEP3_PHASE_SCALE,
        gate_profile=V110_STEP3_GATE_PROFILE, photon_alpha=V110_STEP3_PHOTON_ALPHA,
        head_groups=V110_HEAD_GROUPS, clamp_val=1.27,
    )
    all_results["v110_step3_hard_clamp127"] = v110_clamp_results
    v110_clamp_agg = print_report(
        "v110-step3, hard, K=8 [delta_gate clamp 1.27]",
        v110_clamp_results, K=V110_STEP3_K_MAX, head_groups=V110_HEAD_GROUPS,
    )

    # ============================================================
    # v98 sudoku probe
    # ============================================================
    print(f"\n{'='*70}")
    print("[2/2] Setting up v98 sudoku ...")
    SUDOKU_K = 20
    # Re-attach sudoku params (on same model — share the Pythia backbone)
    attach_sudoku_params(model, hidden=cfg.hidden, n_heads=cfg.n_heads, k_max=SUDOKU_K)
    # cast_layers_fp32 — already done above, no need to redo
    load_sudoku_ckpt(model, V98_CKPT)
    print(f"  Loaded: {V98_CKPT}")
    print(f"  K={SUDOKU_K}")

    sudoku_loader = SudokuLoader(
        V98_VAL, batch_size=1,
        difficulty_filter="medium",
        curriculum=False, seed=SEED,
    )
    # Collect records as (input, gold) pairs from the raw record list
    v98_records = []
    rng = np.random.RandomState(SEED)
    picks = sudoku_loader.records[:N_PUZZLES]
    for rec in picks:
        inp_np = np.array(rec["input"], dtype=np.int32)    # (81,)
        sol_np = np.array(rec["solution"], dtype=np.int32) # (81,)
        v98_records.append((inp_np, sol_np))
    print(f"  Records loaded: {len(v98_records)} (difficulty=medium)")

    v98_results = run_v98_probe(
        model, v98_records, K=SUDOKU_K, head_groups=V98_HEAD_GROUPS,
    )
    all_results["v98_sudoku_medium"] = v98_results
    v98_agg = print_report(
        "v98-sudoku, medium, K=20", v98_results,
        K=SUDOKU_K, head_groups=V98_HEAD_GROUPS,
    )

    # ============================================================
    # Save raw data
    # ============================================================
    out_path = ".cache/interior_probe_v110_v98.json"

    def _convert(obj):
        """Recursively convert numpy types to Python native for JSON serialization."""
        if isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_convert(v) for v in obj]
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating, float)):
            if math.isnan(obj) or math.isinf(obj):
                return None
            return float(obj)
        return obj

    with open(out_path, "w") as f:
        json.dump(_convert(all_results), f)
    print(f"\nRaw data saved to {out_path}")
    print("\nDONE.")


if __name__ == "__main__":
    main()
