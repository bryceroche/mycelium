"""Interior-dynamics diagnostic: attention JSD + residual stream motion per layer per breath.

Probes the BREATHING TRANSFORMER's INTERIOR (per-layer attention distributions
and residual stream state) for the two checkpoints established as project
benchmarks. This is the "transposition test" to the original Qwen-7B telegraph
finding: same JSD instrument, pointed at the student's interior, per layer.

Why interior (not belief-output): yesterday's freeze-breath findings showed that
output beliefs freeze at k≈4 (v110-step3 K=8) and k≈5 (v98 K=20), and the
inference-time delta_gate clamp test showed the damping is upstream of the
gate. This script localizes WHERE in the per-layer attention/residual interior
the dynamics actually freeze.

Two architectures via env var ARCH=v110_step3 (default) or ARCH=v98.

Per-checkpoint runs:
  v110-step3 cont8_step1000 on HARD (failure mass for this arch, cell_acc~0.52)
  v98 prod_final on MEDIUM (failure mass; easy is 79% solved)

Both populations are eval'd: SOLVED puzzles (all unobserved cells correct at
final breath) and FAILED puzzles. Failure-only is the instrument mistake.

Two metrics per layer (4 layers) per breath transition (K-1 transitions):
  M1 - ATTENTION JSD per query row, per head-group, mean and max over rows.
    Head groups:
      v98:  row(0-4) / col(5-9) / box(10-14) / global(15)
      v110: ADD(0-3) / SUB(4-7) / MUL(8-11) / DIV(12-15)
  M2 - RESIDUAL STREAM DELTA per position, split by node type, plus pooled.
    Per-position:
      rel_norm_t = ||x_{k+1,t} - x_{k,t}|| / max(||x_{k,t}||, eps)
      cos_t      = direction consistency with previous transition (only k>0)
    Node types:
      v98:  given cells (input != 0) vs blank cells (input == 0)
      v110: var nodes (idx 0..n_max) vs factor nodes (idx n_max..)

Output: per-checkpoint summary table (one per solved/failed), per-layer rows.
Plus a freeze-breath table per metric, and a 2x2 verdict per layer.

CRITICAL: Eager forwards ONLY. No TinyJit. JIT-unrolled graphs do not expose
intermediates. Batch size 1.

Estimated runtime: 50 puzzles * ~10-15s eager forward = 8-15 min per ckpt.

Usage:
  ARCH=v110_step3 .venv/bin/python scripts/diag_interior_jsd.py
  ARCH=v98        .venv/bin/python scripts/diag_interior_jsd.py
"""
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from tinygrad import Device, Tensor, dtypes

ARCH = os.environ.get("ARCH", "v110_step3")

# ---------------------------------------------------------------------------
# Env defaults for v110-step3 — set BEFORE the module imports them.
# ---------------------------------------------------------------------------

if ARCH == "v110_step3":
    os.environ.setdefault("V110_STEP3_TASK", "1")
    os.environ.setdefault("V110_STEP3_K_MAX", "8")
    os.environ.setdefault("V110_STEP3_N_DIGITS", "5")
    os.environ.setdefault("V110_STEP3_N_MAX", "16")
    os.environ.setdefault("V110_STEP3_F_MAX", "8")
    os.environ.setdefault("V110_STEP3_WAIST_DIM", "512")
    os.environ.setdefault("V110_STEP3_ALTERNATION", "1")
    os.environ.setdefault("V110_STEP3_PHASE_SCALE", "1.0")
elif ARCH == "v98":
    os.environ.setdefault("SUDOKU_TASK", "1")
    os.environ.setdefault("SUDOKU_K_MAX", "20")
else:
    raise SystemExit(f"unknown ARCH={ARCH}; expected v110_step3 or v98")


from mycelium import Config
from mycelium.loader import _load_state, load_breathing


# ---------------------------------------------------------------------------
# Helper functions shared between archs
# ---------------------------------------------------------------------------

EPS = 1e-10


def jsd_distribution(p: np.ndarray, q: np.ndarray, axis: int = -1) -> np.ndarray:
    """Jensen-Shannon divergence between two probability arrays.

    p, q: same shape; last axis is the distribution.
    Returns array with last axis dropped.
    """
    p = p + EPS
    q = q + EPS
    p = p / p.sum(axis=axis, keepdims=True)
    q = q / q.sum(axis=axis, keepdims=True)
    m = 0.5 * (p + q)
    kl_pm = (p * np.log(p / m)).sum(axis=axis)
    kl_qm = (q * np.log(q / m)).sum(axis=axis)
    return 0.5 * kl_pm + 0.5 * kl_qm


def find_freeze_breath(values: list[float], rel_frac: float = 0.05) -> int:
    """First k where values[k] <= rel_frac * values[0]. Returns len(values)-1 if no freeze."""
    if not values:
        return -1
    threshold = max(1e-12, abs(values[0]) * rel_frac)
    for k, v in enumerate(values):
        if v <= threshold:
            return k
    return len(values) - 1


# ---------------------------------------------------------------------------
# Manual layer forward with capture hooks
# ---------------------------------------------------------------------------

def _capture_layer_forward(layer, x, attn_bias, q_rot_cos=1.0, q_rot_sin=0.0,
                            rope: bool = False):
    """Run one BreathingLayer's forward with intermediate capture.

    Mirrors fg_layer_forward_v109pi / sudoku_layer_forward but returns
    (next_x, attention_probs). attention_probs shape: (n_heads, S, S).

    attn_bias: either (n_heads, S, S) [sudoku] or (B, n_heads, S, S) [factor graph].
    Q rotation matches the original v109pi rotation (Pythia-style RoPE-free for
    both archs).
    """
    cfg = layer.cfg
    B, S, H = x.shape
    n_heads = cfg.n_heads
    head_dim = cfg.head_dim

    from mycelium.breathing import _layernorm

    attn_in = _layernorm(x, layer.shared.in_ln_g, layer.shared.in_ln_b,
                          cfg.layer_norm_eps)
    mlp_in = _layernorm(x, layer.shared.post_ln_g, layer.shared.post_ln_b,
                        cfg.layer_norm_eps)
    attn_in_dt = attn_in.cast(x.dtype) if attn_in.dtype != x.dtype else attn_in
    mlp_in_dt = mlp_in.cast(x.dtype) if mlp_in.dtype != x.dtype else mlp_in

    q = (attn_in_dt @ layer.wq + layer.bq).reshape(B, S, n_heads, head_dim).transpose(1, 2)
    k = (attn_in_dt @ layer.wk + layer.bk).reshape(B, S, n_heads, head_dim).transpose(1, 2)
    v = (attn_in_dt @ layer.shared.wv + layer.shared.bv).reshape(B, S, n_heads, head_dim).transpose(1, 2)

    # Per-breath Q rotation. Skipped at default cos=1/sin=0 (sudoku v98 path).
    # v110-step3 (v109pi descendant) always rotates with cos_k/sin_k floats.
    if not (abs(q_rot_cos - 1.0) < 1e-9 and abs(q_rot_sin) < 1e-9):
        # Pairwise rotation across head_dim
        hd_int = int(head_dim)
        assert hd_int % 2 == 0
        q_pairs = q.reshape(B, n_heads, S, hd_int // 2, 2)
        q_even = q_pairs[..., 0]
        q_odd  = q_pairs[..., 1]
        cos_t = Tensor([q_rot_cos], dtype=q.dtype).reshape(1, 1, 1, 1)
        sin_t = Tensor([q_rot_sin], dtype=q.dtype).reshape(1, 1, 1, 1)
        out_even = cos_t * q_even - sin_t * q_odd
        out_odd  = sin_t * q_even + cos_t * q_odd
        out = Tensor.stack(out_even, out_odd, dim=-1)
        q = out.reshape(B, n_heads, S, hd_int)

    scale = 1.0 / math.sqrt(head_dim)
    scores = q @ k.transpose(-2, -1) * scale

    # Broadcast attn_bias to (B, n_heads, S, S)
    bias_t = attn_bias.cast(scores.dtype)
    if bias_t.ndim == 3:
        # (n_heads, S, S) for sudoku — broadcast batch dim
        bias_t = bias_t.reshape(1, n_heads, S, S)
    scores = scores + bias_t
    attn = scores.clip(-1e4, 1e4).softmax(-1)
    ctx = (attn @ v).transpose(1, 2).reshape(B, S, H)
    attn_out = ctx @ layer.shared.wo + layer.shared.bo

    ff = (mlp_in_dt @ layer.w_in + layer.b_in).gelu()
    ffn_out = ff @ layer.shared.w_out + layer.shared.b_out

    out_x = x + attn_out + ffn_out
    return out_x, attn


# ---------------------------------------------------------------------------
# Sudoku v98 — capture forward
# ---------------------------------------------------------------------------

def capture_v98_forward(model, input_cells: Tensor, K: int) -> dict:
    """Run K breaths of v98 sudoku forward; capture per-layer attention and
    residual stream at the END of each layer of each breath.

    Returns dict:
      attns: (K, L=4, n_heads=16, S=81, S=81) numpy fp32 (probabilities)
      resid: (K, L+1=5, S, H) numpy fp32 — residual at INPUT of each layer
             plus after final layer (so we have residual transitions per layer
             across breaths; the per-layer residual evolution).
      cell_logits: (K, S, 9) numpy fp32 (final post-LN @ codebook)
    """
    from mycelium.sudoku import embed_sudoku
    from mycelium.breathing import _layernorm

    state_embed = model.sudoku_state_embed
    position_embed = model.sudoku_position_embed
    attn_bias = model.sudoku_attn_bias
    breath_embed = model.sudoku_breath_embed
    delta_gate = model.sudoku_delta_gate

    x = embed_sudoku(input_cells, state_embed, position_embed)
    x = x.cast(dtypes.half) if x.dtype != dtypes.half else x

    layers = list(model.block.layers)
    assert len(layers) >= 4

    digit_codebook = model.sudoku_digit_codebook

    n_layers = 4
    S = 81

    attns_out = []        # K lists, each L items of (n_heads, S, S)
    resid_out = []        # K lists, each L+1 items of (S, H)
    cell_logits_out = []

    for k in range(K):
        be_k = breath_embed[k].reshape(1, 1, -1).cast(x.dtype)
        x_in = x + be_k
        x_pre = x

        # Capture residual at start of each layer (after breath embed for L0,
        # post-layer for L1..L3, and after L3 (which is the "post-L3" state).
        per_layer_residuals = []  # list of L+1 numpy (S, H)
        h = x_in
        per_layer_residuals.append(h.cast(dtypes.float).realize().numpy()[0])  # input to L0

        per_layer_attns = []  # list of L numpy (n_heads, S, S)
        for li, layer in enumerate(layers[:n_layers]):
            h, attn = _capture_layer_forward(layer, h, attn_bias,
                                              q_rot_cos=1.0, q_rot_sin=0.0)
            per_layer_attns.append(attn.cast(dtypes.float).realize().numpy()[0])
            per_layer_residuals.append(h.cast(dtypes.float).realize().numpy()[0])

        # delta gate update
        gate_k = delta_gate[k].cast(h.dtype).reshape(1, 1, 1)
        delta = h - x_pre
        x = x_pre + gate_k * delta

        # Readout
        x_ln = _layernorm(x, model.ln_f_g, model.ln_f_b, model.cfg.layer_norm_eps).cast(dtypes.float)
        cell_logits_k = (x_ln @ digit_codebook.T.cast(dtypes.float)).realize().numpy()[0]

        attns_out.append(per_layer_attns)
        resid_out.append(per_layer_residuals)
        cell_logits_out.append(cell_logits_k)

    return {
        "attns": np.array(attns_out, dtype=np.float32),       # (K, L, n_heads, S, S)
        "resid": np.array(resid_out, dtype=np.float32),       # (K, L+1, S, H)
        "cell_logits": np.array(cell_logits_out, dtype=np.float32),  # (K, S, 9)
    }


# ---------------------------------------------------------------------------
# v110-step3 — capture forward
# ---------------------------------------------------------------------------

def capture_v110_step3_forward(model, domain_init: Tensor, node_kinds: Tensor,
                                 staging_mask: Tensor, head_op_mask: Tensor,
                                 K: int, n_max: int, f_max: int, n_digits: int,
                                 alternation: bool, phase_scale: float,
                                 gate_profile: str, photon_alpha: float) -> dict:
    """Run K breaths of v110-step3 forward; capture per-layer attention and residual.

    Returns dict similar to capture_v98_forward.
    """
    from mycelium.factor_graph_v107 import embed_factor_graph_v100_aligned
    from mycelium.factor_graph_v109 import _apply_waist_v109
    from mycelium.factor_graph_v110_acc import _acc_notebook_read, _acc_notebook_write
    from mycelium.factor_graph_v110_photon import _photon_gate
    from mycelium.factor_graph_v110_step import V110_STEP_N_HEADS
    from mycelium.breathing import _layernorm

    domain_codebook   = model.fg_v107_domain_codebook
    var_state_embed   = model.fg_v107_var_state_embed
    var_pos_embed     = model.fg_v107_var_pos_embed
    factor_pos_embed  = model.fg_v107_factor_pos_embed
    node_kind_embed   = model.fg_v107_node_kind_embed
    breath_embed      = model.fg_v107_breath_embed
    delta_gate        = model.fg_v107_delta_gate
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
    K_max = int(breath_embed.shape[0])

    breath_phases = [phase_scale * k * math.pi / float(K_max) for k in range(K_max)]
    breath_cos = [math.cos(p) for p in breath_phases]
    breath_sin = [math.sin(p) for p in breath_phases]

    photon_gates = [_photon_gate(k, K_max, gate_profile) for k in range(K_max)]
    binary_gates = [_photon_gate(k, K_max, "binary") for k in range(K_max)]

    tree_cb_flat = tree_codebook.reshape(n_digits * 10, H)
    notebook_slots = []

    attns_out = []
    resid_out = []
    tree_logits_out = []

    n_layers = 4

    for k in range(K):
        x_acc_delta = _acc_notebook_read(
            x, notebook_slots,
            acc_W_q, acc_W_k, acc_W_v, acc_W_o, acc_b_o,
        )
        x = x + x_acc_delta

        be_k = breath_embed[k].reshape(1, 1, -1).cast(x.dtype)
        x_in = x + be_k
        x_pre = x

        stk = staging_mask[:, k, :, :]
        stk_h = stk.reshape(B, 1, T, T).expand(B, V110_STEP_N_HEADS, T, T)
        combined = stk_h.cast(x.dtype) + head_op_mask.cast(x.dtype)  # (B, n_heads, T, T)

        cos_k = breath_cos[k]
        sin_k = breath_sin[k]

        per_layer_residuals = []
        h = x_in
        per_layer_residuals.append(h.cast(dtypes.float).realize().numpy()[0])

        per_layer_attns = []
        for layer in layers[:n_layers]:
            h, attn = _capture_layer_forward(layer, h, combined,
                                              q_rot_cos=cos_k, q_rot_sin=sin_k)
            per_layer_attns.append(attn.cast(dtypes.float).realize().numpy()[0])
            per_layer_residuals.append(h.cast(dtypes.float).realize().numpy()[0])

        # IB codebook
        cb = semantic_codebook.cast(h.dtype)
        tmp = temperature.cast(h.dtype)
        scores = h @ cb.T / tmp.reshape(1, 1, 1)
        weights = scores.clip(-1e4, 1e4).softmax(-1)
        recon = weights @ cb
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
        delta = h_quant - x_pre
        x = x_pre + gate_k * delta

        slot_k = _acc_notebook_write(x, acc_W_write, acc_b_write)
        notebook_slots.append(slot_k)

        x_ln = _layernorm(x, model.ln_f_g, model.ln_f_b,
                          model.cfg.layer_norm_eps).cast(dtypes.float)
        var_x = x_ln[:, :n_max, :]
        tree_logits_flat = var_x @ tree_cb_flat.T.cast(dtypes.float)
        tree_logits_k = tree_logits_flat.reshape(B, n_max, n_digits, 10).realize().numpy()[0]

        attns_out.append(per_layer_attns)
        resid_out.append(per_layer_residuals)
        tree_logits_out.append(tree_logits_k)

    return {
        "attns": np.array(attns_out, dtype=np.float32),       # (K, L, n_heads, T, T)
        "resid": np.array(resid_out, dtype=np.float32),       # (K, L+1, T, H)
        "tree_logits": np.array(tree_logits_out, dtype=np.float32),  # (K, n_max, n_digits, 10)
    }


# ---------------------------------------------------------------------------
# Per-puzzle metric reduction
# ---------------------------------------------------------------------------

def per_puzzle_metrics(captured: dict, head_groups: dict, type_masks: dict,
                       attn_bias: np.ndarray, K: int):
    """Compute per-layer per-breath-transition metrics for ONE puzzle.

    head_groups: dict[group_name, list[head_idx]]
    type_masks:  dict[type_name, np.ndarray[bool] of shape (T,)]  -- positions of that type
    attn_bias:   (n_heads, T, T) or (B, n_heads, T, T) — the additive mask used to
                 determine which keys are unmasked per (head, query) row.

    Returns dict:
      attn_jsd_means: (L, K-1, n_groups)  -- mean over query rows
      attn_jsd_maxes: (L, K-1, n_groups)  -- max over query rows
      group_names: list of group names ordered
      resid_rel_norm: dict[type] -> (L, K-1) ndarray of mean rel_norm per layer per transition
      resid_cos:      dict[type] -> (L, K-1) ndarray of mean cos with prev transition
                                  (k=0 entry undefined → set to NaN)
    """
    attns = captured["attns"]   # (K, L, n_heads, T, T)
    resid = captured["resid"]   # (K, L+1, T, H)

    K_obs, L, n_heads, T, _ = attns.shape
    assert K_obs == K

    # ---- per-row support mask: (n_heads, T, T_keys) bool ----
    if attn_bias.ndim == 3:
        ab = attn_bias                       # (n_heads, T, T)
    else:
        ab = attn_bias[0]                    # (n_heads, T, T)
    support_mask = (ab > -1e3)               # True where key is allowed (bias > -1e3)

    # ---- Attention JSD per group per layer per transition ----
    group_names = list(head_groups.keys())
    n_groups = len(group_names)
    K_minus_1 = K - 1

    attn_jsd_means = np.zeros((L, K_minus_1, n_groups), dtype=np.float32)
    attn_jsd_maxes = np.zeros((L, K_minus_1, n_groups), dtype=np.float32)

    # Vectorized: zero out masked positions, renormalize, then JSD over all T.
    # Masked positions have ~0 probability in p and q (softmax suppresses them
    # by -1e4 bias), so adding EPS there contributes ~0 to JSD after renorm.
    # This is mathematically equivalent to "JSD over support only" up to the
    # tiny EPS contribution from masked positions.
    for li in range(L):
        for k in range(K_minus_1):
            a_k = attns[k, li]           # (n_heads, T, T)
            a_kp1 = attns[k + 1, li]     # (n_heads, T, T)

            for gi, gname in enumerate(group_names):
                head_idxs = head_groups[gname]
                # Stack heads in this group: (g, T, T)
                p = a_k[head_idxs]
                q = a_kp1[head_idxs]
                sm = support_mask[head_idxs].astype(np.float32)   # (g, T, T)
                # Mask out blocked positions (force to zero) then add EPS + renorm
                p_masked = p * sm + EPS
                q_masked = q * sm + EPS
                p_masked = p_masked / p_masked.sum(axis=-1, keepdims=True)
                q_masked = q_masked / q_masked.sum(axis=-1, keepdims=True)
                m_masked = 0.5 * (p_masked + q_masked)
                kl_pm = (p_masked * np.log(p_masked / m_masked)).sum(axis=-1)   # (g, T)
                kl_qm = (q_masked * np.log(q_masked / m_masked)).sum(axis=-1)
                row_jsd = 0.5 * (kl_pm + kl_qm)                                 # (g, T)

                # Suppress rows with support<=1 (JSD=0 trivially)
                n_support = sm.sum(axis=-1)                                     # (g, T)
                row_jsd = np.where(n_support > 1, row_jsd, 0.0)

                # Mean over heads in group, then mean/max over query rows
                group_row_jsd = row_jsd.mean(axis=0)                            # (T,)
                attn_jsd_means[li, k, gi] = float(group_row_jsd.mean())
                attn_jsd_maxes[li, k, gi] = float(group_row_jsd.max())

    # ---- Residual delta per position per layer ----
    # We have L+1 residual slices per breath. Take "per layer" delta as the
    # change in the residual AT THE OUTPUT OF LAYER l between breath k and k+1.
    # Layer index l ∈ [0..L-1] → use resid[:, l+1, :, :] (output of layer l).
    type_names = list(type_masks.keys())
    resid_rel_norm = {tn: np.zeros((L, K_minus_1), dtype=np.float32) for tn in type_names}
    resid_cos      = {tn: np.full((L, K_minus_1), np.nan, dtype=np.float32) for tn in type_names}

    for li in range(L):
        # (K, T, H)
        per_layer_resid = resid[:, li + 1, :, :]
        # deltas: K-1 transitions; delta_k = x_{k+1} - x_k
        deltas = per_layer_resid[1:] - per_layer_resid[:-1]    # (K-1, T, H)
        norms_x = np.linalg.norm(per_layer_resid[:-1], axis=-1)  # (K-1, T)
        norms_d = np.linalg.norm(deltas, axis=-1)                # (K-1, T)
        rel_norms = norms_d / np.maximum(norms_x, 1e-8)          # (K-1, T)
        for k in range(K_minus_1):
            for tn in type_names:
                m = type_masks[tn]
                if not m.any():
                    continue
                resid_rel_norm[tn][li, k] = float(rel_norms[k][m].mean())

        # Direction consistency: cos(delta_k, delta_{k-1}) per position
        # cos_k undefined at k=0 (no previous transition); compute for k>=1
        for k in range(1, K_minus_1):
            d_now = deltas[k]    # (T, H)
            d_prev = deltas[k-1] # (T, H)
            num = (d_now * d_prev).sum(axis=-1)                        # (T,)
            denom = (np.linalg.norm(d_now, axis=-1) *
                     np.linalg.norm(d_prev, axis=-1) + 1e-12)          # (T,)
            cos_t = num / denom                                          # (T,)
            for tn in type_names:
                m = type_masks[tn]
                if not m.any():
                    continue
                resid_cos[tn][li, k] = float(cos_t[m].mean())

    return {
        "attn_jsd_means": attn_jsd_means,
        "attn_jsd_maxes": attn_jsd_maxes,
        "group_names": group_names,
        "resid_rel_norm": resid_rel_norm,
        "resid_cos": resid_cos,
    }


# ---------------------------------------------------------------------------
# Aggregation across many puzzles
# ---------------------------------------------------------------------------

def aggregate_per_layer(metric_list: list[dict], K: int, n_layers: int) -> dict:
    """Average per-puzzle metric dicts into a single summary dict.

    metric_list: list of dicts from per_puzzle_metrics.
    Returns dict with same keys, all averaged across puzzles.
    """
    if not metric_list:
        return None

    group_names = metric_list[0]["group_names"]
    type_names = list(metric_list[0]["resid_rel_norm"].keys())

    K_minus_1 = K - 1
    n_groups = len(group_names)

    am = np.stack([m["attn_jsd_means"] for m in metric_list], axis=0).mean(axis=0)
    ax = np.stack([m["attn_jsd_maxes"] for m in metric_list], axis=0).mean(axis=0)

    resid_rel = {}
    resid_cos = {}
    for tn in type_names:
        rr = np.stack([m["resid_rel_norm"][tn] for m in metric_list], axis=0)
        # All puzzles always have residuals → simple mean
        resid_rel[tn] = rr.mean(axis=0)
        rc = np.stack([m["resid_cos"][tn] for m in metric_list], axis=0)
        # nanmean for cos (k=0 is NaN)
        with np.errstate(invalid="ignore"):
            resid_cos[tn] = np.nanmean(rc, axis=0)

    return {
        "attn_jsd_means": am,
        "attn_jsd_maxes": ax,
        "group_names": group_names,
        "resid_rel_norm": resid_rel,
        "resid_cos": resid_cos,
        "n_puzzles": len(metric_list),
    }


def report_summary(label: str, summary: dict, K: int, n_layers: int):
    """Print per-layer summary table and freeze-breath table and 2x2 verdict."""
    if summary is None or summary["n_puzzles"] == 0:
        print(f"\n  {label}: no puzzles")
        return

    group_names = summary["group_names"]
    type_names = list(summary["resid_rel_norm"].keys())
    n = summary["n_puzzles"]
    K_minus_1 = K - 1

    print(f"\n  {label} (n={n})")
    print("  " + "=" * 78)

    # Per-layer per-breath rows: print every-other or just first/middle/last for compactness
    # The full per-breath data is in summary; print key breaths.
    print(f"\n  Per-layer attention JSD (mean over query rows; "
          f"groups: {', '.join(group_names)})")
    print(f"  {'Layer':<6} | " + " | ".join(f"{g:<14}" for g in group_names))
    for li in range(n_layers):
        line = f"  L{li:<5} | "
        cells = []
        for gi, _ in enumerate(group_names):
            traj = summary["attn_jsd_means"][li, :, gi]
            traj_s = " ".join(f"{v:.3f}" for v in traj)
            cells.append(traj_s)
        # Print per-row trajectory across breaths
        print(line + " | ".join(f"{c:<14}" for c in cells))
    # Print group max trajectory
    print(f"\n  Per-layer attention JSD (max over query rows)")
    print(f"  {'Layer':<6} | " + " | ".join(f"{g:<14}" for g in group_names))
    for li in range(n_layers):
        line = f"  L{li:<5} | "
        cells = []
        for gi, _ in enumerate(group_names):
            traj = summary["attn_jsd_maxes"][li, :, gi]
            traj_s = " ".join(f"{v:.3f}" for v in traj)
            cells.append(traj_s)
        print(line + " | ".join(f"{c:<14}" for c in cells))

    print(f"\n  Per-layer residual rel_norm (mean over positions of each type)")
    print(f"  {'Layer':<6} | " + " | ".join(f"{tn:<14}" for tn in type_names))
    for li in range(n_layers):
        line = f"  L{li:<5} | "
        cells = []
        for tn in type_names:
            traj = summary["resid_rel_norm"][tn][li, :]
            traj_s = " ".join(f"{v:.3f}" for v in traj)
            cells.append(traj_s)
        print(line + " | ".join(f"{c:<14}" for c in cells))

    print(f"\n  Per-layer residual delta-direction cos (k>=1 only; k=0=NaN)")
    print(f"  {'Layer':<6} | " + " | ".join(f"{tn:<14}" for tn in type_names))
    for li in range(n_layers):
        line = f"  L{li:<5} | "
        cells = []
        for tn in type_names:
            traj = summary["resid_cos"][tn][li, :]
            traj_s = " ".join("NaN" if np.isnan(v) else f"{v:.2f}" for v in traj)
            cells.append(traj_s)
        print(line + " | ".join(f"{c:<14}" for c in cells))

    # Freeze breath table
    print(f"\n  Freeze breath per metric per layer (first k where metric "
          f"<= 5% of early value)")
    header_metrics = [f"attn_{g}" for g in group_names] + [f"resid_{tn}" for tn in type_names]
    print(f"  {'Layer':<6} | " + " | ".join(f"{m:<12}" for m in header_metrics))
    freeze_data = {}  # (li, metric_name) -> int
    for li in range(n_layers):
        cells = []
        for gi, gname in enumerate(group_names):
            traj = list(summary["attn_jsd_means"][li, :, gi])
            fk = find_freeze_breath(traj, rel_frac=0.05)
            cells.append(f"k={fk}")
            freeze_data[(li, f"attn_{gname}")] = fk
        for tn in type_names:
            traj = list(summary["resid_rel_norm"][tn][li, :])
            fk = find_freeze_breath(traj, rel_frac=0.05)
            cells.append(f"k={fk}")
            freeze_data[(li, f"resid_{tn}")] = fk
        print(f"  L{li:<5} | " + " | ".join(f"{c:<12}" for c in cells))

    # 2x2 verdict per layer: attention vs residual freeze
    print(f"\n  2x2 verdict per layer (moving = freeze >= K/2; "
          f"K/2 = {K // 2})")
    print(f"  {'Layer':<6} | {'attn_state':<14} | {'resid_state':<14} | verdict")
    for li in range(n_layers):
        attn_fk = max(freeze_data[(li, f"attn_{g}")] for g in group_names)
        resid_fk = max(freeze_data[(li, f"resid_{tn}")] for tn in type_names)
        attn_moving = attn_fk >= K // 2
        resid_moving = resid_fk >= K // 2
        if attn_moving and resid_moving:
            verdict = "interior alive"
        elif attn_moving and not resid_moving:
            verdict = "attn churning over stationary state"
        elif not attn_moving and resid_moving:
            verdict = "non-attn paths"
        else:
            verdict = "architecture-wide dead"
        print(f"  L{li:<5} | "
              f"{'moving' if attn_moving else 'frozen':<14} | "
              f"{'moving' if resid_moving else 'frozen':<14} | {verdict}")


# ---------------------------------------------------------------------------
# Main per-architecture run
# ---------------------------------------------------------------------------

def run_v98():
    from mycelium.sudoku import attach_sudoku_params
    from mycelium.sudoku_data import SudokuLoader
    from scripts.sudoku_train import load_ckpt as load_sudoku_ckpt

    CKPT = os.environ.get("CKPT", ".cache/sudoku_ckpts/v98_prod_final.safetensors")
    VAL = os.environ.get("VAL", ".cache/sudoku_val.jsonl")
    DIFFICULTY = os.environ.get("DIFFICULTY", "medium")
    N_TARGET = int(os.environ.get("N_TARGET", "25"))   # 25 solved + 25 failed
    N_SEEN_MAX = int(os.environ.get("N_SEEN_MAX", "300"))
    K = int(os.environ.get("SUDOKU_K_MAX", "20"))

    print("=" * 78)
    print("Interior JSD probe — v98 Sudoku K=20 MEDIUM")
    print("=" * 78)
    print(f"  ckpt:       {CKPT}")
    print(f"  difficulty: {DIFFICULTY}")
    print(f"  K:          {K}")
    print(f"  N_target:   {N_TARGET} solved + {N_TARGET} failed")
    print(f"  N_seen_max: {N_SEEN_MAX} (cap on puzzles iterated)")
    print()

    cfg = Config()
    print("loading Pythia + sudoku params...")
    sd = _load_state()
    model = load_breathing(cfg, sd=sd)
    del sd
    attach_sudoku_params(model, hidden=cfg.hidden, n_heads=cfg.n_heads, k_max=K)
    load_sudoku_ckpt(model, CKPT)
    Device[Device.DEFAULT].synchronize()

    Tensor.training = False

    # Head groups for v98: row(0-4), col(5-9), box(10-14), global(15)
    head_groups = {
        "row":    [0, 1, 2, 3, 4],
        "col":    [5, 6, 7, 8, 9],
        "box":    [10, 11, 12, 13, 14],
        "global": [15],
    }

    attn_bias_np = model.sudoku_attn_bias.cast(dtypes.float).realize().numpy()  # (n_heads, 81, 81)

    loader = SudokuLoader(VAL, batch_size=1, difficulty_filter=DIFFICULTY,
                           curriculum=False, seed=42)

    solved_metrics = []
    failed_metrics = []

    n_seen = 0
    t0 = time.time()
    for input_cells, gold, picks in loader.iter_eval(batch_size=1):
        if n_seen >= N_SEEN_MAX:
            break
        if len(solved_metrics) >= N_TARGET and len(failed_metrics) >= N_TARGET:
            break

        cap = capture_v98_forward(model, input_cells, K=K)
        # Determine solved/failed: all unobserved cells correct at final breath
        cell_logits_final = cap["cell_logits"][-1]                  # (81, 9)
        pred = cell_logits_final.argmax(axis=-1) + 1                # (81,) digits 1..9
        gold_np = gold.realize().numpy()[0]                          # (81,)
        ic_np = input_cells.realize().numpy()[0]                     # (81,)
        unobs = (ic_np == 0)
        if not unobs.any():
            n_seen += 1
            continue
        all_correct = bool((pred[unobs] == gold_np[unobs]).all())

        # Type masks: given vs blank
        given_mask = (ic_np != 0)
        blank_mask = (ic_np == 0)
        pooled_mask = np.ones(81, dtype=bool)
        type_masks = {
            "given":  given_mask,
            "blank":  blank_mask,
            "pooled": pooled_mask,
        }

        metrics = per_puzzle_metrics(cap, head_groups, type_masks, attn_bias_np, K=K)

        if all_correct and len(solved_metrics) < N_TARGET:
            solved_metrics.append(metrics)
        elif (not all_correct) and len(failed_metrics) < N_TARGET:
            failed_metrics.append(metrics)

        n_seen += 1
        if n_seen % 10 == 0 or n_seen == 1:
            dt = time.time() - t0
            print(f"  [{n_seen:3d}] solved={len(solved_metrics)} "
                  f"failed={len(failed_metrics)} elapsed={dt:.0f}s "
                  f"({dt/n_seen:.1f}s/puzzle)", flush=True)

    print()
    print(f"  total seen: {n_seen}  solved: {len(solved_metrics)}  failed: {len(failed_metrics)}")

    solved_summary = aggregate_per_layer(solved_metrics, K=K, n_layers=4)
    failed_summary = aggregate_per_layer(failed_metrics, K=K, n_layers=4)

    print()
    print("=" * 78)
    print("v98 Sudoku K=20 MEDIUM — INTERIOR DYNAMICS")
    print("=" * 78)
    report_summary("SOLVED", solved_summary, K=K, n_layers=4)
    report_summary("FAILED", failed_summary, K=K, n_layers=4)


def run_v110_step3():
    from mycelium.factor_graph_v110_step3 import (
        V110_STEP3_N_MAX, V110_STEP3_F_MAX, V110_STEP3_N_HEADS,
        V110_STEP3_N_DIGITS, V110_STEP3_K_MAX,
        V110_STEP3_WAIST_DIM, V110_STEP3_ALTERNATION, V110_STEP3_PHASE_SCALE,
        V110_STEP3_GATE_PROFILE, V110_STEP3_PHOTON_ALPHA,
        V110_STEP3_CODEBOOK_N, V110_STEP3_IB_CENTROIDS,
        attach_fg_params_v110_step3,
    )
    from scripts.v110_step3_factor_graph_train import (
        load_ckpt_v110_step3, cast_layers_fp32,
    )
    from mycelium.factor_graph_data_v107 import (
        FactorGraphLoaderV107, _records_to_batch_v107,
    )
    from mycelium.factor_graph_v108 import bins_to_digits_msd

    CKPT = os.environ.get("CKPT",
        ".cache/fg_v110_step3_ckpts/v110_step3_cont8_step1000.safetensors")
    VAL = os.environ.get("VAL", ".cache/factor_graph_test.jsonl")
    DIFFICULTY = os.environ.get("DIFFICULTY", "hard")
    N_TARGET = int(os.environ.get("N_TARGET", "25"))
    N_SEEN_MAX = int(os.environ.get("N_SEEN_MAX", "300"))
    K = V110_STEP3_K_MAX

    print("=" * 78)
    print("Interior JSD probe — v110-step3 K=8 HARD")
    print("=" * 78)
    print(f"  ckpt:       {CKPT}")
    print(f"  difficulty: {DIFFICULTY}")
    print(f"  K:          {K}")
    print(f"  N_target:   {N_TARGET} solved + {N_TARGET} failed")
    print(f"  N_seen_max: {N_SEEN_MAX}")
    print()

    SEED = 42
    np.random.seed(SEED)
    Tensor.manual_seed(SEED)

    cfg = Config()
    print("loading Pythia + v110-step3 params...")
    sd = _load_state()
    model = load_breathing(cfg, sd=sd)
    del sd
    cast_layers_fp32(model)
    attach_fg_params_v110_step3(
        model, hidden=cfg.hidden,
        n_max=V110_STEP3_N_MAX, f_max=V110_STEP3_F_MAX, k_max=K,
        n_digits=V110_STEP3_N_DIGITS, n_code=V110_STEP3_CODEBOOK_N,
        ib_centroids_path=V110_STEP3_IB_CENTROIDS,
        waist_dim=V110_STEP3_WAIST_DIM,
    )
    Device[Device.DEFAULT].synchronize()
    load_ckpt_v110_step3(model, CKPT)
    Device[Device.DEFAULT].synchronize()

    Tensor.training = False

    # Head groups for v110-step3: ADD(0-3), SUB(4-7), MUL(8-11), DIV(12-15)
    head_groups = {
        "ADD": [0, 1, 2, 3],
        "SUB": [4, 5, 6, 7],
        "MUL": [8, 9, 10, 11],
        "DIV": [12, 13, 14, 15],
    }

    val_loader = FactorGraphLoaderV107(
        VAL, batch_size=1,
        difficulty_filter=DIFFICULTY, curriculum=False,
        n_max=V110_STEP3_N_MAX, f_max=V110_STEP3_F_MAX, k_max=K,
        n_heads=V110_STEP3_N_HEADS, seed=SEED + 2,
    )
    records = val_loader.records[:N_SEEN_MAX]

    n_max = V110_STEP3_N_MAX
    f_max = V110_STEP3_F_MAX

    solved_metrics = []
    failed_metrics = []

    t0 = time.time()
    for i, rec in enumerate(records):
        if len(solved_metrics) >= N_TARGET and len(failed_metrics) >= N_TARGET:
            break

        batch_np = _records_to_batch_v107(
            [rec], n_max=n_max, f_max=f_max, k_max=K, n_heads=V110_STEP3_N_HEADS,
        )
        domain_init = Tensor(batch_np["domain_init"], dtype=dtypes.float).contiguous().realize()
        node_kinds  = Tensor(batch_np["node_kinds"], dtype=dtypes.int).contiguous().realize()
        staging     = Tensor(batch_np["staging_mask"], dtype=dtypes.float).contiguous().realize()
        head_mask   = Tensor(batch_np["head_op_mask"], dtype=dtypes.float).contiguous().realize()

        cap = capture_v110_step3_forward(
            model, domain_init, node_kinds, staging, head_mask,
            K=K, n_max=n_max, f_max=f_max, n_digits=V110_STEP3_N_DIGITS,
            alternation=V110_STEP3_ALTERNATION,
            phase_scale=V110_STEP3_PHASE_SCALE,
            gate_profile=V110_STEP3_GATE_PROFILE,
            photon_alpha=V110_STEP3_PHOTON_ALPHA,
        )

        # Solved/failed via final tree_logits
        final_tree = cap["tree_logits"][-1]    # (n_max, n_digits, 10)
        pred_digits = final_tree.argmax(axis=-1)  # (n_max, n_digits)
        gold_digits = bins_to_digits_msd(
            batch_np["gold_bins"], n_digits=V110_STEP3_N_DIGITS,
        )[0]   # (n_max, n_digits)
        obs_mask = batch_np["observed_mask"][0]    # (n_max,)
        unobs_mask = (obs_mask == 0)
        if not unobs_mask.any():
            continue
        cell_eq = (pred_digits == gold_digits).all(axis=-1)  # (n_max,)
        all_correct = bool(cell_eq[unobs_mask].all())

        # Type masks: var_nodes (idx 0..n_max-1) vs factor_nodes (idx n_max..T)
        T = n_max + f_max
        node_kinds_np = batch_np["node_kinds"][0]  # (T,)
        # var nodes are kind 0 (given) or 1 (unknown); factor nodes are kind 2
        var_mask    = (node_kinds_np == 0) | (node_kinds_np == 1)
        factor_mask = (node_kinds_np == 2)
        pooled_mask = np.ones(T, dtype=bool)
        type_masks = {
            "var":    var_mask,
            "factor": factor_mask,
            "pooled": pooled_mask,
        }

        # head_op_mask in batch is (B, n_heads, T, T)
        head_mask_np = batch_np["head_op_mask"]   # (1, n_heads, T, T)

        metrics = per_puzzle_metrics(cap, head_groups, type_masks, head_mask_np, K=K)

        if all_correct and len(solved_metrics) < N_TARGET:
            solved_metrics.append(metrics)
        elif (not all_correct) and len(failed_metrics) < N_TARGET:
            failed_metrics.append(metrics)

        if (i + 1) % 10 == 0 or i == 0:
            dt = time.time() - t0
            print(f"  [{i+1:3d}/{len(records)}] solved={len(solved_metrics)} "
                  f"failed={len(failed_metrics)} elapsed={dt:.0f}s", flush=True)

    print()
    print(f"  total seen: {i+1}  solved: {len(solved_metrics)}  failed: {len(failed_metrics)}")

    solved_summary = aggregate_per_layer(solved_metrics, K=K, n_layers=4)
    failed_summary = aggregate_per_layer(failed_metrics, K=K, n_layers=4)

    print()
    print("=" * 78)
    print("v110-step3 K=8 HARD — INTERIOR DYNAMICS")
    print("=" * 78)
    report_summary("SOLVED", solved_summary, K=K, n_layers=4)
    report_summary("FAILED", failed_summary, K=K, n_layers=4)


def main():
    if ARCH == "v98":
        run_v98()
    elif ARCH == "v110_step3":
        run_v110_step3()


if __name__ == "__main__":
    main()
