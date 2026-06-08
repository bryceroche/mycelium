"""v113: v112b architecture + per-digit input tokens replacing 200-bin domain_codebook.

Motivation (from v108b diagnostic):
  The 200-bin INPUT caps digit precision at pos3-pos4. v108b showed that
  digit-decomposed input (ONE token per variable via summed digit embeddings)
  breaks the attractor, but collapses into the shared backbone. v113 goes
  further: each variable becomes n_digits SEPARATE tokens, each carrying one
  digit embedding + right-aligned RoPE + valid_mask gating.

Architecture change from v112b:
  - DROP fg_v107_domain_codebook input pathway (200-bin one-hot per variable)
  - ADD fg_v113_digit_codebook (10, H) Fourier-init per-digit-value embedding
  - ADD fg_v113_digit_rope_cos/sin (n_digits, H) — frozen sinusoidal right-aligned
    RoPE applied additively to per-digit residual stream
  - T_new = N_MAX * n_digits + F_MAX  (e.g. 16*5+8 = 88 for default N=16, F=8)
  - v112b topology tensor expands to (T_new, latent_dim)
  - valid_mask blocks padding digit positions from being attended to (key masking)
  - Tree codebook readout: token at (var, pos) → tree_codebook[pos] → digit logit

New tensor shapes:
  fg_v113_digit_codebook  : (10, H)           — shared digit value embedding
  fg_v113_digit_rope_cos  : (n_digits, H//2)  — per-array-pos cos table (frozen)
  fg_v113_digit_rope_sin  : (n_digits, H//2)  — per-array-pos sin table (frozen)

Right-aligned RoPE mechanic:
  - MSD-first array layout: array_pos=0 is 10000s digit, array_pos=4 is 1s digit
  - rope_pos = n_digits - 1 - array_pos  (so 1s digit at array_pos=4 gets rope_pos=0)
  - Rotational pairing: x[2i]*cos - x[2i+1]*sin, x[2i]*sin + x[2i+1]*cos
  - Tables indexed by array_pos directly (cos/sin computed with reversed rope_pos)

valid_mask (per variable, per array_pos):
  - Observed: valid if array_pos >= n_digits - len(str(int(value)))
  - Unobserved: all positions valid (predict all 5 digits)
  - Padding digit tokens: blocked as KEYS in attention via -1e4 bias

v112b composability:
  - fg_v115_node_topology resized to (T_new, latent_dim)
  - Per-position residual gate tanh(topology @ W_res_gate) applies per-token
  - Zero-init W_res_gate → identity at step 0

Cold-start only: v112b ckpt shapes are incompatible (different T, no domain_codebook).
"""
from __future__ import annotations

import math
import os
from typing import Any

import numpy as np
from tinygrad import Tensor, dtypes
from tinygrad.engine.jit import TinyJit

from mycelium.factor_graph_v108 import (
    _fourier_orthogonal_init,
    bins_to_digits_msd,
)
from mycelium.factor_graph_v104 import load_ib_centroids
from mycelium.factor_graph_v109pi import fg_layer_forward_v109pi
from mycelium.factor_graph_v109 import _apply_waist_v109
from mycelium.factor_graph_v110_acc import (
    _acc_notebook_read, _acc_notebook_write,
)
from mycelium.factor_graph_v110_photon import _photon_gate
from mycelium.factor_graph_v110_step3 import (
    V110_STEP3_N_MAX, V110_STEP3_F_MAX, V110_STEP3_N_DIGITS,
    V110_STEP3_K_MAX, V110_STEP3_WAIST_DIM,
    V110_STEP3_ALTERNATION, V110_STEP3_PHASE_SCALE,
    V110_STEP3_GATE_PROFILE, V110_STEP3_PHOTON_ALPHA,
    V110_STEP3_N_HEADS,
    V110_STEP3_CODEBOOK_N, V110_STEP3_IB_CENTROIDS,
    V110_STEP3_CALIB_WEIGHT, V110_STEP3_FACTOR_AUX_WEIGHT,
    V110_STEP3_VAR_LOSS_WEIGHT, V110_STEP3_BALANCE_WEIGHT,
    V110_STEP3_UNCERTAINTY_MIN, V110_STEP3_HARD_BREATH_LEVEL,
)
from mycelium.factor_graph_v110_step import (
    V110_STEP_N_HEADS,
)
from mycelium.factor_graph_v112b import (
    V112B_TOPOLOGY_DIM, V112B_BIAS_SCALE_INIT,
)


# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

V113_N_MAX    = int(os.environ.get("V113_N_MAX",    str(V110_STEP3_N_MAX)))
V113_F_MAX    = int(os.environ.get("V113_F_MAX",    str(V110_STEP3_F_MAX)))
V113_K_MAX    = int(os.environ.get("V113_K_MAX",    str(V110_STEP3_K_MAX)))
V113_N_DIGITS = int(os.environ.get("V113_N_DIGITS", str(V110_STEP3_N_DIGITS)))
V113_WAIST_DIM = int(os.environ.get("V113_WAIST_DIM", str(V110_STEP3_WAIST_DIM)))
V113_CODEBOOK_N = int(os.environ.get("V113_CODEBOOK_N", str(V110_STEP3_CODEBOOK_N)))
V113_IB_CENTROIDS = os.environ.get("V113_IB_CENTROIDS", V110_STEP3_IB_CENTROIDS)
# T_new = N_MAX * n_digits + F_MAX
V113_T_MAX    = V113_N_MAX * V113_N_DIGITS + V113_F_MAX
V113_TOPOLOGY_DIM = int(os.environ.get("V113_TOPOLOGY_DIM", str(V112B_TOPOLOGY_DIM)))


# ---------------------------------------------------------------------------
# Right-aligned RoPE tables (frozen sinusoidal)
# ---------------------------------------------------------------------------

def _build_rope_tables(n_digits: int, hidden: int) -> tuple[np.ndarray, np.ndarray]:
    """Build (n_digits, H//2) cos and sin tables for right-aligned RoPE.

    rope_pos = n_digits - 1 - array_pos, so the ONES digit (array_pos = n_digits-1)
    gets rope_pos=0 (highest frequency). The tables are indexed by array_pos directly.

    Using the standard sinusoidal RoPE frequency schedule:
      freq_i = 1 / (10000 ^ (2i / H))  for i in [0, H//2)
    """
    H2 = hidden // 2
    freqs = 1.0 / (10000.0 ** (np.arange(H2, dtype=np.float32) * 2.0 / hidden))
    cos_table = np.zeros((n_digits, H2), dtype=np.float32)
    sin_table = np.zeros((n_digits, H2), dtype=np.float32)
    for arr_pos in range(n_digits):
        rope_pos = n_digits - 1 - arr_pos
        angles = rope_pos * freqs
        cos_table[arr_pos] = np.cos(angles)
        sin_table[arr_pos] = np.sin(angles)
    return cos_table, sin_table


def _apply_rope_additive(x: Tensor, cos_t: Tensor, sin_t: Tensor) -> Tensor:
    """Apply right-aligned RoPE additively to residual stream.

    x:     (B, T_new, H)
    cos_t: (T_new, H//2)  — broadcast over B
    sin_t: (T_new, H//2)  — broadcast over B

    Rotational pairing on the last dimension: treats (x[..., 2i], x[..., 2i+1])
    as a 2D vector and rotates it by angle at the corresponding frequency.
    """
    H = int(x.shape[-1])
    H2 = H // 2
    # Split into even/odd pairs
    x_even = x[..., 0:H:2]      # (B, T, H//2)
    x_odd  = x[..., 1:H:2]      # (B, T, H//2)
    cos_b  = cos_t.reshape(1, -1, H2).cast(x.dtype)   # (1, T, H//2)
    sin_b  = sin_t.reshape(1, -1, H2).cast(x.dtype)   # (1, T, H//2)
    rot_even = x_even * cos_b - x_odd * sin_b
    rot_odd  = x_even * sin_b + x_odd * cos_b
    # Interleave back: stack along new dim then reshape
    B = int(x.shape[0])
    T = int(x.shape[1])
    rot = Tensor.stack(rot_even, rot_odd, dim=-1).reshape(B, T, H)
    return x + rot  # additive (v105.1.2-style)


# ---------------------------------------------------------------------------
# valid_mask: per-variable, per-digit-position
# ---------------------------------------------------------------------------

def build_valid_mask_np(
    gold_values_np: np.ndarray,   # (B, N_MAX) int
    obs_mask_np: np.ndarray,      # (B, N_MAX) int
    n_digits: int = 5,
) -> np.ndarray:
    """Compute (B, N_MAX, n_digits) bool valid_mask.

    Observed var v: valid[v, p] = (p >= n_digits - n_digits_used(gold_values[v]))
    Unobserved var v: valid[v, p] = True  (predict all positions)
    """
    B, N = gold_values_np.shape
    valid = np.ones((B, N, n_digits), dtype=bool)
    for b in range(B):
        for v in range(N):
            if obs_mask_np[b, v] == 1:
                val = int(max(0, gold_values_np[b, v]))
                n_used = max(1, len(str(val)))
                n_used = min(n_used, n_digits)
                # Leading positions (padding zeros) are invalid keys
                for p in range(n_digits - n_used):
                    valid[b, v, p] = False
    return valid


def build_digit_init_v113(
    gold_values_np: np.ndarray,   # (B, N_MAX) int
    obs_mask_np: np.ndarray,      # (B, N_MAX) int
    n_digits: int = 5,
) -> np.ndarray:
    """Build (B, N_MAX, n_digits, 10) one-hot/uniform digit init.

    MSD-first: position 0 = 10000s digit, position 4 = 1s digit.
    Observed: one-hot at the gold digit.
    Unobserved: uniform 1/10.
    """
    B, N = gold_values_np.shape
    di = np.full((B, N, n_digits, 10), 0.1, dtype=np.float32)
    for b in range(B):
        for v in range(N):
            if obs_mask_np[b, v] == 1:
                val = int(max(0, gold_values_np[b, v]))
                # MSD-first decomposition
                digits_msd = []
                rem = val
                for lvl in range(n_digits - 1, -1, -1):
                    place = 10 ** lvl
                    d = rem // place
                    rem = rem % place
                    digits_msd.append(d)
                digits_msd = list(reversed(digits_msd))  # now index 0 = MSD
                for p in range(n_digits):
                    d = int(min(9, max(0, digits_msd[p])))
                    di[b, v, p] = 0.0
                    di[b, v, p, d] = 1.0
    return di


# ---------------------------------------------------------------------------
# Input residual builder: per-digit tokens
# ---------------------------------------------------------------------------

def _build_residual_v113(
    digit_init: Tensor,       # (B, N_MAX, n_digits, 10) — one-hot or uniform
    node_kinds: Tensor,       # (B, T_new) int
    valid_mask: Tensor,       # (B, N_MAX, n_digits) bool/float
    digit_codebook: Tensor,   # (10, H)
    digit_rope_cos: Tensor,   # (n_digits, H//2) frozen
    digit_rope_sin: Tensor,   # (n_digits, H//2) frozen
    var_pos_embed: Tensor,    # (N_MAX, H)
    factor_pos_embed: Tensor, # (F_MAX, H)
    node_kind_embed: Tensor,  # (3, H)
    n_max: int,
    f_max: int,
    n_digits: int,
) -> tuple[Tensor, Tensor]:
    """Build (B, T_new, H) input residual and (B, T_new) key_valid mask.

    T_new = N_MAX * n_digits + F_MAX

    Layout: [var0_d0, var0_d1, ..., var0_d(n_digits-1),
              var1_d0, ...,
              varN_d0, ..., varN_d(n_digits-1),
              fac0, fac1, ..., facF]

    Returns:
      x_in : (B, T_new, H) input residual
      key_valid : (B, T_new) float — 1 if token is valid key, 0 if padding digit
    """
    B   = int(digit_init.shape[0])
    H   = int(digit_codebook.shape[-1])
    H2  = H // 2
    T_new = n_max * n_digits + f_max

    # --- Digit embeddings for variables ---
    # digit_init: (B, N_MAX, n_digits, 10) @ digit_codebook.T (10, H) → (B, N_MAX, n_digits, H)
    di_flat = digit_init.cast(dtypes.float).reshape(B * n_max * n_digits, 10)
    dc      = digit_codebook.cast(dtypes.float)                              # (10, H)
    emb_flat = di_flat @ dc                                                  # (B*N*P, H)
    emb_var  = emb_flat.reshape(B, n_max, n_digits, H)                      # (B, N, P, H)

    # Add var_pos_embed (broadcast over digit positions)
    vpe = var_pos_embed.cast(dtypes.float).reshape(1, n_max, 1, H)          # (1, N, 1, H)
    emb_var = emb_var + vpe                                                  # (B, N, P, H)

    # Reshape to (B, N*P, H) — interleaved as [var0_d0, var0_d1, ..., var0_d(n_digits-1), var1_d0...]
    emb_var_seq = emb_var.reshape(B, n_max * n_digits, H)

    # Apply right-aligned RoPE additively across the T_new sequence (var tokens only)
    # Build per-token rope tables by tiling across n_max variables
    # cos/sin are (n_digits, H//2); we need (N_MAX * n_digits, H//2) by repeating
    cos_tiled = digit_rope_cos.cast(dtypes.float).reshape(1, n_digits, H2).expand(
        n_max, n_digits, H2
    ).reshape(n_max * n_digits, H2)  # (N*P, H//2)
    sin_tiled = digit_rope_sin.cast(dtypes.float).reshape(1, n_digits, H2).expand(
        n_max, n_digits, H2
    ).reshape(n_max * n_digits, H2)  # (N*P, H//2)

    x_even = emb_var_seq[..., 0:H:2]    # (B, N*P, H//2)
    x_odd  = emb_var_seq[..., 1:H:2]    # (B, N*P, H//2)
    cos_b  = cos_tiled.reshape(1, n_max * n_digits, H2)
    sin_b  = sin_tiled.reshape(1, n_max * n_digits, H2)
    rot_even = x_even * cos_b - x_odd * sin_b
    rot_odd  = x_even * sin_b + x_odd * cos_b
    emb_var_rope = Tensor.stack(rot_even, rot_odd, dim=-1).reshape(B, n_max * n_digits, H)
    emb_var_seq  = emb_var_seq + emb_var_rope  # additive RoPE

    # --- Factor tokens ---
    fpe      = factor_pos_embed.cast(dtypes.float).reshape(1, f_max, H)     # (1, F, H)
    fac_zeros = Tensor.zeros(B, f_max, H, dtype=dtypes.float).contiguous()
    emb_fac   = fac_zeros + fpe                                              # (B, F, H)

    # --- Concatenate var_digits + factors ---
    x_in = emb_var_seq.cat(emb_fac, dim=1)                                  # (B, T_new, H)

    # --- Node kind embed ---
    # node_kinds is (B, T_new), values 0=var, 1=unobs_var, 2=factor
    nk_clipped = node_kinds.cast(dtypes.int).clip(0, 2)
    nk_oh  = nk_clipped.one_hot(3).cast(dtypes.float)                       # (B, T_new, 3)
    nk_v   = nk_oh @ node_kind_embed.cast(dtypes.float)                     # (B, T_new, H)
    x_in   = x_in + nk_v

    # --- Key valid mask ---
    # valid_mask: (B, N_MAX, n_digits) bool/float
    # Flatten to (B, N_MAX * n_digits) then append 1s for factor tokens
    vm_var = valid_mask.cast(dtypes.float).reshape(B, n_max * n_digits)     # (B, N*P)
    vm_fac = Tensor.ones(B, f_max, dtype=dtypes.float).contiguous()         # (B, F)
    key_valid = vm_var.cat(vm_fac, dim=1)                                    # (B, T_new)

    return x_in.cast(dtypes.half), key_valid


# ---------------------------------------------------------------------------
# Attention mask expansion: T_old → T_new
# ---------------------------------------------------------------------------

def _expand_mask_to_digit_tokens(
    mask_old: np.ndarray,   # (B, K, T_old, T_old) or (B, N_HEADS, T_old, T_old)
    n_max: int,
    f_max: int,
    n_digits: int,
) -> np.ndarray:
    """Expand mask from T_old = N_MAX + F_MAX to T_new = N_MAX * n_digits + F_MAX.

    Each (var_i, var_j) pair in the old mask becomes an (n_digits, n_digits) block.
    Factor tokens keep their 1-token-each indexing.

    mask_old shape: (B, K_or_H, T_old, T_old)
    Returns: (B, K_or_H, T_new, T_new)
    """
    T_old = n_max + f_max
    T_new = n_max * n_digits + f_max
    B, depth, _, _ = mask_old.shape

    expanded = np.full((B, depth, T_new, T_new), -1e4, dtype=np.float32)

    # Mapping from T_new token idx to (kind, old_idx)
    # var digit tokens: token_idx = v * n_digits + p  → var v
    # factor tokens:    token_idx = n_max*n_digits + f → old_idx = n_max + f

    def new_to_old(i_new):
        if i_new < n_max * n_digits:
            return i_new // n_digits  # var index
        else:
            return n_max + (i_new - n_max * n_digits)  # factor index in T_old

    for b in range(B):
        for d in range(depth):
            for i_new in range(T_new):
                i_old = new_to_old(i_new)
                for j_new in range(T_new):
                    j_old = new_to_old(j_new)
                    expanded[b, d, i_new, j_new] = mask_old[b, d, i_old, j_old]

    return expanded


def _expand_mask_fast(
    mask_old: np.ndarray,   # (B, K_or_H, T_old, T_old)
    n_max: int,
    f_max: int,
    n_digits: int,
) -> np.ndarray:
    """Vectorized version of _expand_mask_to_digit_tokens using gather indexing."""
    T_old = n_max + f_max
    T_new = n_max * n_digits + f_max
    B, depth = mask_old.shape[:2]

    # Build mapping: new_idx → old_idx
    idx = np.zeros(T_new, dtype=np.int32)
    for v in range(n_max):
        for p in range(n_digits):
            idx[v * n_digits + p] = v
    for f in range(f_max):
        idx[n_max * n_digits + f] = n_max + f

    # Gather rows then columns: (B, D, T_new, T_old) → (B, D, T_new, T_new)
    expanded = mask_old[:, :, idx, :][:, :, :, idx]
    return expanded.astype(np.float32)


# ---------------------------------------------------------------------------
# Parameter attachment
# ---------------------------------------------------------------------------

def attach_fg_params_v113(
    model: Any,
    hidden: int,
    n_max: int = V113_N_MAX,
    f_max: int = V113_F_MAX,
    k_max: int = V113_K_MAX,
    n_digits: int = V113_N_DIGITS,
    n_code: int = V113_CODEBOOK_N,
    ib_centroids_path: str = V113_IB_CENTROIDS,
    waist_dim: int = V113_WAIST_DIM,
    topology_dim: int = V113_TOPOLOGY_DIM,
) -> None:
    """Attach all v113 parameters from scratch (cold-start only)."""
    if hasattr(model, "fg_v113_digit_codebook"):
        return

    rng = np.random.RandomState(113001)
    T_new = n_max * n_digits + f_max
    H2    = hidden // 2

    print(f"[v113] cold-start param init: N_MAX={n_max} n_digits={n_digits} "
          f"F_MAX={f_max} T_new={T_new} H={hidden}", flush=True)

    # --- Digit codebook (10, H) Fourier-orthogonal init ---
    dc_np = _fourier_orthogonal_init(10, hidden, seed=113)
    model.fg_v113_digit_codebook = Tensor(dc_np).contiguous().realize()

    # --- Frozen RoPE tables (n_digits, H//2) ---
    cos_np, sin_np = _build_rope_tables(n_digits, hidden)
    model.fg_v113_digit_rope_cos = Tensor(cos_np).contiguous().realize()
    model.fg_v113_digit_rope_sin = Tensor(sin_np).contiguous().realize()

    # --- Var/factor/node_kind pos embeddings ---
    vpe = (rng.randn(n_max, hidden) * 0.02).astype(np.float32)
    model.fg_v113_var_pos_embed = Tensor(vpe).contiguous().realize()
    fpe = (rng.randn(f_max, hidden) * 0.02).astype(np.float32)
    model.fg_v113_factor_pos_embed = Tensor(fpe).contiguous().realize()
    nke = (rng.randn(3, hidden) * 0.02).astype(np.float32)
    model.fg_v113_node_kind_embed = Tensor(nke).contiguous().realize()

    # --- Tree codebook (n_digits, 10, H) Fourier-orthogonal per level ---
    cb = np.zeros((n_digits, 10, hidden), dtype=np.float32)
    for level in range(n_digits):
        cb[level] = _fourier_orthogonal_init(10, hidden, seed=level * 17 + 31)
    model.fg_v113_tree_codebook = Tensor(cb).contiguous().realize()

    # --- Per-breath embedding (orthonormal × 0.5) ---
    rng_be = np.random.RandomState(113002)
    raw_be = rng_be.randn(max(k_max, hidden), hidden).astype(np.float32)
    q_be, _ = np.linalg.qr(raw_be)
    be = (q_be[:k_max] * 0.5).astype(np.float32)
    model.fg_v113_breath_embed = Tensor(be).contiguous().realize()

    # --- Per-breath delta_gate: init 1.0 ---
    model.fg_v113_delta_gate = Tensor.ones((k_max,), dtype=dtypes.float).contiguous().realize()

    # --- Calibration head ---
    cw = (rng.randn(hidden, 1) * 0.02).astype(np.float32)
    model.fg_v113_calib_head_w = Tensor(cw).contiguous().realize()
    model.fg_v113_calib_head_b = Tensor.zeros((1,), dtype=dtypes.float).contiguous().realize()

    # --- IB semantic codebook ---
    cb_ib = load_ib_centroids(ib_centroids_path, n_code=n_code, hidden=hidden)
    model.fg_v113_semantic_codebook = Tensor(cb_ib).contiguous().realize()
    model.fg_v113_delta_gate_quant  = Tensor.zeros((k_max,), dtype=dtypes.float).contiguous().realize()
    model.fg_v113_temperature       = Tensor(
        np.array([1.0], dtype=np.float32)
    ).contiguous().realize()

    # --- Waist (1024 → waist_dim → 1024) ---
    # LoRA-style zero-init so it starts as identity
    W_c = np.zeros((hidden, waist_dim), dtype=np.float32)
    b_c = np.zeros((waist_dim,), dtype=np.float32)
    W_e = np.zeros((waist_dim, hidden), dtype=np.float32)
    b_e = np.zeros((hidden,), dtype=np.float32)
    model.fg_v113_W_compress = Tensor(W_c).contiguous().realize()
    model.fg_v113_b_compress = Tensor(b_c).contiguous().realize()
    model.fg_v113_W_expand   = Tensor(W_e).contiguous().realize()
    model.fg_v113_b_expand   = Tensor(b_e).contiguous().realize()

    # --- Accumulate notebook (v110-acc style) ---
    nb_rng = np.random.RandomState(113003)
    acc_wq = (nb_rng.randn(hidden, hidden) * 0.02).astype(np.float32)
    acc_wk = (nb_rng.randn(hidden, hidden) * 0.02).astype(np.float32)
    acc_wv = (nb_rng.randn(hidden, hidden) * 0.02).astype(np.float32)
    acc_wo = np.zeros((hidden, hidden), dtype=np.float32)
    acc_bo = np.zeros((hidden,),        dtype=np.float32)
    acc_ww = (nb_rng.randn(hidden, hidden) * 0.02).astype(np.float32)
    acc_bw = np.zeros((hidden,), dtype=np.float32)
    model.fg_v113_acc_W_q     = Tensor(acc_wq).contiguous().realize()
    model.fg_v113_acc_W_k     = Tensor(acc_wk).contiguous().realize()
    model.fg_v113_acc_W_v     = Tensor(acc_wv).contiguous().realize()
    model.fg_v113_acc_W_o     = Tensor(acc_wo).contiguous().realize()
    model.fg_v113_acc_b_o     = Tensor(acc_bo).contiguous().realize()
    model.fg_v113_acc_W_write = Tensor(acc_ww).contiguous().realize()
    model.fg_v113_acc_b_write = Tensor(acc_bw).contiguous().realize()

    # --- v112b-style topology tensor (T_new, topology_dim) ---
    topo_np = (rng.randn(T_new, topology_dim) * 0.02).astype(np.float32)
    model.fg_v113_node_topology = Tensor(topo_np).contiguous().realize()
    wres_np = np.zeros((topology_dim, hidden), dtype=np.float32)
    model.fg_v113_W_res_gate   = Tensor(wres_np).contiguous().realize()
    model.fg_v113_attn_bias_scale = Tensor(
        np.array([0.0], dtype=np.float32)
    ).contiguous().realize()

    # --- Phase scale for π-rotation (v109pi style) ---
    model.fg_v113_phase_scale = Tensor(
        np.array([V110_STEP3_PHASE_SCALE], dtype=np.float32)
    ).contiguous().realize()

    n_new_params = (
        10 * hidden +           # digit_codebook
        2 * n_digits * H2 +     # rope tables (frozen but allocated)
        n_max * hidden +        # var_pos_embed
        f_max * hidden +        # factor_pos_embed
        3 * hidden +            # node_kind_embed
        n_digits * 10 * hidden + # tree_codebook
        k_max * hidden +        # breath_embed
        k_max +                 # delta_gate
        hidden + 1 +            # calib head
        n_code * hidden + k_max + 1 +  # ib + delta_gate_quant + temp
        hidden * waist_dim * 2 + waist_dim + hidden +  # waist
        7 * hidden * hidden + 2 * hidden +  # acc notebook (approx)
        T_new * topology_dim + topology_dim * hidden + 1  # topology
    )
    print(
        f"[v113] params attached: T_new={T_new} topology_dim={topology_dim} "
        f"waist_dim={waist_dim}  ≈{n_new_params/1e6:.1f}M v113-specific params",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------

def fg_breathing_forward_v113(
    model: Any,
    digit_init: Tensor,     # (B, N_MAX, n_digits, 10) — one-hot or uniform
    valid_mask: Tensor,     # (B, N_MAX, n_digits) float — 1=valid key, 0=padding
    node_kinds_new: Tensor, # (B, T_new) int — per-token kind
    staging_mask: Tensor,   # (B, K_MAX, T_new, T_new) float — already expanded
    head_op_mask: Tensor,   # (B, N_HEADS, T_new, T_new) float — already expanded
    noise: Tensor,          # (K_MAX, B, T_new, H) — SBP noise
    noise_scale: Tensor,    # scalar
    K: int,
    n_max: int = V113_N_MAX,
    f_max: int = V113_F_MAX,
    n_digits: int = V113_N_DIGITS,
    alternation: bool = V110_STEP3_ALTERNATION,
    phase_scale: float = V110_STEP3_PHASE_SCALE,
    gate_profile: str = V110_STEP3_GATE_PROFILE,
    photon_alpha: float = V110_STEP3_PHOTON_ALPHA,
) -> tuple[list, list, list, list, list]:
    """v113 forward: per-digit input tokens + v112b topology gate.

    Returns (tree_logits_history, var_logits_history, factor_logits_history,
             calib_history, step_mags_history).

    tree_logits_history: K × (B, N_MAX, n_digits, 10)
      — each digit TOKEN's hidden → tree_codebook[pos].T → digit logit
    """
    digit_codebook   = model.fg_v113_digit_codebook
    digit_rope_cos   = model.fg_v113_digit_rope_cos
    digit_rope_sin   = model.fg_v113_digit_rope_sin
    var_pos_embed    = model.fg_v113_var_pos_embed
    factor_pos_embed = model.fg_v113_factor_pos_embed
    node_kind_embed  = model.fg_v113_node_kind_embed
    tree_codebook    = model.fg_v113_tree_codebook
    breath_embed     = model.fg_v113_breath_embed
    delta_gate       = model.fg_v113_delta_gate
    calib_head_w     = model.fg_v113_calib_head_w
    calib_head_b     = model.fg_v113_calib_head_b
    semantic_codebook = model.fg_v113_semantic_codebook
    delta_gate_quant  = model.fg_v113_delta_gate_quant
    temperature       = model.fg_v113_temperature
    W_compress        = model.fg_v113_W_compress
    b_compress        = model.fg_v113_b_compress
    W_expand          = model.fg_v113_W_expand
    b_expand          = model.fg_v113_b_expand
    acc_W_q           = model.fg_v113_acc_W_q
    acc_W_k           = model.fg_v113_acc_W_k
    acc_W_v           = model.fg_v113_acc_W_v
    acc_W_o           = model.fg_v113_acc_W_o
    acc_b_o           = model.fg_v113_acc_b_o
    acc_W_write       = model.fg_v113_acc_W_write
    acc_b_write       = model.fg_v113_acc_b_write
    node_topology     = model.fg_v113_node_topology
    W_res_gate        = model.fg_v113_W_res_gate
    attn_bias_scale   = model.fg_v113_attn_bias_scale

    H     = int(digit_codebook.shape[-1])
    B     = int(digit_init.shape[0])
    T_new = n_max * n_digits + f_max

    # Build input residual + key_valid
    x, key_valid = _build_residual_v113(
        digit_init, node_kinds_new, valid_mask,
        digit_codebook, digit_rope_cos, digit_rope_sin,
        var_pos_embed, factor_pos_embed, node_kind_embed,
        n_max, f_max, n_digits,
    )
    # x: (B, T_new, H) half

    # Apply valid_mask to attention: padding digit tokens (key_valid=0) get -1e4 bias
    # key_valid: (B, T_new); as key bias: (B, 1, 1, T_new) added to attn scores
    key_bias = (1.0 - key_valid.cast(dtypes.float)) * (-1e4)   # (B, T_new)
    key_bias_4d = key_bias.reshape(B, 1, 1, T_new)              # (B, 1, 1, T_new)

    layers = list(model.block.layers)
    K_max  = int(breath_embed.shape[0])
    assert K <= K_max, f"K={K} > K_max={K_max}"

    breath_phases = [phase_scale * k * math.pi / float(K_max) for k in range(K_max)]
    breath_cos    = [math.cos(p) for p in breath_phases]
    breath_sin    = [math.sin(p) for p in breath_phases]

    photon_gates = [_photon_gate(k, K_max, gate_profile) for k in range(K_max)]
    binary_gates = [_photon_gate(k, K_max, "binary")     for k in range(K_max)]

    from mycelium.breathing import _layernorm

    # v113 topology precomputation
    topology_f       = node_topology.cast(dtypes.float)           # (T_new, latent)
    attn_bias_full   = (topology_f @ topology_f.T) * attn_bias_scale.reshape(1, 1).cast(dtypes.float)
    attn_bias_btht   = attn_bias_full.reshape(1, 1, T_new, T_new)
    gate_per_pos     = (topology_f @ W_res_gate.cast(dtypes.float)).tanh()   # (T_new, H)
    gate_multiplier  = (1.0 + gate_per_pos).reshape(1, T_new, H)             # (1, T_new, H)

    tree_logits_history   = []
    var_logits_history    = []
    factor_logits_history = []
    calib_history         = []
    step_mags_history     = []
    notebook_slots: list[Tensor] = []

    # For readout: flatten tree_codebook (n_digits, 10, H) → not flat but per-digit
    # We'll read out per-token: for token at digit-pos p, use tree_codebook[p]
    # Precompute tree_codebook as (n_digits * 10, H) for the flat matmul then reshape
    tree_cb_flat = tree_codebook.reshape(n_digits * 10, H)

    for k in range(K):
        # Accumulate notebook read
        x_acc_delta = _acc_notebook_read(
            x, notebook_slots,
            acc_W_q, acc_W_k, acc_W_v, acc_W_o, acc_b_o,
        )
        x = x + x_acc_delta

        # SBP noise
        x = x + (noise[k] * noise_scale).cast(x.dtype)

        # v113 topology residual gate
        x = x * gate_multiplier.cast(x.dtype)

        be_k  = breath_embed[k].reshape(1, 1, -1).cast(x.dtype)
        x_in  = x + be_k
        x_pre = x

        stk   = staging_mask[:, k, :, :]                                         # (B, T_new, T_new)
        stk_h = stk.reshape(B, 1, T_new, T_new).expand(B, V110_STEP_N_HEADS, T_new, T_new)
        combined = stk_h.cast(x.dtype) + head_op_mask.cast(x.dtype)
        combined = combined + attn_bias_btht.cast(combined.dtype)
        # Apply valid_mask key bias (blocks padding digit tokens as keys)
        combined = combined + key_bias_4d.cast(combined.dtype)

        cos_k = breath_cos[k]
        sin_k = breath_sin[k]
        h = x_in
        for layer in layers[:4]:
            h = fg_layer_forward_v109pi(layer, h, combined, cos_k, sin_k)

        # IB semantic codebook
        cb  = semantic_codebook.cast(h.dtype)
        tmp = temperature.cast(h.dtype)
        scores   = h @ cb.T / tmp.reshape(1, 1, 1)
        weights  = scores.clip(-1e4, 1e4).softmax(-1)
        recon    = weights @ cb
        quantize = recon - h
        gate_quant_k = delta_gate_quant[k].cast(h.dtype).reshape(1, 1, 1)
        h_quant = h + gate_quant_k * quantize

        # Photon waist
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
        step   = gate_k * delta
        x      = x_pre + step

        step_mag_k = step.cast(dtypes.float).square().mean()
        step_mags_history.append(step_mag_k)

        slot_k = _acc_notebook_write(x, acc_W_write, acc_b_write)
        notebook_slots.append(slot_k)

        x_ln   = _layernorm(x, model.ln_f_g, model.ln_f_b,
                            model.cfg.layer_norm_eps).cast(dtypes.float)

        # --- Per-digit token readout ---
        # Variable tokens: (B, N_MAX * n_digits, H)
        var_digit_x = x_ln[:, :n_max * n_digits, :]                             # (B, N*P, H)
        var_digit_x_3d = var_digit_x.reshape(B, n_max, n_digits, H)             # (B, N, P, H)

        # For each digit position p, use tree_codebook[p]: (10, H)
        # tree_cb_flat: (n_digits * 10, H)
        # Flatten to (B*N*P, H) @ (n_digits*10, H).T → (B*N*P, n_digits*10)
        vd_flat    = var_digit_x_3d.reshape(B * n_max * n_digits, H)
        tl_full    = vd_flat @ tree_cb_flat.T.cast(dtypes.float)                # (B*N*P, n_digits*10)
        tl_3d      = tl_full.reshape(B, n_max, n_digits, n_digits, 10)
        # Diagonal: take position p's logits from level p of tree_codebook
        # For token at (n, p), we want tl_3d[b, n, p, p, :] — the p-th tree level
        # Build this via a masked select or by computing directly
        # Simpler: compute per-level separately and stack
        tree_logits_per_level = []
        for p in range(n_digits):
            # For digit token at position p, compute x @ tree_codebook[p].T
            # var_digit_x_3d[:, :, p, :] : (B, N, H)
            tok_p   = var_digit_x_3d[:, :, p, :]                                # (B, N, H)
            cb_p    = tree_codebook[p].cast(dtypes.float)                        # (10, H)
            logit_p = tok_p @ cb_p.T                                             # (B, N, 10)
            tree_logits_per_level.append(logit_p)
        # Stack to (B, N, n_digits, 10)
        tree_logits_k = Tensor.stack(*tree_logits_per_level, dim=2)             # (B, N, P, 10)
        tree_logits_history.append(tree_logits_k)

        # Var logits (for compat; pool over digit tokens → fake domain logit)
        # We don't have a domain codebook, so produce zeros placeholder
        var_logits_k = Tensor.zeros(B, n_max, 1, dtype=dtypes.float).contiguous()
        var_logits_history.append(var_logits_k)

        # Factor tokens readout (use tree_codebook[0] as proxy — factor CE uses bins)
        fac_x = x_ln[:, n_max * n_digits:n_max * n_digits + f_max, :]
        fac_logits_k = fac_x @ tree_cb_flat.T.cast(dtypes.float)               # (B, F, n_digits*10)
        factor_logits_history.append(fac_logits_k)

        # Calibration head
        pool        = x_ln.mean(axis=1)
        calib_logit = pool @ calib_head_w.cast(dtypes.float) + calib_head_b.cast(dtypes.float)
        calib_history.append(calib_logit.reshape(-1).sigmoid())

    return (tree_logits_history, var_logits_history, factor_logits_history,
            calib_history, step_mags_history)


# ---------------------------------------------------------------------------
# Parameter collection + state dict
# ---------------------------------------------------------------------------

def fg_v113_parameters(model: Any) -> list[Tensor]:
    """Collect all trainable parameters for v113 (backbone + v113-specific)."""
    params: list[Tensor] = []
    # Shared Pythia backbone
    sw = model.block.shared
    params += [sw.wv, sw.bv, sw.wo, sw.bo, sw.w_out, sw.b_out,
               sw.in_ln_g, sw.in_ln_b, sw.post_ln_g, sw.post_ln_b]
    for layer in model.block.layers:
        params += [layer.wq, layer.bq, layer.wk, layer.bk, layer.w_in, layer.b_in]
    params += [model.ln_f_g, model.ln_f_b]
    # v113-specific (exclude frozen rope tables)
    params += [
        model.fg_v113_digit_codebook,
        model.fg_v113_var_pos_embed,
        model.fg_v113_factor_pos_embed,
        model.fg_v113_node_kind_embed,
        model.fg_v113_tree_codebook,
        model.fg_v113_breath_embed,
        model.fg_v113_delta_gate,
        model.fg_v113_calib_head_w,
        model.fg_v113_calib_head_b,
        model.fg_v113_semantic_codebook,
        model.fg_v113_delta_gate_quant,
        model.fg_v113_temperature,
        model.fg_v113_W_compress,
        model.fg_v113_b_compress,
        model.fg_v113_W_expand,
        model.fg_v113_b_expand,
        model.fg_v113_acc_W_q,
        model.fg_v113_acc_W_k,
        model.fg_v113_acc_W_v,
        model.fg_v113_acc_W_o,
        model.fg_v113_acc_b_o,
        model.fg_v113_acc_W_write,
        model.fg_v113_acc_b_write,
        model.fg_v113_node_topology,
        model.fg_v113_W_res_gate,
        model.fg_v113_attn_bias_scale,
    ]
    return params


def fg_v113_state_dict(model: Any) -> dict[str, Tensor]:
    """State dict for v113 — includes backbone + v113 params + frozen rope tables."""
    sd: dict[str, Tensor] = {
        "ln_f.g": model.ln_f_g,
        "ln_f.b": model.ln_f_b,
    }
    sw = model.block.shared
    for a in ("wv", "bv", "wo", "bo", "w_out", "b_out",
              "in_ln_g", "in_ln_b", "post_ln_g", "post_ln_b"):
        sd[f"shared.{a}"] = getattr(sw, a)
    for i, layer in enumerate(model.block.layers):
        for a in ("wq", "bq", "wk", "bk", "w_in", "b_in"):
            sd[f"phase{i}.{a}"] = getattr(layer, a)
    for key in (
        "fg_v113_digit_codebook", "fg_v113_digit_rope_cos", "fg_v113_digit_rope_sin",
        "fg_v113_var_pos_embed", "fg_v113_factor_pos_embed", "fg_v113_node_kind_embed",
        "fg_v113_tree_codebook", "fg_v113_breath_embed", "fg_v113_delta_gate",
        "fg_v113_calib_head_w", "fg_v113_calib_head_b",
        "fg_v113_semantic_codebook", "fg_v113_delta_gate_quant", "fg_v113_temperature",
        "fg_v113_W_compress", "fg_v113_b_compress",
        "fg_v113_W_expand", "fg_v113_b_expand",
        "fg_v113_acc_W_q", "fg_v113_acc_W_k", "fg_v113_acc_W_v",
        "fg_v113_acc_W_o", "fg_v113_acc_b_o",
        "fg_v113_acc_W_write", "fg_v113_acc_b_write",
        "fg_v113_node_topology", "fg_v113_W_res_gate", "fg_v113_attn_bias_scale",
        "fg_v113_phase_scale",
    ):
        sd[key] = getattr(model, key)
    return sd


# ---------------------------------------------------------------------------
# JIT training step
# ---------------------------------------------------------------------------

_JIT_V113_CACHE: dict = {}


def _compile_jit_fg_step_v113(
    model: Any,
    opt: Any,
    K: int,
    B: int,
    factor_aux_weight: float = V110_STEP3_FACTOR_AUX_WEIGHT,
    calib_weight: float = V110_STEP3_CALIB_WEIGHT,
    var_loss_weight: float = V110_STEP3_VAR_LOSS_WEIGHT,
    balance_weight: float = V110_STEP3_BALANCE_WEIGHT,
    uncertainty_min: float = V110_STEP3_UNCERTAINTY_MIN,
    hard_breath_level: bool = V110_STEP3_HARD_BREATH_LEVEL,
    alternation: bool = V110_STEP3_ALTERNATION,
    phase_scale: float = V110_STEP3_PHASE_SCALE,
    n_max: int = V113_N_MAX,
    f_max: int = V113_F_MAX,
    n_digits: int = V113_N_DIGITS,
    gate_profile: str = V110_STEP3_GATE_PROFILE,
    photon_alpha: float = V110_STEP3_PHOTON_ALPHA,
    grad_clip: float = 1.0,
):
    key = ("v113", id(model), id(opt), int(K), int(B),
           float(factor_aux_weight), float(calib_weight), float(var_loss_weight),
           float(balance_weight), float(uncertainty_min),
           bool(hard_breath_level), bool(alternation), float(phase_scale),
           int(n_max), int(f_max), int(n_digits),
           str(gate_profile), float(photon_alpha), float(grad_clip))
    if key in _JIT_V113_CACHE:
        return _JIT_V113_CACHE[key]

    fw, aw, vw = float(factor_aux_weight), float(calib_weight), float(var_loss_weight)
    bw, um, gc = float(balance_weight), float(uncertainty_min), float(grad_clip)
    params = opt.params
    T_new  = n_max * n_digits + f_max

    print(
        f"[JIT] compile v113 step: K={K} B={B} T_new={T_new} "
        f"n_digits={n_digits} profile={gate_profile} alpha={photon_alpha}...",
        flush=True,
    )

    @TinyJit
    def _step(
        digit_init: Tensor,      # (B, N_MAX, n_digits, 10)
        valid_mask: Tensor,      # (B, N_MAX, n_digits)
        node_kinds_new: Tensor,  # (B, T_new)
        staging_mask: Tensor,    # (B, K_MAX, T_new, T_new)
        head_op_mask: Tensor,    # (B, N_HEADS, T_new, T_new)
        gold_digits: Tensor,     # (B, N_MAX, n_digits)
        gold_bins: Tensor,       # (B, N_MAX)
        observed_mask: Tensor,   # (B, N_MAX)
        factor_gold_digits: Tensor,  # (B, F_MAX, n_digits)
        factor_valid: Tensor,    # (B, F_MAX)
        noise: Tensor,           # (K_MAX, B, T_new, H)
        noise_scale: Tensor,     # scalar
    ):
        opt.zero_grad()

        tree_lh, _, fac_lh, calib_h, step_mh = fg_breathing_forward_v113(
            model, digit_init, valid_mask, node_kinds_new,
            staging_mask, head_op_mask, noise, noise_scale,
            K=K, n_max=n_max, f_max=f_max, n_digits=n_digits,
            alternation=alternation, phase_scale=phase_scale,
            gate_profile=gate_profile, photon_alpha=photon_alpha,
        )

        unobs_float = (1 - observed_mask.cast(dtypes.float)).reshape(B * n_max)
        n_unobs_sum = unobs_float.sum() + 1e-8
        gd_flat     = gold_digits.cast(dtypes.int).reshape(B * n_max, n_digits)

        var_loss_sum   = Tensor.zeros((), dtype=dtypes.float).contiguous()
        var_weight_sum = 0.0
        per_breath_ce_t: list[Tensor] = []

        for k_idx, tree_logits_k in enumerate(tree_lh):
            weight_k = 1.0 + float(k_idx) / float(max(K - 1, 1))
            tl_flat  = tree_logits_k.reshape(B * n_max, n_digits, 10)

            if hard_breath_level:
                levels_to_use = [k_idx] if k_idx < n_digits else list(range(n_digits))
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

        # Factor aux loss — use digit CE on first digit level of factors
        # fac_lh[k]: (B, F_MAX, n_digits*10) — flatten from tree_cb_flat matmul
        n_valid_factors = factor_valid.cast(dtypes.float).sum() + 1e-8
        fg_flat    = factor_gold_digits.cast(dtypes.int).reshape(B * f_max, n_digits)
        valid_flat = factor_valid.cast(dtypes.float).reshape(B * f_max)

        factor_aux_sum   = Tensor.zeros((), dtype=dtypes.float).contiguous()
        factor_aux_w_sum = 0.0
        for k_aux, fac_logits_k in enumerate(fac_lh):
            w_k_aux = 1.0 + float(k_aux) / float(max(K - 1, 1))
            # fac_logits_k: (B, F_MAX, n_digits*10) from flat matmul
            # Reshape to per-digit logits: (B*F, n_digits, 10)
            ftl = fac_logits_k.reshape(B * f_max, n_digits, 10)
            ce_fbreath_sum = Tensor.zeros((), dtype=dtypes.float).contiguous()
            for level in range(n_digits):
                lev_log  = ftl[:, level, :].log_softmax(axis=-1)
                lev_gold = fg_flat[:, level]
                gold_oh  = lev_gold.one_hot(10).cast(lev_log.dtype)
                nll      = -(lev_log * gold_oh).sum(axis=-1)
                masked   = nll * valid_flat
                ce_fbreath_sum = ce_fbreath_sum + (masked.sum() / n_valid_factors)
            ce_fbreath = ce_fbreath_sum / float(n_digits)
            factor_aux_sum   = factor_aux_sum + ce_fbreath * w_k_aux
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

    _JIT_V113_CACHE[key] = _step
    print(f"[JIT] v113 step ready (cache={len(_JIT_V113_CACHE)})", flush=True)
    return _step


def compile_jit_eval_v113(
    model: Any,
    K: int = V113_K_MAX,
    B: int = 8,
    n_max: int = V113_N_MAX,
    f_max: int = V113_F_MAX,
    n_digits: int = V113_N_DIGITS,
    alternation: bool = V110_STEP3_ALTERNATION,
    phase_scale: float = V110_STEP3_PHASE_SCALE,
    gate_profile: str = V110_STEP3_GATE_PROFILE,
    photon_alpha: float = V110_STEP3_PHOTON_ALPHA,
):
    """JIT'd eval forward for v113. Returns (pred_digits, cell_acc)."""
    T_new = n_max * n_digits + f_max
    H     = 1024
    eval_noise_zeros = Tensor.zeros(K, B, T_new, H, dtype=dtypes.half).contiguous().realize()
    eval_noise_scale = Tensor(
        np.array([0.0], dtype=np.float16)
    ).cast(dtypes.half).contiguous().realize()

    @TinyJit
    def _eval(
        digit_init: Tensor,
        valid_mask: Tensor,
        node_kinds_new: Tensor,
        staging_mask: Tensor,
        head_op_mask: Tensor,
        gold_digits: Tensor,
        observed_mask: Tensor,
    ):
        Tensor.training = False
        tree_lh, _, _, _, _ = fg_breathing_forward_v113(
            model, digit_init, valid_mask, node_kinds_new,
            staging_mask, head_op_mask,
            eval_noise_zeros, eval_noise_scale,
            K=K, n_max=n_max, f_max=f_max, n_digits=n_digits,
            alternation=alternation, phase_scale=phase_scale,
            gate_profile=gate_profile, photon_alpha=photon_alpha,
        )
        final_tree  = tree_lh[-1]
        pred_digits = final_tree.argmax(axis=-1)
        eq_per_pos  = (pred_digits == gold_digits.cast(dtypes.int)).cast(dtypes.float)
        eq          = eq_per_pos.prod(axis=-1)
        unobs       = (1 - observed_mask.cast(dtypes.float))
        cell_acc    = (eq * unobs).sum() / (unobs.sum() + 1e-8)
        return pred_digits.realize(), cell_acc.realize()
    return _eval
