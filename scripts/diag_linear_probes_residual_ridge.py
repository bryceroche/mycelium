"""Linear ridge probe — motion-carries-information test.

For each checkpoint (v98 sudoku K=20, v110-step3 K=8):
  - Extract post-L3 residual at k=4 (EARLY) and k=K-1 (LATE) over train + test sets.
  - Save residuals to .cache/probe_residuals_<tag>.npz.
  - Train 3 fresh linear probes (random-init W, b; linear only):
      Probe-EARLY: x_early @ W + b → logits
      Probe-LATE:  x_late  @ W + b → logits
      Probe-Δ:     (x_late - x_early) @ W + b → logits
  - Eval on held-out split, compute cell_acc.
  - For v110: also split by per-puzzle existing-readout outcome.
  - Save results to .cache/linear_probes_ridge_results.json.

Label spaces:
  v98:  9-way digit per unobserved cell (predictions are 0..8 → digit 1..9).
  v110: 5×10 per-position digit per unobserved var-node.

No JIT, fully eager (re-uses interior probe extraction pattern).
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---- env vars must be set BEFORE heavy imports ----
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
from mycelium.factor_graph_v109pi import _rotate_q_pi
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
    attach_sudoku_params, embed_sudoku,
    SUDOKU_K_MAX as SUDOKU_K_MAX_DEFAULT,
)
from mycelium.sudoku_data import SudokuLoader

# ---- ckpt loaders ----
from scripts.v108_factor_graph_train import cast_layers_fp32
from scripts.v110_acc_factor_graph_train import load_ckpt_v110_acc as load_ckpt_v110_step3
from scripts.sudoku_train import load_ckpt as load_sudoku_ckpt

# ============================================================
# Config
# ============================================================
SEED = 42
EPS = 1e-10

V110_CKPT = ".cache/fg_v110_step3_ckpts/v110_step3_cont8_step1000.safetensors"
V98_CKPT  = ".cache/sudoku_ckpts/v98_prod_final.safetensors"
V110_TRAIN = ".cache/factor_graph_train.jsonl"
V110_TEST  = ".cache/factor_graph_test.jsonl"
V98_TRAIN  = ".cache/sudoku_train.jsonl"
V98_TEST   = ".cache/sudoku_test.jsonl"

# How many puzzles to use for residual extraction (keep manageable for probe training)
# For v98: medium difficulty in train. For v110: hard difficulty.
V98_N_TRAIN   = int(os.environ.get("V98_N_TRAIN",   "500"))
V98_N_TEST    = int(os.environ.get("V98_N_TEST",    "300"))
V110_N_TRAIN  = int(os.environ.get("V110_N_TRAIN",  "500"))
V110_N_TEST   = int(os.environ.get("V110_N_TEST",   "300"))

# Linear probe training params
LR           = float(os.environ.get("PROBE_LR",    "1e-3"))
PROBE_STEPS  = int(os.environ.get("PROBE_STEPS",   "3000"))
PROBE_BATCH  = int(os.environ.get("PROBE_BATCH",   "512"))

SOLVED_THRESHOLD = 0.5


# ============================================================
# Eager residual extraction helpers (re-use interior probe pattern)
# ============================================================

def v110_extract_residuals(
    model, records, K, n_max, f_max, n_digits,
    alternation, phase_scale, gate_profile, photon_alpha,
    k_early: int, k_late: int,
    tag: str,
):
    """
    For each puzzle: run eager forward, collect:
      - post-L3 residual at breath k_early
      - post-L3 residual at breath k_late
      - existing-readout cell_acc (final breath tree_logits)
      - unobserved cell mask
      - gold digit labels (n_max, n_digits)

    Returns:
      x_early_list : list of (n_unobs, H) arrays
      x_late_list  : list of (n_unobs, H) arrays
      y_list       : list of (n_unobs, n_digits) int arrays  (0..9)
      cell_acc_list: list of float  (existing readout)
      puzzle_meta  : list of dicts
    """
    Tensor.training = False

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
    layers = list(model.block.layers)
    K_max  = int(breath_embed.shape[0])

    breath_phases = [phase_scale * k * math.pi / float(K_max) for k in range(K_max)]
    breath_cos = [math.cos(p) for p in breath_phases]
    breath_sin = [math.sin(p) for p in breath_phases]
    photon_gates = [_photon_gate(k, K_max, gate_profile) for k in range(K_max)]
    binary_gates = [_photon_gate(k, K_max, "binary")    for k in range(K_max)]
    tree_cb_flat = tree_codebook.reshape(n_digits * 10, H)

    x_early_list = []
    x_late_list  = []
    y_list       = []
    cell_acc_list = []
    puzzle_meta  = []

    print(f"\n[{tag}] Extracting residuals for {len(records)} puzzles ...")
    t0 = time.time()

    for pi, rec in enumerate(records):
        batch_np = _records_to_batch_v107(
            [rec], n_max=n_max, f_max=f_max, k_max=K,
            n_heads=V110_STEP3_N_HEADS,
        )
        domain_init    = Tensor(batch_np["domain_init"]).cast(dtypes.float).contiguous().realize()
        node_kinds     = Tensor(batch_np["node_kinds"]).cast(dtypes.int).contiguous().realize()
        staging_mask_t = Tensor(batch_np["staging_mask"]).cast(dtypes.float).contiguous().realize()
        head_op_mask_t = Tensor(batch_np["head_op_mask"]).cast(dtypes.float).contiguous().realize()
        obs_mask_np    = np.array(batch_np["observed_mask"])  # (1, n_max)
        gold_bins_np   = np.array(batch_np["gold_bins"])      # (1, n_max)

        # Run eager forward, capture post-L3 residuals at k_early and k_late
        x = embed_factor_graph_v100_aligned(
            domain_init, node_kinds,
            var_pos_embed, factor_pos_embed, node_kind_embed,
            domain_codebook, var_state_embed,
        )
        x = x.cast(dtypes.half) if x.dtype != dtypes.half else x

        notebook_slots: list[Tensor] = []
        x_early_np = None
        x_late_np  = None
        final_tree_logits = None

        for k in range(K):
            x_acc_delta = _acc_notebook_read(
                x, notebook_slots,
                acc_W_q, acc_W_k, acc_W_v, acc_W_o, acc_b_o,
            )
            x = x + x_acc_delta

            be_k  = breath_embed[k].reshape(1, 1, -1).cast(x.dtype)
            x_in  = x + be_k
            x_pre = x

            T = n_max + f_max
            stk      = staging_mask_t[:, k, :, :]
            stk_h    = stk.reshape(1, 1, T, T).expand(1, V110_STEP3_N_HEADS, T, T)
            combined = stk_h.cast(x.dtype) + head_op_mask_t.cast(x.dtype)

            cos_k = breath_cos[k]
            sin_k = breath_sin[k]

            h = x_in
            for li, layer in enumerate(layers[:4]):
                h = _v110_layer_forward_no_hook(layer, h, combined, cos_k, sin_k)

            # Capture post-L3 residual at target breaths
            if k == k_early:
                x_early_np = h.cast(dtypes.float).realize().numpy()[0].copy()  # (T, H)
            if k == k_late:
                x_late_np = h.cast(dtypes.float).realize().numpy()[0].copy()   # (T, H)

            # Continue forward (quantize + waist + delta_gate)
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

            # Compute readout logits at final breath
            if k == K - 1:
                x_ln = _layernorm(x, model.ln_f_g, model.ln_f_b,
                                  model.cfg.layer_norm_eps).cast(dtypes.float)
                var_x = x_ln[:, :n_max, :]
                tree_logits_flat = var_x @ tree_cb_flat.T.cast(dtypes.float)
                tree_logits_k    = tree_logits_flat.reshape(1, n_max, n_digits, 10)
                final_tree_logits = tree_logits_k.realize().numpy()[0]  # (n_max, n_digits, 10)

        assert x_early_np is not None and x_late_np is not None

        # Compute existing-readout cell_acc
        pred_digits = final_tree_logits.argmax(axis=-1)  # (n_max, n_digits)
        gold_digits = bins_to_digits_msd(gold_bins_np[0:1], n_digits=n_digits)[0]  # (n_max, n_digits)
        cell_eq     = (pred_digits == gold_digits).all(axis=-1)  # (n_max,)
        unobs_mask  = (1 - obs_mask_np[0]).astype(bool)  # (n_max,)
        n_unobs     = unobs_mask.sum()
        cell_acc    = float(cell_eq[unobs_mask].sum()) / max(n_unobs, 1)

        # Only keep unobserved positions for probe training
        # x_early_np, x_late_np shape: (T, H) — we want variable positions only: [:n_max]
        x_early_vars = x_early_np[:n_max, :]  # (n_max, H)
        x_late_vars  = x_late_np[:n_max, :]   # (n_max, H)

        x_early_unobs = x_early_vars[unobs_mask, :]  # (n_unobs, H)
        x_late_unobs  = x_late_vars[unobs_mask, :]   # (n_unobs, H)
        y_unobs       = gold_digits[unobs_mask, :]   # (n_unobs, n_digits)

        x_early_list.append(x_early_unobs)
        x_late_list.append(x_late_unobs)
        y_list.append(y_unobs)
        cell_acc_list.append(cell_acc)
        puzzle_meta.append({
            "cell_acc_existing": cell_acc,
            "n_unobs": int(n_unobs),
            "solved": cell_acc >= SOLVED_THRESHOLD,
        })

        if (pi + 1) % 50 == 0 or pi == 0:
            dt = time.time() - t0
            print(f"  [{pi+1:3d}/{len(records)}] dt={dt:.0f}s  cell_acc={cell_acc:.3f}", flush=True)

    return x_early_list, x_late_list, y_list, cell_acc_list, puzzle_meta


def _v110_layer_forward_no_hook(layer, x, attn_bias, q_rot_cos, q_rot_sin):
    """v109pi layer forward without attention extraction (faster)."""
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
    attn   = scores.clip(-1e4, 1e4).softmax(-1)

    ctx = (attn @ v).transpose(1, 2).reshape(B, S, H)
    attn_out = ctx @ layer.shared.wo + layer.shared.bo

    ff      = (mlp_in_dt @ layer.w_in + layer.b_in).gelu()
    ffn_out = ff @ layer.shared.w_out + layer.shared.b_out

    return x + attn_out + ffn_out


def v98_extract_residuals(
    model, records, K, k_early: int, k_late: int, tag: str,
):
    """
    For each v98 puzzle: run eager forward, collect:
      - post-L3 residual at k_early and k_late
      - existing-readout cell_acc (digit_codebook @ final x_ln)
      - unobserved cell mask
      - gold digit labels (81,) 0..8  (1..9 shifted to 0..8)

    Returns:
      x_early_list : list of (n_unobs, H) arrays
      x_late_list  : list of (n_unobs, H) arrays
      y_list       : list of (n_unobs,) int arrays  (0..8)
      cell_acc_list: list of float
      puzzle_meta  : list of dicts
    """
    Tensor.training = False

    state_embed    = model.sudoku_state_embed
    position_embed = model.sudoku_position_embed
    attn_bias_t    = model.sudoku_attn_bias
    breath_embed   = model.sudoku_breath_embed
    delta_gate     = model.sudoku_delta_gate
    digit_codebook = model.sudoku_digit_codebook

    layers = list(model.block.layers)

    # v98: no Q rotation (cos=1, sin=0)
    breath_cos = [1.0] * K
    breath_sin = [0.0] * K

    x_early_list  = []
    x_late_list   = []
    y_list        = []
    cell_acc_list = []
    puzzle_meta   = []

    print(f"\n[{tag}] Extracting residuals for {len(records)} puzzles ...")
    t0 = time.time()

    for pi, (inp_np, gold_np) in enumerate(records):
        # inp_np: (81,) int 0..9 (0=unknown), gold_np: (81,) int 1..9
        input_t = Tensor(inp_np[np.newaxis], dtype=dtypes.int).contiguous().realize()

        x = embed_sudoku(input_t, state_embed, position_embed)
        x = x.cast(dtypes.half) if x.dtype != dtypes.half else x

        x_early_np = None
        x_late_np  = None

        for k in range(K):
            be_k  = breath_embed[k].reshape(1, 1, -1).cast(x.dtype)
            x_in  = x + be_k
            x_pre = x

            cos_k = breath_cos[k]
            sin_k = breath_sin[k]

            h = x_in
            for li, layer in enumerate(layers[:4]):
                h = _v98_layer_forward_no_hook(layer, h, attn_bias_t, cos_k, sin_k)

            if k == k_early:
                x_early_np = h.cast(dtypes.float).realize().numpy()[0].copy()  # (81, H)
            if k == k_late:
                x_late_np = h.cast(dtypes.float).realize().numpy()[0].copy()   # (81, H)

            gate_k = delta_gate[k].cast(h.dtype).reshape(1, 1, 1)
            delta  = h - x_pre
            x      = x_pre + gate_k * delta

        assert x_early_np is not None and x_late_np is not None

        # Final readout
        x_ln = _layernorm(x, model.ln_f_g, model.ln_f_b,
                          model.cfg.layer_norm_eps).cast(dtypes.float)
        cell_logits = (x_ln @ digit_codebook.T.cast(dtypes.float)).realize().numpy()[0]  # (81, 9)
        pred_digits = cell_logits.argmax(axis=-1) + 1  # (81,) 1..9
        cell_acc    = float((pred_digits == gold_np).mean())

        # For v98, predict all 81 cells (no unobs mask — v98 predicts all cells including given)
        # But we should still focus on non-given cells for a fair comparison.
        # given = inp_np > 0; unobserved = inp_np == 0
        unobs_mask = (inp_np == 0)  # (81,) bool
        n_unobs = unobs_mask.sum()

        # Labels: gold_np 1..9 → 0..8
        y_unobs = gold_np[unobs_mask] - 1  # (n_unobs,) int 0..8

        x_early_unobs = x_early_np[unobs_mask, :]  # (n_unobs, H)
        x_late_unobs  = x_late_np[unobs_mask, :]   # (n_unobs, H)

        # Existing-readout cell_acc on unobserved only (matching probe eval)
        pred_unobs = pred_digits[unobs_mask]
        gold_unobs = gold_np[unobs_mask]
        cell_acc_unobs = float((pred_unobs == gold_unobs).mean()) if n_unobs > 0 else 0.0

        x_early_list.append(x_early_unobs)
        x_late_list.append(x_late_unobs)
        y_list.append(y_unobs)
        cell_acc_list.append(cell_acc_unobs)
        puzzle_meta.append({
            "cell_acc_existing": cell_acc_unobs,
            "n_unobs": int(n_unobs),
            "solved": cell_acc >= SOLVED_THRESHOLD,
        })

        if (pi + 1) % 50 == 0 or pi == 0:
            dt = time.time() - t0
            print(f"  [{pi+1:3d}/{len(records)}] dt={dt:.0f}s  cell_acc_unobs={cell_acc_unobs:.3f}", flush=True)

    return x_early_list, x_late_list, y_list, cell_acc_list, puzzle_meta


def _v98_layer_forward_no_hook(layer, x, attn_bias_t, q_rot_cos=1.0, q_rot_sin=0.0):
    """v98 sudoku layer forward without attention extraction."""
    cfg = layer.cfg
    B, S, H = x.shape
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

    ctx = (attn @ v).transpose(1, 2).reshape(B, S, H)
    attn_out = ctx @ layer.shared.wo + layer.shared.bo

    ff      = (mlp_in_dt @ layer.w_in + layer.b_in).gelu()
    ffn_out = ff @ layer.shared.w_out + layer.shared.b_out

    return x + attn_out + ffn_out


# ============================================================
# Linear probe: numpy-based SGD (no tinygrad JIT issues)
# ============================================================

class LinearProbe:
    """Pure numpy linear probe: logits = X @ W + b."""

    def __init__(self, in_dim: int, n_classes: int, rng: np.random.RandomState):
        # Xavier init
        scale = math.sqrt(2.0 / (in_dim + n_classes))
        self.W = rng.randn(in_dim, n_classes).astype(np.float32) * scale
        self.b = np.zeros(n_classes, dtype=np.float32)

    def forward(self, X: np.ndarray) -> np.ndarray:
        """X: (N, D) → logits: (N, C)"""
        return X @ self.W + self.b

    def softmax_ce_loss_and_grad(self, X, y):
        """Cross-entropy loss + gradient."""
        logits = self.forward(X)
        # numerically stable softmax
        logits -= logits.max(axis=1, keepdims=True)
        exp_l = np.exp(logits)
        probs = exp_l / exp_l.sum(axis=1, keepdims=True)
        N = len(y)
        loss = -np.log(probs[np.arange(N), y] + 1e-15).mean()
        # gradient
        dlogits = probs.copy()
        dlogits[np.arange(N), y] -= 1.0
        dlogits /= N
        dW = X.T @ dlogits
        db = dlogits.sum(axis=0)
        return loss, dW, db

    def accuracy(self, X, y) -> float:
        logits = self.forward(X)
        preds = logits.argmax(axis=1)
        return float((preds == y).mean())


def train_linear_probe(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    in_dim: int, n_classes: int,
    rng: np.random.RandomState,
    lr: float = 1e-3,
    n_steps: int = 3000,
    batch_size: int = 512,
    tag: str = "",
) -> tuple[LinearProbe, list[float], float]:
    """Train a fresh linear probe. Returns (probe, loss_curve, final_val_acc)."""
    probe = LinearProbe(in_dim, n_classes, rng)

    # Adam optimizer state
    m_W = np.zeros_like(probe.W)
    v_W = np.zeros_like(probe.W)
    m_b = np.zeros_like(probe.b)
    v_b = np.zeros_like(probe.b)
    beta1, beta2, eps_adam = 0.9, 0.999, 1e-8

    N = len(X_train)
    loss_curve = []
    best_val_acc = -1.0
    best_W = probe.W.copy()
    best_b = probe.b.copy()

    t0 = time.time()
    print(f"  [{tag}] Training {n_steps} steps, N_train={N}, N_val={len(X_val)}, "
          f"in_dim={in_dim}, n_classes={n_classes}", flush=True)

    for step in range(1, n_steps + 1):
        idx = rng.choice(N, size=min(batch_size, N), replace=False)
        Xb = X_train[idx]
        yb = y_train[idx]

        loss, dW, db = probe.softmax_ce_loss_and_grad(Xb, yb)
        loss_curve.append(float(loss))

        t = step
        m_W = beta1 * m_W + (1 - beta1) * dW
        v_W = beta2 * v_W + (1 - beta2) * dW ** 2
        m_b = beta1 * m_b + (1 - beta1) * db
        v_b = beta2 * v_b + (1 - beta2) * db ** 2

        m_W_hat = m_W / (1 - beta1 ** t)
        v_W_hat = v_W / (1 - beta2 ** t)
        m_b_hat = m_b / (1 - beta1 ** t)
        v_b_hat = v_b / (1 - beta2 ** t)

        probe.W -= lr * m_W_hat / (np.sqrt(v_W_hat) + eps_adam)
        probe.b -= lr * m_b_hat / (np.sqrt(v_b_hat) + eps_adam)

        # Evaluate every 500 steps
        if step % 500 == 0 or step == n_steps:
            val_acc = probe.accuracy(X_val, y_val)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_W = probe.W.copy()
                best_b = probe.b.copy()
            print(f"    step={step:5d}  loss={loss:.4f}  val_acc={val_acc:.4f}", flush=True)

    probe.W = best_W
    probe.b = best_b
    dt = time.time() - t0
    print(f"  [{tag}] Done in {dt:.0f}s. Best val_acc={best_val_acc:.4f}", flush=True)
    return probe, loss_curve, best_val_acc


# ============================================================
# Build flat (X, y) arrays from per-puzzle lists
# ============================================================

def build_flat_arrays_v110(x_early_list, x_late_list, y_list):
    """
    Stack all unobserved cells from all puzzles.
    y_list: list of (n_unobs, n_digits) int arrays

    Returns:
      X_early: (N_total, H)
      X_late:  (N_total, H)
      X_delta: (N_total, H)  = X_late - X_early
      y_flat:  (N_total * n_digits,)  int 0..9  — one per (cell, digit_pos)
      cell_offsets: cumulative per-puzzle cell counts, shape (n_puzzles+1,)
                    so puzzle i owns rows [cell_offsets[i]..cell_offsets[i+1]).
                    Each puzzle owns n_unobs[i] cells × n_digits labels.
    Note: we FLATTEN per-position labels so each (cell, pos) is an independent
    classification instance (probe sees one residual vector per cell, predicts
    a 10-way distribution PER POSITION independently). We train n_digits probes
    or we concatenate logits (simpler: train one probe that maps H → 10*n_digits
    and split per position at eval time). Here we train ONE probe per position.
    Actually simplest: for flat cell_acc (all 5 match), train 5 separate probes
    one per digit-position. Return structured.
    Returns dict with per-position arrays.
    """
    n_digits = y_list[0].shape[1] if len(y_list) > 0 else 5
    n_puzzles = len(x_early_list)

    X_early_all = np.concatenate(x_early_list, axis=0)  # (N_cells, H)
    X_late_all  = np.concatenate(x_late_list,  axis=0)
    X_delta_all = X_late_all - X_early_all
    y_all       = np.concatenate(y_list, axis=0)  # (N_cells, n_digits)

    # per-puzzle cell count
    cell_counts = np.array([len(x) for x in x_early_list])
    cell_offsets = np.concatenate([[0], np.cumsum(cell_counts)])

    return {
        "X_early": X_early_all,
        "X_late":  X_late_all,
        "X_delta": X_delta_all,
        "y":       y_all,           # (N_cells, n_digits)
        "cell_offsets": cell_offsets,
        "n_digits": n_digits,
    }


def build_flat_arrays_v98(x_early_list, x_late_list, y_list):
    """
    Stack all unobserved cells from all puzzles.
    y_list: list of (n_unobs,) int arrays 0..8

    Returns dict with flat arrays.
    """
    X_early_all = np.concatenate(x_early_list, axis=0)  # (N_cells, H)
    X_late_all  = np.concatenate(x_late_list,  axis=0)
    X_delta_all = X_late_all - X_early_all
    y_all       = np.concatenate(y_list, axis=0)         # (N_cells,)

    cell_counts = np.array([len(x) for x in x_early_list])
    cell_offsets = np.concatenate([[0], np.cumsum(cell_counts)])

    return {
        "X_early": X_early_all,
        "X_late":  X_late_all,
        "X_delta": X_delta_all,
        "y":       y_all,
        "cell_offsets": cell_offsets,
        "n_digits": 1,  # single label per cell
    }


# ============================================================
# Probe evaluation helpers
# ============================================================

def eval_v98_probes(probes_dict, arrays, puzzle_meta):
    """
    probes_dict: {'early': probe, 'late': probe, 'delta': probe}
    arrays: output of build_flat_arrays_v98
    puzzle_meta: list of dicts with 'solved', 'cell_acc_existing', 'n_unobs'

    Returns dict with per-probe, per-split cell_acc.
    """
    X_early = arrays["X_early"]
    X_late  = arrays["X_late"]
    X_delta = arrays["X_delta"]
    y       = arrays["y"]
    offsets = arrays["cell_offsets"]
    n_puzz  = len(puzzle_meta)

    results = {}
    for probe_name, (probe, X) in [
        ("early", (probes_dict["early"], X_early)),
        ("late",  (probes_dict["late"],  X_late)),
        ("delta", (probes_dict["delta"], X_delta)),
    ]:
        preds = probe.forward(X).argmax(axis=1)  # (N_cells,)

        # Per-puzzle cell accuracy
        per_puzzle_acc = []
        for i in range(n_puzz):
            s, e = offsets[i], offsets[i+1]
            if e <= s:
                per_puzzle_acc.append(0.0)
                continue
            per_puzzle_acc.append(float((preds[s:e] == y[s:e]).mean()))

        # Split by existing-readout outcome
        all_accs    = per_puzzle_acc
        solved_accs = [acc for acc, m in zip(per_puzzle_acc, puzzle_meta) if m["solved"]]
        failed_accs = [acc for acc, m in zip(per_puzzle_acc, puzzle_meta) if not m["solved"]]

        results[probe_name] = {
            "all":    float(np.mean(all_accs))    if all_accs    else float("nan"),
            "solved": float(np.mean(solved_accs)) if solved_accs else float("nan"),
            "failed": float(np.mean(failed_accs)) if failed_accs else float("nan"),
            "n_all":    len(all_accs),
            "n_solved": len(solved_accs),
            "n_failed": len(failed_accs),
        }
    return results


def eval_v110_probes_per_pos(probes_by_pos_dict, arrays, puzzle_meta):
    """
    probes_by_pos_dict: {probe_name: [probe_pos0, ..., probe_pos4]}
    arrays: build_flat_arrays_v110 output
    puzzle_meta: list of dicts

    For cell_acc: all n_digits positions must match.
    Returns per-probe, per-split cell_acc and per-pos digit_acc.
    """
    X_early  = arrays["X_early"]
    X_late   = arrays["X_late"]
    X_delta  = arrays["X_delta"]
    y        = arrays["y"]      # (N_cells, n_digits)
    offsets  = arrays["cell_offsets"]
    n_digits = arrays["n_digits"]
    n_puzz   = len(puzzle_meta)

    X_by_probe = {"early": X_early, "late": X_late, "delta": X_delta}

    results = {}
    for probe_name, probes_list in probes_by_pos_dict.items():
        X = X_by_probe[probe_name]

        # Predict each digit position
        preds_all = np.stack(
            [probe.forward(X).argmax(axis=1) for probe in probes_list],
            axis=1,
        )  # (N_cells, n_digits)

        cell_correct = (preds_all == y).all(axis=1)  # (N_cells,) bool
        digit_correct = (preds_all == y)              # (N_cells, n_digits)

        # Per-puzzle stats
        per_puzzle_cell_acc = []
        per_puzzle_pos_acc  = [[] for _ in range(n_digits)]
        for i in range(n_puzz):
            s, e = offsets[i], offsets[i+1]
            if e <= s:
                per_puzzle_cell_acc.append(0.0)
                for p in range(n_digits):
                    per_puzzle_pos_acc[p].append(0.0)
                continue
            per_puzzle_cell_acc.append(float(cell_correct[s:e].mean()))
            for p in range(n_digits):
                per_puzzle_pos_acc[p].append(float(digit_correct[s:e, p].mean()))

        # Split
        solved_mask = [m["solved"] for m in puzzle_meta]
        failed_mask = [not m["solved"] for m in puzzle_meta]

        def _mean_mask(accs, mask):
            vals = [a for a, m in zip(accs, mask) if m]
            return float(np.mean(vals)) if vals else float("nan")

        results[probe_name] = {
            "cell_acc": {
                "all":    float(np.mean(per_puzzle_cell_acc)),
                "solved": _mean_mask(per_puzzle_cell_acc, solved_mask),
                "failed": _mean_mask(per_puzzle_cell_acc, failed_mask),
            },
            "per_pos_digit_acc": [
                {
                    "all":    float(np.mean(per_puzzle_pos_acc[p])),
                    "solved": _mean_mask(per_puzzle_pos_acc[p], solved_mask),
                    "failed": _mean_mask(per_puzzle_pos_acc[p], failed_mask),
                }
                for p in range(n_digits)
            ],
            "n_all":    len(per_puzzle_cell_acc),
            "n_solved": sum(solved_mask),
            "n_failed": sum(failed_mask),
        }

    return results


# ============================================================
# Save / load npz residuals
# ============================================================

def save_residuals(path, x_early_list, x_late_list, y_list, puzzle_meta,
                   extra_keys=None):
    """Save residuals to npz, flattened per-puzzle. Metadata saved as JSON alongside."""
    X_early = np.concatenate(x_early_list, axis=0).astype(np.float32)
    X_late  = np.concatenate(x_late_list,  axis=0).astype(np.float32)
    # For y: pad to common shape if needed
    if isinstance(y_list[0], np.ndarray) and y_list[0].ndim == 2:
        y_arr = np.concatenate(y_list, axis=0).astype(np.int16)
    else:
        y_arr = np.concatenate(y_list, axis=0).astype(np.int16)
    cell_counts = np.array([len(x) for x in x_early_list], dtype=np.int32)
    np.savez_compressed(
        path,
        X_early=X_early, X_late=X_late, y=y_arr, cell_counts=cell_counts,
    )
    meta_path = path.replace(".npz", "_meta.json")
    # Coerce numpy scalar types to Python natives for JSON compat
    meta_safe = []
    for m in puzzle_meta:
        meta_safe.append({
            "cell_acc_existing": float(m["cell_acc_existing"]),
            "n_unobs":           int(m["n_unobs"]),
            "solved":            bool(m["solved"]),
        })
    with open(meta_path, "w") as f:
        json.dump(meta_safe, f)
    print(f"  Saved residuals to {path} (shape: {X_early.shape})", flush=True)


def load_residuals(path):
    """Load saved residuals back into per-puzzle lists."""
    data = np.load(path)
    X_early_flat = data["X_early"]   # (N, H)
    X_late_flat  = data["X_late"]
    y_flat       = data["y"]
    cell_counts  = data["cell_counts"]

    meta_path = path.replace(".npz", "_meta.json")
    with open(meta_path) as f:
        puzzle_meta = json.load(f)

    x_early_list, x_late_list, y_list = [], [], []
    offset = 0
    for cnt in cell_counts:
        x_early_list.append(X_early_flat[offset:offset+cnt])
        x_late_list.append(X_late_flat[offset:offset+cnt])
        y_list.append(y_flat[offset:offset+cnt])
        offset += cnt

    return x_early_list, x_late_list, y_list, puzzle_meta


# ============================================================
# Main per-ckpt pipeline
# ============================================================

def run_v98_pipeline(model, rng):
    """Full pipeline for v98 sudoku. Returns results dict."""
    K = 20
    k_early = 4
    k_late  = K - 1  # 19

    print("\n" + "="*70)
    print("v98 SUDOKU PIPELINE  K=20  k_early=4  k_late=19")
    print("="*70)

    # ---- Load data ----
    def load_sudoku_records(path, difficulty, n_max):
        import json as _json
        recs = []
        with open(path) as f:
            for line in f:
                r = _json.loads(line)
                if r.get("difficulty") == difficulty:
                    recs.append(r)
                if len(recs) >= n_max:
                    break
        return recs

    train_recs_raw = load_sudoku_records(V98_TRAIN, "medium", V98_N_TRAIN)
    test_recs_raw  = load_sudoku_records(V98_TEST,  "medium", V98_N_TEST)

    def recs_to_pairs(recs):
        pairs = []
        for r in recs:
            inp_np = np.array(r["input"],    dtype=np.int32)
            sol_np = np.array(r["solution"], dtype=np.int32)
            pairs.append((inp_np, sol_np))
        return pairs

    train_pairs = recs_to_pairs(train_recs_raw)
    test_pairs  = recs_to_pairs(test_recs_raw)
    print(f"  Train puzzles (medium): {len(train_pairs)}")
    print(f"  Test  puzzles (medium): {len(test_pairs)}")

    # ---- Extract residuals ----
    train_npz = ".cache/probe_residuals_v98_train.npz"
    test_npz  = ".cache/probe_residuals_v98_test.npz"

    if os.path.exists(train_npz):
        print(f"  Loading cached train residuals from {train_npz}")
        x_early_tr, x_late_tr, y_tr, meta_tr = load_residuals(train_npz)
    else:
        x_early_tr, x_late_tr, y_tr, _, meta_tr = v98_extract_residuals(
            model, train_pairs, K, k_early, k_late, tag="v98-train")
        save_residuals(train_npz, x_early_tr, x_late_tr, y_tr, meta_tr)

    if os.path.exists(test_npz):
        print(f"  Loading cached test residuals from {test_npz}")
        x_early_te, x_late_te, y_te, meta_te = load_residuals(test_npz)
    else:
        x_early_te, x_late_te, y_te, _, meta_te = v98_extract_residuals(
            model, test_pairs, K, k_early, k_late, tag="v98-test")
        save_residuals(test_npz, x_early_te, x_late_te, y_te, meta_te)

    # ---- Baseline existing-readout cell_acc on test ----
    baseline_cell_acc = float(np.mean([m["cell_acc_existing"] for m in meta_te]))
    print(f"\n  Baseline (existing readout) test cell_acc = {baseline_cell_acc:.4f}")

    # ---- Build flat arrays ----
    train_arrs = build_flat_arrays_v98(x_early_tr, x_late_tr, y_tr)
    test_arrs  = build_flat_arrays_v98(x_early_te, x_late_te, y_te)
    H = train_arrs["X_early"].shape[1]
    print(f"  H={H}  N_train_cells={len(train_arrs['X_early'])}  N_test_cells={len(test_arrs['X_early'])}")

    # ---- Train 3 linear probes ----
    probes = {}
    probe_losses = {}
    for probe_name, X_tr, X_te in [
        ("early", train_arrs["X_early"], test_arrs["X_early"]),
        ("late",  train_arrs["X_late"],  test_arrs["X_late"]),
        ("delta", train_arrs["X_delta"], test_arrs["X_delta"]),
    ]:
        probe, losses, val_acc = train_linear_probe(
            X_tr, train_arrs["y"],
            X_te, test_arrs["y"],
            in_dim=H, n_classes=9,
            rng=rng,
            lr=LR, n_steps=PROBE_STEPS, batch_size=PROBE_BATCH,
            tag=f"v98-{probe_name}",
        )
        probes[probe_name] = probe
        probe_losses[probe_name] = losses[-1]

    # ---- Evaluate probes on test ----
    probe_results = eval_v98_probes(probes, test_arrs, meta_te)

    # ---- Format results ----
    n_test    = len(meta_te)
    n_solved  = sum(1 for m in meta_te if m["solved"])
    n_failed  = n_test - n_solved

    result = {
        "ckpt": "v98_sudoku_medium_K20",
        "n_test": n_test, "n_solved": n_solved, "n_failed": n_failed,
        "baseline_cell_acc": baseline_cell_acc,
        "probes": probe_results,
        "final_train_losses": probe_losses,
    }
    return result


def run_v110_pipeline(model, rng):
    """Full pipeline for v110-step3 hard K=8. Returns results dict."""
    K       = V110_STEP3_K_MAX
    n_max   = V110_STEP3_N_MAX
    f_max   = V110_STEP3_F_MAX
    n_digits = V110_STEP3_N_DIGITS
    k_early = 4
    k_late  = K - 1  # 7

    print("\n" + "="*70)
    print(f"v110-step3 HARD PIPELINE  K={K}  k_early={k_early}  k_late={k_late}")
    print("="*70)

    # ---- Load data ----
    def load_fg_records(path, difficulty, n_max_recs):
        import json as _json
        recs = []
        with open(path) as f:
            for line in f:
                r = _json.loads(line)
                if r.get("difficulty") == difficulty:
                    recs.append(r)
                if len(recs) >= n_max_recs:
                    break
        return recs

    train_recs = load_fg_records(V110_TRAIN, "hard", V110_N_TRAIN)
    test_recs  = load_fg_records(V110_TEST,  "hard", V110_N_TEST)
    print(f"  Train puzzles (hard): {len(train_recs)}")
    print(f"  Test  puzzles (hard): {len(test_recs)}")

    # ---- Extract residuals ----
    train_npz = ".cache/probe_residuals_v110_train.npz"
    test_npz  = ".cache/probe_residuals_v110_test.npz"

    extract_kwargs = dict(
        K=K, n_max=n_max, f_max=f_max, n_digits=n_digits,
        alternation=V110_STEP3_ALTERNATION,
        phase_scale=V110_STEP3_PHASE_SCALE,
        gate_profile=V110_STEP3_GATE_PROFILE,
        photon_alpha=V110_STEP3_PHOTON_ALPHA,
        k_early=k_early, k_late=k_late,
    )

    if os.path.exists(train_npz):
        print(f"  Loading cached train residuals from {train_npz}")
        x_early_tr, x_late_tr, y_tr, meta_tr = load_residuals(train_npz)
    else:
        x_early_tr, x_late_tr, y_tr, _, meta_tr = v110_extract_residuals(
            model, train_recs, tag="v110-train", **extract_kwargs)
        save_residuals(train_npz, x_early_tr, x_late_tr, y_tr, meta_tr)

    if os.path.exists(test_npz):
        print(f"  Loading cached test residuals from {test_npz}")
        x_early_te, x_late_te, y_te, meta_te = load_residuals(test_npz)
    else:
        x_early_te, x_late_te, y_te, _, meta_te = v110_extract_residuals(
            model, test_recs, tag="v110-test", **extract_kwargs)
        save_residuals(test_npz, x_early_te, x_late_te, y_te, meta_te)

    # ---- Baseline ----
    baseline_cell_acc = float(np.mean([m["cell_acc_existing"] for m in meta_te]))
    baseline_solved   = [m for m in meta_te if m["solved"]]
    baseline_failed   = [m for m in meta_te if not m["solved"]]
    baseline_cell_acc_solved = float(np.mean([m["cell_acc_existing"] for m in baseline_solved])) if baseline_solved else float("nan")
    baseline_cell_acc_failed = float(np.mean([m["cell_acc_existing"] for m in baseline_failed])) if baseline_failed else float("nan")
    print(f"\n  Baseline (existing readout) test cell_acc:")
    print(f"    all={baseline_cell_acc:.4f}  solved={baseline_cell_acc_solved:.4f}  failed={baseline_cell_acc_failed:.4f}")

    # ---- Build flat arrays ----
    train_arrs = build_flat_arrays_v110(x_early_tr, x_late_tr, y_tr)
    test_arrs  = build_flat_arrays_v110(x_early_te, x_late_te, y_te)
    H = train_arrs["X_early"].shape[1]
    print(f"  H={H}  n_digits={n_digits}")
    print(f"  N_train_cells={len(train_arrs['X_early'])}  N_test_cells={len(test_arrs['X_early'])}")

    # ---- Train 5 linear probes per probe-type (one per digit position) ----
    # probes_by_pos_dict: {probe_name: [probe_p0, ..., probe_p4]}
    probes_by_pos = {}
    probe_losses  = {}
    for probe_name, X_tr, X_te in [
        ("early", train_arrs["X_early"], test_arrs["X_early"]),
        ("late",  train_arrs["X_late"],  test_arrs["X_late"]),
        ("delta", train_arrs["X_delta"], test_arrs["X_delta"]),
    ]:
        probes_this = []
        final_losses = []
        for pos in range(n_digits):
            y_pos_tr = train_arrs["y"][:, pos]  # (N_train_cells,) int 0..9
            y_pos_te = test_arrs["y"][:, pos]
            probe, losses, val_acc = train_linear_probe(
                X_tr, y_pos_tr,
                X_te, y_pos_te,
                in_dim=H, n_classes=10,
                rng=rng,
                lr=LR, n_steps=PROBE_STEPS, batch_size=PROBE_BATCH,
                tag=f"v110-{probe_name}-pos{pos}",
            )
            probes_this.append(probe)
            final_losses.append(losses[-1])
        probes_by_pos[probe_name] = probes_this
        probe_losses[probe_name] = final_losses

    # ---- Evaluate ----
    probe_results = eval_v110_probes_per_pos(probes_by_pos, test_arrs, meta_te)

    n_test   = len(meta_te)
    n_solved = sum(1 for m in meta_te if m["solved"])
    n_failed = n_test - n_solved

    result = {
        "ckpt": "v110_step3_hard_K8",
        "n_test": n_test, "n_solved": n_solved, "n_failed": n_failed,
        "baseline_cell_acc": {
            "all":    baseline_cell_acc,
            "solved": baseline_cell_acc_solved,
            "failed": baseline_cell_acc_failed,
        },
        "probes": probe_results,
        "final_train_losses": probe_losses,
    }
    return result


# ============================================================
# Report formatting
# ============================================================

def format_v98_report(result):
    lines = []
    lines.append(f"\nv98 sudoku medium K=20 (n_test={result['n_test']}, "
                 f"solved={result['n_solved']}, failed={result['n_failed']})")
    lines.append(f"  baseline existing-readout cell_acc: {result['baseline_cell_acc']:.4f}")
    lines.append("")

    p = result["probes"]
    lines.append(f"  Probe-EARLY (residual @ k=4):      cell_acc = {p['early']['all']:.4f}")
    lines.append(f"  Probe-LATE  (residual @ k=19):     cell_acc = {p['late']['all']:.4f}")
    lines.append(f"  Probe-Δ     (Δx = x@19 − x@4):     cell_acc = {p['delta']['all']:.4f}")
    lines.append("")
    lines.append(f"  Train losses (final): early={result['final_train_losses']['early']:.4f}, "
                 f"late={result['final_train_losses']['late']:.4f}, "
                 f"delta={result['final_train_losses']['delta']:.4f}")

    # Verdict
    e_acc = p["early"]["all"]
    l_acc = p["late"]["all"]
    d_acc = p["delta"]["all"]
    b_acc = result["baseline_cell_acc"]
    gap_late_vs_early = l_acc - e_acc
    gap_both_vs_baseline_e = e_acc - b_acc
    gap_both_vs_baseline_l = l_acc - b_acc

    lines.append("")
    lines.append("  VERDICT:")
    if gap_late_vs_early >= 0.05:
        lines.append(f"    OUTCOME 1: probe@late > probe@early by {gap_late_vs_early:.4f} (≥5%).")
        lines.append(f"    Ridge accumulates task info; late-breath drift carries signal.")
    elif gap_both_vs_baseline_e >= 0.05 and gap_both_vs_baseline_l >= 0.05:
        lines.append(f"    OUTCOME 2: probe@late ≈ probe@early, both > existing readout.")
        lines.append(f"    Gap to baseline: early+{gap_both_vs_baseline_e:.4f}, late+{gap_both_vs_baseline_l:.4f}.")
        lines.append(f"    Info was always in residual; codebook path is the bottleneck.")
    elif abs(gap_late_vs_early) < 0.03 and abs(gap_both_vs_baseline_l) < 0.03:
        lines.append(f"    OUTCOME 3: probe@late ≈ probe@early ≈ existing (all within ±3%).")
        lines.append(f"    Ridge is informationally empty.")
    else:
        lines.append(f"    MIXED: late-vs-early gap={gap_late_vs_early:.4f}, "
                     f"late-vs-baseline gap={gap_both_vs_baseline_l:.4f}.")
        lines.append(f"    (does not cleanly land in any single outcome)")

    return "\n".join(lines)


def format_v110_report(result):
    lines = []
    b = result["baseline_cell_acc"]
    lines.append(f"\nv110-step3 hard K=8 (n_test={result['n_test']}, "
                 f"solved={result['n_solved']}, failed={result['n_failed']})")
    lines.append(f"  baseline existing-readout cell_acc: {b['all']:.4f} "
                 f"(solved {b['solved']:.4f}, failed {b['failed']:.4f})")
    lines.append("")

    p = result["probes"]
    header = f"  {'':20s}  {'all':>8s}  {'solved-by-existing':>20s}  {'failed-by-existing':>20s}"
    lines.append(header)
    for probe_key, label in [
        ("early", "Probe-EARLY"),
        ("late",  "Probe-LATE"),
        ("delta", "Probe-Δ"),
    ]:
        ca = p[probe_key]["cell_acc"]
        lines.append(
            f"  {label:20s}  {ca['all']:>8.4f}  {ca['solved']:>20.4f}  {ca['failed']:>20.4f}"
        )
    # Baseline row for reference
    lines.append(
        f"  {'baseline':20s}  {b['all']:>8.4f}  {b['solved']:>20.4f}  {b['failed']:>20.4f}"
    )

    lines.append("")
    lines.append("  Per-position digit_acc (probe-late, all puzzles):")
    for pos_i, pda in enumerate(p["late"]["per_pos_digit_acc"]):
        lines.append(f"    pos{pos_i}: all={pda['all']:.4f}  solved={pda['solved']:.4f}  failed={pda['failed']:.4f}")

    lines.append(f"\n  Train losses (final): early={result['final_train_losses']['early']}, "
                 f"late={result['final_train_losses']['late']}, "
                 f"delta={result['final_train_losses']['delta']}")

    # Verdict
    e_acc = p["early"]["cell_acc"]["all"]
    l_acc = p["late"]["cell_acc"]["all"]
    b_acc = b["all"]
    gap_late_vs_early = l_acc - e_acc
    gap_both_vs_baseline_e = e_acc - b_acc
    gap_both_vs_baseline_l = l_acc - b_acc

    # Discarded-signal test (failed puzzles)
    late_on_failed  = p["late"]["cell_acc"]["failed"]
    base_on_failed  = b["failed"]
    discarded_delta = late_on_failed - base_on_failed

    lines.append("")
    lines.append("  Discarded-signal test (failed puzzles):")
    lines.append(f"    Probe-LATE cell_acc on failed: {late_on_failed:.4f}")
    lines.append(f"    Baseline   cell_acc on failed: {base_on_failed:.4f}")
    lines.append(f"    Δ = {discarded_delta:+.4f}")
    if discarded_delta > 0.05:
        lines.append("    → DISCARDED SIGNAL (late residual carries info that existing readout missed)")
    else:
        lines.append("    → THRASHING / no extractable signal on failed puzzles")

    lines.append("")
    lines.append("  VERDICT:")
    if gap_late_vs_early >= 0.05:
        lines.append(f"    OUTCOME 1: probe@late > probe@early by {gap_late_vs_early:.4f} (≥5%).")
        lines.append(f"    Ridge accumulates task info.")
    elif gap_both_vs_baseline_e >= 0.05 and gap_both_vs_baseline_l >= 0.05:
        lines.append(f"    OUTCOME 2: probe@late ≈ probe@early, both > existing readout.")
        lines.append(f"    Gap: early+{gap_both_vs_baseline_e:.4f}, late+{gap_both_vs_baseline_l:.4f}.")
        lines.append(f"    Codebook path is the bottleneck.")
    elif abs(gap_late_vs_early) < 0.03 and abs(gap_both_vs_baseline_l) < 0.03:
        lines.append(f"    OUTCOME 3: probe@late ≈ probe@early ≈ existing (within ±3%).")
        lines.append(f"    Ridge is empty.")
    else:
        lines.append(f"    MIXED: late-vs-early gap={gap_late_vs_early:.4f}, "
                     f"late-vs-baseline gap={gap_both_vs_baseline_l:.4f}.")
        lines.append(f"    (does not cleanly land in any single outcome)")

    return "\n".join(lines)


# ============================================================
# Entry point
# ============================================================

def main():
    print("="*70)
    print("Linear ridge probe: motion-carries-information test")
    print("="*70)
    print(f"  SEED={SEED}  LR={LR}  PROBE_STEPS={PROBE_STEPS}  BATCH={PROBE_BATCH}")
    print(f"  V98_N_TRAIN={V98_N_TRAIN}  V98_N_TEST={V98_N_TEST}")
    print(f"  V110_N_TRAIN={V110_N_TRAIN}  V110_N_TEST={V110_N_TEST}")

    np.random.seed(SEED)
    rng = np.random.RandomState(SEED)
    Tensor.manual_seed(SEED)
    Tensor.training = False

    # ---- Load shared Pythia backbone ----
    cfg = Config()
    sd  = _load_state()
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

    v110_result = run_v110_pipeline(model, rng)
    all_results["v110_step3_hard_K8"] = v110_result

    # ============================================================
    # v98 sudoku probe
    # ============================================================
    print("\n[2/2] Setting up v98 sudoku ...")
    SUDOKU_K = 20
    attach_sudoku_params(model, hidden=cfg.hidden, n_heads=cfg.n_heads, k_max=SUDOKU_K)
    load_sudoku_ckpt(model, V98_CKPT)
    print(f"  Loaded: {V98_CKPT}")

    v98_result = run_v98_pipeline(model, rng)
    all_results["v98_sudoku_medium_K20"] = v98_result

    # ============================================================
    # Save results
    # ============================================================
    out_path = ".cache/linear_probes_ridge_results.json"
    def _convert(obj):
        if isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_convert(v) for v in obj]
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, (np.floating, float)):
            if math.isnan(obj) or math.isinf(obj):
                return None
            return float(obj)
        return obj

    with open(out_path, "w") as f:
        json.dump(_convert(all_results), f, indent=2)
    print(f"\nRaw results saved to {out_path}")

    # ============================================================
    # Print reports
    # ============================================================
    print("\n" + "="*70)
    print("FINAL REPORTS")
    print("="*70)
    print(format_v98_report(v98_result))
    print()
    print(format_v110_report(v110_result))
    print("\nDONE.")


if __name__ == "__main__":
    main()
