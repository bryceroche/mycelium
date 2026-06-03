"""v105.9 linear probe: does pooled cell_hidden encode the number?

Decisive diagnostic for the train/val plateau on v105.9. Three explanations:
  A. cell_hidden ENCODES the number, decoder CAN'T READ it → fix decoder
  B. cell_hidden ENCODES MAGNITUDE only, not specific value → fix breathing
  C. cell_hidden ENCODES the number, AR decoder OVERFITS train → regularize

This probe trains a SIMPLE linear classifier on cell_hidden → 200-bin label
(same readout shape as v107). Compares probe acc to v107's actual number-bin acc
to determine whether the breathing produces enough number info for a clean
decoder to extract.

ALSO trains a linear regression cell_hidden → log(1 + value) for R² estimate.

Run:
  CKPT=.cache/fg_v105_5_ckpts/v105_9_prod_step2000.safetensors \
    DEV='PCI+AMD' .venv/bin/python -u scripts/diag_v105_9_number_probe.py
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
    apply_hierarchical_ib_codebook,
    apply_projection_waist,
    V105_5_N_HEADS,
)
from mycelium.factor_graph_data_v105_5 import FactorGraphLoaderV105_5
from mycelium.breathing import _layernorm


K_MAX     = int(getenv("V105_5_K_MAX", "8"))
N_DIGITS  = int(getenv("V105_5_N_DIGITS", "5"))
N_MAX     = int(getenv("V105_5_N_MAX", "16"))
F_MAX     = int(getenv("V105_5_F_MAX", "8"))
WAIST     = int(getenv("V105_5_WAIST", "512"))
N_CODE    = int(getenv("V105_5_CODEBOOK_N", "32"))
N_HEADS   = 16

N_BATCHES = int(getenv("N_BATCHES", "60"))
BATCH     = int(getenv("BATCH", "8"))
CKPT      = getenv("CKPT", ".cache/fg_v105_5_ckpts/v105_9_prod_step2000.safetensors")
VAL_PATH  = getenv("VAL_PATH", ".cache/factor_graph_test_loguniform.jsonl")
PROBE_EPOCHS = int(getenv("PROBE_EPOCHS", "300"))
PROBE_LR     = float(getenv("PROBE_LR", "1e-3"))
N_BINS       = int(getenv("N_BINS", "200"))


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


def extract_cell_hidden(model, digit_init, node_kinds, staging_mask, head_op_mask,
                        K=K_MAX, n_max=N_MAX, f_max=F_MAX, n_digits=N_DIGITS):
    """Forward and return pooled cell_hidden at last breath."""
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

    magnitude_head_w    = model.fg_v105_5_magnitude_head_w
    magnitude_head_b    = model.fg_v105_5_magnitude_head_b
    magnitude_centroids = model.fg_v105_5_magnitude_centroids

    B = int(digit_init.shape[0])
    T = n_max * n_digits + f_max

    x = embed_factor_graph_v105_5(
        digit_init, node_kinds,
        var_pos_embed, factor_pos_embed, node_kind_embed,
        digit_codebook, digit_rope_cos, digit_rope_sin,
        n_max=n_max, n_digits=n_digits, f_max=f_max,
    )
    x = x.cast(dtypes.half) if x.dtype != dtypes.half else x
    layers = list(model.block.layers)

    cell_hidden_final = None

    for k in range(K):
        be_k  = breath_embed[k].reshape(1, 1, -1).cast(x.dtype)
        x_in  = x + be_k
        x_pre = x

        stk   = staging_mask[:, k, :, :]
        stk_h = stk.reshape(B, 1, T, T).expand(B, V105_5_N_HEADS, T, T)
        combined = stk_h.cast(x.dtype) + head_op_mask.cast(x.dtype)

        h = x_in
        for layer in layers[:4]:
            h = fg_layer_forward_v105_5(layer, h, combined)

        h = apply_hierarchical_ib_codebook(
            h, ib_codebook, family_centroids, leaf_to_family_oh,
            ib_temperature, delta_gate_quant[k],
        )
        h = apply_projection_waist(h, W_compress, W_expand, b_compress, b_expand)

        gate_k = delta_gate[k].cast(h.dtype).reshape(1, 1, 1)
        delta  = h - x_pre
        x      = x_pre + gate_k * delta

        if k == K - 1:
            x_ln = _layernorm(x, model.ln_f_g, model.ln_f_b,
                              model.cfg.layer_norm_eps).cast(dtypes.float)
            n_var = n_max * n_digits
            var_tokens = x_ln[:, :n_var, :].reshape(B, n_max, n_digits, -1)
            cell_h = var_tokens.mean(axis=2)  # (B, n_max, H)
            mh_w = magnitude_head_w.cast(dtypes.float)
            mh_b = magnitude_head_b.cast(dtypes.float)
            magnitude_logits = cell_h @ mh_w + mh_b.reshape(1, 1, -1)
            magnitude_probs  = magnitude_logits.softmax(axis=-1)
            mc = magnitude_centroids.cast(dtypes.float)
            magnitude_embed = magnitude_probs @ mc
            cell_hidden_final = cell_h + magnitude_embed  # add magnitude

    return cell_hidden_final


def value_to_bin(v, n_bins, max_value=99999):
    """Log-spaced binning for [0, max_value]."""
    if v <= 0:
        return 0
    log_max = math.log(max_value + 1)
    log_v = math.log(v + 1)
    return min(int((log_v / log_max) * n_bins), n_bins - 1)


def softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def train_linear_classifier(X, y, n_classes, epochs=PROBE_EPOCHS, lr=PROBE_LR):
    N, H = X.shape
    rng = np.random.RandomState(0)
    idx = rng.permutation(N)
    n_train = int(N * 0.8)
    tr, te = idx[:n_train], idx[n_train:]
    Xtr, ytr = X[tr], y[tr]
    Xte, yte = X[te], y[te]
    W = rng.randn(H, n_classes).astype(np.float32) * (1.0 / math.sqrt(H))
    Y_oh = np.zeros((len(ytr), n_classes), dtype=np.float32)
    Y_oh[np.arange(len(ytr)), ytr] = 1.0
    m = np.zeros_like(W); v = np.zeros_like(W); beta1, beta2, eps = 0.9, 0.999, 1e-8
    for ep in range(1, epochs + 1):
        logits = Xtr @ W
        p = softmax(logits)
        grad = (Xtr.T @ (p - Y_oh)) / len(ytr)
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad ** 2)
        mh = m / (1 - beta1 ** ep)
        vh = v / (1 - beta2 ** ep)
        W -= lr * mh / (np.sqrt(vh) + eps)
    tr_acc = float(((Xtr @ W).argmax(axis=-1) == ytr).mean())
    te_acc = float(((Xte @ W).argmax(axis=-1) == yte).mean())
    return tr_acc, te_acc


def train_linear_regression(X, y):
    """Closed-form least-squares: w = (X^T X + λI)^-1 X^T y"""
    N, H = X.shape
    lam = 1e-2
    XtX = X.T @ X + lam * np.eye(H, dtype=np.float32)
    Xty = X.T @ y
    w = np.linalg.solve(XtX, Xty)
    yhat = X @ w
    ss_res = ((y - yhat) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1.0 - ss_res / (ss_tot + 1e-8)
    mae = np.abs(y - yhat).mean()
    return float(r2), float(mae)


def main():
    print("=== v105.9 cell_hidden linear probe ===", flush=True)
    print(f"ckpt: {CKPT}", flush=True)
    print(f"n_batches={N_BATCHES} batch={BATCH} → ~{N_BATCHES * BATCH * 4} cells", flush=True)
    print("", flush=True)

    cfg = Config()
    sd  = _load_state()
    model = load_breathing(cfg, sd=sd); del sd
    cast_layers_fp32(model)
    attach_fg_params_v105_5(model, hidden=cfg.hidden,
        n_digits=N_DIGITS, n_max=N_MAX, f_max=F_MAX, k_max=K_MAX,
        waist=WAIST, n_code=N_CODE)
    Device[Device.DEFAULT].synchronize()
    load_ckpt_v105_5(model, CKPT)
    print(f"loaded ckpt OK  hidden={cfg.hidden}", flush=True)

    val_loader = FactorGraphLoaderV105_5(
        VAL_PATH, batch_size=BATCH, difficulty_filter=None, curriculum=False,
        n_max=N_MAX, f_max=F_MAX, k_max=K_MAX, n_heads=N_HEADS,
        n_digits=N_DIGITS, seed=43,
    )

    feats = []
    values = []
    bins = []
    magnitudes = []

    Tensor.training = False
    nb = 0
    for batch in val_loader.iter_eval(batch_size=BATCH):
        cell_hidden = extract_cell_hidden(
            model, batch["digit_init"], batch["node_kinds"],
            batch["staging_mask"], batch["head_op_mask"],
            K=K_MAX, n_max=N_MAX, f_max=F_MAX, n_digits=N_DIGITS,
        )
        ch_np = cell_hidden.realize().numpy()  # (B, n_max, H)
        obs   = batch["observed_mask"].numpy()
        gold_digits = batch["gold_digits"].numpy()  # (B, n_max, n_digits)

        B = ch_np.shape[0]
        for b in range(B):
            nv = int(batch["n_vars_total"][b])
            for v in range(min(nv, N_MAX)):
                if obs[b, v] == 1:
                    continue
                # Reconstruct gold value from LSD-first digits
                gold_val = 0
                for p in range(N_DIGITS):
                    gold_val += int(gold_digits[b, v, p]) * (10 ** p)
                if gold_val < 0:  # skip invalid
                    continue
                feats.append(ch_np[b, v])
                values.append(gold_val)
                bins.append(value_to_bin(gold_val, N_BINS))
                # Magnitude class: 0=1digit, 1=2digit, 2=3digit, 3=4+digit
                if gold_val < 10: mag = 0
                elif gold_val < 100: mag = 1
                elif gold_val < 1000: mag = 2
                else: mag = 3
                magnitudes.append(mag)
        nb += 1
        if nb >= N_BATCHES:
            break
    Tensor.training = True

    X = np.stack(feats).astype(np.float32)
    y_val = np.array(values, dtype=np.float32)
    y_log = np.log1p(y_val)
    y_bin = np.array(bins, dtype=np.int64)
    y_mag = np.array(magnitudes, dtype=np.int64)

    print(f"extracted {len(X)} cells from {nb} batches", flush=True)
    print(f"value range: [{y_val.min():.0f}, {y_val.max():.0f}]  mean={y_val.mean():.0f}", flush=True)
    print("", flush=True)

    # --- Linear classifier on 200-bin label ---
    print(f"=== Linear probe: cell_hidden → {N_BINS}-bin label ===", flush=True)
    tr_acc_bin, te_acc_bin = train_linear_classifier(X, y_bin, N_BINS)
    print(f"  train_acc = {tr_acc_bin:.4f}", flush=True)
    print(f"  test_acc  = {te_acc_bin:.4f}", flush=True)
    print(f"  chance    = {1.0/N_BINS:.4f}", flush=True)
    print("", flush=True)

    # --- Linear classifier on 4-way magnitude ---
    print("=== Linear probe: cell_hidden → 4-way magnitude ===", flush=True)
    tr_acc_mag, te_acc_mag = train_linear_classifier(X, y_mag, 4)
    print(f"  train_acc = {tr_acc_mag:.4f}", flush=True)
    print(f"  test_acc  = {te_acc_mag:.4f}", flush=True)
    print(f"  chance    = 0.25", flush=True)
    print("", flush=True)

    # --- Linear regression on log(value) ---
    print("=== Linear probe: cell_hidden → log(1 + value) ===", flush=True)
    r2, mae = train_linear_regression(X, y_log)
    pred = np.exp(mae) - 1  # rough back-transform
    print(f"  R² (closed-form) = {r2:.4f}", flush=True)
    print(f"  MAE (log space)  = {mae:.4f}  (≈ multiplicative factor of e^MAE)", flush=True)
    print("", flush=True)

    # --- Verdict ---
    print("=== VERDICT ===", flush=True)
    if te_acc_bin > 0.30:
        print(f"  HIGH-encoding (bin acc {te_acc_bin:.2%}):", flush=True)
        print("    cell_hidden contains precise number info.", flush=True)
        print("    The DECODER is the bottleneck (overfits, can't extract).", flush=True)
        print("    → Fix: bigger decoder, dropout, or Fourier codebook warm-start.", flush=True)
    elif te_acc_bin > 0.10:
        print(f"  PARTIAL-encoding (bin acc {te_acc_bin:.2%}):", flush=True)
        print("    cell_hidden has some number info but not enough.", flush=True)
        print("    → Either improve breathing or improve decoder; both contribute.", flush=True)
    else:
        print(f"  LOW-encoding (bin acc {te_acc_bin:.2%}):", flush=True)
        print("    cell_hidden encodes constraint-consistency but not value.", flush=True)
        print("    Breathing produces 'in the 1200 range' but not '1234' precision.", flush=True)
        print("    → Fix: add number-level loss (per-NUMBER CE or MSE) to breathing.", flush=True)

    if te_acc_mag > 0.8:
        print(f"  Magnitude IS well-encoded ({te_acc_mag:.2%}) — breathing knows scale.", flush=True)


if __name__ == "__main__":
    main()
