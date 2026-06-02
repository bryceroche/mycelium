"""v105.4 hidden-state linear-probe diagnostic.

Question: is the digit-prediction failure caused by
  (a) the codebook readout collapsing despite the hidden states having useful info, OR
  (b) the hidden states themselves losing input sensitivity at digit positions?

Method per position p in 0..N_DIGITS-1:
  1.  Tap the final-breath hidden state at every (variable, digit-position) token,
      AFTER the magnitude_embed has been added — i.e. exactly the tensor that
      gets matmul'd against the digit codebook in the production forward.
  2.  Train a 1-layer linear classifier (hidden→10) on (X_p, y_p) for unobserved
      VALID cells across ~30 val batches. 80/20 train/test split.
  3.  Compare probe test acc to the trained codebook acc on the same cells.
  4.  Inspect mean pairwise cosine similarity of hidden states across problems.

Three diagnoses possible per position:
  • linear_probe >> codebook                       → readout collapse.
  • linear_probe ≈ codebook AND cos_sim > 0.9      → hidden state collapse.
  • linear_probe ≈ codebook AND cos_sim < 0.9      → non-linear separability.

Run:
  CKPT=.cache/fg_v105_4_ckpts/v105_4_prod_step3000.safetensors \
    DEV='PCI+AMD' .venv/bin/python -u scripts/diag_v105_4_linear_probe.py

Constraints respected:
  • No edits to mycelium/factor_graph_v105_4.py — the tapped forward below is a
    LOCAL reproduction of `fg_breathing_forward_v105_4`'s last-breath path using
    only the public helpers (`embed_factor_graph_v105_4`,
    `fg_layer_forward_v105_4`, `apply_hierarchical_ib_codebook`,
    `apply_projection_waist`) plus the model's attached v105.4 parameters.
  • Backbone is frozen; only the (1024, 10) probe weights are trained, in numpy.
"""
from __future__ import annotations

import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from tinygrad import Tensor, Device, dtypes
from tinygrad.helpers import getenv

from mycelium import Config
from mycelium.loader import _load_state, load_breathing
from mycelium.factor_graph_v105_4 import (
    attach_fg_params_v105_4,
    load_ckpt_v105_4,
    embed_factor_graph_v105_4,
    fg_layer_forward_v105_4,
    apply_hierarchical_ib_codebook,
    apply_projection_waist,
    V105_4_N_HEADS,
)
from mycelium.factor_graph_data_v105_4 import FactorGraphLoaderV105_4
from mycelium.breathing import _layernorm


# ---------------------------------------------------------------------------
# Config (env-overridable)
# ---------------------------------------------------------------------------

K_MAX     = int(getenv("V105_4_K_MAX", "8"))
N_DIGITS  = int(getenv("V105_4_N_DIGITS", "5"))
N_MAX     = int(getenv("V105_4_N_MAX", "16"))
F_MAX     = int(getenv("V105_4_F_MAX", "8"))
WAIST     = int(getenv("V105_4_WAIST", "512"))
N_CODE    = int(getenv("V105_4_CODEBOOK_N", "32"))
N_HEADS   = 16

N_BATCHES = int(getenv("N_BATCHES", "30"))
BATCH     = int(getenv("BATCH", "8"))
CKPT      = getenv(
    "CKPT", ".cache/fg_v105_4_ckpts/v105_4_prod_step3000.safetensors"
)
VAL_PATH  = getenv("VAL_PATH", ".cache/factor_graph_test_loguniform.jsonl")
DIFF_FILTER = getenv("DIFF_FILTER", "easy") or None

# Probe training hyperparams
PROBE_EPOCHS    = int(getenv("PROBE_EPOCHS", "200"))
PROBE_LR        = float(getenv("PROBE_LR", "1e-3"))
PROBE_TRAIN_FRAC = float(getenv("PROBE_TRAIN_FRAC", "0.8"))
COS_SIM_PAIRS   = int(getenv("COS_SIM_PAIRS", "2000"))   # # random pairs to estimate

POS_LABELS = ["ones", "tens", "hundreds", "thousands", "ten-thousands"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def cast_layers_fp32(model):
    """Cast model layer weights to fp32 (matches diag_v105_4_per_position_acc.py)."""
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


# ---------------------------------------------------------------------------
# Tapped forward: reproduces fg_breathing_forward_v105_4 but returns the
# final-breath per-position hidden state (after magnitude_embed addition).
# Returns:
#   var_hidden_with_mag : (B, N_MAX, N_DIGITS, H) — what codebook actually reads
#   var_hidden_pre_mag  : (B, N_MAX, N_DIGITS, H) — before mag_embed (control)
#   digit_logits        : (B, N_MAX, N_DIGITS, 10) — final-breath codebook output
# ---------------------------------------------------------------------------

def fg_forward_tapped_v105_4(
    model,
    digit_init: Tensor,
    node_kinds: Tensor,
    staging_mask: Tensor,
    head_op_mask: Tensor,
    K: int = K_MAX,
    n_max: int = N_MAX,
    f_max: int = F_MAX,
    n_digits: int = N_DIGITS,
):
    digit_codebook   = model.fg_v105_4_digit_codebook
    digit_rope_cos   = model.fg_v105_4_digit_rope_cos
    digit_rope_sin   = model.fg_v105_4_digit_rope_sin
    var_pos_embed    = model.fg_v105_4_var_pos_embed
    factor_pos_embed = model.fg_v105_4_factor_pos_embed
    node_kind_embed  = model.fg_v105_4_node_kind_embed
    breath_embed     = model.fg_v105_4_breath_embed
    delta_gate       = model.fg_v105_4_delta_gate

    W_compress = model.fg_v105_4_W_compress
    b_compress = model.fg_v105_4_b_compress
    W_expand   = model.fg_v105_4_W_expand
    b_expand   = model.fg_v105_4_b_expand

    ib_codebook       = model.fg_v105_4_ib_codebook
    family_centroids  = model.fg_v105_4_family_centroids
    leaf_to_family_oh = model.fg_v105_4_leaf_to_family_oh
    delta_gate_quant  = model.fg_v105_4_delta_gate_quant
    ib_temperature    = model.fg_v105_4_ib_temperature

    magnitude_head_w    = model.fg_v105_4_magnitude_head_w
    magnitude_head_b    = model.fg_v105_4_magnitude_head_b
    magnitude_centroids = model.fg_v105_4_magnitude_centroids

    B = int(digit_init.shape[0])
    T = n_max * n_digits + f_max

    x = embed_factor_graph_v105_4(
        digit_init, node_kinds,
        var_pos_embed, factor_pos_embed, node_kind_embed,
        digit_codebook, digit_rope_cos, digit_rope_sin,
        n_max=n_max, n_digits=n_digits, f_max=f_max,
    )
    x = x.cast(dtypes.half) if x.dtype != dtypes.half else x

    layers = list(model.block.layers)

    var_tokens_pre_mag = None
    var_tokens_with_mag = None
    digit_logits_final = None

    for k in range(K):
        be_k  = breath_embed[k].reshape(1, 1, -1).cast(x.dtype)
        x_in  = x + be_k
        x_pre = x

        stk   = staging_mask[:, k, :, :]
        stk_h = stk.reshape(B, 1, T, T).expand(B, V105_4_N_HEADS, T, T)
        combined = stk_h.cast(x.dtype) + head_op_mask.cast(x.dtype)

        h = x_in
        for layer in layers[:4]:
            h = fg_layer_forward_v105_4(layer, h, combined)

        h = apply_hierarchical_ib_codebook(
            h, ib_codebook, family_centroids, leaf_to_family_oh,
            ib_temperature, delta_gate_quant[k],
        )
        h = apply_projection_waist(h, W_compress, W_expand, b_compress, b_expand)

        gate_k = delta_gate[k].cast(h.dtype).reshape(1, 1, 1)
        delta  = h - x_pre
        x      = x_pre + gate_k * delta

        # Last breath: tap the per-position hidden state before AND after
        # the magnitude_embed is added.
        if k == K - 1:
            x_ln = _layernorm(
                x, model.ln_f_g, model.ln_f_b, model.cfg.layer_norm_eps,
            ).cast(dtypes.float)
            n_var_tokens = n_max * n_digits
            var_tokens   = x_ln[:, :n_var_tokens, :]
            var_tokens_r = var_tokens.reshape(B, n_max, n_digits, -1)

            cell_hidden = var_tokens_r.mean(axis=2)
            mh_w = magnitude_head_w.cast(dtypes.float)
            mh_b = magnitude_head_b.cast(dtypes.float)
            magnitude_logits = cell_hidden @ mh_w + mh_b.reshape(1, 1, -1)
            magnitude_probs  = magnitude_logits.softmax(axis=-1)
            mc = magnitude_centroids.cast(dtypes.float)
            magnitude_embed_cell = magnitude_probs @ mc
            magnitude_embed_dg = magnitude_embed_cell.reshape(
                B, n_max, 1, -1
            ).expand(B, n_max, n_digits, int(var_tokens_r.shape[-1]))

            var_tokens_pre_mag = var_tokens_r
            var_tokens_with_mag = var_tokens_r + magnitude_embed_dg

            # Match production codebook readout — use AR (LSD-first) since
            # v105_4_prod was trained with V105_4_AR_DIGITS=1 LSD-first.
            cb_fp_all = digit_codebook.cast(dtypes.float)
            ar_logits_list = [None] * n_digits
            cond_accum = Tensor.zeros(
                (B, n_max, int(x_ln.shape[-1])), dtype=dtypes.float,
            ).contiguous()
            ar_cond_scale_t = Tensor(
                np.array([0.5], dtype=np.float32), dtype=dtypes.float,
            ).reshape(1, 1, 1)
            for p in range(n_digits):    # LSD-first
                cb_p = cb_fp_all[p]
                pos_hidden = var_tokens_with_mag[:, :, p, :] + cond_accum
                pos_logits = pos_hidden @ cb_p.T
                ar_logits_list[p] = pos_logits
                pos_probs = pos_logits.softmax(axis=-1)
                pos_embed = pos_probs @ cb_p
                cond_accum = cond_accum + pos_embed * ar_cond_scale_t.cast(pos_embed.dtype)
            digit_logits_final = Tensor.stack(*ar_logits_list, dim=2)

    return var_tokens_pre_mag, var_tokens_with_mag, digit_logits_final


# ---------------------------------------------------------------------------
# Numpy linear probe training (closed-form via Adam on softmax CE)
# ---------------------------------------------------------------------------

def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def train_linear_probe(X: np.ndarray, y: np.ndarray,
                       epochs: int = PROBE_EPOCHS,
                       lr: float = PROBE_LR,
                       seed: int = 0) -> tuple[float, float, np.ndarray]:
    """X: (N, H) float32. y: (N,) int 0..9.

    Returns (train_acc, test_acc, W).  W: (H, 10).
    Linear classifier, no bias, Adam optimizer, cross-entropy.
    """
    rng = np.random.RandomState(seed)
    N, H = X.shape
    n_classes = 10

    # Train/test split
    idx = rng.permutation(N)
    n_train = int(N * PROBE_TRAIN_FRAC)
    tr_idx, te_idx = idx[:n_train], idx[n_train:]
    Xtr, ytr = X[tr_idx], y[tr_idx]
    Xte, yte = X[te_idx], y[te_idx]

    # Initialize weights small (Glorot-ish)
    W = rng.randn(H, n_classes).astype(np.float32) * (1.0 / math.sqrt(H))

    # Adam state
    m = np.zeros_like(W); v = np.zeros_like(W); beta1, beta2, eps = 0.9, 0.999, 1e-8

    # Mini-batch full-batch gradient descent (data is small enough)
    Y_onehot = np.zeros((len(ytr), n_classes), dtype=np.float32)
    Y_onehot[np.arange(len(ytr)), ytr] = 1.0

    for ep in range(1, epochs + 1):
        logits = Xtr @ W                                  # (N, 10)
        probs  = _softmax(logits)
        grad   = (Xtr.T @ (probs - Y_onehot)) / len(ytr) # (H, 10)

        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad ** 2)
        mh = m / (1 - beta1 ** ep)
        vh = v / (1 - beta2 ** ep)
        W -= lr * mh / (np.sqrt(vh) + eps)

    tr_acc = (Xtr @ W).argmax(axis=-1) == ytr
    te_acc = (Xte @ W).argmax(axis=-1) == yte
    return float(tr_acc.mean()), float(te_acc.mean()), W


# ---------------------------------------------------------------------------
# Cosine-similarity sampling
# ---------------------------------------------------------------------------

def estimate_pairwise_cos(X: np.ndarray, n_pairs: int = COS_SIM_PAIRS,
                          seed: int = 0) -> tuple[float, float, float, float]:
    rng = np.random.RandomState(seed)
    N = X.shape[0]
    if N < 2:
        return float("nan"), float("nan"), float("nan"), float("nan")
    # Norm-1 rows
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    Xn = X / norms
    n_pairs = min(n_pairs, N * (N - 1) // 2)
    i = rng.randint(0, N, size=n_pairs)
    j = rng.randint(0, N, size=n_pairs)
    mask = i != j
    i = i[mask]; j = j[mask]
    if len(i) == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    cos = (Xn[i] * Xn[j]).sum(axis=1)
    return (float(cos.mean()), float(cos.std()),
            float(cos.min()), float(cos.max()))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=== v105.4 hidden-state linear-probe diagnostic ===", flush=True)
    print(f"K={K_MAX}  n_digits={N_DIGITS}  n_batches={N_BATCHES}  batch={BATCH}", flush=True)
    print(f"ckpt: {CKPT}", flush=True)
    print(f"val:  {VAL_PATH}  difficulty_filter={DIFF_FILTER}", flush=True)
    print(f"probe: {PROBE_EPOCHS} epochs lr={PROBE_LR}  train_frac={PROBE_TRAIN_FRAC}", flush=True)
    print("", flush=True)

    cfg = Config()
    sd  = _load_state()
    model = load_breathing(cfg, sd=sd); del sd
    cast_layers_fp32(model)
    attach_fg_params_v105_4(
        model, hidden=cfg.hidden,
        n_digits=N_DIGITS, n_max=N_MAX, f_max=F_MAX, k_max=K_MAX,
        waist=WAIST, n_code=N_CODE,
    )
    Device[Device.DEFAULT].synchronize()
    load_ckpt_v105_4(model, CKPT)
    print(f"loaded ckpt OK  hidden={cfg.hidden}", flush=True)

    val_loader = FactorGraphLoaderV105_4(
        VAL_PATH, batch_size=BATCH,
        difficulty_filter=DIFF_FILTER, curriculum=False,
        n_max=N_MAX, f_max=F_MAX, k_max=K_MAX, n_heads=N_HEADS,
        n_digits=N_DIGITS, seed=43,
    )

    H = cfg.hidden

    # ----- Hidden state extraction -----
    Tensor.training = False
    feats_by_pos: list[list[np.ndarray]] = [[] for _ in range(N_DIGITS)]
    labels_by_pos: list[list[int]] = [[] for _ in range(N_DIGITS)]
    cb_correct_by_pos = [0 for _ in range(N_DIGITS)]
    cb_total_by_pos   = [0 for _ in range(N_DIGITS)]

    nb = 0
    for batch in val_loader.iter_eval(batch_size=BATCH):
        var_pre, var_with_mag, digit_logits = fg_forward_tapped_v105_4(
            model, batch["digit_init"], batch["node_kinds"],
            batch["staging_mask"], batch["head_op_mask"],
            K=K_MAX, n_max=N_MAX, f_max=F_MAX, n_digits=N_DIGITS,
        )

        # Realize on CPU as numpy
        hidden_with_mag_np = var_with_mag.realize().numpy()   # (B, N_MAX, N_DIGITS, H)
        cb_pred_np = digit_logits.argmax(axis=-1).realize().numpy()  # (B, N_MAX, N_DIGITS)

        gold = batch["gold_digits"].numpy()             # (B, N_MAX, N_DIGITS)
        obs  = batch["observed_mask"].numpy()           # (B, N_MAX)
        valid = batch["digit_valid_mask"].numpy()       # (B, N_MAX, N_DIGITS)

        B = hidden_with_mag_np.shape[0]
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
                    feats_by_pos[p].append(hidden_with_mag_np[b, v, p])
                    labels_by_pos[p].append(int(gold[b, v, p]))
                    cb_total_by_pos[p] += 1
                    if int(cb_pred_np[b, v, p]) == int(gold[b, v, p]):
                        cb_correct_by_pos[p] += 1

        nb += 1
        if nb >= N_BATCHES:
            break
    Tensor.training = True
    print(f"extracted hidden states from {nb} batches", flush=True)
    for p in range(N_DIGITS):
        print(f"  pos{p} ({POS_LABELS[p]:>14s}): n_cells={len(feats_by_pos[p])}",
              flush=True)

    # ----- Per-position linear probe -----
    print("", flush=True)
    print("=== Training per-position linear probes ===", flush=True)

    rows: list[dict] = []
    for p in range(N_DIGITS):
        Xp = np.stack(feats_by_pos[p], axis=0).astype(np.float32)
        yp = np.asarray(labels_by_pos[p], dtype=np.int64)
        n_cells = len(yp)

        tr_acc, te_acc, _W = train_linear_probe(
            Xp, yp, epochs=PROBE_EPOCHS, lr=PROBE_LR, seed=p,
        )
        cb_acc = cb_correct_by_pos[p] / max(cb_total_by_pos[p], 1)
        cos_mean, cos_std, cos_min, cos_max = estimate_pairwise_cos(
            Xp, n_pairs=COS_SIM_PAIRS, seed=p,
        )

        rows.append({
            "pos": p, "label": POS_LABELS[p],
            "n_cells": n_cells,
            "probe_train": tr_acc, "probe_test": te_acc,
            "codebook": cb_acc,
            "cos_mean": cos_mean, "cos_std": cos_std,
            "cos_min": cos_min, "cos_max": cos_max,
        })

        print(
            f"  pos{p} ({POS_LABELS[p]:>14s})  n={n_cells:>5d}  "
            f"probe_train={tr_acc:.3f}  probe_test={te_acc:.3f}  "
            f"codebook={cb_acc:.3f}  "
            f"cos_mean={cos_mean:.3f}±{cos_std:.3f}  "
            f"[{cos_min:.3f},{cos_max:.3f}]",
            flush=True,
        )

    # ----- Final table -----
    print("", flush=True)
    print("=== SUMMARY TABLE ===", flush=True)
    print(
        f"{'pos':>3}  {'label':>14}  {'n_cells':>8}  "
        f"{'probe_acc':>10}  {'cb_acc':>8}  {'gap':>8}  "
        f"{'cos_mean':>10}  {'cos_std':>8}",
        flush=True,
    )
    print("-" * 90, flush=True)
    for r in rows:
        gap = r["probe_test"] - r["codebook"]
        print(
            f"{r['pos']:>3}  {r['label']:>14}  {r['n_cells']:>8d}  "
            f"{r['probe_test']:>10.3f}  {r['codebook']:>8.3f}  "
            f"{gap:>+8.3f}  {r['cos_mean']:>10.3f}  {r['cos_std']:>8.3f}",
            flush=True,
        )

    # ----- Per-position diagnosis -----
    print("", flush=True)
    print("=== PER-POSITION DIAGNOSIS ===", flush=True)
    verdicts: list[str] = []
    for r in rows:
        probe = r["probe_test"]
        cb    = r["codebook"]
        gap   = probe - cb
        cos   = r["cos_mean"]

        if gap > 0.20:
            verdict = ("readout-collapse",
                       "info in hidden state but codebook readout collapsed")
        elif cos > 0.90 and gap <= 0.20:
            verdict = ("hidden-state-collapse",
                       "hidden states themselves collapsed; no input sensitivity")
        elif cos <= 0.90 and gap <= 0.20:
            verdict = ("non-linear",
                       "hidden states varied but not LINEARLY separable by digit")
        else:
            verdict = ("ambiguous",
                       "doesn't fit a clean bucket")

        verdicts.append(verdict[0])
        print(
            f"  pos{r['pos']} ({r['label']:>14s}):  probe={probe:.3f}  cb={cb:.3f}  "
            f"gap={gap:+.3f}  cos_mean={cos:.3f}  →  {verdict[0]}",
            flush=True,
        )
        print(f"      {verdict[1]}", flush=True)

    # ----- Overall verdict -----
    print("", flush=True)
    print("=== OVERALL VERDICT ===", flush=True)
    n_readout_collapse = sum(v == "readout-collapse" for v in verdicts)
    n_hidden_collapse  = sum(v == "hidden-state-collapse" for v in verdicts)
    n_nonlinear        = sum(v == "non-linear" for v in verdicts)
    n_ambiguous        = sum(v == "ambiguous" for v in verdicts)

    print(f"  readout-collapse positions:       {n_readout_collapse}/{len(rows)}",
          flush=True)
    print(f"  hidden-state-collapse positions:  {n_hidden_collapse}/{len(rows)}",
          flush=True)
    print(f"  non-linear positions:             {n_nonlinear}/{len(rows)}",
          flush=True)
    print(f"  ambiguous positions:              {n_ambiguous}/{len(rows)}",
          flush=True)

    print("", flush=True)
    if n_readout_collapse > n_hidden_collapse + n_nonlinear:
        print("VERDICT: CODEBOOK BOTTLENECK", flush=True)
        print("  → information is in the hidden states; codebook readout is the choke.",
              flush=True)
        print("  Next steps:", flush=True)
        print("    1. Replace per-position codebook with a small MLP head"
              " (hidden→512→10).", flush=True)
        print("    2. Try a VQ-VAE-style readout: codebook + commitment loss.",
              flush=True)
        print("    3. Drop AR conditioning during training (the cond_accum signal"
              " might be poisoning the probe-friendly directions).", flush=True)
    elif n_hidden_collapse > n_readout_collapse + n_nonlinear:
        print("VERDICT: ARCHITECTURE BOTTLENECK", flush=True)
        print("  → hidden states themselves lost input sensitivity"
              " at digit positions.", flush=True)
        print("  Next steps:", flush=True)
        print("    1. Curriculum on a single-digit easy split to bootstrap"
              " digit-channel signal before mixing.", flush=True)
        print("    2. Hybrid with v107 if v107 has a carry channel that prevents collapse.",
              flush=True)
        print("    3. Add a residual digit-identity skip from input → output to"
              " keep the digit channel alive.", flush=True)
    elif n_nonlinear > 0:
        print("VERDICT: NON-LINEAR ENCODING", flush=True)
        print("  → digit information is in the hidden state but not"
              " linearly separable.", flush=True)
        print("  Next steps:", flush=True)
        print("    1. Train a 2-layer MLP probe to confirm.", flush=True)
        print("    2. Replace codebook readout with a deeper head.", flush=True)
    else:
        print("VERDICT: MIXED / INCONCLUSIVE — inspect per-position diagnoses.",
              flush=True)
    print("", flush=True)


if __name__ == "__main__":
    main()
