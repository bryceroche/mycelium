"""probe_pairwise_confidence.py — PAIRWISE-vs-UNARY confidence probe (frozen deducer).

THE DECISION
------------
The frozen graph-coloring deducer (fg_coloring_k16) cannot tell CONSISTENT-CORRECT
from CONSISTENT-WRONG: a locally-plausible assignment whose cells each look fine but
whose JOINT violates a factor.  HYPOTHESIS (Bryce): the per-cell-independent /
product-form readout (cell_logits = layernorm(x) @ codebook.T,
factor_graph_engine.py:386) structurally cannot represent "each cell plausible but
JOINTLY inconsistent", which a PAIRWISE / Bethe readout could.

SUBTLETY: the K=16-breath attention over the factor masks ALREADY mixes neighbour
info into each cell RESIDUAL.  So the per-cell residual MAY already carry the joint
signal even though the product-form readout discards it.  We test THREE probes to
localise WHERE the signal lives, on the FROZEN model:

  C BELIEF-ONLY    : [max_prob, top1-top2 margin, entropy]      (the product/calib baseline)
  A UNARY-RESIDUAL : cell pre-readout residual (x_ln) + belief  (does the residual carry it?)
  B PAIRWISE-RESID : cell residual + AGG factor-neighbour residuals (mean,max) +
                     aggregated neighbour beliefs + belief        (explicit Bethe pairwise)

VERDICT (printed):
  BUILD-BETHE       : B >> A and B >> C  -> build the pairwise/Bethe readout.
  CHEAPER-UNARY-FIX : A ~ B, both >> C   -> residual already carries it; richer unary
                                            readout / better calibration suffices.
  DEEPER-PROBLEM    : A ~ B ~ C (all weak) -> signal not in the frozen residual.
  AMBIGUOUS         : mixed.

SELFTEST_ONLY=1: pure-CPU synthetic harness check.  (1) a PAIRWISE-detectable case
(is_wrong depends on neighbour agreement the unary features lack) -> confirms B >> A.
(2) a UNARY-only control -> confirms A ~ B.  Guards against a harness that can't see
a real B-vs-A difference either way.

RUN (GPU extraction, frozen ckpt):
  DEV=AMD FG_N_VALUES=3 FG_N_INSTANCES=8000 \
      .venv/bin/python3 scripts/probe_pairwise_confidence.py
SELFTEST (CPU):
  SELFTEST_ONLY=1 .venv/bin/python3 scripts/probe_pairwise_confidence.py

DISCIPLINE: frozen model (no backbone training); deterministic (seeded probe init +
instance-level split); pure-numpy probes (CPU); eager batched extraction (no new JIT
graph path); substrate-safe (no float32 literal in any JIT step — extraction reuses
the existing eager forward body).
"""
import ast
import os
import sys

_THIS_FILE = os.path.abspath(__file__)
sys.path.insert(0, os.path.dirname(os.path.dirname(_THIS_FILE)))

import numpy as np


# ==========================================================================
# parse gate
# ==========================================================================
def _ast_parse_ok() -> bool:
    with open(_THIS_FILE) as f:
        src = f.read()
    try:
        ast.parse(src)
        return True
    except SyntaxError as e:
        print(f"[ast.parse] FAILED: {e}", flush=True)
        return False


# ==========================================================================
# Probe machinery (pure numpy, CPU) — shared by SELFTEST and the real run
# ==========================================================================
def _standardize_fit(X):
    """Column z-score; return (Xn, mu, sd) with sd floored to avoid /0."""
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return (X - mu) / sd, mu, sd


def _standardize_apply(X, mu, sd):
    return (X - mu) / sd


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


def fit_logreg(X, y, l2=1.0, n_iter=400, lr=0.5, seed=0):
    """Pure-numpy L2-regularized logistic regression via full-batch gradient descent
    with a fixed step.  Deterministic given seed.  Returns (w, b)."""
    rng = np.random.RandomState(seed)
    n, d = X.shape
    w = rng.randn(d).astype(np.float64) * 0.001
    b = 0.0
    yf = y.astype(np.float64)
    for _ in range(n_iter):
        z = X @ w + b
        p = _sigmoid(z)
        g = p - yf                       # (n,)
        gw = (X.T @ g) / n + (l2 / n) * w
        gb = g.mean()
        w -= lr * gw
        b -= lr * gb
    return w, b


def predict_logreg(X, w, b):
    return _sigmoid(X @ w + b)


def auc_score(y_true, scores):
    """ROC-AUC via the Mann-Whitney U rank statistic.  Handles ties (avg ranks)."""
    y_true = np.asarray(y_true).astype(np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    sorted_scores = scores[order]
    i = 0
    N = len(scores)
    while i < N:
        j = i
        while j + 1 < N and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0     # 1-based average rank for the tie block
        ranks[order[i:j + 1]] = avg_rank
        i = j + 1
    sum_ranks_pos = ranks[y_true == 1].sum()
    u = sum_ranks_pos - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def ece_score(y_true, p_pred, n_bins=10):
    """Expected calibration error (equal-width bins on the predicted prob)."""
    y_true = np.asarray(y_true).astype(np.float64)
    p_pred = np.asarray(p_pred, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    N = len(y_true)
    for bi in range(n_bins):
        lo, hi = edges[bi], edges[bi + 1]
        if bi == n_bins - 1:
            m = (p_pred >= lo) & (p_pred <= hi)
        else:
            m = (p_pred >= lo) & (p_pred < hi)
        cnt = int(m.sum())
        if cnt == 0:
            continue
        conf = p_pred[m].mean()
        acc = y_true[m].mean()
        ece += (cnt / N) * abs(acc - conf)
    return float(ece)


def fit_eval_probe(Xtr, ytr, Xte, yte, hard_mask_te, l2=1.0, seed=0,
                   pca_dim=None, pca_basis=None):
    """Standardize -> (optional PCA) -> logreg -> AUC/ECE overall + hard subset.

    pca_basis: if given (V, mean_vec), project the STANDARDIZED features onto it
    (shared between train/test).  Returns dict of metrics + the predicted test probs.
    """
    Xtr_s, mu, sd = _standardize_fit(Xtr)
    Xte_s = _standardize_apply(Xte, mu, sd)
    if pca_basis is not None:
        V, pmean = pca_basis
        Xtr_s = (Xtr_s - pmean) @ V
        Xte_s = (Xte_s - pmean) @ V
    w, b = fit_logreg(Xtr_s, ytr, l2=l2, seed=seed)
    pte = predict_logreg(Xte_s, w, b)
    auc = auc_score(yte, pte)
    ece = ece_score(yte, pte)
    if hard_mask_te is not None and hard_mask_te.sum() >= 10:
        auc_hard = auc_score(yte[hard_mask_te], pte[hard_mask_te])
    else:
        auc_hard = float("nan")
    return {"auc": auc, "ece": ece, "auc_hard": auc_hard, "p": pte,
            "n_feat": Xtr_s.shape[1]}


def pca_basis_from(X, dim, seed=0):
    """Compute a (standardized-space) PCA basis (V (D,dim), mean (D,)) from X."""
    Xs, _, _ = _standardize_fit(X)
    mean = Xs.mean(axis=0)
    Xc = Xs - mean
    # SVD on the (possibly tall) centered matrix.
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    dim = min(dim, Vt.shape[0])
    V = Vt[:dim].T            # (D, dim)
    return (V, mean)


def pca_reduce_block(block_tr, block_te, dim, seed=0):
    """Fit a standardize+PCA on TRAIN block, apply to TRAIN/TEST.  Returns the
    reduced (train, test) matrices.  No test leakage (basis from train only)."""
    Xtr_s, mu, sd = _standardize_fit(block_tr)
    Xte_s = _standardize_apply(block_te, mu, sd)
    mean = Xtr_s.mean(axis=0)
    Xc = Xtr_s - mean
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    dim = min(dim, Vt.shape[0])
    V = Vt[:dim].T
    return (Xtr_s - mean) @ V, (Xte_s - mean) @ V


# ==========================================================================
# SELFTEST — synthetic pairwise-detectable + unary-only controls
# ==========================================================================
def run_selftest():
    print("=== SELFTEST: probe-fit + AUC machinery (CPU synthetic) ===", flush=True)
    rng = np.random.RandomState(123)

    # ---- sanity: AUC on a known-separable toy ----
    s = np.array([0.1, 0.4, 0.35, 0.8, 0.7, 0.9])
    yl = np.array([0, 0, 0, 1, 1, 1])
    assert abs(auc_score(yl, s) - 1.0) < 1e-9, auc_score(yl, s)
    # tie handling: all-equal scores -> AUC 0.5
    assert abs(auc_score(yl, np.zeros(6)) - 0.5) < 1e-9
    print("  [ok] AUC rank statistic (separable=1.0, ties=0.5)", flush=True)

    # ----------------------------------------------------------------------
    # Build a synthetic "factor graph": N instances, each V cells, chain edges.
    # Each cell has a hidden color (one-hot in the unary block), a noisy belief
    # margin, and an is_wrong label.  TWO regimes:
    #   "pairwise": is_wrong = a NEIGHBOUR has the same color (a not-equal
    #               violation).  This is LINEARLY readable from the neighbour
    #               color-agreement aggregate (collision count over neighbours)
    #               but NOT from own-color + margin alone (the unary block).
    #   "unary"   : is_wrong = a function of the cell's OWN unary features only
    #               (own_extra > 0); no neighbour info needed.
    # The pairwise (B) block carries an explicit per-color "neighbour-has-this-
    # color" max aggregate AND a same-color collision count; the unary (A) block
    # gets own one-hot color + margin + own_extra.  This mirrors the real probe:
    # A = per-cell features; B = A + aggregated neighbour features.
    # ----------------------------------------------------------------------
    def build(regime, n_inst=400, V=14, ncol=3, seed=0):
        r = np.random.RandomState(seed)
        rows = []
        for inst in range(n_inst):
            color = r.randint(0, ncol, size=V)
            own_oh = np.eye(ncol)[color]                      # (V, ncol)
            own_extra = r.randn(V)                            # unary-only signal
            margin = 0.8 + 0.1 * r.randn(V)                  # all cells "confident"
            edges = [(i, i + 1) for i in range(V - 1)]
            # neighbour aggregation
            neigh_color_max = np.zeros((V, ncol))             # any neighbour has color c?
            collision = np.zeros(V)                           # # neighbours with same color
            deg = np.zeros(V)
            for (u, v) in edges:
                neigh_color_max[u] = np.maximum(neigh_color_max[u], own_oh[v])
                neigh_color_max[v] = np.maximum(neigh_color_max[v], own_oh[u])
                if color[u] == color[v]:
                    collision[u] += 1; collision[v] += 1
                deg[u] += 1; deg[v] += 1
            is_wrong = np.zeros(V, dtype=np.int64)
            if regime == "pairwise":
                is_wrong = (collision > 0).astype(np.int64)   # joint not-equal violation
            else:  # unary
                is_wrong = (own_extra > 0.0).astype(np.int64)
            for c in range(V):
                rows.append((inst, own_oh[c], own_extra[c], float(margin[c]),
                             neigh_color_max[c], float(collision[c]), int(is_wrong[c])))
        return rows

    def split_fit(rows, ncol=3):
        insts = sorted({rw[0] for rw in rows})
        rs = np.random.RandomState(7)
        rs.shuffle(insts)
        n_te = max(1, int(0.3 * len(insts)))
        te_set = set(insts[:n_te])

        def pack(in_test):
            sub = [rw for rw in rows if (rw[0] in te_set) == in_test]
            own_oh = np.array([rw[1] for rw in sub])                 # (n, ncol)
            own_extra = np.array([[rw[2]] for rw in sub])            # (n, 1)
            margin = np.array([[rw[3]] for rw in sub])               # (n, 1)
            ncm = np.array([rw[4] for rw in sub])                    # (n, ncol)
            coll = np.array([[rw[5]] for rw in sub])                 # (n, 1)
            y = np.array([rw[6] for rw in sub])
            return own_oh, own_extra, margin, ncm, coll, y

        oh_tr, oe_tr, m_tr, ncm_tr, co_tr, y_tr = pack(False)   # train = NOT in test
        oh_te, oe_te, m_te, ncm_te, co_te, y_te = pack(True)    # test  = in test
        # C belief-only: margin
        Xc_tr, Xc_te = m_tr, m_te
        # A unary: own one-hot + own_extra + margin (per-cell only)
        Xa_tr = np.concatenate([oh_tr, oe_tr, m_tr], axis=1)
        Xa_te = np.concatenate([oh_te, oe_te, m_te], axis=1)
        # B pairwise: A + neighbour-color-max + collision count
        Xb_tr = np.concatenate([oh_tr, oe_tr, m_tr, ncm_tr, co_tr], axis=1)
        Xb_te = np.concatenate([oh_te, oe_te, m_te, ncm_te, co_te], axis=1)
        hard = np.ones(len(y_te), dtype=bool)
        rC = fit_eval_probe(Xc_tr, y_tr, Xc_te, y_te, hard, seed=1)
        rA = fit_eval_probe(Xa_tr, y_tr, Xa_te, y_te, hard, seed=1)
        rB = fit_eval_probe(Xb_tr, y_tr, Xb_te, y_te, hard, seed=1)
        return rC, rA, rB

    # ---- regime 1: PAIRWISE-detectable ----
    rows_p = build("pairwise", seed=11)
    cP, aP, bP = split_fit(rows_p)
    print(f"  [pairwise regime]  C={cP['auc']:.3f}  A={aP['auc']:.3f}  "
          f"B={bP['auc']:.3f}", flush=True)
    assert bP["auc"] > aP["auc"] + 0.05, \
        f"PAIRWISE not detected: B={bP['auc']:.3f} !>> A={aP['auc']:.3f}"
    assert bP["auc"] > 0.65, f"pairwise B AUC too weak: {bP['auc']:.3f}"
    print("    [ok] harness DETECTS a pairwise advantage (B >> A)", flush=True)

    # ---- regime 2: UNARY-only control ----
    rows_u = build("unary", seed=22)
    cU, aU, bU = split_fit(rows_u)
    print(f"  [unary   regime]  C={cU['auc']:.3f}  A={aU['auc']:.3f}  "
          f"B={bU['auc']:.3f}", flush=True)
    assert aU["auc"] > 0.7, f"unary A AUC too weak: {aU['auc']:.3f}"
    assert abs(bU["auc"] - aU["auc"]) < 0.05, \
        f"UNARY control failed: B={bU['auc']:.3f} should ~ A={aU['auc']:.3f}"
    print("    [ok] harness reports A ~ B when signal is purely unary", flush=True)

    print("\nSELFTEST PASSED: harness can see a real B-vs-A difference both ways.",
          flush=True)
    return True


# ==========================================================================
# GPU extraction: model build / ckpt load (mirrors eval_coloring_bands.py)
# ==========================================================================
def cast_layers_fp32(model):
    from tinygrad import dtypes
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


_FG_PARAM_NAMES = [
    "fg_state_embed", "fg_position_embed", "fg_value_codebook",
    "fg_calib_head_w", "fg_calib_head_b", "fg_breath_embed", "fg_delta_gate",
]


def model_state_dict_fg(model) -> dict:
    sd = {"ln_f.g": model.ln_f_g, "ln_f.b": model.ln_f_b}
    sw = model.block.shared
    for a in ("wv", "bv", "wo", "bo", "w_out", "b_out",
              "in_ln_g", "in_ln_b", "post_ln_g", "post_ln_b"):
        sd[f"shared.{a}"] = getattr(sw, a)
    for i, layer in enumerate(model.block.layers):
        for a in ("wq", "bq", "wk", "bk", "w_in", "b_in"):
            sd[f"phase{i}.{a}"] = getattr(layer, a)
    for nm in _FG_PARAM_NAMES:
        sd[nm] = getattr(model, nm)
    return sd


def load_ckpt(model, path: str):
    from tinygrad.nn.state import safe_load
    sd = safe_load(path)
    targets = model_state_dict_fg(model)
    missing = []
    for name, dst in targets.items():
        if name not in sd:
            missing.append(name)
            continue
        src = sd[name].to(dst.device).realize()
        if src.shape != dst.shape:
            try:
                src = src.reshape(dst.shape)
            except Exception:
                missing.append(f"{name} (shape mismatch)")
                continue
        if src.dtype != dst.dtype:
            src = src.cast(dst.dtype)
        dst.assign(src).realize()
    if missing:
        print(f"  ckpt missing {len(missing)} keys: "
              f"{missing[:5]}{'...' if len(missing) > 5 else ''}", flush=True)
    else:
        print(f"  ckpt loaded cleanly ({len(targets)} keys).", flush=True)


# ==========================================================================
# Eager extraction forward — returns final-breath pre-readout residual (x_ln),
# logits, beliefs.  Body copied from factor_breathing_forward; the ONLY change
# is it also returns x_ln (the layernorm(x) that feeds the codebook readout,
# factor_graph_engine.py:384-386).  No new JIT graph path — eager batched.
# ==========================================================================
def extract_forward(model, batch, spec, K):
    from tinygrad import Tensor, dtypes
    from mycelium.factor_masks import (
        build_factor_attn_bias, build_factor_hyperbolic_attn_bias, FG_HYP_MASK,
    )
    from mycelium.kenken import kenken_layer_forward
    from mycelium.breathing import _layernorm
    from mycelium.factor_graph_engine import embed_factor_cells

    N = spec.n_values
    S = spec.s_max
    H = int(model.fg_state_embed.shape[-1])

    state_embed    = model.fg_state_embed
    position_embed = model.fg_position_embed
    breath_embed   = model.fg_breath_embed
    delta_gate     = model.fg_delta_gate
    value_codebook = model.fg_value_codebook

    input_cells       = batch.input_cells
    cell_valid        = batch.cell_valid
    value_domain_mask = batch.value_domain_mask
    membership        = batch.membership
    latent_type       = batch.latent_type

    B = int(input_cells.shape[0])

    if FG_HYP_MASK:
        attn_bias = build_factor_hyperbolic_attn_bias(
            model, membership, latent_type, cell_valid,
            spec.n_heads, spec.n_factor_types, S)
    else:
        attn_bias = build_factor_attn_bias(
            membership, latent_type, cell_valid,
            spec.n_heads, spec.n_factor_types, S)

    value_bias = (1.0 - value_domain_mask) * (-1e4)

    x = embed_factor_cells(input_cells, state_embed, position_embed, N)
    x = x.cast(dtypes.half) if x.dtype != dtypes.half else x

    layers = list(model.block.layers)
    inlet_h = Tensor.zeros((B, S, H), dtype=dtypes.float).cast(x.dtype)

    x_ln_final = None
    logits_final = None
    for k in range(K):
        be_k = breath_embed[k].reshape(1, 1, -1).cast(x.dtype)
        x_in = x + be_k + inlet_h
        x_pre = x
        h = x_in
        for layer in layers[:4]:
            h = kenken_layer_forward(layer, h, attn_bias)
        gate_k = delta_gate[k].cast(h.dtype).reshape(1, 1, 1)
        delta = h - x_pre
        x = x_pre + gate_k * delta
        x_ln = _layernorm(x, model.ln_f_g, model.ln_f_b,
                          model.cfg.layer_norm_eps).cast(dtypes.float)
        cell_logits_k = x_ln @ value_codebook.T.cast(dtypes.float)
        cell_logits_k = cell_logits_k + value_bias.cast(dtypes.float)
        x_ln_final = x_ln
        logits_final = cell_logits_k
    return logits_final, x_ln_final


# ==========================================================================
# Real run
# ==========================================================================
def main():
    import gc
    from tinygrad import Tensor, Device, dtypes
    from tinygrad.helpers import getenv

    CKPT = getenv("FG_CKPT",
                  ".cache/fg_ckpts/fg_coloring_k16/fg_coloring_k16_final.safetensors")
    K = int(getenv("FG_K_MAX", getenv("K", "16")))
    EVAL_BATCH = int(getenv("EVAL_BATCH", getenv("BATCH", "8")))
    SEED = int(getenv("SEED", "42"))
    N_INSTANCES = int(getenv("FG_N_INSTANCES", "8000"))
    S_MAX = int(getenv("FG_S_MAX", "49"))
    N_VALUES = int(getenv("FG_N_VALUES", "3"))
    HARD_DEPTH = int(getenv("HARD_DEPTH", "3"))    # deduction_depth >= 3 = hard band
    MAX_BATCHES = int(getenv("MAX_BATCHES", "0"))  # 0 = all test batches
    PROBE_L2 = float(getenv("PROBE_L2", "1.0"))
    PROBE_SEED = int(getenv("PROBE_SEED", "0"))
    MARGIN_HARD_FRAC = float(getenv("MARGIN_HARD_FRAC", "0.30"))  # bottom-30% margin = hard-subset

    print("=== probe_pairwise_confidence.py — pairwise vs unary readout probe ===",
          flush=True)
    print(f"device={Device.DEFAULT}  ckpt={CKPT}", flush=True)
    print(f"K={K}  EVAL_BATCH={EVAL_BATCH}  seed={SEED}  n_instances={N_INSTANCES}  "
          f"s_max={S_MAX}  k={N_VALUES}  hard_depth>={HARD_DEPTH}", flush=True)

    from mycelium import Config
    from mycelium.loader import _load_state, load_breathing
    from mycelium.factor_graph_engine import FactorGraphSpec, attach_factor_graph_params
    from mycelium.factor_masks import attach_factor_hyperbolic_params, FG_HYP_MASK
    from mycelium.graph_coloring_data import GraphColoringLoader

    spec = FactorGraphSpec(
        s_max=S_MAX, n_values=N_VALUES, n_factor_types=1,
        n_heads=16, k_max=K, has_factor_inlet=False)

    print("loading Pythia-410M -> BreathingTransformer...", flush=True)
    cfg = Config()
    sd = _load_state()
    model = load_breathing(cfg, sd=sd)
    del sd; gc.collect()
    cast_layers_fp32(model)
    attach_factor_graph_params(model, hidden=cfg.hidden, spec=spec)

    if FG_HYP_MASK:
        print("[FG_HYP_MASK=1] building coloring anchor tables ...", flush=True)
        _ref_loader = GraphColoringLoader(
            n_instances=N_INSTANCES, s_max=S_MAX, k_colors=N_VALUES,
            batch_size=EVAL_BATCH, seed=SEED)
        _ref_batch = _ref_loader.sample_batch()
        _mem_np = _ref_batch.membership.realize().numpy()
        _lt_np = _ref_batch.latent_type.realize().numpy()
        attach_factor_hyperbolic_params(
            model, n_heads=spec.n_heads, n_factor_types=spec.n_factor_types,
            s_max=spec.s_max, membership_np=_mem_np, latent_type_np=_lt_np)
        del _ref_loader, _ref_batch, _mem_np, _lt_np

    Device[Device.DEFAULT].synchronize()
    print(f"loading checkpoint: {CKPT}", flush=True)
    load_ckpt(model, CKPT)

    loader = GraphColoringLoader(
        n_instances=N_INSTANCES, s_max=S_MAX, k_colors=N_VALUES,
        batch_size=EVAL_BATCH, seed=SEED)
    n_test = len(loader.test_records)
    print(f"  test set: {n_test} instances", flush=True)

    Tensor.training = False

    # ---- per-cell accumulators (numpy, host) -------------------------------
    rec_inst = []        # instance id (global, per-cell)
    rec_resid = []       # (H,) pre-readout residual
    rec_belief = []      # (N,) softmax belief
    rec_argmax = []      # int
    rec_gold = []        # int (1..N)
    rec_depth = []       # int instance depth
    rec_band = []        # str
    rec_neigh_resid_mean = []   # (H,)
    rec_neigh_resid_max = []    # (H,)
    rec_neigh_belief_mean = []  # (N,)
    rec_neigh_belief_max = []   # (N,)

    inst_counter = 0
    n_cells_total = 0
    cell_ok_total = 0
    seen_keys = set()    # dedup the padded-repeat rows from iter_eval

    print("\nrunning batched extraction forwards...", flush=True)
    n_batches_done = 0
    for batch in loader.iter_eval(batch_size=EVAL_BATCH):
        logits_final, x_ln_final = extract_forward(model, batch, spec, K)
        # readout pieces
        probs = logits_final.softmax(axis=-1).realize().numpy()      # (B,S,N)
        resid = x_ln_final.realize().numpy()                          # (B,S,H)
        cell_valid_np = batch.cell_valid.realize().numpy()            # (B,S)
        gold_np = batch.gold.realize().numpy().astype(np.int32)       # (B,S)
        argmax_np = (logits_final.argmax(axis=-1) + 1).realize().numpy().astype(np.int32)  # (B,S) 1..N
        mem_np = batch.membership.realize().numpy()                   # (B,L,S)
        depth_meta = batch.deduction_depth
        band_meta = batch.band
        n_meta = batch.n

        B_cur = cell_valid_np.shape[0]
        for b in range(B_cur):
            valid = cell_valid_np[b] > 0.5
            nv = int(valid.sum())
            if nv == 0:
                continue
            # dedup the iter_eval padded-repeat instances (last batch repeats recs[0]).
            # key = (n, depth, gold-of-valid-cells fingerprint)
            gv = gold_np[b][valid]
            key = (int(n_meta[b]), int(depth_meta[b]),
                   hash(gv.tobytes()), hash(mem_np[b].tobytes()))
            if key in seen_keys:
                continue
            seen_keys.add(key)

            # adjacency from membership: edge rows have exactly two 1s (u,v).
            mb = mem_np[b]                       # (L, S)
            # neighbours of each valid vertex
            neighbours = [[] for _ in range(S_MAX)]
            for row in mb:
                idx = np.where(row > 0.5)[0]
                if len(idx) == 2:
                    u, v = int(idx[0]), int(idx[1])
                    neighbours[u].append(v)
                    neighbours[v].append(u)

            depth = int(depth_meta[b])
            band = band_meta[b]
            inst_id = inst_counter
            inst_counter += 1

            for v in range(S_MAX):
                if not valid[v]:
                    continue
                nbrs = [w for w in neighbours[v] if valid[w]]
                # aggregate neighbour residuals/beliefs
                if nbrs:
                    nr = resid[b][nbrs]          # (deg, H)
                    nb_belief = probs[b][nbrs]   # (deg, N)
                    nr_mean = nr.mean(axis=0)
                    nr_max = nr.max(axis=0)
                    nbel_mean = nb_belief.mean(axis=0)
                    nbel_max = nb_belief.max(axis=0)
                else:
                    nr_mean = np.zeros(resid.shape[-1], dtype=np.float32)
                    nr_max = np.zeros(resid.shape[-1], dtype=np.float32)
                    nbel_mean = np.zeros(N_VALUES, dtype=np.float32)
                    nbel_max = np.zeros(N_VALUES, dtype=np.float32)

                rec_inst.append(inst_id)
                rec_resid.append(resid[b][v].astype(np.float32))
                rec_belief.append(probs[b][v].astype(np.float32))
                rec_argmax.append(int(argmax_np[b][v]))
                rec_gold.append(int(gold_np[b][v]))
                rec_depth.append(depth)
                rec_band.append(band)
                rec_neigh_resid_mean.append(nr_mean.astype(np.float32))
                rec_neigh_resid_max.append(nr_max.astype(np.float32))
                rec_neigh_belief_mean.append(nbel_mean.astype(np.float32))
                rec_neigh_belief_max.append(nbel_max.astype(np.float32))

                n_cells_total += 1
                if int(argmax_np[b][v]) == int(gold_np[b][v]):
                    cell_ok_total += 1

        n_batches_done += 1
        if n_batches_done % 20 == 0:
            print(f"  [{n_batches_done} batches] cells={n_cells_total} "
                  f"insts={inst_counter} cell_acc={cell_ok_total/max(n_cells_total,1):.3f}",
                  flush=True)
        if MAX_BATCHES and n_batches_done >= MAX_BATCHES:
            break

    # ---- to arrays ----------------------------------------------------------
    inst = np.array(rec_inst, dtype=np.int64)
    resid = np.array(rec_resid, dtype=np.float32)
    belief = np.array(rec_belief, dtype=np.float32)
    argmax = np.array(rec_argmax, dtype=np.int32)
    gold = np.array(rec_gold, dtype=np.int32)
    depth = np.array(rec_depth, dtype=np.int32)
    band = np.array(rec_band)
    nr_mean = np.array(rec_neigh_resid_mean, dtype=np.float32)
    nr_max = np.array(rec_neigh_resid_max, dtype=np.float32)
    nbel_mean = np.array(rec_neigh_belief_mean, dtype=np.float32)
    nbel_max = np.array(rec_neigh_belief_max, dtype=np.float32)

    is_wrong = (argmax != gold).astype(np.int64)
    n_total = len(is_wrong)
    cell_acc = cell_ok_total / max(n_cells_total, 1)
    print(f"\n  extracted cells={n_total}  instances={inst_counter}  "
          f"frac_wrong={is_wrong.mean():.3f}", flush=True)

    # ---- belief features ----------------------------------------------------
    sb = np.sort(belief, axis=1)              # ascending
    max_prob = sb[:, -1]
    margin = sb[:, -1] - sb[:, -2]
    eps = 1e-9
    entropy = -(belief * np.log(belief + eps)).sum(axis=1)
    belief_feats = np.stack([max_prob, margin, entropy], axis=1)   # (n,3)

    # ---- SANITY: cell_acc vs gold (validates the is_wrong label) ------------
    # hard band cell_acc
    hard_inst_mask = depth >= HARD_DEPTH
    easy_inst_mask = depth < HARD_DEPTH
    ca_hard = (argmax[hard_inst_mask] == gold[hard_inst_mask]).mean() if hard_inst_mask.sum() else float("nan")
    ca_easy = (argmax[easy_inst_mask] == gold[easy_inst_mask]).mean() if easy_inst_mask.sum() else float("nan")
    print(f"\n  [SANITY] overall cell_acc={cell_acc:.4f}  "
          f"hard(d>={HARD_DEPTH}) cell_acc={ca_hard:.4f} (n={int(hard_inst_mask.sum())})  "
          f"easy cell_acc={ca_easy:.4f} (n={int(easy_inst_mask.sum())})", flush=True)
    by_band = {}
    for bnd in sorted(set(band.tolist())):
        m = band == bnd
        by_band[bnd] = float((argmax[m] == gold[m]).mean())
    print(f"  [SANITY] cell_acc by band: "
          f"{ {k: round(v,3) for k,v in by_band.items()} }", flush=True)
    by_depth = {}
    for d in sorted(set(depth.tolist())):
        m = depth == d
        by_depth[int(d)] = (float((argmax[m] == gold[m]).mean()), int(m.sum()))
    print(f"  [SANITY] cell_acc by depth (acc,ncells): {by_depth}", flush=True)

    # ---- instance-level split (all cells of an instance -> same split) ------
    rs = np.random.RandomState(SEED)
    uniq = np.unique(inst)
    rs.shuffle(uniq)
    n_te = max(1, int(round(0.30 * len(uniq))))
    te_insts = set(uniq[:n_te].tolist())
    test_mask = np.array([i in te_insts for i in inst])
    train_mask = ~test_mask
    print(f"\n  instance-level split: train cells={int(train_mask.sum())} "
          f"test cells={int(test_mask.sum())} "
          f"(train insts={len(uniq)-n_te} / test insts={n_te})", flush=True)
    print(f"  band mix (test cells): "
          f"{ {k:int((band[test_mask]==k).sum()) for k in sorted(set(band.tolist()))} }",
          flush=True)

    # ---- hard/ambiguous cell subset on TEST: bottom-MARGIN_HARD_FRAC margin --
    te_margin = margin[test_mask]
    if len(te_margin) > 0:
        thr = np.quantile(te_margin, MARGIN_HARD_FRAC)
        hard_cell_mask_te = te_margin <= thr
    else:
        hard_cell_mask_te = np.zeros(0, dtype=bool)
    # ALSO a hard-band (depth>=HARD_DEPTH) test subset
    hard_band_mask_te = depth[test_mask] >= HARD_DEPTH

    y_tr = is_wrong[train_mask]
    y_te = is_wrong[test_mask]
    print(f"  test frac_wrong={y_te.mean():.3f}  "
          f"hard-margin-subset n={int(hard_cell_mask_te.sum())} "
          f"frac_wrong={is_wrong[test_mask][hard_cell_mask_te].mean() if hard_cell_mask_te.sum() else float('nan'):.3f}",
          flush=True)

    # ---- feature blocks -----------------------------------------------------
    # The residual is 1024-dim; with O(10k) cells a raw 1024/3072-dim logreg
    # OVERFITS (smoke showed test-AUC<C + ECE~0.46).  So the PRIMARY A/B probes
    # PCA-reduce the residual blocks to a fixed, well-conditioned dim (basis fit
    # on TRAIN only — no leakage), and crucially A and B get the SAME residual
    # PCA dim, so a B>A win is structural (neighbour aggregates) not extra params.
    # The raw high-dim probes are reported as a SECONDARY diagnostic.
    RES_PCA = int(getenv("RES_PCA_DIM", "64"))   # residual block reduced to this

    # C: belief-only (raw 3-dim)
    Xc_tr = belief_feats[train_mask]; Xc_te = belief_feats[test_mask]

    # reduce the cell residual block (shared component of A and B)
    res_tr_p, res_te_p = pca_reduce_block(resid[train_mask], resid[test_mask],
                                          RES_PCA, seed=PROBE_SEED)
    # reduce the neighbour-residual blocks (B-only) — each to RES_PCA
    nrm_tr_p, nrm_te_p = pca_reduce_block(nr_mean[train_mask], nr_mean[test_mask],
                                          RES_PCA, seed=PROBE_SEED)
    nrx_tr_p, nrx_te_p = pca_reduce_block(nr_max[train_mask], nr_max[test_mask],
                                          RES_PCA, seed=PROBE_SEED)

    # A (primary): reduced cell residual + belief
    Xa_tr = np.concatenate([res_tr_p, belief_feats[train_mask]], axis=1)
    Xa_te = np.concatenate([res_te_p, belief_feats[test_mask]], axis=1)
    # B (primary): reduced cell residual + reduced neigh-residual(mean,max)
    #   + neigh-belief(mean,max) [raw 2N] + belief
    Xb_tr = np.concatenate([res_tr_p, nrm_tr_p, nrx_tr_p,
                            nbel_mean[train_mask], nbel_max[train_mask],
                            belief_feats[train_mask]], axis=1)
    Xb_te = np.concatenate([res_te_p, nrm_te_p, nrx_te_p,
                            nbel_mean[test_mask], nbel_max[test_mask],
                            belief_feats[test_mask]], axis=1)

    print(f"\n  PRIMARY feature dims (residual PCA={RES_PCA}): "
          f"C={Xc_tr.shape[1]}  A={Xa_tr.shape[1]}  B={Xb_tr.shape[1]}", flush=True)

    def report(tag, r):
        print(f"  {tag:30s} AUC={r['auc']:.4f}  AUC_hardMargin={r['auc_hard']:.4f}  "
              f"ECE={r['ece']:.4f}  (d={r['n_feat']})", flush=True)

    print("\n--- PRIMARY PROBES (PCA-reduced residual, instance-split, same L2/seed) ---",
          flush=True)
    rC = fit_eval_probe(Xc_tr, y_tr, Xc_te, y_te, hard_cell_mask_te,
                        l2=PROBE_L2, seed=PROBE_SEED)
    rA = fit_eval_probe(Xa_tr, y_tr, Xa_te, y_te, hard_cell_mask_te,
                        l2=PROBE_L2, seed=PROBE_SEED)
    rB = fit_eval_probe(Xb_tr, y_tr, Xb_te, y_te, hard_cell_mask_te,
                        l2=PROBE_L2, seed=PROBE_SEED)
    report("C belief-only", rC)
    report("A unary-residual", rA)
    report("B pairwise-residual", rB)

    # ---- hard-BAND subset AUC (depth>=HARD_DEPTH) --------------------------
    def auc_on(mask, r):
        if mask.sum() < 10:
            return float("nan")
        return auc_score(y_te[mask], r["p"][mask])
    print("\n  AUC on HARD-BAND subset (depth>=%d, test cells=%d):"
          % (HARD_DEPTH, int(hard_band_mask_te.sum())), flush=True)
    aucC_hb = auc_on(hard_band_mask_te, rC)
    aucA_hb = auc_on(hard_band_mask_te, rA)
    aucB_hb = auc_on(hard_band_mask_te, rB)
    print(f"    C={aucC_hb:.4f}  A={aucA_hb:.4f}  B={aucB_hb:.4f}", flush=True)

    # ---- SECONDARY: raw high-dim residual probes (strong L2) ---------------
    # The full 1024-dim residual + raw neighbour aggregates, with heavier L2.
    # Reported to confirm the PCA reduction isn't hiding signal.
    RAW_L2 = float(getenv("RAW_L2", "20.0"))
    Xa_raw_tr = np.concatenate([resid[train_mask], belief_feats[train_mask]], axis=1)
    Xa_raw_te = np.concatenate([resid[test_mask], belief_feats[test_mask]], axis=1)
    Xb_raw_tr = np.concatenate([resid[train_mask], nr_mean[train_mask], nr_max[train_mask],
                                nbel_mean[train_mask], nbel_max[train_mask],
                                belief_feats[train_mask]], axis=1)
    Xb_raw_te = np.concatenate([resid[test_mask], nr_mean[test_mask], nr_max[test_mask],
                                nbel_mean[test_mask], nbel_max[test_mask],
                                belief_feats[test_mask]], axis=1)
    rA_raw = fit_eval_probe(Xa_raw_tr, y_tr, Xa_raw_te, y_te, hard_cell_mask_te,
                            l2=RAW_L2, seed=PROBE_SEED)
    rB_raw = fit_eval_probe(Xb_raw_tr, y_tr, Xb_raw_te, y_te, hard_cell_mask_te,
                            l2=RAW_L2, seed=PROBE_SEED)
    print(f"\n  SECONDARY raw high-dim probes (L2={RAW_L2}): "
          f"A_raw(d={rA_raw['n_feat']}) AUC={rA_raw['auc']:.4f}  "
          f"B_raw(d={rB_raw['n_feat']}) AUC={rB_raw['auc']:.4f}  "
          f"(B-A={rB_raw['auc']-rA_raw['auc']:+.4f})", flush=True)

    # ---- capacity-matched comparison: reduce A and B to common TOTAL dim ----
    # Reduce the FULL A and B feature matrices to one common PCA dim so a B>A
    # win is not just "more parameters".
    common_dim = int(getenv("PCA_DIM", "32"))
    a_cm_tr, a_cm_te = pca_reduce_block(Xa_tr, Xa_te, common_dim, seed=PROBE_SEED)
    b_cm_tr, b_cm_te = pca_reduce_block(Xb_tr, Xb_te, common_dim, seed=PROBE_SEED)
    rA_pca = fit_eval_probe(a_cm_tr, y_tr, a_cm_te, y_te, hard_cell_mask_te,
                            l2=PROBE_L2, seed=PROBE_SEED)
    rB_pca = fit_eval_probe(b_cm_tr, y_tr, b_cm_te, y_te, hard_cell_mask_te,
                            l2=PROBE_L2, seed=PROBE_SEED)
    print(f"\n  CAPACITY-MATCHED (common PCA dim={common_dim}):", flush=True)
    print(f"    A_pca AUC={rA_pca['auc']:.4f}  AUC_hardMargin={rA_pca['auc_hard']:.4f}",
          flush=True)
    print(f"    B_pca AUC={rB_pca['auc']:.4f}  AUC_hardMargin={rB_pca['auc_hard']:.4f}",
          flush=True)

    # ---- VERDICT -----------------------------------------------------------
    BA = rB["auc"] - rA["auc"]
    BC = rB["auc"] - rC["auc"]
    AC = rA["auc"] - rC["auc"]
    BA_pca = rB_pca["auc"] - rA_pca["auc"]
    BA_hard = (rB["auc_hard"] - rA["auc_hard"])
    MARGIN_THRESH = 0.02       # "meaningful" B-over-A margin
    LIFT_THRESH = 0.03         # residual must clearly beat belief-only

    print(f"\n{'='*64}", flush=True)
    print("  VERDICT", flush=True)
    print(f"{'='*64}", flush=True)
    print(f"  B-A (overall)={BA:+.4f}  B-A (capacity-matched)={BA_pca:+.4f}  "
          f"B-A (hard-margin)={BA_hard:+.4f}", flush=True)
    print(f"  A-C (residual lift over belief)={AC:+.4f}  "
          f"B-C={BC:+.4f}", flush=True)

    pairwise_wins = (BA > MARGIN_THRESH) and (BA_pca > MARGIN_THRESH)
    residual_lifts = (AC > LIFT_THRESH) or (BC > LIFT_THRESH)
    # DEEPER-PROBLEM signature: NEITHER readout recovers signal the 3-dim belief
    # lacks (A~C AND B~C, both within LIFT_THRESH).  This is the load-bearing
    # diagnosis regardless of the ABSOLUTE AUC level — the belief baseline may sit
    # at ~0.70 and the residual still adds nothing.  (An absolute floor <0.60 is a
    # SUFFICIENT but not NECESSARY condition.)
    no_lift = (AC <= LIFT_THRESH) and (BC <= LIFT_THRESH)

    if pairwise_wins and (BC > LIFT_THRESH):
        verdict = "BUILD-BETHE"
        reason = (f"B beats A by {BA:+.3f} overall ({BA_pca:+.3f} capacity-matched, "
                  f"{BA_hard:+.3f} hard-margin) AND beats C by {BC:+.3f}; explicit "
                  f"pairwise structure adds joint-consistency signal NOT in the per-cell "
                  f"residual.")
    elif no_lift:
        verdict = "DEEPER-PROBLEM"
        reason = (f"A~B~C (A={rA['auc']:.3f} B={rB['auc']:.3f} C={rC['auc']:.3f}); the "
                  f"residual adds NOTHING over the 3-dim belief (A-C={AC:+.3f}) and "
                  f"explicit pairwise aggregation adds nothing either (B-C={BC:+.3f}, "
                  f"B-A capacity-matched={BA_pca:+.3f}). Neither a richer unary nor a "
                  f"pairwise/Bethe readout helps — the joint-consistency signal is not "
                  f"recoverable from the frozen residual; the limit is upstream "
                  f"(training/representation), the readout architecture is NOT the gate.")
    elif residual_lifts and (abs(BA) <= MARGIN_THRESH or not pairwise_wins):
        verdict = "CHEAPER-UNARY-FIX"
        reason = (f"A~B (B-A={BA:+.3f}, capacity-matched {BA_pca:+.3f}) but both beat C "
                  f"(A-C={AC:+.3f}, B-C={BC:+.3f}); the K-breath residual already carries "
                  f"the joint signal — a richer non-product unary readout / better "
                  f"calibration suffices, pairwise readout not needed.")
    else:
        verdict = "AMBIGUOUS"
        lean = "toward BUILD-BETHE" if BA > 0 else "toward CHEAPER-UNARY-FIX"
        reason = (f"mixed (B-A={BA:+.3f}, A-C={AC:+.3f}, B-C={BC:+.3f}); leans {lean}. "
                  f"Deciding factor: whether B-A survives capacity matching "
                  f"(got {BA_pca:+.3f}) with more hard instances.")
    print(f"\n  >>> {verdict}", flush=True)
    print(f"  {reason}", flush=True)

    # ---- machine-readable summary line -------------------------------------
    print("\n[RESULT_JSON] " + repr({
        "n_cells_total": int(n_total),
        "n_instances": int(inst_counter),
        "n_test_cells": int(test_mask.sum()),
        "n_train_cells": int(train_mask.sum()),
        "cell_acc": round(float(cell_acc), 4),
        "cell_acc_hard": round(float(ca_hard), 4),
        "cell_acc_easy": round(float(ca_easy), 4),
        "frac_wrong": round(float(is_wrong.mean()), 4),
        "auc_C": round(float(rC["auc"]), 4),
        "auc_A": round(float(rA["auc"]), 4),
        "auc_B": round(float(rB["auc"]), 4),
        "auc_C_hardMargin": round(float(rC["auc_hard"]), 4),
        "auc_A_hardMargin": round(float(rA["auc_hard"]), 4),
        "auc_B_hardMargin": round(float(rB["auc_hard"]), 4),
        "auc_C_hardBand": round(float(aucC_hb), 4),
        "auc_A_hardBand": round(float(aucA_hb), 4),
        "auc_B_hardBand": round(float(aucB_hb), 4),
        "auc_A_pca": round(float(rA_pca["auc"]), 4),
        "auc_B_pca": round(float(rB_pca["auc"]), 4),
        "auc_A_raw": round(float(rA_raw["auc"]), 4),
        "auc_B_raw": round(float(rB_raw["auc"]), 4),
        "ece_C": round(float(rC["ece"]), 4),
        "ece_A": round(float(rA["ece"]), 4),
        "ece_B": round(float(rB["ece"]), 4),
        "B_minus_A": round(float(BA), 4),
        "B_minus_A_pca": round(float(BA_pca), 4),
        "verdict": verdict,
        "by_band_cell_acc": {k: round(v, 4) for k, v in by_band.items()},
    }), flush=True)


# ==========================================================================
if __name__ == "__main__":
    ok = _ast_parse_ok()
    print(f"[ast.parse] astparse_ok={ok}", flush=True)
    if not ok:
        sys.exit(1)
    if int(os.environ.get("SELFTEST_ONLY", "0")) > 0:
        run_selftest()
        sys.exit(0)
    main()
