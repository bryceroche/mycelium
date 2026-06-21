"""probe_svd_collapse.py — Representation collapse / multicollinearity baseline.

Measures rank-collapse and redundancy in the FROZEN single-domain deducer
checkpoints, BEFORE the multi-task general-weights run.  This is the BASELINE
lens for the general-weights experiment: it tells us whether any decorrelation
machinery (codebook-orthogonality penalty, etc.) is needed for the multi-task
run, and gives the numbers to compare a future multi-task checkpoint against.

NON-INVASIVE.  This script NEVER edits mycelium/factor_graph_engine.py or
mycelium/kenken.py.  All intermediate capture is via monkeypatch INSIDE this
file:
  * Per-breath cell representations: monkeypatch mycelium.breathing._layernorm,
    capture ONLY the readout LayerNorm (identified by gamma IS model.ln_f_g —
    the engine's readout uses model.ln_f_g/.ln_f_b; the per-layer LNs use
    layer.shared.in_ln_g/.post_ln_g, so the readout call is unambiguous).
  * Per-head attention OUTPUTS: monkeypatch
    factor_graph_engine.kenken_layer_forward (the engine imported the name into
    its own namespace at module load, so patching there intercepts the call).
    The wrapper recomputes the per-head context EXACTLY as kenken_layer_forward
    does, records it, then calls the original for the true forward (no drift in
    the returned graph).

THE THREE MEASUREMENTS (per checkpoint):
  1. CODEBOOK COLLINEARITY (no forward) — value_codebook (N, H): off-diagonal
     cosine gram (row-normalized), max+mean |off-diag cosine|, and the
     effective rank via singular-value entropy.  This is what the validated
     codebook-orthogonality penalty targets.
  2. BREATH RANK / COLLAPSE — per breath k (0..K-1) over a sample of instances'
     VALID cells: effective rank of the stacked valid-cell readout-LN reps, plus
     the mean consecutive-breath cosine (breath k vs k+1).  Does rank collapse
     and does consecutive-breath cosine -> 1 (static)?
  3. WITHIN-GROUP HEAD REDUNDANCY — per relation-group of same-mask heads (from
     cell_mp_head_allocation): effective rank of the stacked per-head context
     vectors, plus the mean pairwise cosine within the group.  Are same-relation
     heads diverse or redundant?

SELFTEST (CPU): SELFTEST_ONLY=1 sanity-checks the effective-rank + cosine-gram
machinery on synthetic matrices (full-rank Gaussian, rank-1, near-collinear)
before trusting GPU numbers.  No GPU / no checkpoint needed.

USAGE:
  CPU selftest (GPU-free):
    SELFTEST_ONLY=1 .venv/bin/python3 scripts/probe_svd_collapse.py

  GPU run on the frozen checkpoints (AMD):
    DEV=AMD .venv/bin/python3 scripts/probe_svd_collapse.py

  Single checkpoint:
    DEV=AMD PROBE_ONLY=coloring .venv/bin/python3 scripts/probe_svd_collapse.py

Env vars:
  SELFTEST_ONLY   1 -> run only the CPU selftest and exit (default 0).
  PROBE_ONLY      coloring | circuit | circuit_deep | "" (all) — default "".
  PROBE_N_INST    instances to sample for the breath/head measurements (default 64).
  PROBE_BATCH     eval batch size (default 8).
  K               breaths (default 16; the trained K_max).
  PROBE_HEAD_LAYER which transformer layer (0..3) to capture per-head ctx (default 3).
  PROBE_HEAD_BREATH which breath to capture per-head ctx at (default K-1, last).
"""
from __future__ import annotations

import ast
import os
import sys

_THIS_FILE = os.path.abspath(__file__)
_ROOT = os.path.dirname(os.path.dirname(_THIS_FILE))
sys.path.insert(0, _ROOT)

import numpy as np


# ===========================================================================
# ast.parse gate — always runs, even on CPU
# ===========================================================================

def _ast_parse_ok() -> bool:
    with open(_THIS_FILE) as f:
        src = f.read()
    try:
        ast.parse(src)
        return True
    except SyntaxError as e:
        print(f"[ast.parse] FAILED: {e}", flush=True)
        return False


# ===========================================================================
# Core numerical machinery (pure numpy — what the SELFTEST validates)
# ===========================================================================

def effective_rank(mat: np.ndarray) -> tuple[float, int, np.ndarray]:
    """Singular-value-entropy effective rank of a 2D matrix.

    eff_rank = exp(H(p)) where p_i = sigma_i / sum(sigma) is the normalized
    singular-value distribution and H is Shannon entropy in nats.  This is the
    standard "stable rank via spectral entropy" (Roy & Vetterli 2007):
      - full-rank well-conditioned matrix -> eff_rank ~ min(dims)
      - rank-1 matrix                     -> eff_rank ~ 1
      - near-collinear rows               -> eff_rank << min(dims)

    Returns (eff_rank, hard_rank, singular_values).
    `hard_rank` is the count of singular values > 1e-7 * sigma_max (a sanity
    companion, not the headline number).
    """
    mat = np.asarray(mat, dtype=np.float64)
    if mat.ndim != 2:
        raise ValueError(f"effective_rank expects 2D, got {mat.shape}")
    if mat.size == 0 or min(mat.shape) == 0:
        return 0.0, 0, np.array([])
    # SVD on the (n, d) matrix; singular values are non-negative, descending.
    sv = np.linalg.svd(mat, compute_uv=False)
    sv = sv[sv > 0]
    if sv.size == 0:
        return 0.0, 0, sv
    total = sv.sum()
    p = sv / total
    # entropy in nats; guard p=0 (already removed) — use where for safety.
    ent = -np.sum(p * np.log(p))
    eff = float(np.exp(ent))
    hard = int(np.sum(sv > 1e-7 * sv[0]))
    return eff, hard, sv


def cosine_gram(rows: np.ndarray) -> np.ndarray:
    """Row-normalized cosine gram matrix of `rows` (n, d) -> (n, n)."""
    rows = np.asarray(rows, dtype=np.float64)
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    unit = rows / norms
    return unit @ unit.T


def offdiag_cosine_stats(rows: np.ndarray) -> tuple[float, float, np.ndarray]:
    """(max|off-diag|, mean|off-diag|, gram) over the row cosine gram.

    For n<2 rows there is no off-diagonal -> returns (0.0, 0.0, gram).
    """
    gram = cosine_gram(rows)
    n = gram.shape[0]
    if n < 2:
        return 0.0, 0.0, gram
    iu = np.triu_indices(n, k=1)
    off = np.abs(gram[iu])
    return float(off.max()), float(off.mean()), gram


def mean_pairwise_cosine(rows: np.ndarray) -> float:
    """Mean (signed) pairwise cosine over the strict upper triangle."""
    gram = cosine_gram(rows)
    n = gram.shape[0]
    if n < 2:
        return 1.0
    iu = np.triu_indices(n, k=1)
    return float(gram[iu].mean())


def consecutive_cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Mean per-row cosine between paired row sets a, b (both (n, d))."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    na = np.linalg.norm(a, axis=1)
    nb = np.linalg.norm(b, axis=1)
    denom = na * nb
    dots = np.sum(a * b, axis=1)
    good = denom > 1e-12
    if not np.any(good):
        return 1.0
    return float(np.mean(dots[good] / denom[good]))


# ===========================================================================
# SELFTEST (CPU)
# ===========================================================================

def selftest() -> bool:
    print("=== probe_svd_collapse SELFTEST (CPU) ===", flush=True)
    rng = np.random.RandomState(0)
    ok = True

    # 1. Full-rank Gaussian (n=64, d=64) -> eff_rank near min(dims)=64.
    g = rng.randn(64, 64)
    eff_g, hard_g, _ = effective_rank(g)
    print(f"  [full-rank 64x64]   eff_rank={eff_g:.2f}  hard_rank={hard_g} "
          f"(expect eff in ~[40,64], hard=64)", flush=True)
    cond1 = (40.0 <= eff_g <= 64.0) and hard_g == 64
    ok &= cond1

    # 1b. Tall full-rank Gaussian (n=400, d=64) -> eff near 64 (min dim).
    gt = rng.randn(400, 64)
    eff_gt, hard_gt, _ = effective_rank(gt)
    print(f"  [tall 400x64]       eff_rank={eff_gt:.2f}  hard_rank={hard_gt} "
          f"(expect eff in ~[45,64], hard=64)", flush=True)
    cond1b = (45.0 <= eff_gt <= 64.0) and hard_gt == 64
    ok &= cond1b

    # 2. Rank-1 matrix (outer product) -> eff_rank ~ 1, hard_rank = 1.
    u = rng.randn(50, 1)
    w = rng.randn(1, 64)
    r1 = u @ w
    eff_r1, hard_r1, _ = effective_rank(r1)
    print(f"  [rank-1 50x64]      eff_rank={eff_r1:.4f}  hard_rank={hard_r1} "
          f"(expect eff~1.0, hard=1)", flush=True)
    cond2 = abs(eff_r1 - 1.0) < 1e-3 and hard_r1 == 1
    ok &= cond2

    # 2b. cosine: rank-1 rows are all collinear -> mean |off-diag cos| ~ 1.
    # NOTE: u_i can flip sign, so the SIGNED mean pairwise cos averages to ~0;
    # collinearity is captured by the ABSOLUTE off-diagonal cosine (==1 here).
    _, mnabs_r1, _ = offdiag_cosine_stats(r1)
    mpc_r1 = mean_pairwise_cosine(r1)
    print(f"  [rank-1 cosine]     mean|offdiag_cos|={mnabs_r1:.4f}  "
          f"(signed mean={mpc_r1:+.4f}; expect |off|~1.0)", flush=True)
    cond2b = abs(mnabs_r1 - 1.0) < 1e-6
    ok &= cond2b

    # 3. Near-collinear set: rows = base + small noise -> low eff_rank, high cos.
    base = rng.randn(1, 64)
    near = np.tile(base, (40, 1)) + 0.01 * rng.randn(40, 64)
    eff_n, hard_n, _ = effective_rank(near)
    mxc, mnc, _ = offdiag_cosine_stats(near)
    print(f"  [near-collinear]    eff_rank={eff_n:.3f}  max|offdiag_cos|={mxc:.4f}"
          f"  mean|offdiag_cos|={mnc:.4f} (expect eff~1-2, cos>0.99)", flush=True)
    cond3 = eff_n < 2.5 and mnc > 0.99
    ok &= cond3

    # 4. Orthonormal rows -> eff_rank = n exactly, off-diag cosine ~ 0.
    q, _ = np.linalg.qr(rng.randn(64, 64))
    orth = q[:8]  # 8 orthonormal rows in R^64
    eff_o, hard_o, _ = effective_rank(orth)
    mxo, mno, _ = offdiag_cosine_stats(orth)
    print(f"  [orthonormal 8x64]  eff_rank={eff_o:.4f}  max|offdiag_cos|={mxo:.2e}"
          f"  (expect eff~8.0, cos~0)", flush=True)
    cond4 = abs(eff_o - 8.0) < 1e-3 and mxo < 1e-6
    ok &= cond4

    # 5. consecutive_cosine: identical -> 1.0; orthogonal -> ~0.
    cc_same = consecutive_cosine(near, near)
    cc_orth = consecutive_cosine(q[:8], q[8:16])
    print(f"  [consec same]       cos={cc_same:.6f} (expect 1.0)", flush=True)
    print(f"  [consec orth]       cos={cc_orth:.4f} (expect ~0)", flush=True)
    cond5 = abs(cc_same - 1.0) < 1e-6 and abs(cc_orth) < 1e-6
    ok &= cond5

    print(f"\n  SELFTEST {'PASSED' if ok else 'FAILED'}", flush=True)
    return ok


# ===========================================================================
# Verdict helpers
# ===========================================================================

def _eff_rank_frac(eff: float, max_rank: int) -> float:
    return eff / max(max_rank, 1)


def _verdict_codebook(eff: float, n: int, h: int, mean_off: float) -> str:
    """Codebook collapse verdict.  max_rank = min(N, H) = N (N<<H here)."""
    max_rank = min(n, h)
    frac = _eff_rank_frac(eff, max_rank)
    # collapsed: eff rank < 25% of max OR mean |off-diag cos| > 0.9.
    if frac < 0.25 or mean_off > 0.9:
        return f"COLLAPSED (eff_rank={eff:.2f}/{max_rank} = {frac:.0%}, mean|offcos|={mean_off:.3f})"
    return f"HEALTHY (eff_rank={eff:.2f}/{max_rank} = {frac:.0%}, mean|offcos|={mean_off:.3f})"


def _verdict_breath(eff_ranks: list[float], max_rank: int,
                    consec: list[float]) -> str:
    """Breath collapse verdict over the K-breath trajectory."""
    if not eff_ranks:
        return "NO DATA"
    first = eff_ranks[0]
    last = eff_ranks[-1]
    mx = max(eff_ranks)
    mn = min(eff_ranks)
    drop_frac = (mx - last) / max(mx, 1e-6)
    last_frac = _eff_rank_frac(last, max_rank)
    mean_consec = float(np.mean(consec)) if consec else 0.0
    max_consec = float(np.max(consec)) if consec else 0.0
    # collapsed: eff rank drops >50% across breaths AND ends < 25% of max,
    #            OR consecutive-breath cosine pins > 0.98 (static / fixed point).
    rank_collapsed = drop_frac > 0.50 and last_frac < 0.25
    static = mean_consec > 0.98
    if rank_collapsed or static:
        why = []
        if rank_collapsed:
            why.append(f"eff_rank dropped {drop_frac:.0%} (max {mx:.1f} -> last {last:.1f}, "
                       f"{last_frac:.0%} of {max_rank})")
        if static:
            why.append(f"mean consec-breath cos={mean_consec:.3f} > 0.98 (static)")
        return "COLLAPSED (" + "; ".join(why) + ")"
    return (f"HEALTHY (eff_rank range [{mn:.1f},{mx:.1f}], last {last:.1f}={last_frac:.0%} "
            f"of {max_rank}; mean consec-breath cos={mean_consec:.3f}, max={max_consec:.3f})")


def _verdict_head_group(eff: float, n_heads: int, head_dim: int,
                        mean_cos: float) -> str:
    """Within-group head redundancy verdict.  max_rank = min(n_heads, head_dim)."""
    max_rank = min(n_heads, head_dim)
    frac = _eff_rank_frac(eff, max_rank)
    # redundant: eff rank < 25% of max OR mean pairwise cos > 0.9.
    if frac < 0.25 or mean_cos > 0.9:
        return f"REDUNDANT (eff_rank={eff:.2f}/{max_rank} = {frac:.0%}, mean_cos={mean_cos:.3f})"
    return f"HEALTHY (eff_rank={eff:.2f}/{max_rank} = {frac:.0%}, mean_cos={mean_cos:.3f})"


# ===========================================================================
# Measurement 1: codebook collinearity (no forward — read the ckpt tensor)
# ===========================================================================

def probe_codebook(codebook_np: np.ndarray, h: int) -> dict:
    n = codebook_np.shape[0]
    eff, hard, sv = effective_rank(codebook_np)
    mx, mn, gram = offdiag_cosine_stats(codebook_np)
    return {
        "n": n, "h": h,
        "eff_rank": eff, "hard_rank": hard,
        "max_offcos": mx, "mean_offcos": mn,
        "gram": gram, "sv": sv,
    }


# ===========================================================================
# GPU path: build model, install hooks, run breaths, capture intermediates
# ===========================================================================

def _build_model_and_spec(task: str, K: int, s_max: int):
    """Mirror the eval scripts' model build (Pythia-410M -> BreathingTransformer).

    Returns (model, spec, loader, alloc, head_dim, n_heads).
    """
    import gc
    from tinygrad import Device

    from mycelium import Config
    from mycelium.loader import _load_state, load_breathing
    from mycelium.factor_graph_engine import (
        FactorGraphSpec, attach_factor_graph_params, FG_HYP_MASK,
    )
    from mycelium.factor_masks import cell_mp_head_allocation

    if FG_HYP_MASK:
        # The frozen ckpts have NO fg_hyp_anchors_* keys -> they were trained
        # with the boolean mask (FG_HYP_MASK=0).  Refuse to run with the
        # hyperbolic path on, which would inject untrained anchor params.
        raise SystemExit(
            "FG_HYP_MASK=1 but the frozen checkpoints have no anchor tensors; "
            "re-run with FG_HYP_MASK=0 (the boolean-mask path these ckpts used).")

    if task == "coloring":
        from mycelium.graph_coloring_data import GraphColoringLoader
        N_VALUES = int(os.environ.get("FG_N_VALUES", "3"))
        n_factor_types = 1
        n_heads = 16
        loader = GraphColoringLoader(
            n_instances=int(os.environ.get("FG_N_INSTANCES", "8000")),
            s_max=s_max, k_colors=N_VALUES,
            batch_size=int(os.environ.get("PROBE_BATCH", "8")),
            seed=int(os.environ.get("SEED", "42")),
        )
    else:  # circuit / circuit_deep
        from mycelium.circuit_data import CircuitLoader
        gate_types = ("AND", "OR", "NOT")
        N_VALUES = 2
        n_heads = 16
        loader = CircuitLoader(
            n_instances=int(os.environ.get("FG_N_INSTANCES", "8000")),
            s_max=s_max, n_values=N_VALUES,
            batch_size=int(os.environ.get("PROBE_BATCH", "8")),
            seed=int(os.environ.get("SEED", "42")),
            gate_types=gate_types,
        )
        n_factor_types = int(loader.n_factor_types)

    spec = FactorGraphSpec(
        s_max=s_max, n_values=N_VALUES, n_factor_types=n_factor_types,
        n_heads=n_heads, k_max=K, has_factor_inlet=False,
    )

    cfg = Config()
    sd = _load_state()
    model = load_breathing(cfg, sd=sd)
    del sd
    gc.collect()

    # cast_layers_fp32 (mirror of the eval scripts).
    from tinygrad import dtypes as _dt
    def _cast(obj, attr):
        t = getattr(obj, attr)
        if t.dtype == _dt.half:
            setattr(obj, attr, t.cast(_dt.float).contiguous().realize())
    _cast(model.embed, "weight")
    sw = model.block.shared
    for a in ("wv", "bv", "wo", "bo", "w_out", "b_out"):
        _cast(sw, a)
    for layer in model.block.layers:
        for a in ("wq", "bq", "wk", "bk", "w_in", "b_in"):
            _cast(layer, a)

    attach_factor_graph_params(model, hidden=cfg.hidden, spec=spec)
    Device[Device.DEFAULT].synchronize()

    G = max(1, n_heads // 16)
    alloc = cell_mp_head_allocation(n_factor_types, n_heads, G)  # (H,) int64
    head_dim = cfg.hidden // n_heads
    return model, spec, loader, alloc, head_dim, n_heads, cfg


def _load_ckpt(model, path: str):
    """Mirror eval_coloring_bands.load_ckpt (FG param + backbone keys)."""
    from tinygrad.nn.state import safe_load
    sd = safe_load(path)
    fg_names = ["fg_state_embed", "fg_position_embed", "fg_value_codebook",
                "fg_calib_head_w", "fg_calib_head_b", "fg_breath_embed",
                "fg_delta_gate"]
    targets = {"ln_f.g": model.ln_f_g, "ln_f.b": model.ln_f_b}
    sw = model.block.shared
    for a in ("wv", "bv", "wo", "bo", "w_out", "b_out",
              "in_ln_g", "in_ln_b", "post_ln_g", "post_ln_b"):
        targets[f"shared.{a}"] = getattr(sw, a)
    for i, layer in enumerate(model.block.layers):
        for a in ("wq", "bq", "wk", "bk", "w_in", "b_in"):
            targets[f"phase{i}.{a}"] = getattr(layer, a)
    for nm in fg_names:
        targets[nm] = getattr(model, nm)
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
                missing.append(f"{name}(shape)")
                continue
        if src.dtype != dst.dtype:
            src = src.cast(dst.dtype)
        dst.assign(src).realize()
    if missing:
        print(f"  ckpt missing {len(missing)} keys: {missing[:6]}", flush=True)
    else:
        print(f"  ckpt loaded cleanly ({len(targets)} keys).", flush=True)


def run_gpu_probe(task: str, ckpt: str, K: int) -> dict:
    """Build model, install non-invasive hooks, run the breath loop, capture.

    Returns a dict of measurement results for `task`.
    """
    import math
    from tinygrad import Tensor, dtypes
    import mycelium.breathing as breathing_mod
    import mycelium.factor_graph_engine as fge

    S_MAX = int(os.environ.get("FG_S_MAX", "49"))
    HEAD_LAYER = int(os.environ.get("PROBE_HEAD_LAYER", "3"))
    HEAD_BREATH = int(os.environ.get("PROBE_HEAD_BREATH", str(K - 1)))
    N_INST = int(os.environ.get("PROBE_N_INST", "64"))
    BATCH = int(os.environ.get("PROBE_BATCH", "8"))

    print(f"\n{'#'*70}\n#  GPU PROBE: task={task}  ckpt={ckpt}\n{'#'*70}", flush=True)
    model, spec, loader, alloc, head_dim, n_heads, cfg = _build_model_and_spec(
        task, K, S_MAX)
    print(f"  spec: N={spec.n_values} T={spec.n_factor_types} H={n_heads} "
          f"head_dim={head_dim} K={K} s_max={S_MAX}", flush=True)
    print(f"  head allocation (head->relation, -1=global): {alloc.tolist()}",
          flush=True)
    print(f"loading checkpoint: {ckpt}", flush=True)
    _load_ckpt(model, ckpt)
    Tensor.training = False

    # ---- measurement 1: codebook (read tensor directly) --------------------
    cb_np = model.fg_value_codebook.realize().numpy().astype(np.float64)
    cb_res = probe_codebook(cb_np, h=cfg.hidden)

    # ---- HOOK A: per-breath readout-LN reps -------------------------------
    # mycelium.factor_graph_engine imports _layernorm locally as
    #   from mycelium.breathing import _layernorm
    # inside factor_breathing_forward, so we patch mycelium.breathing._layernorm.
    # We capture ONLY the readout call (gamma IS model.ln_f_g) — every per-layer
    # LN uses a DIFFERENT gamma object, so the readout is unambiguous.
    orig_layernorm = breathing_mod._layernorm
    breath_capture: list[np.ndarray] = []     # one (B, S, H) per readout call

    def _patched_layernorm(x, gamma, beta, eps=1e-5):
        out = orig_layernorm(x, gamma, beta, eps)
        if gamma is model.ln_f_g:
            breath_capture.append(out.cast(dtypes.float).realize().numpy())
        return out

    # ---- HOOK B: per-head context at (HEAD_LAYER, HEAD_BREATH) -------------
    # The engine calls kenken_layer_forward(layer, h, attn_bias) for layers 0..3
    # each breath.  We patch factor_graph_engine.kenken_layer_forward (the name
    # the engine resolves).  The wrapper recomputes the per-head context exactly
    # as the original does (LN -> q,k,v -> scores+bias -> softmax -> attn@v) and
    # records the per-head ctx (B, S, n_heads, head_dim) at the target layer, on
    # the target breath only, then calls the original for the true forward.
    orig_kenken_layer = fge.kenken_layer_forward
    # identify layers by object identity -> index.
    layer_index = {id(L): i for i, L in enumerate(model.block.layers)}
    call_state = {"breath": 0, "layer_calls": 0}
    head_capture: dict = {"ctx": None}        # (B, S, n_heads, head_dim) numpy

    def _patched_kenken_layer(layer, x, attn_bias, q_rot_cos=None, q_rot_sin=None):
        li = layer_index.get(id(layer), -1)
        # Each breath issues exactly 4 layer calls (layers[:4]) in order; track
        # the breath index by counting layer-0 calls.
        if li == 0:
            call_state["breath"] = call_state["layer_calls"] // 4
            call_state["layer_calls"] += 1
        else:
            call_state["layer_calls"] += 1
        if (li == HEAD_LAYER and call_state["breath"] == HEAD_BREATH
                and head_capture["ctx"] is None):
            cfgl = layer.cfg
            B, Sx, Hx = x.shape
            attn_in = orig_layernorm(x, layer.shared.in_ln_g,
                                     layer.shared.in_ln_b, cfgl.layer_norm_eps)
            attn_in_dt = attn_in.cast(x.dtype) if attn_in.dtype != x.dtype else attn_in
            q = (attn_in_dt @ layer.wq + layer.bq).reshape(
                B, Sx, cfgl.n_heads, cfgl.head_dim).transpose(1, 2)
            k = (attn_in_dt @ layer.wk + layer.bk).reshape(
                B, Sx, cfgl.n_heads, cfgl.head_dim).transpose(1, 2)
            v = (attn_in_dt @ layer.shared.wv + layer.shared.bv).reshape(
                B, Sx, cfgl.n_heads, cfgl.head_dim).transpose(1, 2)
            scale = 1.0 / math.sqrt(cfgl.head_dim)
            scores = q @ k.transpose(-2, -1) * scale
            scores = scores + attn_bias.cast(scores.dtype)
            attn = scores.clip(-1e4, 1e4).softmax(-1)
            ctx = (attn @ v).transpose(1, 2)              # (B, S, n_heads, head_dim)
            head_capture["ctx"] = ctx.cast(dtypes.float).realize().numpy()
        return orig_kenken_layer(layer, x, attn_bias, q_rot_cos, q_rot_sin)

    # ---- run the forward over a sample of instances ------------------------
    from mycelium.factor_graph_engine import factor_breathing_forward

    # accumulate per-breath reps (valid cells) across instances.
    per_breath_rows: list[list[np.ndarray]] = [[] for _ in range(K)]
    head_group_rows: dict[int, list[np.ndarray]] = {}  # relation -> list of (head_dim,) per (head, cell)
    # We also store per-head reps separately so we can compute group eff-rank
    # over the stacked (n_heads_in_group * n_valid_cells, head_dim).

    breathing_mod._layernorm = _patched_layernorm
    fge.kenken_layer_forward = _patched_kenken_layer
    try:
        done = 0
        for batch in loader.iter_eval(batch_size=BATCH):
            breath_capture.clear()
            head_capture["ctx"] = None
            call_state["breath"] = 0
            call_state["layer_calls"] = 0

            _ = factor_breathing_forward(model, batch, spec, K=K)

            cv = batch.cell_valid.realize().numpy()           # (B, S)
            B_cur = cv.shape[0]
            real = min(B_cur, N_INST - done)                  # cap exactly at N_INST
            # how many readout LN captures? exactly K (one per breath).
            assert len(breath_capture) == K, \
                f"expected {K} readout captures, got {len(breath_capture)}"

            for bi in range(real):
                valid = cv[bi] > 0.5
                if not np.any(valid):
                    continue
                for kk in range(K):
                    reps = breath_capture[kk][bi][valid]      # (n_valid, H)
                    per_breath_rows[kk].append(reps)

            # per-head context (target layer/breath) for valid cells.
            ctx = head_capture["ctx"]                          # (B, S, n_heads, head_dim)
            if ctx is not None:
                for bi in range(real):
                    valid = cv[bi] > 0.5
                    if not np.any(valid):
                        continue
                    # group heads by relation; stack per-head ctx over valid cells.
                    for hh in range(n_heads):
                        rel = int(alloc[hh])
                        head_group_rows.setdefault(rel, []).append(
                            ctx[bi, valid, hh, :])             # (n_valid, head_dim)

            done += real
            if done >= N_INST:
                break
    finally:
        breathing_mod._layernorm = orig_layernorm
        fge.kenken_layer_forward = orig_kenken_layer

    print(f"  captured {done} instances; "
          f"per-breath valid-cell rows assembled.", flush=True)

    # ---- measurement 2: breath rank / collapse -----------------------------
    # For each breath, stack the sampled valid-cell reps (cap rows for SVD cost).
    MAX_ROWS = int(os.environ.get("PROBE_MAX_ROWS", "4000"))
    breath_eff: list[float] = []
    breath_frac: list[float] = []
    breath_stack: list[np.ndarray] = []
    for kk in range(K):
        stk = np.concatenate(per_breath_rows[kk], axis=0)      # (Ntot, H)
        if stk.shape[0] > MAX_ROWS:
            idx = np.linspace(0, stk.shape[0] - 1, MAX_ROWS).astype(int)
            stk = stk[idx]
        breath_stack.append(stk)
        eff, hard, _ = effective_rank(stk)
        max_rank = min(stk.shape[0], stk.shape[1])
        breath_eff.append(eff)
        breath_frac.append(eff / max(max_rank, 1))
    # consecutive-breath cosine: per cell, breath k vs k+1 (use the FIRST
    # instance's valid cells, paired exactly across breaths — same row order).
    # We pair using the per-instance captured reps so rows correspond.
    consec: list[float] = []
    # rebuild paired rows from per_breath_rows: row sets are appended per-instance
    # in the SAME order for every breath, so concatenation preserves the pairing.
    full_breath = [np.concatenate(per_breath_rows[kk], axis=0) for kk in range(K)]
    for kk in range(K - 1):
        consec.append(consecutive_cosine(full_breath[kk], full_breath[kk + 1]))

    breath_max_rank = min(full_breath[0].shape[0], full_breath[0].shape[1])
    breath_res = {
        "eff": breath_eff, "frac": breath_frac, "consec": consec,
        "max_rank": breath_max_rank, "H": cfg.hidden,
        "n_rows": full_breath[0].shape[0],
    }

    # ---- measurement 3: within-group head redundancy -----------------------
    # For each relation group, stack the per-head ctx over (head, valid-cell):
    # rows = (n_heads_in_group * n_valid_cells, head_dim).  Effective rank over
    # that stack tells whether same-relation heads occupy distinct subspaces.
    # ALSO compute the per-head MEAN context vector (one vector per head) and the
    # mean pairwise cosine BETWEEN heads in the group — the direct "are the heads
    # the same" measure.
    head_res: dict[int, dict] = {}
    rel_name = _relation_names(task, spec.n_factor_types)
    for rel, chunks in sorted(head_group_rows.items()):
        # chunks are appended per (instance, head); regroup by head.
        # We appended in order: for each instance, hh=0..n_heads-1.  But chunks
        # only contains heads with this `rel`.  To get per-head mean vectors we
        # recompute from the head_means accumulation below instead.
        pass
    # Recompute head-level structure cleanly: stack ALL per-head ctx rows per
    # relation, and per-head MEAN vectors per relation.
    head_res = _compute_head_groups(head_group_rows, alloc, n_heads, head_dim,
                                    rel_name)

    return {
        "task": task, "ckpt": ckpt, "K": K,
        "codebook": cb_res,
        "breath": breath_res,
        "head": head_res,
        "head_dim": head_dim, "n_heads": n_heads,
        "rel_name": rel_name,
    }


def _relation_names(task: str, T: int) -> dict[int, str]:
    if task == "coloring":
        names = {0: "edge"}
    else:  # circuit: AND,OR,NOT
        names = {0: "AND", 1: "OR", 2: "NOT", 3: "XOR"}
    names = {r: names.get(r, f"type{r}") for r in range(T)}
    names[-1] = "global"
    return names


def _compute_head_groups(head_group_rows, alloc, n_heads, head_dim, rel_name):
    """Per-relation group: eff-rank over stacked per-head ctx rows, and the mean
    pairwise cosine between per-head MEAN context vectors.

    head_group_rows[rel] is a list of (n_valid, head_dim) chunks; chunks were
    appended per (instance, head) for every head with that relation, in head
    order within each instance.  We can't trivially separate by head from the
    flat list, so we ALSO need per-head means.  To keep it simple and exact, we
    stack ALL rows for the group (eff-rank) and reconstruct per-head means by
    re-grouping: the chunks alternate heads in a known cycle per instance.
    """
    # Map relation -> list of head indices (in head order).
    rel_to_heads: dict[int, list[int]] = {}
    for hh in range(n_heads):
        rel_to_heads.setdefault(int(alloc[hh]), []).append(hh)

    out: dict[int, dict] = {}
    MAX_ROWS = int(os.environ.get("PROBE_MAX_ROWS", "4000"))
    for rel, chunks in sorted(head_group_rows.items()):
        heads = rel_to_heads.get(rel, [])
        n_g = len(heads)
        if not chunks:
            continue
        # chunks were appended in cycles of n_g heads per instance.
        # Reconstruct per-head row lists.
        per_head_chunks: list[list[np.ndarray]] = [[] for _ in range(n_g)]
        for ci, ch in enumerate(chunks):
            per_head_chunks[ci % n_g].append(ch)
        # per-head mean context vector (one (head_dim,) per head).
        head_means = []
        for hi in range(n_g):
            allrows = np.concatenate(per_head_chunks[hi], axis=0)  # (Nh, head_dim)
            head_means.append(allrows.mean(axis=0))
        head_means = np.stack(head_means, axis=0)                  # (n_g, head_dim)

        # eff-rank over the full stacked group rows.
        stacked = np.concatenate(chunks, axis=0)                  # (Ntot, head_dim)
        if stacked.shape[0] > MAX_ROWS:
            idx = np.linspace(0, stacked.shape[0] - 1, MAX_ROWS).astype(int)
            stacked = stacked[idx]
        eff, hard, _ = effective_rank(stacked)

        # eff-rank over the per-head MEAN vectors (n_g vectors in head_dim).
        eff_means, _, _ = effective_rank(head_means)

        mean_cos = mean_pairwise_cosine(head_means)
        mx, mn, _ = offdiag_cosine_stats(head_means)

        out[rel] = {
            "name": rel_name.get(rel, f"type{rel}"),
            "heads": heads,
            "n_heads": n_g,
            "eff_rank_rows": eff,           # rank of the full activation stack
            "eff_rank_headmeans": eff_means,  # rank across the n_g head mean vectors
            "max_rank_headmeans": min(n_g, head_dim),
            "mean_cos_headmeans": mean_cos,
            "max_abscos_headmeans": mx,
            "mean_abscos_headmeans": mn,
        }
    return out


# ===========================================================================
# Reporting
# ===========================================================================

def _report(res: dict) -> dict:
    """Print per-checkpoint measurement numbers + verdicts.  Returns verdict dict."""
    task = res["task"]
    cb = res["codebook"]
    br = res["breath"]
    hd = res["head"]
    K = res["K"]
    head_dim = res["head_dim"]

    print(f"\n{'='*72}", flush=True)
    print(f"  RESULTS — {task}  ({res['ckpt']})", flush=True)
    print(f"{'='*72}", flush=True)

    # 1. codebook
    print(f"\n  [1] CODEBOOK COLLINEARITY  (value_codebook N={cb['n']} x H={cb['h']})",
          flush=True)
    print(f"      singular values: {np.array2string(cb['sv'], precision=4)}",
          flush=True)
    print(f"      effective rank   = {cb['eff_rank']:.4f}  (max = N = {cb['n']})",
          flush=True)
    print(f"      hard rank        = {cb['hard_rank']}", flush=True)
    print(f"      max |off-diag cos| = {cb['max_offcos']:.4f}", flush=True)
    print(f"      mean|off-diag cos| = {cb['mean_offcos']:.4f}", flush=True)
    print(f"      off-diag cosine gram:", flush=True)
    g = cb["gram"]
    for i in range(g.shape[0]):
        rowstr = "  ".join(f"{g[i, j]:+.3f}" for j in range(g.shape[1]))
        print(f"        [{rowstr}]", flush=True)
    cb_verdict = _verdict_codebook(cb["eff_rank"], cb["n"], cb["h"], cb["mean_offcos"])
    print(f"      VERDICT: {cb_verdict}", flush=True)

    # 2. breath
    print(f"\n  [2] BREATH RANK / COLLAPSE  (K={K}; {br['n_rows']} valid-cell rows, "
          f"H={br['H']}, max_rank={br['max_rank']})", flush=True)
    print(f"      per-breath effective rank (B0..B{K-1}):", flush=True)
    for r0 in range(0, K, 8):
        chunk = br["eff"][r0:r0 + 8]
        fr = br["frac"][r0:r0 + 8]
        s = "  ".join(f"B{r0+i}={chunk[i]:.1f}({fr[i]:.0%})" for i in range(len(chunk)))
        print(f"        {s}", flush=True)
    print(f"      consecutive-breath cosine (B0->B1 .. B{K-2}->B{K-1}):", flush=True)
    for r0 in range(0, K - 1, 8):
        chunk = br["consec"][r0:r0 + 8]
        s = "  ".join(f"{r0+i}:{chunk[i]:.3f}" for i in range(len(chunk)))
        print(f"        {s}", flush=True)
    br_verdict = _verdict_breath(br["eff"], br["max_rank"], br["consec"])
    print(f"      VERDICT: {br_verdict}", flush=True)

    # 3. head groups
    print(f"\n  [3] WITHIN-GROUP HEAD REDUNDANCY  (per relation; head_dim={head_dim})",
          flush=True)
    head_verdicts = {}
    for rel in sorted(hd.keys()):
        h = hd[rel]
        print(f"      group '{h['name']}' (heads {h['heads']}, n={h['n_heads']}):",
              flush=True)
        print(f"        eff_rank over activation stack       = {h['eff_rank_rows']:.3f} "
              f"(max = head_dim = {head_dim})", flush=True)
        print(f"        eff_rank over per-head mean vectors   = "
              f"{h['eff_rank_headmeans']:.3f} (max = {h['max_rank_headmeans']})",
              flush=True)
        print(f"        mean pairwise cos (head means)        = {h['mean_cos_headmeans']:+.4f}",
              flush=True)
        print(f"        mean |pairwise cos| (head means)      = {h['mean_abscos_headmeans']:.4f}"
              f"  (max {h['max_abscos_headmeans']:.4f})", flush=True)
        v = _verdict_head_group(h["eff_rank_headmeans"], h["n_heads"], head_dim,
                                h["mean_abscos_headmeans"])
        head_verdicts[rel] = v
        print(f"        VERDICT: {v}", flush=True)

    return {"codebook": cb_verdict, "breath": br_verdict, "head": head_verdicts}


# ===========================================================================
# Main
# ===========================================================================

CKPTS = {
    "coloring": ".cache/fg_ckpts/fg_coloring_k16/fg_coloring_k16_final.safetensors",
    "circuit": ".cache/fg_ckpts/fg_circuit_k16/fg_circuit_k16_final.safetensors",
    "circuit_deep": ".cache/fg_ckpts/fg_circuit_deep_k16/fg_circuit_deep_k16_step1200.safetensors",
}


def main():
    if int(os.environ.get("SELFTEST_ONLY", "0")) > 0:
        ok = selftest()
        sys.exit(0 if ok else 1)

    # GPU path: always run the selftest first (cheap, validates the machinery).
    print("Running CPU selftest before GPU probe...", flush=True)
    if not selftest():
        print("SELFTEST FAILED — aborting GPU probe.", flush=True)
        sys.exit(1)

    K = int(os.environ.get("K", os.environ.get("FG_K_MAX", "16")))
    only = os.environ.get("PROBE_ONLY", "").strip()

    tasks = []
    if only:
        tasks = [only]
    else:
        tasks = ["coloring", "circuit"]
        if os.path.exists(CKPTS["circuit_deep"]):
            tasks.append("circuit_deep")

    all_verdicts = {}
    for task in tasks:
        ckpt = CKPTS.get(task)
        if ckpt is None or not os.path.exists(ckpt):
            print(f"[skip] {task}: ckpt not found ({ckpt})", flush=True)
            continue
        res = run_gpu_probe(task, ckpt, K)
        verdicts = _report(res)
        all_verdicts[task] = verdicts

    # ---- overall read -------------------------------------------------------
    print(f"\n{'='*72}\n  OVERALL VERDICT (baseline for the general-weights run)\n{'='*72}",
          flush=True)
    for task, v in all_verdicts.items():
        print(f"\n  {task}:", flush=True)
        print(f"    codebook : {v['codebook']}", flush=True)
        print(f"    breath   : {v['breath']}", flush=True)
        for rel, hv in v["head"].items():
            print(f"    head[{rel}]: {hv}", flush=True)
    print(flush=True)


if __name__ == "__main__":
    ok = _ast_parse_ok()
    print(f"[ast.parse] astparse_ok={ok}", flush=True)
    if not ok:
        sys.exit(1)
    main()
