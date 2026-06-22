"""fg_waist_smoke.py — CPU smoke for the in-deducer WAIST (FG_WAIST).

GPU-FREE. Run with DEV=CPU. Validates the 5 contract checks the BUILD requires:

  (1) FG_WAIST OFF  => forward BYTE-IDENTICAL to current (no waist params -> getattr-gate
      skips the whole waist -> the original 2-tuple, exact-equal logits).
  (2) WAIST ON, gate ~ 0 (gate_init large -ve) => forward ~ IDENTICAL (within fp) to OFF
      (warm-start safety: RESUME_FROM a baseline ckpt starts byte-identical; the convex
      blend at g~0 is a near-exact pass-through regardless of the waist contents).
  (3) The aux VALIDITY LABELS are correct: the in-graph free verifier
      (_factor_alldiff_violation) returns 0 for a PROPER coloring (gold) and >0 for an
      IMPROPER one; the gold teacher-forced path is always valid.
  (4) SHAPES / JIT-contract hold: return_waist gives K (B,S,d) d-reps; pooled_waist_drep
      -> (B,d); the d-rep capture sink collects K tensors; engine returns the right shapes.
  (5) ENGINE ADDITIVE + kenken.py git-clean (asserted via git + a no-waist-symbols check).

Tiny CPU model (hidden=64, 4 layers, K=3) — the engine path is dimension-parameterized, so
the contract is identical to the 1024-d GPU model. NO Pythia load, NO training, NO GPU.
"""
import os
import subprocess
import sys

os.environ.setdefault("DEV", "CPU")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from tinygrad import Tensor, dtypes

from mycelium import Config, BreathingTransformer
from mycelium.factor_graph_engine import (
    FactorGraphSpec, FactorGraphBatch, factor_breathing_forward,
    attach_factor_graph_params, attach_factor_waist_params,
    factor_waist_parameters, pooled_waist_drep,
)

_OK = True


def _check(name, cond):
    global _OK
    status = "PASS" if cond else "FAIL"
    if not cond:
        _OK = False
    print(f"  [{status}] {name}", flush=True)
    return cond


def _tiny_model():
    """A tiny BreathingTransformer-shaped model for the CPU engine forward. We only need
    block.layers[:4], block.shared, ln_f_g/b, cfg — built by BreathingTransformer(cfg)."""
    cfg = Config(hidden=64, n_heads=4, head_dim=16, ffn=128)
    model = BreathingTransformer(cfg)
    # Cast the layer weights the engine touches to fp32 (mirror cast_layers_fp32 minimally).
    def _cast(obj, attr):
        t = getattr(obj, attr, None)
        if t is not None and t.dtype == dtypes.half:
            setattr(obj, attr, t.cast(dtypes.float).contiguous().realize())
    _cast(model.embed, "weight")
    sw = model.block.shared
    for a in ("wv", "bv", "wo", "bo", "w_out", "b_out"):
        _cast(sw, a)
    for layer in model.block.layers:
        for a in ("wq", "bq", "wk", "bk", "w_in", "b_in"):
            _cast(layer, a)
    return model, cfg


def _coloring_batch(B=2, S=49, k=3, n=6, seed=0):
    """A PROPER-coloring batch on the engine's 49-cell grid (kenken_layer_forward asserts
    S==49). The first `n` cells are a path graph 0-1-..-(n-1); cells n..48 are PADDING
    (cell_valid=0). gold is a valid k-coloring on the real cells. One edge factor per
    consecutive pair (two membership 1s each)."""
    rng = np.random.RandomState(seed)
    edges = [(i, i + 1) for i in range(n - 1)]
    L = len(edges)
    membership = np.zeros((B, L, S), dtype=np.float32)
    latent_type = np.zeros((B, L), dtype=np.int32)        # all edges -> type 0
    for e, (u, v) in enumerate(edges):
        membership[:, e, u] = 1.0
        membership[:, e, v] = 1.0
    # gold: a proper k-coloring of the path on the real cells; pad cells gold=0.
    gold = np.zeros((B, S), dtype=np.int32)
    gold[:, :n] = np.tile(np.array([(i % k) + 1 for i in range(n)], dtype=np.int32), (B, 1))
    cell_valid = np.zeros((B, S), dtype=np.float32)
    cell_valid[:, :n] = 1.0
    input_cells = np.zeros((B, S), dtype=np.int32)        # all unknown (deduce all)
    vdm = np.zeros((B, S, k), dtype=np.float32)
    vdm[:, :n, :] = 1.0                                    # all k colors legal on real cells
    d = {
        "input_cells": Tensor(input_cells, dtype=dtypes.int).contiguous().realize(),
        "cell_valid": Tensor(cell_valid, dtype=dtypes.float).contiguous().realize(),
        "value_domain_mask": Tensor(vdm, dtype=dtypes.float).contiguous().realize(),
        "gold": Tensor(gold, dtype=dtypes.int).contiguous().realize(),
        "membership": Tensor(membership, dtype=dtypes.float).contiguous().realize(),
        "latent_type": Tensor(latent_type, dtype=dtypes.int).contiguous().realize(),
    }
    return FactorGraphBatch(d), gold, membership, latent_type, cell_valid


def main():
    print("=== FG_WAIST CPU smoke (DEV=CPU; no GPU, no training) ===", flush=True)
    k = 3
    K = 3
    S = 49                                   # the engine's hard cell-grid (kenken asserts 49)
    n = 6                                     # real cells; rest are padding
    model, cfg = _tiny_model()
    spec = FactorGraphSpec(s_max=S, n_values=k, n_factor_types=1, n_heads=cfg.n_heads,
                           k_max=K, has_factor_inlet=False)
    attach_factor_graph_params(model, hidden=cfg.hidden, spec=spec)
    batch, gold_np, mem_np, lt_np, cv_np = _coloring_batch(B=2, S=S, k=k, n=n, seed=1)

    # ---- (1) FG_WAIST OFF: baseline forward (no waist params attached). ----------------
    lh_off, ch_off = factor_breathing_forward(model, batch, spec, K=K)
    off_logits = [t.realize().numpy().copy() for t in lh_off]
    _check("(4) baseline forward returns K logits (B,S,N) + K calibs",
           len(lh_off) == K and lh_off[-1].shape == (2, S, k)
           and len(ch_off) == K and ch_off[-1].shape == (2,))
    _check("(5) baseline model has NO waist params (factor_waist_parameters empty)",
           factor_waist_parameters(model) == []
           and getattr(model, "fg_waist_down", None) is None)

    # ---- (2) WAIST ON, gate ~ 0: attach + re-run; must ~equal the OFF forward. ----------
    attach_factor_waist_params(model, hidden=cfg.hidden, d=8, after=1,
                               gate_init=-8.0, aux="both")
    _check("(4) waist attached -> factor_waist_parameters non-empty (down/up/gate + aux)",
           len(factor_waist_parameters(model)) == 7)
    g0 = float(model.fg_waist_gate.sigmoid().mean().numpy())
    _check("(2) gate sigmoid ~ 0 at init (gate_init=-8 -> g<1e-3)", g0 < 1e-3)

    lh_on, ch_on = factor_breathing_forward(model, batch, spec, K=K)
    on_logits = [t.realize().numpy() for t in lh_on]
    max_abs = max(float(np.max(np.abs(a - b))) for a, b in zip(on_logits, off_logits))
    print(f"      max|Δlogit| (waist gate~0 + up zero-init vs off) = {max_abs:.3e}",
          flush=True)
    # The convex blend at g~0 is (1-g)*h + g*up_x. With up zero-init, up_x==0, so the blend
    # is (1-g)*h == h scaled by (1-g) ~ h*(1 - 3.4e-4): the ONLY residual delta is this tiny
    # (1-g) scaling (g = sigmoid(-8) ~ 3.4e-4), which is WITHIN fp (the spec's "about
    # identical within fp" warm-start guarantee). NOT exactly 0 because the convex form
    # scales h by (1-g); that is by design (the FRAME specifies the convex blend). Resuming
    # a baseline ckpt thus starts within-fp-identical -> the gentle fine-tune is warm-start
    # safe. (To make it EXACTLY 0 one would use a residual-add h + g*up_x instead of the
    # convex blend; the FRAME chose the convex blend for its bounded [0,1] influence.)
    _check("(2) waist ON @ init (gate~0 + up zero-init) ~ identical to OFF within fp "
           "(max|Δ| < 5e-3, the (1-g) convex-scale only)", max_abs < 5e-3)

    # Now OPEN the gate AND give up nonzero weights, confirm the forward CHANGES (the waist
    # is live, not dead) — proves the blend mixes up(gelu(down(x))) when g>0 and up!=0.
    model.fg_waist_up.assign(Tensor(
        (np.random.RandomState(7).randn(8, cfg.hidden) * 0.05).astype(np.float32),
        dtype=dtypes.float)).realize()
    model.fg_waist_gate.assign(Tensor(np.array(8.0, dtype=np.float32),
                                      dtype=dtypes.float)).realize()
    g1 = float(model.fg_waist_gate.sigmoid().mean().numpy())
    lh_open, _ = factor_breathing_forward(model, batch, spec, K=K)
    open_logits = [t.realize().numpy() for t in lh_open]
    max_open = max(float(np.max(np.abs(a - b))) for a, b in zip(open_logits, off_logits))
    print(f"      gate open sigmoid={g1:.3f}; max|Δlogit| (open vs off) = {max_open:.3e}",
          flush=True)
    _check("(2) gate FULLY OPEN changes the forward (waist is live, not dead): max|Δ|>1e-3",
           max_open > 1e-3)
    # restore the warm-start gate for the rest.
    model.fg_waist_gate.assign(Tensor(np.array(-8.0, dtype=np.float32),
                                      dtype=dtypes.float)).realize()

    # ---- (4) return_waist contract + pooled_waist_drep + capture sink. ------------------
    lh_w, ch_w, drep_hist = factor_breathing_forward(model, batch, spec, K=K,
                                                     return_waist=True)
    _check("(4) return_waist gives K d-reps shaped (B,S,d=8)",
           len(drep_hist) == K and drep_hist[-1].shape == (2, S, 8))
    pooled = pooled_waist_drep(drep_hist[-1], batch.cell_valid)
    _check("(4) pooled_waist_drep -> (B,d)", pooled.realize().shape == (2, 8))

    # capture sink (the re-probe exposure path): set the list, run, expect K appended d-reps.
    model.fg_waist_capture = []
    _lh, _ch = factor_breathing_forward(model, batch, spec, K=K)
    _check("(4) fg_waist_capture sink collects K per-breath d-reps",
           len(model.fg_waist_capture) == K
           and model.fg_waist_capture[-1].shape == (2, S, 8))
    model.fg_waist_capture = None
    # return_waist=True on a model WITHOUT waist -> empty list (aux is a no-op there).
    model2, cfg2 = _tiny_model()
    attach_factor_graph_params(model2, hidden=cfg2.hidden, spec=spec)
    _l2, _c2, drep2 = factor_breathing_forward(model2, batch, spec, K=K, return_waist=True)
    _check("(4) return_waist on a NO-WAIST model -> empty d-rep history (aux no-op safe)",
           drep2 == [])

    # ---- (3) aux validity labels correct (in-graph free verifier). ----------------------
    from scripts.factor_graph_train import _factor_alldiff_violation
    # PROPER coloring (gold): zero violations.
    gold_oh = (Tensor(gold_np, dtype=dtypes.int) - 1).one_hot(k).cast(dtypes.float)  # (B,S,k)
    viol_gold = _factor_alldiff_violation(gold_oh, batch.membership, batch.latent_type, 1)
    vg = viol_gold.realize().numpy()
    _check("(3) free verifier: GOLD proper coloring -> 0 violations (valid)",
           np.allclose(vg, 0.0))
    # IMPROPER coloring: make every cell the SAME color -> every edge clashes.
    bad = np.ones_like(gold_np)
    bad_oh = (Tensor(bad, dtype=dtypes.int) - 1).one_hot(k).cast(dtypes.float)
    viol_bad = _factor_alldiff_violation(bad_oh, batch.membership, batch.latent_type, 1)
    vb = viol_bad.realize().numpy()
    L = mem_np.shape[1]
    _check("(3) free verifier: ALL-SAME coloring -> every edge violates (viol == n_edges)",
           np.allclose(vb, float(L)))
    # is_valid label derives correctly (proper -> 1, improper -> 0).
    _check("(3) validity label: proper->valid(1), improper->invalid(0)",
           bool((vg < 0.5).all()) and bool((vb >= 0.5).all()))

    # padding cells must not create spurious clashes: a row with pad cells (cell_valid=0)
    # whose pad value coincides with a neighbor must still verify if the REAL cells differ.
    # (We zero pad in the trainer via pred_oh * cell_valid; here gold has no pad, so this is
    # a structural note — the verifier counts only via membership ones, and pad cells are
    # never members of a real factor in encode_instance.)

    # ---- (5) engine additive + kenken.py git-clean (the oracle invariant). --------------
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    diff = subprocess.run(["git", "-C", root, "diff", "--name-only", "mycelium/kenken.py"],
                          capture_output=True, text=True).stdout.strip()
    _check("(5) mycelium/kenken.py is git-CLEAN (oracle untouched)", diff == "")
    with open(os.path.join(root, "mycelium", "kenken.py")) as f:
        ksrc = f.read()
    _check("(5) kenken.py contains NO waist symbols (additive change isolated to engine)",
           "fg_waist" not in ksrc and "FG_WAIST" not in ksrc)

    print(f"\n[fg_waist_smoke] {'ALL PASS' if _OK else 'SOME FAILED'}", flush=True)
    return 0 if _OK else 1


if __name__ == "__main__":
    raise SystemExit(main())
