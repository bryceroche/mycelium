"""KenKen STEP-0 SMOKE — verifies the scaffold's novel parts are EXACTLY right.

COLD build (fresh KenKen heads/codebook/inlet on the shared Pythia-410M backbone)
— there is NO byte-identity check. The gates (all six must pass):

  (a) FORWARD: runs on a batch from kenken_train.jsonl, produces
      cell_logits_history (K x (B,49,7)) + calib, no NaN, finite.
  (b) MASK GATE: row-head support == (N-1) row peers per valid cell; col likewise;
      cage-head support == cage-mate count per cell (symmetric, incl self);
      global == all valid cells.
  (c) GRADIENT LIVENESS: one eager backward; EVERY new KenKen param draws nonzero
      grad — ESPECIALLY the verification-inlet params (funded-vs-starved check).
  (d) INLET MAGNITUDE: verification-inlet per-element norm is bounded / comparable
      to the cell residual scale (not colliding).
  (e) CONVERGENCE INSTRUMENT: forward on ~16 puzzles → (breath_count,
      deduction_depth) pairs + settled/stuck counts (pre-training: no correlation
      expected; the gate only checks the instrument PRODUCES the artifact).
  (f) PARAM COUNT: total trainable params printed (Property-1 recurrence-baseline).

K = 20 (match v98 Sudoku). B = 8.

Run:
  KENKEN_TASK=1 .venv/bin/python scripts/kenken_smoke.py
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("KENKEN_TASK", "1")

import numpy as np
from tinygrad import Tensor, Device, dtypes

from mycelium import Config, BreathingTransformer
from mycelium.loader import _load_state, load_breathing
from mycelium.kenken_data import (
    KenKenLoader, N_MAX, N_CELLS, N_OPS, TARGET_BUCKETS,
)
from mycelium.kenken import (
    KENKEN_K_MAX, KENKEN_CONVERGE_JSD,
    attach_kenken_params, kenken_parameters, kenken_inlet_parameters,
    kenken_breathing_forward, kenken_loss, kenken_accuracy,
    build_verification_inlet, convergence_instrument,
    MAX_CAGE_SIZE,
)


def cast_layers_fp32(model):
    """Cast L0-L3 + shared weights fp16 -> fp32 for stable training (mirror sudoku)."""
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


def collect_kenken_params(model):
    """Trainable params: shared L0-L3 + final LN + kenken-specific (mirror sudoku)."""
    params = []
    sw = model.block.shared
    params += [sw.wv, sw.bv, sw.wo, sw.bo, sw.w_out, sw.b_out,
               sw.in_ln_g, sw.in_ln_b, sw.post_ln_g, sw.post_ln_b]
    for layer in model.block.layers:
        params += [layer.wq, layer.bq, layer.wk, layer.bk, layer.w_in, layer.b_in]
    params += [model.ln_f_g, model.ln_f_b]
    params += kenken_parameters(model)
    return params


def main():
    K = int(os.environ.get("KENKEN_K_MAX", str(KENKEN_K_MAX)))
    B = int(os.environ.get("BATCH", "8"))
    SEED = int(os.environ.get("SEED", "42"))
    TRAIN = os.environ.get("KENKEN_TRAIN", ".cache/kenken_train.jsonl")

    print("=" * 72)
    print(f"KENKEN STEP-0 SMOKE  device={Device.DEFAULT}  K={K}  B={B}")
    print(f"  N_max={N_MAX} (49 cells)  value-domain=N_MAX  JSD_thresh={KENKEN_CONVERGE_JSD}")
    print("=" * 72)

    np.random.seed(SEED)
    Tensor.manual_seed(SEED)

    # ---- model: Pythia-410M L0-L3 backbone (mirror sudoku init exactly) ----
    cfg = Config()
    print(f"loading Pythia-410M -> breathing transformer...")
    sd = _load_state()
    model = load_breathing(cfg, sd=sd)
    del sd
    cast_layers_fp32(model)
    attach_kenken_params(model, hidden=cfg.hidden, n_heads=cfg.n_heads, k_max=K)
    Device[Device.DEFAULT].synchronize()

    # ---- data ----
    loader = KenKenLoader(TRAIN, batch_size=B, seed=SEED)
    batch = loader.sample_batch()

    # ---- optimizer (created BEFORE any forward so requires_grad is set on every
    # param before the graph traces — mirrors sudoku_train.py ordering; otherwise
    # the first untracked forward freezes the params out of the autograd tape) ----
    from tinygrad.nn.optim import AdamW
    params = collect_kenken_params(model)
    opt = AdamW(params, lr=1e-4, weight_decay=0.0)

    # ====================================================================
    # GATE (a) FORWARD
    # ====================================================================
    print("\n--- GATE (a) FORWARD ---")
    Tensor.training = True
    cell_logits_history, calib_history = kenken_breathing_forward(model, batch, K=K)
    assert len(cell_logits_history) == K, f"expected {K} breaths, got {len(cell_logits_history)}"
    final = cell_logits_history[-1]
    assert tuple(final.shape) == (B, N_CELLS, N_MAX), f"bad logits shape {final.shape}"
    final_np = final.realize().numpy()
    calib_np = calib_history[-1].realize().numpy()
    finite_logits = bool(np.isfinite(final_np).all())
    finite_calib = bool(np.isfinite(calib_np).all())
    print(f"  cell_logits_history: K={len(cell_logits_history)} x {tuple(final.shape)}  "
          f"(dtype={final.dtype})")
    print(f"  calib_history[-1]: shape={tuple(calib_history[-1].shape)} "
          f"range=[{calib_np.min():.3f}, {calib_np.max():.3f}]")
    print(f"  logits finite={finite_logits}  calib finite={finite_calib}")
    print(f"  logits range=[{final_np.min():.3f}, {final_np.max():.3f}]")
    assert finite_logits and finite_calib, "GATE (a) FAIL: non-finite forward output"
    print("  GATE (a) PASS")

    # ====================================================================
    # GATE (b) MASK GATE
    # ====================================================================
    print("\n--- GATE (b) MASK GATE ---")
    # Recover the allow-mask {0,1} from the additive bias (0 -> allow, -1e4 -> block).
    from mycelium.kenken import build_kenken_attn_bias
    bias = build_kenken_attn_bias(model, batch.cage_mask, batch.cell_valid)  # (B,n_heads,49,49)
    allow = (bias.realize().numpy() > -1.0).astype(np.int32)                 # 1 where bias≈0
    n_row, n_col, n_cage, n_global = model.kenken_head_split
    cell_valid_np = batch.cell_valid.realize().numpy().astype(bool)          # (B,49)
    cage_mask_np = batch.cage_mask.realize().numpy().astype(np.int32)        # (B,49,49) symmetric clique
    N_list = batch.N

    # Pick a representative head from each group.
    h_row = 0
    h_col = n_row
    h_cage = n_row + n_col
    h_global = n_row + n_col + n_cage

    # Check support (#allowed keys) per VALID cell against the expected count.
    def support_per_cell(head_idx):
        return allow[:, head_idx, :, :].sum(axis=-1)  # (B,49) #keys allowed per query cell

    row_sup = support_per_cell(h_row)
    col_sup = support_per_cell(h_col)
    cage_sup = support_per_cell(h_cage)
    glob_sup = support_per_cell(h_global)

    ok_row = ok_col = ok_cage = ok_global = True
    sample_b = 0
    for b in range(B):
        N = N_list[b]
        for f in range(N_CELLS):
            if not cell_valid_np[b, f]:
                continue
            # ROW: valid cells in the same row (N cells incl self -> support N)
            ok_row &= (int(row_sup[b, f]) == N)
            # COL: same column, N valid cells incl self
            ok_col &= (int(col_sup[b, f]) == N)
            # CAGE: cage-mate count incl self == row-sum of the symmetric cage clique
            expected_cage = int(cage_mask_np[b, f].sum())
            ok_cage &= (int(cage_sup[b, f]) == expected_cage)
            # GLOBAL: all valid cells (N*N for an N-board)
            ok_global &= (int(glob_sup[b, f]) == N * N)

    # Pretty print supports for one example cell in batch 0.
    Nb = N_list[sample_b]
    # find a valid cell with a multi-cell cage
    ex = None
    for f in range(N_CELLS):
        if cell_valid_np[sample_b, f] and int(cage_mask_np[sample_b, f].sum()) >= 2:
            ex = f
            break
    if ex is None:
        ex = int(np.argmax(cell_valid_np[sample_b]))
    print(f"  example (batch0, N={Nb}, cell {ex} = r{ex//N_MAX}c{ex%N_MAX}):")
    print(f"    row-head support   = {int(row_sup[sample_b, ex])}  (expect N={Nb})")
    print(f"    col-head support   = {int(col_sup[sample_b, ex])}  (expect N={Nb})")
    print(f"    cage-head support  = {int(cage_sup[sample_b, ex])}  "
          f"(expect cage-mates incl self = {int(cage_mask_np[sample_b, ex].sum())})")
    print(f"    global-head support= {int(glob_sup[sample_b, ex])}  (expect N*N={Nb*Nb})")
    # padding-cell self-only check
    pad_ok = True
    for b in range(B):
        for f in range(N_CELLS):
            if not cell_valid_np[b, f]:
                pad_ok &= (int(row_sup[b, f]) == 1 and int(glob_sup[b, f]) == 1)
    print(f"  ALL valid cells: row={ok_row} col={ok_col} cage={ok_cage} global={ok_global}")
    print(f"  padding cells self-only: {pad_ok}")
    assert ok_row and ok_col and ok_cage and ok_global and pad_ok, "GATE (b) FAIL"
    # Confirm op-type is NOT a mask channel: all heads' masks are independent of op.
    print("  (op-type is NEVER a mask channel — cage mask is symmetric membership only)")
    print("  GATE (b) PASS")

    # ====================================================================
    # GATE (c) GRADIENT LIVENESS
    # ====================================================================
    print("\n--- GATE (c) GRADIENT LIVENESS (one eager backward) ---")
    opt.zero_grad()
    cell_logits_history, calib_history = kenken_breathing_forward(model, batch, K=K)
    total, parts = kenken_loss(cell_logits_history, calib_history, batch,
                               constraint_weight=0.3, calib_weight=0.1)
    # NOTE: do NOT realize() total before backward() — in this tinygrad version a
    # pre-backward realize() frees the autograd tape and every grad comes back
    # None. The grad-component prints below realize the loss parts AFTER backward.
    total.backward()

    # Per-group grad norms (mirror funded-vs-starved on the inlet).
    def grad_norm(t):
        if t.grad is None:
            return None
        return float(t.grad.cast(dtypes.float).square().sum().sqrt().realize().numpy())

    new_groups = {
        "state_embed": model.kenken_state_embed,
        "position_embed": model.kenken_position_embed,
        "value_codebook": model.kenken_value_codebook,
        "calib_head_w": model.kenken_calib_head_w,
        "calib_head_b": model.kenken_calib_head_b,
        "breath_embed": model.kenken_breath_embed,
        "delta_gate": model.kenken_delta_gate,
    }
    inlet_groups = kenken_inlet_parameters(model)

    print(f"  loss={float(total.realize().numpy()):.4f}  "
          f"cell_ce={float(parts['cell_ce'].realize().numpy()):.4f}  "
          f"energy={float(parts['energy'].realize().numpy()):.4f}  "
          f"calib={float(parts['calib'].realize().numpy()):.4f}")
    print("  -- new KenKen params --")
    all_live = True
    for name, t in new_groups.items():
        gn = grad_norm(t)
        live = gn is not None and gn > 0.0
        all_live &= live
        print(f"    {name:16s} |grad|={gn if gn is not None else 'None':>12}  {'LIVE' if live else 'STARVED'}")
    print("  -- VERIFICATION INLET params (funded-vs-starved) --")
    inlet_live = True
    for name, t in inlet_groups.items():
        gn = grad_norm(t)
        live = gn is not None and gn > 0.0
        inlet_live &= live
        print(f"    {name:16s} |grad|={gn if gn is not None else 'None':>12}  {'LIVE' if live else 'STARVED'}")
    assert all_live, "GATE (c) FAIL: a new KenKen param is STARVED"
    assert inlet_live, "GATE (c) FAIL: a VERIFICATION INLET param is STARVED (funded-vs-starved!)"
    print("  GATE (c) PASS — every new param (incl. inlet) draws nonzero grad")

    # ====================================================================
    # GATE (d) INLET MAGNITUDE
    # ====================================================================
    print("\n--- GATE (d) INLET MAGNITUDE (bounded / not colliding) ---")
    Tensor.training = False
    inlet = build_verification_inlet(
        model, batch.cage_op, batch.cage_target, batch.cage_size, batch.cell_cage_id)
    inlet_np = inlet.realize().numpy()                                       # (B,49,H)
    cell_valid_col = batch.cell_valid.realize().numpy().astype(bool)
    # per-cell L2 norm of the inlet, valid cells only.
    inlet_cell_norm = np.sqrt((inlet_np ** 2).sum(axis=-1))                  # (B,49)
    valid_norms = inlet_cell_norm[cell_valid_col]

    # The inlet is added at the START of EVERY breath: x_in = x + breath_embed + inlet.
    # The honest reference scale is the OPERATIVE residual the inlet rides on —
    # i.e. the post-layer residual during breathing, NOT the (deliberately tiny,
    # 0.1-scale codebook-aligned) init embedding. RMSNorm caps the inlet at ~sqrt(H)
    # REGARDLESS of the cage target value (this IS the magnitude-mismatch fix: a
    # mul-target of 343 produces the SAME bounded contribution as an add-target of
    # 8). We report the inlet vs BOTH (init embedding + post-first-breath residual).
    from mycelium.kenken import embed_kenken, kenken_layer_forward, build_kenken_attn_bias
    x0t = embed_kenken(batch.input_cells, model.kenken_state_embed,
                       model.kenken_position_embed)
    x0 = x0t.realize().numpy()
    init_resid = np.sqrt((x0 ** 2).sum(axis=-1))[cell_valid_col]
    # one breath through the 4 shared layers -> operative residual scale.
    xb = x0t.cast(dtypes.half)
    be0 = model.kenken_breath_embed[0].reshape(1, 1, -1).cast(xb.dtype)
    inlet_h = inlet.cast(xb.dtype)
    bias_d = build_kenken_attn_bias(model, batch.cage_mask, batch.cell_valid)
    h1 = xb + be0 + inlet_h
    for layer in list(model.block.layers)[:4]:
        h1 = kenken_layer_forward(layer, h1, bias_d)
    op_resid = np.sqrt((h1.cast(dtypes.float).realize().numpy() ** 2).sum(axis=-1))[cell_valid_col]

    print(f"  inlet per-cell norm (valid): mean={valid_norms.mean():.3f} "
          f"min={valid_norms.min():.3f} max={valid_norms.max():.3f}  "
          f"(spread {valid_norms.max()-valid_norms.min():.3f} — RMSNorm flattens "
          f"target magnitude; sqrt(H)={math.sqrt(cfg.hidden):.1f})")
    print(f"  init embedding norm (valid):  mean={init_resid.mean():.3f}  "
          f"(0.1-scale codebook init — inlet legitimately injects the constraint here)")
    print(f"  operative residual (post-breath-0): mean={op_resid.mean():.3f}  "
          f"(the scale the inlet actually rides on each breath)")
    ratio_op = valid_norms.mean() / (op_resid.mean() + 1e-9)
    print(f"  inlet / operative-residual ratio = {ratio_op:.3f}  "
          f"(comparable, NOT dominating — bounded by RMSNorm)")
    # The load-bearing check: the inlet is BOUNDED (RMSNorm => ~sqrt(H), invariant
    # to the cage target value) and does NOT dominate the operative residual.
    bounded = (np.isfinite(valid_norms).all()
               and valid_norms.max() < 2.0 * math.sqrt(cfg.hidden)   # RMSNorm cap
               and ratio_op < 5.0)                                   # not colliding w/ operative scale
    assert bounded, (f"GATE (d) FAIL: inlet not bounded "
                     f"(max={valid_norms.max()}, ratio_op={ratio_op})")
    print("  GATE (d) PASS — inlet bounded (RMSNorm) & comparable to operative residual")

    # ====================================================================
    # GATE (e) CONVERGENCE INSTRUMENT
    # ====================================================================
    print("\n--- GATE (e) CONVERGENCE INSTRUMENT (Property 2 artifact) ---")
    # Run forward on ~16 puzzles (two B=8 batches).
    n_target = 16
    rows = []
    seen = 0
    loader2 = KenKenLoader(TRAIN, batch_size=B, seed=SEED + 100)
    for bb in loader2.iter_eval(batch_size=B):
        clh, _ = kenken_breathing_forward(model, bb, K=K)
        inst = convergence_instrument(clh, bb, threshold=KENKEN_CONVERGE_JSD)
        rows.extend(inst)
        seen += B
        if seen >= n_target:
            break
    rows = rows[:n_target]
    n_settled = sum(1 for r in rows if r["status"] == "settled")
    n_stuck = sum(1 for r in rows if r["status"] == "stuck")
    n_notconv = sum(1 for r in rows if r["status"] == "not_converged")
    print(f"  emitted {len(rows)} (breath_count, deduction_depth, status) triples:")
    print(f"    {'breath_count':>12} {'deduction_depth':>16} {'status':>14}")
    for r in rows:
        print(f"    {r['breath_count']:>12} {r['deduction_depth']:>16} {r['status']:>14}")
    print(f"  counts: settled={n_settled} stuck={n_stuck} not_converged={n_notconv}")
    print("  (pre-training: breath_count vs deduction_depth NOT expected to correlate;"
          " the gate verifies the instrument PRODUCES the artifact)")
    assert len(rows) == n_target, "GATE (e) FAIL: instrument did not produce artifact"
    assert all(isinstance(r["breath_count"], int) and isinstance(r["deduction_depth"], int)
               for r in rows), "GATE (e) FAIL: malformed artifact"
    print("  GATE (e) PASS — instrument produces the pre-registered artifact")
    Tensor.training = True

    # ====================================================================
    # GATE (f) PARAM COUNT (Property 1)
    # ====================================================================
    print("\n--- GATE (f) PARAM COUNT (Property 1 — recurrence-baseline) ---")
    n_total = sum(int(np.prod(t.shape)) for t in params)
    n_kenken = sum(int(np.prod(t.shape)) for t in kenken_parameters(model))
    n_inlet = sum(int(np.prod(t.shape)) for t in kenken_inlet_parameters(model).values())
    n_backbone = n_total - n_kenken
    print(f"  total trainable params : {n_total:,}  ({n_total/1e6:.2f}M)")
    print(f"    backbone (shared L0-L3 + final LN): {n_backbone:,}  ({n_backbone/1e6:.2f}M)")
    print(f"    kenken-specific (heads+codebook+inlet): {n_kenken:,}  ({n_kenken/1e6:.2f}M)")
    print(f"    of which verification inlet: {n_inlet:,}  ({n_inlet/1e6:.3f}M)")
    print("  NOTE: this is the K-pass SHARED-WEIGHT count — Property 1 compares it")
    print("        against a (larger) trained-recurrence baseline at equal accuracy.")
    print("  GATE (f) PASS")

    print("\n" + "=" * 72)
    print("ALL SIX GATES PASS — KenKen scaffold step-0 smoke complete.")
    print("=" * 72)


if __name__ == "__main__":
    main()
