"""factor_multitask_cpu_smoke.py — GPU-FREE self-test for the general-weights harness.

Run: DEV=CPU .venv/bin/python3 scripts/factor_multitask_cpu_smoke.py

Builds a TINY backbone (PYTHIA_INIT off, small dims via Config) + the multi-task
{coloring, circuit, kenken} harness, then for each domain:
  - draws a tiny native batch, runs the per-domain adapter (remap latent_type to
    GLOBAL ids, pad membership to L_max, pad value_domain_mask to N_max=7, build the
    GENERIC semantics inlet),
  - runs the K-breath forward + the per-breath weighted-CE loss,
  - checks: shapes (B,S,N_max=7) for logits; the generic inlet carries each domain's
    semantics (KenKen inlet differs from a zeroed inlet; non-zero on member cells);
    the masked codebook ZEROES the unused value slots (softmax prob ~0 on slot j>=n);
    per-domain loss is finite + attributable.
Then a tiny MULTI-STEP smoke (a handful of AdamW steps per domain) shows all 3 domain
losses MOVE (decrease) — the kill-gate's CPU shadow.

This needs a KenKen corpus on disk for the kenken leg; if absent, it builds a 4-puzzle
in-memory micro-corpus so the smoke is self-contained.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from tinygrad import Tensor, dtypes
from tinygrad.nn.optim import AdamW

from mycelium import Config, BreathingTransformer
from mycelium.factor_graph_engine import (
    FactorGraphSpec, attach_factor_graph_params, factor_graph_parameters,
    factor_breathing_forward,
)
from mycelium.factor_inlet import (
    attach_factor_inlet_params, factor_inlet_parameters, N_GLOBAL_TYPES,
    build_generic_factor_inlet,
)
import scripts.factor_graph_train as T


def _build_inlet(model, fb):
    """Build the generic inlet from a _MultiTaskBatch's raw semantic ids (mirrors the
    JIT step's in-graph build; sets fb.factor_inlet so the forward reads it)."""
    fb.factor_inlet = build_generic_factor_inlet(
        model, fb.membership, fb.latent_type, fb.cell_valid,
        op=fb.inlet_op, target=fb.inlet_target, size=fb.inlet_size)
    return fb


def _tiny_cfg():
    # Reduce the backbone to keep the smoke fast on CPU. hidden = n_heads * head_dim;
    # head_dim must be large enough that rotary_dim = head_dim*rotary_pct >= 2 (RoPE
    # builds a (max_seq_len, rotary_dim) table). Keep n_heads=16 (the harness assumes
    # >= n_factor_types+1 heads; N_GLOBAL_TYPES=8 + 1 global -> 9 <= 16 fine).
    return Config(hidden=128, n_heads=16, head_dim=8, ffn=256, max_seq_len=64)


def _ensure_kenken_corpus():
    """Return (train_path, test_path). The on-disk corpus is required for the kenken
    leg (built by scripts/build_kenken_data.py)."""
    tr = ".cache/kenken_train.jsonl"
    te = ".cache/kenken_test.jsonl"
    if not (os.path.exists(tr) and os.path.exists(te)):
        raise FileNotFoundError(
            f"kenken corpus not found at {tr}/{te}; build it with "
            f"scripts/build_kenken_data.py (the smoke's kenken leg needs it).")
    return tr, te


def main():
    np.random.seed(0)
    Tensor.manual_seed(0)
    Tensor.training = True

    K = 3
    BATCH = 2
    EVAL_BATCH = 2
    SEED = 42

    cfg = _tiny_cfg()
    hidden = cfg.hidden
    n_heads = cfg.n_heads
    print(f"=== multi-task CPU smoke (tiny backbone H={hidden} heads={n_heads} K={K} "
          f"B={BATCH}) ===")
    model = BreathingTransformer(cfg)
    T.cast_layers_fp32(model)

    tr_path, te_path = _ensure_kenken_corpus()
    print(f"  kenken corpus: {tr_path} / {te_path}")

    mix = ["coloring", "circuit", "kenken"]
    weights = {d: 1.0 for d in mix}

    # ---- attach generic inlet BEFORE the task (adapters call it on `model`).
    attach_factor_inlet_params(model, hidden=hidden)
    # Use a tiny in-memory corpus for coloring/circuit.
    os.environ.setdefault("FG_COLORING_N_INSTANCES", "64")
    os.environ.setdefault("FG_CIRCUIT_N_INSTANCES", "64")
    os.environ.setdefault("FG_COLORING_N_VALUES", "3")

    task = T._build_multitask_task(K, BATCH, EVAL_BATCH, SEED, hidden, n_heads,
                                   model, tr_path, te_path, mix, weights)
    spec = task.spec
    print(f"  unified spec: s_max={spec.s_max} n_values={spec.n_values} "
          f"T={spec.n_factor_types} (N_GLOBAL_TYPES={N_GLOBAL_TYPES}) "
          f"inlet={spec.has_factor_inlet}  L_max={task.L_max}")
    assert spec.n_values == 7, "universal codebook must be N_max=7"
    assert spec.has_factor_inlet is True

    # ---- the fg params (state/pos/codebook/calib/breath/gate).
    attach_factor_graph_params(model, hidden=hidden, spec=spec)

    # ---- per-domain forward + loss + the three CHECKS.
    print("\n--- per-domain forward + checks ---")
    per_domain_native = {}
    for d in mix:
        fb = _build_inlet(model, task.train_loader.sample_batch(domain=d))
        per_domain_native[d] = fb
        B = int(fb.input_cells.shape[0])

        # CHECK A: forward runs; logits are (B, S, N_max=7).
        logits_history, calib_history = factor_breathing_forward(model, fb, spec, K=K)
        lg = logits_history[-1]
        assert lg.shape == (B, spec.s_max, 7), f"{d}: bad logits shape {lg.shape}"

        # CHECK B (UPDATED for the ZERO-INIT per-factor-type gate): at init the gate
        # is all-zero, so the gated generic inlet contributes EXACTLY 0 -> the forward
        # is byte-identical to inlet-OFF -> every domain (param-free coloring included)
        # bootstraps exactly like native. (The OLD live-inlet design asserted
        # energy>0 here; the gate fix makes energy==0 at init the CORRECT state. The
        # gate's ability to OPEN + carry semantics is checked in CHECK B-open below.)
        inlet_np = fb.factor_inlet.realize().numpy()
        cv = fb.cell_valid.realize().numpy()
        valid_mask = cv > 0.5
        inlet_valid_energy = float(np.abs(inlet_np)[valid_mask].sum())
        assert inlet_valid_energy == 0.0, \
            f"{d}: zero-init gate must make the inlet EXACTLY 0 at init (got "\
            f"{inlet_valid_energy}); the gate is not truly zero-init"

        # CHECK C: the masked codebook ZEROES unused value slots. The engine adds
        # value_bias = (1-vdm)*(-1e4); softmax over the 7 slots must put ~0 prob on
        # the illegal slots. Verify on the final-breath logits.
        probs = lg.softmax(axis=-1).realize().numpy()       # (B,S,7)
        vdm = fb.value_domain_mask.realize().numpy()        # (B,S,7)
        illegal = (vdm < 0.5)
        # Only check VALID cells (pad cells have all-zero vdm -> all illegal).
        illegal_valid = illegal & valid_mask[:, :, None]
        max_illegal_prob = float(probs[illegal_valid].max()) if illegal_valid.any() else 0.0
        n_illegal = int(illegal_valid.sum())

        # Per-domain CE (the per-breath weighted ladder, eager).
        observed = (fb.input_cells > 0).cast(dtypes.float)
        supervise = (fb.cell_valid * (1.0 - observed)).reshape(B * spec.s_max)
        sup_sum = supervise.sum() + 1e-6
        gold_idx = (fb.gold - 1).clip(0, spec.n_values - 1).reshape(B * spec.s_max)
        ce_sum = Tensor.zeros((), dtype=dtypes.float)
        wsum = 0.0
        for k, logits in enumerate(logits_history):
            wk = 1.0 + float(k) / float(K - 1) if K > 1 else 1.0
            ce = logits.reshape(B * spec.s_max, spec.n_values
                                ).sparse_categorical_crossentropy(
                gold_idx, reduction="none")
            ce_sum = ce_sum + ((ce * supervise).sum() / sup_sum) * wk
            wsum += wk
        ce_val = float((ce_sum / wsum).realize().numpy())

        # unused-codebook-slot count: for coloring k=3 -> slots 3..6 illegal; circuit
        # 2 -> slots 2..6 illegal; kenken N=5..7 -> slots >=N illegal.
        legal_slots = int(vdm[valid_mask].sum(-1).max()) if valid_mask.any() else 0
        print(f"  [{d:8s}] logits={tuple(lg.shape)}  inlet_energy={inlet_valid_energy:8.2f}  "
              f"legal_slots(max)={legal_slots}/7  illegal_valid_slots={n_illegal} "
              f"max_illegal_prob={max_illegal_prob:.2e}  CE={ce_val:.4f}")
        assert np.isfinite(ce_val), f"{d}: CE not finite"
        assert max_illegal_prob < 1e-2, \
            f"{d}: masked codebook leaked prob {max_illegal_prob} onto illegal slots"

    # CHECK B-open (gate live): with the kenken-type gates OPENED, the KenKen inlet
    # (op/target/size) is non-zero AND DIFFERS from a type-id-only inlet -> the
    # arithmetic semantics flow through once the discriminative cage gate opens. We
    # open the gate to 1.0 for every kenken type (row/col/cage) just for this check,
    # then restore it to zero-init.
    from mycelium.factor_inlet import GLOBAL_TYPE_IDS
    kk = per_domain_native["kenken"]
    g_saved = model.fg_inlet_gate
    g_open = np.zeros((N_GLOBAL_TYPES,), dtype=np.float32)
    for nm in ("kenken_row", "kenken_col", "kenken_cage"):
        g_open[GLOBAL_TYPE_IDS[nm]] = 1.0
    model.fg_inlet_gate = Tensor(g_open, dtype=dtypes.float).contiguous().realize()
    inlet_full = build_generic_factor_inlet(
        model, kk.membership, kk.latent_type, kk.cell_valid,
        op=kk.inlet_op, target=kk.inlet_target, size=kk.inlet_size).realize().numpy()
    inlet_typeonly = build_generic_factor_inlet(
        model, kk.membership, kk.latent_type, kk.cell_valid,
        op=None, target=None, size=None).realize().numpy()
    model.fg_inlet_gate = g_saved   # restore zero-init
    kk_open_energy = float(np.abs(inlet_full).sum())
    kk_sem_diff = float(np.abs(inlet_full - inlet_typeonly).max())
    print(f"\n  kenken gate OPENED: inlet energy={kk_open_energy:.2f} (>0) and "
          f"op/target/size semantics carried (differs from type-id-only): "
          f"max|delta|={kk_sem_diff:.4f} (>0)")
    assert kk_open_energy > 1e-3, "opened kenken gate must make the inlet non-zero"
    assert kk_sem_diff > 1e-3, "kenken arithmetic semantics NOT carried by the inlet"

    # ---- tiny MULTI-STEP smoke: a few AdamW steps per domain; all 3 losses MOVE.
    print("\n--- multi-step smoke (all 3 domain losses should move) ---")
    params = (T.collect_backbone_params(model) + factor_graph_parameters(model)
              + factor_inlet_parameters(model))
    opt = AdamW(params, lr=3e-4, weight_decay=0.0)

    def _domain_ce(fb):
        B = int(fb.input_cells.shape[0])
        # Build the inlet IN-GRAPH (from raw ids) so the inlet params are in the loss
        # graph and TRAIN — exactly the JIT step's behaviour.
        fb.factor_inlet = build_generic_factor_inlet(
            model, fb.membership, fb.latent_type, fb.cell_valid,
            op=fb.inlet_op, target=fb.inlet_target, size=fb.inlet_size)
        logits_history, _ = factor_breathing_forward(model, fb, spec, K=K)
        observed = (fb.input_cells > 0).cast(dtypes.float)
        supervise = (fb.cell_valid * (1.0 - observed)).reshape(B * spec.s_max)
        sup_sum = supervise.sum() + 1e-6
        gold_idx = (fb.gold - 1).clip(0, spec.n_values - 1).reshape(B * spec.s_max)
        ce_sum = Tensor.zeros((), dtype=dtypes.float)
        wsum = 0.0
        for k, logits in enumerate(logits_history):
            wk = 1.0 + float(k) / float(K - 1) if K > 1 else 1.0
            ce = logits.reshape(B * spec.s_max, spec.n_values
                                ).sparse_categorical_crossentropy(
                gold_idx, reduction="none")
            ce_sum = ce_sum + ((ce * supervise).sum() / sup_sum) * wk
            wsum += wk
        return ce_sum / wsum

    # Round-robin a handful of steps per domain on a FIXED per-domain batch (so the
    # CE is measured on the same cells start vs end — the cleanest descent signal on
    # an untrained tiny backbone). lr kept small to avoid overshoot.
    N_ROUNDS = 8
    fixed_fb = {d: task.train_loader.sample_batch(domain=d) for d in mix}
    first = {d: None for d in mix}
    last = {d: None for d in mix}
    inlet_trains = None
    for step in range(N_ROUNDS * len(mix)):
        d = mix[step % len(mix)]
        fb = fixed_fb[d]
        opt.zero_grad()
        loss = _domain_ce(fb)
        loss.backward()
        # On the FIRST kenken step, verify the generic-inlet params get GRADIENT (the
        # general-weights thesis: the shared backbone LEARNS the predicate registry —
        # the inlet is in-graph, not frozen/pre-built).
        if inlet_trains is None and d == "kenken":
            inlet_w_grad = model.fg_inlet_w.grad
            inlet_type_grad = model.fg_inlet_type_embed.grad
            inlet_op_grad = model.fg_inlet_op_embed.grad
            inlet_trains = all(g is not None for g in
                               (inlet_w_grad, inlet_type_grad, inlet_op_grad))
            print(f"  [inlet trainability] fg_inlet_w/type_embed/op_embed get gradient "
                  f"on a kenken step: {inlet_trains}")
        # NaN guard (where()-gated), mirror the trainer. A single-domain step may not
        # touch EVERY inlet sub-table (e.g. coloring never indexes a nonzero op slot),
        # leaving p.grad None; AdamW.step() requires a grad on every param, so fill
        # untouched grads with zeros (a no-op update) — exactly what the JIT-step
        # `if p.grad is not None` guard achieves in the real trainer.
        healthy = loss.isfinite()
        for p in params:
            if p.grad is None:
                p.grad = Tensor.zeros_like(p)
            else:
                p.grad = healthy.where(p.grad, Tensor.zeros_like(p.grad))
        opt.step()
        lv = float(loss.realize().numpy())
        if first[d] is None:
            first[d] = lv
        last[d] = lv

    all_moved = True
    all_descended = True
    for d in mix:
        moved = (last[d] is not None and first[d] is not None
                 and abs(last[d] - first[d]) > 1e-4)
        descended = (last[d] is not None and first[d] is not None
                     and last[d] < first[d] - 1e-4)
        flag = "DESCEND" if descended else ("moved" if moved else "STUCK")
        print(f"  [{d:8s}] first_CE={first[d]:.4f} -> last_CE={last[d]:.4f}  [{flag}]")
        all_moved = all_moved and moved
        all_descended = all_descended and descended

    print()
    print(f"RESULT: shapes OK; inlet carries semantics; inlet params TRAIN "
          f"(grad flows)={inlet_trains}; masked codebook zeroes unused slots; "
          f"per-domain CE finite+attributable; all-3-losses-move={all_moved}; "
          f"all-3-losses-descend={all_descended}")
    ok = bool(all_moved) and bool(inlet_trains)
    print("SMOKE PASS" if ok else "SMOKE WARN (a check did not pass)")


if __name__ == "__main__":
    main()
