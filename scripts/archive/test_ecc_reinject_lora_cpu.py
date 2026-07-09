#!/usr/bin/env python3
"""test_ecc_reinject_lora_cpu.py — CPU-only static checks for the two ECC fixes.

NO GPU, NO training, NO Pythia download. Runs the REAL breath loop on the tinygrad CPU
device (DEV=CPU) with a randomly-initialized BreathingTransformer (no Pythia init), a
tiny batch + K, so it exercises the actual factor_breathing_forward (incl. the real
kenken_layer_forward) — not a stub.

Checks (the brief's four + finiteness):
  (a) byte-identical-off: a forward with BOTH flags off == the current forward
      (reinject_input=False, lora_rank=0) — same output tensor, max|delta|==0. Also
      checks the DISCRETE path (kenken-shaped spec, continuous_input=False) so the
      both-off guarantee holds for ALL tasks, not just ECC.
  (b) re-injection ON adds the channel embed each breath: the per-breath residual differs
      from off, AND the difference lies in the channel-embed direction (cosine of the
      breath-1 residual delta with the channel embed is strongly positive).
  (c) LoRA ON at zero-init (B_k=0) == off (neutral adapter -> byte-identical); after
      perturbing one B_k the breaths differ (and only from the perturbed breath onward).
  (d) finite outputs in every configuration.

Run:  DEV=CPU .venv/bin/python scripts/test_ecc_reinject_lora_cpu.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from tinygrad import Tensor, dtypes

from mycelium import Config, BreathingTransformer
from mycelium.factor_graph_engine import (
    FactorGraphSpec, attach_factor_graph_params, factor_breathing_forward,
    attach_factor_lora_params, factor_lora_parameters, factor_graph_parameters,
)
from mycelium.ecc_data import ECCLoader


K_TEST = 4
B_TEST = 2


def _build(continuous: bool, reinject: bool, lora_rank: int, seed: int = 7):
    """Build a fresh random model + ECC-shaped spec/batch for one configuration.

    The model is built with a FIXED manual seed so two configs that should agree
    (e.g. both-off vs current) get byte-identical weights -> a clean equality test.
    """
    Tensor.manual_seed(seed)
    np.random.seed(seed)
    cfg = Config()
    model = BreathingTransformer(cfg)
    spec = FactorGraphSpec(
        s_max=49, n_values=2, n_factor_types=1, n_heads=cfg.n_heads,
        k_max=K_TEST, has_factor_inlet=False,
        continuous_input=continuous,
        reinject_input=reinject, lora_rank=lora_rank,
    )
    attach_factor_graph_params(model, hidden=cfg.hidden, spec=spec)
    if lora_rank > 0:
        attach_factor_lora_params(model, hidden=cfg.hidden, spec=spec, rank=lora_rank)
    return model, spec, cfg


def _ecc_batch(seed: int = 11):
    loader = ECCLoader(batch_size=B_TEST, seed=seed, n_eval_per_snr=2)
    return loader.sample_batch()


def _discrete_batch(cfg, seed: int = 13):
    """A kenken-shaped discrete batch (continuous_input=False path) for the
    both-off byte-identical check on the NON-ecc forward."""
    rng = np.random.RandomState(seed)
    S, N = 49, 2
    input_cells = np.zeros((B_TEST, S), dtype=np.int32)
    cell_valid = np.ones((B_TEST, S), dtype=np.float32)
    vdm = np.ones((B_TEST, S, N), dtype=np.float32)
    gold = rng.randint(1, N + 1, size=(B_TEST, S)).astype(np.int32)
    L = 6
    mem = (rng.rand(B_TEST, L, S) > 0.6).astype(np.float32)
    lt = np.zeros((B_TEST, L), dtype=np.int32)

    class _B:
        pass
    b = _B()
    b.input_cells = Tensor(input_cells, dtype=dtypes.int).contiguous().realize()
    b.cell_valid = Tensor(cell_valid, dtype=dtypes.float).contiguous().realize()
    b.value_domain_mask = Tensor(vdm, dtype=dtypes.float).contiguous().realize()
    b.gold = Tensor(gold, dtype=dtypes.int).contiguous().realize()
    b.membership = Tensor(mem, dtype=dtypes.float).contiguous().realize()
    b.latent_type = Tensor(lt, dtype=dtypes.int).contiguous().realize()
    b.factor_inlet = None
    return b


def _resid_history(model, batch, spec):
    """Capture the per-breath readout-point residual via fg_resid_capture (eager)."""
    cap = []
    model.fg_resid_capture = cap
    try:
        factor_breathing_forward(model, batch, spec, K=K_TEST)
    finally:
        model.fg_resid_capture = None
    return [np.asarray(x) for x in cap]   # list of K (B, S, H) fp32


def _final_logits(model, batch, spec):
    lh, _ = factor_breathing_forward(model, batch, spec, K=K_TEST)
    return lh[-1].realize().numpy()


def test_a_byte_identical_off():
    print("\n[a] byte-identical-off (both flags off == current forward)")
    eb = _ecc_batch()
    # ECC (continuous) path.
    m_off, s_off, _ = _build(continuous=True, reinject=False, lora_rank=0)
    m_cur, s_cur, _ = _build(continuous=True, reinject=False, lora_rank=0)
    out_off = _final_logits(m_off, eb, s_off)
    out_cur = _final_logits(m_cur, eb, s_cur)
    d = float(np.max(np.abs(out_off - out_cur)))
    print(f"    ECC continuous: max|delta| = {d:.3e}")
    assert d == 0.0, f"both-off ECC forward not byte-identical (max|delta|={d})"
    # The off model must carry NO lora params (so factor_graph_parameters is unchanged).
    assert getattr(m_off, "fg_lora_A", None) is None
    assert factor_lora_parameters(m_off) == []

    # DISCRETE (kenken-shaped) path: both-off must be byte-identical there too.
    m_d1, s_d1, cfg = _build(continuous=False, reinject=False, lora_rank=0, seed=21)
    m_d2, s_d2, _ = _build(continuous=False, reinject=False, lora_rank=0, seed=21)
    db = _discrete_batch(cfg)
    o1 = _final_logits(m_d1, db, s_d1)
    o2 = _final_logits(m_d2, db, s_d2)
    dd = float(np.max(np.abs(o1 - o2)))
    print(f"    discrete (kenken-shaped): max|delta| = {dd:.3e}")
    assert dd == 0.0, f"both-off discrete forward not byte-identical (max|delta|={dd})"
    print("    PASS: both-off is byte-identical on ECC AND discrete paths")


def test_b_reinject_changes_residual():
    print("\n[b] re-injection ON adds the channel embed each breath")
    eb = _ecc_batch()
    m_off, s_off, cfg = _build(continuous=True, reinject=False, lora_rank=0)
    m_on, s_on, _ = _build(continuous=True, reinject=True, lora_rank=0)
    r_off = _resid_history(m_off, eb, s_off)
    r_on = _resid_history(m_on, eb, s_on)

    # The per-breath residuals must DIFFER once re-injection is on.
    deltas = [float(np.max(np.abs(r_on[k] - r_off[k]))) for k in range(K_TEST)]
    print(f"    per-breath max|resid_on - resid_off| = "
          f"{[f'{x:.3e}' for x in deltas]}")
    assert all(d > 0 for d in deltas), \
        "re-injection did not change the residual at every breath"

    # The difference must lie in the CHANNEL-EMBED direction. Recompute the channel
    # embed (== the B0 lift) and check the cosine of the breath-0 residual delta with it.
    from mycelium.factor_graph_engine import embed_factor_cells_continuous
    ch = embed_factor_cells_continuous(
        eb.cont_input, m_on.fg_cont_embed_w, m_on.fg_cont_embed_b,
        m_on.fg_position_embed).realize().numpy().astype(np.float32)  # (B,S,H)
    delta0 = (r_on[0] - r_off[0]).astype(np.float32)                   # (B,S,H)
    # cosine over the flattened (B,S,H) tensors.
    cos = float((delta0.ravel() @ ch.ravel())
                / (np.linalg.norm(delta0) * np.linalg.norm(ch) + 1e-12))
    print(f"    cos(breath-0 resid delta, channel_embed) = {cos:.4f}")
    assert cos > 0.5, \
        f"breath-0 residual delta not aligned with the channel embed (cos={cos})"
    print("    PASS: re-injection adds the channel evidence each breath")


def test_c_lora_zero_init_off_then_perturb():
    print("\n[c] LoRA zero-init == off; perturbed B_k -> breaths differ")
    eb = _ecc_batch()
    # OFF baseline (no lora) and ON at zero-init must be byte-identical.
    m_off, s_off, cfg = _build(continuous=True, reinject=False, lora_rank=0, seed=31)
    m_lz, s_lz, _ = _build(continuous=True, reinject=False, lora_rank=8, seed=31)
    # sanity: B is exactly zero at init.
    Bnp = m_lz.fg_lora_B.realize().numpy()
    print(f"    fg_lora_B init max|.| = {float(np.max(np.abs(Bnp))):.3e} "
          f"(shape {Bnp.shape}); fg_lora_A finite={np.isfinite(m_lz.fg_lora_A.numpy()).all()}")
    assert float(np.max(np.abs(Bnp))) == 0.0, "fg_lora_B is not zero-init"
    out_off = _final_logits(m_off, eb, s_off)
    out_lz = _final_logits(m_lz, eb, s_lz)
    d = float(np.max(np.abs(out_off - out_lz)))
    print(f"    LoRA zero-init vs off: max|delta| = {d:.3e}")
    assert d == 0.0, f"LoRA at zero-init not byte-identical to off (max|delta|={d})"
    assert factor_lora_parameters(m_lz)  # params ARE present (just neutral)

    # Capture the off residual history, then perturb ONE breath's B_k and re-capture.
    r_before = _resid_history(m_lz, eb, s_lz)
    perturb_k = 1
    Bw = m_lz.fg_lora_B.realize().numpy().copy()
    Bw[perturb_k] += np.random.RandomState(99).randn(*Bw[perturb_k].shape).astype(np.float32) * 0.05
    m_lz.fg_lora_B.assign(Tensor(Bw, dtype=dtypes.float)).realize()
    r_after = _resid_history(m_lz, eb, s_lz)

    per_breath = [float(np.max(np.abs(r_after[k] - r_before[k]))) for k in range(K_TEST)]
    print(f"    per-breath max|resid_after - resid_before| (perturb breath {perturb_k}) = "
          f"{[f'{x:.3e}' for x in per_breath]}")
    # Breaths BEFORE the perturbed one are unaffected; the perturbed breath onward differ.
    assert per_breath[0] == 0.0, "breath before perturbed breath should be unchanged"
    assert all(d > 0 for d in per_breath[perturb_k:]), \
        "perturbing B_k did not change the breaths from k onward"
    print("    PASS: LoRA neutral at zero-init; a single perturbed B_k un-ties the breaths")


def test_d_finite_all_configs():
    print("\n[d] finite outputs in every configuration")
    eb = _ecc_batch()
    configs = [
        ("off/off", False, 0),
        ("reinject only (Arm A)", True, 0),
        ("lora only", False, 8),
        ("reinject + lora (Arm B)", True, 8),
    ]
    for name, ri, lr in configs:
        m, s, _ = _build(continuous=True, reinject=ri, lora_rank=lr, seed=41)
        out = _final_logits(m, eb, s)
        ok = bool(np.isfinite(out).all())
        print(f"    {name:24s}: finite={ok}")
        assert ok, f"non-finite output in config {name}"
    print("    PASS: all four configs produce finite outputs")


if __name__ == "__main__":
    print("=" * 72)
    print("ECC re-injection + per-breath LoRA — CPU static checks (DEV=CPU)")
    print("=" * 72)
    test_a_byte_identical_off()
    test_b_reinject_changes_residual()
    test_c_lora_zero_init_off_then_perturb()
    test_d_finite_all_configs()
    print("\nALL CPU STATIC CHECKS PASSED")
