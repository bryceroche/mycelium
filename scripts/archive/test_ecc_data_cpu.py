#!/usr/bin/env python3
"""test_ecc_data_cpu.py — CPU-only static unit test for the ECC / neural-BP port.

NO GPU, NO training, NO Pythia backbone. Runs on the tinygrad CPU device (DEV=CPU).
Asserts the load-bearing data contracts + the continuous-input embed's finiteness.

Run:  DEV=CPU .venv/bin/python scripts/test_ecc_data_cpu.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from tinygrad import Tensor, dtypes

from mycelium.ecc_data import (
    ECCLoader, build_bch_31_16, random_codewords, channel_llr,
    encode_instance, values_to_bits, bits_to_values,
    S_MAX, N_VALUES, N_BITS,
)
from mycelium.factor_graph_engine import (
    FactorGraphSpec, attach_factor_graph_params, embed_factor_cells_continuous,
)


def test_batch_shapes_and_codeword():
    """(1) Build one ECC batch; assert shapes; assert H·(gold bits)=0 mod 2."""
    loader = ECCLoader(batch_size=6, seed=11, n_eval_per_snr=10)
    H = loader.H
    b = loader.sample_batch()

    # Shape contract (FactorGraphBatch + cont_input).
    assert tuple(b.cont_input.shape) == (6, S_MAX), b.cont_input.shape
    assert tuple(b.input_cells.shape) == (6, S_MAX), b.input_cells.shape
    assert tuple(b.cell_valid.shape) == (6, S_MAX), b.cell_valid.shape
    assert tuple(b.value_domain_mask.shape) == (6, S_MAX, N_VALUES)
    assert tuple(b.gold.shape) == (6, S_MAX), b.gold.shape
    assert tuple(b.membership.shape) == (6, loader.n_checks_max, S_MAX)
    assert tuple(b.latent_type.shape) == (6, loader.n_checks_max)

    # cell_valid: exactly N_BITS real cells, rest pad.
    cv = b.cell_valid.numpy()
    assert np.all(cv[:, :N_BITS] == 1.0) and np.all(cv[:, N_BITS:] == 0.0)

    # input_cells all 0 (no GIVEN cells -> every real bit-cell is supervised).
    assert np.all(b.input_cells.numpy() == 0)

    # gold IS a codeword: H · (gold-1 on real cells) == 0 mod 2.
    gold_np = b.gold.numpy().astype(np.int32)
    bits = values_to_bits(gold_np[:, :N_BITS])              # (B, 31)
    synd = (bits @ H.T) & 1                                  # (B, m)
    assert not synd.any(), "gold is NOT a codeword (H·gold_bits != 0)!"
    print(f"  (1) batch shapes OK; H·gold_bits=0 mod 2 for all {bits.shape[0]} "
          f"instances: OK")


def test_llr_sign_convention():
    """(2) The LLR sign convention matches the gold: high-SNR LLR sign -> correct bit.

    BPSK: bit 0 -> x=+1 -> LLR>0; bit 1 -> x=-1 -> LLR<0. So (LLR<0) == bit at high SNR.
    """
    code = build_bch_31_16()
    rng = np.random.default_rng(99)
    C = random_codewords(code["G_sys"], 64, rng)
    LLR, _ = channel_llr(C, 20.0, code["k"] / code["n"], rng)  # high SNR
    hard = (LLR < 0).astype(np.uint8)
    agree = float((hard == C).mean())
    assert agree > 0.99, f"LLR-sign vs gold agreement {agree} too low"

    # Also check the GOLD-VALUE alignment: a high-SNR hard decision argmax over the
    # continuous embed should align with gold value (bit+1). The sign convention is
    # the binding contract; here we confirm it through one encoded instance.
    rec = encode_instance(C[0], LLR[0], code["H_min"], code["H_min"].shape[0])
    decided_bits = (rec["cont_input"][:N_BITS] < 0).astype(np.int32)
    assert np.array_equal(decided_bits, C[0]), "encoded LLR sign != gold bits"
    assert np.array_equal(bits_to_values(decided_bits), rec["gold"][:N_BITS])
    print(f"  (2) high-SNR LLR-sign vs gold agreement={agree:.4f} (>0.99); "
          f"sign->value alignment: OK")


def test_continuous_embed_finite_cpu():
    """(3) The continuous embed produces FINITE hidden on CPU.

    Attach the fg params for the ECC spec (continuous_input=True) on a tiny stub
    'model', then run embed_factor_cells_continuous on a real LLR batch and assert
    the output is finite (no NaN/Inf) and the right shape. No Pythia backbone needed
    — we test the additive core edit in isolation.
    """
    H_DIM = 32  # tiny hidden for the CPU test (the embed is dim-agnostic)
    spec = FactorGraphSpec(s_max=S_MAX, n_values=N_VALUES, n_factor_types=1,
                           n_heads=4, k_max=4, has_factor_inlet=False,
                           continuous_input=True)

    class _M:
        pass
    model = _M()
    attach_factor_graph_params(model, hidden=H_DIM, spec=spec)

    # The cont-embed params MUST have been attached (continuous_input=True).
    assert getattr(model, "fg_cont_embed_w", None) is not None, \
        "fg_cont_embed_w not attached for continuous_input spec!"
    assert tuple(model.fg_cont_embed_w.shape) == (1, H_DIM)
    assert tuple(model.fg_cont_embed_b.shape) == (H_DIM,)
    # cont_embed_w init at codebook scale (~0.1 std) so the lift bootstraps.
    w_std = float(model.fg_cont_embed_w.std().numpy())
    assert 0.03 < w_std < 0.3, f"cont_embed_w std {w_std} not at codebook scale"

    loader = ECCLoader(batch_size=5, seed=3, n_eval_per_snr=5)
    b = loader.sample_batch()
    out = embed_factor_cells_continuous(
        b.cont_input.cast(dtypes.float), model.fg_cont_embed_w,
        model.fg_cont_embed_b, model.fg_position_embed)
    out_np = out.numpy()
    assert out_np.shape == (5, S_MAX, H_DIM), out_np.shape
    assert np.isfinite(out_np).all(), "continuous embed produced non-finite values!"

    # A DISCRETE-input spec must NOT attach cont-embed params (byte-identical-off).
    spec_disc = FactorGraphSpec(s_max=S_MAX, n_values=2, n_factor_types=1,
                                n_heads=4, k_max=4, has_factor_inlet=False)
    assert spec_disc.continuous_input is False
    model2 = _M()
    attach_factor_graph_params(model2, hidden=H_DIM, spec=spec_disc)
    assert getattr(model2, "fg_cont_embed_w", None) is None, \
        "discrete-input spec WRONGLY attached cont-embed params!"
    print(f"  (3) continuous embed finite on CPU "
          f"(shape={out_np.shape}, w_std={w_std:.3f}); discrete spec attaches "
          f"NO cont params (byte-identical-off): OK")


if __name__ == "__main__":
    print("ECC CPU static unit test (DEV=CPU; no GPU, no training, no Pythia)")
    test_batch_shapes_and_codeword()
    test_llr_sign_convention()
    test_continuous_embed_finite_cpu()
    print("ALL ECC CPU STATIC CHECKS PASSED")
