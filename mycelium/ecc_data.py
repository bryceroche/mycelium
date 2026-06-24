"""ecc_data.py — BCH(31,16) ECC / neural-BP data-gen + loader for the deducer.

THE §8.1 SOFT-CONSTRAINT FRONTIER, AS A FACTOR GRAPH.
-----------------------------------------------------
The deducer is "learned loopy BP on a factor graph". An ECC decode is *literally*
loopy BP on the code's Tanner graph: K breaths == K message-passing rounds, the
per-head attention masks (built from `membership`) == the parity-check H topology,
the per-cell continuous input == the channel LLR. This module turns BCH(31,16)
transmissions into the FactorGraphBatch contract the engine already consumes.

WHY BCH(31,16). The kill-gate (scripts/frontier_ecc_bp_gate.py) confirmed a real
CONVERGENT BP gap on the short, high-density (HDPC) regime where classical BP is
known sub-ML and neural-BP is a documented win (Nachmani 2016, Lugosch 2017). n=31
PADS into the engine's fixed 49-cell grid (31 real bit-cells + 18 pad); n=63 trips
the oracle's `assert S == 49`, so BCH(31,16) is the code that fits.

THE FACTOR GRAPH (one transmission == one instance):
  * S = 49 cells. Cells 0..30 = the 31 received BITS; cells 31..48 = padding.
  * N = 2 values (the codebook). BIT 0 -> value 1, BIT 1 -> value 2 (the engine's
    +1 cell-value convention: value 0 = "unknown"; legal values are 1..N).
  * input  = the per-bit channel LLR (a CONTINUOUS scalar per cell), NOT a discrete
    one-hot value. This is THE new thing: discrete tasks (kenken/coloring) one-hot
    the cell value through the codebook; ECC feeds a real LLR -> the continuous
    INPUT embed in the engine (gated by spec.continuous_input).
  * gold   = the 31 transmitted bits mapped to {1,2} (pad cells = 0).
  * membership = the parity-check H as factor rows: row r covers the bit-cells with
    H[r, j] == 1. Padded to 49 columns (the 18 pad cells are never in any check).
  * latent_type = a SINGLE factor type (= 0, "parity") for every real check row;
    pad rows get the global sentinel (>= n_factor_types). n_factor_types = 1.
  * no verification inlet (has_factor_inlet = False) — there is no per-factor
    arithmetic op to verify; the parity relation is carried entirely by the mask.

SIGN CONVENTION (load-bearing — verified by the static check).
  BPSK: bit c in {0,1} -> x = 1 - 2c in {+1,-1}; y = x + noise; LLR = 2y/sigma^2.
  So LLR > 0 favours BIT 0 (x = +1), LLR < 0 favours BIT 1. At high SNR the LLR
  SIGN equals the transmitted bit's negation: bit 0 -> LLR > 0, bit 1 -> LLR < 0.
  The gold value is (bit + 1), so a high-SNR hard decision (bit = LLR < 0) maps to
  the gold value via argmax of the continuous embed — the embed must preserve this
  monotone sign->value relationship (it is a learned 1->H linear, init at codebook
  scale so the task gradient bootstraps it; the per-position scalar->H bootstrap law).

NO new readout, NO new loss, NO inlet — the engine's N=2 value-codebook readout,
per-breath weighted-CE ladder (== BCE on the 2-way), calibration, and v45 reg
stack are ALL reused unchanged. This file is a pure data-layer add; the ONLY core
edit is the continuous input embed in factor_graph_engine.py.

REUSE: build_bch_31_16 + random_codewords + channel_llr live in the gate
(scripts/frontier_ecc_bp_gate.py); we import them so the code/channel are byte-
identical to the validated kill-gate (single source of truth).
"""
from __future__ import annotations

import os
import sys
from typing import Iterator

import numpy as np

from tinygrad import Tensor, dtypes

# Reuse the VALIDATED code construction + channel from the kill-gate (single source
# of truth). The gate lives in scripts/, so make it importable.
_SCRIPTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)
from frontier_ecc_bp_gate import (  # noqa: E402
    build_bch_31_16, random_codewords, channel_llr,
)


# ---------------------------------------------------------------------------
# Constants — the fixed 49-cell grid + the N=2 bit codebook.
# ---------------------------------------------------------------------------
S_MAX = 49          # engine grid (kenken_layer_forward asserts S == 49)
N_VALUES = 2        # the bit codebook: value 1 = bit 0, value 2 = bit 1
N_BITS = 31         # BCH(31,16): 31 transmitted bits -> cells 0..30
N_PAD = S_MAX - N_BITS  # 18 pad cells (31..48)


def bits_to_values(bits: np.ndarray) -> np.ndarray:
    """Map transmitted bits {0,1} -> engine cell values {1,2} (+1 indexing).

    bit 0 -> value 1, bit 1 -> value 2.  Inverse: value-1 == bit.
    """
    return (bits.astype(np.int32) + 1)


def values_to_bits(values: np.ndarray) -> np.ndarray:
    """Map engine cell values {1,2} -> bits {0,1} (the readout inverse)."""
    return (values.astype(np.int32) - 1)


# ---------------------------------------------------------------------------
# Per-instance encoding: one transmission -> the (49,)-padded numpy arrays.
# ---------------------------------------------------------------------------

def encode_instance(cw_bits: np.ndarray, llr_bits: np.ndarray,
                    H: np.ndarray, n_checks_max: int) -> dict:
    """Flatten ONE transmission onto the fixed (S_MAX, n_checks_max) grid.

    cw_bits  : (N_BITS,) uint8  transmitted codeword bits.
    llr_bits : (N_BITS,) float  per-bit channel LLR (the continuous input).
    H        : (m, N_BITS) uint8  parity-check matrix (m = n_checks rows).
    n_checks_max : int  padded number of factor rows (== H.shape[0] here; the
                   loader pins it so the JIT membership topology is static).

    Returns numpy arrays satisfying the FactorGraphBatch contract:
      cont_input        (S_MAX,)            float  per-cell LLR (pad cells = 0).
      input_cells       (S_MAX,)            int    all 0 (no GIVEN cells; the
                                                   decoder observes LLRs, not values
                                                   -> every real bit-cell is supervised).
      cell_valid        (S_MAX,)            float  1.0 on cells 0..30, 0.0 on pad.
      value_domain_mask (S_MAX, N_VALUES)   float  1 on both bit values for real
                                                   cells, 0 on pad cells.
      gold              (S_MAX,)            int    bit+1 on cells 0..30, 0 on pad.
      membership        (n_checks_max, S_MAX) float  parity rows (pad cols/rows 0).
      latent_type       (n_checks_max,)     int    0 (parity) on real rows,
                                                   N_FACTOR_TYPES (sentinel) on pad.
    """
    m = int(H.shape[0])
    assert H.shape[1] == N_BITS, f"H has {H.shape[1]} cols, expected {N_BITS}"
    assert m <= n_checks_max, f"m={m} exceeds n_checks_max={n_checks_max}"

    cont_input = np.zeros((S_MAX,), dtype=np.float32)
    cont_input[:N_BITS] = llr_bits.astype(np.float32)

    input_cells = np.zeros((S_MAX,), dtype=np.int32)  # no GIVEN cells

    cell_valid = np.zeros((S_MAX,), dtype=np.float32)
    cell_valid[:N_BITS] = 1.0

    value_domain_mask = np.zeros((S_MAX, N_VALUES), dtype=np.float32)
    value_domain_mask[:N_BITS, :] = 1.0  # both bit values legal on real cells

    gold = np.zeros((S_MAX,), dtype=np.int32)
    gold[:N_BITS] = bits_to_values(cw_bits)  # bit+1 in {1,2}

    # membership: factor row r covers the bit-cells in check r (H[r,j]==1), padded
    # to S_MAX columns. Pad rows (r >= m) stay all-zero (member of no cell).
    membership = np.zeros((n_checks_max, S_MAX), dtype=np.float32)
    membership[:m, :N_BITS] = H.astype(np.float32)

    # latent_type: 0 = parity for real rows; the global sentinel (= N_FACTOR_TYPES)
    # for pad rows so build_factor_attn_bias treats them as non-relations.
    latent_type = np.full((n_checks_max,), 1, dtype=np.int32)  # 1 == sentinel (T=1)
    latent_type[:m] = 0                                        # 0 == parity

    return dict(
        cont_input=cont_input,
        input_cells=input_cells,
        cell_valid=cell_valid,
        value_domain_mask=value_domain_mask,
        gold=gold,
        membership=membership,
        latent_type=latent_type,
    )


# ---------------------------------------------------------------------------
# Batch object (satisfies the FactorGraphBatch contract + carries cont_input).
# ---------------------------------------------------------------------------

class ECCBatch:
    """A realized batch of ECC tensors.

    Satisfies mycelium.factor_graph_engine.FactorGraphBatch (same tensor attrs +
    shapes/dtypes) PLUS the new continuous-input attribute `cont_input` (B, S) the
    engine reads when spec.continuous_input is True. No factor_inlet (ECC has no
    arithmetic to verify -> spec.has_factor_inlet = False).

    Tensor attributes
    -----------------
    cont_input        Tensor (B, S_MAX)            float — per-cell channel LLR.
    input_cells       Tensor (B, S_MAX)            int   — all 0 (no given cells).
    cell_valid        Tensor (B, S_MAX)            float — 1 real bit / 0 pad.
    value_domain_mask Tensor (B, S_MAX, N_VALUES)  float — 1 both values / 0 pad.
    gold              Tensor (B, S_MAX)            int   — bit+1 (1..2) / 0 pad.
    membership        Tensor (B, n_checks_max, S)  float — parity rows.
    latent_type       Tensor (B, n_checks_max)     int   — 0 parity / 1 pad sentinel.

    Python-side metadata (NOT tensors; eval-only)
    ---------------------------------------------
    snr_db            list[float]  — per-instance Eb/N0 (SNR-stratified eval x-axis).
    """

    def __init__(self, d: dict):
        self.cont_input: Tensor = d["cont_input"]
        self.input_cells: Tensor = d["input_cells"]
        self.cell_valid: Tensor = d["cell_valid"]
        self.value_domain_mask: Tensor = d["value_domain_mask"]
        self.gold: Tensor = d["gold"]
        self.membership: Tensor = d["membership"]
        self.latent_type: Tensor = d["latent_type"]
        # No verification inlet for ECC.
        self.factor_inlet = None
        # python-side metadata
        self.snr_db: list[float] = d.get("snr_db", [0.0] * int(self.gold.shape[0]))
        self.deduction_depth: list[int] = [0] * int(self.gold.shape[0])


# ---------------------------------------------------------------------------
# Loader — generates transmissions on the fly across an SNR range.
# ---------------------------------------------------------------------------

class ECCLoader:
    """In-memory ECC transmission generator for BCH(31,16) over BPSK/AWGN.

    Mirrors GraphColoringLoader / CircuitLoader: an IN-MEMORY generator (NOT path-
    based) that yields FactorGraphBatch-compatible batches directly. The static JIT
    topology width is n_checks_max == H.shape[0] (the parity-check has a fixed shape,
    so this is constant across the whole run).

    TRAIN: random codewords drawn fresh each batch, each instance at a random
    Eb/N0 uniformly in [snr_lo, snr_hi] (default 3..7 dB). Fresh draws = effectively
    infinite data (the standard neural-BP training regime).

    EVAL: a FIXED, SNR-STRATIFIED held-out set generated once at construction (a
    fixed seed distinct from train), `n_eval_per_snr` codewords at each integer SNR
    in eval_snrs (default 3,4,5,6,7). Held-out by SEED (train draws never reuse the
    eval RNG stream); the code is the same but the channel noise is independent.

    Parameters
    ----------
    H_kind      : "min" (the (15,31) BCH-bound H_min, rank 15) — the parity-check
                  used for BOTH membership AND the classical baseline (apples-to-
                  apples). "red" would use a redundant H (more rows); default "min".
    batch_size  : batch size.
    seed        : RNG seed (train sampler). Eval uses seed + 10007 (disjoint stream).
    snr_lo/snr_hi : train Eb/N0 range (dB).
    eval_snrs   : SNR points for the stratified eval set.
    n_eval_per_snr : eval codewords per SNR point.
    """

    def __init__(self, H_kind: str = "min", batch_size: int = 8, seed: int = 0,
                 snr_lo: float = 3.0, snr_hi: float = 7.0,
                 eval_snrs: tuple[float, ...] = (3.0, 4.0, 5.0, 6.0, 7.0),
                 n_eval_per_snr: int = 200):
        self.batch_size = int(batch_size)
        self.snr_lo = float(snr_lo)
        self.snr_hi = float(snr_hi)
        self.eval_snrs = tuple(float(s) for s in eval_snrs)
        self.n_eval_per_snr = int(n_eval_per_snr)

        code = build_bch_31_16()
        self.n = int(code["n"])               # 31
        self.k = int(code["k"])               # 16
        self.G_sys = code["G_sys"]            # (16, 31)
        if str(H_kind).lower() == "red":
            # Redundant H (low-weight dual codewords) — more, sparser checks for BP.
            from frontier_ecc_bp_gate import build_redundant_H_dual
            self.H = build_redundant_H_dual(code["H_min"], self.n)
        else:
            self.H = code["H_min"]            # (15, 31) full-rank BCH-bound H
        self.H_kind = str(H_kind).lower()
        assert self.n == N_BITS, f"code n={self.n} != N_BITS={N_BITS}"
        self.R = self.k / self.n              # code rate (for sigma in the channel)
        self.n_checks_max = int(self.H.shape[0])  # static JIT topology width

        self.rng = np.random.default_rng(seed)

        # Fixed, SNR-stratified held-out eval set (disjoint RNG stream).
        eval_rng = np.random.default_rng(seed + 10007)
        self.eval_records: list[dict] = []
        for snr in self.eval_snrs:
            C = random_codewords(self.G_sys, self.n_eval_per_snr, eval_rng)
            LLR, _ = channel_llr(C, snr, self.R, eval_rng)
            for b in range(self.n_eval_per_snr):
                rec = encode_instance(C[b], LLR[b], self.H, self.n_checks_max)
                rec["snr_db"] = float(snr)
                self.eval_records.append(rec)

        print(f"[ecc_data] BCH({self.n},{self.k}) H_kind={self.H_kind} "
              f"n_checks={self.n_checks_max} R={self.R:.3f}; train SNR "
              f"[{self.snr_lo},{self.snr_hi}]dB; eval {len(self.eval_records)} "
              f"({self.n_eval_per_snr}/SNR @ {list(self.eval_snrs)})", flush=True)

    # -- packing --------------------------------------------------------------

    def _stack(self, encs: list[dict]) -> ECCBatch:
        def stack_int(key):
            return Tensor(np.stack([e[key] for e in encs]).astype(np.int32),
                          dtype=dtypes.int).contiguous().realize()

        def stack_f(key):
            return Tensor(np.stack([e[key] for e in encs]).astype(np.float32),
                          dtype=dtypes.float).contiguous().realize()

        d = {
            "cont_input":        stack_f("cont_input"),
            "input_cells":       stack_int("input_cells"),
            "cell_valid":        stack_f("cell_valid"),
            "value_domain_mask": stack_f("value_domain_mask"),
            "gold":              stack_int("gold"),
            "membership":        stack_f("membership"),
            "latent_type":       stack_int("latent_type"),
            "snr_db":            [e.get("snr_db", 0.0) for e in encs],
        }
        return ECCBatch(d)

    # -- sampling -------------------------------------------------------------

    def sample_batch(self) -> ECCBatch:
        """Draw a fresh batch of random codewords at random SNRs in [lo, hi]."""
        C = random_codewords(self.G_sys, self.batch_size, self.rng)
        encs = []
        for b in range(self.batch_size):
            snr = float(self.rng.uniform(self.snr_lo, self.snr_hi))
            LLR, _ = channel_llr(C[b:b + 1], snr, self.R, self.rng)
            rec = encode_instance(C[b], LLR[0], self.H, self.n_checks_max)
            rec["snr_db"] = snr
            encs.append(rec)
        return self._stack(encs)

    def iter_eval(self, batch_size: int | None = None) -> Iterator[ECCBatch]:
        """Iterate the FIXED held-out eval set in order, padding the last batch."""
        bs = batch_size or self.batch_size
        recs = self.eval_records
        n = len(recs)
        for start in range(0, n, bs):
            batch = list(recs[start:start + bs])
            while len(batch) < bs:
                batch.append(recs[0])
            yield self._stack(batch)

    def __len__(self):
        return len(self.eval_records)


# ---------------------------------------------------------------------------
# CPU smoke (GPU-free): the data-layer self-checks (no engine, no training).
# ---------------------------------------------------------------------------

def _smoke() -> None:
    """Self-check: gold is a codeword (H·gold_bits = 0 mod 2), the LLR sign
    convention matches the gold, and the batch shapes are correct."""
    code = build_bch_31_16()
    H = code["H_min"]
    rng = np.random.default_rng(123)
    C = random_codewords(code["G_sys"], 8, rng)

    # (1) Every generated codeword satisfies H · c^T = 0 (mod 2).
    synd = (C @ H.T) & 1
    assert not synd.any(), "generated codewords are NOT in the code (H·c != 0)!"
    print("  [smoke] H · c^T = 0 (mod 2) for all codewords: OK")

    # (2) High-SNR LLR sign matches the gold: bit 0 -> LLR > 0, bit 1 -> LLR < 0.
    LLR, _ = channel_llr(C, 20.0, code["k"] / code["n"], rng)  # very high SNR
    hard = (LLR < 0).astype(np.uint8)                          # sign decision
    assert (hard == C).mean() > 0.99, "high-SNR LLR sign does not match gold bits!"
    print(f"  [smoke] high-SNR LLR-sign vs gold agreement: "
          f"{(hard == C).mean():.4f} (>0.99): OK")

    # (3) One encoded instance: shapes + the bit<->value mapping round-trips.
    rec = encode_instance(C[0], LLR[0], H, H.shape[0])
    assert rec["cont_input"].shape == (S_MAX,)
    assert rec["gold"].shape == (S_MAX,)
    assert rec["membership"].shape == (H.shape[0], S_MAX)
    # gold value-1 == bit on the real cells.
    assert np.array_equal(values_to_bits(rec["gold"][:N_BITS]), C[0]), \
        "bit<->value mapping does not round-trip!"
    # membership row r restricted to bit-cells equals H row r.
    assert np.array_equal(rec["membership"][:, :N_BITS], H.astype(np.float32))
    print("  [smoke] instance shapes + bit<->value + membership=H: OK")

    # (4) Loader batch.
    loader = ECCLoader(batch_size=4, seed=7, n_eval_per_snr=5)
    b = loader.sample_batch()
    assert tuple(b.cont_input.shape) == (4, S_MAX)
    assert tuple(b.membership.shape) == (4, loader.n_checks_max, S_MAX)
    assert tuple(b.value_domain_mask.shape) == (4, S_MAX, N_VALUES)
    print("  [smoke] ECCLoader.sample_batch shapes: OK")
    print("ecc_data smoke PASSED")


if __name__ == "__main__":
    _smoke()
