"""v105.5 factor graph breathing transformer — v105.4 + per-position FFN.

Extends v105.4 (per-position digit codebooks + magnitude head + hierarchical
IB + soft mag valid mask) with ONE additional change: a per-position FFN
block (PPFFN) inserted ONCE per breath, AFTER the 4 Pythia transformer
layers, BEFORE the rest of the per-breath pipeline.

  ADDITION 5 (v105.5) — PER-POSITION FFN BLOCK (PPFFN)
    The diagnosis on v105.4: linear probe found cos similarity 0.99+ at every
    digit position across problems. The 4 shared Pythia FFNs cannot distinguish
    positions even if attention preserves some per-position signal.
    Fix: 5 SEPARATE FFNs (one per digit position) applied after the 4 Pythia
    layers in each breath, ONLY to digit positions (factor positions pass
    through unchanged). LayerNorm gain/bias and FFN W_in/b_in/W_out/b_out are
    all per-position. W_out + b_out are ZERO-INITIALIZED so the PPFFN is
    inert at step 0 (forward is byte-identical to v105.4 with the same
    warm-start ckpt).

All four v105.4 additions are PRESERVED:

  ADDITION 1 — MAGNITUDE HEAD (4-way per-cell classification)
    For each variable cell, classify "how many digits is this number?":
      class 0 : 1-digit  (value < 10)
      class 1 : 2-digit  (value < 100)
      class 2 : 3-digit  (value < 1000)
      class 3 : 4+ digit (value >= 1000, capped)
    Magnitude probs weight a (4, hidden) centroid table to produce a
    magnitude_embed that is added to each digit's pre-codebook hidden state.

  ADDITION 2 — PER-POSITION DIGIT CODEBOOKS
    Was (10, hidden) shared; now (n_digits, 10, hidden) — one codebook per
    digit position.  Hidden vector for "digit d at position p" can differ
    across positions; the AR loop and reconstruction MSE both use the
    position-specific table.

  ADDITION 3 — HIERARCHICAL IB ATTENTION
    Two-level softmax: family attention (4 ops: ADD/SUB/MUL/DIV) gates leaf
    attention (32 IB leaves). family_centroids initialized from the mean of
    IB centroids within each family; leaf_to_family loaded from the IB tree.

  ADDITION 4 — SOFT MAGNITUDE-DERIVED VALID MASK FOR FACTOR_AUX
    The factor_aux loss uses (gold * soft) per-digit mask, where soft is
    derived from magnitude_softmax → class_to_valid (4, n_digits) mapping.
    Provides gradient on the magnitude head from per-NUMBER reconstruction.
    var_loss still uses the clean gold mask (preserves crisp per-digit CE).

Loss structure (UNCHANGED from v105.4):
  total = var_loss
        + 0.3 * magnitude_loss
        + 1.0 * factor_aux_loss
        + 0.05 * calib_loss
        + 0.01 * energy_loss

Env var gates:
  V105_5_TASK=1                — enable v105.5 forward path
  V105_5_K_MAX=8               — iterative-prefill breaths
  V105_5_N_DIGITS=5
  V105_5_N_MAX=16
  V105_5_F_MAX=8
  V105_5_WAIST=512
  V105_5_CODEBOOK_N=32         — IB leaf codebook entries
  V105_5_N_FAMILIES=4          — fixed at 4 (add/sub/mul/div)
  V105_5_IB_CENTROIDS=.cache/ib_centroids_gsm8k_partial.npz
  V105_5_IB_TREE=.cache/ib_tree_gsm8k_partial.json
  V105_5_ENERGY_WEIGHT=0.01
  V105_5_CALIB_WEIGHT=0.05
  V105_5_FACTOR_AUX_WEIGHT=1.0
  V105_5_MAGNITUDE_WEIGHT=0.3  — α for magnitude_loss
  V105_5_ROPE_BASE=10000
  V105_5_IB_INIT=1
  V105_5_WAIST_LORA_INIT=1
  V105_5_NUMBER_MSE_ONLY=0
  V105_5_AR_DIGITS=1
  V105_5_AR_COND_SCALE=0.5
  V105_5_AR_MSD_FIRST=0
  V105_5_PERPOS_FFN=1           — enable per-position FFN block (NEW v105.5)
  V105_5_PERPOS_FFN_DIM=2048    — FFN intermediate dimension (NEW v105.5)
"""
from __future__ import annotations

import math
import os
import time as _time
from typing import Any

import numpy as np
from tinygrad import Tensor, dtypes
from tinygrad.engine.jit import TinyJit

from mycelium.config import Config

# Re-use helpers from v105 / v105.1.
from mycelium.factor_graph_v105 import (
    OP_ADD, OP_SUB, OP_MUL, OP_DIV,
)
# LSD-first encoding helpers come from the v105.3 data module.
from mycelium.factor_graph_data_v105_5 import (
    value_to_digits_lsd, digits_to_value_lsd,
)
from mycelium.factor_graph_v105_1 import (
    _precompute_digit_rope,
    apply_rope_digit_tg,
)
# IB centroid loader reused from v104.
from mycelium.factor_graph_v104 import load_ib_centroids

__all__ = [
    "OP_ADD", "OP_SUB", "OP_MUL", "OP_DIV",
    "value_to_digits_lsd", "digits_to_value_lsd",
]

# ---------------------------------------------------------------------------
# Env vars
# ---------------------------------------------------------------------------

V105_5_TASK             = int(os.environ.get("V105_5_TASK",             "0")) > 0
V105_5_K_MAX            = int(os.environ.get("V105_5_K_MAX",            "8"))
V105_5_N_DIGITS         = int(os.environ.get("V105_5_N_DIGITS",         "5"))
V105_5_N_MAX            = int(os.environ.get("V105_5_N_MAX",            "16"))
V105_5_F_MAX            = int(os.environ.get("V105_5_F_MAX",             "8"))
V105_5_N_HEADS          = 16   # fixed: Pythia-410M
V105_5_WAIST            = int(os.environ.get("V105_5_WAIST",            "512"))
V105_5_CODEBOOK_N       = int(os.environ.get("V105_5_CODEBOOK_N",       "32"))
V105_5_N_FAMILIES       = 4    # fixed at 4: ADD/SUB/MUL/DIV
V105_5_N_MAGNITUDE      = 4    # fixed at 4: 1/2/3/4+-digit classes
V105_5_IB_CENTROIDS     = os.environ.get(
    "V105_5_IB_CENTROIDS", ".cache/ib_centroids_gsm8k_partial.npz"
)
V105_5_IB_TREE          = os.environ.get(
    "V105_5_IB_TREE", ".cache/ib_tree_gsm8k_partial.json"
)
V105_5_ENERGY_WEIGHT    = float(os.environ.get("V105_5_ENERGY_WEIGHT",   "0.01"))
V105_5_CALIB_WEIGHT     = float(os.environ.get("V105_5_CALIB_WEIGHT",    "0.05"))
V105_5_FACTOR_AUX_WEIGHT = float(os.environ.get("V105_5_FACTOR_AUX_WEIGHT", "1.0"))
V105_5_MAGNITUDE_WEIGHT = float(os.environ.get("V105_5_MAGNITUDE_WEIGHT", "0.3"))
V105_5_ROPE_BASE        = float(os.environ.get("V105_5_ROPE_BASE",       "10000.0"))
V105_5_IB_INIT          = int(os.environ.get("V105_5_IB_INIT",           "1")) > 0
V105_5_WAIST_LORA_INIT  = int(os.environ.get("V105_5_WAIST_LORA_INIT",   "1")) > 0
# v105.13 wave-guide: dimensions [H-PRESERVE_DIMS : H] skip the waist correction.
# 0 = disabled (no mask, regular waist). 512 = preserve last 512 dims of 1024d.
V105_13_WAVEGUIDE_PRESERVE_DIMS = int(os.environ.get(
    "V105_13_WAVEGUIDE_PRESERVE_DIMS", "0"
))
V105_5_FOURIER_INIT     = int(os.environ.get("V105_5_FOURIER_INIT",      "0")) > 0
# Drop per-digit CE entirely, use per-NUMBER MSE on reconstructed value as the
# sole variable supervision.
V105_5_NUMBER_MSE_ONLY  = int(os.environ.get("V105_5_NUMBER_MSE_ONLY",   "0")) > 0
# Autoregressive digit decoding. Each digit's logits condition on the soft
# (softmax) predictions of all previously committed digit positions.
V105_5_AR_DIGITS        = int(os.environ.get("V105_5_AR_DIGITS",         "0")) > 0
# Scale of the soft prediction's embedding contribution when conditioning the
# next digit's hidden state. Smaller = milder conditioning, larger = stronger.
V105_5_AR_COND_SCALE    = float(os.environ.get("V105_5_AR_COND_SCALE",   "0.5"))
# AR iteration direction (in LSD-first array layout):
#   0 = LSD-first (default; array index 0 → N-1 — ones first, condition upward)
#   1 = MSD-first (array index N-1 → 0 — ten-thousands first, condition downward)
V105_5_AR_MSD_FIRST     = int(os.environ.get("V105_5_AR_MSD_FIRST",      "0")) > 0
# NEW v105.5 — Per-position FFN block (PPFFN).
# Runs ONCE per breath after the 4 Pythia transformer layers and BEFORE the AR
# digit decoding / IB codebook / waist / delta_gate. Provides position-specific
# transformation that survives attention averaging (5 separate FFNs, one per
# digit position).
V105_5_PERPOS_FFN       = int(os.environ.get("V105_5_PERPOS_FFN",        "1")) > 0
V105_5_PERPOS_FFN_DIM   = int(os.environ.get("V105_5_PERPOS_FFN_DIM",  "2048"))
# v105.6 — per-position W_in for L0 FFN (REPLACEMENT, not add-alongside).
# When enabled, at L0 the FFN's W_in is per-digit-position for digit tokens
# while factor tokens go through the shared Pythia w_in path. There is NO
# shared Pythia w_in fallback for digit tokens at L0 → the gradient MUST flow
# through the per-position weights, eliminating the "gradient suppression"
# trap observed with PPFFN (which trained but stayed functionally inert).
V105_6_PERPOS_L0        = int(os.environ.get("V105_6_PERPOS_L0",          "0")) > 0
# DIAGNOSTIC — aux loss that explicitly penalizes high cos_sim between digit
# positions within each (unobserved, valid) cell at the terminal hidden state
# (the tensor right before the digit codebook readout: var_tokens + magnitude
# embed of the LAST breath). Set > 0 to enable. Test: can the model find a
# low-cos_sim solution when EXPLICITLY pressured to? If yes → architecture is
# capable but unmotivated. If no → architectural ceiling, pivot warranted.
V105_AUX_DISTINCT_WEIGHT = float(os.environ.get("V105_AUX_DISTINCT_WEIGHT", "0.0"))

# v105.5 var_loss weight — multiplier on the per-digit CE (or NUMBER_MSE_ONLY
# variant) summed over breaths. Default 1.0 (unchanged behaviour). Set to 0
# when V105_8_PER_NUMBER_READOUT=1 (per-number CE supersedes per-digit CE).
V105_5_VAR_LOSS_WEIGHT  = float(os.environ.get("V105_5_VAR_LOSS_WEIGHT",  "1.0"))

# v105.8 — PER-NUMBER readout. Per-layer cos_sim diagnostic on v105.6
# confirmed terminal hidden state cos_sim ~ 1.0 across digit positions of
# each cell (mean-field collapse). Per-digit CE loss is what creates that
# pressure: position-specific codebook entries decode digits from a SHARED
# hidden state, so the model gets the same gradient signal as if the
# distribution were a product of independent per-digit marginals.
#
# v105.8 keeps the per-position token architecture but switches to a single
# per-NUMBER CE on a (n_bins, H) codebook applied to the mean of the
# terminal hidden state across digit positions. Positions are now ALLOWED
# to collapse (the mean is still a valid number representation); the 5
# per-position tokens act as thinking slots rather than digit decoders.
#
# When enabled, V105_5_VAR_LOSS_WEIGHT, V105_5_MAGNITUDE_WEIGHT,
# V105_5_ENERGY_WEIGHT, V105_AUX_DISTINCT_WEIGHT should be forced to 0
# (the training driver does this Python-side; the JIT respects the
# resulting weights).

# v105.11 — DROP-THE-CODEBOOK with log-number-MSE through AR digit decoder.
#
# v105.10 diagnostics (Jun 3) showed three coherent findings:
#   1. In-distribution 5-token win is real (v105.10 vs v107: +16.8pt on val[medium]).
#   2. OOD compositionality FAILED (per-digit acc at chance on [10010, 99998]).
#   3. AR conditioning was DECORATIVE (consistency test UNRESPONSIVE — d2 ignored
#      d1 clamps; each digit was predicted independently from cell_hidden).
#
# Result 3 explains result 2: independent per-position classifiers can't
# generalize compositionally. The codebook provided an "easy path" for the
# breathing → cell_hidden was optimized for codebook readout, not digit
# extraction.
#
# v105.11 removes the codebook entirely, replacing its precision signal with
# log-number-MSE on the AR-reconstructed value. Per-digit CE and log-MSE share
# the SAME path through the AR decoder. Three mechanisms force the AR
# conditioning to be load-bearing:
#   Mechanism 1: stronger cond_scale (V105_9_AR_COND_SCALE=2.0 in the launcher).
#   Mechanism 2: concat-and-project conditioning (V105_11_CONCAT_COND=1).
#   Mechanism 3: conditional dropout on cell_hidden (V105_11_COND_DROPOUT>0).
#
# When V105_11_NUMBER_MSE=1, mutex/relaxation logic forces:
#   V105_8_PER_NUMBER_READOUT = 0  (no codebook)
#   V105_9_AR_DIGIT_DECODER   = 1  (need the AR path)
#   V105_10_DUAL_READOUT      = 0  (no codebook to dual with)
# We do this by overriding os.environ BEFORE reading V105_8/9/10 below, so
# module-level constants reflect the v105.11 contract and mutex logic stays
# in one place.
V105_11_NUMBER_MSE      = int(os.environ.get("V105_11_NUMBER_MSE",      "0")) > 0
V105_11_NUMBER_MSE_BETA = float(os.environ.get("V105_11_NUMBER_MSE_BETA", "1.0"))
V105_11_CONCAT_COND     = int(os.environ.get("V105_11_CONCAT_COND",     "0")) > 0
V105_11_COND_DROPOUT    = float(os.environ.get("V105_11_COND_DROPOUT",  "0.0"))
# Mechanism 4 (Jun 3): LayerNorm cell_hidden and prev_embed_sum BEFORE combining,
# equalizing their magnitudes. Mechanisms 1/2/3 all failed because cell_hidden
# (mag ~32 = sqrt(H)) dominated d_embed (mag ~2) — model could ignore d_embed
# as noise. LN equalizes both to mag ~sqrt(H), forcing the decoder to process
# both components.
V105_11_LN_COND         = int(os.environ.get("V105_11_LN_COND",         "0")) > 0

# v105.12 — FINAL COMPOSITIONALITY EXPERIMENT
#
# Combines every proven v105 mechanism (v105.8 codebook precision, v105.9 AR
# digit decoder, v105.10 dual readout, v105.11 log-MSE + LN_COND) with three
# new principled additions targeting the remaining failure modes from v105.10
# (OOD compositional failure) and v105.11 (in-dist plateau despite RESPONSIVE
# conditioning):
#
#   Change 1 — PREFILL ISOLATE: in v105.10/v105.11, the magnitude head +
#     per-position digit codebook + magnitude_embed addition all fire at
#     EVERY breath, with magnitude_embed feeding back into var_tokens_r
#     and thus the next breath's input. With V105_12_PREFILL_ISOLATE=1,
#     these readout-side computations are gated to k == K-1 only. The
#     breathing loop becomes pure constraint propagation; decode happens
#     once at the end. Forward-only architectural change.
#
#   Change 2 — FOURIER DECODE INIT: replace the QR-random init of
#     fg_v105_5_digit_codebook with a Fourier basis. For position p, digit
#     d, hidden dim k: codebook[p, d, 2k] = cos(2π·d·(k+1)/10),
#     codebook[p, d, 2k+1] = sin(2π·d·(k+1)/10); normalized by 1/sqrt(H).
#     With PREFILL_ISOLATE the codebook is touched ONLY at the last breath
#     (not washed out by Pythia layers during breathing).
#
#   Change 3 — CODEBOOK ANNEAL: drives a runtime-Tensor weight on the
#     number_codebook CE loss (v105.8 path). Schedule lives in the
#     training driver:
#       steps 0    – 2000:   weight = 1.0   (bootstrap precision)
#       steps 2000 – 5000:   linear 1.0 → 0.15
#       steps 5000 – 15000:  weight = 0.15  (maintenance)
#     This is the enable flag; the actual weight comes through the JIT
#     as a Tensor input argument so step time is unchanged.
V105_12_PREFILL_ISOLATE     = int(os.environ.get("V105_12_PREFILL_ISOLATE",     "0")) > 0
V105_12_FOURIER_DECODE_INIT = int(os.environ.get("V105_12_FOURIER_DECODE_INIT", "0")) > 0
V105_12_CODEBOOK_ANNEAL     = int(os.environ.get("V105_12_CODEBOOK_ANNEAL",     "0")) > 0
V105_12_ENABLED = (
    V105_12_PREFILL_ISOLATE or V105_12_FOURIER_DECODE_INIT or V105_12_CODEBOOK_ANNEAL
)

# v105.12 enables BOTH v105.8 (number codebook) and v105.9 (AR digit decoder)
# concurrently — same dual-readout pattern as v105.10, with the v105.11 LN_COND
# fix and the three v105.12 mechanisms layered on top. This precedes the
# v105.11 NUMBER_MSE mutex so v105.12 wins when both are active.
if V105_12_ENABLED:
    os.environ["V105_8_PER_NUMBER_READOUT"] = "1"
    os.environ["V105_9_AR_DIGIT_DECODER"]   = "1"
    os.environ["V105_10_DUAL_READOUT"]      = "1"

if V105_11_NUMBER_MSE and not V105_12_ENABLED:
    os.environ["V105_8_PER_NUMBER_READOUT"] = "0"
    os.environ["V105_9_AR_DIGIT_DECODER"]   = "1"
    os.environ["V105_10_DUAL_READOUT"]      = "0"

V105_8_PER_NUMBER_READOUT = int(os.environ.get("V105_8_PER_NUMBER_READOUT", "0")) > 0
V105_8_N_NUMBER_BINS      = int(os.environ.get("V105_8_N_NUMBER_BINS",      "200"))

# v105.9 — AR DIGIT DECODER over POOLED cell_hidden.
#
# Diagnosis: v105.8 (200-bin number codebook) works (same as v107) but loses
# digit compositionality — it must memorize 40k pairwise (value, bin) entries
# for 4-digit numbers and cannot generalize to unseen magnitudes without
# retraining the codebook. v105.9 keeps the digit-decomposition WIN by moving
# digits from REPRESENTATION (per-position tokens, where they collapse) to
# READOUT (a SMALL AR digit decoder that reads from pooled cell_hidden).
#
# After K=8 breaths, at the LAST breath:
#   cell_hidden = var_tokens_with_mag.mean(axis=2)          # (B, n_max, H)
#   For each digit position p (LSD-first), condition on cell_hidden + sum of
#   soft embeddings of LOWER digits, then read out via the existing
#   per-position digit codebook[p] (shape (10, H)):
#     logits_p = (cell_hidden + cond) @ codebook[p].T
#     probs_p  = softmax(logits_p)
#     cond     = cond + AR_COND_SCALE * (probs_p @ codebook[p])
#
# The per-digit CE loss flows through ONE pooled hidden state (not 5
# conflicting per-position signals). Model encodes "the number is X" in
# cell_hidden; the small AR decoder extracts digits.
#
# Mutually exclusive with V105_8_PER_NUMBER_READOUT (per-NUMBER codebook),
# EXCEPT when v105.10 dual-readout mode is enabled.
V105_9_AR_DIGIT_DECODER   = int(os.environ.get("V105_9_AR_DIGIT_DECODER", "0")) > 0
V105_9_AR_COND_SCALE      = float(os.environ.get("V105_9_AR_COND_SCALE", "0.5"))

# v105.10 — DUAL READOUT (v105.8 + v105.9 simultaneously).
#
# Combines v105.8's 200-bin per-NUMBER readout AND v105.9's AR digit decoder
# in a single architecture. Both readouts are computed from the same pooled
# cell_hidden at the final breath:
#   number_logits = cell_hidden @ number_codebook.T     # v105.8 path
#   digit_logits_pooled = AR-decode(cell_hidden, digit_codebook)  # v105.9 path
#
# Loss:
#   total += number_ce_loss + V105_10_DIGIT_WEIGHT * digit_ce_loss
#
# Hypothesis: number_CE provides precise-value gradient that makes
# cell_hidden a clean representation; digit_CE trains the decoder to extract
# compositional digits from that representation. The killer experiment is OOD
# generalization: train on [0, 9999], test on [10000, 99999]. The 200-bin
# codebook has no bins above 9999 → 0% on OOD. The digit decoder reads each
# digit independently → can attempt 5-digit numbers it never saw.
#
# When V105_10_DUAL_READOUT=1, the standard mutex check between
# V105_8_PER_NUMBER_READOUT and V105_9_AR_DIGIT_DECODER is relaxed: both
# paths must be enabled (the launcher sets all three flags together).
V105_10_DUAL_READOUT      = int(os.environ.get("V105_10_DUAL_READOUT", "0")) > 0
V105_10_DIGIT_WEIGHT      = float(os.environ.get("V105_10_DIGIT_WEIGHT", "0.3"))

if V105_8_PER_NUMBER_READOUT and V105_9_AR_DIGIT_DECODER and not V105_10_DUAL_READOUT:
    raise RuntimeError(
        "v105.8 (PER_NUMBER_READOUT) and v105.9 (AR_DIGIT_DECODER) are "
        "mutually exclusive. Set at most one of V105_8_PER_NUMBER_READOUT / "
        "V105_9_AR_DIGIT_DECODER to 1, OR set V105_10_DUAL_READOUT=1 to enable "
        "both simultaneously (v105.10 dual-readout mode)."
    )
if V105_10_DUAL_READOUT and not (V105_8_PER_NUMBER_READOUT and V105_9_AR_DIGIT_DECODER):
    raise RuntimeError(
        "V105_10_DUAL_READOUT=1 requires BOTH V105_8_PER_NUMBER_READOUT=1 AND "
        "V105_9_AR_DIGIT_DECODER=1 (v105.10 = v105.8 + v105.9 dual readout)."
    )


# ---------------------------------------------------------------------------
# IB tree loader: build leaf_to_family mapping (32 → 4) from JSON
# ---------------------------------------------------------------------------

_OP_NAME_TO_IDX = {"ADD": 0, "SUB": 1, "MUL": 2, "DIV": 3}


def _load_leaf_to_family(
    path: str, n_code: int = 32, n_families: int = 4
) -> tuple[np.ndarray, list[str]]:
    """Load leaf_to_family int array (n_code,) from IB tree JSON.

    The JSON has a 'leaves' list; each leaf has an 'op' field in {ADD,SUB,MUL,DIV}.
    Leaf order in the JSON matches the npz ordering and thus the codebook index.

    Returns:
      leaf_to_family : (n_code,) int array, values in [0, n_families)
      leaf_ids       : list of leaf id strings (for diagnostic)
    """
    import json
    leaf_to_family = np.zeros(n_code, dtype=np.int32)
    leaf_ids: list[str] = []

    if not os.path.exists(path):
        # Fall back to uniform 8 per family.
        print(
            f"[v105.5] IB tree JSON missing at {path}; falling back to "
            f"uniform 8-per-family leaf_to_family assignment.",
            flush=True,
        )
        per_fam = max(1, n_code // n_families)
        for i in range(n_code):
            leaf_to_family[i] = min(i // per_fam, n_families - 1)
        return leaf_to_family, [f"FAKE.{i}" for i in range(n_code)]

    try:
        with open(path) as f:
            tree = json.load(f)
        leaves = tree.get("leaves", [])
        if len(leaves) < n_code:
            print(
                f"[v105.5] IB tree has {len(leaves)} leaves < n_code={n_code}; "
                f"falling back to uniform 8-per-family.",
                flush=True,
            )
            per_fam = max(1, n_code // n_families)
            for i in range(n_code):
                leaf_to_family[i] = min(i // per_fam, n_families - 1)
            return leaf_to_family, [f"FAKE.{i}" for i in range(n_code)]
        for i in range(n_code):
            leaf = leaves[i]
            op = str(leaf.get("op", "ADD")).upper()
            leaf_to_family[i] = _OP_NAME_TO_IDX.get(op, 0)
            leaf_ids.append(str(leaf.get("leaf_id", f"L{i}")))
    except Exception as e:
        print(
            f"[v105.5] IB tree JSON load failed ({e}); falling back to "
            f"uniform 8-per-family.",
            flush=True,
        )
        per_fam = max(1, n_code // n_families)
        for i in range(n_code):
            leaf_to_family[i] = min(i // per_fam, n_families - 1)
        leaf_ids = [f"FAKE.{i}" for i in range(n_code)]

    return leaf_to_family, leaf_ids


def _fourier_digit_codebook(n_digits: int, hidden: int) -> np.ndarray:
    """Cyclic-Fourier init for digit_codebook.

    Maps each digit d to phases on a circle: phase_d = 2π·d/n_digits.
    The codebook fills hidden dims with [cos(d·freq·phase), sin(d·freq·phase)]
    pairs cycling through the Nyquist-limited frequencies 1..n_digits/2.
    """
    cb = np.zeros((n_digits, hidden), dtype=np.float32)
    n_unique_freqs = max(n_digits // 2, 1)  # 5 for n_digits=10
    n_pairs = hidden // 2
    for k in range(n_pairs):
        freq = (k % n_unique_freqs) + 1
        for d in range(n_digits):
            phase = 2.0 * np.pi * d * freq / n_digits
            cb[d, 2 * k]     = np.cos(phase)
            cb[d, 2 * k + 1] = np.sin(phase)
    norms = np.linalg.norm(cb, axis=1, keepdims=True)
    cb = cb / (norms + 1e-8)
    return cb.astype(np.float32)


def _v105_12_fourier_decode_init(n_digits_pos: int, hidden: int) -> np.ndarray:
    """v105.12 Fourier init for per-position digit_codebook (n_digits_pos, 10, H).

    For each position p, each digit d in 0..9, each k in 0..H//2-1:
        angle = 2π·d·(k+1)/10
        codebook[p, d, 2k]   = cos(angle)
        codebook[p, d, 2k+1] = sin(angle)
    Then scale globally by 1/sqrt(H) so codebook norms are O(1) (unit-ish).

    The per-position structure is IDENTICAL — same Fourier basis at every
    digit position. The codebook stays learnable so it can adapt slowly;
    Fourier is the starting basis. Combined with V105_12_PREFILL_ISOLATE=1,
    the codebook is touched ONLY at the decode breath and is not washed out
    by Pythia layers during the iterative breathing loop.
    """
    cb = np.zeros((n_digits_pos, 10, hidden), dtype=np.float32)
    n_pairs = hidden // 2
    for p in range(n_digits_pos):
        for d in range(10):
            for k in range(n_pairs):
                angle = 2.0 * np.pi * d * (k + 1) / 10.0
                cb[p, d, 2 * k]     = float(np.cos(angle))
                cb[p, d, 2 * k + 1] = float(np.sin(angle))
    # Scale 0.1/sqrt(H) — not 1/sqrt(H). At 1/sqrt(H) the codebook entries
    # have unit norm, and combined with LN'd cell_hidden (norm sqrt(H)≈32)
    # the inner products reach magnitude ~32, saturating softmax and creating
    # pool_ce explosions (~43) early in training. 0.1× preserves the Fourier
    # geometry (cosine similarities are scale-invariant) while keeping initial
    # logit magnitudes in the well-behaved softmax range (~3).
    cb = cb * (0.1 / float(np.sqrt(hidden)))
    return cb.astype(np.float32)


# ---------------------------------------------------------------------------
# Component 1+2: Embedding with digit-axis RoPE (LSD layout — no reversal)
# ---------------------------------------------------------------------------

def embed_factor_graph_v105_5(
    digit_init: Tensor,       # (B, N_MAX, N_DIGITS, 10) — one-hot / uniform
    node_kinds: Tensor,        # (B, T_MAX) int
    var_pos_embed: Tensor,     # (N_MAX, H)
    factor_pos_embed: Tensor,  # (F_MAX, H)
    node_kind_embed: Tensor,   # (3, H)
    digit_codebook: Tensor,    # (N_DIGITS, 10, H) — per-position codebooks (v105.5)
    digit_rope_cos: Tensor,    # (N_DIGITS, H)
    digit_rope_sin: Tensor,    # (N_DIGITS, H)
    n_max: int,
    n_digits: int,
    f_max: int,
) -> Tensor:
    """Build (B, T_MAX, H) hidden states with digit-axis RoPE.

    v105.5: digit_codebook is now (n_digits, 10, H) — PER-POSITION.

    Component 1: digit_codebook[p] (10, H) — distinct codebook per position p.
    Component 2: apply_rope_digit_tg — each position p gets rotated by p*freq.

    LSD layout: array index 0 = ones digit ⇒ RoPE position 0 = no rotation.
                array index N-1 = most significant ⇒ maximum rotation.

    raw[b, v, p] = digit_codebook[p, digit_value] + var_pos_embed[v]
    embed[b, v, p] = RoPE(raw[b, v, p], p)
    """
    B = int(digit_init.shape[0])
    H = int(var_pos_embed.shape[1])

    # Per-position codebook contraction: digit_init has shape (B, N_MAX, N_DIGITS, 10)
    # and digit_codebook is (N_DIGITS, 10, H).  We contract on dim 10 per position p.
    # Reshape both to align the digit-position axis for einsum-like behavior.
    di_cb = digit_init.cast(digit_codebook.dtype)             # (B, N_MAX, N_DIGITS, 10)
    # Broadcast multiply on (B, N_MAX, N_DIGITS, 10, H):
    cb_bcast = digit_codebook.reshape(1, 1, n_digits, 10, H)  # (1, 1, N_DIGITS, 10, H)
    di_bcast = di_cb.reshape(B, n_max, n_digits, 10, 1)        # (B, N_MAX, N_DIGITS, 10, 1)
    var_digit_state = (di_bcast * cb_bcast).sum(axis=3)        # (B, N_MAX, N_DIGITS, H)

    vpe = var_pos_embed.reshape(1, n_max, 1, H).cast(var_digit_state.dtype)
    raw = var_digit_state + vpe.expand(B, n_max, n_digits, H)  # (B, N_MAX, N_DIGITS, H)

    # Apply digit RoPE (Component 2) — LSD layout, no reversal
    var_digit_h = apply_rope_digit_tg(
        raw, digit_rope_cos, digit_rope_sin, n_digits=n_digits, hidden=H
    )  # (B, N_MAX, N_DIGITS, H)

    var_tokens = var_digit_h.reshape(B, n_max * n_digits, H)

    # Factor positions (unchanged)
    factor_pos = factor_pos_embed.reshape(1, f_max, H).cast(var_tokens.dtype).expand(B, f_max, H)
    x = var_tokens.cat(factor_pos, dim=1)  # (B, T, H)

    # Node-kind embedding
    nk_clamped = node_kinds.clip(0, 2)
    nk_oh      = nk_clamped.one_hot(3).cast(x.dtype)
    nk_emb     = nk_oh @ node_kind_embed.cast(x.dtype)
    x = x + nk_emb

    return x


# ---------------------------------------------------------------------------
# One transformer layer (identical to v105.1.2)
# ---------------------------------------------------------------------------

def fg_layer_forward_v105_5(
    layer: Any,
    x: Tensor,          # (B, T_MAX, H)
    attn_bias: Tensor,  # (B, N_HEADS, T_MAX, T_MAX)
) -> Tensor:
    """Run one BreathingLayer with per-head factor-graph attention mask."""
    cfg = layer.cfg
    B, S, H = x.shape
    n_heads  = cfg.n_heads
    head_dim = cfg.head_dim

    from mycelium.breathing import _layernorm
    attn_in = _layernorm(x, layer.shared.in_ln_g, layer.shared.in_ln_b, cfg.layer_norm_eps)
    mlp_in  = _layernorm(x, layer.shared.post_ln_g, layer.shared.post_ln_b, cfg.layer_norm_eps)
    attn_in_dt = attn_in.cast(x.dtype) if attn_in.dtype != x.dtype else attn_in
    mlp_in_dt  = mlp_in.cast(x.dtype) if mlp_in.dtype != x.dtype else mlp_in

    q = (attn_in_dt @ layer.wq + layer.bq).reshape(B, S, n_heads, head_dim).transpose(1, 2)
    k = (attn_in_dt @ layer.wk + layer.bk).reshape(B, S, n_heads, head_dim).transpose(1, 2)
    v = (attn_in_dt @ layer.shared.wv + layer.shared.bv).reshape(B, S, n_heads, head_dim).transpose(1, 2)

    scale  = 1.0 / math.sqrt(head_dim)
    scores = q @ k.transpose(-2, -1) * scale
    scores = scores + attn_bias.cast(scores.dtype)
    attn   = scores.clip(-1e4, 1e4).softmax(-1)
    ctx    = (attn @ v).transpose(1, 2).reshape(B, S, H)
    attn_out = ctx @ layer.shared.wo + layer.shared.bo

    ff      = (mlp_in_dt @ layer.w_in + layer.b_in).gelu()
    ffn_out = ff @ layer.shared.w_out + layer.shared.b_out

    return x + attn_out + ffn_out


# ---------------------------------------------------------------------------
# v105.6 ADDITION — Per-position W_in for L0 FFN (REPLACEMENT path).
# ---------------------------------------------------------------------------

def fg_layer_forward_v105_6_l0(
    layer: Any,
    x: Tensor,                # (B, T_MAX, H)
    attn_bias: Tensor,        # (B, N_HEADS, T_MAX, T_MAX)
    perpos_w_in: Tensor,      # (n_digits, H, ffn_int)
    perpos_b_in: Tensor,      # (n_digits, ffn_int)
    n_max: int,
    n_digits: int,
    f_max: int,
) -> Tensor:
    """L0 layer forward with per-position W_in for digit tokens.

    Identical to fg_layer_forward_v105_5 EXCEPT the FFN's W_in is split:
      - Digit tokens (first n_max*n_digits sequence positions): use
        per-digit-position W_in / b_in. No shared Pythia w_in fallback.
      - Factor tokens (remaining f_max sequence positions): use the
        shared Pythia w_in/b_in path (layer.w_in, layer.b_in).

    The FFN's W_out remains shared (layer.shared.w_out) so we keep the
    Pythia warm-start on the output projection. Per-position W_in is
    initialized as Pythia's w_in + tiny Gaussian noise so step-0 forward
    is nearly identical to v105.5 with the same warm-start.

    Args:
      layer       : BreathingLayer (provides shared LNs, attn weights, w_out, etc.)
      x           : (B, T, H) residual stream
      attn_bias   : per-head attention mask (B, N_HEADS, T, T)
      perpos_w_in : (n_digits, H, ffn_int) per-position FFN W_in
      perpos_b_in : (n_digits, ffn_int)    per-position FFN b_in
      n_max, n_digits, f_max : token layout (T = n_max*n_digits + f_max)

    Returns:
      x_new : (B, T, H) updated residual stream.
    """
    cfg = layer.cfg
    B, S, H = x.shape
    n_heads  = cfg.n_heads
    head_dim = cfg.head_dim
    n_var    = n_max * n_digits

    from mycelium.breathing import _layernorm
    attn_in = _layernorm(x, layer.shared.in_ln_g, layer.shared.in_ln_b, cfg.layer_norm_eps)
    mlp_in  = _layernorm(x, layer.shared.post_ln_g, layer.shared.post_ln_b, cfg.layer_norm_eps)
    attn_in_dt = attn_in.cast(x.dtype) if attn_in.dtype != x.dtype else attn_in
    mlp_in_dt  = mlp_in.cast(x.dtype) if mlp_in.dtype != x.dtype else mlp_in

    # Attention sublayer (unchanged from v105.5).
    q = (attn_in_dt @ layer.wq + layer.bq).reshape(B, S, n_heads, head_dim).transpose(1, 2)
    k = (attn_in_dt @ layer.wk + layer.bk).reshape(B, S, n_heads, head_dim).transpose(1, 2)
    v = (attn_in_dt @ layer.shared.wv + layer.shared.bv).reshape(B, S, n_heads, head_dim).transpose(1, 2)

    scale  = 1.0 / math.sqrt(head_dim)
    scores = q @ k.transpose(-2, -1) * scale
    scores = scores + attn_bias.cast(scores.dtype)
    attn   = scores.clip(-1e4, 1e4).softmax(-1)
    ctx    = (attn @ v).transpose(1, 2).reshape(B, S, H)
    attn_out = ctx @ layer.shared.wo + layer.shared.bo

    # FFN sublayer: split digit-token-path (per-position W_in) and
    # factor-token-path (shared Pythia W_in).
    mlp_in_var = mlp_in_dt[:, :n_var, :].reshape(B, n_max, n_digits, H)  # (B, n_max, n_digits, H)
    mlp_in_fac = mlp_in_dt[:, n_var:, :]                                  # (B, f_max, H)

    ffn_int = int(perpos_w_in.shape[-1])

    # Per-position FFN W_in for digit tokens. Inner compute in fp32 for
    # numerical stability inside JIT (same pattern as apply_perpos_ffn).
    pp_w_in_f  = perpos_w_in.cast(dtypes.float)   # (n_digits, H, ffn_int)
    pp_b_in_f  = perpos_b_in.cast(dtypes.float)   # (n_digits, ffn_int)

    ff_var_per_pos: list[Tensor] = []
    for p in range(n_digits):
        in_p = mlp_in_var[:, :, p, :].cast(dtypes.float)  # (B, n_max, H)
        W_p  = pp_w_in_f[p]                                # (H, ffn_int)
        b_p  = pp_b_in_f[p].reshape(1, 1, -1)              # (1, 1, ffn_int)
        ff_p = (in_p @ W_p + b_p).gelu()                   # (B, n_max, ffn_int)
        ff_var_per_pos.append(ff_p)

    # Stack along digit-position axis then collapse back into the var-token axis.
    # (B, n_max, n_digits, ffn_int) → (B, n_max*n_digits, ffn_int)
    ff_var = Tensor.stack(*ff_var_per_pos, dim=2).reshape(B, n_var, ffn_int)
    # Cast back to model dtype to match shared.w_out path.
    ff_var = ff_var.cast(x.dtype)

    # Shared FFN W_in for factor tokens.
    ff_fac = (mlp_in_fac @ layer.w_in + layer.b_in).gelu()  # (B, f_max, ffn_int)

    # Concat digit + factor FFN intermediate activations along seq axis.
    ff = Tensor.cat(ff_var, ff_fac, dim=1)                  # (B, T, ffn_int)

    # Shared W_out (preserves Pythia warm-start on output projection).
    ffn_out = ff @ layer.shared.w_out + layer.shared.b_out

    return x + attn_out + ffn_out


# ---------------------------------------------------------------------------
# Component 3: Projection waist 1024 → 512 → 1024 (LoRA-style)
# ---------------------------------------------------------------------------

def apply_projection_waist(
    h: Tensor,         # (B, T, H)
    W_compress: Tensor,  # (H, waist)
    W_expand: Tensor,    # (waist, H)   — zero-initialized
    b_compress: Tensor,  # (waist,)
    b_expand: Tensor,    # (H,)
    waveguide_mask: Tensor | None = None,  # optional (1, 1, H) — 1.0 commit, 0.0 preserve
) -> Tensor:
    """Projection waist with LoRA-style zero init.

    At init: W_expand = 0 → quantize = 0 → output = h (byte-identical to no-waist).
    After unlock: h → 512d compressed → GELU → 1024d correction (added as residual).

    Wave-guide variant (v105.13): if waveguide_mask is provided, the LoRA
    correction is zeroed out for dimensions where the mask is 0. Those
    dimensions pass through unchanged each breath — the "preserve channel"
    that carries fine-grained info without compression. The commit channel
    (mask=1) is updated as normal. Byte-identical at init regardless of mask.
    """
    wc = W_compress.cast(h.dtype)
    bc = b_compress.reshape(1, 1, -1).cast(h.dtype)
    we = W_expand.cast(h.dtype)
    be = b_expand.reshape(1, 1, -1).cast(h.dtype)

    waist_h  = (h @ wc + bc).gelu()   # (B, T, waist)
    quantize = waist_h @ we + be       # (B, T, H) — zero at init (W_expand=0)
    if waveguide_mask is not None:
        quantize = quantize * waveguide_mask.cast(quantize.dtype)
    return h + quantize                # residual; = h at init


# ---------------------------------------------------------------------------
# Component 4: IB semantic codebook soft projection (LoRA-style gate)
# ---------------------------------------------------------------------------

def apply_ib_codebook(
    h: Tensor,              # (B, T, H)
    codebook: Tensor,        # (N_CODE, H)
    temperature: Tensor,     # () scalar
    delta_gate_quant_k: Tensor,  # () — zero at init
) -> Tensor:
    """IB soft codebook projection with zero-init gate.

    At init: delta_gate_quant = 0 → h_quant = h (byte-identical to no-codebook).
    After unlock: soft nearest-codebook reconstruction is blended in as residual.
    """
    cb  = codebook.cast(h.dtype)
    tmp = temperature.cast(h.dtype)

    scores   = h @ cb.T / tmp.reshape(1, 1, 1)     # (B, T, N_CODE)
    weights  = scores.clip(-1e4, 1e4).softmax(-1)  # (B, T, N_CODE)
    recon    = weights @ cb                          # (B, T, H)
    quantize = recon - h                            # delta: toward codebook
    gate_k   = delta_gate_quant_k.cast(h.dtype).reshape(1, 1, 1)
    return h + gate_k * quantize                    # = h at init


def apply_hierarchical_ib_codebook(
    h: Tensor,                       # (B, T, H)
    leaf_codebook: Tensor,            # (N_CODE, H)
    family_centroids: Tensor,         # (N_FAMILIES, H)
    leaf_to_family_oh: Tensor,        # (N_CODE, N_FAMILIES) float — frozen one-hot
    temperature: Tensor,              # () scalar
    delta_gate_quant_k: Tensor,       # () — zero at init
) -> Tensor:
    """Hierarchical IB soft codebook with family gating (v105.5 addition 3).

    Two-level soft attention:
      family_weights = softmax(h @ family_centroids^T)   (B, T, F)
      leaf_weights_flat = softmax(h @ leaf_codebook^T)   (B, T, N)
      # Multiply each leaf weight by its family weight, then renormalize:
      fam_for_each_leaf = family_weights @ leaf_to_family_oh^T  (B, T, N)
      hierarchical_weights = leaf_weights_flat * fam_for_each_leaf
      hierarchical_weights /= sum + 1e-8
      ib_context = hierarchical_weights @ leaf_codebook  (B, T, H)
      h_new = h + delta_gate_quant_k * (ib_context - h)

    At init: delta_gate_quant_k = 0 → output = h (byte-identical to v105.3 step-0).
    """
    cb     = leaf_codebook.cast(h.dtype)            # (N, H)
    famcb  = family_centroids.cast(h.dtype)         # (F, H)
    l2f_oh = leaf_to_family_oh.cast(h.dtype)        # (N, F)
    tmp    = temperature.cast(h.dtype)

    # 1. Family attention
    fam_scores  = h @ famcb.T / tmp.reshape(1, 1, 1)              # (B, T, F)
    fam_weights = fam_scores.clip(-1e4, 1e4).softmax(-1)          # (B, T, F)

    # 2. Leaf attention (flat)
    leaf_scores  = h @ cb.T / tmp.reshape(1, 1, 1)                # (B, T, N)
    leaf_w_flat  = leaf_scores.clip(-1e4, 1e4).softmax(-1)        # (B, T, N)

    # 3. Reweight leaves by their family weight (gather via one-hot matmul).
    # family_weights: (B, T, F), l2f_oh: (N, F) → fam_for_each_leaf: (B, T, N)
    fam_for_each_leaf = fam_weights @ l2f_oh.T                    # (B, T, N)

    hierarchical = leaf_w_flat * fam_for_each_leaf
    norm         = hierarchical.sum(axis=-1, keepdim=True) + 1e-8
    hierarchical = hierarchical / norm

    # 4. Apply
    ib_context = hierarchical @ cb                                # (B, T, H)
    quantize   = ib_context - h
    gate_k     = delta_gate_quant_k.cast(h.dtype).reshape(1, 1, 1)
    return h + gate_k * quantize                                  # = h at init


# ---------------------------------------------------------------------------
# v105.5 ADDITION — Per-position FFN block (PPFFN)
# ---------------------------------------------------------------------------

def apply_perpos_ffn(
    x: Tensor,                # (B, T, H)
    ppffn_ln_g: Tensor,        # (n_digits, H)
    ppffn_ln_b: Tensor,        # (n_digits, H)
    ppffn_W_in: Tensor,        # (n_digits, H, intermediate)
    ppffn_b_in: Tensor,        # (n_digits, intermediate)
    ppffn_W_out: Tensor,       # (n_digits, intermediate, H)  — zero at init
    ppffn_b_out: Tensor,       # (n_digits, H)                 — zero at init
    n_max: int,
    n_digits: int,
    f_max: int,
    eps: float = 1e-5,
) -> Tensor:
    """Per-position FFN block.

    Applies a separate FFN (LayerNorm + W_in → GELU → W_out) per digit
    POSITION (0..n_digits-1).  Only digit tokens (the first n_max*n_digits
    sequence positions) are affected — factor positions pass through
    unchanged.

    At init: W_out = 0, b_out = 0  →  per-position residual contribution = 0,
    so the forward pass is byte-identical to v105.4 with the same warm-start.

    Args:
      x : (B, T, H) hidden state — T = n_max*n_digits + f_max
      ppffn_* : per-position weight tensors (see signature)

    Returns:
      x_new : (B, T, H) with digit positions residually updated.
    """
    B = int(x.shape[0])
    n_var_tokens = n_max * n_digits

    # Carve out digit positions: (B, n_max*n_digits, H) → (B, n_max, n_digits, H)
    x_var   = x[:, :n_var_tokens, :]
    x_var_r = x_var.reshape(B, n_max, n_digits, -1)  # (B, n_max, n_digits, H)
    H       = int(x_var_r.shape[-1])

    # FP32 internal compute for numerical stability inside JIT.
    x32 = x_var_r.cast(dtypes.float)

    ffn_outs: list[Tensor] = []
    for p in range(n_digits):
        xp = x32[:, :, p, :]                        # (B, n_max, H)

        # LayerNorm with per-position gain/bias (FP32 internal).
        mean = xp.mean(axis=-1, keepdim=True)
        var  = ((xp - mean) ** 2).mean(axis=-1, keepdim=True)
        xp_ln = (xp - mean) * (var + eps).rsqrt()
        g = ppffn_ln_g[p].cast(dtypes.float)        # (H,)
        b = ppffn_ln_b[p].cast(dtypes.float)        # (H,)
        xp_ln = xp_ln * g.reshape(1, 1, H) + b.reshape(1, 1, H)

        # Per-position FFN: (H → intermediate) → GELU → (intermediate → H).
        W_in_p  = ppffn_W_in[p].cast(dtypes.float)   # (H, intermediate)
        b_in_p  = ppffn_b_in[p].cast(dtypes.float)   # (intermediate,)
        W_out_p = ppffn_W_out[p].cast(dtypes.float)  # (intermediate, H)
        b_out_p = ppffn_b_out[p].cast(dtypes.float)  # (H,)

        hp = (xp_ln @ W_in_p + b_in_p.reshape(1, 1, -1)).gelu()   # (B, n_max, intermediate)
        yp = hp @ W_out_p + b_out_p.reshape(1, 1, -1)              # (B, n_max, H) — zero at init
        ffn_outs.append(yp)

    # Stack: (B, n_max, n_digits, H), cast back to x dtype.
    ffn_stack = Tensor.stack(*ffn_outs, dim=2).cast(x.dtype)  # (B, n_max, n_digits, H)

    # Residual add only on digit positions.
    x_var_new = (x_var_r + ffn_stack).reshape(B, n_var_tokens, H)
    x_fac     = x[:, n_var_tokens:, :]
    return x_var_new.cat(x_fac, dim=1)


# ---------------------------------------------------------------------------
# Constraint energy (LSD place values)
# ---------------------------------------------------------------------------

def _expected_value_v105_5(digit_logits: Tensor, n_digits: int) -> Tensor:
    """Compute expected integer value from per-digit logit distributions (LSD layout).

    digit_logits: (..., N_DIGITS, 10)
    Returns: (...,) float — expected value = Σ_p E[digit_p] × 10^p

    LSD place values: array index p has place 10^p (ones at idx 0, ten-thousands at idx N-1).
    """
    probs   = digit_logits.softmax(-1)  # (..., N_DIGITS, 10)
    d_vals  = Tensor(np.arange(10, dtype=np.float32)).cast(probs.dtype)  # (10,)
    exp_dig = (probs * d_vals).sum(axis=-1)  # (..., N_DIGITS)
    place_vals = Tensor(
        np.array([10 ** p for p in range(n_digits)], dtype=np.float32)
    ).cast(exp_dig.dtype)  # (N_DIGITS,) — LSD place values
    return (exp_dig * place_vals).sum(axis=-1)  # (...)


def constraint_energy_v105_5(
    digit_logits_final: Tensor,   # (B, N_MAX, N_DIGITS, 10)
    factor_types: Tensor,          # (B, F_MAX) int
    factor_args: Tensor,           # (B, F_MAX, 3) int
    n_max: int = V105_5_N_MAX,
    f_max: int = V105_5_F_MAX,
    n_digits: int = V105_5_N_DIGITS,
) -> Tensor:
    """Expected-value constraint energy (differentiable, inside JIT) — LSD place values.

    Same formula as constraint_energy_v105 but with LSD-first place values.
    For each factor: E_factor = (E[result] - op(E[arg1], E[arg2]))^2, aggregated
    over valid factors and normalized.
    """
    B = int(digit_logits_final.shape[0])

    # Compute expected value for all variables: (B, N_MAX) — LSD-aware
    ev = _expected_value_v105_5(digit_logits_final, n_digits)  # (B, N_MAX)

    # Gather arg/result expected values via one-hot.
    fa_clamped = factor_args.cast(dtypes.int).clip(0, n_max - 1)  # (B, F_MAX, 3)
    fa_oh = fa_clamped.reshape(B, f_max * 3).one_hot(n_max)       # (B, F_MAX*3, N_MAX)
    ev_bc = ev.reshape(B, 1, n_max).cast(dtypes.float)             # (B, 1, N_MAX)
    gathered = (fa_oh.cast(dtypes.float) * ev_bc).sum(axis=-1)    # (B, F_MAX*3)
    gathered_r = gathered.reshape(B, f_max, 3)                     # (B, F_MAX, 3)
    ev_arg1   = gathered_r[:, :, 0]
    ev_arg2   = gathered_r[:, :, 1]
    ev_result = gathered_r[:, :, 2]

    ev_add = ev_arg1 + ev_arg2
    ev_sub = ev_arg1 - ev_arg2
    ev_mul = ev_arg1 * ev_arg2
    ev_div = ev_arg1 / (ev_arg2.abs() + 1.0)

    ft_clamped = factor_types.cast(dtypes.int).clip(0, 3)
    ft_oh      = ft_clamped.one_hot(4).cast(dtypes.float)
    ev_expected_stack = ev_add.reshape(B, f_max, 1).cat(
        ev_sub.reshape(B, f_max, 1),
        ev_mul.reshape(B, f_max, 1),
        ev_div.reshape(B, f_max, 1),
        dim=-1,
    )
    ev_expected = (ft_oh * ev_expected_stack).sum(axis=-1)

    valid = (factor_types >= 0).cast(dtypes.float)
    residual    = ev_result - ev_expected
    rel_err     = residual.abs() / (ev_expected.abs() + 1.0)
    rel_err_clipped = rel_err.clip(0.0, 10.0)
    energy      = rel_err_clipped * valid
    n_valid     = valid.sum() + 1e-8
    return energy.sum() / n_valid


# ---------------------------------------------------------------------------
# Iterative prefill loop — full stack
# ---------------------------------------------------------------------------

def fg_breathing_forward_v105_5(
    model: Any,
    digit_init: Tensor,       # (B, N_MAX, N_DIGITS, 10)
    node_kinds: Tensor,        # (B, T_MAX) int
    staging_mask: Tensor,      # (B, K_MAX, T_MAX, T_MAX)
    head_op_mask: Tensor,      # (B, N_HEADS, T_MAX, T_MAX)
    K: int,
    n_max: int = V105_5_N_MAX,
    f_max: int = V105_5_F_MAX,
    n_digits: int = V105_5_N_DIGITS,
) -> tuple[
    list[Tensor], list[Tensor], list[Tensor], list[Tensor],
    Tensor, "Tensor | None", "Tensor | None",
]:
    """K iterative-prefill breaths with v105.5 additions.

    Per breath k:
      1. Breath embedding added.
      2. Combined staging + op mask built.
      3. 4 transformer layers run.
      4. Hierarchical IB codebook soft projection (Component 4 v105.5)
         — family attention gates leaf attention.
      5. Projection waist compression  (Component 3 — zero effect at step 0).
      6. Delta gate residual update.
      7. Per-cell magnitude head: 4-way "n_digits" classification, used to
         construct a magnitude_embed added to each digit-position hidden state.
      8. Readout: per-position digit codebook → AR digit logits; factor logits;
         calibration; magnitude logits.
      9. (v105.8 only) At final breath, compute per-NUMBER logits by mean-
         pooling over digit positions and projecting through the
         number_codebook. Per-NUMBER CE replaces per-digit CE in the loss.
     10. (v105.9 only) At final breath, mean-pool var_tokens over digit
         positions to get cell_hidden, then run a small AR digit decoder
         that conditions each digit's logits on cell_hidden + accumulated
         soft embeddings of lower digits. Reuses the existing per-position
         digit codebook as the readout matrices. Per-digit CE on this
         pooled-AR output replaces per-position digit CE in the loss.

    Returns:
      digit_logits_history     : list[K] of (B, N_MAX, N_DIGITS, 10)
      factor_logits_history    : list[K] of (B, F_MAX, N_DIGITS, 10)
      calib_history            : list[K] of (B,)
      magnitude_logits_history : list[K] of (B, N_MAX, N_MAGNITUDE=4)
      terminal_var_hidden      : (B, N_MAX, N_DIGITS, H) — final-breath
        var_tokens AFTER magnitude_embed add (the tensor that the digit
        codebook matmul reads from). Exposed for aux distinctness loss.
      number_logits_final      : (B, N_MAX, N_NUMBER_BINS) at the LAST
        breath, or None when V105_8_PER_NUMBER_READOUT is disabled / the
        number codebook is not attached.
      digit_logits_pooled_final: (B, N_MAX, N_DIGITS, 10) at the LAST breath
        — pooled-cell-hidden AR digit decoder output. None when
        V105_9_AR_DIGIT_DECODER is disabled.
    """
    assert hasattr(model, "fg_v105_5_digit_codebook"), \
        "model has no v105.5 params; was attach_fg_params_v105_5 called?"

    # Component 1+2 params
    # v105.5: digit_codebook now has shape (N_DIGITS, 10, H) — per-position.
    digit_codebook   = model.fg_v105_5_digit_codebook    # (N_DIGITS, 10, H)
    digit_rope_cos   = model.fg_v105_5_digit_rope_cos    # (N_DIGITS, H)
    digit_rope_sin   = model.fg_v105_5_digit_rope_sin    # (N_DIGITS, H)
    var_pos_embed    = model.fg_v105_5_var_pos_embed      # (N_MAX, H)
    factor_pos_embed = model.fg_v105_5_factor_pos_embed  # (F_MAX, H)
    node_kind_embed  = model.fg_v105_5_node_kind_embed   # (3, H)
    breath_embed     = model.fg_v105_5_breath_embed      # (K_max, H)
    delta_gate       = model.fg_v105_5_delta_gate        # (K_max,)
    calib_head_w     = model.fg_v105_5_calib_head_w      # (H, 1)
    calib_head_b     = model.fg_v105_5_calib_head_b      # (1,)

    # Component 3: projection waist params
    W_compress = model.fg_v105_5_W_compress  # (H, waist)
    b_compress = model.fg_v105_5_b_compress  # (waist,)
    W_expand   = model.fg_v105_5_W_expand    # (waist, H) — zero at init
    b_expand   = model.fg_v105_5_b_expand    # (H,)

    # Component 4 (v105.5): hierarchical IB codebook
    ib_codebook       = model.fg_v105_5_ib_codebook        # (N_CODE, H)
    family_centroids  = model.fg_v105_5_family_centroids    # (N_FAMILIES, H)
    leaf_to_family_oh = model.fg_v105_5_leaf_to_family_oh   # (N_CODE, N_FAMILIES) frozen
    delta_gate_quant  = model.fg_v105_5_delta_gate_quant   # (K_max,) — zero at init
    ib_temperature    = model.fg_v105_5_ib_temperature     # ()

    # NEW v105.5 — magnitude head
    magnitude_head_w   = model.fg_v105_5_magnitude_head_w    # (H, N_MAG)
    magnitude_head_b   = model.fg_v105_5_magnitude_head_b    # (N_MAG,)
    magnitude_centroids = model.fg_v105_5_magnitude_centroids # (N_MAG, H)

    # NEW v105.5 — per-position FFN block (PPFFN)
    ppffn_ln_g  = getattr(model, "fg_v105_5_ppffn_ln_g",  None)   # (N_DIGITS, H)
    ppffn_ln_b  = getattr(model, "fg_v105_5_ppffn_ln_b",  None)   # (N_DIGITS, H)
    ppffn_W_in  = getattr(model, "fg_v105_5_ppffn_W_in",  None)   # (N_DIGITS, H, INT)
    ppffn_b_in  = getattr(model, "fg_v105_5_ppffn_b_in",  None)   # (N_DIGITS, INT)
    ppffn_W_out = getattr(model, "fg_v105_5_ppffn_W_out", None)   # (N_DIGITS, INT, H) — zero at init
    ppffn_b_out = getattr(model, "fg_v105_5_ppffn_b_out", None)   # (N_DIGITS, H)      — zero at init

    # NEW v105.6 — per-position W_in for L0 FFN (replacement at L0).
    v6_l0_w_in = getattr(model, "fg_v105_6_l0_perpos_w_in", None)  # (N_DIGITS, H, ffn_int)
    v6_l0_b_in = getattr(model, "fg_v105_6_l0_perpos_b_in", None)  # (N_DIGITS, ffn_int)

    # NEW v105.8 — per-NUMBER codebook for the per-cell readout (collapse-tolerant).
    number_codebook = getattr(model, "fg_v105_8_number_codebook", None)  # (N_BINS, H) | None

    B = int(digit_init.shape[0])
    T = n_max * n_digits + f_max
    H = int(var_pos_embed.shape[1])

    # Initial embedding: per-position digit codebook + RoPE on digit positions
    x = embed_factor_graph_v105_5(
        digit_init, node_kinds,
        var_pos_embed, factor_pos_embed, node_kind_embed,
        digit_codebook, digit_rope_cos, digit_rope_sin,
        n_max=n_max, n_digits=n_digits, f_max=f_max,
    )
    x = x.cast(dtypes.half) if x.dtype != dtypes.half else x

    layers = list(model.block.layers)
    K_max  = int(breath_embed.shape[0])
    assert K <= K_max, f"K={K} exceeds K_max={K_max}"

    from mycelium.breathing import _layernorm

    digit_logits_history     = []
    factor_logits_history    = []
    calib_history            = []
    magnitude_logits_history = []
    terminal_var_hidden: Tensor | None = None
    # v105.8 — per-NUMBER readout computed at LAST breath only (see step 9 below).
    number_logits_final: Tensor | None = None
    # v105.9 — pooled-cell AR digit decoder output at LAST breath only
    # (see step 10 below).
    digit_logits_pooled_final: Tensor | None = None

    for k in range(K):
        # 1. Breath embedding
        be_k  = breath_embed[k].reshape(1, 1, -1).cast(x.dtype)
        x_in  = x + be_k
        x_pre = x

        # 2. Combined mask (B, N_HEADS, T, T)
        stk   = staging_mask[:, k, :, :]     # (B, T, T)
        stk_h = stk.reshape(B, 1, T, T).expand(B, V105_5_N_HEADS, T, T)
        combined = stk_h.cast(x.dtype) + head_op_mask.cast(x.dtype)

        # 2.5. Per-position FFN block (MOVED to BEFORE L0 as of 2026-06-02
        # after per-layer cos_sim diagnostic revealed L0 does the bulk of
        # the collapse (input cos 0.74 → post-L0 cos 0.90). Placing PPFFN
        # BEFORE L0 amplifies per-position distinction so L0's averaging
        # starts from a wider spread. Output is zero at init (W_out + b_out
        # zero-initialized), so step-0 forward is byte-identical to v105.4.
        if V105_5_PERPOS_FFN and ppffn_W_out is not None:
            x_in = apply_perpos_ffn(
                x_in, ppffn_ln_g, ppffn_ln_b,
                ppffn_W_in, ppffn_b_in, ppffn_W_out, ppffn_b_out,
                n_max=n_max, n_digits=n_digits, f_max=f_max,
                eps=model.cfg.layer_norm_eps,
            )

        # 3. Four transformer layers.
        # v105.6: L0 may use per-position W_in for digit tokens (replacement
        # path — no shared Pythia w_in fallback at digit positions). Gated
        # by V105_6_PERPOS_L0; the per-position tensors are attached iff
        # the env var is set, so the hasattr() check is a defensive
        # belt-and-suspenders against partial state.
        h = x_in
        if V105_6_PERPOS_L0 and v6_l0_w_in is not None:
            h = fg_layer_forward_v105_6_l0(
                layers[0], h, combined,
                v6_l0_w_in, v6_l0_b_in,
                n_max=n_max, n_digits=n_digits, f_max=f_max,
            )
        else:
            h = fg_layer_forward_v105_5(layers[0], h, combined)
        for layer in layers[1:4]:
            h = fg_layer_forward_v105_5(layer, h, combined)

        # 4. Hierarchical IB semantic codebook soft projection (v105.5 addition 3)
        h = apply_hierarchical_ib_codebook(
            h, ib_codebook, family_centroids, leaf_to_family_oh,
            ib_temperature, delta_gate_quant[k],
        )

        # 5. Projection waist compression  (Component 3)
        # v105.13 wave-guide: if model has a waveguide_mask attached, the LoRA
        # correction is masked so the preserve channel passes through untouched.
        waveguide_mask = getattr(model, "fg_v105_5_waveguide_mask", None)
        h = apply_projection_waist(
            h, W_compress, W_expand, b_compress, b_expand,
            waveguide_mask=waveguide_mask,
        )

        # 6. Delta gate residual update
        gate_k = delta_gate[k].cast(h.dtype).reshape(1, 1, 1)
        delta  = h - x_pre
        x      = x_pre + gate_k * delta

        # 7. Per-cell magnitude head (NEW v105.5)
        # v105.12 PREFILL_ISOLATE: when enabled, the magnitude head + magnitude_embed
        # addition + per-position digit codebook readout all fire ONLY at the
        # decode breath (k == K-1). The breathing loop becomes pure constraint
        # propagation; the readout pipeline runs once at the end.
        x_ln = _layernorm(x, model.ln_f_g, model.ln_f_b, model.cfg.layer_norm_eps).cast(dtypes.float)
        n_var_tokens = n_max * n_digits
        var_tokens   = x_ln[:, :n_var_tokens, :]
        var_tokens_r = var_tokens.reshape(B, n_max, n_digits, -1)   # (B, N_MAX, N_DIGITS, H)

        is_decode_breath = (k == K - 1)
        run_readout      = (not V105_12_PREFILL_ISOLATE) or is_decode_breath

        if run_readout:
            # cell_hidden = mean over the 5 digit positions for that cell.
            cell_hidden = var_tokens_r.mean(axis=2)                      # (B, N_MAX, H)
            mh_w = magnitude_head_w.cast(dtypes.float)
            mh_b = magnitude_head_b.cast(dtypes.float)
            magnitude_logits = cell_hidden @ mh_w + mh_b.reshape(1, 1, -1)  # (B, N_MAX, N_MAG)
            magnitude_logits_history.append(magnitude_logits)

            magnitude_probs  = magnitude_logits.softmax(axis=-1)          # (B, N_MAX, N_MAG)
            mc = magnitude_centroids.cast(dtypes.float)                   # (N_MAG, H)
            magnitude_embed_cell = magnitude_probs @ mc                    # (B, N_MAX, H)
            # Broadcast magnitude_embed across all digit positions of each cell.
            magnitude_embed_dg = magnitude_embed_cell.reshape(
                B, n_max, 1, -1
            ).expand(B, n_max, n_digits, int(var_tokens_r.shape[-1]))     # (B, N_MAX, N_DIGITS, H)

            var_tokens_r = var_tokens_r + magnitude_embed_dg               # add magnitude_embed
        else:
            # v105.12 PREFILL_ISOLATE early breath: emit zero placeholders for
            # magnitude_logits to keep history length == K (JIT tuple stability).
            # var_tokens_r is NOT modified (no magnitude_embed feedback into
            # the residual stream's read-out tap). Tensor.zeros is OK inside JIT
            # because shapes are static.
            magnitude_logits = Tensor.zeros(
                (B, n_max, V105_5_N_MAGNITUDE), dtype=dtypes.float,
            ).contiguous()
            magnitude_logits_history.append(magnitude_logits)

        # Save terminal hidden state from the LAST breath for the aux
        # distinctness loss. This is the tensor the digit codebook readout
        # consumes — same tap point as the linear-probe diagnostic.
        if is_decode_breath:
            terminal_var_hidden = var_tokens_r

        # 8a. Per-position digit codebook readout (v105.5 addition 2).
        # digit_codebook: (N_DIGITS, 10, H). For AR, per-iter codebook is digit_codebook[p].
        cb_fp_all = digit_codebook.cast(dtypes.float)  # (N_DIGITS, 10, H)

        if run_readout:
            if V105_5_AR_DIGITS:
                ar_logits_list: list[Tensor] = [None] * n_digits  # type: ignore
                cond_accum = Tensor.zeros(
                    (B, n_max, int(x_ln.shape[-1])), dtype=dtypes.float
                ).contiguous()
                ar_cond_scale_t = Tensor(
                    np.array([float(V105_5_AR_COND_SCALE)], dtype=np.float32),
                    dtype=dtypes.float,
                ).reshape(1, 1, 1)

                if V105_5_AR_MSD_FIRST:
                    ar_iter = range(n_digits - 1, -1, -1)
                else:
                    ar_iter = range(n_digits)   # LSD-first default

                for p in ar_iter:
                    cb_p = cb_fp_all[p]                                    # (10, H)
                    pos_hidden = var_tokens_r[:, :, p, :] + cond_accum     # (B, N_MAX, H)
                    pos_logits = pos_hidden @ cb_p.T                       # (B, N_MAX, 10)
                    ar_logits_list[p] = pos_logits
                    pos_probs = pos_logits.softmax(axis=-1)                 # (B, N_MAX, 10)
                    pos_embed = pos_probs @ cb_p                           # (B, N_MAX, H)
                    cond_accum = cond_accum + pos_embed * ar_cond_scale_t.cast(pos_embed.dtype)

                digit_logits_k = Tensor.stack(*ar_logits_list, dim=2)      # (B, N_MAX, N_DIGITS, 10)
            else:
                # Parallel — but with per-position codebook each iteration is independent.
                parallel_logits_list: list[Tensor] = []
                for p in range(n_digits):
                    cb_p = cb_fp_all[p]                                    # (10, H)
                    pos_hidden = var_tokens_r[:, :, p, :]                  # (B, N_MAX, H)
                    pos_logits = pos_hidden @ cb_p.T                       # (B, N_MAX, 10)
                    parallel_logits_list.append(pos_logits)
                digit_logits_k = Tensor.stack(*parallel_logits_list, dim=2)
        else:
            # v105.12 PREFILL_ISOLATE early breath: zero placeholder.
            digit_logits_k = Tensor.zeros(
                (B, n_max, n_digits, 10), dtype=dtypes.float,
            ).contiguous()

        digit_logits_history.append(digit_logits_k)

        # 8b. Factor digit logits — also use per-position codebook.
        fac_tokens   = x_ln[:, n_var_tokens:n_var_tokens + f_max, :]
        # We don't add magnitude_embed for factor cells (factors are not variables
        # — they carry result digits but their cell_hidden derivation differs).
        if run_readout:
            fac_logits_list: list[Tensor] = []
            for p in range(n_digits):
                cb_p = cb_fp_all[p]                                        # (10, H)
                fac_logits_p = fac_tokens @ cb_p.T                          # (B, F_MAX, 10)
                fac_logits_list.append(fac_logits_p)
            fac_logits_k = Tensor.stack(*fac_logits_list, dim=2)           # (B, F_MAX, N_DIGITS, 10)
        else:
            # v105.12 PREFILL_ISOLATE early breath: zero placeholder.
            fac_logits_k = Tensor.zeros(
                (B, f_max, n_digits, 10), dtype=dtypes.float,
            ).contiguous()
        factor_logits_history.append(fac_logits_k)

        # 8c. Calibration head (always run — depends only on x_ln, not the readout).
        pool        = x_ln.mean(axis=1)
        calib_logit = pool @ calib_head_w.cast(dtypes.float) + calib_head_b.cast(dtypes.float)
        calib_k     = calib_logit.reshape(-1).sigmoid()
        calib_history.append(calib_k)

        # 9. (v105.8) Per-NUMBER readout at LAST breath only.
        # cell_hidden = mean of post-magnitude var_tokens_r across digit positions.
        # number_logits = cell_hidden @ number_codebook.T : (B, N_MAX, N_BINS)
        # This is intentionally a SHARED projection across the 5 digit positions:
        # positions ARE allowed to collapse (mean is still a number); the per-position
        # tokens act as thinking slots, not digit decoders. Training-side: per-number
        # CE on bin index (see _compile_jit_fg_step_v105_5).
        if k == K - 1 and V105_8_PER_NUMBER_READOUT and number_codebook is not None:
            number_cell_hidden = var_tokens_r.mean(axis=2)               # (B, N_MAX, H)
            ncb = number_codebook.cast(dtypes.float)                      # (N_BINS, H)
            number_logits_final = number_cell_hidden @ ncb.T              # (B, N_MAX, N_BINS)

        # 10. (v105.9) Pooled-cell AR digit decoder at LAST breath only.
        # Reads from cell_hidden (mean of post-magnitude var_tokens across the
        # 5 digit positions) and runs the existing AR chain (same algorithm
        # as step 8a) but starting from a single (B, N_MAX, H) hidden state
        # per cell. Per-digit CE loss on this output flows back through ONE
        # pooled hidden state — model encodes "the number is X" in
        # cell_hidden; this small decoder extracts the digits sequentially.
        # LSD-first (matches v105.5 convention): position p=0 is ones.
        #
        # v105.11 additions (gated by env vars):
        #   - V105_11_CONCAT_COND=1: replace additive cond_pool with
        #     concat(cell_hidden, cond_pool) @ W_concat. Forces the model
        #     to process prior digit embeddings (they're in d_p's INPUT
        #     dimensions, not optional). Mechanism 2 from the v105.11 design.
        #   - V105_11_COND_DROPOUT>0: during training, randomly zero out
        #     cell_hidden_pool for positions p >= 1 per (batch, cell).
        #     Forces d_p to learn to use d_{p-1} when cell_hidden is gone.
        #     Mechanism 3 from the v105.11 design.
        if k == K - 1 and V105_9_AR_DIGIT_DECODER:
            cell_hidden_pool = var_tokens_r.mean(axis=2)                # (B, N_MAX, H)
            pooled_logits_list: list[Tensor] = [None] * n_digits  # type: ignore
            H_local = int(x_ln.shape[-1])
            cond_pool = Tensor.zeros(
                (B, n_max, H_local), dtype=dtypes.float,
            ).contiguous()
            ar_cond_scale_pool = Tensor(
                np.array([float(V105_9_AR_COND_SCALE)], dtype=np.float32),
                dtype=dtypes.float,
            ).reshape(1, 1, 1)

            # v105.11 Mechanism 2: concat-and-project conditioning.
            v11_concat_W = getattr(model, "fg_v105_11_concat_W", None)
            v11_concat_enabled = (
                V105_11_CONCAT_COND and v11_concat_W is not None
            )
            v11_concat_W_f = (
                v11_concat_W.cast(dtypes.float) if v11_concat_enabled else None
            )

            # v105.11 Mechanism 3: conditional dropout on cell_hidden for
            # positions p >= 1. Sample one Bernoulli mask of shape
            # (B, n_max, n_digits, 1) once before the loop; index per p.
            # Mask is 1 at p=0 unconditionally (no dropout on the first digit).
            v11_dropout_enabled = (
                V105_11_COND_DROPOUT > 0.0
                and bool(Tensor.training)
            )
            v11_dropout_mask: Tensor | None = None
            if v11_dropout_enabled:
                rand_mask = (
                    Tensor.rand(B, n_max, n_digits, 1, dtype=dtypes.float)
                    > float(V105_11_COND_DROPOUT)
                ).cast(dtypes.float)
                # Force p=0 to keep cell_hidden (no dropout on ones digit).
                # Build a (1, 1, n_digits, 1) mask with mask[..., 0, :] = 1.
                # by combining np-init "force_keep" tensor with the rand mask.
                # We compute the mask as (rand_mask OR force_keep_p0).
                # force_keep_p0_np[0,0,0,0] = 1; all others = 0.
                # Then mask = rand_mask + force_keep_p0 - (rand_mask * force_keep_p0)
                # is an OR equivalent for binary {0,1} masks. We just clamp <= 1.
                force_p0_np = np.zeros((1, 1, n_digits, 1), dtype=np.float32)
                force_p0_np[0, 0, 0, 0] = 1.0
                force_p0_t = Tensor(
                    force_p0_np, dtype=dtypes.float
                ).contiguous()
                v11_dropout_mask = (rand_mask + force_p0_t).clip(0.0, 1.0)

            # v105.11 Mechanism 4: LayerNorm cell_hidden and cond_pool before
            # combining to equalize magnitudes. Without LN, cell_hidden (mag
            # ~sqrt(H)=32) dominates cond_pool (mag ~cond_scale=2) by 16×,
            # making the d_embed signal ignorable. LN forces both to mag
            # ~sqrt(H), so the decoder must process both components.
            # Uses model.ln_f_g/ln_f_b (existing trained final-LN params) for
            # both — both vectors live in the same 1024d residual-stream space.
            v11_ln_enabled = bool(V105_11_LN_COND)
            ln_g_f = model.ln_f_g.cast(dtypes.float) if v11_ln_enabled else None
            ln_b_f = model.ln_f_b.cast(dtypes.float) if v11_ln_enabled else None
            ln_eps = model.cfg.layer_norm_eps

            # LSD-first iteration: p=0 (ones) first.
            for p in range(n_digits):
                cb_p = cb_fp_all[p]                                       # (10, H)

                # Optional conditional dropout on cell_hidden (mechanism 3).
                if v11_dropout_mask is not None:
                    drop_p = v11_dropout_mask[:, :, p, :]                  # (B, n_max, 1)
                    cell_hidden_eff = cell_hidden_pool * drop_p
                else:
                    cell_hidden_eff = cell_hidden_pool

                # Mechanism 4: LN both components before combining (equalize mags).
                if v11_ln_enabled:
                    cell_hidden_eff = _layernorm(
                        cell_hidden_eff, ln_g_f, ln_b_f, ln_eps,
                    )
                    cond_pool_for_combine = _layernorm(
                        cond_pool, ln_g_f, ln_b_f, ln_eps,
                    )
                else:
                    cond_pool_for_combine = cond_pool

                # Conditioning: additive (default) or concat-and-project (v105.11 m2).
                if v11_concat_enabled:
                    cat_in = Tensor.cat(
                        cell_hidden_eff, cond_pool_for_combine, dim=-1
                    )                                                       # (B, N_MAX, 2H)
                    pooled_in = cat_in @ v11_concat_W_f                      # (B, N_MAX, H)
                else:
                    pooled_in = cell_hidden_eff + cond_pool_for_combine      # (B, N_MAX, H)

                pooled_log  = pooled_in @ cb_p.T                            # (B, N_MAX, 10)
                pooled_logits_list[p] = pooled_log
                pooled_prob = pooled_log.softmax(axis=-1)
                pooled_emb  = pooled_prob @ cb_p                            # (B, N_MAX, H)
                cond_pool   = cond_pool + pooled_emb * ar_cond_scale_pool.cast(pooled_emb.dtype)
            digit_logits_pooled_final = Tensor.stack(
                *pooled_logits_list, dim=2
            )                                                              # (B, N_MAX, N_DIGITS, 10)

    assert terminal_var_hidden is not None, \
        "terminal_var_hidden was not set; K must be >= 1"
    return (
        digit_logits_history, factor_logits_history,
        calib_history, magnitude_logits_history,
        terminal_var_hidden, number_logits_final,
        digit_logits_pooled_final,
    )


# ---------------------------------------------------------------------------
# Parameter attachment
# ---------------------------------------------------------------------------

def attach_fg_params_v105_5(
    model: Any,
    hidden: int,
    n_digits: int = V105_5_N_DIGITS,
    n_max: int = V105_5_N_MAX,
    f_max: int = V105_5_F_MAX,
    k_max: int | None = None,
    waist: int | None = None,
    n_code: int | None = None,
    rope_base: float | None = None,
    ib_centroids_path: str | None = None,
    ib_tree_path: str | None = None,
    ib_init: bool | None = None,
    waist_lora_init: bool | None = None,
    n_families: int = V105_5_N_FAMILIES,
    n_magnitude: int = V105_5_N_MAGNITUDE,
) -> None:
    """Attach v105.5 params to model.

    v105.5 additions vs v105.3:
      • per-position digit_codebook (n_digits, 10, hidden)
      • family_centroids (n_families, hidden) initialized from IB tree means
      • leaf_to_family_oh (n_code, n_families) frozen one-hot lookup
      • magnitude_head_w (hidden, n_magnitude)
      • magnitude_head_b (n_magnitude,) — zeros
      • magnitude_centroids (n_magnitude, hidden) — small random init

    All v105.5 additions are wired so that initial behavior reduces to v105.3
    where possible. The hierarchical IB and magnitude head produce non-zero
    output at step 0 (cannot trivially zero-out since they're additive in
    the residual stream), but their effect is small because:
      - delta_gate_quant = 0  → hierarchical IB has zero residual contribution
      - magnitude_centroids ≈ 0.02 std → magnitude_embed is small
    """
    if k_max is None:
        k_max = V105_5_K_MAX
    if waist is None:
        waist = V105_5_WAIST
    if n_code is None:
        n_code = V105_5_CODEBOOK_N
    if rope_base is None:
        rope_base = V105_5_ROPE_BASE
    if ib_centroids_path is None:
        ib_centroids_path = V105_5_IB_CENTROIDS
    if ib_tree_path is None:
        ib_tree_path = V105_5_IB_TREE
    if ib_init is None:
        ib_init = V105_5_IB_INIT
    if waist_lora_init is None:
        waist_lora_init = V105_5_WAIST_LORA_INIT

    rng = np.random.RandomState(20013)

    # -----------------------------------------------------------------------
    # Components 1+2: PER-POSITION digit codebook (v105.5 addition 2)
    # + frozen RoPE tables (LSD layout)
    # -----------------------------------------------------------------------
    # Build n_digits independent codebooks, each (10, hidden).
    # Init order of precedence: v105.12 Fourier > V105_5_FOURIER_INIT > QR-random.
    if V105_12_FOURIER_DECODE_INIT:
        # v105.12 Fourier basis — same structure at every position. Combined
        # with V105_12_PREFILL_ISOLATE this codebook is touched ONLY at the
        # decode breath, so the structure isn't washed out by Pythia layers.
        dc_per_pos = _v105_12_fourier_decode_init(
            n_digits_pos=n_digits, hidden=hidden,
        )
    else:
        dc_per_pos = np.zeros((n_digits, 10, hidden), dtype=np.float32)
        for p in range(n_digits):
            if V105_5_FOURIER_INIT:
                dc_per_pos[p] = _fourier_digit_codebook(n_digits=10, hidden=hidden)
            else:
                rng_p = np.random.RandomState(20013 + p)
                raw_dc = rng_p.randn(max(hidden, 10), hidden).astype(np.float32)
                q_dc, _ = np.linalg.qr(raw_dc)
                dc_per_pos[p] = q_dc[:10].astype(np.float32) * 1.0
    model.fg_v105_5_digit_codebook = Tensor(dc_per_pos, dtype=dtypes.float).contiguous()

    # LSD-first array layout: array index i ↔ RoPE position i naturally.
    cos_t, sin_t = _precompute_digit_rope(n_digits, hidden, base=rope_base)
    model.fg_v105_5_digit_rope_cos = Tensor(cos_t, dtype=dtypes.float).contiguous()
    model.fg_v105_5_digit_rope_sin = Tensor(sin_t, dtype=dtypes.float).contiguous()

    # -----------------------------------------------------------------------
    # Position / kind embeddings
    # -----------------------------------------------------------------------
    vp = rng.randn(n_max, hidden).astype(np.float32) * 0.02
    model.fg_v105_5_var_pos_embed = Tensor(vp, dtype=dtypes.float).contiguous()

    fp_emb = rng.randn(f_max, hidden).astype(np.float32) * 0.02
    model.fg_v105_5_factor_pos_embed = Tensor(fp_emb, dtype=dtypes.float).contiguous()

    nk = rng.randn(3, hidden).astype(np.float32) * 0.02
    model.fg_v105_5_node_kind_embed = Tensor(nk, dtype=dtypes.float).contiguous()

    # -----------------------------------------------------------------------
    # Calibration head
    # -----------------------------------------------------------------------
    cw = (rng.randn(hidden, 1) * 0.02).astype(np.float32)
    model.fg_v105_5_calib_head_w = Tensor(cw, dtype=dtypes.float).contiguous()
    model.fg_v105_5_calib_head_b = Tensor.zeros((1,), dtype=dtypes.float).contiguous()

    # -----------------------------------------------------------------------
    # Per-breath embeddings + delta gate
    # -----------------------------------------------------------------------
    rng_be = np.random.RandomState(20014)
    raw_be = rng_be.randn(max(k_max, hidden), hidden).astype(np.float32)
    q_be, _ = np.linalg.qr(raw_be)
    be = q_be[:k_max].astype(np.float32) * 0.5
    model.fg_v105_5_breath_embed = Tensor(be, dtype=dtypes.float).contiguous()
    model.fg_v105_5_delta_gate   = Tensor.ones((k_max,), dtype=dtypes.float).contiguous()

    # -----------------------------------------------------------------------
    # Component 3: Projection waist  (LoRA-style)
    # -----------------------------------------------------------------------
    W_c = (rng.randn(hidden, waist) * 0.02).astype(np.float32)
    model.fg_v105_5_W_compress = Tensor(W_c, dtype=dtypes.float).contiguous()
    model.fg_v105_5_b_compress = Tensor.zeros((waist,), dtype=dtypes.float).contiguous()

    if waist_lora_init:
        model.fg_v105_5_W_expand = Tensor.zeros((waist, hidden), dtype=dtypes.float).contiguous()
    else:
        We = (rng.randn(waist, hidden) * 0.02).astype(np.float32)
        model.fg_v105_5_W_expand = Tensor(We, dtype=dtypes.float).contiguous()
    model.fg_v105_5_b_expand = Tensor.zeros((hidden,), dtype=dtypes.float).contiguous()

    # -----------------------------------------------------------------------
    # Component 4 (v105.5): hierarchical IB semantic codebook
    # -----------------------------------------------------------------------
    if ib_init:
        cb_np = load_ib_centroids(ib_centroids_path, n_code=n_code, hidden=hidden)
    else:
        raw_cb = rng.randn(max(hidden, n_code), hidden).astype(np.float32)
        q_cb, _ = np.linalg.qr(raw_cb)
        cb_np = q_cb[:n_code].astype(np.float32) * 0.5
        print(f"[v105.5] IB init disabled; random QR codebook ({n_code}, {hidden})")

    model.fg_v105_5_ib_codebook = Tensor(cb_np, dtype=dtypes.float).contiguous()
    model.fg_v105_5_delta_gate_quant = Tensor.zeros((k_max,), dtype=dtypes.float).contiguous()
    model.fg_v105_5_ib_temperature   = Tensor(
        np.array([1.0], dtype=np.float32), dtype=dtypes.float
    ).contiguous()

    # Build leaf_to_family from IB tree JSON.
    l2f_np, leaf_ids = _load_leaf_to_family(
        ib_tree_path, n_code=n_code, n_families=n_families
    )
    # Frozen int → frozen one-hot float for matmul-style gather.
    l2f_oh = np.zeros((n_code, n_families), dtype=np.float32)
    for i in range(n_code):
        l2f_oh[i, int(l2f_np[i])] = 1.0
    model.fg_v105_5_leaf_to_family    = Tensor(l2f_np, dtype=dtypes.int).contiguous()
    model.fg_v105_5_leaf_to_family_oh = Tensor(l2f_oh, dtype=dtypes.float).contiguous()

    # family_centroids = mean of IB centroids in each family.
    family_centroids = np.zeros((n_families, hidden), dtype=np.float32)
    counts = np.zeros((n_families,), dtype=np.float32)
    for i in range(n_code):
        fam = int(l2f_np[i])
        family_centroids[fam] += cb_np[i]
        counts[fam] += 1.0
    for f in range(n_families):
        if counts[f] > 0:
            family_centroids[f] /= counts[f]
        else:
            family_centroids[f] = rng.randn(hidden).astype(np.float32) * 0.02
    model.fg_v105_5_family_centroids = Tensor(family_centroids, dtype=dtypes.float).contiguous()

    # -----------------------------------------------------------------------
    # NEW v105.5 — magnitude head + magnitude_centroids
    # -----------------------------------------------------------------------
    mh_w = (rng.randn(hidden, n_magnitude) * 0.02).astype(np.float32)
    model.fg_v105_5_magnitude_head_w = Tensor(mh_w, dtype=dtypes.float).contiguous()
    model.fg_v105_5_magnitude_head_b = Tensor.zeros((n_magnitude,), dtype=dtypes.float).contiguous()
    mag_c = (rng.randn(n_magnitude, hidden) * 0.02).astype(np.float32)
    model.fg_v105_5_magnitude_centroids = Tensor(mag_c, dtype=dtypes.float).contiguous()

    # -----------------------------------------------------------------------
    # NEW v105.5 — Per-position FFN block (PPFFN).
    #
    # 5 separate FFNs (LayerNorm + W_in → GELU → W_out), one per digit position.
    # W_out and b_out are ZERO-INITIALIZED so the PPFFN's residual contribution
    # is identically zero at step 0 — forward pass byte-identical to v105.4.
    #
    # W_in : small-scale Kaiming-style random (N(0, scale²·2/H))
    # b_in : zeros
    # ln_g : ones (LayerNorm gain)
    # ln_b : zeros (LayerNorm bias)
    #
    # ATTACH-GATED by V105_5_PERPOS_FFN: when disabled (e.g. v105.6), the PPFFN
    # tensors are NOT attached, so they don't enter `fg_v105_5_parameters()` and
    # don't trigger the AdamW "missing grad" assertion. (Previously the params
    # were always attached but only USED in forward when PERPOS_FFN=1.)
    # -----------------------------------------------------------------------
    intermediate = int(V105_5_PERPOS_FFN_DIM)
    ppffn_params = 0
    if V105_5_PERPOS_FFN:
        rng_ppffn = np.random.RandomState(20015)

        # W_in : Kaiming-like init.  Scale chosen so initial activations are O(1).
        kaiming_scale = math.sqrt(2.0 / max(hidden, 1))
        pp_W_in_np = (
            rng_ppffn.randn(n_digits, hidden, intermediate).astype(np.float32)
            * kaiming_scale * 0.02
        )
        model.fg_v105_5_ppffn_W_in = Tensor(pp_W_in_np, dtype=dtypes.float).contiguous()
        model.fg_v105_5_ppffn_b_in = Tensor.zeros(
            (n_digits, intermediate), dtype=dtypes.float
        ).contiguous()

        # W_out : small Kaiming-like init (was ZEROS, but PPFFN was gradient-locked
        # at zero — trained W_out norm 2.45 only, per-element mean_abs 5.8e-4, so
        # PPFFN contributed functionally zero output. Non-zero init lets gradient
        # flow through both W_in and W_out paths from step 0. Per-element scale ~
        # 6e-4 — small enough that step-0 forward is nearly identical to baseline.)
        kaiming_scale_out = math.sqrt(2.0 / max(intermediate, 1))
        pp_W_out_np = (
            rng_ppffn.randn(n_digits, intermediate, hidden).astype(np.float32)
            * kaiming_scale_out * 0.02
        )
        model.fg_v105_5_ppffn_W_out = Tensor(pp_W_out_np, dtype=dtypes.float).contiguous()
        # b_out stays zero — bias adds nothing useful at init.
        model.fg_v105_5_ppffn_b_out = Tensor.zeros(
            (n_digits, hidden), dtype=dtypes.float
        ).contiguous()

        # LayerNorm: gain=1, bias=0 per position.
        pp_ln_g_np = np.ones((n_digits, hidden), dtype=np.float32)
        pp_ln_b_np = np.zeros((n_digits, hidden), dtype=np.float32)
        model.fg_v105_5_ppffn_ln_g = Tensor(pp_ln_g_np, dtype=dtypes.float).contiguous()
        model.fg_v105_5_ppffn_ln_b = Tensor(pp_ln_b_np, dtype=dtypes.float).contiguous()

        ppffn_params = (
            n_digits * hidden * intermediate            # W_in
            + n_digits * intermediate                   # b_in
            + n_digits * intermediate * hidden          # W_out
            + n_digits * hidden                         # b_out
            + 2 * n_digits * hidden                     # ln_g + ln_b
        )

    # -----------------------------------------------------------------------
    # NEW v105.6 — Per-position W_in for L0 FFN (REPLACEMENT, not add-alongside).
    #
    # When V105_6_PERPOS_L0 is enabled, the L0 layer's FFN W_in is split:
    #   • Digit tokens → per-position W_in / b_in (n_digits independent matrices)
    #   • Factor tokens → shared Pythia w_in / b_in (unchanged)
    # No shared Pythia w_in fallback exists for digit tokens at L0, so the
    # gradient MUST flow through per-position weights (no path-suppression).
    #
    # Init: copy of Pythia L0's w_in for every position, plus tiny Gaussian
    # noise (0.1% of weight std). Without noise, all 5 positions would be
    # identical at step 0 and evolve identically — same gradient-locked
    # trap as zero-init PPFFN. With noise, symmetry breaks while step-0
    # forward stays very close to the warm-start baseline.
    # -----------------------------------------------------------------------
    v6_l0_params = 0
    if V105_6_PERPOS_L0:
        # Pythia weights are often fp16 — promote to fp32 BEFORE numpy
        # statistics to avoid fp32-reduction overflow on the 4M-element
        # w_in tensor (observed: `abs(w_fp16).std()` → inf when computed
        # via numpy's default fp32 accumulator).
        pythia_l0_w_in_np = (
            model.block.layers[0].w_in.numpy().astype(np.float32)
        )  # (H, ffn_int)
        pythia_l0_b_in_np = (
            model.block.layers[0].b_in.numpy().astype(np.float32)
        )  # (ffn_int,)
        ffn_int_l0 = int(pythia_l0_w_in_np.shape[1])
        # Use fp64 accumulator for std — fp32 std on 4M elements can overflow.
        w_std = float(np.abs(pythia_l0_w_in_np.astype(np.float64)).std())
        noise_scale = 0.001 * w_std  # 0.1% of weight std

        rng_l0 = np.random.RandomState(20016)
        perpos_w_in_np = np.stack([
            pythia_l0_w_in_np
            + rng_l0.randn(*pythia_l0_w_in_np.shape).astype(np.float32) * noise_scale
            for _ in range(n_digits)
        ], axis=0).astype(np.float32)  # (n_digits, H, ffn_int)
        perpos_b_in_np = np.stack(
            [pythia_l0_b_in_np for _ in range(n_digits)], axis=0
        ).astype(np.float32)            # (n_digits, ffn_int)

        model.fg_v105_6_l0_perpos_w_in = Tensor(
            perpos_w_in_np, dtype=dtypes.float,
        ).contiguous()
        model.fg_v105_6_l0_perpos_b_in = Tensor(
            perpos_b_in_np, dtype=dtypes.float,
        ).contiguous()

        v6_l0_params = (
            n_digits * hidden * ffn_int_l0    # W_in
            + n_digits * ffn_int_l0            # b_in
        )
        print(
            f"[v105.6] perpos L0 w_in init: pythia_w_std={w_std:.6f} "
            f"noise_scale={noise_scale:.6f}",
            flush=True,
        )

    # -----------------------------------------------------------------------
    # NEW v105.8 — per-NUMBER codebook (n_bins, hidden) for per-cell readout.
    # Attached only when V105_8_PER_NUMBER_READOUT is enabled. Random
    # orthonormal init (× 0.1) follows v107's domain_codebook scale.
    # -----------------------------------------------------------------------
    v8_number_params = 0
    if V105_8_PER_NUMBER_READOUT:
        n_bins = int(V105_8_N_NUMBER_BINS)
        rng_v8 = np.random.RandomState(20017)
        raw_ncb = rng_v8.randn(max(hidden, n_bins), hidden).astype(np.float32)
        q_ncb, _ = np.linalg.qr(raw_ncb)
        ncb_unit = q_ncb[:n_bins].astype(np.float32)               # (n_bins, hidden) orthonormal
        model.fg_v105_8_number_codebook = Tensor(
            ncb_unit * 0.1, dtype=dtypes.float
        ).contiguous()
        v8_number_params = n_bins * hidden

    # -----------------------------------------------------------------------
    # NEW v105.11 — concat-and-project conditioning projection (Mechanism 2).
    #
    # Replaces the additive cond_pool with concat(cell_hidden, cond_pool)
    # projected to H. Forces the AR decoder to process prior digit embeddings
    # because they're in d_p's INPUT dimensions, not optionally added.
    #
    # Init: block-stacked identity matrices [[I]; [I]] so step-0 forward gives
    #   (cell_hidden || cond_pool) @ W = cell_hidden + cond_pool
    # which matches the additive baseline. Training-time gradient flows
    # independently into the cell_hidden block and the cond_pool block.
    #
    # Attached only when V105_11_CONCAT_COND is enabled.
    # -----------------------------------------------------------------------
    v11_concat_params = 0
    if V105_11_CONCAT_COND:
        # (2H, H) block-stacked identity: top half = I, bottom half = I.
        # Result: (cell_hidden, cond_pool) @ W = cell_hidden + cond_pool at init.
        eye_H = np.eye(hidden, dtype=np.float32)
        concat_W_np = np.concatenate([eye_H, eye_H], axis=0)  # (2H, H)
        model.fg_v105_11_concat_W = Tensor(
            concat_W_np, dtype=dtypes.float
        ).contiguous()
        v11_concat_params = 2 * hidden * hidden

    T = n_max * n_digits + f_max
    if V105_12_FOURIER_DECODE_INIT:
        _dc_init_label = "v105.12 FOURIER (1/sqrt(H) scale)"
    elif V105_5_FOURIER_INIT:
        _dc_init_label = "FOURIER"
    else:
        _dc_init_label = "QR-random"
    print(
        f"[v105.5] params attached:\n"
        f"  digit_codebook=(N_DIGITS={n_digits}, 10, {hidden}) PER-POSITION "
        f"init={_dc_init_label}, "
        f"digit_rope (N_DIGITS={n_digits}, H={hidden}, base={rope_base:.0f}) [FROZEN]\n"
        f"  LSD layout: idx 0=ones (RoPE pos 0)\n"
        f"  loss_mode={'NUMBER_MSE_ONLY' if V105_5_NUMBER_MSE_ONLY else 'per-digit CE'}\n"
        f"  ar_digits={V105_5_AR_DIGITS} (cond_scale={V105_5_AR_COND_SCALE if V105_5_AR_DIGITS else 'N/A'}, "
        f"dir={'MSD-first' if V105_5_AR_MSD_FIRST else 'LSD-first'})\n"
        f"  var_pos_embed=({n_max},{hidden}), factor_pos_embed=({f_max},{hidden})\n"
        f"  waist=({hidden}→{waist}→{hidden}), W_expand={'ZEROS' if waist_lora_init else 'random'}\n"
        f"  ib_codebook=({n_code},{hidden}), family_centroids=({n_families},{hidden})\n"
        f"  leaf_to_family[:8]={l2f_np[:8].tolist()}\n"
        f"  leaf_ids[:4]={leaf_ids[:4]}\n"
        f"  magnitude_head=({hidden},{n_magnitude}), centroids=({n_magnitude},{hidden})\n"
        f"  PPFFN: enabled={V105_5_PERPOS_FFN} intermediate={intermediate} "
        f"params={ppffn_params/1e6:.1f}M [W_out,b_out ZEROS → inert at step 0]\n"
        f"  v105.6 PERPOS_L0: enabled={V105_6_PERPOS_L0} "
        f"params={v6_l0_params/1e6:.1f}M [W_in=copy(Pythia L0)+noise, b_in=copy(Pythia L0)]\n"
        f"  v105.8 PER_NUMBER_READOUT: enabled={V105_8_PER_NUMBER_READOUT} "
        f"n_bins={V105_8_N_NUMBER_BINS if V105_8_PER_NUMBER_READOUT else 'N/A'} "
        f"params={v8_number_params/1e6:.1f}M [random orthonormal × 0.1]\n"
        f"  v105.11 NUMBER_MSE: enabled={V105_11_NUMBER_MSE} beta={V105_11_NUMBER_MSE_BETA} "
        f"concat_cond={V105_11_CONCAT_COND} cond_dropout={V105_11_COND_DROPOUT}\n"
        f"  v105.11 concat_W: enabled={V105_11_CONCAT_COND} "
        f"params={v11_concat_params/1e6:.1f}M [block-stacked identity init → additive at step 0]\n"
        f"  v105.12: prefill_isolate={V105_12_PREFILL_ISOLATE} "
        f"fourier_decode_init={V105_12_FOURIER_DECODE_INIT} "
        f"codebook_anneal={V105_12_CODEBOOK_ANNEAL}\n"
        f"  delta_gate_quant=ZEROS, ib_init={ib_init}\n"
        f"  T={T} (N_MAX*N_DIGITS+F_MAX={n_max}*{n_digits}+{f_max}), K_max={k_max}",
        flush=True,
    )

    # v105.13 wave-guide mask: dims [H-PRESERVE_DIMS:H] receive no LoRA correction
    # from the waist. Mask is a fixed (non-trainable) tensor on the model.
    if V105_13_WAVEGUIDE_PRESERVE_DIMS > 0:
        preserve = int(V105_13_WAVEGUIDE_PRESERVE_DIMS)
        if preserve >= hidden:
            raise ValueError(
                f"V105_13_WAVEGUIDE_PRESERVE_DIMS={preserve} >= hidden={hidden}; "
                "preserve channel must be strictly smaller than residual"
            )
        commit_count = hidden - preserve
        mask_np = np.concatenate([
            np.ones(commit_count, dtype=np.float32),
            np.zeros(preserve, dtype=np.float32),
        ]).reshape(1, 1, hidden)
        model.fg_v105_5_waveguide_mask = Tensor(
            mask_np, dtype=dtypes.float
        ).contiguous().realize()
        print(
            f"  v105.13 WAVE-GUIDE: enabled — commit_dims=[0,{commit_count}) "
            f"(LoRA-corrected), preserve_dims=[{commit_count},{hidden}) (skip)",
            flush=True,
        )


def fg_v105_5_parameters(model: Any) -> list[Tensor]:
    """Trainable v105.5 factor-graph-specific params.

    Deliberately excludes frozen tables:
      - digit_rope_cos, digit_rope_sin  (precomputed)
      - leaf_to_family, leaf_to_family_oh  (frozen mapping)

    Includes all learnable params across the v105.4 stack + the v105.5
    per-position FFN block (PPFFN).
    """
    params = [
        # Components 1+2
        model.fg_v105_5_digit_codebook,     # (N_DIGITS, 10, H) per-position
        model.fg_v105_5_var_pos_embed,
        model.fg_v105_5_factor_pos_embed,
        model.fg_v105_5_node_kind_embed,
        model.fg_v105_5_calib_head_w,
        model.fg_v105_5_calib_head_b,
        model.fg_v105_5_breath_embed,
        model.fg_v105_5_delta_gate,
        # Component 3: projection waist
        model.fg_v105_5_W_compress,
        model.fg_v105_5_b_compress,
        model.fg_v105_5_W_expand,
        model.fg_v105_5_b_expand,
        # Component 4 (v105.5): hierarchical IB codebook
        model.fg_v105_5_ib_codebook,
        model.fg_v105_5_family_centroids,
        model.fg_v105_5_delta_gate_quant,
        model.fg_v105_5_ib_temperature,
        # v105.4 additions: magnitude head + centroids
        model.fg_v105_5_magnitude_head_w,
        model.fg_v105_5_magnitude_head_b,
        model.fg_v105_5_magnitude_centroids,
    ]
    # NEW v105.5 — per-position FFN block (PPFFN)
    if hasattr(model, "fg_v105_5_ppffn_W_in"):
        params += [
            model.fg_v105_5_ppffn_ln_g,
            model.fg_v105_5_ppffn_ln_b,
            model.fg_v105_5_ppffn_W_in,
            model.fg_v105_5_ppffn_b_in,
            model.fg_v105_5_ppffn_W_out,
            model.fg_v105_5_ppffn_b_out,
        ]
    # NEW v105.6 — per-position W_in at L0
    if hasattr(model, "fg_v105_6_l0_perpos_w_in"):
        params += [
            model.fg_v105_6_l0_perpos_w_in,
            model.fg_v105_6_l0_perpos_b_in,
        ]
    # NEW v105.8 — per-NUMBER codebook
    if hasattr(model, "fg_v105_8_number_codebook"):
        params += [model.fg_v105_8_number_codebook]
    # NEW v105.11 — concat-cond projection (Mechanism 2)
    if hasattr(model, "fg_v105_11_concat_W"):
        params += [model.fg_v105_11_concat_W]
    return params


def fg_v105_5_state_dict(model: Any) -> dict[str, Tensor]:
    """Full state dict (frozen tables included for checkpoint self-containment)."""
    sd = {
        # Components 1+2
        "fg_v105_5.digit_codebook":   model.fg_v105_5_digit_codebook,  # (N_DIGITS, 10, H)
        "fg_v105_5.digit_rope_cos":   model.fg_v105_5_digit_rope_cos,   # frozen
        "fg_v105_5.digit_rope_sin":   model.fg_v105_5_digit_rope_sin,   # frozen
        "fg_v105_5.var_pos_embed":    model.fg_v105_5_var_pos_embed,
        "fg_v105_5.factor_pos_embed": model.fg_v105_5_factor_pos_embed,
        "fg_v105_5.node_kind_embed":  model.fg_v105_5_node_kind_embed,
        "fg_v105_5.calib_head_w":     model.fg_v105_5_calib_head_w,
        "fg_v105_5.calib_head_b":     model.fg_v105_5_calib_head_b,
        "fg_v105_5.breath_embed":     model.fg_v105_5_breath_embed,
        "fg_v105_5.delta_gate":       model.fg_v105_5_delta_gate,
        # Component 3
        "fg_v105_5.W_compress":       model.fg_v105_5_W_compress,
        "fg_v105_5.b_compress":       model.fg_v105_5_b_compress,
        "fg_v105_5.W_expand":         model.fg_v105_5_W_expand,
        "fg_v105_5.b_expand":         model.fg_v105_5_b_expand,
        # Component 4 (v105.5): hierarchical IB
        "fg_v105_5.ib_codebook":          model.fg_v105_5_ib_codebook,
        "fg_v105_5.family_centroids":     model.fg_v105_5_family_centroids,
        "fg_v105_5.leaf_to_family":       model.fg_v105_5_leaf_to_family,       # frozen
        "fg_v105_5.leaf_to_family_oh":    model.fg_v105_5_leaf_to_family_oh,    # frozen
        "fg_v105_5.delta_gate_quant":     model.fg_v105_5_delta_gate_quant,
        "fg_v105_5.ib_temperature":       model.fg_v105_5_ib_temperature,
        # v105.4 additions: magnitude head + centroids
        "fg_v105_5.magnitude_head_w":     model.fg_v105_5_magnitude_head_w,
        "fg_v105_5.magnitude_head_b":     model.fg_v105_5_magnitude_head_b,
        "fg_v105_5.magnitude_centroids":  model.fg_v105_5_magnitude_centroids,
    }
    # NEW v105.5 — per-position FFN block (PPFFN)
    if hasattr(model, "fg_v105_5_ppffn_W_in"):
        sd["fg_v105_5.ppffn_ln_g"]  = model.fg_v105_5_ppffn_ln_g
        sd["fg_v105_5.ppffn_ln_b"]  = model.fg_v105_5_ppffn_ln_b
        sd["fg_v105_5.ppffn_W_in"]  = model.fg_v105_5_ppffn_W_in
        sd["fg_v105_5.ppffn_b_in"]  = model.fg_v105_5_ppffn_b_in
        sd["fg_v105_5.ppffn_W_out"] = model.fg_v105_5_ppffn_W_out
        sd["fg_v105_5.ppffn_b_out"] = model.fg_v105_5_ppffn_b_out
    # NEW v105.6 — per-position W_in at L0
    if hasattr(model, "fg_v105_6_l0_perpos_w_in"):
        sd["fg_v105_6.l0_perpos_w_in"] = model.fg_v105_6_l0_perpos_w_in
        sd["fg_v105_6.l0_perpos_b_in"] = model.fg_v105_6_l0_perpos_b_in
    # NEW v105.8 — per-NUMBER codebook
    if hasattr(model, "fg_v105_8_number_codebook"):
        sd["fg_v105_8.number_codebook"] = model.fg_v105_8_number_codebook
    # NEW v105.11 — concat-cond projection (Mechanism 2)
    if hasattr(model, "fg_v105_11_concat_W"):
        sd["fg_v105_11.concat_W"] = model.fg_v105_11_concat_W
    return sd


# ---------------------------------------------------------------------------
# Warm-start from v104 backbone checkpoint (or v105.1.2)
# ---------------------------------------------------------------------------

def load_ckpt_v105_5(model: Any, path: str) -> None:
    """Load backbone (shared.*, phase*.*, ln_f.*) from any v104-family checkpoint.

    v105.3-specific params (digit_codebook, waist, ib_codebook, RoPE tables,
    breath_embed, etc.) are NOT loaded — they are fresh-initialized so the warm-start
    preserves LoRA/zero-gate guarantees.
    """
    from tinygrad.nn.state import safe_load

    sd = safe_load(path)

    backbone_keys = [
        ("ln_f.g", model.ln_f_g),
        ("ln_f.b", model.ln_f_b),
    ]
    sw = model.block.shared
    for a in ("wv", "bv", "wo", "bo", "w_out", "b_out",
              "in_ln_g", "in_ln_b", "post_ln_g", "post_ln_b"):
        backbone_keys.append((f"shared.{a}", getattr(sw, a)))
    for i, layer in enumerate(model.block.layers):
        for a in ("wq", "bq", "wk", "bk", "w_in", "b_in"):
            backbone_keys.append((f"phase{i}.{a}", getattr(layer, a)))

    # Also copy v104's IB codebook if present in the checkpoint
    if "fg_v104.codebook" in sd and hasattr(model, "fg_v105_5_ib_codebook"):
        src = sd["fg_v104.codebook"]
        dst = model.fg_v105_5_ib_codebook
        if src.shape == dst.shape:
            dst.assign(src.to(dst.device).cast(dst.dtype)).realize()
            print(f"  copied fg_v104.codebook → fg_v105_5.ib_codebook", flush=True)

    loaded  = []
    missing = []
    for name, dst in backbone_keys:
        if name not in sd:
            missing.append(name)
            continue
        src = sd[name].to(dst.device).realize()
        if src.shape != dst.shape:
            try:
                src = src.reshape(dst.shape)
            except Exception:
                missing.append(f"{name}(shape mismatch)")
                continue
        if src.dtype != dst.dtype:
            src = src.cast(dst.dtype)
        dst.assign(src).realize()
        loaded.append(name)

    print(
        f"  loaded {len(loaded)}/{len(backbone_keys)} backbone keys "
        f"from {os.path.basename(path)}",
        flush=True,
    )
    if missing:
        print(f"  missing {len(missing)} keys: {missing[:5]}", flush=True)

    # v105.6 — load per-position L0 W_in/b_in if present in checkpoint and
    # the model has the attribute attached. (Allows resuming a v105.6 run.)
    v6_keys = [
        ("fg_v105_6.l0_perpos_w_in", "fg_v105_6_l0_perpos_w_in"),
        ("fg_v105_6.l0_perpos_b_in", "fg_v105_6_l0_perpos_b_in"),
    ]
    v6_loaded = 0
    for sd_key, attr_name in v6_keys:
        if sd_key in sd and hasattr(model, attr_name):
            dst = getattr(model, attr_name)
            src = sd[sd_key].to(dst.device).realize()
            if src.shape == dst.shape:
                if src.dtype != dst.dtype:
                    src = src.cast(dst.dtype)
                dst.assign(src).realize()
                v6_loaded += 1
    if v6_loaded:
        print(f"  loaded {v6_loaded}/{len(v6_keys)} v105.6 perpos-L0 keys", flush=True)

    # v105.8 — load per-NUMBER codebook from any source matching shape. Order:
    #   1. fg_v105_8.number_codebook (resuming a v105.8 run)
    #   2. fg_v107.domain_codebook    (v107 200-bin codebook — exact bin scheme match)
    #   3. fg_v100.domain_codebook    (v104's 100-bin codebook — only if n_bins=100)
    # If none match, keeps the freshly-initialised random orthonormal codebook.
    if hasattr(model, "fg_v105_8_number_codebook"):
        dst = model.fg_v105_8_number_codebook
        v8_warm_src = None
        v8_warm_label = None
        for sd_key, label in [
            ("fg_v105_8.number_codebook", "v105.8 resume"),
            ("fg_v107.domain_codebook",    "v107 domain_codebook"),
            ("fg_v100.domain_codebook",    "v100/v104 domain_codebook"),
        ]:
            if sd_key in sd and sd[sd_key].shape == dst.shape:
                v8_warm_src = sd[sd_key]
                v8_warm_label = label
                break
        if v8_warm_src is not None:
            src = v8_warm_src.to(dst.device).realize()
            if src.dtype != dst.dtype:
                src = src.cast(dst.dtype)
            dst.assign(src).realize()
            print(
                f"  warm-started fg_v105_8.number_codebook from {v8_warm_label} "
                f"(shape {tuple(dst.shape)})",
                flush=True,
            )

    # v105.11 — load concat-cond projection if present in checkpoint AND the
    # model has the attribute attached (i.e. V105_11_CONCAT_COND=1 at boot).
    if hasattr(model, "fg_v105_11_concat_W"):
        sd_key = "fg_v105_11.concat_W"
        if sd_key in sd:
            dst = model.fg_v105_11_concat_W
            src = sd[sd_key].to(dst.device).realize()
            if src.shape == dst.shape:
                if src.dtype != dst.dtype:
                    src = src.cast(dst.dtype)
                dst.assign(src).realize()
                print(f"  loaded {sd_key} (shape {tuple(dst.shape)})", flush=True)


# ---------------------------------------------------------------------------
# JIT training step
# ---------------------------------------------------------------------------

_JIT_V105_5_CACHE: dict = {}


def _compile_jit_fg_step_v105_5(
    model: Any,
    opt: Any,
    K: int,
    B: int,
    factor_aux_weight: float = V105_5_FACTOR_AUX_WEIGHT,
    calib_weight: float = V105_5_CALIB_WEIGHT,
    energy_weight: float = V105_5_ENERGY_WEIGHT,
    magnitude_weight: float = V105_5_MAGNITUDE_WEIGHT,
    aux_distinct_weight: float = V105_AUX_DISTINCT_WEIGHT,
    var_loss_weight: float = V105_5_VAR_LOSS_WEIGHT,
    n_max: int = V105_5_N_MAX,
    f_max: int = V105_5_F_MAX,
    n_digits: int = V105_5_N_DIGITS,
    n_magnitude: int = V105_5_N_MAGNITUDE,
    grad_clip: float = 1.0,
):
    """Compile a TinyJit'd train step for v105.5.

    Loss structure (additions marked NEW):
      var_loss         — per-breath weighted CE on unobserved variable digits
                         (scaled by var_loss_weight; force 0 for v105.8)
      magnitude_loss   — NEW: per-breath weighted CE on 4-way magnitude class
      factor_aux_loss  — per-NUMBER relative MSE on reconstructed value
                         multiplied by SOFT-magnitude-derived valid mask
      calib_loss       — MSE calibration head
      energy_loss      — expected-value constraint energy (LSD place values)
      aux_distinct     — DIAGNOSTIC: mean off-diagonal cos_sim of digit-position
                         hidden states within each unobserved cell (the terminal
                         var_tokens_with_mag tensor). Pressures positions apart.
                         Off by default (weight=0); the step is byte-identical
                         to the baseline when weight==0.
      number_ce_loss   — NEW v105.8: per-NUMBER CE on the final-breath
                         number_logits (cell_hidden mean @ number_codebook.T).
                         Computed only when V105_8_PER_NUMBER_READOUT=1.
                         When v105.8 enabled, var_loss/magnitude/energy
                         should be Python-side zeroed and per-NUMBER CE
                         replaces them as the variable supervision.
      var_loss_pooled  — NEW v105.9: per-digit CE on the final-breath
                         pooled-cell AR digit logits
                         (digit_logits_pooled_final). Computed only when
                         V105_9_AR_DIGIT_DECODER=1. Mask is per-cell ×
                         per-digit-position (digit_valid_mask) — same as
                         standard per-digit CE but logits come from pooled
                         hidden state. When v105.9 enabled, the standard
                         var_loss is set to 0 by the training driver
                         (var_loss_weight=0); var_loss_pooled is the variable
                         supervision and contributes unconditionally to
                         total_ce with weight 1.0.
    """
    n_code = int(model.fg_v105_5_ib_codebook.shape[0])
    v8_enabled = bool(V105_8_PER_NUMBER_READOUT) and hasattr(
        model, "fg_v105_8_number_codebook"
    )
    n_number_bins = (
        int(model.fg_v105_8_number_codebook.shape[0]) if v8_enabled else 0
    )
    v9_enabled = bool(V105_9_AR_DIGIT_DECODER)
    # v105.10 dual readout: when enabled, the pooled-AR digit CE is weighted
    # by V105_10_DIGIT_WEIGHT (instead of the default 1.0 used by pure v105.9).
    v10_enabled = bool(V105_10_DUAL_READOUT)
    digit_weight_for_pooled = float(V105_10_DIGIT_WEIGHT) if v10_enabled else 1.0
    # v105.11: log-number-MSE through the AR-reconstructed value. Requires
    # the pooled-AR digit decoder (v105.9). The mutex logic at module load
    # already enforced V105_9=1 when V105_11=1, but we double-check here.
    v11_enabled = bool(V105_11_NUMBER_MSE) and v9_enabled
    v11_beta    = float(V105_11_NUMBER_MSE_BETA) if v11_enabled else 0.0
    v11_ln      = bool(V105_11_LN_COND)
    # v105.12 codebook annealing: when enabled, the codebook weight (multiplier
    # on number_ce_loss) is a RUNTIME Tensor input to the JIT step, not a
    # compiled constant. This lets the training driver pass per-step weights
    # from the anneal schedule without triggering JIT recompilation. Cache key
    # uses bool(v105_12_anneal) only — the actual float weight is dynamic.
    v105_12_anneal = bool(V105_12_CODEBOOK_ANNEAL)
    v105_12_prefill_iso = bool(V105_12_PREFILL_ISOLATE)
    v105_12_fourier_init = bool(V105_12_FOURIER_DECODE_INIT)
    key = ("v105_5", id(model), id(opt), int(K), int(B),
           float(factor_aux_weight), float(calib_weight), float(energy_weight),
           float(magnitude_weight), float(aux_distinct_weight),
           float(var_loss_weight),
           int(n_max), int(f_max), int(n_digits), int(n_magnitude),
           float(grad_clip), int(n_code), bool(v8_enabled), int(n_number_bins),
           bool(v9_enabled), bool(v10_enabled), float(digit_weight_for_pooled),
           bool(v11_enabled), float(v11_beta), bool(v11_ln),
           bool(v105_12_prefill_iso), bool(v105_12_fourier_init),
           bool(v105_12_anneal))
    if key in _JIT_V105_5_CACHE:
        return _JIT_V105_5_CACHE[key]

    aw     = float(calib_weight)
    fw     = float(factor_aux_weight)
    ew     = float(energy_weight)
    mw     = float(magnitude_weight)
    adw    = float(aux_distinct_weight)
    vlw    = float(var_loss_weight)
    gc     = float(grad_clip)
    params = opt.params

    print(
        f"[JIT] compile v105.5 fg step: K={K} B={B} n_digits={n_digits} "
        f"T={n_max * n_digits + f_max} aw={aw} fw={fw} ew={ew} mw={mw} "
        f"adw={adw} vlw={vlw} gc={gc} n_code={n_code} "
        f"v105.8={v8_enabled} n_number_bins={n_number_bins} "
        f"v105.9={v9_enabled} v105.10={v10_enabled} "
        f"digit_weight_for_pooled={digit_weight_for_pooled} "
        f"v105.11={v11_enabled} v11_beta={v11_beta}...",
        flush=True,
    )

    # Build the (4, n_digits) class_to_valid mapping for the soft mask:
    #   class 0 (1-digit):  [1, 0, 0, 0, 0]
    #   class 1 (2-digit):  [1, 1, 0, 0, 0]
    #   class 2 (3-digit):  [1, 1, 1, 0, 0]
    #   class 3 (4-digit):  [1, 1, 1, 1, 0]
    c2v_np = np.zeros((n_magnitude, n_digits), dtype=np.float32)
    for cls in range(n_magnitude):
        n_valid_for_cls = min(cls + 1, n_digits)
        c2v_np[cls, :n_valid_for_cls] = 1.0
    class_to_valid_t = Tensor(c2v_np, dtype=dtypes.float).contiguous()

    # Pre-compute aux distinctness off-diagonal mask as a constant tensor
    # OUTSIDE the JIT step (avoids per-step Tensor(np_array) creation
    # which was causing ~10x slowdown). Only used when adw > 0.
    _aux_off_mask = None
    if adw > 0:
        _aux_off_mask_np = (1.0 - np.eye(n_digits)).astype(np.float32)
        _aux_off_mask    = Tensor(_aux_off_mask_np, dtype=dtypes.float).reshape(
            1, 1, n_digits, n_digits
        ).contiguous()

    # v105.11 closure variables — pre-compute place_values and digit_values
    # OUTSIDE the JIT step (same pattern as _aux_off_mask). These are used
    # to reconstruct the soft expected number from per-position digit probs.
    # LSD-first place values [10^0, 10^1, ..., 10^(n_digits-1)].
    _v11_place_values = None
    _v11_digit_values = None
    if v11_enabled:
        _pv_np = np.array(
            [10.0 ** p for p in range(n_digits)], dtype=np.float32
        )
        _v11_place_values = Tensor(_pv_np, dtype=dtypes.float).reshape(
            1, 1, n_digits
        ).contiguous()
        _dv_np = np.arange(10, dtype=np.float32)
        _v11_digit_values = Tensor(_dv_np, dtype=dtypes.float).reshape(
            1, 1, 1, 10
        ).contiguous()

    @TinyJit
    def _step(
        digit_init: Tensor,
        node_kinds: Tensor,
        staging_mask: Tensor,
        head_op_mask: Tensor,
        gold_digits: Tensor,
        observed_mask: Tensor,
        factor_gold_dg: Tensor,
        factor_valid: Tensor,
        factor_types: Tensor,
        factor_args: Tensor,
        digit_valid_mask: Tensor,           # (B, N_MAX, N_DIGITS) float
        factor_digit_valid_mask: Tensor,    # (B, F_MAX, N_DIGITS) float
        magnitude_target: Tensor,           # (B, N_MAX) int  — gold magnitude class
        number_bin_target: Tensor,          # (B, N_MAX) int  — v105.8 per-NUMBER target
        codebook_weight: Tensor,            # (1,) float — v105.12 anneal multiplier on number_ce
    ):
        opt.zero_grad()

        (digit_logits_history, factor_logits_history,
         calib_history, magnitude_logits_history,
         terminal_var_hidden, number_logits_final,
         digit_logits_pooled_final) = \
            fg_breathing_forward_v105_5(
                model, digit_init, node_kinds, staging_mask, head_op_mask,
                K=K, n_max=n_max, f_max=f_max, n_digits=n_digits,
            )

        # --- Main loss on unobserved variables ---
        # LSD layout: valid positions are the LEADING n_actual_digits per variable.
        # The trailing positions are leading-zero padding above the most-significant
        # digit and must NOT contribute to the loss.
        unobs_float   = (1 - observed_mask.cast(dtypes.float))         # (B, N_MAX)
        unobs_dg      = unobs_float.reshape(B, n_max, 1).expand(B, n_max, n_digits)
        combined_mask = unobs_dg * digit_valid_mask                     # (B, N_MAX, N_DIGITS)
        n_active      = combined_mask.sum() + 1e-8

        gold_flat   = gold_digits.cast(dtypes.int).reshape(B * n_max * n_digits)
        cmask_flat  = combined_mask.reshape(B * n_max * n_digits)

        var_loss_sum   = Tensor.zeros((), dtype=dtypes.float).contiguous()
        var_weight_sum = 0.0
        per_breath_ce_t: list[Tensor] = []

        # LSD place values for number reconstruction: [10^0, 10^1, ..., 10^(N-1)]
        _place_values_np_var = [float(10 ** i) for i in range(n_digits)]
        _place_values_t_var  = Tensor(_place_values_np_var, dtype=dtypes.float).reshape(1, 1, n_digits)
        _digit_vals_t_var    = Tensor([float(i) for i in range(10)], dtype=dtypes.float)

        if V105_5_NUMBER_MSE_ONLY:
            # ============================================================
            # NUMBER-ONLY LOSS PATH (V105_5_NUMBER_MSE_ONLY=1)
            # ============================================================
            _var_gold_float    = gold_digits.cast(dtypes.float)         # (B, N_MAX, N_DIGITS)
            _var_gold_masked   = _var_gold_float * digit_valid_mask
            var_gold_numbers   = (_var_gold_masked * _place_values_t_var).sum(axis=-1)  # (B, N_MAX)
            var_rel_denom      = var_gold_numbers.abs() + 1.0                            # (B, N_MAX)

            is_real_var_loss   = (digit_valid_mask.sum(axis=-1) > 0).cast(dtypes.float)  # (B, N_MAX)
            unobs_real_var     = unobs_float * is_real_var_loss                          # (B, N_MAX)
            n_unobs_real_var   = unobs_real_var.sum() + 1e-8

            for k, dig_logits in enumerate(digit_logits_history):
                weight_k    = 1.0 + float(k) / float(max(K - 1, 1))
                probs_k     = dig_logits.softmax(axis=-1)               # (B, N_MAX, N_DIGITS, 10)
                exp_digit_k = (probs_k * _digit_vals_t_var.reshape(1, 1, 1, 10)).sum(axis=-1)
                exp_digit_m = exp_digit_k * digit_valid_mask             # mask invalid positions
                pred_number = (exp_digit_m * _place_values_t_var).sum(axis=-1)            # (B, N_MAX)
                rel_err  = ((pred_number - var_gold_numbers) / var_rel_denom).clip(-5.0, 5.0)
                sq_err   = rel_err * rel_err
                sq_err_m = sq_err * unobs_real_var.cast(sq_err.dtype)
                ce_k     = sq_err_m.sum() / n_unobs_real_var

                per_breath_ce_t.append(ce_k)
                var_loss_sum   = var_loss_sum + ce_k * weight_k
                var_weight_sum += weight_k
        else:
            # ============================================================
            # PER-DIGIT CE PATH (default behavior)
            # ============================================================
            for k, dig_logits in enumerate(digit_logits_history):
                weight_k    = 1.0 + float(k) / float(max(K - 1, 1))
                logits_flat = dig_logits.reshape(B * n_max * n_digits, 10)
                log_probs   = logits_flat.log_softmax(axis=-1)
                gold_oh     = gold_flat.one_hot(10).cast(log_probs.dtype)
                nll         = -(log_probs * gold_oh).sum(axis=-1)
                masked_nll  = nll * cmask_flat.cast(nll.dtype)
                ce_k        = masked_nll.sum() / n_active
                per_breath_ce_t.append(ce_k)
                var_loss_sum   = var_loss_sum + ce_k * weight_k
                var_weight_sum += weight_k

        var_loss = var_loss_sum / float(var_weight_sum)

        # --- Magnitude loss (NEW v105.5) — per-breath weighted 4-way CE ---
        is_real_var = (digit_valid_mask.sum(axis=-1) > 0).cast(dtypes.float)    # (B,N)
        unobs_real  = unobs_float * is_real_var                                  # (B,N)
        n_unobs_real = unobs_real.sum() + 1e-8

        mag_target_int = magnitude_target.cast(dtypes.int).clip(0, n_magnitude - 1)
        mag_target_flat = mag_target_int.reshape(B * n_max)

        mag_loss_sum   = Tensor.zeros((), dtype=dtypes.float).contiguous()
        mag_weight_sum = 0.0
        for k, mag_logits_k in enumerate(magnitude_logits_history):
            weight_k    = 1.0 + float(k) / float(max(K - 1, 1))
            mlogits_flat = mag_logits_k.reshape(B * n_max, n_magnitude)
            log_probs    = mlogits_flat.log_softmax(axis=-1)
            tgt_oh       = mag_target_flat.one_hot(n_magnitude).cast(log_probs.dtype)
            nll          = -(log_probs * tgt_oh).sum(axis=-1)            # (B*N,)
            unobs_flat   = unobs_real.reshape(B * n_max)
            masked_nll   = nll * unobs_flat.cast(nll.dtype)
            ce_k         = masked_nll.sum() / n_unobs_real
            mag_loss_sum = mag_loss_sum + ce_k * weight_k
            mag_weight_sum += weight_k
        magnitude_loss = mag_loss_sum / float(mag_weight_sum)

        # --- Per-NUMBER factor auxiliary loss (relative MSE) ---
        # LSD-first place values: index 0 = 10^0, ..., index N-1 = 10^(N-1).
        # v105.5 addition 4: multiply factor_digit_valid_mask by a soft mask
        # derived from the FINAL-breath magnitude prediction over the RESULT
        # variable of each factor. The resulting mask is per-factor.
        n_valid_fac  = factor_valid.sum() + 1e-8

        # LSD place values: [10^0, 10^1, ..., 10^(N-1)]
        place_values_np = [float(10 ** i) for i in range(n_digits)]
        place_values_t  = Tensor(place_values_np, dtype=dtypes.float).reshape(1, 1, n_digits)

        digit_vals_t    = Tensor([float(i) for i in range(10)], dtype=dtypes.float)

        gold_dg_float   = factor_gold_dg.cast(dtypes.float)                       # (B, F, D)
        gold_dg_masked  = gold_dg_float * factor_digit_valid_mask                  # zero invalid
        gold_numbers    = (gold_dg_masked * place_values_t).sum(axis=-1)          # (B, F)
        rel_denom       = gold_numbers.abs() + 1.0                                 # (B, F)

        # Build per-factor SOFT valid mask from final-breath magnitude over the
        # result variable.  factor_args[:, :, 2] is the result idx; gather the
        # magnitude_softmax of the corresponding variable, then mix with class_to_valid.
        final_mag_logits = magnitude_logits_history[-1]                           # (B, N_MAX, N_MAG)
        final_mag_probs  = final_mag_logits.softmax(axis=-1)                       # (B, N_MAX, N_MAG)
        res_idx          = factor_args[:, :, 2].cast(dtypes.int).clip(0, n_max - 1) # (B, F_MAX)
        # Gather via one-hot: (B, F_MAX, N_MAX) @ (B, N_MAX, N_MAG) → (B, F_MAX, N_MAG)
        res_oh = res_idx.one_hot(n_max).cast(dtypes.float)                        # (B, F_MAX, N_MAX)
        fac_mag_probs = res_oh @ final_mag_probs                                   # (B, F_MAX, N_MAG)
        # class_to_valid: (N_MAG, N_DIGITS)
        fac_soft_valid = fac_mag_probs @ class_to_valid_t.cast(fac_mag_probs.dtype) # (B, F_MAX, N_DIGITS)
        # Combine soft + gold mask (multiplicative) — gradient on magnitude head
        # comes through fac_soft_valid; gold mask still enforces ground truth.
        fac_combined_mask = factor_digit_valid_mask * fac_soft_valid               # (B, F_MAX, N_DIGITS)

        factor_aux_sum   = Tensor.zeros((), dtype=dtypes.float).contiguous()
        factor_aux_w_sum = 0.0

        for k_aux, fac_logits_k in enumerate(factor_logits_history):
            w_k_aux = 1.0 + float(k_aux) / float(max(K - 1, 1))

            fac_probs   = fac_logits_k.softmax(axis=-1)                            # (B,F,D,10)
            exp_digit   = (fac_probs * digit_vals_t.reshape(1, 1, 1, 10)).sum(axis=-1)
            exp_digit_m = exp_digit * fac_combined_mask                            # SOFT-masked
            pred_number = (exp_digit_m * place_values_t).sum(axis=-1)

            rel_err  = ((pred_number - gold_numbers) / rel_denom).clip(-5.0, 5.0)
            sq_err   = rel_err * rel_err
            sq_err_m = sq_err * factor_valid.cast(sq_err.dtype)
            fac_ce_k = sq_err_m.sum() / n_valid_fac

            factor_aux_sum   = factor_aux_sum + fac_ce_k * w_k_aux
            factor_aux_w_sum += w_k_aux

        factor_aux_loss = factor_aux_sum / float(factor_aux_w_sum)

        # --- Constraint energy (LSD place values) ---
        final_dig_logits = digit_logits_history[-1]
        energy_loss = constraint_energy_v105_5(
            final_dig_logits, factor_types, factor_args,
            n_max=n_max, f_max=f_max, n_digits=n_digits,
        )

        # --- Calibration ---
        final_pred_dg = digit_logits_history[-1].argmax(axis=-1).detach()           # (B,N,D)
        dg_eq    = (final_pred_dg == gold_digits.cast(dtypes.int)).cast(dtypes.float)
        dg_match_or_invalid = dg_eq * digit_valid_mask + (1.0 - digit_valid_mask)
        var_eq   = dg_match_or_invalid.min(axis=-1)                                 # (B,N)
        eq_unobs    = var_eq * unobs_real
        n_unobs_per = unobs_real.sum(axis=-1) + 1e-8
        correct     = eq_unobs.sum(axis=-1) / n_unobs_per

        calib_loss_sum = Tensor.zeros((), dtype=dtypes.float).contiguous()
        for k, calib_k in enumerate(calib_history):
            prog       = float(k) / float(max(K - 1, 1))
            target_k   = 0.5 + (correct - 0.5) * prog
            calib_loss_sum = calib_loss_sum + ((calib_k - target_k) ** 2).mean()
        calib_loss = calib_loss_sum / float(K)

        # --- Magnitude accuracy diagnostic ---
        final_mag_pred = magnitude_logits_history[-1].argmax(axis=-1).detach()      # (B, N_MAX)
        mag_eq         = (final_mag_pred == magnitude_target.cast(dtypes.int)).cast(dtypes.float)
        mag_eq_unobs   = mag_eq * unobs_real
        mag_acc        = (mag_eq_unobs.sum() / n_unobs_real).detach()

        # --- Metrics ---
        cell_acc  = (eq_unobs.sum() / (unobs_real.sum() + 1e-8)).detach()
        query_acc = correct.mean().detach()

        # --- Aux per-position distinctness loss (DIAGNOSTIC, gated by adw) ---
        # Tests whether the model CAN find low-cos_sim hidden states when
        # explicitly pressured to. Computed on `terminal_var_hidden` (the
        # tensor the digit codebook reads from) at the LAST breath only.
        # Computation: pairwise off-diagonal cosine similarity across the
        # n_digits positions of each cell; averaged over unobserved+valid
        # cells. Minimizing drives positions apart (lower cos = more
        # distinct).
        if adw > 0:
            # Cos_sim formulation (bounded [0, 1]). Reverted from L2 distance
            # because L2 is unbounded — model gamed it by inflating hidden
            # state magnitudes (distinct=-14422 in v5 smoke). Cos_sim is the
            # right formulation; the ~10s/step overhead is acceptable.
            hidden_f = terminal_var_hidden.cast(dtypes.float)              # (B, n_max, n_digits, H)
            norms    = ((hidden_f * hidden_f).sum(axis=-1, keepdim=True) + 1e-6).sqrt()
            unit     = hidden_f / norms                                    # (B, n_max, n_digits, H)
            gram     = unit @ unit.transpose(-1, -2)                       # (B, n_max, n_digits, n_digits)
            n_off    = float(n_digits * (n_digits - 1))
            cell_distinctness = (gram * _aux_off_mask).sum(axis=(-1, -2)) / n_off  # (B, n_max)
            # Weight by (unobserved AND valid-real) cells — same notion as
            # is_real_var * unobs_float used for var_loss.
            cell_weight = unobs_real                                        # (B, n_max)
            aux_distinct_loss = (cell_distinctness * cell_weight).sum() / (cell_weight.sum() + 1e-8)
        else:
            # Placeholder zero (so the return tuple is fixed). No graph
            # contribution.
            aux_distinct_loss = Tensor.zeros((), dtype=dtypes.float).contiguous()

        # --- v105.8 per-NUMBER CE loss (only when enabled) ---
        # number_logits_final is None when v105.8 disabled or codebook not attached.
        # Always emit a number_ce_loss scalar to keep the JIT return tuple stable;
        # when disabled, it's a constant zero and contributes nothing to grad.
        if v8_enabled and number_logits_final is not None:
            # Per-cell weighting: unobserved AND valid-real (same as factor_aux).
            num_logits_flat = number_logits_final.reshape(B * n_max, n_number_bins)
            num_logprobs    = num_logits_flat.log_softmax(axis=-1)
            num_gold_flat   = number_bin_target.cast(dtypes.int).clip(
                0, n_number_bins - 1
            ).reshape(B * n_max)
            num_gold_oh     = num_gold_flat.one_hot(n_number_bins).cast(num_logprobs.dtype)
            num_nll         = -(num_logprobs * num_gold_oh).sum(axis=-1)        # (B*N,)
            unobs_real_flat = unobs_real.reshape(B * n_max).cast(num_nll.dtype)
            num_masked_nll  = num_nll * unobs_real_flat
            number_ce_loss  = num_masked_nll.sum() / n_unobs_real

            # Per-NUMBER accuracy diagnostic.
            num_pred       = number_logits_final.argmax(axis=-1).detach()        # (B, N_MAX)
            num_eq         = (num_pred == number_bin_target.cast(dtypes.int)
                              ).cast(dtypes.float)
            num_eq_unobs   = num_eq * unobs_real
            number_acc     = (num_eq_unobs.sum() / n_unobs_real).detach()
        else:
            number_ce_loss = Tensor.zeros((), dtype=dtypes.float).contiguous()
            number_acc     = Tensor.zeros((), dtype=dtypes.float).contiguous()

        # --- v105.9 pooled-AR per-digit CE loss (only when enabled) ---
        # digit_logits_pooled_final is None when v105.9 disabled. Always emit
        # a var_loss_pooled scalar to keep the JIT return tuple stable.
        # Mask: (unobserved AND valid-real) × digit_valid_mask, same notion as
        # the standard per-digit CE — but logits come from one pooled hidden
        # state per cell, not 5 per-position hiddens. The pooled-AR loss is
        # computed at the LAST breath only (weight 1.0); earlier breaths
        # don't have a pooled output, so per-breath ladder is implicit in the
        # K iterative refinements of cell_hidden itself.
        if v9_enabled and digit_logits_pooled_final is not None:
            pool_logits_flat = digit_logits_pooled_final.reshape(
                B * n_max * n_digits, 10
            )
            pool_logprobs    = pool_logits_flat.log_softmax(axis=-1)
            pool_gold_oh     = gold_flat.one_hot(10).cast(pool_logprobs.dtype)
            pool_nll         = -(pool_logprobs * pool_gold_oh).sum(axis=-1)
            pool_masked_nll  = pool_nll * cmask_flat.cast(pool_nll.dtype)
            var_loss_pooled  = pool_masked_nll.sum() / n_active

            # Pooled per-cell accuracy diagnostic.
            pool_pred        = digit_logits_pooled_final.argmax(axis=-1).detach()  # (B,N,D)
            pool_dg_eq       = (pool_pred == gold_digits.cast(dtypes.int)).cast(dtypes.float)
            pool_dg_or_invalid = pool_dg_eq * digit_valid_mask + (1.0 - digit_valid_mask)
            pool_var_eq      = pool_dg_or_invalid.min(axis=-1)                     # (B,N)
            pool_eq_unobs    = pool_var_eq * unobs_real
            pooled_cell_acc  = (pool_eq_unobs.sum() / (unobs_real.sum() + 1e-8)).detach()
        else:
            var_loss_pooled = Tensor.zeros((), dtype=dtypes.float).contiguous()
            pooled_cell_acc = Tensor.zeros((), dtype=dtypes.float).contiguous()

        # --- v105.11 log-number-MSE through the AR-reconstructed value ---
        # Computed only when v11_enabled (V105_11_NUMBER_MSE=1 AND v9_enabled).
        # Always emit a num_mse scalar to keep the JIT return tuple stable.
        # Closure vars: _v11_place_values (1, 1, n_digits), _v11_digit_values
        # (1, 1, 1, 10) — both float, pre-built outside JIT to avoid the
        # per-step Tensor(np.array) creation overhead.
        if v11_enabled and digit_logits_pooled_final is not None:
            # Soft expected digit per position: (B, n_max, n_digits)
            v11_probs    = digit_logits_pooled_final.softmax(axis=-1)
            v11_exp_dig  = (v11_probs * _v11_digit_values).sum(axis=-1)
            # Reconstruct number (LSD-first): (B, n_max)
            v11_rec_N    = (v11_exp_dig * _v11_place_values).sum(axis=-1)
            # Gold number from gold_digits (LSD-first).
            v11_gold_dg  = gold_digits.cast(dtypes.float)                # (B, n_max, n_digits)
            v11_gold_N   = (v11_gold_dg * _v11_place_values).sum(axis=-1)  # (B, n_max)
            # Log-MSE balanced across orders of magnitude (clip negatives so
            # we never feed a negative argument to .log()).
            v11_log_rec  = (1.0 + v11_rec_N.clip(0.0, 1e8)).log()
            v11_log_gold = (1.0 + v11_gold_N.clip(0.0, 1e8)).log()
            v11_sq       = (v11_log_rec - v11_log_gold) ** 2
            # Mask: (unobserved AND valid-real) — same notion as factor_aux
            # and v105.8 number_ce_loss. unobs_real is (B, n_max) float.
            num_mse      = (v11_sq * unobs_real).sum() / (unobs_real.sum() + 1e-8)
        else:
            num_mse = Tensor.zeros((), dtype=dtypes.float).contiguous()

        # --- Total loss ---
        # var_loss is scaled by vlw (set to 0 by training driver for v105.8/v105.9).
        total_ce = (
            vlw * var_loss
            + mw * magnitude_loss
            + fw * factor_aux_loss
            + aw * calib_loss
            + ew * energy_loss
        )
        if adw > 0:
            total_ce = total_ce + adw * aux_distinct_loss
        if v8_enabled:
            # v105.12 codebook anneal: multiply by runtime weight Tensor.
            # When V105_12_CODEBOOK_ANNEAL=0 the training driver passes 1.0
            # so this is a no-op multiply. Reshape to scalar so the broadcast
            # produces a scalar total_ce contribution.
            cw_scalar = codebook_weight.reshape(()).cast(number_ce_loss.dtype)
            total_ce = total_ce + cw_scalar * number_ce_loss
        if v9_enabled:
            # v105.9 default weight: 1.0. v105.10 dual readout: V105_10_DIGIT_WEIGHT
            # (a Python-side scalar baked into the JIT graph via digit_weight_for_pooled).
            total_ce = total_ce + digit_weight_for_pooled * var_loss_pooled
        if v11_enabled:
            # v105.11 log-number-MSE on the AR-reconstructed value. Per-digit
            # CE (var_loss_pooled above) and log-number-MSE share the SAME
            # path through the AR decoder.
            total_ce = total_ce + v11_beta * num_mse
        total_ce.backward()

        healthy = total_ce.isfinite().cast(dtypes.float)
        for p in params:
            if p.grad is not None:
                p.grad = p.grad * healthy.cast(p.grad.dtype)

        if gc > 0:
            sq_sum = Tensor.zeros((), dtype=dtypes.float).contiguous()
            for p in params:
                if p.grad is not None:
                    sq_sum = sq_sum + p.grad.cast(dtypes.float).square().sum()
            grad_norm  = (sq_sum + 1e-12).sqrt()
            clip_coef  = (Tensor(gc, dtype=dtypes.float) / (grad_norm + 1e-6)).minimum(
                Tensor(1.0, dtype=dtypes.float)
            )
            for p in params:
                if p.grad is not None:
                    p.grad = p.grad * clip_coef.cast(p.grad.dtype)

        opt.step()

        return (
            total_ce.realize(),
            healthy.realize(),
            var_loss.realize(),
            factor_aux_loss.realize(),
            calib_loss.realize(),
            energy_loss.realize(),
            magnitude_loss.realize(),
            mag_acc.realize(),
            cell_acc.realize(),
            query_acc.realize(),
            aux_distinct_loss.realize(),
            number_ce_loss.realize(),
            number_acc.realize(),
            var_loss_pooled.realize(),
            pooled_cell_acc.realize(),
            num_mse.realize(),
            *(ce.realize() for ce in per_breath_ce_t),
        )

    _JIT_V105_5_CACHE[key] = _step
    print(
        f"[JIT] v105.5 fg step ready (cache={len(_JIT_V105_5_CACHE)}); "
        f"first call compiles...",
        flush=True,
    )
    return _step


def _compile_jit_fg_eval_v105_5(
    model: Any,
    K: int,
    B: int,
    n_max: int = V105_5_N_MAX,
    f_max: int = V105_5_F_MAX,
    n_digits: int = V105_5_N_DIGITS,
):
    """Compile a TinyJit'd eval step (forward only, no gradient).

    Takes digit_valid_mask so cell_acc treats invalid (leading-zero padding)
    positions as automatically correct (consistent with the train loss).
    """
    key = ("eval_v105_5", id(model), int(K), int(B), int(n_max), int(f_max), int(n_digits))
    if key in _JIT_V105_5_CACHE:
        return _JIT_V105_5_CACHE[key]

    print(f"[JIT] compile v105.5 fg eval: K={K} B={B} n_digits={n_digits}...", flush=True)

    @TinyJit
    def _eval(
        digit_init: Tensor,
        node_kinds: Tensor,
        staging_mask: Tensor,
        head_op_mask: Tensor,
        gold_digits: Tensor,
        observed_mask: Tensor,
        digit_valid_mask: Tensor,
    ):
        (digit_logits_history, _, _, _, _, _, _) = fg_breathing_forward_v105_5(
            model, digit_init, node_kinds, staging_mask, head_op_mask,
            K=K, n_max=n_max, f_max=f_max, n_digits=n_digits,
        )
        final_logits  = digit_logits_history[-1]
        pred_dg       = final_logits.argmax(axis=-1)
        dg_eq         = (pred_dg == gold_digits.cast(pred_dg.dtype)).cast(dtypes.float)
        dg_or_invalid = dg_eq * digit_valid_mask + (1.0 - digit_valid_mask)
        var_eq        = dg_or_invalid.min(axis=-1)
        is_real_var   = (digit_valid_mask.sum(axis=-1) > 0).cast(dtypes.float)
        unobs_real    = (1 - observed_mask.cast(dtypes.float)) * is_real_var
        cell_acc      = (var_eq * unobs_real).sum() / (unobs_real.sum() + 1e-8)
        return pred_dg.realize(), cell_acc.realize()

    _JIT_V105_5_CACHE[key] = _eval
    print(f"[JIT] v105.5 eval ready (cache={len(_JIT_V105_5_CACHE)})", flush=True)
    return _eval


# ---------------------------------------------------------------------------
# v105.8 — per-NUMBER eval JIT
# ---------------------------------------------------------------------------

def _compile_jit_fg_eval_v105_8(
    model: Any,
    K: int,
    B: int,
    n_max: int = V105_5_N_MAX,
    f_max: int = V105_5_F_MAX,
    n_digits: int = V105_5_N_DIGITS,
):
    """v105.8 per-NUMBER eval JIT. Returns:
      pred_bin : (B, N_MAX) int — argmax over number bins.
      cell_acc : float — fraction of unobserved cells whose predicted bin matches
                 gold_bin_target. (No per-digit decode; eval is at the bin level.)
    """
    assert hasattr(model, "fg_v105_8_number_codebook"), \
        "v105.8 number_codebook not attached; attach_fg_params_v105_5 with "\
        "V105_8_PER_NUMBER_READOUT=1 must be called first"

    key = ("eval_v105_8", id(model), int(K), int(B), int(n_max),
           int(f_max), int(n_digits))
    if key in _JIT_V105_5_CACHE:
        return _JIT_V105_5_CACHE[key]

    print(f"[JIT] compile v105.8 fg eval (per-NUMBER): K={K} B={B}...", flush=True)

    @TinyJit
    def _eval(
        digit_init: Tensor,
        node_kinds: Tensor,
        staging_mask: Tensor,
        head_op_mask: Tensor,
        number_bin_target: Tensor,        # (B, N_MAX) int — gold bin index
        observed_mask: Tensor,            # (B, N_MAX) int
        digit_valid_mask: Tensor,         # (B, N_MAX, N_DIGITS) — for unobs_real
    ):
        (_, _, _, _, _, number_logits_final, _) = fg_breathing_forward_v105_5(
            model, digit_init, node_kinds, staging_mask, head_op_mask,
            K=K, n_max=n_max, f_max=f_max, n_digits=n_digits,
        )
        pred_bin    = number_logits_final.argmax(axis=-1)                        # (B, N_MAX)
        bin_eq      = (pred_bin == number_bin_target.cast(pred_bin.dtype)
                       ).cast(dtypes.float)
        is_real_var = (digit_valid_mask.sum(axis=-1) > 0).cast(dtypes.float)
        unobs_real  = (1 - observed_mask.cast(dtypes.float)) * is_real_var
        cell_acc    = (bin_eq * unobs_real).sum() / (unobs_real.sum() + 1e-8)
        return pred_bin.realize(), cell_acc.realize()

    _JIT_V105_5_CACHE[key] = _eval
    print(f"[JIT] v105.8 eval ready (cache={len(_JIT_V105_5_CACHE)})", flush=True)
    return _eval


# ---------------------------------------------------------------------------
# v105.9 — pooled-AR digit-decoder eval JIT
# ---------------------------------------------------------------------------

def _compile_jit_fg_eval_v105_9(
    model: Any,
    K: int,
    B: int,
    n_max: int = V105_5_N_MAX,
    f_max: int = V105_5_F_MAX,
    n_digits: int = V105_5_N_DIGITS,
):
    """v105.9 pooled-AR digit-decoder eval JIT.

    Reads digit_logits_pooled_final (the last element of the forward return)
    and computes per-cell accuracy via the same "all digits match" rule used
    in v105.5 eval — but the logits come from the pooled-cell AR decoder
    rather than per-position hidden states.

    Returns:
      pred_dg  : (B, N_MAX, N_DIGITS) int — argmax over digit values.
      cell_acc : float — fraction of unobserved-and-valid cells whose
        predicted digits all match the gold digits (treating padded
        leading-zero positions as automatically correct via digit_valid_mask).
    """
    key = ("eval_v105_9", id(model), int(K), int(B), int(n_max),
           int(f_max), int(n_digits))
    if key in _JIT_V105_5_CACHE:
        return _JIT_V105_5_CACHE[key]

    print(f"[JIT] compile v105.9 fg eval (pooled-AR digits): K={K} B={B}...", flush=True)

    @TinyJit
    def _eval(
        digit_init: Tensor,
        node_kinds: Tensor,
        staging_mask: Tensor,
        head_op_mask: Tensor,
        gold_digits: Tensor,
        observed_mask: Tensor,
        digit_valid_mask: Tensor,
    ):
        (_, _, _, _, _, _, digit_logits_pooled_final) = fg_breathing_forward_v105_5(
            model, digit_init, node_kinds, staging_mask, head_op_mask,
            K=K, n_max=n_max, f_max=f_max, n_digits=n_digits,
        )
        pred_dg       = digit_logits_pooled_final.argmax(axis=-1)
        dg_eq         = (pred_dg == gold_digits.cast(pred_dg.dtype)).cast(dtypes.float)
        dg_or_invalid = dg_eq * digit_valid_mask + (1.0 - digit_valid_mask)
        var_eq        = dg_or_invalid.min(axis=-1)
        is_real_var   = (digit_valid_mask.sum(axis=-1) > 0).cast(dtypes.float)
        unobs_real    = (1 - observed_mask.cast(dtypes.float)) * is_real_var
        cell_acc      = (var_eq * unobs_real).sum() / (unobs_real.sum() + 1e-8)
        return pred_dg.realize(), cell_acc.realize()

    _JIT_V105_5_CACHE[key] = _eval
    print(f"[JIT] v105.9 eval ready (cache={len(_JIT_V105_5_CACHE)})", flush=True)
    return _eval
