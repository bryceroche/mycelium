"""Mycelium v4 breathing transformer.

Architecture matches Pythia-410M (GPT-NeoX) so weights load cleanly:
  - Standard 2-weight FFN (w_in -> GELU -> w_out), biases everywhere
  - Two LayerNorms per block (input_LN, post_attn_LN), parallel residual
  - Partial RoPE: only first rotary_dim=16 of head_dim=64 dims are rotated

Mycelium v4 modifications:
  - 4 phase-specific layers (RISE/PEAK/FALL/TROUGH) sharing V/O/FFN-out/LNs
  - Phase-specific Q, K, FFN-input projection (and biases)
  - π-cycled RoPE: per-head + per-loop phase offset applied to Q only
  - Sine-wave temperature modulation per phase
  - Gated running integral across loops (controller stub: gate=1)
"""
import math
import os
from typing import List

import numpy as np
from tinygrad import Tensor, dtypes
from tinygrad.nn import Embedding
from tinygrad.engine.jit import TinyJit

from mycelium.config import Config
from mycelium.lookup_table import LookupTable
from mycelium.controller import Controller
from mycelium.calibration import ConfidenceHead


# Ablation flags — read once at module import. Each disables one closed-loop
# component for Phase 2/3 directional screening. Default off: behavior matches
# the un-ablated architecture exactly when none are set.
ABLATE_TEMP        = int(os.environ.get("ABLATE_TEMP", "0")) > 0          # pin temperature multiplier to 1.0
ABLATE_STEP_MULT   = int(os.environ.get("ABLATE_STEP_MULT", "0")) > 0     # pin RoPE step_mult to 1.0
ABLATE_GATE        = int(os.environ.get("ABLATE_GATE", "0")) > 0          # pin integration gate to 1.0 (uniform breath weighting)
ABLATE_INTEGRATION = int(os.environ.get("ABLATE_INTEGRATION", "0")) > 0   # no cross-breath integral; last-breath-only
ABLATE_NOTEBOOK    = int(os.environ.get("ABLATE_NOTEBOOK", "0")) > 0      # clear notebook before every controller call
ABLATE_ROTATION    = int(os.environ.get("ABLATE_ROTATION", "0")) > 0      # uniform RoPE phase (no per-head / per-loop offset)

# Sine-baseline temperature schedule (diffusion noise-schedule analog).
# Default OFF for backward compat. When SINE_TEMP=1, each breath gets a structural
# baseline temperature interpolated from SINE_TEMP_MAX (warm, breath 0) to
# SINE_TEMP_MIN (cool, breath max_loops-1) via cosine half-period. The controller's
# learned temp_mult MULTIPLIES the baseline as a problem-specific perturbation.
# This is the forcing function the architecture was missing — without it, every
# breath operates at the same temperature and there's no structural reason for
# per-breath outputs to differ (the v4 diagnostic confirmed: per-example loss is
# flat after breath 2).
SINE_TEMP          = int(os.environ.get("SINE_TEMP", "0")) > 0
SINE_TEMP_MAX      = float(os.environ.get("SINE_TEMP_MAX", "4.0"))
SINE_TEMP_MIN      = float(os.environ.get("SINE_TEMP_MIN", "0.5"))

# RoPE inter-breath rotation period. Default 0 = preserve existing behavior
# (loop_phase = l * π / max_loops, half-cycle over max_loops breaths, no wrap).
# When > 0: loop_phase = l * 2π / ROTATION_PERIOD, geometry returns to start
# every ROTATION_PERIOD breaths. ROTATION_PERIOD=4 with max_loops=8 gives two
# revolutions per pass (discovery → verification structure) and 50% per-breath
# head-phase overlap (vs current 87.5%).
ROTATION_PERIOD    = int(os.environ.get("ROTATION_PERIOD", "0"))

# Zero ONLY the per-breath rotation, keeping per-head spread intact. Tests the
# "depth from integration alone" hypothesis — does the architecture work as
# stacked disks (every breath sees the same 16 angular positions) instead of
# a helix? Distinct from ABLATE_ROTATION which kills BOTH per-head + per-breath.
ABLATE_BREATH_ROTATION = int(os.environ.get("ABLATE_BREATH_ROTATION", "0")) > 0

# Active axial conditioning — diffusion-style breath-time embedding. When enabled,
# adds a learned per-breath vector to the residual stream at the start of each
# breath, so the model explicitly knows "I'm at breath b". Closes the axial-vs-
# angular gap (we have per-head angular conditioning via RoPE, but no axial
# conditioning across breaths). Always present in state_dict for ckpt symmetry;
# only added to forward when env var is set.
#
# BREATH_TIME_INIT_SCALE controls the init magnitude. v9 used 0.02 (Pythia-style)
# which proved too disruptive to warm-starts — the random vectors added noise at
# every breath, the model couldn't both adapt to the new signal AND undo the
# representation damage. Setting to 0.0 (zero-init) means initial behavior is
# identical to the no-embed model; the gradient builds up breath_embed gradually
# as the loss rewards using it. Small positive values (e.g. 0.001) are an
# intermediate option.
BREATH_TIME_EMBED        = int(os.environ.get("BREATH_TIME_EMBED", "0")) > 0
BREATH_TIME_INIT_SCALE   = float(os.environ.get("BREATH_TIME_INIT_SCALE", "0.02"))

# Cross-breath handoff — relay-race baton between consecutive breaths. At the end
# of each breath, a learned linear projection produces a "handoff vector" from
# the breath's output. That vector is added to the next breath's input. Distinct
# from integration (which is a long running sum) and breath_embed (which is an
# unconditional positional signal): handoff is CONDITIONAL on the previous
# breath's content and fast-decay (just one breath of memory). Zero-init so
# initial behavior matches v11; gradient builds up the projection gradually.
CROSS_BREATH_HANDOFF     = int(os.environ.get("CROSS_BREATH_HANDOFF", "0")) > 0

# Learned helix pitch. Replaces the fixed per-breath rotation rate
# (loop_phase = l * π/max_loops or = 0) with a single learnable scalar.
# alpha(h, l) = head_phase[h] + l * pitch  where pitch is a model parameter.
# Zero-init: initial behavior matches ABLATE_BREATH_ROTATION=1 exactly (no per-breath
# rotation). Gradient discovers if/how much pitch the helix wants.
# Single scalar Tensor; trained jointly with the transformer.
LEARN_PITCH              = int(os.environ.get("LEARN_PITCH", "0")) > 0

# Constant-radius projection (kill DC growth, force rep onto cylinder).
# Direct test of the helix metaphor: a helix lives at constant distance R
# from its central axis. Currently the rep's L2 norm grows 3.9 → 6.4 across
# 8 breaths — model spirals outward, not orbiting on a cylinder.
#
# Mechanism: after each breath, blend the rep with a magnitude-normalized
# version using a learnable mix scalar (zero-init: no projection initially).
# If mix learns toward 1, the model WANTS to be on the cylinder (helix
# confirmed). If mix stays near 0, radial growth was useful (helix wrong).
#
#   x_normalized = x * (target_norm / x.norm(axis=-1))
#   x = (1 - mix_alpha) * x + mix_alpha * x_normalized
#
# Zero-init mix_alpha means initial forward is identical to no-projection.
CONSTANT_RADIUS          = int(os.environ.get("CONSTANT_RADIUS", "0")) > 0

# Per-layer angular pitch (within-breath rotation).
# Each of the 4 phase layers gets a different RoPE rotation: layer_idx * scale.
# The "helix completes one full rotation per breath via 4 quarter-turns" hypothesis.
# With TARGET=π/2, scale ramps from 0 → π/2 over RAMP_STEPS, then holds.
# Ramp gives smooth onset (no warm-start shock) while forcing exploration of the
# target value (unlike learnable scalar which might stay at zero).
LAYER_PITCH_TARGET       = float(os.environ.get("LAYER_PITCH_TARGET", "0.0"))   # π/2 ≈ 1.5708
LAYER_PITCH_RAMP_STEPS   = int(os.environ.get("LAYER_PITCH_RAMP_STEPS", "500"))

# Per-(layer, head) fixed pitch (v23a). When enabled, each layer gets a per-layer
# angular offset designed to maximally decorrelate the 4 layers × 16 heads = 64
# (layer, head) positions on the unit circle. Each layer is shifted by l * π/64
# relative to the previous, putting the layers at 4 distinct angular "lanes" with
# the base π-cycled head spacing of π/16 preserved within each lane.
#
# Designed to break the head-collision resonance discovered in v22: at LAYER_PITCH
# multiples of π/8 ≈ 2× head_phase_spacing, all heads in a non-zero layer landed
# on positions occupied by another layer's heads (causing dips at step 500, 1000).
#
# v23a init is FIXED (no learnable component); buffer not in parameters() so frozen.
# Tests whether *just having decorrelated head positions* recovers depth-helps.
PER_HEAD_PITCH           = int(os.environ.get("PER_HEAD_PITCH", "0")) > 0

# v24 "photon" components — each breath = one full wave cycle. Three coupled
# pieces, each independently togglable for ablation:
#   PER_BREATH_TEMP: temperature oscillates within each breath (warm at layer 0,
#     cool at layer n_phases//2, warming at layer n_phases-1). Each breath does
#     one full explore→commit→re-explore cycle.
#   BREATH_NORM_OSC: rep L2 norm follows the same wave via time-varying CRP
#     target (CONSTANT_RADIUS must also be on). Coupled "B field" perpendicular
#     to the angular "E field" (per-head pitch).
#   NOTEBOOK_V24: 512d notebook persisting across breaths within one forward
#     pass. Writes at the cool/sharp layer (the "measurement"), reads at the
#     warm/broad layer (the "next inhale"). Zero-init projections.
PER_BREATH_TEMP          = int(os.environ.get("PER_BREATH_TEMP", "0")) > 0
BREATH_NORM_OSC          = int(os.environ.get("BREATH_NORM_OSC", "0")) > 0
NOTEBOOK_V24             = int(os.environ.get("NOTEBOOK_V24", "0")) > 0

# v38 B-field: photon-analogy perpendicular axis to rotation. A single Information
# Bottleneck (1024 → BFIELD_WAIST → 1024) inserted between layers L1 and L2 of
# each breath. The rep is forced through a narrow waist once per breath cycle —
# one compression per rotation step, phase-locked to the angular E field.
#
# Forward: residual + GELU bottleneck.
#   compressed   = (x @ proj_down)          # (B, T, waist)
#   activated    = compressed.gelu()
#   decompressed = activated @ proj_up + bias
#   out          = x + decompressed
#
# Init: proj_down random Pythia-scale (0.02), proj_up + bias zero. With proj_up=0
# the initial decompressed=0 and forward output equals input — preserves any
# warm-start exactly. Gradient still flows to proj_up via the non-zero proj_down,
# then propagates to proj_down once proj_up moves (LoRA-style asymmetric init).
#
# 0 = disabled (params still allocated minimally for state_dict symmetry).
BFIELD_WAIST             = int(os.environ.get("BFIELD_WAIST", "0"))

# v38a CFG-style residual scale for the B-field. Multiplies the bottleneck's
# decompressed contribution before adding to the residual. At training time
# default 1.0 (standard residual). At inference time can be swept up — directly
# implements CFG extrapolation:
#   final = x + α · decompressed
#         = (1-α)·uncond + α·cond     [uncond=x, cond=x+decompressed]
# Stored as a Tensor scalar on BreathingBlock so it can be reassigned at runtime
# without JIT recompile (the Tensor is a graph input, not a baked constant).
BFIELD_ALPHA             = float(os.environ.get("BFIELD_ALPHA", "1.0"))

# v39 B-field placement and enforcement.
#
# BFIELD_END_OF_BREATH: when 1, the waist fires AFTER L3 (not between L1 and L2).
#   Each breath: L0 → L1 → L2 → L3 → waist → next breath input.
#   The breath's "thinking" is complete in 1024d before being committed via the
#   bottleneck. Maps to the inhale-exhale rhythm.
# BFIELD_ENFORCED: when 1, the waist REPLACES x (no residual skip). The
#   bottleneck output IS the breath's output. Forces all info through the
#   compressed scale — makes B-field load-bearing instead of an additive aux.
# BFIELD_SIN_MOD: when 1, the integral accumulates breath outputs weighted by
#   sin((l + 0.5) · π / n_loops) — a symmetric bell that peaks in the middle
#   of the breath sequence and never goes to zero at the endpoints (l=0 gives
#   ~0.195, l=n/2 gives ~0.98). The "slow cycle" envelope across breaths,
#   running over the per-breath fast cycle of explore (E in layers) → commit
#   (B at waist).
# BFIELD_AUX_WEIGHT: when > 0, applies op-classification CE on the 512d
#   compressed tensor at the "=" position. Mirrors LOOKUP_AUX_WEIGHT but at
#   the waist scale. Forces the bottleneck to encode op-discriminative info,
#   so the rep can't be informationally hollow under enforced mode.
BFIELD_END_OF_BREATH     = int(os.environ.get("BFIELD_END_OF_BREATH", "0")) > 0
BFIELD_ENFORCED          = int(os.environ.get("BFIELD_ENFORCED", "0")) > 0
BFIELD_SIN_MOD           = int(os.environ.get("BFIELD_SIN_MOD", "0")) > 0
BFIELD_AUX_WEIGHT        = float(os.environ.get("BFIELD_AUX_WEIGHT", "0.0"))

# v40 fresh-embedding-each-breath. When 1, each breath's INPUT is the original
# token embedding plus notebook context (read at breath start), NOT the previous
# breath's output. Breaks the rep-flow chain that caused v39's A=8 collapse:
# under enforced mode the previous breath's output went through the bottleneck,
# and chaining 8 such compressions degraded information catastrophically.
# With BREATHE_FRESH_INPUT=1, each breath sees a fresh embedding + REPLACE
# notebook context (RNN-style memory). The bottleneck still applies but only
# affects ONE breath's output at a time — no chain.
# Decouples computation (rep flow, fresh per breath) from memory (notebook).
BREATHE_FRESH_INPUT      = int(os.environ.get("BREATHE_FRESH_INPUT", "0")) > 0

# v24b notebook tuning. Default 0.0 = zero-init (v24a behavior). Small positive
# random init breaks the "stuck at zero" trap: with zero-init projections, the
# notebook output is exactly 0 and gradient signal through it is exactly 0 too
# — model never learns to use it. Pythia-style scale 0.02 gives a meaningful
# initial contribution (~1% of rep magnitude) so gradient can refine.
NOTEBOOK_INIT_SCALE      = float(os.environ.get("NOTEBOOK_INIT_SCALE", "0.0"))
# Write source: "mean" (default, mean-pool over seq) or "attn" (attention-weighted
# pool with a learnable query vector — model picks which positions to commit).
NOTEBOOK_POOL_MODE       = os.environ.get("NOTEBOOK_POOL_MODE", "mean")
# Dual notebook (v24c): when ON, adds a second 512d notebook with REPLACE semantics
# alongside the existing ACCUMULATE notebook. Replace notebook: nb_r = new_write
# (RNN-style hidden state, bounded magnitude, recent-only memory). Accumulate
# notebook: nb_a = nb_a + new_write (running sum, long-term memory, magnitude
# grows). Both read at breath start, both written at breath end. Model learns
# which is useful via separate read projections; an end-of-run ablation reveals
# load-bearing role.
NOTEBOOK_DUAL            = int(os.environ.get("NOTEBOOK_DUAL", "0")) > 0
# Ablation toggle (default ON for back-compat): when 0, the ACCUMULATE notebook's
# read/write contributions are skipped at runtime. Used to isolate the role of
# replace vs accumulate without retraining. Both notebooks still present in
# state_dict/parameters so ckpts round-trip; just the forward contribution is
# gated. Combined with NOTEBOOK_DUAL=1 this enables "replace only" eval.
NOTEBOOK_ACCUMULATE_ENABLED = int(os.environ.get("NOTEBOOK_ACCUMULATE_ENABLED", "1")) > 0
# v26: thread the notebook state across cycles in multi_cycle_train_step. When ON
# and n_cycles >= 2, cycle N's forward pass is seeded with cycle N-1's final
# notebook(s) instead of zeros. The notebook becomes a true cross-cycle "scratch
# pad" the model writes as it completes each reasoning step. Tests the train/
# inference alignment hypothesis from the v24c notebook ablation.
CROSS_CYCLE_NOTEBOOK     = int(os.environ.get("CROSS_CYCLE_NOTEBOOK", "0")) > 0
# Notebook initial STATE scale. 0.0 (default) = zero-init each forward (current).
# >0 = random N(0, scale²) init each forward — regularization so the model can't
# memorize "the initial state is zero." Crucial for cross-cycle notebook to work:
# without this, cross-cycle states (non-zero from prior cycle) are OOD vs the
# zero-init training distribution. Scale 0.5 roughly matches expected end-of-cycle
# notebook magnitude (writes have std ~0.4 with 0.02 projection init × prompt norm ~30).
NOTEBOOK_STATE_INIT_SCALE = float(os.environ.get("NOTEBOOK_STATE_INIT_SCALE", "0.0"))
# v28: prototype retrieval. The lookup table is extended with a values matrix
# (n_entries, hidden) — each "prime operation" entry now has both a KEY (where
# the basin sits in rep-space) and a VALUE (the ideal rep at the basin floor).
# At the end of the breath loop, query the lookup, compute weighted-sum of
# values via softmax(match_scores), and add to the final rep. Continuous
# Hopfield network over a learned prototype library — the model's energy
# landscape made explicit. Active when LOOKUP_VALUE_INJECT=1.
LOOKUP_VALUE_INJECT      = int(os.environ.get("LOOKUP_VALUE_INJECT", "0")) > 0
# Mix coefficient on the retrieved value contribution. 1.0 = full addition,
# 0.1 = gentle injection. Start at 1.0 since values are random-init small.
LOOKUP_VALUE_SCALE       = float(os.environ.get("LOOKUP_VALUE_SCALE", "1.0"))
# v30: Full 2π RoPE range (default π). When ROPE_FULL_CIRCLE=1, head phases span
# the full circle (2π/n_heads spacing instead of π/n_heads) — gives heads 2x more
# angular separation. Per-head pitch increment doubles correspondingly (π/32
# instead of π/64). Requires retraining from Pythia (weights are tuned to π range).
ROPE_FULL_CIRCLE         = int(os.environ.get("ROPE_FULL_CIRCLE", "0")) > 0
# Dropout rate applied after attn_out and ffn_out in each BreathingLayer. Default
# 0.0 (no dropout — preserves prior behavior). 0.1 is a standard transformer
# value; useful as anti-overfit regularization especially for our cycle-
# decomposed training where each piece is memorizable.
DROPOUT_RATE             = float(os.environ.get("DROPOUT_RATE", "0.0"))
# Lookup table information bottleneck. When >0, values are stored as (n_entries,
# IB_DIM) instead of (n_entries, hidden), with a learned (IB_DIM, hidden) projection
# expanding to hidden dim. Forces a compressed representation that can't encode
# example-specific details — only "procedure shape" survives the bottleneck.
LOOKUP_IB_DIM            = int(os.environ.get("LOOKUP_IB_DIM", "0"))


def _sine_temp_baseline(loop_idx: int, n_loops: int) -> float:
    """Cosine half-period temperature baseline. SINE_TEMP_MAX (warm) at loop_idx=0,
    SINE_TEMP_MIN (cool) at loop_idx=n_loops-1. Returns 1.0 if SINE_TEMP disabled
    (preserves original behavior)."""
    if not SINE_TEMP:
        return 1.0
    if n_loops <= 1:
        return (SINE_TEMP_MAX + SINE_TEMP_MIN) / 2.0
    cosine_phase = loop_idx * math.pi / (n_loops - 1)
    return ((SINE_TEMP_MAX + SINE_TEMP_MIN) / 2.0 +
            (SINE_TEMP_MAX - SINE_TEMP_MIN) / 2.0 * math.cos(cosine_phase))


def _per_layer_temp_within_breath(layer_idx: int, n_phases: int) -> float:
    """v24 photon-mode: full cosine wave per breath. One breath = one wavelength.
    Layer 0 = peak (warm), layer n_phases//2 = trough (cool), layer n_phases-1 =
    mid (warming for next breath). Continuous across breath boundaries.

    temp(k) = temp_mid + temp_amp * cos(2π * k / n_phases)
    """
    if n_phases <= 1:
        return (SINE_TEMP_MAX + SINE_TEMP_MIN) / 2.0
    temp_mid = (SINE_TEMP_MAX + SINE_TEMP_MIN) / 2.0
    temp_amp = (SINE_TEMP_MAX - SINE_TEMP_MIN) / 2.0
    return temp_mid + temp_amp * math.cos(2 * math.pi * layer_idx / n_phases)


def _initial_notebook_state(B: int, nb_dim: int) -> Tensor:
    """Return the initial notebook state tensor for a fresh forward pass.
    Zeros (default) when NOTEBOOK_STATE_INIT_SCALE=0.0, otherwise random Gaussian
    at the configured scale. Per-forward randomness regularizes the model to
    handle ANY notebook state — required for cross-cycle threading where state
    from prior cycle is non-zero.
    """
    if NOTEBOOK_STATE_INIT_SCALE > 0.0:
        return (Tensor.randn(B, nb_dim, dtype=dtypes.float) * NOTEBOOK_STATE_INIT_SCALE).contiguous()
    return Tensor.zeros((B, nb_dim), dtype=dtypes.float)


def _per_layer_norm_scale_within_breath(layer_idx: int, n_phases: int) -> float:
    """v24 photon-mode: rep norm follows the same wave as temperature. Layer 0 =
    peak amplitude (1.0×), layer n_phases//2 = collapse (NORM_MIN×), layer
    n_phases-1 = mid (recovering). Multiplies the CRP target_norm.
    """
    if n_phases <= 1:
        return 1.0
    NORM_MIN, NORM_MAX = 0.4, 1.0
    norm_mid = (NORM_MAX + NORM_MIN) / 2.0
    norm_amp = (NORM_MAX - NORM_MIN) / 2.0
    return norm_mid + norm_amp * math.cos(2 * math.pi * layer_idx / n_phases)

_active_ablations = [n for n, v in [
    ("TEMP", ABLATE_TEMP), ("STEP_MULT", ABLATE_STEP_MULT), ("GATE", ABLATE_GATE),
    ("INTEGRATION", ABLATE_INTEGRATION), ("NOTEBOOK", ABLATE_NOTEBOOK),
    ("ROTATION", ABLATE_ROTATION)] if v]
if _active_ablations:
    print(f"[ABLATE] active: {_active_ablations}", flush=True)
if SINE_TEMP:
    print(f"[SINE_TEMP] schedule: {SINE_TEMP_MAX} → {SINE_TEMP_MIN} (cosine half-period)", flush=True)
if ROTATION_PERIOD > 0:
    print(f"[ROTATION_PERIOD] {ROTATION_PERIOD}-breath cycle (loop_phase = l * 2π/{ROTATION_PERIOD})", flush=True)
if ABLATE_BREATH_ROTATION:
    print(f"[ABLATE_BREATH_ROTATION] per-breath rotation disabled (per-head spread preserved)", flush=True)
if BREATH_TIME_EMBED:
    init_str = f"init_scale={BREATH_TIME_INIT_SCALE}" + (" (zero-init)" if BREATH_TIME_INIT_SCALE == 0.0 else "")
    print(f"[BREATH_TIME_EMBED] active axial conditioning ({init_str})", flush=True)
if CROSS_BREATH_HANDOFF:
    print(f"[CROSS_BREATH_HANDOFF] zero-init handoff projection between breaths (relay-race baton)", flush=True)
if LEARN_PITCH:
    print(f"[LEARN_PITCH] helix pitch as learned scalar (zero-init; gradient discovers rotation rate)", flush=True)
if CONSTANT_RADIUS:
    print(f"[CONSTANT_RADIUS] cylinder projection at breath end (zero-init mix; gradient discovers if helix wants it)", flush=True)
if LAYER_PITCH_TARGET > 0.0:
    print(f"[LAYER_PITCH] per-layer rotation ramp 0 → {LAYER_PITCH_TARGET:.4f} rad over {LAYER_PITCH_RAMP_STEPS} steps", flush=True)
if PER_HEAD_PITCH:
    print(f"[PER_HEAD_PITCH] frozen per-(layer, head) offsets (max-decorrelation: layer l = l * π/64)", flush=True)
if PER_BREATH_TEMP:
    print(f"[PER_BREATH_TEMP] photon-mode temp wave per breath (layer 0 warm → layer n//2 cool → layer n-1 warming)", flush=True)
if BREATH_NORM_OSC:
    print(f"[BREATH_NORM_OSC] photon-mode rep norm oscillates per breath (CRP target scales 0.4× → 1.0× per wave)", flush=True)
if LOOKUP_VALUE_INJECT:
    print(f"[LOOKUP_VALUE_INJECT] prototype retrieval active, scale={LOOKUP_VALUE_SCALE}", flush=True)
if ROPE_FULL_CIRCLE:
    print(f"[ROPE_FULL_CIRCLE] head_phase uses full 2π range (vs default π)", flush=True)
if DROPOUT_RATE > 0:
    print(f"[DROPOUT] rate={DROPOUT_RATE} applied after attn + ffn (training only)", flush=True)
if LOOKUP_IB_DIM > 0:
    print(f"[LOOKUP_IB_DIM] information bottleneck at K={LOOKUP_IB_DIM} (lookup values projected through low-dim)", flush=True)
if NOTEBOOK_V24:
    init_str = f"init_scale={NOTEBOOK_INIT_SCALE}" + (" (zero-init)" if NOTEBOOK_INIT_SCALE == 0.0 else " (random)")
    nb_str = "DUAL (accumulate + replace)" if NOTEBOOK_DUAL else "single (accumulate)"
    if not NOTEBOOK_ACCUMULATE_ENABLED:
        nb_str += " [accumulate DISABLED — replace-only mode]"
    if CROSS_CYCLE_NOTEBOOK:
        nb_str += " + CROSS-CYCLE threading (v26)"
    if NOTEBOOK_STATE_INIT_SCALE > 0.0:
        nb_str += f" + state_init=N(0,{NOTEBOOK_STATE_INIT_SCALE}²)"
    print(f"[NOTEBOOK_V24] 512d notebook: {nb_str}, pool_mode={NOTEBOOK_POOL_MODE}, {init_str}", flush=True)


# ---------- partial RoPE with π cycling ----------

def _rope_base(seq_len: int, rotary_dim: int, base: int):
    """Half-rotation RoPE table for the rotated portion only.

    cos/sin: shape (seq_len, rotary_dim). The full head_dim is split into
    [rotated | unrotated]; only the first rotary_dim slots get the cos/sin.
    """
    half = rotary_dim // 2
    inv_freq = Tensor([1.0 / (base ** (2 * i / rotary_dim)) for i in range(half)], dtype=dtypes.float)
    pos = Tensor.arange(seq_len, dtype=dtypes.float)
    angles = pos.reshape(-1, 1) * inv_freq.reshape(1, -1)            # (seq, half)
    angles_full = Tensor.cat(angles, angles, dim=-1)                 # (seq, rotary_dim)
    return angles_full.cos().contiguous(), angles_full.sin().contiguous()


def _rotate(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Half-rotation RoPE applied to the rotated slice only.

    x:        (..., rotary_dim)
    cos, sin: broadcast to x's shape
    """
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    rotated = Tensor.cat(-x2, x1, dim=-1)
    return x * cos + rotated * sin


class RoPE:
    """π-cycled rotary position embedding, partial (Pythia-style 25%).

    Standard RoPE on both Q and K for relative position. Then Q gets an extra
    rotation by alpha(h, l) = h*pi/n_heads + l*pi/max_loops. Q-only application
    means q·k shifts by alpha (uniform offset on both would cancel).
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        rd = cfg.rotary_dim
        cos, sin = _rope_base(cfg.max_seq_len, rd, cfg.rope_base)
        # Realize so the tables are real buffers (not lazy ops). Lazy ops here
        # tangle with autograd and silently swallow gradients on consumer tensors.
        self.cos = cos.reshape(1, 1, cfg.max_seq_len, rd).contiguous().realize()
        self.sin = sin.reshape(1, 1, cfg.max_seq_len, rd).contiguous().realize()

        # v30: when ROPE_FULL_CIRCLE=1, use full 2π circle (2x spacing). Default π.
        head_phase_range = 2 * math.pi if ROPE_FULL_CIRCLE else math.pi
        head_phase = [h * head_phase_range / cfg.n_heads for h in range(cfg.n_heads)]
        if ABLATE_ROTATION:
            head_phase = [0.0] * cfg.n_heads  # no per-head phase diversity
        # Fixed head-phase tensor (the "R" of the helix — angular structure).
        # Used by the learned-pitch path to compute alpha on the fly.
        self.head_phase_t = Tensor(head_phase, dtype=dtypes.float).contiguous().realize()

        # Learned pitch — per-head rotation rate. Multi-scale helical analysis:
        # each head can learn its own pitch, giving 16 different helical scales.
        # v13b used (1,) shared scalar — neutral result. v14 = (n_heads,) for per-head.
        # Zero-init: initially no per-breath rotation, matching v8/v11/v12 setup.
        # Always present in state_dict for ckpt symmetry; only flows through forward
        # when LEARN_PITCH=1. When 0, the precomputed alpha tables are used (existing
        # behavior is preserved exactly).
        self.pitch = Tensor.zeros((cfg.n_heads,), dtype=dtypes.float).contiguous()

        # Precomputed alpha tables — used in the fast path when LEARN_PITCH=0.
        self.alpha_cos: List[Tensor] = []
        self.alpha_sin: List[Tensor] = []
        for l in range(cfg.max_loops):
            if ABLATE_ROTATION:
                loop_phase = 0.0
            elif ABLATE_BREATH_ROTATION:
                loop_phase = 0.0  # per-head spread preserved; only inter-breath drift zeroed
            elif ROTATION_PERIOD > 0:
                loop_phase = l * 2 * math.pi / ROTATION_PERIOD
            else:
                loop_phase = l * math.pi / cfg.max_loops
            alphas = [hp + loop_phase for hp in head_phase]
            ac = Tensor(alphas, dtype=dtypes.float).cos().reshape(1, cfg.n_heads, 1, 1).contiguous().realize()
            asn = Tensor(alphas, dtype=dtypes.float).sin().reshape(1, cfg.n_heads, 1, 1).contiguous().realize()
            self.alpha_cos.append(ac)
            self.alpha_sin.append(asn)

    def _alpha_at(self, loop_idx: int, target_dtype):
        """Compute alpha_cos and alpha_sin for the given breath.

        When LEARN_PITCH=1: alpha = head_phase + loop_idx * pitch (Tensor scalar).
        Gradient flows back to self.pitch via this path.

        When LEARN_PITCH=0: use precomputed alpha_cos[loop_idx] (no gradient).
        Matches v12 behavior exactly.
        """
        if LEARN_PITCH and not ABLATE_ROTATION:
            alpha = self.head_phase_t + float(loop_idx) * self.pitch       # (n_heads,)
            ac = alpha.cos().reshape(1, self.cfg.n_heads, 1, 1).cast(target_dtype)
            asn = alpha.sin().reshape(1, self.cfg.n_heads, 1, 1).cast(target_dtype)
            return ac, asn
        return (self.alpha_cos[loop_idx].cast(target_dtype),
                self.alpha_sin[loop_idx].cast(target_dtype))

    def apply_at_tensor_pos(self, q: Tensor, k: Tensor, loop_idx: int, t_pos_t: Tensor,
                             alpha: tuple | None = None):
        """Cached generation: S=1, t_pos as Tensor (shape () or (1,) for shared
        position across batch, or (B,) for per-batch positions). Builds the position-
        specific cos/sin via mask + sum over the full table.
        """
        cfg = self.cfg
        rd = cfg.rotary_dim
        max_len = int(self.cos.shape[2])
        # Determine if t_pos is per-batch or scalar
        per_batch = (t_pos_t.ndim == 1 and int(t_pos_t.shape[0]) > 1)
        B = int(t_pos_t.shape[0]) if per_batch else 1

        pos = Tensor.arange(max_len)
        if per_batch:
            # Per-batch position: mask shape (B, 1, max_len, 1)
            mask = (pos.reshape(1, max_len) == t_pos_t.reshape(B, 1)).reshape(B, 1, max_len, 1)
        else:
            mask = (pos == t_pos_t).reshape(1, 1, max_len, 1)
        cos_at = (self.cos * mask.cast(self.cos.dtype)).sum(axis=2, keepdim=True).cast(q.dtype)
        sin_at = (self.sin * mask.cast(self.sin.dtype)).sum(axis=2, keepdim=True).cast(q.dtype)

        q_rot, q_pass = q[..., :rd], q[..., rd:]
        k_rot, k_pass = k[..., :rd], k[..., rd:]
        q_rot = _rotate(q_rot, cos_at, sin_at)
        k_rot = _rotate(k_rot, cos_at, sin_at)

        if alpha is not None:
            ac, asn = alpha
        else:
            ac, asn = self._alpha_at(loop_idx, q.dtype)
        q_rot = _rotate(q_rot, ac, asn)

        return Tensor.cat(q_rot, q_pass, dim=-1), Tensor.cat(k_rot, k_pass, dim=-1)

    def apply(self, q: Tensor, k: Tensor, loop_idx: int, start_pos: int = 0,
              alpha: tuple | None = None):
        """q, k: (B, n_heads, seq, head_dim). Rotate first rotary_dim slots, leave rest.

        start_pos: offset into the position table (for KV-cached generation, when
        the new token sits at position T_past, not 0).
        """
        S = q.shape[2]
        rd = self.cfg.rotary_dim
        cos = self.cos[:, :, start_pos:start_pos + S, :].cast(q.dtype)
        sin = self.sin[:, :, start_pos:start_pos + S, :].cast(q.dtype)

        q_rot, q_pass = q[..., :rd], q[..., rd:]
        k_rot, k_pass = k[..., :rd], k[..., rd:]

        q_rot = _rotate(q_rot, cos, sin)
        k_rot = _rotate(k_rot, cos, sin)

        # π-cycled phase offset on Q only (rotated portion).
        # When alpha is provided (hoisted from breathe_once), use it directly — saves
        # 4× redundant computation across the 4 phase layers in a breath.
        if alpha is not None:
            ac, asn = alpha
        else:
            ac, asn = self._alpha_at(loop_idx, q.dtype)
        q_rot = _rotate(q_rot, ac, asn)

        q = Tensor.cat(q_rot, q_pass, dim=-1)
        k = Tensor.cat(k_rot, k_pass, dim=-1)
        return q, k


# ---------- weight initializers ----------

def _linear_w(in_dim: int, out_dim: int, dtype=dtypes.half) -> Tensor:
    """Pythia init: normal(0, 0.02). Stored (in, out) so x @ w needs no transpose."""
    return (Tensor.randn(in_dim, out_dim, dtype=dtype) * 0.02).contiguous()


def _bias(dim: int, dtype=dtypes.half) -> Tensor:
    return Tensor.zeros(dim, dtype=dtype).contiguous()


def _layernorm(x: Tensor, gamma: Tensor, beta: Tensor, eps: float = 1e-5) -> Tensor:
    """LayerNorm with FP32 internal compute, casts output back to input dtype."""
    in_dt = x.dtype
    x32 = x.cast(dtypes.float)
    mean = x32.mean(axis=-1, keepdim=True)
    var = ((x32 - mean) ** 2).mean(axis=-1, keepdim=True)
    out = (x32 - mean) * (var + eps).rsqrt()
    return (out * gamma + beta).cast(in_dt)


# ---------- shared & phase-specific weights ----------

class SharedWeights:
    """Weights tied across all 4 phase layers — initialized from Pythia layer 0."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.wv = _linear_w(cfg.hidden, cfg.hidden)
        self.bv = _bias(cfg.hidden)
        self.wo = _linear_w(cfg.hidden, cfg.hidden)
        self.bo = _bias(cfg.hidden)
        self.w_out = _linear_w(cfg.ffn, cfg.hidden)             # FFN dense_4h_to_h
        self.b_out = _bias(cfg.hidden)
        # Pythia has separate input_layernorm and post_attention_layernorm (use_parallel_residual=True).
        self.in_ln_g = Tensor.ones(cfg.hidden, dtype=dtypes.float).contiguous()
        self.in_ln_b = Tensor.zeros(cfg.hidden, dtype=dtypes.float).contiguous()
        self.post_ln_g = Tensor.ones(cfg.hidden, dtype=dtypes.float).contiguous()
        self.post_ln_b = Tensor.zeros(cfg.hidden, dtype=dtypes.float).contiguous()

    def parameters(self):
        return [self.wv, self.bv, self.wo, self.bo, self.w_out, self.b_out,
                self.in_ln_g, self.in_ln_b, self.post_ln_g, self.post_ln_b]


class BreathingLayer:
    """One phase of the breath — RISE, PEAK, FALL, or TROUGH.

    Phase-specific: Q, K, FFN-in (and their biases). Shared: V, O, FFN-out, LNs.
    Temperature: T = exp(temp_amp * sin(phase)); divides QK^T inside softmax.
    """

    def __init__(self, cfg: Config, phase: float, shared: SharedWeights, rope: RoPE):
        self.cfg = cfg
        self.phase = phase
        self.shared = shared
        self.rope = rope

        # Phase-specific
        self.wq = _linear_w(cfg.hidden, cfg.hidden)
        self.bq = _bias(cfg.hidden)
        self.wk = _linear_w(cfg.hidden, cfg.hidden)
        self.bk = _bias(cfg.hidden)
        self.w_in = _linear_w(cfg.hidden, cfg.ffn)              # FFN dense_h_to_4h
        self.b_in = _bias(cfg.ffn)

        # Sine-wave temperature: T = exp(amp * sin(phase))
        self.temperature = math.exp(cfg.temp_amp * math.sin(phase))
        self.attn_scale = 1.0 / (math.sqrt(cfg.head_dim) * self.temperature)

    def parameters(self):
        return [self.wq, self.bq, self.wk, self.bk, self.w_in, self.b_in]

    def __call__(self, x: Tensor, loop_idx: int, attn_mask: Tensor | None = None,
                 temp_mult: Tensor | float = 1.0, alpha: tuple | None = None) -> Tensor:
        return self._forward(x, loop_idx, kv_cache=None, return_kv=False,
                             attn_mask=attn_mask, temp_mult=temp_mult, alpha=alpha)[0]

    def forward_with_kv(self, x: Tensor, loop_idx: int, attn_mask: Tensor | None = None,
                        temp_mult: float = 1.0, alpha: tuple | None = None):
        """Full-sequence forward that also returns the post-RoPE K, V tensors.

        attn_mask: optional (B, S) bool/{0,1} tensor — 1 for valid, 0 for padding.
        When provided, padding positions don't influence attention (added as -inf to
        scores) and don't get gradient signal.
        alpha: optional precomputed (ac, asn) tuple; hoisted from the breath loop
        so all 4 layers in a breath share one alpha computation.
        """
        return self._forward(x, loop_idx, kv_cache=None, return_kv=True, attn_mask=attn_mask,
                             temp_mult=temp_mult, alpha=alpha)

    def forward_cached_step(self, x_new: Tensor, loop_idx: int, kv_cache):
        """Single-token (S=1) forward with cached past K/V. Returns (out, (k_full, v_full))."""
        return self._forward(x_new, loop_idx, kv_cache=kv_cache, return_kv=True)

    def forward_cached_step_batched(self, x_new: Tensor, loop_idx: int,
                                    k_buf: Tensor, v_buf: Tensor, t_pos_t: Tensor,
                                    prompt_mask: Tensor | None = None):
        """Batched single-token cached forward.

        x_new:        (B, 1, H)
        k_buf, v_buf: (B, n_heads, max_seq_len, head_dim) — buffers shared across batch
        t_pos_t:      0-dim or shape (1,) Tensor — uniform write position
        prompt_mask:  (B, max_seq_len) — 1 where the cache position holds a valid (non-pad)
                      prompt token. The new-token slot itself is added to the valid mask
                      via the causal `pos <= t_pos_t` comparison (no need to update the
                      prompt_mask between calls because future slots are zero by default
                      and t_pos_t monotonically advances).
        """
        cfg = self.cfg
        max_len = int(k_buf.shape[2])
        B = int(x_new.shape[0])

        attn_in = _layernorm(x_new, self.shared.in_ln_g, self.shared.in_ln_b, cfg.layer_norm_eps)
        mlp_in = _layernorm(x_new, self.shared.post_ln_g, self.shared.post_ln_b, cfg.layer_norm_eps)
        attn_in_dt = attn_in.cast(x_new.dtype) if attn_in.dtype != x_new.dtype else attn_in
        mlp_in_dt = mlp_in.cast(x_new.dtype) if mlp_in.dtype != x_new.dtype else mlp_in

        q_new = (attn_in_dt @ self.wq + self.bq).reshape(B, 1, cfg.n_heads, cfg.head_dim).transpose(1, 2)
        k_new = (attn_in_dt @ self.wk + self.bk).reshape(B, 1, cfg.n_heads, cfg.head_dim).transpose(1, 2)
        v_new = (attn_in_dt @ self.shared.wv + self.shared.bv).reshape(B, 1, cfg.n_heads, cfg.head_dim).transpose(1, 2)

        q_new, k_new = self.rope.apply_at_tensor_pos(q_new, k_new, loop_idx, t_pos_t)

        pos = Tensor.arange(max_len)
        per_batch = (t_pos_t.ndim == 1 and int(t_pos_t.shape[0]) > 1)
        if per_batch:
            write_at = (pos.reshape(1, max_len) == t_pos_t.reshape(B, 1)).reshape(B, 1, max_len, 1)
            causal = (pos.reshape(1, max_len) <= t_pos_t.reshape(B, 1)).reshape(B, 1, 1, max_len)
        else:
            write_at = (pos == t_pos_t).reshape(1, 1, max_len, 1)
            causal = (pos <= t_pos_t).reshape(1, 1, 1, max_len)
        k_new_b = k_new.expand(B, cfg.n_heads, max_len, cfg.head_dim)
        v_new_b = v_new.expand(B, cfg.n_heads, max_len, cfg.head_dim)
        k_buf_new = write_at.where(k_new_b, k_buf)
        v_buf_new = write_at.where(v_new_b, v_buf)

        # Per-batch t_pos already excludes Phase-A padding via causal mask (we only
        # attend to positions < t_pos_t[b], and shorter prompts have their first
        # generated token overwrite the Phase-A padding before any later token tries
        # to attend to it). prompt_mask kept as optional knob for caller-provided masking.
        if prompt_mask is not None:
            pmask = prompt_mask.reshape(B, 1, 1, max_len).cast(dtypes.bool)
            valid = causal & pmask
        else:
            valid = causal

        scores = q_new @ k_buf_new.transpose(-2, -1) * self.attn_scale
        scores = valid.where(scores, Tensor(-float("inf"), dtype=scores.dtype))
        attn = scores.softmax(-1).cast(v_buf_new.dtype)
        ctx = (attn @ v_buf_new).transpose(1, 2).reshape(B, 1, cfg.hidden)
        attn_out = ctx @ self.shared.wo + self.shared.bo

        ff = (mlp_in_dt @ self.w_in + self.b_in).gelu()
        ffn_out = ff @ self.shared.w_out + self.shared.b_out

        out = x_new + attn_out + ffn_out
        return out, k_buf_new, v_buf_new

    def forward_cached_step_jit(self, x_new: Tensor, loop_idx: int,
                                k_buf: Tensor, v_buf: Tensor, t_pos_t: Tensor):
        """Single-token cached forward, t_pos as Tensor scalar (JIT-replay friendly).
        Same compute as forward_cached_step_fixed but no Python-int slicing — every op
        has stable shape so a single TinyJit graph handles all positions."""
        cfg = self.cfg
        max_len = int(k_buf.shape[2])
        attn_dtype = x_new.dtype

        attn_in = _layernorm(x_new, self.shared.in_ln_g, self.shared.in_ln_b, cfg.layer_norm_eps)
        mlp_in = _layernorm(x_new, self.shared.post_ln_g, self.shared.post_ln_b, cfg.layer_norm_eps)
        attn_in_dt = attn_in.cast(attn_dtype) if attn_in.dtype != attn_dtype else attn_in
        mlp_in_dt = mlp_in.cast(attn_dtype) if mlp_in.dtype != attn_dtype else mlp_in

        q_new = (attn_in_dt @ self.wq + self.bq).reshape(1, 1, cfg.n_heads, cfg.head_dim).transpose(1, 2)
        k_new = (attn_in_dt @ self.wk + self.bk).reshape(1, 1, cfg.n_heads, cfg.head_dim).transpose(1, 2)
        v_new = (attn_in_dt @ self.shared.wv + self.shared.bv).reshape(1, 1, cfg.n_heads, cfg.head_dim).transpose(1, 2)

        q_new, k_new = self.rope.apply_at_tensor_pos(q_new, k_new, loop_idx, t_pos_t)

        pos = Tensor.arange(max_len)
        write_at = (pos == t_pos_t).reshape(1, 1, max_len, 1)
        k_new_b = k_new.expand(1, cfg.n_heads, max_len, cfg.head_dim)
        v_new_b = v_new.expand(1, cfg.n_heads, max_len, cfg.head_dim)
        k_buf_new = write_at.where(k_new_b, k_buf)
        v_buf_new = write_at.where(v_new_b, v_buf)

        valid = (pos <= t_pos_t).reshape(1, 1, 1, max_len)
        scores = q_new @ k_buf_new.transpose(-2, -1) * self.attn_scale
        scores = valid.where(scores, Tensor(-float("inf"), dtype=scores.dtype))
        attn = scores.softmax(-1).cast(v_buf_new.dtype)
        ctx = (attn @ v_buf_new).transpose(1, 2).reshape(1, 1, cfg.hidden)
        attn_out = ctx @ self.shared.wo + self.shared.bo

        ff = (mlp_in_dt @ self.w_in + self.b_in).gelu()
        ffn_out = ff @ self.shared.w_out + self.shared.b_out

        out = x_new + attn_out + ffn_out
        return out, k_buf_new, v_buf_new

    def forward_cached_step_fixed(self, x_new: Tensor, loop_idx: int,
                                  k_buf: Tensor, v_buf: Tensor, t_pos):
        """Single-token forward with FIXED-SHAPE K/V buffers + position-mask attention.

        t_pos may be a Python int or a 0-dim Tensor (the latter is required for
        TinyJit replay to handle multiple positions with one compiled graph).
        """
        cfg = self.cfg
        max_len = int(k_buf.shape[2])
        attn_dtype = x_new.dtype

        attn_in = _layernorm(x_new, self.shared.in_ln_g, self.shared.in_ln_b, cfg.layer_norm_eps)
        mlp_in = _layernorm(x_new, self.shared.post_ln_g, self.shared.post_ln_b, cfg.layer_norm_eps)
        attn_in_dt = attn_in.cast(attn_dtype) if attn_in.dtype != attn_dtype else attn_in
        mlp_in_dt = mlp_in.cast(attn_dtype) if mlp_in.dtype != attn_dtype else mlp_in

        q_new = (attn_in_dt @ self.wq + self.bq).reshape(1, 1, cfg.n_heads, cfg.head_dim).transpose(1, 2)
        k_new = (attn_in_dt @ self.wk + self.bk).reshape(1, 1, cfg.n_heads, cfg.head_dim).transpose(1, 2)
        v_new = (attn_in_dt @ self.shared.wv + self.shared.bv).reshape(1, 1, cfg.n_heads, cfg.head_dim).transpose(1, 2)

        # NOTE: For now t_pos is treated as a Python int (RoPE uses .start_pos slicing
        # which must resolve at trace time). When t_pos is a Tensor, this path will
        # need to use full-length cos/sin tables and apply with masking. For initial
        # JIT smoke we keep it as int and accept N JIT compiles for first N positions.
        q_new, k_new = self.rope.apply(q_new, k_new, loop_idx, start_pos=t_pos)

        pos = Tensor.arange(max_len, dtype=dtypes.int)
        write_at = (pos == t_pos).reshape(1, 1, max_len, 1)
        k_new_b = k_new.expand(1, cfg.n_heads, max_len, cfg.head_dim)
        v_new_b = v_new.expand(1, cfg.n_heads, max_len, cfg.head_dim)
        k_buf_new = write_at.where(k_new_b, k_buf)
        v_buf_new = write_at.where(v_new_b, v_buf)

        valid = (pos <= t_pos).reshape(1, 1, 1, max_len)
        scores = q_new @ k_buf_new.transpose(-2, -1) * self.attn_scale
        scores = valid.where(scores, Tensor(-float("inf"), dtype=scores.dtype))
        attn = scores.softmax(-1).cast(v_buf_new.dtype)
        ctx = (attn @ v_buf_new).transpose(1, 2).reshape(1, 1, cfg.hidden)
        attn_out = ctx @ self.shared.wo + self.shared.bo

        ff = (mlp_in_dt @ self.w_in + self.b_in).gelu()
        ffn_out = ff @ self.shared.w_out + self.shared.b_out

        out = x_new + attn_out + ffn_out
        return out, k_buf_new, v_buf_new

    def _forward(self, x: Tensor, loop_idx: int, kv_cache, return_kv: bool,
                 attn_mask: Tensor | None = None, temp_mult: Tensor | float = 1.0,
                 alpha: tuple | None = None):
        cfg = self.cfg
        B, S, H = x.shape

        attn_in = _layernorm(x, self.shared.in_ln_g, self.shared.in_ln_b, cfg.layer_norm_eps)
        mlp_in = _layernorm(x, self.shared.post_ln_g, self.shared.post_ln_b, cfg.layer_norm_eps)
        attn_in_dt = attn_in.cast(x.dtype) if attn_in.dtype != x.dtype else attn_in
        mlp_in_dt = mlp_in.cast(x.dtype) if mlp_in.dtype != x.dtype else mlp_in

        q = (attn_in_dt @ self.wq + self.bq).reshape(B, S, cfg.n_heads, cfg.head_dim).transpose(1, 2)
        k = (attn_in_dt @ self.wk + self.bk).reshape(B, S, cfg.n_heads, cfg.head_dim).transpose(1, 2)
        v = (attn_in_dt @ self.shared.wv + self.shared.bv).reshape(B, S, cfg.n_heads, cfg.head_dim).transpose(1, 2)

        # Adaptive temperature: built-in sine schedule × 1/temp_mult (higher mult → softer attention).
        # When temp_mult=1.0 (default), behavior is identical to the original fixed sine schedule.
        if isinstance(temp_mult, Tensor):
            scale = self.attn_scale * (1.0 / temp_mult).cast(q.dtype).reshape(B, 1, 1, 1)
        else:
            scale = self.attn_scale / float(temp_mult)

        if kv_cache is None:
            q, k = self.rope.apply(q, k, loop_idx, start_pos=0, alpha=alpha)
            scores = q @ k.transpose(-2, -1) * scale
            mask = Tensor.ones(S, S, dtype=scores.dtype).tril().reshape(1, 1, S, S)
            scores = scores.masked_fill(mask == 0, float("-inf"))
            if attn_mask is not None:
                # attn_mask shape: (B, S) — 1 valid, 0 padding. Broadcast to (B, 1, 1, S).
                key_mask = attn_mask.reshape(B, 1, 1, S).cast(dtypes.bool)
                scores = key_mask.where(scores, Tensor(-float("inf"), dtype=scores.dtype))
            attn = scores.softmax(-1).cast(v.dtype)
            ctx = (attn @ v).transpose(1, 2).reshape(B, S, H)
            attn_out = ctx @ self.shared.wo + self.shared.bo
            ff = (mlp_in_dt @ self.w_in + self.b_in).gelu()
            ffn_out = ff @ self.shared.w_out + self.shared.b_out
            # Dropout (training only — Tensor.dropout respects Tensor.training)
            if DROPOUT_RATE > 0:
                attn_out = attn_out.dropout(DROPOUT_RATE)
                ffn_out = ffn_out.dropout(DROPOUT_RATE)
            out = x + attn_out + ffn_out
            return (out, (k, v)) if return_kv else (out, None)

        # Cached path — single new token (S==1) attending over (cached past + itself).
        k_past, v_past = kv_cache
        t_past = int(k_past.shape[2])
        q, k = self.rope.apply(q, k, loop_idx, start_pos=t_past, alpha=alpha)
        k_full = Tensor.cat(k_past, k, dim=2)        # (B, n_heads, T_past+1, head_dim)
        v_full = Tensor.cat(v_past, v, dim=2)
        # No causal mask: new token can attend to all past + itself.
        scores = q @ k_full.transpose(-2, -1) * scale
        attn = scores.softmax(-1).cast(v_full.dtype)
        ctx = (attn @ v_full).transpose(1, 2).reshape(B, S, H)
        attn_out = ctx @ self.shared.wo + self.shared.bo
        ff = (mlp_in_dt @ self.w_in + self.b_in).gelu()
        ffn_out = ff @ self.shared.w_out + self.shared.b_out
        out = x + attn_out + ffn_out
        return out, (k_full, v_full)


# ---------- block: 4 phases × N loops + integration ----------

class BreathingBlock:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.shared = SharedWeights(cfg)
        self.rope = RoPE(cfg)
        # Phases trace one full sine wave: 0, π/2, π, 3π/2.
        phases = [i * 2 * math.pi / cfg.n_phases for i in range(cfg.n_phases)]
        self.layers = [BreathingLayer(cfg, ph, self.shared, self.rope) for ph in phases]
        # Diffusion-style breath-time embedding (axial conditioning). Always created
        # so the parameter list is stable across env-var toggles; only added to the
        # residual stream when BREATH_TIME_EMBED=1.
        #
        # Init magnitude controlled by BREATH_TIME_INIT_SCALE (env var, default 0.02).
        # Zero-init (=0.0) preserves warm-start behavior; gradient builds the embed
        # gradually as loss rewards using it. Small positive values introduce a
        # weak prior signal from the start.
        if BREATH_TIME_INIT_SCALE > 0.0:
            self.breath_embed = (Tensor.randn(cfg.max_loops, cfg.hidden, dtype=dtypes.float)
                                  * BREATH_TIME_INIT_SCALE).contiguous()
        else:
            self.breath_embed = Tensor.zeros((cfg.max_loops, cfg.hidden), dtype=dtypes.float).contiguous()

        # Cross-breath handoff projection (relay-race baton). Zero-init means initial
        # handoff = 0 → forward behavior identical to no-handoff. Gradient builds it up.
        # Always present in state_dict for ckpt symmetry; only threaded into the breath
        # loop when CROSS_BREATH_HANDOFF=1.
        self.handoff_w = Tensor.zeros((cfg.hidden, cfg.hidden), dtype=dtypes.float).contiguous()
        self.handoff_b = Tensor.zeros((cfg.hidden,), dtype=dtypes.float).contiguous()

        # Constant-radius projection scalars. zero-init mix → no projection initially.
        # Target norm starts at ~30 (typical activation L2 norm for 1024-d fp16 reps).
        # Both shape (1,) for AdamW compatibility.
        self.crp_mix_alpha = Tensor.zeros((1,), dtype=dtypes.float).contiguous()
        self.crp_target_norm = (Tensor.ones((1,), dtype=dtypes.float) * 30.0).contiguous()

        # Per-layer pitch scale — NOT learnable. Ramped from 0 → LAYER_PITCH_TARGET
        # over LAYER_PITCH_RAMP_STEPS by the training script (via .assign()).
        # Buffer Tensor — included in state_dict for ckpt symmetry, NOT in parameters.
        self.layer_pitch_scale = Tensor.zeros((1,), dtype=dtypes.float).contiguous()

        # Per-(layer, head) fixed pitch (v23a). Buffer shape (n_layers, n_heads) so
        # later versions can vary per-head (v23b: learnable bounded). v23a init:
        # uniform per-layer offset l * π/64 for layer l (all heads in layer get
        # the same offset). Each layer at its own angular "lane" — 4 lanes spaced
        # π/64 apart, each lane keeping the base π-cycled head structure.
        #
        # 4 layers × 16 heads = 64 unique (layer, head) angular positions in [0, π + 3π/64).
        # Adjacent positions spaced π/64 ≈ 2.8° apart — finer than v22's collision-
        # prone uniform layer rotation.
        #
        # NOT in parameters() — frozen. Always in state_dict for ckpt symmetry.
        # v30: when ROPE_FULL_CIRCLE=1, head_phase spacing doubles (2π/n_heads vs
        # π/n_heads), so layer increment doubles too (π/32 vs π/64) to maintain
        # quarter-spacing within the new head range.
        pitch_range = 2 * math.pi if ROPE_FULL_CIRCLE else math.pi
        ph_init = [[l * pitch_range / (cfg.n_phases * cfg.n_heads) for _ in range(cfg.n_heads)]
                   for l in range(cfg.n_phases)]
        self.per_head_pitch = Tensor(ph_init, dtype=dtypes.float).contiguous()
        # Precomputed cos/sin tables — eliminates per-breath cos/sin compute.
        # Shape (n_layers, 1, n_heads, 1, 1) to match alpha broadcast shape on indexing.
        # Realize so JIT sees these as constant buffers, not lazy ops (lazy ops
        # under per-breath JIT recomputation triggered MEMVIOL on first v23 attempt).
        ph_t = Tensor(ph_init, dtype=dtypes.float)
        self.per_head_pitch_cos = ph_t.cos().reshape(cfg.n_phases, 1, cfg.n_heads, 1, 1).contiguous().realize()
        self.per_head_pitch_sin = ph_t.sin().reshape(cfg.n_phases, 1, cfg.n_heads, 1, 1).contiguous().realize()

        # v24 notebook (the "measurement record"). 512d state, written after each
        # breath, read before each breath. Always in parameters for ckpt symmetry.
        #
        # Init:
        #   NOTEBOOK_INIT_SCALE=0 (default): zero-init projections (v24a behavior).
        #     With zero output and zero gradient, model can get stuck never using
        #     the notebook — observed empirically in v24 step 1500 ablation.
        #   NOTEBOOK_INIT_SCALE>0: small random init (Pythia-style 0.02). Initial
        #     read contribution is ~1% of rep magnitude — gradient gets signal.
        # Write source (NOTEBOOK_POOL_MODE):
        #   "mean": mean-pool over sequence (v24a behavior; loses position info)
        #   "attn": attention-weighted pool with learnable query vector — model
        #     picks which token positions to commit to the notebook.
        NB_DIM = 512
        self.nb_dim = NB_DIM
        if NOTEBOOK_INIT_SCALE > 0.0:
            s = NOTEBOOK_INIT_SCALE
            self.notebook_write_w = (Tensor.randn(cfg.hidden, NB_DIM, dtype=dtypes.float) * s).contiguous()
            self.notebook_read_w  = (Tensor.randn(NB_DIM, cfg.hidden, dtype=dtypes.float) * s).contiguous()
        else:
            self.notebook_write_w = Tensor.zeros((cfg.hidden, NB_DIM), dtype=dtypes.float).contiguous()
            self.notebook_read_w  = Tensor.zeros((NB_DIM, cfg.hidden), dtype=dtypes.float).contiguous()
        self.notebook_write_b = Tensor.zeros((NB_DIM,), dtype=dtypes.float).contiguous()
        self.notebook_read_b  = Tensor.zeros((cfg.hidden,), dtype=dtypes.float).contiguous()
        # Attention-pool query vector for write source "attn". Zero-init means
        # initial attention scores are uniform → falls back to mean-pool. Gradient
        # learns which positions matter (e.g., "=", end-of-answer, etc.).
        self.notebook_write_query = Tensor.zeros((cfg.hidden,), dtype=dtypes.float).contiguous()
        # v24c dual notebook — REPLACE-semantics partner to the accumulate notebook
        # above. Same shape, same init pattern. Active only when NOTEBOOK_DUAL=1;
        # otherwise present in state_dict for ckpt symmetry but gradient is inert.
        # Distinct update rule: nb_r = new_write (full overwrite each breath).
        # Bounded magnitude, RNN-style hidden state, complement to accumulator's
        # long-term memory.
        if NOTEBOOK_INIT_SCALE > 0.0:
            s = NOTEBOOK_INIT_SCALE
            self.notebook_rep_write_w = (Tensor.randn(cfg.hidden, NB_DIM, dtype=dtypes.float) * s).contiguous()
            self.notebook_rep_read_w  = (Tensor.randn(NB_DIM, cfg.hidden, dtype=dtypes.float) * s).contiguous()
        else:
            self.notebook_rep_write_w = Tensor.zeros((cfg.hidden, NB_DIM), dtype=dtypes.float).contiguous()
            self.notebook_rep_read_w  = Tensor.zeros((NB_DIM, cfg.hidden), dtype=dtypes.float).contiguous()
        self.notebook_rep_write_b = Tensor.zeros((NB_DIM,), dtype=dtypes.float).contiguous()
        self.notebook_rep_read_b  = Tensor.zeros((cfg.hidden,), dtype=dtypes.float).contiguous()
        self.notebook_rep_query = Tensor.zeros((cfg.hidden,), dtype=dtypes.float).contiguous()

        # v38 B-field IB bottleneck — single waist between L1 and L2 per breath.
        # Allocate at max(1, BFIELD_WAIST) so state_dict shapes are consistent
        # across runs even when disabled; gradient is inert when BFIELD_WAIST=0.
        bf_w = max(1, BFIELD_WAIST)
        self.bfield_waist_dim = bf_w
        # proj_down always random Pythia-scale.
        self.bfield_proj_down = (Tensor.randn(cfg.hidden, bf_w, dtype=dtypes.float) * 0.02).contiguous()
        # proj_up: zero-init in residual mode (preserves warm-start, gradient activates
        # via the non-zero proj_down — LoRA pattern). Random init in enforced mode
        # because output = decompressed; zero proj_up would zero the breath's output.
        if BFIELD_ENFORCED:
            # 1/sqrt(bf_w) keeps the projected magnitudes similar to x — preserves
            # rough rep magnitude through the bottleneck on the first forward.
            up_scale = (1.0 / max(bf_w, 1)) ** 0.5
            self.bfield_proj_up = (Tensor.randn(bf_w, cfg.hidden, dtype=dtypes.float) * up_scale).contiguous()
        else:
            self.bfield_proj_up = Tensor.zeros((bf_w, cfg.hidden), dtype=dtypes.float).contiguous()
        self.bfield_bias      = Tensor.zeros((cfg.hidden,), dtype=dtypes.float).contiguous()
        # v38a CFG-style residual scale. Tensor (1,) so it's a JIT graph input
        # and can be reassigned via .assign() at inference without recompile.
        self.bfield_alpha     = (Tensor.ones((1,), dtype=dtypes.float) * BFIELD_ALPHA).contiguous()

    def apply_bfield_waist(self, x: Tensor, return_compressed: bool = False) -> Tensor:
        """B-field IB bottleneck with optional CFG-alpha residual scale.

        Residual mode (BFIELD_ENFORCED=0):
            out = x + α · decompressed     (zero-init proj_up → identity at step 0)
        Enforced mode (BFIELD_ENFORCED=1):
            out = decompressed             (no skip — rep must flow through waist)

        When return_compressed=True, returns (out, compressed) for aux supervision
        on the 512d intermediate.
        """
        x_f = x.cast(dtypes.float)
        compressed = x_f @ self.bfield_proj_down
        activated = compressed.gelu()
        decompressed = activated @ self.bfield_proj_up + self.bfield_bias
        if BFIELD_ENFORCED:
            out = decompressed.cast(x.dtype)
        else:
            out = (x_f + self.bfield_alpha * decompressed).cast(x.dtype)
        if return_compressed:
            return out, compressed.cast(x.dtype)
        return out

    def parameters(self):
        ps = list(self.shared.parameters())
        for layer in self.layers:
            ps.extend(layer.parameters())
        ps.append(self.breath_embed)
        ps.append(self.handoff_w)
        ps.append(self.handoff_b)
        ps.append(self.rope.pitch)
        ps.append(self.crp_mix_alpha)
        ps.append(self.crp_target_norm)
        ps.append(self.notebook_write_w)
        ps.append(self.notebook_write_b)
        ps.append(self.notebook_read_w)
        ps.append(self.notebook_read_b)
        ps.append(self.notebook_write_query)
        ps.append(self.notebook_rep_write_w)
        ps.append(self.notebook_rep_write_b)
        ps.append(self.notebook_rep_read_w)
        ps.append(self.notebook_rep_read_b)
        ps.append(self.notebook_rep_query)
        ps.append(self.bfield_proj_down)
        ps.append(self.bfield_proj_up)
        ps.append(self.bfield_bias)
        return ps

    def compute_handoff(self, x: Tensor) -> Tensor:
        """Linear projection of a breath output to its handoff vector for the next
        breath. Zero-init means this returns ~0 initially; gradient learns useful values."""
        x_f = x.cast(dtypes.float)
        return (x_f @ self.handoff_w + self.handoff_b).cast(x.dtype)

    def breathe_once(self, x: Tensor, loop_idx: int, temp_mult: float = 1.0,
                     return_waist_compressed: bool = False):
        """One breath: 4 layers sequentially. temp_mult scales attention softness
        per breath (used by the sine-baseline schedule when SINE_TEMP=1).

        When BREATH_TIME_EMBED=1, adds the learned per-breath axial embedding to
        the residual stream at the start of the breath — the model explicitly
        knows "I'm at breath loop_idx" (analogous to diffusion timestep conditioning).

        Perf: alpha is computed ONCE per breath here and shared across the 4 layers
        (vs 4× redundantly inside each layer's rope.apply call). 4× reduction in
        per-breath alpha overhead.

        v24 photon mode (PER_BREATH_TEMP=1, BREATH_NORM_OSC=1): each breath is one
        full wave. Temperature warm (layer 0) → cool (layer n//2) → warming. Rep norm
        follows the same wave via time-varying CRP target. Across-breath SINE_TEMP
        baseline is OVERRIDDEN by the per-layer wave when PER_BREATH_TEMP=1.
        """
        alpha = self.rope._alpha_at(loop_idx, x.dtype)
        if BREATH_TIME_EMBED:
            x = x + self.breath_embed[loop_idx].reshape(1, 1, -1).cast(x.dtype)
        ac_base, asn_base = alpha
        n_phases = self.cfg.n_phases
        for layer_idx, layer in enumerate(self.layers):
            # --- pitch (v23a / legacy) ---
            if PER_HEAD_PITCH and layer_idx > 0:
                cos_o = self.per_head_pitch_cos[layer_idx].cast(x.dtype)
                sin_o = self.per_head_pitch_sin[layer_idx].cast(x.dtype)
                ac_layer = ac_base * cos_o - asn_base * sin_o
                asn_layer = ac_base * sin_o + asn_base * cos_o
                layer_alpha = (ac_layer, asn_layer)
            elif LAYER_PITCH_TARGET > 0.0 and layer_idx > 0:
                offset_angle = (self.layer_pitch_scale * float(layer_idx)).cast(dtypes.float)
                cos_o = offset_angle.cos().reshape(1, 1, 1, 1).cast(x.dtype)
                sin_o = offset_angle.sin().reshape(1, 1, 1, 1).cast(x.dtype)
                ac_layer = ac_base * cos_o - asn_base * sin_o
                asn_layer = ac_base * sin_o + asn_base * cos_o
                layer_alpha = (ac_layer, asn_layer)
            else:
                layer_alpha = alpha
            # --- per-layer temperature (v24) — overrides across-breath baseline ---
            if PER_BREATH_TEMP:
                layer_temp = _per_layer_temp_within_breath(layer_idx, n_phases)
            else:
                layer_temp = temp_mult
            x = layer(x, loop_idx, temp_mult=layer_temp, alpha=layer_alpha)
            # --- per-layer norm oscillation (v24) — applied between layers ---
            if BREATH_NORM_OSC and CONSTANT_RADIUS:
                scale = _per_layer_norm_scale_within_breath(layer_idx, n_phases)
                x_f = x.cast(dtypes.float)
                x_norm = (x_f.square().sum(axis=-1, keepdim=True) + 1e-6).sqrt()
                target = self.crp_target_norm * scale
                mix = self.crp_mix_alpha
                x_proj = x_f * (target / x_norm)
                x = (x_f * (1.0 - mix) + x_proj * mix).cast(x.dtype)
            # --- v38 B-field IB bottleneck: mid-breath waist (after L1) ---
            # v39: when BFIELD_END_OF_BREATH=1, this fires AFTER L3 instead (below).
            if BFIELD_WAIST > 0 and layer_idx == 1 and not BFIELD_END_OF_BREATH:
                x = self.apply_bfield_waist(x)
        # --- v39 B-field end-of-breath waist (after all 4 layers complete) ---
        waist_compressed = None
        if BFIELD_WAIST > 0 and BFIELD_END_OF_BREATH:
            if return_waist_compressed:
                x, waist_compressed = self.apply_bfield_waist(x, return_compressed=True)
            else:
                x = self.apply_bfield_waist(x)
        # End-of-breath CRP — only when per-layer oscillation is OFF (otherwise the
        # last layer's per-layer CRP already did this).
        if CONSTANT_RADIUS and not BREATH_NORM_OSC:
            x_f = x.cast(dtypes.float)
            x_norm = (x_f.square().sum(axis=-1, keepdim=True) + 1e-6).sqrt()
            target = self.crp_target_norm
            mix = self.crp_mix_alpha
            x_proj = x_f * (target / x_norm)
            x = (x_f * (1.0 - mix) + x_proj * mix).cast(x.dtype)
        if return_waist_compressed:
            return x, waist_compressed
        return x

    def breathe(self, x: Tensor, n_loops: int,
                initial_notebook: Tensor | None = None,
                initial_notebook_r: Tensor | None = None,
                return_notebook: bool = False):
        """Loop the 4-layer breath n_loops times with gated running integral.

        Stub controller: gate=1 every breath, always continue. Returns the normalized
        integral (running mean) of all breath outputs. When SINE_TEMP=1 each breath
        operates at its scheduled temperature. When CROSS_BREATH_HANDOFF=1, the
        previous breath's output is projected and added to the next breath's input
        (relay-race baton). When NOTEBOOK_V24=1, a 512d notebook accumulates across
        breaths: read at breath start (the "inhale" with prior measurements), write
        at breath end (the post-collapse committed state).

        v26 cross-cycle notebook: pass initial_notebook (and initial_notebook_r when
        dual) to seed from a prior cycle's final state. Set return_notebook=True to
        get the post-run notebook(s) for threading into the next cycle.
        """
        x_emb = x  # v40: frozen embedding, reused each breath in fresh mode
        integral = Tensor.zeros_like(x)
        gate_sum = 0.0
        handoff = None
        notebook = None        # accumulate notebook (always when NOTEBOOK_V24)
        notebook_r = None      # replace notebook (only when NOTEBOOK_DUAL)
        if NOTEBOOK_V24:
            B = x.shape[0]
            notebook = initial_notebook if initial_notebook is not None else _initial_notebook_state(B, self.nb_dim)
            if NOTEBOOK_DUAL:
                notebook_r = initial_notebook_r if initial_notebook_r is not None else _initial_notebook_state(B, self.nb_dim)
        for l in range(n_loops):
            # v40 fresh-input: each breath starts from the original embedding
            if BREATHE_FRESH_INPUT:
                x_in = x_emb
            else:
                x_in = x
                if CROSS_BREATH_HANDOFF and handoff is not None:
                    x_in = x_in + handoff
            # Read notebook before breath
            if NOTEBOOK_V24:
                B = x_in.shape[0]
                if NOTEBOOK_ACCUMULATE_ENABLED:
                    read_vec = (notebook @ self.notebook_read_w + self.notebook_read_b)
                    x_in = x_in + read_vec.reshape(B, 1, -1).cast(x_in.dtype)
                if NOTEBOOK_DUAL:
                    read_vec_r = (notebook_r @ self.notebook_rep_read_w + self.notebook_rep_read_b)
                    x_in = x_in + read_vec_r.reshape(B, 1, -1).cast(x_in.dtype)
            x = self.breathe_once(x_in, l, temp_mult=_sine_temp_baseline(l, n_loops))
            # Write notebook after breath. Pool source determined by NOTEBOOK_POOL_MODE.
            if NOTEBOOK_V24:
                x_f = x.cast(dtypes.float)
                if NOTEBOOK_POOL_MODE == "attn":
                    scores = (x_f * self.notebook_write_query.reshape(1, 1, -1)).sum(axis=-1)
                    weights = scores.softmax(axis=-1).reshape(B, -1, 1)
                    x_pool = (x_f * weights).sum(axis=1)
                else:
                    x_pool = x_f.mean(axis=1)
                if NOTEBOOK_ACCUMULATE_ENABLED:
                    notebook = notebook + (x_pool @ self.notebook_write_w + self.notebook_write_b)
                if NOTEBOOK_DUAL:
                    # Replace notebook uses its own attn query (potentially picking different positions)
                    if NOTEBOOK_POOL_MODE == "attn":
                        scores_r = (x_f * self.notebook_rep_query.reshape(1, 1, -1)).sum(axis=-1)
                        weights_r = scores_r.softmax(axis=-1).reshape(B, -1, 1)
                        x_pool_r = (x_f * weights_r).sum(axis=1)
                    else:
                        x_pool_r = x_pool  # share the mean-pooled source
                    # REPLACE semantics: overwrite the entire notebook each breath
                    notebook_r = x_pool_r @ self.notebook_rep_write_w + self.notebook_rep_write_b
            if CROSS_BREATH_HANDOFF:
                handoff = self.compute_handoff(x)
            # v39 sin-modulated integral envelope across breaths (heartbeat)
            if BFIELD_SIN_MOD:
                beta_l = math.sin((l + 0.5) * math.pi / float(n_loops))
            else:
                beta_l = 1.0
            integral = integral + beta_l * x
            gate_sum += beta_l
        if return_notebook:
            return integral / gate_sum, notebook, notebook_r
        return integral / gate_sum


# ---------- top-level model ----------

class BreathingTransformer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.embed = Embedding(cfg.vocab_size, cfg.hidden)
        self.block = BreathingBlock(cfg)
        # Final LN on the integrated representation.
        self.ln_f_g = Tensor.ones(cfg.hidden, dtype=dtypes.float).contiguous()
        self.ln_f_b = Tensor.zeros(cfg.hidden, dtype=dtypes.float).contiguous()
        # Output head (untied — Pythia has separate embed_out weight, no bias).
        self.embed_out = _linear_w(cfg.hidden, cfg.vocab_size)
        # Closed-loop component #4: prime-operation lookup table. 16 entries × 1024d.
        # Random orthogonal init; joint-trained via aux op-classification CE so the
        # entries align with the model's actual operation directions.
        self.lookup_table = LookupTable(n_entries=cfg.n_lookup_entries,
                                        hidden=cfg.hidden,
                                        seed=cfg.seed_lookup)
        # Closed-loop component #5: the controller. State reader + decision heads.
        # Step B scaffold; notebook (Step C) and adaptive wiring (Step D) follow.
        self.controller = Controller(cfg)
        # Per-step optimal-stopping calibration head. Reads the rep at a step's
        # "=" position and emits scalar confidence in (0,1). Trained jointly with
        # the transformer on the REINFORCE optimal-stopping objective.
        self.confidence_head = ConfidenceHead(cfg.hidden)
        # v39 waist head: classifies op label from the 512d compressed tensor at
        # the "=" position. Trained jointly via BFIELD_AUX_WEIGHT — forces the
        # B-field bottleneck to encode op-discriminative information in enforced
        # mode. Pythia-scale init; bias zero.
        bf_w = max(1, BFIELD_WAIST)
        self.waist_head_w = (Tensor.randn(bf_w, 4, dtype=dtypes.float) * 0.02).contiguous()
        self.waist_head_b = Tensor.zeros((4,), dtype=dtypes.float).contiguous()

    def parameters(self):
        """Parameters trained on the main loss (transformer + lookup table +
        confidence head). The controller has gradient separation per the spec —
        its parameters are returned by controller_parameters() and trained by a
        separate optimizer (Step F) via REINFORCE on outcomes + auxiliary signals."""
        return ([self.embed.weight, self.ln_f_g, self.ln_f_b, self.embed_out]
                + self.block.parameters()
                + self.lookup_table.parameters()
                + self.confidence_head.parameters()
                + [self.waist_head_w, self.waist_head_b])

    def controller_parameters(self):
        """Controller-only parameters. Trained via a separate optimizer with a
        non-overlapping signal (gradient separation enforced by construction —
        the main loss never reaches these params)."""
        return self.controller.parameters()

    def state_dict(self) -> dict:
        """Single source of truth for ckpt save/load. New components register here."""
        sd = {
            "embed.weight": self.embed.weight,
            "embed_out": self.embed_out,
            "ln_f.g": self.ln_f_g,
            "ln_f.b": self.ln_f_b,
            "lookup_table.weight": self.lookup_table.weight,
            "lookup_table.values": self.lookup_table.values,
            "lookup_table.value_proj_up": self.lookup_table.value_proj_up,
        }
        sw = self.block.shared
        for a in ("wv", "bv", "wo", "bo", "w_out", "b_out",
                 "in_ln_g", "in_ln_b", "post_ln_g", "post_ln_b"):
            sd[f"shared.{a}"] = getattr(sw, a)
        for i, layer in enumerate(self.block.layers):
            for a in ("wq", "bq", "wk", "bk", "w_in", "b_in"):
                sd[f"phase{i}.{a}"] = getattr(layer, a)
        sd["block.breath_embed"] = self.block.breath_embed
        sd["block.handoff_w"] = self.block.handoff_w
        sd["block.handoff_b"] = self.block.handoff_b
        sd["block.rope.pitch"] = self.block.rope.pitch
        sd["block.crp_mix_alpha"] = self.block.crp_mix_alpha
        sd["block.crp_target_norm"] = self.block.crp_target_norm
        sd["block.layer_pitch_scale"] = self.block.layer_pitch_scale
        sd["block.per_head_pitch"] = self.block.per_head_pitch
        sd["block.notebook_write_w"] = self.block.notebook_write_w
        sd["block.notebook_write_b"] = self.block.notebook_write_b
        sd["block.notebook_read_w"] = self.block.notebook_read_w
        sd["block.notebook_read_b"] = self.block.notebook_read_b
        sd["block.notebook_write_query"] = self.block.notebook_write_query
        sd["block.notebook_rep_write_w"] = self.block.notebook_rep_write_w
        sd["block.notebook_rep_write_b"] = self.block.notebook_rep_write_b
        sd["block.notebook_rep_read_w"] = self.block.notebook_rep_read_w
        sd["block.notebook_rep_read_b"] = self.block.notebook_rep_read_b
        sd["block.notebook_rep_query"] = self.block.notebook_rep_query
        sd["block.bfield_proj_down"] = self.block.bfield_proj_down
        sd["block.bfield_proj_up"] = self.block.bfield_proj_up
        sd["block.bfield_bias"] = self.block.bfield_bias
        sd["block.bfield_alpha"] = self.block.bfield_alpha
        sd["waist_head_w"] = self.waist_head_w
        sd["waist_head_b"] = self.waist_head_b
        sd.update(self.controller.state_dict())
        sd.update(self.confidence_head.state_dict())
        return sd

    def load_state_dict(self, sd_ck: dict, strict: bool = False) -> dict:
        """Load tensors from sd_ck into the model. With strict=False, missing keys
        in sd_ck are skipped (current weights kept) — important for resuming from
        ckpts saved before new components (e.g., lookup_table) existed.

        Returns a dict {missing: [...], unexpected: [...]} for visibility.
        """
        from tinygrad import Device
        targets = self.state_dict()
        missing, unexpected = [], []
        for name, dst in targets.items():
            if name not in sd_ck:
                missing.append(name)
                if strict:
                    raise KeyError(f"missing key in checkpoint: {name}")
                continue
            src = sd_ck[name].to(dst.device).realize()
            # If reshape would fail (different total element count), skip and keep init.
            # Used for resuming when component dims change (e.g., LOOKUP_IB_DIM change).
            src_elems = 1
            for d in src.shape: src_elems *= int(d)
            dst_elems = 1
            for d in dst.shape: dst_elems *= int(d)
            if src_elems != dst_elems:
                missing.append(f"{name} (shape mismatch: ckpt {tuple(src.shape)} vs model {tuple(dst.shape)})")
                continue
            if src.shape != dst.shape: src = src.reshape(dst.shape)
            if src.dtype != dst.dtype: src = src.cast(dst.dtype)
            dst.assign(src).realize()
        for name in sd_ck:
            if name not in targets:
                unexpected.append(name)
        Device[Device.DEFAULT].synchronize()
        return {"missing": missing, "unexpected": unexpected}

    def hidden_states(self, tokens: Tensor, n_loops: int, return_per_loop: bool = False):
        """Forward pass returning hidden states. If return_per_loop, returns a list of
        states after each breath (length n_loops+1, including the embedded input).
        Otherwise returns the final integrated representation (post final LN).
        """
        x = self.embed(tokens).cast(dtypes.half)
        if not return_per_loop:
            x = self.block.breathe(x, n_loops)
            final = _layernorm(x, self.ln_f_g, self.ln_f_b, self.cfg.layer_norm_eps)
            # v28: same prototype retrieval as breathe_with_lookup
            if LOOKUP_VALUE_INJECT:
                scores = self.lookup_table(final).cast(dtypes.float)
                retrieved = self.lookup_table.retrieve(scores).cast(final.dtype)
                final = final + LOOKUP_VALUE_SCALE * retrieved
            return final
        # explicit loop for diagnostics
        states = [x]
        integral = Tensor.zeros_like(x)
        gate_sum = 0.0
        handoff = None
        for l in range(n_loops):
            if CROSS_BREATH_HANDOFF and handoff is not None:
                x = x + handoff
            x = self.block.breathe_once(x, l, temp_mult=_sine_temp_baseline(l, n_loops))
            if CROSS_BREATH_HANDOFF:
                handoff = self.block.compute_handoff(x)
            states.append(x)
            # v39 sin-modulated integral envelope
            if BFIELD_SIN_MOD:
                beta_l = math.sin((l + 0.5) * math.pi / float(n_loops))
            else:
                beta_l = 1.0
            integral = integral + beta_l * x
            gate_sum += beta_l
        final = _layernorm(integral / gate_sum, self.ln_f_g, self.ln_f_b, self.cfg.layer_norm_eps)
        return states, final

    def __call__(self, tokens: Tensor, n_loops: int) -> Tensor:
        return self.hidden_states(tokens, n_loops, return_per_loop=False)

    def forward_with_notebook(self, tokens: Tensor, n_loops: int,
                               initial_notebook: Tensor | None = None,
                               initial_notebook_r: Tensor | None = None):
        """v26 cross-cycle forward: like __call__ but accepts an initial notebook
        state and returns (final_hidden, notebook, notebook_r). Used in
        multi_cycle_train_step to thread the notebook from cycle N to cycle N+1.
        Without notebook threading (when initial_notebook=None), behavior is
        identical to __call__ (notebook starts at zeros).
        """
        x = self.embed(tokens).cast(dtypes.half)
        rep, nb, nb_r = self.block.breathe(x, n_loops,
                                            initial_notebook=initial_notebook,
                                            initial_notebook_r=initial_notebook_r,
                                            return_notebook=True)
        final = _layernorm(rep, self.ln_f_g, self.ln_f_b, self.cfg.layer_norm_eps)
        return final, nb, nb_r

    def breathe_with_lookup_jit(self, tokens: Tensor, n_loops: int):
        """JIT-cached version of breathe_with_lookup, returning only (final_hidden,
        last_breath_match_weights). The per-breath lists aren't returned because
        TinyJit doesn't return Python lists; we keep only what the training step
        actually consumes (the LAST breath's match weights, used by the aux CE).

        Cached per n_loops in self._jit_breathe_forwards. First call at each
        n_loops compiles (~30-60s); subsequent calls replay as a single graph.

        Bypasses the per-step py_overhead growth we observed (870ms → 2274ms in
        100 steps without JIT). The JIT replays a fixed-shape compiled graph,
        so no lazy Python state accumulates between calls.
        """
        n_loops = int(n_loops)
        if not hasattr(self, "_jit_breathe_forwards"):
            self._jit_breathe_forwards = {}
        if n_loops not in self._jit_breathe_forwards:
            n_loops_captured = n_loops

            @TinyJit
            def _fwd(toks: Tensor):
                final, match_weights, _ = self.breathe_with_lookup(toks, n_loops_captured)
                return final, match_weights[-1]

            self._jit_breathe_forwards[n_loops] = _fwd
        return self._jit_breathe_forwards[n_loops](tokens)

    def call_jit(self, tokens: Tensor, n_loops: int) -> Tensor:
        """JIT-cached version of __call__ (the plain forward without lookup table
        per-breath queries). Used for cycles that don't need the aux loss.
        Cached per n_loops in self._jit_forwards."""
        n_loops = int(n_loops)
        if not hasattr(self, "_jit_forwards"):
            self._jit_forwards = {}
        if n_loops not in self._jit_forwards:
            n_loops_captured = n_loops

            @TinyJit
            def _fwd(toks: Tensor):
                return self(toks, n_loops_captured)

            self._jit_forwards[n_loops] = _fwd
        return self._jit_forwards[n_loops](tokens)

    def breathe_controlled(self, tokens: Tensor, max_loops: int, notebook,
                           rep_position: int = -1, detach_rep_for_ctrl: bool = True,
                           detach_decisions_into_transformer: bool = False,
                           adaptive: bool = False, min_loops: int = 1,
                           return_per_breath_reps: bool = False):
        """Closed-loop adaptive breathing — the full 7/7 system in action.

        Per breath:
          1. Run the 4 layer-passes at the current temperature (multiplier from controller).
          2. Add this breath's contribution to the running integral, weighted by gate.
          3. Read the integrated rep at rep_position (the 'controller's eyes').
          4. Run the controller(rep, notebook) → page is appended to notebook,
             attention reads tree of prior pages, decision heads emit
             {temperature, gate, stop_logit} for the NEXT breath.
          5. If adaptive=True and l+1 >= min_loops and mean(stop_logit) > 0, break.

        Adaptive stopping (inference-only by default — training keeps adaptive=False
        so loss computation is straightforward over a fixed unrolled loop):
          - adaptive=True: after each breath's controller call, halt if the batch-mean
            stop_logit crosses zero. Adds one .numpy() sync per breath; cheap at
            inference. Each breath has access to the stop_logit it emitted, so the
            controller can learn (via compute-penalty) to halt early on easy problems.
          - min_loops: guarantees at least this many breaths run before early-stop is
            considered. Default 1 — the controller can't bail at breath 0.

        Gradient separation:
          - detach_rep_for_ctrl=True: the rep fed into the controller is detached,
            so the controller's loss can't update transformer params.
          - detach_decisions_into_transformer=False: the controller's outputs
            (temperature, gate) flow into the transformer's computation WITH
            gradient. This is correct for controller training (controller learns
            from how its decisions affected the transformer's behavior). For
            main-loss training of the transformer, set True so transformer
            gradient doesn't update controller params.

        Returns:
          final_hidden: (B, T, hidden) post final LN — the same surface as __call__
          decisions:    list of dicts (one per breath taken)
          n_breaths:    int — actual number of breaths run (≤ max_loops)
          match_weights: list of (B, T, n_entries) per-breath lookup table queries
        """
        cfg = self.cfg
        x = self.embed(tokens).cast(dtypes.half)
        B = int(x.shape[0])
        notebook.clear()

        integral = Tensor.zeros_like(x)
        gate_total = Tensor.zeros((B,), dtype=dtypes.float).realize()
        decisions_per_breath = []
        match_weights = []
        integrated_per_breath = []   # only populated if return_per_breath_reps

        # Initial decisions (from raw input) — controller's "first look" before any breathing
        rep = x[:, rep_position, :].cast(dtypes.float)
        if detach_rep_for_ctrl:
            rep = rep.detach()
        decisions = self.controller(rep, notebook=notebook)
        decisions_per_breath.append(decisions)

        # Adaptive phase index — accumulated as a float across breaths. RoPE table
        # is integer-indexed, so we round and clamp into [0, max_loops-1] when
        # querying. Uniform default (step_mult=1.0) reproduces the existing
        # 0,1,2,...,max_loops-1 sequence exactly.
        phase_idx_float = 0.0
        actual_n_breaths = max_loops

        for l in range(max_loops):
            temp_mult = decisions["temperature"]                      # (B,)
            gate = decisions["gate"]                                  # (B,)
            step_mult = decisions["step_mult"]                        # (B,)
            if ABLATE_TEMP:
                # Pin to 1.0 but keep gradient graph connected (zero grad flows
                # back through the controller's temperature head). Replacing with
                # Tensor.ones_like severs the path and breaks ctrl_opt.step.
                temp_mult = temp_mult * 0.0 + 1.0
            if ABLATE_STEP_MULT:
                step_mult = step_mult * 0.0 + 1.0
            if ABLATE_GATE:
                gate = gate * 0.0 + 1.0

            # Sine-baseline temperature schedule (diffusion noise-schedule analog).
            # cosine half-period from SINE_TEMP_MAX at l=0 to SINE_TEMP_MIN at
            # l=max_loops-1. Controller's temp_mult multiplies as a perturbation.
            if SINE_TEMP:
                if max_loops > 1:
                    cosine_phase = l * math.pi / (max_loops - 1)
                    sine_baseline = (
                        (SINE_TEMP_MAX + SINE_TEMP_MIN) / 2.0
                        + (SINE_TEMP_MAX - SINE_TEMP_MIN) / 2.0 * math.cos(cosine_phase)
                    )
                else:
                    sine_baseline = (SINE_TEMP_MAX + SINE_TEMP_MIN) / 2.0
                temp_mult = temp_mult * sine_baseline
            if detach_decisions_into_transformer:
                temp_mult = temp_mult.detach()
                gate = gate.detach()
                step_mult = step_mult.detach()

            # Per-batch step_mult is averaged across the batch for the (shared)
            # RoPE phase index. Per-batch fractional indexing would require
            # interpolating cos/sin tables; this is the simpler v1.
            step_avg = float(step_mult.mean().realize().numpy())
            current_loop_idx = max(0, min(int(round(phase_idx_float)), cfg.max_loops - 1))

            # Run the 4-layer breath at this breath's temperature + adaptive phase
            for layer in self.block.layers:
                x = layer(x, current_loop_idx, temp_mult=temp_mult)
            phase_idx_float += step_avg

            # Add to integral with gate weighting.
            # Ablation: when ABLATE_INTEGRATION, overwrite instead of accumulate
            # (last-breath-only — no cross-breath memory in the integral path).
            if ABLATE_INTEGRATION:
                integral = x.cast(dtypes.float) * gate.cast(dtypes.float).reshape(B, 1, 1)
                gate_total = gate.cast(dtypes.float)
            else:
                integral = integral + x.cast(dtypes.float) * gate.cast(dtypes.float).reshape(B, 1, 1)
                gate_total = gate_total + gate.cast(dtypes.float)

            # Per-breath lookup-table query against the running integral, normalized
            running = integral / (gate_total + 1e-6).reshape(B, 1, 1)
            running_normed = _layernorm(running, self.ln_f_g, self.ln_f_b, cfg.layer_norm_eps)
            match_weights.append(self.lookup_table(running_normed))
            if return_per_breath_reps:
                integrated_per_breath.append(running_normed)

            # Controller reads the running integral and emits decisions for next breath.
            # Ablation: when ABLATE_NOTEBOOK, clear notebook before each call so the
            # controller never sees prior-breath pages (no cross-breath memory).
            rep = running_normed[:, rep_position, :].cast(dtypes.float)
            if detach_rep_for_ctrl:
                rep = rep.detach()
            if ABLATE_NOTEBOOK:
                notebook.clear()
            decisions = self.controller(rep, notebook=notebook)
            decisions_per_breath.append(decisions)

            # Adaptive early-stop: after we've run at least min_loops breaths, halt
            # when the controller's batch-mean stop_logit crosses zero. One sync per
            # breath; only enabled at inference.
            if adaptive and (l + 1) >= min_loops:
                stop_mean = float(decisions["stop_logit"].mean().realize().numpy())
                if stop_mean > 0.0:
                    actual_n_breaths = l + 1
                    break

        # Final integrated rep: gate-weighted mean
        final = integral / (gate_total + 1e-6).reshape(B, 1, 1)
        final = _layernorm(final, self.ln_f_g, self.ln_f_b, cfg.layer_norm_eps)
        if return_per_breath_reps:
            return final, decisions_per_breath, actual_n_breaths, match_weights, integrated_per_breath
        return final, decisions_per_breath, actual_n_breaths, match_weights

    def breathe_with_lookup(self, tokens: Tensor, n_loops: int,
                             initial_notebook: Tensor | None = None,
                             initial_notebook_r: Tensor | None = None,
                             return_notebook: bool = False,
                             return_waist_compressed: bool = False):
        """Forward pass returning (final_hidden, per_breath_match_weights, integrated_per_breath).

        Queries the model's lookup table once per breath against the running integral
        normalized to date. Returned shapes:
          final_hidden:                (B, T, hidden)         — same as __call__
          per_breath_match_weights:    list of n_loops × (B, T, n_entries)
          integrated_per_breath:       list of n_loops × (B, T, hidden) post-LN

        Used by the controller to read per-breath operation matches and by training
        to apply auxiliary lookup-CE loss against ground-truth op labels.

        v26 cross-cycle notebook: pass initial_notebook(/_r) to seed from a prior
        cycle's final state. Set return_notebook=True to return the final notebook
        state(s) for threading into the next cycle.
        """
        x_emb = self.embed(tokens).cast(dtypes.half)  # v40: frozen embedding, reused each breath in fresh mode
        x = x_emb
        integral = Tensor.zeros_like(x)
        match_weights = []
        integrated_per_breath = []
        waist_compressed_per_breath = []  # v39: 512d compressed at end-of-breath waist
        handoff = None
        notebook = None
        notebook_r = None
        weight_sum = 0.0  # v39 sin-modulation: accumulated weights for the integral
        if NOTEBOOK_V24:
            B = x.shape[0]
            notebook = initial_notebook if initial_notebook is not None else _initial_notebook_state(B, self.block.nb_dim)
            if NOTEBOOK_DUAL:
                notebook_r = initial_notebook_r if initial_notebook_r is not None else _initial_notebook_state(B, self.block.nb_dim)
        for l in range(n_loops):
            # v40 fresh-input mode: each breath starts from the original embedding
            # plus notebook context. Breaks the rep-flow chain that caused v39's
            # A=8 collapse under enforced bottleneck.
            if BREATHE_FRESH_INPUT:
                x_in = x_emb
            else:
                x_in = x
                if CROSS_BREATH_HANDOFF and handoff is not None:
                    x_in = x_in + handoff
            if NOTEBOOK_V24:
                if NOTEBOOK_ACCUMULATE_ENABLED:
                    read_vec = (notebook @ self.block.notebook_read_w + self.block.notebook_read_b)
                    x_in = x_in + read_vec.reshape(x_in.shape[0], 1, -1).cast(x_in.dtype)
                if NOTEBOOK_DUAL:
                    read_vec_r = (notebook_r @ self.block.notebook_rep_read_w + self.block.notebook_rep_read_b)
                    x_in = x_in + read_vec_r.reshape(x_in.shape[0], 1, -1).cast(x_in.dtype)
            if return_waist_compressed:
                x, waist_compressed = self.block.breathe_once(x_in, l, temp_mult=_sine_temp_baseline(l, n_loops),
                                                                return_waist_compressed=True)
                waist_compressed_per_breath.append(waist_compressed)
            else:
                x = self.block.breathe_once(x_in, l, temp_mult=_sine_temp_baseline(l, n_loops))
            if NOTEBOOK_V24:
                x_f = x.cast(dtypes.float)
                if NOTEBOOK_POOL_MODE == "attn":
                    scores = (x_f * self.block.notebook_write_query.reshape(1, 1, -1)).sum(axis=-1)
                    weights = scores.softmax(axis=-1).reshape(x.shape[0], -1, 1)
                    x_pool = (x_f * weights).sum(axis=1)
                else:
                    x_pool = x_f.mean(axis=1)
                if NOTEBOOK_ACCUMULATE_ENABLED:
                    notebook = notebook + (x_pool @ self.block.notebook_write_w + self.block.notebook_write_b)
                if NOTEBOOK_DUAL:
                    if NOTEBOOK_POOL_MODE == "attn":
                        scores_r = (x_f * self.block.notebook_rep_query.reshape(1, 1, -1)).sum(axis=-1)
                        weights_r = scores_r.softmax(axis=-1).reshape(x.shape[0], -1, 1)
                        x_pool_r = (x_f * weights_r).sum(axis=1)
                    else:
                        x_pool_r = x_pool
                    notebook_r = x_pool_r @ self.block.notebook_rep_write_w + self.block.notebook_rep_write_b
            if CROSS_BREATH_HANDOFF:
                handoff = self.block.compute_handoff(x)
            # v39 sin-modulated integral: bell-shaped weighting that never hits
            # zero at the endpoints (l=0 gives sin(π/(2·n_loops)), peaks at the
            # middle breath). Heartbeat-across-breaths envelope.
            if BFIELD_SIN_MOD:
                beta_l = math.sin((l + 0.5) * math.pi / float(n_loops))
            else:
                beta_l = 1.0
            integral = integral + beta_l * x
            weight_sum = weight_sum + beta_l
            running = integral / weight_sum
            running_normed = _layernorm(running, self.ln_f_g, self.ln_f_b, self.cfg.layer_norm_eps)
            integrated_per_breath.append(running_normed)
            match_weights.append(self.lookup_table(running_normed))
        # Final hidden — bit-for-bit equal to __call__'s output when no sin mod
        final = _layernorm(integral / weight_sum,
                           self.ln_f_g, self.ln_f_b, self.cfg.layer_norm_eps)
        # v28: prototype retrieval. Query lookup at every position, weighted-sum of
        # values, add to final. The model picks up "ideal-rep-for-this-class" guidance.
        if LOOKUP_VALUE_INJECT:
            scores = self.lookup_table(final).cast(dtypes.float)      # (B, T, n_entries)
            retrieved = self.lookup_table.retrieve(scores).cast(final.dtype)  # (B, T, hidden)
            final = final + LOOKUP_VALUE_SCALE * retrieved
        if return_notebook and return_waist_compressed:
            return final, match_weights, integrated_per_breath, notebook, notebook_r, waist_compressed_per_breath
        if return_notebook:
            return final, match_weights, integrated_per_breath, notebook, notebook_r
        if return_waist_compressed:
            return final, match_weights, integrated_per_breath, waist_compressed_per_breath
        return final, match_weights, integrated_per_breath

    def cached_generate_batch(self, batch_prompt_ids: list, n_loops: int, max_new: int,
                               stop_token_ids=None, stop_seq=None,
                               vocab_active: int = 50277,
                               cache_max_len: int | None = None) -> list:
        """Batched cached generation. Processes B prompts in parallel with shared
        K/V cache buffers.

        cache_max_len: dimension of the K/V buffers (defaults to cfg.max_seq_len).
        For short generations (e.g. L3-spaced arithmetic with ~30-token sequences),
        passing a smaller value (e.g. 32) reduces cache size 16× and unlocks much
        larger batch sizes within the GPU's memory budget. Must be >= max_prompt
        + max_new.
        """
        from tinygrad import Tensor as _T
        cfg = self.cfg
        n_layers = cfg.n_phases
        B = len(batch_prompt_ids)
        stop_set = set(stop_token_ids or [])
        seq = list(stop_seq or [])
        seq_len = len(seq)

        real_lens = [len(p) for p in batch_prompt_ids]
        max_prompt = max(real_lens)
        # cache_max_len defaults to model max but can be much smaller for short gens
        if cache_max_len is None:
            cache_max_len = cfg.max_seq_len
        assert max_prompt + max_new <= cache_max_len, (
            f"cache_max_len={cache_max_len} too small for prompt={max_prompt} + new={max_new}"
        )
        assert cache_max_len <= cfg.max_seq_len, "cache_max_len cannot exceed RoPE table size"
        max_len = cache_max_len

        # Right-pad prompts to max_prompt with PAD=0
        padded = [p + [0] * (max_prompt - len(p)) for p in batch_prompt_ids]
        prompts_t = _T(padded, dtype=dtypes.int).realize()

        # Phase A attention mask: 1 for real prompt positions [0..real_len), 0 for padding.
        prompt_attn_mask_phase_a_np = np.zeros((B, max_prompt), dtype=np.int32)
        for b, rl in enumerate(real_lens):
            prompt_attn_mask_phase_a_np[b, :rl] = 1
        attn_mask_phase_a = _T(prompt_attn_mask_phase_a_np, dtype=dtypes.int).realize()

        # gen_mask is unused now (per-batch t_pos handles correctness) but kept for compat
        gen_mask_np = np.ones((B, max_len), dtype=np.int32)
        gen_attn_mask = _T(gen_mask_np, dtype=dtypes.int).contiguous().realize()

        # ---- Stage 1: Phase A breathing on padded prompts ----
        # Pass attn_mask only when there's actual padding (mixed prompt lengths). For
        # uniform-length batches the mask is all-1s and we skip it to keep the code path
        # identical to the uncached forward — easier correctness verification.
        same_len = all(rl == real_lens[0] for rl in real_lens)
        attn_mask_arg = None if same_len else attn_mask_phase_a

        x_emb = self.embed(prompts_t).cast(dtypes.half)  # v40: frozen embedding for fresh-input mode
        x = x_emb
        integral = Tensor.zeros_like(x)
        cache_k = [[None] * n_loops for _ in range(n_layers)]
        cache_v = [[None] * n_loops for _ in range(n_layers)]
        n_phases = cfg.n_phases
        # v23a/v24 eval-time alignment: mirror breathe_once's per-(layer, head) pitch,
        # per-layer temp, per-layer norm, and notebook read/write so eval geometry
        # matches train geometry exactly. Without this, the model trained with these
        # mechanisms would be evaluated under v15-baseline geometry — train/eval mismatch.
        notebook = None
        notebook_r = None
        if NOTEBOOK_V24:
            notebook = _initial_notebook_state(B, self.block.nb_dim)
            if NOTEBOOK_DUAL:
                notebook_r = _initial_notebook_state(B, self.block.nb_dim)
        for loop in range(n_loops):
            # v40 fresh-input: reset to embedding each breath, then add notebook context
            if BREATHE_FRESH_INPUT:
                x = x_emb
            # Notebook read at breath start
            if NOTEBOOK_V24:
                if NOTEBOOK_ACCUMULATE_ENABLED:
                    read_vec = (notebook @ self.block.notebook_read_w + self.block.notebook_read_b)
                    x = x + read_vec.reshape(B, 1, -1).cast(x.dtype)
                if NOTEBOOK_DUAL:
                    read_vec_r = (notebook_r @ self.block.notebook_rep_read_w + self.block.notebook_rep_read_b)
                    x = x + read_vec_r.reshape(B, 1, -1).cast(x.dtype)
            if BREATH_TIME_EMBED:
                x = x + self.block.breath_embed[loop].reshape(1, 1, -1).cast(x.dtype)
            # Base alpha for this breath (same for all layers, mutated by per-head pitch below)
            base_alpha = self.block.rope._alpha_at(loop, x.dtype)
            ac_base, asn_base = base_alpha
            tm_breath = _sine_temp_baseline(loop, n_loops)
            for li, layer in enumerate(self.block.layers):
                # Per-(layer, head) pitch (v23a) — same as breathe_once
                if PER_HEAD_PITCH and li > 0:
                    cos_o = self.block.per_head_pitch_cos[li].cast(x.dtype)
                    sin_o = self.block.per_head_pitch_sin[li].cast(x.dtype)
                    ac_layer = ac_base * cos_o - asn_base * sin_o
                    asn_layer = ac_base * sin_o + asn_base * cos_o
                    layer_alpha = (ac_layer, asn_layer)
                else:
                    layer_alpha = base_alpha
                # Per-layer temp (v24)
                layer_temp = _per_layer_temp_within_breath(li, n_phases) if PER_BREATH_TEMP else tm_breath
                x, (k_part, v_part) = layer.forward_with_kv(x, loop_idx=loop,
                                                             attn_mask=attn_mask_arg,
                                                             temp_mult=layer_temp,
                                                             alpha=layer_alpha)
                pad_n = max_len - max_prompt
                k_full = k_part.pad(((0, 0), (0, 0), (0, pad_n), (0, 0))).contiguous().realize()
                v_full = v_part.pad(((0, 0), (0, 0), (0, pad_n), (0, 0))).contiguous().realize()
                cache_k[li][loop] = k_full
                cache_v[li][loop] = v_full
                # Per-layer norm oscillation (v24)
                if BREATH_NORM_OSC and CONSTANT_RADIUS:
                    scale = _per_layer_norm_scale_within_breath(li, n_phases)
                    x_f = x.cast(dtypes.float)
                    x_norm = (x_f.square().sum(axis=-1, keepdim=True) + 1e-6).sqrt()
                    target = self.block.crp_target_norm * scale
                    mix = self.block.crp_mix_alpha
                    x_proj = x_f * (target / x_norm)
                    x = (x_f * (1.0 - mix) + x_proj * mix).cast(x.dtype)
                # v38 B-field — mid-breath waist after L1 (when not end-of-breath)
                if BFIELD_WAIST > 0 and li == 1 and not BFIELD_END_OF_BREATH:
                    x = self.block.apply_bfield_waist(x)
            # v39 end-of-breath waist
            if BFIELD_WAIST > 0 and BFIELD_END_OF_BREATH:
                x = self.block.apply_bfield_waist(x)
            # End-of-breath CRP — only when per-layer oscillation is OFF
            if CONSTANT_RADIUS and not BREATH_NORM_OSC:
                x_f = x.cast(dtypes.float)
                x_norm = (x_f.square().sum(axis=-1, keepdim=True) + 1e-6).sqrt()
                target = self.block.crp_target_norm
                mix = self.block.crp_mix_alpha
                x_proj = x_f * (target / x_norm)
                x = (x_f * (1.0 - mix) + x_proj * mix).cast(x.dtype)
            # Notebook write at breath end
            if NOTEBOOK_V24:
                x_f = x.cast(dtypes.float)
                if NOTEBOOK_POOL_MODE == "attn":
                    scores = (x_f * self.block.notebook_write_query.reshape(1, 1, -1)).sum(axis=-1)
                    weights = scores.softmax(axis=-1).reshape(B, -1, 1)
                    x_pool = (x_f * weights).sum(axis=1)
                else:
                    x_pool = x_f.mean(axis=1)
                if NOTEBOOK_ACCUMULATE_ENABLED:
                    notebook = notebook + (x_pool @ self.block.notebook_write_w + self.block.notebook_write_b)
                if NOTEBOOK_DUAL:
                    if NOTEBOOK_POOL_MODE == "attn":
                        scores_r = (x_f * self.block.notebook_rep_query.reshape(1, 1, -1)).sum(axis=-1)
                        weights_r = scores_r.softmax(axis=-1).reshape(B, -1, 1)
                        x_pool_r = (x_f * weights_r).sum(axis=1)
                    else:
                        x_pool_r = x_pool
                    notebook_r = x_pool_r @ self.block.notebook_rep_write_w + self.block.notebook_rep_write_b
            # v39 sin-modulated integral envelope (mirrors breathe_with_lookup)
            if BFIELD_SIN_MOD:
                beta_l = math.sin((loop + 0.5) * math.pi / float(n_loops))
            else:
                beta_l = 1.0
            integral = integral + beta_l * x
        if BFIELD_SIN_MOD:
            # Closed-form sum of sin((l+0.5)·π/n) for l=0..n-1; precompute as Python scalar
            _wsum_stage1 = sum(math.sin((l + 0.5) * math.pi / float(n_loops)) for l in range(n_loops))
            integrated_rep = (integral / _wsum_stage1).realize()
        else:
            integrated_rep = (integral / float(n_loops)).realize()

        # First token: per-batch gather at real_lens[i] - 1.
        # Build per-batch index: pos == (real_len - 1)
        pos_arange = Tensor.arange(max_prompt).reshape(1, max_prompt, 1)
        last_idx = (Tensor(real_lens, dtype=dtypes.int).reshape(B, 1, 1) - 1)
        last_mask = (pos_arange == last_idx).cast(dtypes.half)  # (B, max_prompt, 1)
        h_at_last = (integrated_rep * last_mask).sum(axis=1, keepdim=True)  # (B, 1, H)
        h_normed = _layernorm(h_at_last, self.ln_f_g, self.ln_f_b, cfg.layer_norm_eps)
        # v28: prototype retrieval — match h_normed (1, hidden) against lookup keys,
        # weighted sum of values, add to h_normed. Mirrors breathe_with_lookup behavior.
        if LOOKUP_VALUE_INJECT:
            scores = self.lookup_table(h_normed).cast(dtypes.float)  # (B, 1, n_entries)
            retrieved = self.lookup_table.retrieve(scores).cast(h_normed.dtype)  # (B, 1, hidden)
            h_normed = h_normed + LOOKUP_VALUE_SCALE * retrieved
        logits = (h_normed @ self.embed_out).cast(dtypes.float)
        logits = logits[:, :, :vocab_active]
        first_ids = logits.argmax(axis=-1).realize().numpy().reshape(B)
        outs = [[int(first_ids[b])] for b in range(B)]
        is_done = [False] * B
        for b in range(B):
            if outs[b][0] in stop_set:
                is_done[b] = True
            elif seq_len > 0 and outs[b][-seq_len:] == seq:
                is_done[b] = True
        if all(is_done):
            return outs

        # ---- Stage 2: batched per-token generation, JIT-fused ----
        # JIT body: embed(prev_id) → breathing → argmax → next_id. Both embed and
        # argmax inside JIT cuts ~2 kernel launches per step and lets the compiler
        # fuse the embedding lookup with the first matmul + the final logit projection
        # with the argmax reduction.
        #
        # JITs are cached in a dict keyed on (B, n_loops, vocab_active) so a typical
        # eval that sweeps multiple loop counts (EVAL_LOOPS=[1,2,4,8]) compiles each
        # graph once at the first eval and replays them at zero compile cost on every
        # subsequent eval cycle.
        if not hasattr(self, "_cached_batch_jits"):
            self._cached_batch_jits = {}
        jit_key = (B, n_loops, vocab_active)
        if jit_key not in self._cached_batch_jits:
            import time as _t_jit
            _jit_compile_start = _t_jit.perf_counter()
            print(f"[JIT] compile cached_generate_batch: B={B} n_loops={n_loops} vocab={vocab_active}...", flush=True)
            ln_g = self.ln_f_g
            ln_b_t = self.ln_f_b
            embed_out = self.embed_out
            embed_w = self.embed.weight
            layers = self.block.layers
            block_local = self.block
            layer_norm_eps = cfg.layer_norm_eps
            n_loops_local = n_loops
            n_layers_local = n_layers
            B_local = B
            vocab_active_local = vocab_active

            @TinyJit
            def _step(prev_id_t, t_pos_t, notebook_in, notebook_r_in, *kv_flat):
                total = n_layers_local * n_loops_local
                ck = list(kv_flat[:total])
                cv = list(kv_flat[total:])
                # In-graph embedding lookup
                x_new = embed_w[prev_id_t].cast(dtypes.half)
                x = x_new
                integral = Tensor.zeros(B_local, 1, cfg.hidden, dtype=dtypes.half).contiguous()
                new_ck = [None] * total
                new_cv = [None] * total
                notebook = notebook_in
                notebook_r = notebook_r_in
                # v39 sin envelope normalization (closed-form) — Python scalar baked into JIT graph
                if BFIELD_SIN_MOD:
                    _w_sum_local = sum(math.sin((l + 0.5) * math.pi / float(n_loops_local)) for l in range(n_loops_local))
                else:
                    _w_sum_local = float(n_loops_local)
                for loop in range(n_loops_local):
                    # v40 fresh-input: each breath starts from the new token's embedding
                    if BREATHE_FRESH_INPUT:
                        x = x_new
                    # v40 Notebook reads at breath start (mirror Stage 1 / training)
                    if NOTEBOOK_V24:
                        if NOTEBOOK_ACCUMULATE_ENABLED:
                            read_vec = (notebook @ block_local.notebook_read_w + block_local.notebook_read_b)
                            x = x + read_vec.reshape(B_local, 1, -1).cast(x.dtype)
                        if NOTEBOOK_DUAL:
                            read_vec_r = (notebook_r @ block_local.notebook_rep_read_w + block_local.notebook_rep_read_b)
                            x = x + read_vec_r.reshape(B_local, 1, -1).cast(x.dtype)
                    for li in range(n_layers_local):
                        idx = li * n_loops_local + loop
                        x, k_new, v_new = layers[li].forward_cached_step_batched(
                            x, loop, ck[idx], cv[idx], t_pos_t, None  # per-batch t_pos handles masking
                        )
                        new_ck[idx] = k_new
                        new_cv[idx] = v_new
                    # v39 end-of-breath waist
                    if BFIELD_WAIST > 0 and BFIELD_END_OF_BREATH:
                        x = block_local.apply_bfield_waist(x)
                    # v40 Notebook writes at breath end
                    if NOTEBOOK_V24:
                        x_f = x.cast(dtypes.float)
                        if NOTEBOOK_POOL_MODE == "attn":
                            scores = (x_f * block_local.notebook_write_query.reshape(1, 1, -1)).sum(axis=-1)
                            weights = scores.softmax(axis=-1).reshape(B_local, -1, 1)
                            x_pool = (x_f * weights).sum(axis=1)
                        else:
                            x_pool = x_f.mean(axis=1)
                        if NOTEBOOK_ACCUMULATE_ENABLED:
                            notebook = notebook + (x_pool @ block_local.notebook_write_w + block_local.notebook_write_b)
                        if NOTEBOOK_DUAL:
                            if NOTEBOOK_POOL_MODE == "attn":
                                scores_r = (x_f * block_local.notebook_rep_query.reshape(1, 1, -1)).sum(axis=-1)
                                weights_r = scores_r.softmax(axis=-1).reshape(B_local, -1, 1)
                                x_pool_r = (x_f * weights_r).sum(axis=1)
                            else:
                                x_pool_r = x_pool
                            notebook_r = x_pool_r @ block_local.notebook_rep_write_w + block_local.notebook_rep_write_b
                    # Sin-modulated accumulation
                    if BFIELD_SIN_MOD:
                        beta_l = math.sin((loop + 0.5) * math.pi / float(n_loops_local))
                    else:
                        beta_l = 1.0
                    integral = integral + beta_l * x
                integrated = integral / _w_sum_local
                x_normed = _layernorm(integrated, ln_g, ln_b_t, layer_norm_eps)
                logits = x_normed @ embed_out  # (B, 1, vocab) — half is fine for argmax
                logits_active = logits[:, :, :vocab_active_local]
                next_id_t = logits_active.argmax(axis=-1).cast(dtypes.int).realize()  # (B, 1)
                return (next_id_t, notebook, notebook_r, *new_ck, *new_cv)

            self._cached_batch_jits[jit_key] = _step
            print(f"[JIT] cached_generate_batch graph registered for replay "
                  f"(cache size={len(self._cached_batch_jits)}) — first call will compile lazily.", flush=True)

        jit_step = self._cached_batch_jits[jit_key]

        # Per-batch t_pos: each batch item starts at its real prompt's last position + 1.
        # All advance by 1 each step, so t_pos[b] = real_lens[b] + step.
        step = 0
        t_pos_per = [real_lens[b] for b in range(B)]
        t_pos_t = _T(t_pos_per, dtype=dtypes.int).contiguous().realize()
        # Persistent prev_id_t buffer (B, 1) seeded with first generated tokens
        prev_id_t = _T([[outs[b][0]] for b in range(B)], dtype=dtypes.int).contiguous().realize()
        # v40: thread Stage 1's final notebook state into Stage 2.
        # If notebook isn't active, use zero placeholders (JIT graph still expects them).
        if NOTEBOOK_V24 and notebook is not None:
            notebook_state = notebook.contiguous().realize()
        else:
            notebook_state = _initial_notebook_state(B, self.block.nb_dim).contiguous().realize()
        if NOTEBOOK_V24 and NOTEBOOK_DUAL and notebook_r is not None:
            notebook_r_state = notebook_r.contiguous().realize()
        else:
            notebook_r_state = _initial_notebook_state(B, self.block.nb_dim).contiguous().realize()
        packed_kv = (
            [cache_k[li][lp] for li in range(n_layers) for lp in range(n_loops)]
            + [cache_v[li][lp] for li in range(n_layers) for lp in range(n_loops)]
        )

        total = n_layers * n_loops
        for _ in range(max_new - 1):
            # Stop if any batch item has run out of cache slots
            if max(t_pos_per) >= max_len:
                break

            outputs = jit_step(prev_id_t, t_pos_t, notebook_state, notebook_r_state, *packed_kv)
            next_id_t = outputs[0]
            notebook_state = outputs[1].contiguous().realize()
            notebook_r_state = outputs[2].contiguous().realize()
            new_kv = outputs[3:]
            for li in range(n_layers):
                for lp in range(n_loops):
                    idx = li * n_loops + lp
                    cache_k[li][lp] = new_kv[idx]
                    cache_v[li][lp] = new_kv[total + idx]
            packed_kv = (
                [cache_k[li][lp] for li in range(n_layers) for lp in range(n_loops)]
                + [cache_v[li][lp] for li in range(n_layers) for lp in range(n_loops)]
            )

            step += 1
            t_pos_per = [real_lens[b] + step for b in range(B)]
            t_pos_t.assign(_T(t_pos_per, dtype=dtypes.int)).realize()

            # Single sync per step: pull next_ids to CPU for stop check
            next_ids = next_id_t.numpy().reshape(B)
            # Feed back into next step via persistent buffer
            prev_id_t.assign(next_id_t).realize()

            for b in range(B):
                if is_done[b]:
                    continue
                outs[b].append(int(next_ids[b]))
                if int(next_ids[b]) in stop_set:
                    is_done[b] = True
                elif seq_len > 0 and outs[b][-seq_len:] == seq:
                    is_done[b] = True
            if all(is_done):
                break

        return outs

    def cached_generate(self, prompt_ids: list, n_loops: int, max_new: int,
                        stop_token_ids=None, stop_seq=None, vocab_active: int = 50277):
        """Fast cached inference that bit-for-bit matches the uncached path.

        Per-loop, per-layer K/V cache. Each new token does N cached breaths through 4
        layers each (vs uncached: N × 4 × full_seq forward operations per token). The
        integral is recomputed for the new token only by accumulating its per-loop
        outputs.

        Cache: (n_phases × n_loops) × {K, V} fixed-shape buffers of (1, n_heads,
        max_seq_len, head_dim). Total ~32 MB at N=8.
        """
        from tinygrad import Tensor as _T
        cfg = self.cfg
        max_len = cfg.max_seq_len
        n_layers = cfg.n_phases
        stop_set = set(stop_token_ids or [])
        seq = list(stop_seq or [])
        seq_len = len(seq)
        prompt_len = len(prompt_ids)
        assert prompt_len <= max_len, f"prompt {prompt_len} exceeds max_seq_len {max_len}"

        # ---- Stage 1: full breathing on prompt, save K/V at every (layer, loop) ----
        # cache[layer_idx][loop_idx] = (K_buf, V_buf) padded to max_len
        x_emb = self.embed(_T([prompt_ids], dtype=dtypes.int).realize()).cast(dtypes.half)
        x = x_emb
        integral = Tensor.zeros_like(x)
        cache_k = [[None] * n_loops for _ in range(n_layers)]
        cache_v = [[None] * n_loops for _ in range(n_layers)]
        for loop in range(n_loops):
            # v40 fresh-input: each breath starts from the original embedding
            if BREATHE_FRESH_INPUT:
                x = x_emb
            for li, layer in enumerate(self.block.layers):
                x, (k_part, v_part) = layer.forward_with_kv(x, loop_idx=loop)
                pad_n = max_len - int(k_part.shape[2])
                k_full = k_part.pad(((0, 0), (0, 0), (0, pad_n), (0, 0))).contiguous().realize()
                v_full = v_part.pad(((0, 0), (0, 0), (0, pad_n), (0, 0))).contiguous().realize()
                cache_k[li][loop] = k_full
                cache_v[li][loop] = v_full
            # v39 end-of-breath waist
            if BFIELD_WAIST > 0 and BFIELD_END_OF_BREATH:
                x = self.block.apply_bfield_waist(x)
            # v39 sin-modulated integral
            if BFIELD_SIN_MOD:
                beta_l = math.sin((loop + 0.5) * math.pi / float(n_loops))
            else:
                beta_l = 1.0
            integral = integral + beta_l * x
        if BFIELD_SIN_MOD:
            _wsum = sum(math.sin((l + 0.5) * math.pi / float(n_loops)) for l in range(n_loops))
            integrated_rep = (integral / _wsum).realize()
        else:
            integrated_rep = (integral / float(n_loops)).realize()

        # First token: project integrated_rep[:, -1, :] (matches training exactly).
        h_normed = _layernorm(integrated_rep, self.ln_f_g, self.ln_f_b, cfg.layer_norm_eps)
        logits = (h_normed[:, -1:, :] @ self.embed_out).cast(dtypes.float)
        logits = logits[:, :, :vocab_active]
        next_id = int(logits.argmax(axis=-1).realize().numpy()[0, 0])
        out = [next_id]
        if next_id in stop_set:
            return out
        if seq_len > 0 and out[-seq_len:] == seq:
            return out

        # ---- Stage 2: per-token incremental generation, JIT with in-place cache updates ----
        # The 32 cached forwards fuse into ~4 batched kernels via TinyJit. The remaining
        # bottleneck is data movement: returning new cache tensors from the JIT triggers
        # 32 × 2MB AMD<-AMD copies per token. We eliminate these by mutating the cache
        # buffers IN-PLACE via .assign() inside the JIT — only logits is returned.
        if not hasattr(self, "_cached_token_jit") or getattr(self, "_cached_jit_n_loops", None) != n_loops:
            ln_g = self.ln_f_g
            ln_b = self.ln_f_b
            embed_out = self.embed_out
            layers = self.block.layers
            layer_norm_eps = cfg.layer_norm_eps
            n_loops_local = n_loops
            n_layers_local = n_layers

            block_local = self.block
            @TinyJit
            def _token_step(x_new, t_pos_t, *kv_flat):
                total = n_layers_local * n_loops_local
                ck = list(kv_flat[:total])
                cv = list(kv_flat[total:])
                integral = Tensor.zeros(1, 1, cfg.hidden, dtype=dtypes.half).contiguous()
                new_ck = [None] * total
                new_cv = [None] * total
                x = x_new
                # v39 sin envelope normalization
                if BFIELD_SIN_MOD:
                    _w_sum_tok = sum(math.sin((l + 0.5) * math.pi / float(n_loops_local)) for l in range(n_loops_local))
                else:
                    _w_sum_tok = float(n_loops_local)
                for loop in range(n_loops_local):
                    # v40 fresh-input: each breath starts from the new token's embedding
                    if BREATHE_FRESH_INPUT:
                        x = x_new
                    for li in range(n_layers_local):
                        idx = li * n_loops_local + loop
                        x, k_new, v_new = layers[li].forward_cached_step_jit(
                            x, loop, ck[idx], cv[idx], t_pos_t
                        )
                        new_ck[idx] = k_new
                        new_cv[idx] = v_new
                        # v38 B-field — mid-breath waist after L1 (when not end-of-breath)
                        if BFIELD_WAIST > 0 and li == 1 and not BFIELD_END_OF_BREATH:
                            x = block_local.apply_bfield_waist(x)
                    # v39 end-of-breath waist
                    if BFIELD_WAIST > 0 and BFIELD_END_OF_BREATH:
                        x = block_local.apply_bfield_waist(x)
                    # v39 sin-modulated accumulation
                    if BFIELD_SIN_MOD:
                        beta_l = math.sin((loop + 0.5) * math.pi / float(n_loops_local))
                    else:
                        beta_l = 1.0
                    integral = integral + beta_l * x
                integrated = integral / _w_sum_tok
                x_normed = _layernorm(integrated, ln_g, ln_b, layer_norm_eps)
                logits = (x_normed @ embed_out).cast(dtypes.float).realize()
                return (logits, *new_ck, *new_cv)

            self._cached_token_jit = _token_step
            self._cached_jit_n_loops = n_loops

        t_pos = prompt_len
        t_pos_t = Tensor([t_pos], dtype=dtypes.int).contiguous().realize()
        packed_kv = (
            [cache_k[li][lp] for li in range(n_layers) for lp in range(n_loops)]
            + [cache_v[li][lp] for li in range(n_layers) for lp in range(n_loops)]
        )
        total_n = n_layers * n_loops

        for _ in range(max_new - 1):
            if t_pos >= max_len:
                break
            tok_t = _T([[next_id]], dtype=dtypes.int).realize()
            new_emb = self.embed(tok_t).cast(dtypes.half).contiguous().realize()
            outputs = self._cached_token_jit(new_emb, t_pos_t, *packed_kv)
            logits = outputs[0]
            new_kv = outputs[1:]
            for li in range(n_layers):
                for lp in range(n_loops):
                    idx = li * n_loops + lp
                    cache_k[li][lp] = new_kv[idx]
                    cache_v[li][lp] = new_kv[total_n + idx]
            packed_kv = (
                [cache_k[li][lp] for li in range(n_layers) for lp in range(n_loops)]
                + [cache_v[li][lp] for li in range(n_layers) for lp in range(n_loops)]
            )

            t_pos += 1
            t_pos_t.assign(Tensor([t_pos], dtype=dtypes.int)).realize()

            logits_active = logits[:, :, :vocab_active]
            next_id = int(logits_active.argmax(axis=-1).realize().numpy()[0, 0])
            out.append(next_id)
            if next_id in stop_set:
                break
            if seq_len > 0 and len(out) >= seq_len and out[-seq_len:] == seq:
                break
        return out
