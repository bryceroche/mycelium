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

# v77b: orthogonal breath_embed initialization. Zero-init (v9 / v77) leaves the
# breaths nearly indistinguishable after 1500 training steps (typical L2 norms
# 0.08-0.17). At v77 we observed that gradient from breath 0 (Layer 0 verbal
# target) and breath 5 (Layer 5 DAG target) effectively cancel each other —
# different supervision targets pulling on the SAME (near-zero) embedding.
#
# When BREATH_EMBED_ORTHO_INIT > 0.0, init breath_embed as orthogonal vectors
# (one row per breath) at the configured L2 norm. Each breath gets a unique,
# linearly-independent signal at meaningful magnitude — separating the gradients.
#
# Override happens POST-LOAD in scripts/l3_train.py so warm-start ckpts don't
# overwrite the orthogonal init with their tiny trained values.
BREATH_EMBED_ORTHO_INIT  = float(os.environ.get("BREATH_EMBED_ORTHO_INIT", "0.0"))

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

# v83 (2026-05-27) Per-breath VARIED WAIST WIDTH — the B field becomes a precision
# ladder phase-locked to the E field rotation. The fixed BFIELD_WAIST=512 channel
# bandwidth is replaced with a per-breath schedule: at breath b, only the first
# k_b channels of the 512d waist are kept (the rest are zeroed by mask BEFORE
# GELU + codebook injection). The waist masks the wide proj_down/proj_up matrices
# (no new params); shape (K, BFIELD_WAIST). When unset, behavior is byte-identical
# to v82 (mask is all-ones).
#
# Example schedule: "64,256,384,512,512" with K=5 breaths.
#   B0: 64 channels  — narrow bottleneck for skeleton / OPs / op-magnitude only
#   B1: 256 channels — coarse content (types depth-1, args magnitude)
#   B2: 384 channels — refinement (types depth-2)
#   B3-B4: 512 channels — full precision
#
# Combined with V83_ANYTIME_SUPERVISION below, this lets capable students "read
# ahead" — emit full precision early — bounded by the physical channel limit.
BFIELD_WAIST_SCHEDULE    = os.environ.get("BFIELD_WAIST_SCHEDULE", "")


def _parse_bfield_waist_schedule(spec: str, max_value: int) -> list[int] | None:
    """Parse "64,256,384,512,512" into [64, 256, 384, 512, 512]. Returns None when
    spec is empty. Clamps each entry to [1, max_value]."""
    spec = (spec or "").strip()
    if not spec:
        return None
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    out: list[int] = []
    for p in parts:
        v = int(p)
        if v < 1:
            v = 1
        if v > max_value:
            v = max_value
        out.append(v)
    return out

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

# v44 doubled-layers architecture. When 1, instantiates TWO sets of 4 phase-layers
# (layers_a and layers_b). Breaths in [0, max_loops/2) use Set A; breaths in
# [max_loops/2, max_loops) use Set B. With ROPE_FULL_CIRCLE=1 (rotation 0→2π over
# max_loops breaths), this makes Set A active during sin>0 half and Set B during
# sin<0 half — phase-locked E (rotation) ⊥ B (set switch) with zero crossings at
# breaths 0 and max_loops/2. Both sets init from Pythia L0-L3 (identical at step 0,
# gradient differentiates them). Doubles params (~127M → ~254M) but forward compute
# is unchanged (each breath still uses 4 layers, just from a different set).
DOUBLED_LAYERS           = int(os.environ.get("DOUBLED_LAYERS", "0")) > 0

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
# Stochastic depth on breath integral contributions. Per-breath Bernoulli keep:
# integral_contribution = beta_l * keep_l * x where keep_l is 1/(1-p) when kept,
# 0 when dropped. Preserves E[integral] (ResNet-style scaling), so output has the
# right expectation. l3_train.py writes a new (max_loops,) keep-mask each training
# step via assign() on BreathingBlock.stoch_keep_mask (preserves JIT graph identity).
# Inference (Tensor.training=False) path skips the mask read entirely.
STOCH_DEPTH_P            = float(os.environ.get("STOCH_DEPTH_P", "0.0"))
# Whether to read/write the notebook inside cached_generate_batch's Stage 2 (the
# per-token autoregressive decode JIT'd _step). Training has no autoregressive
# decode — training updates notebook only during the breath loop on the prompt
# (Stage 1 equivalent). v40+ added notebook reads/writes to Stage 2 to "mirror
# training," but for any model trained where Stage 2 was decode-only (v24c and
# earlier), this is severe train/eval mismatch: ~30 generated tokens × 8 breaths
# = 240 extra notebook updates the model has never seen. Manifests as low
# teacher-forced val loss + 0% generation accuracy + garbage output.
# Default 0 (Stage 2 decode does NOT touch the notebook) restores v24c-compatible
# behavior. Set to 1 only for models explicitly trained with this mode.
STAGE2_NOTEBOOK          = int(os.environ.get("STAGE2_NOTEBOOK", "0")) > 0
# v61 DAG notebook env vars
NOTEBOOK_DAG             = int(os.environ.get("NOTEBOOK_DAG", "0")) > 0
NOTEBOOK_DAG_N_HEADS     = int(os.environ.get("NOTEBOOK_DAG_N_HEADS", "4"))
NOTEBOOK_DAG_POS_EMBED   = int(os.environ.get("NOTEBOOK_DAG_POS_EMBED", "1")) > 0
# v68 (2026-05-22) TWO_PHASE breath: replace symmetric 4-layer breath with asymmetric
# EXPAND (4 layers, warm) + COMPRESS (2 layers, cool) structure. Embodies inhale-exhale
# rhythm architecturally. Each phase has its own SharedWeights (V/O/FFN-out/LNs).
# Temperature is structural (not scheduled): EXPAND=2.0, COMPRESS=0.7.
TWO_PHASE                = int(os.environ.get("TWO_PHASE", "0")) > 0
EXPAND_LAYERS            = int(os.environ.get("EXPAND_LAYERS", "4"))
COMPRESS_LAYERS          = int(os.environ.get("COMPRESS_LAYERS", "2"))
EXPAND_TEMP              = float(os.environ.get("EXPAND_TEMP", "2.0"))
COMPRESS_TEMP            = float(os.environ.get("COMPRESS_TEMP", "0.7"))

# v69 (2026-05-23) COLLAPSE — JPEG/MP3-inspired lossy compression at the waist.
# Replaces v66's B-field MLP waist with: codebook match (transform) → gate (quantize)
# → 128d projection (encode) → residual block. Each step decoded by the WaistController
# provides dense per-breath gradient back through the whole compression pipeline.
#
# Diagnostic 2026-05-22 night: math problems in Pythia's embedding space live on a
# ~128d submanifold (96.2% energy in K=128 at the layer-0 INPUT). Waist=128 matches
# the actual signal dimensionality. Independent SharedWeights don't help (refuted by
# per-layer rank diagnostic). Single SharedWeights, single residual block.
COLLAPSE_V69             = int(os.environ.get("COLLAPSE_V69", "0")) > 0
COLLAPSE_WAIST_DIM       = int(os.environ.get("COLLAPSE_WAIST_DIM", "128"))
COLLAPSE_CODEBOOK_N      = int(os.environ.get("COLLAPSE_CODEBOOK_N", "256"))
COLLAPSE_TAU             = float(os.environ.get("COLLAPSE_TAU", "1.0"))
COLLAPSE_GATE_BIAS       = float(os.environ.get("COLLAPSE_GATE_BIAS", "2.0"))
COLLAPSE_ENTROPY_REG     = float(os.environ.get("COLLAPSE_ENTROPY_REG", "0.01"))

# v70 (2026-05-23) COLLAPSE refined — fixes v69's K≥4 regression.
# Diagnostic 2026-05-23: v66 waist-position rank_95 trajectory is 263→192→157→135→118→103
# across K=6 breaths. v69's 128d was below natural rank → 65% K=6 retention → catastrophic.
# v70 design:
#   - Fixed 512d waist (matches v66, JIT-safe)
#   - Codebook 256 entries, conditioned gate emits importance (B, T, 512) per dim
#   - Gate input: (prototype || breath_embed[breath_idx])  → learns breath-dependent compression
#   - Budget-violation sparsity penalty: target_frac = 0.8 − 0.1·breath_idx
#     (lagging-safe: only penalizes if importance.mean() > target; never punishes under-budget)
#   - α=0 init, copy v66's bfield_proj_down/up/bias for byte-identical warm-start
COLLAPSE_V70             = int(os.environ.get("COLLAPSE_V70", "0")) > 0
COLLAPSE_V70_WAIST_DIM   = int(os.environ.get("COLLAPSE_V70_WAIST_DIM", "512"))
COLLAPSE_V70_CODEBOOK_N  = int(os.environ.get("COLLAPSE_V70_CODEBOOK_N", "256"))
COLLAPSE_V70_TAU         = float(os.environ.get("COLLAPSE_V70_TAU", "0.5"))
COLLAPSE_V70_BREATH_DIM  = int(os.environ.get("COLLAPSE_V70_BREATH_DIM", "64"))
COLLAPSE_V70_BUDGET_START = float(os.environ.get("COLLAPSE_V70_BUDGET_START", "0.80"))
COLLAPSE_V70_BUDGET_DECAY = float(os.environ.get("COLLAPSE_V70_BUDGET_DECAY", "0.10"))
COLLAPSE_V70_BUDGET_MIN  = float(os.environ.get("COLLAPSE_V70_BUDGET_MIN", "0.10"))
COLLAPSE_V70_SPARSITY_WEIGHT = float(os.environ.get("COLLAPSE_V70_SPARSITY_WEIGHT", "0.1"))
COLLAPSE_V70_GATE_BIAS   = float(os.environ.get("COLLAPSE_V70_GATE_BIAS", "4.6"))  # sigmoid(4.6)≈0.99

# v71 (2026-05-23) COLLAPSE — fixes v70's three failure modes:
#   1. SPARSITY_WEIGHT 0.1 → 1.0 (10× stronger; gate now feels pressure to close).
#      At v70's W=0.1: sparsity loss was (0.19²)·0.1 ≈ 0.0036/breath ≈ 0.01 total
#      vs CE ≈ 1.75 (0.6% of loss). At W=1.0: ~10% of CE (comparable; gradient real).
#   2. GATE_BIAS 4.6 → 1.0 (sigmoid(1)=0.73 vs sigmoid(4.6)=0.99). Gate starts
#      closer to budget target (0.80 at breath 0); needs less travel; avoids sigmoid
#      saturation at init (gradient near-zero in the 0.99 regime).
#   3. K-means codebook init (vs v70's randn × 0.02). Random codebook with τ=0.5
#      still produces near-uniform softmax for 512d random vectors → no symmetry
#      breaking at init. K-means centers from real v66 waist data → differentiated
#      entries from step 0.
#
# Controller-input shift fix (architectural):
#   v70's waist_compressed = (residual × importance) + prototype  → controller sees
#   compressed_x · importance · 1 + prototype · (1 − importance)
#   At init importance≈0.99 → ≈ compressed_x + 0.01·prototype (small but non-zero;
#   as codebook drifts the controller's input shifts away from v66's known-good shape).
#
#   v71's waist_compressed = compressed_x × importance (NO prototype add-back to
#   controller's read). At init this is exactly importance(=0.73 with GATE_BIAS=1.0)
#   × compressed_x — a uniform scaling of v66's signal, no codebook-dependent shift.
#   The codebook contribution still flows into the decompression path:
#       decompressed = (waist_compressed + prototype) @ proj_up + bias
#   so values learn additive content; importance learns where to keep signal vs
#   where to drop it. The two pathways are decoupled at the controller's read.
COLLAPSE_V71             = int(os.environ.get("COLLAPSE_V71", "0")) > 0
COLLAPSE_V71_WAIST_DIM   = int(os.environ.get("COLLAPSE_V71_WAIST_DIM", "512"))
COLLAPSE_V71_CODEBOOK_N  = int(os.environ.get("COLLAPSE_V71_CODEBOOK_N", "256"))
COLLAPSE_V71_TAU         = float(os.environ.get("COLLAPSE_V71_TAU", "0.5"))
COLLAPSE_V71_BREATH_DIM  = int(os.environ.get("COLLAPSE_V71_BREATH_DIM", "64"))
COLLAPSE_V71_BUDGET_START = float(os.environ.get("COLLAPSE_V71_BUDGET_START", "0.80"))
COLLAPSE_V71_BUDGET_DECAY = float(os.environ.get("COLLAPSE_V71_BUDGET_DECAY", "0.10"))
COLLAPSE_V71_BUDGET_MIN  = float(os.environ.get("COLLAPSE_V71_BUDGET_MIN", "0.10"))
COLLAPSE_V71_SPARSITY_WEIGHT = float(os.environ.get("COLLAPSE_V71_SPARSITY_WEIGHT", "1.0"))  # 10× v70
COLLAPSE_V71_GATE_BIAS   = float(os.environ.get("COLLAPSE_V71_GATE_BIAS", "1.0"))  # sigmoid(1)≈0.73
COLLAPSE_V71_KMEANS_INIT_PATH = os.environ.get("COLLAPSE_V71_KMEANS_INIT_PATH", "")

# v65 per-breath prompt refresh: x_in = prev_breath_output + α × original_prompt_emb.
# Skip connection from raw embeddings to every breath. Diagnosed root cause: the
# 512d waist compression destroys entity identity (rename diagnostic 0/20 grounded).
# The refresh carries identity through the lossy bottleneck while waist carries reasoning.
# α=0.0 disables. α=0.1 is the principled starting point per the design rationale.
PROMPT_REFRESH_ALPHA     = float(os.environ.get("PROMPT_REFRESH_ALPHA", "0.0"))
# v65 boundary auxiliary loss: at each step-k position, predict "is next token ####?"
# via a small head on the breath's hidden state. Supervised by gold positions.
# Forces the model to learn segment timing explicitly. Diagnosed root cause: model
# emits ~2 segments regardless of K=2..6 (76% segment-shortfall).
BOUNDARY_AUX_WEIGHT      = float(os.environ.get("BOUNDARY_AUX_WEIGHT", "0.0"))
# Hybrid-heads quadrature (v46). When 1, the second half of heads in each layer
# (heads n_heads/2 .. n_heads-1) get an additional π/2 phase offset on top of
# the standard PER_HEAD_PITCH offset. Provides quadrature pairs: when "phase 0
# heads" are at rotation angle θ, "phase 1 heads" are at θ + π/2. Combined view
# has constant-magnitude rotating field (circular polarization analogy) instead
# of linear oscillation that goes through zero-crossings. Cheapest test of the
# photon analogy — zero added params and zero added compute. Default 0 preserves
# v23a behavior (all heads in a layer share the same offset).
QUADRATURE_HEADS         = int(os.environ.get("QUADRATURE_HEADS", "0")) > 0
# Quadrature ramp — gradually increase the second-half head offset from 0 to π/2
# over the first QUADRATURE_RAMP_STEPS training steps so the model's W_O can adapt
# as the geometry diverges. v46 take 1 confirmed that applying full π/2 to a v45
# warm-start causes -35 to -41 point collapse (model's learned head combination
# weights expect all-same-offset heads). 0 = no ramp (full π/2 from step 0).
QUADRATURE_RAMP_STEPS    = int(os.environ.get("QUADRATURE_RAMP_STEPS", "0"))
# Across-layer quadrature (v47): when > 0, the per-layer offset step changes
# from the v23a default (pitch_range/(n_phases*n_heads) ≈ π/64) to the given
# target. With target=π/2 and n_phases=4, layers sit at {0, π/2, π, 3π/2} —
# four-corner quadrature spread across the 4 breath layers, all heads within
# a layer keeping the SAME offset. Distinct from within-layer (per-head) split
# in QUADRATURE_HEADS which collapsed v46 take 1 because W_O wasn't trained to
# handle mismatched heads. The phase-shift sweep on v46b step 750 showed
# uniform offsets up to π/4 cost <3 points — across-layer keeps that property
# per-layer-locally while varying the *layer-to-layer* phase difference.
ACROSS_LAYER_PITCH_TARGET = float(os.environ.get("ACROSS_LAYER_PITCH_TARGET", "0.0"))
# Ramp from the v23a base step to ACROSS_LAYER_PITCH_TARGET over this many
# steps, so warm-start models can adapt as layer-to-layer phase difference
# grows. 0 = no ramp (init directly at target — cold-start mode).
ACROSS_LAYER_PITCH_RAMP_STEPS = int(os.environ.get("ACROSS_LAYER_PITCH_RAMP_STEPS", "0"))
# Per-layer offset override: comma-separated list of n_phases radian values.
# When set (non-empty), overrides BOTH the v23a default and any ACROSS_LAYER_PITCH
# config — these specific offsets are used for the layers (all heads in a layer
# share the same offset). Validated 2026-05-18 by per-layer offset sweep that
# alternating/triangle/symmetric patterns within tolerance preserve accuracy.
# Example: "0,0.1963,0.3927,0.1963" = triangle-small (0, π/16, π/8, π/16).
PER_LAYER_OFFSETS_RADIANS = os.environ.get("PER_LAYER_OFFSETS_RADIANS", "")
# v50 learnable codebook at the IB waist (item #1/#4 from laundry list).
# When WAIST_CODEBOOK_N > 0, allocates N learnable keys + values at the
# B-field waist (dim = max(1, BFIELD_WAIST)). After compression to bf_w, the
# compressed state queries the codebook via dot-product attention, retrieves
# a weighted-sum of values, adds (scaled by WAIST_CODEBOOK_INJECT_WEIGHT)
# to the compressed state before GELU. Values init at zero so the
# contribution is identity at step 0 (graceful warm-start); gradient + the
# main CE shape both keys and values during training. This gives the model
# a "discrete operation library" at the bottleneck — instead of compressing
# to a continuous 256d, it commits toward one of N attractor entries.
WAIST_CODEBOOK_N             = int(os.environ.get("WAIST_CODEBOOK_N", "0"))
WAIST_CODEBOOK_INJECT_WEIGHT = float(os.environ.get("WAIST_CODEBOOK_INJECT_WEIGHT", "1.0"))
# v52 Stage 1 — per-breath decode supervision. When PER_BREATH_DECODE=1, the
# training step reads each breath's end-of-breath output, decodes via ln_f +
# embed_out, and supervises against the gen_target tokens for THAT step. This
# enforces "the waist commits to a partial-answer state decodable at each
# breath" — the depth-helps signal becomes architecturally required.
# Companion to BFIELD_WAIST > 0 (waist active) and BFIELD_END_OF_BREATH=1
# (waist runs at the END of each breath, just before the per-breath output
# is captured for supervision).
PER_BREATH_DECODE = int(os.environ.get("PER_BREATH_DECODE", "0")) > 0
# v54 (2026-05-19) Phase 1 — Controller as supervision conduit. When 1, the
# WaistController fires once per breath, reading the compressed waist (512d)
# and cross-attending over the prompt embeddings, outputting partial-answer
# logits via the SAME embed_out as the main model (tied). Per-breath CE
# supervises the controller's predictions on step k's gen_target. Gradient
# flows back through the controller → into the waist → shapes the main
# model's rep space to encode info the controller can use.
#
# Why a controller: forces the waist to be USEFUL. The controller can't
# predict the partial answer without informative waist content. The "thinking
# in rep space, decode only at end" objective gets implicit supervision via
# the controller's text predictions.
CONTROLLER_DECODE = int(os.environ.get("CONTROLLER_DECODE", "0")) > 0
# Controller depth (number of cross-attn layers). 1-2 typical; v78b uses 4 for
# extra operand-binding capacity.
CONTROLLER_N_LAYERS = int(os.environ.get("CONTROLLER_N_LAYERS", "1"))
if CONTROLLER_N_LAYERS != 1:
    print(f"[CONTROLLER_N_LAYERS] {CONTROLLER_N_LAYERS}", flush=True)
# v72 Pointer-network copy mechanism at the WaistController. Addresses the
# entity-tracking bottleneck (rename diagnostic: 0/20 grounded on v66 because
# waist compression destroys entity identity). When WAIST_COPY=1, the
# controller produces an additional "copy distribution" over prompt positions,
# combined with the vocab softmax via a sigmoid gate. The model can POINT to a
# prompt position and copy that token directly. State-dict-stable: params are
# always allocated; gradient is inert when WAIST_COPY=0 (forward path skips).
WAIST_COPY        = int(os.environ.get("WAIST_COPY", "0")) > 0
# Bias for gate at init. sigmoid(-2.0) = 0.12 — copy starts SUPPRESSED, not 50/50.
# Rationale: with random-init copy attention, p_copy is noise. At gate=0.5, noise
# corrupts 50% of every prediction and CE pushes gate NEGATIVE (kills copy before
# it can learn). At gate=0.12, only 12% noise contamination — model mostly generates
# (matching v66 behavior), copy attention learns quietly in the background, gate
# pulls UP on entity tokens once copy attention has learned where to point.
WAIST_COPY_GATE_BIAS_INIT = float(os.environ.get("WAIST_COPY_GATE_BIAS_INIT", "-2.0"))
# Aux loss directly supervising copy attention. When y_target appears in the prompt at
# position i, copy_attn[t, i] should be high. Without this, the copy mechanism never
# bootstraps — copy_attn stays random, p_copy stays low, gate stays closed (the
# v69/v70 dead-codebook pattern). The aux loss is self-limiting: once copy_attn learns
# where to point, CE makes p_copy[y] high → gate naturally opens → aux contribution
# fades to near-zero. Default 1.0 — comparable to main CE, dominant during warm-up.
WAIST_COPY_AUX_WEIGHT = float(os.environ.get("WAIST_COPY_AUX_WEIGHT", "1.0"))
# Hidden dim for the copy Q/K projections. Small (128) — cheap, low-rank
# attention space dedicated to "does this prompt token match what I want to
# emit next?". Independent of the controller's H_ctrl.
WAIST_COPY_HIDDEN = int(os.environ.get("WAIST_COPY_HIDDEN", "128"))
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
# v75 (2026-05-23) Diffusion paradigm — bounded per-breath rep delta.
# When MAX_STEP_SIZE > 0, the per-token L2 norm of (breathe_once output - input)
# is capped at MAX_STEP_SIZE. Forces each breath to be a gradual refinement
# rather than a giant leap; total transformation = integral of K small steps
# (analogous to diffusion's β_t schedule). 0.0 = disabled (v66 behavior).
# Companion to PER_BREATH_FULL_ANSWER and NOTEBOOK_NO_DETACH (both in l3_training.py).
MAX_STEP_SIZE            = float(os.environ.get("MAX_STEP_SIZE", "0.0"))
# v75 (2026-05-24) Half-cosine step-size schedule across K breaths.
# When MAX_STEP_BASE > 0 (priority over MAX_STEP_SIZE), per-breath bound is:
#   max_step_k = MAX_STEP_MIN + (MAX_STEP_BASE - MAX_STEP_MIN) * cos(π/2 * k / (K-1))
# breath 0 = MAX_STEP_BASE (wide exploration, basin landing);
# breath K-1 = MAX_STEP_MIN (tight commit, refinement). Matches diffusion's β_t
# coarse-to-fine schedule. K=1 edge case: returns MAX_STEP_BASE.
MAX_STEP_BASE            = float(os.environ.get("MAX_STEP_BASE", "0.0"))
MAX_STEP_MIN             = float(os.environ.get("MAX_STEP_MIN",  "0.1"))
# v75 Gradient flow through the notebook. The standard NOTEBOOK_V24 path does
# NOT detach the notebook write — gradient already flows breath-to-breath via
# `notebook = notebook + (x_pool @ W + b)`. This env var is a no-op for current
# v66 architecture (kept for v75-aggressive variants that might re-introduce a
# detach to remove). Default 0 preserves current behavior.
NOTEBOOK_NO_DETACH       = int(os.environ.get("NOTEBOOK_NO_DETACH", "0")) > 0

# v78 (2026-05-24) Per-head model codebook. Each of the n_heads attention heads
# gets its own copy of an N-cell codebook (N_HEAD_CELLS, head_dim). When
# V78_HEAD_CODEBOOK=1, each head's K and V are extended with the head's codebook
# entries (concat along the sequence axis), so the head attends to BOTH sequence
# positions and codebook cells. Init: ONE base codebook (randn × 0.02) replicated
# across heads — shared init, independent training (each head's slice evolves
# independently). Storage is always allocated for state-dict symmetry; gradient
# inert when V78_HEAD_CODEBOOK=0. Default N_HEAD_CELLS=12 (4 ops × 3 dag step
# types) — conceptual basis; cells learn their own meaning end-to-end.
V78_HEAD_CODEBOOK        = int(os.environ.get("V78_HEAD_CODEBOOK", "0")) > 0
V78_HEAD_CODEBOOK_N      = int(os.environ.get("V78_HEAD_CODEBOOK_N", "12"))

# v78b (2026-05-25) Attention supervision. When WAIST_ATTN_SUPERVISION=1, the
# WaistController stashes the post-softmax attention weights of its LAST
# cross-attn layer (mean over heads). The trainer reads these via
# `model.waist_controller._last_cross_attn` and supervises them to peak at
# matching digit positions in the prompt via the WAIST_ATTN_AUX_WEIGHT-scaled
# aux loss. v77 diagnosed wrong operands ("x0 = 2 + 2" vs "x0 = 2 + 1") — the
# cross-attention was available but not directed; this gives it a direction.
# Default 0 preserves v66-v78 behavior; the stash slot is None when off.
WAIST_ATTN_SUPERVISION   = int(os.environ.get("WAIST_ATTN_SUPERVISION", "0")) > 0
if WAIST_ATTN_SUPERVISION:
    _WAIST_ATTN_AUX_WEIGHT_LOG = float(os.environ.get("WAIST_ATTN_AUX_WEIGHT", "0.5"))
    print(f"[WAIST_ATTN_SUPERVISION] active, weight={_WAIST_ATTN_AUX_WEIGHT_LOG}", flush=True)

# v79 (2026-05-25) Causal masks during TRAINING to plug lookahead leaks.
# Two leaks fixed by these masks:
#   1. WaistController cross-attn KV is `embed(tokens)`, which during training
#      contains the gold answer tokens at positions >= prompt_len. With no mask,
#      cross-attn at any decode position can see them — the model learns to
#      "cheat" by attending to the gold span. At eval those positions are zero
#      → degenerate output.
#   2. notebook_write_query attention-pools over the WHOLE sequence; post-breath
#      x at gold-answer positions carries gradient information about the gold.
#      Notebook reads that → leaks gold across breaths.
#
# When V79_CAUSAL_MASKS=1, the trainer builds a per-example kv_mask that is 1.0
# at positions [0, prompt_len) and 0.0 elsewhere, then passes it to both:
#   - WaistController.forward(..., kv_mask=kv_mask)
#   - breathe_with_lookup(..., notebook_pool_mask=kv_mask)  (added in v79)
# Eval-time kv_mask use (eval_v77_dag.py) was added in v78c; v79 extends the
# same idea to training so train and eval geometries match.
V79_CAUSAL_MASKS         = int(os.environ.get("V79_CAUSAL_MASKS", "0")) > 0
if V79_CAUSAL_MASKS:
    print(f"[CAUSAL_MASK] cross-attn + notebook masked to prompt range during training", flush=True)


# v81 (2026-05-26) Multi-head WaistController + main-attn answer-span masking.
#
# Two coupled changes:
#   1. MULTI_HEAD_WAIST=1: WaistController emits FOUR parallel logit heads
#      (ops / types / args1 / args2), one per list in v81's "4-list separated by |"
#      training targets. All heads share the cross-attn backbone; each has its own
#      pre-projection MLP before the shared embed_out vocab projection.
#   2. V81_MAIN_ATTN_MASK=1: thread kv_mask as `main_attn_mask` into
#      `breathe_with_lookup`. The mask blocks main-self-attention keys at
#      answer-span positions AND zeros the input embeddings there. Combined with
#      V79's notebook + cross-attn masks, this makes per-position predictions
#      strictly prompt-conditional (no teacher-forcing leak). Audited via
#      scripts/diag_v81_masking_audit.py — MUST pass before training.
MULTI_HEAD_WAIST         = int(os.environ.get("MULTI_HEAD_WAIST", "0")) > 0
V81_MAIN_ATTN_MASK       = int(os.environ.get("V81_MAIN_ATTN_MASK", "0")) > 0
if MULTI_HEAD_WAIST:
    print(f"[MULTI_HEAD_WAIST] WaistController emits 4 parallel heads (ops/types/args1/args2)", flush=True)
if V81_MAIN_ATTN_MASK:
    print(f"[V81_MAIN_ATTN_MASK] main-attn answer-span masking active", flush=True)


# v85 (2026-05-27) Differentiable queryable structures.
#
# Replaces v82-v84's text-based DAG supervision with STRUCTURED slot supervision.
# Numbers/verbs are queryable differentiable tensors. The model binds DAG args to
# actual prompt-number positions via pointer attention.
#
# Architecture per breath:
#   - K_max=10 DAG slots, each with: ops (4), types (32), 2 args, is_active (1)
#   - Each slot is a query (slot_pos_embed[k] + waist_compressed_pooled)
#   - Per-slot heads: ops_logits, types_logits, args_logits[2] over (N_max + K_max),
#     is_active_logits (sigmoid)
#   - All breaths emit SAME structure (no per-breath specialization)
#   - Slots fill in PARALLEL per breath (no AR)
#
# Codebooks (learnable):
#   - ops_codebook (4, h)        — small random init
#   - types_codebook (32, h)     — init from .cache/ib_centroids.npz when available
#
# When V85_QUERYABLE=1, the slot decoder fires at EVERY breath. Each breath gets
# per-slot CE supervision on (ops, types, args1, args2, is_active). This replaces
# the v82/v84 per-breath full-sequence CE.
V85_QUERYABLE        = int(os.environ.get("V85_QUERYABLE", "0")) > 0
V85_K_MAX            = int(os.environ.get("V85_K_MAX", "10"))
V85_N_MAX            = int(os.environ.get("V85_N_MAX", "20"))   # max prompt numbers
V85_TYPES_N          = int(os.environ.get("V85_TYPES_N", "32"))
V85_OPS_N            = 4
if V85_QUERYABLE:
    print(f"[V85_QUERYABLE] differentiable queryable structures — K_max={V85_K_MAX} N_max={V85_N_MAX} types_N={V85_TYPES_N}", flush=True)

# v86 (2026-05-27) Targeted fixes to v85's args-binding bottleneck.
#
# v85 args heads read a mean-pooled waist → broadcast same context to all slots
# → loses positional info. v86 makes each slot cross-attend over the FULL waist
# sequence to compute its own positional context, then uses that context for
# the pointer logits. The per-slot signal comes from `slot_pos_embed` which
# already differentiates the slot queries (existing v85 piece).
#
# v86 also adds a positive-class weight to the active head BCE so slots are
# predicted active more often (v85 was too conservative — most problems
# decoded to empty DAGs).
V86_ARGS_CROSS_ATTN  = int(os.environ.get("V86_ARGS_CROSS_ATTN", "0")) > 0
V86_ACTIVE_POS_WEIGHT = float(os.environ.get("V86_ACTIVE_POS_WEIGHT", "5.0"))
if V86_ARGS_CROSS_ATTN:
    print(f"[V86_ARGS_CROSS_ATTN] per-slot cross-attn over full waist sequence for args binding", flush=True)
if V85_QUERYABLE:
    print(f"[V86_ACTIVE_POS_WEIGHT] active-head BCE pos_weight={V86_ACTIVE_POS_WEIGHT}", flush=True)

# v95 (2026-05-28) Operand-position attention supervision for the AR v80 paradigm.
#
# v80_prod_step400 ceiling: 78% DAG parse / 1.7% accuracy. Two failure modes
# diagnosed:
#   (a) wrong-operand binding (model emits SOME prompt number but the wrong one)
#   (b) undefined variable references (emits x3 when only x0 defined)
#
# v95 attacks (a): for each AR-output position that emits a number-token in the
# gold L6 (e.g. `Ġ50`, `Ġ12`, `Ġ60`), force the WaistController's cross-attention
# at THAT output position to peak at the prompt position where that number was
# originally mentioned (as a digit-spaced sequence: `Ġ5 Ġ0`).
#
# This is a complement to (not replacement for) the existing WAIST_ATTN_SUPERVISION
# which supervises single-digit-token positions. v95 covers WHOLE-NUMBER tokens —
# the dominant content tokens in the L6 DAG. Targets are MERGED into the same
# attn_target_t / attn_mask_t tensors (no new JIT inputs needed; same stash).
#
# Default 0 preserves v80 behavior. When V95_OPERAND_AUX=1 AND v80 data is
# loaded, the data-annotation step extends attn_target/attn_mask to cover
# whole-number-token output positions.
V95_OPERAND_AUX        = int(os.environ.get("V95_OPERAND_AUX", "0")) > 0
V95_OPERAND_AUX_WEIGHT = float(os.environ.get("V95_OPERAND_AUX_WEIGHT", "0.5"))
if V95_OPERAND_AUX:
    print(f"[V95_OPERAND_AUX] operand-position attention supervision weight={V95_OPERAND_AUX_WEIGHT}", flush=True)

# v87 (2026-05-27) Slot-symmetry fix.
#
# v86 hit 100% parse but 0% accuracy because ALL slots attend to the same prompt
# position (JSD 0.001 across slots vs ceiling 0.69). Root cause: per-slot
# positional embeddings exist but are too weak (slot_pos_embed at 0.02 scale,
# v86_args_slot_pos zero-init) to break the GELU-pooled slot_query symmetry.
#
# v87 fix: init the per-slot positional embeddings at a meaningful scale so
# slots START differentiated. Both `slot_pos_embed` and `v86_args_slot_pos` use
# this scale. Uniform(-scale, scale) — same family as the original 0.02 init.
#
# When warm-starting from a v86 ckpt whose saved slot_pos_embed has the small
# v85/v86 values, the trainer can reinitialize the params AFTER load_state_dict
# by setting V87_REINIT_SLOT_POS=1.
V87_SLOT_POS_INIT_SCALE = float(os.environ.get("V87_SLOT_POS_INIT_SCALE", "0.0"))
if V87_SLOT_POS_INIT_SCALE > 0.0:
    print(f"[V87_SLOT_POS_INIT_SCALE] slot_pos_embed + v86_args_slot_pos init scale {V87_SLOT_POS_INIT_SCALE}", flush=True)

# v89 (2026-05-27) Supervised attention for args binding.
#
# v88 fixed the K/V projection collapse (slots now produce diverse attention,
# pairwise JSD lifted 8×) but accuracy stayed at 0% because the args POINTER
# projections collapsed to a degenerate "slot k picks index 20+k" pattern (the
# dag-ref region of the unified pointer space). Even though cross-attn found
# different positions per slot, those positions didn't translate to NUMBER
# positions in the prompt.
#
# v89 fix: add an auxiliary loss that directly supervises the cross-attn
# distribution to peak at the GOLD number position from Haiku data. For each
# DAG slot k whose gold args[i].source == "numbers", the slot's args[i]
# cross-attn distribution should peak at the token-position spanning that
# number. CE on softmax(slot_q @ prompt_k.T) against one-hot at the gold
# position. This is supervision on the CROSS-ATTN DIST itself, not on the
# downstream pointer logits — directly training the attention pattern, which
# the pointer projection then reads through.
#
# To enable per-arg supervision (args1 and args2 have different gold target
# positions per slot), v89 splits the v86 single shared cross-attn into TWO
# parallel cross-attns: one for args1, one for args2. Each gets its own K and
# V projection. The Q projection is shared (slot_query is the same). Output:
# slot_args_ctx_args1 feeds args1_q_w; slot_args_ctx_args2 feeds args2_q_w.
#
# When V89_SUPERVISED_ATTN=0: the v86 single-shared cross-attn path is
# preserved unchanged (warm-start safe).
# When V89_SUPERVISED_ATTN=1: split into two cross-attns. The new K/V projs
# are initialized at scale V89_PROJ_INIT_SCALE (default 0.02). For warm-start,
# the trainer can OPTIONALLY copy the v86 K/V proj values into both args1 and
# args2 projections (V89_INHERIT_V86=1) so we start where v88 left off and
# the supervised loss reshapes them.
V89_SUPERVISED_ATTN = int(os.environ.get("V89_SUPERVISED_ATTN", "0")) > 0
V89_PROJ_INIT_SCALE = float(os.environ.get("V89_PROJ_INIT_SCALE", "0.02"))
if V89_SUPERVISED_ATTN:
    print(f"[V89_SUPERVISED_ATTN] split args1/args2 cross-attn + supervised attn loss "
          f"(proj init scale={V89_PROJ_INIT_SCALE})", flush=True)

# v91 (2026-05-27) Simplified args pathway — collapse the 5-matmul args chain
# into 1 matmul that mirrors the ops_codebook mechanism. Audit on v90 step 100
# showed 10-20x gradient attenuation through args projections (ops_codebook
# grad_L2 ≈ 8.6, args projections ≈ 0.4-0.9). Root cause: 5 trainable matmul
# transforms + 2 softmaxes downstream of waist for args vs 1 + 1 for ops.
#
# v91 replaces the cross-attn + pointer chain with a single einsum:
#   args_codebook = concat(numbers_emb, slot_query)            # (B, N_max+K_max, H)
#   arg_query = slot_query + arg_pos_emb[i]                    # (B, K_max, 2, H)
#   args_logits = einsum("bkih,bjh->bkij", arg_query, args_codebook)
# arg_pos_emb is a learnable (2, H) tensor distinguishing args1 vs args2.
#
# When V91_SIMPLIFIED_ARGS=1:
#   - forward() uses the simplified path (no cross-attn, no pointer projections)
#   - deprecated tensors (args1_q_w, args2_q_w, args_k_w, v86_args_q_proj,
#     v86_args_k_proj, v86_args_v_proj, v86_args_slot_pos, v89_args1_*, v89_args2_*)
#     stay ALLOCATED for state-dict compat but are NOT in parameters().
#   - args_attn / args1_attn / args2_attn / args1_attn_scores / args2_attn_scores
#     are NOT emitted (no cross-attn to read).
#   - v89 supervised attention aux loss is no-op (gated in trainer).
V91_SIMPLIFIED_ARGS = int(os.environ.get("V91_SIMPLIFIED_ARGS", "0")) > 0
if V91_SIMPLIFIED_ARGS:
    print(f"[V91_SIMPLIFIED_ARGS] simplified args pathway: single-matmul codebook lookup "
          f"(deprecated: args*_q_w, args_k_w, v86_args_*, v89_args*)", flush=True)

# v96 (2026-05-28) CONSOLIDATION TABLE — compress the DELTA, not the state.
#
# Per breath:
#   delta = x_out - x_in
#   importance = sigmoid(delta @ gate_w + gate_b)            # per-dim gate
#   delta_q = delta * importance                              # quantized delta
#   pool = softmax(delta_q · breath_embed[k])                 # attention pool over T
#   delta_pooled = (pool * delta_q).sum(T)                    # (B, hidden)
#   artifact = pack(ops_logits, types_logits, conf, summary)  # (B, 165)
#   table[:, k, :] = artifact
#
# The WaistController at the FINAL breath reads the table as additional KV.
# Per-breath supervision on (ops_logits, types_logits, confidence) breaks the
# v85 template attractor by giving EVERY breath its own credit signal.
#
# Implementation lives in mycelium/v96.py + hooks here + l3_training.py for
# the per-row CE loss. ALWAYS allocated for state_dict symmetry; gradient
# inert (zero-init paths) when V96_CONSOLIDATION=0.
V96_CONSOLIDATION = int(os.environ.get("V96_CONSOLIDATION", "0")) > 0
if V96_CONSOLIDATION:
    print(f"[V96_CONSOLIDATION] consolidation-table architecture ON: per-breath delta → "
          f"gated quantization → attention pool → packed artifact (165d) → "
          f"WaistController reads table KV at final breath.", flush=True)

# v96.2 (2026-05-28) Per-breath TEMPERATURE DECAY on ops/types logits — broad early,
# sharp late. Bombe-inspired: early breaths keep many candidate selections alive,
# late breaths commit. Default OFF for v96.1 byte-compat; gate-on with
# V96_TEMPERATURE_DECAY=1 in v96.2.
V96_TEMPERATURE_DECAY = int(os.environ.get("V96_TEMPERATURE_DECAY", "0")) > 0
V96_T_START = float(os.environ.get("V96_T_START", "2.0"))
V96_T_END   = float(os.environ.get("V96_T_END", "0.3"))
if V96_TEMPERATURE_DECAY:
    print(f"[V96_TEMPERATURE_DECAY] per-breath sharpening ON: T_start={V96_T_START}, "
          f"T_end={V96_T_END}, linear (floor at T_end).", flush=True)

# v97 (2026-05-28) CALIBRATION HEAD — pure auxiliary loss on a read-only head.
# Bombe-inspired self-assessment WITHOUT the v96 consolidation table feedback loop.
#
# Architecture: small Linear(1024 -> 1) head that predicts P(model's final answer
# correct) from each breath's pooled hidden state. Per-breath target is a linear
# progression: B0 = 0.5 (uncertain), B_{K-1} = sigmoid-of-(-final_pb_ce) proxy.
# The model learns to self-assess. The calibration head READS the breath's hidden
# state but DOES NOT WRITE back to any state subsequent breaths read — pure
# forward-then-loss; no architectural feedback. This is the key distinction from
# v96 (which had table reads flowing through the residual stream into next breath's
# writes — structural positive feedback that drove waist_norm 1.4 → 710).
#
# Pure auxiliary loss; never modifies the residual stream. AR-correct masking
# preserved (V81_MAIN_ATTN_MASK=0 for AR generation).
V97_CALIBRATION = int(os.environ.get("V97_CALIBRATION", "0")) > 0
if V97_CALIBRATION:
    print(f"[V97_CALIBRATION] calibration head ON: per-breath self-assessment via "
          f"Linear(1024 -> 1) read-only head. Aux loss only — no feedback into "
          f"residual stream.", flush=True)


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


def _make_orthogonal_breath_embed(max_loops: int, hidden: int, norm: float, seed: int = 42) -> np.ndarray:
    """v77b orthogonal breath_embed init. Returns shape (max_loops, hidden) where
    each row has L2 = norm and rows are mutually orthogonal (when max_loops <= hidden).
    Uses QR decomposition on a Gaussian random matrix; deterministic with seed.

    Diagnosed from v77 step 1500: breath_embed rows had L2 0.08-0.17 after 1500
    steps from zero-init — the breaths can't differentiate, so gradients from
    Layer-0 supervision (breath 0) and Layer-5 supervision (breath 5) fight on
    the same near-zero embedding. Orthogonal init at meaningful norm gives each
    breath a unique, linearly-independent signal from step 0.
    """
    rng = np.random.RandomState(seed)
    random_matrix = rng.randn(hidden, max_loops).astype(np.float32)
    Q, _ = np.linalg.qr(random_matrix)            # (hidden, max_loops) orthonormal columns
    ortho_embed = (Q.T * norm).astype(np.float32) # (max_loops, hidden), each row L2 = norm
    return ortho_embed


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
if BREATH_EMBED_ORTHO_INIT > 0.0:
    print(f"[BREATH_EMBED_ORTHO_INIT] ortho L2={BREATH_EMBED_ORTHO_INIT} per breath (post-load override)", flush=True)
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
if NOTEBOOK_DAG:
    pe_str = " + slot_pos_embed" if NOTEBOOK_DAG_POS_EMBED else ""
    print(f"[NOTEBOOK_DAG] active: D_nb=512, n_heads={NOTEBOOK_DAG_N_HEADS}, causal cross-attn{pe_str}", flush=True)
if PROMPT_REFRESH_ALPHA > 0.0:
    print(f"[PROMPT_REFRESH] α={PROMPT_REFRESH_ALPHA} — skip-connection from raw prompt_emb into every breath's input", flush=True)
if TWO_PHASE:
    print(f"[TWO_PHASE] EXPAND={EXPAND_LAYERS} layers @ temp={EXPAND_TEMP} | COMPRESS={COMPRESS_LAYERS} layers @ temp={COMPRESS_TEMP} — structural inhale-exhale", flush=True)
if COLLAPSE_V69:
    print(f"[COLLAPSE_V69] waist={COLLAPSE_WAIST_DIM}d, codebook={COLLAPSE_CODEBOOK_N} entries, τ={COLLAPSE_TAU}, gate_bias={COLLAPSE_GATE_BIAS}, entropy_reg={COLLAPSE_ENTROPY_REG}", flush=True)
if COLLAPSE_V70:
    print(f"[COLLAPSE_V70] waist={COLLAPSE_V70_WAIST_DIM}d, codebook={COLLAPSE_V70_CODEBOOK_N} entries, τ={COLLAPSE_V70_TAU}, "
          f"breath_dim={COLLAPSE_V70_BREATH_DIM}, budget={COLLAPSE_V70_BUDGET_START}-{COLLAPSE_V70_BUDGET_DECAY}·k≥{COLLAPSE_V70_BUDGET_MIN}, "
          f"sparsity_w={COLLAPSE_V70_SPARSITY_WEIGHT}, gate_bias={COLLAPSE_V70_GATE_BIAS}", flush=True)
if COLLAPSE_V71:
    print(f"[COLLAPSE_V71] waist={COLLAPSE_V71_WAIST_DIM}d, codebook={COLLAPSE_V71_CODEBOOK_N} entries, τ={COLLAPSE_V71_TAU}, "
          f"breath_dim={COLLAPSE_V71_BREATH_DIM}, budget={COLLAPSE_V71_BUDGET_START}-{COLLAPSE_V71_BUDGET_DECAY}·k≥{COLLAPSE_V71_BUDGET_MIN}, "
          f"sparsity_w={COLLAPSE_V71_SPARSITY_WEIGHT} (10×v70), gate_bias={COLLAPSE_V71_GATE_BIAS} (sigmoid={1/(1+math.exp(-COLLAPSE_V71_GATE_BIAS)):.3f}), "
          f"controller_reads=compressed_x·importance (no prototype add-back)", flush=True)
if BOUNDARY_AUX_WEIGHT > 0.0:
    print(f"[BOUNDARY_AUX] weight={BOUNDARY_AUX_WEIGHT} — BCE on per-breath boundary head (predict next-token=####)", flush=True)
if MAX_STEP_BASE > 0.0:
    print(f"[STEP_SCHEDULE] cosine: base={MAX_STEP_BASE} -> min={MAX_STEP_MIN} across K breaths (half-cosine: k=0 wide, k=K-1 tight)", flush=True)
elif MAX_STEP_SIZE > 0.0:
    print(f"[MAX_STEP_SIZE] v75 bounded per-breath delta: per-token L2 norm of (breath_out - breath_in) capped at {MAX_STEP_SIZE}", flush=True)
if NOTEBOOK_NO_DETACH:
    print(f"[NOTEBOOK_NO_DETACH] gradient through notebook write path enabled (no-op in standard NOTEBOOK_V24 — write already non-detached)", flush=True)
if V78_HEAD_CODEBOOK:
    print(f"[V78_HEAD_CODEBOOK] per-head model codebook active: {V78_HEAD_CODEBOOK_N} cells × n_heads heads (each head extends K/V with its own codebook)", flush=True)


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

        # v78 (2026-05-24) Per-head model codebook. Shape (n_heads, N, head_dim).
        # Each head gets its own copy of an N-cell codebook (init: ONE base codebook
        # randn × 0.02 replicated across heads). Always allocated for state-dict
        # symmetry; only used in attention when V78_HEAD_CODEBOOK=1.
        #
        # When active, the codebook entries are concatenated to K and V along the
        # sequence-position axis: each head attends to (T + N) positions instead
        # of T. The codebook acts as 'always-on memory cells' the head can route
        # information through, distinct from the prompt/answer-text positions.
        n_head_cells = max(1, V78_HEAD_CODEBOOK_N)
        # Deterministic seed combining the layer's phase and a global salt so
        # different (phase, codebook) instances start at different random tensors,
        # but the same phase-layer always starts at the same place.
        _phase_int = int(round(phase * 1e6))
        _rng = np.random.RandomState(2078 + (_phase_int & 0x7fffffff))
        _base_cb = (_rng.randn(n_head_cells, cfg.head_dim).astype(np.float32) * 0.02)
        # Tile across n_heads — shared init, independent training (each head's
        # slice evolves independently under gradient descent).
        _tiled = np.broadcast_to(_base_cb[None, :, :], (cfg.n_heads, n_head_cells, cfg.head_dim)).copy()
        self.v78_head_codebook = Tensor(_tiled, dtype=dtypes.float).contiguous()

        # Sine-wave temperature: T = exp(amp * sin(phase))
        self.temperature = math.exp(cfg.temp_amp * math.sin(phase))
        self.attn_scale = 1.0 / (math.sqrt(cfg.head_dim) * self.temperature)

    def parameters(self):
        # v78 head_codebook NOT included here — it's gated via collect_params() in
        # l3_train.py (only added to the optimizer when V78_HEAD_CODEBOOK=1, since
        # otherwise its gradient is None and AdamW asserts on missing grads).
        # State_dict registration is separate (always present for ckpt symmetry).
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
                                    prompt_mask: Tensor | None = None,
                                    alpha: tuple | None = None,
                                    temp_mult: float = 1.0):
        """Batched single-token cached forward.

        x_new:        (B, 1, H)
        k_buf, v_buf: (B, n_heads, max_seq_len, head_dim) — buffers shared across batch
        t_pos_t:      0-dim or shape (1,) Tensor — uniform write position
        prompt_mask:  (B, max_seq_len) — 1 where the cache position holds a valid (non-pad)
                      prompt token. The new-token slot itself is added to the valid mask
                      via the causal `pos <= t_pos_t` comparison (no need to update the
                      prompt_mask between calls because future slots are zero by default
                      and t_pos_t monotonically advances).
        alpha:        optional (ac, asn) Q-rotation tuple. When None, uses the breath's
                      default alpha for loop_idx. Pass the per-(layer, head) pitch alpha
                      here for parity with Stage 1's forward_with_kv when PER_HEAD_PITCH=1.
        temp_mult:    attention temperature multiplier. Matches the per-breath SINE_TEMP
                      schedule applied in Stage 1's forward_with_kv. Default 1.0 (legacy).
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

        q_new, k_new = self.rope.apply_at_tensor_pos(q_new, k_new, loop_idx, t_pos_t, alpha=alpha)

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

        if prompt_mask is not None:
            pmask = prompt_mask.reshape(B, 1, 1, max_len).cast(dtypes.bool)
            valid = causal & pmask
        else:
            valid = causal

        scale = self.attn_scale / float(temp_mult)
        scores = q_new @ k_buf_new.transpose(-2, -1) * scale
        scores = valid.where(scores, Tensor(-float("inf"), dtype=scores.dtype))
        # Clamp pre-softmax scores to prevent inf/NaN at head_dim=256 (H=2048).
        # Model is fp32; clip(-1e4,1e4) keeps scores well below overflow while
        # preserving relative attention — exp(10000) would overflow fp32, exp(1e4)→inf.
        attn = scores.clip(-1e4, 1e4).softmax(-1)
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
        # Clamp pre-softmax scores to prevent inf/NaN at head_dim=256 (H=2048).
        # Model is fp32; clip(-1e4,1e4) keeps scores well below overflow while
        # preserving relative attention — exp(10000) would overflow fp32, exp(1e4)→inf.
        attn = scores.clip(-1e4, 1e4).softmax(-1)
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
        # Clamp pre-softmax scores to prevent inf/NaN at head_dim=256 (H=2048).
        # Model is fp32; clip(-1e4,1e4) keeps scores well below overflow while
        # preserving relative attention — exp(10000) would overflow fp32, exp(1e4)→inf.
        attn = scores.clip(-1e4, 1e4).softmax(-1)
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
            # v78 head codebook: each head attends to T sequence positions + N codebook cells.
            # Codebook entries are NOT causal-masked (always visible) and NOT padding-masked.
            # K/V get extended along the sequence axis with the codebook (per-head, broadcast over batch).
            if V78_HEAD_CODEBOOK:
                N_cb = int(self.v78_head_codebook.shape[1])
                # cb shape: (n_heads, N_cb, head_dim). Cast to attention dtype, broadcast to batch.
                cb = self.v78_head_codebook.cast(k.dtype).reshape(1, cfg.n_heads, N_cb, cfg.head_dim)
                cb_b = cb.expand(B, cfg.n_heads, N_cb, cfg.head_dim)
                # Cat then force contiguous (AMD quirk: cat(view, expand) can slow down matmul).
                k_ext = Tensor.cat(k, cb_b, dim=2).contiguous()
                v_ext = Tensor.cat(v, cb_b, dim=2).contiguous()
            else:
                k_ext, v_ext = k, v
                N_cb = 0
            scores = q @ k_ext.transpose(-2, -1) * scale
            # Causal mask only over the sequence positions (first S keys). Codebook
            # cells (keys S..S+N_cb-1) are always visible → mask 1.0 there.
            if N_cb > 0:
                seq_mask = Tensor.ones(S, S, dtype=scores.dtype).tril()                 # (S, S)
                cb_mask = Tensor.ones(S, N_cb, dtype=scores.dtype)                       # (S, N_cb) all visible
                mask = Tensor.cat(seq_mask, cb_mask, dim=1).reshape(1, 1, S, S + N_cb)
            else:
                mask = Tensor.ones(S, S, dtype=scores.dtype).tril().reshape(1, 1, S, S)
            scores = scores.masked_fill(mask == 0, float("-inf"))
            if attn_mask is not None:
                # attn_mask shape: (B, S) — 1 valid, 0 padding. Broadcast to (B, 1, 1, S).
                # When codebook is on, extend with all-1s over the N_cb codebook positions.
                if N_cb > 0:
                    ones_cb = Tensor.ones(B, N_cb, dtype=attn_mask.dtype)
                    attn_mask_ext = Tensor.cat(attn_mask, ones_cb, dim=1)
                else:
                    attn_mask_ext = attn_mask
                key_mask = attn_mask_ext.reshape(B, 1, 1, S + N_cb).cast(dtypes.bool)
                scores = key_mask.where(scores, Tensor(-float("inf"), dtype=scores.dtype))
            # Clamp pre-softmax scores to prevent inf/NaN at head_dim=256 (H=2048).
            attn = scores.clip(-1e4, 1e4).softmax(-1)
            ctx = (attn @ v_ext).transpose(1, 2).reshape(B, S, H)
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
        # Clamp pre-softmax scores to prevent inf/NaN at head_dim=256 (H=2048).
        attn = scores.clip(-1e4, 1e4).softmax(-1)
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
        # v44 doubled-layers: second set of 4 phase-layers for the sin<0 half of
        # the rotation cycle. Pythia-init copied to both sets at load time; both
        # always present in state_dict for ckpt symmetry (gradient inert when
        # DOUBLED_LAYERS=0 — never used in forward).
        self.layers_b = [BreathingLayer(cfg, ph, self.shared, self.rope) for ph in phases]
        # v68 TWO_PHASE: explicit EXPAND (4 warm layers) + COMPRESS (2 cool layers)
        # with INDEPENDENT SharedWeights per phase. Each set has its own V/O/FFN-out/LNs.
        # When TWO_PHASE=0, these allocate but never activate — state_dict stable.
        if TWO_PHASE:
            self.expand_shared = SharedWeights(cfg)
            self.compress_shared = SharedWeights(cfg)
            # phase angles for expand layers (continue the sine wave) and compress layers
            expand_phases = [i * 2 * math.pi / cfg.n_phases for i in range(EXPAND_LAYERS)]
            compress_phases = [(EXPAND_LAYERS + i) * 2 * math.pi / cfg.n_phases for i in range(COMPRESS_LAYERS)]
            self.expand_layers = [BreathingLayer(cfg, ph, self.expand_shared, self.rope) for ph in expand_phases]
            self.compress_layers = [BreathingLayer(cfg, ph, self.compress_shared, self.rope) for ph in compress_phases]
        else:
            # Minimal placeholders so state_dict signatures stay stable across configs.
            self.expand_shared = SharedWeights(cfg)
            self.compress_shared = SharedWeights(cfg)
            self.expand_layers = []
            self.compress_layers = []
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

        # Stochastic depth keep-mask buffer (one float per breath slot). Default
        # all-ones (no-op). l3_train.py rewrites this each training step when
        # STOCH_DEPTH_P > 0; the JIT'd graph captures the buffer reference and
        # picks up the new values on replay (same pattern as layer_pitch_scale).
        # Not learnable; included in state_dict for ckpt symmetry.
        self.stoch_keep_mask = Tensor.ones((cfg.max_loops,), dtype=dtypes.float).contiguous()

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
        # Three mutually-exclusive per-layer offset modes (PER_LAYER_OFFSETS_RADIANS
        # has highest precedence, then ACROSS_LAYER_PITCH_TARGET, then v23a default):
        # 1. PER_LAYER_OFFSETS_RADIANS: arbitrary 4 values (e.g., triangle-small).
        # 2. ACROSS_LAYER_PITCH_TARGET: monotonic l * target (with optional ramp).
        # 3. v23a default: monotonic l * (pitch_range / (n_phases * n_heads)).
        # All preserve "all heads within a layer share offset" — the lesson from
        # v46 within-layer split where heterogeneous heads broke W_O.
        base_layer_step = pitch_range / (cfg.n_phases * cfg.n_heads)
        explicit_offsets = None
        if PER_LAYER_OFFSETS_RADIANS:
            parts = [float(x) for x in PER_LAYER_OFFSETS_RADIANS.split(",")]
            if len(parts) != cfg.n_phases:
                raise ValueError(f"PER_LAYER_OFFSETS_RADIANS needs {cfg.n_phases} values, got {len(parts)}")
            explicit_offsets = parts
        if ACROSS_LAYER_PITCH_TARGET > 0.0:
            init_layer_step = base_layer_step if ACROSS_LAYER_PITCH_RAMP_STEPS > 0 else ACROSS_LAYER_PITCH_TARGET
        else:
            init_layer_step = base_layer_step
        half_heads = cfg.n_heads // 2
        init_scale = 0.0 if (QUADRATURE_HEADS and QUADRATURE_RAMP_STEPS > 0) else (1.0 if QUADRATURE_HEADS else 0.0)
        self._quadrature_half_heads = half_heads
        self._quadrature_pitch_range = pitch_range
        self._base_layer_step = base_layer_step  # for ramp logic in trainer
        def _head_offset(layer_idx, head_idx, scale, layer_step):
            if explicit_offsets is not None:
                base = explicit_offsets[layer_idx]
            else:
                base = layer_idx * layer_step
            if QUADRATURE_HEADS and head_idx >= half_heads:
                return base + (math.pi / 2) * scale
            return base
        # v68 TWO_PHASE: total layers per breath is EXPAND_LAYERS + COMPRESS_LAYERS (default 6).
        # Otherwise stays at cfg.n_phases (4). Per-(layer, head) pitch buffer sized accordingly.
        n_positions = (EXPAND_LAYERS + COMPRESS_LAYERS) if TWO_PHASE else cfg.n_phases
        ph_init = [[_head_offset(l, h, init_scale, init_layer_step) for h in range(cfg.n_heads)]
                   for l in range(n_positions)]
        self.per_head_pitch = Tensor(ph_init, dtype=dtypes.float).contiguous()
        # Precomputed cos/sin tables — eliminates per-breath cos/sin compute.
        # Shape (n_layers, 1, n_heads, 1, 1) to match alpha broadcast shape on indexing.
        # Realize so JIT sees these as constant buffers, not lazy ops (lazy ops
        # under per-breath JIT recomputation triggered MEMVIOL on first v23 attempt).
        # When QUADRATURE_RAMP_STEPS > 0, l3_train.py updates these per step via
        # .assign() to ramp the quadrature offset; JIT graph captures the buffer
        # reference and picks up new values on replay (same pattern as layer_pitch_scale).
        ph_t = Tensor(ph_init, dtype=dtypes.float)
        self.per_head_pitch_cos = ph_t.cos().reshape(n_positions, 1, cfg.n_heads, 1, 1).contiguous().realize()
        self.per_head_pitch_sin = ph_t.sin().reshape(n_positions, 1, cfg.n_heads, 1, 1).contiguous().realize()

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

        # v61 (2026-05-21) DAG notebook params. Storage allocated per-forward.
        # nb_dag_o_w ZERO-INIT → DAG read = 0 at step 0 (warm-start safe).
        nb_h = max(1, NOTEBOOK_DAG_N_HEADS)
        assert NB_DIM % nb_h == 0, f"NB_DIM={NB_DIM} must be divisible by NOTEBOOK_DAG_N_HEADS={nb_h}"
        self.nb_dag_n_heads = nb_h
        self.nb_dag_head_dim = NB_DIM // nb_h
        self.nb_dag_q_w     = (Tensor.randn(cfg.hidden, NB_DIM, dtype=dtypes.float) * 0.02).contiguous()
        self.nb_dag_q_b     = Tensor.zeros((NB_DIM,), dtype=dtypes.float).contiguous()
        self.nb_dag_k_w     = (Tensor.randn(NB_DIM, NB_DIM, dtype=dtypes.float) * 0.02).contiguous()
        self.nb_dag_k_b     = Tensor.zeros((NB_DIM,), dtype=dtypes.float).contiguous()
        self.nb_dag_v_w     = (Tensor.randn(NB_DIM, NB_DIM, dtype=dtypes.float) * 0.02).contiguous()
        self.nb_dag_v_b     = Tensor.zeros((NB_DIM,), dtype=dtypes.float).contiguous()
        self.nb_dag_o_w     = Tensor.zeros((NB_DIM, cfg.hidden), dtype=dtypes.float).contiguous()
        self.nb_dag_o_b     = Tensor.zeros((cfg.hidden,), dtype=dtypes.float).contiguous()
        self.nb_dag_write_w = (Tensor.randn(cfg.hidden, NB_DIM, dtype=dtypes.float) * 0.02).contiguous()
        self.nb_dag_write_b = Tensor.zeros((NB_DIM,), dtype=dtypes.float).contiguous()
        self.nb_dag_write_query = Tensor.zeros((cfg.hidden,), dtype=dtypes.float).contiguous()
        self.nb_dag_pos_embed = (Tensor.randn(cfg.max_loops, NB_DIM, dtype=dtypes.float) * 0.02).contiguous()

        # v69 COLLAPSE — JPEG-inspired lossy compression. Always allocated for state-dict
        # symmetry; gradient inert when COLLAPSE_V69=0 (α=0 init makes them no-ops anyway).
        #   codebook_keys/values: (N, hidden) — transform stage (match against prototypes)
        #   gate_w/b: (hidden, hidden) / (hidden,) — quantize stage (importance per dim,
        #     conditioned on prototype). Bias init COLLAPSE_GATE_BIAS=+2 → sigmoid≈0.88
        #     at init → keep most dims initially, sharpen via gradient.
        #   proj_down/up: (hidden, waist_dim) / (waist_dim, hidden) — encode stage
        #   alpha: (1,) — residual scale, zero-init (LoRA-style, gradient builds it)
        cb_n = max(1, COLLAPSE_CODEBOOK_N)
        cb_d = COLLAPSE_WAIST_DIM
        self.collapse_codebook_keys   = (Tensor.randn(cb_n, cfg.hidden, dtype=dtypes.float) * 0.02).contiguous()
        self.collapse_codebook_values = (Tensor.randn(cb_n, cfg.hidden, dtype=dtypes.float) * 0.02).contiguous()
        self.collapse_gate_w          = (Tensor.randn(cfg.hidden, cfg.hidden, dtype=dtypes.float) * 0.02).contiguous()
        self.collapse_gate_b          = (Tensor.ones((cfg.hidden,), dtype=dtypes.float) * COLLAPSE_GATE_BIAS).contiguous()
        self.collapse_proj_down       = (Tensor.randn(cfg.hidden, cb_d, dtype=dtypes.float) * 0.02).contiguous()
        self.collapse_proj_up         = (Tensor.randn(cb_d, cfg.hidden, dtype=dtypes.float) * 0.02).contiguous()
        self.collapse_alpha           = Tensor.zeros((1,), dtype=dtypes.float).contiguous()
        # Last-call cache (for entropy logging) — written by apply_collapse_v69, read by trainer.
        # Not a learnable param.
        self._collapse_last_match_entropy = None

        # --- v70 COLLAPSE: refined collapse with breath-conditioned gate + budget sparsity ---
        # Allocated regardless of COLLAPSE_V70 flag for state-dict symmetry. Gradient inert
        # when COLLAPSE_V70=0 (params not in forward path; opt skips them via collect_params).
        # PERF (2026-05-23): codebook + gate moved from INPUT dim (1024) to WAIST dim (512).
        # Cuts 3 of the 5 new v70 matmuls in half. The codebook now matches against the
        # compressed rep (after proj_down) — same pattern as v66's waist_codebook.
        cb_n70 = max(1, COLLAPSE_V70_CODEBOOK_N)
        cb_d70 = COLLAPSE_V70_WAIST_DIM
        be_d70 = COLLAPSE_V70_BREATH_DIM
        # proj_down/up at v66's B-field dims (1024 ↔ 512). When warm-starting from v66,
        # the trainer copies bfield_proj_down/up/bias into these slots.
        self.collapse_v70_proj_down = (Tensor.randn(cfg.hidden, cb_d70, dtype=dtypes.float) * 0.02).contiguous()
        self.collapse_v70_proj_up   = (Tensor.randn(cb_d70, cfg.hidden, dtype=dtypes.float) * 0.02).contiguous()
        self.collapse_v70_bias      = Tensor.zeros((cfg.hidden,), dtype=dtypes.float).contiguous()
        # Codebook at WAIST dim (cb_d70=512). Keys are random small, values zero so prototype=0
        # at init → residual=compressed_x → gate sees zero → output unchanged at α=0.
        self.collapse_v70_codebook_keys   = (Tensor.randn(cb_n70, cb_d70, dtype=dtypes.float) * 0.02).contiguous()
        self.collapse_v70_codebook_values = Tensor.zeros((cb_n70, cb_d70), dtype=dtypes.float).contiguous()
        # Gate: SPLIT into two params (avoid slice-view backward scatter-add on AMD).
        #   gate_w_proto: (waist, waist) — gates on the codebook prototype (at waist dim)
        #   gate_w_breath: (be_d, waist) — gates on the breath embedding
        # Sum + bias → sigmoid → per-dim importance over waist dim.
        self.collapse_v70_gate_w_proto  = (Tensor.randn(cb_d70, cb_d70, dtype=dtypes.float) * 0.02).contiguous()
        self.collapse_v70_gate_w_breath = (Tensor.randn(be_d70, cb_d70, dtype=dtypes.float) * 0.02).contiguous()
        self.collapse_v70_gate_b        = (Tensor.ones((cb_d70,), dtype=dtypes.float) * COLLAPSE_V70_GATE_BIAS).contiguous()
        # Legacy unified gate_w kept for ckpt-load symmetry; ignored when COLLAPSE_V70 active.
        # Sized to never be in the forward path under v70.
        self.collapse_v70_gate_w        = (Tensor.zeros((1, 1), dtype=dtypes.float)).contiguous()
        # Per-breath embedding (zero-init → no breath-conditioning at step 0)
        self.collapse_v70_breath_embed = Tensor.zeros((cfg.max_loops, be_d70), dtype=dtypes.float).contiguous()
        # Residual scale α — zero-init means correction has no effect at step 0
        # (model output is byte-identical to no-collapse forward).
        self.collapse_v70_alpha     = Tensor.zeros((1,), dtype=dtypes.float).contiguous()
        # Diagnostics (written by apply_collapse_v70 at breath 0; read by trainer).
        self._collapse_v70_last_match_entropy = None
        self._collapse_v70_last_importance_mean = None
        # v70 per-breath sparsity LIST (reset on breath 0). Trainer sums after K-loop.
        # Python list — append at breath k, sum once outside the K-loop. Avoids the
        # K-deep self.X = self.X + s chain that fragments AMD's kernel scheduling.
        self._collapse_v70_sparsity_list = None

        # --- v71 COLLAPSE: same pipeline as v70, but three fixes + cleaner controller signal.
        # Allocated regardless of COLLAPSE_V71 flag for state-dict symmetry. Gradient inert
        # when COLLAPSE_V71=0 (params not in forward path; opt skips them via collect_params).
        # FIXES vs v70:
        #   1. SPARSITY_WEIGHT 0.1 → 1.0 (10× stronger) → gate feels pressure to close.
        #   2. GATE_BIAS 4.6 → 1.0 → sigmoid(1)=0.73, closer to budget target.
        #   3. K-means codebook init (via COLLAPSE_V71_KMEANS_INIT_PATH) → break symmetry.
        # ARCHITECTURAL: controller reads (compressed_x × importance) — NO prototype add-back.
        # This keeps the controller's input shape stable at init (no codebook-dependent drift).
        # The codebook participates via the decompression path (decompressed has prototype add).
        cb_n71 = max(1, COLLAPSE_V71_CODEBOOK_N)
        cb_d71 = COLLAPSE_V71_WAIST_DIM
        be_d71 = COLLAPSE_V71_BREATH_DIM
        # proj_down/up at v66's B-field dims (1024 ↔ 512). When warm-starting from v66,
        # the trainer copies bfield_proj_down/up/bias into these slots.
        self.collapse_v71_proj_down = (Tensor.randn(cfg.hidden, cb_d71, dtype=dtypes.float) * 0.02).contiguous()
        self.collapse_v71_proj_up   = (Tensor.randn(cb_d71, cfg.hidden, dtype=dtypes.float) * 0.02).contiguous()
        self.collapse_v71_bias      = Tensor.zeros((cfg.hidden,), dtype=dtypes.float).contiguous()
        # Codebook at WAIST dim. Keys default-init random small; warm-start replaces with k-means
        # centers. Values zero-init so prototype=0 at start (decompression's prototype-add is
        # a no-op at init; gradient builds it).
        self.collapse_v71_codebook_keys   = (Tensor.randn(cb_n71, cb_d71, dtype=dtypes.float) * 0.02).contiguous()
        self.collapse_v71_codebook_values = Tensor.zeros((cb_n71, cb_d71), dtype=dtypes.float).contiguous()
        # Gate: SPLIT into two params (same AMD-safe pattern as v70).
        self.collapse_v71_gate_w_proto  = (Tensor.randn(cb_d71, cb_d71, dtype=dtypes.float) * 0.02).contiguous()
        self.collapse_v71_gate_w_breath = (Tensor.randn(be_d71, cb_d71, dtype=dtypes.float) * 0.02).contiguous()
        self.collapse_v71_gate_b        = (Tensor.ones((cb_d71,), dtype=dtypes.float) * COLLAPSE_V71_GATE_BIAS).contiguous()
        # Per-breath embedding (zero-init → no breath-conditioning at step 0)
        self.collapse_v71_breath_embed = Tensor.zeros((cfg.max_loops, be_d71), dtype=dtypes.float).contiguous()
        # Residual scale α — zero-init means correction has no effect at step 0
        # (model output is byte-identical to no-collapse forward).
        self.collapse_v71_alpha     = Tensor.zeros((1,), dtype=dtypes.float).contiguous()
        # Diagnostics (written by apply_collapse_v71 at breath 0; read by trainer).
        self._collapse_v71_last_match_entropy = None
        self._collapse_v71_last_importance_mean = None
        # v71 per-breath sparsity LIST (same pattern as v70).
        self._collapse_v71_sparsity_list = None

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
        # v83 (2026-05-27) per-breath waist width mask. Shape (max_loops, bf_w).
        # When BFIELD_WAIST_SCHEDULE is set, the mask zeros out (bf_w - k_b)
        # channels per breath BEFORE GELU + codebook injection. When unset, the
        # mask is all-ones (byte-identical to v82). Non-learnable (not in
        # parameters()), constructed from the env var at module load.
        sched = _parse_bfield_waist_schedule(BFIELD_WAIST_SCHEDULE, bf_w)
        mask_np = np.ones((cfg.max_loops, bf_w), dtype=np.float32)
        if sched is not None:
            # Pad/truncate the schedule to max_loops. If schedule shorter than
            # max_loops, hold the LAST value (typical: last breaths run at full
            # precision); if longer, truncate. Both keep mask shape (max_loops, bf_w).
            for b in range(cfg.max_loops):
                if b < len(sched):
                    k_b = sched[b]
                else:
                    k_b = sched[-1] if sched else bf_w
                if k_b < bf_w:
                    mask_np[b, k_b:] = 0.0
        self.bfield_waist_mask = Tensor(mask_np, dtype=dtypes.float).contiguous()
        # Diagnostic / logging convenience
        self._bfield_waist_schedule = sched
        if sched is not None:
            print(f"[v83 BFIELD_WAIST_SCHEDULE] per-breath waist widths: {sched} "
                  f"(max waist={bf_w}, max_loops={cfg.max_loops})", flush=True)

        # v53 (2026-05-19) learnable codebook at the IB waist.
        #
        # Init regime:
        #   keys:   randn × 0.02   — small random for entry diversity, so attention
        #                            scores aren't all identical at step 0.
        #   values: ZERO           — guarantees zero contribution at step 0 (the codebook
        #                            adds nothing to compressed → bit-identical to no-
        #                            codebook warm-start). Same logic as bfield_proj_up
        #                            zero-init (proven to bootstrap successfully in v38).
        #
        # Gradient bootstrap: at step 0, scores from random keys give non-uniform softmax
        # weights. Gradient flows to VALUES via weights^T × grad_retrieved (non-zero).
        # Once values are non-zero, gradient flows to keys. Clean cold-start of the
        # codebook on top of any warm-started model.
        cb_n = max(1, WAIST_CODEBOOK_N)
        self.waist_codebook_keys = (Tensor.randn(cb_n, bf_w, dtype=dtypes.float) * 0.02).contiguous()
        self.waist_codebook_values = Tensor.zeros((cb_n, bf_w), dtype=dtypes.float).contiguous()

    def apply_bfield_waist(self, x: Tensor, return_compressed: bool = False,
                             loop_idx: int | None = None) -> Tensor:
        """B-field IB bottleneck with optional CFG-alpha residual scale.

        Residual mode (BFIELD_ENFORCED=0):
            out = x + α · decompressed     (zero-init proj_up → identity at step 0)
        Enforced mode (BFIELD_ENFORCED=1):
            out = decompressed             (no skip — rep must flow through waist)

        When return_compressed=True, returns (out, compressed) for aux supervision
        on the 512d intermediate.

        v83 (2026-05-27): when BFIELD_WAIST_SCHEDULE is set and loop_idx is provided,
        applies the per-breath waist-width mask BEFORE the codebook injection +
        GELU. Channels above the per-breath threshold are zeroed → narrow waist
        in early breaths, full waist in later breaths. Mask is non-learnable, so
        the projection params still receive gradient on the kept channels.
        """
        x_f = x.cast(dtypes.float)
        compressed = x_f @ self.bfield_proj_down
        # v83: per-breath waist-width mask. When the schedule is unset, mask is
        # all-ones (no-op). When set + loop_idx provided, mask zeros channels
        # above k_b. Multiplying before the codebook injection AND GELU ensures
        # both downstream paths see a narrower effective waist.
        if self._bfield_waist_schedule is not None and loop_idx is not None:
            mask_row = self.bfield_waist_mask[loop_idx].reshape(1, 1, -1).cast(dtypes.float)
            compressed = compressed * mask_row
        # v50 codebook injection — query learnable keys, retrieve weighted values,
        # add to compressed state before GELU. Values are zero-init so contribution
        # is identity at step 0 (warm-start compatible).
        if WAIST_CODEBOOK_N > 0 and WAIST_CODEBOOK_INJECT_WEIGHT > 0.0:
            # compressed: (B, T, bf_w)  keys: (N, bf_w)  values: (N, bf_w)
            scores = compressed @ self.waist_codebook_keys.T
            weights = scores.softmax(axis=-1)
            retrieved = weights @ self.waist_codebook_values
            compressed = compressed + WAIST_CODEBOOK_INJECT_WEIGHT * retrieved
            # v83: re-mask after codebook injection so retrieved values can't
            # "leak" through the closed channels (values are zero-init initially
            # but become non-zero with training).
            if self._bfield_waist_schedule is not None and loop_idx is not None:
                compressed = compressed * mask_row
        activated = compressed.gelu()
        decompressed = activated @ self.bfield_proj_up + self.bfield_bias
        if BFIELD_ENFORCED:
            out = decompressed.cast(x.dtype)
        else:
            out = (x_f + self.bfield_alpha * decompressed).cast(x.dtype)
        if return_compressed:
            return out, compressed.cast(x.dtype)
        return out

    def apply_collapse_v69(self, x: Tensor, return_compressed: bool = False) -> Tensor:
        """v69 collapse: JPEG-inspired lossy compression at the waist.

        Pipeline:
          1. TRANSFORM   — codebook matching: softmax((x @ keys) / τ) @ values = prototype
          2. RESIDUAL    — residual = x − prototype  (what doesn't fit any prototype)
          3. QUANTIZE    — importance = sigmoid(prototype @ gate_w + gate_b)
                          residual_gated = residual × importance
          4. ENCODE      — waist_compressed = residual_gated @ proj_down  (1024 → 128)
          5. RECONSTRUCT — correction = prototype + (waist_compressed @ proj_up)
                          out = x + α × correction   (α=0 init → step 0 ≈ identity)

        Per-breath supervision: WaistController reads waist_compressed and decodes step-k
        tokens. Gradient flows back through the entire pipeline. The gate learns
        "what dimensions matter" from the consequences of dropping them.

        Confidence (free): residual.norm() — how much the prototype DOESN'T explain.
        Match entropy (logged): −Σ p log p of match_weights — how peaked the matching is.
        """
        x_f = x.cast(dtypes.float)
        # 1. TRANSFORM — codebook match
        # scores: (B, T, N), match_weights: (B, T, N), prototype: (B, T, hidden)
        scores = (x_f @ self.collapse_codebook_keys.T) / (float(self.collapse_codebook_keys.shape[-1]) ** 0.5)
        match_weights = (scores / COLLAPSE_TAU).softmax(axis=-1)
        prototype = match_weights @ self.collapse_codebook_values

        # 2. RESIDUAL
        residual = x_f - prototype

        # 3. QUANTIZE — operation-conditioned gate
        # importance: (B, T, hidden) — values in (0, 1) via sigmoid
        importance = (prototype @ self.collapse_gate_w + self.collapse_gate_b).sigmoid()
        residual_gated = residual * importance

        # 4. ENCODE — compress to waist_dim
        waist_compressed = residual_gated @ self.collapse_proj_down  # (B, T, waist_dim)

        # 5. RECONSTRUCT
        residual_decompressed = waist_compressed @ self.collapse_proj_up  # (B, T, hidden)
        correction = prototype + residual_decompressed
        out = (x_f + self.collapse_alpha * correction).cast(x.dtype)

        # Cache last-call match_weights entropy for logging (one scalar, cheap)
        # entropy = -Σ p log p, averaged across (B, T)
        ent = -(match_weights * (match_weights + 1e-12).log()).sum(axis=-1).mean()
        self._collapse_last_match_entropy = ent

        if return_compressed:
            return out, waist_compressed.cast(x.dtype)
        return out

    def apply_collapse_v70(self, x: Tensor, breath_idx: int, return_compressed: bool = False) -> Tensor:
        """v70 collapse: lagging-safe lossy compression at WAIST dim (perf-tuned).

        Pipeline (codebook + gate operate at WAIST dim, not input):
          1. COMPRESS    — compressed_x = x @ proj_down                           (B, T, 512)
          2. TRANSFORM   — prototype = softmax(compressed_x @ keys.T / τ) @ values (B, T, 512)
          3. RESIDUAL    — residual = compressed_x − prototype                     (B, T, 512)
          4. GATE        — importance = sigmoid(prototype @ Wp + be @ Wb + b)      (B, T, 512)
          5. WAIST       — waist_compressed = (residual × importance) + prototype  (B, T, 512)
          6. DECOMPRESS  — decompressed = waist_compressed @ proj_up + bias        (B, T, 1024)
          7. RESIDUAL    — out = x + α × decompressed   (α=0 init → identity)

        Sparsity loss (stored in self._collapse_v70_sparsity_list, summed by trainer):
          target_frac = max(BUDGET_MIN, BUDGET_START − BUDGET_DECAY · breath_idx)
          violation = relu(importance.mean() − target_frac)
          loss = SPARSITY_WEIGHT · violation²

        PERF changes vs v70-orig:
          - codebook/gate moved to waist dim (3 matmuls cut in half)
          - gate_w split into two separate params (no slice-view backward scatter-add)
          - sparsity collected in Python list, summed by trainer outside the per-breath chain
        """
        x_f = x.cast(dtypes.float)
        # 1. COMPRESS — project x to waist dim
        compressed_x = x_f @ self.collapse_v70_proj_down  # (B, T, 512)

        # 2. TRANSFORM — codebook match against compressed rep
        cb_d = float(self.collapse_v70_codebook_keys.shape[-1])
        scores = (compressed_x @ self.collapse_v70_codebook_keys.T) / (cb_d ** 0.5)  # (B, T, N)
        match_weights = (scores / COLLAPSE_V70_TAU).softmax(axis=-1)  # (B, T, N)
        prototype = match_weights @ self.collapse_v70_codebook_values  # (B, T, waist)

        # 3. RESIDUAL — what's not explained by any codebook prototype
        residual = compressed_x - prototype  # (B, T, waist)

        # 4. GATE — breath-conditioned per-dim importance over the waist dim.
        # Two separate matmuls (no Tensor.cat with expanded view); no slice-view backward.
        be_idx = max(0, min(breath_idx, self.collapse_v70_breath_embed.shape[0] - 1))
        be_vec = self.collapse_v70_breath_embed[be_idx]  # (be_d,)
        gate_proto = prototype @ self.collapse_v70_gate_w_proto                       # (B, T, waist)
        gate_breath = (be_vec.reshape(1, -1) @ self.collapse_v70_gate_w_breath).reshape(1, 1, -1)  # (1, 1, waist)
        importance = (gate_proto + gate_breath + self.collapse_v70_gate_b).sigmoid()  # (B, T, waist)

        # 5. WAIST — gate the residual, then add prototype back (codebook contribution kept).
        # Empirically: gating the WHOLE rep collapses information at high sparsity pressure.
        # Gating only the residual lets the codebook's structured contribution survive.
        waist_compressed = (residual * importance) + prototype  # (B, T, waist)

        # 6. DECOMPRESS
        decompressed = waist_compressed @ self.collapse_v70_proj_up + self.collapse_v70_bias  # (B, T, hidden)

        # 7. RESIDUAL block
        out = (x_f + self.collapse_v70_alpha * decompressed).cast(x.dtype)

        # --- diagnostics (computed only at breath 0; cheap reductions) ---
        if breath_idx == 0:
            ent = -(match_weights * (match_weights + 1e-12).log()).sum(axis=-1).mean()
            self._collapse_v70_last_match_entropy = ent
            self._collapse_v70_last_importance_mean = importance.mean()

        # --- sparsity loss: per-breath, collected in a Python list ---
        # Budget tightens with breath_idx. relu makes under-budget free (lagging-safe).
        imp_mean = importance.mean()
        target_frac = max(COLLAPSE_V70_BUDGET_MIN,
                          COLLAPSE_V70_BUDGET_START - COLLAPSE_V70_BUDGET_DECAY * breath_idx)
        violation = (imp_mean - target_frac).relu()
        sparsity_this = (violation * violation) * COLLAPSE_V70_SPARSITY_WEIGHT
        if breath_idx == 0:
            self._collapse_v70_sparsity_list = [sparsity_this]
        else:
            # In the JIT trace, each call appends a fresh graph node to a fresh list.
            # The trainer reads the list and sums it once after the K-loop.
            self._collapse_v70_sparsity_list.append(sparsity_this)

        if return_compressed:
            return out, waist_compressed.cast(x.dtype)
        return out

    def apply_collapse_v71(self, x: Tensor, breath_idx: int, return_compressed: bool = False) -> Tensor:
        """v71 collapse: v70 refined with stronger sparsity, lower gate bias, k-means codebook,
        and a cleaner controller signal at init.

        Pipeline:
          1. COMPRESS    — compressed_x = x @ proj_down                              (B, T, 512)
          2. TRANSFORM   — prototype = softmax(compressed_x @ keys.T / τ) @ values   (B, T, 512)
          3. GATE        — importance = sigmoid(prototype @ Wp + be @ Wb + b)        (B, T, 512)
          4. WAIST       — waist_compressed = compressed_x × importance               (B, T, 512)
                          ← controller reads this; NO prototype add-back. At init
                            (zero-init values → prototype=0; GATE_BIAS=1 → sigmoid=0.73)
                            this is a uniform scale of v66's compressed signal, no
                            codebook-dependent shift in the controller's input.
          5. DECOMPRESS  — decompressed = (waist_compressed + prototype) @ proj_up + bias
                          ← codebook participates here, additive to the gated rep.
                            With α=0 at init the whole correction is masked out → v66-identical.
          6. RESIDUAL    — out = x + α × decompressed   (α=0 init → identity)

        Sparsity loss (stored in self._collapse_v71_sparsity_list, summed by trainer):
          target_frac = max(BUDGET_MIN, BUDGET_START − BUDGET_DECAY · breath_idx)
          violation = relu(importance.mean() − target_frac)
          loss = SPARSITY_WEIGHT · violation²
        """
        x_f = x.cast(dtypes.float)
        # 1. COMPRESS — project x to waist dim
        compressed_x = x_f @ self.collapse_v71_proj_down  # (B, T, 512)

        # 2. TRANSFORM — codebook match against compressed rep
        cb_d = float(self.collapse_v71_codebook_keys.shape[-1])
        scores = (compressed_x @ self.collapse_v71_codebook_keys.T) / (cb_d ** 0.5)  # (B, T, N)
        match_weights = (scores / COLLAPSE_V71_TAU).softmax(axis=-1)  # (B, T, N)
        prototype = match_weights @ self.collapse_v71_codebook_values  # (B, T, waist)

        # 3. GATE — breath-conditioned per-dim importance over the waist dim.
        # Two separate matmuls (no Tensor.cat with expanded view); no slice-view backward.
        be_idx = max(0, min(breath_idx, self.collapse_v71_breath_embed.shape[0] - 1))
        be_vec = self.collapse_v71_breath_embed[be_idx]  # (be_d,)
        gate_proto = prototype @ self.collapse_v71_gate_w_proto                         # (B, T, waist)
        gate_breath = (be_vec.reshape(1, -1) @ self.collapse_v71_gate_w_breath).reshape(1, 1, -1)  # (1, 1, waist)
        importance = (gate_proto + gate_breath + self.collapse_v71_gate_b).sigmoid()    # (B, T, waist)

        # 4. WAIST — pure multiplicative gating of the compressed rep.
        # No prototype add-back here: the controller reads exactly importance × compressed_x.
        # At init (importance≈0.73 uniform), this is a uniform scaling of v66's signal —
        # identical signal SHAPE to v66, just attenuated. As importance learns, dims drop
        # asymmetrically. The codebook contributes through decompression below, not here.
        waist_compressed = compressed_x * importance  # (B, T, waist)

        # 5. DECOMPRESS — codebook contributes via additive prototype before projection up.
        # At init (values=0 → prototype=0), this is identical to waist_compressed @ proj_up + bias.
        decompressed = (waist_compressed + prototype) @ self.collapse_v71_proj_up + self.collapse_v71_bias  # (B, T, hidden)

        # 6. RESIDUAL block
        out = (x_f + self.collapse_v71_alpha * decompressed).cast(x.dtype)

        # --- diagnostics (computed only at breath 0; cheap reductions) ---
        if breath_idx == 0:
            ent = -(match_weights * (match_weights + 1e-12).log()).sum(axis=-1).mean()
            self._collapse_v71_last_match_entropy = ent
            self._collapse_v71_last_importance_mean = importance.mean()

        # --- sparsity loss: per-breath, collected in a Python list ---
        # Budget tightens with breath_idx. relu makes under-budget free (lagging-safe).
        imp_mean = importance.mean()
        target_frac = max(COLLAPSE_V71_BUDGET_MIN,
                          COLLAPSE_V71_BUDGET_START - COLLAPSE_V71_BUDGET_DECAY * breath_idx)
        violation = (imp_mean - target_frac).relu()
        sparsity_this = (violation * violation) * COLLAPSE_V71_SPARSITY_WEIGHT
        if breath_idx == 0:
            self._collapse_v71_sparsity_list = [sparsity_this]
        else:
            self._collapse_v71_sparsity_list.append(sparsity_this)

        if return_compressed:
            return out, waist_compressed.cast(x.dtype)
        return out

    def dag_notebook_init_storage(self, B: int) -> Tensor:
        """v61 DAG: allocate per-forward storage. Shape (B, max_loops, NB_DIM)."""
        return Tensor.zeros((B, self.cfg.max_loops, self.nb_dim), dtype=dtypes.float).contiguous()

    def dag_notebook_read(self, x_in: Tensor, storage: Tensor, breath_idx: int) -> Tensor:
        """v61 DAG: multi-head cross-attention from x_in to storage with causal mask."""
        B = x_in.shape[0]
        T = x_in.shape[1]
        D = self.nb_dim
        h = self.nb_dag_n_heads
        hd = self.nb_dag_head_dim
        max_loops = self.cfg.max_loops

        x_f = x_in.cast(dtypes.float)
        storage_kv = storage
        if NOTEBOOK_DAG_POS_EMBED:
            storage_kv = storage_kv + self.nb_dag_pos_embed.reshape(1, max_loops, D)

        q = x_f @ self.nb_dag_q_w + self.nb_dag_q_b           # (B, T, D)
        k = storage_kv @ self.nb_dag_k_w + self.nb_dag_k_b    # (B, max_loops, D)
        v = storage_kv @ self.nb_dag_v_w + self.nb_dag_v_b    # (B, max_loops, D)

        q = q.reshape(B, T, h, hd).permute(0, 2, 1, 3)
        k = k.reshape(B, max_loops, h, hd).permute(0, 2, 1, 3)
        v = v.reshape(B, max_loops, h, hd).permute(0, 2, 1, 3)

        scores = (q @ k.transpose(-2, -1)) * (hd ** -0.5)

        slot_idx = Tensor.arange(max_loops, dtype=dtypes.float)
        mask = (slot_idx < float(breath_idx)).reshape(1, 1, 1, max_loops).cast(dtypes.float)
        scores = scores + (1.0 - mask) * (-1e9)

        weights = scores.softmax(axis=-1)
        attended = weights @ v
        attended = attended.permute(0, 2, 1, 3).reshape(B, T, D)

        read_vec = attended @ self.nb_dag_o_w + self.nb_dag_o_b
        return read_vec.cast(x_in.dtype)

    def dag_notebook_write(self, x: Tensor, storage: Tensor, breath_idx: int) -> Tensor:
        """v61 DAG: attn-pool x, project to NB_DIM, write to slot via one-hot mask."""
        B = x.shape[0]
        D = self.nb_dim
        max_loops = self.cfg.max_loops

        x_f = x.cast(dtypes.float)
        scores = (x_f * self.nb_dag_write_query.reshape(1, 1, -1)).sum(axis=-1)
        weights = scores.softmax(axis=-1).reshape(B, -1, 1)
        x_pool = (x_f * weights).sum(axis=1)
        write_vec = x_pool @ self.nb_dag_write_w + self.nb_dag_write_b

        slot_idx = Tensor.arange(max_loops, dtype=dtypes.float)
        slot_hot = (slot_idx == float(breath_idx)).reshape(1, max_loops, 1).cast(dtypes.float)
        new_storage = storage * (1.0 - slot_hot) + slot_hot * write_vec.reshape(B, 1, D)
        return new_storage

    def parameters(self):
        # v68 TWO_PHASE: include ONLY active layer sets in optimizer params.
        # State_dict still includes all (for ckpt symmetry); but unused params
        # would have None grads → opt.step() unwrap assertion fails. Exclude them here.
        ps = []
        if TWO_PHASE:
            ps.extend(self.expand_shared.parameters())
            ps.extend(self.compress_shared.parameters())
            for layer in self.expand_layers:
                ps.extend(layer.parameters())
            for layer in self.compress_layers:
                ps.extend(layer.parameters())
        else:
            ps.extend(self.shared.parameters())
            for layer in self.layers:
                ps.extend(layer.parameters())
            # v44 doubled-layers: include set B params for ckpt symmetry
            for layer in self.layers_b:
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
        # v61 DAG notebook
        ps.append(self.nb_dag_q_w)
        ps.append(self.nb_dag_q_b)
        ps.append(self.nb_dag_k_w)
        ps.append(self.nb_dag_k_b)
        ps.append(self.nb_dag_v_w)
        ps.append(self.nb_dag_v_b)
        ps.append(self.nb_dag_o_w)
        ps.append(self.nb_dag_o_b)
        ps.append(self.nb_dag_write_w)
        ps.append(self.nb_dag_write_b)
        ps.append(self.nb_dag_write_query)
        ps.append(self.nb_dag_pos_embed)
        # v69 collapse mechanism — always in optimizer when allocated (inert at α=0 init).
        ps.append(self.collapse_codebook_keys)
        ps.append(self.collapse_codebook_values)
        ps.append(self.collapse_gate_w)
        ps.append(self.collapse_gate_b)
        ps.append(self.collapse_proj_down)
        ps.append(self.collapse_proj_up)
        ps.append(self.collapse_alpha)
        # v70 collapse — codebook + gate at waist dim. Gate split into proto/breath weights.
        ps.append(self.collapse_v70_codebook_keys)
        ps.append(self.collapse_v70_codebook_values)
        ps.append(self.collapse_v70_proj_down)
        ps.append(self.collapse_v70_proj_up)
        ps.append(self.collapse_v70_bias)
        ps.append(self.collapse_v70_gate_w_proto)
        ps.append(self.collapse_v70_gate_w_breath)
        ps.append(self.collapse_v70_gate_b)
        ps.append(self.collapse_v70_breath_embed)
        ps.append(self.collapse_v70_alpha)
        # legacy unified gate_w — kept for state-dict symmetry, never in forward path.
        # Excluded from parameters() so the optimizer doesn't touch it.
        # v71 collapse — same structural shape as v70; refined fixes per the v71 design.
        ps.append(self.collapse_v71_codebook_keys)
        ps.append(self.collapse_v71_codebook_values)
        ps.append(self.collapse_v71_proj_down)
        ps.append(self.collapse_v71_proj_up)
        ps.append(self.collapse_v71_bias)
        ps.append(self.collapse_v71_gate_w_proto)
        ps.append(self.collapse_v71_gate_w_breath)
        ps.append(self.collapse_v71_gate_b)
        ps.append(self.collapse_v71_breath_embed)
        ps.append(self.collapse_v71_alpha)
        ps.append(self.bfield_proj_down)
        ps.append(self.bfield_proj_up)
        ps.append(self.bfield_bias)
        ps.append(self.waist_codebook_keys)
        ps.append(self.waist_codebook_values)
        return ps

    def compute_handoff(self, x: Tensor) -> Tensor:
        """Linear projection of a breath output to its handoff vector for the next
        breath. Zero-init means this returns ~0 initially; gradient learns useful values."""
        x_f = x.cast(dtypes.float)
        return (x_f @ self.handoff_w + self.handoff_b).cast(x.dtype)

    def breathe_once(self, x: Tensor, loop_idx: int, temp_mult: float = 1.0,
                     return_waist_compressed: bool = False, n_loops: int | None = None,
                     attn_mask: Tensor | None = None):
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

        v75 MAX_STEP_SIZE > 0 / MAX_STEP_BASE > 0: capture the breath input here,
        then at the end of the function clip the per-token L2 norm of (output - input)
        to a per-breath bound. Bounded refinement: forces gradual evolution across K
        breaths. When MAX_STEP_BASE > 0, the bound follows a half-cosine schedule
        (k=0 base, k=K-1 min); else MAX_STEP_SIZE is used as a constant bound.
        n_loops is the K for this forward pass (used by the cosine schedule); falls
        back to cfg.max_loops if not supplied.

        v81 (2026-05-26) attn_mask: optional (B, S) float, 1.0 at valid (prompt)
        positions, 0.0 at answer-span positions. When provided, ALL self-attention
        layers within the breath restrict keys to mask=1 positions only. This blocks
        the teacher-forcing leak where main-attn at answer-span positions can read
        previously-emitted gold tokens via the standard causal triangular mask.
        Combined with V79's kv_mask on the cross-attn and notebook_pool_mask, this
        is what makes the v81 multi-head WaistController paradigm independent of the
        gold tokens in the input sequence (verified via diag_v81_masking_audit.py).
        """
        # v75: capture the breath input BEFORE any transformation (incl. breath_embed)
        # so the delta = (final output) - (breath input) over the FULL breath.
        _step_bound_active = MAX_STEP_BASE > 0.0 or MAX_STEP_SIZE > 0.0
        breath_in = x if _step_bound_active else None
        # Per-breath bound: half-cosine schedule when MAX_STEP_BASE > 0, else constant.
        if MAX_STEP_BASE > 0.0:
            _K = int(n_loops) if n_loops is not None else int(self.cfg.max_loops)
            if _K > 1:
                _max_step_k = MAX_STEP_MIN + (MAX_STEP_BASE - MAX_STEP_MIN) * math.cos(
                    math.pi / 2.0 * float(loop_idx) / float(_K - 1)
                )
            else:
                _max_step_k = MAX_STEP_BASE
        else:
            _max_step_k = MAX_STEP_SIZE
        alpha = self.rope._alpha_at(loop_idx, x.dtype)
        if BREATH_TIME_EMBED:
            x = x + self.breath_embed[loop_idx].reshape(1, 1, -1).cast(x.dtype)
        ac_base, asn_base = alpha
        n_phases = self.cfg.n_phases
        # v68 TWO_PHASE: explicit EXPAND (warm, broad) + COMPRESS (cool, sharp) phases.
        # When TWO_PHASE=1, iterate expand_layers at EXPAND_TEMP then compress_layers
        # at COMPRESS_TEMP. Layer indices flow 0..EXPAND_LAYERS-1 for pitch lookup,
        # then EXPAND_LAYERS..EXPAND_LAYERS+COMPRESS_LAYERS-1.
        if TWO_PHASE:
            active_layers = list(self.expand_layers) + list(self.compress_layers)
            per_layer_temp_override = ([EXPAND_TEMP] * EXPAND_LAYERS
                                       + [COMPRESS_TEMP] * COMPRESS_LAYERS)
        # v44 doubled-layers: pick Set A or Set B based on breath index
        # (first half-cycle → A, second half-cycle → B). Maps to E/B alternation
        # when ROPE_FULL_CIRCLE=1 (rotation 0→2π over max_loops breaths).
        elif DOUBLED_LAYERS and loop_idx >= (self.cfg.max_loops // 2):
            active_layers = self.layers_b
            per_layer_temp_override = None
        else:
            active_layers = self.layers
            per_layer_temp_override = None
        for layer_idx, layer in enumerate(active_layers):
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
            # --- per-layer temperature ---
            # v68 TWO_PHASE: structural temps (EXPAND_TEMP for first 4, COMPRESS_TEMP for last 2)
            # v24 PER_BREATH_TEMP: within-breath wave (different from TWO_PHASE; mutually exclusive)
            if per_layer_temp_override is not None:
                layer_temp = per_layer_temp_override[layer_idx]
            elif PER_BREATH_TEMP:
                layer_temp = _per_layer_temp_within_breath(layer_idx, n_phases)
            else:
                layer_temp = temp_mult
            x = layer(x, loop_idx, attn_mask=attn_mask, temp_mult=layer_temp, alpha=layer_alpha)
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
                x = self.apply_bfield_waist(x, loop_idx=loop_idx)
        # --- v71/v70/v69 COLLAPSE: lossy compression replaces B-field waist ---
        # v71 (current): v70 refined — stronger sparsity, lower gate bias, k-means codebook,
        #                cleaner controller signal (no prototype add-back to controller's read).
        # v70: fixed 512d waist + breath-conditioned gate + budget-violation sparsity.
        # v69 (legacy):  128d waist + uniform gate. All replace v66's B-field MLP.
        # v71 takes precedence if both flags are set.
        waist_compressed = None
        if COLLAPSE_V71:
            if return_waist_compressed:
                x, waist_compressed = self.apply_collapse_v71(x, loop_idx, return_compressed=True)
            else:
                x = self.apply_collapse_v71(x, loop_idx)
        elif COLLAPSE_V70:
            if return_waist_compressed:
                x, waist_compressed = self.apply_collapse_v70(x, loop_idx, return_compressed=True)
            else:
                x = self.apply_collapse_v70(x, loop_idx)
        elif COLLAPSE_V69:
            if return_waist_compressed:
                x, waist_compressed = self.apply_collapse_v69(x, return_compressed=True)
            else:
                x = self.apply_collapse_v69(x)
        elif BFIELD_WAIST > 0 and BFIELD_END_OF_BREATH:
            if return_waist_compressed:
                x, waist_compressed = self.apply_bfield_waist(x, return_compressed=True, loop_idx=loop_idx)
            else:
                x = self.apply_bfield_waist(x, loop_idx=loop_idx)
        # End-of-breath CRP — only when per-layer oscillation is OFF (otherwise the
        # last layer's per-layer CRP already did this).
        if CONSTANT_RADIUS and not BREATH_NORM_OSC:
            x_f = x.cast(dtypes.float)
            x_norm = (x_f.square().sum(axis=-1, keepdim=True) + 1e-6).sqrt()
            target = self.crp_target_norm
            mix = self.crp_mix_alpha
            x_proj = x_f * (target / x_norm)
            x = (x_f * (1.0 - mix) + x_proj * mix).cast(x.dtype)
        # v75 (2026-05-23, schedule 2026-05-24) Bounded per-breath delta. After
        # the full breath (layers + waist + CRP), clip the per-token L2 norm of
        # the residual delta. This is the diffusion-style "small step" constraint:
        # rep_K = rep_0 + Σ small_step_k. When MAX_STEP_BASE > 0, _max_step_k follows
        # a half-cosine schedule across K breaths (basin landing → refinement); else
        # _max_step_k == MAX_STEP_SIZE (legacy constant). When both are 0,
        # _step_bound_active is False, breath_in is None, and this block is skipped
        # (default v66 behavior, no extra ops). Use dtypes.float for the L2 reduction
        # (no .cast(dtypes.float32) — AM driver quirk). The clip().minimum() keeps
        # the scale ≤ 1, so the delta is shrunk only when |delta_l2| > _max_step_k.
        if _step_bound_active and breath_in is not None:
            out_dtype = x.dtype
            delta = (x - breath_in).cast(dtypes.float)
            delta_l2 = (delta.square().sum(axis=-1, keepdim=True) + 1e-12).sqrt()
            scale = (_max_step_k / (delta_l2 + 1e-6)).minimum(1.0)
            delta_clipped = delta * scale
            x = (breath_in.cast(dtypes.float) + delta_clipped).cast(out_dtype)
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
            x = self.breathe_once(x_in, l, temp_mult=_sine_temp_baseline(l, n_loops), n_loops=n_loops)
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
            # Stochastic depth: per-breath keep-mask scaling. Mask is 0 when dropped
            # and 1/(1-p) when kept, so E[contribution] is unchanged (ResNet-style).
            # gate_sum unchanged → output is an unbiased estimator of the no-drop mean.
            if STOCH_DEPTH_P > 0.0 and Tensor.training:
                keep_l = self.stoch_keep_mask[l].cast(x.dtype)
                integral = integral + beta_l * keep_l * x
            else:
                integral = integral + beta_l * x
            gate_sum += beta_l
        if return_notebook:
            return integral / gate_sum, notebook, notebook_r
        return integral / gate_sum


# ---------- v54 WaistController (Phase 1) ----------

class WaistController:
    """Small cross-attention text decoder that reads (compressed waist, prompt
    embeddings) → outputs vocab logits per position. Fires once per breath
    when CONTROLLER_DECODE=1.

    Architecture:
        waist_proj_up:  bf_w → hidden (e.g., 512 → 1024)
        for layer in cross_attn_layers (default 1):
            pre-LN, then cross-attn(Q=waist_proj, K/V=prompt_emb), residual
            pre-LN, then FFN (4× hidden), residual
        decode via TIED model.embed_out → vocab logits

    Trainable params (1 layer, hidden=1024, n_heads=16): ~13M (4×1024² for QKVO +
    2×1024×4096 for FFN + small projs). Tied embed_out → no extra decoder params.
    """
    def __init__(self, cfg, waist_dim: int, n_layers: int = 1):
        # Decoupled controller width: cfg.controller_hidden may differ from cfg.hidden.
        # At Pythia-410M (cfg.hidden=1024, default controller_hidden=1024), behavior is
        # identical to before. At Pythia-1B (cfg.hidden=2048, controller_hidden=1024),
        # cross-attn K/V are rectangular (2048→1024) and a final up-projection
        # (1024→2048) ensures embed_out compatibility (embed_out is cfg.hidden × vocab).
        H_base = cfg.hidden                           # input prompt dim (full base width)
        H_ctrl = getattr(cfg, "controller_hidden", H_base)  # internal controller width
        self.n_heads = cfg.n_heads
        self.head_dim = H_ctrl // cfg.n_heads          # head_dim from controller width
        self.cfg = cfg
        self.waist_dim = waist_dim
        self.n_layers = n_layers
        self.H_base = H_base
        self.H_ctrl = H_ctrl
        # Project waist up to controller_hidden (not base hidden).
        self.waist_up_w = (Tensor.randn(waist_dim, H_ctrl, dtype=dtypes.float) * 0.02).contiguous()
        self.waist_up_b = Tensor.zeros((H_ctrl,), dtype=dtypes.float).contiguous()
        # n_layers cross-attn blocks. Q in H_ctrl; K/V projected from H_base → H_ctrl.
        self.layers = []
        for _ in range(n_layers):
            self.layers.append({
                "wq":  (Tensor.randn(H_ctrl, H_ctrl, dtype=dtypes.float) * 0.02).contiguous(),
                "wk":  (Tensor.randn(H_base, H_ctrl, dtype=dtypes.float) * 0.02).contiguous(),  # rectangular
                "wv":  (Tensor.randn(H_base, H_ctrl, dtype=dtypes.float) * 0.02).contiguous(),  # rectangular
                "wo":  Tensor.zeros((H_ctrl, H_ctrl), dtype=dtypes.float).contiguous(),
                "wf1": (Tensor.randn(H_ctrl, H_ctrl * 4, dtype=dtypes.float) * 0.02).contiguous(),
                "wf2": Tensor.zeros((H_ctrl * 4, H_ctrl), dtype=dtypes.float).contiguous(),
                "ln1_g": Tensor.ones((H_ctrl,), dtype=dtypes.float).contiguous(),
                "ln1_b": Tensor.zeros((H_ctrl,), dtype=dtypes.float).contiguous(),
                "ln2_g": Tensor.ones((H_ctrl,), dtype=dtypes.float).contiguous(),
                "ln2_b": Tensor.zeros((H_ctrl,), dtype=dtypes.float).contiguous(),
            })
        # Final up-projection from controller_hidden → base_hidden so embed_out
        # (cfg.hidden × vocab) can be applied. When H_ctrl == H_base this is just
        # identity-shape; we still allocate the matrix so the state_dict has stable
        # keys across configs (zero-init so it starts as a learned identity-ish op).
        if H_ctrl != H_base:
            self.final_up_w = (Tensor.randn(H_ctrl, H_base, dtype=dtypes.float) * 0.02).contiguous()
            self.final_up_b = Tensor.zeros((H_base,), dtype=dtypes.float).contiguous()
        else:
            self.final_up_w = None  # not needed; saves params for 410M
            self.final_up_b = None
        # v64 K-position embedding: (max_loops, max_loops, waist_dim) tensor indexed
        # by (K_total - 1, k_idx). NON-ZERO init (randn × 0.02) so the embed has
        # immediate contribution → gradient signal can propagate. v63 had zero-init
        # which created a dead-bootstrap problem (no contribution → no gradient).
        # Slightly perturbs warm-start at step 0 but k_pos_embed magnitude is small
        # (0.02 vs waist values typically 0.1-1.0).
        K_POS_INIT = float(os.environ.get("K_POS_INIT_SCALE", "0.02"))
        if K_POS_INIT > 0.0:
            self.k_pos_embed = (Tensor.randn(cfg.max_loops, cfg.max_loops, waist_dim, dtype=dtypes.float) * K_POS_INIT).contiguous()
        else:
            self.k_pos_embed = Tensor.zeros((cfg.max_loops, cfg.max_loops, waist_dim), dtype=dtypes.float).contiguous()
        # v72 Copy (pointer-network) params — ALWAYS allocated for state_dict symmetry.
        # Gradient is inert unless WAIST_COPY=1 (the forward path skips the copy graph
        # when the env var is off; the L2 reg in the trainer keeps these defined).
        # copy_q_w: project decoder hidden → small copy-attn space (H_ctrl → H_c)
        # copy_k_w: project prompt embeddings → same space (H_base → H_c)
        # copy_gate_w: linear → 1; sigmoid gives mixing weight (0 = vocab only, 1 = copy only)
        H_c = WAIST_COPY_HIDDEN
        self.copy_h = H_c
        self.copy_q_w    = (Tensor.randn(H_ctrl, H_c, dtype=dtypes.float) * 0.02).contiguous()
        self.copy_k_w    = (Tensor.randn(H_base, H_c, dtype=dtypes.float) * 0.02).contiguous()
        # Gate weight: zero-init (no input-dependence at start). Gate BIAS is initialized
        # NEGATIVE so sigmoid(bias) ≈ 0.12, suppressing copy by default. Copy attention
        # learns quietly under low gate; once it has signal, gate pulls up per-token.
        # Without the negative bias, random-init p_copy noise drives gate to 0 before
        # copy attention can learn (dead-bootstrap, same failure as v69/v70 codebooks).
        self.copy_gate_w = Tensor.zeros((H_ctrl, 1), dtype=dtypes.float).contiguous()
        self.copy_gate_b = (Tensor.ones((1,), dtype=dtypes.float) * WAIST_COPY_GATE_BIAS_INIT).contiguous()
        # v81 (2026-05-26) Multi-head WaistController.
        # Four parallel heads (ops / types / args1 / args2). Each adds its own learned
        # additive offset to the shared post-cross-attn hidden (H_ctrl) then shares the
        # tied embed_out projection (H_base × vocab).
        # SIMPLIFIED vs the original MLP design (2 matmuls per head per breath = too
        # many graph nodes for the AMD JIT capture phase, which hung post-step-0):
        # each head just learns a single per-head additive vector (size H_ctrl).
        # zero-init so heads start with identical logits at step 0; CE gradient through
        # different per-head label arrays drives the per-head specialization.
        # ALWAYS allocated for state_dict symmetry; forward path uses them only when
        # MULTI_HEAD_WAIST=1.
        self.head_names = ["ops", "types", "args1", "args2"]
        self.head_mlps = []
        for _hi in range(4):
            self.head_mlps.append({
                "w1": Tensor.zeros((H_ctrl, H_ctrl), dtype=dtypes.float).contiguous(),  # unused, kept for ckpt symmetry
                "b1": Tensor.zeros((H_ctrl,), dtype=dtypes.float).contiguous(),         # additive offset (size H_ctrl)
                "w2": Tensor.zeros((H_ctrl, H_ctrl), dtype=dtypes.float).contiguous(),  # unused, kept for ckpt symmetry
                "b2": Tensor.zeros((H_ctrl,), dtype=dtypes.float).contiguous(),         # additive offset (size H_ctrl)
            })
        # Stashed outputs (set in forward when WAIST_COPY=1). Used by the trainer
        # to compute the mixed CE loss without changing the return signature.
        self._last_copy_attn = None
        self._last_copy_gate = None
        self._last_copy_scores = None  # pre-softmax scores for stable log_softmax in aux loss
        # v78b: stash post-softmax cross-attn weights of the LAST cross-attn layer
        # (head-mean) when WAIST_ATTN_SUPERVISION=1. Shape (B, T_q, T_kv). Used by
        # the trainer's attention-supervision aux loss to direct the cross-attention
        # toward matching digit positions in the prompt.
        self._last_cross_attn = None

    def forward(self, waist_compressed: Tensor, prompt_emb: Tensor, embed_out: Tensor,
                 k_idx: int | None = None, K_total: int | None = None,
                 prompt_dropout_mask: Tensor | None = None,
                 prompt_tokens: Tensor | None = None,
                 kv_mask: Tensor | None = None,
                 force_single_head: bool = False,
                 v96_table_kv: Tensor | None = None) -> Tensor:
        """waist_compressed: (B, T_q, waist_dim) — Q sequence length T_q can be 1 or full T
           prompt_emb:        (B, T_kv, H_base) — main model's embedding of the prompt
           embed_out:         (H_base, vocab) — main model's TIED output projection
           prompt_tokens:     (B, T_kv) int — required when WAIST_COPY=1 so callers
                              (trainer / eval) can scatter copy-attn back to vocab IDs.
                              When omitted, copy components are not stashed and behavior
                              matches v66/v71 exactly.
           kv_mask:           (B, T_kv) float — 1.0 at valid KV positions, 0.0 at invalid
                              (positions past `current_len` in autoregressive eval). When
                              provided, applies additive -1e4 to cross-attn scores at
                              invalid positions, eliminating "ghost" attention to zero-pad
                              positions that contain stale EOS/zero embeddings.
                              **This fixes the train-eval mismatch where training has gold
                              answer tokens in the answer-span positions of `prompt_emb`
                              but eval has zeros — see v78b inference bug 2026-05-25.**
                              When omitted, behavior matches all pre-v78c callers exactly.
        Returns: (B, T_q, vocab). T_q matches the Q input length (1 at inference per-position).
        Side effect when WAIST_COPY=1 AND prompt_tokens is not None:
           self._last_copy_attn = (B, T_q, T_kv) float — attention over prompt positions
           self._last_copy_gate = (B, T_q, 1)    float — mixing gate in (0, 1)
        """
        B = waist_compressed.shape[0]
        T_q = waist_compressed.shape[1]
        T_kv = prompt_emb.shape[1]
        H_ctrl = self.H_ctrl
        # v63 K-position embedding: add (K_total-1, k_idx) embedding to waist before
        # projecting up. Zero-init means initial behavior matches v60-take-2.
        if k_idx is not None and K_total is not None:
            kpos = self.k_pos_embed[K_total - 1, k_idx].reshape(1, 1, -1)  # (1, 1, waist_dim)
            waist_compressed = waist_compressed + kpos.cast(waist_compressed.dtype)
        # Project waist up to controller_hidden.
        x = (waist_compressed @ self.waist_up_w + self.waist_up_b).cast(dtypes.float)
        prompt_f = prompt_emb.cast(dtypes.float)
        # v96 (2026-05-28) optionally CONCATENATE the consolidation table KV stream
        # along the sequence axis BEFORE the cross-attn layer loop.
        # v96_table_kv: (B, K_table, H_base) — already projected by the caller.
        # kv_mask is extended to cover the table positions (all 1.0 = valid).
        if v96_table_kv is not None:
            T_table = v96_table_kv.shape[1]
            # Cast to match prompt_f dtype and concat along seq axis.
            v96_table_kv_f = v96_table_kv.cast(prompt_f.dtype)
            prompt_f = Tensor.cat(prompt_f, v96_table_kv_f, dim=1)
            T_kv = T_kv + T_table
            if kv_mask is not None:
                # Extend the kv_mask with all-1.0 for the table positions.
                table_mask = Tensor.ones((B, T_table), dtype=kv_mask.dtype)
                kv_mask = Tensor.cat(kv_mask, table_mask, dim=1)
        # v63 prompt-dropout: zero the prompt embeddings when mask is 0 (training-time
        # randomized). Forces the controller to learn from waist alone occasionally,
        # enabling proper CFG at inference later. mask shape: scalar Tensor (1,).
        if prompt_dropout_mask is not None:
            prompt_f = prompt_f * prompt_dropout_mask.cast(prompt_f.dtype).reshape(1, 1, 1)
        # v78b: reset the cross-attn stash slot every forward. The loop writes it
        # for the last layer when WAIST_ATTN_SUPERVISION=1; otherwise it stays None.
        self._last_cross_attn = None
        n_attn_layers = len(self.layers)
        for layer_idx, layer in enumerate(self.layers):
            # Pre-LN cross-attn (Q from x at H_ctrl, K/V from prompt at H_base → H_ctrl).
            x_n = _layernorm(x, layer["ln1_g"], layer["ln1_b"], self.cfg.layer_norm_eps)
            kv = prompt_f
            q = x_n @ layer["wq"]
            k = kv  @ layer["wk"]  # (B, T_kv, H_base) → (B, T_kv, H_ctrl)
            v = kv  @ layer["wv"]
            # Multi-head reshape — all internal heads work in controller_hidden.
            q = q.reshape(B, T_q,  self.n_heads, self.head_dim).transpose(1, 2)
            k = k.reshape(B, T_kv, self.n_heads, self.head_dim).transpose(1, 2)
            v = v.reshape(B, T_kv, self.n_heads, self.head_dim).transpose(1, 2)
            scale = self.head_dim ** -0.5
            scores = (q @ k.transpose(-2, -1)) * scale
            # v78c (2026-05-25) Apply optional kv_mask: at invalid KV positions, subtract
            # 1e4 from the score so the post-softmax attention there is ~0. Reshape mask
            # to broadcast over (B, n_heads, T_q, T_kv). Mask shape: (B, T_kv).
            if kv_mask is not None:
                # mask=1 valid, mask=0 invalid → additive penalty for invalid positions
                penalty = (1.0 - kv_mask.cast(scores.dtype)).reshape(B, 1, 1, T_kv) * (-1e4)
                scores = scores + penalty
            # Clamp pre-softmax scores to prevent inf/NaN at larger head_dim.
            attn_weights = scores.clip(-1e4, 1e4).softmax(axis=-1)   # (B, n_heads, T_q, T_kv)
            # v78b: stash head-mean attention of the LAST cross-attn layer for the
            # attention-supervision aux loss. Only when WAIST_ATTN_SUPERVISION=1, only on
            # the last layer (the layer immediately before the tied vocab head fires).
            if WAIST_ATTN_SUPERVISION and (layer_idx == n_attn_layers - 1):
                # Mean over heads → (B, T_q, T_kv). Gradient flows back through this stash
                # because we are inside the same forward graph the trainer wraps in JIT.
                self._last_cross_attn = attn_weights.mean(axis=1)
            attn = attn_weights @ v
            attn = attn.transpose(1, 2).reshape(B, T_q, H_ctrl)
            x = x + attn @ layer["wo"]
            # Pre-LN FFN
            x_n2 = _layernorm(x, layer["ln2_g"], layer["ln2_b"], self.cfg.layer_norm_eps)
            ffn = (x_n2 @ layer["wf1"]).gelu() @ layer["wf2"]
            x = x + ffn
        # --- v72 Copy attention (computed BEFORE the up-projection, on x at H_ctrl). ---
        # When WAIST_COPY=1 AND prompt_tokens passed, compute pointer attention over the
        # prompt positions and the sigmoid gate. Stashed on the instance so the trainer /
        # eval can mix into the final distribution without changing this method's signature.
        # When OFF, params still exist but are not touched in the forward graph (the L2 reg
        # in the trainer covers the gradient).
        if WAIST_COPY and prompt_tokens is not None:
            # x: (B, T_q, H_ctrl); prompt_f: (B, T_kv, H_base)
            copy_q = x @ self.copy_q_w                                  # (B, T_q, H_c)
            copy_k = prompt_f @ self.copy_k_w                           # (B, T_kv, H_c)
            copy_scale = float(self.copy_h) ** -0.5
            copy_scores = (copy_q @ copy_k.transpose(-1, -2)) * copy_scale   # (B, T_q, T_kv)
            # Mask out padding positions in the prompt. We use prompt_tokens == 0 as the
            # padding indicator (matches encoder's pad token, see scripts/l3_train.py).
            # Subtract a large additive mask BEFORE softmax to drive masked-position attn → 0.
            pad_mask = (prompt_tokens == 0).cast(dtypes.float)          # (B, T_kv) — 1.0 at pads
            additive = (pad_mask * -1e4).reshape(B, 1, T_kv)            # (B, 1, T_kv) → broadcasts to (B, T_q, T_kv)
            copy_scores = (copy_scores + additive).clip(-1e4, 1e4)
            copy_attn = copy_scores.softmax(axis=-1)                    # (B, T_q, T_kv)
            copy_gate = (x @ self.copy_gate_w + self.copy_gate_b).sigmoid()  # (B, T_q, 1)
            self._last_copy_attn = copy_attn
            self._last_copy_gate = copy_gate
            # Stash pre-softmax scores for numerically-stable log_softmax in the trainer's aux loss.
            self._last_copy_scores = copy_scores
        else:
            self._last_copy_attn = None
            self._last_copy_gate = None
            self._last_copy_scores = None
        # v81 (2026-05-26) Multi-head path: 4 parallel heads (ops/types/args1/args2)
        # each producing its own vocab logits through the shared tied embed_out.
        # The shared cross-attn backbone (x at H_ctrl) feeds all heads; each head adds
        # its own per-head learned BIAS VECTOR (b1 + b2) — no per-head matmul (the
        # MLP design hung the AMD JIT capture phase post-step-0). Heads start with
        # identical zero-bias logits and diverge through per-head CE supervision.
        # final_up_w handles the H_ctrl → H_base projection (shared across heads).
        # force_single_head=True allows callers (e.g. earlier-breath supervision) to skip
        # multi-head decode to reduce graph complexity. The K-1 breath always uses
        # multi-head (when MULTI_HEAD_WAIST=1) — that's the breath whose output assembles
        # the v81 B6 text.
        if MULTI_HEAD_WAIST and not force_single_head:
            head_logits = {}
            # Project once (shared); then add per-head bias before embed_out.
            if self.final_up_w is not None:
                x_proj_shared = x @ self.final_up_w + self.final_up_b   # (B, T_q, H_base)
                # Per-head additive bias must be in H_base. Use a thin map from H_ctrl
                # to H_base — but to keep AMD JIT happy, just use the b2 vector in H_ctrl
                # and broadcast-add to x_proj_shared via final_up_w (still 1 matmul/head).
                # Simpler: just add b1 vector directly to x (H_ctrl) and re-project.
                for hi, name in enumerate(self.head_names):
                    mlp = self.head_mlps[hi]
                    bias = (mlp["b1"] + mlp["b2"]).reshape(1, 1, -1)  # (1, 1, H_ctrl)
                    x_h = x + bias
                    x_h_proj = x_h @ self.final_up_w + self.final_up_b
                    head_logits[name] = x_h_proj @ embed_out.cast(dtypes.float)
            else:
                # H_ctrl == H_base; embed_out (H_base × vocab) applies directly.
                for hi, name in enumerate(self.head_names):
                    mlp = self.head_mlps[hi]
                    bias = (mlp["b1"] + mlp["b2"]).reshape(1, 1, -1)  # (1, 1, H_ctrl)
                    x_h = x + bias
                    head_logits[name] = x_h @ embed_out.cast(dtypes.float)
            return head_logits
        # Single-head legacy path.
        if self.final_up_w is not None:
            x = x @ self.final_up_w + self.final_up_b
        # Tied decode head — use main model's embed_out (H_base × vocab) for vocab projection.
        logits = x @ embed_out.cast(dtypes.float)
        return logits

    def parameters(self):
        ps = [self.waist_up_w, self.waist_up_b]
        for layer in self.layers:
            for v in layer.values():
                ps.append(v)
        if self.final_up_w is not None:
            ps.extend([self.final_up_w, self.final_up_b])
        ps.append(self.k_pos_embed)  # v63
        # v72 copy params — always listed for state-dict symmetry. Caller
        # (collect_params in scripts/l3_train.py) gates optimizer inclusion on
        # WAIST_COPY=1 so AdamW doesn't assert on None grad when the copy path is off.
        ps.extend([self.copy_q_w, self.copy_k_w, self.copy_gate_w, self.copy_gate_b])
        return ps

    def multi_head_parameters(self):
        """v81 multi-head MLP params. Separately listed so collect_params() can
        gate inclusion on MULTI_HEAD_WAIST=1 (AdamW would assert on None grad
        if these were always in the optimizer but the forward path skipped them)."""
        ps = []
        for mlp in self.head_mlps:
            ps.extend([mlp["w1"], mlp["b1"], mlp["w2"], mlp["b2"]])
        return ps


# ---------- v85 Queryable Slot Decoder (2026-05-27) ----------

class V85SlotDecoder:
    """Differentiable queryable structures: K_max DAG slots × per-slot heads.

    For each breath, reads `waist_compressed` (B, T, waist_dim) and a per-example
    pool of "queryable structures" (numbers_emb, ops_codebook, types_codebook),
    and emits per-slot logits for:
      - ops_logits (B, K_max, 4)
      - types_logits (B, K_max, 32)
      - args_logits[2] (B, K_max, N_max + K_max) — two heads, one per arg position
      - is_active_logits (B, K_max, 1) — sigmoid for soft insert/delete

    Slots fill in PARALLEL per breath. No AR. All breaths emit the same target
    structure. Soft commit: no argmax inside the breath loop.

    Architecture (parameter-frugal for AMD JIT):
      - slot_pos_embed (K_max, h) — learnable, identifies each slot k
      - slot_q_proj (waist_dim → h) — projects waist_compressed (mean-pooled
        over T at prompt positions) into slot query space
      - h_combine: slot_query[k] = (slot_pos_embed[k] + waist_pooled @ slot_q_proj).LN.gelu
      - ops_head_w (h → 4)
      - types_head_w (h → 32)
      - args_pointer_q_w (h → h_p)
      - args_pointer_k_w (h → h_p) shared across "numbers" / "dag_slots" keys
      - active_head_w (h → 1)

      Args pointer attention:
        slot_q_p = slot_query @ args_pointer_q_w        # (B, K_max, h_p)
        numbers_k = numbers_emb @ args_pointer_k_w      # (B, N_max, h_p)
        slot_k = slot_query @ args_pointer_k_w          # (B, K_max, h_p) — for dag args
        all_keys = concat(numbers_k, slot_k)            # (B, N_max + K_max, h_p)
        args_scores = (slot_q_p @ all_keys.T) * scale   # (B, K_max, N_max + K_max)
        causal mask for dag refs: slot k can only ref dag[k'] for k' < k.

    Two arg positions per slot. We have TWO pointer heads (args1_q_w, args2_q_w)
    that share the keys.

    `numbers_emb` is built outside this module from token-span pooling at the
    encoder side (see scripts/build_v85_data.py and l3_training.py for the pipe).
    """

    def __init__(self, cfg, waist_dim: int, K_max: int, N_max: int,
                 types_N: int = 32, h_pointer: int = 64):
        H = cfg.hidden
        self.cfg = cfg
        self.waist_dim = waist_dim
        self.K_max = K_max
        self.N_max = N_max
        self.types_N = types_N
        self.ops_N = 4
        self.h_pointer = h_pointer

        # Slot position embedding (per slot index). Small random init so slots
        # start distinguishable. Each row is a 1024d vector.
        #
        # v87 (2026-05-27): when V87_SLOT_POS_INIT_SCALE>0, use a meaningful
        # uniform init scale so slots START strongly differentiated. v86's
        # 0.02-scale randn was too weak — the GELU-pooled slot_query collapsed
        # symmetric across slots (cross-slot JSD ~0.001 at training). Bigger
        # init breaks symmetry before the args-binding gradient gets a chance
        # to fight it. Uniform family preserves the same scale-by-scale ablation.
        if V87_SLOT_POS_INIT_SCALE > 0.0:
            self.slot_pos_embed = Tensor.uniform(
                K_max, H, low=-V87_SLOT_POS_INIT_SCALE, high=V87_SLOT_POS_INIT_SCALE,
                dtype=dtypes.float).contiguous()
        else:
            self.slot_pos_embed = (Tensor.randn(K_max, H, dtype=dtypes.float) * 0.02).contiguous()

        # Waist pool projection. Reads the waist_compressed at a SINGLE pooled
        # position (controller reads in 1024d space the same way the controller
        # already does). 1024 -> 1024 zero-init so the slot_query starts equal
        # to slot_pos_embed alone (warm-start safe).
        # NOTE: waist_dim is e.g. 512 here.
        self.waist_pool_proj_w = Tensor.zeros((waist_dim, H), dtype=dtypes.float).contiguous()
        self.waist_pool_proj_b = Tensor.zeros((H,), dtype=dtypes.float).contiguous()

        # Per-slot pre-output LN (gain + bias). Stabilizes the per-slot vector
        # before the four output heads.
        self.slot_ln_g = Tensor.ones((H,), dtype=dtypes.float).contiguous()
        self.slot_ln_b = Tensor.zeros((H,), dtype=dtypes.float).contiguous()

        # Codebooks (queryable structures). Learnable.
        # ops_codebook: 4 × H, small random init.
        self.ops_codebook = (Tensor.randn(self.ops_N, H, dtype=dtypes.float) * 0.02).contiguous()
        # types_codebook: 32 × H. Init from IB centroids when available (in
        # trainer post-init), else random.
        self.types_codebook = (Tensor.randn(self.types_N, H, dtype=dtypes.float) * 0.02).contiguous()
        # Per-codebook scale for logit calibration.
        # Apply via @codebook.T direct projection: slot_q @ codebook.T → (B, K_max, N).

        # Output heads.
        # ops/types are matmuls against the codebooks; no separate weight matrix.
        self.ops_head_b = Tensor.zeros((self.ops_N,), dtype=dtypes.float).contiguous()
        self.types_head_b = Tensor.zeros((self.types_N,), dtype=dtypes.float).contiguous()

        # Args pointer pieces (two arg positions per slot).
        self.args1_q_w = (Tensor.randn(H, h_pointer, dtype=dtypes.float) * 0.02).contiguous()
        self.args2_q_w = (Tensor.randn(H, h_pointer, dtype=dtypes.float) * 0.02).contiguous()
        self.args_k_w  = (Tensor.randn(H, h_pointer, dtype=dtypes.float) * 0.02).contiguous()

        # v86 (2026-05-27) Per-slot cross-attn over full waist sequence for args.
        # When V86_ARGS_CROSS_ATTN=1, args1/args2 first build a per-slot context
        # via cross-attn(slot_query, waist_full), then use THAT context for the
        # pointer logits. The cross-attn keys/values project waist_dim → H so
        # the result is a per-slot 1024d vector in the same space as slot_query,
        # which can then go through args1_q_w / args2_q_w for the pointer dot
        # products with numbers/dag keys.
        #
        # Init scheme:
        #   - v86_args_q_proj: small random (slot_query -> h_pointer-d query).
        #   - v86_args_k_proj: zero-init -> attn is uniform initially -> waist_ctx
        #     starts equal to mean(waist_full @ v86_args_v_proj). This is benign
        #     for warm-start (v86_args_v_proj is also zero-init, so waist_ctx=0
        #     and the args path falls back to the original v85 behavior).
        #   - v86_args_v_proj: zero-init -> waist_ctx = 0 at start.
        # With v_proj zero, args path uses slot_query (unchanged from v85). The
        # gradient lifts both q and v together as the model learns to bind.
        h_p_attn = h_pointer  # use same width for cross-attn q/k dim
        self.v86_args_q_proj = (Tensor.randn(H, h_p_attn, dtype=dtypes.float) * 0.02).contiguous()
        self.v86_args_k_proj = Tensor.zeros((waist_dim, h_p_attn), dtype=dtypes.float).contiguous()
        self.v86_args_v_proj = Tensor.zeros((waist_dim, H), dtype=dtypes.float).contiguous()
        # v89 (2026-05-27): split args1/args2 cross-attn for supervised attention.
        # Two parallel cross-attentions (separate K/V projections) so args1 and
        # args2 can peak at DIFFERENT positions. When V89_SUPERVISED_ATTN=0 these
        # are inert (the v86 single-shared attn path is used instead) but are
        # always allocated so the parameter list / state_dict are stable. The
        # trainer can optionally inherit values from v86 K/V projections via
        # V89_INHERIT_V86=1 after load_state_dict.
        if V89_SUPERVISED_ATTN:
            self.v89_args1_k_proj = (Tensor.randn(waist_dim, h_p_attn, dtype=dtypes.float) * V89_PROJ_INIT_SCALE).contiguous()
            self.v89_args1_v_proj = (Tensor.randn(waist_dim, H, dtype=dtypes.float) * V89_PROJ_INIT_SCALE).contiguous()
            self.v89_args2_k_proj = (Tensor.randn(waist_dim, h_p_attn, dtype=dtypes.float) * V89_PROJ_INIT_SCALE).contiguous()
            self.v89_args2_v_proj = (Tensor.randn(waist_dim, H, dtype=dtypes.float) * V89_PROJ_INIT_SCALE).contiguous()
        else:
            self.v89_args1_k_proj = Tensor.zeros((waist_dim, h_p_attn), dtype=dtypes.float).contiguous()
            self.v89_args1_v_proj = Tensor.zeros((waist_dim, H), dtype=dtypes.float).contiguous()
            self.v89_args2_k_proj = Tensor.zeros((waist_dim, h_p_attn), dtype=dtypes.float).contiguous()
            self.v89_args2_v_proj = Tensor.zeros((waist_dim, H), dtype=dtypes.float).contiguous()
        # Per-slot positional embedding ADDED to slot_query before the args
        # cross-attn (zero-init so warm-start is byte-identical; the existing
        # slot_pos_embed in slot_query already differentiates slots so this is
        # a redundant safeguard if slot_pos_embed degenerates).
        #
        # v87 (2026-05-27): when V87_SLOT_POS_INIT_SCALE>0, init at meaningful
        # uniform scale matching slot_pos_embed. Provides a second, additive
        # source of slot differentiation directly INSIDE the args cross-attn
        # query — independent of the slot_pos_embed→LN→GELU pathway which may
        # have collapsed the per-slot signal.
        if V87_SLOT_POS_INIT_SCALE > 0.0:
            self.v86_args_slot_pos = Tensor.uniform(
                K_max, H, low=-V87_SLOT_POS_INIT_SCALE, high=V87_SLOT_POS_INIT_SCALE,
                dtype=dtypes.float).contiguous()
        else:
            self.v86_args_slot_pos = Tensor.zeros((K_max, H), dtype=dtypes.float).contiguous()

        # v91 (2026-05-27) Args position embedding — (2, H) zero-init. Distinguishes
        # args1 (index 0) from args2 (index 1) when building the per-arg query in
        # the simplified pathway. Zero-init means args1 == args2 query at step 0,
        # so the args1/args2 supervised CE gradients will pull these in different
        # directions immediately.
        self.arg_pos_emb = Tensor.zeros((2, H), dtype=dtypes.float).contiguous()

        # Active head (binary). Initialized at zero-bias → sigmoid(0) = 0.5 (neutral).
        self.active_head_w = Tensor.zeros((H, 1), dtype=dtypes.float).contiguous()
        self.active_head_b = Tensor.zeros((1,), dtype=dtypes.float).contiguous()

    def forward(self, waist_compressed: Tensor, numbers_emb: Tensor,
                numbers_mask: Tensor, waist_full: Tensor = None,
                waist_full_mask: Tensor = None):
        """Compute per-slot logits for one breath.

        Args:
          waist_compressed: (B, T, waist_dim) — current breath's compressed waist.
            Usually the caller passes (B, 1, waist_dim) mean-pooled over prompt
            positions; the forward internally mean-pools again (no-op if T=1).
          numbers_emb: (B, N_max, H) — per-number embedding (zero-padded for
            non-existent numbers). Constructed outside (token-span pool).
          numbers_mask: (B, N_max) float — 1.0 at valid number positions, 0.0 at pads.
          waist_full: (B, T_full, waist_dim) — full waist sequence used by v86
            args cross-attn. Only consulted when V86_ARGS_CROSS_ATTN is on AND
            this is not None.
          waist_full_mask: (B, T_full) — 1.0 at valid (prompt) positions, 0.0
            at pads. Used to mask the args cross-attn.

        Returns dict:
          ops_logits:    (B, K_max, 4)
          types_logits:  (B, K_max, 32)
          args1_logits:  (B, K_max, N_max + K_max)
          args2_logits:  (B, K_max, N_max + K_max)
          active_logits: (B, K_max)   — raw logits, sigmoid at loss time
          slot_state:    (B, K_max, H) — used for diagnostics
          args_attn:     (B, K_max, T_full) — per-slot cross-attn weights (v86
            only, else absent). For diagnostics. v89: returns the args1+args2
            average for backward-compat with existing diag scripts.
          args1_attn:    (B, K_max, T_full) — per-arg cross-attn (v89 only;
            supervised by gold args1 token position).
          args2_attn:    (B, K_max, T_full) — same for args2.

        Soft commit: no argmax. All outputs are mixtures. At eval, the decoder
        argmaxes after the final breath.
        """
        B = waist_compressed.shape[0]
        T = waist_compressed.shape[1]
        H = self.cfg.hidden
        K_max = self.K_max
        N_max = self.N_max

        # Pool the waist over T positions (mean across the whole sequence). For
        # prompt-only conditioning, the caller should pre-mask the waist by the
        # prompt_mask before passing in. Cheaper than per-slot cross-attn.
        # mean-pool: (B, waist_dim)
        waist_f = waist_compressed.cast(dtypes.float)
        waist_pooled = waist_f.mean(axis=1)  # (B, waist_dim)
        # Project to H.
        waist_proj = waist_pooled @ self.waist_pool_proj_w + self.waist_pool_proj_b  # (B, H)
        # Broadcast-add to slot_pos_embed.
        # slot_query[k] = slot_pos_embed[k] + waist_proj
        slot_query = self.slot_pos_embed.reshape(1, K_max, H) + waist_proj.reshape(B, 1, H)
        # LN then GELU.
        slot_query = _layernorm(slot_query, self.slot_ln_g, self.slot_ln_b,
                                 self.cfg.layer_norm_eps).gelu()  # (B, K_max, H)

        # Ops logits (matmul against ops_codebook).
        ops_logits = slot_query @ self.ops_codebook.T + self.ops_head_b.reshape(1, 1, -1)
        # Types logits (matmul against types_codebook).
        types_logits = slot_query @ self.types_codebook.T + self.types_head_b.reshape(1, 1, -1)

        # v91 (2026-05-27): SIMPLIFIED args pathway — single einsum mirroring
        # ops_codebook. See module-level V91_SIMPLIFIED_ARGS doc. Early-return
        # before the v86/v89 cross-attn and pointer projections.
        if V91_SIMPLIFIED_ARGS:
            # args_codebook = concat(numbers_emb, slot_query) — per-problem dynamic.
            # numbers_emb: (B, N_max, H), slot_query: (B, K_max, H)
            numbers_f = numbers_emb.cast(dtypes.float)
            args_codebook = numbers_f.cat(slot_query, dim=1)   # (B, N_max + K_max, H)
            # arg_query = slot_query + arg_pos_emb broadcast over slot dim.
            # slot_query: (B, K_max, H) -> (B, K_max, 1, H)
            # arg_pos_emb: (2, H) -> (1, 1, 2, H)
            arg_query = (slot_query.reshape(B, K_max, 1, H)
                         + self.arg_pos_emb.reshape(1, 1, 2, H))   # (B, K_max, 2, H)
            # Args logits: one einsum, no intermediate projection.
            # arg_query (B, K_max, 2, H) · args_codebook (B, N_max+K_max, H)
            args_logits_both = Tensor.einsum(
                "bkih,bjh->bkij", arg_query, args_codebook)        # (B, K_max, 2, N_max+K_max)

            # Mask construction (same logic as legacy path).
            ones_k = Tensor.ones((K_max, K_max), dtype=dtypes.float)
            ltri = ones_k.tril()
            eye = Tensor.eye(K_max, dtype=dtypes.float)
            strict_ltri = ltri - eye   # (K_max, K_max), 1.0 below-diagonal
            strict_ltri_b = strict_ltri.reshape(1, K_max, K_max)
            num_valid = numbers_mask.cast(dtypes.float).reshape(B, 1, N_max)
            num_valid_bk = num_valid.expand(B, K_max, N_max)
            slot_valid_bk = strict_ltri_b.expand(B, K_max, K_max)
            all_valid = num_valid_bk.cat(slot_valid_bk, dim=2)   # (B, K_max, N_max+K_max)
            args_penalty = (1.0 - all_valid) * (-1e4)              # (B, K_max, N_max+K_max)

            # Same penalty applied to BOTH arg positions.
            args1_scores = (args_logits_both[:, :, 0, :] + args_penalty).clip(-1e4, 1e4)
            args2_scores = (args_logits_both[:, :, 1, :] + args_penalty).clip(-1e4, 1e4)

            # Active logits (unchanged from legacy path).
            active_logits = (slot_query @ self.active_head_w + self.active_head_b).reshape(B, K_max)

            return {
                "ops_logits":    ops_logits,
                "types_logits":  types_logits,
                "args1_logits":  args1_scores,
                "args2_logits":  args2_scores,
                "active_logits": active_logits,
                "slot_state":    slot_query,
            }

        # v86: per-slot cross-attn over full waist sequence to compute per-slot
        # positional context. Falls back to slot_query when v86 is off OR when
        # the caller didn't pass waist_full.
        #
        # v89 (2026-05-27): when V89_SUPERVISED_ATTN=1 AND we have waist_full,
        # SPLIT the cross-attn into two parallel attentions — one for args1 and
        # one for args2 — each with its OWN K/V projection. This lets the
        # supervised loss train args1 attn to peak at the gold args1 number
        # position AND args2 attn to peak at the gold args2 number position
        # (different positions per slot). Q-projection is shared (slot_query
        # the same for both arg positions). Output: TWO ctx tensors fed into
        # args1_q_w and args2_q_w respectively. Diagnostic: args1_attn,
        # args2_attn returned in the output dict for inspection and aux loss.
        args_attn_diag = None
        args1_attn_w = None
        args2_attn_w = None
        slot_query_for_args1 = slot_query
        slot_query_for_args2 = slot_query
        if V86_ARGS_CROSS_ATTN and (waist_full is not None):
            waist_full_f = waist_full.cast(dtypes.float)
            T_full = waist_full.shape[1]
            scale_attn = float(self.h_pointer) ** -0.5
            # Per-slot query: slot_query + per-slot pos emb (init scale set by V87
            # when v87 active, zero-init otherwise — see slot_pos init paths).
            q_in = slot_query + self.v86_args_slot_pos.reshape(1, K_max, H)
            q_attn = q_in @ self.v86_args_q_proj                 # (B, K_max, h_p_attn)
            # Mask construction shared across all cross-attn variants below.
            if waist_full_mask is not None:
                mask_kt = waist_full_mask.cast(dtypes.float).reshape(B, 1, T_full)
                attn_penalty = (1.0 - mask_kt) * (-1e4)
            else:
                attn_penalty = None

            if V89_SUPERVISED_ATTN:
                # Two parallel cross-attns: separate K/V for args1 and args2.
                # args1 path:
                k_attn1 = waist_full_f @ self.v89_args1_k_proj   # (B, T_full, h_p_attn)
                v_attn1 = waist_full_f @ self.v89_args1_v_proj   # (B, T_full, H)
                a1_scores = (q_attn @ k_attn1.transpose(-2, -1)) * scale_attn
                if attn_penalty is not None:
                    a1_scores_masked = (a1_scores + attn_penalty).clip(-1e4, 1e4)
                else:
                    a1_scores_masked = a1_scores
                attn_w1 = a1_scores_masked.softmax(axis=-1)      # (B, K_max, T_full)
                slot_args1_ctx = attn_w1 @ v_attn1               # (B, K_max, H)
                # args2 path:
                k_attn2 = waist_full_f @ self.v89_args2_k_proj
                v_attn2 = waist_full_f @ self.v89_args2_v_proj
                a2_scores = (q_attn @ k_attn2.transpose(-2, -1)) * scale_attn
                if attn_penalty is not None:
                    a2_scores_masked = (a2_scores + attn_penalty).clip(-1e4, 1e4)
                else:
                    a2_scores_masked = a2_scores
                attn_w2 = a2_scores_masked.softmax(axis=-1)
                slot_args2_ctx = attn_w2 @ v_attn2
                args1_attn_w = attn_w1
                args2_attn_w = attn_w2
                # Also stash masked pre-softmax scores so trainer can compute
                # supervised CE via sparse_categorical_crossentropy directly.
                args1_attn_scores_v89 = a1_scores_masked
                args2_attn_scores_v89 = a2_scores_masked
                # diag: average for backward compat (eval/diag scripts read 'args_attn')
                args_attn_diag = (attn_w1 + attn_w2) * 0.5
                slot_query_for_args1 = slot_query + slot_args1_ctx
                slot_query_for_args2 = slot_query + slot_args2_ctx
            else:
                # v86 single shared cross-attn (warm-start-safe legacy path).
                k_attn = waist_full_f @ self.v86_args_k_proj         # (B, T_full, h_p_attn)
                v_attn = waist_full_f @ self.v86_args_v_proj         # (B, T_full, H)
                attn_scores = (q_attn @ k_attn.transpose(-2, -1)) * scale_attn
                if attn_penalty is not None:
                    attn_scores = (attn_scores + attn_penalty).clip(-1e4, 1e4)
                attn_w = attn_scores.softmax(axis=-1)
                slot_args_ctx = attn_w @ v_attn
                args_attn_diag = attn_w
                # The pointer "query" for args1/args2 is now the per-slot context
                # (v86_args_v_proj zero-init means this starts at 0 — neutral).
                # Use a residual: slot_query + slot_args_ctx so that the args still
                # have a usable per-slot signal even at warm-start (slot_query has
                # the per-slot info via slot_pos_embed).
                slot_query_for_args1 = slot_query + slot_args_ctx
                slot_query_for_args2 = slot_query + slot_args_ctx

        # Args pointer attention.
        # numbers_emb: (B, N_max, H). Pad numbers (mask=0) get a large negative
        # contribution via additive penalty after softmax score computation.
        numbers_f = numbers_emb.cast(dtypes.float)
        # All keys = concat(numbers_emb, slot_query) → (B, N_max + K_max, H)
        # Use pointer projections.
        slot_k = slot_query @ self.args_k_w        # (B, K_max, h_p)
        num_k  = numbers_f @ self.args_k_w         # (B, N_max, h_p)
        all_k  = num_k.cat(slot_k, dim=1)          # (B, N_max + K_max, h_p)

        scale = float(self.h_pointer) ** -0.5

        # args1 pointer — uses args1-specific slot_query (v89) or shared (v86 path).
        slot_q1 = slot_query_for_args1 @ self.args1_q_w      # (B, K_max, h_p)
        args1_scores = (slot_q1 @ all_k.transpose(-2, -1)) * scale   # (B, K_max, N_max + K_max)

        # args2 pointer — uses args2-specific slot_query (v89) or shared (v86 path).
        slot_q2 = slot_query_for_args2 @ self.args2_q_w
        args2_scores = (slot_q2 @ all_k.transpose(-2, -1)) * scale

        # Build the mask over (N_max + K_max) keys.
        # Numbers section: valid where numbers_mask == 1.
        # Slot section: valid where target_slot_idx < query_slot_idx (causal — can
        # reference earlier slots only). To realize this without per-row Python
        # loops, build a (K_max, K_max) lower-triangular STRICT mask:
        #   ltri_strict[q_idx, k_idx] = 1 if k_idx < q_idx else 0
        # Broadcast: numbers part is (B, K_max, N_max) — same mask per q,
        # slot part is (B, K_max, K_max) — causal mask.
        # We construct this mask numerically inside this forward (single small
        # tensor op — Tensor.tril produces a triangular).
        ones_k = Tensor.ones((K_max, K_max), dtype=dtypes.float)
        # Strict lower triangle (k_idx < q_idx): off-diagonal lower triangle.
        # tril includes diagonal; we want STRICT lower → tril - eye.
        ltri = ones_k.tril()
        eye = Tensor.eye(K_max, dtype=dtypes.float)
        strict_ltri = ltri - eye   # (K_max, K_max), 1.0 below-diagonal
        # Reshape for broadcasting in the (B, K_max, N_max + K_max) score tensor.
        strict_ltri_b = strict_ltri.reshape(1, K_max, K_max)

        # Numbers mask: (B, 1, N_max). 1 valid, 0 pad.
        num_valid = numbers_mask.cast(dtypes.float).reshape(B, 1, N_max)

        # Build combined validity: (B, K_max, N_max + K_max).
        # Numbers segment is the SAME for every query slot k → broadcast over K_max.
        # Slot segment is per-query (causal).
        num_valid_bk = num_valid.expand(B, K_max, N_max)
        slot_valid_bk = strict_ltri_b.expand(B, K_max, K_max)
        all_valid = num_valid_bk.cat(slot_valid_bk, dim=2)  # (B, K_max, N_max + K_max)

        # Apply additive -1e4 at invalid positions.
        # NOTE: at q_idx=0, ALL slot keys are invalid (strict_ltri row 0 is all
        # zero). That's fine — slot 0 should only refer to numbers.
        args_penalty = (1.0 - all_valid) * (-1e4)
        args1_scores = (args1_scores + args_penalty).clip(-1e4, 1e4)
        args2_scores = (args2_scores + args_penalty).clip(-1e4, 1e4)

        # Active logits.
        active_logits = (slot_query @ self.active_head_w + self.active_head_b).reshape(B, K_max)

        out = {
            "ops_logits":    ops_logits,
            "types_logits":  types_logits,
            "args1_logits":  args1_scores,
            "args2_logits":  args2_scores,
            "active_logits": active_logits,
            "slot_state":    slot_query,
        }
        if args_attn_diag is not None:
            out["args_attn"] = args_attn_diag
        # v89 (2026-05-27): emit per-arg cross-attn distributions for aux
        # supervision. Only present when V89_SUPERVISED_ATTN=1 (else None).
        if args1_attn_w is not None:
            out["args1_attn"] = args1_attn_w
            out["args2_attn"] = args2_attn_w
            # Pre-softmax masked scores — used by the trainer to compute
            # sparse_categorical_crossentropy against the gold token positions.
            out["args1_attn_scores"] = args1_attn_scores_v89
            out["args2_attn_scores"] = args2_attn_scores_v89
        return out

    def parameters(self):
        # v91 (2026-05-27): when V91_SIMPLIFIED_ARGS=1, only arg_pos_emb participates
        # in the args pathway; the legacy projections (args*_q_w, args_k_w,
        # v86_args_*, v89_args*) stay allocated for state-dict compat but are NOT
        # in parameters() so AdamW doesn't track them and their grads stay None.
        if V91_SIMPLIFIED_ARGS:
            return [
                self.slot_pos_embed,
                self.waist_pool_proj_w, self.waist_pool_proj_b,
                self.slot_ln_g, self.slot_ln_b,
                self.ops_codebook, self.ops_head_b,
                self.types_codebook, self.types_head_b,
                self.arg_pos_emb,
                self.active_head_w, self.active_head_b,
            ]
        return [
            self.slot_pos_embed,
            self.waist_pool_proj_w, self.waist_pool_proj_b,
            self.slot_ln_g, self.slot_ln_b,
            self.ops_codebook, self.ops_head_b,
            self.types_codebook, self.types_head_b,
            self.args1_q_w, self.args2_q_w, self.args_k_w,
            # v86 per-slot cross-attn for args binding (always in opt list when
            # V85_QUERYABLE so AdamW has gradients; inert when V86_ARGS_CROSS_ATTN=0).
            self.v86_args_q_proj, self.v86_args_k_proj, self.v86_args_v_proj,
            self.v86_args_slot_pos,
            # v89 split args1/args2 cross-attn K/V (always allocated for state-dict
            # symmetry; inert when V89_SUPERVISED_ATTN=0 since they're zero-init).
            self.v89_args1_k_proj, self.v89_args1_v_proj,
            self.v89_args2_k_proj, self.v89_args2_v_proj,
            self.active_head_w, self.active_head_b,
        ]


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
        # v54 WaistController — small cross-attention decoder over (waist, prompt).
        # Allocated always (state_dict symmetry); only USED when CONTROLLER_DECODE=1.
        # bf_w = waist dim (max 1 for symmetry if BFIELD_WAIST=0).
        # v69/v70/v71: waist dim for the controller depends on which collapse mechanism is active.
        if COLLAPSE_V71:
            _bf_w_for_wc = COLLAPSE_V71_WAIST_DIM
        elif COLLAPSE_V70:
            _bf_w_for_wc = COLLAPSE_V70_WAIST_DIM
        elif COLLAPSE_V69:
            _bf_w_for_wc = COLLAPSE_WAIST_DIM
        else:
            _bf_w_for_wc = max(1, BFIELD_WAIST)
        self.waist_controller = WaistController(cfg, waist_dim=_bf_w_for_wc, n_layers=CONTROLLER_N_LAYERS)
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
        # v65 boundary head: per-position scalar predicting "is the next token ####?".
        # Applied to per_breath_x[k] (1024d hidden state at end of breath k). Always
        # allocated for state-dict symmetry; gradient is inert when BOUNDARY_AUX_WEIGHT=0.
        self.boundary_head_w = (Tensor.randn(cfg.hidden, 1, dtype=dtypes.float) * 0.02).contiguous()
        self.boundary_head_b = Tensor.zeros((1,), dtype=dtypes.float).contiguous()

        # v85 (2026-05-27) Queryable slot decoder. Always allocated for state-dict
        # symmetry; only USED when V85_QUERYABLE=1 (the trainer/eval gates the
        # forward path on the env var). When OFF, gradient is inert.
        self.v85_slot_decoder = V85SlotDecoder(
            cfg, waist_dim=_bf_w_for_wc, K_max=V85_K_MAX, N_max=V85_N_MAX,
            types_N=V85_TYPES_N)
        # v85 numbers/verbs span encoder is just a token-span pool — no learnable
        # weights needed beyond what the model already has (we pool embed_in
        # rows over character-aligned token spans, externally in the trainer).

        # v96 (2026-05-28) CONSOLIDATION TABLE params — ALWAYS allocated for
        # state_dict symmetry; gradient is inert when V96_CONSOLIDATION=0
        # (the forward path skips them; the L2 reg in the trainer keeps
        # gradients defined so AdamW doesn't assert).
        from mycelium.v96 import make_v96_params as _v96_make
        v96p = _v96_make(cfg.hidden)
        self.v96_gate_w         = v96p["v96_gate_w"]
        self.v96_gate_b         = v96p["v96_gate_b"]
        self.v96_ops_codebook   = v96p["v96_ops_codebook"]
        self.v96_types_codebook = v96p["v96_types_codebook"]
        self.v96_summary_proj   = v96p["v96_summary_proj"]
        self.v96_table_kv_proj  = v96p["v96_table_kv_proj"]
        # v96.1: scale gate on table contribution (zero-init scalar)
        self.v96_table_alpha    = v96p["v96_table_alpha"]
        # v96.2 (2026-05-28) constraint check heads. Bombe-inspired elimination:
        # the model self-supervises on whether its own row's raw_summary respects
        # (a) causal reference validity and (b) non-commutative arg ordering.
        # Trained via constraint losses in the trainer.
        self.v96_ref_validity_head_w = v96p["v96_ref_validity_head_w"]
        self.v96_ref_validity_head_b = v96p["v96_ref_validity_head_b"]
        self.v96_arg_order_head_w    = v96p["v96_arg_order_head_w"]
        self.v96_arg_order_head_b    = v96p["v96_arg_order_head_b"]

        # v97 (2026-05-28) CALIBRATION HEAD — Bombe-inspired self-assessment.
        # Small Linear(1024 -> 1) head; reads each breath's attention-pooled
        # hidden state (post-final-LN, prompt-range pool via breath_embed[k] as
        # query) and predicts P(final answer correct). Pure aux loss — output
        # is NEVER fed back into the residual stream. Compatible with v96 OFF
        # paradigm. Always allocated for state_dict symmetry; participates in
        # opt list only when V97_CALIBRATION=1.
        h = cfg.hidden
        # Standard randn 0.02 init, bias zero — predict logit 0 → sigmoid=0.5 at start.
        self.v97_calib_head_w = (Tensor.randn(h, 1, dtype=dtypes.float) * 0.02).contiguous()
        self.v97_calib_head_b = Tensor.zeros((1,), dtype=dtypes.float).contiguous()

    def parameters(self):
        """Parameters trained on the main loss (transformer + lookup table +
        confidence head). The controller has gradient separation per the spec —
        its parameters are returned by controller_parameters() and trained by a
        separate optimizer (Step F) via REINFORCE on outcomes + auxiliary signals."""
        ps = ([self.embed.weight, self.ln_f_g, self.ln_f_b, self.embed_out]
                + self.block.parameters()
                + self.lookup_table.parameters()
                + self.confidence_head.parameters()
                + [self.waist_head_w, self.waist_head_b,
                   self.boundary_head_w, self.boundary_head_b]  # v65 boundary head
                + self.waist_controller.parameters())
        # v85 slot decoder params — listed here when V85_QUERYABLE is on so AdamW
        # actually trains them. When off they're idle (no gradient signal).
        if V85_QUERYABLE:
            ps = ps + self.v85_slot_decoder.parameters()
        # v96 consolidation-table params — listed here when V96_CONSOLIDATION is on
        # so AdamW trains them. State-dict registration is separate (always present
        # for ckpt symmetry — see state_dict()).
        if V96_CONSOLIDATION:
            ps = ps + [self.v96_gate_w, self.v96_gate_b,
                        self.v96_ops_codebook, self.v96_types_codebook,
                        self.v96_summary_proj, self.v96_table_kv_proj,
                        self.v96_table_alpha,    # v96.1
                        # v96.2 constraint check heads.
                        self.v96_ref_validity_head_w, self.v96_ref_validity_head_b,
                        self.v96_arg_order_head_w,    self.v96_arg_order_head_b]
        # v97 calibration head — listed when V97_CALIBRATION is on so AdamW trains it.
        # State-dict registration is always-on (see state_dict()).
        if V97_CALIBRATION:
            ps = ps + [self.v97_calib_head_w, self.v97_calib_head_b]
        return ps

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
            # v78 per-head model codebook — always saved for ckpt symmetry
            sd[f"phase{i}.v78_head_codebook"] = layer.v78_head_codebook
        # v44 doubled-layers: Set B params under phase{i}_b.* keys
        for i, layer in enumerate(self.block.layers_b):
            for a in ("wq", "bq", "wk", "bk", "w_in", "b_in"):
                sd[f"phase{i}_b.{a}"] = getattr(layer, a)
            sd[f"phase{i}_b.v78_head_codebook"] = layer.v78_head_codebook
        # v68 TWO_PHASE: expand_shared, compress_shared, and per-set phase layers.
        # Empty lists when TWO_PHASE=0 → no extra keys.
        sw_exp = self.block.expand_shared
        sw_cmp = self.block.compress_shared
        for a in ("wv", "bv", "wo", "bo", "w_out", "b_out",
                  "in_ln_g", "in_ln_b", "post_ln_g", "post_ln_b"):
            sd[f"expand_shared.{a}"] = getattr(sw_exp, a)
            sd[f"compress_shared.{a}"] = getattr(sw_cmp, a)
        for i, layer in enumerate(self.block.expand_layers):
            for a in ("wq", "bq", "wk", "bk", "w_in", "b_in"):
                sd[f"expand_phase{i}.{a}"] = getattr(layer, a)
            sd[f"expand_phase{i}.v78_head_codebook"] = layer.v78_head_codebook
        for i, layer in enumerate(self.block.compress_layers):
            for a in ("wq", "bq", "wk", "bk", "w_in", "b_in"):
                sd[f"compress_phase{i}.{a}"] = getattr(layer, a)
            sd[f"compress_phase{i}.v78_head_codebook"] = layer.v78_head_codebook
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
        # v61 DAG notebook
        sd["block.nb_dag_q_w"]         = self.block.nb_dag_q_w
        sd["block.nb_dag_q_b"]         = self.block.nb_dag_q_b
        sd["block.nb_dag_k_w"]         = self.block.nb_dag_k_w
        sd["block.nb_dag_k_b"]         = self.block.nb_dag_k_b
        sd["block.nb_dag_v_w"]         = self.block.nb_dag_v_w
        sd["block.nb_dag_v_b"]         = self.block.nb_dag_v_b
        sd["block.nb_dag_o_w"]         = self.block.nb_dag_o_w
        sd["block.nb_dag_o_b"]         = self.block.nb_dag_o_b
        sd["block.nb_dag_write_w"]     = self.block.nb_dag_write_w
        sd["block.nb_dag_write_b"]     = self.block.nb_dag_write_b
        sd["block.nb_dag_write_query"] = self.block.nb_dag_write_query
        sd["block.nb_dag_pos_embed"]   = self.block.nb_dag_pos_embed
        # v69 collapse
        sd["block.collapse_codebook_keys"]   = self.block.collapse_codebook_keys
        sd["block.collapse_codebook_values"] = self.block.collapse_codebook_values
        sd["block.collapse_gate_w"]          = self.block.collapse_gate_w
        sd["block.collapse_gate_b"]          = self.block.collapse_gate_b
        sd["block.collapse_proj_down"]       = self.block.collapse_proj_down
        sd["block.collapse_proj_up"]         = self.block.collapse_proj_up
        sd["block.collapse_alpha"]           = self.block.collapse_alpha
        # v70 collapse — waist-dim codebook + split gate. Registered for ckpt save/load.
        sd["block.collapse_v70_codebook_keys"]   = self.block.collapse_v70_codebook_keys
        sd["block.collapse_v70_codebook_values"] = self.block.collapse_v70_codebook_values
        sd["block.collapse_v70_proj_down"]       = self.block.collapse_v70_proj_down
        sd["block.collapse_v70_proj_up"]         = self.block.collapse_v70_proj_up
        sd["block.collapse_v70_bias"]            = self.block.collapse_v70_bias
        sd["block.collapse_v70_gate_w_proto"]    = self.block.collapse_v70_gate_w_proto
        sd["block.collapse_v70_gate_w_breath"]   = self.block.collapse_v70_gate_w_breath
        sd["block.collapse_v70_gate_b"]          = self.block.collapse_v70_gate_b
        sd["block.collapse_v70_gate_w"]          = self.block.collapse_v70_gate_w  # legacy, kept for symmetry
        sd["block.collapse_v70_breath_embed"]    = self.block.collapse_v70_breath_embed
        sd["block.collapse_v70_alpha"]           = self.block.collapse_v70_alpha
        # v71 collapse — refined v70 (stronger sparsity, lower gate bias, k-means init,
        # cleaner controller signal). Registered for ckpt save/load.
        sd["block.collapse_v71_codebook_keys"]   = self.block.collapse_v71_codebook_keys
        sd["block.collapse_v71_codebook_values"] = self.block.collapse_v71_codebook_values
        sd["block.collapse_v71_proj_down"]       = self.block.collapse_v71_proj_down
        sd["block.collapse_v71_proj_up"]         = self.block.collapse_v71_proj_up
        sd["block.collapse_v71_bias"]            = self.block.collapse_v71_bias
        sd["block.collapse_v71_gate_w_proto"]    = self.block.collapse_v71_gate_w_proto
        sd["block.collapse_v71_gate_w_breath"]   = self.block.collapse_v71_gate_w_breath
        sd["block.collapse_v71_gate_b"]          = self.block.collapse_v71_gate_b
        sd["block.collapse_v71_breath_embed"]    = self.block.collapse_v71_breath_embed
        sd["block.collapse_v71_alpha"]           = self.block.collapse_v71_alpha
        sd["block.bfield_proj_down"] = self.block.bfield_proj_down
        sd["block.bfield_proj_up"] = self.block.bfield_proj_up
        sd["block.bfield_bias"] = self.block.bfield_bias
        sd["block.waist_codebook_keys"] = self.block.waist_codebook_keys
        sd["block.waist_codebook_values"] = self.block.waist_codebook_values
        # v54 waist controller
        sd["wc.waist_up_w"] = self.waist_controller.waist_up_w
        sd["wc.waist_up_b"] = self.waist_controller.waist_up_b
        for i, layer in enumerate(self.waist_controller.layers):
            for k, v in layer.items():
                sd[f"wc.layer{i}.{k}"] = v
        # v62 decoupled-controller final up-projection (present only when H_ctrl != H_base)
        if self.waist_controller.final_up_w is not None:
            sd["wc.final_up_w"] = self.waist_controller.final_up_w
            sd["wc.final_up_b"] = self.waist_controller.final_up_b
        sd["wc.k_pos_embed"] = self.waist_controller.k_pos_embed  # v63
        # v72 copy mechanism — always saved (state-dict symmetry). Loaded weights
        # are zero-init / small random when ckpt predates v72.
        sd["wc.copy_q_w"]    = self.waist_controller.copy_q_w
        sd["wc.copy_k_w"]    = self.waist_controller.copy_k_w
        sd["wc.copy_gate_w"] = self.waist_controller.copy_gate_w
        sd["wc.copy_gate_b"] = self.waist_controller.copy_gate_b
        # v81 (2026-05-26) Multi-head WaistController params — ALWAYS saved (state-dict symmetry).
        # Ckpts predating v81 will load_state_dict with these as missing keys → kept at
        # zero-residual init (forward behavior matches single-head exactly when MLP=0).
        for hi, name in enumerate(self.waist_controller.head_names):
            mlp = self.waist_controller.head_mlps[hi]
            sd[f"wc.head_{name}.w1"] = mlp["w1"]
            sd[f"wc.head_{name}.b1"] = mlp["b1"]
            sd[f"wc.head_{name}.w2"] = mlp["w2"]
            sd[f"wc.head_{name}.b2"] = mlp["b2"]
        sd["block.bfield_alpha"] = self.block.bfield_alpha
        sd["waist_head_w"] = self.waist_head_w
        sd["waist_head_b"] = self.waist_head_b
        sd["boundary_head_w"] = self.boundary_head_w  # v65
        sd["boundary_head_b"] = self.boundary_head_b  # v65
        # v85 (2026-05-27) Queryable slot decoder — ALWAYS saved (state-dict
        # symmetry). Ckpts predating v85 load with these as missing keys → kept
        # at init (zero-init waist proj + small random codebooks + zero-init
        # active head, so initial behavior is benign).
        sd["v85.slot_pos_embed"]      = self.v85_slot_decoder.slot_pos_embed
        sd["v85.waist_pool_proj_w"]   = self.v85_slot_decoder.waist_pool_proj_w
        sd["v85.waist_pool_proj_b"]   = self.v85_slot_decoder.waist_pool_proj_b
        sd["v85.slot_ln_g"]           = self.v85_slot_decoder.slot_ln_g
        sd["v85.slot_ln_b"]           = self.v85_slot_decoder.slot_ln_b
        sd["v85.ops_codebook"]        = self.v85_slot_decoder.ops_codebook
        sd["v85.ops_head_b"]          = self.v85_slot_decoder.ops_head_b
        sd["v85.types_codebook"]      = self.v85_slot_decoder.types_codebook
        sd["v85.types_head_b"]        = self.v85_slot_decoder.types_head_b
        sd["v85.args1_q_w"]           = self.v85_slot_decoder.args1_q_w
        sd["v85.args2_q_w"]           = self.v85_slot_decoder.args2_q_w
        sd["v85.args_k_w"]            = self.v85_slot_decoder.args_k_w
        sd["v85.active_head_w"]       = self.v85_slot_decoder.active_head_w
        sd["v85.active_head_b"]       = self.v85_slot_decoder.active_head_b
        # v86 (2026-05-27) Per-slot cross-attn for args binding — ALWAYS saved
        # (state-dict symmetry). Ckpts predating v86 load with these as missing
        # keys → kept at init (small random q + zero-init k/v + zero-init slot_pos
        # → cross-attn contribution is 0 at warm-start, so behavior matches v85).
        sd["v85.v86_args_q_proj"]     = self.v85_slot_decoder.v86_args_q_proj
        sd["v85.v86_args_k_proj"]     = self.v85_slot_decoder.v86_args_k_proj
        sd["v85.v86_args_v_proj"]     = self.v85_slot_decoder.v86_args_v_proj
        sd["v85.v86_args_slot_pos"]   = self.v85_slot_decoder.v86_args_slot_pos
        # v91 (2026-05-27) arg_pos_emb — always saved; zero-init when missing from
        # legacy ckpts so warm-start preserves v90 behavior at step 0.
        sd["v85.arg_pos_emb"]         = self.v85_slot_decoder.arg_pos_emb
        # v96 (2026-05-28) consolidation-table params — ALWAYS saved (state-dict
        # symmetry). Ckpts predating v96 load with these as missing keys → kept
        # at init (zero-init gate_w + small-random codebooks + zero-init paths,
        # so V96 forward path is benign warm-start when needed).
        sd["v96.gate_w"]              = self.v96_gate_w
        sd["v96.gate_b"]              = self.v96_gate_b
        sd["v96.ops_codebook"]        = self.v96_ops_codebook
        sd["v96.types_codebook"]      = self.v96_types_codebook
        sd["v96.summary_proj"]        = self.v96_summary_proj
        sd["v96.table_kv_proj"]       = self.v96_table_kv_proj
        sd["v96.table_alpha"]         = self.v96_table_alpha    # v96.1
        # v96.2 constraint check heads (state_dict symmetry).
        sd["v96.ref_validity_head_w"] = self.v96_ref_validity_head_w
        sd["v96.ref_validity_head_b"] = self.v96_ref_validity_head_b
        sd["v96.arg_order_head_w"]    = self.v96_arg_order_head_w
        sd["v96.arg_order_head_b"]    = self.v96_arg_order_head_b
        # v97 (2026-05-28) calibration head — ALWAYS saved (state_dict symmetry).
        # Ckpts predating v97 load with these as missing keys → kept at init
        # (randn 0.02 weights + zero bias → sigmoid output ≈ 0.5 at warm-start,
        # benign with the V97_CALIB_WEIGHT scaling at default 0.1).
        sd["v97.calib_head_w"]        = self.v97_calib_head_w
        sd["v97.calib_head_b"]        = self.v97_calib_head_b
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
            x = self.block.breathe_once(x, l, temp_mult=_sine_temp_baseline(l, n_loops), n_loops=n_loops)
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
                             return_waist_compressed: bool = False,
                             return_per_breath_x: bool = False,
                             notebook_pool_mask: Tensor | None = None,
                             main_attn_mask: Tensor | None = None,
                             return_v96_artifacts: bool = False,
                             return_v97_calib: bool = False):
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

        v79 (2026-05-25) notebook_pool_mask: shape (B, T) float, 1.0 at valid
        positions (typically [0, prompt_len)) and 0.0 elsewhere. When provided
        AND NOTEBOOK_POOL_MODE == "attn", applies an additive -1e4 penalty to the
        notebook write attention scores at masked positions, so the notebook only
        ever reads from prompt positions. Plugs the v78b training-time leak where
        post-breath x at gold-answer positions carries gradient information about
        the gold span — the notebook would otherwise read it.

        v96 (2026-05-28) return_v96_artifacts: when True AND V96_CONSOLIDATION=1,
        also returns:
          - v96_artifacts_per_breath: list of n_loops × V96Artifact (unpacked view)
          - v96_table_packed:         (B, n_loops, 165) — the consolidation table

        Per breath the v96 artifact is computed from the (input, output) pair of
        that breath via the gated-quantize → attention-pool → codebook pipeline.

        v97 (2026-05-28) return_v97_calib: when True AND V97_CALIBRATION=1, also
        returns:
          - v97_calib_logits_per_breath: list of n_loops × (B,) calibration logits.
        Each breath: attention-pool the breath-output state x over T using
        breath_embed[l] as the query (prompt-range only via notebook_pool_mask if
        provided), project pooled state through v97_calib_head_w/b → (B,) logit.
        The trainer applies sigmoid + MSE against per-breath progression targets.
        Pure forward signal — NEVER feeds back into any subsequent breath's input.
        """
        x_emb = self.embed(tokens).cast(dtypes.half)  # v40: frozen embedding, reused each breath in fresh mode
        # v81 (2026-05-26) main_attn_mask doubles as an embedding mask: at answer-span
        # positions (mask=0), zero the input embedding so the residual stream carries
        # NO information about the gold/garbage token sitting there. Combined with the
        # main-attn key mask and the cross-attn / notebook mask, this makes the v81
        # paradigm fully prompt-conditional — each position's prediction is independent
        # of the teacher-forced tokens in the answer span. Verified by audit pos=N≥prompt_len.
        if main_attn_mask is not None:
            x_emb = x_emb * main_attn_mask.cast(x_emb.dtype).reshape(x_emb.shape[0], -1, 1)
        x = x_emb
        integral = Tensor.zeros_like(x)
        match_weights = []
        integrated_per_breath = []
        waist_compressed_per_breath = []  # v39: 512d compressed at end-of-breath waist
        per_breath_x = []  # v52 Stage 1: end-of-breath outputs for per-breath supervision
        v96_artifacts_per_breath = []  # v96: per-breath unpacked artifacts
        v96_packed_rows = []  # v96: per-breath packed rows (B, 165)
        v97_calib_logits_per_breath = []  # v97: per-breath calibration logits (B,)
        handoff = None
        notebook = None
        notebook_r = None
        weight_sum = 0.0  # v39 sin-modulation: accumulated weights for the integral
        # v70: apply_collapse_v70 resets self.block._collapse_v70_accum_sparsity on
        # breath_idx==0 and accumulates over the K-loop. The trainer reads the sum
        # after forward (it's a graph node belonging to this forward's trace).
        if NOTEBOOK_V24:
            B = x.shape[0]
            notebook = initial_notebook if initial_notebook is not None else _initial_notebook_state(B, self.block.nb_dim)
            if NOTEBOOK_DUAL:
                notebook_r = initial_notebook_r if initial_notebook_r is not None else _initial_notebook_state(B, self.block.nb_dim)
        dag_storage = None
        if NOTEBOOK_DAG:
            B = x.shape[0]
            dag_storage = self.block.dag_notebook_init_storage(B)
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
            # v65 per-breath prompt refresh: skip connection from raw prompt_emb.
            # Refreshes entity identity at every breath through the lossy waist.
            if PROMPT_REFRESH_ALPHA > 0.0:
                x_in = x_in + (PROMPT_REFRESH_ALPHA * x_emb).cast(x_in.dtype)
            if NOTEBOOK_V24:
                if NOTEBOOK_ACCUMULATE_ENABLED:
                    read_vec = (notebook @ self.block.notebook_read_w + self.block.notebook_read_b)
                    x_in = x_in + read_vec.reshape(x_in.shape[0], 1, -1).cast(x_in.dtype)
                if NOTEBOOK_DUAL:
                    read_vec_r = (notebook_r @ self.block.notebook_rep_read_w + self.block.notebook_rep_read_b)
                    x_in = x_in + read_vec_r.reshape(x_in.shape[0], 1, -1).cast(x_in.dtype)
            if NOTEBOOK_DAG and dag_storage is not None:
                # v61: causal cross-attention over prior breaths' summaries.
                # Read is 0 at l=0 (no priors) and at step 0 (W_o zero-init).
                dag_read = self.block.dag_notebook_read(x_in, dag_storage, l)
                x_in = x_in + dag_read
            # v96 (2026-05-28) capture x_in BEFORE breathe_once for delta computation.
            # Note: x_in already incorporates breath_embed via breathe_once internally,
            # but for v96 we want the PRE-breath_embed input so the delta captures
            # everything the breath added (including embed offset). The cleanest
            # delta = (x AFTER breath) - (x_in BEFORE breath).
            v96_x_in_capture = x_in if V96_CONSOLIDATION else None
            if return_waist_compressed:
                x, waist_compressed = self.block.breathe_once(x_in, l, temp_mult=_sine_temp_baseline(l, n_loops),
                                                                return_waist_compressed=True, n_loops=n_loops,
                                                                attn_mask=main_attn_mask)
                waist_compressed_per_breath.append(waist_compressed)
            else:
                x = self.block.breathe_once(x_in, l, temp_mult=_sine_temp_baseline(l, n_loops), n_loops=n_loops,
                                              attn_mask=main_attn_mask)
            # v96 (2026-05-28) compute artifact from (x_in, x) → packed row → append.
            if V96_CONSOLIDATION and v96_x_in_capture is not None:
                from mycelium.v96 import compute_v96_artifact as _v96_compute, pack_artifact as _v96_pack
                # Use notebook_pool_mask as the v96 attention-pool mask (same
                # prompt-range restriction — prevents pooling over answer-span
                # positions during training).
                _v96_mask = notebook_pool_mask
                # v96.2: per-breath temperature for ops/types logits sharpening.
                # Linear from V96_T_START at B0 down to V96_T_END at B_{K-1}.
                if V96_TEMPERATURE_DECAY:
                    if n_loops <= 1:
                        _T_k = V96_T_END
                    else:
                        _T_k = V96_T_START * (1.0 - float(l) / float(n_loops - 1))
                        if _T_k < V96_T_END:
                            _T_k = V96_T_END
                else:
                    _T_k = None
                v96_art = _v96_compute(
                    v96_x_in_capture, x,
                    self.block.breath_embed[l],
                    self.v96_gate_w, self.v96_gate_b,
                    self.v96_ops_codebook, self.v96_types_codebook,
                    self.v96_summary_proj,
                    pool_mask=_v96_mask,
                    temperature=_T_k,
                )
                v96_artifacts_per_breath.append(v96_art)
                v96_packed_rows.append(_v96_pack(v96_art))  # (B, 165)
            # v97 (2026-05-28) calibration head — read-only pool over breath output.
            # Attention-pool x using breath_embed[l] as query, project to scalar logit.
            # No feedback into the residual stream: the result is appended to a
            # standalone list returned to the trainer for an aux loss.
            if V97_CALIBRATION and return_v97_calib:
                # Cast pool computation to float32 for numerical stability of softmax
                # (consistent with v96 — already-float in our paths).
                x_f_v97 = x.cast(dtypes.float)
                pool_q_v97 = self.block.breath_embed[l].cast(dtypes.float).reshape(1, 1, -1)
                scores_v97 = (x_f_v97 * pool_q_v97).sum(axis=-1)               # (B, T)
                if notebook_pool_mask is not None:
                    scores_v97 = scores_v97 + (1.0 - notebook_pool_mask.cast(scores_v97.dtype)) * (-1e4)
                w_v97 = scores_v97.softmax(axis=-1).reshape(scores_v97.shape[0], -1, 1)  # (B, T, 1)
                pooled_v97 = (x_f_v97 * w_v97).sum(axis=1)                      # (B, H)
                calib_logit = (pooled_v97 @ self.v97_calib_head_w
                                + self.v97_calib_head_b.reshape(1, -1)).reshape(pooled_v97.shape[0])  # (B,)
                v97_calib_logits_per_breath.append(calib_logit)
            if NOTEBOOK_V24:
                x_f = x.cast(dtypes.float)
                # v79 notebook causal mask: when notebook_pool_mask provided, mask
                # out non-prompt positions in attention-pool BEFORE softmax. mask=1
                # at valid (prompt) positions, mask=0 elsewhere. Mean-pool unaffected.
                if NOTEBOOK_POOL_MODE == "attn":
                    scores = (x_f * self.block.notebook_write_query.reshape(1, 1, -1)).sum(axis=-1)
                    if notebook_pool_mask is not None:
                        scores = scores + (1.0 - notebook_pool_mask.cast(scores.dtype)) * (-1e4)
                    weights = scores.softmax(axis=-1).reshape(x.shape[0], -1, 1)
                    x_pool = (x_f * weights).sum(axis=1)
                else:
                    x_pool = x_f.mean(axis=1)
                if NOTEBOOK_ACCUMULATE_ENABLED:
                    notebook = notebook + (x_pool @ self.block.notebook_write_w + self.block.notebook_write_b)
                if NOTEBOOK_DUAL:
                    if NOTEBOOK_POOL_MODE == "attn":
                        scores_r = (x_f * self.block.notebook_rep_query.reshape(1, 1, -1)).sum(axis=-1)
                        if notebook_pool_mask is not None:
                            scores_r = scores_r + (1.0 - notebook_pool_mask.cast(scores_r.dtype)) * (-1e4)
                        weights_r = scores_r.softmax(axis=-1).reshape(x.shape[0], -1, 1)
                        x_pool_r = (x_f * weights_r).sum(axis=1)
                    else:
                        x_pool_r = x_pool
                    notebook_r = x_pool_r @ self.block.notebook_rep_write_w + self.block.notebook_rep_write_b
            if NOTEBOOK_DAG and dag_storage is not None:
                # v61: write attn-pooled end-of-breath rep to slot l (one-hot mask).
                dag_storage = self.block.dag_notebook_write(x, dag_storage, l)
            if CROSS_BREATH_HANDOFF:
                handoff = self.block.compute_handoff(x)
            # v52 Stage 1: capture end-of-breath state for per-breath supervision.
            if return_per_breath_x:
                per_breath_x.append(x)
            # v39 sin-modulated integral: bell-shaped weighting that never hits
            # zero at the endpoints (l=0 gives sin(π/(2·n_loops)), peaks at the
            # middle breath). Heartbeat-across-breaths envelope.
            if BFIELD_SIN_MOD:
                beta_l = math.sin((l + 0.5) * math.pi / float(n_loops))
            else:
                beta_l = 1.0
            # Stochastic depth — see BreathingBlock.breathe() for the rationale.
            if STOCH_DEPTH_P > 0.0 and Tensor.training:
                keep_l = self.block.stoch_keep_mask[l].cast(x.dtype)
                integral = integral + beta_l * keep_l * x
            else:
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
        # v96: assemble the consolidation table as a single (B, n_loops, 165) tensor.
        # Stacking the list of (B, 165) rows along a new axis 1 — done OUTSIDE the
        # JIT body once per forward (no per-breath syncs).
        v96_table_packed = None
        if V96_CONSOLIDATION and v96_packed_rows:
            # Tensor.stack along a new axis 1 (between batch and packed-dim).
            # Each row is (B, 165); stack → (B, K, 165).
            v96_table_packed = Tensor.stack(*v96_packed_rows, dim=1)
        if return_per_breath_x:
            # v52 Stage 1: simplified return when per-breath supervision is the consumer.
            # Caller gets per-breath end-of-breath outputs + match weights for op-aux.
            # v54: optionally also returns per-breath waist_compressed for controller decode.
            # v96 (2026-05-28): when return_v96_artifacts=True, append the per-breath
            # artifact list + packed table at the END of the tuple. Backward-compatible.
            # v97 (2026-05-28): when return_v97_calib=True, append v97 calib logits list
            # at the very end.
            base_tuple = (final, match_weights, per_breath_x)
            if return_waist_compressed:
                base_tuple = base_tuple + (waist_compressed_per_breath,)
            if return_v96_artifacts:
                base_tuple = base_tuple + (v96_artifacts_per_breath, v96_table_packed)
            if return_v97_calib:
                base_tuple = base_tuple + (v97_calib_logits_per_breath,)
            return base_tuple
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
            # v44 doubled-layers: pick Set A or Set B based on breath index
            if DOUBLED_LAYERS and loop >= (cfg.max_loops // 2):
                active_layers = self.block.layers_b
            else:
                active_layers = self.block.layers
            for li, layer in enumerate(active_layers):
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
                    x = self.block.apply_bfield_waist(x, loop_idx=loop)
            # v39 end-of-breath waist
            if BFIELD_WAIST > 0 and BFIELD_END_OF_BREATH:
                x = self.block.apply_bfield_waist(x, loop_idx=loop)
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
            layers_b = self.block.layers_b  # v44 doubled-layers
            block_local = self.block
            layer_norm_eps = cfg.layer_norm_eps
            n_loops_local = n_loops
            n_layers_local = n_layers
            max_loops_local = cfg.max_loops  # for doubled-layers split point
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
                    # Notebook reads — gated on STAGE2_NOTEBOOK (default off).
                    # Training does NOT have autoregressive decode, so updating
                    # notebook per generated token is OOD for any pre-v40 model.
                    if NOTEBOOK_V24 and STAGE2_NOTEBOOK:
                        if NOTEBOOK_ACCUMULATE_ENABLED:
                            read_vec = (notebook @ block_local.notebook_read_w + block_local.notebook_read_b)
                            x = x + read_vec.reshape(B_local, 1, -1).cast(x.dtype)
                        if NOTEBOOK_DUAL:
                            read_vec_r = (notebook_r @ block_local.notebook_rep_read_w + block_local.notebook_rep_read_b)
                            x = x + read_vec_r.reshape(B_local, 1, -1).cast(x.dtype)
                    # v44 doubled-layers: pick Set A or Set B based on breath index
                    if DOUBLED_LAYERS and loop >= (max_loops_local // 2):
                        active_layers_local = layers_b
                    else:
                        active_layers_local = layers
                    for li in range(n_layers_local):
                        idx = li * n_loops_local + loop
                        x, k_new, v_new = active_layers_local[li].forward_cached_step_batched(
                            x, loop, ck[idx], cv[idx], t_pos_t, None  # per-batch t_pos handles masking
                        )
                        new_ck[idx] = k_new
                        new_cv[idx] = v_new
                    # v39 end-of-breath waist
                    if BFIELD_WAIST > 0 and BFIELD_END_OF_BREATH:
                        x = block_local.apply_bfield_waist(x, loop_idx=loop)
                    # Notebook writes — gated on STAGE2_NOTEBOOK (default off, same reasoning as reads).
                    if NOTEBOOK_V24 and STAGE2_NOTEBOOK:
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

    def _get_seg_buffers(self, B: int, K: int, max_len: int, waist_dim: int, nb_dim: int):
        """Allocate (or retrieve memoized) persistent buffers for closure-based JIT.

        Memoized by (B, K, max_len, waist_dim) — one allocation per distinct shape combo,
        reused across calls with the same shapes (same batch size and cache budget).

        Returns a dict of named Tensors. All are realized contiguous buffers so they can
        be closed over by a @TinyJit function and written via .assign().
        """
        key = (B, K, max_len, waist_dim)
        if not hasattr(self, "_seg_buf_cache"):
            self._seg_buf_cache = {}
        if key in self._seg_buf_cache:
            return self._seg_buf_cache[key]

        H = self.cfg.hidden
        n_layers = self.cfg.n_phases
        # KV cache: [n_layers][K] each (B, n_heads, max_len, head_dim)
        n_heads = self.cfg.n_heads
        head_dim = self.cfg.head_dim
        cache_k = [[Tensor.zeros((B, n_heads, max_len, head_dim), dtype=dtypes.float).contiguous().realize()
                    for _ in range(K)] for _ in range(n_layers)]
        cache_v = [[Tensor.zeros((B, n_heads, max_len, head_dim), dtype=dtypes.float).contiguous().realize()
                    for _ in range(K)] for _ in range(n_layers)]
        # Per-breath waist buffers: K × (B, max_len, waist_dim)
        waist_per_breath = [Tensor.zeros((B, max_len, waist_dim), dtype=dtypes.float).contiguous().realize()
                            for _ in range(K)]
        # Prompt embedding buffer: (B, max_len, H)
        prompt_emb_buf = Tensor.zeros((B, max_len, H), dtype=dtypes.float).contiguous().realize()
        # Notebook state buffers
        notebook_state = _initial_notebook_state(B, nb_dim).contiguous().realize()
        notebook_r_state = _initial_notebook_state(B, nb_dim).contiguous().realize()
        # Per-token decode inputs: prev token id (B, 1) and current position (B,)
        prev_id_t = Tensor.zeros((B, 1), dtype=dtypes.int).contiguous().realize()
        t_pos_t = Tensor.zeros((B,), dtype=dtypes.int).contiguous().realize()
        # Argmax output buffer: (K, B) — JIT writes here and caller reads
        argmax_buf = Tensor.zeros((K, B), dtype=dtypes.int).contiguous().realize()
        # Per-example step counter tracking #### boundaries (updated inside JIT)
        step_counter_t = Tensor.zeros((B,), dtype=dtypes.int).contiguous().realize()

        bufs = {
            "cache_k": cache_k,   # list[list[Tensor]]
            "cache_v": cache_v,
            "waist_per_breath": waist_per_breath,
            "prompt_emb_buf": prompt_emb_buf,
            "notebook_state": notebook_state,
            "notebook_r_state": notebook_r_state,
            "prev_id_t": prev_id_t,
            "t_pos_t": t_pos_t,
            "argmax_buf": argmax_buf,
            "step_counter_t": step_counter_t,
        }
        self._seg_buf_cache[key] = bufs
        return bufs

    def cached_generate_segmented(self, batch_prompt_ids: list, n_loops: int, max_new: int,
                                     decode_fn,
                                     stop_token_ids=None, stop_seq=None,
                                     cache_max_len: int | None = None,
                                     waist_dim: int | None = None) -> list:
        """KV-cached inference for the per-breath supervision paradigm (v54+).

        Closure-based fused JIT design (v2): all per-token work happens inside a
        zero-arg @TinyJit that closes over persistent state buffers. No Python work
        per token after the first two JIT calls (cnt=0 eager, cnt=1 capture+exec,
        cnt>=2 pure replay). Mirrors the optimizer JIT pattern.

        decode_fn signature (UPDATED from v1):
            next_ids = decode_fn(
                argmax_buf,    # Tensor (K, B) int — argmax token IDs per breath per example
                t_pos_per,     # list[int], length B — current position per example
                decoded_so_far,# list[list[int]] — tokens decoded per example
            )
            # Returns: numpy array shape (B,) int32 — next token id per example.

        The caller picks the right breath per example via #### count in decode_fn.

        Required model config: BFIELD_WAIST > 0, BFIELD_END_OF_BREATH=1,
        CONTROLLER_DECODE=1.
        """
        from tinygrad import Tensor as _T
        cfg = self.cfg
        n_layers = cfg.n_phases
        B = len(batch_prompt_ids)
        stop_set = set(stop_token_ids or [])
        seq = list(stop_seq or [])
        seq_len = len(seq)

        assert BFIELD_WAIST > 0 and BFIELD_END_OF_BREATH, (
            "cached_generate_segmented requires BFIELD_WAIST>0 and BFIELD_END_OF_BREATH=1"
        )
        if waist_dim is None:
            waist_dim = BFIELD_WAIST

        real_lens = [len(p) for p in batch_prompt_ids]
        max_prompt = max(real_lens)
        if cache_max_len is None:
            cache_max_len = cfg.max_seq_len
        assert max_prompt + max_new <= cache_max_len
        assert cache_max_len <= cfg.max_seq_len
        max_len = cache_max_len

        # Get or allocate persistent buffers for this (B, K, max_len, waist_dim) combo
        bufs = self._get_seg_buffers(B, n_loops, max_len, waist_dim, self.block.nb_dim)
        cache_k = bufs["cache_k"]
        cache_v = bufs["cache_v"]
        waist_per_breath_buf = bufs["waist_per_breath"]
        prompt_emb_buf_t = bufs["prompt_emb_buf"]
        notebook_state_t = bufs["notebook_state"]
        notebook_r_state_t = bufs["notebook_r_state"]
        prev_id_t = bufs["prev_id_t"]
        t_pos_t = bufs["t_pos_t"]
        argmax_buf = bufs["argmax_buf"]
        step_counter_t = bufs["step_counter_t"]

        # Pad ALL prompts to max_len (not just max_prompt) so Stage 1 JIT has FIXED shapes.
        # This allows Stage 1 to compile once and replay fast across all batches.
        padded_full = np.zeros((B, max_len), dtype=np.int32)
        for b, p in enumerate(batch_prompt_ids):
            padded_full[b, :len(p)] = p
        prompts_t = _T(padded_full, dtype=dtypes.int).realize()

        # Attention mask: 1 for real prompt positions, 0 for padding.
        # Shape: (B, max_len) — needed to avoid padding positions corrupting attention.
        prompt_attn_mask_np = np.zeros((B, max_len), dtype=np.int32)
        for b, rl in enumerate(real_lens):
            prompt_attn_mask_np[b, :rl] = 1
        attn_mask_full = _T(prompt_attn_mask_np, dtype=dtypes.int).realize()
        # Use mask only when there's actual padding (same_len batches + max_len pads)
        # Always pass mask since max_len > max_prompt in general
        attn_mask_arg = attn_mask_full

        n_phases = cfg.n_phases

        # === Stage 1: JIT-compiled prefill over fixed-shape (B, max_len) prompts ===
        # Build or retrieve the Stage 1 JIT. Key: (B, K, max_len, waist_dim).
        # The JIT takes (prompts_t, attn_mask_t) as inputs and writes K/V, waist, notebook
        # into the CLOSURE-captured persistent buffers. All shapes are fixed → one compile.
        if not hasattr(self, "_seg_stage1_jits"):
            self._seg_stage1_jits = {}
        s1_key = (B, n_loops, max_len, waist_dim)
        if s1_key not in self._seg_stage1_jits:
            print(f"[JIT] registering Stage 1 prefill JIT: B={B} K={n_loops} max_len={max_len}", flush=True)
            embed_w_s1 = self.embed.weight
            block_s1 = self.block
            n_loops_s1 = n_loops
            n_layers_s1 = n_layers
            n_phases_s1 = n_phases
            max_loops_s1 = cfg.max_loops
            B_s1 = B
            waist_dim_s1 = waist_dim
            max_len_s1 = max_len
            _ck_s1 = cache_k
            _cv_s1 = cache_v
            _waist_s1 = waist_per_breath_buf
            _nb_s1 = notebook_state_t
            _nbr_s1 = notebook_r_state_t

            @TinyJit
            def _stage1_jit(prompts_in, attn_mask_in):
                x_emb_s1 = embed_w_s1[prompts_in].cast(dtypes.half)
                x_s1 = x_emb_s1
                notebook_s1 = _initial_notebook_state(B_s1, block_s1.nb_dim)
                notebook_r_s1 = _initial_notebook_state(B_s1, block_s1.nb_dim)
                handoff_s1 = None
                for loop in range(n_loops_s1):
                    if BREATHE_FRESH_INPUT:
                        x_in_s1 = x_emb_s1
                    else:
                        x_in_s1 = x_s1
                        if CROSS_BREATH_HANDOFF and handoff_s1 is not None:
                            x_in_s1 = x_in_s1 + handoff_s1
                    # v65 per-breath prompt refresh (eval-side parity with breathe_with_lookup).
                    if PROMPT_REFRESH_ALPHA > 0.0:
                        x_in_s1 = x_in_s1 + (PROMPT_REFRESH_ALPHA * x_emb_s1).cast(x_in_s1.dtype)
                    if NOTEBOOK_V24:
                        if NOTEBOOK_ACCUMULATE_ENABLED:
                            rv = (notebook_s1 @ block_s1.notebook_read_w + block_s1.notebook_read_b)
                            x_in_s1 = x_in_s1 + rv.reshape(B_s1, 1, -1).cast(x_in_s1.dtype)
                        if NOTEBOOK_DUAL:
                            rv_r = (notebook_r_s1 @ block_s1.notebook_rep_read_w + block_s1.notebook_rep_read_b)
                            x_in_s1 = x_in_s1 + rv_r.reshape(B_s1, 1, -1).cast(x_in_s1.dtype)
                    if BREATH_TIME_EMBED:
                        x_in_s1 = x_in_s1 + block_s1.breath_embed[loop].reshape(1, 1, -1).cast(x_in_s1.dtype)
                    base_alpha_s1 = block_s1.rope._alpha_at(loop, x_in_s1.dtype)
                    ac_base_s1, asn_base_s1 = base_alpha_s1
                    tm_s1 = _sine_temp_baseline(loop, n_loops_s1)
                    if DOUBLED_LAYERS and loop >= (max_loops_s1 // 2):
                        active_s1 = block_s1.layers_b
                    else:
                        active_s1 = block_s1.layers
                    x_s1 = x_in_s1
                    for li in range(n_layers_s1):
                        if PER_HEAD_PITCH and li > 0:
                            cos_o = block_s1.per_head_pitch_cos[li].cast(x_s1.dtype)
                            sin_o = block_s1.per_head_pitch_sin[li].cast(x_s1.dtype)
                            la_s1 = (ac_base_s1 * cos_o - asn_base_s1 * sin_o,
                                     ac_base_s1 * sin_o + asn_base_s1 * cos_o)
                        else:
                            la_s1 = base_alpha_s1
                        lt_s1 = _per_layer_temp_within_breath(li, n_phases_s1) if PER_BREATH_TEMP else tm_s1
                        x_s1, (k_part_s1, v_part_s1) = active_s1[li].forward_with_kv(
                            x_s1, loop_idx=loop, attn_mask=attn_mask_in,
                            temp_mult=lt_s1, alpha=la_s1)
                        # Write K/V directly (no padding needed — already at max_len shape)
                        _ck_s1[li][loop].assign(k_part_s1)
                        _cv_s1[li][loop].assign(v_part_s1)
                        if BREATH_NORM_OSC and CONSTANT_RADIUS:
                            scale_s1 = _per_layer_norm_scale_within_breath(li, n_phases_s1)
                            x_f_s1 = x_s1.cast(dtypes.float)
                            x_norm_s1 = (x_f_s1.square().sum(axis=-1, keepdim=True) + 1e-6).sqrt()
                            target_s1 = block_s1.crp_target_norm * scale_s1
                            x_s1 = (x_f_s1 * (1.0 - block_s1.crp_mix_alpha) +
                                    x_f_s1 * (target_s1 / x_norm_s1) * block_s1.crp_mix_alpha).cast(x_s1.dtype)
                        if BFIELD_WAIST > 0 and li == 1 and not BFIELD_END_OF_BREATH:
                            x_s1 = block_s1.apply_bfield_waist(x_s1, loop_idx=loop)
                    # End-of-breath waist with compressed capture
                    x_s1, compressed_s1 = block_s1.apply_bfield_waist(x_s1, return_compressed=True, loop_idx=loop)
                    _waist_s1[loop].assign(compressed_s1.cast(dtypes.float))
                    # End-of-breath CRP
                    if CONSTANT_RADIUS and not BREATH_NORM_OSC:
                        x_f_s1 = x_s1.cast(dtypes.float)
                        x_norm_s1 = (x_f_s1.square().sum(axis=-1, keepdim=True) + 1e-6).sqrt()
                        x_s1 = (x_f_s1 * (1.0 - block_s1.crp_mix_alpha) +
                                x_f_s1 * (block_s1.crp_target_norm / x_norm_s1) * block_s1.crp_mix_alpha).cast(x_s1.dtype)
                    # Notebook write
                    if NOTEBOOK_V24:
                        x_f_s1 = x_s1.cast(dtypes.float)
                        if NOTEBOOK_POOL_MODE == "attn":
                            sc = (x_f_s1 * block_s1.notebook_write_query.reshape(1, 1, -1)).sum(axis=-1)
                            wt = sc.softmax(axis=-1).reshape(B_s1, -1, 1)
                            xp = (x_f_s1 * wt).sum(axis=1)
                        else:
                            xp = x_f_s1.mean(axis=1)
                        if NOTEBOOK_ACCUMULATE_ENABLED:
                            notebook_s1 = notebook_s1 + (xp @ block_s1.notebook_write_w + block_s1.notebook_write_b)
                        if NOTEBOOK_DUAL:
                            if NOTEBOOK_POOL_MODE == "attn":
                                sc_r = (x_f_s1 * block_s1.notebook_rep_query.reshape(1, 1, -1)).sum(axis=-1)
                                wt_r = sc_r.softmax(axis=-1).reshape(B_s1, -1, 1)
                                xp_r = (x_f_s1 * wt_r).sum(axis=1)
                            else:
                                xp_r = xp
                            notebook_r_s1 = xp_r @ block_s1.notebook_rep_write_w + block_s1.notebook_rep_write_b
                    if CROSS_BREATH_HANDOFF:
                        handoff_s1 = block_s1.compute_handoff(x_s1)
                # Write notebook states to persistent buffers
                _nb_s1.assign(notebook_s1.cast(dtypes.float) if NOTEBOOK_V24 else notebook_s1)
                _nbr_s1.assign(notebook_r_s1.cast(dtypes.float) if (NOTEBOOK_V24 and NOTEBOOK_DUAL) else notebook_r_s1)
                return (
                    *_ck_s1[0],  # K[layer0][all loops]
                    *_cv_s1[0],
                    *_ck_s1[1],
                    *_cv_s1[1],
                    *_ck_s1[2],
                    *_cv_s1[2],
                    *_ck_s1[3],
                    *_cv_s1[3],
                    *_waist_s1,
                    _nb_s1,
                    _nbr_s1,
                )

            self._seg_stage1_jits[s1_key] = _stage1_jit

        stage1_jit = self._seg_stage1_jits[s1_key]
        # Run Stage 1 JIT — fills cache_k, cache_v, waist_per_breath, notebook state
        stage1_jit(prompts_t, attn_mask_full)

        # Initialize prompt_emb_buf from the same padded_full tokens used in Stage 1.
        # Use the SAME padded_full array (already has prompt tokens + token-0 padding).
        # This ensures prompt_emb_buf[b, :real_lens[b]] = embed(prompt_tokens) and
        # prompt_emb_buf[b, real_lens[b]:] = embed(token_0), matching the eager path.
        tokens_buf_init = _T(padded_full, dtype=dtypes.int).contiguous().realize()
        prompt_emb_init = self.embed(tokens_buf_init).cast(dtypes.float)
        prompt_emb_buf_t.assign(prompt_emb_init).realize()

        # === Build the closure-based Stage 2 JIT (once per shape key) ===
        if not hasattr(self, "_seg_closure_jits"):
            self._seg_closure_jits = {}
        jit_key = (B, n_loops, max_len, waist_dim)
        if jit_key not in self._seg_closure_jits:
            print(f"[JIT] registering closure-based cached_generate_segmented: "
                  f"B={B} K={n_loops} max_len={max_len} waist_dim={waist_dim}", flush=True)
            embed_w = self.embed.weight
            embed_out_local = self.embed_out
            waist_ctrl = self.waist_controller
            layers = self.block.layers
            layers_b = self.block.layers_b
            block_local = self.block
            n_loops_local = n_loops
            n_layers_local = n_layers
            max_loops_local = cfg.max_loops
            B_local = B
            waist_dim_local = waist_dim
            max_len_local = max_len
            vocab_active_local = 50277
            # Capture persistent buffer references in the closure
            _cache_k = cache_k
            _cache_v = cache_v
            _waist_per_breath = waist_per_breath_buf
            _prompt_emb_buf = prompt_emb_buf_t
            _notebook_state = notebook_state_t
            _notebook_r_state = notebook_r_state_t
            _prev_id_t = prev_id_t
            _t_pos_t = t_pos_t
            _argmax_buf = argmax_buf
            _step_counter = step_counter_t
            _hash_tok_id = 1835  # token ID for '####'

            @TinyJit
            def _step_closure():
                # Embed previous token
                x_new = embed_w[_prev_id_t].cast(dtypes.half)  # (B, 1, H)
                x = x_new
                notebook = _notebook_state
                notebook_r = _notebook_r_state
                handoff_inner = None
                waists_inner = []
                for loop in range(n_loops_local):
                    if BREATHE_FRESH_INPUT:
                        x_in = x_new
                    else:
                        x_in = x
                        if CROSS_BREATH_HANDOFF and handoff_inner is not None:
                            x_in = x_in + handoff_inner
                    if NOTEBOOK_V24 and STAGE2_NOTEBOOK:
                        if NOTEBOOK_ACCUMULATE_ENABLED:
                            read_vec = (notebook @ block_local.notebook_read_w + block_local.notebook_read_b)
                            x_in = x_in + read_vec.reshape(B_local, 1, -1).cast(x_in.dtype)
                        if NOTEBOOK_DUAL:
                            read_vec_r = (notebook_r @ block_local.notebook_rep_read_w + block_local.notebook_rep_read_b)
                            x_in = x_in + read_vec_r.reshape(B_local, 1, -1).cast(x_in.dtype)
                    if BREATH_TIME_EMBED:
                        x_in = x_in + block_local.breath_embed[loop].reshape(1, 1, -1).cast(x_in.dtype)
                    if DOUBLED_LAYERS and loop >= (max_loops_local // 2):
                        active_layers_local = layers_b
                    else:
                        active_layers_local = layers
                    # Per-breath alpha and temperature
                    base_alpha = block_local.rope._alpha_at(loop, x_in.dtype)
                    ac_base, asn_base = base_alpha
                    tm_breath = _sine_temp_baseline(loop, n_loops_local)
                    x = x_in
                    for li in range(n_layers_local):
                        if PER_HEAD_PITCH and li > 0:
                            cos_o = block_local.per_head_pitch_cos[li].cast(x.dtype)
                            sin_o = block_local.per_head_pitch_sin[li].cast(x.dtype)
                            ac_layer = ac_base * cos_o - asn_base * sin_o
                            asn_layer = ac_base * sin_o + asn_base * cos_o
                            layer_alpha = (ac_layer, asn_layer)
                        else:
                            layer_alpha = base_alpha
                        layer_temp = _per_layer_temp_within_breath(li, n_layers_local) if PER_BREATH_TEMP else tm_breath
                        x, k_new, v_new = active_layers_local[li].forward_cached_step_batched(
                            x, loop, _cache_k[li][loop], _cache_v[li][loop], _t_pos_t, None,
                            alpha=layer_alpha, temp_mult=layer_temp,
                        )
                        # Write updated K/V into persistent cache buffers
                        _cache_k[li][loop].assign(k_new)
                        _cache_v[li][loop].assign(v_new)
                    # End-of-breath waist with compressed capture
                    x, compressed = block_local.apply_bfield_waist(x, return_compressed=True, loop_idx=loop)
                    waists_inner.append(compressed.cast(dtypes.float))  # (B, 1, waist_dim)
                    # Notebook writes (gated — default STAGE2_NOTEBOOK=0)
                    if NOTEBOOK_V24 and STAGE2_NOTEBOOK:
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
                    if CROSS_BREATH_HANDOFF:
                        handoff_inner = block_local.compute_handoff(x)
                # Build scatter mask once (reused for waist, emb, and gather)
                positions = Tensor.arange(max_len_local).reshape(1, max_len_local, 1)
                t_pos_resh = _t_pos_t.reshape(B_local, 1, 1)
                scatter_mask = (positions == t_pos_resh).cast(dtypes.float)  # (B, max_len, 1)
                inv_mask = (1.0 - scatter_mask)
                # Scatter new token embedding into prompt_emb_buf FIRST so the controller
                # sees embed(prev_id_t) at t_pos — matching the original code's semantics.
                x_emb_new = embed_w[_prev_id_t].cast(dtypes.float)  # (B, 1, H)
                x_emb_broadcast = x_emb_new.expand(B_local, max_len_local, x_emb_new.shape[-1])
                _prompt_emb_buf.assign(
                    scatter_mask * x_emb_broadcast + inv_mask * _prompt_emb_buf
                )
                # Scatter new waist values into persistent buffers
                for k in range(n_loops_local):
                    new_w = waists_inner[k].expand(B_local, max_len_local, waist_dim_local)
                    _waist_per_breath[k].assign(
                        scatter_mask * new_w + inv_mask * _waist_per_breath[k]
                    )
                # Update notebook state buffers
                _notebook_state.assign(notebook)
                _notebook_r_state.assign(notebook_r)
                # Controller decode: use fresh waists (no scatter-gather round-trip).
                tokens_per_breath = []
                for k in range(n_loops_local):
                    wk_at_pos = waists_inner[k]  # (B, 1, waist_dim) — fresh from this step
                    lk = waist_ctrl.forward(wk_at_pos, _prompt_emb_buf, embed_out_local)
                    tk = lk[:, :, :vocab_active_local].argmax(axis=-1).reshape(B_local)
                    tokens_per_breath.append(tk)
                result = Tensor.stack(*tokens_per_breath, dim=0)  # (K, B) int
                _argmax_buf.assign(result)
                # Update step counter: check if _prev_id_t contained #### and bump counter.
                # This is done AFTER computing argmax for this step (the current prev_id is
                # the token that was just placed at t_pos, not yet the one we're predicting).
                # So we check _prev_id_t (the token we just processed) for #### and update
                # the counter for the NEXT step.
                is_hash = (_prev_id_t.reshape(B_local) == _hash_tok_id).cast(dtypes.int)  # (B,)
                new_counter = (_step_counter + is_hash).clip(0, n_loops_local - 1)
                _step_counter.assign(new_counter)
                # Select token for this step using the CURRENT (pre-update) step counter.
                # Build one-hot for step counter selection: (B, K)
                k_arange = Tensor.arange(n_loops_local).reshape(1, n_loops_local)
                step_onehot = (k_arange == _step_counter.reshape(B_local, 1)).cast(dtypes.int)  # (B, K)
                # result is (K, B). Transpose to (B, K), pick per-example: sum(result.T * onehot, dim=-1)
                result_t = result.transpose(0, 1)  # (B, K)
                selected = (result_t * step_onehot).sum(axis=-1).cast(dtypes.int)  # (B,)
                # Update prev_id_t with selected token and increment t_pos_t
                _prev_id_t.assign(selected.reshape(B_local, 1))
                _t_pos_t.assign(_t_pos_t + 1)
                # Return ALL assigned tensors so Tensor.realize() executes ALL stores.
                return (
                    _argmax_buf,
                    *_waist_per_breath,
                    _prompt_emb_buf,
                    _notebook_state,
                    _notebook_r_state,
                    _step_counter,
                    _prev_id_t,
                    _t_pos_t,
                )

            self._seg_closure_jits[jit_key] = _step_closure

        jit_step_closure = self._seg_closure_jits[jit_key]

        # === First token decode using prompt-side waist (no Stage 2 JIT yet) ===
        # Use a simple eager decode for the first token (matches previous behavior).
        decoded_so_far = [list(p) for p in batch_prompt_ids]
        t_pos_first_per = [real_lens[b] - 1 for b in range(B)]
        # Eager first-token decode: gather waist at t_pos_first_per, run controller
        positions_np = np.array(t_pos_first_per, dtype=np.int32)
        t_pos_first_t = _T(positions_np, dtype=dtypes.int).realize()
        pos_arange = Tensor.arange(max_len).reshape(1, max_len, 1)
        gather_mask_first = (pos_arange == t_pos_first_t.reshape(B, 1, 1)).cast(dtypes.float)
        tokens_first = []
        for k in range(n_loops):
            wk_at = (waist_per_breath_buf[k] * gather_mask_first).sum(axis=1, keepdim=True)
            # v63: pass (k_idx, K_total) for K-pos embed lookup.
            lk = self.waist_controller.forward(wk_at, prompt_emb_buf_t, self.embed_out,
                                                 k_idx=k, K_total=n_loops)
            tk = lk[:, :, :50277].argmax(axis=-1).reshape(B)
            tokens_first.append(tk)
        argmax_first = Tensor.stack(*tokens_first, dim=0).realize().numpy()  # (K, B)

        # First token: always use breath 0 (step_counter starts at 0, no #### seen yet)
        first_ids = np.array([argmax_first[0, b] for b in range(B)], dtype=np.int32)
        argmax_buf.assign(_T(argmax_first, dtype=dtypes.int)).realize()
        outs = [[int(first_ids[b])] for b in range(B)]
        for b in range(B):
            decoded_so_far[b].append(int(first_ids[b]))
        is_done = [False] * B
        for b in range(B):
            if outs[b][0] in stop_set:
                is_done[b] = True
            elif seq_len > 0 and outs[b][-seq_len:] == seq:
                is_done[b] = True
        if all(is_done):
            return outs

        # Initialize Stage 2 persistent state:
        # - prev_id_t = first generated token (to be written at real_lens[b])
        # - t_pos_t = real_lens[b] (first write position)
        # - step_counter_t = 0 for all examples (no #### seen yet)
        prev_id_t.assign(_T(first_ids.reshape(B, 1), dtype=dtypes.int)).realize()
        t_pos_per = [real_lens[b] for b in range(B)]
        t_pos_t.assign(_T(np.array(t_pos_per, dtype=np.int32), dtype=dtypes.int)).realize()
        step_counter_t.assign(_T(np.zeros(B, dtype=np.int32), dtype=dtypes.int)).realize()

        # === Stage 2: closure-based fused JIT decode loop ===
        # The JIT body:
        #   1. Embeds _prev_id_t, runs K-breath forward at _t_pos_t
        #   2. Scatters waist/emb into persistent buffers
        #   3. Controller decode → (K, B) argmax per breath
        #   4. Checks _prev_id_t for #### → updates _step_counter for next step
        #      BUT selects token using the CURRENT (pre-update) step counter
        #   5. Assigns selected token to _prev_id_t (for next step)
        #   6. Increments _t_pos_t
        # After the JIT call, _prev_id_t holds the selected token for this step.
        # One numpy() read per step for EOS/stop detection.
        for _step_idx in range(max_new - 1):
            if min(t_pos_per) >= max_len:  # use min to stop when ALL are at capacity
                break

            jit_step_closure()

            # Read selected tokens — ONE numpy() call per step (vs 2 in the unfused path)
            next_ids = prev_id_t.numpy().reshape(B)
            # t_pos has been incremented inside JIT; track Python-side via +1
            t_pos_per = [tp + 1 for tp in t_pos_per]

            for b in range(B):
                if is_done[b]:
                    continue
                tok_b = int(next_ids[b])
                outs[b].append(tok_b)
                decoded_so_far[b].append(tok_b)
                if tok_b in stop_set:
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
            # v44 doubled-layers: pick Set A or Set B based on breath index
            if DOUBLED_LAYERS and loop >= (cfg.max_loops // 2):
                active_layers = self.block.layers_b
            else:
                active_layers = self.block.layers
            for li, layer in enumerate(active_layers):
                x, (k_part, v_part) = layer.forward_with_kv(x, loop_idx=loop)
                pad_n = max_len - int(k_part.shape[2])
                k_full = k_part.pad(((0, 0), (0, 0), (0, pad_n), (0, 0))).contiguous().realize()
                v_full = v_part.pad(((0, 0), (0, 0), (0, pad_n), (0, 0))).contiguous().realize()
                cache_k[li][loop] = k_full
                cache_v[li][loop] = v_full
            # v39 end-of-breath waist
            if BFIELD_WAIST > 0 and BFIELD_END_OF_BREATH:
                x = self.block.apply_bfield_waist(x, loop_idx=loop)
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
            layers_b = self.block.layers_b  # v44 doubled-layers
            max_loops_local = cfg.max_loops
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
                    # v44 doubled-layers: pick Set A or Set B based on breath index
                    if DOUBLED_LAYERS and loop >= (max_loops_local // 2):
                        active_layers_local = layers_b
                    else:
                        active_layers_local = layers
                    for li in range(n_layers_local):
                        idx = li * n_loops_local + loop
                        x, k_new, v_new = active_layers_local[li].forward_cached_step_jit(
                            x, loop, ck[idx], cv[idx], t_pos_t
                        )
                        new_ck[idx] = k_new
                        new_cv[idx] = v_new
                        # v38 B-field — mid-breath waist after L1 (when not end-of-breath)
                        if BFIELD_WAIST > 0 and li == 1 and not BFIELD_END_OF_BREATH:
                            x = block_local.apply_bfield_waist(x, loop_idx=loop)
                    # v39 end-of-breath waist
                    if BFIELD_WAIST > 0 and BFIELD_END_OF_BREATH:
                        x = block_local.apply_bfield_waist(x, loop_idx=loop)
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
