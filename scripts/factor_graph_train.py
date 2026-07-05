"""factor_graph_train.py — GENERAL trainer for the Mycelium factor-graph engine.

The first TRAINING driver for mycelium/factor_graph_engine.py (so far only the
forwards were exercised). It clones the STRUCTURE of scripts/kenken_train.py —
the TinyJit'd train step, the _compile_step cache-key discipline (every shape /
runtime-determining constant in the key), the AdamW optimizer over
factor_graph_parameters(model)+backbone, global-norm grad clip, per-step logging
(loss, per-breath-CE ladder, cell_acc/puzzle_acc), periodic held-out eval,
checkpoint save — but drives the GENERAL engine instead of the KenKen-specific
forward, so a NEW factor-graph domain is a TASK SWITCH, not a new trainer.

GENERAL ENGINE (mycelium/factor_graph_engine.py):
  attach_factor_graph_params(model, hidden, spec)  — allocate fg_* params
  factor_breathing_forward(model, batch, spec, K)  — (logits_hist, calib_hist)
  factor_loss(logits_hist, calib_hist, batch, spec, ...) -> (total, parts)
  factor_accuracy(logits_final, batch, spec)       -> (cell_acc, puzzle_acc)
  factor_graph_parameters(model)                   — trainable fg params
  FactorGraphBatch / FactorGraphSpec / make_kenken_factor_batch

TASK SWITCH (FG_TASK env, default kenken):
  kenken   spec=FactorGraphSpec(s_max=49, n_values=7, n_factor_types=3,
                                n_heads=16, k_max=K, has_factor_inlet=True).
           KenKenLoader (mycelium.kenken_data) → make_kenken_factor_batch with a
           PRE-BUILT verification inlet (build_verification_inlet); the per-domain
           constraint-energy plug is kenken_constraint_energy. This is also the
           TRAIN-parity check vs the v98 KenKen executor (same loader, same inlet,
           same energy → the general engine is byte-identical at matching knobs).
  coloring spec=FactorGraphSpec(s_max=<FG_S_MAX>, n_values=FG_N_VALUES,
                                n_factor_types=1, n_heads=16, k_max=K,
                                has_factor_inlet=False).
           GraphColoringLoader (mycelium.graph_coloring_data — IN-MEMORY generator,
           NOT path-based) produces FactorGraphBatch-compatible batches directly
           (input_cells, membership (B,L,s_max), latent_type, cell_valid,
           value_domain_mask (B,s_max,k), gold (B,s_max)). No verification inlet,
           no domain constraint energy (CE + calib only). The import is lazy
           (inside the coloring branch) so a missing data module never breaks the
           kenken path or the trainer's CPU import.
  circuit  spec=FactorGraphSpec(s_max=<FG_S_MAX>, n_values=2,
                                n_factor_types=3 (AND/OR/NOT; 4 when XOR on via
                                FG_CIRCUIT_XOR=1), n_heads=16, k_max=K,
                                has_factor_inlet=False).
           CircuitLoader (mycelium.circuit_data — IN-MEMORY generator, NOT
           path-based; the RUNG-1 deduction-depth testbed: a Boolean circuit is a
           DAG, so accuracy-vs-level is THE deduction-depth read) produces
           FactorGraphBatch-compatible batches directly (input_cells, membership
           (B,L,49) with one factor per GATE, latent_type {0..T-1 gate types /
           T=global sentinel for padding gate rows}, cell_valid, value_domain_mask
           (B,49,2) [two boolean values], gold (B,49)) plus python-side per-NODE
           lvl + per-instance circuit_depth/band/n metadata. No verification inlet,
           no domain constraint energy (CE + calib only). Lazy import so a missing
           data module never breaks the kenken/coloring paths or the CPU import.

JIT cache key (PORT from kenken_train) includes EVERY shape/runtime-determining
constant AND the spec params: task, s_max, n_values, n_factor_types,
has_factor_inlet, K, B, plus the loss knobs (constraint_weight, calib_weight,
ortho_lambda, label_smoothing, stoch_depth_p, grad_clip). Different specs ARE
different graphs — a stale graph on a wrong spec is silent corruption.

DIMS are parameterized: hidden is read from the backbone Config (cfg.hidden);
nothing hardcodes 1024, so a future 4k-dim backbone swap is a config change.

SUBSTRATE (tinygrad + AMD AM driver):
  - No dtypes.float32 Tensor literal baked as a JIT graph constant (build float
    consts as numpy then wrap — the engine + masks already follow this).
  - scores.clip(-1e4,1e4) lives inside kenken_layer_forward (the engine threads
    attn_bias through it) — the loop never re-clips.
  - NaN guard via where()-gated SELECT (NOT multiply: NaN*0=NaN poisons Adam),
    single-kernel total.isfinite(), NO per-param isnan() loop (AM-driver segfault).
  - Toggles + spec params in the JIT cache key; no host sync inside the JIT body.

Env vars:
  FG_TASK=kenken|coloring        task switch (default kenken)
  FG_K_MAX / K=16                number of iterative-prefill breaths
  BATCH=8
  STEPS=2000
  LR=3e-5
  RUN_NAME=fg_<task>
  RESUME_FROM=                  warm-start from a saved fg ckpt (default COLD)
  PYTHIA_INIT=1                 init L0-L3 from Pythia-410M (default 1)
  SEED=42
  CKPT_EVERY=200  EVAL_EVERY=100  LOG_EVERY=10  PER_BREATH_CE_EVERY=50  GC_EVERY=50
  EVAL_BATCHES=20  EVAL_BATCH=<BATCH>
  GRAD_CLIP=0.0
  FG_CONSTRAINT_WEIGHT=0.0      weight on the domain constraint-energy plug
  FG_CALIB_WEIGHT=0.1
  FG_ORTHO_LAMBDA=0.0           codebook-orthogonality penalty (0 = off)
  LABEL_SMOOTHING=0.0  WEIGHT_DECAY=0.05  STOCH_DEPTH_P=0.0
  --- kenken task ---
  FG_TRAIN=.cache/kenken_train.jsonl   FG_TEST=.cache/kenken_test.jsonl
  KENKEN_N_CAGES_MAX=<corpus max>
  --- coloring task ---
  FG_S_MAX=<grid size>          sequence length (n vertices padded)
  FG_N_VALUES=<k>               number of colors (codebook size)
  FG_N_FACTOR_TYPES=1           coloring has one relation (edge); default 1
  FG_N_INSTANCES=8000           total corpus size (GraphColoringLoader generates in-memory)
  --- circuit task ---
  FG_S_MAX=49                   sequence length (n gate/input nodes padded; engine
                                asserts S==49, so keep 49)
  FG_N_INSTANCES=8000           total corpus size (CircuitLoader generates in-memory)
  FG_CIRCUIT_XOR=0              include XOR gates (T=4 when on; default off -> T=3
                                AND/OR/NOT). spec.n_factor_types follows this flag.
  FG_CIRCUIT_GATE_TYPES=        optional explicit gate-type list (comma-separated,
                                e.g. "and,or,not"); overrides FG_CIRCUIT_XOR. The
                                loader sizes T from this; the spec follows the loader.
"""
import gc
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from tinygrad import Tensor, Device, dtypes
from tinygrad.helpers import getenv
from tinygrad.engine.jit import TinyJit
from tinygrad.nn.optim import AdamW
from tinygrad.nn.state import safe_save, safe_load

from mycelium import Config, BreathingTransformer
from mycelium.loader import _load_state, load_breathing
from mycelium.factor_graph_engine import (
    FactorGraphSpec, FactorGraphBatch,
    attach_factor_graph_params, factor_graph_parameters,
    factor_breathing_forward, factor_loss, factor_accuracy,
    make_kenken_factor_batch,
    attach_factor_waist_params, factor_waist_parameters, pooled_waist_drep,
    attach_factor_lora_params, factor_lora_parameters,
    FG_WAIST_DIM, FG_WAIST_AFTER, FG_WAIST_GATE_INIT,
    FG_HYP_MASK, FG_HYP_FREEZE,
)
from mycelium.factor_masks import (
    attach_factor_hyperbolic_params,
    clamp_factor_hyp_tangent_norms,
    FG_HYP_RELAX, FG_HYP_EUCLID,
)
from mycelium.factor_inlet import (
    GLOBAL_TYPE_IDS, N_GLOBAL_TYPES,
    attach_factor_inlet_params, factor_inlet_param_names,
    factor_inlet_parameters, build_generic_factor_inlet,
)


# ---------------------------------------------------------------------------
# fp32 cast of the Pythia L0-L3 weights (mirror kenken_train.cast_layers_fp32).
# Only WEIGHTS go to fp32; the inter-breath activation carry stays fp16 (the
# validated v98 recipe — the engine casts x to half in the forward).
# ---------------------------------------------------------------------------

def cast_layers_fp32(model):
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


def collect_backbone_params(model) -> list[Tensor]:
    """The shared L0-L3 attn/FFN + final LN params (the backbone trainables).

    Mirror of kenken_train.collect_kenken_params' backbone half, MINUS the
    kenken-specific heads (those are the fg_* params from factor_graph_parameters).
    Token embedding / embed_out / lookup / controllers / notebooks are untouched
    by the factor forward and are excluded.
    """
    params: list[Tensor] = []
    sw = model.block.shared
    params += [sw.wv, sw.bv, sw.wo, sw.bo, sw.w_out, sw.b_out,
               sw.in_ln_g, sw.in_ln_b, sw.post_ln_g, sw.post_ln_b]
    for layer in model.block.layers:
        params += [layer.wq, layer.bq, layer.wk, layer.bk, layer.w_in, layer.b_in]
    params += [model.ln_f_g, model.ln_f_b]
    return params


# ---------------------------------------------------------------------------
# Checkpoint state dict — backbone (shared + L0-L3 + final LN) + fg params.
# ---------------------------------------------------------------------------

_FG_PARAM_NAMES = [
    "fg_state_embed", "fg_position_embed", "fg_value_codebook",
    "fg_calib_head_w", "fg_calib_head_b", "fg_breath_embed", "fg_delta_gate",
]
# CONTINUOUS-INPUT embed params (only present when spec.continuous_input -> ECC). Saved
# only when attached, so a discrete-input (kenken/coloring/circuit) ckpt is byte-identical
# (it never carries these keys); load_ckpt keeps init for absent keys.
_FG_CONT_NAMES = [
    "fg_cont_embed_w", "fg_cont_embed_b",
]
# PER-BREATH LoRA params (only present when FG_LORA_RANK>0 / attach_factor_lora_params
# was called). Saved only when attached, so a no-LoRA ckpt is byte-identical (it never
# carries these keys) and a LoRA ckpt round-trips them. load_ckpt keeps init for absent keys.
_FG_LORA_NAMES = [
    "fg_lora_A", "fg_lora_B",
]
# In-deducer WAIST params (only present when FG_WAIST=1 / attach_factor_waist_params was
# called). Saved only when attached, so a baseline (no-waist) ckpt is byte-identical (it
# never carries these keys) and a waist ckpt round-trips them. fg_waist_aux_* are present
# only for aux in {classify, both}; load_ckpt keeps init for absent keys.
_FG_WAIST_NAMES = [
    "fg_waist_down", "fg_waist_down_b", "fg_waist_up", "fg_waist_up_b",
    "fg_waist_gate", "fg_waist_aux_w", "fg_waist_aux_b",
]
# kenken verification-inlet params (only present when the kenken inlet is attached).
_KENKEN_INLET_NAMES = [
    "kenken_op_embed", "kenken_target_embed", "kenken_size_embed",
    "kenken_inlet_w", "kenken_inlet_b",
]
# Generic semantics-as-input inlet params (only present in the MULTI-TASK path,
# attach_factor_inlet_params). Saved only when attached, so single-domain ckpts are
# byte-identical (they never carry these keys).
_GENERIC_INLET_NAMES = factor_inlet_param_names()


def model_state_dict_fg(model) -> dict:
    sd = {"ln_f.g": model.ln_f_g, "ln_f.b": model.ln_f_b}
    sw = model.block.shared
    for a in ("wv", "bv", "wo", "bo", "w_out", "b_out",
              "in_ln_g", "in_ln_b", "post_ln_g", "post_ln_b"):
        sd[f"shared.{a}"] = getattr(sw, a)
    for i, layer in enumerate(model.block.layers):
        for a in ("wq", "bq", "wk", "bk", "w_in", "b_in"):
            sd[f"phase{i}.{a}"] = getattr(layer, a)
    for nm in _FG_PARAM_NAMES:
        sd[nm] = getattr(model, nm)
    # CONTINUOUS-INPUT embed params (saved only when attached -> discrete ckpts byte-identical).
    for nm in _FG_CONT_NAMES:
        t = getattr(model, nm, None)
        if t is not None:
            sd[nm] = t
    # PER-BREATH LoRA params (saved only when attached -> no-LoRA ckpts byte-identical).
    for nm in _FG_LORA_NAMES:
        t = getattr(model, nm, None)
        if t is not None:
            sd[nm] = t
    # In-deducer WAIST params (saved only when attached -> baseline ckpts byte-identical).
    for nm in _FG_WAIST_NAMES:
        t = getattr(model, nm, None)
        if t is not None:
            sd[nm] = t
    # Verification-inlet params are saved only when attached (kenken task).
    for nm in _KENKEN_INLET_NAMES:
        t = getattr(model, nm, None)
        if t is not None:
            sd[nm] = t
    # Generic semantics-as-input inlet params (only present in the multi-task path).
    for nm in _GENERIC_INLET_NAMES:
        t = getattr(model, nm, None)
        if t is not None:
            sd[nm] = t
    # Per-type hyperbolic anchor tables (only present when FG_HYP_MASK=1).
    # Saved regardless of FG_HYP_FREEZE so a RELAXED run's trained anchors
    # are preserved and a round-trip load_ckpt restores them exactly.
    if FG_HYP_MASK:
        t_idx = 0
        while True:
            anchors = getattr(model, f"fg_hyp_anchors_{t_idx}", None)
            if anchors is None:
                if t_idx > 64:
                    break
                t_idx += 1
                continue
            sd[f"fg_hyp_anchors_{t_idx}"] = anchors
            t_idx += 1
    return sd


def load_ckpt(model, path: str, strict: bool = False):
    sd = safe_load(path)
    targets = model_state_dict_fg(model)
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
                missing.append(f"{name} (shape mismatch)")
                continue
        if src.dtype != dst.dtype:
            src = src.cast(dst.dtype)
        dst.assign(src).realize()
    if missing:
        # keep-init is fine for a warm-start (new params start at init on purpose), but
        # in eval-only it is a FALSE-RESULT generator: the model scores with init weights
        # and the number looks plausible (2026-07-04: a silent 43-key fallback on a
        # hidden=2048 ckpt scored chance as if it were the model).
        if strict:
            raise RuntimeError(
                f"eval-only ckpt load is INCOMPLETE: {len(missing)} keys kept init "
                f"({missing[:5]}{'...' if len(missing) > 5 else ''}) — wrong ckpt for "
                f"this model build (check hidden dims / task spec in measured_config.json)")
        print(f"  ckpt missing {len(missing)} keys (kept init): "
              f"{missing[:3]}{'...' if len(missing) > 3 else ''}")


# ---------------------------------------------------------------------------
# WAIST VALIDITY-SHAPING AUX (FG_WAIST_AUX) — the research-uncertain crux.
# ---------------------------------------------------------------------------
# The FREE EXACT verifier (a coloring is proper iff no edge factor's two members share a
# color; generically: no all-different factor has a repeated value) is computed IN-GRAPH
# from membership + the predicted per-cell argmax, so it costs no extra forward and gives a
# per-instance VALID/INVALID label for the model's OWN output. Gold is valid by definition.
#
# classify : a head on the POOLED waist d-rep predicts validity (class-weighted BCE).
#            VALID exemplars = GOLD teacher-forced (input_cells=gold -> output==gold ->
#            always valid -> its waist d-rep is a guaranteed-valid silhouette). INVALID
#            exemplars = the ACTUAL forward's waist d-rep labeled by the in-graph verifier
#            on its own argmax. The aux grad flows through the waist (down/up + gate),
#            shaping the d-rep common mode to be validity-aware. DISCRIMINATIVE: makes the
#            rep SEPARABLE; may NOT move the output distribution (the honest risk).
# attract  : GENERATIVE shaping (the p-raising candidate). Maintain a running VALID CENTROID
#            of gold-TF waist d-reps (an EMA buffer the trainer owns, detached). Add a term
#            that PULLS the ACTUAL forward's pooled d-rep toward that centroid, WEIGHTED by
#            how near-valid the output is (more pull when the output already satisfies most
#            factors), biasing the deduction toward the good common mode. This shapes the
#            ENERGY, not just a classifier boundary -> most likely to actually raise p.
# both     : classify + attract.
# none     : no aux term (pure-CE waist; the ablation).
#
# ALL aux terms are ZERO-able (aux=='none' -> the whole block is dead) and gate behind
# use_waist (no waist params -> no aux). The gold-TF second forward shares the SAME weights
# (iterative-prefill recipe) so it is a cheap extra K-breath pass; both forwards see the
# SAME masks (same membership) so the d-reps live in one space.


def _factor_alldiff_violation(pred_onehot: "Tensor", membership: "Tensor",
                              latent_type: "Tensor", n_factor_types: int) -> "Tensor":
    """Per-instance count of all-different VIOLATIONS over real (typed) factors, IN-GRAPH.

    pred_onehot : (B, S, N) one-hot of the predicted value per cell (argmax; detached).
    membership  : (B, L, S) — factor membership (real factors have >=2 ones; pad rows 0).
    latent_type : (B, L) int — a REAL relation iff 0 <= t < n_factor_types (the global
                  sentinel id >= n_factor_types marks padding). All current relation types
                  (coloring edge, kenken row/col/cage) are ALL-DIFFERENT constraints, so the
                  generic verifier is: for each real factor, count(value) <= 1 for every
                  value. counts = membership @ pred_onehot -> (B, L, N); a violation is any
                  count >= 2. Returns (B,) total violations -> 0 == proper/valid.

    This is the EXACT free verifier (_coloring_proper_np's edge check generalized to cliques)
    expressed as tensor ops -> no host sync, JIT-safe, no float32 literal."""
    B = int(membership.shape[0])
    L = int(membership.shape[1])
    is_real = (latent_type < n_factor_types).cast(dtypes.float).reshape(B, L, 1)  # (B,L,1)
    counts = membership.cast(dtypes.float) @ pred_onehot.cast(dtypes.float)        # (B,L,N)
    # excess over the all-different cap of 1 per value, only on real factors.
    excess = (counts - 1.0).maximum(0.0) * is_real                                 # (B,L,N)
    return excess.sum(axis=(1, 2))                                                 # (B,)


# ---------------------------------------------------------------------------
# COLORING OUTPUT-SPACE SOFT-VIOLATION ENERGY (the differentiable verifier plug).
# ---------------------------------------------------------------------------
# A differentiable RELAXATION of the coloring verifier, minimized alongside CE to steer
# the OUTPUT toward valid colorings (vs a discriminative head that only makes a rep
# separable). It REUSES the generic all-different form (_factor_alldiff_violation): an
# edge is a 2-member not-equal factor, so for each color c the membership@probs mass over
# the edge's two endpoints is <=1 iff they disagree; relu(mass-1) is the soft collision
# (positive only when both endpoints put overlapping mass on the same color). Summed over
# real edge factors per instance -> the soft "adjacent vertices share a color" penalty.
#
# THE KEY (differentiability): apply it to the SOFTMAX PROBS (the caller hands
# `final_probs = logits_history[-1].softmax(...)`), NOT an argmax one-hot. The waist aux
# used a DETACHED argmax one-hot (a label); here the gradient must flow to the probs, so
# there is NO .detach()/argmax in this path. membership/latent_type are inputs (no grad),
# so grad flows only into final_probs -> the readout -> the breathing weights.
#
# SIGN (the load-bearing property): the energy is NON-NEGATIVE (relu) and ZERO for a valid
# coloring (no edge has both endpoints on one color -> every per-color mass <=1 -> relu=0)
# and POSITIVE for violations (a shared color drives an endpoint pair's mass toward 2 ->
# relu>0). The loss ADDS constraint_weight*energy and the optimizer MINIMIZES, so reducing
# the energy REDUCES collisions => pushes toward valid. (Verified: gold-onehot -> ~0;
# all-same-color -> large positive.)
#
# GENERALITY: this lives in the trainer/domain layer (like kenken_constraint_energy), NOT
# the engine core, and reuses the GENERIC _factor_alldiff_violation helper — no coloring-
# only specifics leak into the core. It is a factory so T (= spec.n_factor_types, the
# real-vs-sentinel cutoff the helper needs) is captured at task-build time; the returned
# callable matches the caller's exact `fn(final_probs, batch) -> (B,)` contract.

def make_coloring_constraint_energy(n_factor_types: int):
    """Return a coloring soft-violation energy fn closing over T (= n_factor_types).

    The returned callable has the exact constraint-energy contract the trainer's loss
    graph calls (`fn(final_probs, batch) -> (B,)`): same signature as
    kenken_constraint_energy. `final_probs` is the FINAL-breath SOFTMAX over values
    (B, S, N) — already value-domain masked by the readout's value_bias — so it is
    differentiable end-to-end. Pad cells are zeroed (their prob mass must not create a
    spurious collision) BEFORE the all-different reduction; this is dtype-preserving and
    JIT-safe (no host sync, no dtypes.float32 literal)."""
    T = int(n_factor_types)

    def coloring_constraint_energy(final_probs: "Tensor", batch) -> "Tensor":
        # final_probs: (B, S, N) softmax over values (differentiable; NOT argmax'd).
        B = int(final_probs.shape[0])
        S = int(final_probs.shape[1])
        # Zero pad cells so padding never contributes a spurious clash (mirrors the waist
        # verifier's pad-mask). cell_valid is an input (no grad); this is a SELECT-by-mask
        # multiply on the LIVE probs, so gradient still flows to the valid cells' probs.
        cv = batch.cell_valid.reshape(B, S, 1).cast(final_probs.dtype)
        pv = final_probs * cv                                            # (B, S, N)
        # Reuse the GENERIC all-different soft violation on the SOFTMAX (differentiable).
        return _factor_alldiff_violation(pv, batch.membership, batch.latent_type, T)  # (B,)

    return coloring_constraint_energy


# ---------------------------------------------------------------------------
# KENKEN CAGE-ARITHMETIC SOFT ENERGY (trainer-local; the oracle stays untouched).
#
# kenken_constraint_energy (mycelium/kenken.py) is row/col AllDiff ONLY — it does
# NOT re-derive cage arithmetic (its docstring's "cage soft-AllDiff" is a no-op in
# the code). This factory adds a DIFFERENTIABLE cage-arithmetic surrogate on the
# SOFTMAX PROBS via per-cell EXPECTED VALUES, segment-summed per cage from
# cell_cage_id. It lives in the trainer/domain layer (like make_coloring_*), reads
# ONLY existing batch attributes, and NEVER touches the oracle / engine / data
# layer. Gated by FG_ENERGY_CAGE (default 0 = OFF -> the trainer is byte-identical
# to the current row/col-only behavior).
#
# TARGET RECONSTRUCTION (load-bearing): batch.kenken_cage_target carries the
# LOG-BUCKET id (0..31 from kenken_data.target_to_bucket), NOT the raw integer
# target. We invert the bucketing to a bucket-center magnitude
#   t_hat = TARGET_MAX ** (bucket / (TARGET_BUCKETS-1))
# (the inverse of b = floor(log t / log TARGET_MAX * (B-1))). This is a pure tensor
# op on the int JIT input cast to float — no host sync, no .numpy() in the graph.
# The reconstruction is lossy (bucket centers, not exact targets), so the cage
# energy is a SHAPING surrogate, not an exact verifier; constraint_weight controls
# its strength. (Constants mirror kenken_data: TARGET_BUCKETS=32, TARGET_MAX=1000.)
#
# SIGN: every op term is a squared error (or |.|-based), so a SATISFIED cage gives
# ~0 and a VIOLATED cage gives POSITIVE energy; the loss ADDS constraint_weight *
# energy and the optimizer MINIMIZES -> reducing it pushes toward arithmetic
# feasibility. Mirrors build_kenken_data.cage_ok (the GOLD reference) relaxed via
# E[v] per cell.

# Constants (mirror mycelium/kenken_data: TARGET_BUCKETS / TARGET_MAX). Kept local
# so the trainer never imports the data layer just for two scalars.
_KK_TARGET_BUCKETS = 32
_KK_TARGET_MAX = 1000.0


def make_kenken_cage_constraint_energy(n_cages_max: int, n_values: int = 7):
    """Return a differentiable KenKen cage-arithmetic soft-violation energy fn.

    Closes over n_cages_max (pinned at task-build) and n_values (codebook size). The
    returned callable matches the trainer's constraint-energy contract exactly:
    `fn(final_probs, batch) -> (B,)`, same signature as kenken_constraint_energy and
    the coloring energy. `final_probs` is the per-breath SOFTMAX over values
    (B, S, N) (differentiable; already value-domain-masked by the readout's
    value_bias). Reads batch.kenken_cage_op/target/size + batch.kenken_cell_cage_id.

    PURE TENSOR OPS (JIT-safe: no host sync, no .numpy()/.realize(), no
    dtypes.float32 literal). Returns (B,) cage energy ready for .mean() in the loss
    graph (or summed alongside the row/col energy)."""
    C = int(n_cages_max)
    N = int(n_values)
    eps = 1e-4

    def kenken_cage_constraint_energy(final_probs: "Tensor", batch) -> "Tensor":
        B = int(final_probs.shape[0])
        S = int(final_probs.shape[1])

        # (1) Per-cell EXPECTED VALUE E[v] = sum_v v * p(v). The value axis index n
        # (0..N-1) decodes to digit (n+1) (codebook col 0 = digit 1). Build the digit
        # weights from numpy (no float literal baked as a graph const).
        digits = Tensor(np.arange(1, N + 1, dtype=np.float32).reshape(1, 1, N),
                        dtype=dtypes.float)                                   # (1,1,N)
        cv = batch.cell_valid.reshape(B, S, 1).cast(dtypes.float)            # (B,S,1)
        pv = final_probs.cast(dtypes.float) * cv                            # zero pad cells
        exp_val = (pv * digits).sum(axis=-1)                                 # (B,S) E[v], pad cells ~0

        # (2) Segment membership: one-hot of cell->cage on cell_cage_id (-1 pad ->
        # all-zero row -> contributes to no cage). (B, S, C).
        cell_cage_id = batch.kenken_cell_cage_id                             # (B,S) int (-1 pad)
        oh = cell_cage_id.one_hot(C).cast(dtypes.float)                      # (B,S,C)

        # (3a) Per-cage SUM of expected values (add / given use this).
        #   cage_sum[b,c] = sum_s oh[b,s,c] * exp_val[b,s]
        cage_sum = (oh * exp_val.reshape(B, S, 1)).sum(axis=1)               # (B,C)

        # (3b) Per-cage PRODUCT via exp(sum(log(E[v]+eps))) over MEMBER cells only.
        # Non-member cells (oh=0) contribute 0 to the log-sum -> factor 1. Guard the
        # log arg with eps (E[v] in [0, N], pad cells exp_val~0 but are non-members).
        log_ev = (exp_val + eps).log().reshape(B, S, 1)                      # (B,S,1)
        cage_logprod = (oh * log_ev).sum(axis=1)                             # (B,C)
        cage_prod = cage_logprod.exp()                                       # (B,C)

        # (3c) Per-cage extremes for sub/div (2-cell cages). Max member E[v] and the
        # SECOND value (= sum - max for a 2-cell cage). Build max via a large negative
        # bias on non-members so the row max ignores them.
        neg = (1.0 - oh) * (-1e4)                                            # (B,S,C) -inf on non-members
        member_ev = exp_val.reshape(B, S, 1) + neg                          # (B,S,C)
        cage_max = member_ev.max(axis=1)                                     # (B,C) largest member E[v]
        cage_min = cage_sum - cage_max                                       # (B,C) the OTHER cell (2-cell)

        # (4) Reconstruct the integer target from the log-bucket id (bucket center).
        bucket = batch.kenken_cage_target.cast(dtypes.float)                 # (B,C) bucket id 0..31
        frac = bucket / float(_KK_TARGET_BUCKETS - 1)                        # (B,C) in [0,1]
        # t_hat = TARGET_MAX ** frac = exp(frac * log TARGET_MAX) — pure tensor op on
        # the int JIT input (no full_like constant, no host sync).
        target = (frac * float(math.log(_KK_TARGET_MAX))).exp()              # (B,C) ~ raw target

        # (5) Per-op masks from the integer op id (0=given,1=add,2=sub,3=mul,4=div).
        m_given = (batch.kenken_cage_op == 0).cast(dtypes.float)            # (B,C)
        m_add = (batch.kenken_cage_op == 1).cast(dtypes.float)
        m_sub = (batch.kenken_cage_op == 2).cast(dtypes.float)
        m_mul = (batch.kenken_cage_op == 3).cast(dtypes.float)
        m_div = (batch.kenken_cage_op == 4).cast(dtypes.float)

        # (6) RELATIVE squared arithmetic error per cage, scale-free via 1/(target+1):
        # ((value - target)/(target+1))^2. ABSOLUTE error is unusable: at init (near-
        # uniform probs, E[v]~mid) a mul/add cage reads hundreds-to-thousands, and the
        # scale varies ~100x across ops/targets (a target-2 div vs a target-1000 mul),
        # so a single cw cannot balance it and the term swamps CE + the row/col energy.
        # Relative error puts every cage on a comparable O(1) scale. SAT->0, VIOL->+.
        inv_t = 1.0 / (target + 1.0)                                        # (B,C) scale-free norm
        def _rel(v):                                                        # ((v-target)*inv_t)^2
            d = (v - target) * inv_t
            return d * d
        #   given/add : value = sum E[v];  mul : prod E[v];  sub (2-cell): max-min
        r_given = _rel(cage_sum)
        r_add = _rel(cage_sum)
        r_mul = _rel(cage_prod)
        r_sub = _rel(cage_max - cage_min)
        #   div   : the KenKen div clue is a DISJUNCTION (a/b==target OR b/a==target), so
        #   the relaxation is the MIN of the two directional rel-errors (whichever matches
        #   drives it to 0) — NOT a sum/average, which double-penalizes the always-wrong
        #   direction on gold. cage_max>=cage_min so ratio_hi=max/min>=1 is canonical.
        ratio_hi = cage_max / (cage_min + eps)
        ratio_lo = (cage_min + eps) / (cage_max + eps)
        r_div = _rel(ratio_hi).minimum(_rel(ratio_lo))

        # (7) Select the active op's rel-error per cage; BOUND each to [0,1) via r/(1+r)
        # so no single (huge, early) cage dominates — linear ~r near SAT (clean gradient),
        # saturating when far (stable). Mask to REAL cages (size>0).
        r = (m_given * r_given + m_add * r_add + m_sub * r_sub
             + m_mul * r_mul + m_div * r_div)                              # (B,C)
        bounded = r / (1.0 + r)                                            # (B,C) in [0,1)
        real_cage = (batch.kenken_cage_size > 0).cast(dtypes.float)        # (B,C)
        bounded = bounded * real_cage                                      # zero padding cages

        # (8) MEAN over real cages -> (B,) energy in [0,1), comparable to the row/col
        # term's natural scale (~0 at uniform, O(few) under collisions) -> a fair cw.
        n_real = real_cage.sum(axis=1).maximum(1.0)                        # (B,) >=1
        return bounded.sum(axis=1) / n_real                               # (B,)

    return kenken_cage_constraint_energy


# ---------------------------------------------------------------------------
# PER-BREATH ENERGY WAVE — the weight schedule w_k over K breaths.
#
# Three modes (env FG_ENERGY_WAVE, default "off"):
#   off         : final breath only (w_{K-1}=1, else 0) -> the per-breath loop is
#                 NEVER traced (the OFF branch is a python compile-time decision in
#                 the loss graph), so the graph is BYTE-IDENTICAL to the current code.
#   monotonic   : w_k = (1 + k/(K-1)) / K  -> normalized so sum_k w_k = 1 (mirrors the
#                 CE-ladder weight; sum of (1+k/(K-1)) over k = K, divide by K).
#   oscillating : w_k = |cos(pi*k*cycles/K)| normalized so sum_k w_k = 1 (V-cycle:
#                 emphasize coarse k=0 + fine k=K, de-emphasize the middle).
#
# All three are normalized so a CONSTANT per-breath energy yields the SAME total
# magnitude as the off control (a fair A/B). The weights are PYTHON floats computed
# from numpy constants + the unrolled int loop index k (no Tensor branch, no float
# literal baked as a graph const). The schedule is a JIT-keyed compile-time constant.
# ---------------------------------------------------------------------------

def _energy_wave_weights(K: int, mode: str, cycle_depth: int = 2) -> list:
    """Return python floats [w_0..w_{K-1}], normalized so sum == 1.0.

    off: final-breath only (but this helper is NOT called on the off path — the off
    path stays the original no-loop final-breath code for byte-identical tracing).
    Included for completeness / unit-test symmetry."""
    K = int(K)
    mode = str(mode).strip().lower()
    if K <= 1:
        return [1.0]
    if mode == "off":
        return [0.0] * (K - 1) + [1.0]
    if mode == "monotonic":
        raw = [1.0 + float(k) / float(K - 1) for k in range(K)]
    elif mode == "oscillating":
        cyc = int(cycle_depth)
        raw = [abs(math.cos(math.pi * float(k) * float(cyc) / float(K)))
               for k in range(K)]
    else:
        raise ValueError(f"unknown FG_ENERGY_WAVE mode {mode!r}")
    s = sum(raw)
    if s <= 0.0:
        # degenerate (all-zero, e.g. an oscillating schedule that hits only zeros):
        # fall back to final-breath only so the term is still defined + comparable.
        return [0.0] * (K - 1) + [1.0]
    return [w / s for w in raw]


# ---------------------------------------------------------------------------
# JIT train step — where()-gated NaN guard (PORT from kenken_train).
# ---------------------------------------------------------------------------

_JIT_FG_CACHE: dict = {}


def _compile_jit_fg_step(model, opt, spec: FactorGraphSpec, task: str,
                         K: int, B: int,
                         constraint_weight: float, calib_weight: float,
                         ortho_lambda: float = 0.0,
                         grad_clip: float = 0.0,
                         label_smoothing: float = 0.0,
                         stoch_depth_p: float = 0.0,
                         constraint_energy_fn=None,
                         has_inlet: bool = False,
                         multitask: bool = False, mix_key: str = "",
                         waist_on: bool = False, waist_aux: str = "none",
                         waist_aux_w: float = 0.0, waist_attract_w: float = 0.0,
                         valid_centroid: "Tensor | None" = None,
                         centroid_momentum: float = 0.99):
    """Compile + return a TinyJit'd train step for the general factor forward.

    JIT cache key includes EVERY shape/runtime-determining constant AND the spec
    params (PORT #3). task + s_max + n_values + n_factor_types + has_factor_inlet
    are part of the key because DIFFERENT SPECS ARE DIFFERENT GRAPHS — a stale
    graph reading a wrong-shaped membership / value-domain / inlet buffer is silent
    corruption. The loss knobs (constraint_weight, calib_weight, ortho_lambda,
    label_smoothing, stoch_depth_p, grad_clip) change the traced graph body and so
    are also keyed.

    Inputs (stable shapes; passed as realized Tensors):
      input_cells (B,S) int, gold (B,S) int, cell_valid (B,S) f,
      value_domain_mask (B,S,N) f, membership (B,L,S) f, latent_type (B,L) int,
      factor_inlet (B,S,H) f  (zeros when has_inlet=False — always a JIT input so
                               the signature is stable; only READ when has_inlet),
      stoch_keep (K,) f.

    Returns realized scalars:
      total, healthy, cell_ce, energy, calib, cell_acc, puzzle_acc, ortho,
      *per_breath_ce[K], *per_breath_calib[K].
    """
    key = (id(model), id(opt), str(task),
           int(spec.s_max), int(spec.n_values), int(spec.n_factor_types),
           bool(spec.has_factor_inlet),
           # CONTINUOUS-INPUT (ECC / §8.1) FLIPS THE TRACED BODY (the engine takes the
           # LLR->H continuous embed vs the discrete one-hot path), so it MUST be keyed —
           # a stale discrete-input graph reading cont_input is silent corruption. True
           # only for the ECC spec; False (default) for kenken/coloring/circuit/multi ->
           # their key is unchanged (byte-identical, old graphs stay cache-compatible).
           bool(getattr(spec, "continuous_input", False)),
           # ECC FIXES (the two diagnosed one-shot-collapse fixes) FLIP THE TRACED BODY:
           # reinject_input re-adds the channel embed every breath (extra per-breath add),
           # lora_rank>0 inserts the K per-breath low-rank adapters. Both MUST be keyed —
           # a stale graph under a flipped toggle is silent corruption. Default off (False,
           # 0) -> the key is unchanged from current (byte-identical, old graphs stay
           # cache-compatible). INDEPENDENT toggles -> all 4 (off/off, on/off, off/on,
           # on/on) get distinct graphs.
           bool(getattr(spec, "reinject_input", False)),
           int(getattr(spec, "lora_rank", 0)),
           int(K), int(B),
           float(constraint_weight), float(calib_weight), float(ortho_lambda),
           float(grad_clip), float(label_smoothing), float(stoch_depth_p),
           bool(has_inlet),
           # Multi-task is a DISTINCT graph: FG_MULTITASK flips the spec (N_max=7
           # codebook, T=N_GLOBAL_TYPES, generic inlet) and the batch shapes (L padded
           # to L_max). FG_MIX hashes the domain set so different mixes never share a
           # JIT graph. When OFF (default) these are (False, "") — the single-domain
           # key is unchanged, so old single-task graphs stay cache-compatible.
           bool(multitask), str(mix_key),
           # WAIST + aux flip the traced graph body (the per-breath waist blend + the aux
           # loss terms), so they MUST be keyed — a stale graph under a flipped waist/aux is
           # silent corruption. The valid-centroid buffer is an in-place-assigned closure
           # object (its CONTENTS update per replay), keyed by presence (id), not value.
           bool(waist_on), str(waist_aux), float(waist_aux_w), float(waist_attract_w),
           float(centroid_momentum),
           id(valid_centroid) if valid_centroid is not None else 0,
           bool(FG_HYP_MASK), bool(FG_HYP_FREEZE),
           # Rung-2 relax knobs CHANGE THE TRACED GRAPH BODY (exp_0 vs raw coord,
           # euclid vs hyp distance, soft-block vs saturated alpha), so they MUST be
           # keyed — a stale graph under a flipped knob is silent corruption.
           bool(FG_HYP_RELAX), bool(FG_HYP_EUCLID),
           # KENKEN in-graph verification-inlet build FLIPS the traced body (the inlet
           # is built from kk_* cage features INSIDE the graph vs read as a constant), so
           # it MUST be keyed. True only for single-domain kenken (has_inlet & !multitask);
           # False for coloring/circuit/multi -> their key is unchanged (byte-identical).
           bool(has_inlet and not multitask),
           # PER-BREATH ENERGY WAVE knobs CHANGE THE TRACED GRAPH BODY (off = final-breath
           # only, no loop; monotonic/oscillating = a K-step weighted accumulation), so they
           # MUST be keyed. Default "off" -> the key is the literal "off" string, identical
           # across existing single/multi graphs (which never set FG_ENERGY_WAVE). The cycle
           # depth is keyed ONLY for the oscillating schedule (it is inert otherwise) to
           # avoid spurious cache misses. FG_ENERGY_CAGE is reflected in the
           # constraint_energy_fn IDENTITY (wired at task-build), and is keyed here as a
           # bool too so an A/B flip can never silently replay a stale graph.
           str(getenv("FG_ENERGY_WAVE", "off")).strip().lower(),
           (int(getenv("FG_ENERGY_CYCLE_DEPTH", "2"))
            if str(getenv("FG_ENERGY_WAVE", "off")).strip().lower() == "oscillating"
            else 0),
           bool(int(getenv("FG_ENERGY_CAGE", "0")) > 0))
    if key in _JIT_FG_CACHE:
        return _JIT_FG_CACHE[key]

    cw = float(constraint_weight)
    aw = float(calib_weight)
    olam = float(ortho_lambda)
    use_ortho = olam > 0.0
    gc_val = float(grad_clip)
    ls = float(label_smoothing)
    sd_p = float(stoch_depth_p)
    use_stoch = sd_p > 0.0
    use_energy = constraint_energy_fn is not None and cw > 0.0
    N = int(spec.n_values)
    S = int(spec.s_max)
    T = int(spec.n_factor_types)
    jit_params = opt.params

    # PER-BREATH ENERGY WAVE (compile-time constants -> baked into the traced graph
    # body). FG_ENERGY_WAVE in {off,monotonic,oscillating}; default "off" = the
    # current FINAL-breath-only energy (the per-breath loop is NEVER traced). The
    # weights are precomputed python floats so the loss graph contains only constant
    # multiplies + the existing softmax/energy ops. The mode + cycle depth are in the
    # JIT cache key (different schedules never share a graph).
    energy_wave_mode = str(getenv("FG_ENERGY_WAVE", "off")).strip().lower()
    energy_cycle_depth = int(getenv("FG_ENERGY_CYCLE_DEPTH", "2"))
    use_energy_wave = use_energy and energy_wave_mode != "off"
    if use_energy_wave:
        wave_w = _energy_wave_weights(K, energy_wave_mode, energy_cycle_depth)

    # KENKEN IN-GRAPH VERIFICATION INLET (the frozen-inlet fix). In SINGLE-DOMAIN kenken
    # (has_inlet=True, NOT multitask), build the verification inlet INSIDE this JIT step
    # from the RAW cage features (kk_cage_op/target/size + kk_cell_cage_id) so the inlet
    # params (op/target/size embeds + W + b) are in the differentiated loss graph and get
    # GRADIENT every step — mirroring the v98 oracle (kenken.py builds it in-graph; its
    # docstring: the inlet "must be LIVE at init so gradient visits the inlet"). When the
    # inlet was pre-built data-side (the regression) the params were a JIT CONSTANT and
    # never trained. This is a COMPILE-TIME constant (baked into the traced body, NOT a
    # runtime python branch): multitask=False + has_inlet=True is true ONLY for kenken.
    # coloring/circuit (has_inlet=False) + multitask (generic inlet branch) never enter it,
    # so those graphs are byte-identical.
    build_kk_inlet = has_inlet and not multitask
    if build_kk_inlet:
        from mycelium.kenken import build_verification_inlet

    # CONTINUOUS-INPUT (ECC / §8.1) — compile-time constant baked into the traced body.
    # True ONLY for the ECC spec; the engine then reads batch.cont_input (the per-cell
    # LLR) and takes the LLR->H continuous embed instead of the discrete one-hot path.
    # When False (default) the engine never touches cont_input -> the JIT input is an
    # unused placeholder (zeros) and the graph is byte-identical to current.
    continuous_input = bool(getattr(spec, "continuous_input", False))

    # WAIST aux gating (compile-time constants -> baked into the traced graph body).
    aux_mode = str(waist_aux) if waist_on else "none"
    use_classify = waist_on and aux_mode in ("classify", "both") and waist_aux_w > 0.0
    use_attract = waist_on and aux_mode in ("attract", "both") and waist_attract_w > 0.0
    use_waist_aux = use_classify or use_attract
    aux_w = float(waist_aux_w)
    attract_w = float(waist_attract_w)
    cmom = float(centroid_momentum)

    print(f"[JIT] compile fg step: task={task} S={S} N={N} "
          f"T={T} inlet={has_inlet} K={K} B={B} cw={cw} aw={aw} "
          f"ortho={olam} clip={gc_val} ls={ls} stoch_depth_p={sd_p} "
          f"waist={waist_on} aux={aux_mode} aux_w={aux_w} attract_w={attract_w}...",
          flush=True)

    @TinyJit
    def _step(input_cells: Tensor, gold: Tensor, cell_valid: Tensor,
              value_domain_mask: Tensor, membership: Tensor, latent_type: Tensor,
              factor_inlet: Tensor, stoch_keep: Tensor,
              inlet_op: Tensor, inlet_target: Tensor, inlet_size: Tensor,
              head_type_oh: Tensor, head_is_global: Tensor,
              kk_cage_op: Tensor, kk_cage_target: Tensor, kk_cage_size: Tensor,
              kk_cell_cage_id: Tensor, cont_input: Tensor):
        opt.zero_grad()

        # Lightweight batch shim so the engine reads per-instance tensors by attr.
        # All tensor attrs are JIT inputs (re-traced each replay).
        class _B:
            pass
        batch = _B()
        batch.input_cells = input_cells
        # CONTINUOUS INPUT (ECC): the per-cell LLR the engine reads when
        # spec.continuous_input is True. Always attached (zeros placeholder for non-ECC
        # so the JIT signature is stable); only the continuous-input engine branch READS
        # it (a compile-time branch on the spec flag, not a runtime domain branch).
        batch.cont_input = cont_input
        batch.gold = gold
        batch.cell_valid = cell_valid
        batch.value_domain_mask = value_domain_mask
        batch.membership = membership
        batch.latent_type = latent_type
        # KENKEN raw cage features (also consumed by the optional FG_ENERGY_CAGE
        # arithmetic energy). ALWAYS attached (placeholder zeros for non-kenken,
        # never read there); the cage energy is wired only for the single-domain
        # kenken task (build_kenken_task), so non-kenken graphs never trace it.
        batch.kenken_cage_op = kk_cage_op
        batch.kenken_cage_target = kk_cage_target
        batch.kenken_cage_size = kk_cage_size
        batch.kenken_cell_cage_id = kk_cell_cage_id

        # MULTI-TASK: build the GENERIC inlet IN-GRAPH from the raw per-latent semantic
        # ids (op/target/size), so the inlet params are in the loss graph and TRAIN every
        # step (the general-weights thesis). SINGLE-DOMAIN: the engine's pre-built
        # factor_inlet path is unchanged (the inlet tensor is passed through as before).
        # inlet_op/target/size are ALWAYS passed (zeros when single-domain) so the JIT
        # signature is stable; only the multitask branch reads them — a compile-time
        # constant, NOT a runtime python branch on domain-id.
        if multitask:
            batch.factor_inlet = build_generic_factor_inlet(
                model, membership, latent_type, cell_valid,
                op=inlet_op, target=inlet_target, size=inlet_size)
            # PER-BATCH head allocation (THE FIX): the multitask batch routes the engine
            # to build_factor_attn_bias_multitask via these tensors. Always set in the
            # multitask graph (a compile-time branch on `multitask`, NOT on domain-id);
            # their VALUES (which domain's allocation) vary per replay without recompile.
            batch.head_type_oh = head_type_oh
            batch.head_is_global = head_is_global
        elif build_kk_inlet:
            # SINGLE-DOMAIN KENKEN: build the verification inlet IN-GRAPH (the fix). The
            # raw cage features (kk_*) are fixed-shape JIT inputs; build_verification_inlet
            # is pure tensor ops (one_hot @ table, RMSNorm) -> the inlet params land in the
            # loss graph and get gradient. factor_inlet (the prebuilt tensor) is now an
            # unused zeros placeholder kept only for a stable JIT signature.
            inlet49 = build_verification_inlet(
                model, kk_cage_op, kk_cage_target, kk_cage_size, kk_cell_cage_id)  # (B,49,H)
            if S > 49:
                # DUAL-VIEW (s_max=98): the oracle verification inlet is hard-49 and applies
                # only to the primal cells; zero-pad the dual positions so factor_inlet is
                # (B, S, H). S is the compile-time spec constant -> this branch is baked
                # (the S=49 single-view path takes the else => byte-identical).
                pad = Tensor.zeros(inlet49.shape[0], S - 49, inlet49.shape[-1],
                                   dtype=inlet49.dtype)
                batch.factor_inlet = inlet49.cat(pad, dim=1)
            else:
                batch.factor_inlet = inlet49
            # Single-domain: do NOT set head_type_oh/head_is_global -> the engine takes
            # the original build_factor_attn_bias path (byte-identical).
        else:
            batch.factor_inlet = factor_inlet if has_inlet else None
            # Single-domain: do NOT set head_type_oh/head_is_global -> the engine takes
            # the original build_factor_attn_bias path (byte-identical).

        # Forward: K constant → loop unrolls → static graph topology. When the WAIST aux is
        # on, ALSO collect the per-breath waist d-rep history (return_waist=True) so the aux
        # objective can shape the d-rep in-graph. return_waist=False otherwise -> byte-
        # identical 2-tuple, no extra graph nodes.
        if use_waist_aux:
            logits_history, calib_history, waist_drep_history = factor_breathing_forward(
                model, batch, spec, K=K,
                stoch_keep=(stoch_keep if use_stoch else None), return_waist=True)
        else:
            logits_history, calib_history = factor_breathing_forward(
                model, batch, spec, K=K,
                stoch_keep=(stoch_keep if use_stoch else None))
            waist_drep_history = []

        # ---- Per-breath weighted-CE ladder (the training loss; inlined to expose
        # per-breath scalars in the JIT return). Supervise VALID & UNOBSERVED cells,
        # value-domain masked. Run-2 MASKED label smoothing: the logits carry a -1e4
        # bias on illegal values, so naive label_smoothing would spread mass onto the
        # -1e4 classes; we smooth ONLY over the legal value domain and mask the
        # per-class (target*logp) term by vdm. ls=0 reduces to the original NLL.
        observed = (input_cells > 0).cast(dtypes.float)                     # (B,S)
        supervise = cell_valid * (1.0 - observed)                          # (B,S)
        sup_sum = supervise.sum() + 1e-6

        cell_loss_sum = Tensor.zeros((), dtype=dtypes.float).contiguous()
        weight_sum = 0.0
        per_breath_ce_losses = []
        if ls > 0.0:
            gold_idx2d = (gold - 1).clip(0, N - 1)                          # (B,S)
            gold_oh = gold_idx2d.one_hot(N).cast(dtypes.float)              # (B,S,N)
            vdm = value_domain_mask.cast(dtypes.float)                      # (B,S,N)
            n_legal = vdm.sum(axis=-1, keepdim=True) + 1e-6                 # (B,S,1)
            smooth_target = gold_oh * (1.0 - ls) + (vdm / n_legal) * ls     # (B,S,N)
            for k, logits in enumerate(logits_history):
                weight_k = 1.0 + float(k) / float(K - 1) if K > 1 else 1.0
                logp = logits.cast(dtypes.float).log_softmax(axis=-1)       # (B,S,N)
                ce_per_cell = -(smooth_target * logp * vdm).sum(axis=-1)    # (B,S)
                ce_k = (ce_per_cell * supervise).sum() / sup_sum
                per_breath_ce_losses.append(ce_k)
                cell_loss_sum = cell_loss_sum + ce_k * weight_k
                weight_sum += weight_k
        else:
            gold_idx = (gold - 1).clip(0, N - 1).reshape(B * S)            # (B*S,)
            supervise_flat = supervise.reshape(B * S)
            for k, logits in enumerate(logits_history):
                weight_k = 1.0 + float(k) / float(K - 1) if K > 1 else 1.0
                ce_elems = logits.reshape(B * S, N).sparse_categorical_crossentropy(
                    gold_idx, reduction="none")                            # (B*S,)
                ce_k = (ce_elems * supervise_flat).sum() / sup_sum
                per_breath_ce_losses.append(ce_k)
                cell_loss_sum = cell_loss_sum + ce_k * weight_k
                weight_sum += weight_k
        cell_loss = cell_loss_sum / float(weight_sum)

        # Constraint energy (domain plug or zero). KenKen passes
        # kenken_constraint_energy (+ optional cage energy); coloring passes None.
        #
        # FG_ENERGY_WAVE gates the energy schedule (compile-time constant -> baked in):
        #   off (default): FINAL breath only — BYTE-IDENTICAL to the original code (no
        #                  loop, no extra graph nodes; the wave branch is not traced).
        #   monotonic/oscillating: PER-BREATH accumulation with normalized weights w_k
        #                  (sum w_k == 1), so a constant energy matches the off-control
        #                  magnitude (a fair A/B). The loop is unrolled at trace time (K
        #                  constant); each breath does its own softmax + constraint call.
        if use_energy:
            if not use_energy_wave:
                # CONTROL path (final-breath only) — identical to the prior code.
                final_probs = logits_history[-1].softmax(axis=-1)
                energy = constraint_energy_fn(final_probs, batch).mean()
            else:
                # WAVE path (per-breath, weighted). wave_w sums to 1.0.
                energy = Tensor.zeros((), dtype=dtypes.float).contiguous()
                for k, logits in enumerate(logits_history):
                    probs_k = logits.softmax(axis=-1)                       # (B,S,N)
                    energy_k = constraint_energy_fn(probs_k, batch).mean()  # scalar
                    energy = energy + float(wave_w[k]) * energy_k
        else:
            energy = Tensor.zeros((), dtype=dtypes.float).contiguous()

        # Calibration with detached argmax-correctness target (valid cells).
        final_argmax = (logits_history[-1].argmax(axis=-1) + 1).detach()    # (B,S)
        eq = (final_argmax == gold).cast(dtypes.float)                      # (B,S)
        eq_valid = eq * cell_valid + (1.0 - cell_valid)                     # pad = match
        correct = eq_valid.prod(axis=-1)                                    # (B,) 0/1
        calib_loss_sum = Tensor.zeros((), dtype=dtypes.float).contiguous()
        per_breath_calib_means = []
        for k, calib_k in enumerate(calib_history):
            progression = float(k) / float(K - 1) if K > 1 else 1.0
            target_k = 0.5 + (correct - 0.5) * progression
            calib_loss_sum = calib_loss_sum + ((calib_k - target_k) ** 2).mean()
            per_breath_calib_means.append(calib_k.mean())
        calib_loss = calib_loss_sum / float(K)

        # Train accuracy (detached) over VALID cells.
        eq_v = eq * cell_valid                                              # (B,S)
        n_valid = cell_valid.sum() + 1e-6
        train_cell_acc = (eq_v.sum() / n_valid).detach()
        train_puzzle_acc = eq_valid.prod(axis=-1).mean().detach()

        # ---- WAIST VALIDITY-SHAPING AUX ----------------------------------------------
        # Computed only when use_waist_aux (compile-time constant). Off -> waist_aux reports
        # 0.0 and `total` is the original CE+energy+calib (waist runs as a pure-CE bypass).
        waist_aux_loss = Tensor.zeros((), dtype=dtypes.float).contiguous()
        if use_waist_aux:
            # (1) FREE in-graph verifier on the ACTUAL forward's argmax -> per-instance
            # VALID label. A coloring/clique is proper iff zero all-different violations.
            pred_oh = final_argmax.one_hot(N + 1)[:, :, 1:].cast(dtypes.float)  # (B,S,N) drop the 0 row
            # zero out padding cells so pad never contributes a spurious clash.
            pred_oh = (pred_oh * cell_valid.reshape(B, S, 1).cast(dtypes.float)).detach()
            viol = _factor_alldiff_violation(pred_oh, membership, latent_type, T)  # (B,)
            is_valid_pred = (viol < 0.5).cast(dtypes.float).detach()            # (B,) 1=proper

            # ACTUAL forward's pooled waist d-rep (final breath) — the rep to shape.
            d_actual = pooled_waist_drep(waist_drep_history[-1], cell_valid)    # (B, d)

            # (2) GOLD teacher-forced forward (input_cells=gold -> output==gold -> ALWAYS
            # valid) — its waist d-rep is the guaranteed-valid silhouette. SAME shared
            # weights (iterative-prefill), SAME masks (same membership). Cheap extra pass.
            class _GB:
                pass
            gbatch = _GB()
            gbatch.input_cells = gold
            gbatch.gold = gold
            gbatch.cell_valid = cell_valid
            gbatch.value_domain_mask = value_domain_mask
            gbatch.membership = membership
            gbatch.latent_type = latent_type
            # cont_input only matters under continuous_input (ECC); the WAIST aux is
            # NOT used for ECC (FG_WAIST defaults off), so this branch is never traced
            # there. Attach the live cont_input so the shim is well-formed if ever on.
            gbatch.cont_input = cont_input
            if multitask:
                gbatch.factor_inlet = batch.factor_inlet
                gbatch.head_type_oh = head_type_oh
                gbatch.head_is_global = head_is_global
            elif build_kk_inlet:
                # Reuse the in-graph kenken inlet (same membership/features -> same inlet)
                # so the gold-TF pass also reads the trainable inlet.
                gbatch.factor_inlet = batch.factor_inlet
            else:
                gbatch.factor_inlet = factor_inlet if has_inlet else None
            _gl, _gc, gold_drep_history = factor_breathing_forward(
                model, gbatch, spec, K=K, stoch_keep=None, return_waist=True)
            d_gold = pooled_waist_drep(gold_drep_history[-1], cell_valid)       # (B, d)

            # (a) CLASSIFY: class-weighted BCE on the pooled d-rep. POSITIVES = gold-TF d-rep
            # (label 1) + actual d-rep where the verifier says proper (label is_valid_pred);
            # NEGATIVES = actual d-rep where improper. Pos is the minority -> upweight it.
            if use_classify:
                aw_w = model.fg_waist_aux_w.cast(dtypes.float)                  # (d,1)
                ab_w = model.fg_waist_aux_b.cast(dtypes.float)                  # (1,)
                # actual reps: label = verifier flag.
                logit_act = (d_actual.cast(dtypes.float) @ aw_w + ab_w).reshape(B)  # (B,)
                # gold reps: label = 1 (always valid).
                logit_gold = (d_gold.cast(dtypes.float) @ aw_w + ab_w).reshape(B)   # (B,)
                # class weights: positives (valid) upweighted by (1-p)/p style; use a fixed
                # POS_W so the minority valid class isn't drowned (env-tunable upstream).
                pos_w = 3.0
                def _bce(logit, label, w_pos):
                    # stable BCE-with-logits; weight positives by w_pos.
                    sp = logit.maximum(0.0)
                    bce = sp - logit * label + (1.0 + (-logit.abs()).exp()).log()
                    wt = label * w_pos + (1.0 - label)
                    return (bce * wt).sum() / (wt.sum() + 1e-6)
                bce_act = _bce(logit_act, is_valid_pred, pos_w)
                bce_gold = _bce(logit_gold, Tensor.ones((B,), dtype=dtypes.float), pos_w)
                classify_loss = 0.5 * (bce_act + bce_gold)
                waist_aux_loss = waist_aux_loss + aux_w * classify_loss

            # (b) ATTRACT (generative): pull the ACTUAL d-rep toward the running VALID
            # CENTROID, weighted by near-validity (1 - normalized violation), so the term
            # pulls HARDEST when the output already satisfies most factors -> biases the
            # deduction into the good common mode (energy shaping, not a classifier line).
            if use_attract and valid_centroid is not None:
                # near-valid weight in [0,1]: 1 when zero violations, decaying with viol.
                near = (1.0 / (1.0 + viol)).detach().reshape(B, 1)             # (B,1)
                centroid = valid_centroid.cast(dtypes.float).reshape(1, -1)    # (1,d) detached buffer
                diff = d_actual.cast(dtypes.float) - centroid                  # (B,d)
                attract_loss = ((diff * diff).sum(axis=-1) * near.reshape(B)).sum() / (
                    near.sum() + 1e-6)
                waist_aux_loss = waist_aux_loss + attract_w * attract_loss
                # EMA-update the centroid from the GOLD d-rep (guaranteed valid), DETACHED.
                # In-place assign on the closure buffer -> returned in the tuple so the JIT
                # does not drop the assign (closure-assign-must-return quirk).
                new_centroid = (cmom * valid_centroid.cast(dtypes.float)
                                + (1.0 - cmom) * d_gold.cast(dtypes.float).mean(axis=0)).detach()
                valid_centroid.assign(new_centroid)

        total = cell_loss + cw * energy + aw * calib_loss + waist_aux_loss

        # Codebook-orthogonality penalty (rotates collinear value rows apart). OFF
        # (olam==0) => no term, `ortho` reports 0.0, total byte-identical.
        if use_ortho:
            from mycelium.kenken import codebook_ortho_penalty
            ortho = codebook_ortho_penalty(model.fg_value_codebook)
            total = total + olam * ortho
        else:
            ortho = Tensor.zeros((), dtype=dtypes.float).contiguous()
        total.backward()

        # ---- NaN guard — where()-gated SELECT (PORT). multiply-gating passes NaN
        # through (NaN*0=NaN), poisoning Adam moments forever. where() SELECTS:
        # cond False → exact 0 regardless of NaN. Single-kernel isfinite() per
        # param, NO per-param isnan() loop (AM-driver safe).
        healthy_b = total.isfinite()
        healthy = healthy_b.cast(dtypes.float)
        _nograd_shapes = []
        for p in jit_params:
            if p.grad is None:
                # A param untouched by THIS step's graph (e.g. a verification-inlet or
                # other sub-table not indexed by this batch) gets no grad; AdamW.step()
                # requires a grad on every param, so fill zeros (a no-op update). This
                # happens for KenKen too — the assumption that single-domain touches ALL
                # its params is false (the verification inlet has unindexed sub-tables) —
                # so apply it for EVERY task, not just multitask.
                _nograd_shapes.append(tuple(int(s) for s in p.shape))
                p.grad = Tensor.zeros_like(p)
            else:
                p.grad = healthy_b.where(p.grad, Tensor.zeros_like(p.grad))
        if _nograd_shapes:
            print(f"  [no-grad params zero-filled this step: {_nograd_shapes}]", flush=True)

        # Optional global-norm gradient clipping (single sq_sum kernel; AM-safe).
        if gc_val > 0:
            sq_sum = Tensor.zeros((), dtype=dtypes.float).contiguous()
            for p in jit_params:
                if p.grad is not None:
                    sq_sum = sq_sum + p.grad.cast(dtypes.float).square().sum()
            grad_norm = (sq_sum + 1e-12).sqrt()
            clip_coef = (Tensor(gc_val, dtype=dtypes.float) / (grad_norm + 1e-6)).minimum(
                Tensor(1.0, dtype=dtypes.float))
            for p in jit_params:
                if p.grad is not None:
                    p.grad = p.grad * clip_coef.cast(p.grad.dtype)

        opt.step()

        # The valid-centroid buffer is in-place-assigned above (attract mode); include it in
        # the return so the JIT does NOT drop the assign (closure-assign-must-return quirk).
        # It is APPENDED last so the caller's outs[8:8+K] / outs[8+K:8+2*K] slicing is
        # unchanged; the caller reads waist_aux from outs[8+2*K] and ignores the centroid.
        centroid_ret = (valid_centroid.realize() if (use_attract and valid_centroid is not None)
                        else Tensor.zeros((), dtype=dtypes.float).contiguous().realize())
        return (
            total.realize(),
            healthy.realize(),
            cell_loss.realize(),
            energy.realize(),
            calib_loss.realize(),
            train_cell_acc.realize(),
            train_puzzle_acc.realize(),
            ortho.realize(),
            *(ce.realize() for ce in per_breath_ce_losses),
            *(c.realize() for c in per_breath_calib_means),
            waist_aux_loss.realize(),
            centroid_ret,
        )

    _JIT_FG_CACHE[key] = _step
    print(f"[JIT] fg step ready (cache size={len(_JIT_FG_CACHE)}); "
          f"first call compiles (~60-90s)…", flush=True)
    return _step


# ---------------------------------------------------------------------------
# PERMUTATION AUGMENTATION (FG_PERM_AUG) — data-side vertex RELABEL of each
# instance + its gold. Engine + oracle UNTOUCHED: the engine receives an already-
# relabeled FactorGraphBatch and builds the relabeled masks from the relabeled
# membership with NO change.
#
# THE INVARIANT (why this is a valid relabel, not a corrupted problem):
#   The engine's mask is A_t = m_t^T @ m_t over the membership CELL axis
#   (factor_masks.build_factor_attn_bias). The residual stream is indexed by the
#   SAME cell axis, and gold supervises that same axis. So if ONE permutation pi
#   of an instance's REAL cells is applied CONSISTENTLY to EVERY per-cell tensor —
#   input_cells, cell_valid, value_domain_mask, gold (all cell axis = last axis)
#   AND membership (cell axis = axis 2 of (B,L,s_max); the latent/factor axis L and
#   latent_type are UNCHANGED, a factor is the same factor, only its member CELL
#   indices relabel) — the result is the SAME graph under a vertex relabeling: the
#   permuted gold is a valid solution of the permuted graph (the verifier accepts
#   it) and the per-head masks the engine builds are the relabeled masks.
#
# DIRECTION CONVENTION (ONE, applied identically everywhere): GATHER along the cell
#   axis with `perm` — new[..., i] = old[..., perm[i]] for cell-axis tensors, and
#   new_membership[:, :, i] = old_membership[:, :, perm[i]]. `perm` is a permutation
#   of the REAL cell indices; pad cells (cell_valid == 0) map to themselves (identity
#   on the pad block), so padding stays pad. The inverse pi^{-1} maps a permuted-graph
#   prediction back to the original vertex order (round-trip == identity).
#
# STRENGTH KNOB (the TENSION guard): FG_PERM_AUG_FRAC in [0,1] = the fraction of
#   instances in a batch that get a FRESH random relabel (the rest stay identity).
#   FRAC=0 == OFF. Gentle augmentation keeps view-diversity (the load-bearing B
#   bucket) intact; full equivariance would collapse it — the harness's view-success-
#   count metric measures exactly this, pre vs post.
#
# ADDITIVITY: FG_PERM_AUG=0 (default) -> permute_factor_batch is NEVER called, the
#   batch is byte-identical, and NO augmentation RNG is drawn (the loader / stoch-
#   depth RNG streams are untouched) -> training is byte-identical to current.
# ---------------------------------------------------------------------------

FG_PERM_AUG = int(getenv("FG_PERM_AUG", "0")) > 0
FG_PERM_AUG_FRAC = float(getenv("FG_PERM_AUG_FRAC", "1.0"))
FG_PERM_AUG_SEED = int(getenv("FG_PERM_AUG_SEED", "1234"))


def _valid_cell_indices(cell_valid_row: np.ndarray) -> np.ndarray:
    """The REAL (non-pad) cell indices of one instance, derived from cell_valid.

    The general source of truth (works for coloring's contiguous valid prefix AND
    any domain whose valid cells are scattered): a permutation relabels ONLY the
    valid cells among themselves; pad cells (cell_valid == 0) stay fixed."""
    return np.nonzero(cell_valid_row > 0.5)[0]


def permute_factor_batch(fb, rng: "np.random.RandomState", frac: float):
    """Return a NEW batch-like object with FG_PERM_AUG applied (engine-ready).

    For each instance independently, with probability `frac`, draw a random
    permutation pi of its REAL cell indices (from cell_valid) and GATHER every
    per-cell tensor along the cell axis by pi (new[...,i] = old[...,perm[i]]);
    membership is gathered on its CELL axis (axis 2), with the factor axis (L) and
    latent_type left UNCHANGED. Instances NOT selected (prob 1-frac) are copied
    identity. Pad cells map to themselves (identity on the pad block).

    Reads the tensors via .numpy() and re-wraps fresh Tensors (same shapes/dtypes
    as the source batch), so the returned object satisfies the FactorGraphBatch
    contract and the engine needs no change. Python-side metadata (n / n_edges /
    band / deduction_depth) is carried through UNCHANGED (a relabel does not change
    n, edge count, band, or DSATUR depth).

    Returns a lightweight shim object exposing the same tensor attrs the engine
    reads (input_cells, cell_valid, value_domain_mask, gold, membership,
    latent_type, factor_inlet) plus whatever metadata attrs were present."""
    ic = fb.input_cells.realize().numpy()                       # (B, S) int
    cv = fb.cell_valid.realize().numpy()                        # (B, S) f
    vdm = fb.value_domain_mask.realize().numpy()               # (B, S, N) f
    gold = fb.gold.realize().numpy()                            # (B, S) int
    mem = fb.membership.realize().numpy()                       # (B, L, S) f
    Bn, S = ic.shape

    # per-instance index map P (B, S): P[b] is the GATHER index along the cell axis.
    P = np.tile(np.arange(S, dtype=np.int64), (Bn, 1))          # identity default
    for b in range(Bn):
        if rng.random_sample() >= frac:
            continue                                            # identity (not augmented)
        idx = _valid_cell_indices(cv[b])                        # real cell indices
        if idx.size <= 1:
            continue                                            # nothing to relabel
        pi = rng.permutation(idx.size)                          # permute the real cells
        # GATHER convention: new position idx[j] takes old position idx[pi[j]].
        P[b, idx] = idx[pi]

    # apply the SAME P[b] to every per-cell tensor (cell axis = last for ic/cv/gold,
    # axis -2 for vdm (B,S,N), axis 2 for membership (B,L,S)).
    new_ic = np.take_along_axis(ic, P, axis=1)                  # (B, S)
    new_cv = np.take_along_axis(cv, P, axis=1)                  # (B, S) — pads unchanged
    new_gold = np.take_along_axis(gold, P, axis=1)             # (B, S)
    new_vdm = np.take_along_axis(vdm, P[:, :, None], axis=1)   # (B, S, N)
    # membership cell axis is axis 2; broadcast P over the L (factor) axis.
    new_mem = np.take_along_axis(mem, P[:, None, :], axis=2)   # (B, L, S)

    class _PermBatch:
        pass
    out = _PermBatch()
    out.input_cells = Tensor(new_ic.astype(np.int32),
                             dtype=dtypes.int).contiguous().realize()
    out.cell_valid = Tensor(new_cv.astype(np.float32),
                            dtype=dtypes.float).contiguous().realize()
    out.value_domain_mask = Tensor(new_vdm.astype(np.float32),
                                   dtype=dtypes.float).contiguous().realize()
    out.gold = Tensor(new_gold.astype(np.int32),
                      dtype=dtypes.int).contiguous().realize()
    out.membership = Tensor(new_mem.astype(np.float32),
                            dtype=dtypes.float).contiguous().realize()
    # latent_type + factor_inlet + the raw semantic ids are NOT on the cell axis —
    # a factor is the SAME factor (its type/op/target/size unchanged), only its
    # member CELL indices relabel (already done via membership). Pass them through.
    out.latent_type = fb.latent_type
    # factor_inlet (B, S, H) IS on the cell axis (KenKen's per-cell verification inlet,
    # keyed by cell_cage_id) -> it MUST be gathered by the SAME P, else after a relabel
    # it points at the wrong cells. coloring/circuit: factor_inlet is None -> pass through.
    _finlet = getattr(fb, "factor_inlet", None)
    if _finlet is not None:
        _finlet_np = _finlet.realize().numpy()                      # (B, S, H)
        _new_finlet = np.take_along_axis(_finlet_np, P[:, :, None], axis=1)
        out.factor_inlet = Tensor(_new_finlet.astype(np.float32),
                                  dtype=dtypes.float).contiguous().realize()
    else:
        out.factor_inlet = None
    for attr in ("inlet_op", "inlet_target", "inlet_size",
                 "head_type_oh", "head_is_global",
                 "deduction_depth", "n", "n_edges", "band", "domain"):
        if hasattr(fb, attr):
            setattr(out, attr, getattr(fb, attr))
    # KENKEN raw cage features (in-graph inlet build). cage_op/target/size are PER-CAGE
    # (not on the cell axis) -> a cell relabel does not change a cage's op/target/size, so
    # pass them through UNCHANGED. cell_cage_id IS on the cell axis (B,49) -> it MUST be
    # gathered by the SAME P, else after a relabel the inlet scatters to the wrong cells.
    for attr in ("kenken_cage_op", "kenken_cage_target", "kenken_cage_size"):
        if hasattr(fb, attr):
            setattr(out, attr, getattr(fb, attr))
    _kcid = getattr(fb, "kenken_cell_cage_id", None)
    if _kcid is not None:
        _kcid_np = _kcid.realize().numpy()                          # (B, 49) int
        _new_kcid = np.take_along_axis(_kcid_np, P, axis=1)         # (B, 49)
        out.kenken_cell_cage_id = Tensor(_new_kcid.astype(np.int32),
                                         dtype=dtypes.int).contiguous().realize()
    return out


# ---------------------------------------------------------------------------
# Task adapters — turn a domain loader's native batch into a FactorGraphBatch
# and pull the JIT-input tuple out of it. Each task is ONE small function so the
# train/eval loops are domain-agnostic.
# ---------------------------------------------------------------------------

class _Task:
    """Holds the per-domain glue: spec, loaders, to_factor_batch, jit_inputs,
    constraint_energy_fn, has_inlet."""
    def __init__(self, spec, train_loader, test_loader, to_factor_batch,
                 constraint_energy_fn, has_inlet, eval_iter):
        self.spec = spec
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.to_factor_batch = to_factor_batch       # native_batch -> FactorGraphBatch
        self.constraint_energy_fn = constraint_energy_fn
        self.has_inlet = has_inlet
        self.eval_iter = eval_iter                   # () -> iterator of native batches
        # Multi-task fields (default OFF -> single-domain path is byte-identical).
        self.is_multitask = False
        self.mix = None
        self.adapters = None
        self.eval_loaders = None
        self.L_max = None


def _zeros_inlet(B, S, H):
    return Tensor(np.zeros((B, S, H), dtype=np.float32),
                  dtype=dtypes.float).contiguous().realize()


def _build_kenken_task(K, BATCH, EVAL_BATCH, SEED, hidden, n_heads, model,
                       train_path, test_path):
    """KenKen task: KenKenLoader + make_kenken_factor_batch + verification inlet +
    kenken_constraint_energy. Also the train-parity check vs v98."""
    from mycelium.kenken import (
        attach_kenken_params, build_verification_inlet, kenken_constraint_energy,
    )
    from mycelium.kenken_data import KenKenLoader, load_jsonl

    spec = FactorGraphSpec(s_max=49, n_values=7, n_factor_types=3,
                           n_heads=n_heads, k_max=K, has_factor_inlet=True)

    # The verification inlet uses kenken-specific embedding tables (op/target/size +
    # projection). attach_kenken_params allocates them (alongside its own heads,
    # which the general forward ignores — the general forward reads ONLY fg_*). We
    # attach ONLY for the inlet tables.
    attach_kenken_params(model, hidden=hidden, n_heads=n_heads, k_max=K)

    # n_cages_max pinned to the max over BOTH corpora so train+eval JIT topology
    # is identical (PORT #4).
    train_recs = load_jsonl(train_path)
    test_recs = load_jsonl(test_path)
    corpus_n_cages_max = max(
        max(len(r["cages"]) for r in train_recs),
        max(len(r["cages"]) for r in test_recs),
    )
    n_cages_max = int(getenv("KENKEN_N_CAGES_MAX", str(corpus_n_cages_max)))
    assert n_cages_max >= corpus_n_cages_max, (
        f"KENKEN_N_CAGES_MAX={n_cages_max} < corpus max {corpus_n_cages_max}")
    print(f"  kenken n_cages_max (train+test) = {n_cages_max}")

    train_loader = KenKenLoader(train_path, batch_size=BATCH, seed=SEED,
                                n_cages_max=n_cages_max)
    test_loader = KenKenLoader(test_path, batch_size=EVAL_BATCH, seed=SEED + 1,
                               n_cages_max=n_cages_max)

    def to_factor_batch(kb):
        # DO NOT pre-build the inlet (the frozen-inlet regression): if the inlet is built
        # here (data-side, eager) it enters the JIT as a CONSTANT and its params get NO
        # gradient. Instead leave factor_inlet=None and carry the RAW cage features so the
        # JIT step (_step, build_kk_inlet branch) builds the inlet IN-GRAPH -> the inlet
        # params (op/target/size embeds + W + b) are in the loss graph and TRAIN. Mirrors
        # the v98 oracle (kenken.py builds build_verification_inlet inside its JIT).
        fb = make_kenken_factor_batch(kb, spec)   # factor_inlet stays None
        fb.kenken_cage_op = kb.cage_op
        fb.kenken_cage_target = kb.cage_target
        fb.kenken_cage_size = kb.cage_size
        fb.kenken_cell_cage_id = kb.cell_cage_id
        return fb

    # CAGE-ARITHMETIC SOFT ENERGY (FG_ENERGY_CAGE, default 0 = OFF). When OFF the plug
    # is the v98-compatible row/col-only kenken_constraint_energy (oracle, UNTOUCHED) ->
    # the loss graph is BYTE-IDENTICAL to current. When ON, the plug SUMS the row/col
    # oracle energy with the trainer-local differentiable cage-arithmetic surrogate
    # (both receive the SAME probs). Pure trainer-local wiring — the oracle is never
    # modified and the cage fn lives in this file.
    use_cage = int(getenv("FG_ENERGY_CAGE", "0")) > 0
    if use_cage:
        cage_energy_fn = make_kenken_cage_constraint_energy(
            n_cages_max, n_values=spec.n_values)
        print(f"  FG_ENERGY_CAGE=1 -> cage-arithmetic energy ON "
              f"(row/col + cage, n_cages_max={n_cages_max})")

        def kenken_energy(probs, batch):
            return kenken_constraint_energy(probs, batch) + cage_energy_fn(probs, batch)
    else:
        # OFF (default): row/col only -> byte-identical to the current behavior.
        kenken_energy = kenken_constraint_energy

    return _Task(
        spec=spec, train_loader=train_loader, test_loader=test_loader,
        to_factor_batch=to_factor_batch,
        constraint_energy_fn=kenken_energy,
        has_inlet=True,
        eval_iter=lambda: test_loader.iter_eval(batch_size=EVAL_BATCH),
    )


def _build_dual_kenken_task(K, BATCH, EVAL_BATCH, SEED, hidden, n_heads, model,
                            train_path, test_path):
    """DUAL-VIEW KenKen task (Bryce's multi-view/channeling generality probe).

    s_max=98 (49 primal cells + 49 dual variables), n_factor_types=6 (primal
    row/col/cage + dual value-alldiff + dual row-alldiff + channeling). The PRIMAL
    half reproduces _build_kenken_task's encoding exactly, so primal-cell solve is
    directly comparable to the 0.796 single-view baseline (primal_s_max=49 makes the
    eval report primal-only as the headline). The verification inlet is the SAME
    oracle build (at 49), zero-padded to 98 in the JIT step. No constraint energy
    (cw default 0). Lives entirely in the general path; oracle kenken.py untouched.
    """
    from mycelium.kenken import attach_kenken_params
    from mycelium.kenken_data import load_jsonl
    from mycelium.kenken_dual_data import KenKenDualLoader, S_DUAL

    channel_msg = bool(int(getenv("FG_CHANNEL_MSG", 0)))
    spec = FactorGraphSpec(s_max=S_DUAL, n_values=7, n_factor_types=6,
                           n_heads=n_heads, k_max=K, has_factor_inlet=True,
                           primal_s_max=49, channel_messages=channel_msg)
    print(f"  dual-kenken channel_messages={channel_msg} "
          f"(FG_CHANNEL_MSG; explicit bidirectional BP messages)")

    # inlet embed tables (op/target/size + projection) — same as single-view kenken.
    attach_kenken_params(model, hidden=hidden, n_heads=n_heads, k_max=K)

    # n_cages_max pinned across BOTH corpora so the membership L (=35+n_cages_max) and
    # the per-cage tensors have a static JIT topology.
    train_recs = load_jsonl(train_path)
    test_recs = load_jsonl(test_path)
    corpus_n_cages_max = max(
        max(len(r["cages"]) for r in train_recs),
        max(len(r["cages"]) for r in test_recs),
    )
    n_cages_max = int(getenv("KENKEN_N_CAGES_MAX", str(corpus_n_cages_max)))
    assert n_cages_max >= corpus_n_cages_max, (
        f"KENKEN_N_CAGES_MAX={n_cages_max} < corpus max {corpus_n_cages_max}")
    print(f"  dual-kenken n_cages_max={n_cages_max} -> L={35 + n_cages_max} factors, "
          f"s_max={S_DUAL}")

    train_loader = KenKenDualLoader(train_path, batch_size=BATCH, seed=SEED,
                                    n_cages_max=n_cages_max)
    test_loader = KenKenDualLoader(test_path, batch_size=EVAL_BATCH, seed=SEED + 1,
                                   n_cages_max=n_cages_max)

    def to_factor_batch(db):
        # KenKenDualBatch already satisfies the FactorGraphBatch contract (input_cells/
        # cell_valid/value_domain_mask/gold/membership/latent_type). Add the inlet alias
        # attrs (read by _jit_inputs + the eager eval inlet build) + a None factor_inlet
        # (built in-graph / eagerly in eval). cell_cage_id stays (B,49) for the oracle inlet.
        db.factor_inlet = None
        db.kenken_cage_op = db.cage_op
        db.kenken_cage_target = db.cage_target
        db.kenken_cage_size = db.cage_size
        db.kenken_cell_cage_id = db.cell_cage_id           # (B,49) primal, for the inlet
        return db

    return _Task(
        spec=spec, train_loader=train_loader, test_loader=test_loader,
        to_factor_batch=to_factor_batch,
        constraint_energy_fn=None,                          # no soft energy (cw default 0)
        has_inlet=True,
        eval_iter=lambda: test_loader.iter_eval(batch_size=EVAL_BATCH),
    )


def _build_coloring_task(K, BATCH, EVAL_BATCH, SEED, n_heads):
    """Graph-coloring task: GraphColoringLoader (IN-MEMORY, NOT path-based) produces
    FactorGraphBatch-compatible batches directly (input_cells, membership (B,L,s_max),
    latent_type, cell_valid, value_domain_mask (B,s_max,k), gold (B,s_max)). No
    verification inlet, no domain constraint energy. The import is LAZY so a missing
    data module never breaks the kenken path or the trainer's CPU import.

    ONE loader owns the internal train/test split (mirrors KenKenLoader). sample_batch()
    draws from train_records; iter_eval() iterates test_records. The JIT topology width
    (n_edges_max) is fixed to loader.n_edges_max after corpus generation."""
    from mycelium.graph_coloring_data import GraphColoringLoader

    s_max = int(getenv("FG_S_MAX", "49"))
    n_values = int(getenv("FG_N_VALUES", "0"))
    assert n_values > 0, "coloring task requires FG_N_VALUES (k colors, e.g. FG_N_VALUES=3)"
    n_factor_types = int(getenv("FG_N_FACTOR_TYPES", "1"))  # coloring: 1 edge relation
    n_instances = int(getenv("FG_N_INSTANCES", "8000"))

    spec = FactorGraphSpec(s_max=s_max, n_values=n_values,
                           n_factor_types=n_factor_types, n_heads=n_heads,
                           k_max=K, has_factor_inlet=False)

    # ONE in-memory loader: generates the corpus, splits train/test internally.
    # train_loader.sample_batch() -> GraphColoringBatch (train).
    # train_loader.iter_eval()    -> iterator of GraphColoringBatch (test).
    # n_edges_max is fixed to the corpus max — the static JIT topology width.
    loader = GraphColoringLoader(
        n_instances=n_instances,
        s_max=s_max,
        k_colors=n_values,
        batch_size=BATCH,
        seed=SEED,
    )

    def to_factor_batch(cb):
        # GraphColoringBatch already satisfies the FactorGraphBatch contract (same
        # tensor attrs); pass through directly.
        return cb

    # OUTPUT-SPACE SOFT-VIOLATION ENERGY (the differentiable verifier plug). Wired so a
    # FG_CONSTRAINT_WEIGHT>0 fine-tune steers the output toward valid colorings; at the
    # default FG_CONSTRAINT_WEIGHT=0 the loss graph's `use_energy` gate (cw>0.0) leaves it
    # INERT (never traced) -> byte-identical to the previous constraint_energy_fn=None.
    coloring_energy = make_coloring_constraint_energy(spec.n_factor_types)

    return _Task(
        spec=spec,
        train_loader=loader,
        test_loader=loader,   # same object; eval_iter uses loader.iter_eval()
        to_factor_batch=to_factor_batch,
        constraint_energy_fn=coloring_energy,
        has_inlet=False,
        eval_iter=lambda: loader.iter_eval(batch_size=EVAL_BATCH),
    )


def _build_circuit_task(K, BATCH, EVAL_BATCH, SEED, n_heads):
    """Boolean-circuit task: CircuitLoader (IN-MEMORY, NOT path-based) produces
    FactorGraphBatch-compatible batches directly (input_cells, membership
    (B,L,49) — one factor per GATE, latent_type {0..T-1 gate types / T=global
    sentinel for padding}, cell_valid, value_domain_mask (B,49,2), gold (B,49)).
    No verification inlet, no domain constraint energy (CE + calib only). The
    import is LAZY so a missing data module never breaks the kenken/coloring paths
    or the trainer's CPU import.

    THE RUNG-1 TESTBED. Unlike KenKen (flat lateral cliques) and coloring (binary
    not-equal edges), a Boolean circuit is a DAG: a gate's output is a deduction
    over its inputs, and a deep circuit chains many such deductions. The per-NODE
    `lvl` (topological depth) is therefore the deduction-depth axis the eval reads.

    ONE loader owns the internal train/test split (mirrors GraphColoringLoader /
    KenKenLoader). sample_batch() draws from train; iter_eval() iterates test. The
    JIT topology width (n_gates_max) is fixed to loader.n_gates_max after corpus
    generation. The number of gate (factor) TYPES T is owned by the loader — the
    spec follows the loader so the membership/latent_type the engine sees always
    matches spec.n_factor_types (a mismatch is silent corruption, PORT #3)."""
    from mycelium.circuit_data import CircuitLoader

    s_max = int(getenv("FG_S_MAX", "49"))
    n_values = 2                                    # Boolean: {0, 1}
    n_instances = int(getenv("FG_N_INSTANCES", "8000"))

    # Gate-type set: explicit list overrides the XOR toggle. The loader owns T;
    # spec.n_factor_types is read from loader AFTER construction (PORT #3 guard).
    # Keys MUST be UPPERCASE — CircuitLoader keys on ('AND','OR','NOT','XOR').
    gate_types_env = getenv("FG_CIRCUIT_GATE_TYPES", "").strip()
    use_xor = int(getenv("FG_CIRCUIT_XOR", "0")) > 0
    if gate_types_env:
        gate_types: tuple[str, ...] = tuple(g.strip().upper() for g in gate_types_env.split(",") if g.strip())
    elif use_xor:
        gate_types = ("AND", "OR", "NOT", "XOR")
    else:
        gate_types = ("AND", "OR", "NOT")          # loader default

    # Band selection: FG_CIRCUIT_BANDS overrides the default D2..D5 set.
    # Deep bands (D6..D16) are generated via generate_skinny_instance() inside
    # circuit_data.generate_corpus() when any band is in DEEP_BANDS.
    # Example: FG_CIRCUIT_BANDS=D4,D5,D6,D8,D10,D12,D14,D16 for deep-mix training.
    circuit_bands_env = getenv("FG_CIRCUIT_BANDS", "").strip()
    if circuit_bands_env:
        from mycelium.circuit_data import _ALL_BAND_TARGET_D
        circuit_bands: list[str] | None = [
            b.strip().upper() for b in circuit_bands_env.split(",") if b.strip()
        ]
        # Validate band names.
        for cb in circuit_bands:
            if cb not in _ALL_BAND_TARGET_D:
                raise ValueError(
                    f"FG_CIRCUIT_BANDS: unknown band {cb!r}; "
                    f"valid bands: {sorted(_ALL_BAND_TARGET_D.keys())}")
    else:
        circuit_bands = None   # defaults to BANDS (D2..D5) inside CircuitLoader

    # ONE in-memory loader: generates the corpus, splits train/test internally,
    # fixes n_gates_max (the static JIT topology width) and owns the gate-type set.
    loader = CircuitLoader(
        n_instances=n_instances,
        s_max=s_max,
        n_values=n_values,
        batch_size=BATCH,
        seed=SEED,
        gate_types=gate_types,
        bands=circuit_bands,
    )

    # T = number of non-global gate types — read from the loader (authoritative).
    n_factor_types = int(loader.n_factor_types)

    spec = FactorGraphSpec(s_max=s_max, n_values=n_values,
                           n_factor_types=n_factor_types, n_heads=n_heads,
                           k_max=K, has_factor_inlet=False)

    print(f"  circuit: n_gates_max={loader.n_gates_max} "
          f"n_factor_types(T)={n_factor_types} "
          f"gate_types={getattr(loader, 'gate_types', gate_types)} "
          f"bands={circuit_bands or 'default(D2..D5)'}")

    def to_factor_batch(cb):
        # CircuitBatch already satisfies the FactorGraphBatch contract (same
        # tensor attrs); pass through directly.
        return cb

    return _Task(
        spec=spec,
        train_loader=loader,
        test_loader=loader,   # same object; eval_iter uses loader.iter_eval()
        to_factor_batch=to_factor_batch,
        constraint_energy_fn=None,
        has_inlet=False,
        eval_iter=lambda: loader.iter_eval(batch_size=EVAL_BATCH),
    )


def _build_ecc_task(K, BATCH, EVAL_BATCH, SEED, n_heads):
    """ECC / neural-BP task (BCH(31,16); the §8.1 soft-constraint frontier).

    The deducer AS A LEARNED BP DECODER: K breaths == K message-passing rounds, the
    per-head attention masks (from membership) == the parity-check H topology, the
    per-cell CONTINUOUS input == the channel LLR.  ECCLoader (mycelium.ecc_data —
    IN-MEMORY generator, NOT path-based) produces FactorGraphBatch-compatible batches
    directly (PLUS the new cont_input attr the continuous-input engine path reads).
    The import is LAZY so a missing data module never breaks the other tasks or the
    trainer's CPU import.

    spec: s_max=49, n_values=2 (the bit codebook), n_factor_types=1 (one parity
    relation), n_heads=16, has_factor_inlet=False (no arithmetic to verify),
    continuous_input=True (THE flag that routes the engine to the LLR->H embed).

    NO new readout / loss / inlet: the N=2 value-codebook readout (BER = argmax+1 ==
    gold on real bit-cells), the per-breath weighted-CE ladder (== BCE on the 2-way),
    the calibration head, and the v45 reg stack are ALL reused unchanged.  The ONE
    new tensor in the loss graph is the continuous embed (handled in the engine).

    ONE loader owns train (fresh random codewords across SNR) + a FIXED, SNR-
    stratified held-out eval set.  The JIT topology width (n_checks_max == H rows)
    is fixed by the parity-check shape -> static.
    """
    from mycelium.ecc_data import ECCLoader

    H_kind = getenv("ECC_H_KIND", "min").strip().lower()
    snr_lo = float(getenv("ECC_SNR_LO", "3.0"))
    snr_hi = float(getenv("ECC_SNR_HI", "7.0"))
    eval_snrs = tuple(float(s) for s in
                      getenv("ECC_EVAL_SNRS", "3,4,5,6,7").split(",") if s.strip())
    n_eval_per_snr = int(getenv("ECC_EVAL_PER_SNR", "200"))

    spec = FactorGraphSpec(s_max=49, n_values=2, n_factor_types=1,
                           n_heads=n_heads, k_max=K, has_factor_inlet=False,
                           continuous_input=True)

    loader = ECCLoader(H_kind=H_kind, batch_size=BATCH, seed=SEED,
                       snr_lo=snr_lo, snr_hi=snr_hi, eval_snrs=eval_snrs,
                       n_eval_per_snr=n_eval_per_snr)

    print(f"  ecc: n_checks_max={loader.n_checks_max} n_values=2 "
          f"continuous_input=True H_kind={H_kind} "
          f"train_SNR=[{snr_lo},{snr_hi}]dB eval_SNRs={list(eval_snrs)}")

    def to_factor_batch(eb):
        # ECCBatch already satisfies the FactorGraphBatch contract (same tensor
        # attrs) PLUS the cont_input attr; pass through directly.
        return eb

    return _Task(
        spec=spec,
        train_loader=loader,
        test_loader=loader,   # same object; eval_iter uses loader.iter_eval()
        to_factor_batch=to_factor_batch,
        constraint_energy_fn=None,   # no domain energy (CE + calib only, like circuit)
        has_inlet=False,
        eval_iter=lambda: loader.iter_eval(batch_size=EVAL_BATCH),
    )


# ---------------------------------------------------------------------------
# MULTI-TASK (GENERAL-WEIGHTS) HARNESS  — FG_MULTITASK=1 / FG_TASK=multi.
#
# ONE dense shared Pythia-410M backbone co-trained on {coloring, circuit, kenken}
# under ONE unified spec. The model distinguishes domains/factor-types ONLY from the
# INPUT: the membership topology + the generic semantics inlet (per-factor GLOBAL
# type-id, + KenKen op/target/size) + the universal masked value-codebook. NO MoE,
# NO LoRA, NO routing, NO per-domain weight/branch in the backbone.
#
# ADDITIVE: default OFF -> the single-domain FG_TASK={coloring,circuit,kenken} path is
# byte-identical (this whole section is dead code unless FG_MULTITASK=1).
# ---------------------------------------------------------------------------

# Unified spec constants (the general-weights contract).
MT_S_MAX = 49                  # all domains lay on the 7x7 = 49-cell grid.
MT_N_MAX = 7                   # universal codebook size (KenKen size; the max).
MT_T = N_GLOBAL_TYPES          # global factor-type count (mask-builder T).
MT_SENTINEL = N_GLOBAL_TYPES   # global padding/sentinel latent_type id.


def _circuit_gate_global_id(gate_name: str) -> int:
    """Map a circuit gate-type name (AND/OR/NOT/XOR) to its GLOBAL type id."""
    return GLOBAL_TYPE_IDS[f"circuit_{gate_name.lower()}"]


def _np_int(t: Tensor) -> np.ndarray:
    return t.realize().numpy().astype(np.int32)


def _np_f(t: Tensor) -> np.ndarray:
    return t.realize().numpy().astype(np.float32)


def _pad_membership_np(mem: np.ndarray, L_max: int) -> np.ndarray:
    """Pad (B, L, S) membership rows to (B, L_max, S) with all-zero pad rows."""
    B, L, S = mem.shape
    if L == L_max:
        return mem
    assert L <= L_max, f"membership L={L} exceeds L_max={L_max}"
    out = np.zeros((B, L_max, S), dtype=np.float32)
    out[:, :L, :] = mem
    return out


def _pad_latent_type_np(lt: np.ndarray, L_max: int) -> np.ndarray:
    """Pad (B, L) latent_type to (B, L_max) with the global sentinel id."""
    B, L = lt.shape
    if L == L_max:
        return lt
    out = np.full((B, L_max), MT_SENTINEL, dtype=np.int32)
    out[:, :L] = lt
    return out


def _pad_vdm_np(vdm: np.ndarray) -> np.ndarray:
    """Pad (B, S, n) value_domain_mask to (B, S, MT_N_MAX) with 0 on unused slots."""
    B, S, n = vdm.shape
    if n == MT_N_MAX:
        return vdm
    assert n <= MT_N_MAX, f"value_domain_mask n={n} exceeds N_max={MT_N_MAX}"
    out = np.zeros((B, S, MT_N_MAX), dtype=np.float32)
    out[:, :, :n] = vdm
    return out


class _MultiTaskBatch:
    """A unified, single-domain batch padded to the multi-task JIT topology.

    Satisfies the FactorGraphBatch contract (same tensor attrs). Built by a per-domain
    adapter: latent_type remapped to GLOBAL ids, membership padded to L_max, vdm padded
    to N_max=7. Carries `domain` for per-domain logging.

    THE INLET IS TRAINABLE: instead of a pre-built (detached) factor_inlet tensor, the
    multi-task batch carries the RAW per-latent semantic id tensors (op/target/size,
    B x L_max int). The JIT step builds the generic inlet IN-GRAPH from these via
    build_generic_factor_inlet, so the inlet params are in the loss graph and get
    gradient every step (the general-weights thesis: the shared backbone LEARNS the
    predicate registry). This mirrors kenken_train.py, which builds its verification
    inlet inside the JIT — NOT the general engine's pre-built (frozen) inlet path.

    factor_inlet stays None on a _MultiTaskBatch (the in-graph build supersedes it);
    eval (eager, not jitted) builds the inlet eagerly via build_generic_factor_inlet
    so the trained inlet params are read at eval time too.
    """
    def __init__(self, input_cells, cell_valid, value_domain_mask, gold,
                 membership, latent_type, domain, deduction_depth,
                 inlet_op, inlet_target, inlet_size,
                 head_type_oh=None, head_is_global=None):
        self.input_cells = input_cells
        self.cell_valid = cell_valid
        self.value_domain_mask = value_domain_mask
        self.gold = gold
        self.membership = membership
        self.latent_type = latent_type
        self.factor_inlet = None             # built in-graph (JIT) / eager (eval).
        self.domain = domain
        self.deduction_depth = deduction_depth
        # Raw per-latent semantic ids (B, L_max) int — the inlet INPUTS.
        self.inlet_op = inlet_op
        self.inlet_target = inlet_target
        self.inlet_size = inlet_size
        # PER-BATCH head->GLOBAL-type allocation tensors (THE HEAD-ALLOC FIX).
        # head_type_oh: (H, T) one-hot over global types; head_is_global: (H,1,1).
        # Computed host-side per domain (the domain's NATIVE allocation mapped to its
        # PRESENT global types); the engine routes to build_factor_attn_bias_multitask
        # when these are present. Single-domain batches never set them.
        self.head_type_oh = head_type_oh
        self.head_is_global = head_is_global


def _present_global_types(domain: str,
                          gate_types: "tuple[str, ...] | None") -> "list[int]":
    """The GLOBAL factor-type ids PRESENT in a pure single-domain batch of `domain`.

    Sorted so the native head allocation lays contiguous per-type head blocks in a
    stable order. coloring -> [edge]; circuit -> the gate-type ids in gate_types
    order; kenken -> [row, col, cage]. These drive native_head_alloc_for_present_types
    (the head-alloc fix): coloring P=1 -> 15 edge-heads; kenken/circuit P=3 -> 5/5/5.
    """
    if domain == "coloring":
        return [GLOBAL_TYPE_IDS["coloring_edge"]]
    if domain == "circuit":
        assert gate_types is not None
        return [_circuit_gate_global_id(g) for g in gate_types]
    if domain == "kenken":
        return [GLOBAL_TYPE_IDS["kenken_row"], GLOBAL_TYPE_IDS["kenken_col"],
                GLOBAL_TYPE_IDS["kenken_cage"]]
    raise ValueError(f"unknown multi-task domain {domain!r}")


def _build_multitask_adapter(domain: str, model, L_max: int, hidden: int,
                             gate_types: "tuple[str, ...] | None" = None,
                             n_heads: int = 16, n_factor_types: int = MT_T):
    """Return a callable native_batch -> _MultiTaskBatch for one domain.

    The adapter does the FOUR domain-agnosticizations that turn a per-domain native
    batch into the unified contract:
      1. REMAP latent_type to GLOBAL type ids (so the shared mask-builder + inlet
         separate every domain's factor-types FROM INPUT).
      2. PAD membership (B,L,S) -> (B,L_max,S) and latent_type (B,L) -> (B,L_max) so
         the JIT topology is stable regardless of which domain is sampled.
      3. PAD value_domain_mask (B,S,n) -> (B,S,7) (universal masked codebook): unused
         value slots are 0 -> the engine's (1-vdm)*(-1e4) bias zeroes them in-graph.
      4. BUILD the generic semantics inlet (per-latent GLOBAL type-id, + KenKen
         op/target/size; coloring/circuit carry the type-id alone).

    Remapping/padding are done on the NUMPY side (data loading), NOT in the JIT graph
    — no python branch on domain-id inside the backbone (the dense-only rule). The
    adapter emits the RAW per-latent semantic id tensors (op/target/size, B x L_max);
    the JIT step builds the generic inlet IN-GRAPH from them (so the inlet params train).
    coloring/circuit carry no arithmetic -> their op/target/size are all-zero (row 0 of
    each table = the "no-param" slot); only the GLOBAL type-id distinguishes them.
    """
    def _zero_sem(B):
        z = np.zeros((B, L_max), dtype=np.int32)
        return (Tensor(z, dtype=dtypes.int).contiguous().realize(),
                Tensor(z.copy(), dtype=dtypes.int).contiguous().realize(),
                Tensor(z.copy(), dtype=dtypes.int).contiguous().realize())

    # PER-DOMAIN NATIVE HEAD ALLOCATION (THE FIX), computed ONCE (constant per domain).
    # present_global_types -> native_head_alloc_for_present_types -> the (H,) global-type
    # array -> the (head_type_oh, head_is_global) tensors the engine routes on. These are
    # the SAME for every batch of this domain (the allocation depends only on which types
    # are present, not on the instance), so we build them once and re-attach per batch.
    from mycelium.factor_masks import (
        native_head_alloc_for_present_types, head_alloc_to_tensors,
    )
    present = _present_global_types(domain, gate_types)
    head_global_type = native_head_alloc_for_present_types(present, n_heads)
    head_type_oh_t, head_is_global_t = head_alloc_to_tensors(
        head_global_type, n_factor_types)
    head_type_oh_t = head_type_oh_t.realize()
    head_is_global_t = head_is_global_t.realize()

    if domain == "coloring":
        from mycelium.graph_coloring_data import LTYPE_EDGE
        edge_gid = GLOBAL_TYPE_IDS["coloring_edge"]

        def adapt(cb):
            B = int(cb.input_cells.shape[0])
            lt = _np_int(cb.latent_type)                      # (B, L) local: 0 edge / 1 pad
            # Remap: local edge (0) -> global edge id; everything else -> sentinel.
            lt_g = np.where(lt == LTYPE_EDGE, edge_gid, MT_SENTINEL).astype(np.int32)
            lt_g = _pad_latent_type_np(lt_g, L_max)
            mem = _pad_membership_np(_np_f(cb.membership), L_max)
            vdm = _pad_vdm_np(_np_f(cb.value_domain_mask))
            membership_t = Tensor(mem, dtype=dtypes.float).contiguous().realize()
            latent_type_t = Tensor(lt_g, dtype=dtypes.int).contiguous().realize()
            vdm_t = Tensor(vdm, dtype=dtypes.float).contiguous().realize()
            op_t, tgt_t, sz_t = _zero_sem(B)
            return _MultiTaskBatch(
                input_cells=cb.input_cells, cell_valid=cb.cell_valid,
                value_domain_mask=vdm_t, gold=cb.gold,
                membership=membership_t, latent_type=latent_type_t,
                domain="coloring", deduction_depth=cb.deduction_depth,
                inlet_op=op_t, inlet_target=tgt_t, inlet_size=sz_t,
                head_type_oh=head_type_oh_t, head_is_global=head_is_global_t)
        return adapt

    if domain == "circuit":
        assert gate_types is not None, "circuit adapter needs gate_types for the remap"
        # Local gate-type idx i -> global id by NAME (loader's gate_types order).
        local_to_global = {i: _circuit_gate_global_id(g) for i, g in enumerate(gate_types)}
        # Build a lookup vector indexed by local idx in [0, T_local]; sentinel -> sentinel.
        T_local = len(gate_types)
        remap_vec = np.full((T_local + 1,), MT_SENTINEL, dtype=np.int32)
        for i in range(T_local):
            remap_vec[i] = local_to_global[i]

        def adapt(cb):
            B = int(cb.input_cells.shape[0])
            lt = _np_int(cb.latent_type)                      # (B, L) local 0..T-1 / T pad
            lt_clipped = np.clip(lt, 0, T_local)              # T = sentinel row
            lt_g = remap_vec[lt_clipped].astype(np.int32)     # (B, L) global ids
            lt_g = _pad_latent_type_np(lt_g, L_max)
            mem = _pad_membership_np(_np_f(cb.membership), L_max)
            vdm = _pad_vdm_np(_np_f(cb.value_domain_mask))
            membership_t = Tensor(mem, dtype=dtypes.float).contiguous().realize()
            latent_type_t = Tensor(lt_g, dtype=dtypes.int).contiguous().realize()
            vdm_t = Tensor(vdm, dtype=dtypes.float).contiguous().realize()
            op_t, tgt_t, sz_t = _zero_sem(B)
            return _MultiTaskBatch(
                input_cells=cb.input_cells, cell_valid=cb.cell_valid,
                value_domain_mask=vdm_t, gold=cb.gold,
                membership=membership_t, latent_type=latent_type_t,
                domain="circuit", deduction_depth=cb.deduction_depth,
                inlet_op=op_t, inlet_target=tgt_t, inlet_size=sz_t,
                head_type_oh=head_type_oh_t, head_is_global=head_is_global_t)
        return adapt

    if domain == "kenken":
        from mycelium.kenken_data import N_MAX
        row_gid = GLOBAL_TYPE_IDS["kenken_row"]
        col_gid = GLOBAL_TYPE_IDS["kenken_col"]
        cage_gid = GLOBAL_TYPE_IDS["kenken_cage"]
        # The KenKen factor batch lays latents as [N_MAX rows | N_MAX cols | C cages].
        # Local type 0=row, 1=col, 2=cage -> global row/col/cage ids.
        local_to_global_kk = np.array([row_gid, col_gid, cage_gid], dtype=np.int32)

        def adapt(kb_pair):
            # kb_pair = (KenKenBatch, FactorGraphBatch-from-make_kenken_factor_batch).
            # Reuse make_kenken_factor_batch's membership/latent_type (rows/cols/cages),
            # REMAP local 0/1/2 -> global ids, and emit the per-latent op/target/size ids
            # aligned to the cage latents (the JIT builds the inlet in-graph from them).
            kb, fb = kb_pair
            lt = _np_int(fb.latent_type)                      # (B, L) local 0/1/2
            lt_g = local_to_global_kk[np.clip(lt, 0, 2)].astype(np.int32)
            lt_g = _pad_latent_type_np(lt_g, L_max)
            mem = _pad_membership_np(_np_f(fb.membership), L_max)
            vdm = _pad_vdm_np(_np_f(fb.value_domain_mask))    # already 7 wide -> no-op
            membership_t = Tensor(mem, dtype=dtypes.float).contiguous().realize()
            latent_type_t = Tensor(lt_g, dtype=dtypes.int).contiguous().realize()
            vdm_t = Tensor(vdm, dtype=dtypes.float).contiguous().realize()

            # Per-latent op/target/size: rows/cols carry no arithmetic (id 0), cages
            # carry the real cage_op/cage_target/cage_size. Build (B, L_max) int.
            B = int(kb.cage_op.shape[0])
            C = int(kb.cage_op.shape[1])
            n_rowcol = 2 * N_MAX
            op_np = np.zeros((B, L_max), dtype=np.int32)
            tgt_np = np.zeros((B, L_max), dtype=np.int32)
            sz_np = np.zeros((B, L_max), dtype=np.int32)
            op_np[:, n_rowcol:n_rowcol + C] = _np_int(kb.cage_op)
            tgt_np[:, n_rowcol:n_rowcol + C] = _np_int(kb.cage_target)
            sz_np[:, n_rowcol:n_rowcol + C] = _np_int(kb.cage_size)
            op_t = Tensor(op_np, dtype=dtypes.int).contiguous().realize()
            tgt_t = Tensor(tgt_np, dtype=dtypes.int).contiguous().realize()
            sz_t = Tensor(sz_np, dtype=dtypes.int).contiguous().realize()
            return _MultiTaskBatch(
                input_cells=kb.input_cells, cell_valid=kb.cell_valid,
                value_domain_mask=vdm_t, gold=kb.gold,
                membership=membership_t, latent_type=latent_type_t,
                domain="kenken", deduction_depth=kb.deduction_depth,
                inlet_op=op_t, inlet_target=tgt_t, inlet_size=sz_t,
                head_type_oh=head_type_oh_t, head_is_global=head_is_global_t)
        return adapt

    raise ValueError(f"unknown multi-task domain {domain!r}")


class _MultiTaskLoader:
    """Round-robin / weight-sampled domain loader. Each batch is PURE single-domain.

    Holds {domain -> (native_loader, adapter)} and a weight vector. sample_batch()
    samples a domain ~ weights, draws its native batch, and adapts it to the unified
    _MultiTaskBatch contract. eval_iter(domain) iterates ONE domain's test set.
    """
    def __init__(self, domain_loaders: dict, adapters: dict, weights: dict,
                 seed: int):
        self.domains = list(domain_loaders.keys())
        self.loaders = domain_loaders
        self.adapters = adapters
        w = np.array([float(weights[d]) for d in self.domains], dtype=np.float64)
        self.probs = (w / w.sum()).tolist()
        self.rng = np.random.RandomState(seed + 7)

    def sample_domain(self) -> str:
        idx = self.rng.choice(len(self.domains), p=self.probs)
        return self.domains[idx]

    def sample_batch(self, domain: "str | None" = None):
        d = domain or self.sample_domain()
        native = self.loaders[d].sample_batch()
        return self.adapters[d](native)


def _build_multitask_task(K, BATCH, EVAL_BATCH, SEED, hidden, n_heads, model,
                          train_path, test_path, mix, weights):
    """Build the unified multi-task _Task (the GENERAL-WEIGHTS harness).

    Steps:
      1. Build each sub-domain's NATIVE loader (KenKen file-based; coloring/circuit
         in-memory). The KenKen native loader returns KenKenBatch; we wrap it so the
         adapter receives (kb, make_kenken_factor_batch(kb)).
      2. PROBE each domain's L (membership rows) by drawing one batch; set L_max =
         max over domains -> the stable JIT topology width.
      3. UNIFIED spec: s_max=49, n_values=7, n_factor_types=N_GLOBAL_TYPES, n_heads,
         has_factor_inlet=True. (Attach generic-inlet + fg params happens in main.)
      4. Per-domain adapters + the _MultiTaskLoader.

    The constraint_energy_fn is None for the multi-task run (KenKen's energy is a
    metric, not a per-step loss here; keeping it out keeps the JIT signature uniform).
    """
    # ---- unified spec FIRST (we need has_factor_inlet=True before any inlet build).
    spec = FactorGraphSpec(s_max=MT_S_MAX, n_values=MT_N_MAX,
                           n_factor_types=MT_T, n_heads=n_heads, k_max=K,
                           has_factor_inlet=True)

    # ---- native loaders + raw (un-padded) membership widths.
    native_loaders: dict = {}
    raw_specs: dict = {}      # domain -> dict with L, gate_types, etc.

    if "coloring" in mix:
        from mycelium.graph_coloring_data import GraphColoringLoader
        s_max = MT_S_MAX
        n_values = int(getenv("FG_COLORING_N_VALUES", getenv("FG_N_VALUES", "3")))
        assert 0 < n_values <= MT_N_MAX, \
            f"coloring k={n_values} must be in 1..{MT_N_MAX} for the universal codebook"
        n_instances = int(getenv("FG_COLORING_N_INSTANCES",
                                 getenv("FG_N_INSTANCES", "8000")))
        cl = GraphColoringLoader(n_instances=n_instances, s_max=s_max,
                                 k_colors=n_values, batch_size=BATCH, seed=SEED)
        native_loaders["coloring"] = cl
        raw_specs["coloring"] = {"L": int(cl.n_edges_max), "gate_types": None,
                                 "n_values": n_values}

    if "circuit" in mix:
        from mycelium.circuit_data import CircuitLoader
        s_max = MT_S_MAX
        n_instances = int(getenv("FG_CIRCUIT_N_INSTANCES",
                                 getenv("FG_N_INSTANCES", "8000")))
        gate_types_env = getenv("FG_CIRCUIT_GATE_TYPES", "").strip()
        use_xor = int(getenv("FG_CIRCUIT_XOR", "0")) > 0
        if gate_types_env:
            gtypes = tuple(g.strip().upper() for g in gate_types_env.split(",") if g.strip())
        elif use_xor:
            gtypes = ("AND", "OR", "NOT", "XOR")
        else:
            gtypes = ("AND", "OR", "NOT")
        circ = CircuitLoader(n_instances=n_instances, s_max=s_max, n_values=2,
                             batch_size=BATCH, seed=SEED, gate_types=gtypes)
        native_loaders["circuit"] = circ
        raw_specs["circuit"] = {"L": int(circ.n_gates_max),
                                "gate_types": tuple(circ.gate_types), "n_values": 2}

    if "kenken" in mix:
        from mycelium.kenken import attach_kenken_params  # noqa: F401 (inlet not used)
        from mycelium.kenken_data import KenKenLoader, load_jsonl, N_MAX
        kk_spec = FactorGraphSpec(s_max=49, n_values=7, n_factor_types=3,
                                  n_heads=n_heads, k_max=K, has_factor_inlet=True)
        train_recs = load_jsonl(train_path)
        test_recs = load_jsonl(test_path)
        corpus_n_cages_max = max(
            max(len(r["cages"]) for r in train_recs),
            max(len(r["cages"]) for r in test_recs),
        )
        n_cages_max = int(getenv("KENKEN_N_CAGES_MAX", str(corpus_n_cages_max)))
        assert n_cages_max >= corpus_n_cages_max
        kk_train = KenKenLoader(train_path, batch_size=BATCH, seed=SEED,
                                n_cages_max=n_cages_max)
        kk_test = KenKenLoader(test_path, batch_size=EVAL_BATCH, seed=SEED + 1,
                               n_cages_max=n_cages_max)

        class _KKWrap:
            """Wrap KenKenLoader so sample_batch returns (kb, factor_batch)."""
            def __init__(self, loader):
                self.loader = loader
                self._kk_spec = kk_spec
            def sample_batch(self):
                kb = self.loader.sample_batch()
                fb = make_kenken_factor_batch(kb, self._kk_spec)
                return (kb, fb)
            def iter_eval(self, batch_size=None):
                for kb in self.loader.iter_eval(batch_size=batch_size or EVAL_BATCH):
                    yield (kb, make_kenken_factor_batch(kb, self._kk_spec))

        kk_train_w = _KKWrap(kk_train)
        kk_test_w = _KKWrap(kk_test)
        native_loaders["kenken"] = kk_train_w
        # L for kenken = 7 rows + 7 cols + n_cages_max cages.
        raw_specs["kenken"] = {"L": int(N_MAX + N_MAX + n_cages_max),
                               "gate_types": None, "n_values": 7,
                               "test_loader": kk_test_w}

    # ---- L_max = max membership width over the mix (stable JIT topology).
    L_max = max(rs["L"] for rs in raw_specs.values())
    print(f"  [multi] L_max={L_max}  per-domain L="
          f"{{{', '.join(f'{d}:{raw_specs[d]['L']}' for d in mix)}}}", flush=True)

    # ---- per-domain adapters.
    adapters: dict = {}
    for d in mix:
        adapters[d] = _build_multitask_adapter(
            d, model, L_max, hidden, gate_types=raw_specs[d]["gate_types"],
            n_heads=n_heads, n_factor_types=spec.n_factor_types)

    mt_loader = _MultiTaskLoader(native_loaders, adapters, weights, SEED)

    # ---- eval test loaders (per-domain).
    eval_loaders: dict = {}
    for d in mix:
        if d == "kenken":
            eval_loaders[d] = raw_specs[d]["test_loader"]
        else:
            eval_loaders[d] = native_loaders[d]   # in-memory loader owns the split

    # Pack everything the train/eval loop needs onto the _Task.
    task = _Task(
        spec=spec, train_loader=mt_loader, test_loader=mt_loader,
        to_factor_batch=lambda x: x,             # mt_loader already returns _MultiTaskBatch
        constraint_energy_fn=None, has_inlet=True,
        eval_iter=None,                          # multi-task uses per-domain eval below
    )
    task.is_multitask = True
    task.mix = list(mix)
    task.adapters = adapters
    task.eval_loaders = eval_loaders
    task.L_max = L_max
    return task


def _zeros_sem(B, L):
    z = np.zeros((B, L), dtype=np.int32)
    return Tensor(z, dtype=dtypes.int).contiguous().realize()


def _jit_inputs(fb: FactorGraphBatch, spec: FactorGraphSpec, has_inlet: bool,
                hidden: int, stoch_keep: Tensor):
    """Pull the stable JIT-input tuple out of a FactorGraphBatch. The pre-built inlet
    AND the raw semantic ids are ALWAYS passed (zeros when unused) so the JIT signature
    is stable across single-domain / multi-task. Single-domain reads factor_inlet (the
    pre-built tensor); multi-task reads inlet_op/target/size (builds the inlet in-graph).

    The PER-BATCH head-allocation tensors (head_type_oh (H,T), head_is_global (H,1,1))
    are ALSO always passed (zeros when single-domain) so the JIT signature is stable;
    only the multitask graph reads them (compile-time branch on `multitask`)."""
    B = int(fb.input_cells.shape[0])
    if has_inlet and fb.factor_inlet is not None:
        inlet = fb.factor_inlet.cast(dtypes.float)
    else:
        inlet = _zeros_inlet(B, spec.s_max, hidden)
    L = int(fb.membership.shape[1])
    inlet_op = getattr(fb, "inlet_op", None)
    if inlet_op is None:
        inlet_op = _zeros_sem(B, L)
        inlet_target = _zeros_sem(B, L)
        inlet_size = _zeros_sem(B, L)
    else:
        inlet_target = fb.inlet_target
        inlet_size = fb.inlet_size
    head_type_oh = getattr(fb, "head_type_oh", None)
    if head_type_oh is None:
        H = int(spec.n_heads)
        T = int(spec.n_factor_types)
        head_type_oh = Tensor(np.zeros((H, T), dtype=np.float32),
                              dtype=dtypes.float).contiguous().realize()
        head_is_global = Tensor(np.zeros((H, 1, 1), dtype=np.float32),
                                dtype=dtypes.float).contiguous().realize()
    else:
        head_is_global = fb.head_is_global
    # KENKEN raw cage features (verification-inlet build IN-GRAPH). ALWAYS passed (zeros
    # placeholder when absent) so the JIT signature is stable across tasks; only the
    # single-domain kenken graph READS them (the build_kk_inlet compile-time branch).
    # Shapes are run-fixed (C = n_cages_max pinned by the loader; cell_cage_id is (B,49)).
    kk_cage_op = getattr(fb, "kenken_cage_op", None)
    if kk_cage_op is not None:
        kk_cage_target = fb.kenken_cage_target
        kk_cage_size = fb.kenken_cage_size
        kk_cell_cage_id = fb.kenken_cell_cage_id
    else:
        # Placeholder for non-kenken (never read). (B,1) cages + (B,S) cell ids; -1 cell id
        # so build_verification_inlet (if ever traced) would zero every cell. Stable within
        # a task: coloring/circuit/multi never carry these attrs, so the shape is constant.
        # MUST be DISTINCT buffer objects: TinyJit rejects identical input buffers
        # ("duplicate inputs to JIT"), so op/target/size each get their own zeros tensor
        # (reusing one z1 for all three broke every non-kenken task incl. coloring + ECC).
        kk_cage_op = Tensor(np.zeros((B, 1), dtype=np.int32), dtype=dtypes.int).contiguous().realize()
        kk_cage_target = Tensor(np.zeros((B, 1), dtype=np.int32), dtype=dtypes.int).contiguous().realize()
        kk_cage_size = Tensor(np.zeros((B, 1), dtype=np.int32), dtype=dtypes.int).contiguous().realize()
        kk_cell_cage_id = Tensor(np.full((B, spec.s_max), -1, dtype=np.int32),
                                 dtype=dtypes.int).contiguous().realize()
    # CONTINUOUS INPUT (ECC / §8.1): the per-cell LLR the engine reads when
    # spec.continuous_input is True. ALWAYS passed (zeros placeholder when the batch
    # carries no cont_input, e.g. kenken/coloring/circuit) so the JIT signature is
    # stable across tasks; only the continuous-input engine branch READS it.
    cont_input = getattr(fb, "cont_input", None)
    if cont_input is None:
        cont_input = Tensor(np.zeros((B, spec.s_max), dtype=np.float32),
                            dtype=dtypes.float).contiguous().realize()
    else:
        cont_input = cont_input.cast(dtypes.float)
    return (fb.input_cells, fb.gold, fb.cell_valid, fb.value_domain_mask,
            fb.membership, fb.latent_type, inlet, stoch_keep,
            inlet_op, inlet_target, inlet_size,
            head_type_oh, head_is_global,
            kk_cage_op, kk_cage_target, kk_cage_size, kk_cell_cage_id,
            cont_input)


# ---------------------------------------------------------------------------
# Evaluation (eager forward; per-breath CE is eval-only, OUT of the JIT step).
# ---------------------------------------------------------------------------

def evaluate(task: _Task, K: int, max_batches: int) -> dict:
    Tensor.training = False
    spec = task.spec
    # PRIMAL/DUAL split: headline acc is over [0, P); dual acc over [P, s_max). For every
    # task except dual-view KenKen, primal_s_max is None => P == s_max => acc over ALL
    # positions (byte-identical eval). For dual-view KenKen P=49 => the 0.796 comparison.
    P = getattr(spec, "primal_s_max", None) or spec.s_max
    cell_eq_sum = 0.0
    n_cells = 0
    puzzle_eq_sum = 0
    n_puzzles = 0
    dual_eq_sum = 0.0
    n_dual = 0
    pb_ce_first = None

    n_batches = 0
    for native in task.eval_iter():
        fb = task.to_factor_batch(native)
        # KENKEN eval: to_factor_batch leaves factor_inlet=None (the in-graph training
        # build supersedes it). Build the verification inlet EAGERLY here from the carried
        # raw cage features so eval reads the TRAINED inlet params (mirrors the multitask
        # eval path). Other tasks carry no kenken_cage_op -> this is a kenken-only branch.
        if spec.has_factor_inlet and getattr(fb, "kenken_cage_op", None) is not None \
                and fb.factor_inlet is None:
            from mycelium.kenken import build_verification_inlet
            inlet = build_verification_inlet(
                model, fb.kenken_cage_op, fb.kenken_cage_target,
                fb.kenken_cage_size, fb.kenken_cell_cage_id)                 # (B,49,H)
            if int(inlet.shape[1]) < spec.s_max:                            # dual-view pad
                pad = Tensor.zeros(inlet.shape[0], spec.s_max - int(inlet.shape[1]),
                                   inlet.shape[-1], dtype=inlet.dtype)
                inlet = inlet.cat(pad, dim=1)
            fb.factor_inlet = inlet.realize()
        logits_history, _ = factor_breathing_forward(model, fb, spec, K=K)
        final_logits = logits_history[-1]

        cell_valid_np = fb.cell_valid.realize().numpy()                     # (B,S)
        gold_np = fb.gold.realize().numpy().astype(np.int32)                # (B,S)
        pred_np = (final_logits.argmax(axis=-1) + 1).realize().numpy().astype(np.int32)
        eq_np = ((pred_np == gold_np).astype(np.float32) * cell_valid_np)   # (B,S)

        Bn = int(cell_valid_np.shape[0])
        for b in range(Bn):
            valid = cell_valid_np[b] > 0.5
            valid_p = valid.copy(); valid_p[P:] = False        # primal positions [0,P)
            nvp = int(valid_p.sum())
            if nvp == 0:
                continue
            cell_eq_sum += float(eq_np[b][:P].sum())
            n_cells += nvp
            # puzzle solved == ALL primal cells correct (the actual KenKen solve)
            puzzle_eq_sum += int(np.all(pred_np[b][valid_p] == gold_np[b][valid_p]))
            n_puzzles += 1
            if P < spec.s_max:                                  # dual positions [P, s_max)
                valid_d = valid.copy(); valid_d[:P] = False
                dual_eq_sum += float(eq_np[b][P:].sum())
                n_dual += int(valid_d.sum())

        if pb_ce_first is None:
            # Per-breath CE on the FIRST eval batch only (eval-only; .realize()s).
            observed = (fb.input_cells > 0).cast(dtypes.float)
            supervise = (fb.cell_valid * (1.0 - observed)).reshape(Bn * spec.s_max)
            sup_sum = supervise.sum() + 1e-6
            gold_idx = (fb.gold - 1).clip(0, spec.n_values - 1).reshape(Bn * spec.s_max)
            pb = []
            for logits in logits_history:
                ce = logits.reshape(Bn * spec.s_max, spec.n_values
                                    ).sparse_categorical_crossentropy(
                    gold_idx, reduction="none")
                pb.append(float(((ce * supervise).sum() / sup_sum).realize().numpy()))
            pb_ce_first = pb

        n_batches += 1
        if n_batches >= max_batches:
            break

    Tensor.training = True
    out = {
        "cell_acc": cell_eq_sum / max(n_cells, 1),
        "puzzle_acc": puzzle_eq_sum / max(n_puzzles, 1),
        "n_puzzles": n_puzzles,
        "per_breath_ce": pb_ce_first or [],
    }
    if n_dual > 0:
        out["dual_cell_acc"] = dual_eq_sum / n_dual    # dual-view (column) prediction acc
    return out


def evaluate_multitask(task: _Task, K: int, max_batches: int) -> dict:
    """Per-domain eval for the multi-task harness. Returns {domain: {cell_acc, ...}}.

    Each domain is evaluated SEPARATELY on its own test set (the adapter pads + builds
    the inlet just like training). The per-domain split is what the convergence
    instrument correlates against each domain's own deduction-depth axis."""
    Tensor.training = False
    spec = task.spec
    out: dict = {}
    for d in task.mix:
        adapt = task.adapters[d]
        loader = task.eval_loaders[d]
        cell_eq_sum = 0.0
        n_cells = 0
        puzzle_eq_sum = 0
        n_puzzles = 0
        pb_ce_first = None
        n_batches = 0
        for native in loader.iter_eval():
            fb = adapt(native)
            # Build the generic inlet eagerly (same in-graph op the JIT step uses) so
            # eval reads the TRAINED inlet params.
            fb.factor_inlet = build_generic_factor_inlet(
                model, fb.membership, fb.latent_type, fb.cell_valid,
                op=fb.inlet_op, target=fb.inlet_target, size=fb.inlet_size).realize()
            logits_history, _ = factor_breathing_forward(model, fb, spec, K=K)
            final_logits = logits_history[-1]
            cell_valid_np = fb.cell_valid.realize().numpy()
            gold_np = fb.gold.realize().numpy().astype(np.int32)
            pred_np = (final_logits.argmax(axis=-1) + 1).realize().numpy().astype(np.int32)
            eq_np = ((pred_np == gold_np).astype(np.float32) * cell_valid_np)
            Bn = int(cell_valid_np.shape[0])
            for b in range(Bn):
                valid = cell_valid_np[b] > 0.5
                nv = int(valid.sum())
                if nv == 0:
                    continue
                cell_eq_sum += float(eq_np[b].sum())
                n_cells += nv
                puzzle_eq_sum += int(np.all(pred_np[b][valid] == gold_np[b][valid]))
                n_puzzles += 1
            if pb_ce_first is None:
                observed = (fb.input_cells > 0).cast(dtypes.float)
                supervise = (fb.cell_valid * (1.0 - observed)).reshape(Bn * spec.s_max)
                sup_sum = supervise.sum() + 1e-6
                gold_idx = (fb.gold - 1).clip(0, spec.n_values - 1).reshape(Bn * spec.s_max)
                pb = []
                for logits in logits_history:
                    ce = logits.reshape(Bn * spec.s_max, spec.n_values
                                        ).sparse_categorical_crossentropy(
                        gold_idx, reduction="none")
                    pb.append(float(((ce * supervise).sum() / sup_sum).realize().numpy()))
                pb_ce_first = pb
            n_batches += 1
            if n_batches >= max_batches:
                break
        out[d] = {
            "cell_acc": cell_eq_sum / max(n_cells, 1),
            "puzzle_acc": puzzle_eq_sum / max(n_puzzles, 1),
            "n_puzzles": n_puzzles,
            "per_breath_ce": pb_ce_first or [],
        }
    Tensor.training = True
    return out


def _print_eval_table(res: dict, K: int) -> None:
    dual_str = (f" dual_cell_acc={res['dual_cell_acc']:.3f}"
                if "dual_cell_acc" in res else "")
    print(f"  test: cell_acc={res['cell_acc']:.3f} "
          f"puzzle_acc={res['puzzle_acc']:.3f} n={res['n_puzzles']}{dual_str}", flush=True)
    pbe = res["per_breath_ce"]
    if pbe:
        if K <= 8:
            pbe_str = " ".join(f"{v:.2f}" for v in pbe)
        else:
            pbe_str = (" ".join(f"{v:.2f}" for v in pbe[:4]) + " ... "
                       + " ".join(f"{v:.2f}" for v in pbe[-4:]))
        print(f"    eval per_breath_ce[B0..B{K-1}]: {pbe_str}", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

model = None  # set in main(); evaluate() reads it as a module global (eval-only)


def main():
    global model

    TASK = getenv("FG_TASK", "kenken").strip().lower()
    # FG_MULTITASK=1 OR FG_TASK=multi triggers the general-weights harness.
    MULTITASK = int(getenv("FG_MULTITASK", "0")) > 0 or TASK == "multi"
    if MULTITASK:
        TASK = "multi"
    assert TASK in ("kenken", "dual_kenken", "coloring", "circuit", "ecc", "multi"), \
        f"FG_TASK must be kenken|dual_kenken|coloring|circuit|ecc|multi, got {TASK!r}"
    # The mix + weights (default all three, equal weight).
    MIX = [m.strip().lower() for m in getenv("FG_MIX", "coloring,circuit,kenken").split(",")
           if m.strip()]
    if MULTITASK:
        for m in MIX:
            assert m in ("coloring", "circuit", "kenken"), \
                f"FG_MIX domain {m!r} unsupported (SAT is search-tier; excluded)"
    MIX_WEIGHTS_ENV = getenv("FG_MIX_WEIGHTS", "").strip()  # e.g. "coloring:1,circuit:1,kenken:1"
    MIX_WEIGHTS = {m: 1.0 for m in MIX}
    if MIX_WEIGHTS_ENV:
        for pair in MIX_WEIGHTS_ENV.split(","):
            if ":" in pair:
                k_, v_ = pair.split(":")
                if k_.strip().lower() in MIX_WEIGHTS:
                    MIX_WEIGHTS[k_.strip().lower()] = float(v_)

    K = int(getenv("FG_K_MAX", getenv("K", "16")))
    BATCH = int(getenv("BATCH", 8))
    STEPS = int(getenv("STEPS", 2000))
    LR = float(getenv("LR", "3e-5"))
    CKPT_EVERY = int(getenv("CKPT_EVERY", 200))
    EVAL_EVERY = int(getenv("EVAL_EVERY", 100))
    LOG_EVERY = int(getenv("LOG_EVERY", 10))
    PER_BREATH_CE_EVERY = int(getenv("PER_BREATH_CE_EVERY", 50))
    GC_EVERY = int(getenv("GC_EVERY", 50))
    RUN_NAME = getenv("RUN_NAME", getenv("CKPT_LABEL", f"fg_{TASK}"))
    RESUME_FROM = getenv("RESUME_FROM", "")
    PYTHIA_INIT = int(getenv("PYTHIA_INIT", 1)) > 0
    SEED = int(getenv("SEED", 42))
    GRAD_CLIP = float(getenv("GRAD_CLIP", "0.0"))
    EVAL_BATCHES = int(getenv("EVAL_BATCHES", 20))
    EVAL_BATCH = int(getenv("EVAL_BATCH", BATCH))
    EVAL_ONLY = int(getenv("FG_EVAL_ONLY", 0)) > 0 or STEPS == 0

    # Loss knobs.
    CONSTRAINT_WEIGHT = float(getenv("FG_CONSTRAINT_WEIGHT", "0.0"))
    CALIB_WEIGHT = float(getenv("FG_CALIB_WEIGHT", "0.1"))
    ORTHO_LAMBDA = float(getenv("FG_ORTHO_LAMBDA", "0.0"))
    LABEL_SMOOTHING = float(getenv("LABEL_SMOOTHING", "0.0"))
    WEIGHT_DECAY = float(getenv("WEIGHT_DECAY", "0.05"))
    STOCH_DEPTH_P = float(getenv("STOCH_DEPTH_P", "0.0"))

    # ---- IN-DEDUCER WAIST (FG_WAIST) knobs. Default OFF -> byte-identical training.
    FG_WAIST = int(getenv("FG_WAIST", "0")) > 0
    FG_WAIST_DIM_ENV = int(getenv("FG_WAIST_DIM", str(FG_WAIST_DIM)))
    FG_WAIST_AFTER_ENV = int(getenv("FG_WAIST_AFTER", str(FG_WAIST_AFTER)))
    FG_WAIST_GATE_INIT_ENV = float(getenv("FG_WAIST_GATE_INIT", str(FG_WAIST_GATE_INIT)))
    FG_WAIST_AUX = getenv("FG_WAIST_AUX", "classify").strip().lower()
    assert FG_WAIST_AUX in ("none", "classify", "attract", "both"), \
        f"FG_WAIST_AUX must be none|classify|attract|both, got {FG_WAIST_AUX!r}"
    FG_WAIST_AUX_W = float(getenv("FG_WAIST_AUX_W", "0.1"))         # classify weight
    FG_WAIST_ATTRACT_W = float(getenv("FG_WAIST_ATTRACT_W", "0.01"))  # attract weight
    FG_WAIST_CENTROID_MOM = float(getenv("FG_WAIST_CENTROID_MOM", "0.99"))

    # ---- ECC FIXES (the two diagnosed one-shot-collapse fixes; INDEPENDENT toggles).
    # FG_ECC_REINJECT=1  -> per-breath channel re-injection (re-add the B0 channel embed
    #                       to the residual at the start of every breath; no new params).
    # FG_LORA_RANK=r (0)  -> per-breath rank-r LoRA adapters (K unique low-rank residual
    #                       transforms; zero-init B -> neutral at step 0). Default 0 = OFF.
    # Both default OFF -> the spec fields are False/0 and the engine + trainer are
    # byte-identical to current for ALL tasks (incl. ECC-without-flags).
    FG_ECC_REINJECT = int(getenv("FG_ECC_REINJECT", "0")) > 0
    FG_LORA_RANK = int(getenv("FG_LORA_RANK", "0"))

    # Default paths for the kenken task only. Coloring uses an in-memory generator
    # (GraphColoringLoader); FG_TRAIN / FG_TEST are not read by the coloring branch.
    FG_TRAIN = getenv("FG_TRAIN", ".cache/kenken_train.jsonl")
    FG_TEST = getenv("FG_TEST", ".cache/kenken_test.jsonl")

    print("=== Mycelium factor-graph training (general engine) ===")
    print(f"task={TASK}  device={Device.DEFAULT}  B={BATCH}  K={K}  "
          f"steps={STEPS}  lr={LR}")
    print(f"constraint_weight={CONSTRAINT_WEIGHT}  calib_weight={CALIB_WEIGHT}  "
          f"ortho_lambda={ORTHO_LAMBDA}  grad_clip={GRAD_CLIP}")
    print(f"REG: label_smoothing={LABEL_SMOOTHING}  weight_decay={WEIGHT_DECAY}  "
          f"stoch_depth_p={STOCH_DEPTH_P}")
    if TASK in ("kenken", "dual_kenken"):
        print(f"train_path={FG_TRAIN}  test_path={FG_TEST}")
    elif TASK == "coloring":
        print(f"coloring corpus: FG_N_INSTANCES={getenv('FG_N_INSTANCES','8000')} "
              f"s_max={getenv('FG_S_MAX','49')} k={getenv('FG_N_VALUES','?')} (in-memory)")
    else:
        print(f"circuit corpus: FG_N_INSTANCES={getenv('FG_N_INSTANCES','8000')} "
              f"s_max={getenv('FG_S_MAX','49')} n_values=2 "
              f"xor={getenv('FG_CIRCUIT_XOR','0')} "
              f"gate_types={getenv('FG_CIRCUIT_GATE_TYPES','(default)')} (in-memory)")
    print(f"warm-start={'COLD' if not RESUME_FROM else RESUME_FROM}")
    print()

    np.random.seed(SEED)
    Tensor.manual_seed(SEED)

    # ---- backbone (Pythia L0-L3). hidden read from cfg → no 1024 hardcode.
    # ENV-OVERRIDABLE base dims so a wider Pythia (e.g. 1.4B at h=2048) can be the
    # base WITHOUT hardcoding 2k. HARD CONTRACT: with FG_HIDDEN/FG_HEAD_DIM/FG_FFN/
    # FG_N_HEADS_BASE UNSET, getenv returns the int defaults below (= Config()'s
    # 410M dims: hidden=1024, head_dim=64, ffn=4096, n_heads=16) → Config(...) is
    # byte-identical to Config() → the 410M path (all existing runs/ckpts) is
    # unchanged. For 1.4B set FG_HIDDEN=2048 FG_HEAD_DIM=128 FG_FFN=8192 and point
    # PYTHIA_WEIGHTS at the 1.4B safetensors (rotary_pct=0.25 → rotary_dim=32).
    cfg = Config(
        hidden=int(getenv("FG_HIDDEN", 1024)),
        head_dim=int(getenv("FG_HEAD_DIM", 64)),
        ffn=int(getenv("FG_FFN", 4096)),
        n_heads=int(getenv("FG_N_HEADS_BASE", 16)),
    )
    print(f"loading Pythia (h={cfg.hidden} head_dim={cfg.head_dim} ffn={cfg.ffn} "
          f"n_heads={cfg.n_heads}) -> breathing transformer (PYTHIA_INIT={PYTHIA_INIT})...")
    if PYTHIA_INIT:
        sd = _load_state()
        model = load_breathing(cfg, sd=sd)
        del sd
    else:
        model = BreathingTransformer(cfg)
    cast_layers_fp32(model)
    hidden = cfg.hidden
    n_heads = cfg.n_heads

    # ---- build the task FIRST when it needs to attach inlet params before the fg
    # params (kenken). Then attach the general factor-graph params (fg_*).
    if TASK == "kenken":
        task = _build_kenken_task(K, BATCH, EVAL_BATCH, SEED, hidden, n_heads,
                                  model, FG_TRAIN, FG_TEST)
    elif TASK == "dual_kenken":
        task = _build_dual_kenken_task(K, BATCH, EVAL_BATCH, SEED, hidden, n_heads,
                                       model, FG_TRAIN, FG_TEST)
    elif TASK == "coloring":
        task = _build_coloring_task(K, BATCH, EVAL_BATCH, SEED, n_heads)
    elif TASK == "circuit":
        task = _build_circuit_task(K, BATCH, EVAL_BATCH, SEED, n_heads)
    elif TASK == "ecc":
        task = _build_ecc_task(K, BATCH, EVAL_BATCH, SEED, n_heads)
    else:   # multi (GENERAL-WEIGHTS harness)
        # The generic inlet params MUST be attached BEFORE the task is built (the
        # per-domain adapters call build_generic_factor_inlet on `model`).
        attach_factor_inlet_params(model, hidden=hidden)
        print(f"  [multi] generic inlet attached: type_table_rows={MT_T + 1} "
              f"(N_GLOBAL_TYPES={N_GLOBAL_TYPES}); mix={MIX} weights={MIX_WEIGHTS}",
              flush=True)
        task = _build_multitask_task(K, BATCH, EVAL_BATCH, SEED, hidden, n_heads,
                                     model, FG_TRAIN, FG_TEST, MIX, MIX_WEIGHTS)
    spec = task.spec

    # ---- ECC FIXES: stamp the two INDEPENDENT toggles onto the spec (the engine reads
    # spec.reinject_input / spec.lora_rank as compile-time python branches; both default
    # off -> byte-identical). These are spec fields (mirroring continuous_input) so the
    # eval driver constructs an identical spec from the same env and reads the same ckpt.
    spec.reinject_input = bool(FG_ECC_REINJECT)
    spec.lora_rank = int(FG_LORA_RANK)
    if FG_ECC_REINJECT or FG_LORA_RANK > 0:
        print(f"  [ECC fixes] reinject_input={spec.reinject_input} "
              f"lora_rank={spec.lora_rank}", flush=True)

    # The GENERAL factor-graph params (fg_state_embed / fg_position_embed /
    # fg_value_codebook / fg_calib_head / fg_breath_embed / fg_delta_gate).
    attach_factor_graph_params(model, hidden=hidden, spec=spec)

    # PER-BREATH LoRA (FG_LORA_RANK>0): attach the K rank-r zero-init-B adapters. OFF
    # (rank 0) -> NOT attached -> the engine's getattr-gate skips LoRA entirely -> forward
    # byte-identical, training byte-identical, ckpt byte-identical.
    if FG_LORA_RANK > 0:
        attach_factor_lora_params(model, hidden=hidden, spec=spec, rank=FG_LORA_RANK)
        n_lora = 2 * int(spec.k_max) * int(FG_LORA_RANK) * hidden
        print(f"  [FG_LORA_RANK={FG_LORA_RANK}] per-breath LoRA attached: "
              f"K={spec.k_max} r={FG_LORA_RANK} H={hidden} "
              f"({n_lora/1e6:.2f}M params; B_k zero-init -> neutral at step 0)",
              flush=True)

    # IN-DEDUCER WAIST (FG_WAIST=1): attach the zero-init-gated convex-blend waist params.
    # OFF (default) -> NOT attached -> the engine's getattr-gate skips the waist entirely
    # -> forward byte-identical, training byte-identical, ckpt byte-identical.
    if FG_WAIST:
        attach_factor_waist_params(
            model, hidden=hidden, d=FG_WAIST_DIM_ENV, after=FG_WAIST_AFTER_ENV,
            gate_init=FG_WAIST_GATE_INIT_ENV, aux=FG_WAIST_AUX)
        import math as _math
        g0 = 1.0 / (1.0 + _math.exp(-FG_WAIST_GATE_INIT_ENV))
        print(f"  [FG_WAIST=1] waist attached: d={FG_WAIST_DIM_ENV} "
              f"after_layer={FG_WAIST_AFTER_ENV} gate_init={FG_WAIST_GATE_INIT_ENV} "
              f"(g0=sigmoid(gate_init)={g0:.2e} -> warm-start ~ identity blend) "
              f"aux={FG_WAIST_AUX} aux_w={FG_WAIST_AUX_W} attract_w={FG_WAIST_ATTRACT_W}",
              flush=True)

    # FG_HYP_MASK=1: build the per-type Poincaré anchor tables and attach them.
    # We need a REPRESENTATIVE membership/latent_type to determine G_t per type.
    # Use the first training batch drawn from the task's loader.
    # NOTE: hyperbolic masks are Tier-2 research, out of scope for the multi-task
    # general-weights harness — incompatible (one anchor field per type can't span a
    # cross-domain type set with one representative batch). Disallow the combination.
    if FG_HYP_MASK and task.is_multitask:
        raise ValueError("FG_HYP_MASK=1 is not supported with FG_MULTITASK (Tier-2 "
                         "hyperbolic masks are single-domain; disable one).")
    if FG_HYP_MASK:
        _ref_native = task.train_loader.sample_batch()
        _ref_fb = task.to_factor_batch(_ref_native)
        _mem_np = _ref_fb.membership.realize().numpy()       # (B_ref, L, s_max)
        _lt_np  = _ref_fb.latent_type.realize().numpy()      # (B_ref, L)
        print(f"  [FG_HYP_MASK=1] attach_factor_hyperbolic_params "
              f"(dim={os.environ.get('FG_HYP_DIM','48')}, "
              f"rho={os.environ.get('FG_HYP_RHO','0.7')}, freeze={FG_HYP_FREEZE})")
        attach_factor_hyperbolic_params(
            model,
            n_heads=spec.n_heads,
            n_factor_types=spec.n_factor_types,
            s_max=spec.s_max,
            membership_np=_mem_np,
            latent_type_np=_lt_np,
        )
        del _ref_native, _ref_fb, _mem_np, _lt_np

    Device[Device.DEFAULT].synchronize()

    # ---- params: backbone + fg params (+ kenken inlet tables when present).
    params = collect_backbone_params(model) + factor_graph_parameters(model)
    # IN-DEDUCER WAIST params (down/up/gate + optional aux head). Empty when not attached.
    params += factor_waist_parameters(model)
    # PER-BREATH LoRA params (A/B). Empty when not attached (FG_LORA_RANK=0).
    params += factor_lora_parameters(model)
    # KenKen verification-inlet params (trained — they're LIVE at init, not gated).
    # In the multi-task path these are NOT attached (the generic inlet replaces them).
    for nm in _KENKEN_INLET_NAMES:
        t = getattr(model, nm, None)
        if t is not None:
            params.append(t)
    # Generic semantics-as-input inlet params (multi-task only; LIVE at init).
    if task.is_multitask:
        params += factor_inlet_parameters(model)

    # FG_HYP_MASK=1 + FG_HYP_FREEZE=0 (Step 3 relax): add the hyperbolic anchor
    # params as a separate optimizer param group.  When FG_HYP_FREEZE=1 (default,
    # Step 2 frozen-confirm) they are NOT added — they are fixed constants and the
    # optimizer never touches them.
    if FG_HYP_MASK and not FG_HYP_FREEZE:
        from mycelium.factor_masks import cell_mp_head_allocation, CELL_MP_HEAD_GLOBAL
        _G = max(1, spec.n_heads // 16)
        _alloc = cell_mp_head_allocation(spec.n_factor_types, spec.n_heads, _G)
        _used_types = sorted({int(t) for t in _alloc if int(t) != CELL_MP_HEAD_GLOBAL})
        for _t in _used_types:
            _anchors = getattr(model, f"fg_hyp_anchors_{_t}", None)
            if _anchors is not None:
                params.append(_anchors)
        print(f"  [FG_HYP_FREEZE=0] added hyperbolic anchor params to optimizer "
              f"(types={_used_types})")
    n_params = sum(int(np.prod(t.shape)) for t in params)
    print(f"  trainable params: {n_params/1e6:.1f}M  (spec: s_max={spec.s_max} "
          f"n_values={spec.n_values} T={spec.n_factor_types} "
          f"inlet={spec.has_factor_inlet})")

    if RESUME_FROM:
        print(f"resuming from fg ckpt: {RESUME_FROM}")
        load_ckpt(model, RESUME_FROM, strict=EVAL_ONLY)
        print("  loaded.")

    opt = AdamW(params, lr=LR, weight_decay=WEIGHT_DECAY)

    # ---- ckpt dir + provenance
    run_dir = os.path.join(".cache/fg_ckpts", RUN_NAME)
    os.makedirs(run_dir, exist_ok=True)
    measured_config = {
        "arch_version": "factor_graph_v1",
        "engine": "factor_graph_engine",
        "task": TASK,
        "spec": {
            "s_max": spec.s_max, "n_values": spec.n_values,
            "n_factor_types": spec.n_factor_types, "n_heads": spec.n_heads,
            "k_max": spec.k_max, "has_factor_inlet": spec.has_factor_inlet,
        },
        "base": "Pythia-410M",
        "backbone": "pythia", "n_pythia_layers": 4,
        "hidden": hidden, "n_heads": n_heads,
        "K": K, "B": BATCH, "LR": LR, "steps": STEPS, "seed": SEED,
        "warm_start": "none" if not RESUME_FROM else RESUME_FROM,
        "constraint_weight": CONSTRAINT_WEIGHT, "calib_weight": CALIB_WEIGHT,
        "ortho_lambda": ORTHO_LAMBDA, "grad_clip": GRAD_CLIP,
        "reg_stack": {
            "label_smoothing": LABEL_SMOOTHING, "weight_decay": WEIGHT_DECAY,
            "stoch_depth_p": STOCH_DEPTH_P,
        },
        "perm_aug": {
            "enabled": FG_PERM_AUG, "frac": FG_PERM_AUG_FRAC,
            "seed": FG_PERM_AUG_SEED,
        },
        "waist": {
            "enabled": FG_WAIST, "dim": FG_WAIST_DIM_ENV, "after": FG_WAIST_AFTER_ENV,
            "gate_init": FG_WAIST_GATE_INIT_ENV, "aux": FG_WAIST_AUX,
            "aux_w": FG_WAIST_AUX_W, "attract_w": FG_WAIST_ATTRACT_W,
            "centroid_mom": FG_WAIST_CENTROID_MOM,
        },
        **({"train_path": FG_TRAIN, "test_path": FG_TEST} if TASK == "kenken"
           else {"multitask": True, "mix": MIX, "mix_weights": MIX_WEIGHTS,
                 "L_max": task.L_max, "n_global_factor_types": N_GLOBAL_TYPES,
                 "generic_inlet": True, "universal_codebook_N": MT_N_MAX,
                 **({"train_path": FG_TRAIN, "test_path": FG_TEST}
                    if "kenken" in MIX else {})}
           if TASK == "multi"
           else {"n_instances": int(getenv("FG_N_INSTANCES", "8000")),
                 "corpus": "in-memory",
                 **({"circuit_xor": int(getenv("FG_CIRCUIT_XOR", "0")) > 0,
                     "circuit_gate_types": getenv("FG_CIRCUIT_GATE_TYPES", "")}
                    if TASK == "circuit" else {})}),
        "trainable_params_M": round(n_params / 1e6, 3),
        "ckpt_dir": run_dir,
    }
    with open(os.path.join(run_dir, "measured_config.json"), "w") as f:
        json.dump(measured_config, f, indent=2)
    print(f"  wrote provenance: {os.path.join(run_dir, 'measured_config.json')}")

    # ---- EVAL-ONLY path (FG_EVAL_ONLY=1 or STEPS=0).
    if EVAL_ONLY:
        Tensor.training = False
        print(f"\n=== EVAL-ONLY (no training; ckpt={RESUME_FROM or 'COLD'}) ===")
        t_eval = time.time()
        if task.is_multitask:
            res = evaluate_multitask(task, K=K, max_batches=EVAL_BATCHES)
            for d in task.mix:
                print(f"  [{d}]", end=" ")
                _print_eval_table(res[d], K)
        else:
            res = evaluate(task, K=K, max_batches=EVAL_BATCHES)
            _print_eval_table(res, K)
        print(f"  (eval-only done in {time.time() - t_eval:.1f}s; NO ckpt written)",
              flush=True)
        return

    # ---- WAIST attract: the running VALID CENTROID buffer (fixed (d,), assigned in-place
    # inside the JIT, returned in the step tuple per the assign-must-return quirk). Built
    # only for attract/both; None otherwise (the step keys on its presence). NOT loaded from
    # ckpt — it warms up via the in-graph EMA from gold-TF d-reps (a few steps to settle).
    valid_centroid = None
    if FG_WAIST and FG_WAIST_AUX in ("attract", "both"):
        valid_centroid = Tensor(np.zeros((FG_WAIST_DIM_ENV,), dtype=np.float32),
                                dtype=dtypes.float).contiguous().realize()

    # ---- JIT compile the train step.  Multi-task: keyed with (multitask, mix_key)
    # so the unified graph is cached separately from any single-domain graph.
    Tensor.training = True
    mix_key = ",".join(sorted(task.mix)) if task.is_multitask else ""
    step_fn = _compile_jit_fg_step(
        model, opt, spec, TASK, K=K, B=BATCH,
        constraint_weight=CONSTRAINT_WEIGHT, calib_weight=CALIB_WEIGHT,
        ortho_lambda=ORTHO_LAMBDA, grad_clip=GRAD_CLIP,
        label_smoothing=LABEL_SMOOTHING, stoch_depth_p=STOCH_DEPTH_P,
        constraint_energy_fn=task.constraint_energy_fn,
        has_inlet=task.has_inlet,
        multitask=task.is_multitask, mix_key=mix_key,
        waist_on=FG_WAIST, waist_aux=FG_WAIST_AUX,
        waist_aux_w=FG_WAIST_AUX_W, waist_attract_w=FG_WAIST_ATTRACT_W,
        valid_centroid=valid_centroid, centroid_momentum=FG_WAIST_CENTROID_MOM,
    )

    # ---- stochastic-depth keep-mask RNG (fed as a JIT input so the drop pattern
    # varies WITHOUT recompiling; all-ones when off — the branch ignores it).
    sd_rng = np.random.RandomState(SEED + 99)
    ones_keep = Tensor(np.ones((K,), dtype=np.float32),
                       dtype=dtypes.float).contiguous().realize()

    def _draw_stoch_keep():
        if STOCH_DEPTH_P <= 0.0:
            return ones_keep
        kept = (sd_rng.rand(K) >= STOCH_DEPTH_P).astype(np.float32)
        scale = kept / (1.0 - STOCH_DEPTH_P)
        return Tensor(scale.astype(np.float32),
                      dtype=dtypes.float).contiguous().realize()

    # ---- PERMUTATION-AUGMENTATION RNG (FG_PERM_AUG). A DEDICATED stream so it does
    # NOT perturb the loader / stoch-depth RNG draws — when FG_PERM_AUG is OFF this
    # object is constructed but NEVER advanced (permute_factor_batch is not called),
    # so the batch + every other RNG stream is byte-identical to current training.
    perm_aug_rng = np.random.RandomState(FG_PERM_AUG_SEED)
    if FG_PERM_AUG:
        print(f"  [FG_PERM_AUG=1] vertex-relabel augmentation ON: frac="
              f"{FG_PERM_AUG_FRAC} seed={FG_PERM_AUG_SEED} (data-side relabel of each "
              f"instance + gold; engine untouched).", flush=True)

    # ---- train loop (deferred logging: accumulate ON-GPU, one .numpy() / LOG_EVERY).
    print("\ntraining...\n")
    t0 = time.time()
    log_acc = None
    log_n = 0
    _prof = int(getenv("FG_PROFILE", "0")) > 0   # FG_PROFILE: split CPU-prep vs GPU-step (perf audit)
    _prof_cpu = 0.0; _prof_gpu = 0.0; _prof_n = 0
    # Per-domain loss/acc tracking (multi-task only; CPU-side floats accumulated per
    # LOG_EVERY window). Keeps the JIT step domain-agnostic — domain attribution is
    # external (each batch is pure single-domain, so its scalars belong to one domain).
    mt_dom_acc = {d: {"loss": 0.0, "cell_acc": 0.0, "n": 0} for d in (task.mix or [])}

    for step in range(1, STEPS + 1):
        if _prof: _pc0 = time.perf_counter()
        if task.is_multitask:
            cur_domain = task.train_loader.sample_domain()
            fb = task.train_loader.sample_batch(domain=cur_domain)
        else:
            cur_domain = None
            native = task.train_loader.sample_batch()
            fb = task.to_factor_batch(native)
        # FG_PERM_AUG: relabel each instance's REAL cells (+ gold) on the data side,
        # BEFORE _jit_inputs pulls the engine inputs. Off (default) -> byte-identical
        # (call skipped, no augmentation RNG drawn). Composes with the multitask path
        # (the relabel is on the cell axis only; latent_type / inlet are unchanged).
        if FG_PERM_AUG:
            fb = permute_factor_batch(fb, perm_aug_rng, FG_PERM_AUG_FRAC)
        ins = _jit_inputs(fb, spec, task.has_inlet, hidden, _draw_stoch_keep())
        if _prof: _pc1 = time.perf_counter()
        outs = step_fn(*ins)
        if _prof:
            outs[0].realize()                       # force the GPU step to finish (sync)
            _pc2 = time.perf_counter()
            if step > 1:                            # skip step 1 (JIT compile)
                _prof_cpu += _pc1 - _pc0
                _prof_gpu += _pc2 - _pc1
                _prof_n += 1

        # RUNG-2 RIM GUARD (spec §7): under relaxation the hyperbolic anchors are TANGENT
        # params and the d_hyp backward's 1/(1-|z|^2) explodes near the ball boundary.
        # Clamp |v| so |z|=tanh(|v|) stays off the rim, AFTER each optimizer step (the
        # step runs inside the JIT; this in-place assign runs outside it). No-op unless
        # FG_HYP_MASK=1 and FG_HYP_FREEZE=0 (anchors are in the optimizer == relaxing).
        if FG_HYP_MASK and not FG_HYP_FREEZE and FG_HYP_RELAX:
            clamp_factor_hyp_tangent_norms(model)

        total_t      = outs[0]
        healthy_t    = outs[1]
        cell_ce_t    = outs[2]
        energy_t     = outs[3]
        calib_t      = outs[4]
        cell_acc_t   = outs[5]
        puzzle_acc_t = outs[6]
        ortho_t      = outs[7]
        pb_ce_ts     = outs[8:8 + K]
        pb_calib_ts  = outs[8 + K:8 + 2 * K]
        # WAIST aux scalar at 8+2K (centroid follows at 8+2K+1; ignored by the host loop —
        # it exists only so the JIT does not drop the in-place centroid assign).
        waist_aux_t  = outs[8 + 2 * K] if len(outs) > 8 + 2 * K else None

        cur = total_t.reshape(1).cat(cell_ce_t.reshape(1), energy_t.reshape(1),
                                     calib_t.reshape(1), healthy_t.reshape(1),
                                     cell_acc_t.reshape(1))
        log_acc = cur.realize() if log_acc is None else (log_acc + cur).realize()
        log_n += 1

        # Per-domain accumulation (multi-task): one host sync of (cell_ce, cell_acc)
        # per step — attributed to the sampled domain. Cheap (two scalars).
        if task.is_multitask:
            dce = float(cell_ce_t.numpy())
            dca = float(cell_acc_t.numpy())
            md = mt_dom_acc[cur_domain]
            md["loss"] += dce
            md["cell_acc"] += dca
            md["n"] += 1

        if step % LOG_EVERY == 0:
            v = log_acc.numpy()  # the ONLY host sync in the hot loop (per LOG_EVERY)
            loss_a, cell_ce_a, energy_a, calib_a, healthy_a, cell_acc_a = (
                float(x) for x in v)
            n_skips = int(round(log_n - healthy_a))
            if n_skips > 0:
                print(f"[NaN-skip] {n_skips} step(s) in [{step-log_n+1}..{step}] had "
                      f"NaN grad — where()-gated, Adam moments protected", flush=True)
            dt = time.time() - t0
            print(f"[step {step:5d}] loss={loss_a/log_n:.4f} "
                  f"cell_ce={cell_ce_a/log_n:.4f} energy={energy_a/log_n:.4f} "
                  f"calib={calib_a/log_n:.4f}  ({dt:.1f}s, {dt/step:.2f}s/step)",
                  flush=True)
            if _prof and _prof_n:
                _pt = _prof_cpu + _prof_gpu
                print(f"  [PROFILE] cpu_prep={_prof_cpu/_prof_n*1e3:.1f}ms "
                      f"gpu_step={_prof_gpu/_prof_n*1e3:.1f}ms "
                      f"gpu_idle_on_cpu_prep={_prof_cpu/_pt*100:.1f}%  (n={_prof_n})",
                      flush=True)
            if task.is_multitask:
                parts = []
                for d in task.mix:
                    md = mt_dom_acc[d]
                    if md["n"] > 0:
                        parts.append(f"{d}[ce={md['loss']/md['n']:.3f} "
                                     f"acc={md['cell_acc']/md['n']:.3f} n={md['n']}]")
                    else:
                        parts.append(f"{d}[--]")
                print(f"    per-domain: {'  '.join(parts)}", flush=True)
                mt_dom_acc = {d: {"loss": 0.0, "cell_acc": 0.0, "n": 0}
                              for d in task.mix}
            log_acc = None
            log_n = 0

        if step % PER_BREATH_CE_EVERY == 0:
            pb_ce = [float(t.numpy()) for t in pb_ce_ts]
            pb_calib = [float(t.numpy()) for t in pb_calib_ts]
            if K <= 8:
                pb_ce_str = " ".join(f"{v:.2f}" for v in pb_ce)
                pb_calib_str = " ".join(f"{v:.2f}" for v in pb_calib)
            else:
                pb_ce_str = (" ".join(f"{v:.2f}" for v in pb_ce[:4]) + " ... "
                             + " ".join(f"{v:.2f}" for v in pb_ce[-4:]))
                pb_calib_str = (" ".join(f"{v:.2f}" for v in pb_calib[:4]) + " ... "
                                + " ".join(f"{v:.2f}" for v in pb_calib[-4:]))
            ortho_str = (f" ortho={float(ortho_t.numpy()):.4f}"
                         if ORTHO_LAMBDA > 0 else "")
            waist_str = ""
            if FG_WAIST:
                g_now = float(model.fg_waist_gate.sigmoid().mean().numpy())
                aux_now = (float(waist_aux_t.numpy()) if waist_aux_t is not None else 0.0)
                waist_str = f" waist_gate={g_now:.4f} waist_aux={aux_now:.4f}"
            print(f"  per_breath_ce[B0..B{K-1}]:    {pb_ce_str}  "
                  f"(train cell_acc={float(cell_acc_t.numpy()):.3f} "
                  f"puzzle_acc={float(puzzle_acc_t.numpy()):.3f}{ortho_str}{waist_str})",
                  flush=True)
            print(f"  per_breath_calib[B0..B{K-1}]: {pb_calib_str}", flush=True)

        if step % EVAL_EVERY == 0:
            print(f"  evaluating on test ({EVAL_BATCHES} batches × B={EVAL_BATCH})...",
                  flush=True)
            if task.is_multitask:
                res = evaluate_multitask(task, K=K, max_batches=EVAL_BATCHES)
                for d in task.mix:
                    print(f"  [{d}]", end=" ")
                    _print_eval_table(res[d], K)
            else:
                res = evaluate(task, K=K, max_batches=EVAL_BATCHES)
                _print_eval_table(res, K)

        if step % CKPT_EVERY == 0:
            ckpt_path = os.path.join(run_dir, f"{RUN_NAME}_step{step}.safetensors")
            safe_save(model_state_dict_fg(model), ckpt_path)
            print(f"  saved {ckpt_path}", flush=True)

        if step % GC_EVERY == 0:
            gc.collect()

    ckpt_path = os.path.join(run_dir, f"{RUN_NAME}_final.safetensors")
    safe_save(model_state_dict_fg(model), ckpt_path)
    print(f"\ndone. saved {ckpt_path}", flush=True)


if __name__ == "__main__":
    main()
