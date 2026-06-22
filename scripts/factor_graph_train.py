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


def load_ckpt(model, path: str):
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
        print(f"  ckpt missing {len(missing)} keys (kept init): "
              f"{missing[:3]}{'...' if len(missing) > 3 else ''}")


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
                         multitask: bool = False, mix_key: str = ""):
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
           bool(FG_HYP_MASK), bool(FG_HYP_FREEZE),
           # Rung-2 relax knobs CHANGE THE TRACED GRAPH BODY (exp_0 vs raw coord,
           # euclid vs hyp distance, soft-block vs saturated alpha), so they MUST be
           # keyed — a stale graph under a flipped knob is silent corruption.
           bool(FG_HYP_RELAX), bool(FG_HYP_EUCLID))
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
    jit_params = opt.params

    print(f"[JIT] compile fg step: task={task} S={S} N={N} "
          f"T={spec.n_factor_types} inlet={has_inlet} K={K} B={B} cw={cw} aw={aw} "
          f"ortho={olam} clip={gc_val} ls={ls} stoch_depth_p={sd_p}...", flush=True)

    @TinyJit
    def _step(input_cells: Tensor, gold: Tensor, cell_valid: Tensor,
              value_domain_mask: Tensor, membership: Tensor, latent_type: Tensor,
              factor_inlet: Tensor, stoch_keep: Tensor,
              inlet_op: Tensor, inlet_target: Tensor, inlet_size: Tensor,
              head_type_oh: Tensor, head_is_global: Tensor):
        opt.zero_grad()

        # Lightweight batch shim so the engine reads per-instance tensors by attr.
        # All tensor attrs are JIT inputs (re-traced each replay).
        class _B:
            pass
        batch = _B()
        batch.input_cells = input_cells
        batch.gold = gold
        batch.cell_valid = cell_valid
        batch.value_domain_mask = value_domain_mask
        batch.membership = membership
        batch.latent_type = latent_type

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
        else:
            batch.factor_inlet = factor_inlet if has_inlet else None
            # Single-domain: do NOT set head_type_oh/head_is_global -> the engine takes
            # the original build_factor_attn_bias path (byte-identical).

        # Forward: K constant → loop unrolls → static graph topology.
        logits_history, calib_history = factor_breathing_forward(
            model, batch, spec, K=K,
            stoch_keep=(stoch_keep if use_stoch else None))

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
        # kenken_constraint_energy; coloring passes None (CE + calib only).
        if use_energy:
            final_probs = logits_history[-1].softmax(axis=-1)
            energy = constraint_energy_fn(final_probs, batch).mean()
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

        total = cell_loss + cw * energy + aw * calib_loss

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
        for p in jit_params:
            if p.grad is None:
                # A param untouched by THIS step's graph (e.g. an inlet sub-table whose
                # value slot no domain in the batch indexed) gets no grad; AdamW.step()
                # requires a grad on every param. Fill with zeros (a no-op update). This
                # is multitask-only in effect (single-domain touches all its params).
                if multitask:
                    p.grad = Tensor.zeros_like(p)
            else:
                p.grad = healthy_b.where(p.grad, Tensor.zeros_like(p.grad))

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
        inlet = build_verification_inlet(
            model, kb.cage_op, kb.cage_target, kb.cage_size, kb.cell_cage_id)
        return make_kenken_factor_batch(kb, spec, prebuilt_inlet=inlet)

    return _Task(
        spec=spec, train_loader=train_loader, test_loader=test_loader,
        to_factor_batch=to_factor_batch,
        constraint_energy_fn=kenken_constraint_energy,
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

    return _Task(
        spec=spec,
        train_loader=loader,
        test_loader=loader,   # same object; eval_iter uses loader.iter_eval()
        to_factor_batch=to_factor_batch,
        constraint_energy_fn=None,
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
    return (fb.input_cells, fb.gold, fb.cell_valid, fb.value_domain_mask,
            fb.membership, fb.latent_type, inlet, stoch_keep,
            inlet_op, inlet_target, inlet_size,
            head_type_oh, head_is_global)


# ---------------------------------------------------------------------------
# Evaluation (eager forward; per-breath CE is eval-only, OUT of the JIT step).
# ---------------------------------------------------------------------------

def evaluate(task: _Task, K: int, max_batches: int) -> dict:
    Tensor.training = False
    spec = task.spec
    cell_eq_sum = 0.0
    n_cells = 0
    puzzle_eq_sum = 0
    n_puzzles = 0
    pb_ce_first = None

    n_batches = 0
    for native in task.eval_iter():
        fb = task.to_factor_batch(native)
        logits_history, _ = factor_breathing_forward(model, fb, spec, K=K)
        final_logits = logits_history[-1]

        cell_valid_np = fb.cell_valid.realize().numpy()                     # (B,S)
        gold_np = fb.gold.realize().numpy().astype(np.int32)                # (B,S)
        pred_np = (final_logits.argmax(axis=-1) + 1).realize().numpy().astype(np.int32)
        eq_np = ((pred_np == gold_np).astype(np.float32) * cell_valid_np)   # (B,S)

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
    return {
        "cell_acc": cell_eq_sum / max(n_cells, 1),
        "puzzle_acc": puzzle_eq_sum / max(n_puzzles, 1),
        "n_puzzles": n_puzzles,
        "per_breath_ce": pb_ce_first or [],
    }


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
    print(f"  test: cell_acc={res['cell_acc']:.3f} "
          f"puzzle_acc={res['puzzle_acc']:.3f} n={res['n_puzzles']}", flush=True)
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
    assert TASK in ("kenken", "coloring", "circuit", "multi"), \
        f"FG_TASK must be kenken|coloring|circuit|multi, got {TASK!r}"
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
    if TASK == "kenken":
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

    # ---- backbone (Pythia-410M L0-L3). hidden read from cfg → no 1024 hardcode.
    cfg = Config()
    print(f"loading Pythia-410M -> breathing transformer (PYTHIA_INIT={PYTHIA_INIT})...")
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
    elif TASK == "coloring":
        task = _build_coloring_task(K, BATCH, EVAL_BATCH, SEED, n_heads)
    elif TASK == "circuit":
        task = _build_circuit_task(K, BATCH, EVAL_BATCH, SEED, n_heads)
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

    # The GENERAL factor-graph params (fg_state_embed / fg_position_embed /
    # fg_value_codebook / fg_calib_head / fg_breath_embed / fg_delta_gate).
    attach_factor_graph_params(model, hidden=hidden, spec=spec)

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
        load_ckpt(model, RESUME_FROM)
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
    # Per-domain loss/acc tracking (multi-task only; CPU-side floats accumulated per
    # LOG_EVERY window). Keeps the JIT step domain-agnostic — domain attribution is
    # external (each batch is pure single-domain, so its scalars belong to one domain).
    mt_dom_acc = {d: {"loss": 0.0, "cell_acc": 0.0, "n": 0} for d in (task.mix or [])}

    for step in range(1, STEPS + 1):
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
        outs = step_fn(*ins)

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
            print(f"  per_breath_ce[B0..B{K-1}]:    {pb_ce_str}  "
                  f"(train cell_acc={float(cell_acc_t.numpy()):.3f} "
                  f"puzzle_acc={float(puzzle_acc_t.numpy()):.3f}{ortho_str})",
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
