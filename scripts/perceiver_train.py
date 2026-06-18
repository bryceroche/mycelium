"""Perceiver-Poincaré BRICK-1 trainer — the make-or-break anchored-perceiver test.

Sibling of scripts/kenken_train.py. REUSES the validated harness (Pythia-410M
L0-L3 loader, fp32 weight cast, the v200 where()-gated NaN grad-guard, the JIT
cache key discipline, AdamW + grad-clip + tangent-norm rim guard) and the
validated machinery (mycelium/perceiver_poincare.py forward/anchor/engagement;
mycelium/kenken.py loss ladder + convergence_instrument + accuracy;
mycelium/perceiver_poincare_data.py for the curriculum batches + constraint
membership). NO waist, NO notebook, NO MCTS (bricks 3-4).

THE BRICK-1 DELIVERABLES (this script):
  (a) compiles on the AM driver (d_hyp cross-attn READ/WRITE + Pythia THINK + readout)
  (b) t=0 ANCHOR-CHECK: does the anchored routing reproduce the factor-graph
      membership? Runs BOTH ball-paths (single + per_constraint), reports the
      membership match, SELECTS the path (single if it matches, else per_constraint)
  (c) ~40-60 training steps: loss descends OFF CHANCE (cell_acc > 1/N)
  (d) ENGAGEMENT (THE KILL METRIC): READ/WRITE select_norm + entropy logged per
      step; flags if it flatlines toward 0 (dead) or stays alive (cured)
  (e) no NaN, grads finite (the where()-gated guard + tangent rim guard)

Env vars:
  PERCEIVER_TASK=1
  PERCEIVER_K_MAX=20
  PERCEIVER_BALL_PATH=auto    auto => run the t=0 check + select; or force single|per_constraint
  PERCEIVER_TAU / PERCEIVER_RHO / PERCEIVER_DIM / PERCEIVER_N_GLOBAL (see module)
  BATCH=8  STEPS=60  LR=3e-5  SEED=42
  RUN_NAME=perceiver_brick1_smoke
  LOG_EVERY=5  EVAL_EVERY=0 (0=off in smoke)  GRAD_CLIP=1.0  MAX_ZNORM=0.9
  KENKEN_TRAIN/KENKEN_TEST (the corpus)
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
from tinygrad.nn.state import safe_save

from mycelium import Config, BreathingTransformer
from mycelium.loader import _load_state, load_breathing
from mycelium.kenken_data import N_CELLS, N_MAX, load_jsonl
from mycelium.kenken import kenken_constraint_energy, kenken_accuracy, convergence_instrument
from mycelium.perceiver_poincare import (
    PERCEIVER_K_MAX, PERCEIVER_TAU, PERCEIVER_RHO, PERCEIVER_DIM,
    PERCEIVER_N_GLOBAL,
    PERCEIVER_HOIST_BIAS, PERCEIVER_FP16_THINK, PERCEIVER_DEFUSE_BREATH,
    PERCEIVER_FAST_GRADNORM, PERCEIVER_THINK_RENORM,
    PERCEIVER_NOTEBOOK, PERCEIVER_PI_ROPE,
    PERCEIVER_PI_ROPE_QK, PERCEIVER_PI_ROPE_ANGLE_SCALE, PERCEIVER_PI_ROPE_PERHEAD,
    PERCEIVER_SILHOUETTE, PERCEIVER_SIL_DIM, PERCEIVER_NB_PIROPE,
    PERCEIVER_SHARP_REG, PERCEIVER_SHARP_REG_LAMBDA,
    PERCEIVER_FREEZE_ROUTING,
    attach_perceiver_params, perceiver_parameters, perceiver_deduction_parameters,
    perceiver_state_dict,
    perceiver_breathing_forward, t0_anchor_check, clamp_perceiver_tangent_norms,
    perceiver_gphi_parameters, perceiver_active_cell_coords,
    gphi_param_snapshot, gphi_drift, latent_coords,
)
from mycelium.perceiver_poincare_data import PerceiverLoader, latent_capacity


# ---- fp32 cast (mirror kenken_train.cast_layers_fp32) ------------------------

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


def collect_params(model, ball_path: str = "single") -> list[Tensor]:
    """Trainable: shared L0-L3 attn/FFN (the THINK layers) + final LN + perceiver.

    When PERCEIVER_FREEZE_ROUTING=0 (default/brick-2): perceiver_parameters includes
    g_phi + active-path cell coords + deduction params (the full unfrozen set).
    When PERCEIVER_FREEZE_ROUTING=1: perceiver_deduction_parameters EXCLUDES g_phi +
    cell coords so the routing geometry is frozen at the t=0 anchor. The THINK layers
    and final LN are always trained (they are never routing)."""
    params: list[Tensor] = []
    sw = model.block.shared
    params += [sw.wv, sw.bv, sw.wo, sw.bo, sw.w_out, sw.b_out,
               sw.in_ln_g, sw.in_ln_b, sw.post_ln_g, sw.post_ln_b]
    for layer in model.block.layers:
        params += [layer.wq, layer.bq, layer.wk, layer.bk, layer.w_in, layer.b_in]
    params += [model.ln_f_g, model.ln_f_b]
    if PERCEIVER_FREEZE_ROUTING:
        params += perceiver_deduction_parameters(model, ball_path=ball_path)
    else:
        params += perceiver_parameters(model, ball_path=ball_path)
    return params


def model_state_dict(model) -> dict:
    sd = {"ln_f.g": model.ln_f_g, "ln_f.b": model.ln_f_b}
    sw = model.block.shared
    for a in ("wv", "bv", "wo", "bo", "w_out", "b_out",
              "in_ln_g", "in_ln_b", "post_ln_g", "post_ln_b"):
        sd[f"shared.{a}"] = getattr(sw, a)
    for i, layer in enumerate(model.block.layers):
        for a in ("wq", "bq", "wk", "bk", "w_in", "b_in"):
            sd[f"phase{i}.{a}"] = getattr(layer, a)
    sd.update(perceiver_state_dict(model))
    return sd


# ---- JIT train step — v200 where()-gated grad-guard --------------------------

_JIT_CACHE: dict = {}


def _compile_step(model, opt, K: int, B: int, L: int, ball_path: str,
                  constraint_weight: float, calib_weight: float,
                  grad_clip: float = 0.0):
    """Compile the TinyJit train step. JIT cache key includes EVERY shape/runtime
    constant: K, B, L (the latent count — the perceiver's new shape axis),
    ball_path (single vs per_constraint => different graph body), the loss weights,
    grad_clip. The d_hyp cross-attn READ/WRITE + Pythia THINK + readout all live
    inside (re-trace with inputs). Engagement scalars (the kill metric) are
    appended to the return so they cost one .numpy() per LOG_EVERY, no per-step sync.

    BRICK-2: the return also carries the per-field UNIFORM FLOORs (floor_read,
    floor_write, read_max_floor, write_max_floor — the recalibrated kill metric's
    discriminator) and the g_phi grad norm + max|z| (the now-unfrozen routing's
    liveness probes). All computed in-graph from the validity masks + grads.
    """
    # PERF FIX A/B/C toggles in the cache key: each (HOIST_BIAS, FP16_THINK,
    # DEFUSE_BREATH) config builds a structurally DIFFERENT graph (hoisted vs
    # in-loop bias; fp32 vs fp16 THINK + renorm; fused vs .contiguous()-fragmented
    # breath backward), so they must each compile their own graph — no silent
    # retrace. DEFUSE_BREATH changes ONLY fusion grouping (byte-identical values),
    # but the realize barriers produce a different kernel graph, so it must key.
    # THINK_RENORM adds an fp32 RMSNorm op at the THINK seam (different graph body
    # vs the no-renorm fp32 path), so it must key too — otherwise an =1 run would
    # silently reuse the =0 (no-renorm) graph.
    # PERCEIVER_NOTEBOOK adds the READ/WRITE notebook cross-attn ops (+ the K-slot
    # accumulate carry) to the breath body; PERCEIVER_PI_ROPE adds the per-breath
    # rotation in the THINK attention. Each builds a structurally DIFFERENT graph
    # (extra ops / extra closed-over params), so both MUST key — an =1 run must not
    # silently reuse the =0 graph.
    # The three PI_ROPE sub-knobs each change the rotation's GRAPH BODY and must key:
    #   PI_ROPE_QK adds the K-rotation ops; PI_ROPE_PERHEAD swaps scalar cos/sin for
    #   per-head cos/sin Tensors (a different op + closed-over const); ANGLE_SCALE
    #   bakes a different compile-time angle constant per breath (1.0 default is the
    #   current angle bit-for-bit). ANGLE_SCALE is keyed as a float so distinct
    #   scales compile distinct graphs (and 1.0 reuses the current graph).
    # PERCEIVER_SILHOUETTE adds the W_sil side-channel projection + conditionally
    #   replaces _nb_write with _nb_write_sil in the breath body (structurally
    #   different graph body: different closed-over params + different source dim).
    #   SIL_DIM bakes as a compile-time constant in the rotation reshape; keyed as
    #   an int so distinct dims compile distinct graphs.
    # PERCEIVER_NB_PIROPE adds _pi_rope_rotate_q calls in the notebook READ (Q-only)
    #   and on the write source, each with a per-breath compile-time angle constant
    #   (structurally different graph body from the un-rotated notebook path).
    # PERCEIVER_SHARP_REG adds the entropy-reg graph into the breath loop (different
    # graph body vs the no-reg path: extra log_softmax ops + entropy reductions +
    # accumulation into sharp_reg_sum). It MUST key so =1 does not silently reuse the
    # =0 graph. PERCEIVER_SHARP_REG_LAMBDA bakes as a compile-time Python float
    # constant inside the JIT (_srl in perceiver_breathing_forward) — a different
    # lambda bakes a different scale constant, so it also keys (same discipline as
    # PERCEIVER_PI_ROPE_ANGLE_SCALE). lambda=0.0 + SHARP_REG=1 still keys uniquely
    # (the entropy graph is still built; the trainer skips adding to total via the
    # Python-level gate `if PERCEIVER_SHARP_REG and _srl > 0`).
    # PERCEIVER_FREEZE_ROUTING=1: the optimizer is built over the DEDUCTION-ONLY param
    # set (g_phi + cell coords excluded). opt.step() calls Adam update ops ONLY for
    # the params in opt.params — a smaller set -> a different graph body than the
    # full brick-2 set. MUST key so =0 and =1 never silently share a compiled graph.
    # When =0 the key appends False -> the =0 key is distinct from any future =1 key.
    key = (id(model), id(opt), int(K), int(B), int(L), str(ball_path),
           float(constraint_weight), float(grad_clip),
           bool(PERCEIVER_HOIST_BIAS), bool(PERCEIVER_FP16_THINK),
           bool(PERCEIVER_DEFUSE_BREATH), bool(PERCEIVER_FAST_GRADNORM),
           bool(PERCEIVER_THINK_RENORM),
           bool(PERCEIVER_NOTEBOOK), bool(PERCEIVER_PI_ROPE),
           bool(PERCEIVER_PI_ROPE_QK), float(PERCEIVER_PI_ROPE_ANGLE_SCALE),
           bool(PERCEIVER_PI_ROPE_PERHEAD),
           bool(PERCEIVER_SILHOUETTE), int(PERCEIVER_SIL_DIM),
           bool(PERCEIVER_NB_PIROPE),
           bool(PERCEIVER_SHARP_REG), float(PERCEIVER_SHARP_REG_LAMBDA),
           bool(PERCEIVER_FREEZE_ROUTING))
    # PERCEIVER_FREEZE_ROUTING: when =1 opt.params is the DEDUCTION-ONLY set (g_phi
    # + cell coords excluded). opt.step() operates on a smaller param list -> a
    # different graph body than the full brick-2 set. MUST key so =0 and =1 compile
    # separate graphs and never silently reuse each other. When =0 the key appends
    # False -> the brick-2 key is extended by one False but is otherwise identical
    # to the pre-FREEZE_ROUTING key (no false-cache-miss against old compiled graphs).
    if key in _JIT_CACHE:
        return _JIT_CACHE[key]

    cw = float(constraint_weight)
    gc_val = float(grad_clip)
    jit_params = opt.params
    # BRICK-2: the now-UNFROZEN routing tensors, closed over for the in-graph grad
    # liveness probe (g_phi encoder) + the rim-guard headroom probe (active coords).
    gphi_param_list = perceiver_gphi_parameters(model)
    active_coord_list = perceiver_active_cell_coords(model, ball_path)
    print(f"[JIT] compile perceiver step: K={K} B={B} L={L} ball_path={ball_path} "
          f"cw={cw} clip={gc_val} ...", flush=True)

    @TinyJit
    def _step(input_cells: Tensor, gold: Tensor, cell_valid: Tensor,
              value_domain_mask: Tensor, latent_membership: Tensor,
              latent_valid: Tensor, latent_type: Tensor):
        opt.zero_grad()

        class _B:
            pass
        batch = _B()
        batch.input_cells = input_cells
        batch.gold = gold
        batch.cell_valid = cell_valid
        batch.value_domain_mask = value_domain_mask
        batch.latent_membership = latent_membership
        batch.latent_valid = latent_valid
        batch.latent_type = latent_type

        cell_logits_history, eng_history, sharp_reg = perceiver_breathing_forward(
            model, batch, K=K, ball_path=ball_path, collect_engagement=True)

        # ---- per-breath weighted CE ladder (REUSE the kenken supervision form):
        # supervise VALID & UNOBSERVED cells, value-domain masked. ----
        observed = (input_cells > 0).cast(dtypes.float)                 # (B,49)
        supervise = cell_valid * (1.0 - observed)                      # (B,49)
        sup_sum = supervise.sum() + 1e-6
        gold_idx = (gold - 1).clip(0, N_MAX - 1).reshape(B * N_CELLS)
        supervise_flat = supervise.reshape(B * N_CELLS)

        cell_loss_sum = Tensor.zeros((), dtype=dtypes.float).contiguous()
        weight_sum = 0.0
        per_breath_ce = []
        for k, logits in enumerate(cell_logits_history):
            weight_k = 1.0 + float(k) / float(K - 1) if K > 1 else 1.0
            ce_elems = logits.reshape(B * N_CELLS, N_MAX).sparse_categorical_crossentropy(
                gold_idx, reduction="none")
            ce_k = (ce_elems * supervise_flat).sum() / sup_sum
            per_breath_ce.append(ce_k)
            cell_loss_sum = cell_loss_sum + ce_k * weight_k
            weight_sum += weight_k
        cell_loss = cell_loss_sum / float(weight_sum)

        final_probs = cell_logits_history[-1].softmax(axis=-1)
        energy = kenken_constraint_energy(final_probs, batch).mean()

        # train accuracy (detached) over valid cells.
        final_argmax = (cell_logits_history[-1].argmax(axis=-1) + 1).detach()
        eq = (final_argmax == gold).cast(dtypes.float)                  # (B,49)
        eq_v = eq * cell_valid
        n_valid = cell_valid.sum() + 1e-6
        train_cell_acc = (eq_v.sum() / n_valid).detach()
        eq_valid = eq * cell_valid + (1.0 - cell_valid)
        train_puzzle_acc = eq_valid.prod(axis=-1).mean().detach()

        total = cell_loss + cw * energy
        # SHARPNESS REG: add lambda * (mean_read_entropy + mean_write_entropy) to
        # the training loss BEFORE backward so the gradient flows to z_latent /
        # z_cell / g_phi / the coords (the whole point — push the routing sharp).
        # sharp_reg is the per-breath mean of (H_read + H_write) from the forward;
        # lambda (_srl_val) bakes as a compile-time Python float constant (substrate-
        # legal; same pattern as the constraint_weight cw above). Gate on both
        # toggles at Python level so default-off is byte-identical (no extra add op
        # when off; sharp_reg is Tensor.zeros(()) and never participates in the graph
        # when PERCEIVER_SHARP_REG=0; the gate here ensures it does not modify total
        # even when SHARP_REG=1+lambda=0 is used as a diagnostic-only mode).
        _srl_val = float(PERCEIVER_SHARP_REG_LAMBDA)
        if PERCEIVER_SHARP_REG and _srl_val > 0:
            total = total + _srl_val * sharp_reg
        total.backward()

        # ---- NaN guard — where()-gated SELECT (never multiply-gate) ----
        healthy_b = total.isfinite()
        healthy = healthy_b.cast(dtypes.float)
        for p in jit_params:
            if p.grad is not None:
                p.grad = healthy_b.where(p.grad, Tensor.zeros_like(p.grad))

        # ---- ENGAGEMENT (THE KILL METRIC): mean over breaths of the READ/WRITE
        # select_norm + entropy + max-attn. Computed inside the graph, returned. ----
        def _avg(keyname):
            s = Tensor.zeros((), dtype=dtypes.float).contiguous()
            for e in eng_history:
                s = s + e[keyname]
            return s / float(len(eng_history))
        eng_read_sel = _avg("read_select_norm")
        eng_read_ent = _avg("read_entropy")
        eng_read_max = _avg("read_max")
        eng_write_sel = _avg("write_select_norm")
        eng_write_ent = _avg("write_entropy")
        eng_write_max = _avg("write_max")

        # ---- BRICK-2 KILL-METRIC RECALIBRATION: the per-field UNIFORM FLOOR.
        # select_norm = mean L2 norm of a softmax row; a DEAD-FLAT uniform routing
        # over n cells/latents floors at 1/sqrt(n) (NOT 0) — so the brick-1
        # "ALIVE if >1e-3" verdict is non-discriminating. The discriminating floor:
        #   floor_read  = 1/sqrt(mean # valid cells per puzzle)   (read softmax is
        #                 over ALL valid cells, so the uniform L2 is 1/sqrt(n_cells))
        #   floor_write = 1/sqrt(mean # valid latents per puzzle)
        # ALIVE iff select_norm CLEARLY above the floor; DEAD iff within ~10% of it.
        # Computed in-graph from the validity masks (no host branch), returned.
        n_cells_mean = cell_valid.cast(dtypes.float).sum(axis=-1).mean()       # mean valid cells/puzzle
        n_lat_mean = latent_valid.cast(dtypes.float).sum(axis=-1).mean()       # mean valid latents/puzzle
        floor_read = 1.0 / (n_cells_mean + 1e-6).sqrt()
        floor_write = 1.0 / (n_lat_mean + 1e-6).sqrt()
        # also the 1/S uniform-max reference (a flat row's max-attn = 1/n).
        read_max_floor = 1.0 / (n_cells_mean + 1e-6)
        write_max_floor = 1.0 / (n_lat_mean + 1e-6)

        # ---- BRICK-2 g_phi DRIFT/LIVENESS: the g_phi GRAD NORM (the now-UNFROZEN
        # routing's gradient pull). Non-zero => g_phi co-adapts the deduction. ----
        gphi_grads = [p.grad for p in gphi_param_list if p.grad is not None]
        if gphi_grads:
            gphi_sq = Tensor.zeros((), dtype=dtypes.float).contiguous()
            for g in gphi_grads:
                gphi_sq = gphi_sq + g.cast(dtypes.float).square().sum()
            gphi_grad_norm = (gphi_sq + 1e-12).sqrt()
        else:
            gphi_grad_norm = Tensor.zeros((), dtype=dtypes.float).contiguous()
        # max |z| over the active cell coords (the rim-guard headroom probe).
        max_z = Tensor.zeros((), dtype=dtypes.float).contiguous()
        for t in active_coord_list:
            tn = t.cast(dtypes.float).pow(2).sum(axis=-1).sqrt()  # |v| per row
            zr = tn.tanh().max()                                  # |z|=tanh(|v|)
            max_z = max_z.maximum(zr)

        # BRICK-2 OBSERVABILITY (purely additive): max |z| over the g_phi-bearing
        # LATENT ball points. max_z above watches only the CELL coords, so a latent
        # pushed toward the rim by g_phi (to maximize orthogonal capacity) would be
        # INVISIBLE. latent_coords = _exp0_map(base + corr): base = segment-mean of
        # the constraint's cell tangents, corr = the (zero-init) g_phi DeepSets
        # correction, mapped tangent->ball by exp_0. These POST-exp0 BALL points
        # carry corr, so their L2 norm is the true latent ball-norm (NOT tanh(|v|):
        # they are already ball coords; NOT the tangent `tan`; NOT base-only; NOT
        # cells). Computed AFTER backward() as a metrics-only subgraph (no grad,
        # no RNG, single-kernel reduction, no float32 literal) -> does not touch
        # loss/grads/max_z. |z|<1 by construction (exp_0 tanh) but -> rim possible.
        z_latent_probe = latent_coords(model, latent_membership, ball_path,
                                       latent_type)               # (B,L,dim) ball
        max_latent_z = z_latent_probe.cast(dtypes.float).pow(2).sum(axis=-1).sqrt().max()

        # optional global-norm grad clip (single sq_sum kernel).
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
        else:
            sq_sum = Tensor.zeros((), dtype=dtypes.float).contiguous()
            for p in jit_params:
                if p.grad is not None:
                    sq_sum = sq_sum + p.grad.cast(dtypes.float).square().sum()
            grad_norm = (sq_sum + 1e-12).sqrt()

        opt.step()

        # PERF FIX D (PERCEIVER_FAST_GRADNORM): the in-graph grad-clip ABOVE is
        # byte-identical either way (grad_norm is still computed and STILL drives
        # clip_coef -> the grads are scaled identically). The ONLY difference is
        # whether grad_norm is MATERIALISED as a returned scalar. When =1 it is
        # OMITTED from the return tuple (so the all-param sq_sum stays consumed
        # in-graph for the clip and never becomes the standalone occ-0 mega-kernel).
        # gphi_grad_norm (small ~0.1M-param reduction, the brick-2 watch-item) is
        # KEPT in both modes.
        head = (
            total.realize(), healthy.realize(), cell_loss.realize(),
            energy.realize(), train_cell_acc.realize(), train_puzzle_acc.realize(),
        )
        gn = () if PERCEIVER_FAST_GRADNORM else (grad_norm.realize(),)
        # sharp_reg: the per-breath mean of (coupled_read_reg + blanket_write_reg).
        # coupled_read_reg = (1/n_valid) * sum_i( H_read_i * |z_latent_i|^2 ) per breath.
        # blanket_write_reg = mean_cell H_write per breath.  Logged to trajectory JSONL
        # so the magnitude of the coupled term can be tracked across the run.
        # When PERCEIVER_SHARP_REG=0 this is Tensor.zeros(()) — safe to realize always.
        tail = (
            eng_read_sel.realize(), eng_read_ent.realize(), eng_read_max.realize(),
            eng_write_sel.realize(), eng_write_ent.realize(), eng_write_max.realize(),
            floor_read.realize(), floor_write.realize(),
            read_max_floor.realize(), write_max_floor.realize(),
            gphi_grad_norm.realize(), max_z.realize(), max_latent_z.realize(),
            sharp_reg.realize(),
            *(ce.realize() for ce in per_breath_ce),
        )
        return head + gn + tail

    _JIT_CACHE[key] = _step
    print(f"[JIT] perceiver step ready; first call compiles (~60-90s)…", flush=True)
    return _step


def main():
    assert int(getenv("PERCEIVER_TASK", 0)) > 0, "PERCEIVER_TASK=1 must be set"

    K = int(getenv("PERCEIVER_K_MAX", str(PERCEIVER_K_MAX)))
    BATCH = int(getenv("BATCH", 8))
    STEPS = int(getenv("STEPS", 60))
    LR = float(getenv("LR", "3e-5"))
    SEED = int(getenv("SEED", 42))
    LOG_EVERY = int(getenv("LOG_EVERY", 5))
    EVAL_EVERY = int(getenv("EVAL_EVERY", 0))
    EVAL_BATCHES = int(getenv("EVAL_BATCHES", 5))
    GC_EVERY = int(getenv("GC_EVERY", 20))
    GRAD_CLIP = float(getenv("GRAD_CLIP", "1.0"))
    MAX_ZNORM = float(getenv("MAX_ZNORM", "0.9"))
    RUN_NAME = getenv("RUN_NAME", "perceiver_brick1_smoke")
    BALL_PATH_ENV = getenv("PERCEIVER_BALL_PATH", "auto").strip().lower()
    CONSTRAINT_WEIGHT = float(getenv("PERCEIVER_CONSTRAINT_WEIGHT", "0.3"))
    CALIB_WEIGHT = float(getenv("PERCEIVER_CALIB_WEIGHT", "0.0"))

    KENKEN_TRAIN = getenv("KENKEN_TRAIN", ".cache/kenken_train.jsonl")
    KENKEN_TEST = getenv("KENKEN_TEST", ".cache/kenken_test.jsonl")

    _mode_tag = ("FREEZE-ROUTING (routing FROZEN at anchor, deduction trains)"
                 if PERCEIVER_FREEZE_ROUTING
                 else "BRICK-2 (deduction test: g_phi UNFROZEN, routing co-adapts)")
    print(f"=== Perceiver-Poincaré {_mode_tag} ===")
    print(f"device={Device.DEFAULT}  B={BATCH}  K={K}  steps={STEPS}  lr={LR}")
    print(f"tau={PERCEIVER_TAU}  rho={PERCEIVER_RHO}  dim={PERCEIVER_DIM}  "
          f"n_global={PERCEIVER_N_GLOBAL}  ball_path={BALL_PATH_ENV}")
    print(f"grad_clip={GRAD_CLIP}  max_znorm={MAX_ZNORM}  "
          f"constraint_weight={CONSTRAINT_WEIGHT}")
    print(f"train={KENKEN_TRAIN}  test={KENKEN_TEST}")
    print()

    np.random.seed(SEED)
    Tensor.manual_seed(SEED)

    # n_cages_max over BOTH corpora (stable JIT shape).
    train_recs = load_jsonl(KENKEN_TRAIN)
    test_recs = load_jsonl(KENKEN_TEST)
    corpus_n_cages_max = max(
        max(len(r["cages"]) for r in train_recs),
        max(len(r["cages"]) for r in test_recs),
    )
    N_CAGES_MAX = int(getenv("KENKEN_N_CAGES_MAX", str(corpus_n_cages_max)))
    L_MAX = latent_capacity(N_CAGES_MAX, PERCEIVER_N_GLOBAL)
    print(f"n_cages_max={N_CAGES_MAX}  -> L_max (latents) = "
          f"{2*N_MAX} row/col + {N_CAGES_MAX} cage + {PERCEIVER_N_GLOBAL} global = {L_MAX}")

    cfg = Config()
    print(f"loading Pythia-410M -> breathing transformer (THINK layers)...")
    sd = _load_state()
    model = load_breathing(cfg, sd=sd)
    del sd
    cast_layers_fp32(model)
    attach_perceiver_params(model, hidden=cfg.hidden, n_heads=cfg.n_heads,
                            L_max=L_MAX, k_max=K)
    Device[Device.DEFAULT].synchronize()

    train_loader = PerceiverLoader(KENKEN_TRAIN, batch_size=BATCH, seed=SEED,
                                   n_cages_max=N_CAGES_MAX)
    test_loader = PerceiverLoader(KENKEN_TEST, batch_size=BATCH, seed=SEED + 1,
                                  n_cages_max=N_CAGES_MAX)

    # ---- (b) t=0 ANCHOR-CHECK: run BOTH ball-paths on a fresh batch, report the
    # membership match, SELECT the path. NO training. ----
    Tensor.training = False
    probe = train_loader.sample_batch()
    print("\n=== t=0 ANCHOR-CHECK (does anchored routing == factor-graph membership?) ===")
    chk_single = t0_anchor_check(model, probe, "single")
    chk_per = t0_anchor_check(model, probe, "per_constraint")
    for nm, chk in (("single", chk_single), ("per_constraint", chk_per)):
        print(f"  [{nm:>14}] n_constraint_latents={chk['n_constraint_latents']:4d}  "
              f"topk_recall={chk['topk_recall']:.3f}  "
              f"membership_match={chk['membership_match']:.3f}  "
              f"mean_in_attn={chk['mean_in_attn']:.3f}", flush=True)
    # Selection: single if its membership_match is high enough (>= threshold), else
    # fall back to per_constraint (the Tier-2 proven path).
    MATCH_THRESH = float(getenv("PERCEIVER_T0_MATCH_THRESH", "0.95"))
    if BALL_PATH_ENV in ("single", "per_constraint"):
        ball_path = BALL_PATH_ENV
        reason = f"FORCED by env PERCEIVER_BALL_PATH={BALL_PATH_ENV}"
    elif chk_single["membership_match"] >= MATCH_THRESH:
        ball_path = "single"
        reason = (f"single membership_match {chk_single['membership_match']:.3f} "
                  f">= {MATCH_THRESH} -> single unified ball holds")
    else:
        ball_path = "per_constraint"
        reason = (f"single membership_match {chk_single['membership_match']:.3f} "
                  f"< {MATCH_THRESH} -> FALL BACK to per-constraint routing")
    print(f"  SELECTED ball_path = {ball_path}  ({reason})", flush=True)
    selected_chk = chk_single if ball_path == "single" else chk_per
    print(f"  selected-path t=0 membership_match = {selected_chk['membership_match']:.3f} "
          f"(topk_recall {selected_chk['topk_recall']:.3f})", flush=True)
    Tensor.training = True

    # ---- build the optimizer over ONLY the active ball-path's params (the inactive
    # cell field has no grad path -> would trip AdamW's grad-is-None assert).
    # When PERCEIVER_FREEZE_ROUTING=1, collect_params calls perceiver_deduction_parameters
    # (excludes g_phi + cell coords); when =0 it calls perceiver_parameters (full set). ----
    params = collect_params(model, ball_path=ball_path)
    n_params = sum(int(np.prod(t.shape)) for t in params)
    print(f"  trainable params ({ball_path}): {n_params/1e6:.1f}M")

    # ---- PARAM-SET PARTITION AUDIT (printed in BOTH modes so the partition is always
    # on record; assertion logic differs by mode). ----
    pid = {id(p) for p in params}
    gphi_total = len(perceiver_gphi_parameters(model))
    gphi_in = [p for p in perceiver_gphi_parameters(model) if id(p) in pid]
    coords_in = [p for p in perceiver_active_cell_coords(model, ball_path) if id(p) in pid]
    inactive_coords = ([model.perc_cell_v_row, model.perc_cell_v_col, model.perc_cell_v_cage]
                       if ball_path == "single" else [model.perc_cell_v])
    inactive_in = [p for p in inactive_coords if id(p) in pid]
    think_in = sum(1 for p in (
        [model.block.shared.wv, model.block.shared.bv, model.block.shared.wo,
         model.block.shared.bo, model.block.shared.w_out, model.block.shared.b_out]
        + [getattr(l, a) for l in model.block.layers
           for a in ("wq", "bq", "wk", "bk", "w_in", "b_in")]) if id(p) in pid)
    delta_gate_in = 1 if id(model.perc_delta_gate) in pid else 0

    if PERCEIVER_FREEZE_ROUTING:
        # FROZEN-ROUTING: g_phi and coords MUST be excluded; deduction MUST be present.
        print(f"  FREEZE-ROUTING param-set:"
              f"  THINK={think_in} tensors"
              f"  readout=[value_codebook,state_embed,position_embed,ln_f]"
              f"  delta_gate={delta_gate_in}"
              f"  type_embed={1 if id(model.perc_latent_type_embed) in pid else 0}"
              f"  breath_embed={1 if id(model.perc_breath_embed) in pid else 0}"
              f"  --- EXCLUDED (frozen at anchor): ---"
              f"  g_phi={len(gphi_in)}/{gphi_total} (expect 0)"
              f"  active_coords({ball_path})={len(coords_in)} (expect 0)"
              f"  inactive_coords_present={len(inactive_in)} (expect 0)")
        assert len(gphi_in) == 0, \
            f"g_phi IS in the optimizer under FREEZE_ROUTING=1 ({len(gphi_in)} tensors present — bug)"
        assert len(coords_in) == 0, \
            f"active cell coords ARE in the optimizer under FREEZE_ROUTING=1 ({len(coords_in)} present — bug)"
        assert len(inactive_in) == 0, "inactive cell field is in the optimizer (dead param)"
        assert delta_gate_in == 1, "delta_gate not in optimizer (deduction param must be trained)"
    else:
        # BRICK-2 UNFREEZE CONFIRM: the optimizer trains the FULL anchored perceiver =
        # {THINK (Pythia L0-L3) + readout + delta_gate + g_phi + active-path coords}.
        print(f"  UNFREEZE param-set: THINK={think_in} tensors  readout=[value_codebook,"
              f"state_embed,position_embed,ln_f]  delta_gate={delta_gate_in}"
              f"  g_phi={len(gphi_in)}/{gphi_total} tensors"
              f"  active_coords({ball_path})={len(coords_in)}  inactive_coords_present={len(inactive_in)}")
        assert len(gphi_in) == gphi_total and len(gphi_in) > 0, \
            "g_phi NOT fully in optimizer (brick-2 requires UNFROZEN g_phi)"
        assert len(coords_in) == len(perceiver_active_cell_coords(model, ball_path)), \
            "active-path cell coords NOT in optimizer (brick-2 requires UNFROZEN coords)"
        assert len(inactive_in) == 0, "inactive cell field is in the optimizer (dead param)"
        assert delta_gate_in == 1, "delta_gate not in optimizer"

    opt = AdamW(params, lr=LR, weight_decay=0.0)

    run_dir = os.path.join(".cache/perceiver_ckpts", RUN_NAME)
    os.makedirs(run_dir, exist_ok=True)
    _routing_desc = ("FROZEN at anchor (g_phi+coords excluded from optimizer)"
                     if PERCEIVER_FREEZE_ROUTING else
                     "UNFROZEN (g_phi+coords trained, routing co-adapts)")
    provenance = {
        "arch_version": ("perceiver_poincare_freeze_routing"
                         if PERCEIVER_FREEZE_ROUTING else "perceiver_poincare_brick2"),
        "core": f"perceiver_anchored_deduction (routing {_routing_desc})",
        "base": "Pythia-410M (L0-L3 THINK layers, shared every breath)",
        "K": K, "B": BATCH, "LR": LR, "steps": STEPS, "seed": SEED,
        "L_max": L_MAX, "n_cages_max": N_CAGES_MAX, "n_global": PERCEIVER_N_GLOBAL,
        "tau": PERCEIVER_TAU, "rho": PERCEIVER_RHO, "dim": PERCEIVER_DIM,
        "ball_path": ball_path, "ball_path_reason": reason,
        "t0_anchor_check": {"single": chk_single, "per_constraint": chk_per,
                            "selected": ball_path, "match_thresh": MATCH_THRESH},
        "trainable_params_M": round(n_params / 1e6, 3),
        "constraint_weight": CONSTRAINT_WEIGHT,
        "freeze_routing": bool(PERCEIVER_FREEZE_ROUTING),
        "param_set_desc": (
            "DEDUCTION ONLY: THINK(Pythia L0-L3) + readout + delta_gate + "
            "type_embed + breath_embed (g_phi + cell_coords FROZEN at anchor)"
            if PERCEIVER_FREEZE_ROUTING else
            "FULL: THINK(Pythia L0-L3) + readout + delta_gate + g_phi + "
            f"active-path coords ({ball_path})"
        ),
        "kill_metric": "floor-relative (select_norm vs per-field uniform floor "
                       "1/sqrt(S); ALIVE > 1.3*floor OR max >> 1/S)",
        "trajectory": "trajectory.jsonl (per-LOG_EVERY)",
        "bricks_excluded": "waist (brick3), notebook+MCTS (brick4)",
    }
    with open(os.path.join(run_dir, "measured_config.json"), "w") as f:
        json.dump(provenance, f, indent=2)
    print(f"  wrote provenance: {os.path.join(run_dir, 'measured_config.json')}")

    step_fn = _compile_step(model, opt, K=K, B=BATCH, L=L_MAX, ball_path=ball_path,
                            constraint_weight=CONSTRAINT_WEIGHT,
                            calib_weight=CALIB_WEIGHT, grad_clip=GRAD_CLIP)

    chance = 1.0 / float(N_MAX)
    print(f"\n=== TRAINING ({STEPS} steps; chance cell_acc ~= 1/{N_MAX} = {chance:.3f}) ===")
    print(f"  KILL METRIC (recalibrated): select_norm vs the per-field "
          f"UNIFORM FLOOR 1/sqrt(S). DEAD if within ~10% of floor AND max near 1/S; "
          f"ALIVE only CLEARLY above (sel>1.3*floor OR max>>1/S).")
    if PERCEIVER_FREEZE_ROUTING:
        print(f"  GOAL (FREEZE-ROUTING): does cell_acc CLIMB beyond the brick-2 plateau "
              f"(0.37)? routing FROZEN at anchor -> tests if routing DRIFT was the bug.\n"
              f"  g_phi_grad_norm should be ~0 (excluded from opt; grads exist but unused).\n")
    else:
        print(f"  GOAL (brick-2): does cell_acc CLIMB once g_phi unfreezes (the deduction "
              f"test)? + g_phi drift > 0 (routing co-adapts).\n")
    t0 = time.time()
    # engagement trajectory for the final verdict.
    eng_traj = []
    acc_traj = []
    max_latent_z_traj = []  # latent rim-norm probe (purely additive)
    nan_steps = 0
    first_eng = None
    last_floors = {"read": 0.0, "write": 0.0, "read_max": 0.0, "write_max": 0.0}

    # g_phi DRIFT probe — snapshot the g_phi encoder BEFORE training.
    # FREEZE_ROUTING=0 (brick-2): g_phi is in the optimizer and should MOVE (rho drift > 0).
    # FREEZE_ROUTING=1: g_phi is EXCLUDED from the optimizer; drift should stay ~0 (only
    # gradient accumulation noise — no Adam update applied). A non-zero drift here would
    # mean the freeze is broken.
    gphi_snap0 = gphi_param_snapshot(model)

    # BRICK-2: per-LOG_EVERY trajectory JSONL (auditable artifact, not just stdout).
    traj_path = os.path.join(run_dir, "trajectory.jsonl")
    traj_f = open(traj_path, "w")
    print(f"  trajectory JSONL -> {traj_path}", flush=True)

    # Deferred logging (kenken perf idiom): pack the scalars into ONE on-GPU tensor
    # per step (one .realize() enqueue, NO host block) and a single .numpy() at
    # LOG_EVERY — instead of many GPU->CPU syncs per step (the sync storm that,
    # interleaved with the in-place clamp on the large K-breath graph, hung the AM
    # driver on replay 2). Order: total, healthy, cell_loss, cell_acc, grad_norm,
    # read[sel,ent,max], write[sel,ent,max], floor[read,write,read_max,write_max],
    # gphi_grad_norm, max_z, max_latent_z.
    for step in range(1, STEPS + 1):
        batch = train_loader.sample_batch()
        outs = step_fn(
            batch.input_cells, batch.gold, batch.cell_valid,
            batch.value_domain_mask, batch.latent_membership,
            batch.latent_valid, batch.latent_type,
        )
        # PERF FIX D: under PERCEIVER_FAST_GRADNORM the all-param grad_norm scalar
        # is NOT returned (it stays consumed in-graph for the byte-identical clip).
        # Re-insert a sentinel at its position so the pack/unpack/log indices below
        # stay IDENTICAL in both modes (the in-graph clip is unaffected either way).
        if PERCEIVER_FAST_GRADNORM:
            grad_norm_t = Tensor(float("nan"))
            outs = outs[:6] + (grad_norm_t,) + outs[6:]
        # Output tuple layout (after the FAST_GRADNORM sentinel re-insert):
        # [0-5]  total, healthy, cell_loss, energy, cell_acc, puzzle_acc
        # [6]    grad_norm (or nan sentinel)
        # [7-12] eng: eR_sel, eR_ent, eR_max, eW_sel, eW_ent, eW_max
        # [13-16] floors: floor_read, floor_write, read_max_floor, write_max_floor
        # [17-19] gphi_gn, max_z, max_latent_z
        # [20]   sharp_reg (per-breath mean H_read + H_write; 0.0 when reg is off)
        # [21:]  per_breath_ce
        (total_t, healthy_t, cell_loss_t, energy_t, cell_acc_t, puzzle_acc_t,
         grad_norm_t, eR_sel, eR_ent, eR_max, eW_sel, eW_ent, eW_max,
         floor_read_t, floor_write_t, read_max_floor_t, write_max_floor_t,
         gphi_gn_t, max_z_t, max_latent_z_t, sharp_reg_t) = outs[:21]
        per_breath_ce_t = list(outs[21:])  # already realized in _step; host read only

        # tangent rim guard (Tier-2 §7) AFTER opt.step. BRICK-2 FIX: scoped to the
        # ACTIVE ball_path's cell fields ONLY (the brick-1 caveat: it touched the
        # inactive field too). Gated on MAX_ZNORM>0 (the in-place .assign clamp).
        if MAX_ZNORM > 0:
            clamp_perceiver_tangent_norms(model, max_znorm=MAX_ZNORM,
                                          ball_path=ball_path)

        packed = Tensor.stack(
            total_t.reshape(()), healthy_t.reshape(()), cell_loss_t.reshape(()),
            cell_acc_t.reshape(()), grad_norm_t.reshape(()),
            eR_sel.reshape(()), eR_ent.reshape(()), eR_max.reshape(()),
            eW_sel.reshape(()), eW_ent.reshape(()), eW_max.reshape(()),
            floor_read_t.reshape(()), floor_write_t.reshape(()),
            read_max_floor_t.reshape(()), write_max_floor_t.reshape(()),
            gphi_gn_t.reshape(()), max_z_t.reshape(()), max_latent_z_t.reshape(()),
            sharp_reg_t.reshape(()),
            *(c.reshape(()) for c in per_breath_ce_t),
        ).realize()  # one enqueue, no host block
        # packed layout: [0-4] loss/healthy/ce/acc/gn; [5-10] eng; [11-14] floors;
        # [15-17] gphi_gn/max_z/max_latent_z; [18] sharp_reg; [19:] per_breath_ce

        if step % LOG_EVERY == 0 or step == 1:
            v = packed.numpy()  # the only host sync this step
            loss_v, healthy_v, ce_v, acc_v, gn_v = (float(x) for x in v[:5])
            eng = {
                "read_select_norm": float(v[5]), "read_entropy": float(v[6]),
                "read_max": float(v[7]), "write_select_norm": float(v[8]),
                "write_entropy": float(v[9]), "write_max": float(v[10]),
            }
            floor_read_v, floor_write_v = float(v[11]), float(v[12])
            read_max_floor_v, write_max_floor_v = float(v[13]), float(v[14])
            gphi_gn_v, max_z_v = float(v[15]), float(v[16])
            max_latent_z_v = float(v[17])
            sharp_reg_v = float(v[18])
            per_breath_ce_v = [float(x) for x in v[19:19 + len(per_breath_ce_t)]]
            last_floors = {"read": floor_read_v, "write": floor_write_v,
                           "read_max": read_max_floor_v,
                           "write_max": write_max_floor_v}
            if healthy_v < 0.5:
                nan_steps += 1
            eng_traj.append(eng)
            acc_traj.append(acc_v)
            max_latent_z_traj.append(max_latent_z_v)
            if first_eng is None:
                first_eng = eng
            # floor-relative ALIVE/DEAD (the recalibrated kill metric).
            r_alive = _alive_vs_floor(eng["read_select_norm"], floor_read_v,
                                      eng["read_max"], read_max_floor_v)
            w_alive = _alive_vs_floor(eng["write_select_norm"], floor_write_v,
                                      eng["write_max"], write_max_floor_v)
            dt = time.time() - t0
            gn_str = "n/a(fast)" if PERCEIVER_FAST_GRADNORM else f"{gn_v:.3e}"
            sreg_str = (f"{sharp_reg_v:.4f}" if PERCEIVER_SHARP_REG else "off")
            print(f"[step {step:3d}] loss={loss_v:.4f} cell_ce={ce_v:.4f} "
                  f"cell_acc={acc_v:.3f} grad_norm={gn_str} "
                  f"sharp_reg={sreg_str} "
                  f"gphi_grad={gphi_gn_v:.3e} max_z={max_z_v:.3f} "
                  f"max_latent_z={max_latent_z_v:.3f} "
                  f"({dt:.1f}s, {dt/step:.2f}s/step)", flush=True)
            print(f"          READ  sel={eng['read_select_norm']:.4f} "
                  f"floor={floor_read_v:.4f} max={eng['read_max']:.3f} "
                  f"(1/S={read_max_floor_v:.3f}) -> {r_alive}", flush=True)
            print(f"          WRITE sel={eng['write_select_norm']:.4f} "
                  f"floor={floor_write_v:.4f} max={eng['write_max']:.3f} "
                  f"(1/S={write_max_floor_v:.3f}) -> {w_alive}", flush=True)
            # BRICK-2 trajectory persistence (one JSONL line per LOG_EVERY).
            rec = {
                "step": step, "cell_acc": acc_v, "loss": loss_v,
                "cell_ce": ce_v,
                # PERF FIX D: grad_norm is not measured under FAST_GRADNORM (the
                # clip is byte-identical, only the logged scalar is dropped) -> log
                # None so the JSONL stays strict-JSON-parseable (no bare NaN).
                "grad_norm": (None if PERCEIVER_FAST_GRADNORM else gn_v),
                # sharp_reg: per-breath mean of (coupled_read_reg + blanket_write_reg).
                # coupled_read_reg = mean_i(H_read_i * |z_latent_i|^2) over valid latents.
                # blanket_write_reg = mean_cell(H_write). Non-zero only when REG=1.
                # 0.0 when off (Tensor.zeros(()) / K — always present in JSONL).
                "sharp_reg": sharp_reg_v,
                "read_select_norm": eng["read_select_norm"],
                "write_select_norm": eng["write_select_norm"],
                "read_max": eng["read_max"], "write_max": eng["write_max"],
                "read_entropy": eng["read_entropy"],
                "write_entropy": eng["write_entropy"],
                "floor_read": floor_read_v, "floor_write": floor_write_v,
                "read_max_floor": read_max_floor_v,
                "write_max_floor": write_max_floor_v,
                "read_alive": r_alive, "write_alive": w_alive,
                "gphi_grad_norm": gphi_gn_v, "max_z": max_z_v,
                "max_latent_z": max_latent_z_v,
                "per_breath_ce": per_breath_ce_v,
                "nan": (healthy_v < 0.5),
            }
            traj_f.write(json.dumps(rec) + "\n")
            traj_f.flush()

        if EVAL_EVERY > 0 and step % EVAL_EVERY == 0:
            _quick_eval(model, test_loader, K, ball_path, EVAL_BATCHES)

        if step % GC_EVERY == 0:
            gc.collect()

    traj_f.close()

    # ---- final verdict (floor-relative kill-metric + g_phi drift) ----
    _verdict_tag = "FREEZE-ROUTING VERDICT" if PERCEIVER_FREEZE_ROUTING else "BRICK-2 VERDICT"
    print(f"\n=== {_verdict_tag} ===")
    first_acc = acc_traj[0]
    last_acc = float(np.mean(acc_traj[-min(5, len(acc_traj)):]))
    print(f"  cell_acc: step1={first_acc:.3f} -> last5_mean={last_acc:.3f} "
          f"(chance={chance:.3f})  OFF-CHANCE={'YES' if last_acc > chance + 0.02 else 'NO'}")
    r0 = first_eng["read_select_norm"]
    rN = float(np.mean([e["read_select_norm"] for e in eng_traj[-min(5, len(eng_traj)):]]))
    w0 = first_eng["write_select_norm"]
    wN = float(np.mean([e["write_select_norm"] for e in eng_traj[-min(5, len(eng_traj)):]]))
    rmaxN = float(np.mean([e["read_max"] for e in eng_traj[-min(5, len(eng_traj)):]]))
    wmaxN = float(np.mean([e["write_max"] for e in eng_traj[-min(5, len(eng_traj)):]]))
    fr, fw = last_floors["read"], last_floors["write"]
    frm, fwm = last_floors["read_max"], last_floors["write_max"]
    # the RECALIBRATED kill metric (vs the per-field UNIFORM floor 1/sqrt(S)).
    print(f"  --- RECALIBRATED KILL-METRIC (vs uniform floor 1/sqrt(S)) ---")
    print(f"  READ  select_norm: {r0:.4f} -> {rN:.4f}  floor={fr:.4f} "
          f"(ratio {rN/max(fr,1e-9):.2f}x)  read_max={rmaxN:.3f} (1/S={frm:.3f}) "
          f"-> {_alive_vs_floor(rN, fr, rmaxN, frm)}")
    print(f"  WRITE select_norm: {w0:.4f} -> {wN:.4f}  floor={fw:.4f} "
          f"(ratio {wN/max(fw,1e-9):.2f}x)  write_max={wmaxN:.3f} (1/S={fwm:.3f}) "
          f"-> {_alive_vs_floor(wN, fw, wmaxN, fwm)}")
    print(f"  (DEAD = within 10% of floor AND max near 1/S; ALIVE = sel>1.3*floor "
          f"OR max>>1/S)")
    # g_phi DRIFT probe.
    # FREEZE_ROUTING=0 (brick-2): drift > 0 means routing co-adapts (expected).
    # FREEZE_ROUTING=1: drift should be ~0 (only gradient noise, no Adam update applied).
    #   A large drift under FREEZE_ROUTING=1 would mean the freeze is broken.
    drift = gphi_drift(model, gphi_snap0)
    _drift_label = ("g_phi DRIFT (FREEZE mode: should be ~0 — optimizer excluded)"
                    if PERCEIVER_FREEZE_ROUTING else
                    "g_phi DRIFT (brick-2: unfrozen routing co-adapt; expect >0)")
    print(f"  --- {_drift_label} ---")
    print(f"  drift_total={drift['drift_total']:.4e}  drift_rho(output)="
          f"{drift['drift_rho']:.4e}  -> g_phi {'MOVES (check freeze!)' if (PERCEIVER_FREEZE_ROUTING and drift['drift_total'] > 1e-6) else ('MOVES' if drift['drift_total'] > 0 else 'FROZEN')}")
    # latent rim-norm probe — does g_phi push a LATENT toward the rim (|z|->1)?
    if max_latent_z_traj:
        mlz0 = max_latent_z_traj[0]
        mlzN = float(np.max(max_latent_z_traj[-min(5, len(max_latent_z_traj)):]))
        mlz_all = float(np.max(max_latent_z_traj))
        print(f"  --- LATENT BALL-NORM (latent rim probe) ---")
        print(f"  max_latent_z: step1={mlz0:.3f} -> last5_max={mlzN:.3f}  "
              f"run_max={mlz_all:.3f}  (rim if -> 1.0)")
    if PERCEIVER_FREEZE_ROUTING:
        print(f"  FREEZE-ROUTING: routing geometry HELD at anchor; cell_acc > 0.37 = "
              f"routing drift WAS the brick-2 plateau cause.")
    print(f"  NaN steps: {nan_steps}/{STEPS}  (where()-gated guard)")
    print(f"  selected ball_path: {ball_path}  "
          f"(t=0 membership_match={selected_chk['membership_match']:.3f})")
    print(f"  trajectory: {os.path.join(run_dir, 'trajectory.jsonl')}")

    ckpt_path = os.path.join(run_dir, f"{RUN_NAME}_final.safetensors")
    safe_save(model_state_dict(model), ckpt_path)
    print(f"\ndone. saved {ckpt_path}", flush=True)


def _alive_vs_floor(select_norm: float, floor: float, max_attn: float,
                    max_floor: float) -> str:
    """The BRICK-2 floor-relative ALIVE/DEAD verdict (the make-or-break instrument).

    select_norm floors at 1/sqrt(S) (a dead-flat uniform routing) and CANNOT reach
    0 — so the brick-1 'ALIVE if >1e-3' flag false-positives a dead run. The
    discriminating test:
      DEAD  if select_norm is within ~10% of the floor AND max_attn is near 1/S
             (the routing is indistinguishable from uniform).
      ALIVE if select_norm CLEARLY above the floor (> 1.3*floor) OR max_attn >> 1/S
             (>= 2x the uniform max — the routing PEAKS on specific cells/latents).
    """
    clearly_above = select_norm > 1.3 * floor
    peaked_max = max_attn > 2.0 * max_floor
    near_floor = select_norm <= 1.1 * floor
    flat_max = max_attn <= 1.5 * max_floor
    if clearly_above or peaked_max:
        return "ALIVE"
    if near_floor and flat_max:
        return "DEAD"
    return "MARGINAL"


def _quick_eval(model, loader, K, ball_path, max_batches):
    Tensor.training = False
    cell_eq = 0.0
    n_cells = 0
    nb = 0
    for batch in loader.iter_eval(batch_size=loader.batch_size):
        cell_logits_history, _, _ = perceiver_breathing_forward(
            model, batch, K=K, ball_path=ball_path, collect_engagement=False)
        acc, _ = kenken_accuracy(cell_logits_history[-1], batch)
        cv = batch.cell_valid.realize().numpy()
        cell_eq += acc * float(cv.sum())
        n_cells += float(cv.sum())
        nb += 1
        if nb >= max_batches:
            break
    print(f"    [eval] cell_acc={cell_eq/max(n_cells,1):.3f} (n_batches={nb})", flush=True)
    Tensor.training = True


if __name__ == "__main__":
    main()
