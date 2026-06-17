"""Perceiver-Poincaré kernel-efficiency profiling harness.

PURPOSE: localize the ~20x per-breath slowdown (perceiver ~3.1 s/breath at K=8 vs
v98 KenKen ~0.14 s/breath at K~20) by timing each logical phase of ONE K-breath
forward+backward step separately and reporting per-phase ms + kernel count + GB moved.

The ~20x is NOT explained by FLOPs (THINK 2.33e9 vs v98 2.49e9; 46 latents vs 49
cells; cross-attends add only +0.2%) and NOT a missing easy JIT win (all four already
confirmed). This harness instruments each phase to find the kernel-efficiency
root-cause:

  Phase 0 – COORD BUILD  : d_hyp cross (latent_coords + _d_hyp_cross), built once
             before the K loop.  Suspects: FP32 cast + pow+sum+gram on (B,L,49,dim);
             the diff_sq -> relu -> reciprocal -> log + isfinite chain fires many
             small kernels.
  Phase 1 – READ cross-attend : _cross_attend(d_read, cell_hidden, ...) per breath.
             Suspects: softmax over 49 cells per latent is small / un-fused; the
             cell_hidden cast from fp32 might re-materialize each breath.
  Phase 2 – THINK (Pythia L0-L3): _latent_layer_forward x4 over (B,L,H) in FP32.
             Suspects: L~46 latents (vs 49 cells in v98) -> TINY matmuls poorly
             occupying the 7900XTX; 4-layer fp32 self-attn on (B,16,L,L)=(8,16,46,46)
             -> 5888-element softmax, BW-starved; no fp16 packing (v98 runs fp16 for
             its cells).
  Phase 3 – WRITE cross-attend : _cross_attend(d_write, latent_hidden, ...) per breath.
  Phase 4 – READOUT          : layernorm + codebook matmul per breath.
  Phase 5 – BACKWARD         : backward() over all K breaths.
  Phase 6 – OPT STEP         : AdamW.step().

METHODOLOGY: each phase is isolated by calling .realize() on its output tensor(s)
before the next phase begins, forcing the AM driver to flush and giving GlobalCounters
a clean per-phase snapshot. The JIT is NOT used for phase profiling (TinyJit fuses
the full step into one dispatch, making per-phase attribution impossible). We run in
EAGER mode with Tensor.training=True; a warm-up pass (3 forward passes, no timing)
first to page in weights.

SUBSTRATE LAWS RESPECTED:
  - No dtypes.float32 literal in the phase bodies (inherited from perceiver_poincare).
  - where()-gated NaN guard for backward.
  - .realize() is the barrier; no .contiguous() stacking inside loops.
  - Single-kernel isfinite for grad health.

USAGE:
  PERCEIVER_TASK=1 K=8 BATCH=8 \\
  KENKEN_TRAIN=.cache/kenken_train_curriculum.jsonl \\
  KENKEN_TEST=.cache/kenken_test_curriculum.jsonl \\
  .venv/bin/python scripts/perceiver_perf_profile.py

Optional env overrides:
  K=8          number of breaths to profile (default 8, matching the slow 24.6 s/step)
  BATCH=8      batch size
  WARMUP=3     number of warm-up forward passes before timing
  REPS=3       number of timed repetitions to average over
  BALL_PATH=single|per_constraint
  PERCEIVER_N_GLOBAL=4   (must match what the training run uses)
  KENKEN_N_CAGES_MAX=N   (auto-detected from corpus if not set)
"""
from __future__ import annotations

import os
import sys
import time
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from tinygrad import Tensor, Device, dtypes
from tinygrad import GlobalCounters
from tinygrad.helpers import getenv
from tinygrad.nn.optim import AdamW

from mycelium import Config, BreathingTransformer
from mycelium.loader import _load_state, load_breathing
from mycelium.kenken_data import N_CELLS, N_MAX, load_jsonl
from mycelium.kenken import kenken_constraint_energy
from mycelium.perceiver_poincare import (
    PERCEIVER_K_MAX, PERCEIVER_TAU, PERCEIVER_RHO, PERCEIVER_DIM,
    PERCEIVER_N_GLOBAL,
    attach_perceiver_params, perceiver_parameters,
    perceiver_breathing_forward,
    latent_coords, cell_coords,
    _d_hyp_cross, _cross_attend, _latent_self_attn_bias, _latent_layer_forward,
)
from mycelium.perceiver_poincare_data import PerceiverLoader, PerceiverBatch, latent_capacity
from mycelium.kenken import embed_kenken


# ---------------------------------------------------------------------------
# fp32 cast (mirror perceiver_train.cast_layers_fp32)
# ---------------------------------------------------------------------------

def cast_layers_fp32(model: Any) -> None:
    def _cast(obj: Any, attr: str) -> None:
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


# ---------------------------------------------------------------------------
# Snapshot helper
# ---------------------------------------------------------------------------

def _snap() -> dict:
    """Return a snapshot of GlobalCounters accumulated values."""
    return {
        "kernel_count": GlobalCounters.kernel_count,
        "global_ops": GlobalCounters.global_ops,
        "global_mem": GlobalCounters.global_mem,
        "time_sum_s": GlobalCounters.time_sum_s,
    }


def _delta(before: dict, after: dict) -> dict:
    return {k: after[k] - before[k] for k in before}


def _barrier(*tensors: Tensor) -> None:
    """Force all tensors to realize (flush GPU), updating GlobalCounters."""
    for t in tensors:
        t.realize()
    Device[Device.DEFAULT].synchronize()


# ---------------------------------------------------------------------------
# Per-phase forward — EAGER (no TinyJit) so each .realize() is a clean barrier
# ---------------------------------------------------------------------------

def profile_one_step(model: Any, batch: PerceiverBatch, K: int,
                     ball_path: str, params: list[Tensor],
                     opt: AdamW) -> dict[str, dict]:
    """Run one full perceiver train step in eager mode, timing each phase.

    Returns a dict phase_name -> {wall_ms, kernel_count, global_ops, global_mem_GB,
    gc_time_ms} where gc_time_ms is what GlobalCounters.time_sum_s accumulated
    (kernel-level timing, requires DEBUG>=2 to be non-zero; the wall_ms is always
    valid).
    """
    from mycelium.breathing import _layernorm

    cfg = model.cfg

    state_embed    = model.perc_state_embed
    position_embed = model.perc_position_embed
    breath_embed   = model.perc_breath_embed
    latent_type_embed = model.perc_latent_type_embed
    value_codebook = model.perc_value_codebook

    tau = PERCEIVER_TAU

    input_cells       = batch.input_cells
    gold              = batch.gold
    cell_valid        = batch.cell_valid
    value_domain_mask = batch.value_domain_mask
    membership        = batch.latent_membership
    latent_valid      = batch.latent_valid
    latent_type       = batch.latent_type

    Bn = int(input_cells.shape[0])
    L  = int(membership.shape[1])

    results: dict[str, dict] = {}

    # -----------------------------------------------------------------------
    # OPT ZERO_GRAD (not timed — housekeeping before phase 0)
    # -----------------------------------------------------------------------
    opt.zero_grad()

    # -----------------------------------------------------------------------
    # PHASE 0 — COORD BUILD: latent_coords (g_phi + segment-mean + exp_0) +
    #           _d_hyp_cross -> d_read, d_write.
    # For "single" this is one cross-distance (B,L,49); for "per_constraint"
    # three cross-distances. The key suspects:
    #   • _d_hyp_cross: sa.pow(2).sum + gram (B,L,dim)@(B,dim,49) + diff_sq.relu
    #     + log(arg+sqrt(inner)) — each sub-op is a separate kernel on AMD
    #   • latent_coords: membership@phi_cells matmul (B,L,49)@(49,W) + rho MLP
    #   • dtype: entirely FP32 (boundary-gradient safety); no fp16 packing.
    # -----------------------------------------------------------------------
    GlobalCounters.reset()
    t0 = time.perf_counter()

    z_latent = latent_coords(model, membership, ball_path, latent_type)  # (B,L,dim)
    if ball_path == "single":
        z_cell = cell_coords(model, "single")
        z_cell_b = z_cell.reshape(1, N_CELLS, -1).expand(Bn, N_CELLS, -1)
        d_read = _d_hyp_cross(z_latent, z_cell_b)    # (B,L,49)
        d_write = d_read.transpose(1, 2)              # (B,49,L)
        _barrier(d_read, d_write)
    else:
        z_cell_row  = cell_coords(model, "row").reshape(1, N_CELLS, -1).expand(Bn, N_CELLS, -1)
        z_cell_col  = cell_coords(model, "col").reshape(1, N_CELLS, -1).expand(Bn, N_CELLS, -1)
        z_cell_cage = cell_coords(model, "cage").reshape(1, N_CELLS, -1).expand(Bn, N_CELLS, -1)
        d_row  = _d_hyp_cross(z_latent, z_cell_row)
        d_col  = _d_hyp_cross(z_latent, z_cell_col)
        d_cage = _d_hyp_cross(z_latent, z_cell_cage)
        t = latent_type.clip(0, 2)
        is_row  = (t == 0).cast(dtypes.float).reshape(Bn, L, 1)
        is_col  = (t == 1).cast(dtypes.float).reshape(Bn, L, 1)
        is_cage = (t == 2).cast(dtypes.float).reshape(Bn, L, 1)
        d_read  = d_row * is_row + d_col * is_col + d_cage * is_cage
        d_write = d_read.transpose(1, 2)
        _barrier(d_read, d_write)

    wall_phase0 = (time.perf_counter() - t0) * 1e3
    snap_phase0 = _snap()
    results["0_COORD_BUILD"] = {
        "wall_ms": wall_phase0,
        "kernel_count": snap_phase0["kernel_count"],
        "global_ops": snap_phase0["global_ops"],
        "global_mem_GB": snap_phase0["global_mem"] / 1e9,
        "gc_time_ms": snap_phase0["time_sum_s"] * 1e3,
        "note": "latent_coords (g_phi + exp0) + _d_hyp_cross once; FP32 throughout",
    }

    # -----------------------------------------------------------------------
    # CELL EMBEDDING (static; not a timed phase — but realize so it's paged in)
    # -----------------------------------------------------------------------
    cell_embed = embed_kenken(input_cells, state_embed, position_embed)
    cell_embed = cell_embed.cast(dtypes.float)
    _barrier(cell_embed)

    # latent hidden init
    type_oh = latent_type.clip(0, 3).one_hot(4).cast(dtypes.float)
    latent_hidden = type_oh @ latent_type_embed.cast(dtypes.float)
    latent_hidden = latent_hidden * latent_valid.reshape(Bn, L, 1).cast(dtypes.float)
    _barrier(latent_hidden)

    value_bias = (1.0 - value_domain_mask) * (-1e4)
    _barrier(value_bias)

    layers = list(model.block.layers)

    cell_logits_history = []
    phase_read_ms   = []
    phase_think_ms  = []
    phase_write_ms  = []
    phase_readout_ms = []
    phase_read_kc   = []
    phase_think_kc  = []
    phase_write_kc  = []
    phase_readout_kc = []
    phase_read_mem  = []
    phase_think_mem = []
    phase_write_mem = []
    phase_readout_mem = []
    phase_read_ops  = []
    phase_think_ops = []
    phase_write_ops = []
    phase_readout_ops = []

    cell_hidden = cell_embed   # (B,49,H)

    for k in range(K):
        be_k = breath_embed[k].reshape(1, 1, -1).cast(dtypes.float)

        # === PHASE 1 — READ cross-attend ===
        # _cross_attend: softmax over 49 cells, then attn @ cell_hidden.
        # Suspect: softmax on small (B,L,49) = (8,46,49)=18032 elements ->
        # un-fused with the preceding -d/tau addition; the cast of cell_hidden
        # (fp32) may produce a copy each breath if the lazy graph can't reuse.
        GlobalCounters.reset()
        t_r0 = time.perf_counter()

        ctx_l, read_attn = _cross_attend(d_read, cell_hidden, cell_valid, tau)
        latent_in = latent_hidden + ctx_l + be_k
        latent_in = latent_in * latent_valid.reshape(Bn, L, 1)
        _barrier(ctx_l, latent_in)

        t_r1 = time.perf_counter()
        snap_r = _snap()
        phase_read_ms.append((t_r1 - t_r0) * 1e3)
        phase_read_kc.append(snap_r["kernel_count"])
        phase_read_mem.append(snap_r["global_mem"] / 1e9)
        phase_read_ops.append(snap_r["global_ops"])

        # === PHASE 2 — THINK (4 Pythia L0-L3 layers in FP32) ===
        # Suspect: (B,L,H)=(8,46,1024); QKV matmuls are (8*16,46,64)^2 ->
        # 46x46 attention; L=46 << 64-head, so the GPU is occupancy-starved.
        # FP32 (not fp16) because of fp16 overflow on late breaths (see L447-450
        # in perceiver_poincare.py). v98 KenKen runs fp16; this runs fp32 -> AMD
        # fp32 vs fp16 throughput on bandwidth-bound ops can be 2x.
        # _latent_self_attn_bias is also rebuilt inside the loop each breath
        # (a per-batch (B,n_heads,L,L) bias from latent_valid masks); it is an
        # allocation + a few elementwise ops -> a hidden per-breath kernel burst.
        GlobalCounters.reset()
        t_t0 = time.perf_counter()

        attn_bias = _latent_self_attn_bias(latent_valid, cfg.n_heads)
        h = latent_in.cast(dtypes.float)
        for layer in layers[:4]:
            h = _latent_layer_forward(layer, h, attn_bias)
        gate_k = model.perc_delta_gate[k].cast(dtypes.float).reshape(1, 1, 1)
        latent_hidden = latent_hidden + gate_k * (h - latent_hidden)
        latent_hidden = latent_hidden * latent_valid.reshape(Bn, L, 1)
        _barrier(latent_hidden)

        t_t1 = time.perf_counter()
        snap_t = _snap()
        phase_think_ms.append((t_t1 - t_t0) * 1e3)
        phase_think_kc.append(snap_t["kernel_count"])
        phase_think_mem.append(snap_t["global_mem"] / 1e9)
        phase_think_ops.append(snap_t["global_ops"])

        # === PHASE 3 — WRITE cross-attend ===
        # Symmetric with READ but (B,49,L): softmax over L=46 latents per cell.
        # Suspect: same as READ; 49*L >> L*49 in different transposition ->
        # different kernel-fusion profile; memory layout after transpose may
        # force an extra copy.
        GlobalCounters.reset()
        t_w0 = time.perf_counter()

        ctx_c, write_attn = _cross_attend(d_write, latent_hidden, latent_valid, tau)
        cell_hidden = cell_embed + ctx_c
        cell_hidden = cell_hidden * cell_valid.reshape(Bn, N_CELLS, 1)
        _barrier(ctx_c, cell_hidden)

        t_w1 = time.perf_counter()
        snap_w = _snap()
        phase_write_ms.append((t_w1 - t_w0) * 1e3)
        phase_write_kc.append(snap_w["kernel_count"])
        phase_write_mem.append(snap_w["global_mem"] / 1e9)
        phase_write_ops.append(snap_w["global_ops"])

        # === PHASE 4 — READOUT ===
        # layernorm(cell_hidden) + codebook matmul (B,49,H)@(H,7) + value_bias.
        GlobalCounters.reset()
        t_o0 = time.perf_counter()

        x_ln = _layernorm(cell_hidden, model.ln_f_g, model.ln_f_b,
                          cfg.layer_norm_eps).cast(dtypes.float)
        cell_logits_k = x_ln @ value_codebook.T.cast(dtypes.float)
        cell_logits_k = cell_logits_k + value_bias.cast(dtypes.float)
        _barrier(cell_logits_k)

        t_o1 = time.perf_counter()
        snap_o = _snap()
        phase_readout_ms.append((t_o1 - t_o0) * 1e3)
        phase_readout_kc.append(snap_o["kernel_count"])
        phase_readout_mem.append(snap_o["global_mem"] / 1e9)
        phase_readout_ops.append(snap_o["global_ops"])

        cell_logits_history.append(cell_logits_k)

    # -----------------------------------------------------------------------
    # Build the loss for BACKWARD/OPT.
    #
    # CRITICAL (the bug that kept this profiler from ever running): the per-phase
    # forward above .realize()'s every intermediate (the per-phase barrier that
    # gives GlobalCounters a clean snapshot). In this tinygrad, .realize() swaps a
    # tensor's uop for a materialised BUFFER uop, so backward()'s toposort of the
    # loss no longer reaches the leaf params' compute-graph uops -> EVERY p.grad
    # comes back None -> AdamW's unwrap(t.grad) asserts. (Verified: a barriered
    # forward yields 18/18 None grads; an un-barriered one yields 0/18.)
    #
    # The forward-PHASE measurements (kernels/FLOPs/GB/wall) are still valid — the
    # barrier is exactly what makes them attributable. We only need a tape that
    # survives to the leaves for the BACKWARD measurement, so we rebuild the loss
    # from a single un-barriered forward via the real entry point. This recompute
    # is NOT inside any timed phase; only its backward()/opt.step() are timed, and
    # those carry the same kernel/FLOP/GB cost as the real run.
    # -----------------------------------------------------------------------
    rebuilt_history, _ = perceiver_breathing_forward(
        model, batch, K=K, ball_path=ball_path, collect_engagement=False)

    observed = (input_cells > 0).cast(dtypes.float)
    supervise = cell_valid * (1.0 - observed)
    sup_sum = supervise.sum() + 1e-6
    gold_idx = (gold - 1).clip(0, N_MAX - 1).reshape(Bn * N_CELLS)
    supervise_flat = supervise.reshape(Bn * N_CELLS)

    cell_loss_sum = Tensor.zeros((), dtype=dtypes.float).contiguous()
    weight_sum = 0.0
    for k, logits in enumerate(rebuilt_history):
        weight_k = 1.0 + float(k) / float(K - 1) if K > 1 else 1.0
        ce_elems = logits.reshape(Bn * N_CELLS, N_MAX).sparse_categorical_crossentropy(
            gold_idx, reduction="none")
        ce_k = (ce_elems * supervise_flat).sum() / sup_sum
        cell_loss_sum = cell_loss_sum + ce_k * weight_k
        weight_sum += weight_k
    cell_loss = cell_loss_sum / float(weight_sum)

    # -----------------------------------------------------------------------
    # PHASE 5 — BACKWARD
    # The backward graph traverses K breaths of READ+THINK+WRITE+READOUT plus
    # the coord-build graph (latent_coords gradient flows through g_phi, the
    # segment-mean, and exp_0's tanh boundary terms). This is the most expensive
    # phase to debug because tinygrad materializes ALL intermediate activations
    # saved for backward (no gradient checkpointing). The key suspects:
    #   • K * 4-layer THINK backward: each layer's backward is ~3 matmuls
    #     (dQ/dK/dV + dAttn + dFF) over (B,L,H) in FP32 -> 8 * 8 * 3 * 4 = 96
    #     matmuls, each 46x1024 -> very small.
    #   • _d_hyp_cross backward: diff_sq.relu gradient (where() gate, no issue
    #     from the arccosh boundary guard's inner.sqrt() floor 1e-10).
    #   • The per-breath attn_bias rebuild (_latent_self_attn_bias) is an
    #     allocation inside the graph: K * attn_bias forward + backward -> 2K
    #     extra allocation events; on AMD this may not be fused.
    # -----------------------------------------------------------------------
    # NOTE: we deliberately do NOT realize cell_loss before backward(). In this
    # tinygrad, .realize() swaps a tensor's uop for a materialised BUFFER uop, so
    # realizing the loss (or any forward activation) severs the autograd tape and
    # backward() returns all-None grads (verified: realize-then-backward -> 54/54
    # None). So the BACKWARD phase below necessarily includes the forward RECOMPUTE
    # of the rebuilt graph (forward kernels + grad kernels fused) — which is in
    # fact how the real JIT'd _step runs it (forward+backward in one realize).
    # FLOP/GB/kernel READ GUIDANCE: BACKWARD here ~= (1x forward recompute + the
    # backward); subtract the phase-0..4 forward totals to isolate the pure-grad
    # cost if needed. The forward-phase numbers (0-4) remain clean (barriered).
    # -----------------------------------------------------------------------
    GlobalCounters.reset()
    t_b0 = time.perf_counter()

    cell_loss.backward()
    # NaN guard (where()-gated, not multiply — substrate law).
    healthy_b = cell_loss.isfinite()
    for p in params:
        if p.grad is not None:
            p.grad = healthy_b.where(p.grad, Tensor.zeros_like(p.grad))
    # Force grad materialization before timing the opt step.
    grad_tensors = [p.grad for p in params if p.grad is not None]
    if grad_tensors:
        _barrier(*grad_tensors)

    t_b1 = time.perf_counter()
    snap_b = _snap()
    results["5_BACKWARD"] = {
        "wall_ms": (t_b1 - t_b0) * 1e3,
        "kernel_count": snap_b["kernel_count"],
        "global_ops": snap_b["global_ops"],
        "global_mem_GB": snap_b["global_mem"] / 1e9,
        "gc_time_ms": snap_b["time_sum_s"] * 1e3,
        "note": f"backward over {K} breaths (READ+THINK+WRITE+READOUT) + coord-build; "
                "suspect: K*4-layer FP32 backward; _d_hyp_cross arccosh backward; "
                f"K={K} per-breath attn_bias reallocations",
    }

    # -----------------------------------------------------------------------
    # PHASE 6 — OPT STEP (AdamW parameter update)
    # -----------------------------------------------------------------------
    GlobalCounters.reset()
    t_s0 = time.perf_counter()

    opt.step()
    Device[Device.DEFAULT].synchronize()

    t_s1 = time.perf_counter()
    snap_s = _snap()
    results["6_OPT_STEP"] = {
        "wall_ms": (t_s1 - t_s0) * 1e3,
        "kernel_count": snap_s["kernel_count"],
        "global_ops": snap_s["global_ops"],
        "global_mem_GB": snap_s["global_mem"] / 1e9,
        "gc_time_ms": snap_s["time_sum_s"] * 1e3,
        "note": "AdamW.step() over all trainable params",
    }

    # -----------------------------------------------------------------------
    # CONSOLIDATED FULL-STEP COUNTERS (the authoritative per-step kernel count).
    # A single un-barriered forward + backward + opt.step in ONE GlobalCounters
    # window — the eager dispatch total for a whole training step. This is the
    # number the dispatch-bound diagnosis hinges on (kernels/step, kernels/breath).
    # It is measured separately because the per-phase windows above reset between
    # phases and the BACKWARD phase double-counts the forward recompute. EAGER, so
    # it OVER-counts vs the JIT (which fuses) — but the kernel COUNT is the signal.
    # -----------------------------------------------------------------------
    opt.zero_grad()
    GlobalCounters.reset()
    t_fs0 = time.perf_counter()
    fs_hist, _ = perceiver_breathing_forward(
        model, batch, K=K, ball_path=ball_path, collect_engagement=False)
    fs_loss_sum = Tensor.zeros((), dtype=dtypes.float).contiguous()
    fs_wsum = 0.0
    for k, logits in enumerate(fs_hist):
        wk = 1.0 + float(k) / float(K - 1) if K > 1 else 1.0
        ce_e = logits.reshape(Bn * N_CELLS, N_MAX).sparse_categorical_crossentropy(
            gold_idx, reduction="none")
        fs_loss_sum = fs_loss_sum + (ce_e * supervise_flat).sum() / sup_sum * wk
        fs_wsum += wk
    fs_loss = fs_loss_sum / float(fs_wsum)
    fs_loss.backward()
    fs_healthy = fs_loss.isfinite()
    for p in params:
        if p.grad is not None:
            p.grad = fs_healthy.where(p.grad, Tensor.zeros_like(p.grad))
    opt.step()
    Device[Device.DEFAULT].synchronize()
    t_fs1 = time.perf_counter()
    snap_fs = _snap()
    results["_FULL_STEP_CLEAN"] = {
        "wall_ms": (t_fs1 - t_fs0) * 1e3,
        "kernel_count": snap_fs["kernel_count"],
        "global_ops": snap_fs["global_ops"],
        "global_mem_GB": snap_fs["global_mem"] / 1e9,
        "gc_time_ms": snap_fs["time_sum_s"] * 1e3,
        "note": "fwd+bwd+opt in ONE eager window; AUTHORITATIVE kernels/step + GB/step",
    }

    # Aggregate per-breath phases (sum over K breaths).
    def _agg(phase_id: str, ms_list: list, kc_list: list, mem_list: list,
             ops_list: list, note: str) -> None:
        results[phase_id] = {
            "wall_ms": float(sum(ms_list)),
            "kernel_count": int(sum(kc_list)),
            "global_ops": int(sum(ops_list)),
            "global_mem_GB": float(sum(mem_list)),
            "gc_time_ms": 0.0,
            "per_breath_ms": [round(x, 2) for x in ms_list],
            "per_breath_kc": [int(x) for x in kc_list],
            "note": note,
        }

    _agg("1_READ_x{K}", phase_read_ms, phase_read_kc, phase_read_mem, phase_read_ops,
         note=f"_cross_attend(d_read, cell_hidden) x{K} breaths; softmax over S=49 cells; "
              "FP32; suspect: small softmax un-fused, cell_hidden cast copy each breath")
    _agg("2_THINK_x{K}", phase_think_ms, phase_think_kc, phase_think_mem, phase_think_ops,
         note=f"_latent_self_attn_bias rebuild + 4x _latent_layer_forward in FP32 x{K} "
              f"breaths; (B={Bn},L={L},H=1024); CORE SUSPECT: L={L}<<head_dim=64 -> "
              "tiny matmuls, poor GPU occupancy; FP32 vs v98 fp16 -> ~2x BW cost on AMD")
    _agg("3_WRITE_x{K}", phase_write_ms, phase_write_kc, phase_write_mem, phase_write_ops,
         note=f"_cross_attend(d_write, latent_hidden) x{K} breaths; softmax over S={L} latents; "
              "FP32; suspect: transpose of d_read -> memory layout change -> extra copy")
    _agg("4_READOUT_x{K}", phase_readout_ms, phase_readout_kc, phase_readout_mem, phase_readout_ops,
         note=f"layernorm + (B,49,1024)@(1024,7) codebook matmul x{K} breaths; FP32")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    assert int(getenv("PERCEIVER_TASK", 0)) > 0, \
        "Set PERCEIVER_TASK=1 before running this script."

    K        = int(getenv("K", str(PERCEIVER_K_MAX)))
    BATCH    = int(getenv("BATCH", 8))
    WARMUP   = int(getenv("WARMUP", 3))
    REPS     = int(getenv("REPS", 3))
    BALL_PATH_ENV = getenv("BALL_PATH", "single").strip().lower()
    KENKEN_TRAIN = getenv("KENKEN_TRAIN", ".cache/kenken_train_curriculum.jsonl")
    KENKEN_TEST  = getenv("KENKEN_TEST",  ".cache/kenken_test_curriculum.jsonl")
    SEED     = int(getenv("SEED", 42))

    print("=" * 72)
    print(f"Perceiver-Poincaré perf profiler  K={K}  B={BATCH}  "
          f"ball_path={BALL_PATH_ENV}  warmup={WARMUP}  reps={REPS}")
    print(f"device={Device.DEFAULT}  seed={SEED}")
    print("=" * 72)

    np.random.seed(SEED)
    Tensor.manual_seed(SEED)

    train_recs = load_jsonl(KENKEN_TRAIN)
    test_recs  = load_jsonl(KENKEN_TEST)
    corpus_n_cages_max = max(
        max(len(r["cages"]) for r in train_recs),
        max(len(r["cages"]) for r in test_recs),
    )
    N_CAGES_MAX = int(getenv("KENKEN_N_CAGES_MAX", str(corpus_n_cages_max)))
    L_MAX = latent_capacity(N_CAGES_MAX, PERCEIVER_N_GLOBAL)
    print(f"n_cages_max={N_CAGES_MAX}  L_max={L_MAX}  "
          f"(7 rows + 7 cols + {N_CAGES_MAX} cages + {PERCEIVER_N_GLOBAL} global)")

    cfg = Config()
    print("Loading Pythia-410M ...", flush=True)
    sd = _load_state()
    model = load_breathing(cfg, sd=sd)
    del sd
    cast_layers_fp32(model)
    attach_perceiver_params(model, hidden=cfg.hidden, n_heads=cfg.n_heads,
                            L_max=L_MAX, k_max=K)
    Device[Device.DEFAULT].synchronize()

    loader = PerceiverLoader(KENKEN_TRAIN, batch_size=BATCH, seed=SEED,
                             n_cages_max=N_CAGES_MAX)

    ball_path = BALL_PATH_ENV if BALL_PATH_ENV in ("single", "per_constraint") else "single"

    # Mirror the REAL trainer's trainable set (THINK L0-L3 attn/FFN + final LN +
    # active-path perceiver params) so the OPT-STEP phase reflects the live run —
    # the perceiver-only params are ~0.1M; the THINK layers dominate opt.step.
    from scripts.perceiver_train import collect_params
    params = collect_params(model, ball_path=ball_path)
    n_params = sum(int(np.prod(t.shape)) for t in params)
    print(f"Trainable params ({ball_path}): {n_params/1e6:.1f}M "
          f"(THINK L0-L3 + final LN + perceiver)")

    opt = AdamW(params, lr=3e-5, weight_decay=0.0)

    Tensor.training = True

    # -----------------------------------------------------------------------
    # Warm-up: a few forward passes (no backward) to page in weights / JIT caches
    # -----------------------------------------------------------------------
    print(f"\nWarm-up ({WARMUP} forward passes, no timing) ...", flush=True)
    for _ in range(WARMUP):
        batch = loader.sample_batch()
        z_l = latent_coords(model, batch.latent_membership, ball_path, batch.latent_type)
        if ball_path == "single":
            z_c = cell_coords(model, "single").reshape(1, N_CELLS, -1).expand(
                BATCH, N_CELLS, -1)
            dr = _d_hyp_cross(z_l, z_c)
            dr.realize()
        Device[Device.DEFAULT].synchronize()
    print("Warm-up done.", flush=True)

    # -----------------------------------------------------------------------
    # Timed reps
    # -----------------------------------------------------------------------
    all_results: list[dict] = []
    for rep in range(REPS):
        batch = loader.sample_batch()
        print(f"\n--- Rep {rep+1}/{REPS} ---", flush=True)
        t_step_start = time.perf_counter()
        results = profile_one_step(model, batch, K=K, ball_path=ball_path,
                                   params=params, opt=opt)
        t_step_total = (time.perf_counter() - t_step_start) * 1e3
        results["_TOTAL_STEP"] = {"wall_ms": t_step_total}
        all_results.append(results)

    # -----------------------------------------------------------------------
    # Aggregate over REPS and print report
    # -----------------------------------------------------------------------
    print("\n" + "=" * 72)
    print(f"PERCEIVER PERF REPORT  K={K}  B={BATCH}  L={L_MAX}  "
          f"ball_path={ball_path}  (avg over {REPS} reps)")
    print("=" * 72)

    phase_order = [
        "0_COORD_BUILD",
        "1_READ_x{K}", "2_THINK_x{K}", "3_WRITE_x{K}", "4_READOUT_x{K}",
        "5_BACKWARD", "6_OPT_STEP", "_FULL_STEP_CLEAN", "_TOTAL_STEP",
    ]

    rows = []
    total_wall = 0.0
    for phase in phase_order:
        vals = [r[phase] for r in all_results if phase in r]
        if not vals:
            continue
        avg_wall  = float(np.mean([v["wall_ms"] for v in vals]))
        avg_kc    = float(np.mean([v.get("kernel_count", 0) for v in vals]))
        avg_ops   = float(np.mean([v.get("global_ops", 0) for v in vals]))
        avg_memGB = float(np.mean([v.get("global_mem_GB", 0.0) for v in vals]))
        avg_gc    = float(np.mean([v.get("gc_time_ms", 0.0) for v in vals]))
        if phase not in ("_TOTAL_STEP", "_FULL_STEP_CLEAN"):
            total_wall += avg_wall
        rows.append((phase, avg_wall, avg_kc, avg_ops, avg_memGB, avg_gc))

    fmt = "{:<18s}  {:>9s}  {:>8s}  {:>9s}  {:>8s}  {:>9s}  {:>6s}"
    print(fmt.format("phase", "wall_ms", "kernels", "GFLOP", "mem_GB", "gc_ms", "%total"))
    print("-" * 78)
    step_ms = float(np.mean([r["_TOTAL_STEP"]["wall_ms"] for r in all_results]))
    for phase, wall, kc, ops, mem, gc in rows:
        pct = f"{100*wall/step_ms:.1f}%" if phase not in ("_TOTAL_STEP", "_FULL_STEP_CLEAN") else "--"
        print(fmt.format(phase[:18], f"{wall:.1f}", f"{int(kc)}", f"{ops/1e9:.2f}",
                         f"{mem:.3f}", f"{gc:.1f}", pct))

    print("-" * 78)
    # The dispatch-bound diagnosis: kernels/step + kernels/breath (the AUTHORITATIVE
    # full-step number, not the sum of the double-counting per-phase windows).
    fs_kc = float(np.mean([r["_FULL_STEP_CLEAN"]["kernel_count"] for r in all_results]))
    fs_gb = float(np.mean([r["_FULL_STEP_CLEAN"]["global_mem_GB"] for r in all_results]))
    fs_gflop = float(np.mean([r["_FULL_STEP_CLEAN"]["global_ops"] for r in all_results])) / 1e9
    fwd_kc = float(np.mean([sum(r[p]["kernel_count"] for p in
        ("0_COORD_BUILD", "1_READ_x{K}", "2_THINK_x{K}", "3_WRITE_x{K}", "4_READOUT_x{K}"))
        for r in all_results]))
    print(f"\n[DISPATCH DIAGNOSIS]  full eager step (fwd+bwd+opt):")
    print(f"  kernels/step  = {fs_kc:.0f}   ->  kernels/breath = {fs_kc/K:.0f}  (K={K})")
    print(f"  forward-only kernels/step = {fwd_kc:.0f}  ->  fwd kernels/breath = {fwd_kc/K:.0f}")
    print(f"  GFLOP/step = {fs_gflop:.1f}    GB-moved/step = {fs_gb:.2f}")
    # Reference: v98 KenKen executor at K~20 = ~0.14 s/breath -> per-step at K=8 ~ 1.1 s
    # Perceiver target: 24.6 s/step at K=8 -> 3.1 s/breath; goal is to find which phase
    # accounts for most of the ~22x gap vs the reference.
    print(f"\nTotal step wall time (avg): {step_ms:.1f} ms  = {step_ms/1000:.2f} s")
    print(f"Per-breath equiv (step/{K}): {step_ms/K:.1f} ms")
    print(f"Reference: v98 KenKen ~140 ms/breath at K~20  -> target speedup: "
          f"~{step_ms/K/140:.1f}x gap remains")
    print()
    # Print per-breath breakdown for the loop phases
    for phase in ["1_READ_x{K}", "2_THINK_x{K}", "3_WRITE_x{K}", "4_READOUT_x{K}"]:
        for r in all_results[:1]:
            if phase in r and "per_breath_ms" in r[phase]:
                pb = r[phase]["per_breath_ms"]
                print(f"  {phase:<22s}  per-breath (ms): "
                      f"min={min(pb):.1f}  max={max(pb):.1f}  "
                      f"avg={np.mean(pb):.1f}")
    print()
    # Print notes (hypotheses) for the dominant phases
    dominant = sorted([(w, ph) for ph, w, *_ in rows
                       if ph not in ("_TOTAL_STEP", "_FULL_STEP_CLEAN")],
                      reverse=True)[:3]
    print("Top-3 slowest phases (hypotheses):")
    for w, ph in dominant:
        for r in all_results[:1]:
            if ph in r:
                print(f"  [{ph}]  {w:.1f} ms")
                print(f"    => {r[ph].get('note', '')}")
    print()
    print("Interpretation guide:")
    print("  • If THINK dominates: FP32 tiny-L matmuls (L=46<<64) is the culprit.")
    print("    Fix: cast THINK to fp16 + rescale delta_gate or use bf16 if AMD supports.")
    print("  • If COORD_BUILD dominates: _d_hyp_cross kernel fragmentation.")
    print("    Fix: fuse pow+sum+gram+relu+log into one custom kernel (or hoist d_read")
    print("    outside the JIT by treating it as a per-step constant input).")
    print("  • If BACKWARD >> FORWARD: activation checkpointing or K reduction.")
    print("  • If READ/WRITE dominate: softmax fusion; consider sparse top-k routing.")
    print("  • If kernel_count >> expected: tinygrad can't fuse small ops -> use")
    print("    .contiguous() after each major sub-op to force a clean fusion boundary.")


if __name__ == "__main__":
    main()
