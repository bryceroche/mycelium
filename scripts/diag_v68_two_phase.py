"""Diagnostic for v68 TWO_PHASE norm explosion.

Two tasks in priority order:
  Task 1: Verify v66 baseline (TWO_PHASE=0) still works with current code.
           Loads v66 step 3000 ckpt, runs a 3-example forward, checks norms.
  Task 2: Trace per-layer norms in TWO_PHASE=1 forward to locate explosion.
           Loads patched v68 ckpt, adds norm logging inside breathe_once.

Usage:
  # Task 1 (v66 baseline):
  TASK=1 DEV=PCI+AMD TWO_PHASE=0 python scripts/diag_v68_two_phase.py

  # Task 2 (v68 norm trace):
  TASK=2 DEV=PCI+AMD TWO_PHASE=1 python scripts/diag_v68_two_phase.py
"""
import os, sys, math

# Always insert MAIN project root first so mycelium/* modules resolve .cache relative to main project.
_WORKTREE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MAIN_ROOT = "/home/bryce/mycelium"
# Use main root's mycelium package if it has .cache, otherwise fall back to worktree
if os.path.exists(os.path.join(_MAIN_ROOT, ".cache", "pythia-410m", "model.safetensors")):
    sys.path.insert(0, _MAIN_ROOT)
else:
    sys.path.insert(0, _WORKTREE_ROOT)

TASK = int(os.environ.get("TASK", "1"))

# ---- architecture env vars (must be set before importing breathing) ----
# These match the v66 / v68 launcher exactly.
os.environ.setdefault("NOTEBOOK_DAG", "0")
os.environ.setdefault("CONTROLLER_DECODE", "1")
os.environ.setdefault("CONTROLLER_N_LAYERS", "2")
os.environ.setdefault("PER_BREATH_DECODE", "1")
os.environ.setdefault("BFIELD_WAIST", "512")
os.environ.setdefault("BFIELD_END_OF_BREATH", "1")
os.environ.setdefault("BFIELD_ENFORCED", "0")
os.environ.setdefault("BFIELD_ALPHA", "1.0")
os.environ.setdefault("WAIST_CODEBOOK_N", "64")
os.environ.setdefault("WAIST_CODEBOOK_INJECT_WEIGHT", "1.0")
os.environ.setdefault("NOTEBOOK_V24", "1")
os.environ.setdefault("NOTEBOOK_ACCUMULATE_ENABLED", "0")
os.environ.setdefault("NOTEBOOK_DUAL", "1")
os.environ.setdefault("NOTEBOOK_POOL_MODE", "attn")
os.environ.setdefault("NOTEBOOK_INIT_SCALE", "0.02")
os.environ.setdefault("STOCH_DEPTH_P", "0.10")
os.environ.setdefault("LABEL_SMOOTHING", "0.1")
os.environ.setdefault("WEIGHT_DECAY", "0.05")
os.environ.setdefault("PER_HEAD_PITCH", "1")
os.environ.setdefault("CONSTANT_RADIUS", "1")
os.environ.setdefault("BREATH_TIME_EMBED", "1")
os.environ.setdefault("BREATH_TIME_INIT_SCALE", "0.0")
os.environ.setdefault("CROSS_BREATH_HANDOFF", "1")
os.environ.setdefault("ABLATE_BREATH_ROTATION", "1")
os.environ.setdefault("QUADRATURE_HEADS", "0")
os.environ.setdefault("PROMPT_REFRESH_ALPHA", "0.1")
os.environ.setdefault("BOUNDARY_AUX_WEIGHT", "0.1")
os.environ.setdefault("BOUNDARY_POS_WEIGHT", "5.0")
os.environ.setdefault("SCHED_SAMPLE_RATE", "0.3")

# v66 uses SINE_TEMP=1; v68 drops it.
if os.environ.get("TWO_PHASE", "0") == "0":
    os.environ.setdefault("SINE_TEMP", "1")
    os.environ.setdefault("SINE_TEMP_MAX", "2.0")
    os.environ.setdefault("SINE_TEMP_MIN", "0.7")
else:
    os.environ.setdefault("SINE_TEMP", "0")
    os.environ.setdefault("EXPAND_LAYERS", "4")
    os.environ.setdefault("COMPRESS_LAYERS", "2")
    os.environ.setdefault("EXPAND_TEMP", "2.0")
    os.environ.setdefault("COMPRESS_TEMP", "0.7")

import numpy as np
from tinygrad import Tensor, Device, dtypes
from tinygrad.nn.state import safe_load

from mycelium import Config
from mycelium.loader import load_breathing, _load_state
from mycelium.data import load_tokenizer
from mycelium.l3_data import space_digits

# ---- locate project root based on cache availability ----
# Support running from either the main project dir or worktree dir.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_CANDIDATE_ROOTS = [
    os.path.dirname(_SCRIPT_DIR),                           # worktree root
    "/home/bryce/mycelium",                                 # main project root
]
_PROJECT_ROOT = next(
    (r for r in _CANDIDATE_ROOTS if os.path.exists(os.path.join(r, ".cache", "pythia-410m", "model.safetensors"))),
    _CANDIDATE_ROOTS[0]
)
print(f"    project root (for .cache): {_PROJECT_ROOT}")
os.environ.setdefault("PYTHIA_WEIGHTS", os.path.join(_PROJECT_ROOT, ".cache", "pythia-410m", "model.safetensors"))

# ---- checkpoints ----
V66_CKPT      = os.path.join(_PROJECT_ROOT, ".cache/gsm8k_steps_ckpts/v66_sched_sampling_step3000.safetensors")
V68_CKPT      = os.path.join(_PROJECT_ROOT, ".cache/gsm8k_steps_ckpts/v66_step3000_two_phase.safetensors")
V68_CKPT_FIXB = os.path.join(_PROJECT_ROOT, ".cache/gsm8k_steps_ckpts/v66_step3000_two_phase_fixb.safetensors")

WT = os.environ.get("TWO_PHASE", "0")
FIX = os.environ.get("FIX", "")  # "B" for fix-b ckpt
if WT == "0":
    CKPT = V66_CKPT
elif FIX == "B":
    CKPT = V68_CKPT_FIXB
else:
    CKPT = V68_CKPT

print(f"=== diag_v68_two_phase.py  TASK={TASK}  TWO_PHASE={WT} ===")
print(f"    ckpt: {CKPT}")
print(f"    device: {Device.DEFAULT}")

# ---- build model ----
print("\nBuilding model...")
cfg = Config()
sd_pythia = _load_state()
model = load_breathing(cfg, sd=sd_pythia)
del sd_pythia

print(f"Loading checkpoint {CKPT}...")
ckpt_sd = safe_load(CKPT)
info = model.load_state_dict(ckpt_sd, strict=False)
missing  = info.get("missing", [])
unexpected = info.get("unexpected", [])
print(f"  missing  ({len(missing)}): {missing[:8]}")
print(f"  unexpected ({len(unexpected)}): {unexpected[:8]}")
del ckpt_sd
Device[Device.DEFAULT].synchronize()

Tensor.training = False

# ---- helper: compute tensor norm (fp32 for safety) ----
def norm_of(t: Tensor) -> float:
    return float(t.cast(dtypes.float).square().sum(-1).mean().sqrt().numpy())

# ---- helper: short token sequence ----
tok = load_tokenizer()
TEST_PROBLEM = "Janet ' s ducks lay 1 6 eggs per day. She eats 3 for breakfast and bakes 4 cookies for her friends daily. How many eggs does she have left ?"
tokens = tok.encode(TEST_PROBLEM).ids[:64]  # cap at 64 tokens


# ======================================================================
# Task 1 — v66 baseline regression check
# ======================================================================
if TASK == 1:
    print("\n" + "="*60)
    print("TASK 1: v66 baseline norm + quick forward check")
    print("="*60)
    print(f"Input tokens: {len(tokens)}")
    print(f"TWO_PHASE env: {os.environ.get('TWO_PHASE', '0')}")

    # Run single breathe_once and inspect norms
    toks = Tensor([tokens], dtype=dtypes.int32)
    x = model.embed(toks)  # (1, T, H)
    x = x.cast(dtypes.float)
    print(f"\nEmbed output norm:  {norm_of(x):.3f}")

    n_loops = 4  # v66 trains at TRAIN_LOOPS=4
    print(f"\nRunning {n_loops} breathe_once calls (individual), tracing norms:")
    from mycelium.breathing import _sine_temp_baseline

    # Monkey-patch breathe_once to log per-layer norms
    block = model.block
    import mycelium.breathing as bmod

    # Store original breathe_once
    _orig_breathe_once = block.breathe_once.__func__

    # Patched version that logs per-layer norms
    def _patched_breathe_once(self, x_in, loop_idx, temp_mult=1.0, return_waist_compressed=False):
        from mycelium.breathing import (
            TWO_PHASE, EXPAND_LAYERS, COMPRESS_LAYERS, EXPAND_TEMP, COMPRESS_TEMP,
            DOUBLED_LAYERS, PER_HEAD_PITCH, LAYER_PITCH_TARGET, SINE_TEMP,
            BREATH_TIME_EMBED, PER_BREATH_TEMP, BREATH_NORM_OSC, CONSTANT_RADIUS,
            BFIELD_WAIST, BFIELD_END_OF_BREATH,
            _per_layer_temp_within_breath, _per_layer_norm_scale_within_breath
        )
        x = x_in
        alpha = self.rope._alpha_at(loop_idx, x.dtype)
        if BREATH_TIME_EMBED:
            x = x + self.breath_embed[loop_idx].reshape(1, 1, -1).cast(x.dtype)
        ac_base, asn_base = alpha
        n_phases = self.cfg.n_phases
        if TWO_PHASE:
            active_layers = list(self.expand_layers) + list(self.compress_layers)
            per_layer_temp_override = ([EXPAND_TEMP] * EXPAND_LAYERS + [COMPRESS_TEMP] * COMPRESS_LAYERS)
            phase_tags = ["EXP"] * EXPAND_LAYERS + ["CMP"] * COMPRESS_LAYERS
        elif DOUBLED_LAYERS and loop_idx >= (self.cfg.max_loops // 2):
            active_layers = self.layers_b
            per_layer_temp_override = None
            phase_tags = ["B"] * len(self.layers_b)
        else:
            active_layers = self.layers
            per_layer_temp_override = None
            phase_tags = ["  "] * len(self.layers)

        print(f"  breath {loop_idx} input norm: {norm_of(x):.4f}", flush=True)
        for layer_idx, (layer, tag) in enumerate(zip(active_layers, phase_tags)):
            if PER_HEAD_PITCH and layer_idx > 0:
                cos_o = self.per_head_pitch_cos[layer_idx].cast(x.dtype)
                sin_o = self.per_head_pitch_sin[layer_idx].cast(x.dtype)
                ac_layer = ac_base * cos_o - asn_base * sin_o
                asn_layer = ac_base * sin_o + asn_base * cos_o
                layer_alpha = (ac_layer, asn_layer)
            elif LAYER_PITCH_TARGET > 0.0 and layer_idx > 0:
                import math as _math
                offset_angle = (self.layer_pitch_scale * float(layer_idx)).cast(dtypes.float)
                cos_o = offset_angle.cos().reshape(1, 1, 1, 1).cast(x.dtype)
                sin_o = offset_angle.sin().reshape(1, 1, 1, 1).cast(x.dtype)
                ac_layer = ac_base * cos_o - asn_base * sin_o
                asn_layer = ac_base * sin_o + asn_base * cos_o
                layer_alpha = (ac_layer, asn_layer)
            else:
                layer_alpha = alpha
            if per_layer_temp_override is not None:
                layer_temp = per_layer_temp_override[layer_idx]
            elif PER_BREATH_TEMP:
                layer_temp = _per_layer_temp_within_breath(layer_idx, n_phases)
            else:
                layer_temp = temp_mult

            x_before = x
            x = layer(x, loop_idx, temp_mult=layer_temp, alpha=layer_alpha)
            n_before = norm_of(x_before)
            n_after  = norm_of(x)
            delta = x - x_before
            delta_n  = norm_of(delta)
            print(f"    layer {layer_idx} [{tag}] temp={layer_temp:.2f}: "
                  f"norm {n_before:.4f} → {n_after:.4f}  (delta={delta_n:.4f})", flush=True)

            if BREATH_NORM_OSC and CONSTANT_RADIUS:
                scale = _per_layer_norm_scale_within_breath(layer_idx, n_phases)
                x_f = x.cast(dtypes.float)
                x_norm = (x_f.square().sum(axis=-1, keepdim=True) + 1e-6).sqrt()
                target = self.crp_target_norm * scale
                mix = self.crp_mix_alpha
                x_proj = x_f * (target / x_norm)
                x = (x_f * (1.0 - mix) + x_proj * mix).cast(x.dtype)

            if BFIELD_WAIST > 0 and layer_idx == 1 and not BFIELD_END_OF_BREATH:
                x = self.apply_bfield_waist(x)

        waist_compressed = None
        if BFIELD_WAIST > 0 and BFIELD_END_OF_BREATH:
            if return_waist_compressed:
                x, waist_compressed = self.apply_bfield_waist(x, return_compressed=True)
            else:
                x = self.apply_bfield_waist(x)
        if CONSTANT_RADIUS and not BREATH_NORM_OSC:
            x_f = x.cast(dtypes.float)
            x_norm = (x_f.square().sum(axis=-1, keepdim=True) + 1e-6).sqrt()
            target = self.crp_target_norm
            mix = self.crp_mix_alpha
            x_proj = x_f * (target / x_norm)
            x = (x_f * (1.0 - mix) + x_proj * mix).cast(x.dtype)

        print(f"  breath {loop_idx} output norm: {norm_of(x):.4f}", flush=True)
        if return_waist_compressed:
            return x, waist_compressed
        return x

    import types
    block.breathe_once = types.MethodType(_patched_breathe_once, block)

    x_cur = x
    for l in range(n_loops):
        temp = _sine_temp_baseline(l, n_loops)
        x_cur = block.breathe_once(x_cur, l, temp_mult=temp)
        Device[Device.DEFAULT].synchronize()

    print(f"\nFinal output norm after {n_loops} breaths: {norm_of(x_cur):.4f}")
    print("\nDone. If norms stay below ~50 across all layers, v66 baseline is OK.")


# ======================================================================
# Task 2 — v68 TWO_PHASE norm trace
# ======================================================================
elif TASK == 2:
    print("\n" + "="*60)
    print("TASK 2: v68 TWO_PHASE per-layer norm trace")
    print("="*60)
    from mycelium.breathing import (
        TWO_PHASE as TP, EXPAND_LAYERS as EL, COMPRESS_LAYERS as CL,
        EXPAND_TEMP as ET, COMPRESS_TEMP as CT
    )
    print(f"  TWO_PHASE={TP}  EXPAND={EL}@{ET}  COMPRESS={CL}@{CT}")

    if not TP:
        print("ERROR: TWO_PHASE is not 1 in loaded module — aborting Task 2.")
        sys.exit(1)

    toks = Tensor([tokens], dtype=dtypes.int32)
    x = model.embed(toks).cast(dtypes.float)
    print(f"\nEmbed output norm:  {norm_of(x):.3f}")

    block = model.block
    # Check shared weights equality
    exp_wv  = float(block.expand_shared.wv.cast(dtypes.float).abs().mean().numpy())
    cmp_wv  = float(block.compress_shared.wv.cast(dtypes.float).abs().mean().numpy())
    are_eq  = (block.expand_shared.wv - block.compress_shared.wv).abs().max().numpy()
    print(f"\n  expand_shared.wv mean abs: {exp_wv:.5f}")
    print(f"  compress_shared.wv mean abs: {cmp_wv:.5f}")
    print(f"  |expand_shared.wv - compress_shared.wv| max: {float(are_eq):.6f}  (should be 0.0 if cloned)")

    # Also compare expand_phase0 == v66.phase0 (by proxy: compare to what we loaded)
    print(f"\n  expand_phase0.wq[:3,:3]: {block.expand_layers[0].wq[:3,:3].cast(dtypes.float).numpy()}")
    print(f"  compress_phase0.wq[:3,:3]: {block.compress_layers[0].wq[:3,:3].cast(dtypes.float).numpy()}")

    import types
    from mycelium.breathing import (
        DOUBLED_LAYERS, PER_HEAD_PITCH, LAYER_PITCH_TARGET,
        BREATH_TIME_EMBED, PER_BREATH_TEMP, BREATH_NORM_OSC, CONSTANT_RADIUS,
        BFIELD_WAIST, BFIELD_END_OF_BREATH,
        _per_layer_temp_within_breath, _per_layer_norm_scale_within_breath
    )

    def _patched_breathe_once_v68(self, x_in, loop_idx, temp_mult=1.0, return_waist_compressed=False):
        x = x_in
        alpha = self.rope._alpha_at(loop_idx, x.dtype)
        if BREATH_TIME_EMBED:
            x = x + self.breath_embed[loop_idx].reshape(1, 1, -1).cast(x.dtype)
        ac_base, asn_base = alpha

        active_layers = list(self.expand_layers) + list(self.compress_layers)
        per_layer_temp_override = ([ET] * EL + [CT] * CL)
        phase_tags = ["EXP"] * EL + ["CMP"] * CL

        print(f"\n  --- breath {loop_idx} --- input norm={norm_of(x):.4f}", flush=True)
        for layer_idx, (layer, tag) in enumerate(zip(active_layers, phase_tags)):
            if PER_HEAD_PITCH and layer_idx > 0:
                cos_o = self.per_head_pitch_cos[layer_idx].cast(x.dtype)
                sin_o = self.per_head_pitch_sin[layer_idx].cast(x.dtype)
                ac_layer = ac_base * cos_o - asn_base * sin_o
                asn_layer = ac_base * sin_o + asn_base * cos_o
                layer_alpha = (ac_layer, asn_layer)
            else:
                layer_alpha = alpha
            layer_temp = per_layer_temp_override[layer_idx]

            x_before_norm = norm_of(x)
            x = layer(x, loop_idx, temp_mult=layer_temp, alpha=layer_alpha)
            x_after_norm  = norm_of(x)
            delta_norm    = norm_of(x - x_in)
            print(f"    layer {layer_idx} [{tag}] T={layer_temp:.1f}: "
                  f"norm {x_before_norm:.4f} → {x_after_norm:.4f}  Δ_from_input={delta_norm:.4f}",
                  flush=True)
            Device[Device.DEFAULT].synchronize()

            if BFIELD_WAIST > 0 and layer_idx == 1 and not BFIELD_END_OF_BREATH:
                x = self.apply_bfield_waist(x)

        waist_compressed = None
        if BFIELD_WAIST > 0 and BFIELD_END_OF_BREATH:
            n_before = norm_of(x)
            if return_waist_compressed:
                x, waist_compressed = self.apply_bfield_waist(x, return_compressed=True)
            else:
                x = self.apply_bfield_waist(x)
            print(f"    waist (end-of-breath): norm {n_before:.4f} → {norm_of(x):.4f}", flush=True)
        if CONSTANT_RADIUS and not BREATH_NORM_OSC:
            n_before = norm_of(x)
            x_f = x.cast(dtypes.float)
            x_norm = (x_f.square().sum(axis=-1, keepdim=True) + 1e-6).sqrt()
            target = self.crp_target_norm
            mix = self.crp_mix_alpha
            x_proj = x_f * (target / x_norm)
            x = (x_f * (1.0 - mix) + x_proj * mix).cast(x.dtype)
            print(f"    CRP: {n_before:.4f} → {norm_of(x):.4f}  "
                  f"(mix={float(mix.numpy()[0]):.4f}, target={float(target.numpy()[0]):.4f})", flush=True)

        print(f"  --- breath {loop_idx} output norm={norm_of(x):.4f} ---", flush=True)
        if return_waist_compressed:
            return x, waist_compressed
        return x

    block.breathe_once = types.MethodType(_patched_breathe_once_v68, block)

    n_loops = 3
    print(f"\nRunning {n_loops} breaths (no integration — raw breathe_once calls):")
    x_cur = x
    for l in range(n_loops):
        x_cur = block.breathe_once(x_cur, l, temp_mult=1.0)
        Device[Device.DEFAULT].synchronize()
        if not x_cur.isfinite().all().numpy():
            print(f"\n  *** NaN/Inf detected after breath {l} — stopping ***")
            break

    print(f"\nFinal norm: {norm_of(x_cur):.4f}")
    print("\nDone.")

else:
    print(f"Unknown TASK={TASK}. Use TASK=1 or TASK=2.")
    sys.exit(1)
