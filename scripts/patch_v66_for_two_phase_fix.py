"""Patched v66 → v68 warm-start: Fix B-revised (zero-init compress output projections).

Diagnosis:
  - v68 norm explosion is NOT a warm-start incompatibility (compress weights working sensibly
    on layer-3-output). The compress layers at T=0.7 add ~8-9 norm units per layer — same order
    as expand layers at T=2.0.
  - The fundamental issue: 6 layers total (4 expand + 2 compress) accumulate too much norm
    because CRP (crp_mix_alpha=0.055) barely constrains it.

Fix B-revised strategy:
  - compress_shared.wo (attn output projection) → ZEROS
  - compress_shared.b_out (FFN output bias) → ZEROS
  - compress_shared.w_out (FFN output projection) → ZEROS
  - All other compress_shared attrs (wv, bv, in_ln_g, in_ln_b, post_ln_g, post_ln_b) → cloned from v66
  - compress_phase{0,1}.{wq,bq,wk,bk,w_in,b_in} → cloned from v66 phase{2,3} (unchanged)

Net effect: At init, compress layers are identity (zero residual contribution). Gradient
builds up wo/w_out over training, gradually enabling the compress path. This is exactly
the LoRA-style zero-init approach that v11, v12, v24c etc. validated for new components.

Also sets crp_mix_alpha=0.3 (was 0.055) to give CRP a head-start on the larger 6-layer path.
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tinygrad.nn.state import safe_load, safe_save
from tinygrad import Tensor, dtypes

SRC = "/home/bryce/mycelium/.cache/gsm8k_steps_ckpts/v66_sched_sampling_step3000.safetensors"
DST = "/home/bryce/mycelium/.cache/gsm8k_steps_ckpts/v66_step3000_two_phase_fixb.safetensors"

SHARED_ATTRS = ("wv", "bv", "wo", "bo", "w_out", "b_out",
                "in_ln_g", "in_ln_b", "post_ln_g", "post_ln_b")
PHASE_ATTRS = ("wq", "bq", "wk", "bk", "w_in", "b_in")

# Attrs we zero-init in compress_shared (output projections → zero residual contribution at init)
COMPRESS_ZERO_ATTRS = ("wo", "bo", "w_out", "b_out")

print(f"Loading {SRC}")
sd = safe_load(SRC)
print(f"  {len(sd)} keys")

new_sd = {}

# 1. EXPAND: exact clone of v66 (unchanged)
for attr in SHARED_ATTRS:
    new_sd[f"expand_shared.{attr}"] = sd[f"shared.{attr}"]
for i in range(4):
    for attr in PHASE_ATTRS:
        new_sd[f"expand_phase{i}.{attr}"] = sd[f"phase{i}.{attr}"]

# 2. COMPRESS: V/K/LNs from v66, output projections ZEROED
for attr in SHARED_ATTRS:
    if attr in COMPRESS_ZERO_ATTRS:
        # Zero-init: same shape as the v66 tensor but all zeros
        t = sd[f"shared.{attr}"]
        print(f"  compress_shared.{attr}: ZERO-INIT (shape {t.shape}, dtype {t.dtype})")
        new_sd[f"compress_shared.{attr}"] = Tensor.zeros(t.shape, dtype=t.dtype).contiguous()
    else:
        new_sd[f"compress_shared.{attr}"] = sd[f"shared.{attr}"]

for i in range(2):
    src_phase = i + 2  # phase 2 → compress 0, phase 3 → compress 1
    for attr in PHASE_ATTRS:
        new_sd[f"compress_phase{i}.{attr}"] = sd[f"phase{src_phase}.{attr}"]

# 3. All other keys pass through.
n_pass = 0
for k, v in sd.items():
    new_sd[k] = v
    n_pass += 1

# 4. Boost crp_mix_alpha: v66's was trained to 0.055 (barely does anything).
#    With 6 layers, CRP needs to be more active. Set to 0.3.
if "block.crp_mix_alpha" in new_sd:
    old_val = float(new_sd["block.crp_mix_alpha"].numpy()[0])
    new_sd["block.crp_mix_alpha"] = Tensor([0.3], dtype=dtypes.float).contiguous()
    print(f"  crp_mix_alpha: {old_val:.4f} → 0.3 (boosted for 6-layer path)")
elif "crp_mix_alpha" in new_sd:
    old_val = float(new_sd["crp_mix_alpha"].numpy()[0])
    new_sd["crp_mix_alpha"] = Tensor([0.3], dtype=dtypes.float).contiguous()
    print(f"  crp_mix_alpha: {old_val:.4f} → 0.3 (boosted for 6-layer path)")
else:
    print("  WARNING: crp_mix_alpha key not found in ckpt — CRP boost skipped")

print(f"\nPassed through {n_pass} v66 keys + added expand/compress blocks")
print(f"  New expand_shared keys: {sum(1 for k in new_sd if k.startswith('expand_shared.'))}")
print(f"  New compress_shared keys: {sum(1 for k in new_sd if k.startswith('compress_shared.'))}")
print(f"  New expand_phase keys: {sum(1 for k in new_sd if k.startswith('expand_phase'))}")
print(f"  New compress_phase keys: {sum(1 for k in new_sd if k.startswith('compress_phase'))}")
print(f"  Total keys in new ckpt: {len(new_sd)}")

safe_save(new_sd, DST)
src_size = os.path.getsize(SRC) / 1e9
dst_size = os.path.getsize(DST) / 1e9
print(f"\nSaved {DST}")
print(f"  size: {src_size:.2f}GB → {dst_size:.2f}GB")
print("\nFix B-revised: compress output projections zero-initialized.")
print("Compress layers start as identity; gradient builds up wo/w_out over training.")
