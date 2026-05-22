"""Patch v66 step 3000 ckpt for v68 TWO_PHASE warm-start.

Mapping:
  v66 'shared.{a}'        → v68 'expand_shared.{a}'  (Set A V/O/FFN-out)
  v66 'shared.{a}'        → v68 'compress_shared.{a}' (Set B cloned — specialize via training)
  v66 'phase{i}.{a}'      → v68 'expand_phase{i}.{a}'  (i=0..3, full EXPAND stack from v66)
  v66 'phase{2|3}.{a}'    → v68 'compress_phase{0|1}.{a}' (last 2 phases cloned for COMPRESS)
  Other keys (boundary_head, notebook, controller, codebook, lookup_table) → unchanged

v66's per_head_pitch is shape (4, n_heads). v68's is (6, n_heads). The patch script
DROPS v66's pitch from the new ckpt — v68 model uses default v23a init for all 6 positions.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tinygrad.nn.state import safe_load, safe_save

SRC = ".cache/gsm8k_steps_ckpts/v66_sched_sampling_step3000.safetensors"
DST = ".cache/gsm8k_steps_ckpts/v66_step3000_two_phase.safetensors"

SHARED_ATTRS = ("wv", "bv", "wo", "bo", "w_out", "b_out",
                "in_ln_g", "in_ln_b", "post_ln_g", "post_ln_b")
PHASE_ATTRS = ("wq", "bq", "wk", "bk", "w_in", "b_in")

print(f"Loading {SRC}")
sd = safe_load(SRC)
print(f"  {len(sd)} keys")

new_sd = {}

# 1. EXPAND: Set A's 4 layers — clone of v66's 4 phase layers + shared weights
for attr in SHARED_ATTRS:
    new_sd[f"expand_shared.{attr}"] = sd[f"shared.{attr}"]
for i in range(4):
    for attr in PHASE_ATTRS:
        new_sd[f"expand_phase{i}.{attr}"] = sd[f"phase{i}.{attr}"]

# 2. COMPRESS: Set B's 2 layers — clone of v66's shared + phases 2,3 (later/distillation)
for attr in SHARED_ATTRS:
    new_sd[f"compress_shared.{attr}"] = sd[f"shared.{attr}"]
for i in range(2):
    src_phase = i + 2  # phase 2 → compress 0, phase 3 → compress 1
    for attr in PHASE_ATTRS:
        new_sd[f"compress_phase{i}.{attr}"] = sd[f"phase{src_phase}.{attr}"]

# 3. All other keys (notebook, boundary, controller, codebook, lookup, embed, ln_f, etc.) pass through.
#    EXCLUDE the old 'shared.*' and 'phase*.*' keys since v68 model has different keys for those.
#    Keep them in the new ckpt anyway — load_state_dict with strict=False will ignore unrecognized keys.
n_pass = 0
for k, v in sd.items():
    if k.startswith("shared.") or (k.startswith("phase") and not k.startswith("phases_")):
        # These map to expand/compress already; pass through too (model still has self.shared and self.layers)
        new_sd[k] = v
        n_pass += 1
    else:
        new_sd[k] = v
        n_pass += 1

print(f"Passed through {n_pass} v66 keys")
print(f"  New expand_shared keys: {sum(1 for k in new_sd if k.startswith('expand_shared.'))}")
print(f"  New compress_shared keys: {sum(1 for k in new_sd if k.startswith('compress_shared.'))}")
print(f"  New expand_phase keys: {sum(1 for k in new_sd if k.startswith('expand_phase'))}")
print(f"  New compress_phase keys: {sum(1 for k in new_sd if k.startswith('compress_phase'))}")
print(f"  Total keys in new ckpt: {len(new_sd)}")

safe_save(new_sd, DST)
src_size = os.path.getsize(SRC) / 1e9
dst_size = os.path.getsize(DST) / 1e9
print(f"\nSaved {DST}")
print(f"  size: {src_size:.2f}GB → {dst_size:.2f}GB (delta from new expand/compress shared + phase clones)")
