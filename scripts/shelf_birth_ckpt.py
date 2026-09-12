"""Write a birth checkpoint for the shelf readout: balV242's trained keys + the
fresh (birth) sh_* params at the env's ALG_SHELF_B0, so the eval loader (which
hard-errors on key mismatch) can read the UNTRAINED shelf. usage:
ALG_SHELF=1 ALG_SHELF_B0=5 <family env> shelf_birth_ckpt.py src.safetensors dst.safetensors"""
import sys; sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import phase1_algebra_head as H
from tinygrad.nn.state import safe_load, safe_save
src, dst = sys.argv[1], sys.argv[2]
p = H.build_params(0); sd = safe_load(src); n = 0
for k in p:
    if k in sd and tuple(sd[k].shape) == tuple(p[k].shape):
        p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize(); n += 1
fresh = [k for k in p if k not in sd]
assert all(k.startswith("sh_") for k in fresh), f"unexpected fresh keys {fresh}"
safe_save(p, dst); print(f"[shelf-birth] {n}/{len(p)} keys from {src}; fresh {fresh} (B0={H.ALG_SHELF_B0}) -> {dst}")
