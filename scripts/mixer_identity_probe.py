"""mixer_identity_probe.py -- THE TYPED MIXER / THE SHOAL MIXER identity proof (2026-10-10).

Unlike THE LOOPED TRANSFORMER's identity probe (which compares two DIFFERENT architectures,
since ALG_DEPTH genuinely replaces the mixer's output BY VALUE), this organ's two new roads
read their env flags fresh on every call (os.environ.get inside _mixer_edge_mask /
_mixer_typed_edges / _mixer_shoal_edges -- never cached at import time), so ONE process can
toggle them between forward() calls on the SAME params/inputs/slot_mask and expect TRUE
bit-identity to the baseline (role8, both flags unset):

  typed-allones:  ALG_TYPED_MIXER=1 ALG_TYPED_ALLONES=1  -- the gate's own test hook, forces
                  _mixer_typed_edges to return None (the TRUE no-op, not an "all-ones tensor"
                  multiplied through -- _mixer_apply_edges skips the multiply+renormalize
                  entirely when E is None, so there is no float32 ULP drift to chase).
  shoal24:        ALG_SHOAL_MIXER=24 -- k >= L_TOT (24) also returns None from
                  _mixer_shoal_edges for the identical reason.

Both must therefore be EXACTLY (maxdiff 0.0) bit-identical to the role8 baseline -- not
"close", not "within float tolerance". If this probe reports any nonzero diff, the None-vs-
ones design note was violated somewhere (check for a stray `.maximum(eye)` or similar applied
even when E is None).

Env: SR_CKPT (optional warm checkpoint, default .cache/sharp_balV242.safetensors), GP_ROWS
(default 0:8), plus the family envs (role8/SURF8 etc.) exported by the caller -- this script
never sets ALG_TYPED_MIXER/ALG_SHOAL_MIXER/ALG_TYPED_ALLONES itself at import time (it flips
os.environ between calls, in-process, since those flags are read fresh every call).
"""
import os, sys
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np
import phase1_algebra_head as H
from phase1_algebra_head import (build_params, forward, load_alg, build_slot_masks,
                                  ident_build_array, T_ALG)
from tinygrad import Tensor, dtypes

vs, vst, vtk, vg, vse = load_alg("train")
lo, hi = (int(x) for x in os.environ.get("GP_ROWS", "0:8").split(":"))
sl = np.arange(lo, hi); B = len(sl)

p = build_params(0)
ckpt = os.environ.get("SR_CKPT", ".cache/sharp_balV242.safetensors")
if ckpt and os.path.exists(ckpt):
    from tinygrad.nn.state import safe_load
    sd = safe_load(ckpt)
    for k in p:
        if k in sd and tuple(sd[k].shape) == tuple(p[k].shape):
            p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
    print(f"[mixer-identity] warm from {ckpt}")
else:
    print(f"[mixer-identity] fresh init throughout (ckpt {ckpt!r} not found)")

ts = Tensor(vst[sl].astype(np.float32), dtype=dtypes.float)
tk = Tensor(vtk[sl].astype(np.float32), dtype=dtypes.float)
se = Tensor(vse[sl].astype(np.int32), dtype=dtypes.int)
_idt = Tensor(ident_build_array([vs[int(i)] for i in sl], T_ALG), dtype=dtypes.int) \
    if (H.ALG_BUSREG or H.ALG_IDKEY) else None

o0 = forward(p, ts, tk, se, ident=_idt)
onp0 = {k: o0[k].realize().numpy() for k in ("fat", "args", "res")}
mk = Tensor(build_slot_masks(onp0, vse[sl].astype(np.int32)), dtype=dtypes.float)


def dump():
    o = forward(p, ts, tk, se, slot_mask=mk, ident=_idt)
    out = {}
    for k, v in o.items():
        if k == "breaths":
            continue
        if hasattr(v, "numpy"):
            out[k] = v.realize().numpy()
    if "breaths" in o:
        for bi, ob in enumerate(o["breaths"]):
            for k, v in ob.items():
                if hasattr(v, "numpy"):
                    out[f"breath{bi}__{k}"] = v.realize().numpy()
    return out


def diff(a, b, tag):
    common = sorted(set(a) & set(b))
    only_a = sorted(set(a) - set(b)); only_b = sorted(set(b) - set(a))
    maxd = 0.0; worst = None
    for k in common:
        if a[k].size == 0:
            continue
        d = float(np.abs(a[k].astype("float64") - b[k].astype("float64")).max())
        if d > maxd:
            maxd = d; worst = k
    print(f"[mixer-identity] {tag}: maxdiff={maxd:.3e} over {len(common)} common keys "
          f"(only_a={only_a} only_b={only_b}, worst={worst})")
    print(f"[mixer-identity] {tag} PASS: {maxd == 0.0}")
    return maxd == 0.0


for k in ("ALG_TYPED_MIXER", "ALG_TYPED_ALLONES", "ALG_SHOAL_MIXER", "ALG_MIXER_OFF"):
    os.environ.pop(k, None)
base = dump()
print("[mixer-identity] baseline (role8, no mixer flags) dumped", len(base), "tensors")

os.environ["ALG_TYPED_MIXER"] = "1"
os.environ["ALG_TYPED_ALLONES"] = "1"
typed_allones = dump()
ok1 = diff(base, typed_allones, "typed-allones vs role8")
os.environ.pop("ALG_TYPED_MIXER", None)
os.environ.pop("ALG_TYPED_ALLONES", None)

os.environ["ALG_SHOAL_MIXER"] = str(H.L_TOT)
shoal24 = dump()
ok2 = diff(base, shoal24, "shoal24 vs role8")
os.environ.pop("ALG_SHOAL_MIXER", None)

print(f"[mixer-identity] ALL PASS: {ok1 and ok2}")
