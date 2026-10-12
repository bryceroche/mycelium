"""mixer_grad_probe.py -- THE TYPED MIXER / SHOAL MIXER / MIXER SEVERED grad probe
(2026-10-10; the depth_grad_probe.py / paircmp_grad_probe.py convention: eager, no TinyJit, so
a bare .numpy() read never races a JIT capture pass).

Builds params fresh (optionally warm from SR_CKPT), forwards the fixture's first GP_ROWS rows
through the SAME two-pass (open -> build_slot_masks -> masked) do_train uses, takes ONE
backward of _loss_single at the final breath, and reports:

  MX_ARM=typed  -- ALG_TYPED_MIXER=1 must already be exported by the caller. W_bq/W_bk/W_bv/
                   W_bo must each receive a NONZERO gradient (the road is live, not a dead
                   mask multiply).
  MX_ARM=shoal  -- ALG_SHOAL_MIXER=<k> must already be exported. Same check.
  MX_ARM=mixoff -- ALG_MIXER_OFF=1 must already be exported. W_bq/W_bk/W_bv/W_bo (and, when
                   present, mh_wo/fed_mx_hg) must receive an EXACTLY ZERO (defined, not None)
                   gradient, AND at least one other param (waist_w) must receive a nonzero
                   gradient -- the None-grad lesson: every param stays in the graph with a
                   real forward pass; only the CONTRIBUTION is zero-multiplied away.

Env: SR_CKPT (optional warm checkpoint), GP_ROWS (default 0:8), MX_ARM (required), plus the
family envs (role8/SURF8 etc.) exported by the caller, matching the gate fixture exactly.
"""
import os, sys
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np
import phase1_algebra_head as H
from phase1_algebra_head import (build_params, forward, load_alg, build_slot_masks,
                                  _loss_single, ident_build_array, T_ALG)
from tinygrad import Tensor, dtypes

ARM = os.environ.get("MX_ARM", "")
assert ARM in ("typed", "shoal", "mixoff"), f"MX_ARM must be typed/shoal/mixoff, got {ARM!r}"
if ARM == "typed":
    assert int(os.environ.get("ALG_TYPED_MIXER", "0")), "MX_ARM=typed needs ALG_TYPED_MIXER=1 exported"
    WATCH = ("W_bq", "W_bk", "W_bv", "W_bo")
    EXPECT_NONZERO = True
elif ARM == "shoal":
    assert int(os.environ.get("ALG_SHOAL_MIXER", "0")), "MX_ARM=shoal needs ALG_SHOAL_MIXER=<k> exported"
    WATCH = ("W_bq", "W_bk", "W_bv", "W_bo")
    EXPECT_NONZERO = True
else:
    assert int(os.environ.get("ALG_MIXER_OFF", "0")), "MX_ARM=mixoff needs ALG_MIXER_OFF=1 exported"
    WATCH = ("W_bq", "W_bk", "W_bv", "W_bo")
    EXPECT_NONZERO = False

vs, vst, vtk, vg, vse = load_alg("train")
lo, hi = (int(x) for x in os.environ.get("GP_ROWS", "0:8").split(":"))
sl = np.arange(lo, hi); B = len(sl)

p = build_params(0)
for extra in ("mh_wo", "fed_mx_hg", "alt_g"):
    if extra in p:
        WATCH = WATCH + (extra,)

ckpt = os.environ.get("SR_CKPT", "")
if ckpt:
    from tinygrad.nn.state import safe_load
    sd = safe_load(ckpt)
    _miss = [k for k in p if k not in sd]
    for k in p:
        if k in sd:
            p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
    print(f"[mixer-grad] ARM={ARM} warm from {ckpt}: {len(_miss)} params stay at fresh init: {_miss[:6]}", flush=True)
else:
    print(f"[mixer-grad] ARM={ARM} no SR_CKPT given -- fresh init throughout", flush=True)

ts = Tensor(vst[sl].astype(np.float32), dtype=dtypes.float)
tk = Tensor(vtk[sl].astype(np.float32), dtype=dtypes.float)
se = Tensor(vse[sl].astype(np.int32), dtype=dtypes.int)
_idt = Tensor(ident_build_array([vs[int(i)] for i in sl], T_ALG), dtype=dtypes.int) \
    if (H.ALG_BUSREG or H.ALG_IDKEY) else None

o0 = forward(p, ts, tk, se, ident=_idt)
onp0 = {k: o0[k].realize().numpy() for k in ("fat", "args", "res")}
mk = Tensor(build_slot_masks(onp0, vse[sl].astype(np.int32)), dtype=dtypes.float)

g = {}
for k in vg:
    a = vg[k][sl]
    g[k] = Tensor(a.astype(np.float32) if a.dtype.kind == "f" else a.astype(np.int32),
                  dtype=dtypes.float if a.dtype.kind == "f" else dtypes.int)
if "is_lit_f" not in g and "is_lit" in g:
    g["is_lit_f"] = g["is_lit"]

o = forward(p, ts, tk, se, slot_mask=mk, ident=_idt)
K = int(os.environ.get("ALG_BREATH", "7"))
assert "breaths" in o and len(o["breaths"]) == K, ("breaths", len(o.get("breaths", [])), K)
full = dict(o, **o["breaths"][K - 1])
loss = _loss_single(full, g, level=3)

for t in p.values():
    t.grad = None
loss.backward()

print(f"[mixer-grad] rows {lo}:{hi} loss={float(loss.numpy()):.6f}")

watch_ok = True
for k in WATCH:
    gr = p[k].grad
    if gr is None:
        print(f"[mixer-grad] {k}: grad=None  <-- the None-grad lesson, FAIL")
        watch_ok = False
        continue
    gv = float((gr * gr).sum().numpy())
    is_zero = gv == 0.0
    ok = (not is_zero) if EXPECT_NONZERO else is_zero
    watch_ok = watch_ok and ok
    print(f"[mixer-grad] {k}: grad_sq={gv:.6e} ({'nonzero' if not is_zero else 'EXACT ZERO'}) "
          f"-> {'OK' if ok else 'MISMATCH'} (expected {'nonzero' if EXPECT_NONZERO else 'exact zero'})")

other_nonzero = 0
other_total = 0
for k, t in p.items():
    if k in WATCH:
        continue
    other_total += 1
    gr = t.grad
    if gr is not None and float((gr * gr).sum().numpy()) > 0.0:
        other_nonzero += 1
print(f"[mixer-grad] other params: {other_nonzero}/{other_total} with a nonzero grad "
      f"(at least one other param moving is required for MX_ARM=mixoff)")
other_ok = (other_nonzero > 0) if ARM == "mixoff" else True

print(f"[mixer-grad] ARM={ARM} PASS WATCH: {watch_ok}  PASS OTHER: {other_ok}  "
      f"PASS ALL: {watch_ok and other_ok}")
