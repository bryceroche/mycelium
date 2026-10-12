"""waist_skip_grad_probe.py -- THE WAIST SKIP, FORM A's grad probe (2026-10-09; the
selfmatch_grad_probe.py / sort_grad_probe.py convention: eager, no TinyJit, so a bare
.numpy() read never races a JIT capture pass). Builds params fresh with ALG_WAIST_SKIP=1
(waist_skip_theta starts at ALG_WAIST_SKIP_INIT, default -4.0 -- LIVE from birth, asserted
below, NOT zero), forwards the fixture's first GP_ROWS rows through the SAME two-pass (open
-> build_slot_masks -> masked) do_train uses, takes ONE backward of _loss_single at the final
breath, and prints waist_skip_theta's gradient norm: a nonzero grad proves the bypass's road
is live (differentiable, reachable from the standard loss with no new BCE term, per the build
spec "no new loss") -- the NO-GRAD FENCE inside do_train's real step() already enforces this on
every real training run (an assert naming the starved param), so this script is a diagnostic
confirmation, not the only proof.
Env: WS_CKPT (optional warm checkpoint; missing keys stay at fresh init, stated) + the family
envs (ALG_POLAR=1 ALG_POLAR_D=128 ALG_WAIST_SKIP=1 required) + GP_ROWS (default 0:8).
"""
import os, sys
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np
import phase1_algebra_head as H
from phase1_algebra_head import (build_params, forward, load_alg, build_slot_masks,
                                  _loss_single, ident_build_array, T_ALG)
from tinygrad import Tensor, dtypes

assert int(os.environ.get("ALG_WAIST_SKIP", "0")), \
    "ALG_WAIST_SKIP=1 is required -- otherwise waist_skip_theta never exists in p"

vs, vst, vtk, vg, vse = load_alg("train")
lo, hi = (int(x) for x in os.environ.get("GP_ROWS", "0:8").split(":"))
sl = np.arange(lo, hi); B = len(sl)

p = build_params(0)
assert "waist_skip_theta" in p, "ALG_WAIST_SKIP did not register waist_skip_theta"
_init = float(os.environ.get("ALG_WAIST_SKIP_INIT", "-4.0"))
_theta0 = p["waist_skip_theta"].numpy()
assert np.all(_theta0 == _init), \
    f"waist_skip_theta must be born EXACTLY at ALG_WAIST_SKIP_INIT={_init} (saw {_theta0[:4]}...)"
_g0 = 1.0 / (1.0 + np.exp(-_init))
print(f"[waistskip-grad] theta born at {_init} -> g = sigmoid(theta) = {_g0:.6f} "
      f"(LIVE from birth, the knob law -- not zero)")

ckpt = os.environ.get("WS_CKPT", "")
if ckpt:
    from tinygrad.nn.state import safe_load
    sd = safe_load(ckpt)
    _miss = [k for k in p if k not in sd]
    for k in p:
        if k in sd:
            p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
    print(f"[waistskip-grad] warm from {ckpt}: {len(_miss)} params stay at fresh init: {_miss[:6]}", flush=True)
else:
    print("[waistskip-grad] no WS_CKPT given -- fresh init throughout", flush=True)

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

o = forward(p, ts, tk, se, slot_mask=mk, ident=_idt)   # fresh graph, a backward consumes it
K = int(os.environ.get("ALG_BREATH", "7"))
assert "breaths" in o and len(o["breaths"]) == K, ("breaths", len(o.get("breaths", [])), K)
full = dict(o, **o["breaths"][K - 1])                  # the final breath -- the ladder's own last word
loss = _loss_single(full, g, level=3)

for t in p.values():
    t.grad = None
loss.backward()

tg = p["waist_skip_theta"].grad
assert tg is not None, \
    "waist_skip_theta has NO gradient -- the waist skip's road is dead, not just quiet"
tgn = tg.numpy()
norm = float(np.sqrt((tgn * tgn).sum()))
n_nonzero = int((tgn != 0.0).sum())
print(f"[waistskip-grad] rows {lo}:{hi} loss={float(loss.numpy()):.6f}")
print(f"[waistskip-grad] waist_skip_theta.grad norm={norm:.6e} "
      f"({n_nonzero}/{tgn.size} dims with a nonzero grad)")
print(f"[waistskip-grad] PASS ROAD: {norm > 0}")
