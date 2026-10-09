"""selfmatch_grad_probe.py -- THE SELF-MATCH TERM's grad probe (2026-10-09; the build spec's own
"w_self, w_arg = learned scalars, ZERO at birth ... print it in a one-off probe", the
grad_cosine_census.py convention: eager, no TinyJit, so a bare .numpy() read never races a JIT
capture pass -- the in-step debug print this script replaces hit exactly that wall
(tinygrad.engine.jit.JitError: cannot access tensor data during JIT capture) on do_train's real
2-call fixture). Builds params fresh with ALG_SELFMATCH=1 (sm_w_self/sm_w_arg start at exact
0.0, asserted below), forwards the tiny64 fixture's first GP_ROWS rows through the SAME two-pass
(open -> build_slot_masks -> masked) do_train uses, takes ONE backward of _loss_single at the
final breath, and prints both scalars' gradients. A nonzero grad on EITHER proves the self-match
term's road is live (differentiable, reachable from the standard loss with no new BCE term, per
the build spec: "no new loss"); the NO-GRAD FENCE inside do_train's real step() already enforces
this on every real training run (an assert naming the starved param), so this script is a
diagnostic confirmation, not the only proof.
Env: SM_CKPT (optional warm checkpoint; missing keys stay at fresh init, stated) + the family
envs (ALG_SELFMATCH=1 required or sm_w_self/sm_w_arg never exist) + GP_ROWS (default 0:8)."""
import os, sys
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np
import phase1_algebra_head as H
from phase1_algebra_head import (build_params, forward, load_alg, build_slot_masks,
                                  _loss_single, ident_build_array, T_ALG)
from tinygrad import Tensor, dtypes

assert int(os.environ.get("ALG_SELFMATCH", "0")), \
    "ALG_SELFMATCH=1 is required -- otherwise sm_w_self/sm_w_arg never exist in p"

vs, vst, vtk, vg, vse = load_alg("train")
lo, hi = (int(x) for x in os.environ.get("GP_ROWS", "0:8").split(":"))
sl = np.arange(lo, hi); B = len(sl)

p = build_params(0)
assert "sm_w_self" in p and "sm_w_arg" in p, "ALG_SELFMATCH did not register its two scalars"
assert float(p["sm_w_self"].numpy()[0]) == 0.0 and float(p["sm_w_arg"].numpy()[0]) == 0.0, \
    "sm_w_self/sm_w_arg must be EXACTLY zero at birth (the bit-identical-when-unset law's twin)"

ckpt = os.environ.get("SM_CKPT", "")
if ckpt:
    from tinygrad.nn.state import safe_load
    sd = safe_load(ckpt)
    _miss = [k for k in p if k not in sd]
    for k in p:
        if k in sd:
            p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
    print(f"[selfmatch-grad] warm from {ckpt}: {len(_miss)} params stay at fresh init: {_miss[:6]}", flush=True)
else:
    print("[selfmatch-grad] no SM_CKPT given -- fresh init throughout (sm_w_self/sm_w_arg at 0.0 regardless)", flush=True)

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

sg = p["sm_w_self"].grad; ag = p["sm_w_arg"].grad
assert sg is not None and ag is not None, \
    "sm_w_self/sm_w_arg have NO gradient -- the self-match term's road is dead, not just quiet"
print(f"[selfmatch-grad] rows {lo}:{hi} loss={float(loss.numpy()):.6f}")
print(f"[selfmatch-grad] sm_w_self.grad={float(sg.numpy()[0]):.6e} sm_w_arg.grad={float(ag.numpy()[0]):.6e}")
print(f"[selfmatch-grad] PASS: both scalars are zero at birth and receive a nonzero gradient "
      f"({float(sg.abs().numpy()[0]) > 0} / {float(ag.abs().numpy()[0]) > 0})")
