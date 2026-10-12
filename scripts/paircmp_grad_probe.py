"""scripts/paircmp_grad_probe.py -- THE PAIRWISE COMPARATOR ROAD's grad probe (2026-10-09; the
SELFMATCH grad probe's own convention: eager, no TinyJit, so a bare .numpy() read never races a
JIT capture pass -- the self-match term's in-step debug print hit exactly that wall before this
pattern was adopted). Builds params fresh with ALG_PAIRCMP=1 (pc_w1/pc_b1 small-random with the
head's seed, pc_w2/pc_b2 ALSO small-random per the coordinator's post-SM_241 knob-law correction
-- NOT zero-init; bit-identity to the unset arm comes from ALG_PAIRCMP=0 never allocating these
params, not from a zero output layer), forwards the tiny64 fixture's first GP_ROWS rows through
the SAME two-pass (open -> build_slot_masks -> masked) do_train uses, takes ONE backward of
_loss_single at the final breath, and prints both the hidden and output layers' grad norms. A
nonzero grad on EITHER proves the comparator's road is live and reachable from the standard loss
with no new BCE term (per the build spec: "no new loss").
Env: PC_CKPT (optional warm checkpoint; missing keys stay at fresh init, stated) + the family
envs (ALG_PAIRCMP=1 required or pc_w1/pc_b1/pc_w2/pc_b2 never exist) + GP_ROWS (default 0:8)."""
import os, sys
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np
import phase1_algebra_head as H
from phase1_algebra_head import (build_params, forward, load_alg, build_slot_masks,
                                  _loss_single, ident_build_array, T_ALG)
from tinygrad import Tensor, dtypes

assert int(os.environ.get("ALG_PAIRCMP", "0")), \
    "ALG_PAIRCMP=1 is required -- otherwise pc_w1/pc_b1/pc_w2/pc_b2 never exist in p"

vs, vst, vtk, vg, vse = load_alg("train")
lo, hi = (int(x) for x in os.environ.get("GP_ROWS", "0:8").split(":"))
sl = np.arange(lo, hi); B = len(sl)

p = build_params(0)
for k in ("pc_w1", "pc_b1", "pc_w2", "pc_b2"):
    assert k in p, f"ALG_PAIRCMP did not register {k}"
print(f"[paircmp-grad] pc_w1 std={float(p['pc_w1'].numpy().std()):.6e} "
      f"pc_w2 std={float(p['pc_w2'].numpy().std()):.6e} (both small-random, NOT zero -- the "
      f"knob-law correction after SM_241's autopsy)", flush=True)

ckpt = os.environ.get("PC_CKPT", "")
if ckpt:
    from tinygrad.nn.state import safe_load
    sd = safe_load(ckpt)
    _miss = [k for k in p if k not in sd]
    for k in p:
        if k in sd:
            p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
    print(f"[paircmp-grad] warm from {ckpt}: {len(_miss)} params stay at fresh init: {_miss[:6]}", flush=True)
else:
    print("[paircmp-grad] no PC_CKPT given -- fresh init throughout", flush=True)

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

g1, gb1, g2, gb2 = (p["pc_w1"].grad, p["pc_b1"].grad, p["pc_w2"].grad, p["pc_b2"].grad)
assert g1 is not None and g2 is not None, \
    "pc_w1/pc_w2 have NO gradient -- the comparator's road is dead, not just quiet"
n1, n2 = float(g1.numpy().std()), float(g2.numpy().std())
nb1 = float(np.linalg.norm(gb1.numpy())) if gb1 is not None else float("nan")
nb2 = float(gb2.numpy()[0]) if gb2 is not None else float("nan")
print(f"[paircmp-grad] rows {lo}:{hi} loss={float(loss.numpy()):.6f}")
print(f"[paircmp-grad] pc_w1.grad std={n1:.6e} pc_w2.grad std={n2:.6e} "
      f"pc_b1.grad norm={nb1:.6e} pc_b2.grad={nb2:.6e}")
print(f"[paircmp-grad] PASS: both layers receive a nonzero gradient "
      f"(pc_w1: {n1 > 0} / pc_w2: {n2 > 0})")
