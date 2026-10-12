"""depth_grad_probe.py -- THE LOOPED TRANSFORMER's grad probe (2026-10-09; the sort_grad_probe.py
convention: eager, no TinyJit, so a bare .numpy() read never races a JIT capture pass). Builds
params fresh with ALG_DEPTH=<N>, forwards the fixture's first GP_ROWS rows through the SAME
two-pass (open -> build_slot_masks -> masked) do_train uses, takes ONE backward of _loss_single
at the final breath, and prints the stack's own dp{i}_* params' gradient norm (proves the road is
LIVE, not just present) plus, per block, which tensors are nonzero at step 0 (the ReZero
signature: ONLY wo/xwo/ffn_w2 + their biases should be nonzero -- everything else sits strictly
upstream of a zero multiply and gets EXACTLY zero gradient at this first step, by construction).
Env: SR_CKPT (optional warm checkpoint; missing keys stay at fresh init) + the family envs
(ALG_DEPTH=<N> required) + GP_ROWS (default 0:8).
"""
import os, sys
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np
import phase1_algebra_head as H
from phase1_algebra_head import (build_params, forward, load_alg, build_slot_masks,
                                  _loss_single, ident_build_array, T_ALG)
from tinygrad import Tensor, dtypes

assert int(os.environ.get("ALG_DEPTH", "0")) > 0, \
    "ALG_DEPTH=<N> (N>=1) is required -- otherwise dp0_wq/.../ffn_w2 never exist in p"

vs, vst, vtk, vg, vse = load_alg("train")
lo, hi = (int(x) for x in os.environ.get("GP_ROWS", "0:8").split(":"))
sl = np.arange(lo, hi); B = len(sl)

p = build_params(0)
assert "dp0_wq" in p, "ALG_DEPTH did not register the looped transformer's own params"
_dp_names = [k for k in p if k.startswith("dp")]
print(f"[depth-grad] {len(_dp_names)} looped-transformer tensors "
      f"({sum(int(np.prod(p[k].shape)) for k in _dp_names):,} params)", flush=True)

ckpt = os.environ.get("SR_CKPT", "")
if ckpt:
    from tinygrad.nn.state import safe_load
    sd = safe_load(ckpt)
    _miss = [k for k in p if k not in sd]
    for k in p:
        if k in sd:
            p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
    print(f"[depth-grad] warm from {ckpt}: {len(_miss)} params stay at fresh init: {_miss[:6]}", flush=True)
else:
    print("[depth-grad] no SR_CKPT given -- fresh init throughout", flush=True)

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

dp_grad_sq = 0.0
n_nonzero = 0
per_block_nonzero = {}
for k in _dp_names:
    gr = p[k].grad
    if gr is None:
        continue
    gv = float((gr * gr).sum().numpy())
    dp_grad_sq += gv
    blk = k.split("_", 1)[0]   # "dp0", "dp1", ...
    per_block_nonzero.setdefault(blk, []).append((k, gv))
    if gv > 0:
        n_nonzero += 1
dp_grad_norm = dp_grad_sq ** 0.5
print(f"[depth-grad] rows {lo}:{hi} loss={float(loss.numpy()):.6f}")
print(f"[depth-grad] looped-transformer grad norm={dp_grad_norm:.6e} "
      f"({n_nonzero}/{len(_dp_names)} tensors with a nonzero grad)")
print(f"[depth-grad] PASS ROAD: {dp_grad_norm > 0}")

EXPECT_NONZERO_AT_STEP0 = ("wo", "wo_b", "xwo", "xwo_b", "ffn_w2", "ffn_b2")
rezero_ok = True
for blk, items in sorted(per_block_nonzero.items()):
    nz = sorted(k.split("_", 1)[1] for k, gv in items if gv > 0)
    expect = sorted(EXPECT_NONZERO_AT_STEP0)
    ok = (nz == expect)
    rezero_ok = rezero_ok and ok
    print(f"[depth-grad] {blk}: nonzero-at-step0={nz} (expected {expect}) -> {'OK' if ok else 'MISMATCH'}")
print(f"[depth-grad] PASS REZERO: {rezero_ok}")
