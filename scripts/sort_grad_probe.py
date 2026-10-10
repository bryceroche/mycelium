"""sort_grad_probe.py -- THE SORTING ROOM form 2's grad probe (2026-10-09; the selfmatch_grad_probe.py
convention: eager, no TinyJit, so a bare .numpy() read never races a JIT capture pass). Builds params
fresh with ALG_SORT=<N> (and, if ALG_SORT_KEY=1, the owner-key gain -- p["sort_key_gain"] starts at
exactly 0.0 unless ALG_SORT_KEY_GAIN overrides the init), forwards the fixture's first GP_ROWS rows
through the SAME two-pass (open -> build_slot_masks -> masked) do_train uses, takes ONE backward of
_loss_single at the final breath, and prints:
  - the sorting room's own attention/FFN params' gradient norm (proves the road is LIVE, not just
    present -- a zero-grad road would be the mandatory-road law's violation in reverse: a trained
    road nobody trains);
  - if ALG_SORT_KEY=1: sort_key_gain's gradient (must be nonzero -- the zero-at-birth law's twin,
    exactly the selfmatch precedent's sm_w_self/sm_w_arg check).
Env: SR_CKPT (optional warm checkpoint; missing keys stay at fresh init) + the family envs
(ALG_SORT=<N> required) + GP_ROWS (default 0:8).
"""
import os, sys
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np
import phase1_algebra_head as H
from phase1_algebra_head import (build_params, forward, load_alg, build_slot_masks,
                                  _loss_single, ident_build_array, T_ALG)
from tinygrad import Tensor, dtypes

assert int(os.environ.get("ALG_SORT", "0")) > 0, \
    "ALG_SORT=<N> (N>=1) is required -- otherwise sort{i}_wq/.../ffn_w2 never exist in p"

vs, vst, vtk, vg, vse = load_alg("train")
lo, hi = (int(x) for x in os.environ.get("GP_ROWS", "0:8").split(":"))
sl = np.arange(lo, hi); B = len(sl)

p = build_params(0)
assert "sort0_wq" in p, "ALG_SORT did not register the sorting room's own params"
_sort_param_names = [k for k in p if k.startswith("sort")]
print(f"[sort-grad] {len(_sort_param_names)} sorting-room tensors "
      f"({sum(int(np.prod(p[k].shape)) for k in _sort_param_names):,} params)", flush=True)
_has_key = "sort_key_gain" in p
if _has_key:
    assert float(p["sort_key_gain"].numpy()[0]) == 0.0, \
        "sort_key_gain must be EXACTLY zero at birth by default (ALG_SORT_KEY_GAIN unset)"

ckpt = os.environ.get("SR_CKPT", "")
if ckpt:
    from tinygrad.nn.state import safe_load
    sd = safe_load(ckpt)
    _miss = [k for k in p if k not in sd]
    for k in p:
        if k in sd:
            p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
    print(f"[sort-grad] warm from {ckpt}: {len(_miss)} params stay at fresh init: {_miss[:6]}", flush=True)
else:
    print("[sort-grad] no SR_CKPT given -- fresh init throughout", flush=True)

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

sort_grad_sq = 0.0
n_nonzero = 0
for k in _sort_param_names:
    gr = p[k].grad
    if gr is None:
        continue
    gv = float((gr * gr).sum().numpy())
    sort_grad_sq += gv
    if gv > 0:
        n_nonzero += 1
sort_grad_norm = sort_grad_sq ** 0.5
print(f"[sort-grad] rows {lo}:{hi} loss={float(loss.numpy()):.6f}")
print(f"[sort-grad] sorting-room grad norm={sort_grad_norm:.6e} "
      f"({n_nonzero}/{len(_sort_param_names)} tensors with a nonzero grad)")
print(f"[sort-grad] PASS ROAD: {sort_grad_norm > 0}")

if _has_key:
    kg = p["sort_key_gain"].grad
    assert kg is not None, "sort_key_gain has NO gradient -- the owner-key shuffle's road is dead"
    kgv = float(kg.numpy()[0])
    print(f"[sort-grad] sort_key_gain.grad={kgv:.6e}")
    print(f"[sort-grad] PASS KEY: {kgv != 0.0}")
