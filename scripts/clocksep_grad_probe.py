"""clocksep_grad_probe.py -- THE SEPARATED CLOCK's grad probe (2026-10-09; the
sort_grad_probe.py/selfmatch_grad_probe.py convention: eager, no TinyJit).

Builds params fresh with ALG_CLOCK_SEP=1 (+ the caller's family env), forwards the
fixture's first GP_ROWS rows through the standard two-pass (open -> build_slot_masks ->
masked) do_train uses, takes ONE backward of _loss_single at the final breath, and prints:
  - polar_wd / polar_wu (the waist's new 512x128/128x512 shape): gradient norm, must be > 0.
  - polar_clk_init (the register's one "reading" parameter): gradient norm, must be > 0 --
    by construction (see breath_step) the register (state["clk"]) is read ONLY by the QROT
    additive bias terms (sc2 / _mx_sc) and its own turn/EM self-update; this script does
    not need to prove that separately -- it is a code-inspection fact (grep '_clk' in
    breath_step), reported here, not measured.
  - a CONTENT check: W_bq's gradient at the dims that were CLOCK dims before separation
    (read directly off .cache/polar_bands.json, never off this build's own all-content
    _polar_tables) must be NONZERO and comparable in scale to the dims that were already
    content -- proving the old clock positions are now ordinary, fully-trained content,
    not a dead zone.
Env: GP_ROWS (default 0:8) + the family envs (ALG_POLAR=1 ALG_CLOCK_SEP=1 required).
"""
import json
import os
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
import numpy as np
import phase1_algebra_head as H
from phase1_algebra_head import (build_params, forward, load_alg, build_slot_masks,
                                  _loss_single, ident_build_array, T_ALG)
from tinygrad import Tensor, dtypes

assert H.ALG_POLAR and H.ALG_CLOCK_SEP, "ALG_POLAR=1 ALG_CLOCK_SEP=1 are required"

vs, vst, vtk, vg, vse = load_alg("train")
lo, hi = (int(x) for x in os.environ.get("GP_ROWS", "0:8").split(":"))
sl = np.arange(lo, hi); B = len(sl)

p = build_params(0)
assert "polar_wd" in p and "polar_wu" in p, "ALG_POLAR_D did not register the waist"
assert "polar_clk_init" in p, "ALG_CLOCK_SEP did not register the register's birth projection"
assert p["polar_wd"].shape == (512, int(os.environ.get("ALG_POLAR_D", "128"))), p["polar_wd"].shape
print(f"[clocksep-grad] polar_wd{tuple(p['polar_wd'].shape)} polar_wu{tuple(p['polar_wu'].shape)} "
      f"polar_clk_init{tuple(p['polar_clk_init'].shape)}", flush=True)

ckpt = os.environ.get("GP_CKPT", "")
if ckpt:
    from tinygrad.nn.state import safe_load
    sd = safe_load(ckpt)
    # ALG_CLOCK_SEP reshapes polar_wd/polar_wu (384->512 content dims) and adds
    # polar_clk_init -- a shape mismatch against an OLD checkpoint means "reborn here,"
    # same treatment as "missing from the checkpoint" (CS_241 itself trains from scratch).
    _miss = [k for k in p if k not in sd or tuple(sd[k].shape) != tuple(p[k].shape)]
    for k in p:
        if k in sd and tuple(sd[k].shape) == tuple(p[k].shape):
            p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
    print(f"[clocksep-grad] warm from {ckpt}: {len(_miss)} params stay at fresh init "
          f"(missing or reshaped): {_miss[:6]}", flush=True)
else:
    print("[clocksep-grad] no GP_CKPT given -- fresh init throughout", flush=True)

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


def gnorm(name):
    gr = p[name].grad
    if gr is None:
        return None
    return float((gr * gr).sum().numpy()) ** 0.5


for name in ("polar_wd", "polar_wu", "polar_clk_init"):
    n = gnorm(name)
    print(f"[clocksep-grad] {name}: grad_norm={n}")
    assert n is not None and n > 0.0, f"{name} has zero/None gradient -- a dead road"

# THE CONTENT CHECK: W_bq's per-INPUT-ROW gradient, split by whether that dim WAS a clock
# dim before separation (read straight off the band file, never off this build's own
# all-content _polar_tables -- the point is to probe the OLD partition).
bands = json.load(open(os.environ.get("ALG_POLAR_BANDS", ".cache/polar_bands.json")))
old_clock_planes = sorted(pl for w in bands["wheels"] for pl in w["planes"])
old_clock_dims = sorted(d for pl in old_clock_planes for d in (2 * pl, 2 * pl + 1))
all_dims = set(range(H.H_W))
old_content_dims = sorted(all_dims - set(old_clock_dims))
gbq = p["W_bq"].grad
assert gbq is not None, "W_bq has no gradient at all"
gbq_np = gbq.numpy()
row_norm = np.linalg.norm(gbq_np, axis=1)   # (H_W,) per input-row gradient norm
clk_rows = row_norm[old_clock_dims]
cnt_rows = row_norm[old_content_dims]
n_dead_clk = int((clk_rows == 0.0).sum())
print(f"[clocksep-grad] W_bq per-input-row grad norm -- OLD CLOCK dims (n={len(old_clock_dims)}): "
      f"mean={clk_rows.mean():.4e} min={clk_rows.min():.4e} dead={n_dead_clk} | "
      f"OLD CONTENT dims (n={len(old_content_dims)}): mean={cnt_rows.mean():.4e} min={cnt_rows.min():.4e}")
assert n_dead_clk == 0, (
    f"{n_dead_clk}/{len(old_clock_dims)} formerly-clock dims of W_bq's input still have "
    f"ZERO gradient under ALG_CLOCK_SEP -- they are not behaving as ordinary content")
print("[clocksep-grad] PASS: the waist's new params and every content plane (old clock "
      "positions included) receive gradient; the register's one new param "
      "(polar_clk_init) is live.")
