"""funnel_grad_probe.py -- THE FUNNEL GATE's grad probe (2026-10-10; the waist_skip_grad_probe.py
/ selfmatch_grad_probe.py convention: eager, no TinyJit, so a bare .numpy() read never races a
JIT capture pass). Builds params fresh with ALG_FUNNEL=1 (funnel_skip_theta[K_B, 3] and
funnel_rank_theta[K_B, POLAR_D] both start at ALG_FUNNEL_INIT, default -4.0 -- LIVE from birth,
asserted below, NOT zero), forwards the fixture's first GP_ROWS rows through the SAME two-pass
(open -> build_slot_masks -> masked) do_train uses, takes ONE backward of _loss_single at the
final breath, and prints BOTH theta tables' per-breath-row gradient norms: every breath row
1..K_B-1 (the rows breath_step's loop actually indexes) must show a nonzero grad on BOTH
families -- row 0 (the grounding read's row, never read by the loop) is reported but not
asserted, the SAME standing note breath_emb's own row 0 carries (T6 clock audit S2). The
NO-GRAD FENCE inside do_train's real step() already enforces this on every real training run
(an assert naming the starved param), so this script is a diagnostic confirmation, not the
only proof.
Env: FG_CKPT (optional warm checkpoint; missing keys stay at fresh init, stated) + the family
envs (ALG_POLAR=1 ALG_POLAR_D=128 ALG_FUNNEL=1 required) + GP_ROWS (default 0:8).
"""
import os, sys
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np
import phase1_algebra_head as H
from phase1_algebra_head import (build_params, forward, load_alg, build_slot_masks,
                                  _loss_single, ident_build_array, T_ALG)
from tinygrad import Tensor, dtypes

assert int(os.environ.get("ALG_FUNNEL", "0")), \
    "ALG_FUNNEL=1 is required -- otherwise funnel_skip_theta/funnel_rank_theta never exist in p"

vs, vst, vtk, vg, vse = load_alg("train")
lo, hi = (int(x) for x in os.environ.get("GP_ROWS", "0:8").split(":"))
sl = np.arange(lo, hi); B = len(sl)

p = build_params(0)
assert "funnel_skip_theta" in p and "funnel_rank_theta" in p, \
    "ALG_FUNNEL did not register funnel_skip_theta/funnel_rank_theta"
K_B = int(os.environ.get("ALG_BREATH", "1"))
assert p["funnel_skip_theta"].shape == (K_B, 3), p["funnel_skip_theta"].shape
assert p["funnel_rank_theta"].shape == (K_B, H.POLAR_D), p["funnel_rank_theta"].shape
_init = float(os.environ.get("ALG_FUNNEL_INIT", "-4.0"))
_sk0 = p["funnel_skip_theta"].numpy(); _rk0 = p["funnel_rank_theta"].numpy()
assert np.all(_sk0 == _init), \
    f"funnel_skip_theta must be born EXACTLY at ALG_FUNNEL_INIT={_init} (saw {_sk0[:2]}...)"
assert np.all(_rk0 == _init), \
    f"funnel_rank_theta must be born EXACTLY at ALG_FUNNEL_INIT={_init} (saw {_rk0[0, :4]}...)"
_g0 = 1.0 / (1.0 + np.exp(-_init)); _r0 = 1.0 - _g0
print(f"[funnel-grad] theta born at {_init} -> g_skip = sigmoid(theta) = {_g0:.6f}, "
      f"rank r = 1 - sigmoid(theta) = {_r0:.6f} (LIVE from birth, the knob law -- not zero/one)")

ckpt = os.environ.get("FG_CKPT", "")
if ckpt:
    from tinygrad.nn.state import safe_load
    sd = safe_load(ckpt)
    _miss = [k for k in p if k not in sd]
    for k in p:
        if k in sd:
            p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
    print(f"[funnel-grad] warm from {ckpt}: {len(_miss)} params stay at fresh init: {_miss[:6]}", flush=True)
else:
    print("[funnel-grad] no FG_CKPT given -- fresh init throughout", flush=True)

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
assert "breaths" in o and len(o["breaths"]) == K_B, ("breaths", len(o.get("breaths", [])), K_B)
full = dict(o, **o["breaths"][K_B - 1])                # the final breath -- the ladder's own last word
loss = _loss_single(full, g, level=3)

for t in p.values():
    t.grad = None
loss.backward()

sk_g = p["funnel_skip_theta"].grad
rk_g = p["funnel_rank_theta"].grad
assert sk_g is not None, "funnel_skip_theta has NO gradient -- the funnel skip road is dead, not just quiet"
assert rk_g is not None, "funnel_rank_theta has NO gradient -- the funnel rank road is dead, not just quiet"
sk_gn = sk_g.numpy(); rk_gn = rk_g.numpy()
print(f"[funnel-grad] rows {lo}:{hi} loss={float(loss.numpy()):.6f}")

_all_ok_skip = True; _all_ok_rank = True
for kb in range(K_B):
    _sn = float(np.sqrt((sk_gn[kb] ** 2).sum()))
    _rn = float(np.sqrt((rk_gn[kb] ** 2).sum()))
    _tag = "(breath 0, grounding read -- never indexed by the loop, not asserted)" if kb == 0 else ""
    print(f"[funnel-grad] breath {kb}: skip_grad_norm={_sn:.6e} rank_grad_norm={_rn:.6e} {_tag}")
    if kb >= 1:
        _all_ok_skip = _all_ok_skip and (_sn > 0)
        _all_ok_rank = _all_ok_rank and (_rn > 0)

print(f"[funnel-grad] PASS ROAD SKIP (every breath 1..{K_B - 1} nonzero): {_all_ok_skip}")
print(f"[funnel-grad] PASS ROAD RANK (every breath 1..{K_B - 1} nonzero): {_all_ok_rank}")
