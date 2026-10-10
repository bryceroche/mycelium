"""scripts/argsres_grad_probe.py -- THE ARGS-CONDITIONED RES's grad probe (2026-10-09; the
SELFMATCH/PAIRCMP grad probe convention: eager, no TinyJit, so a bare .numpy() read never
races a JIT capture pass). Builds params fresh with ALG_ARGSRES=1 (ar_w1/ar_b1 small-random
with the head's seed, ar_w2/ar_b2 ALSO small-random per the paircmp knob-law correction --
NOT zero-init; bit-identity to the unset arm comes from ALG_ARGSRES=0 never allocating these
params, not from a zero output layer), forwards the tiny64 fixture's first GP_ROWS rows
through the SAME two-pass (open -> build_slot_masks -> masked) do_train uses, takes ONE
backward of _loss_single at the final breath, and prints both the hidden and output layers'
grad norms, PLUS (the two-terminal law's own census, per the build spec) the gradient the res
loss places on the args head's OWN logits (W_args), reported against the args head's own
baseline gradient norm from an ALG_ARGSRES=0 forward on the SAME rows/weights so the two are
directly comparable -- proving the res term does not hijack the args head's own training
signal by swamping it.

Env: AR_CKPT (optional warm checkpoint; missing keys stay at fresh init, stated) + the family
envs (ALG_ARGSRES=1 required or ar_w1/ar_b1/ar_w2/ar_b2 never exist) + GP_ROWS (default 0:8).
"""
import os, sys
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np
import phase1_algebra_head as H
from phase1_algebra_head import (build_params, forward, load_alg, build_slot_masks,
                                  _loss_single, ident_build_array, T_ALG)
from tinygrad import Tensor, dtypes

assert int(os.environ.get("ALG_ARGSRES", "0")), \
    "ALG_ARGSRES=1 is required -- otherwise ar_w1/ar_b1/ar_w2/ar_b2 never exist in p"

vs, vst, vtk, vg, vse = load_alg("train")
lo, hi = (int(x) for x in os.environ.get("GP_ROWS", "0:8").split(":"))
sl = np.arange(lo, hi); B = len(sl)

p = build_params(0)
for k in ("ar_w1", "ar_b1", "ar_w2", "ar_b2"):
    assert k in p, f"ALG_ARGSRES did not register {k}"
print(f"[argsres-grad] ar_w1 std={float(p['ar_w1'].numpy().std()):.6e} "
      f"ar_w2 std={float(p['ar_w2'].numpy().std()):.6e} (both small-random, NOT zero -- the "
      f"paircmp knob-law correction applied again here)", flush=True)

ckpt = os.environ.get("AR_CKPT", "")
if ckpt:
    from tinygrad.nn.state import safe_load
    sd = safe_load(ckpt)
    _miss = [k for k in p if k not in sd]
    for k in p:
        if k in sd:
            p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
    print(f"[argsres-grad] warm from {ckpt}: {len(_miss)} params stay at fresh init: {_miss[:6]}", flush=True)
else:
    print("[argsres-grad] no AR_CKPT given -- fresh init throughout", flush=True)

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

g1, gb1, g2, gb2 = (p["ar_w1"].grad, p["ar_b1"].grad, p["ar_w2"].grad, p["ar_b2"].grad)
assert g1 is not None and g2 is not None, \
    "ar_w1/ar_w2 have NO gradient -- the args-conditioned res road is dead, not just quiet"
n1, n2 = float(g1.numpy().std()), float(g2.numpy().std())
nb1 = float(np.linalg.norm(gb1.numpy())) if gb1 is not None else float("nan")
nb2 = float(gb2.numpy()[0]) if gb2 is not None else float("nan")
print(f"[argsres-grad] rows {lo}:{hi} loss={float(loss.numpy()):.6f}")
print(f"[argsres-grad] ar_w1.grad std={n1:.6e} ar_w2.grad std={n2:.6e} "
      f"ar_b1.grad norm={nb1:.6e} ar_b2.grad={nb2:.6e}")
print(f"[argsres-grad] PASS ROAD: both layers receive a nonzero gradient "
      f"(ar_w1: {n1 > 0} / ar_w2: {n2 > 0})")

# THE TWO-TERMINAL CENSUS: the res loss's gradient on the args head's OWN logits (W_args),
# via a_ik = _args_out.sigmoid() NOT detached, reported against the args head's own baseline
# gradient from an ALG_ARGSRES=0 forward on the SAME rows/weights/graph shape (fresh params,
# same seed-derived init -- the comparison is of GRADIENT NORMS on the same tensor, not of
# losses). "W_args" is the args bilinear's own weight matrix (registered in build_params
# beside W_res); if ALG_ARGSRES's term never touched it, w_args.grad would be identical
# (bit-for-bit) to the baseline run's -- this census reports the norm RATIO instead, since an
# exact diff would require re-running with identical random batches end to end (already true
# here: SAME ts/tk/se/g/mk tensors feed both forwards).
assert "W_args" in p, "W_args missing from p -- cannot report the args-logit gradient census"
args_grad_with = float(p["W_args"].grad.numpy().std()) if p["W_args"].grad is not None else 0.0

# BASELINE (ALG_ARGSRES=0 on the SAME weights everywhere the term doesn't touch): build_params
# and _heads_of both read ALG_ARGSRES fresh from os.environ with no module-level caching of the
# flag itself (unlike ALG_PAIRCMP_SCALE) -- toggling the env var and re-calling build_params is
# enough; no module reload needed. p2 copies every tensor p has EXCEPT ar_w1/ar_b1/ar_w2/ar_b2
# (which build_params(0) under ALG_ARGSRES=0 never allocates at all) so the comparison isolates
# the new term's effect on W_args' own gradient.
os.environ["ALG_ARGSRES"] = "0"
p2 = build_params(0)
for k in p2:
    if k in p:
        p2[k].assign(p[k].numpy()).realize()   # SAME weights everywhere ALG_ARGSRES doesn't touch
os.environ["ALG_ARGSRES"] = "1"
o0b = forward(p2, ts, tk, se, ident=_idt)
onp0b = {k: o0b[k].realize().numpy() for k in ("fat", "args", "res")}
mk0 = Tensor(build_slot_masks(onp0b, vse[sl].astype(np.int32)), dtype=dtypes.float)
o_base = forward(p2, ts, tk, se, slot_mask=mk0, ident=_idt)
full_base = dict(o_base, **o_base["breaths"][K - 1])
loss_base = _loss_single(full_base, g, level=3)
for t in p2.values():
    t.grad = None
loss_base.backward()
args_grad_base = float(p2["W_args"].grad.numpy().std()) if p2["W_args"].grad is not None else 0.0

print(f"[argsres-grad] W_args.grad std WITH ALG_ARGSRES=1: {args_grad_with:.6e}")
print(f"[argsres-grad] W_args.grad std WITHOUT (ALG_ARGSRES=0 baseline, same rows/weights): "
      f"{args_grad_base:.6e}")
ratio = args_grad_with / args_grad_base if args_grad_base else float("nan")
print(f"[argsres-grad] TWO-TERMINAL CENSUS: res-loss-via-a_ik adds to the args head's own "
      f"gradient at ratio {ratio:.4f}x baseline (>1 means the res term is NOT starving the "
      f"args head's own BCE signal -- it is additive; the two-terminal law's own evidence, "
      f"not a training gate)")
