"""sort_feed_grad_probe.py -- THE SORTING ROOM'S FEEDBACK FORM's grad probe (SRF_241,
2026-10-10; the sort_hg_grad_probe.py / grad_census.py conventions combined, eager, no TinyJit).
Two claims, both read off the SAME fixture and ckpt:

(1) THE REZERO CASCADE (one level, the flat sorting room's own precedent, applied to the NEW
    cross-attention branch in all four blocks): at step 0, sort{i}_xwo(+bias) is the first thing
    backprop touches from the live residual gradient and is LIVE (nonzero); sort{i}_xwq/xwk/xwv
    and the pre-cross LayerNorm sort{i}_lnx_g/_b sit STRICTLY upstream of that same zero multiply
    and get EXACTLY zero gradient at this first step, waking at step 1 once sort{i}_xwo moves
    off zero (PASS A below). PASS B (one real AdamW step later) must show them live.

(2) THE SLOT-SIDE GRADIENT (the arm's own "two-terminal" requirement: the cross-attention
    branch's keys/values are NOT detached from `cur`, so the slots' own loss gradient should
    reach them through this branch too, once it wakes): THE GRADIENT TAP (H._SRF_GTAP, the
    grad_census.py idiom) adds a zero, requires_grad leaf onto the branch's own kx/vx at a named
    breath, isolating dL/d(cross) THROUGH THIS BRANCH ALONE from `cur`'s other (mixer) gradient
    paths. Reads EXACTLY zero at step 0 (ReZero, see (1)) and nonzero at PASS B -- "the cross-
    attention branch live after one step; the slots receive gradient through the keys," read
    literally.

Env: SR_CKPT (optional warm checkpoint) + the family envs (ALG_SORT=4 ALG_SORT_FEED=1 required)
+ GP_ROWS (default 0:8) + GP_LR (default 1e-4) + GP_KB (breath to tap, default 1).
"""
import os, sys
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np
import phase1_algebra_head as H
from phase1_algebra_head import (build_params, forward, load_alg, build_slot_masks,
                                  _loss_single, H_W, N_HEADS, L_FAC)
from tinygrad import Tensor, dtypes

assert int(os.environ.get("ALG_SORT", "0")) == 4, "ALG_SORT=4 required"
assert int(os.environ.get("ALG_SORT_FEED", "0")) == 1, "ALG_SORT_FEED=1 required -- the cross-attention branch this probe reads"

vs, vst, vtk, vg, vse = load_alg("train")
lo, hi = (int(x) for x in os.environ.get("GP_ROWS", "0:8").split(":"))
LR = float(os.environ.get("GP_LR", "1e-4"))
GP_KB = int(os.environ.get("GP_KB", "1"))


def _rows(lo, hi):
    sl = np.arange(lo, hi)
    B = len(sl)
    ts = Tensor(vst[sl].astype(np.float32), dtype=dtypes.float)
    tk = Tensor(vtk[sl].astype(np.float32), dtype=dtypes.float)
    se = Tensor(vse[sl].astype(np.int32), dtype=dtypes.int)
    return sl, B, ts, tk, se


def _loss_for(p, sl, B, ts, tk, se):
    o0 = forward(p, ts, tk, se)
    onp0 = {k: o0[k].realize().numpy() for k in ("fat", "args", "res")}
    mk = Tensor(build_slot_masks(onp0, vse[sl].astype(np.int32)), dtype=dtypes.float)
    o = forward(p, ts, tk, se, slot_mask=mk)
    K = int(os.environ.get("ALG_BREATH", "7"))
    assert "breaths" in o and len(o["breaths"]) == K
    full = dict(o, **o["breaths"][K - 1])
    g = {}
    for k in vg:
        a = vg[k][sl]
        g[k] = Tensor(a.astype(np.float32) if a.dtype.kind == "f" else a.astype(np.int32),
                      dtype=dtypes.float if a.dtype.kind == "f" else dtypes.int)
    if "is_lit_f" not in g and "is_lit" in g:
        g["is_lit_f"] = g["is_lit"]
    return _loss_single(full, g, level=3)


p = build_params(0)
assert "sort0_xwq" in p and "sort0_xwo" in p, "ALG_SORT_FEED did not register the cross-attention branch's own params"
_srf_names = [k for k in p if k.startswith("sort") and ("xw" in k or "lnx" in k)]
print(f"[sort-feed-grad] {len(_srf_names)} cross-attention tensors "
      f"({sum(int(np.prod(p[k].shape)) for k in _srf_names):,} params)", flush=True)
for i in range(int(os.environ["ALG_SORT"])):
    assert float(np.abs(p[f"sort{i}_xwo"].numpy()).sum()) == 0.0, f"sort{i}_xwo must be EXACTLY zero at birth"

ckpt = os.environ.get("SR_CKPT", "")
if ckpt:
    from tinygrad.nn.state import safe_load
    sd = safe_load(ckpt)
    miss = [k for k in p if k not in sd]
    for k in p:
        if k in sd:
            p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
    print(f"[sort-feed-grad] warm from {ckpt}: {len(miss)} params stay at fresh init: {miss[:6]}", flush=True)
else:
    print("[sort-feed-grad] no SR_CKPT -- fresh init throughout", flush=True)

N_BLOCKS = int(os.environ["ALG_SORT"])
XWO = [f"sort{i}_{n}" for i in range(N_BLOCKS) for n in ("xwo", "xwo_b")]
XW_UP = [f"sort{i}_{n}" for i in range(N_BLOCKS)
         for n in ("xwq", "xwq_b", "xwk", "xwk_b", "xwv", "xwv_b", "lnx_g", "lnx_b")]


def _grad_norms():
    out = {}
    for k in _srf_names:
        g = p[k].grad
        out[k] = float((g * g).sum().numpy()) ** 0.5 if g is not None else -1.0
    return out


def _report(tag, norms, expect_zero, expect_live):
    ez = all(norms.get(k, -1.0) == 0.0 for k in expect_zero)
    el = all(norms.get(k, -1.0) > 0.0 for k in expect_live)
    print(f"[sort-feed-grad] {tag}: expect-zero-group all exactly 0: {ez} | expect-live-group all >0: {el}")
    for k in expect_zero:
        if norms.get(k, -1.0) != 0.0:
            print(f"    UNEXPECTED NONZERO (expected exactly 0): {k}={norms.get(k)}")
    for k in expect_live:
        if not (norms.get(k, -1.0) > 0.0):
            print(f"    UNEXPECTED ZERO (expected >0): {k}={norms.get(k)}")
    return ez, el


def _gn(t):
    return 0.0 if t.grad is None else float((t.grad.detach() ** 2).sum().numpy()) ** 0.5


# ---- PASS A: step 0, fresh/warm params, no optimizer step yet -- THE GRADIENT TAP armed at GP_KB.
# ONE leaf, shaped like `cross` itself (B, L_FAC, H_W) -- not the post-reshape per-head kx/vx --
# added in _sort_block BEFORE either the key or value projection reads `cross`, so the leaf's
# own gradient after backward() is dL/d(cross) through this branch, whichever of kx/vx carries
# it (a tap added AFTER kx is already computed from the OLD `cross` is dead code for kx's own
# path -- found the hard way, fixed in _sort_block, not re-introduced here). ----
sl, B, ts, tk, se = _rows(lo, hi)
tap_leaf = Tensor(np.zeros((B, L_FAC, H_W), np.float32), requires_grad=True)
H._SRF_GTAP = {GP_KB: tap_leaf}
for t in p.values():
    t.grad = None
lossA = _loss_for(p, sl, B, ts, tk, se)
lossA.backward()
normsA = _grad_norms()
print(f"[sort-feed-grad] PASS A rows {lo}:{hi} loss={float(lossA.numpy()):.6f}")
ezA, elA = _report("PASS A (step 0)", normsA, expect_zero=XW_UP, expect_live=XWO)
tap_A = _gn(tap_leaf)
print(f"[sort-feed-grad] PASS A gradient tap (breath {GP_KB}): ||dL/d(cross leaf)||={tap_A:.6e} "
      f"(0.0 expected -- ReZero's own cascade: sort{{i}}_xwo is exactly zero, so dL/d(cross)=0 "
      f"through this branch at step 0)")
print(f"[sort-feed-grad] PASS A road: xwo(+b) LIVE={elA} | xwq/xwk/xwv/lnx EXACTLY ZERO={ezA} "
      f"(the flat sorting room's own cold-init cascade, applied to the cross-attention branch)")

# ---- one real AdamW step (gentle-continuation LR) on the SAME loss/grad already computed ----
from tinygrad.nn.optim import AdamW
Tensor.training = True
_frz = {"r_gain"} if "r_gain" in p else set()
opt = AdamW([v for k, v in p.items() if k not in _frz and v.grad is not None], lr=LR, weight_decay=0.01)
opt.step()
xwo_after = {k: float(np.abs(p[k].numpy()).sum()) for k in XWO if k.endswith("xwo")}
print(f"[sort-feed-grad] after one AdamW step (lr={LR}): sum|xwo| per block = {xwo_after} "
      f"(0.0 anywhere would mean the step did not move that block's door)")

# ---- PASS B: fresh forward+backward (new rows), post-step params, a FRESH gradient tap at the SAME breath ----
sl2, B2, ts2, tk2, se2 = _rows(hi, hi + (hi - lo))
tap_leaf2 = Tensor(np.zeros((B2, L_FAC, H_W), np.float32), requires_grad=True)
H._SRF_GTAP = {GP_KB: tap_leaf2}
for t in p.values():
    t.grad = None
lossB = _loss_for(p, sl2, B2, ts2, tk2, se2)
lossB.backward()
normsB = _grad_norms()
print(f"[sort-feed-grad] PASS B rows {hi}:{hi + (hi - lo)} loss={float(lossB.numpy()):.6f}")
_, elB = _report("PASS B (post-step)", normsB, expect_zero=[], expect_live=XW_UP + XWO)
tap_B = _gn(tap_leaf2)
print(f"[sort-feed-grad] PASS B gradient tap (breath {GP_KB}): ||dL/d(cross leaf)||={tap_B:.6e} "
      f"(nonzero expected -- the branch has woken)")
H._SRF_GTAP = None

pass_road = elA and ezA
pass_tap_zero_at_birth = (tap_A == 0.0)
pass_tap_live_after = (tap_B > 0.0)
print(f"[sort-feed-grad] PASS ROAD (step-0 ReZero signature matches the identity-at-birth proof): {pass_road}")
print(f"[sort-feed-grad] PASS TAP (gradient tap exactly 0 at step 0, nonzero after one step -- "
      f"'the cross-attention branch live after one step; the slots receive gradient through the "
      f"keys', read literally): {pass_tap_zero_at_birth and pass_tap_live_after}")
print(f"[sort-feed-grad] PASS CASCADE (every block's xwq/xwk/xwv/lnx now live post-step): {elB}")
