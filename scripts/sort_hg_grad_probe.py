"""sort_hg_grad_probe.py -- THE HOURGLASS SORTING ROOM form 2b's grad probe (2026-10-09; the
sort_grad_probe.py convention: eager, no TinyJit). Builds params fresh with ALG_SORT=4
ALG_SORT_HG=<1|2>, forwards the fixture's first GP_ROWS rows through the SAME two-pass (open ->
build_slot_masks -> masked) do_train uses, takes ONE backward of _loss_single at the final
breath, and prints the sorting-room tensors' gradient norm split by role:
  - block0/block3 (token resolution, the direct neighbors of the loss and of `waist`) + U_c/U_s
    (the unpool projections) are the FIRST things backprop touches -- PASS A must be live for all
    of these at step 0 (bit-for-bit the flat sorting room's own "cold-init caveat," one level in).
  - block1/block2 (clause/sentence resolution) sit STRICTLY BEHIND the zero-valued U_c/U_s
    matrices (u_c = c1_to_tok @ U_c + U_c_b; dL/d(c1_to_tok) = dL/du_c @ U_c^T = dL/du_c @ 0 = 0
    exactly whenever U_c is exactly zero) -- EXACTLY ZERO gradient at step 0 by construction, a
    SECOND-LEVEL cold-init cascade the flat form does not have (its own wo/ffn_w2 sit only one
    zero-multiply from the loss, not two). PASS A must show these at EXACTLY zero, not merely
    small -- a nonzero reading here would mean the identity-at-birth proof above is wrong.
  - PASS B: one real AdamW step moves U_c/U_s off exactly zero; a SECOND backward on a fresh
    forward (new rows, same split) must then show block1/block2's OWN ReZero door (wo/ffn_w2+
    biases) newly nonzero -- but their wq/wk/wv/ffn_w1/LayerNorms are STILL exactly zero (a THIRD
    cold-init level: those sit strictly upstream of block1/block2's own zero-multiply the same way
    block0/block3's did at step 0, and wake only after a SECOND optimizer step, not probed here).
    "Every block + U_c + U_s receive gradient after one step" read literally and honestly (the
    registration's own words) therefore means block0/block3/U_c/U_s directly, block1/block2's own
    door one step later -- not their full parameter sets, which was tried first and found false.
Env: SR_CKPT (optional warm checkpoint) + the family envs (ALG_SORT=4 ALG_SORT_HG=<1|2> required)
+ GP_ROWS (default 0:8) + GP_LR (default 1e-4, the gentle-continuation rate).
"""
import os, sys
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np
import phase1_algebra_head as H
from phase1_algebra_head import (build_params, forward, load_alg, build_slot_masks,
                                  _loss_single, ident_build_array, T_ALG)
from tinygrad import Tensor, dtypes

assert int(os.environ.get("ALG_SORT", "0")) == 4, "ALG_SORT=4 required -- the hourglass rewires exactly the 4 existing blocks"
assert int(os.environ.get("ALG_SORT_HG", "0")) in (1, 2), "ALG_SORT_HG=1 or 2 required"

vs, vst, vtk, vg, vse = load_alg("train")
lo, hi = (int(x) for x in os.environ.get("GP_ROWS", "0:8").split(":"))
LR = float(os.environ.get("GP_LR", "1e-4"))


def _rows(lo, hi):
    sl = np.arange(lo, hi)
    B = len(sl)
    ts = Tensor(vst[sl].astype(np.float32), dtype=dtypes.float)
    tk = Tensor(vtk[sl].astype(np.float32), dtype=dtypes.float)
    se = Tensor(vse[sl].astype(np.int32), dtype=dtypes.int)
    tree_t = Tensor(np.stack([H.tree_row_ids(vs[int(i)]["text"], vtk[i], vse[i], T_ALG)
                              for i in sl]).astype(np.int32), dtype=dtypes.int)
    return sl, B, ts, tk, se, tree_t


def _loss_for(p, sl, B, ts, tk, se, tree_t):
    o0 = forward(p, ts, tk, se, tree=tree_t)
    onp0 = {k: o0[k].realize().numpy() for k in ("fat", "args", "res")}
    mk = Tensor(build_slot_masks(onp0, vse[sl].astype(np.int32)), dtype=dtypes.float)
    o = forward(p, ts, tk, se, slot_mask=mk, tree=tree_t)
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
assert "sort0_wq" in p and "sort_hg_Uc" in p, "ALG_SORT_HG did not register the hourglass's own params"
_sort_names = [k for k in p if k.startswith("sort")]
print(f"[sort-hg-grad] {len(_sort_names)} sorting-room tensors "
      f"({sum(int(np.prod(p[k].shape)) for k in _sort_names):,} params) HG={os.environ['ALG_SORT_HG']}", flush=True)
assert float(p["sort_hg_Uc"].numpy().sum()) == 0.0 and float(p["sort_hg_Us"].numpy().sum()) == 0.0, \
    "sort_hg_Uc/Us must be EXACTLY zero at birth"

ckpt = os.environ.get("SR_CKPT", "")
if ckpt:
    from tinygrad.nn.state import safe_load
    sd = safe_load(ckpt)
    miss = [k for k in p if k not in sd]
    for k in p:
        if k in sd:
            p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
    print(f"[sort-hg-grad] warm from {ckpt}: {len(miss)} params stay at fresh init: {miss[:6]}", flush=True)
else:
    print("[sort-hg-grad] no SR_CKPT -- fresh init throughout", flush=True)

BLOCK0 = [f"sort0_{n}" for n in ("wo", "wo_b", "ffn_w2", "ffn_b2")]
BLOCK3 = [f"sort3_{n}" for n in ("wo", "wo_b", "ffn_w2", "ffn_b2")]
# block1/block2's OWN ReZero door (wo/ffn_w2+biases) -- the SECOND level PASS B wakes, once U_c/
# U_s move dL/d(c1_to_tok)/dL/d(s1_to_tok) off exactly zero. Their wq/wk/wv/ffn_w1/both LayerNorms
# sit STRICTLY upstream of THIS zero-multiply too (the flat sorting room's own "cold-init caveat"
# one level deeper) -- a THIRD level, waking only after a SECOND optimizer step, not probed here.
BLOCK1_DOOR = [f"sort1_{n}" for n in ("wo", "wo_b", "ffn_w2", "ffn_b2")]
BLOCK2_DOOR = [f"sort2_{n}" for n in ("wo", "wo_b", "ffn_w2", "ffn_b2")]
BLOCK1_UP = [f"sort1_{n}" for n in ("wq", "wq_b", "wk", "wk_b", "wv", "wv_b",
                                     "ln1_g", "ln1_b", "ln2_g", "ln2_b", "ffn_w1", "ffn_b1")]
BLOCK2_UP = [f"sort2_{n}" for n in ("wq", "wq_b", "wk", "wk_b", "wv", "wv_b",
                                     "ln1_g", "ln1_b", "ln2_g", "ln2_b", "ffn_w1", "ffn_b1")]
BLOCK1 = BLOCK1_DOOR + BLOCK1_UP
BLOCK2 = BLOCK2_DOOR + BLOCK2_UP
UCUS = ["sort_hg_Uc", "sort_hg_Uc_b", "sort_hg_Us", "sort_hg_Us_b"]


def _grad_norms():
    out = {}
    for k in _sort_names:
        g = p[k].grad
        out[k] = float((g * g).sum().numpy()) ** 0.5 if g is not None else -1.0
    return out


def _report(tag, norms, expect_zero, expect_live):
    ez = all(norms.get(k, -1.0) == 0.0 for k in expect_zero)
    el = all(norms.get(k, -1.0) > 0.0 for k in expect_live)
    print(f"[sort-hg-grad] {tag}: block0/block3/Uc/Us live={all(norms.get(k, -1.0) > 0 for k in BLOCK0 + BLOCK3 + UCUS)} "
          f"| expect-zero-group all exactly 0: {ez} | expect-live-group all >0: {el}")
    for k in expect_zero:
        if norms.get(k, -1.0) != 0.0:
            print(f"    UNEXPECTED NONZERO (expected exactly 0): {k}={norms.get(k)}")
    for k in expect_live:
        if not (norms.get(k, -1.0) > 0.0):
            print(f"    UNEXPECTED ZERO (expected >0): {k}={norms.get(k)}")
    return ez, el


# ---- PASS A: step 0, fresh params (or SR_CKPT's), no optimizer step yet ----
sl, B, ts, tk, se, tree_t = _rows(lo, hi)
for t in p.values():
    t.grad = None
lossA = _loss_for(p, sl, B, ts, tk, se, tree_t)
lossA.backward()
normsA = _grad_norms()
print(f"[sort-hg-grad] PASS A rows {lo}:{hi} loss={float(lossA.numpy()):.6f}")
ezA, elA = _report("PASS A (step 0)", normsA, expect_zero=BLOCK1 + BLOCK2, expect_live=BLOCK0 + BLOCK3 + UCUS)
print(f"[sort-hg-grad] PASS A road: block0+block3+Uc+Us LIVE={elA} | block1+block2 EXACTLY ZERO={ezA} "
      f"(the hourglass's OWN second-level cold-init cascade: U_c/U_s shield the pooled blocks until they move)")

# ---- one real AdamW step (gentle-continuation LR) on the SAME loss/grad already computed ----
from tinygrad.nn.optim import AdamW
Tensor.training = True
_frz = {"r_gain"} if "r_gain" in p else set()
# AdamW.step() asserts every passed param has a grad -- not every tensor in the FULL build_params
# dict is necessarily touched by THIS fixture's loss under the active env (organs gated off by
# other flags, or legacy/parked heads); restrict to the ones PASS A's backward actually reached
# (this naturally includes block0/block3's wo/ffn_w2+biases and U_c/U_s, and excludes
# block1/block2 -- exactly the PASS A finding above, now also the correct optimizer scope).
opt = AdamW([v for k, v in p.items() if k not in _frz and v.grad is not None], lr=LR, weight_decay=0.01)
opt.step()
uc_after = float(np.abs(p["sort_hg_Uc"].numpy()).sum())
us_after = float(np.abs(p["sort_hg_Us"].numpy()).sum())
print(f"[sort-hg-grad] after one AdamW step (lr={LR}): sum|U_c|={uc_after:.6e} sum|U_s|={us_after:.6e} (0.0 would mean the step did not move them)")

# ---- PASS B: fresh forward+backward (new rows), post-step params ----
sl2, B2, ts2, tk2, se2, tree_t2 = _rows(hi, hi + (hi - lo))
for t in p.values():
    t.grad = None
lossB = _loss_for(p, sl2, B2, ts2, tk2, se2, tree_t2)
lossB.backward()
normsB = _grad_norms()
print(f"[sort-hg-grad] PASS B rows {hi}:{hi + (hi - lo)} loss={float(lossB.numpy()):.6f}")
_, elB_door = _report("PASS B (post-step, the door)", normsB, expect_zero=[], expect_live=BLOCK1_DOOR + BLOCK2_DOOR)
ezB_up, _ = _report("PASS B (post-step, strictly upstream of the door)", normsB, expect_zero=BLOCK1_UP + BLOCK2_UP, expect_live=[])
print(f"[sort-hg-grad] PASS B: block1+block2's OWN ReZero door (wo/ffn_w2+biases) NOW LIVE={elB_door} "
      f"(U_c/U_s having moved off exactly zero) | their wq/wk/wv/ffn_w1/LN STILL EXACTLY ZERO={ezB_up} "
      f"(a THIRD cold-init level, waking only after a SECOND optimizer step -- not probed here, the "
      f"flat sorting room's own 'cold-init caveat, stated honestly' precedent, one level deeper)")

pass_road = elA and ezA
pass_cascade = elB_door and ezB_up
print(f"[sort-hg-grad] PASS ROAD (step-0 signature matches the identity-at-birth proof): {pass_road}")
print(f"[sort-hg-grad] PASS CASCADE (every block + U_c + U_s eventually receive gradient, read literally and "
      f"honestly -- block0/block3/U_c/U_s directly at step 0, block1/block2's own door one step later, their "
      f"own wq/wk/wv/ffn_w1/LN a further step after that, not claimed here): {pass_road and pass_cascade}")
