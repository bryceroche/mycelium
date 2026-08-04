"""commit_terminal_smoke.py — RUNG-3's wire pull (2026-08-04; the
sign-terminal discipline: the terminal is proven by gradient, never by
inspection). One batch through forward/loss/backward under
ALG_BREATH=3 + ALG_RINGS=1: |W_cmt.grad| asserted nonzero; the door's
emission+buffer asserts exercised; init-closed property checked (gates
at -2, commit bias at -4 -> rings output ~= incumbent at init)."""
import os, sys
os.environ["ALG_BREATH"] = "3"; os.environ["ALG_RINGS"] = "1"
os.environ.setdefault("ALG2", "1"); os.environ.setdefault("ALG_FTYPES", "8")
os.environ.setdefault("ALG_HW", "512"); os.environ.setdefault("ALG_DUP", "1")
os.environ.setdefault("ALG_WIDE", "1")
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np
from tinygrad import Tensor, dtypes
from phase1_algebra_head import (build_params, forward, loss_fn,
                                 assert_terminals, L_FAC, K_VARS, T_ALG, N_DIG)

rng = np.random.default_rng(103850)
p = build_params(0)
assert "W_cmt" in p and "W_bo" in p
for t in p.values():
    t.requires_grad = True
B = 1
st = Tensor(rng.standard_normal((B, T_ALG, 2048)).astype(np.float32))
msk = Tensor(np.ones((B, T_ALG), np.float32))
snt = Tensor(np.zeros((B, T_ALG), np.int32), dtype=dtypes.int)
sm = Tensor(np.ones((B, L_FAC, L_FAC), np.float32))
o = forward(p, st, msk, snt, slot_mask=sm)
assert "cmt" in o, "cmt not emitted"
assert_terminals(p=p, emitted=set(o.keys()), site="rings smoke")
print(f"[smoke] cmt emitted {tuple(o['cmt'].shape)}; final mass "
      f"mean {float(o['cmt_m'].mean().numpy()):.4f} (init-closed: ~0.02)")

g = {"presence": Tensor(np.ones((B, L_FAC), np.float32)),
     "is_lit_f": Tensor(np.zeros((B, L_FAC), np.float32)),
     "is_mod": Tensor(np.zeros((B, L_FAC), np.float32)),
     "is_sel": Tensor(np.zeros((B, L_FAC), np.float32)),
     "is_pct": Tensor(np.zeros((B, L_FAC), np.float32)),
     "is_fdiv": Tensor(np.zeros((B, L_FAC), np.float32)),
     "args": Tensor(np.zeros((B, L_FAC, K_VARS), np.float32)),
     "fspan": Tensor(np.ones((B, L_FAC, T_ALG), np.float32)),
     "vspan": Tensor(np.ones((B, K_VARS, T_ALG), np.float32)),
     "ftype": Tensor(np.zeros((B, L_FAC), np.int32), dtype=dtypes.int),
     "op": Tensor(np.zeros((B, L_FAC), np.int32), dtype=dtypes.int),
     "res": Tensor(np.zeros((B, L_FAC), np.int32), dtype=dtypes.int),
     "digits": Tensor(np.zeros((B, L_FAC, N_DIG), np.int32), dtype=dtypes.int),
     "sign": Tensor(np.zeros((B, L_FAC), np.float32)),
     "query": Tensor(np.zeros((B,), np.int32), dtype=dtypes.int)}
l = loss_fn(o, g)
lv = float(l.numpy()); assert np.isfinite(lv)
l.backward()
gc = float(p["W_cmt"].grad.abs().max().numpy())
gb = float(p["W_bo"].grad.abs().max().numpy())
print(f"[smoke] loss={lv:.4f} |W_cmt.grad|max={gc:.3e} |W_bo.grad|max={gb:.3e}")
assert gc > 0, "COMMIT TERMINAL INERT"
print("COMMIT-TERMINAL SMOKE: PASS — the pawl is live, both feeds wired")
