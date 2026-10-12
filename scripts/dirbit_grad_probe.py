"""dirbit_grad_probe.py — THE DIRECTION BIT's grad-norm probe (2026-10-08, zero GPU, DEV=CPU;
extended 2026-10-09 for FORM 2, ALG_DIR=2). THE TWO-TERMINAL LAW's own assert-at-the-door form,
built the way earlier one-off builders proved their terminal's grad flows before trusting a full
gate (phase1_algebra_head.py's selftest() is the precedent this ports): build_params(ALG_DIR=<mode>),
one synthetic batch with a NON-trivial arg_dir/dir_mask gold (so the BCE term is not an all-zero
no-op), forward (no dirgold passed -- FORM 2's res-mask branch is a no-op without it, by design;
this probe is about the "dir" BCE term, unchanged across modes) -> loss_fn -> backward, then print
h_dir's and h_dir_b's grad norm. Asserts > 0 — the fence this task's gate table quotes. Respects a
pre-set ALG_DIR (mode 1 or 2); defaults to 1.

usage: DEV=CPU ALG_DIR=1 .venv/bin/python3 scripts/dirbit_grad_probe.py
       DEV=CPU ALG_DIR=2 .venv/bin/python3 scripts/dirbit_grad_probe.py
"""
import os
os.environ.setdefault("DEV", "CPU")
os.environ.setdefault("ALG_DIR", "1")
import numpy as np
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase1_algebra_head import (build_params, forward, loss_fn, L_FAC, K_VARS,
                                  T_ALG, H_TRUNK)
from tinygrad import Tensor, dtypes

_MODE = os.environ["ALG_DIR"]
p = build_params(0)
assert "h_dir" in p and "h_dir_b" in p, f"ALG_DIR={_MODE} did not allocate h_dir/h_dir_b"
B = 2
rng = np.random.RandomState(0)
o = forward(p, Tensor(rng.randn(B, T_ALG, H_TRUNK).astype(np.float32) * .1),
            Tensor(np.ones((B, T_ALG), np.float32)),
            Tensor(np.zeros((B, T_ALG), np.int32), dtype=dtypes.int))
assert "dir" in o, f"ALG_DIR={_MODE} but _heads_of emitted no 'dir' key"

# a synthetic row: factor 0 forward (dir=0), factor 1 inverse (dir=1), both
# relation slots dir_mask=1 (cleanly classified) -- a non-trivial gold so
# the BCE term is not an all-zero no-op that would pass a grad check for
# the wrong reason.
arg_dir = np.zeros((B, L_FAC), np.float32); arg_dir[:, 1] = 1.0
dir_mask = np.zeros((B, L_FAC), np.float32); dir_mask[:, 0] = 1.0; dir_mask[:, 1] = 1.0
is_rel = np.zeros((B, L_FAC), np.float32); is_rel[:, 0] = 1.0; is_rel[:, 1] = 1.0
presence = np.zeros((B, L_FAC), np.float32); presence[:, :2] = 1.0

g = {"presence": Tensor(presence),
     "is_lit_f": Tensor(np.zeros((B, L_FAC), np.float32)),
     "args": Tensor(np.zeros((B, L_FAC, K_VARS), np.float32)),
     "fspan": Tensor(np.ones((B, L_FAC, T_ALG), np.float32)),
     "vspan": Tensor(np.ones((B, K_VARS, T_ALG), np.float32)),
     "ftype": Tensor(np.zeros((B, L_FAC), np.int32), dtype=dtypes.int),
     "op": Tensor(np.zeros((B, L_FAC), np.int32), dtype=dtypes.int),
     "res": Tensor(np.zeros((B, L_FAC), np.int32), dtype=dtypes.int),
     "digits": Tensor(np.zeros((B, L_FAC, 3), np.int32), dtype=dtypes.int),
     "query": Tensor(np.zeros((B,), np.int32), dtype=dtypes.int),
     "is_rel": Tensor(is_rel),
     "arg_dir": Tensor(arg_dir),
     "dir_mask": Tensor(dir_mask)}

l = loss_fn(o, g)
# NOTE (2026-10-08, this tinygrad install): calling l.numpy() BEFORE
# l.backward() yields grad=None on EVERY param, not just h_dir/h_dir_b --
# reproduced on the UNMODIFIED selftest() in both the main tree and this
# worktree (realize-via-numpy detaches the graph on read in the currently
# installed tinygrad; selftest()'s own `lv = float(l.numpy()); ...;
# l.backward()` order is itself broken by this, pre-existing, not
# introduced here). backward() FIRST, read the loss value after.
l.backward()
lv = float(l.numpy())
assert np.isfinite(lv), f"loss not finite: {lv}"
gn_w = float(p["h_dir"].grad.abs().max().numpy()) if p["h_dir"].grad is not None else -1.0
gn_b = float(p["h_dir_b"].grad.abs().max().numpy()) if p["h_dir_b"].grad is not None else -1.0
print(f"[dirbit-grad-probe] ALG_DIR={_MODE} loss={lv:.4f}  h_dir.grad.abs().max()={gn_w:.6g}  "
      f"h_dir_b.grad.abs().max()={gn_b:.6g}")
assert gn_w > 0, "h_dir grad is exactly zero -- the two-terminal law's door failed"
assert gn_b > 0, "h_dir_b grad is exactly zero -- the two-terminal law's door failed"
print(f"[dirbit-grad-probe] OK (ALG_DIR={_MODE}): h_dir/h_dir_b grads flow (> 0)")
