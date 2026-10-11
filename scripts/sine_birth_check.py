"""sine_birth_check.py -- is THE SINE BREATH's stack the identity at ReZero birth?

At birth every block's wo / xwo / ffn_w2 is zero, so a residual block returns its input and the
stack must return `cur` (that is what the gate's identity proof claims). Read-only: imports the head
as a library, calls _depth_stack_sine directly on random states, prints ||out|| / ||cur||.

Run from the repo root, once per skip mode (the head reads ALG_* at import):
  for sk in peak none boundary; do
    ALG_SINE_SKIP=$sk ALG_SINE=sine ALG_DEPTH=4 ALG_POLAR=1 ALG_POLAR_D=128 ALG_BREATH=7 DEV=CPU \
      .venv/bin/python3 sine_birth_check.py
  done
Observed (2026-10-11, origin/main 0c823f5; a stand-in orthonormal waist init, irrelevant at birth):
  peak      kb=1: 2.000x   kb=2: 2.000x   (out == 2*cur exactly)
  none      kb=1: 1.000x   kb=2: 1.000x   (exact identity)
  boundary  kb=1: 1.000x   kb=2: 2.000x   (the carry is the previous output, which already rides in cur)
"""
import os
import sys

sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np
import phase1_algebra_head as H
from tinygrad import Tensor

p = H.build_params(0)
rng = np.random.default_rng(0)
B, L, T = 2, 48, 256
cur = Tensor(rng.standard_normal((B, L, H.H_W)).astype(np.float32))
waist = Tensor(rng.standard_normal((B, T, H.H_W)).astype(np.float32))
tokmask = Tensor(np.ones((B, T), np.float32))
slot_mask = Tensor(np.ones((B, L, L), np.float32))
state = {}
c = cur.numpy()
for kb in (1, 2):
    out = H._depth_stack_sine(p, cur, slot_mask, waist, tokmask, B, kb, state).numpy()
    print(f"[birth] ALG_SINE={H.ALG_SINE} skip={H.ALG_SINE_SKIP} kb={kb}: "
          f"||out||/||cur|| = {np.linalg.norm(out) / np.linalg.norm(c):.4f}   "
          f"out == cur: {np.allclose(out, c, atol=1e-6)}   out == 2*cur: {np.allclose(out, 2 * c, atol=1e-5)}")
