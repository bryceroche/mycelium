"""complex_fence.py -- THE COMPLEX-TENSOR FENCE (2026-09-24; registered 09-23). The head's own rotate
(phase1_algebra_head.rot2_interleaved — the clock frame, the router's unbind, the garage's snap, the
register all run through it) is asserted equal to the bus's formal language (mycelium/complex_tensor.py:
tg_rotate on tinygrad, and bind/unbind on numpy complex64 by the layout law) on random tensors. Runs in
every CPU gate; a drift between the specification and the implementation fails loudly. Zero GPU.
usage: DEV=CPU .venv/bin/python3 scripts/complex_fence.py"""
import os, sys
os.environ.setdefault("DEV", "CPU")
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np
from tinygrad import Tensor
from mycelium import complex_tensor as CT
from phase1_algebra_head import rot2_interleaved

rng = np.random.default_rng(0x2b1d)
P = 256
v = rng.standard_normal((4, 24, 2 * P)).astype(np.float32)
theta = rng.uniform(0, 2 * np.pi, P).astype(np.float32)
c, s = np.cos(theta).astype(np.float32), np.sin(theta).astype(np.float32)

head = rot2_interleaved(Tensor(v), Tensor(c), Tensor(s)).numpy()
lib_tg = CT.tg_rotate(Tensor(v), Tensor(c), Tensor(s)).numpy()
lib_np = CT.lower(CT.bind(CT.lift(v), CT.phasor(theta)))
back = rot2_interleaved(Tensor(head), Tensor(c), Tensor(-s)).numpy()      # unbind: the conjugate

d1 = float(np.abs(head - lib_tg).max()); d2 = float(np.abs(head - lib_np).max()); d3 = float(np.abs(back - v).max())
mod = float(np.abs(np.linalg.norm(head.reshape(4, 24, P, 2), axis=-1) - np.linalg.norm(v.reshape(4, 24, P, 2), axis=-1)).max())
print(f"[complex-fence] head vs tg_rotate max|d| {d1:.2e} | head vs numpy bind max|d| {d2:.2e} | unbind(bind(v)) - v max|d| {d3:.2e} | modulus drift {mod:.2e}")
assert d1 < 1e-5 and d2 < 1e-4 and d3 < 1e-4 and mod < 1e-4, "[complex-fence] FAILED: the head's rotate drifted from the bus's formal language"
print("[complex-fence] OK: the specification and the implementation agree")
