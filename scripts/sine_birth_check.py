"""sine_birth_check.py -- THE SINE BREATH's own identity-at-birth gate step (2026-10-11, after
the skip bug: ledger 08:52). Calls `_depth_stack_sine` DIRECTLY on random states (not through
forward(), which silently never runs the breath loop at all without a slot_mask -- the exact
reason the original gate's depth_identity_probe.py-based "0.0 diff" was vacuous: see the ledger
entry and the module comment above ALG_SINE's own definition in phase1_algebra_head.py). At
ReZero birth every block's wo/xwo/ffn_w2 is zero, so the stack must return `cur` bit-for-bit
unchanged at EVERY breath, under THE ARM'S OWN env -- this is now the only accepted proof of
identity-at-birth; a probe's forward()-level dump diff is not (the gate rule this bug writes).

Run once per arm's exact env (reads ALG_* at import):
  env <SINE4's own FAM+S8+SINEX> .venv/bin/python3 scripts/sine_birth_check.py
  env <ANTI4's own FAM+S8+ANTIX> .venv/bin/python3 scripts/sine_birth_check.py
  env <SEAL's own FAM+S8+SEALX, ALG_SINE_SEAL_SEVER=1> .venv/bin/python3 scripts/sine_birth_check.py

Exits 0 and prints PASS iff `out == cur` bitwise (atol=1e-6) at EVERY breath kb=1..ALG_BREATH-1
(the full loop depth, not just kb=1,2 as the original diagnostic sampled) under the arm's own
env; exits 1 and prints FAIL otherwise, printing the ratio for the first failing breath.
ALG_SINE_SEAL_SEVER=1 (SEAL_241's own door) is READ but not exercised by this script directly --
it lives outside `_depth_stack_sine` entirely (in forward()'s own breath loop, beside
`_polar_waist()`'s call site), so this script's own call to `_depth_stack_sine` is identical
whether or not that flag is set; it is printed for the record so a SEAL run's own gate log line
is self-describing.
"""
import os
import sys

sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np
import phase1_algebra_head as H
from tinygrad import Tensor

assert H.ALG_SINE, "ALG_SINE=<sine|anti|flat> required (caller's own arm env)"
assert H.ALG_SINE_SKIP == "none", f"ALG_SINE_SKIP={H.ALG_SINE_SKIP!r} is refused when ALG_SINE is set"

p = H.build_params(0)
rng = np.random.default_rng(0)
B, L, T = 2, 48, 256
cur = Tensor(rng.standard_normal((B, L, H.H_W)).astype(np.float32))
waist = Tensor(rng.standard_normal((B, T, H.H_W)).astype(np.float32))
tokmask = Tensor(np.ones((B, T), np.float32))
slot_mask = Tensor(np.ones((B, L, L), np.float32))
state = {}
c = cur.numpy()

K_B = int(os.environ.get("ALG_BREATH", "7"))
all_ok = True
worst = 0.0
for kb in range(1, K_B):
    out = H._depth_stack_sine(p, cur, slot_mask, waist, tokmask, B, kb, state).numpy()
    ratio = float(np.linalg.norm(out) / np.linalg.norm(c))
    ok = bool(np.allclose(out, c, atol=1e-6))
    maxdiff = float(np.abs(out - c).max())
    worst = max(worst, maxdiff)
    all_ok = all_ok and ok
    print(f"[birth] ALG_SINE={H.ALG_SINE} skip={H.ALG_SINE_SKIP} seal_sever={H.ALG_SINE_SEAL_SEVER} "
          f"kb={kb}: ||out||/||cur|| = {ratio:.4f}   out == cur: {ok}   maxdiff={maxdiff:.3e}")

print(f"[birth] {'PASS' if all_ok else 'FAIL'} (ALG_SINE={H.ALG_SINE} skip={H.ALG_SINE_SKIP} "
      f"seal_sever={H.ALG_SINE_SEAL_SEVER}): out == cur bitwise at every breath kb=1..{K_B - 1}: "
      f"{all_ok} (worst maxdiff over all breaths: {worst:.3e})")
sys.exit(0 if all_ok else 1)
