"""sine_seal_rank.py -- THE SEAL RANK, REDESIGNED (2026-10-11, after the skip bug: ledger 08:52).
The original version tested _depth_stack_sine's own "boundary" carry, which no longer exists
(ALG_SINE_SKIP="peak"/"boundary" are refused -- see the module comment above ALG_SINE in
phase1_algebra_head.py). THE NEW SEAL PLACEBO (SEAL_241) lives entirely OUTSIDE
_depth_stack_sine: ALG_SINE_SEAL_SEVER=1 severs `_polar_waist()`'s own call in forward()'s
breath loop (the SAME "waist" sever THE WAIST READ, 2026-09-24 already reads with, now union'd
into `_SEVER` under its own training-flag name). This script tests THAT function directly, on
its own real linear-algebra terms -- no training needed, no birth/ReZero property involved: the
waist's reconstruction `_cn = (u @ W_down) @ W_up` is a composition through a width-128
bottleneck, so its JACOBIAN w.r.t. `u`'s content block has rank <= ALG_POLAR_D=128 by
CONSTRUCTION, at every step, trained or not -- a structural bound, not a birth-only one (the
crystal's own squeeze is linear-algebraically never information-preserving above rank 128
REGARDLESS of training; only severing it removes the bound). With the sever
(ALG_SINE_SEAL_SEVER=1), `_polar_waist` is never called at all (content passes straight through
unchanged -- full rank, bounded only by the content block's own width, 384). Finite-difference
SVD on the REAL `_polar_waist` function, small batch, no training.
"""
import os
import sys

sys.path.insert(0, "."); sys.path.insert(0, "scripts")
os.environ.setdefault("DEV", "CPU")
import numpy as np

import phase1_algebra_head as H
from tinygrad import Tensor

assert H.ALG_POLAR and H.POLAR_D == 128, "ALG_POLAR=1 ALG_POLAR_D=128 required"

EPS = 1e-3
B, L, C = 1, 2, 384   # a tiny body: 2 "slots" (really just rows), the 384 content dims only


def _waist_rank(severed):
    p = H.build_params(0)
    csel, n = H._paircmp_content_sel()
    assert n == 384
    rng = np.random.default_rng(5)
    u0 = rng.standard_normal((B, L, H.H_W)).astype(np.float32)
    # zero the clock dims of the probe so the content-only Jacobian is clean (the waist leaves
    # clock dims bitwise untouched either way -- not what this test is about)
    gk = H._polar_sink()[3].numpy()
    u0 = u0 * (1.0 - gk).reshape(1, 1, -1)

    n_c = C
    xs = np.repeat(u0.reshape(1, -1), 2 * n_c, axis=0)
    cdims = np.flatnonzero((1.0 - gk) > 0.5)
    assert len(cdims) == 384, len(cdims)
    for j, d in enumerate(cdims):
        xs[2 * j, d] += EPS
        xs[2 * j + 1, d] -= EPS
    xs = xs.reshape(2 * n_c, L, H.H_W)
    state = {}
    if severed:
        # THE SEAL PLACEBO: _polar_waist is never called -- content passes straight through.
        out = xs
    else:
        out = H._polar_waist(Tensor(xs), p, state).numpy()
    y = out.reshape(2 * n_c, -1).astype(np.float64)
    J = ((y[0::2] - y[1::2]) / (2 * EPS)).T
    sv = np.linalg.svd(J, compute_uv=False)
    rank = int((sv > 1e-2 * max(sv[0], 1e-12)).sum())
    return rank, n_c, sv[:5]


def main():
    print(f"[seal-rank] ALG_POLAR_D={H.POLAR_D}  content dims probed={C}  structural bound={H.POLAR_D}")
    for severed, label in ((False, "not-severed (the waist runs)"), (True, "severed (ALG_SINE_SEAL_SEVER=1)")):
        rank, n_c, sv5 = _waist_rank(severed)
        bound = H.POLAR_D
        verdict = ("sealed: rank bounded by the crystal" if rank <= bound
                   else "BYPASSED: the crystal is optional")
        print(f"[seal-rank] {label:32s} rank {rank:4d} / {n_c}  (crystal bound {bound})  "
              f"top sv {[round(float(x), 4) for x in sv5]}  {verdict}")


if __name__ == "__main__":
    main()
