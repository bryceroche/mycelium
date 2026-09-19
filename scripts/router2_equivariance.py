"""scripts/router2_equivariance.py — THE EQUIVARIANCE CHECK for ALG_ROUTER=2
(THE BUS-NATIVE ROUTER, 2026-09-19 THE SURFACE PIVOT).

Standalone (no network, no training data). Verifies:
  1. unbind(bind(x, theta), theta) == x — the phasor rotation is exactly
     invertible (bind = rotate by +theta, unbind = rotate by -theta).
  2. score(rotate(bus), rotate(key)) == score(bus, key) for the router's
     OWN score function, under an INDEPENDENT random rotation applied
     per-plane, identically to the bus vector and the token key — the
     rotation-equivariance the brief requires "by construction".

THE DESIGN DECISION this file documents (see the same note at
ALG_ROUTER=2's params build in scripts/phase1_algebra_head.py): W_rq2 is
a PER-PLANE complex scalar (a (P,2) real/imag pair per plane), not a
dense 512x512 mixer. This is what makes check #2 hold EXACTLY under
INDEPENDENT per-plane rotations: complex multiplication commutes with
rotation within the SAME plane, but a dense mixer that combines
different planes only commutes with a single GLOBAL phase shared by
every plane, not with independently-varying per-plane angles (the
theta_arg1/arg2/res/given vectors are exactly that: 256 independent
angles). Check #3 below is a negative control that makes this concrete:
a dense plane-mixing projection in the query path visibly BREAKS the
property under the same rotation. W_rk2 (the key's H_W->512 projection)
is a free dense matrix and is deliberately NOT part of the rotated
quantity here: the check rotates the KEY VECTOR directly (as the
brief's "rotate ... token keys" describes), so nothing about W_rk2's
structure enters the property — dot products are invariant under any
common orthogonal map applied to both operands, independent of how
either operand was produced upstream.
"""
import os
os.environ.setdefault("DEV", "CPU")
import numpy as np
from tinygrad import Tensor


def rot2(v, c, s):
    """The head's own interleaved-real phasor rotation (_rot2 in
    phase1_algebra_head.py), reproduced verbatim: bind = rotate by
    +theta (c=cos(theta), s=sin(theta)); unbind = rotate by -theta
    (c=cos(theta), s=-sin(theta))."""
    vr = v.reshape(*v.shape[:-1], v.shape[-1] // 2, 2)
    x, y = vr[..., 0], vr[..., 1]
    return Tensor.stack(x * c - y * s, x * s + y * c, dim=-1).reshape(*v.shape)


def main():
    rng = np.random.default_rng(0)
    P = 256
    B, L, T = 2, 5, 7

    # ---- check 1: unbind(bind(x, theta), theta) == x ----
    x = Tensor(rng.standard_normal((B, L, 2 * P)).astype(np.float32))
    theta = rng.uniform(0, 2 * np.pi, P).astype(np.float32)
    c, s = np.cos(theta).astype(np.float32), np.sin(theta).astype(np.float32)
    bound = rot2(x, Tensor(c), Tensor(s))
    unbound = rot2(bound, Tensor(c), Tensor(-s))
    err1 = float((unbound - x).abs().max().numpy())
    assert err1 < 1e-4, f"unbind(bind(x,theta),theta) != x: max err {err1}"
    print(f"[1] unbind(bind(x,theta),theta) == x: max err {err1:.3e}  PASS")

    # ---- check 2: rotation equivariance of the router's score ----
    bus = Tensor(rng.standard_normal((B, L, 2 * P)).astype(np.float32))
    key = Tensor(rng.standard_normal((B, T, 2 * P)).astype(np.float32))
    # the field's role angle (stand-in for theta_arg1/arg2/res, or the
    # learned theta_given) — 256 INDEPENDENT angles, exactly the shape
    # the real theta_* vectors carry
    theta_f = rng.uniform(0, 2 * np.pi, P).astype(np.float32)
    cf, sf = np.cos(theta_f).astype(np.float32), np.sin(theta_f).astype(np.float32)
    # W_rq2: a per-plane complex scalar, RANDOM here (not the birth
    # identity init) — a real test of the general claim, not just of
    # the do-nothing case
    wr = rng.standard_normal(P).astype(np.float32)
    wi = rng.standard_normal(P).astype(np.float32)

    def score(bus_, key_):
        q = rot2(bus_, Tensor(cf), Tensor(-sf))     # unbind by theta_f
        q = rot2(q, Tensor(wr), Tensor(wi))         # W_rq2, the per-plane scale
        return (q @ key_.transpose(-2, -1)) / np.sqrt(P)

    s0 = score(bus, key).numpy()

    # an INDEPENDENT random rotation per plane, applied IDENTICALLY to
    # the bus vector and the token key (the brief's "rotate both by the
    # same per-plane angles")
    phi = rng.uniform(0, 2 * np.pi, P).astype(np.float32)
    cphi, sphi = np.cos(phi).astype(np.float32), np.sin(phi).astype(np.float32)
    bus_r = rot2(bus, Tensor(cphi), Tensor(sphi))
    key_r = rot2(key, Tensor(cphi), Tensor(sphi))
    s1 = score(bus_r, key_r).numpy()

    err2 = float(np.abs(s0 - s1).max())
    assert err2 < 1e-4, f"rotation equivariance broken: max err {err2}"
    print(f"[2] score(rotate(bus), rotate(key)) == score(bus, key): "
          f"max err {err2:.3e}  PASS")

    # ---- negative control (documents WHY W_rq2 is diagonal rather than
    # a dense mixer, instead of asserting the design choice silently) ----
    Wd = Tensor((rng.standard_normal((2 * P, 2 * P)) / np.sqrt(2 * P)).astype(np.float32))

    def score_dense(bus_, key_):
        q = rot2(bus_, Tensor(cf), Tensor(-sf))
        q = q @ Wd                                  # a DENSE, plane-mixing "W_rq2"
        return (q @ key_.transpose(-2, -1)) / np.sqrt(P)

    d0 = score_dense(bus, key).numpy()
    d1 = score_dense(bus_r, key_r).numpy()
    errd = float(np.abs(d0 - d1).max())
    print(f"[3] (negative control) a DENSE plane-mixing projection breaks "
          f"equivariance under INDEPENDENT per-plane rotations: max err "
          f"{errd:.3e} (expected LARGE — confirms why the real W_rq2 is "
          f"the per-plane diagonal complex scalar, not a dense mixer)")
    assert errd > 1e-2, "expected the dense control to break equivariance"

    print("router2_equivariance: ALL CHECKS PASS")


if __name__ == "__main__":
    main()
