"""complex_tensor.py — THE BUS'S NATIVE TONGUE AS A LIBRARY (2026-08-28,
Bryce's slower-is-faster gut): FHRR/rotational-binding ops over both
numpy (native complex64) and tinygrad (interleaved-real R^{2P}) — one
API, two backends, the C/R isomorphism enforced by construction.
Shaped for eventual tinygrad upstreaming (userspace, zero core changes).

Layout law: an R^{2P} interleaved-real vector [..., x_p, y_p, ...]
IS the complex vector z_p = x_p + i*y_p. All ops preserve modulus
(rotation-only) — the reason bindings survive every RMS norm.
"""
import numpy as np

def lift(v):
    """R^{2P} interleaved-real -> C^P (numpy)."""
    v2 = np.asarray(v).reshape(*np.shape(v)[:-1], -1, 2)
    return (v2[..., 0] + 1j * v2[..., 1]).astype(np.complex64)

def lower(z):
    """C^P -> R^{2P} interleaved-real (numpy)."""
    return np.stack([z.real, z.imag], -1).reshape(*z.shape[:-1], -1).astype(np.float32)

def phasor(theta):
    """role angles -> unit phasor e^{i theta} (C^P)."""
    return np.exp(1j * np.asarray(theta)).astype(np.complex64)

def bind(z, role_phasor):
    """binding = elementwise complex multiplication (phase addition)."""
    return z * role_phasor

def unbind(z, role_phasor):
    """unbinding = multiplication by the conjugate."""
    return z * np.conj(role_phasor)

def cleanup(z, codebook_c):
    """nearest code by Re<z, c> (the R^{2P} cosine numerator's isomorph)."""
    zn = z / (np.sqrt((np.abs(z) ** 2).sum(-1, keepdims=True)) + 1e-9)
    return (zn @ np.conj(codebook_c).T).real.argmax(-1)

def bind_clause_real(cb_real, thetas, ids):
    """gold-builder helper: superpose role-bound codes, all-real in/out.
    cb_real: (N, 2P) codebook; thetas: {role: (P,)}; ids: {role: idx}."""
    z = None
    for role, idx in ids.items():
        t = bind(lift(cb_real[idx]), phasor(thetas[role]))
        z = t if z is None else z + t
    return lower(z)

# ---- tinygrad side: same ops on interleaved-real Tensors ----
def tg_rotate(x, cos_t, sin_t):
    """tinygrad: rotate interleaved-real (..., 2P) by per-plane angles.
    cos_t/sin_t: (P,) numpy or Tensor. Pure mul/add — JIT-safe."""
    from tinygrad import Tensor
    P = x.shape[-1] // 2
    xr = x.reshape(*x.shape[:-1], P, 2)
    a, b = xr[..., 0], xr[..., 1]
    c = cos_t if isinstance(cos_t, Tensor) else Tensor(cos_t.astype('float32'))
    s = sin_t if isinstance(sin_t, Tensor) else Tensor(sin_t.astype('float32'))
    return Tensor.stack(a * c - b * s, a * s + b * c, dim=-1).reshape(*x.shape)

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    cb = rng.standard_normal((32, 128)).astype(np.float32)
    cb /= np.linalg.norm(cb, axis=1, keepdims=True)
    th = rng.uniform(0, 2*np.pi, 64)
    z = bind(lift(cb[3]), phasor(th))
    assert cleanup(unbind(z, phasor(th)), lift(cb)) == 3
    v = bind_clause_real(cb, {"a": th}, {"a": 7})
    assert cleanup(unbind(lift(v), phasor(th)), lift(cb)) == 7
    # tg_rotate coverage (audit 2026-08-30: was dead + untested — a future
    # caller must inherit a VERIFIED sign convention)
    import os as _os
    _os.environ.setdefault("DEV", "CPU")
    from tinygrad import Tensor as _T
    _x = rng.standard_normal(128).astype(np.float32)
    _got = tg_rotate(_T(_x.reshape(1, 128)), np.cos(th).astype(np.float32),
                     np.sin(th).astype(np.float32)).numpy()[0]
    _ref = lower(bind(lift(_x), phasor(th)))
    assert np.abs(_got - _ref).max() < 1e-5, "tg_rotate sign convention broken"
    print("complex_tensor: self-tests pass (incl. tg_rotate == bind)")
