"""rotor_clock.py — THE MASTER CLOCK (2026-08-29, word given).

ONE CLOCK, THREE GEARS, SEPARATE BANDS. This module is the single source
of truth for every phase table in the three-rotor stack (CLAUDE.md S4;
docs/rotational_bus.md S0/S5). S2 (recurrent bus) imports its cycle keys
from here; S3 (breath-phase attention rotation) imports its schedule from
here. Sync is enforced by the import graph, not by convention — two organs
deriving time independently WILL drift the first time one is edited.

Laws embedded here:
  * FREQUENCIES FROZEN, GAINS LEARNABLE — every rate below is a pinned
    constant; a learnable rate is a Goodhart door (the network escapes a
    rotation cheapest by learning delta -> 0). Gates (sw_g-style,
    zero-init) live in the head, never here.
  * THE T^3 ODOMETER — three wave-pairs at geared rates (1x, 2x, 4x):
    wheel 1 is the coarse address (alias-free across the breath window by
    construction, self-tested below); wheels 2-3 are fine/redundant.
  * SEPARATE BANDS — plane allocations are declared here so no rotor can
    silently claim another's spectrum. The token rotor (trunk RoPE) is
    frozen and internal; it owns nothing here and nothing here touches it.
  * DELIBERATE DE-SYNC — the bus stays blind to token position (the
    frame-free-graph law); this module offers NO token-indexed phases.
  * Audits are OBSERVATIONAL (Goodhart fence): demodulation/aliasing
    readers below never enter any loss.
"""
import math

import numpy as np

# ---------------------------------------------------------------- constants
K_BREATH = 7                     # the breath window (ALG_BREATH)
N_WHEELS = 3                     # the T^3: three antiphase pairs
GEARS = (1, 2, 4)                # geared rates per wheel (coarse -> fine)
DELTA = 2.0 * math.pi / 8.0      # base quantum: 45deg — wheel 1 is unique
                                 # for k = 0..7 >= K_BREATH (alias-free)
CYCLE_SEED = 13                  # per-plane bus cycle rates (frozen)
SCRAMBLE_SEED = 1013             # the scramble-control table (kill-criterion arm)

# Band allocation for S3 (the breath rotor's reserved attention planes).
# Head dim = ALG_HW / N_HEADS = 512/8 = 64 -> 32 rotation pairs; the breath
# rotor owns the TOP 8 pairs; pairs 0-23 stay static (token-derived
# geometry survives rotation).
HEAD_DIM_PAIRS = 32
BREATH_BAND_PAIRS = tuple(range(24, 32))


# ------------------------------------------------------------------ the T^3
def phase_of(k, gears=GEARS, delta=DELTA):
    """The master phase vector at breath k: one angle per wheel (radians,
    wrapped). THE single time authority — derive, never duplicate."""
    return np.array([(g * delta * k) % (2.0 * math.pi) for g in gears],
                    np.float32)


def wheel_table(K=K_BREATH, gears=GEARS, delta=DELTA):
    """(K, N_WHEELS) table of angles for the whole breath window."""
    return np.stack([phase_of(k, gears, delta) for k in range(K)])


# -------------------------------------------------- bus rotor: cycle keying
def cycle_rates(P, seed=CYCLE_SEED):
    """Frozen per-plane rates for the bus's breath-cycle keying (FHRR
    style): plane p advances by rates[p] per breath. Uniform in
    [0.25, 1.75]*DELTA — bounded away from 0 (no stationary planes) and
    from 2pi-degeneracy."""
    rng = np.random.default_rng(seed)
    return (rng.uniform(0.25, 1.75, P) * DELTA).astype(np.float32)


def cycle_phasor(k, P, seed=CYCLE_SEED):
    """e^{i k * rates} as complex64 (P,) — multiply to BIND a write at
    breath k; conjugate to unbind."""
    return np.exp(1j * k * cycle_rates(P, seed)).astype(np.complex64)


def cycle_cos_sin(K, P, seed=CYCLE_SEED):
    """(K, P) cos and sin tables, float32 — the tinygrad-facing form for
    interleaved-real rotation (x' = x c - y s ; y' = x s + y c)."""
    th = np.outer(np.arange(K, dtype=np.float32), cycle_rates(P, seed))
    return np.cos(th).astype(np.float32), np.sin(th).astype(np.float32)


# ------------------------------------------- breath rotor: attention rotation
def breath_qk_angles(K=K_BREATH, pairs=BREATH_BAND_PAIRS):
    """(K, len(pairs)) rotation angles for the reserved attention pairs:
    reserved pair j turns with wheel j % N_WHEELS. Multiplicative entry
    only (geometry, not bias — the alpha null is scope-tagged additive)."""
    wt = wheel_table(K)
    return np.stack([wt[:, j % N_WHEELS] for j in range(len(pairs))], 1)


def scrambled_wheel_table(K=K_BREATH, seed=SCRAMBLE_SEED):
    """The control arm's table: same marginal angles, order shuffled per
    wheel. A load-bearing clock HURTS when scrambled; a decorative one
    doesn't (the standing tripwire from the alpha lesson)."""
    rng = np.random.default_rng(seed)
    wt = wheel_table(K).copy()
    for w in range(wt.shape[1]):
        rng.shuffle(wt[:, w])
    return wt


# ------------------------------------------------- audits (observational)
def demod_breath(wire_c, P, K=K_BREATH, seed=CYCLE_SEED):
    """THE ODOMETER TEST: which breath wrote this wire? Counter-rotate by
    every candidate k, score by peak cleanup coherence proxy (energy
    concentration of the counter-rotated vector's mean phasor). Returns
    argmax k. Observational only — never in a loss."""
    scores = []
    for k in range(K):
        z = wire_c * np.conj(cycle_phasor(k, P, seed))
        scores.append(float(np.abs(z.sum()) / (np.abs(z).sum() + 1e-9)))
    return int(np.argmax(scores)), scores


def aliasing_confusion(P, K=K_BREATH, seed=CYCLE_SEED, trials=64, rng_seed=7):
    """Write random unit content at breath j, demod across k: (K, K)
    row-normalized confusion. Off-diagonal mass = gearing failure."""
    rng = np.random.default_rng(rng_seed)
    M = np.zeros((K, K))
    for _ in range(trials):
        c = rng.standard_normal(P) + 1j * rng.standard_normal(P)
        c = (c / np.abs(c)).astype(np.complex64)
        for j in range(K):
            k_hat, _ = demod_breath(c * cycle_phasor(j, P, seed), P, K, seed)
            M[j, k_hat] += 1
    return M / trials


# ------------------------------------------------------------- self-tests
def _selftest():
    # 1. wheel-1 alias-free across the window
    w1 = [phase_of(k)[0] for k in range(K_BREATH)]
    assert len({round(a, 6) for a in w1}) == K_BREATH, "wheel-1 aliases"
    # 2. full T^3 vector unique per breath
    vecs = {tuple(np.round(phase_of(k), 6)) for k in range(K_BREATH)}
    assert len(vecs) == K_BREATH, "T^3 odometer aliases"
    # 3. cycle keys distinguishable: min pairwise phasor distance
    P = 256
    ph = [cycle_phasor(k, P) for k in range(K_BREATH)]
    dmin = min(np.abs(ph[a] - ph[b]).mean()
               for a in range(K_BREATH) for b in range(a + 1, K_BREATH))
    assert dmin > 0.3, f"cycle keys too close ({dmin:.3f})"
    # 4. the odometer reads its own writes perfectly on clean wires
    M = aliasing_confusion(P, trials=16)
    assert float(np.trace(M)) / K_BREATH > 0.99, "odometer misreads clean writes"
    # 5. scramble differs from the true table
    assert not np.allclose(scrambled_wheel_table(), wheel_table()), "scramble no-op"
    return True


_selftest()

if __name__ == "__main__":
    print(f"[rotor_clock] K={K_BREATH} gears={GEARS} delta={math.degrees(DELTA):.0f}deg")
    print(f"[rotor_clock] wheel table (deg):\n{np.degrees(wheel_table()).round(0)}")
    M = aliasing_confusion(256)
    print(f"[rotor_clock] odometer diag mean: {float(np.trace(M))/K_BREATH:.3f} "
          f"(off-diag max {float((M - np.diag(np.diag(M))).max()):.3f})")
    print("[rotor_clock] ALL SELF-TESTS PASS — one clock, three gears, separate bands")
