"""rotor_clock.py — THE MASTER CLOCK (2026-08-29; SEXTET CLOCK, word given).

ONE CLOCK, THREE GEARS, SEPARATE BANDS. This module is the single source
of truth for every phase table in the three-rotor stack (CLAUDE.md S4;
docs/rotational_bus.md S0/S5). S2 (recurrent bus) imports its cycle keys
from here; S3 (breath-phase attention rotation) imports its schedule from
here. Sync is enforced by the import graph, not by convention.

THE SEXTET CLOCK (canon-harmonized): the six helical waves ARE the clock.
The breath window is breath-0 (the raw bank read) + SIX loop breaths
(kb = 1..6). Quantum = 60 degrees; loop breath k carries tick (k-1):
phase = (k-1) * 60deg — six ticks, six wave phases, zero aliasing.
**BREATH-0 IS OUTSIDE TIME**: the register-invariant raw state (the S1
two-tap law) is UNCLOCKED by construction — it exists before the clock
starts; S2 never writes at breath 0; asking for its phase is an error.

Wheels (each with a JOB, not just a rate):
  wheel 0 — THE BREATH HAND: (k-1)*60deg, unique across the six loop
            breaths (the address).
  wheel 1 — PARITY: 2x rate (period 3), cheap redundancy/error margin.
  wheel 2 — THE PASS WHEEL: advances 60deg per FULL PASS; static this
            era (single-pass), reserved for multi-pass deliberation —
            hierarchical time on one torus.

Laws embedded here:
  * FREQUENCIES FROZEN, GAINS LEARNABLE — no rate here is a parameter
    (a learnable rate is a Goodhart door: delta -> 0 escapes rotation).
  * SEPARATE BANDS — plane allocations declared here; no rotor claims
    another's spectrum. The token rotor (trunk RoPE) owns nothing here.
  * DELIBERATE DE-SYNC — no token-indexed phases exist in this module
    (the frame-free-graph law: the bus stays blind to position).
  * Audits are OBSERVATIONAL (Goodhart fence) — never in any loss.
"""
import math

import numpy as np

# ---------------------------------------------------------------- constants
K_BREATH = 7                     # states: breath-0 + six loop breaths
N_LOOP = 6                       # clocked loop breaths (kb = 1..6)
N_WHEELS = 3
QUANTUM = 2.0 * math.pi / 6.0    # 60 degrees — the sextet quantum
GEARS = (1, 2)                   # breath hand, parity (pass wheel is
                                 # pass-indexed, not breath-geared)
CYCLE_SEED = 13                  # per-plane bus cycle rates (frozen)
SCRAMBLE_SEED = 1013             # scramble-control table (kill-criterion arm)

# Band allocation for S3. TRUTH-MAINTAINED 2026-08-30: the live bands are
# AUDIT-DRAFTED (spectrum audit ranks all 32 pairs by q-k utility; the
# mandate takes the slackest, electives the next; the most load-bearing
# stay static) and travel via the ROT_BANDS json (.cache/rot_band_draft
# .json) consumed by the head. BREATH_BAND_PAIRS below is the LEGACY fixed
# band (v0's reserved 24-31) kept only for breath_qk_angles() callers;
# it does NOT describe the live allocation.
HEAD_DIM_PAIRS = 32
BREATH_BAND_PAIRS = tuple(range(24, 32))    # legacy (v0); live = ROT_BANDS


def is_clocked(k):
    """Loop breaths 1..6 are clocked; breath-0 is outside time."""
    return 1 <= k <= N_LOOP


# ------------------------------------------------------------------ the T^3
def phase_of(k, pass_idx=0):
    """The master phase vector at loop breath k (1..6): [breath hand,
    parity, pass wheel] in radians, wrapped. THE single time authority.
    Breath-0 is outside time — asking for its phase is an error."""
    if not is_clocked(k):
        raise ValueError(f"breath {k} is outside time (clocked: 1..{N_LOOP})")
    t = k - 1
    return np.array([(GEARS[0] * QUANTUM * t) % (2.0 * math.pi),
                     (GEARS[1] * QUANTUM * t) % (2.0 * math.pi),
                     (QUANTUM * pass_idx) % (2.0 * math.pi)], np.float32)


def wheel_table(pass_idx=0):
    """(N_LOOP, N_WHEELS) angles for the clocked window."""
    return np.stack([phase_of(k, pass_idx) for k in range(1, N_LOOP + 1)])


# -------------------------------------------------- bus rotor: cycle keying
def cycle_rates(P, seed=CYCLE_SEED):
    """Frozen per-plane rates for the bus's breath-cycle keying (FHRR
    style): plane p advances by rates[p] per tick. Uniform in
    [0.25, 1.75]*QUANTUM — bounded off 0 (no stationary planes) and off
    degeneracy."""
    rng = np.random.default_rng(seed)
    return (rng.uniform(0.25, 1.75, P) * QUANTUM).astype(np.float32)


def cycle_phasor(k, P, seed=CYCLE_SEED):
    """e^{i (k-1) * rates} as complex64 (P,) — multiply to BIND a write at
    loop breath k; conjugate to unbind. Breath-0 is never written."""
    if not is_clocked(k):
        raise ValueError(f"breath {k} is outside time — the bus never writes it")
    return np.exp(1j * (k - 1) * cycle_rates(P, seed)).astype(np.complex64)


def cycle_cos_sin(P, seed=CYCLE_SEED):
    """(N_LOOP, P) cos and sin tables (ticks 0..5), float32 — the
    tinygrad-facing form for interleaved-real rotation
    (x' = x c - y s ; y' = x s + y c)."""
    th = np.outer(np.arange(N_LOOP, dtype=np.float32), cycle_rates(P, seed))
    return np.cos(th).astype(np.float32), np.sin(th).astype(np.float32)


# ------------------------------------------- breath rotor: attention rotation
def breath_qk_angles(pass_idx=0, pairs=BREATH_BAND_PAIRS):
    """(N_LOOP, len(pairs)) rotation angles for the reserved attention
    pairs: reserved pair j turns with wheel j % N_WHEELS (pass-wheel pairs
    are static this era — reserved spectrum, dormant). Multiplicative
    entry only (geometry, not bias — the alpha null is scope-tagged
    additive)."""
    wt = wheel_table(pass_idx)
    return np.stack([wt[:, j % N_WHEELS] for j in range(len(pairs))], 1)


def scrambled_wheel_table(seed=SCRAMBLE_SEED):
    """The control arm's table: same marginal angles, order shuffled per
    wheel. A load-bearing clock HURTS when scrambled; a decorative one
    doesn't (the alpha lesson as a standing tripwire)."""
    rng = np.random.default_rng(seed)
    wt = wheel_table().copy()
    for w in range(wt.shape[1]):
        rng.shuffle(wt[:, w])
    return wt


# ------------------------------------------------- audits (observational)
def demod_breath(wire_c, refs_c, seed=CYCLE_SEED):
    """THE ODOMETER TEST: which loop breath wrote this wire?
    Counter-rotate by every candidate k (1..6) and score against the KNOWN
    reference set (cleanup needs a dictionary — random content has no
    self-coherence; draft-1 of this audit failed its own self-test on
    exactly that). refs_c: (n_refs, P) complex candidates. Returns
    (k_hat, scores). Observational only."""
    P = wire_c.shape[-1]
    R = np.atleast_2d(refs_c)
    Rn = R / (np.linalg.norm(R, axis=-1, keepdims=True) + 1e-9)
    scores = []
    for k in range(1, N_LOOP + 1):
        z = wire_c * np.conj(cycle_phasor(k, P, seed))
        zn = z / (np.linalg.norm(z) + 1e-9)
        scores.append(float(np.abs(zn @ np.conj(Rn).T).max()))
    return 1 + int(np.argmax(scores)), scores


def aliasing_confusion(P, seed=CYCLE_SEED, trials=64, rng_seed=7):
    """Write random unit content at loop breath j, demod across k:
    (N_LOOP, N_LOOP) row-normalized confusion (index 0 = breath 1).
    Off-diagonal mass = gearing failure."""
    rng = np.random.default_rng(rng_seed)
    M = np.zeros((N_LOOP, N_LOOP))
    for _ in range(trials):
        c = rng.standard_normal(P) + 1j * rng.standard_normal(P)
        c = (c / np.abs(c)).astype(np.complex64)
        for j in range(1, N_LOOP + 1):
            k_hat, _ = demod_breath(c * cycle_phasor(j, P, seed), c, seed)
            M[j - 1, k_hat - 1] += 1
    return M / trials


# ------------------------------------------------------------- self-tests
def _selftest():
    # 1. the breath hand is unique across the six loop breaths
    w0 = [phase_of(k)[0] for k in range(1, N_LOOP + 1)]
    assert len({round(a, 6) for a in w0}) == N_LOOP, "breath hand aliases"
    # 2. full vector unique per loop breath (pass fixed)
    vecs = {tuple(np.round(phase_of(k), 6)) for k in range(1, N_LOOP + 1)}
    assert len(vecs) == N_LOOP, "T^3 odometer aliases"
    # 3. breath-0 is outside time
    for fn in (lambda: phase_of(0), lambda: cycle_phasor(0, 8)):
        try:
            fn(); raise AssertionError("breath-0 accepted a clock read")
        except ValueError:
            pass
    # 4. cycle keys distinguishable across ticks
    P = 256
    ph = [cycle_phasor(k, P) for k in range(1, N_LOOP + 1)]
    dmin = min(np.abs(ph[a] - ph[b]).mean()
               for a in range(N_LOOP) for b in range(a + 1, N_LOOP))
    assert dmin > 0.3, f"cycle keys too close ({dmin:.3f})"
    # 5. the odometer reads its own clean writes perfectly
    M = aliasing_confusion(P, trials=16)
    assert float(np.trace(M)) / N_LOOP > 0.99, "odometer misreads clean writes"
    # 6. scramble differs from the true table
    assert not np.allclose(scrambled_wheel_table(), wheel_table()), "scramble no-op"
    # 7. the pass wheel advances across passes, wheels 0-1 unmoved
    a, b = phase_of(3, pass_idx=0), phase_of(3, pass_idx=1)
    assert np.allclose(a[:2], b[:2]) and not np.isclose(a[2], b[2]), \
        "pass wheel broken"
    return True


_selftest()

if __name__ == "__main__":
    print(f"[rotor_clock] SEXTET: {N_LOOP} loop breaths @ "
          f"{math.degrees(QUANTUM):.0f}deg, breath-0 outside time")
    print(f"[rotor_clock] wheel table (deg):\n{np.degrees(wheel_table()).round(0)}")
    M = aliasing_confusion(256)
    print(f"[rotor_clock] odometer diag mean: {float(np.trace(M))/N_LOOP:.3f} "
          f"(off-diag max {float((M - np.diag(np.diag(M))).max()):.3f})")
    print("[rotor_clock] ALL SELF-TESTS PASS — the six waves ARE the clock")
