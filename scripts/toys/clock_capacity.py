"""clock_capacity.py -- IS THE CLOCK A KEY TO STORAGE CAPACITY? (2026-10-16; Bryce's gut, zero GPU, numpy, 3 seeds.)
D = 256 phase planes; items = random value phasors (cleanup vs 1000); N items stored, one per tick t = 0..N-1.
  RANDOM      one vector, each item bound to an independent random key                (the ~20-item desk)
  CLOCK6      one vector, key(t) = the six-wave clock: 6 frequencies, each REPEATED across 256/6 planes
              (the clock as it is: 6 independent planes of information)
  CLOCK256    one vector, key(t) = e^{i w_d t} with 256 distinct frequencies (the clock LIFTED, RoPE-style)
  GOLDEN256   CLOCK256 with golden-ratio-spaced frequencies (w_d = 2 pi frac(d * phi))
  THETAGAMMA  the time-multiplexed form: R registers; tick t writes register t mod R with a CLOCK256 key
              (the clock picks WHICH register is live; Lisman-Idiart's theta-gamma code)
  MULTISCALE  geometric frequencies (periods 2..1000 ticks), RoPE-style (added after the first read: a full-band clock
              is a HASH -- its half-tick kernel is ~0 everywhere, so nearest-behind reads 0.000)
Also NEAREST-BEHIND: query the clock at t + 0.5 (between ticks) -- does the code retrieve tick t or t+1 (a smooth
time kernel = 'the latest event behind me') vs a random key (no notion of nearby time)?
"""
import numpy as np

D, NV = 256, 1000
PHI = (1 + 5 ** 0.5) / 2


def cos(Q, K):
    return np.real((Q / np.linalg.norm(Q, axis=-1, keepdims=True)) @ (K / np.linalg.norm(K, axis=-1, keepdims=True)).conj().T)


def keys(kind, rng, ts):
    ts = np.asarray(ts, float)[:, None]
    if kind == "RANDOM":
        return np.exp(1j * rng.uniform(0, 2 * np.pi, (len(ts), D)))
    if kind == "CLOCK6":
        w = rng.uniform(0.05, 1.0, 6) * 2 * np.pi
        w = np.repeat(w, -(-D // 6))[:D]
        return np.exp(1j * w * ts)
    if kind == "CLOCK256":
        return np.exp(1j * rng.uniform(0, 2 * np.pi, D) * ts)
    if kind == "GOLDEN256":
        return np.exp(1j * 2 * np.pi * ((np.arange(1, D + 1) * PHI) % 1.0) * ts)
    if kind == "MULTISCALE":   # RoPE-style geometric frequencies: periods 2 .. 1000 ticks (fast = exact, slow = proximity)
        return np.exp(1j * (2 * np.pi / np.geomspace(2, 1000, D)) * ts)
    raise ValueError(kind)


def recall(kind, rng, N, R=1):
    vals = np.exp(1j * rng.uniform(0, 2 * np.pi, (NV, D)))
    vi = rng.choice(NV, N, replace=N > NV)
    base = "CLOCK256" if kind == "THETAGAMMA" else kind
    seed = rng.integers(1 << 30)
    K = keys(base, np.random.default_rng(seed), range(N))
    reg = np.arange(N) % R
    M = np.zeros((R, D), complex)
    np.add.at(M, reg, K * vals[vi])
    got = cos(M[reg] * K.conj(), vals).argmax(1)
    return float((got == vi).mean())


def nearest_behind(kind, rng, N=20):
    seed = rng.integers(1 << 30)
    K = keys(kind, np.random.default_rng(seed), range(N))
    if kind == "RANDOM":
        return float("nan")
    Q = keys(kind, np.random.default_rng(seed), np.arange(N - 1) + 0.5)
    top = cos(Q, K).argmax(1)
    return float(np.isin(top - np.arange(N - 1), (0, 1)).mean())


def main():
    Ns = (10, 20, 30, 50, 100, 200)
    print("recall of N items (one per tick), mean of 3 seeds:")
    for kind, R in (("RANDOM", 1), ("CLOCK6", 1), ("CLOCK256", 1), ("GOLDEN256", 1), ("MULTISCALE", 1), ("THETAGAMMA", 8), ("THETAGAMMA", 16)):
        r = [np.mean([recall(kind, np.random.default_rng(s), N, R) for s in range(3)]) for N in Ns]
        tag = f"{kind}{'/R' + str(R) if kind == 'THETAGAMMA' else ''}"
        print(f"  {tag:14s} " + " ".join(f"N{N}={x:.2f}" for N, x in zip(Ns, r)))
    print("nearest-behind (a between-ticks query lands on an adjacent tick):")
    for kind in ("CLOCK6", "CLOCK256", "GOLDEN256", "MULTISCALE"):
        print(f"  {kind:14s} {np.mean([nearest_behind(kind, np.random.default_rng(s)) for s in range(3)]):.3f} (chance ~0.10 at N=20)")


if __name__ == "__main__":
    main()
