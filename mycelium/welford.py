"""mycelium/welford.py -- THE WELFORD ATLAS's accumulator (2026-10-07, Bryce: "the centroids need
some love, and Welford stats").

A streaming (count, mean, M2 -> variance) accumulator, vectorised over an arbitrary number of
dims, with an optional exponential-decay variant (for a quantity whose circuits keep moving, per
CLAUDE.md's "never mix generations' coordinates" -- a decayed cell forgets its oldest members
instead of averaging a rotation into nonsense) and a lossless parallel MERGE (Chan et al. 1979)
for the exact (non-decayed) form. A WelfordLibrary is a plain dict of string key -> Welford cell,
with save/load to a single npz (one array stack per field, keys as an object array) -- the "clean
library of centroids over time" the ask names.

THE FENCE this module itself never crosses: it is a MONITOR's accumulator, not a decoder -- nothing
here reads a model checkpoint, calls forward(), or touches a training loop. Callers decide what
feeds it (gold-graded rows only, per the 2026-10-07 ruling); this module only accumulates and
answers "how far is x from what I've seen" (variance/radius), never "what is x".

usage:
    from mycelium.welford import Welford, WelfordLibrary
    w = Welford(dim=384)
    w.update(x)                 # x: (D,) or (N,D)
    w.mean, w.variance, w.count

    lib = WelfordLibrary()
    lib.get_or_create("given", dim=384).update(x)
    lib.save(".cache/welford_atlas_X.npz")
    lib2 = WelfordLibrary.load(".cache/welford_atlas_X.npz")
"""
import numpy as np


class Welford:
    """One streaming accumulator over a fixed dim. decay=None -> exact Welford (Chan's online
    algorithm); decay in (0,1) -> exponentially-weighted mean/second-moment (the cell forgets its
    oldest members at rate `decay`; `count` becomes an EFFECTIVE count, saturating at 1/decay, not
    a true tally -- documented, never silently conflated with the exact form's n)."""

    __slots__ = ("dim", "decay", "n", "mean", "m2")

    def __init__(self, dim, decay=None):
        assert decay is None or 0.0 < decay < 1.0, decay
        self.dim = int(dim)
        self.decay = None if decay is None else float(decay)
        self.n = 0.0
        self.mean = np.zeros(self.dim, dtype=np.float64)
        self.m2 = np.zeros(self.dim, dtype=np.float64)

    def update(self, x):
        """x: (dim,) or (N, dim). Exact form processes rows one at a time (Welford's update is
        inherently sequential per-sample for the running mean/M2 pair); decayed form likewise."""
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x[None, :]
        assert x.shape[-1] == self.dim, (x.shape, self.dim)
        for row in x:
            if self.decay is None:
                self.n += 1.0
                d1 = row - self.mean
                self.mean = self.mean + d1 / self.n
                d2 = row - self.mean
                self.m2 = self.m2 + d1 * d2
            else:
                a = self.decay
                self.n = (1.0 - a) * self.n + 1.0
                d1 = row - self.mean
                self.mean = self.mean + a * d1
                d2 = row - self.mean
                self.m2 = (1.0 - a) * self.m2 + a * d1 * d2
        return self

    @property
    def count(self):
        return self.n

    @property
    def variance(self):
        """Per-dim variance. Exact form: the usual sample variance (M2/(n-1), n>=2; 0 for n<2).
        Decayed form: m2 IS already a decayed mean-square -- returned as-is."""
        if self.decay is not None:
            return self.m2.copy()
        if self.n < 2:
            return np.zeros(self.dim, dtype=np.float64)
        return self.m2 / (self.n - 1.0)

    @property
    def std(self):
        return np.sqrt(np.maximum(self.variance, 0.0))

    def merge(self, other):
        """In-place parallel merge (Chan et al.'s exact formula) when both cells are exact Welford
        accumulators over the SAME dim; for either/both decayed, falls back to an n-weighted
        average of (mean, m2) -- an approximation, stated here, never passed off as exact."""
        assert self.dim == other.dim, (self.dim, other.dim)
        if other.n == 0:
            return self
        if self.n == 0:
            self.n, self.mean, self.m2 = other.n, other.mean.copy(), other.m2.copy()
            self.decay = other.decay
            return self
        if self.decay is not None or other.decay is not None:
            tot = self.n + other.n
            self.mean = (self.mean * self.n + other.mean * other.n) / tot
            self.m2 = (self.m2 * self.n + other.m2 * other.n) / tot
            self.n = tot
            return self
        n_a, n_b = self.n, other.n
        delta = other.mean - self.mean
        tot = n_a + n_b
        self.mean = self.mean + delta * (n_b / tot)
        self.m2 = self.m2 + other.m2 + delta * delta * (n_a * n_b / tot)
        self.n = tot
        return self

    def cos_to_mean(self, x):
        """mean cosine similarity of x (N,dim) or (dim,) to this cell's current mean -- the
        library's "radius" statistic (THE ASK's own definition: "the mean cosine of members to
        the mean"), computed on demand, never stored as a running (and therefore stale) quantity."""
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x[None, :]
        mu = self.mean
        num = x @ mu
        den = np.linalg.norm(x, axis=-1) * (np.linalg.norm(mu) + 1e-12) + 1e-12
        return num / den


class WelfordLibrary(dict):
    """A named collection of Welford cells (string key -> Welford). Plain dict subclass; adds
    get_or_create + npz save/load. All cells in one library are assumed to share `dim` (asserted
    at save time) -- a library is one SPACE (e.g. "content state" or "retina clause"), never a mix."""

    def get_or_create(self, key, dim, decay=None):
        cell = self.get(key)
        if cell is None:
            cell = Welford(dim, decay=decay)
            self[key] = cell
        else:
            assert cell.dim == dim, (key, cell.dim, dim)
        return cell

    def save(self, path, extra_meta=""):
        keys = sorted(self.keys())
        if not keys:
            np.savez(path, keys=np.array([], dtype=object), meta=str(extra_meta))
            return
        dims = {self[k].dim for k in keys}
        assert len(dims) == 1, f"WelfordLibrary.save: mixed dims in one library: {dims}"
        dim = dims.pop()
        n = np.array([self[k].n for k in keys], dtype=np.float64)
        mean = np.stack([self[k].mean for k in keys], axis=0)
        m2 = np.stack([self[k].m2 for k in keys], axis=0)
        decay = np.array([(-1.0 if self[k].decay is None else self[k].decay) for k in keys], dtype=np.float64)
        np.savez(path, keys=np.array(keys, dtype=object), n=n, mean=mean, m2=m2, decay=decay,
                 dim=np.int64(dim), meta=str(extra_meta))

    @classmethod
    def load(cls, path):
        z = np.load(path, allow_pickle=True)
        lib = cls()
        keys = [str(k) for k in z["keys"]]
        if not keys:
            return lib
        for i, k in enumerate(keys):
            dec = float(z["decay"][i])
            cell = Welford(int(z["dim"]), decay=(None if dec < 0 else dec))
            cell.n = float(z["n"][i])
            cell.mean = z["mean"][i].astype(np.float64).copy()
            cell.m2 = z["m2"][i].astype(np.float64).copy()
            lib[k] = cell
        return lib
